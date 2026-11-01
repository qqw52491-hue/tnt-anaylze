#!/usr/bin/env python3
"""Read-only, optional TNT live angle stream for HUD.

Output protocol:
  TNTANGLE <pid> <self-uid> <fire-angle-0..180> <control-value> <qt-rotation> <mirror-0-or-1>
  TNTANGLE-                 # stale/unverified/not in battle
  TNTORIGIN <pid> <uid> <sample-unix-ms> <fire-angle> <left> <player-x> <player-y> <origin-x> <origin-y> <scale:1|2>
  TNTORIGIN-                # correction unavailable; only with --fire-origin

The candidate angle follows the reconstructed client's fireAngle() math,
but uses a Qt scene-transform cache that may be stale. Never write memory,
control the game, restart processes or perform an action on the player's behalf.
The caller selects optional origin output with --fire-origin; angle source
selection and fallback remain the HUD's responsibility.
"""
import argparse
import json
import math
import os
import queue
import subprocess
import struct
import sys
import threading
import time

from probe_live_xy_readonly import ADB, DEV, probe
from probe_live_angle_readonly import discover_self, quick_sample, sample
import probe_fire_origin_readonly as origin


class OriginStream:
    """Optional origin samples in the existing reader, never a third poller.

    Two identical geometric poses are required. Uniform scales 1 and 2 select
    separately calibrated projectile models; intermediate scales wait for stability.
    """
    def __init__(self):
        self.cache = None
        self.previous = None
        self.next_read = 0.0
        self.observation = None

    def poll(self, angle_cache, angle_row, snapshot=None, full=False):
        self.observation = None
        if angle_cache is None:
            self.cache = self.previous = None
            return "TNTORIGIN-"
        if angle_row is None:
            self.previous = None
            return "TNTORIGIN-"
        key = lambda c: (c["pid"], c["uid"], c["sprite"], c["panel"])
        if self.cache is not None and key(self.cache) != key(angle_cache):
            self.cache = self.previous = None
        if self.cache is None:
            if not full:
                return "TNTORIGIN-"
            try:
                self.cache = origin.discover(snapshot, angle_cache)
            except (RuntimeError, ValueError, OSError, struct.error, subprocess.TimeoutExpired):
                return "TNTORIGIN-"
        now = time.monotonic()
        if now < self.next_read:
            return None
        self.next_read = now + .35
        started = time.time_ns() // 1000000
        try:
            row = origin.sample(self.cache, with_aim=True)
            ended = time.time_ns() // 1000000
            self.observation = dict(row, sample_start_unix_ms=started, sample_end_unix_ms=ended,
                                    source_eligible=False)
            # Only geometry determines firePos. Control values and unrelated
            # Qt cache/padding bytes can change without moving the muzzle.
            geometry = [[n['item'], n['parent'], n['pos'], n['local_transform']] for n in row['nodes']]
            signature = json.dumps([row['pid'], row['uid'], geometry,
                                    row['fire_pos_rounded'], row['player_graphics_xy']],
                                   sort_keys=True, allow_nan=False)
            stable = self.previous is not None and self.previous[0] == signature and now - self.previous[1] <= 1.0
            self.previous = (signature, now)
            px, py = row["player_graphics_xy"]
            x, y = row["fire_pos_rounded"]
            sx,sy=row['linear_scale_xy']
            supported_scale=next((s for s in (1.0,2.0) if abs(sx-s)<=1e-6 and abs(sy-s)<=1e-6),None)
            checks = [(not stable, 'pose_not_stable'),
                      (ended - started > 250, 'read_too_slow'),
                      (supported_scale is None, 'unsupported_scale'),
                      (supported_scale is not None and abs(abs(row['linear_determinant'])-supported_scale**2)>1e-6, 'nonorthogonal_transform'),
                      (not 0 <= row['fire_angle'] <= 180, 'invalid_angle'),
                      (row['fire_angle'] != angle_row['provisional_fire_angle'], 'angle_source_disagrees'),
                      (math.hypot(x-px, y-py) > 40*(supported_scale or 1), 'muzzle_outside_range')]
            reasons = [name for failed, name in checks if failed]
            self.observation.update(source_rejection_reasons=reasons,
                                    angle_stream_fire_angle=angle_row['provisional_fire_angle'])
            if reasons:
                return "TNTORIGIN-"
            self.observation['source_eligible'] = True
            return (f"TNTORIGIN {row['pid']} {row['uid']} {started} {row['fire_angle']} "
                    f"{int(row['heading_left'])} {px!r} {py!r} {x} {y} {supported_scale:.1f}")
        except (RuntimeError, ValueError, OSError, struct.error, subprocess.TimeoutExpired) as exc:
            self.observation = dict(status='unverified', pid=angle_cache['pid'], uid=angle_cache['uid'],
                                    sample_start_unix_ms=started,
                                    sample_end_unix_ms=time.time_ns() // 1000000,
                                    source_eligible=False, reason=str(exc)[:180])
            self.cache = self.previous = None
            return "TNTORIGIN-"


class BackgroundVerifier:
    """One daemon verification job; hot sampling never waits on it."""
    def __init__(self):
        self.thread = None
        self.results = queue.Queue(maxsize=1)

    def start(self, work):
        if self.thread is not None:
            return False

        def run():
            try:
                result = dict(value=work())
            except Exception as exc:
                result = dict(error=exc)
            self.results.put_nowait(result)

        self.thread = threading.Thread(target=run, daemon=True, name='tnt-scene-verifier')
        self.thread.start()
        return True

    def take(self):
        try:
            result = self.results.get_nowait()
        except queue.Empty:
            return None
        self.thread = None
        return result


def verify_scene(allow_reconnect):
    started = time.monotonic()
    discovered = probe()
    recovered = discovered.get('status') == 'game_not_running' and allow_reconnect
    if recovered and recover_adb_transport():
        discovered = probe()
    cache = discover_self(discovered) if discovered.get('status') == 'coordinates_found' else None
    completed = time.monotonic()
    return dict(discovered=discovered, cache=cache, verified_at=completed,
                duration_ms=(completed - started) * 1000, recovery_attempted=recovered)


def encode_line(cache, row):
    if row.get("status") != "angle_value_found":
        return "TNTANGLE-"
    pid = cache["pid"]
    uid = cache["uid"]
    fire = row["provisional_fire_angle"]
    raw = row["angle_value"]
    rotation = row["qt_rotation_degrees"]
    flipped = row["flipped_x_cached"]
    if (type(pid) is not int or pid <= 0 or type(uid) is not int or uid <= 0
            or type(fire) is not int or not (0 <= fire <= 180)
            or type(raw) is not int or not (0 <= raw <= 180)
            or not math.isfinite(rotation)
            or type(flipped) is not bool):
        return "TNTANGLE-"
    return f"TNTANGLE {pid} {uid} {fire} {raw} {rotation!r} {int(flipped)}"


def origin_status_line(row):
    if row is None:
        return None
    reasons = row.get('source_rejection_reasons', [])
    if row.get('source_eligible'):
        code = 'ready'
    elif 'unsupported_scale' in reasons:
        code = 'scale'
    elif 'angle_source_disagrees' in reasons:
        code = 'angle'
    elif 'pose_not_stable' in reasons:
        code = 'moving'
    else:
        code = 'read'
    scale = max(row.get('linear_scale_xy', [1.0, 1.0]))
    return f"TNTORIGINSTATUS {row['pid']} {row['uid']} {row['sample_start_unix_ms']} {code} {scale:.6f}"


def should_full_check(cache, now, next_full, last_change, last_full,
                      max_full_age=20.0, quiet_seconds=0.8):
    """Schedule background full verification, deferring during active aiming.

    Every quick read still verifies the panel, own sprite/UID and Angle pointer.
    Full ABI/object discovery is deferred while actively aiming, for at most
    max_full_age seconds since the last successful full discovery. The caller
    expires cache authority at that deadline even if a background job stalls.
    """
    if cache is None:
        return True
    if now < next_full:
        return False
    if now - last_full >= max_full_age:
        return True
    return now - last_change >= quiet_seconds


def transient_sampling_error(exc):
    """Retry one quick read after transport/short-read hiccups, not identity failures."""
    return (isinstance(exc, subprocess.TimeoutExpired) or
            isinstance(exc, RuntimeError) and
            (str(exc).startswith("quick angle read incomplete") or
             str(exc).startswith("root command failed")))


def recover_adb_transport():
    """Reconnect adb transport only; never starts, stops or modifies the game."""
    try:
        devices = subprocess.run([ADB, "devices"], capture_output=True,
                                 text=True, timeout=4)
        if devices.returncode != 0:
            return False
        if any(line.strip().startswith(DEV + " ") and "device" in line
               for line in devices.stdout.splitlines()):
            return False  # ADB is working; game might simply be closed.
        res = subprocess.run([ADB, "connect", DEV], capture_output=True,
                             text=True, timeout=5)
        print(f"[ANGLE] adb reconnect: {res.stdout.strip()[:120]}", file=sys.stderr, flush=True)
        return res.returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interval", type=float, default=0.09)
    parser.add_argument("--full-check-seconds", type=float, default=3.0)
    parser.add_argument("--max-samples", type=int, default=0,
                        help="bounded manual diagnostic: 0 means stream until parent exits")
    parser.add_argument("--fire-origin", action="store_true",
                        help="also emit guarded normal/expanded-scale TNTORIGIN rows")
    parser.add_argument("--record-origin", action="store_true",
                        help="emit full timestamped TNTPOSE snapshots for private offline recording")
    args = parser.parse_args()
    if not 0.05 <= args.interval <= 10:
        parser.error("interval must be 0.05..10 seconds")
    if not 1 <= args.full_check_seconds <= 10:
        parser.error("full-check-seconds must be 1..10")
    if not 0 <= args.max_samples <= 1000:
        parser.error("max-samples must be 0..1000")

    parent = os.getppid()
    cache = None
    next_full = 0.0
    last_full = 0.0
    last_change = 0.0
    last_angle_value = None
    transient_failures = 0
    next_adb_reconnect = 0.0
    previous_status = None
    samples = 0
    measurement_start = time.monotonic()
    good_count = 0
    max_read_ms = 0.0
    max_full_ms = 0.0
    max_gap_ms = 0.0
    last_good = None
    last_measured = None
    origin_stream = OriginStream() if args.fire_origin else None
    verifier = BackgroundVerifier()
    scene_status = "verifying_scene"
    try:
        while os.getppid() == parent and (not args.max_samples or samples < args.max_samples):
            started = time.monotonic()
            did_full_check = False
            result = None
            discovered = None
            try:
                now = started
                completed = verifier.take()
                if completed is not None:
                    did_full_check = True
                    next_full = now + args.full_check_seconds
                    if 'error' in completed:
                        raise completed['error']
                    check = completed['value']
                    discovered = check['discovered']
                    cache = check['cache']
                    last_full = check['verified_at'] if cache else 0.0
                    max_full_ms = max(max_full_ms, check['duration_ms'])
                    scene_status = discovered.get('status', 'not_in_battle')
                    if check['recovery_attempted']:
                        next_adb_reconnect = now + 10
                # Do not extend trust in a cache when full verification stalls.
                if cache is not None and now - last_full > 20:
                    cache = None
                    scene_status = 'full_verification_expired'
                if ((cache is not None or now >= next_full)
                        and should_full_check(cache, now, next_full, last_change, last_full)):
                    allowed = now >= next_adb_reconnect
                    verifier.start(lambda allowed=allowed: verify_scene(allowed))
                if cache is None:
                    result = None
                    status = scene_status
                    line = "TNTANGLE-"
                else:
                    result = sample(cache) if did_full_check else quick_sample(cache)
                    status = result["status"]
                    line = encode_line(cache, result)
                    if line.startswith("TNTANGLE "):
                        transient_failures = 0
                        sampled_at = time.monotonic()
                        good_count += 1
                        if last_good is not None:
                            max_gap_ms = max(max_gap_ms, (sampled_at - last_good) * 1000)
                        last_good = sampled_at
                        value = result["angle_value"]
                        if value != last_angle_value:
                            last_change = sampled_at
                            last_angle_value = value
                        last_measured = (value, result["qt_rotation_degrees"],
                                         result["provisional_fire_angle"])
                        elapsed_ms = (sampled_at - started) * 1000
                        max_read_ms = max(max_read_ms, elapsed_ms)
            except Exception as exc:
                # One failed ADB/short read can be transport contention with
                # the independent XY probe. Discard that sample immediately,
                # but allow ONE guarded fast-read retry without an expensive
                # synchronous scene scan. Never retry an identity mismatch.
                retry_fast = (cache is not None and not did_full_check
                              and transient_failures == 0
                              and transient_sampling_error(exc))
                transient_failures = transient_failures + 1 if retry_fast else 0
                if not retry_fast:
                    cache = None
                    last_angle_value = None
                line = "TNTANGLE-"
                status = f"unverified:{str(exc)[:100]}"
            if status != previous_status:
                print(f"[ANGLE] {status}", file=sys.stderr, flush=True)
                previous_status = status
            print(line, flush=True)
            if origin_stream is not None:
                origin_line = origin_stream.poll(cache, result, discovered, did_full_check)
                status_line = origin_status_line(origin_stream.observation)
                if status_line is not None:
                    print(status_line, flush=True)
                elif origin_line == 'TNTORIGIN-':
                    print('TNTORIGINSTATUS-', flush=True)
                if args.record_origin and origin_stream.observation is not None:
                    print('TNTPOSE ' + json.dumps(origin_stream.observation, allow_nan=False,
                                                 separators=(',', ':')), flush=True)
                if origin_line is not None:
                    print(origin_line, flush=True)
            samples += 1
            now = time.monotonic()
            period = now - measurement_start
            if period >= 5:
                print(f"[ANGLE] samples_per_sec={good_count / period:.2f} "
                      f"max_read_ms={max_read_ms:.0f} max_full_ms={max_full_ms:.0f} "
                      f"max_gap_ms={max_gap_ms:.0f} "
                      f"last_raw_rotation_fire={last_measured} "
                      f"(full verification in background; cache trust limit 20s)",
                      file=sys.stderr, flush=True)
                measurement_start = now
                good_count = 0
                max_read_ms = 0.0
                max_full_ms = 0.0
                max_gap_ms = 0.0
            time.sleep(args.interval if cache else max(0.6, args.interval))
    except (BrokenPipeError, KeyboardInterrupt):
        return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
