#!/usr/bin/env python3
"""Read-only, optional TNT live angle stream for HUD.

Output protocol:
  TNTANGLE <pid> <self-uid> <fire-angle-0..180> <control-value> <qt-rotation> <mirror-0-or-1>
  TNTANGLE-                 # stale/unverified/not in battle

The candidate angle follows the reconstructed client's fireAngle() math,
but uses a Qt scene-transform cache that may be stale. Never write memory,
control the game, restart processes or perform an action on the player's behalf.
The caller must opt in and retain OCR fallback.
"""
import argparse
import math
import os
import subprocess
import sys
import time

from probe_live_xy_readonly import ADB, DEV, probe
from probe_live_angle_readonly import discover_self, quick_sample, sample


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


def should_full_check(cache, now, next_full, last_change, last_full,
                      max_full_age=20.0, quiet_seconds=0.8):
    """Avoid blocking the hot sampling loop mid-hold when angle is changing.

    Every quick read still verifies the panel, own sprite/UID and Angle pointer.
    Full ABI/object discovery is deferred while actively aiming, for at most
    max_full_age seconds since the last successful full discovery.
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
    try:
        while os.getppid() == parent and (not args.max_samples or samples < args.max_samples):
            started = time.monotonic()
            did_full_check = False
            try:
                now = started
                if should_full_check(cache, now, next_full, last_change, last_full):
                    discovered = probe()
                    if discovered.get("status") == "game_not_running" and now >= next_adb_reconnect:
                        # game_not_running also occurs when adb transport disappears.
                        # Probe transport and reconnect only when adb has no device.
                        next_adb_reconnect = now + 10
                        if recover_adb_transport():
                            discovered = probe()
                    cache = discover_self(discovered) if discovered.get("status") == "coordinates_found" else None
                    last_full = time.monotonic() if cache else 0.0
                    next_full = time.monotonic() + args.full_check_seconds
                    did_full_check = cache is not None
                if cache is None:
                    result = None
                    status = discovered.get("status", "not_in_battle")
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
                        if did_full_check:
                            max_full_ms = max(max_full_ms, elapsed_ms)
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
            samples += 1
            now = time.monotonic()
            period = now - measurement_start
            if period >= 5:
                print(f"[ANGLE] samples_per_sec={good_count / period:.2f} "
                      f"max_read_ms={max_read_ms:.0f} max_full_ms={max_full_ms:.0f} "
                      f"max_gap_ms={max_gap_ms:.0f} "
                      f"last_raw_rotation_fire={last_measured} "
                      f"(full scan deferred while aiming; hard limit 20s)",
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
