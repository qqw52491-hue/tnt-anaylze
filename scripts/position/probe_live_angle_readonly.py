#!/usr/bin/env python3
"""Read-only research probe for the current local aiming Angle value.

Does not move the mouse, fire, inject, write game memory, restart the emulator,
or change HUD state. No continuous monitoring unless --watch-seconds is given.

Research provenance (2026-10-17 reconstructed libgame / Qt5Widgets):
  BattleSpritePlayerSelf::BattleSpritePlayerSelf -> self+0x1d8 = Angle*
  Angle::setValue(int) -> Angle+0xa0 = i32 current value
  Angle::setRange(...) -> Angle+0xa4 = upper, +0xa8 = lower
  BattleSpritePlayerSelf::fireAngle() uses Angle+0xa0 AND Qt rotation/flip.
Thus value != guaranteed final world-space firing angle. Compare with UI/FireReq.
"""
import argparse
import json
import math
import re
import struct
import sys
import time

from probe_live_xy_readonly import Memory, PANEL_BATTLE_VPTR_VA, canonical, probe

SELF_VTABLE_VA = 0xA647E8 + 0x10
ANGLE_VTABLE_VA = 0xA64F98 + 0x10
SET_VALUE_WRITE_VA = 0x84A33C
SET_VALUE_WRITE_CODE = bytes.fromhex("74 a2 00 b9")  # str w20,[x19,#0xa0]
FIRE_ANGLE_LOAD_VA = 0x8423D0
FIRE_ANGLE_LOAD_CODE = bytes.fromhex("02 a1 40 bd")  # ldr s2,[x8,#0xa0]


def fire_angle_candidates(value, rotation):
    """The two branches in BattleSpritePlayerSelf::fireAngle() (direction unresolved)."""
    if not (math.isfinite(rotation) and -360 <= rotation <= 360):
        raise ValueError("invalid Qt rotation")
    return {
        "unflipped": math.trunc(value - rotation),
        "flipped": math.trunc(180.0 - rotation - value),
    }


def decode_angle_triplet(buf):
    if len(buf) != 12:
        raise ValueError("angle read is not 12 bytes")
    value, upper, lower = struct.unpack("<iii", buf)
    if not (-360 <= lower <= upper <= 360 and lower <= value <= upper):
        raise ValueError(f"angle triplet is out of range: {value=}, {lower=}, {upper=}")
    return value, lower, upper


def discover_self(snapshot):
    if snapshot.get("status") != "coordinates_found":
        raise RuntimeError(snapshot.get("status", "no_verified_battle"))
    pid = snapshot["pid"]
    mem = Memory(pid)
    bias = int(snapshot["panel_expected_vptr"], 16) - PANEL_BATTLE_VPTR_VA
    if bias <= 0 or bias % 0x1000:
        raise RuntimeError("invalid image mapping bias")
    if mem.read(bias + SET_VALUE_WRITE_VA, 4) != SET_VALUE_WRITE_CODE:
        raise RuntimeError("Angle::setValue ABI changed; refusing to read")
    if mem.read(bias + FIRE_ANGLE_LOAD_VA, 4) != FIRE_ANGLE_LOAD_CODE:
        raise RuntimeError("fireAngle ABI changed; refusing to read")
    found = []
    for player in snapshot["players"]:
        sprite = int(player["sprite_va"], 16)
        if canonical(mem.u64(sprite)) != bias + SELF_VTABLE_VA:
            continue
        angle = canonical(mem.u64(sprite + 0x1D8))
        if angle == 0 or canonical(mem.u64(angle)) != bias + ANGLE_VTABLE_VA:
            raise RuntimeError("self Angle object identity invalid")
        # fireAngle() queries the rotation of the QGraphicsItem at
        # self+0x80 (BattleSpriteRotatable* with QGraphicsItem at +0x10).
        rotatable = canonical(mem.u64(sprite + 0x80))
        if rotatable == 0:
            raise RuntimeError("self rotation item not available")
        private = canonical(mem.u64(rotatable + 0x18))
        if private == 0:
            raise RuntimeError("rotation QGraphicsItemPrivate unavailable")
        transform = canonical(mem.u64(private + 0x90))
        # GraphicsHelper::isFlippedX() calls sceneTransform() for self+0x78
        # and compares the first QTransform double (m11) to zero. Qt's
        # sceneTransform() copies cached matrix from QGraphicsItemPrivate+0xa0.
        flip_item_owner = canonical(mem.u64(sprite + 0x78))
        if flip_item_owner == 0:
            raise RuntimeError("mirror QGraphicsItem unavailable")
        flip_private = canonical(mem.u64(flip_item_owner + 0x18))
        if flip_private == 0:
            raise RuntimeError("mirror QGraphicsItemPrivate unavailable")
        found.append({
            "pid": pid,
            "uid": player["player_id"],
            "panel": int(snapshot["panel_va"], 16),
            "panel_expected_vptr": int(snapshot["panel_expected_vptr"], 16),
            "sprite": sprite,
            "role": int(player["role_va"], 16),
            "angle": angle,
            "self_vptr": bias + SELF_VTABLE_VA,
            "angle_vptr": bias + ANGLE_VTABLE_VA,
            "rotatable": rotatable,
            "rotation_private": private,
            "rotation_transform": transform,
            "flip_item_owner": flip_item_owner,
            "flip_private": flip_private,
        })
    if len(found) != 1:
        raise RuntimeError(f"expected exactly one verified self sprite, found {len(found)}")
    return found[0]


def sample(cache):
    """One batched read; every sample rechecks identities before publishing."""
    fields = [
        (cache["sprite"], 8),
        (cache["sprite"] + 0xC8, 8),
        (cache["sprite"] + 0x1D8, 8),
        (cache["role"] + 0x68, 4),
        (cache["angle"], 8),
        (cache["angle"] + 0xA0, 12),
        (cache["sprite"] + 0x80, 8),
        (cache["rotatable"] + 0x18, 8),
        (cache["rotation_private"] + 0x90, 8),
        (cache["sprite"] + 0x78, 8),
        (cache["flip_item_owner"] + 0x18, 8),
        (cache["flip_private"] + 0xA0, 8),
    ]
    if cache["rotation_transform"]:
        fields.append((cache["rotation_transform"] + 0x60, 8))
    # Guard against battle panel being destroyed/replaced between full probes.
    fields.append((cache["panel"], 8))
    cmd = " ; ".join(
        f"dd if=/proc/{cache['pid']}/mem bs=1 skip={canonical(addr)} count={n} "
        f"2>/dev/null | od -v -An -tx1" for addr, n in fields
    )
    raw = Memory(cache["pid"]).root_cmd("'" + cmd + "'", timeout=8)
    tokens = re.findall(r"(?<![0-9a-fA-F])[0-9a-fA-F]{2}(?![0-9a-fA-F])", raw)
    buf = bytes(int(t, 16) for t in tokens)
    if len(buf) != sum(n for _, n in fields):
        raise RuntimeError(f"incomplete memory snapshot {len(buf)} bytes")
    parts = []
    index = 0
    for _, n in fields:
        parts.append(buf[index:index+n])
        index += n
    sprite_vptr = canonical(struct.unpack("<Q", parts[0])[0])
    linked_role = canonical(struct.unpack("<Q", parts[1])[0])
    linked_angle = canonical(struct.unpack("<Q", parts[2])[0])
    role_id = struct.unpack("<i", parts[3])[0]
    angle_vptr = canonical(struct.unpack("<Q", parts[4])[0])
    if (sprite_vptr != cache["self_vptr"]
            or linked_role != cache["role"]
            or linked_angle != cache["angle"]
            or role_id != cache["uid"]
            or angle_vptr != cache["angle_vptr"]):
        raise RuntimeError("sprite/role/Angle identity changed; discard old pointers")
    if canonical(struct.unpack("<Q", parts[-1])[0]) != cache["panel_expected_vptr"]:
        raise RuntimeError("battle panel replaced; discard cached angle")
    value, lower, upper = decode_angle_triplet(parts[5])
    linked_rotatable = canonical(struct.unpack("<Q", parts[6])[0])
    linked_private = canonical(struct.unpack("<Q", parts[7])[0])
    linked_transform = canonical(struct.unpack("<Q", parts[8])[0])
    if (linked_rotatable != cache["rotatable"]
            or linked_private != cache["rotation_private"]
            or linked_transform != cache["rotation_transform"]):
        raise RuntimeError("Qt rotation object changed; fresh discovery required")
    flip_item_owner = canonical(struct.unpack("<Q", parts[9])[0])
    flip_private = canonical(struct.unpack("<Q", parts[10])[0])
    scene_m11 = struct.unpack("<d", parts[11])[0]
    if (flip_item_owner != cache["flip_item_owner"]
            or flip_private != cache["flip_private"]):
        raise RuntimeError("Qt mirror item changed; fresh discovery required")
    if not math.isfinite(scene_m11) or abs(scene_m11) > 100:
        raise RuntimeError("invalid cached scene transform")
    rotation = struct.unpack("<d", parts[12])[0] if cache["rotation_transform"] else 0.0
    candidates = fire_angle_candidates(value, rotation)
    inferred_flipped = scene_m11 < 0
    provisional = candidates["flipped" if inferred_flipped else "unflipped"]
    return {
        "status": "angle_value_found",
        "angle_value": value,
        "qt_rotation_degrees": rotation,
        "fire_angle_candidates": candidates,
        "scene_m11_cached": round(scene_m11, 6),
        "flipped_x_cached": inferred_flipped,
        "provisional_fire_angle": provisional,
        "provisional_only": "Qt sceneTransform cache may be stale; validate with UI/FireReq",
        "allowed_range": [lower, upper],
        "source": "BattleSpritePlayerSelf+0x1d8 -> Angle+0xa0",
        "meaning": "internal aiming control value; fireAngle also applies Qt rotation/flip",
        "verified_live_identity": True,
    }


def quick_sample(cache):
    """Lean guarded read between full ABI/vtable checks (in stream only).

    Eight short memory ranges instead of the full pointer-chain read. A live
    panel, unchanged self vptr, linked angle pointer, and role UID are checked
    on EVERY sample. Full discovery and complete sample re-run periodically.
    """
    fields = [
        (cache["panel"], 8),
        (cache["sprite"], 8),
        (cache["sprite"] + 0x1D8, 8),
        (cache["role"] + 0x68, 4),
        (cache["angle"] + 0xA0, 12),
        (cache["rotation_private"] + 0x90, 8),
        (cache["flip_private"] + 0xA0, 8),
    ]
    if cache["rotation_transform"]:
        fields.append((cache["rotation_transform"] + 0x60, 8))
    command = " ; ".join(
        f"dd if=/proc/{cache['pid']}/mem bs=1 skip={canonical(addr)} count={n} "
        f"2>/dev/null | od -v -An -tx1" for addr, n in fields
    )
    raw = Memory(cache["pid"]).root_cmd("'" + command + "'", timeout=8)
    tokens = re.findall(r"(?<![0-9a-fA-F])[0-9a-fA-F]{2}(?![0-9a-fA-F])", raw)
    buf = bytes(int(t, 16) for t in tokens)
    if len(buf) != sum(n for _, n in fields):
        raise RuntimeError(f"quick angle read incomplete {len(buf)} bytes")
    parts = []
    pos = 0
    for _, length in fields:
        parts.append(buf[pos:pos+length])
        pos += length

    panel_vptr = canonical(struct.unpack("<Q", parts[0])[0])
    sprite_vptr = canonical(struct.unpack("<Q", parts[1])[0])
    linked_angle = canonical(struct.unpack("<Q", parts[2])[0])
    role_id = struct.unpack("<i", parts[3])[0]
    transform = canonical(struct.unpack("<Q", parts[5])[0])
    if (panel_vptr != cache["panel_expected_vptr"]
            or sprite_vptr != cache["self_vptr"]
            or linked_angle != cache["angle"]
            or role_id != cache["uid"]
            or transform != cache["rotation_transform"]):
        raise RuntimeError("angle scene identity/transform changed; re-discover")

    value, lower, upper = decode_angle_triplet(parts[4])
    scene_m11 = struct.unpack("<d", parts[6])[0]
    rotation = (struct.unpack("<d", parts[7])[0]
                if cache["rotation_transform"] else 0.0)
    if not math.isfinite(scene_m11) or abs(scene_m11) > 100:
        raise RuntimeError("invalid cached mirror scene matrix")
    candidates = fire_angle_candidates(value, rotation)
    flipped = scene_m11 < 0
    return {
        "status": "angle_value_found",
        "angle_value": value,
        "qt_rotation_degrees": rotation,
        "fire_angle_candidates": candidates,
        "scene_m11_cached": round(scene_m11, 6),
        "flipped_x_cached": flipped,
        "provisional_fire_angle": candidates["flipped" if flipped else "unflipped"],
        "provisional_only": "Qt sceneTransform cache may lag scene updates",
        "allowed_range": [lower, upper],
        "source": "verified BattleSpritePlayerSelf -> Angle+0xa0 (fast sample)",
        "verified_live_identity": True,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--watch-seconds", type=float, default=0,
                        help="optional bounded read-only observations, max 30s")
    parser.add_argument("--interval", type=float, default=0.25)
    args = parser.parse_args()
    if not (0 <= args.watch_seconds <= 30 and 0.1 <= args.interval <= 10):
        parser.error("watch-seconds must be 0..30 and interval 0.1..10")
    deadline = time.monotonic() + args.watch_seconds
    cache = None
    next_check = 0.0
    last_key = None
    try:
        while True:
            now = time.monotonic()
            if cache is None or now >= next_check:
                state = probe()
                if state.get("status") != "coordinates_found":
                    print(json.dumps({"status": state.get("status", "unavailable")},
                                     ensure_ascii=False), flush=True)
                    break
                cache = discover_self(state)
                next_check = time.monotonic() + 3
            result = sample(cache)
            key = (result["angle_value"], result["qt_rotation_degrees"],
                   result["flipped_x_cached"], result["scene_m11_cached"],
                   tuple(result["allowed_range"]))
            if args.watch_seconds == 0 or key != last_key:
                print(json.dumps(result, ensure_ascii=False), flush=True)
            last_key = key
            if args.watch_seconds == 0 or time.monotonic() >= deadline:
                break
            time.sleep(args.interval)
    except (ValueError, RuntimeError, OSError) as exc:
        print(json.dumps({"status": "unverified", "reason": str(exc)},
                         ensure_ascii=False), flush=True)
        return 1
    except KeyboardInterrupt:
        return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
