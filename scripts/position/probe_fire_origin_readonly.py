#!/usr/bin/env python3
"""Read-only reconstruction of the client's firePos() from local Qt transforms.

Research output only: no HUD override, function calls in the game, injection,
memory writes, device-file writes, game launch, or restart. Cached scene matrices
are not used to generate the position. Every sample checks the same live object
chain; Qt graphics-transform lists and projective transforms are unsupported.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import struct
import subprocess
import time

from probe_live_xy_readonly import Memory, canonical, probe
from probe_live_angle_readonly import discover_self, decode_angle_triplet, fire_angle_candidates

GAME_FIRE_POS = 0x842430
GAME_FIRE_POS_CODE = bytes.fromhex(
    "08 ec 40 f9 00 40 00 91 13 41 00 91 81 55 ee 97 "
    "e1 03 00 aa e2 23 00 91 e0 03 13 aa ff ff 00 a9 6c 8e ee 97")
QT_GUARDS = {
    0x47AF14: bytes.fromhex("08 04 40 f9 00 41 40 f9 c0 03 5f d6"),
    0x47EBBC: bytes.fromhex("09 04 40 f9 29 49 40 f9 49 01 00 b4"),
    0x47EBF4: bytes.fromhex("08 04 40 f9 08 49 40 f9 68 00 00 b4 00 31 40 fd"),
    0x47EDB0: bytes.fromhex("08 04 40 f9 08 49 40 f9 68 00 00 b4 00 2d 40 fd"),
    0x47F3F4: bytes.fromhex("08 04 40 f9 08 49 40 f9 68 00 00 b4 00 85 46 6d"),
    0x490D60: bytes.fromhex("a0 86 46 6d e0 83 01 91 a6 a1 f3 97"),
}


def identity():
    return ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))


def multiply(a, b):
    return tuple(tuple(sum(a[i][k] * b[k][j] for k in range(3))
                       for j in range(3)) for i in range(3))


def translate(x, y):
    return ((1.0, 0.0, x), (0.0, 1.0, y), (0.0, 0.0, 1.0))


def transform_data(data):
    if len(data) != 0x88:
        raise ValueError("Qt transform data size mismatch")
    # Qt's actual nine-double memory order is affine-six, then m13/m23/m33,
    # verified by its constructor/map machine instructions, not row-major nine.
    m11, m12, m21, m22, dx, dy, m13, m23, m33 = struct.unpack_from("<9d", data)
    scale, rotation, ox, oy = struct.unpack_from("<4d", data, 0x58)
    if not all(math.isfinite(v) for v in (m11, m12, m21, m22, dx, dy, m13, m23, m33,
                                        scale, rotation, ox, oy)):
        raise ValueError("nonfinite Qt transform")
    if abs(m13) > 1e-12 or abs(m23) > 1e-12 or abs(m33 - 1) > 1e-12:
        raise ValueError("projective transform unsupported")
    if not (0.01 <= abs(scale) <= 10 and abs(rotation) <= 3600
            and abs(ox) <= 100000 and abs(oy) <= 100000):
        raise ValueError("Qt transform outside supported range")
    base = ((m11, m21, dx), (m12, m22, dy), (0.0, 0.0, 1.0))
    if data[0x80] not in (0, 1):
        raise ValueError("invalid Qt only-transform flag")
    only = bool(data[0x80])
    if not only:
        radians = math.radians(rotation)
        co, si = math.cos(radians), math.sin(radians)
        rotation_scale = ((scale * co, -scale * si, 0.0),
                          (scale * si, scale * co, 0.0), (0.0, 0.0, 1.0))
        base = multiply(base, multiply(translate(ox, oy),
                                      multiply(rotation_scale, translate(-ox, -oy))))
    return base, {"scale": scale, "rotation": rotation, "origin": [ox, oy],
                  "only_base_transform": only}


def batch(mem, ranges):
    command = " ; ".join(
        f"dd if=/proc/{mem.pid}/mem bs=1 skip={canonical(a)} count={n} "
        f"2>/dev/null | od -v -An -tx1" for a, n in ranges)
    text = mem.root_cmd("'" + command + "'", timeout=8)
    values = re.findall(r"(?<![0-9a-fA-F])[0-9a-fA-F]{2}(?![0-9a-fA-F])", text)
    data = bytes(int(v, 16) for v in values)
    if len(data) != sum(n for _, n in ranges):
        raise RuntimeError("incomplete origin snapshot")
    parts = []
    offset = 0
    for _, n in ranges:
        parts.append(data[offset:offset + n])
        offset += n
    return parts


def discover(snapshot, angle_cache=None):
    cache = dict(angle_cache) if angle_cache is not None else discover_self(snapshot)
    mem = Memory(cache["pid"])
    bias = cache["self_vptr"] - 0xA647F8
    if mem.read(bias + GAME_FIRE_POS, len(GAME_FIRE_POS_CODE)) != GAME_FIRE_POS_CODE:
        raise RuntimeError("firePos implementation changed")
    qt_bases = []
    for line in mem.maps().splitlines():
        match = re.match(r"^([0-9a-f]+)-[0-9a-f]+\s+r-xp\s+00000000\s+\S+\s+\S+\s+.*libQt5Widgets_arm64-v8a\.so$", line)
        if match:
            qt_bases.append(int(match.group(1), 16))
    if len(qt_bases) != 1:
        raise RuntimeError("Qt mapping ambiguous")
    for va, code in QT_GUARDS.items():
        if mem.read(qt_bases[0] + va, len(code)) != code:
            raise RuntimeError(f"Qt transform ABI mismatch at {va:#x}")
    sprite_private = canonical(mem.u64(cache["sprite"] + 0x18))
    target = canonical(mem.u64(sprite_private + 0x80))
    if target == 0:
        raise RuntimeError("self has no coordinate parent")
    node = cache["angle"] + 0x10
    chain = []
    visited = set()
    while node != target:
        if not node or node in visited or len(chain) >= 8:
            raise RuntimeError("origin ancestry invalid or too deep")
        visited.add(node)
        private = canonical(mem.u64(node + 8))
        if not private:
            raise RuntimeError("missing item private")
        data = mem.read(private + 0x58, 0x40)
        parent = canonical(struct.unpack_from("<Q", data, 0x28)[0])
        transform = canonical(struct.unpack_from("<Q", data, 0x38)[0])
        row = {"item": node, "private": private, "parent": parent, "transform": transform,
               "list": None}
        if transform:
            local = mem.read(transform, 0x88)
            row["list"] = canonical(struct.unpack_from("<Q", local, 0x78)[0])
            if not row["list"]:
                raise RuntimeError("invalid Qt transform list")
        chain.append(row)
        node = parent
    if cache["sprite"] + 0x10 not in visited:
        raise RuntimeError("Angle ancestry does not include verified self")
    cache.update(chain=chain, target=target, sprite_private=sprite_private)
    return cache


def sample(cache, with_aim=False):
    mem = Memory(cache["pid"])
    ranges = [(cache["panel"], 8), (cache["sprite"], 8),
              (cache["sprite"] + 0xC8, 8), (cache["role"] + 0x68, 4),
              (cache["sprite"] + 0x1D8, 8), (cache["angle"], 8)]
    for row in cache["chain"]:
        ranges.extend([(row["item"] + 8, 8), (row["private"] + 0x58, 0x40)])
        if row["transform"]:
            ranges.extend([(row["transform"], 0x88), (row["list"] + 8, 8)])
    if with_aim:
        ranges.append((cache["angle"] + 0xA0, 12))
    parts = batch(mem, ranges)
    pointer = lambda data: canonical(struct.unpack("<Q", data)[0])
    if (pointer(parts[0]) != cache["panel_expected_vptr"]
            or pointer(parts[1]) != cache["self_vptr"]
            or pointer(parts[2]) != cache["role"]
            or struct.unpack("<i", parts[3])[0] != cache["uid"]
            or pointer(parts[4]) != cache["angle"]
            or pointer(parts[5]) != cache["angle_vptr"]):
        raise RuntimeError("live origin identity changed")
    index = 6
    combined = identity()
    nodes = []
    player_xy = None
    for row in cache["chain"]:
        private, data = parts[index:index + 2]
        index += 2
        if (pointer(private) != row["private"]
                or canonical(struct.unpack_from("<Q", data, 0x28)[0]) != row["parent"]
                or canonical(struct.unpack_from("<Q", data, 0x38)[0]) != row["transform"]):
            raise RuntimeError("origin object/parent/transform changed")
        x, y = struct.unpack_from("<dd", data)
        if not all(math.isfinite(v) and abs(v) <= 100000 for v in (x, y)):
            raise RuntimeError("origin item position invalid")
        local, description = identity(), {"scale": 1.0, "rotation": 0.0}
        if row["transform"]:
            transform, bounds = parts[index:index + 2]
            index += 2
            if canonical(struct.unpack_from("<Q", transform, 0x78)[0]) != row["list"]:
                raise RuntimeError("Qt transform list changed")
            begin, end = struct.unpack("<ii", bounds)
            if begin != end or not (0 <= begin <= 4096):
                raise RuntimeError("nonempty Qt graphics-transform list unsupported")
            local, description = transform_data(transform)
        combined = multiply(multiply(translate(x, y), local), combined)
        nodes.append({"item": row["item"], "parent": row["parent"], "pos": [x, y],
                      "local_transform": local, **description})
        if row["item"] == cache["sprite"] + 0x10:
            player_xy = [x, y]
    x, y = combined[0][2], combined[1][2]
    scale_x = math.hypot(combined[0][0], combined[1][0])
    scale_y = math.hypot(combined[0][1], combined[1][1])
    if not all(math.isfinite(v) and abs(v) <= 1000000 for v in (x, y)):
        raise RuntimeError("computed origin invalid")
    result = {"status": "fire_origin_candidate", "pid": cache["pid"], "uid": cache["uid"],
            "fire_pos_float": [x, y], "fire_pos_rounded": [math.floor(x + .5), math.floor(y + .5)],
            "player_graphics_xy": player_xy, "linear_scale_xy": [scale_x, scale_y],
            "linear_determinant": combined[0][0] * combined[1][1] - combined[0][1] * combined[1][0],
            "nodes": nodes, "snapshot_digest": hashlib.sha256(b"".join(parts)).hexdigest(),
            "source": "verified firePos Angle.mapToItem(self.parentItem,zero), local Qt transform chain",
            "research_only": True}
    if with_aim:
        result["snapshot_ranges"] = [dict(address=a, size=n, hex=b.hex())
                                     for (a, n), b in zip(ranges, parts)]
        value, _, _ = decode_angle_triplet(parts[-1])
        ground = next((n for n in nodes if n["item"] == cache["rotatable"] + 0x10), None)
        if ground is None:
            raise RuntimeError("rotation item missing from origin ancestry")
        left = combined[0][0] < 0
        result.update(heading_left=left, control_value=value,
                      fire_angle=fire_angle_candidates(value, ground["rotation"])[
                          "flipped" if left else "unflipped"])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--watch-seconds", type=float, default=0)
    parser.add_argument("--interval", type=float, default=.25)
    args = parser.parse_args()
    if not (0 <= args.watch_seconds <= 120 and .15 <= args.interval <= 10):
        parser.error("watch-seconds=0..120, interval=.15..10")
    deadline = time.monotonic() + args.watch_seconds
    cache = None
    while True:
        started = time.time_ns() // 1000000
        try:
            if cache is None:
                cache = discover(probe())
            result = sample(cache)
            result.update(sample_start_unix_ms=started, sample_end_unix_ms=time.time_ns() // 1000000)
        except (RuntimeError, ValueError, OSError, struct.error, subprocess.TimeoutExpired) as exc:
            result = {"status": "unverified", "reason": str(exc)[:180]}
            cache = None
        print(json.dumps(result, ensure_ascii=False), flush=True)
        if not args.watch_seconds or time.monotonic() >= deadline:
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
