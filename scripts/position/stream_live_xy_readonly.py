#!/usr/bin/env python3
"""将 BattleSprite QGraphicsItem::pos 的绘制坐标持续输出到 stdout。

输出协议（行模式）：
  TNTXY <pid> <count> <player_id>:<x>:<y> ...
  TNTXY-  # 离开战斗、校验失败或暂时无数据时清除覆盖层

不修改游戏、不注入、不启动/停止模拟器、不写设备文件。
仅当 HUD 显式启用 TNT_LIVE_XY=1 时，才由 HUD 启动。
父进程消失则退出，stdout 管道关闭也会自动退出。
"""
import argparse
import math
import re
import struct
import os
import sys
import time

from probe_live_xy_readonly import Memory, canonical, probe


def encode_line(result):
    if result.get("status") != "coordinates_found":
        return "TNTXY-"
    pid = result.get("pid")
    players = result.get("players", [])
    if not isinstance(pid, int) or not isinstance(players, list) or not players:
        return "TNTXY-"
    rows = []
    ids = set()
    for player in players:
        if not player.get("matches_battle_role_index"):
            return "TNTXY-"
        # 只取实际图形坐标，不能再回退到 BattleRole 逻辑 Y。
        pid0 = player.get("player_id")
        gx, gy = player.get("graphics_x"), player.get("graphics_y")
        if type(pid0) is not int or not all(type(v) in (int, float) for v in (gx, gy)):
            return "TNTXY-"
        if not (0 < pid0 < (1 << 32) and -100000 <= gx <= 100000 and -100000 <= gy <= 100000):
            return "TNTXY-"
        x, y = int(round(gx)), int(round(gy))
        if pid0 in ids:
            return "TNTXY-"
        ids.add(pid0)
        rows.append(f"{pid0}:{x}:{y}")
    return f"TNTXY {pid} {len(rows)} " + " ".join(rows)


def quick_sample(cache):
    """Full probe cached role pointers; batch read Qt private pos + identity guards."""
    if cache.get("status") != "coordinates_found":
        raise RuntimeError("no valid cache")
    pid = cache["pid"]
    rows = cache["players"]
    if not (1 <= len(rows) <= 8):
        raise RuntimeError("unexpected role count")
    panel = int(cache["panel_va"], 16)
    expect_vptr = int(cache["panel_expected_vptr"], 16)
    ranges = [(panel, 8)]
    for row in rows:
        sprite = int(row["sprite_va"], 16)
        role = int(row["role_va"], 16)
        private = int(row["graphics_private_va"], 16)
        ranges.extend([(sprite + 0x18, 8), (sprite + 0xc8, 8),
                       (role + 0x68, 4), (private + 0x58, 16)])
    # All addresses are integers from a previously verified QHash/vtable walk.
    cmd = " ; ".join(
        f"dd if=/proc/{pid}/mem bs=1 skip={canonical(a)} count={length} "
        f"2>/dev/null | od -v -An -tx1" for a, length in ranges)
    # adb shell itself parses semicolons before su -c unless the inner command
    # is quoted; without these quotes only the first read runs as root.
    raw = Memory(pid).root_cmd("'" + cmd + "'", timeout=8)
    tokens = re.findall(r"(?<![0-9a-fA-F])[0-9a-fA-F]{2}(?![0-9a-fA-F])", raw)
    data = bytes(int(token, 16) for token in tokens)
    if len(data) != sum(length for _, length in ranges):
        raise RuntimeError(f"quick read incomplete {len(data)}/{sum(length for _, length in ranges)} bytes")
    pieces = []
    at = 0
    for _, length in ranges:
        pieces.append(data[at:at+length])
        at += length
    if canonical(struct.unpack("<Q", pieces[0])[0]) != expect_vptr:
        raise RuntimeError("battle panel changed")
    current = []
    for i, row in enumerate(rows):
        idx = 1 + i*4
        private = canonical(struct.unpack("<Q", pieces[idx])[0])
        role = canonical(struct.unpack("<Q", pieces[idx+1])[0])
        role_id = struct.unpack("<I", pieces[idx+2])[0]
        gx, gy = struct.unpack("<dd", pieces[idx+3])
        if (private != int(row["graphics_private_va"],16)
                or role != int(row["role_va"],16)
                or role_id != row["player_id"]):
            raise RuntimeError("graphics role identity changed")
        if not (math.isfinite(gx) and math.isfinite(gy)
                and -100000 <= gx <= 100000 and -100000 <= gy <= 100000):
            raise RuntimeError("invalid Qt graphics XY")
        current.append({"player_id":role_id,"graphics_x":gx,"graphics_y":gy,
                        "matches_battle_role_index":True})
    return {"status":"coordinates_found", "pid":pid, "players":current}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--interval", type=float, default=0.06)
    p.add_argument("--full-check-seconds", type=float, default=3.0)
    args = p.parse_args()
    if not (0.04 <= args.interval <= 30):
        p.error("interval must be between 0.04 and 30 seconds")
    if not (1 <= args.full_check_seconds <= 10):
        p.error("full-check-seconds must be between 1 and 10 seconds")
    parent = os.getppid()
    previous_status = None
    previous_line = None
    last_position_report = 0.0
    cache = None
    next_full_check = 0.0
    try:
        while os.getppid() == parent:
            try:
                if cache is None or time.monotonic() >= next_full_check:
                    result = probe()
                    cache = result if result.get("status") == "coordinates_found" else None
                    next_full_check = time.monotonic() + args.full_check_seconds
                else:
                    result = quick_sample(cache)
                line = encode_line(result)
                status = result.get("status", "unknown")
            except Exception as exc:
                cache = None
                line = "TNTXY-"
                status = f"error:{str(exc)[:100]}"
            if status != previous_status:
                print(f"[XY] {status}", file=sys.stderr, flush=True)
                previous_status = status
            now = time.monotonic()
            # 诊断 stderr 不参与 stdout 坐标协议；防止看不到数据变化。
            if line.startswith("TNTXY ") and now - last_position_report >= 3.0 and (
                line != previous_line or now - last_position_report >= 12.0
            ):
                print(f"[XY] positions {line}", file=sys.stderr, flush=True)
                last_position_report = now
            previous_line = line
            print(line, flush=True)
            time.sleep(args.interval if cache is not None else max(args.interval, 0.6))
    except (BrokenPipeError, KeyboardInterrupt):
        return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
