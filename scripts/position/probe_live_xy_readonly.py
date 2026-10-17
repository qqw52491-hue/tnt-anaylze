#!/usr/bin/env python3
"""只读探测新版 TNT 当前对局内玩家 XY。
读取 BlueStacks 现存进程的 /proc/<pid>/maps + /proc/<pid>/mem；
不注入/修改内存、不设置系统属性、不重启 APK、不写远端文件。

用法: python3 probe_live_xy_readonly.py [--watch-seconds 0] [--interval 2]
默认仅执行一次；watch 只在本次显式命令运行期间轮询。
"""
import argparse
import json
import math
import re
import struct
import subprocess
import time

ADB = "/Applications/BlueStacks.app/Contents/MacOS/hd-adb"
DEV = "127.0.0.1:5555"
PACKAGE = "com.grapefruit.game.tnt"
SU = "/system/xbin/bstk/su"
RX_BYTES = 0xA0D000
SET_POS_VA = 0x7E5D00
SET_POS_CODE = bytes.fromhex("28 00 40 f9 29 00 80 52 09 2c 01 39 08 44 00 f9 ac ec ef 17")
# Guard the exact Qt ABI used by graphics coordinates: qGraphicsItem::pos
# {ldr x8,[x0,#8]; ldp d0,d1,[x8,#0x58]; ret}. If Qt changes,
# decline to read graphics positions and let HUD fall back to network.
QT_POS_VA = 0x47E24C
QT_POS_CODE = bytes.fromhex("08 04 40 f9 00 85 45 6d c0 03 5f d6")
SINGLETON_VA = 0xA8FF70
LAYER_UI_VPTR_VA = 0xA19EB0
PANEL_BATTLE_VPTR_VA = 0xA61E80


def adb(*args, timeout=8):
    return subprocess.run([ADB, "-s", DEV, *args], capture_output=True,
                          text=True, timeout=timeout)


def as_uint64(buf, off=0):
    return struct.unpack_from("<Q", buf, off)[0]


def canonical(value):
    # Android 13 heap objects in this emulator carry a nonzero TBI tag.
    return value & 0x00FFFFFFFFFFFFFF


class Memory:
    def __init__(self, pid):
        self.pid = int(pid)

    def root_cmd(self, cmd, timeout=8):
        out = adb("shell", SU, "-c", cmd, timeout=timeout)
        if out.returncode:
            raise RuntimeError(f"root command failed: {out.stderr.strip()[:120]}")
        return out.stdout

    def maps(self):
        return self.root_cmd(f"cat /proc/{self.pid}/maps")

    def read(self, address, length):
        if length < 1 or length > 8192:
            raise ValueError("memory request length out of bounds")
        address = canonical(int(address))
        # Process is only read, no ptrace / writes. Hex transport avoids shell CR/LF changes.
        output = self.root_cmd(
            f"dd if=/proc/{self.pid}/mem bs=1 skip={address} count={length} "
            f"2>/dev/null | od -v -An -tx1", timeout=10)
        tokens = re.findall(r"(?<![0-9a-fA-F])[0-9a-fA-F]{2}(?![0-9a-fA-F])", output)
        data = bytes(int(v, 16) for v in tokens)
        if len(data) != length:
            raise RuntimeError(f"memory short read at 0x{address:x}: {len(data)}/{length}")
        return data

    def u64(self, address):
        return as_uint64(self.read(address, 8))

    def qhash(self, address, max_nodes=128):
        address = canonical(address)
        hdr = self.read(address, 0x28)
        buckets = canonical(as_uint64(hdr, 8))
        count = struct.unpack_from("<I", hdr, 0x14)[0]
        n = struct.unpack_from("<I", hdr, 0x20)[0]
        seed = struct.unpack_from("<I", hdr, 0x24)[0]
        if not 1 <= n <= 4096 or count > max_nodes:
            raise RuntimeError(f"unexpected QHash metadata at 0x{address:x}: {count=}, {n=}")
        table = self.read(buckets, 8*n)
        result = {}
        seen = set()
        for i in range(n):
            node = canonical(as_uint64(table, i*8))
            for _ in range(max_nodes):
                if node == address:
                    break
                if node in seen:
                    break
                seen.add(node)
                item = self.read(node, 24)
                nxt, h, key, value = struct.unpack("<QIIQ", item)
                if (h != ((key ^ seed) & 0xffffffff) or h % n != i):
                    raise RuntimeError(f"invalid QHash node key={key}, bucket={i}")
                result[key] = value
                node = canonical(nxt)
            else:
                raise RuntimeError("QHash traversal exceeded node cap")
        if len(result) != count:
            raise RuntimeError(f"QHash item count mismatch {len(result)}/{count}")
        return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--watch-seconds", type=int, default=0)
    parser.add_argument("--interval", type=float, default=2)
    args = parser.parse_args()
    if args.watch_seconds < 0 or args.watch_seconds > 180:
        parser.error("watch-seconds must be in [0,180]")
    stop = time.monotonic() + args.watch_seconds
    last_state = None
    while True:
        result = probe()
        state = result.get("status")
        if args.watch_seconds == 0 or state != last_state or state == "coordinates_found":
            print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
        if args.watch_seconds == 0 or time.monotonic() >= stop:
            break
        if state == "coordinates_found":
            # Capturing one successful reading is enough for this read-only experiment.
            break
        last_state = state
        time.sleep(max(0.5, args.interval))


def probe():
    procs = adb("shell", "pidof", PACKAGE).stdout.strip().split()
    if not procs:
        return {"status": "game_not_running"}
    for pid in procs:
        mem = Memory(pid)
        try:
            maps = mem.maps()
        except (OSError, RuntimeError, subprocess.TimeoutExpired):
            continue
        if "libgame_arm64-v8a.so" not in maps:
            continue
        candidates = []
        lines = maps.splitlines()
        qt_bases = []
        for line in lines:
            qt = re.match(
                r"^([0-9a-f]+)-[0-9a-f]+\s+r-xp\s+00000000\s+\S+\s+\S+\s+.*libQt5Widgets_arm64-v8a\.so$",
                line,
            )
            if qt:
                qt_bases.append(int(qt.group(1), 16))
        if not qt_bases:
            return {"status": "qt_widgets_mapping_not_found", "pid": int(pid)}
        for line in lines:
            m = re.match(r"^([0-9a-f]+)-([0-9a-f]+)\s+r-xp\s+\S+\s+\S+\s+\S+\s+\[anon:\.bss\]$", line)
            if m and int(m.group(2),16)-int(m.group(1),16)==RX_BYTES:
                candidates.append(int(m.group(1),16))
        for bias in candidates:
            try:
                if mem.read(bias+SET_POS_VA,len(SET_POS_CODE)) != SET_POS_CODE:
                    continue
                if not any(
                    mem.read(qt_base+QT_POS_VA,len(QT_POS_CODE)) == QT_POS_CODE
                    for qt_base in qt_bases
                ):
                    return {"status": "qt_graphics_abi_mismatch", "pid": int(pid)}
                layer = canonical(mem.u64(bias+SINGLETON_VA))
                if layer == 0:
                    return {"status":"layer_ui_not_created","pid":int(pid),"image_bias":hex(bias)}
                actual_vptr = canonical(mem.u64(layer))
                if actual_vptr != bias+LAYER_UI_VPTR_VA:
                    return {"status":"layer_ui_type_mismatch","pid":int(pid)}
                panels = mem.qhash(mem.u64(layer+0x60))
                if 12 not in panels:
                    return {"status":"not_in_battle","pid":int(pid),
                            "panel_count":len(panels),"battle_panel_present":False,
                            "verified_live_private_elf":True}
                panel = canonical(panels[12])
                pvptr = canonical(mem.u64(panel))
                if pvptr != bias+PANEL_BATTLE_VPTR_VA:
                    return {"status":"battle_panel_type_mismatch","pid":int(pid),
                            "vtable_expected":hex(bias+PANEL_BATTLE_VPTR_VA),
                            "vtable_actual":hex(pvptr)}
                sprite_hash = mem.qhash(mem.u64(panel+0x90))
                battle = canonical(mem.u64(panel+0x148))
                logical_roles = mem.qhash(mem.u64(battle+0xb0)) if battle else {}
                rows=[]
                for player_id, raw_sprite in sorted(sprite_hash.items()):
                    sprite=canonical(raw_sprite)
                    role=canonical(mem.u64(sprite+0xc8))
                    if not role:
                        continue
                    rid=struct.unpack("<i",mem.read(role+0x68,4))[0]
                    xy=mem.read(role+0x88,8)
                    x,y=struct.unpack("<ii",xy)
                    if rid!=player_id or not (-100000 <= x <= 100000 and -100000 <= y <= 100000):
                        raise RuntimeError(f"role ID / coordinates validation failed for key {player_id}")
                    other=canonical(logical_roles.get(player_id,0))
                    # 真实走路显示坐标由 MovingLeft/Right::onValueChanged 写入
                    # QGraphicsItem::setPos(sprite+0x10)；Qt5Widgets ARM64
                    # QGraphicsItem::pos(): d_ptr=[item+0x8]; QPointF=[d_ptr+0x58].
                    graphics_item_private=canonical(mem.u64(sprite+0x18))
                    if not graphics_item_private:
                        raise RuntimeError(f"missing QGraphicsItemPrivate for role {player_id}")
                    gx,gy=struct.unpack("<dd",mem.read(graphics_item_private+0x58,16))
                    if not (math.isfinite(gx) and math.isfinite(gy)
                            and -100000 <= gx <= 100000 and -100000 <= gy <= 100000):
                        raise RuntimeError(f"invalid rendered QGraphicsItem coordinates for role {player_id}")
                    rows.append({"player_id":player_id,"x":x,"y":y,
                                 "graphics_x":gx,"graphics_y":gy,
                                 "graphics_private_va":hex(graphics_item_private),
                                 "sprite_va":hex(sprite),"role_va":hex(role),
                                 "matches_battle_role_index":other==role})
                if not rows:
                    return {"status":"battle_panel_present_no_players",
                            "pid":int(pid),"role_map_count":len(sprite_hash)}
                return {"status":"coordinates_found","pid":int(pid),
                        "source":"in-process BattleRole logical QPoint + QGraphicsItem::pos",
                        "role_count":len(rows),"players":rows,
                        "panel_va":hex(panel),"panel_expected_vptr":hex(bias+PANEL_BATTLE_VPTR_VA),
                        "note":"graphics_x/graphics_y 是绘制对象的当前位置；x/y 是逻辑坐标。Qt pos 属于父项局部坐标，已有 BattleSpriteRole::onRolePosChanged 将逻辑坐标写入该位置。"}
            except (RuntimeError, OSError, struct.error, subprocess.TimeoutExpired) as exc:
                return {"status":"probe_error","pid":int(pid),
                        "error":str(exc)[:180]}
    return {"status":"private_image_not_mapped"}

if __name__ == "__main__":
    main()
