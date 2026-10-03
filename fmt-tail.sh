#!/bin/bash
# 查看 FMT+ 观察记录(只读 HUD 隧道原始记录;需先跑 HUD 打一局产生记录)
# 用法: ./fmt-tail.sh [行数,默认100]
BIN=/tmp/tnt-tunnel-records.bin
if [ ! -f "$BIN" ]; then
    echo "❌ 没有 $BIN — 先启动 HUD 打一局产生隧道记录再试"
    exit 1
fi
strings "$BIN" | grep -E 'FMT\+|QNUM\+|QTEXT\+' | tail -n "${1:-100}"
