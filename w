#!/bin/bash
# 极简风速差分状态机(最短路径,内部自动 SNAP→DIFF→RATIO;筛成0会自动回退,不会丢候选):
#   ./w 8.3     第一次看到风速 → 自动快照
#   ./w -1.2    风速变了再输 → 自动找变化地址(负号自动取绝对值)
#   ./w 2.5     再次变化 → 自动按比值收敛候选
#   ./w hits    候选少了随时看当前值
#   ./w reset   归零重来
set -u

DIR="$(cd "$(dirname "$0")" && pwd)"
SCAN="$DIR/scan.sh"
STATE=/tmp/tnt-wind-scan-state

usage() {
    echo "用法: ./w <风速数字> | hits | status | reset"
    exit 1
}

[ $# -eq 1 ] || usage

case "$1" in
    reset)
        rm -f "$STATE"
        echo "已重置;看到稳定风速后运行 ./w <数字>"
        exit 0
        ;;
    hits)
        exec "$SCAN" HITS
        ;;
    status)
        if [ -f "$STATE" ]; then
            cat "$STATE"
        else
            echo "未开始"
        fi
        exit 0
        ;;
esac

# 数字参数: -?整数或小数(也接受 + 前缀),其他拒收
if ! [[ "$1" =~ ^[+-]?([0-9]+(\.[0-9]+)?|\.[0-9]+)$ ]]; then
    usage
fi
NEW=$(awk -v v="$1" 'BEGIN{v+=0; print (v < 0 ? -v : v)}')

if [ ! -f "$STATE" ]; then
    OUT=$("$SCAN" SNAP)
    printf '%s\n' "$OUT"
    if printf '%s\n' "$OUT" | grep -q "# SNAP:"; then
        echo "SNAP $NEW" > "$STATE"
        echo "✅ 已快照;风速变化后运行: ./w <新数字>"
    else
        echo "❌ 快照失败;状态未更新,请重试"
    fi
    exit 0
fi

MODE=$(awk '{print $1}' "$STATE")
OLD=$(awk '{print $2}' "$STATE")
if [ -z "$OLD" ] || ! [[ "$OLD" =~ ^[0-9]+(\.[0-9]+)?$ ]]; then
    echo "状态文件异常;运行 ./w reset 后重新开始"
    exit 1
fi

if [ "$MODE" = "SNAP" ]; then
    OUT=$("$SCAN" DIFF)
    printf '%s\n' "$OUT"
    if printf '%s\n' "$OUT" | grep -qE "# DIFF changed: [1-9][0-9]* hits"; then
        echo "RATIO $NEW" > "$STATE"
        echo "✅ 已找到变化地址;风速再次变化后运行: ./w <新数字>"
    else
        echo "❌ 变化地址为 0 或执行失败;状态未更新,可重试或 ./w reset"
    fi
    exit 0
fi

if [ "$MODE" = "RATIO" ]; then
    if awk -v a="$OLD" -v b="$NEW" 'BEGIN{exit !(a==b)}'; then
        echo "风速没变,跳过,等变化后再运行"
        exit 0
    fi
    if awk -v a="$OLD" -v b="$NEW" 'BEGIN{exit !(a==0||b==0)}'; then
        # 有一边是 0 算不出比值,降级用 DIFF 按变化继续收敛
        OUT=$("$SCAN" DIFF)
        printf '%s\n' "$OUT"
        if printf '%s\n' "$OUT" | grep -qE "# DIFF changed: [1-9][0-9]* hits"; then
            echo "RATIO $NEW" > "$STATE"
            echo "✅ 已按变化继续收敛;风速再次变化后运行: ./w <新数字>"
        elif printf '%s\n' "$OUT" | grep -q "# RESTORED:"; then
            echo "本轮归零但已自动回退;确认风速稳定后可重试,或等下一次变化再输入"
        else
            echo "候选归零;运行 ./w reset 后重新开始"
        fi
        exit 0
    fi
    OUT=$("$SCAN" RATIO "$OLD" "$NEW")
    printf '%s\n' "$OUT"
    if printf '%s\n' "$OUT" | grep -qE "# RATIO .*: [1-9][0-9]*/[1-9][0-9]* kept"; then
        echo "RATIO $NEW" > "$STATE"
        KEPT=$(printf '%s\n' "$OUT" | sed -n 's/^# RATIO .*: \([0-9][0-9]*\)\/[0-9][0-9]* kept.*/\1/p' | head -1)
        echo "✅ 候选继续收敛;风速再次变化后运行: ./w <新数字> (候选少时 ./w hits)"
        if [ -n "$KEPT" ] && [ "$KEPT" -le 10 ]; then
            echo "候选已很少,自动显示:"
            "$SCAN" HITS || true
        fi
    elif printf '%s\n' "$OUT" | grep -q "# RESTORED:"; then
        echo "本轮归零但已自动回退;确认风速稳定后可重试,或等下一次变化再输入"
    else
        echo "候选归零;运行 ./w reset 后重新开始"
    fi
    exit 0
fi

echo "状态文件异常;运行 ./w reset 后重新开始"
exit 1
