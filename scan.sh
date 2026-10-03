#!/bin/bash
# 风速内存扫描(推荐直接用 ./w <当前风速>,自动走完 快照→差分→比值 三步):
#   ./scan.sh F1 -8.3          第一回合全扫(f32/f64/int×10)
#   ./scan.sh FN 12.5          后续回合筛上次命中
#   ./scan.sh SEED 129040      搜网络种子在内存中的位置(含上下文)
#   ./scan.sh SNAP             快照全部可写内存(差分扫描基准)
#   ./scan.sh DIFF             找"变了的"地址(首跑对快照,之后逐次缩小)
#   ./scan.sh DIFF same        找"没变的"地址
#   ./scan.sh RATIO 1 2        命中按比值 b/a 过滤(±15%,f32/f64/i32 自动)
#   ./scan.sh EQ -8.3          命中按等于 v 过滤(f32/f64/i32/int×10 自动)
#   ./scan.sh HITS             看前 100 个命中地址的当前值
#   ./scan.sh WATCH f64 7a...  锁定地址后自动读风速进网络记录
#   ./scan.sh WATCH off        停止
#   ./scan.sh RD 7a0c1f0000    读某地址64字节
# 屏幕只看箭头方向: 箭头向左 → 填负数, 向右 → 填正数
ADB=/Applications/BlueStacks.app/Contents/MacOS/hd-adb
DEV=127.0.0.1:5555
CTL=/data/local/tmp/tnt_ctl
OUT=/data/local/tmp/tnt_scan.out

if [ $# -lt 1 ]; then
    echo "用法: ./scan.sh F1|FN|SEED|SNAP|DIFF|RATIO|EQ|HITS|WATCH|RD <参数>"
    exit 1
fi

CMD="$1"
HITS_BEFORE=0
# 破坏性筛选(FN/DIFF/RATIO/EQ)前先备份当前候选;归零时自动回滚
case "$CMD" in
    FN|DIFF|RATIO|EQ)
        HITS_BEFORE=$("$ADB" -s "$DEV" shell "wc -l < /data/local/tmp/tnt_hits" 2>/dev/null | tr -d ' \r\n\t')
        [[ "$HITS_BEFORE" =~ ^[0-9]+$ ]] || HITS_BEFORE=0
        if [ "$HITS_BEFORE" -gt 0 ]; then
            if ! "$ADB" -s "$DEV" shell "cp /data/local/tmp/tnt_hits /data/local/tmp/tnt_hits.bak && chmod 666 /data/local/tmp/tnt_hits.bak" >/dev/null 2>&1; then
                echo "❌ 无法备份候选,本次筛选已取消"
                exit 1
            fi
        fi
        ;;
esac

"$ADB" -s "$DEV" shell "echo \"$*\" > $CTL" || { echo "❌ adb 不通"; exit 1; }
sleep 3
echo "=== $* 结果 ==="
RESULT=$("$ADB" -s "$DEV" shell "cat $OUT" 2>/dev/null)
printf '%s\n' "$RESULT"

ZEROED=0
case "$CMD" in
    FN)
        printf '%s\n' "$RESULT" | grep -qE "# FN .*: 0/[1-9][0-9]* 存活" && ZEROED=1
        ;;
    RATIO)
        printf '%s\n' "$RESULT" | grep -qE "# RATIO .*: 0/[1-9][0-9]* kept" && ZEROED=1
        ;;
    EQ)
        printf '%s\n' "$RESULT" | grep -qE "# EQ .*: 0/[1-9][0-9]* kept" && ZEROED=1
        ;;
    DIFF)
        printf '%s\n' "$RESULT" | grep -qE "# DIFF (changed|same): 0 hits" && ZEROED=1
        ;;
esac
if [ "$ZEROED" = "1" ] && [ "$HITS_BEFORE" -gt 0 ]; then
    if "$ADB" -s "$DEV" shell "cp /data/local/tmp/tnt_hits.bak /data/local/tmp/tnt_hits && chmod 666 /data/local/tmp/tnt_hits" >/dev/null 2>&1; then
        echo "# RESTORED: 已自动恢复上一次的 $HITS_BEFORE 个候选"
    else
        echo "❌ 候选归零且回滚失败"
        exit 1
    fi
fi
