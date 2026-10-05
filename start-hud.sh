#!/bin/bash
# TNT HUD 一键启动:wrap 注入检查 + adb 隧道 + HUD
# 用法:  ./start-hud.sh [--restart]
#   --restart  游戏在跑但需要重载(.so 更新/未注入)时,明确授权 force-stop+重启;
#              默认绝不停止正在运行的游戏,只打印延迟重载说明并非零退出
# 自检模式(不拉起 HUD):  TNT_NO_LAUNCH=1 ./start-hud.sh
set -u

RESTART=0
for arg in "$@"; do
    case "$arg" in
        --restart) RESTART=1 ;;
        *)
            echo "用法: $0 [--restart]" >&2
            exit 2
            ;;
    esac
done

ADB=/Applications/BlueStacks.app/Contents/MacOS/hd-adb
DEV=127.0.0.1:5555
PKG=com.grapefruit.game.tnt
WRAP=/data/local/tmp/sniff-wrap.sh
PORT=19001
DIR="$(cd "$(dirname "$0")" && pwd)"

ok()  { echo "✅ $*"; }
bad() { echo "❌ $*"; }
info(){ echo "…  $*"; }

echo "===== TNT HUD 启动检查 ====="

# ---------- 1. ADB 连通 ----------
if ! "$ADB" -s "$DEV" shell echo ok >/dev/null 2>&1; then
    info "ADB 未连接,尝试 adb connect..."
    "$ADB" connect "$DEV" >/dev/null 2>&1
    sleep 2
fi
if "$ADB" -s "$DEV" shell echo ok >/dev/null 2>&1; then
    ok "ADB 已连接 $DEV"
else
    bad "ADB 连不上 $DEV"
    echo "   → 检查 BlueStacks 设置→高级→Android Debug Bridge 是否开启,然后重跑本脚本"
    exit 1
fi

# ---------- 2. wrap 属性 ----------
CUR_WRAP=$("$ADB" -s "$DEV" shell "getprop wrap.$PKG" 2>/dev/null | tr -d '\r')
if [ "$CUR_WRAP" = "$WRAP" ]; then
    ok "wrap 属性已就位"
else
    "$ADB" -s "$DEV" shell "setprop wrap.$PKG $WRAP" >/dev/null 2>&1
    CUR_WRAP=$("$ADB" -s "$DEV" shell "getprop wrap.$PKG" 2>/dev/null | tr -d '\r')
    if [ "$CUR_WRAP" = "$WRAP" ]; then
        ok "wrap 已重新设置"
    else
        bad "setprop wrap 失败 — Android 侧可能不允许,检查 ADB 状态"
        exit 1
    fi
fi

# ---------- 2.5 wrap 脚本 + .so 同步(带 TNT_RAND=1,激活 RNG 钩子) ----------
LOCAL_WRAP="/Users/wx/Downloads/tnt-apk-analysis/sniff-wrap.sh"
LOCAL_SO="/Users/wx/Downloads/tnt-apk-analysis/libtntsniff.so"
SO_UPDATED=0
if [ -f "$LOCAL_WRAP" ]; then
    "$ADB" -s "$DEV" push "$LOCAL_WRAP" "$WRAP" >/dev/null 2>&1 \
        && "$ADB" -s "$DEV" shell "chmod 755 $WRAP" >/dev/null 2>&1 \
        && ok "wrap 脚本已同步(TNT_RAND=1)"
fi
if [ -f "$LOCAL_SO" ]; then
    LOCAL_MD5=$(md5 -q "$LOCAL_SO" 2>/dev/null)
    # 读不到远端 md5 时视为不同,照常 push
    REMOTE_MD5=$("$ADB" -s "$DEV" shell "md5sum /data/local/tmp/libtntsniff.so 2>/dev/null" | awk '{print $1}' | tr -d '\r')
    if [ "$LOCAL_MD5" = "$REMOTE_MD5" ]; then
        ok "libtntsniff.so 已是最新版,跳过 push"
    else
        # 不能直接覆盖一个可能被进程映射中的 .so:先推 .new 再设备内 mv -f 原子替换
        if ! "$ADB" -s "$DEV" push "$LOCAL_SO" /data/local/tmp/libtntsniff.so.new >/dev/null 2>&1; then
            bad "libtntsniff.so push 失败"
            exit 1
        fi
        if ! "$ADB" -s "$DEV" shell "chmod 755 /data/local/tmp/libtntsniff.so.new && mv -f /data/local/tmp/libtntsniff.so.new /data/local/tmp/libtntsniff.so" >/dev/null 2>&1; then
            bad "libtntsniff.so 设备内替换失败"
            exit 1
        fi
        SO_UPDATED=1
        ok "libtntsniff.so 已同步(新版)"
    fi
fi
# 内存扫描文件:app 只能写已存在的全局可写文件,预建
"$ADB" -s "$DEV" shell 'touch /data/local/tmp/tnt_ctl /data/local/tmp/tnt_scan.out /data/local/tmp/tnt_hits /data/local/tmp/tnt_hits.bak /data/local/tmp/tnt_snap.dat /data/local/tmp/tnt_snap.idx; chmod 666 /data/local/tmp/tnt_ctl /data/local/tmp/tnt_scan.out /data/local/tmp/tnt_hits /data/local/tmp/tnt_hits.bak /data/local/tmp/tnt_snap.dat /data/local/tmp/tnt_snap.idx' >/dev/null 2>&1 \
    && ok "memscan 文件已就位"

# ---------- 3. 游戏进程 + .so 注入 ----------
game_pid() { "$ADB" -s "$DEV" shell "pidof $PKG" 2>/dev/null | tr -d '\r' | awk '{print $1}'; }
# rand=1 + fmt=1 + memscan=7 才表示进程里跑的是当前 Qt hook 观察版;旧标记视为旧版,要重启吃新 .so
so_loaded() { "$ADB" -s "$DEV" shell "logcat -d -s TntSniff" 2>/dev/null | grep -q "loaded pid=$1 rand=1 fmt=1 memscan=7"; }

# 明确授权(--restart)才允许的重启路径:force-stop + monkey 拉起并验证注入
restart_game() {
    "$ADB" -s "$DEV" shell "am force-stop $PKG; sleep 1; monkey -p $PKG -c android.intent.category.LAUNCHER 1" >/dev/null 2>&1
    sleep 7
    PID=$(game_pid)
    if [ -n "$PID" ] && so_loaded "$PID"; then
        ok "游戏已重启 pid=$PID,.so 注入成功"
    else
        bad ".so 仍未加载 — 手工看: $ADB -s $DEV shell 'logcat -d -s TntSniff'"
        exit 1
    fi
}

# 默认路径:绝不碰正在运行的游戏,打印延迟重载说明后非零退出
defer_reload() {
    bad "游戏运行中 pid=$1 但需要重载($2),默认不停止正在运行的游戏"
    echo "   → 请在游戏里手动退出到桌面后重跑本脚本,或明确授权:"
    echo "      ./start-hud.sh --restart   # 会中断当前对局，可能影响积分；确认安全后才使用"
    exit 1
}

PID=$(game_pid)
if [ -n "$PID" ] && [ "$SO_UPDATED" = "1" ]; then
    if [ "$RESTART" = "1" ]; then
        info ".so 已更新，按明确授权重启游戏加载新版；会中断当前对局..."
        restart_game
    else
        defer_reload "$PID" "新版 libtntsniff.so 待加载"
    fi
elif [ -n "$PID" ] && so_loaded "$PID"; then
    ok "游戏运行中 pid=$PID,libtntsniff 已注入"
elif [ -n "$PID" ]; then
    if [ "$RESTART" = "1" ]; then
        info "游戏在跑(pid=$PID)但 .so 未注入 — 按明确授权重启，会中断当前对局..."
        restart_game
    else
        defer_reload "$PID" "libtntsniff.so 未注入"
    fi
else
    info "游戏未运行,正在启动..."
    "$ADB" -s "$DEV" shell "monkey -p $PKG -c android.intent.category.LAUNCHER 1" >/dev/null 2>&1
    sleep 7
    PID=$(game_pid)
    if [ -n "$PID" ] && so_loaded "$PID"; then
        ok "游戏已启动 pid=$PID,.so 注入成功"
    else
        bad "游戏没起来或 .so 未注入 — 请先在 BlueStacks 里手动打开游戏再重跑"
        exit 1
    fi
fi

# ---------- 4. adb reverse 隧道 ----------
"$ADB" -s "$DEV" reverse "tcp:$PORT" "tcp:$PORT" >/dev/null 2>&1
if "$ADB" -s "$DEV" reverse --list 2>/dev/null | grep -q "tcp:$PORT"; then
    ok "reverse 转发 tcp:$PORT 就绪"
else
    bad "adb reverse 设置失败 — 隧道不可用"
    exit 1
fi

# ---------- 5. 端口占用 ----------
if lsof -nP -tiTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1; then
    HOLDER=$(lsof -nP -iTCP:"$PORT" -sTCP:LISTEN | tail -1 | awk '{print $1" pid="$2}')
    info "端口 $PORT 被 $HOLDER 占用,正在结束它..."
    lsof -nP -tiTCP:"$PORT" -sTCP:LISTEN | xargs kill 2>/dev/null
    sleep 1
fi
ok "端口 $PORT 空闲"

# ---------- 6. HUD ----------
if [ ! -x "$DIR/target/debug/live_gui" ]; then
    info "live_gui 未编译,先 cargo build..."
    (cd "$DIR" && cargo build --bin live_gui) || { bad "编译失败"; exit 1; }
fi

echo ""
echo "=========================================="
echo " 全部就绪。HUD 马上启动:"
echo "   ① 回车或输入2选模式"
echo "   ② 框选左上角小地图(按住拖一个矩形)"
echo "   ③ 框选左下角圆盘角度数字"
echo " 窗口弹出后进一局战斗即可。别关 HUD 窗口!"
echo "   本终端会持续打 📡 NET 日志和 lag 延迟值"
echo "   文本日志: /tmp/tnt-hud-live.log"
echo "   原始网络记录: /tmp/tnt-tunnel-records.bin"
echo "=========================================="
[ "${TNT_NO_LAUNCH:-0}" = "1" ] && { ok "自检完成(未启动 HUD)"; exit 0; }
# 新录制前归档上一次的日志/记录(时间戳+PID 后缀),不删除任何历史数据
ARCHIVE_SUFFIX="$(date +%Y%m%d-%H%M%S).$$"
for f in /tmp/tnt-hud-live.log /tmp/tnt-tunnel-records.bin; do
    if [ -f "$f" ]; then
        if ! mv "$f" "$f.$ARCHIVE_SUFFIX"; then
            bad "无法归档 ${f}，已停止启动以保护旧记录"
            exit 1
        fi
        info "已归档 $f → $f.$ARCHIVE_SUFFIX"
    fi
done
cd "$DIR" && ./target/debug/live_gui 2>&1 | tee /tmp/tnt-hud-live.log
