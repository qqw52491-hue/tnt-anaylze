# Qt SSL 抓包库

`tnt_sniff.c` 从原 `/Users/wx/Downloads/tnt-apk-analysis/tnt_sniff.c`
迁入维护，保留原 socket 采集和旧观察功能。TLS 增量在 `tnt_tls.h`。
`start-hud.sh` 自动编译并优先使用 `target/android/libtntsniff.so`。

2026-10-16 更新后的录制连接到 `43.248.190.45:8889`，ServerHello
选择 `supported_versions=0x0304`（TLS 1.3）。旧录制使用 8888 的明文
WebSocket。新 APK 的 QtNetwork 导入 `_ZN8QLibrary7resolveEPKc`；
[Qt 源码](https://code.qt.io/cgit/qt/qtbase.git/tree/src/network/ssl/qsslsocket_openssl_symbols.cpp?h=5.15)
通过这一入口查找 OpenSSL 函数，因此仅导出同名 SSL_read/SSL_write 不够。

新版在 QLibrary 实例的 resolve 入口替换成功解析的 SSL_read、SSL_write、
SSL_read_ex、SSL_write_ex、SSL_free 和 SSL_clear 指针。其余查找原样转发。
SSL 调用只执行一次，保持返回值、errno 和实际成功字节数；失败/重试不记录。
不调用 OpenSSL 错误队列接口，不修改密钥、证书验证、协议或物理模型。

最初 tls=1 版只使用 RTLD_NEXT，Android 局部加载的 QtCore 不在其可见范围，
造成所有 SSL 符号查找失败，不能用于实际游戏。tls=2 增加普通 NOLOAD 查找，
仍无法跨过游戏 Java classloader namespace。tls=3 先查已有 QtCore，必要时
通过 Bionic 的可选 `__loader_dlopen` 入口传入原 Qt 调用地址，在调用方的
namespace 中 NOLOAD 查找原函数；禁止重复加载另一份 QtCore。
该 Bionic 入口动态查找，已在当前 Android 13/API 33 设备验证可用；不是
跨 Android 版本的公共 NDK API。变更后必须验证实际游戏日志，不能以替身
测试通过就宣称实机兼容。

当前设备已通过用户授权重新加载 tls=3，实际游戏日志成功解析 SSL_read、
SSL_write、SSL_free、SSL_clear，用户确认登录成功。真实战斗的风速/血量
已取得事件（血量字段大小等仍需屏幕验证）。若崩溃后留下孤儿进程，普通 force-stop 可能失效；
启动脚本仅在显式 --restart 路径尝试用本设备的 `/system/xbin/bstk/su`
清理本游戏残留进程，不会在默认路径停止游戏。

已取得真实游戏明文，新版 zlib 内包头改为 `[u32BE body_len][u16BE name_len]`，
旧版是 `[u32LE body_len][u32LE name_len]`。Rust 同时支持两者并校验长度；
body_len 都不含首个四字节长度本身。TLS 通了不代表应用层格式保持不变。

新 BE16 Action 的 f4 直接携带有符号 wind10，不再使用旧 XOR。用户独立报告
的四个自己回合风速 10.6/7.1/13.2/13.8，分别匹配同期 raw=-106/71/132/138
的绝对值/10，包括 speed=70、order 变化。此为实测配对证据，并非新版客户端
反汇编证明；风向符号仍需与屏幕箭头独立对照。旧 LE32 包继续原 XOR。

Qt 使用 memory BIO，不能依靠 SSL_get_fd 得到 socket。每个 SSL 对象的
生命周期分配一个高位为 1 的合成流 ID；释放/成功 clear 后关闭该流。
它不是 OS fd，也不证明某个 IP、账号或战斗归属。并发上限 128 个对象，
超限时跳过采集，不干扰实际 SSL 操作。隧道缓冲区仍可能丢记录。

隧道维持原 21 字节头，新增 kind=11（SSL 成功发送明文）、12（解密明文）。
kind=2/3 仍保存 socket 原始字节，可能是密文。Rust Tracker 为两种来源
分别维护 WS 解码器。录制 manifest 和事件中的 raw_record_kind 记录来源。
旧读取工具需要支持新增 kind 才能离线解析 SSL 明文。

## 构建和验证

```bash
bash native/build-sniffer.sh
cc -std=gnu11 -Wall -Wextra -Werror -pthread native/test_tls.c -o /tmp/tnt-test-tls
/tmp/tnt-test-tls
cargo test --lib
cargo build --bin live_gui --bin net_live --bin dump_wind_fields
bash native/test-android.sh
./target/debug/net_live replay recordings/<session>/raw.bin
```

默认 NDK 为 `~/Library/Android/sdk/ndk/27.0.12077973`，可通过
`ANDROID_NDK_HOME` 覆盖。Android 集成测试在独立临时进程中运行，使用
游戏随包 OpenSSL、memory BIO、临时自签名证书和人工构造的风速消息，
验证真实 TLS 1.3 → Qt ABI 动态拦截 → 隧道 → WS/zlib/Protobuf → 风速 7.1。
QLibrary 调用方使用 ABI 测试替身，置于独立的 Android SHARED namespace 内，
并以 RTLD_LOCAL 加载依赖，复现实际游戏的两层可见性边界。
旧 tls=1/tls=2 二进制在该回归测试中失败，tls=3 通过。
该测试不是实际 QtNetwork 或游戏对局测试。
不连接游戏服务器，不读取账号数据，不启动或停止游戏。
测试使用独立 tcp:19002 转发，结束后移除测试转发和唯一的设备临时目录。
`target/tls-integration` 保留人工测试记录和测试证书，勿作为真实对局证据。
`TNT_TUNNEL_PORT` 只供隔离测试覆盖默认隧道端口 19001。

运行中的旧进程不会自动加载新版。必须由用户明确授权重启游戏，或用户
自行退出后再启动。看到 `tls=3 resolved SSL_read/SSL_write` 表示入口已接入；
HUD 提示“已收到 SSL 明文”表示取得明文，之后仍需真实对局验证风速和其他字段。
未获得明文前不能宣称新游戏协议保持不变。
