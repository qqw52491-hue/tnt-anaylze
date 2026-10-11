# TNT 弹道 HUD

Rust+OpenCV 弹弹堂辅助 HUD。`./start-hud.sh` 一键启动隧道与 HUD。
不要自动停止正在运行的游戏;重启必须由用户明确授权。
网络层: 43.248.190.45:8888 WebSocket 二进制帧 = zlib(Protobuf)。解析在 `src/proto.rs`+`src/net.rs`(Tracker/隧道记录格式见注释)。

## 验证

```bash
cargo test --lib
cargo build --bin live_gui --bin net_live --bin dump_wind_fields
```

## 风速解码(客户端运行时代码已恢复,2026-10)

BattleNotify type-8 的 Action: f1=round, f2=repeated order, f3=speed,
f4=raw wind。客户端 BattleCommandAction::init 使用有符号 32 位 XOR:

```
wind10 = i32(raw) ^ i32(round) ^ i32(speed) ^ XOR(每个 i32(order))
世界风速 = wind10 / 10.0
```

不先取 abs(raw),不截低8位,不只用 order[0]。完整列表随玩家退出而变化,
因此旧的 C_battle 并非通用局内常数;负风的旧模型也可能产生 0.2 等偏差。
这些偏差不能一律归咎于语音识别。

已有32条历史 Action 的计算已冻结在
`/Users/wx/Downloads/tnt-apk-analysis/wind_decode_validation.json`。
其中3局各有1条手动报告可独立比较大小,7.1/9.0/7.8均完全匹配。
SETTLED/applied 行不是独立真值。完整实时屏幕对照仍待正常游戏验证。
HUD 使用解出的世界风向结合目标方位/F翻转转换为顺逆风。
语音/手动输入保留为覆盖,不再用于推测局密钥。

## 配对采样(自动)

`live_gui` 运行中: 每条 SEED 写 `wind_pairs.csv`(applied列=自动预测值);
语音/回车确认风速自动记 VOICE/MANUAL 行。正常玩即可攒数据。
`analyze_wind.py` 是旧模型研究工具,其推测的密钥不能覆盖客户端算法。

```bash
python3 analyze_wind.py            # 配对 raw↔wind, 列出每回合 key
```

## 禁用项

- 不要视觉吸附(用户明确否决)
- OCR 流光风字不可靠;`./w`/`scan.sh` 内存扫描已否决(全是假候选)
- vsnprintf/Qt setNum hook 无效(UI 走自定义渲染,只见 IP 格式化)

## 语音

`TNT_VOICE_TURBO=1 ./start-hud.sh` 常驻模型 ~1.3s;默认档 SenseVoice+Whisper 双保险。

## 弹道研究录制（原始数据，不自动拟合）

新版 HUD 在项目 `recordings/<unix-ms>-<pid>-<index>/` 自动保存：
- `raw.bin`：原隧道格式全部记录，含游戏端时间、进程和连接信息，可离线重放。
- `events.jsonl`：实际 FireReq、Fire 通知、风和玩家位置事件；保留完整力度精度、方向、连接、事件序号和原始记录偏移。
- `manifest.json`：格式版本和限制。

`TNT_RECORD_DIR` 可改录制根目录；`TNT_TRACE_PATH` 可指定原始文件，但已存在时拒绝覆盖并提示录制失败。
数据可能含游戏账号、聊天等隐私信息，目录已加入 gitignore，不要上传公开仓库。
请求和通知不自动配对；Fire 坐标含义、缺省字段、炮弹路径和落点未核实，`fit_ready=false`。
`player_transport` 是玩家飞行路径，不是炮弹轨迹；HUD 上下文是缓存快照，不是独立真值。
录制功能本身不修改物理模型或主动启动/重启游戏。文件实时写入，但不是逐条 fsync 的断电保护。当前默认物理模型见下节。

## 弹道模型更新

默认采用 recorded-frozen-v1；旧版完整保留在 src/physics/legacy.rs。
TNT_PHYSICS=legacy 可回退。说明见 PHYSICS_MODELS.md。
新 manifest 记录模型名；运行中的旧 HUD 不热更新，不自动重启游戏。
