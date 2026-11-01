# TNT 弹道 HUD

Rust+OpenCV 弹弹堂辅助 HUD。`./start-hud.sh` 一键启动隧道与 HUD。
不要自动停止正在运行的游戏;重启必须由用户明确授权。
网络层: 旧版 43.248.190.45:8888 明文 WebSocket；2026-10-16 新版改用同 IP 的 8889，实测 TLS 1.3。
WebSocket 二进制帧仍含 zlib；解压后的旧包头为 u32LE 长度+u32LE 名长，
新版实测为 u32BE 长度+u16BE 名长。两种均已支持。
抓包源码在 `native/`，Qt SSL 明文 kind=11/12、合成 SSL 流 ID（非 OS fd），见 `native/README.md`。
解析在 `src/proto.rs`+`src/net.rs`(Tracker/隧道记录格式见注释)。

## 验证

```bash
cargo test --lib
cargo build --bin live_gui --bin net_live --bin dump_wind_fields
```

## 风速解码(客户端运行时代码已恢复,2026-10)

2026-10-16 新版 BE16 包：Action f4 已直接传有符号 wind10，不再套旧 XOR。
用户独立报告的 4 个自己回合风速 10.6/7.1/13.2/13.8，与同期 f4
-106/71/132/138 的绝对值/10 全部匹配（含 speed=70 和 order 变化）。
这些报告验证大小；新版世界风向符号仍待屏幕箭头独立对照。
以下 XOR 仅用于旧 LE32 包，历史 32 条冻结样本保持原算法不变。

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
- `origin_samples.jsonl`：同次运行的炮口探针原始行和接收时间。TNTPOSE 含采样起止时间、
  人物/炮口位置、角度/朝向、完整本地变换和原始读取段；TNTORIGIN/- 保留源侧通过/清除。
  probe_line 是转义字符串，TNTPOSE 后再解析JSON；保留无效及双尺度样本，不当成真值或HUD已应用。

`TNT_RECORD_DIR` 可改录制根目录；`TNT_TRACE_PATH` 可指定原始文件，但已存在时拒绝覆盖并提示录制失败。
数据可能含游戏账号、聊天等隐私信息，目录已加入 gitignore，不要上传公开仓库。
炮口与发射需离线按同进程/身份及时间先后配对，只用请求前新鲜样本，不拿事后首点选择预测。
请求和通知不自动配对；Fire 坐标含义、缺省字段、炮弹路径和落点未核实，`fit_ready=false`。
`player_transport` 是玩家飞行路径，不是炮弹轨迹；HUD 上下文是缓存快照，不是独立真值。
录制功能本身不修改物理模型或主动启动/重启游戏。文件实时写入，但不是逐条 fsync 的断电保护。当前默认物理模型见下节。

## HUD 布局与目标选择（2026-11-01）

画布830x680。地图在左侧392x398视口内居中等比例显示，右侧390；主题在 src/hud_theme.rs。
推荐力度置顶，角度/风速卡片、8人名单分区；标尺和手动点位在地图下方。画面EXIT已移除。
地图有边距：图像、角色/轨迹、地图点选与手动标尺逆变换必须共用 map_display_rect，不能漏掉偏移。
名单按UID排序，列表/地图/默认目标/距离/风向共用 selected_target_id，不得各自遍历HashMap。
列表按下/松开复核UID及对局，拖出或角色集合变化不得误选。保留显式手动标点和标尺操作。
`live_gui --preview-layout <新PNG路径> [recent|expanded|scale|waiting]` 仅离线示例，不启动窗口、采样或游戏。
说明见 HUD_LAYOUT_20261101.md；不得为查看版式自动关闭/重启正在运行的游戏。
macOS生命周期在 src/hud_window.rs，只为本进程同名窗口启用原生关闭按钮；imshow前检查关闭，
最小化/隐藏不退出。原生关闭和Esc只退出HUD，不操作游戏。独立关闭测试见文档。

## 弹道模型更新

默认采用 recorded-calibrated-v2（2026-10-22 常规轨迹校准）；前一版完整保留在
src/physics/recorded_frozen.rs，TNT_PHYSICS=frozen-v1 可回退；更早版本保留在
src/physics/legacy.rs，TNT_PHYSICS=legacy 可回退。说明见 PHYSICS_MODELS.md。
常规模型使用223条拟合，另140/20条按日期分组验证。2026-11-01接入独立的放大炮弹
recorded-expanded-v1（src/physics/expanded.rs），使用历史34条训练参数，不以新验证数据重拟合。
发射前Qt均匀尺度1/2自动选择普通/放大模型，炮口、力度、角度求解、绘图与缓存键必须一致。
81条历史及22条后续轨迹复核，另两炮发射前尺度2及炮口均匹配实际首点，绝对RMSE为0.536/0.648。
player_transport不属于此模型，不能混用验证。详情见EXPANDED_CALIBRATION_20261101.md。
历史首点对齐误差与发射前炮口绝对误差需区分，均不代表实时命中率。
离线工具在 examples/export_ballistics.rs 和 scripts/physics/；不自动修改运行时参数。
新 manifest 记录模型名；运行中的旧 HUD 不热更新，不自动重启游戏。

网络点位的距离/高差在小地图绘制取整前计算（2026-11-01）；显示像素只用于绘图。
手动标点仍按选定标尺计算。炮口按已恢复 firePos() 的本地 Qt 变换链计算，不猜固定偏移、
不用场景缓存矩阵。同一姿态3炮的发射前预测与真实首点完全一致，跨地图/姿态验证仍有限。
角度流可附带 TNTORIGIN：均匀尺度1或2、两次稳定、600ms内，身份/人物位置/实际发射角度
匹配才作为实时建议应用。失败须区分体型变化中、角度/位置未匹配或读数更新，不得笼统写“校准中”。
短缺帧且输入完全未变、最近源状态ready时，允许原采样年龄<=1.2s的明确标注“上次建议·更新中”参考值；
这不延长600ms实时炮口时效，不得当新数据用于验证或拟合。状态变化/过旧撤下，禁止切回人物起点交替输出。
手动角度/手动点、OCR、legacy、显式关闭炮口时维持人物起点估算；TNT_FIRE_ORIGIN=0 可关闭。
完整场景复验在单个后台线程执行，快采样不中断；完整缓存授权20秒、炮口样本600ms均不得延长。
源侧两次稳定只比较实际几何，不把控制值或无关缓存字节变化当作炮口移动。见 HUD_STABILITY_20261101.md。
普通模型参数未改；新增独立放大炮弹模型。旧HUD不热更新，不能自动重启。
见EXPANDED_CALIBRATION_20261101.md、BALLISTICS_REVIEW_20261101.md与scripts/position/README.md。

2026-11-01反抛修复：禁止用人物left/heading_left与目标方位作拦截。MEM保留0..180实际世界发射角，
0..90折叠仅用于显示；计算/轨迹不得强制朝向目标。实际世界角必须与新鲜炮口样本一致。
角度加入轨迹与近期建议缓存；顺逆风按玩家→目标换算世界风，不能用炮口dx反转近身风向。
对初速远离目标的风返分支按实际轨迹下降交点求解，无解如实提示，不给目标镜像的假建议。
连续ready不清空仍新鲜、同身份/尺度炮口，也不刷新其时间戳；非ready、身份/尺度变化仍清除。
见BACKSHOT_ADVICE_FIX_20261101.md。不自动重启游戏或HUD。
