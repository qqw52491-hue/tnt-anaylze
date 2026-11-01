# TNT 我方实时 X/Y（默认启用）

2026-10-22 性能检查后，`./start-hud.sh` 会先检查源码更新并默认启动优化编译的 `target/release/live_gui`；需要调试版时用 `TNT_HUD_PROFILE=debug ./start-hud.sh`。下文直接运行 `target/debug/live_gui` 的示例仍可用于调试，但不会自动更新编译。详见根目录 `PERFORMANCE_REVIEW_20261022.md`。

HUD 的 `live_gui` 默认启动只读 Android 内存探针，以 `QGraphicsItem::pos()` 作为我方角色的实时图形坐标，支持走路、坡面和飞行。敌方保留已有网络位置解析，原始网络录制不改写。

**正常启动（默认已启用）：**
```bash
./target/debug/live_gui
```

**临时关闭，只用原来的网络定位：**
```bash
TNT_LIVE_XY=0 ./target/debug/live_gui
```

**也可显式开启：** `TNT_LIVE_XY=1 ./target/debug/live_gui`

脚本 `probe_live_xy_readonly.py` 负责每次完整校验游戏私有 ELF 指令、Qt5Widgets `QGraphicsItem::pos()` 实现指令、LayerUI/PanelBattle 虚表和哈希表的玩家身份。脚本 `stream_live_xy_readonly.py` 快速采样并每约 3 秒重新完整校验，不修改游戏或设备。游戏未启动、未处于战斗、版本/ABI 校验失败、读错/超时/过期时会输出清除标记，HUD 自动使用原网络坐标。

游戏或 Qt 版本升级后，内部 ELF 映射、虚表、Sprite 结构、Qt 私有对象偏移或 Android 包名变化都可能导致探针暂时不可用。它不会尝试猜测新地址，而会回退，需要重新分析并更新探针。未来若遇到 Y 不更新，先看 `[XY]` 和 `XY HUD: applied=...` 的诊断输出。

## 实时角度（默认 MEM，角度 OCR 完全关闭）

新版已有独立的只读角度探针，可从当前游戏的 `BattleSpritePlayerSelf -> Angle`
读取角度控制值、Qt 图形旋转与镜像缓存，再按客户端 `fireAngle()` 公式
推算开火角度。**此值尚未完成各种地形/左右移动的实战交叉验证，不保证准确**。

**普通启动已默认启用内存角度**（仅需重开 HUD，不需要重启游戏）：

```bash
cd /Users/wx/Downloads/tnt-comput
./target/debug/live_gui
```

在 HUD 原有“自动角度同步”按钮的**右侧**新增独立的 `MEM / OCR` 按钮（同一行，最右侧）。

- `MEM`（默认）：仅使用内存角度，**不运行角度 OCR，不截图角度区域，也不加载旧识别器**。无效或过期时明确显示无新读数，不拿上回合或 OCR 的旧数值冒充当前角度。
- `OCR`：手动点击后才启用旧表盘识别；第一次切换时会要求框选一次角度表盘，以后同一次 HUD 运行中不再重复框选。
- 切换只改变角度**来源**，不会启停游戏、改变坐标、风速、网络模块，也不会取消用户手动角度。
- 如需连内存采样进程都禁用，可在启动 HUD 时设置 `TNT_LIVE_ANGLE=0 ./target/debug/live_gui`。此时 UI 固定 OCR；下次正常启动恢复 MEM 默认。

在 MEM 模式下，`stream_live_angle_readonly.py` 以只读方式流式采集。HUD
**只接受来自经过战斗面板、我方 BattleSpritePlayerSelf 和 Angle 虚表校验、且采集时间不超过 2 秒**的数值；如果网络已经明确识别出我方 UID，则还必须与网络 UID 一致。**网络 `my_id=None` 不再被误判为 UID 冲突。** 将
`FireReq` 的世界方向角折回 0..90° 的弹道抬角用于显示，例如110°→70°；计算与绘图另保留完整世界发射角，不能丢掉左右方向。
界面显示 `内存角`；若读取失效会显示“内存角暂无新读数”，不会后台偷偷使用 OCR。只有手动点击 `OCR` 才会切换到旧表盘识别，切回 MEM 便停止 OCR 截图和识别。
表盘识别的旧值超过 2 秒未更新时，会标识“保留旧值”，不伪装成新读数。
用户设置的手动角度和语音角度仍优先，不被覆盖。

如果界面一直显示“内存角暂无新读数”，先看启动 HUD 的终端：`[ANGLE] not_in_battle` 表示当前不是战斗界面，`[ANGLE] game_not_running` 也可能是 BlueStacks ADB 断线。新版角度流会在无 ADB 设备时每隔至少 10 秒尝试一次**仅重连 ADB**（绝不重启游戏），并输出 `[ANGLE] adb reconnect`。HUD 还会每约 5 秒打印 `ANGLE HUD: no_verified_sample / uid_mismatch / sample_stale / sample_ready / sample_ready_local`，分别表示无有效角度样本、与**已知**网络身份冲突、过期、网络身份已核对、本地已验证但网络身份尚未知。`sample_ready_local` 在内存探针验证身份成功时也允许显示“内存角”，不依赖网络隧道是否连接；无论哪种就绪状态，读数仍是待实测的候选角度。

**网络通道是另一条独立链路。** 直接运行 `./target/debug/live_gui` 只会绑定本地 19001 端口，**不会执行 `start-hud.sh` 的 ADB reverse 配置**；如果日志持续 `conns=0 map=0x0`，应单独检查调试桥：`/Applications/BlueStacks.app/Contents/MacOS/hd-adb -s 127.0.0.1:5555 reverse --list`。缺少 `tcp:19001 tcp:19001` 时，可用 `... reverse tcp:19001 tcp:19001` 恢复转发；这不会重启游戏，但仅设置转发也无法保证游戏的网络采集模块已经加载。**网络断开不会再导致已验证的本地角度强制回退，但网络风/世界地图等仍需要网络链路。**

**启动和性能规则：**现在直接 `./start-hud.sh` 即默认小地图+MEM，不再停在大/小地图输入选择，也不弹角度 OCR 框选；小地图优先使用 `~/.config/tnt-comput/minimap_region.txt` 记录的上次位置（重启电脑后仍保留），旧模板匹配也会尝试自动定位，实在没有校准数据时才要求框选一次小地图（之后会缓存位置）。屏幕分辨率、窗口位置变动时使用 `TNT_RESELECT_MINIMAP=1 ./start-hud.sh` 重新校准。旧大地图模式可以在 HUD 内按原有 BIG MAP 按钮切换，或者启动时设 `TNT_MAP_MODE=big`。地图背景截图间隔从 **250ms 降至 500ms**，对地图刷新率更重视时可设 `TNT_MAP_CAPTURE_MS=250`，希望更省资源可设 `TNT_MAP_CAPTURE_MS=1000`（允许范围 250–3000 毫秒）；内存角度与网络位置采样不受该选项影响。无需角度 OCR 时截图线程睡眠从 20ms 放宽到 60ms。

**内存角开关：**不设置 `TNT_LIVE_ANGLE` 就默认 MEM；`TNT_LIVE_ANGLE=0` 为强制关闭内存采样，启动 HUD 后会首次请求 OCR 框选；`TNT_LIVE_ANGLE=1` 为显式启用。
已存在的实时 X/Y、风速、网络录制和游戏内存均不受这项实验影响。
**快速调角优化：** 内存角度的普通轮询已从约 14 段独立读取减至 **8 段**，依然每次核对战斗面板、我方 UID 和角度指针。以前每隔约 3 秒同步执行一次耗时的完整扫描，可能在长按角度键时阻塞更新；现在检测到内部角度还在持续变化，就推迟完整扫描到停止变化至少 0.8 秒后，**最长 20 秒强制复验一次**。遇到临时 ADB 超时/短读，只丢弃当帧并容许一次有身份校验的快速重试；真正 UID/对象不匹配仍立即清理缓存。终端每约 5 秒打印 `[ANGLE] samples_per_sec=... max_read_ms=... max_full_ms=... max_gap_ms=...`，分别显示采样频率、最慢读取、完整扫描耗时、连续有效样本的最大间隔。**目前只完成 8 项自动化角度单测，没有在长按实战里证明零延迟、无跳帧。**

**HUD 人物点位渲染未纳入这些角度单元测试，也尚未完成带实际游戏画面的端到端测试。** 代码里的网络人物点位仍须 `map_w/map_h`、玩家网络身份和网络位置存在；`mem_roles=2` 但 `conns=0 map=0x0` 时不会自动出现人物点位。请不要把角度测试成功当成坐标/风/界面已恢复。

**坐标和风速恢复是独立问题。** 以前的只读检查曾发现游戏进程未加载原网络采集模块 `libtntsniff.so`；即使 ADB reverse 已配置，`conns=0、map=0x0` 仍意味着网络没有向 HUD 传数据。现有 `start-hud.sh` 会检查模块，默认**不会**重启当前游戏；只有用户明确选择 `TNT_LIVE_ANGLE=1 ./start-hud.sh --restart` 才会执行可能中断对局的恢复流程。**本次角度优化没有修改坐标、风速、网络程序，也没有自动重启游戏。**

**内存角度仍使用 Qt 场景缓存选择左右分支，这个缓存可能滞后。**
遇到与画面或实际 `FIREREQ` 不一致时，立刻关闭实验开关并保留日志，
不要把该数值当成最终可信角度。详情见根目录
`ANGLE_RUNTIME_FINDINGS_20261017.md`。

注意：只有退出并重新打开 **HUD** 才能加载重新编译的 Rust 程序；**不需要退出或重启游戏**。运行中的 HUD 不会热更新。

## 炮弹起点（2026-11-01）

恢复的客户端 `firePos()` 会把 Angle 子图形的原点映射到人物父坐标系，再按客户端规则取整。
`probe_fire_origin_readonly.py` 重建本地 Qt 变换链，包含镜像、坡面旋转和旋转中心；不采用固定炮口偏移，
不调用游戏函数，不使用缓存场景矩阵。可单次只读诊断，或用 `--watch-seconds 30` 有界观察；这不会更新HUD。

HUD 默认在原角度进程附加 `--fire-origin`，每至少350ms尝试一次炮口读取，无新增常驻进程。
行格式为 `TNTORIGIN pid uid sample-unix-ms fire-angle left player-x player-y origin-x origin-y scale`（scale为1或2，旧无末字段行按1兼容）；
`TNTORIGIN-` 立即清理候选。源侧要求两次相同快照、均匀正交尺度1或2、读取不超过250ms、
当前角度与原角度读取一致；尺度2的炮口范围同步扩大，非均匀/动画中间尺度继续拒绝；HUD要求总样本年龄不超过600ms、UID及人物坐标（各轴容差1世界单位）匹配、
当前实际世界发射角与炮口样本匹配；人物朝向不限制目标方位，允许反抛。只在 recorded 模型、MEM、纯网络点位路径使用；自动MEM纯网络路径过期/失效时等待新鲜炮口，禁止切回人物起点。
手动点、OCR、legacy使用人物起点。均匀尺度2选择独立放大炮弹模型，
恢复1时切回普通模型，力度/轨迹共用该选择；详见EXPANDED_CALIBRATION_20261101.md。

关闭炮口修正用 `TNT_FIRE_ORIGIN=0 ./start-hud.sh`，原有角度、坐标、风速功能保留。
当前证据是同一姿态的3炮发射前起点与实际首点完全一致；跨地图/姿态以及新GUI画面的端到端验证仍有限。
完整发现过程和读取延迟限制见根目录 `BALLISTICS_REVIEW_20261101.md`。正常重开HUD后生效，游戏不必退出。

离线检查（不连接设备）：
```bash
python3 -m unittest discover -s scripts/position -p 'test_*readonly.py'
```

**静止力度跳变修正：** 上述早期“失效沿用人物起点”的策略在自动MEM网络模式下已取消，
因为短暂过期会使两套起点的推荐力度交替。现在自动模式失效会显示具体读数/姿态更新原因；
完整场景复验改为单个后台线程，前台持续读取，完整缓存授权仍以20秒为上限，炮口仍600ms过期。
显式手动角度/点位、OCR、legacy及关闭炮口继续人物起点估算。记录包含源侧拒绝原因。
短暂数据缺失不会用“冻结旧力度”冒充新读数。当前结果和验证见 `HUD_STABILITY_20261101.md`。

当前界面版式见 `HUD_LAYOUT_20261101.md`。新增 `TNTORIGINSTATUS pid uid sample-unix-ms code scale`
与清除行 `TNTORIGINSTATUS-`，供UI区别ready/scale/angle/moving/read；同身份、新鲜时间才接受提示。
只用于解释不可用原因，不放宽实时炮口门槛或自动启用特殊模型。
短缺帧可显示输入完全一致的“上次建议·更新中”参考值，原采样最多1.2秒；不可当成新采样。

正常 HUD 还会传入 `--record-origin`：在同一私有录制目录自动写入 `origin_samples.jsonl`，
与 `raw.bin`、`events.jsonl` 共用会话。外层每行含主机接收时间与原始 `probe_line` 字符串；
`TNTPOSE ` 后为JSON，包含采样起止时间、人物/炮口位置、角度/朝向、逐节点本地矩阵、
原始读取段（地址/长度/十六进制）及源侧 `source_eligible`。即使双尺度、未稳定或读取失败也保留记录，
并保留 `TNTORIGIN-` 清除标记；这些是观测值，不是独立真值，源侧通过也不代表HUD最终应用。
离线对照必须限定同一进程/身份/对局，只选请求之前的新鲜样本，并核对清除标记及延迟。
不能拿发射后的姿态或实际首点去反选最佳起点。文件权限600，目录700，禁止上传公开仓库。
录制不额外读取游戏内存，不添加采样进程，不改变已有源侧/界面回退条件；磁盘故障会提示并停止本次录制。

连续ready状态保留仍新鲜炮口，不延长采样时效；反抛修复见根目录BACKSHOT_ADVICE_FIX_20261101.md。
