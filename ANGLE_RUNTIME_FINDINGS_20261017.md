# TNT 新版客户端：实时角度的原生数据来源（2026-10-17）

## 结论与验证程度

**已在当前 BlueStacks 战斗进程里通过只读方式读取到我方的 Angle 控件整数角度。** 这不再是 OCR 或扫描 1..90 的碰运气候选值。

- 首次战斗读取：`Angle.value = 40`，范围 `40..67`。
- 随后的战斗读取：`Angle.value = 43`，范围仍是 `40..67`。
- 再次战斗读取：`Angle.value = 53`，范围 `35..75`；说明脚本能使用新的战斗对象状态，不能由此单独证明连续调角响应。
- 8 秒连续短采样成功，期间显示 `43` 未变化；没有人为控制游戏或尝试发射。
- 已确认对应对象的精确类型，`BattleSpritePlayerSelf` 和 `Angle` 两层虚表均匹配当前库的运行时映像偏移。
- 该值**已确认是游戏内部角度控件值**，尚未在所有地形/镜像/方向上实测等同于屏幕展示的最终角度或 `FireReq.angle`。不能立即据此覆盖现有 HUD 角度。
- 后续同步读取了 `QGraphicsItem::rotation()` 和镜像图形项的 Qt scene transform 缓存：曾实测 `Angle.value=49 / rotation=28° / 候选未镜像21°、镜像103°`；又读到 `Angle.value=54 / rotation=20° / scene_m11=-0.9396926207859084`，据此镜像候选为 `106°`；另一时刻 `Angle.value=58 / rotation=20° / scene_m11≈-0.939693`，**临时计算**镜像发射角 `102°`。这证明运行时旋转可与控件值不同，但不等于实际发射角已由独立样本验证。

## 原生代码依据

分析文件：`target/rea-deep-20261017/reconstructed-elf-20261017/libgame-mapped-analysis.so`。这是 **仅供分析的重建 ELF**，不是可执行部署文件；用其真实导出符号、函数指令和 PLT 重定位交叉核对。

| 函数/指令 | 新版 ELF VA | 已验证行为 |
| --- | --- | --- |
| `BattleSpritePlayerSelf::BattleSpritePlayerSelf` | `0x8412e0` | `0x841358` 将新建 `Angle*` 存到 self `+0x1d8` |
| `Angle::setValue(int)` | `0x84a2f0` | `0x84a318` 比较旧值、`0x84a33c` 写入当前值到 `Angle+0xa0` |
| `Angle::setRange(int,int,int)` | `0x84a224` | `Angle+0xa4` 上界、`Angle+0xa8` 下界 |
| `BattleSpritePlayerSelf::setAngle(bool)` | `0x841800` | 从 `self+0x1d8` 取指针，读取 `Angle+0xa0`，按方向增减，再通过 `Angle::setValue` 设值 |
| `BattleSpritePlayerSelf::fireAngle()` | `0x84239c` | `0x8423cc..0x8423f8` 读取 `Angle+0xa0` 并结合图形旋转/镜像计算发射角 |
| `QGraphicsItem::rotation()` | QtWidgets `0x47ebf4` | 读 `item+0x8` private，再读 `private+0x90` transform；非空时返回 `transform+0x60` double |
| `GraphicsHelper::isFlippedX(QGraphicsItem*)` | `0x526844` | 基于场景变换的第一个元素是否小于 0 判断水平翻转 |
| `QGraphicsItem::sceneTransform()` | QtWidgets `0x47dd88` | `0x47ddc4..0x47dde4` 从 `QGraphicsItemPrivate+0xa0` 复制 Qt 场景矩阵；函数先执行缓存更新，故脚本直接读取该缓存可能在特定时刻仍旧 |

关键指针链：

```text
战斗 PanelBattle 的 sprite_hash[我方 id]
  -> BattleSpritePlayerSelf*（运行时主虚表 = image_bias + 0xa647f8）
  -> self+0x1d8 -> Angle*（运行时主虚表 = image_bias + 0xa64fa8）
  -> Angle+0xa0 -> i32 当前控件角度
  -> Angle+0xa4 -> i32 上界
  -> Angle+0xa8 -> i32 下界
```

`fireAngle()` 的静态公式（`trunc` 表示朝零取整）大致为：

```text
r = QGraphicsItem::rotation(self+0x80 的图形项)
a = *(int32*)(*(self+0x1d8)+0xa0)
if GraphicsHelper::isFlippedX(self+0x78 的图形项):
    return trunc(180.0 - r - a)
else:
    return trunc(a - r)
```

这里的 `isFlippedX` 是场景变换判断，不能用“角色在地图左/右侧”未经核验地代替。`BattleConsole::updateAngle` 也结合 Qt 旋转计算文本角度，因此 **Angle.value 与视觉展示/游戏发送角度之间可能有变换差**，尤其在倾斜地形或镜像状态中。

## 当前新增的只读试验工具

`scripts/position/probe_live_angle_readonly.py`

单次读取：

```bash
cd /Users/wx/Downloads/tnt-comput
python3 scripts/position/probe_live_angle_readonly.py
```

最多 30 秒的变化观察（**角度、图形旋转、镜像缓存或角度范围**变化时打印；无需按角度键才允许脚本重新计算）：

```bash
python3 scripts/position/probe_live_angle_readonly.py --watch-seconds 30 --interval 0.25
```

工具使用现有 `probe_live_xy_readonly.py` 的进程/战斗对象发现机制，不修改内存或控制游戏。额外检查 `Angle::setValue` 和 `fireAngle` 的关键机器指令，验证我方 `BattleSpritePlayerSelf` 及 `Angle` 虚表，批量读取时每次复核角色 ID 和指针身份。战斗退出/对象变化会丢弃旧指针，避免返回虚假角度。

新版脚本增加 `qt_rotation_degrees`、`scene_m11_cached`、`flipped_x_cached`、`fire_angle_candidates`（未镜像/镜像两套）及 `provisional_fire_angle`。**注意：`provisional_fire_angle` 根据缓存场景矩阵选择分支，属于有明确逆向依据但尚未和实际 FireReq 核实的候选结果。** Qt 的 `sceneTransform()` getter 会先确保场景矩阵缓存有效，而脚本本身只读缓存，不会触发该更新；所以如果左右移动后 UI 不刷新，仍需检查读数是否过期。

**后续已将内存角度设为默认 MEM，并新增 MEM/OCR 切换按钮。** 为减少性能开销，正常启动 MEM 时**完全不运行**旧角度 OCR（不截取表盘、不创建识别器、不弹角度区域选择）；内存读数失效会明确显示无新读数，只有手动点击 OCR 后才执行一次角度框选与旧识别。默认小地图、无启动模式选择，小地图 ROI 优先自动复用上次校准的持久位置（无缓存时可能需要手工框选一次），背景截图频率由约 4Hz 降到约 2Hz。`TNT_LIVE_ANGLE=0` 可强制禁用内存采样，保留以前的 OCR。保留原坐标、风速、网络录制和物理模型代码，没有停止/重启游戏或 HUD，也未提交或推送已有改动。

从现有 HUD 运行日志看到 `NET FIREREQ angle=110`，表盘识别为 `70`～`71`；两者符合折角关系 `180−110=70`。另一次运行时独立采样读到内部值 `65`、旋转 `5`、镜像缓存为负，按客户端公式也得到候选发射角 `110`。**这些样本没有建立同一时刻的完整配对记录**，所以只是支持该公式，而非证明每个回合都完全正确。

## 下一步判断是否能安全接入 HUD

1. 在战斗中手动连续调整角度，观察脚本输出 `Angle.value` 是否逐度变化，确认延迟及更新频率；不要让脚本代替玩家操作。
2. 现已补读 Qt 图形旋转和镜像缓存。对照画面角度时，分别记录 `angle_value`、`qt_rotation_degrees`、`scene_m11_cached`、`provisional_fire_angle`；观察每回合 11°、13°、2° 等差值是否等于当时旋转的影响。不要假设固定加减某个数字。
3. 只手动左右移动**但不调整瞄准角度**时，观察 `qt_rotation_degrees`/`scene_m11_cached` 是否变。如果变而 `angle_value` 不变，则脚本会主动输出新候选；若游戏自己的角度条不刷新，这两种显示可能不一致。还需结合真实 `FireReq.angle` 比较发射瞬间结果；它不能代替瞄准过程中的实时样本。
4. **本地内存角默认启用，OCR 默认关闭，只在点击 MEM/OCR 按钮切至 OCR 后启用。** 正常建议由 `./start-hud.sh` 启动以保留网络采集初始化，`./target/debug/live_gui` 仅运行独立 HUD。用户已反馈快速连按调角正常，但并非所有斜坡/镜像/回合场景均经过端到端验证；存在偏差时可手动切 OCR 对照，或用 `TNT_LIVE_ANGLE=0` 禁用内存探针。

## 本轮测试与环境

- 真实只读进程发现曾返回 `coordinates_found`，可以识别两名角色；其中一名的虚表与 `BattleSpritePlayerSelf` 精确匹配。
- 新工具 `python3 -m py_compile`、角度值/边界解析用例通过。
- 新工具单次真实读取返回：
  `{"status":"angle_value_found","angle_value":43,"allowed_range":[40,67],...}`。
- 连续采样 8 秒未见值变化，不把它解释成“连续变化已经实证”。
- 游戏有时处于大厅，旧的探针会返回 `not_in_battle`；这是正常安全边界，不能重用上一局对象地址。
