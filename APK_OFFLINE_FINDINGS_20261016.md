# 新版 APK 离线检查：移动 Y 与弹道入口

日期：2026-10-16。范围：读取用户提供的 APK，并复核本机已有的旧版运行时快照。

## 本轮结论

**移动 Y 有具体线索，完整弹道公式没有恢复。** 旧版客户端的移动请求 `MoveReq` 本来就只声明 round 和 X；客户端内部仍计算、更新二维位置，移动通知也携带二维位置。这解释了为什么只观察自己发出的移动请求会看到 X。此处的汇编证据来自旧版快照，不能直接宣称新版逻辑完全相同。

当前 HUD 源码已经接收服务器移动通知中的 X/Y；但收到仅含 X 的 `MoveReq` 时会保留之前的 Y。若实际移动期间没有新的二维位置通知，缓存 Y 就不能随地形立即变化。本轮没有用猜测的地形算法替换它。

新版 APK 的 Java 启动入口有保护壳，核心库的初始化/JNI 入口使用解释器包装，公开资源目录中的 3,571 个文件全部只有一个字节。本轮没有解出这些资源的真实内容，也没有取得新版完整游戏逻辑。因此不能声称已经找到新版重力、风力、阻力或力度到初速度的权威公式。

现有 HUD、抓包库和物理模型未修改；没有安装或执行 APK，也没有控制、停止或重启游戏/模拟器。

## 输入及版本身份

输入文件：`/Users/wx/Downloads/This is Not TNT v261016.1.APK`。

| 项目 | 实际读取结果 |
| --- | --- |
| APK SHA256 | `e1927b52986c019946b5fe52b81af98ee4fe083accd6517df6205c565e86c12b` |
| 包名 | `com.grapefruit.game.tnt` |
| Android versionName | `8.1`；文件名中的 `v261016.1` 另行保留，不混为同一个字段 |
| ZIP 成员数 | 3,636 |
| 核心库 | `lib/arm64-v8a/libgame_arm64-v8a.so`，11,506,128 字节 |
| 核心库 SHA256 | `15ad9b7d65d8f66c80f5d8f72400c6a1e6fe55380094cfdd07ae85273dfbc1a9` |

核心库与本机此前保存的 `target/tls-integration/libgame-20261016.so` 哈希完全一致。这次分析的是今晚抓包修复涉及的同一个磁盘核心库，并非另一份旧安装包。

证据保存在本项目 `target/apk-offline-20261016/`。该目录被 Git 忽略，以下相对证据文件名均指向它。

## 新版 APK 中实际看到的保护结构

1. Manifest 的 Application 为 `tnt.emag.tiurfeparg.moc.StubApp`。可直接反汇编的 DEX 只定义四个类：Configuration、DtcLoader、StubApp、a。`DtcLoader` 在 DEX 文件偏移 `0x2ecc` 调用 `System.loadLibrary("jgdtc")`；StubApp 保存 `libjgcggt` 名称并通过原生接口继续初始化。它们提供启动/加载逻辑，没有直接给出弹道公式。
2. `assets/libjgcggt_a64.so` 内存在 `libjiagu.so`、`JIAGU_ENCRYPTED_DEX_NAME`、`com/stub/StubApp`、`qihooCrash` 等明文字符串。这支持存在相关加固组件的判断；没有据此认定确切产品版本、全部保护机制或加密算法。
3. 核心库的普通节区信息不完整。通过 ELF 程序头和 `PT_DYNAMIC` 读取可得到 45,414 个动态符号记录，其中大量名字是无业务含义的标识。不能靠搜索 `wind`、`gravity` 等名字判断算法是否存在。
4. **在新版磁盘核心库上确认了一条解释器调用链：**

   ```text
   JNI_OnLoad，ELF 虚拟地址 0x530e50，84 字节
     x0 = 0x545c0c；w1 = 0xb58；x2 = 0x6dc470
     0x530e80: bl 0x2f1390
     PLT 使用 GOT 槽 0x6bfda8
     DT_JMPREL 将该槽绑定到 interpreter_wrap_int64_t，0x53d3ec
     包装函数在 0x53d4ac 调用 0x53b790
   ```

   `DT_INIT_ARRAY` 的磁盘值为零，必须同时读重定位；其中 RELATIVE 重定位恢复出的初始化入口为 `0x530dd0`，该入口也调用上述包装器。这里的地址均为新版 ELF 相对虚拟地址，**不是正在运行的进程地址**。没有把数据块误当 AArch64 指令，也没有把一个解释器入口推广成“所有游戏函数都已虚拟化”的结论。本轮没有恢复解释器指令语义和内部游戏模块。
5. `bttets/` 下 3,571 个文件全部为一字节，包含全部 247 个 `.pack`，不能当作真实地图、配置或物理参数表读取。ZIP 中未找到扩展名为 `.lua`、`.js`、`.qml`、`.json` 的源文件。两个 `raw/libnativesoHelper_*.ogg` 的文件头都不是 `OggS`；大的一个为 844,600,748 字节。其实际容器格式、内容与资源路径的映射未恢复，不能凭后缀说它是音频，也不能直接断言其中是哪一种加密。

对应证据：`manifest-tree.txt`、`dex-new-disassembly.txt`、`protection-strings.json`、`elf-metadata.json`、`static-entry-disassembly.txt`、`static-entry-metadata.json`、`native-dispatch-evidence.txt`、`resource-check.json`。

## 移动 Y：旧版快照的确定证据

这一节只使用先前已有的 `/Users/wx/Downloads/tnt-apk-analysis/dump/tntdump/seg_*.bin`，没有读取实时进程。以下绝对地址仅适用于该快照。

| 旧版函数/数据 | 地址 | 可以确认的内容 |
| --- | --- | --- |
| `MoveReq::kRoundFieldNumber` | `0x7584a61b20` | 读出的 u32 为 1 |
| `MoveReq::kXFieldNumber` | `0x7584a61b24` | 读出的 u32 为 2 |
| `MoveReq::SerializeWithCachedSizes` | `0x7584742ea0` | 声明字段只序列化对象 `+0x18` 的 round、`+0x1c` 的 X；未知字段转发不代表声明了 Y |
| `BattleSpritePlayerSelf::syncPosition` | `0x758494b4d4` | 调用位置读取路径后，在 `0x758494b584` 只把 d0 转成整数，随后存入请求 `+0x1c`；没有把 d1 写入请求 |
| `BattleNotify_Move::SerializeWithCachedSizes` | `0x7584725cf0` | 字段 1=id，字段 2=pos 子消息，字段 3=fall；字段名常量另行读取核对 |
| `BattleCommandMove::init` | `0x75848e8440` | 从 pos 对象 `+0x18/+0x1c` 读取两个 32 位坐标，写到命令 `+0x10/+0x14` |
| `BattleSpritePlayerSelf::moveTo(int,int,int)` | `0x758494c1b8` | 将前两个整型参数分别转成 double，成对传给位置设置路径 |
| `BattleSpritePlayerSelf::onRolePosChanged` | `0x758494b76c` | 从角色对象 `+0x88` 取得成对的有符号 32 位坐标，转换后传给相同的位置设置路径 |

左右移动函数还会在客户端更新 Y。`moveRight(double)` 位于 `0x758494c278`：步进分支按 `0.008333333333333333`（1/120）这个 double 常量处理输入时间；在每个水平步进内取得一个采样结果，并把返回的整型高度差加到当前 Y，再提交成对坐标。该常量只描述这条移动路径，不能当作炮弹或服务器物理时钟。

另外，`BattleTerrain::getCollisionPos(int,int,int&) const` 位于 `0x7584917f64`，会沿纵向扫描四字节像素的第 4 个字节，达到阈值后把当前 Y 写入输出引用。这符合地形透明度/碰撞掩码扫描的结构。阈值来自未保存的 GOT 区域；像素格式、全部坐标变换和移动采样调用目标未完整解析，不能把此描述当作可以直接复制进 HUD 的完整地形算法。

**仍缺少的环节：** 保存的旧快照没有覆盖多个 PLT/GOT 槽及回调表，例如位置读取槽 `0x7584b6d800`、位置设置槽 `0x7584b6f400`。因此对这些调用只描述参数与数据流，没有猜测具体 Qt/游戏 API 名称。`fall()` 涉及异步动作与回调，也没有恢复成完整重力递推。

对应证据：`movement-field-numbers.json`、`move-serializers.txt`、`movement-disassembly.txt`、`movement-proto-symbols.txt`。

## 当前 HUD 为什么可能出现“移动只变 X”

本轮只读核对了现有实现：

- `src/proto.rs:354` 已把 type-11 移动通知解析为 `Move { id, x, y }`；`src/bin/live_gui.rs:215` 会同时替换 X/Y。
- `src/proto.rs:410` 从 `MoveReq` 读取字段 2 的 X；`src/bin/live_gui.rs:218` 会更新自己的 X，并保留缓存的 Y。
- 玩家属性更新也可提供二维位置；飞行事件可提供玩家路径。玩家飞行路径不能当作炮弹轨迹。

因此，现有实现并非完全没有 Y 输入；缺口是**只看到自己发送的 X 时，无法独立知道地形造成的新 Y**。要确认新版游戏实际何时回传 Y，还需要对应时段的通知与游戏画面对照。仅凭旧汇编和现有解析代码，不能断言是漏包、字段变动或服务器根本不回传。

可行的后续顺序是先核对正常游戏中已记录的二维位置通知与时序；若仍缺少即时 Y，再研究新版真实地形数据和局内地形变化。此处只记录路线，没有开始新一轮实时采样，也没有采用被用户否决的视觉吸附或内存扫描。

## 弹道公式：目前能确认到哪里

本轮重新导出了旧版相关函数的反汇编，保存在 `ballistics-reference-disassembly.txt`，并与已有的 `/Users/wx/Downloads/tnt-apk-analysis/ballistics_runtime_analysis.md` 对照。

- 旧版 `Firing::update(double)`，`0x7584967be8`：累加时间、按 30 倍处理轨迹点索引，随后在相邻坐标间做 `p = p0 + (p1-p0) * (剩余时间*30)`。这是收到坐标后的播放插值；不能拿它冒充完整弹道方程，也不能由此认定服务器模拟频率为 30 Hz。
- 旧版普通发射请求路径 `BattleConsole::onForceClicked`，`0x7584957ad4`：请求力度为 `100 * 控件值`。`FireReq` 序列化器独立确认字段 4 为该 double。这个量尚未被证明等于炮弹初速度，也没有给出用户经验中固定力度偏差的来源。
- **尚未恢复：** 新版角度/力度/风速到全轨迹的计算，权威重力、阻力、风力系数，碰撞/爆炸/特殊弹种规则及其取整行为。

可能需要先还原新版受保护的逻辑/资源，再判断客户端是否持有完整预测器。若权威模拟只在服务端，APK 自身不会包含那份服务端实现；目前没有证据可以确认它一定只在服务端，也没有证据证明完整公式必在 APK 内。

TLS 抓包恢复说明端点明文可读，不等于本地保护资源或所有内部算法已经还原。风速恢复的成功也不能保证弹道公式的恢复难度相同。

## 复核与本轮边界

主要工具是本机 Android build-tools 的 aapt/dexdump、Python 标准库、Capstone，以及已有的只读快照分析脚本。没有把 APK、抓包或账号数据发送到外部服务。

可复查命令（在本项目目录运行）：

```bash
python3 target/apk-offline-20261016/inspect_static.py
python3 /Users/wx/Downloads/tnt-apk-analysis/inspect_runtime.py --symbols 'MoveReq|BattleNotify_Move'
python3 /Users/wx/Downloads/tnt-apk-analysis/inspect_runtime.py --function 'BattleSpritePlayerSelf.*(syncPosition|moveTo|moveLeft|moveRight)|BattleTerrain.*getCollisionPos'
```

`inventory.json` 保存了输入、抽取成员及读取前的 SHA256。结束时已用 `verify_integrity.py` 重新计算并比对：原 APK、31 个 Rust 源文件、10 个已有快照文件和 8 个抽取成员全部一致，结果见 `verification.json`。本轮没有功能代码变动，未重复执行编译或游戏验证。

本轮在已确认保护入口、资源占位结构、旧版移动数据流及具体证据缺口后结束。没有进行未经验证的参数替换，也没有把尚未解出的内容报告为“破解成功”。
