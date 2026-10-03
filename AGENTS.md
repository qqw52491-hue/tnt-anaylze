# TNT 弹道 HUD

Rust+OpenCV 弹弹堂辅助 HUD。`./start-hud.sh` 一键:推 .so→重启游戏→起隧道→HUD。
网络层: 43.248.190.45:8888 WebSocket 二进制帧 = zlib(Protobuf)。解析在 `src/proto.rs`+`src/net.rs`(Tracker/隧道记录格式见注释)。

## 验证

```bash
cargo test --lib                    # 14 项
cargo build --bin live_gui --bin net_live --bin dump_wind_fields
```

## 风速解码(已确认结构,2025-10)

BattleNotify type-8 inner f4 = 带符号 i64。**符号=风向**(已验证)。
大小解码模型(3局独立验证,C为局内常数):

```
wind10 = (abs(raw) & 0xff) ^ C_battle ^ round
C_battle = 每局恒定的XOR密钥: 15xxx局=0x02, 146xx=0x1B, 154xx=0x1D, 128xxx≈0x19x
```

验证结果: 154xx局10连回合中6组逐位精确、4组差±0.2-0.4(疑ASR听错小数尾数)。
C来源仍未知(消息字段推不出)→**运行时用一次用户报风自举**:
live_gui 在每次语音/手动确认时解 `C=(|raw|&0xff)^w10^round`,之后每回合
自动填风速(我方回合才覆盖 st.wind,符号=箭头×方位×F翻转,标 [N])。
用户重报即更新C → 自纠错。环境变量无需开启,默认启用。

已排除: %1024/10、线性同余、位移/乘法/灰码、round^2。
注意: 8位模型上限25.5;若实测出现>25.5的大风需扩到10bit再验证
(128xxx局的40.2/43.8疑为ASR幻听未确认)。

## 配对采样(自动)

`live_gui` 运行中: 每条 SEED 写 `wind_pairs.csv`(applied列=自动预测值);
语音/回车确认风速自动记 VOICE/MANUAL 行。正常玩即可攒数据。

```bash
python3 analyze_wind.py            # 配对 raw↔wind, 列出每回合 key
```

## 禁用项

- 不要视觉吸附(用户明确否决)
- OCR 流光风字不可靠;`./w`/`scan.sh` 内存扫描已否决(全是假候选)
- vsnprintf/Qt setNum hook 无效(UI 走自定义渲染,只见 IP 格式化)

## 语音

`TNT_VOICE_TURBO=1 ./start-hud.sh` 常驻模型 ~1.3s;默认档 SenseVoice+Whisper 双保险。
