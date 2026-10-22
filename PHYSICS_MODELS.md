# HUD 弹道模型切换

默认：`recorded-calibrated-v2`，使用 2026-10-22 校准后冻结的常规轨迹参数。
`src/physics/recorded.rs` 是新模型；`src/physics/legacy.rs` 完整保留替换前公式及测试。
上一版 `recorded-frozen-v1` 原文件完整保留在 `src/physics/recorded_frozen.rs`。
`src/physics.rs` 是统一接口，力度求解与绘图选择同一模型。
本次只更新参数，积分顺序、力度求解、绘制与模型范围保持原实现。
校准依据与限制见 `BALLISTICS_CALIBRATION_20261022.md`。

## 启用

编译后关闭旧 HUD，再运行 `./start-hud.sh`。只换 HUD，不必为公式切换重启游戏。
运行中的旧 HUD 不支持热更新；启动日志会打印 `[PHYSICS] model=recorded-calibrated-v2`。
不要传 `--restart`，除非确实希望并明确授权重启游戏。

## 回退

关闭 HUD 后运行 `TNT_PHYSICS=frozen-v1 ./start-hud.sh` 回退到上一版，无需重新编译。
`TNT_PHYSICS=legacy ./start-hud.sh` 回退到更早的表格拟合版。
显式选新模型可运行 `TNT_PHYSICS=new ./start-hud.sh`。
模型选择在进程启动后固定；新的录制 manifest 会记录模型名。

## 坐标与限制

模型原始单位是服务器世界坐标，每一步是轨迹采样间隔，未证明是帧或秒。
HUD 对外仍使用 y 向上、距离单位；默认每单位80个世界坐标。
`TNT_WPU` 的正有限值用于统一坐标转换，默认80。
左右朝向与世界风/相对风转换沿用已有 HUD 逻辑。
此前小误差基于真实炮弹首点对齐；当前前端仍从玩家/手动点起算，
尚未修复炮口偏移或玩家坐标延迟，不能保证同样的实战瞄准误差。
碰撞、地形和特殊武器的行为没有因此被恢复。
保留现有角度/力度求解范围；没有自动重新拟合，也没有自动发射。
原始录制不会覆盖；旧 manifest 不修改。
新参数在常规尺度独立数据上改善约 0.2 个世界坐标单位的轨迹误差；这不是屏幕像素或命中率改善。
双尺度轨迹的运行时识别、炮口起点换算、碰撞后的轨迹仍未解决，不能宣称所有场景均已校准。
