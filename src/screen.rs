//! 跨平台屏幕抓图与交互式区域选择。
//!
//! ⚠️ 本文件不含任何弹道 / 风力 / 力度公式。
//!
//! 为什么需要它：
//!   - macOS 上原来的 `get_slurp_geometry()` 直接返回写死的 (0, 0, 800, 600)，
//!     根本没有选区；而且主流程里第二步还无条件调了 Linux 的 `grim`。
//!   - macOS 没有 slurp 这种能直接返回坐标的选区工具，`screencapture -i`
//!     只能存图、不告诉你坐标。所以这里自己实现：先全屏截图，
//!     再在 OpenCV 窗口里让用户拖框，用的全是项目里已经跑通的 highgui API。
//!
//! Retina 坑：`screencapture -R` 吃的是**逻辑点**，而截出来的 PNG 是**物理像素**，
//! 两者在 Retina 上差 2 倍。不换算的话选区会整个偏一倍。

use opencv::{core, highgui, imgcodecs, imgproc, prelude::*};
use std::process::Command;
use std::sync::{Arc, Mutex};

/// (x, y, width, height)，单位是截图命令所需的坐标系：
/// macOS 为逻辑点，Linux 为像素。
pub type Geometry = (i32, i32, i32, i32);

/// 后台轮询抓图的临时文件路径 (小地图, 右下角数值区)。
pub fn capture_paths() -> (&'static str, &'static str) {
    #[cfg(target_os = "macos")]
    {
        ("/tmp/tnt_map.png", "/tmp/tnt_power.png")
    }
    #[cfg(not(target_os = "macos"))]
    {
        ("/tmp/tnt_map.ppm", "/tmp/tnt_power.ppm")
    }
}

// ---------------------------------------------------------------------------
// 全屏截图
// ---------------------------------------------------------------------------

#[cfg(target_os = "macos")]
fn capture_full_screen(path: &str) -> bool {
    Command::new("sh")
        .arg("-c")
        .arg(format!("screencapture -x -t png {}", path))
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

#[cfg(not(target_os = "macos"))]
fn capture_full_screen(path: &str) -> bool {
    // Wayland 优先 grim，不行再试 X11 的 import
    let ok = Command::new("sh")
        .arg("-c")
        .arg(format!("grim {}", path))
        .status()
        .map(|s| s.success())
        .unwrap_or(false);
    if ok {
        return true;
    }
    Command::new("sh")
        .arg("-c")
        .arg(format!("import -window root {}", path))
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

/// 按区域抓图到文件。先写 .tmp 再 mv，避免读到写一半的图。
pub fn capture_rect_to_file(geo: Geometry, path: &str) {
    #[cfg(target_os = "macos")]
    let cmd = format!(
        "screencapture -R {},{},{},{} -x -t png {}.tmp && mv {}.tmp {}",
        geo.0, geo.1, geo.2, geo.3, path, path, path
    );
    #[cfg(not(target_os = "macos"))]
    let cmd = format!(
        "grim -g \"{},{} {}x{}\" -t ppm {}.tmp && mv {}.tmp {}",
        geo.0, geo.1, geo.2, geo.3, path, path, path
    );

    let _ = Command::new("sh").arg("-c").arg(cmd).status();
}

// ---------------------------------------------------------------------------
// Retina / HiDPI 倍率
// ---------------------------------------------------------------------------

/// 桌面的逻辑尺寸（点）。macOS 通过 osascript 问 Finder。
#[cfg(target_os = "macos")]
fn logical_screen_size() -> Option<(i32, i32)> {
    let out = Command::new("osascript")
        .arg("-e")
        .arg("tell application \"Finder\" to get bounds of window of desktop")
        .output()
        .ok()?;
    let s = String::from_utf8(out.stdout).ok()?;
    let nums: Vec<i32> = s
        .trim()
        .split(',')
        .filter_map(|p| p.trim().parse::<i32>().ok())
        .collect();
    if nums.len() == 4 {
        let w = nums[2] - nums[0];
        let h = nums[3] - nums[1];
        if w > 0 && h > 0 {
            return Some((w, h));
        }
    }
    None
}

#[cfg(not(target_os = "macos"))]
fn logical_screen_size() -> Option<(i32, i32)> {
    None
}

/// 物理像素 -> 逻辑点 的倍率。Retina 上通常是 2.0。
fn pixel_to_point_ratio(screenshot_width: i32) -> f64 {
    if let Some((lw, _)) = logical_screen_size() {
        if lw > 0 {
            let r = screenshot_width as f64 / lw as f64;
            // 只接受合理范围，防止 osascript 返回奇怪值时把坐标算飞
            if (0.9..=4.0).contains(&r) {
                return r;
            }
        }
    }
    // 拿不到就猜：超宽基本就是 Retina
    if screenshot_width >= 2560 { 2.0 } else { 1.0 }
}

// ---------------------------------------------------------------------------
// 交互式选区
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Default)]
struct DragState {
    start: Option<core::Point>,
    cur: Option<core::Point>,
    dragging: bool,
    has_box: bool,
}

/// 在全屏截图上拖框选区。
///
/// 操作：按住左键拖出矩形 -> 松开 -> Enter/空格 确认，ESC 取消，不满意可以重拖。
/// 返回的坐标已经换算成截图命令需要的单位（macOS 为逻辑点）。
fn select_region_interactive(title: &str) -> Option<Geometry> {
    let tmp = "/tmp/tnt_fullscreen.png";
    if !capture_full_screen(tmp) {
        eprintln!("❌ 全屏截图失败，无法进入选区");
        return None;
    }

    let full = imgcodecs::imread(tmp, imgcodecs::IMREAD_COLOR).ok()?;
    if full.empty() {
        eprintln!("❌ 截图读取失败：{}", tmp);
        return None;
    }

    let img_w = full.cols();
    let img_h = full.rows();
    let ratio = pixel_to_point_ratio(img_w);

    // 把全屏图缩到能放进窗口的尺寸
    let max_w = 1400.0;
    let max_h = 850.0;
    let disp_scale = (max_w / img_w as f64)
        .min(max_h / img_h as f64)
        .min(1.0);
    let disp_w = ((img_w as f64 * disp_scale).round() as i32).max(1);
    let disp_h = ((img_h as f64 * disp_scale).round() as i32).max(1);

    let mut base = core::Mat::default();
    imgproc::resize(
        &full,
        &mut base,
        core::Size::new(disp_w, disp_h),
        0.0,
        0.0,
        imgproc::INTER_AREA,
    )
    .ok()?;

    let win = "Select Region";
    highgui::named_window(win, highgui::WINDOW_AUTOSIZE).ok()?;

    let drag = Arc::new(Mutex::new(DragState::default()));
    let drag_cb = drag.clone();

    highgui::set_mouse_callback(
        win,
        Some(Box::new(move |event, x, y, _flags| {
            let Ok(mut d) = drag_cb.lock() else { return };
            if event == highgui::EVENT_LBUTTONDOWN {
                d.start = Some(core::Point::new(x, y));
                d.cur = Some(core::Point::new(x, y));
                d.dragging = true;
                d.has_box = false;
            } else if event == highgui::EVENT_MOUSEMOVE {
                if d.dragging {
                    d.cur = Some(core::Point::new(x, y));
                }
            } else if event == highgui::EVENT_LBUTTONUP {
                if d.dragging {
                    d.cur = Some(core::Point::new(x, y));
                    d.dragging = false;
                    d.has_box = true;
                }
            }
        })),
    )
    .ok()?;

    println!("🖱️  {}：按住左键拖框 -> 松开 -> [Enter/空格] 确认，[ESC] 取消，不满意可重拖", title);

    let result: Option<Geometry>;
    loop {
        let d = *drag.lock().ok()?;

        let mut canvas = base.try_clone().ok()?;

        if let (Some(s), Some(c)) = (d.start, d.cur) {
            let r = core::Rect::new(
                s.x.min(c.x),
                s.y.min(c.y),
                (s.x - c.x).abs(),
                (s.y - c.y).abs(),
            );
            let color = if d.has_box {
                core::Scalar::new(0.0, 255.0, 0.0, 0.0)
            } else {
                core::Scalar::new(0.0, 255.0, 255.0, 0.0)
            };
            let _ = imgproc::rectangle(&mut canvas, r, color, 2, imgproc::LINE_8, 0);

            // 实时显示真实（未缩放）尺寸
            let real_w = (r.width as f64 / disp_scale / ratio).round() as i32;
            let real_h = (r.height as f64 / disp_scale / ratio).round() as i32;
            let label = format!("{}x{}", real_w, real_h);
            let _ = imgproc::put_text(
                &mut canvas,
                &label,
                core::Point::new(r.x, (r.y - 6).max(14)),
                imgproc::FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
                imgproc::LINE_AA,
                false,
            );
        }

        let hint = format!("{}  |  Drag a box, then [Enter] confirm / [ESC] cancel", title);
        let _ = imgproc::put_text(
            &mut canvas,
            &hint,
            core::Point::new(12, 26),
            imgproc::FONT_HERSHEY_SIMPLEX,
            0.6,
            core::Scalar::new(0.0, 255.0, 0.0, 0.0),
            2,
            imgproc::LINE_AA,
            false,
        );

        highgui::imshow(win, &canvas).ok()?;
        let key = highgui::wait_key(20).unwrap_or(-1);

        if key == 27 {
            result = None;
            break;
        }
        if key == 13 || key == 10 || key == 32 {
            if !d.has_box {
                println!("⚠️  还没有拖出框，先按住左键拖一个矩形");
                continue;
            }
            let (Some(s), Some(c)) = (d.start, d.cur) else {
                continue;
            };

            // 显示坐标 -> 截图物理像素
            let px_x = (s.x.min(c.x) as f64 / disp_scale).round();
            let px_y = (s.y.min(c.y) as f64 / disp_scale).round();
            let px_w = ((s.x - c.x).abs() as f64 / disp_scale).round();
            let px_h = ((s.y - c.y).abs() as f64 / disp_scale).round();

            if px_w < 8.0 || px_h < 8.0 {
                println!("⚠️  选区太小了，重新拖一个");
                continue;
            }

            // 物理像素 -> 截图命令的坐标系（macOS 是逻辑点）
            let gx = (px_x / ratio).round() as i32;
            let gy = (px_y / ratio).round() as i32;
            let gw = ((px_w / ratio).round() as i32).max(1);
            let gh = ((px_h / ratio).round() as i32).max(1);

            // 夹回屏幕范围内
            let max_gx = (img_w as f64 / ratio).round() as i32;
            let max_gy = (img_h as f64 / ratio).round() as i32;
            let gx = gx.clamp(0, (max_gx - 1).max(0));
            let gy = gy.clamp(0, (max_gy - 1).max(0));
            let gw = gw.min((max_gx - gx).max(1));
            let gh = gh.min((max_gy - gy).max(1));

            println!("✅ {} 已选定: x={} y={} w={} h={} (倍率 {:.1})", title, gx, gy, gw, gh, ratio);
            result = Some((gx, gy, gw, gh));
            break;
        }

        // 窗口被关掉就当取消
        let visible = highgui::get_window_property(win, highgui::WND_PROP_VISIBLE).unwrap_or(1.0);
        if visible < 1.0 {
            result = None;
            break;
        }
    }

    let _ = highgui::destroy_window(win);
    let _ = highgui::wait_key(1);
    result
}

/// Linux 上优先用 slurp（直接返回坐标，体验最好）。
#[cfg(target_os = "linux")]
fn try_slurp() -> Option<Geometry> {
    let output = Command::new("slurp").output().ok()?;
    if !output.status.success() {
        return None;
    }
    let geo_str = String::from_utf8(output.stdout).ok()?;
    let parts: Vec<&str> = geo_str.trim().split_whitespace().collect();
    if parts.len() == 2 {
        let xy: Vec<i32> = parts[0].split(',').filter_map(|s| s.parse().ok()).collect();
        let wh: Vec<i32> = parts[1].split('x').filter_map(|s| s.parse().ok()).collect();
        if xy.len() == 2 && wh.len() == 2 && wh[0] > 0 && wh[1] > 0 {
            return Some((xy[0], xy[1], wh[0], wh[1]));
        }
    }
    None
}

#[cfg(not(target_os = "linux"))]
fn try_slurp() -> Option<Geometry> {
    None
}

/// 对外的统一入口：选一个屏幕区域。
///
/// Linux：先试 slurp，没装就回退到交互式选区。
/// macOS：直接交互式选区（以前是写死的 800x600）。
pub fn select_region(title: &str) -> Option<Geometry> {
    if let Some(g) = try_slurp() {
        return Some(g);
    }
    select_region_interactive(title)
}
