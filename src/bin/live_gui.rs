use opencv::{core, highgui, imgcodecs, imgproc, prelude::*};
use std::process::Command;
use std::sync::{Arc, Mutex};
use std::thread;

#[derive(Debug, Clone, Copy, PartialEq)]
enum EditMode {
    None,
    P1,
    E1,
    DrawRuler1,
    DrawRuler2,
}

#[derive(Debug, Clone, Copy)]
struct AppState {
    edit_mode: EditMode,
    manual_p1: Option<core::Point>,
    manual_e1: Option<core::Point>,
    manual_cam_rect: Option<core::Rect>,
    drag_start: Option<core::Point>,
    current_angle: f64,
    wind: f64,
    locked_px_per_unit: Option<f64>,
    /// 大地图模式:整张地图的宽度代表多少游戏距离(12/16/18/20/22)
    map_units: f64,
    /// 当前是否大地图模式(运行时可切换)
    big_map: bool,
    /// 显示缩放系数:canvas 像素 / 截图像素,切换模式时重算
    disp_scale: f64,
    /// 当前地图源尺寸(截图像素)
    src_w: i32,
    src_h: i32,
    auto_angle: bool,
    is_fixed_angle: bool,
    exit_requested: bool,
    switch_requested: bool,
}

use tnt_comput::physics::*;

fn compute_fixed_trajectory(dx_units: f64, dy_units: f64, angle_deg: f64, wind: f64) -> Option<(f64, f64)> {
    let mut eff_angle = angle_deg;
    if eff_angle > 90.0 {
        eff_angle = 180.0 - eff_angle;
    }

    let wind_power = wind; // 用户输入的是相对风力（正=顺风，负=逆风），物理引擎统一按向右打计算，所以直接传入即可
    let dist = dx_units.abs();

    // 调用全新的底层物理引擎，同时传入高低差 dy_units
    let final_power = power_for_angle(eff_angle, dist, dy_units, wind_power)?;
    Some((final_power, angle_deg))
}

fn compute_trajectory(dx_units: f64, dy_units: f64, angle_deg: f64, wind: f64) -> Option<(f64, f64)> {
    let mut eff_angle = angle_deg;
    if eff_angle > 90.0 {
        eff_angle = 180.0 - eff_angle;
    }

    let is_reverse = dx_units < 0.0;
    let wind_power = wind; // 用户输入的是相对风力（正=顺风，负=逆风）
    let dist = dx_units.abs();

    // 使用新的 power_for_angle 和 calc_angle 传递 dy_units
    let base_power = power_for_angle(eff_angle, dist, dy_units, 0.0)?;
    let mut final_angle = calc_angle(dist, dy_units, base_power, wind_power, eff_angle);

    final_angle = final_angle.clamp(15.0, 89.0);

    if is_reverse {
        final_angle = 180.0 - final_angle;
    }

    Some((base_power, final_angle))
}

fn draw_btn(
    canvas: &mut core::Mat,
    rect: core::Rect,
    label: &str,
    is_active: bool,
) -> opencv::Result<()> {
    let color = if is_active {
        core::Scalar::new(0.0, 200.0, 255.0, 0.0)
    } else {
        core::Scalar::new(80.0, 80.0, 80.0, 0.0)
    };
    let text_color = if is_active {
        core::Scalar::new(0.0, 0.0, 0.0, 0.0)
    } else {
        core::Scalar::new(255.0, 255.0, 255.0, 0.0)
    };

    imgproc::rectangle(canvas, rect, color, -1, imgproc::LINE_8, 0)?;
    imgproc::rectangle(
        canvas,
        rect,
        core::Scalar::new(200.0, 200.0, 200.0, 0.0),
        1,
        imgproc::LINE_8,
        0,
    )?;

    let mut baseline = 0;
    let size = imgproc::get_text_size(label, imgproc::FONT_HERSHEY_SIMPLEX, 0.5, 1, &mut baseline)?;
    let text_x = rect.x + (rect.width - size.width) / 2;
    let text_y = rect.y + (rect.height + size.height) / 2;

    imgproc::put_text(
        canvas,
        label,
        core::Point::new(text_x, text_y),
        imgproc::FONT_HERSHEY_SIMPLEX,
        0.5,
        text_color,
        1,
        imgproc::LINE_AA,
        false,
    )?;
    Ok(())
}

fn is_inside(x: i32, y: i32, rect: core::Rect) -> bool {
    x >= rect.x && x <= rect.x + rect.width && y >= rect.y && y <= rect.y + rect.height
}

/// 用户交互式框选截图后，通过全屏截图 + 模板匹配反推出实际的屏幕坐标
fn find_screen_position(crop_img: &core::Mat) -> Option<(i32, i32, i32, i32)> {
    let full_path = "/tmp/tnt_full_screen.png";
    // 静默全屏截图
    #[cfg(target_os = "macos")]
    let _ = Command::new("screencapture").arg("-x").arg(full_path).status();
    #[cfg(target_os = "linux")]
    let _ = Command::new("sh").arg("-c").arg(format!("grim {}", full_path)).status();

    let full_img = imgcodecs::imread(full_path, imgcodecs::IMREAD_COLOR).ok()?;
    if full_img.empty() || crop_img.cols() > full_img.cols() || crop_img.rows() > full_img.rows() {
        return None;
    }

    let mut match_result = core::Mat::default();
    imgproc::match_template(&full_img, crop_img, &mut match_result, imgproc::TM_CCOEFF_NORMED, &core::no_array()).ok()?;
    let mut max_val = 0.0;
    let mut max_loc = core::Point::new(0, 0);
    core::min_max_loc(&match_result, None, Some(&mut max_val), None, Some(&mut max_loc), &core::no_array()).ok()?;

    if max_val > 0.5 {
        println!("✅ 模板匹配成功 (score={:.2})，屏幕坐标: ({},{}) {}x{}", max_val, max_loc.x, max_loc.y, crop_img.cols(), crop_img.rows());
        Some((max_loc.x, max_loc.y, crop_img.cols(), crop_img.rows()))
    } else {
        println!("⚠️  模板匹配得分过低 ({:.2})，使用 (0,0) 作为默认坐标", max_val);
        None
    }
}

#[cfg(target_os = "linux")]
fn select_crop_interactive(path: &str) -> bool {
    Command::new("sh")
        .arg("-c")
        .arg(format!("grim -g \"$(slurp)\" {}", path))
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

#[cfg(target_os = "macos")]
fn select_crop_interactive(path: &str) -> bool {
    Command::new("screencapture")
        .arg("-i")
        .arg(path)
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

#[cfg(target_os = "linux")]
fn capture_rect_to_file(geo: (i32, i32, i32, i32), path: &str) {
    let _ = Command::new("sh")
        .arg("-c")
        .arg(format!("grim -g \"{},{} {}x{}\" -t ppm {}.tmp && mv {}.tmp {}", geo.0, geo.1, geo.2, geo.3, path, path, path))
        .status();
}

#[cfg(target_os = "macos")]
fn capture_rect_to_file(geo: (i32, i32, i32, i32), path: &str) {
    let _ = Command::new("screencapture")
        .arg("-R")
        .arg(format!("{},{},{},{}", geo.0, geo.1, geo.2, geo.3))
        .arg("-x")
        .arg(path)
        .status();
}

fn main() -> opencv::Result<()> { // Recognizer moved to bg thread
    #[cfg(target_os = "macos")]
    println!("=== 🍎 Mac OS 环境检测成功，已自动切换原生 screencapture 截图引擎 ===");

    println!("👉 [模式选择] 回车 = 小地图模式(实时刷新); 输入 2 = 大地图模式(框选整张静态地图):");
    let mut mode_buf = String::new();
    let _ = std::io::stdin().read_line(&mut mode_buf);
    let big_map_mode = mode_buf.trim() == "2";

    if big_map_mode {
        println!("👉 [步骤 1/2] 请框选【整张游戏地图】区域(整屏宽度 = 12/16/18/20/22 按钮)
    提示: 画面实时刷新;点我方/敌方后直接出力度");
    } else {
        println!("👉 [步骤 1/2] 请在屏幕上框选【左上角小地图】区域...");
    }
    let map_crop_path = "/tmp/tnt_selected_map.png";
    select_crop_interactive(map_crop_path);

    let initial_img = match imgcodecs::imread(map_crop_path, imgcodecs::IMREAD_COLOR) {
        Ok(m) if !m.empty() => m,
        _ => {
            println!("❌ 抓取【小地图】区域失败或取消！");
            return Ok(());
        }
    };
    let t_w = initial_img.cols();
    let t_h = initial_img.rows();
    // 通过模板匹配反推屏幕坐标，回退到 (0,0)
    let map_geo = find_screen_position(&initial_img).unwrap_or((0, 0, t_w, t_h));
    println!("📍 小地图屏幕区域: ({},{}) {}x{}", map_geo.0, map_geo.1, map_geo.2, map_geo.3);

    println!("👉 [步骤 2/2] 请在屏幕上框选【右下角/角度/数值】区域...");
    let power_crop_path = "/tmp/tnt_selected_power.png";
    select_crop_interactive(power_crop_path);

    let power_img = imgcodecs::imread(power_crop_path, imgcodecs::IMREAD_COLOR)
        .ok()
        .filter(|m| !m.empty());
    let power_geo = power_img.as_ref().and_then(|img| find_screen_position(img));
    if let Some(pg) = power_geo {
        println!("📍 数值区域屏幕坐标: ({},{}) {}x{}", pg.0, pg.1, pg.2, pg.3);
    }


    let window_name = "TNT Assistant HUD";
    highgui::named_window(window_name, highgui::WINDOW_AUTOSIZE)?;

    // 显示缩放:大地图缩到 ~452px 宽(和小地图显示尺寸一致),小地图沿用放大规则
    let scale = if big_map_mode {
        (452.0 / t_w as f64).min(2.0)
    } else if t_w > 600 {
        1.0
    } else {
        2.0
    };

    let app_state = Arc::new(Mutex::new(AppState {
        edit_mode: EditMode::None,
        manual_p1: None,
        manual_e1: None,
        manual_cam_rect: None,
        drag_start: None,
        current_angle: 45.0,
        wind: 0.0,
        locked_px_per_unit: None,
        map_units: 12.0,
        big_map: big_map_mode,
        disp_scale: scale,
        src_w: t_w,
        src_h: t_h,
        auto_angle: true,
        is_fixed_angle: true,
        exit_requested: false,
        switch_requested: false,
    }));

    let map_w_display = (t_w as f64 * scale) as i32;

    // 大地图模式:整屏宽默认 12 距,启动即自动锁尺(12/16/18/20/22 按钮可切换)
    if big_map_mode {
        app_state.lock().unwrap().locked_px_per_unit = Some(t_w as f64 / 12.0);
    }

    let btn_p1 = core::Rect::new(map_w_display + 20, 30, 110, 40);
    let btn_e1 = core::Rect::new(map_w_display + 140, 30, 110, 40);
    // 大/小地图模式切换(点击后弹交互框选新区域)
    let btn_mode_switch = core::Rect::new(map_w_display + 260, 30, 110, 40);

    let btn_lock_ruler = core::Rect::new(map_w_display + 20, 80, 230, 40);
    let btn_draw_ruler = core::Rect::new(map_w_display + 20, 130, 230, 35);

    let btn_exit = core::Rect::new(map_w_display + 150, 5, 100, 30);

    let btn_clear = core::Rect::new(map_w_display + 20, 210, 230, 25);

    // 大地图模式:整屏距离档位(小地图模式下仅占位,点击无效)
    let btn_u12 = core::Rect::new(map_w_display + 20, 175, 52, 30);
    let btn_u16 = core::Rect::new(map_w_display + 77, 175, 52, 30);
    let btn_u18 = core::Rect::new(map_w_display + 134, 175, 52, 30);
    let btn_u20 = core::Rect::new(map_w_display + 191, 175, 52, 30);
    let btn_u22 = core::Rect::new(map_w_display + 248, 175, 52, 30);

    // Preset Angle Buttons
    let btn_a20 = core::Rect::new(map_w_display + 20, 240, 50, 30);
    let btn_a30 = core::Rect::new(map_w_display + 80, 240, 50, 30);
    let btn_a45 = core::Rect::new(map_w_display + 140, 240, 50, 30);
    let btn_a50 = core::Rect::new(map_w_display + 200, 240, 50, 30);

    let btn_a60 = core::Rect::new(map_w_display + 20, 275, 50, 30);
    let btn_a65 = core::Rect::new(map_w_display + 80, 275, 50, 30);
    let btn_a70 = core::Rect::new(map_w_display + 140, 275, 50, 30);
    let btn_a75 = core::Rect::new(map_w_display + 200, 275, 50, 30);

    // Fine-tune buttons
    let btn_ang_m5 = core::Rect::new(map_w_display + 20, 315, 40, 35);
    let btn_ang_minus = core::Rect::new(map_w_display + 65, 315, 35, 35);
    let rect_ang_text = core::Rect::new(map_w_display + 105, 315, 60, 35);
    let btn_ang_plus = core::Rect::new(map_w_display + 170, 315, 35, 35);
    let btn_ang_p5 = core::Rect::new(map_w_display + 210, 315, 40, 35);

    let btn_wind_m1 = core::Rect::new(map_w_display + 20, 355, 45, 30);
    let btn_wind_m01 = core::Rect::new(map_w_display + 75, 355, 45, 30);
    let rect_wind_text = core::Rect::new(map_w_display + 125, 355, 60, 30);
    let btn_wind_p01 = core::Rect::new(map_w_display + 190, 355, 45, 30);
    let btn_wind_p1 = core::Rect::new(map_w_display + 245, 355, 45, 30);

    let btn_auto_angle = core::Rect::new(map_w_display + 20, 395, 230, 30);

    let state_cb = app_state.clone();
    highgui::set_mouse_callback(
        window_name,
        Some(Box::new(move |event, x, y, _flags| {
            let mut st = state_cb.lock().unwrap();

            if event == highgui::EVENT_LBUTTONDOWN {
                // Buttons
                if is_inside(x, y, btn_p1) {
                    st.edit_mode = if st.edit_mode == EditMode::P1 {
                        EditMode::None
                    } else {
                        EditMode::P1
                    };
                } else if is_inside(x, y, btn_e1) {
                    st.edit_mode = if st.edit_mode == EditMode::E1 {
                        EditMode::None
                    } else {
                        EditMode::E1
                    };
                } else if is_inside(x, y, btn_lock_ruler) {
                    if st.locked_px_per_unit.is_some() {
                        st.locked_px_per_unit = None;
                    } else {
                        st.locked_px_per_unit = Some(0.0);
                    }
                } else if is_inside(x, y, btn_draw_ruler) {
                    st.edit_mode = if st.edit_mode == EditMode::DrawRuler1
                        || st.edit_mode == EditMode::DrawRuler2
                    {
                        EditMode::None
                    } else {
                        EditMode::DrawRuler1
                    };
                } else if is_inside(x, y, btn_clear) {
                    // 只清标记。比例尺锁定和标记无关,保留(清了就会出现
                    // "明明锁定了却提示请先锁定"的问题)
                    st.manual_p1 = None;
                    st.manual_e1 = None;
                    st.manual_cam_rect = None;
                    st.edit_mode = EditMode::None;
                } else if is_inside(x, y, btn_a20) {
                    st.current_angle = 20.0;
                    st.auto_angle = false;
                } else if is_inside(x, y, btn_a30) {
                    st.current_angle = 30.0;
                    st.auto_angle = false;
                } else if is_inside(x, y, btn_a45) {
                    st.current_angle = 45.0;
                    st.auto_angle = false;
                } else if is_inside(x, y, btn_a50) {
                    st.current_angle = 50.0;
                    st.auto_angle = false;
                } else if is_inside(x, y, btn_a60) {
                    st.current_angle = 60.0;
                    st.auto_angle = false;
                } else if is_inside(x, y, btn_a65) {
                    st.current_angle = 65.0;
                    st.auto_angle = false;
                } else if is_inside(x, y, btn_a70) {
                    st.current_angle = 70.0;
                    st.auto_angle = false;
                } else if is_inside(x, y, btn_a75) {
                    st.current_angle = 75.0;
                    st.auto_angle = false;
                } else if is_inside(x, y, btn_ang_m5) {
                    st.current_angle = (st.current_angle - 5.0).max(0.0);
                    st.auto_angle = false;
                } else if is_inside(x, y, btn_ang_minus) {
                    st.current_angle = (st.current_angle - 1.0).max(0.0);
                    st.auto_angle = false;
                } else if is_inside(x, y, btn_ang_plus) {
                    st.current_angle = (st.current_angle + 1.0).min(180.0);
                    st.auto_angle = false;
                } else if is_inside(x, y, btn_ang_p5) {
                    st.current_angle = (st.current_angle + 5.0).min(180.0);
                    st.auto_angle = false;
                } else if is_inside(x, y, btn_auto_angle) {
                    st.auto_angle = true;
                } else if is_inside(x, y, btn_wind_m1) {
                    st.wind -= 1.0;
                } else if is_inside(x, y, btn_wind_m01) {
                    st.wind -= 0.1;
                } else if is_inside(x, y, btn_wind_p01) {
                    st.wind += 0.1;
                } else if is_inside(x, y, btn_wind_p1) {
                    st.wind += 1.0;
                } else if is_inside(x, y, btn_u12)
                    || is_inside(x, y, btn_u16)
                    || is_inside(x, y, btn_u18)
                    || is_inside(x, y, btn_u20)
                    || is_inside(x, y, btn_u22)
                {
                    // 大地图模式:整屏宽度 = 所选距离,立刻按新档位重锁尺子。
                    // 小地图模式不响应(比例尺走手动标尺),只吞掉这次点击。
                    let units = if is_inside(x, y, btn_u12) {
                        12.0
                    } else if is_inside(x, y, btn_u16) {
                        16.0
                    } else if is_inside(x, y, btn_u18) {
                        18.0
                    } else if is_inside(x, y, btn_u20) {
                        20.0
                    } else {
                        22.0
                    };
                    if st.big_map {
                        st.map_units = units;
                        st.locked_px_per_unit = Some(st.src_w as f64 / units);
                    }
                } else if is_inside(x, y, btn_mode_switch) {
                    // 运行时切换大/小地图:标记 switch_requested,
                    // 由 UI 线程弹交互框选(回调里不能阻塞,会死锁)
                    st.switch_requested = true;
                } else if is_inside(x, y, btn_exit) {
                    st.exit_requested = true;
                }
                // Map click
                else if x < map_w_display {
                    let pt = core::Point::new(x, y);
                    match st.edit_mode {
                        EditMode::P1 => {
                            st.manual_p1 = Some(pt);
                            st.edit_mode = EditMode::None;
                        }
                        EditMode::E1 => {
                            st.manual_e1 = Some(pt);
                            st.edit_mode = EditMode::None;
                        }
                        EditMode::DrawRuler1 => {
                            st.drag_start = Some(pt);
                            st.edit_mode = EditMode::DrawRuler2;
                        }
                        EditMode::DrawRuler2 => {
                            if let Some(start) = st.drag_start {
                                let min_x = start.x.min(pt.x);
                                let max_x = start.x.max(pt.x);
                                let min_y = start.y.min(pt.y);
                                let max_y = start.y.max(pt.y);
                                st.manual_cam_rect = Some(core::Rect::new(
                                    (min_x as f64 / st.disp_scale) as i32,
                                    (min_y as f64 / st.disp_scale) as i32,
                                    ((max_x - min_x) as f64 / st.disp_scale) as i32,
                                    ((max_y - min_y) as f64 / st.disp_scale) as i32,
                                ));
                            }
                            st.drag_start = None;
                            st.edit_mode = EditMode::None;
                            // Auto-lock the ruler with the newly drawn box (0.0 triggers evaluation in drawing loop)
                            st.locked_px_per_unit = Some(0.0);
                        }
                        _ => {}
                    }
                }
            } else if event == highgui::EVENT_MOUSEMOVE {
                if st.edit_mode == EditMode::DrawRuler2 {
                    if let Some(start) = st.drag_start {
                        let min_x = start.x.min(x);
                        let max_x = start.x.max(x);
                        let min_y = start.y.min(y);
                        let max_y = start.y.max(y);
                        // Store in original minimap coordinates to show live preview
                        st.manual_cam_rect = Some(core::Rect::new(
                            (min_x as f64 / st.disp_scale) as i32,
                            (min_y as f64 / st.disp_scale) as i32,
                            ((max_x - min_x) as f64 / st.disp_scale) as i32,
                            ((max_y - min_y) as f64 / st.disp_scale) as i32,
                        ));
                    }
                }
            }
        })),
    )?;

    let is_running = Arc::new(std::sync::atomic::AtomicBool::new(true));
    let r_clone = is_running.clone();

    let cap_time_ms = Arc::new(std::sync::Mutex::new(0u128));
    let cap_time_ms_clone = cap_time_ms.clone();

    // 地图截图区域放共享变量:运行时切换模式后,后台线程下一轮就用新区域
    let map_geo_shared = Arc::new(std::sync::Mutex::new(map_geo));
    let map_geo_shared_clone = map_geo_shared.clone();
    let power_geo_clone = power_geo.clone();

    let shared_map = Arc::new(std::sync::Mutex::new(None::<core::Mat>));
    let shared_map_clone = shared_map.clone();
    let shared_recognized_angle = Arc::new(std::sync::Mutex::new(None::<i32>));
    let shared_recognized_angle_clone = shared_recognized_angle.clone();

    let app_state_bg = app_state.clone();

    thread::spawn(move || {
        let recognizer =
            tnt_comput::ui::UiRecognizer::new("src/templates").expect("Failed to init recognizer");
        let mut last_recog_time = std::time::Instant::now();

        #[cfg(target_os = "linux")]
        let (map_path, power_path) = ("/tmp/tnt_map.ppm", "/tmp/tnt_power.ppm");
        #[cfg(target_os = "macos")]
        let (map_path, power_path) = ("/tmp/tnt_map.png", "/tmp/tnt_power.png");

        while r_clone.load(std::sync::atomic::Ordering::Relaxed) {
            let t0 = std::time::Instant::now();

            // 实测(2560x1440):单次小区域截图 ~51ms,而大图的 PNG 编解码要贵得多,
            // 所以"并集成一张大图"反而不如"各自抓小图"快。
            // 地图每轮都抓(小地图/大地图都实时),保证点击跟手;
            // 角度框 200ms 一次(OCR ~5Hz,同步足够)。
            capture_rect_to_file(*map_geo_shared_clone.lock().unwrap(), map_path);
            if let Ok(m) = imgcodecs::imread(map_path, imgcodecs::IMREAD_COLOR) {
                if !m.empty() {
                    if let Ok(mut lock) = shared_map_clone.lock() {
                        *lock = Some(m);
                    }
                }
            }

            if let Some(pg) = power_geo_clone {
                if last_recog_time.elapsed().as_millis() > 200 {
                    capture_rect_to_file(pg, power_path);
                    if let Ok(p) = imgcodecs::imread(power_path, imgcodecs::IMREAD_COLOR) {
                        if !p.empty() {
                            if let Ok(Some(val)) = recognizer.recognize_angle_digit(&p) {
                                if let Ok(mut lock) = shared_recognized_angle_clone.lock() {
                                    *lock = Some(val);
                                }
                                // Auto-sync angle if enabled
                                if val >= 10 && val <= 90 {
                                    if let Ok(mut m_state) = app_state_bg.lock() {
                                        if m_state.auto_angle {
                                            m_state.current_angle = val as f64;
                                        }
                                    }
                                }
                            }
                        }
                    }
                    last_recog_time = std::time::Instant::now();
                }
            }

            if let Ok(mut lock) = cap_time_ms_clone.lock() {
                *lock = t0.elapsed().as_millis();
            }
            std::thread::sleep(std::time::Duration::from_millis(40));
        }
    });

    let mut first_show = true;

    // 立即显示初始化画面，防止 Mac 窗口引擎死锁
    let mut init_canvas = core::Mat::new_rows_cols_with_default(
        400,
        800,
        core::CV_8UC3,
        core::Scalar::new(30.0, 30.0, 30.0, 0.0),
    )?;
    imgproc::put_text(
        &mut init_canvas,
        "Initializing Background Capture...",
        core::Point::new(50, 200),
        imgproc::FONT_HERSHEY_SIMPLEX,
        1.0,
        core::Scalar::new(0.0, 255.0, 0.0, 0.0),
        2,
        imgproc::LINE_AA,
        false,
    )?;
    highgui::imshow(window_name, &init_canvas)?;
    highgui::set_window_property(window_name, highgui::WND_PROP_TOPMOST, 1.0)?;
    highgui::wait_key(100)?;

    let mut wind_input_buf = String::new();
    // 弹道缓存：点位/角度/风没变就不重算。定角模式约 70 次全程仿真，
    // 不可达时更是 600+ 次采样，每帧重算是纯浪费。
    let mut traj_memo: Option<((f64, f64, f64, f64, bool), Option<(f64, f64)>)> = None;
    let mut fps_t0 = std::time::Instant::now();
    let mut last_toggle_time = std::time::Instant::now() - std::time::Duration::from_secs(1); // 防抖时间戳

    loop {
        let loop_t0 = std::time::Instant::now();

        let img = {
            let lock = shared_map.lock().unwrap();
            lock.as_ref().and_then(|m| m.try_clone().ok())
        };

        let t_io = loop_t0.elapsed().as_millis();
        let t1 = std::time::Instant::now();

        let power_recognized_val = *shared_recognized_angle.lock().unwrap();

        let t_recog = t1.elapsed().as_millis();
        let t2 = std::time::Instant::now();

        let canvas_w = map_w_display + 310;
        let st = *app_state.lock().unwrap();
        // 模式切换后这些会变,每帧从状态里取(遮蔽外层的启动值)
        let t_w = st.src_w;
        let t_h = st.src_h;
        let scale = st.disp_scale;
        let map_h_display = (t_h as f64 * scale) as i32;
        let canvas_h_target = map_h_display.max(580);
        let mut canvas = core::Mat::new_rows_cols_with_default(
            canvas_h_target,
            canvas_w,
            core::CV_8UC3,
            core::Scalar::new(30.0, 30.0, 30.0, 0.0),
        )?;

        if let Some(minimap) = img {
            let mut map_display = core::Mat::default();
            imgproc::resize(
                &minimap,
                &mut map_display,
                core::Size::new(map_w_display, (t_h as f64 * scale) as i32),
                0.0,
                0.0,
                imgproc::INTER_LINEAR,
            )?;

            for y in 0..map_display.rows() {
                if let (Ok(src_row), Ok(dst_row)) = (
                    map_display.at_row::<core::Vec3b>(y),
                    canvas.at_row_mut::<core::Vec3b>(y),
                ) {
                    let len = src_row.len();
                    dst_row[0..len].copy_from_slice(src_row);
                }
            }



            let cam_rect = st.manual_cam_rect.unwrap_or(core::Rect::new(0, 0, t_w, t_h));

            let to_scr = |p: core::Point| {
                core::Point::new((p.x as f64 * scale) as i32, (p.y as f64 * scale) as i32)
            };

            // 绘制摄像机白框（黄框代表手动）
            let cam_p1 = to_scr(core::Point::new(cam_rect.x, cam_rect.y));
            let cam_p2 = to_scr(core::Point::new(
                cam_rect.x + cam_rect.width,
                cam_rect.y + cam_rect.height,
            ));
            if st.manual_cam_rect.is_some() && st.locked_px_per_unit.is_none() {
                let box_color = core::Scalar::new(0.0, 255.0, 255.0, 0.0);
                let _ = imgproc::rectangle(
                    &mut canvas,
                    core::Rect::new(cam_p1.x, cam_p1.y, cam_p2.x - cam_p1.x, cam_p2.y - cam_p1.y),
                    box_color,
                    2,
                    imgproc::LINE_8,
                    0,
                );

                let ruler_width = cam_p2.x - cam_p1.x;
                for i in 1..12 {
                    let tick_x = cam_p1.x + (ruler_width as f64 * (i as f64 / 12.0)) as i32;
                    let _ = imgproc::line(&mut canvas, core::Point::new(tick_x, cam_p1.y), core::Point::new(tick_x, cam_p1.y + 10), box_color, 1, imgproc::LINE_AA, 0);
                    let _ = imgproc::line(&mut canvas, core::Point::new(tick_x, cam_p2.y - 10), core::Point::new(tick_x, cam_p2.y), box_color, 1, imgproc::LINE_AA, 0);
                }

                let cam_txt = format!("CAMERA {}x{}", cam_rect.width, cam_rect.height);
                let _ = imgproc::put_text(&mut canvas, &cam_txt, core::Point::new(cam_p1.x, (cam_p1.y - 5).max(10)), imgproc::FONT_HERSHEY_SIMPLEX, 0.4, core::Scalar::new(0.0, 255.0, 255.0, 0.0), 1, imgproc::LINE_8, false);
            }

            let mut current_px_per_unit = cam_rect.width as f64 / 12.0;
            if let Some(locked) = st.locked_px_per_unit {
                if locked == 0.0 {
                    // 大地图且没画过手动标尺:整屏宽 = map_units 距;
                    // 画了标尺就仍按标尺算,两种都能用
                    let v = if st.big_map && st.manual_cam_rect.is_none() {
                        t_w as f64 / st.map_units
                    } else {
                        current_px_per_unit
                    };
                    current_px_per_unit = v;
                    if let Ok(mut m_state) = app_state.lock() {
                        m_state.locked_px_per_unit = Some(v);
                    }
                } else {
                    current_px_per_unit = locked;
                }
            }
            let px_per_unit = current_px_per_unit;

            // 全手动模式：图像识别只剩角度数字 OCR，点位全部靠点击标注。
            let p1 = st.manual_p1;
            let e1 = st.manual_e1;

            let draw_pt =
                |c: &mut core::Mat, pt: core::Point, label: &str, is_red: bool, is_manual: bool| {
                    // 手动点击的坐标已经是 canvas 像素，直接画。
                    let cx = pt.x;
                    let cy = pt.y;
                    let color = if is_red {
                        core::Scalar::new(50.0, 50.0, 255.0, 0.0)
                    } else {
                        core::Scalar::new(255.0, 255.0, 0.0, 0.0)
                    };
                    let thickness = if is_manual { -1 } else { 2 };
                    // radius 6 for smaller dots
                    let _ = imgproc::circle(
                        c,
                        core::Point::new(cx, cy),
                        6,
                        color,
                        thickness,
                        imgproc::LINE_AA,
                        0,
                    );
                    let _ = imgproc::put_text(
                        c,
                        label,
                        core::Point::new(cx - 15, cy - 12),
                        imgproc::FONT_HERSHEY_SIMPLEX,
                        0.6,
                        core::Scalar::new(255.0, 255.0, 255.0, 0.0),
                        2,
                        imgproc::LINE_AA,
                        false,
                    );
                };

            if let Some(p) = p1 {
                draw_pt(&mut canvas, p, "我方 (My)", false, st.manual_p1.is_some());
            }
            if let Some(e) = e1 {
                draw_pt(&mut canvas, e, "敌方 (Enemy)", true, st.manual_e1.is_some());
            }

            let mut y_offset = 480;
            let mut draw_result = |p: core::Point, e: core::Point| {
                // Ensure ruler is locked
                if st.locked_px_per_unit.is_none() {
                    imgproc::rectangle(
                        &mut canvas,
                        core::Rect::new(map_w_display + 10, y_offset - 35, 250, 80),
                        core::Scalar::new(0.0, 0.0, 50.0, 0.0),
                        -1,
                        imgproc::LINE_8,
                        0,
                    )
                    .unwrap();
                    imgproc::rectangle(
                        &mut canvas,
                        core::Rect::new(map_w_display + 10, y_offset - 35, 250, 80),
                        core::Scalar::new(0.0, 0.0, 255.0, 0.0),
                        2,
                        imgproc::LINE_8,
                        0,
                    )
                    .unwrap();
                    let _ = imgproc::put_text(
                        &mut canvas,
                        "请先锁定距离尺!",
                        core::Point::new(map_w_display + 20, y_offset + 10),
                        imgproc::FONT_HERSHEY_SIMPLEX,
                        0.8,
                        core::Scalar::new(0.0, 0.0, 255.0, 0.0),
                        2,
                        imgproc::LINE_AA,
                        false,
                    );
                    return;
                }

                // Convert pixel diff to original minimap scale, then apply px_per_unit
                let orig_dx = (e.x - p.x) as f64 / scale;
                let dx = orig_dx / px_per_unit;

                let orig_dy = -(e.y - p.y) as f64 / scale;
                let dy = orig_dy / px_per_unit;

                let key = (dx, dy, st.current_angle, st.wind, st.is_fixed_angle);
                let trajectory_res = match traj_memo {
                    Some((k, v)) if k == key => v,
                    _ => {
                        let r = if st.is_fixed_angle {
                            compute_fixed_trajectory(dx, dy, st.current_angle, st.wind)
                        } else {
                            compute_trajectory(dx, dy, st.current_angle, st.wind)
                        };
                        traj_memo = Some((key, r));
                        r
                    }
                };

                match trajectory_res {
                    Some((force, final_angle)) => {
                        // Draw a beautiful background box for the force recommendation
                        imgproc::rectangle(
                            &mut canvas,
                            core::Rect::new(map_w_display + 10, y_offset - 35, 300, 80),
                            core::Scalar::new(0.0, 50.0, 0.0, 0.0),
                            -1,
                            imgproc::LINE_8,
                            0,
                        )
                        .unwrap();
                        imgproc::rectangle(
                            &mut canvas,
                            core::Rect::new(map_w_display + 10, y_offset - 35, 300, 80),
                            core::Scalar::new(0.0, 255.0, 0.0, 0.0),
                            2,
                            imgproc::LINE_8,
                            0,
                        )
                        .unwrap();

                        let mode_str = if st.is_fixed_angle {
                            "[定角打法]"
                        } else {
                            "[变角打法]"
                        };
                        let title = format!(
                            "{}  风力: {:.1}  X距: {:.1}  Y高: {:.1}",
                            mode_str,
                            st.wind,
                            dx.abs(),
                            dy
                        );
                        let _ = imgproc::put_text(
                            &mut canvas,
                            &title,
                            core::Point::new(map_w_display + 15, y_offset - 10),
                            imgproc::FONT_HERSHEY_SIMPLEX,
                            0.45,
                            core::Scalar::new(200.0, 200.0, 200.0, 0.0),
                            1,
                            imgproc::LINE_AA,
                            false,
                        );

                        let res_txt = if st.is_fixed_angle {
                            format!("锁定: {:.0}° 力度: {:.1} 2/3: {:.1}", final_angle, force, force * 2.0 / 3.0)
                        } else {
                            format!("推荐: {:.0}° 力度: {:.1} 2/3: {:.1}", final_angle, force, force * 2.0 / 3.0)
                        };
                        let _ = imgproc::put_text(
                            &mut canvas,
                            &res_txt,
                            core::Point::new(map_w_display + 15, y_offset + 15),
                            imgproc::FONT_HERSHEY_SIMPLEX,
                            0.55,
                            core::Scalar::new(0.0, 255.0, 0.0, 0.0),
                            2,
                            imgproc::LINE_AA,
                            false,
                        );

                        // Draw trajectory dots on the minimap
                        let mut draw_angle = final_angle;
                        let is_reverse = e.x < p.x;
                        if is_reverse && draw_angle <= 90.0 {
                            draw_angle = 180.0 - draw_angle;
                        } else if !is_reverse && draw_angle > 90.0 {
                            draw_angle = 180.0 - draw_angle;
                        }

                        // 物理引擎原生支持真实的物理世界坐标系（角度>90代表向左，风向带符号）
                        // 用户输入的是相对风力（正=顺风），但在画图时，我们要把它转成绝对世界的风向。
                        // 如果向左打，顺风就是向左吹（绝对世界里的负风向）。
                        let sim_wind = if is_reverse { -st.wind } else { st.wind };
                        let path = tnt_comput::physics::simulate_path(draw_angle, force, sim_wind);
                        for (sim_x, sim_y) in path {
                            let img_x = p.x as f64 + sim_x * scale * px_per_unit;
                            let img_y = p.y as f64 - sim_y * scale * px_per_unit;

                            // Stop if out of bounds of the minimap
                            if img_x < 0.0 || img_x > map_w_display as f64 || img_y > map_h_display as f64 {
                                break;
                            }
                            if img_y >= 0.0 {
                                let _ = imgproc::circle(
                                    &mut canvas,
                                    core::Point::new(img_x as i32, img_y as i32),
                                    2,
                                    core::Scalar::new(255.0, 255.0, 0.0, 0.0),
                                    -1,
                                    imgproc::LINE_AA,
                                    0,
                                );
                            }
                        }
                    }
                    None => {
                        // Draw Unreachable box
                        imgproc::rectangle(
                            &mut canvas,
                            core::Rect::new(map_w_display + 10, y_offset - 35, 300, 80),
                            core::Scalar::new(0.0, 0.0, 50.0, 0.0),
                            -1,
                            imgproc::LINE_8,
                            0,
                        )
                        .unwrap();
                        let _ = imgproc::put_text(
                            &mut canvas,
                            "❌ 目标不可达 (Unreachable)",
                            core::Point::new(map_w_display + 20, y_offset + 5),
                            imgproc::FONT_HERSHEY_SIMPLEX,
                            0.7,
                            core::Scalar::new(0.0, 0.0, 255.0, 0.0),
                            2,
                            imgproc::LINE_AA,
                            false,
                        );
                    }
                }

                if let Some(pval) = power_recognized_val {
                    let power_txt = format!("右下角实时读数: {}", pval);
                    let _ = imgproc::put_text(
                        &mut canvas,
                        &power_txt,
                        core::Point::new(map_w_display + 15, y_offset + 38),
                        imgproc::FONT_HERSHEY_SIMPLEX,
                        0.55,
                        core::Scalar::new(0.0, 255.0, 255.0, 0.0),
                        2,
                        imgproc::LINE_AA,
                        false,
                    );
                }

                y_offset += 100;

                let pt_p = core::Point::new(p.x, p.y);
                let pt_e = core::Point::new(e.x, e.y);
                let pt_corner = core::Point::new(e.x, p.y);
                // Draw horizontal line (X distance)
                let _ = imgproc::line(
                    &mut canvas,
                    pt_p,
                    pt_corner,
                    core::Scalar::new(0.0, 255.0, 255.0, 0.0),
                    1,
                    imgproc::LINE_AA,
                    0,
                );
                // Draw vertical line (Y distance)
                let _ = imgproc::line(
                    &mut canvas,
                    pt_corner,
                    pt_e,
                    core::Scalar::new(255.0, 100.0, 0.0, 0.0),
                    1,
                    imgproc::LINE_AA,
                    0,
                );
            };

            if let (Some(p), Some(e)) = (p1, e1) {
                draw_result(p, e);
            }
        } else if img.is_none() {
            imgproc::put_text(
                &mut canvas,
                "Waiting for Capture...",
                core::Point::new(50, 50),
                imgproc::FONT_HERSHEY_SIMPLEX,
                0.8,
                core::Scalar::new(0.0, 0.0, 255.0, 0.0),
                2,
                imgproc::LINE_AA,
                false,
            )?;
        }

        let pval_str = if st.auto_angle {
            if let Some(val) = power_recognized_val {
                format!("实机同步: 开启 ({})", val)
            } else {
                "实机同步: 监听中...".to_string()
            }
        } else {
            "实机同步: 已暂停 (点击恢复)".to_string()
        };
        let _ = draw_btn(
            &mut canvas,
            btn_auto_angle,
            &pval_str,
            st.auto_angle,
        );

        draw_btn(
            &mut canvas,
            btn_p1,
            "我方 (My)",
            st.edit_mode == EditMode::P1,
        )?;
        draw_btn(
            &mut canvas,
            btn_e1,
            "敌方 (Enemy)",
            st.edit_mode == EditMode::E1,
        )?;
        draw_btn(
            &mut canvas,
            btn_mode_switch,
            if st.big_map { "MINI MAP" } else { "BIG MAP" },
            false,
        )?;

        let lock_label = if st.locked_px_per_unit.is_some() {
            "[已锁定] 解锁尺子"
        } else {
            "[未锁定] 锁定距离尺"
        };
        draw_btn(
            &mut canvas,
            btn_lock_ruler,
            lock_label,
            st.locked_px_per_unit.is_some(),
        )?;

        let ruler_lbl = if st.edit_mode == EditMode::DrawRuler1 {
            "[步骤1] 点击左上角"
        } else if st.edit_mode == EditMode::DrawRuler2 {
            "[步骤2] 点击右下角"
        } else {
            "手动框选 1屏幕宽"
        };
        draw_btn(
            &mut canvas,
            btn_draw_ruler,
            ruler_lbl,
            st.edit_mode == EditMode::DrawRuler1 || st.edit_mode == EditMode::DrawRuler2,
        )?;

        draw_btn(&mut canvas, btn_clear, "清空手动标记", false)?;

        // 大地图整屏距离档位(小地图模式下灰显不响应)
        draw_btn(
            &mut canvas,
            btn_u12,
            "12",
            st.big_map && (st.map_units - 12.0).abs() < 0.1,
        )?;
        draw_btn(
            &mut canvas,
            btn_u16,
            "16",
            st.big_map && (st.map_units - 16.0).abs() < 0.1,
        )?;
        draw_btn(
            &mut canvas,
            btn_u18,
            "18",
            st.big_map && (st.map_units - 18.0).abs() < 0.1,
        )?;
        draw_btn(
            &mut canvas,
            btn_u20,
            "20",
            st.big_map && (st.map_units - 20.0).abs() < 0.1,
        )?;
        draw_btn(
            &mut canvas,
            btn_u22,
            "22",
            st.big_map && (st.map_units - 22.0).abs() < 0.1,
        )?;
        // Draw preset angle buttons
        draw_btn(
            &mut canvas,
            btn_a20,
            "20°",
            (st.current_angle - 20.0).abs() < 0.5,
        )?;
        draw_btn(
            &mut canvas,
            btn_a30,
            "30°",
            (st.current_angle - 30.0).abs() < 0.5,
        )?;
        draw_btn(
            &mut canvas,
            btn_a45,
            "45°",
            (st.current_angle - 45.0).abs() < 0.5,
        )?;
        draw_btn(
            &mut canvas,
            btn_a50,
            "50°",
            (st.current_angle - 50.0).abs() < 0.5,
        )?;
        draw_btn(
            &mut canvas,
            btn_a60,
            "60°",
            (st.current_angle - 60.0).abs() < 0.5,
        )?;
        draw_btn(
            &mut canvas,
            btn_a65,
            "65°",
            (st.current_angle - 65.0).abs() < 0.5,
        )?;
        draw_btn(
            &mut canvas,
            btn_a70,
            "70°",
            (st.current_angle - 70.0).abs() < 0.5,
        )?;
        draw_btn(
            &mut canvas,
            btn_a75,
            "75°",
            (st.current_angle - 75.0).abs() < 0.5,
        )?;

        draw_btn(&mut canvas, btn_ang_m5, "-5", false)?;
        draw_btn(&mut canvas, btn_ang_minus, "-1", false)?;
        draw_btn(
            &mut canvas,
            rect_ang_text,
            &format!("{:.0}°", st.current_angle),
            true,
        )?;
        draw_btn(&mut canvas, btn_ang_plus, "+1", false)?;
        draw_btn(&mut canvas, btn_ang_p5, "+5", false)?;

        draw_btn(&mut canvas, btn_wind_m1, "-1.0", false)?;
        draw_btn(&mut canvas, btn_wind_m01, "-0.1", false)?;

        let wind_str = if !wind_input_buf.is_empty() {
            format!("缓冲: {}_", wind_input_buf)
        } else {
            format!("风: {:.1}", st.wind)
        };
        let _ = imgproc::put_text(
            &mut canvas,
            &wind_str,
            core::Point::new(rect_wind_text.x - 10, rect_wind_text.y + 20),
            imgproc::FONT_HERSHEY_SIMPLEX,
            0.45,
            core::Scalar::new(0.0, 255.0, 255.0, 0.0),
            1,
            imgproc::LINE_AA,
            false,
        );
        draw_btn(&mut canvas, btn_wind_p01, "+0.1", false)?;
        draw_btn(&mut canvas, btn_wind_p1, "+1.0", false)?;

        let hint_txt = "提示: 数字[回车]=风速, [空格]=角度, 支持负号";
        let _ = imgproc::put_text(
            &mut canvas,
            hint_txt,
            core::Point::new(map_w_display + 5, 420),
            imgproc::FONT_HERSHEY_SIMPLEX,
            0.38,
            core::Scalar::new(180.0, 255.0, 180.0, 0.0),
            1,
            imgproc::LINE_AA,
            false,
        );

        imgproc::rectangle(
            &mut canvas,
            btn_exit,
            core::Scalar::new(0.0, 0.0, 220.0, 0.0),
            -1,
            imgproc::LINE_8,
            0,
        )?;
        imgproc::put_text(
            &mut canvas,
            "X EXIT",
            core::Point::new(btn_exit.x + 15, btn_exit.y + 20),
            imgproc::FONT_HERSHEY_SIMPLEX,
            0.6,
            core::Scalar::new(255.0, 255.0, 255.0, 0.0),
            2,
            imgproc::LINE_AA,
            false,
        )?;

        let t_ui = t2.elapsed().as_millis();
        let bg_cap_ms = *cap_time_ms.lock().unwrap();
        let fps = 1000.0 / fps_t0.elapsed().as_millis().max(1) as f64;
        fps_t0 = std::time::Instant::now();
        
        let perf_txt = format!("FPS:{:.0} | IO:{} Rec:{} UI:{} BG:{}", fps, t_io, t_recog, t_ui, bg_cap_ms);
        let _ = imgproc::put_text(
            &mut canvas,
            &perf_txt,
            core::Point::new(map_w_display + 5, 435),
            imgproc::FONT_HERSHEY_SIMPLEX,
            0.55,
            core::Scalar::new(255.0, 100.0, 100.0, 0.0),
            2,
            imgproc::LINE_AA,
            false,
        );

        highgui::imshow(window_name, &canvas)?;
        if first_show {
            highgui::set_window_property(window_name, highgui::WND_PROP_TOPMOST, 1.0)?;
            first_show = false;
        }

        let key = highgui::wait_key(15)?;
        let debounce_ok = last_toggle_time.elapsed() > std::time::Duration::from_millis(300);
        let is_visible =
            highgui::get_window_property(window_name, highgui::WND_PROP_VISIBLE).unwrap_or(1.0);
        let exit_req = app_state.lock().unwrap().exit_requested;
        if key == 27 || key == 'q' as i32 || is_visible < 1.0 || exit_req {
            break;
        } else if key == 13 || key == 10 {
            // Enter: Set Wind
            if !wind_input_buf.is_empty() {
                if let Ok(w) = wind_input_buf.parse::<f64>() {
                    let mut st = app_state.lock().unwrap();
                    st.wind = w;
                }
                wind_input_buf.clear();
            }
        } else if key == 32 {
            // Space: Set Angle
            if !wind_input_buf.is_empty() {
                if let Ok(a) = wind_input_buf.parse::<f64>() {
                    let mut st = app_state.lock().unwrap();
                    st.current_angle = a.clamp(0.0, 180.0);
                    st.auto_angle = false;
                }
                wind_input_buf.clear();
            }
        } else if key == 8 || key == 127 {
            // Backspace
            wind_input_buf.pop();
        } else if debounce_ok && (key == 'z' as i32 || key == 'Z' as i32) {
            // 快捷键 Z: 切换我方标注模式 (EditMode::P1)
            let mut st = app_state.lock().unwrap();
            st.edit_mode = if st.edit_mode == EditMode::P1 { EditMode::None } else { EditMode::P1 };
            last_toggle_time = std::time::Instant::now();
        } else if debounce_ok && (key == 'x' as i32 || key == 'X' as i32) {
            // 快捷键 X: 切换敌方标注模式 (EditMode::E1)
            let mut st = app_state.lock().unwrap();
            st.edit_mode = if st.edit_mode == EditMode::E1 { EditMode::None } else { EditMode::E1 };
            last_toggle_time = std::time::Instant::now();
        } else if debounce_ok && (key == 'c' as i32 || key == 'C' as i32) {
            // 快捷键 C: 清空手动标记
            let mut st = app_state.lock().unwrap();
            st.manual_p1 = None;
            st.manual_e1 = None;
            st.edit_mode = EditMode::None;
            last_toggle_time = std::time::Instant::now();
        } else if debounce_ok && (key == 'r' as i32 || key == 'R' as i32) {
            // 快捷键 R: 锁定/解锁距离尺
            let mut st = app_state.lock().unwrap();
            if st.locked_px_per_unit.is_some() {
                st.locked_px_per_unit = None;
            } else {
                st.locked_px_per_unit = Some(0.0);
            }
            last_toggle_time = std::time::Instant::now();
        } else if debounce_ok && (key == 'm' as i32 || key == 'M' as i32) {
            let mut st = app_state.lock().unwrap();
            st.is_fixed_angle = !st.is_fixed_angle;
            last_toggle_time = std::time::Instant::now();
        } else if key == 65362 || key == 0x260000 || key == 82 {
            // Up arrow
            if let Ok(mut m_state) = app_state.lock() {
                m_state.current_angle = (m_state.current_angle + 1.0).min(180.0);
                m_state.auto_angle = false;
            }
        } else if key == 65364 || key == 0x280000 || key == 84 {
            // Down arrow
            if let Ok(mut m_state) = app_state.lock() {
                m_state.current_angle = (m_state.current_angle - 1.0).max(0.0);
                m_state.auto_angle = false;
            }
        } else if key > 0 {
            let ch = (key & 0xFF) as u8 as char;
            if ch.is_ascii_digit() || ch == '.' || ch == '-' {
                wind_input_buf.push(ch);
            }
        }

        // 运行时切换大/小地图模式:交互框选新区域 → 换图源 → 重算缩放和比例尺。
        // 点位属于旧图,全部作废;大地图自动按档位锁尺,小地图重新画尺子。
        if app_state.lock().unwrap().switch_requested {
            app_state.lock().unwrap().switch_requested = false;
            let to_big = !app_state.lock().unwrap().big_map;
            println!(
                "👉 [切换模式] 请框选【{}】区域...",
                if to_big { "整张游戏地图" } else { "左上角小地图" }
            );
            let crop_path = "/tmp/tnt_mode_switch.png";
            if select_crop_interactive(crop_path) {
                if let Ok(new_img) = imgcodecs::imread(crop_path, imgcodecs::IMREAD_COLOR) {
                    if !new_img.empty() {
                        let nw = new_img.cols();
                        let nh = new_img.rows();
                        let geo = find_screen_position(&new_img).unwrap_or((0, 0, nw, nh));
                        *map_geo_shared.lock().unwrap() = geo;
                        *shared_map.lock().unwrap() = None; // 丢弃旧区域画面,等新图
                        let new_scale = (map_w_display as f64 / nw as f64).min(2.0);
                        let mut stg = app_state.lock().unwrap();
                        stg.big_map = to_big;
                        stg.src_w = nw;
                        stg.src_h = nh;
                        stg.disp_scale = new_scale;
                        stg.manual_p1 = None;
                        stg.manual_e1 = None;
                        stg.manual_cam_rect = None;
                        if to_big {
                            stg.locked_px_per_unit = Some(nw as f64 / stg.map_units);
                        } else {
                            stg.locked_px_per_unit = None;
                        }
                        println!(
                            "✅ 已切换到{} ({}x{}, 显示缩放 {:.2})",
                            if to_big { "大地图" } else { "小地图" },
                            nw,
                            nh,
                            new_scale
                        );
                    }
                }
            } else {
                println!("⚠️ 未完成框选,保持原模式");
            }
        }
    }

    is_running.store(false, std::sync::atomic::Ordering::Relaxed);
    std::process::exit(0);
}
