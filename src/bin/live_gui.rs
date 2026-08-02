use opencv::{core, highgui, imgcodecs, imgproc, prelude::*};
use std::sync::{Arc, Mutex};
use std::thread;

use tnt_comput::screen;

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
    map_locked: bool,
    auto_detect: bool,
    auto_angle: bool,
    is_fixed_angle: bool,
    exit_requested: bool,
}

use tnt_comput::physics::*;

fn compute_fixed_trajectory(dx_units: f64, dy_units: f64, angle_deg: f64, wind: f64) -> (f64, f64) {
    let mut eff_angle = angle_deg;
    let is_reverse = eff_angle > 90.0;
    if is_reverse {
        eff_angle = 180.0 - eff_angle;
    }

    let wind_power = wind * (if is_reverse { -1.0 } else { 1.0 });
    let dist = dx_units.abs();

    // 调用全新的底层物理引擎，同时传入高低差 dy_units
    let final_power = power_for_angle(eff_angle, dist, dy_units, wind_power).unwrap_or(100.0);
    (final_power.clamp(1.0, 100.0), angle_deg)
}

fn compute_trajectory(dx_units: f64, dy_units: f64, angle_deg: f64, wind: f64) -> (f64, f64) {
    let mut eff_angle = angle_deg;
    let is_reverse = eff_angle > 90.0;
    if is_reverse {
        eff_angle = 180.0 - eff_angle;
    }

    let wind_power = wind * (if is_reverse { -1.0 } else { 1.0 });
    let dist = dx_units.abs();

    // 使用新的 power_for_angle 和 calc_angle 传递 dy_units
    let base_power = power_for_angle(eff_angle, dist, dy_units, 0.0).unwrap_or(100.0);
    let mut final_angle = calc_angle(dist, dy_units, base_power, wind_power, eff_angle);

    final_angle = final_angle.clamp(15.0, 89.0);

    if is_reverse {
        final_angle = 180.0 - final_angle;
    }

    (base_power.clamp(1.0, 100.0), final_angle)
}

// 【呼吸灯/闪烁帧差法】：通过前后两帧小地图相减，0.1ms 自动无视地图杂色背景，瞬间精确定位我方位置！
fn detect_breathing_dots(prev: &core::Mat, curr: &core::Mat) -> opencv::Result<Vec<core::Point>> {
    if prev.empty() || curr.empty() || prev.size()? != curr.size()? {
        return Ok(Vec::new());
    }

    let mut diff = core::Mat::default();
    core::absdiff(prev, curr, &mut diff)?;

    let mut gray_diff = core::Mat::default();
    imgproc::cvt_color(
        &diff,
        &mut gray_diff,
        imgproc::COLOR_BGR2GRAY,
        0,
        core::AlgorithmHint::ALGO_HINT_DEFAULT,
    )?;

    let mut mask = core::Mat::default();
    imgproc::threshold(&gray_diff, &mut mask, 20.0, 255.0, imgproc::THRESH_BINARY)?;

    let mut contours = core::Vector::<core::Vector<core::Point>>::new();
    imgproc::find_contours(
        &mask,
        &mut contours,
        imgproc::RETR_EXTERNAL,
        imgproc::CHAIN_APPROX_SIMPLE,
        core::Point::new(0, 0),
    )?;

    let mut pts = Vec::new();
    for i in 0..contours.len() {
        let contour = contours.get(i)?;
        let rect = opencv::geometry::bounding_rect(&contour)?;
        let area = rect.width * rect.height;
        if area >= 4 && area <= 400 && rect.width <= 25 && rect.height <= 25 {
            let center = core::Point::new(rect.x + rect.width / 2, rect.y + rect.height / 2);
            pts.push(center);
        }
    }
    pts.sort_by(|a, b| a.x.cmp(&b.x));
    Ok(pts)
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

fn main() -> opencv::Result<()> {
    #[cfg(target_os = "macos")]
    println!("=== 🍎 Mac OS 环境检测成功，已自动切换底层抓图引擎 ===");

    // FIX: macOS 上原来这里直接返回写死的 (0,0,800,600)，根本没有选区；
    // 现在走 screen::select_region，先全屏截图再拖框，并自动处理 Retina 倍率。
    println!("👉 [步骤 1/2] 请框选【左上角小地图】区域...");
    let Some(map_geo) = screen::select_region("步骤 1/2 小地图") else {
        println!("❌ 已取消选区，退出。");
        return Ok(());
    };
    let t_w = map_geo.2;
    let t_h = map_geo.3;

    println!("👉 [步骤 2/2] 请框选【右下角 角度/力度 数值】区域...");
    let power_geo = screen::select_region("步骤 2/2 角度力度数值");

    // FIX: 原来这里无条件调用 grim（没有 cfg 门控），macOS 上必然 command not found。
    if let Some(p_geo) = power_geo {
        screen::capture_rect_to_file(p_geo, "/tmp/tnt_selected_power.png");
    } else {
        println!("⚠️  未选择数值区域，角度自动识别将不可用（可手动设角度）。");
    }

    let window_name = "TNT Assistant HUD";
    highgui::named_window(window_name, highgui::WINDOW_AUTOSIZE)?;

    let app_state = Arc::new(Mutex::new(AppState {
        edit_mode: EditMode::None,
        manual_p1: None,
        manual_e1: None,
        manual_cam_rect: None,
        drag_start: None,
        current_angle: 45.0,
        wind: 0.0,
        locked_px_per_unit: None,
        map_locked: true,
        auto_detect: true,
        auto_angle: true,
        is_fixed_angle: true,
        exit_requested: false,
    }));

    let scale = if t_w > 600 { 1.0 } else { 2.0 };
    let map_w_display = (t_w as f64 * scale) as i32;

    let btn_p1 = core::Rect::new(map_w_display + 20, 30, 110, 40);
    let btn_e1 = core::Rect::new(map_w_display + 140, 30, 110, 40);

    let btn_lock_ruler = core::Rect::new(map_w_display + 20, 80, 230, 40);
    let btn_draw_ruler = core::Rect::new(map_w_display + 20, 130, 230, 35);

    let btn_exit = core::Rect::new(map_w_display + 150, 5, 100, 30);

    // New feature buttons
    let btn_auto_detect = core::Rect::new(map_w_display + 20, 175, 110, 30);
    let btn_lock_map = core::Rect::new(map_w_display + 140, 175, 110, 30);

    let btn_clear = core::Rect::new(map_w_display + 20, 210, 230, 25);

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
    let map_geo_clone = map_geo;
    highgui::set_mouse_callback(
        window_name,
        Some(Box::new(move |event, x, y, _flags| {
            let mut st = state_cb.lock().unwrap();
            let t_w = map_geo_clone.2;
            let scale = if t_w > 600 { 1.0 } else { 2.0 };
            let map_w_display = (t_w as f64 * scale) as i32;

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
                } else if is_inside(x, y, btn_auto_detect) {
                    st.auto_detect = !st.auto_detect;
                } else if is_inside(x, y, btn_lock_map) {
                    st.map_locked = !st.map_locked;
                } else if is_inside(x, y, btn_clear) {
                    st.manual_p1 = None;
                    st.manual_e1 = None;
                    st.manual_cam_rect = None; // Also clear manual rect!
                    st.edit_mode = EditMode::None;
                    st.locked_px_per_unit = None; // Reset lock too
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
                                    (min_x as f64 / scale) as i32,
                                    (min_y as f64 / scale) as i32,
                                    ((max_x - min_x) as f64 / scale) as i32,
                                    ((max_y - min_y) as f64 / scale) as i32,
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
                            (min_x as f64 / scale) as i32,
                            (min_y as f64 / scale) as i32,
                            ((max_x - min_x) as f64 / scale) as i32,
                            ((max_y - min_y) as f64 / scale) as i32,
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

    let map_geo_bg = map_geo;
    let power_geo_bg = power_geo;

    let shared_map = Arc::new(std::sync::Mutex::new(None::<core::Mat>));
    let shared_map_clone = shared_map.clone();
    let shared_detected_pt = Arc::new(std::sync::Mutex::new(None::<core::Point>));
    let shared_detected_pt_clone = shared_detected_pt.clone();
    let shared_recognized_angle = Arc::new(std::sync::Mutex::new(None::<i32>));
    let shared_recognized_angle_clone = shared_recognized_angle.clone();

    let app_state_bg = app_state.clone();

    thread::spawn(move || {
        let recognizer =
            tnt_comput::ui::UiRecognizer::new("src/templates").expect("Failed to init recognizer");
        let mut prev_minimap: Option<core::Mat> = None;
        let mut last_recog_time = std::time::Instant::now();

        // 角度多帧投票：连续 3 帧里过半才写入，消除偶发跳数字
        let mut angle_voter = tnt_comput::detect::ValueVoter::new(3);

        let (map_path, power_path) = screen::capture_paths();

        while r_clone.load(std::sync::atomic::Ordering::Relaxed) {
            let t0 = std::time::Instant::now();

            screen::capture_rect_to_file(map_geo_bg, map_path);

            if let Ok(m) = imgcodecs::imread(map_path, imgcodecs::IMREAD_COLOR) {
                if !m.empty() {
                    // Background breathing dots detection
                    if m.cols() <= 1600 && m.rows() <= 1200 {
                        if let Some(ref prev) = prev_minimap {
                            if let Ok(pts) = detect_breathing_dots(prev, &m) {
                                if let Ok(mut lock) = shared_detected_pt_clone.lock() {
                                    *lock = if !pts.is_empty() { Some(pts[0]) } else { None };
                                }
                            }
                        }
                    }
                    prev_minimap = m.try_clone().ok();

                    if let Ok(mut lock) = shared_map_clone.lock() {
                        *lock = Some(m);
                    }
                }
            }

            if let Some(p_geo) = power_geo_bg {
                screen::capture_rect_to_file(p_geo, power_path);

                if let Ok(p_mat) = imgcodecs::imread(power_path, imgcodecs::IMREAD_COLOR) {
                    // FIX: 原来的上限是 300x200，而 Retina 截图是 2 倍物理像素，
                    // 稍大一点的选区就会让角度识别静默地永远不执行。
                    if !p_mat.empty() && p_mat.cols() <= 1600 && p_mat.rows() <= 900 {
                        if last_recog_time.elapsed().as_millis() > 100 {
                            if let Ok((Some(val), conf)) =
                                recognizer.recognize_angle_digit_conf(&p_mat)
                            {
                                if conf > 0.88 {
                                    if let Some(stable) = angle_voter.push(val) {
                                        if let Ok(mut lock) = shared_recognized_angle_clone.lock() {
                                            *lock = Some(stable);
                                        }
                                        // Auto-sync angle if enabled
                                        if stable >= 10 && stable <= 90 {
                                            if let Ok(mut m_state) = app_state_bg.lock() {
                                                if m_state.auto_angle {
                                                    m_state.current_angle = stable as f64;
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                            last_recog_time = std::time::Instant::now();
                        }
                    }
                }
            }

            if let Ok(mut lock) = cap_time_ms_clone.lock() {
                *lock = t0.elapsed().as_millis();
            }
            std::thread::sleep(std::time::Duration::from_millis(40)); // 25 FPS bg capture
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
    let mut last_stable_p1: Option<core::Point> = None;
    let mut fps_t0 = std::time::Instant::now();

    loop {
        let loop_t0 = std::time::Instant::now();

        let img = {
            let lock = shared_map.lock().unwrap();
            lock.as_ref().and_then(|m| m.try_clone().ok())
        };

        let t_io = loop_t0.elapsed().as_millis();
        let t1 = std::time::Instant::now();

        let power_recognized_val = *shared_recognized_angle.lock().unwrap();
        let raw_detected_p = *shared_detected_pt.lock().unwrap();

        let t_recog = t1.elapsed().as_millis();
        let t2 = std::time::Instant::now();

        let canvas_w = map_w_display + 270;
        let canvas_h_target = ((t_h as f64 * scale) as i32).max(580);
        let mut canvas = core::Mat::new_rows_cols_with_default(
            canvas_h_target,
            canvas_w,
            core::CV_8UC3,
            core::Scalar::new(30.0, 30.0, 30.0, 0.0),
        )?;
        let st = *app_state.lock().unwrap();

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

            // 【防抖抗闪内存锁】：呼吸灯熄灭的暗周期自动保持上一次坐标；同原地闪烁时微小震荡自动平滑，消除画面闪烁！
            let final_auto_p1 = if let Some(new_pt) = raw_detected_p {
                if let Some(last_pt) = last_stable_p1 {
                    let dx = (new_pt.x - last_pt.x) as f64;
                    let dy = (new_pt.y - last_pt.y) as f64;
                    if (dx * dx + dy * dy) < 400.0 {
                        // 20px 阈值范围内认为是原地呼吸
                        Some(last_pt)
                    } else {
                        last_stable_p1 = Some(new_pt);
                        Some(new_pt)
                    }
                } else {
                    last_stable_p1 = Some(new_pt);
                    Some(new_pt)
                }
            } else {
                last_stable_p1
            };

            let auto_p = if let Some(p) = final_auto_p1 {
                vec![p]
            } else {
                Vec::new()
            };
            let auto_e: Vec<core::Point> = Vec::new();
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
                    if let Ok(mut m_state) = app_state.lock() {
                        m_state.locked_px_per_unit = Some(current_px_per_unit);
                    }
                } else {
                    current_px_per_unit = locked;
                }
            }
            let px_per_unit = current_px_per_unit;

            let p1 = st.manual_p1.or_else(|| {
                auto_p.get(0).map(|p| {
                    core::Point::new((p.x as f64 * scale) as i32, (p.y as f64 * scale) as i32)
                })
            });
            let e1 = st.manual_e1.or_else(|| {
                auto_e.get(0).map(|p| {
                    core::Point::new((p.x as f64 * scale) as i32, (p.y as f64 * scale) as i32)
                })
            });

            let draw_pt =
                |c: &mut core::Mat, pt: core::Point, label: &str, is_red: bool, is_manual: bool| {
                    // pt is now EXACTLY in canvas pixels (whether from manual click or auto_p scaled up)
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

                let (force, final_angle) = if st.is_fixed_angle {
                    compute_fixed_trajectory(dx, dy, st.current_angle, st.wind)
                } else {
                    compute_trajectory(dx, dy, st.current_angle, st.wind)
                };

                // Draw a beautiful background box for the force recommendation
                imgproc::rectangle(
                    &mut canvas,
                    core::Rect::new(map_w_display + 10, y_offset - 35, 260, 80),
                    core::Scalar::new(0.0, 50.0, 0.0, 0.0),
                    -1,
                    imgproc::LINE_8,
                    0,
                )
                .unwrap();
                imgproc::rectangle(
                    &mut canvas,
                    core::Rect::new(map_w_display + 10, y_offset - 35, 260, 80),
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
                    format!("锁定角度: {:.0}°   力度: {:.1}", final_angle, force)
                } else {
                    format!("推荐角度: {:.0}°   力度: {:.1}", final_angle, force)
                };
                let _ = imgproc::put_text(
                    &mut canvas,
                    &res_txt,
                    core::Point::new(map_w_display + 15, y_offset + 15),
                    imgproc::FONT_HERSHEY_SIMPLEX,
                    0.6,
                    core::Scalar::new(0.0, 255.0, 0.0, 0.0),
                    2,
                    imgproc::LINE_AA,
                    false,
                );

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
        let _ = draw_btn(&mut canvas, btn_auto_angle, &pval_str, st.auto_angle);

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

        let lock_lbl = if st.locked_px_per_unit.is_some() {
            "[已锁定] 解锁尺子"
        } else {
            "[未锁定] 锁定距离尺"
        };
        draw_btn(
            &mut canvas,
            btn_lock_ruler,
            lock_lbl,
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

        let auto_lbl = if st.auto_detect {
            "自动识别: 开"
        } else {
            "自动识别: 关"
        };
        draw_btn(&mut canvas, btn_auto_detect, auto_lbl, st.auto_detect)?;

        let map_lbl = if st.map_locked {
            "地图区域: 锁定"
        } else {
            "地图区域: 追踪"
        };
        draw_btn(&mut canvas, btn_lock_map, map_lbl, st.map_locked)?;

        draw_btn(&mut canvas, btn_clear, "清空手动标记", false)?;
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

        let hint_txt = "提示: 敲数字后 [回车]=风速, [空格]=角度";
        let _ = imgproc::put_text(
            &mut canvas,
            hint_txt,
            core::Point::new(map_w_display + 5, 385),
            imgproc::FONT_HERSHEY_SIMPLEX,
            0.4,
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
        } else if key == '1' as i32 {
            // 快捷键 1: 切换我方标注模式 (EditMode::P1)
            let mut st = app_state.lock().unwrap();
            st.edit_mode = if st.edit_mode == EditMode::P1 { EditMode::None } else { EditMode::P1 };
        } else if key == '2' as i32 {
            // 快捷键 2: 切换敌方标注模式 (EditMode::E1)
            let mut st = app_state.lock().unwrap();
            st.edit_mode = if st.edit_mode == EditMode::E1 { EditMode::None } else { EditMode::E1 };
        } else if key == 'c' as i32 || key == 'C' as i32 {
            // 快捷键 C: 清空手动标记
            let mut st = app_state.lock().unwrap();
            st.manual_p1 = None;
            st.manual_e1 = None;
            st.edit_mode = EditMode::None;
        } else if key == 'r' as i32 || key == 'R' as i32 {
            // 快捷键 R: 锁定/解锁距离尺
            let mut st = app_state.lock().unwrap();
            if st.locked_px_per_unit.is_some() {
                st.locked_px_per_unit = None;
            } else {
                st.locked_px_per_unit = Some(0.0);
            }
        } else if key == 'a' as i32 || key == 'A' as i32 {
            // 快捷键 A: 切换自动识别开关
            let mut st = app_state.lock().unwrap();
            st.auto_detect = !st.auto_detect;
        } else if key == 'l' as i32 || key == 'L' as i32 {
            // 快捷键 L: 锁定/追踪地图区域
            let mut st = app_state.lock().unwrap();
            st.map_locked = !st.map_locked;
        } else if key == 'm' as i32 || key == 'M' as i32 {
            let mut st = app_state.lock().unwrap();
            st.is_fixed_angle = !st.is_fixed_angle;
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
    }

    is_running.store(false, std::sync::atomic::Ordering::Relaxed);
    std::process::exit(0);
}
