use opencv::{
    core,
    highgui,
    imgcodecs,
    imgproc,
    prelude::*,
};
use std::process::Command;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

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
    is_fixed_angle: bool,
}

use tnt_comput::physics::*;

fn compute_fixed_trajectory(dx_units: f64, dy_units: f64, angle_deg: f64, wind: f64) -> (f64, f64) {
    let mut eff_angle = angle_deg;
    let is_reverse = eff_angle > 90.0;
    if is_reverse { eff_angle = 180.0 - eff_angle; }
    
    let wind_power = wind * (if is_reverse { -1.0 } else { 1.0 });
    let mut dist = dx_units.abs();
    
    // 取消高低差的计算，只使用X距
    let mut eff_dist = dist;
    if eff_dist < 0.0 { eff_dist = 0.0; }
    
    let final_power = calc_power(eff_angle, eff_dist, wind_power);
    (final_power.clamp(1.0, 100.0), angle_deg)
}

fn compute_trajectory(dx_units: f64, dy_units: f64, angle_deg: f64, wind: f64) -> (f64, f64) {
    let mut eff_angle = angle_deg;
    let is_reverse = eff_angle > 90.0;
    if is_reverse { eff_angle = 180.0 - eff_angle; }
    
    let wind_power = wind * (if is_reverse { -1.0 } else { 1.0 });
    let mut dist = dx_units.abs();
    
    // 取消高低差的计算，只使用X距
    let mut eff_dist = dist;
    if eff_dist < 0.0 { eff_dist = 0.0; }
    
    let base_power = calc_power(eff_angle, eff_dist, 0.0);
    let mut final_angle = calc_angle(eff_dist, base_power, wind_power, eff_angle);
    
    final_angle = final_angle.clamp(15.0, 89.0);
    
    if is_reverse {
        final_angle = 180.0 - final_angle;
    }
    
    (base_power.clamp(1.0, 100.0), final_angle)
}

fn find_dots(minimap: &core::Mat, is_red: bool) -> opencv::Result<Vec<core::Point>> {
    let mut hsv = core::Mat::default();
    imgproc::cvt_color(minimap, &mut hsv, imgproc::COLOR_BGR2HSV, 0, core::AlgorithmHint::ALGO_HINT_DEFAULT)?;

    let (lower1, upper1, lower2, upper2) = if is_red {
        (core::Scalar::new(0.0, 80.0, 80.0, 0.0), core::Scalar::new(20.0, 255.0, 255.0, 0.0),
         core::Scalar::new(160.0, 80.0, 80.0, 0.0), core::Scalar::new(180.0, 255.0, 255.0, 0.0))
    } else {
        (core::Scalar::new(80.0, 80.0, 80.0, 0.0), core::Scalar::new(150.0, 255.0, 255.0, 0.0),
         core::Scalar::new(80.0, 80.0, 80.0, 0.0), core::Scalar::new(150.0, 255.0, 255.0, 0.0))
    };

    let mut mask1 = core::Mat::default();
    let mut mask2 = core::Mat::default();
    let mut mask = core::Mat::default();
    core::in_range(&hsv, &lower1, &upper1, &mut mask1)?;
    core::in_range(&hsv, &lower2, &upper2, &mut mask2)?;
    core::bitwise_or(&mask1, &mask2, &mut mask, &core::no_array())?;

    let mut contours = core::Vector::<core::Vector<core::Point>>::new();
    imgproc::find_contours(&mask, &mut contours, imgproc::RETR_EXTERNAL, imgproc::CHAIN_APPROX_SIMPLE, core::Point::new(0, 0))?;

    let mut pts = Vec::new();
    for i in 0..contours.len() {
        let contour = contours.get(i)?;
        let rect = opencv::geometry::bounding_rect(&contour)?;
        let area = rect.width * rect.height;

        if area >= 9 && area <= 600 && rect.width <= 30 && rect.height <= 30 {
            let aspect = rect.width as f64 / rect.height as f64;
            if aspect > 0.3 && aspect < 3.0 {
                let mut dark_pixels = 0;
                let mut total_edge = 0;
                let start_x = (rect.x - 1).max(0);
                let start_y = (rect.y - 1).max(0);
                let end_x = (rect.x + rect.width + 1).min(minimap.cols() - 1);
                let end_y = (rect.y + rect.height + 1).min(minimap.rows() - 1);

                for y in start_y..=end_y {
                    for x in start_x..=end_x {
                        if x < rect.x || x >= rect.x + rect.width || y < rect.y || y >= rect.y + rect.height {
                            total_edge += 1;
                            let p = minimap.at_2d::<core::Vec3b>(y, x)?;
                            if p[0] < 120 && p[1] < 120 && p[2] < 120 { dark_pixels += 1; }
                        }
                    }
                }

                if total_edge > 0 && (dark_pixels as f64 / total_edge as f64) > 0.10 {
                    pts.push(core::Point::new(rect.x + rect.width / 2, rect.y + rect.height / 2));
                }
            }
        }
    }
    pts.sort_by(|a, b| a.x.cmp(&b.x));
    Ok(pts)
}

fn detect_camera_frame(minimap: &core::Mat) -> opencv::Result<core::Rect> {
    let rows = minimap.rows() as usize;
    let cols = minimap.cols() as usize;

    if rows > 500 || cols > 600 {
        return Ok(core::Rect::new(0, 0, 90, 60));
    }

    let mut gray = core::Mat::default();
    imgproc::cvt_color(minimap, &mut gray, imgproc::COLOR_BGR2GRAY, 0, core::AlgorithmHint::ALGO_HINT_DEFAULT)?;
    let mut clahe = imgproc::create_clahe(4.0, core::Size::new(8, 8))?;
    let mut enhanced = core::Mat::default();
    clahe.apply(&gray, &mut enhanced)?;
    let data = enhanced.data_bytes()?;

    let mut max_score = -1.0;
    let mut best = core::Rect::new(0, 0, 90, 60);

    for w in (75..=115).step_by(8) {
        for h in (50..=80).step_by(8) {
            for y1 in (0..rows.saturating_sub(h)).step_by(6) {
                let y2 = y1 + h;
                for x1 in (0..cols.saturating_sub(w)).step_by(6) {
                    let x2 = x1 + w;
                    let mut score = 0.0;
                    for x in (x1..x2).step_by(2) {
                        score += (data[y1 * cols + x] as f64 - data[(y1 + 1) * cols + x] as f64).abs();
                        score += (data[y2 * cols + x] as f64 - data[(y2.saturating_sub(1)) * cols + x] as f64).abs();
                    }
                    for y in (y1..y2).step_by(2) {
                        score += (data[y * cols + x1] as f64 - data[y * cols + x1 + 1] as f64).abs();
                        score += (data[y * cols + x2] as f64 - data[y * cols + x2.saturating_sub(1)] as f64).abs();
                    }
                    let ts = score / (2.0 * (w + h) as f64);
                    if ts > max_score {
                        max_score = ts;
                        best = core::Rect::new(x1 as i32, y1 as i32, w as i32, h as i32);
                    }
                }
            }
        }
    }
    Ok(best)
}

fn draw_btn(canvas: &mut core::Mat, rect: core::Rect, label: &str, is_active: bool) -> opencv::Result<()> {
    let color = if is_active { core::Scalar::new(0.0, 200.0, 255.0, 0.0) } else { core::Scalar::new(80.0, 80.0, 80.0, 0.0) };
    let text_color = if is_active { core::Scalar::new(0.0, 0.0, 0.0, 0.0) } else { core::Scalar::new(255.0, 255.0, 255.0, 0.0) };
    
    imgproc::rectangle(canvas, rect, color, -1, imgproc::LINE_8, 0)?;
    imgproc::rectangle(canvas, rect, core::Scalar::new(200.0, 200.0, 200.0, 0.0), 1, imgproc::LINE_8, 0)?;
    
    let mut baseline = 0;
    let size = imgproc::get_text_size(label, imgproc::FONT_HERSHEY_SIMPLEX, 0.5, 1, &mut baseline)?;
    let text_x = rect.x + (rect.width - size.width) / 2;
    let text_y = rect.y + (rect.height + size.height) / 2;
    
    imgproc::put_text(canvas, label, core::Point::new(text_x, text_y), imgproc::FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, imgproc::LINE_AA, false)?;
    Ok(())
}

fn is_inside(x: i32, y: i32, rect: core::Rect) -> bool {
    x >= rect.x && x <= rect.x + rect.width && y >= rect.y && y <= rect.y + rect.height
}

fn main() -> opencv::Result<()> {
    println!("👉 [步骤 1/2] 请务必【只框选左上角的小地图区域】(你框选什么，GUI就显示什么，请不要截全屏!)...");
    let crop_path = "/tmp/tnt_selected_area.png";
    let _ = Command::new("screencapture").arg("-i").arg(crop_path).status();

    let initial_img = match imgcodecs::imread(crop_path, imgcodecs::IMREAD_COLOR) {
        Ok(m) if !m.empty() => m,
        _ => return Ok(()),
    };
    let t_w = initial_img.cols();
    let t_h = initial_img.rows();
    let template = initial_img.try_clone()?;

    let window_name = "TNT Assistant HUD";
    highgui::start_window_thread()?;
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
        is_fixed_angle: false,
    }));

    let scale = if t_w > 600 { 1.0 } else { 2.0 };
    let map_w_display = (t_w as f64 * scale) as i32;
    
    let btn_p1 = core::Rect::new(map_w_display + 20, 30, 110, 40);
    let btn_e1 = core::Rect::new(map_w_display + 140, 30, 110, 40);
    
    let btn_lock_ruler = core::Rect::new(map_w_display + 20, 80, 230, 40);
    let btn_draw_ruler = core::Rect::new(map_w_display + 20, 130, 230, 35);
    
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

    let state_cb = app_state.clone();
    let template_clone = template.clone();
    highgui::set_mouse_callback(window_name, Some(Box::new(move |event, x, y, _flags| {
        let mut st = state_cb.lock().unwrap();
        let scale = if t_w > 600 { 1.0 } else { 2.0 };
        let t_w = template_clone.cols();
        let map_w_display = (t_w as f64 * scale) as i32;

        if event == highgui::EVENT_LBUTTONDOWN {
            // Buttons
            if is_inside(x, y, btn_p1) { st.edit_mode = if st.edit_mode == EditMode::P1 { EditMode::None } else { EditMode::P1 }; }
            else if is_inside(x, y, btn_e1) { st.edit_mode = if st.edit_mode == EditMode::E1 { EditMode::None } else { EditMode::E1 }; }
            else if is_inside(x, y, btn_lock_ruler) {
                if st.locked_px_per_unit.is_some() {
                    st.locked_px_per_unit = None;
                } else {
                    st.locked_px_per_unit = Some(0.0);
                }
            }
            else if is_inside(x, y, btn_draw_ruler) {
                st.edit_mode = if st.edit_mode == EditMode::DrawRuler1 || st.edit_mode == EditMode::DrawRuler2 { EditMode::None } else { EditMode::DrawRuler1 };
            }
            else if is_inside(x, y, btn_auto_detect) {
                st.auto_detect = !st.auto_detect;
            }
            else if is_inside(x, y, btn_lock_map) {
                st.map_locked = !st.map_locked;
            }
            else if is_inside(x, y, btn_clear) {
                st.manual_p1 = None; st.manual_e1 = None;
                st.manual_cam_rect = None; // Also clear manual rect!
                st.edit_mode = EditMode::None;
                st.locked_px_per_unit = None; // Reset lock too
            }
            else if is_inside(x, y, btn_a20) { st.current_angle = 20.0; }
            else if is_inside(x, y, btn_a30) { st.current_angle = 30.0; }
            else if is_inside(x, y, btn_a45) { st.current_angle = 45.0; }
            else if is_inside(x, y, btn_a50) { st.current_angle = 50.0; }
            else if is_inside(x, y, btn_a60) { st.current_angle = 60.0; }
            else if is_inside(x, y, btn_a65) { st.current_angle = 65.0; }
            else if is_inside(x, y, btn_a70) { st.current_angle = 70.0; }
            else if is_inside(x, y, btn_a75) { st.current_angle = 75.0; }
            else if is_inside(x, y, btn_ang_m5) { st.current_angle = (st.current_angle - 5.0).max(0.0); }
            else if is_inside(x, y, btn_ang_minus) { st.current_angle = (st.current_angle - 1.0).max(0.0); }
            else if is_inside(x, y, btn_ang_plus) { st.current_angle = (st.current_angle + 1.0).min(180.0); }
            else if is_inside(x, y, btn_ang_p5) { st.current_angle = (st.current_angle + 5.0).min(180.0); }
            else if is_inside(x, y, btn_wind_m1) { st.wind -= 1.0; }
            else if is_inside(x, y, btn_wind_m01) { st.wind -= 0.1; }
            else if is_inside(x, y, btn_wind_p01) { st.wind += 0.1; }
            else if is_inside(x, y, btn_wind_p1) { st.wind += 1.0; }
            // Map click
            else if x < map_w_display {
                let pt = core::Point::new(x, y);
                match st.edit_mode {
                    EditMode::P1 => { st.manual_p1 = Some(pt); st.edit_mode = EditMode::None; }
                    EditMode::E1 => { st.manual_e1 = Some(pt); st.edit_mode = EditMode::None; }
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
                                ((max_y - min_y) as f64 / scale) as i32
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
                        ((max_y - min_y) as f64 / scale) as i32
                    ));
                }
            }
        }
    })))?;

    let full_path = "/tmp/tnt_fullscreen.png";
    let temp_path = "/tmp/tnt_temp.png";
    let is_running = Arc::new(std::sync::atomic::AtomicBool::new(true));
    let r_clone = is_running.clone();
    let fp = full_path.to_string();
    let tp = temp_path.to_string();
    thread::spawn(move || {
        while r_clone.load(std::sync::atomic::Ordering::Relaxed) {
            let _ = Command::new("screencapture").arg("-x").arg(&tp).status();
            let _ = std::fs::rename(&tp, &fp);
            thread::sleep(Duration::from_millis(100));
        }
    });

    let mut first_show = true;
    let mut last_valid_loc = core::Point::new(0, 0);

    // 立即显示初始化画面，防止 Mac 窗口引擎死锁
    let mut init_canvas = core::Mat::new_rows_cols_with_default(400, 800, core::CV_8UC3, core::Scalar::new(30.0, 30.0, 30.0, 0.0))?;
    imgproc::put_text(&mut init_canvas, "Initializing Background Capture...", core::Point::new(50, 200), imgproc::FONT_HERSHEY_SIMPLEX, 1.0, core::Scalar::new(0.0, 255.0, 0.0, 0.0), 2, imgproc::LINE_AA, false)?;
    highgui::imshow(window_name, &init_canvas)?;
    highgui::set_window_property(window_name, highgui::WND_PROP_TOPMOST, 1.0)?;
    highgui::wait_key(100)?;

    loop {
        let full_img = match imgcodecs::imread(full_path, imgcodecs::IMREAD_COLOR) {
            Ok(m) if !m.empty() => Some(m),
            _ => None,
        };

        let mut max_val = 0.0;
        let mut max_loc = core::Point::new(0, 0);
        let mut img = None;

        if let Some(ref f_img) = full_img {
            if template.cols() <= f_img.cols() && template.rows() <= f_img.rows() {
                let map_locked = app_state.lock().unwrap().map_locked;
                if !map_locked || (last_valid_loc.x == 0 && last_valid_loc.y == 0) {
                    let mut match_result = core::Mat::default();
                    if imgproc::match_template(f_img, &template, &mut match_result, imgproc::TM_CCOEFF_NORMED, &core::no_array()).is_ok() {
                        let _ = core::min_max_loc(&match_result, None, Some(&mut max_val), None, Some(&mut max_loc), &core::no_array());
                        if max_val > 0.35 {
                            last_valid_loc = max_loc;
                        }
                    }
                }
            }

            if last_valid_loc.x + t_w <= f_img.cols() && last_valid_loc.y + t_h <= f_img.rows() {
                if let Ok(roi) = core::Mat::roi(f_img, core::Rect::new(last_valid_loc.x, last_valid_loc.y, t_w, t_h)) {
                    img = roi.try_clone().ok();
                }
            }
        }

        let canvas_w = map_w_display + 270;
        let canvas_h_target = ((t_h as f64 * scale) as i32).max(580);
        let mut canvas = core::Mat::new_rows_cols_with_default(canvas_h_target, canvas_w, core::CV_8UC3, core::Scalar::new(30.0, 30.0, 30.0, 0.0))?;
        let st = *app_state.lock().unwrap();

        if let Some(minimap) = img {
            let mut map_display = core::Mat::default();
            imgproc::resize(&minimap, &mut map_display, core::Size::new(map_w_display, (t_h as f64 * scale) as i32), 0.0, 0.0, imgproc::INTER_LINEAR)?;
            
            for y in 0..map_display.rows() {
                if let (Ok(src_row), Ok(dst_row)) = (map_display.at_row::<core::Vec3b>(y), canvas.at_row_mut::<core::Vec3b>(y)) {
                    for x in 0..map_display.cols() {
                        dst_row[x as usize] = src_row[x as usize];
                    }
                }
            }

            let auto_p = if st.auto_detect { find_dots(&minimap, false).unwrap_or_default() } else { Vec::new() };
            let auto_e = if st.auto_detect { find_dots(&minimap, true).unwrap_or_default() } else { Vec::new() };
            let cam_rect = st.manual_cam_rect.unwrap_or_else(|| {
                detect_camera_frame(&minimap).unwrap_or(core::Rect::new(0, 0, t_w, t_h))
            });

            let to_scr = |p: core::Point| core::Point::new((p.x as f64 * scale) as i32, (p.y as f64 * scale) as i32);

            // 绘制摄像机白框（黄框代表手动）
            let cam_p1 = to_scr(core::Point::new(cam_rect.x, cam_rect.y));
            let cam_p2 = to_scr(core::Point::new(cam_rect.x + cam_rect.width, cam_rect.y + cam_rect.height));
            let box_color = if st.manual_cam_rect.is_some() { core::Scalar::new(0.0, 255.0, 255.0, 0.0) } else { core::Scalar::new(0.0, 255.0, 0.0, 0.0) };
            let _ = imgproc::rectangle(&mut canvas, core::Rect::new(cam_p1.x, cam_p1.y, cam_p2.x - cam_p1.x, cam_p2.y - cam_p1.y), box_color, 2, imgproc::LINE_8, 0);
            
            // 绘制 12 等分刻度（纯粹视觉辅助，让用户能直观看到 12 距的刻度线）
            let ruler_width = cam_p2.x - cam_p1.x;
            for i in 1..12 {
                let tick_x = cam_p1.x + (ruler_width as f64 * (i as f64 / 12.0)) as i32;
                let _ = imgproc::line(&mut canvas, core::Point::new(tick_x, cam_p1.y), core::Point::new(tick_x, cam_p1.y + 10), box_color, 1, imgproc::LINE_AA, 0);
                let _ = imgproc::line(&mut canvas, core::Point::new(tick_x, cam_p2.y - 10), core::Point::new(tick_x, cam_p2.y), box_color, 1, imgproc::LINE_AA, 0);
            }

            let cam_txt = format!("CAMERA {}x{}", cam_rect.width, cam_rect.height);
            let _ = imgproc::put_text(&mut canvas, &cam_txt, core::Point::new(cam_p1.x, (cam_p1.y - 5).max(10)), imgproc::FONT_HERSHEY_SIMPLEX, 0.4, core::Scalar::new(0.0, 255.0, 0.0, 0.0), 1, imgproc::LINE_8, false);

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

            let p1 = st.manual_p1.or_else(|| auto_p.get(0).map(|p| core::Point::new((p.x as f64 * scale) as i32, (p.y as f64 * scale) as i32)));
            let e1 = st.manual_e1.or_else(|| auto_e.get(0).map(|p| core::Point::new((p.x as f64 * scale) as i32, (p.y as f64 * scale) as i32)));
            
            let draw_pt = |c: &mut core::Mat, pt: core::Point, label: &str, is_red: bool, is_manual: bool| {
                // pt is now EXACTLY in canvas pixels (whether from manual click or auto_p scaled up)
                let cx = pt.x;
                let cy = pt.y;
                let color = if is_red { core::Scalar::new(50.0, 50.0, 255.0, 0.0) } else { core::Scalar::new(255.0, 255.0, 0.0, 0.0) };
                let thickness = if is_manual { -1 } else { 2 };
                // radius 6 for smaller dots
                let _ = imgproc::circle(c, core::Point::new(cx, cy), 6, color, thickness, imgproc::LINE_AA, 0);
                let _ = imgproc::put_text(c, label, core::Point::new(cx - 15, cy - 12), imgproc::FONT_HERSHEY_SIMPLEX, 0.6, core::Scalar::new(255.0, 255.0, 255.0, 0.0), 2, imgproc::LINE_AA, false);
            };

            if let Some(p) = p1 { draw_pt(&mut canvas, p, "我方 (My)", false, st.manual_p1.is_some()); }
            if let Some(e) = e1 { draw_pt(&mut canvas, e, "敌方 (Enemy)", true, st.manual_e1.is_some()); }

            let mut y_offset = 430;
            let mut draw_result = |p: core::Point, e: core::Point| {
                // Ensure ruler is locked
                if st.locked_px_per_unit.is_none() {
                    imgproc::rectangle(&mut canvas, core::Rect::new(map_w_display + 10, y_offset - 35, 250, 80), core::Scalar::new(0.0, 0.0, 50.0, 0.0), -1, imgproc::LINE_8, 0).unwrap();
                    imgproc::rectangle(&mut canvas, core::Rect::new(map_w_display + 10, y_offset - 35, 250, 80), core::Scalar::new(0.0, 0.0, 255.0, 0.0), 2, imgproc::LINE_8, 0).unwrap();
                    let _ = imgproc::put_text(&mut canvas, "请先锁定距离尺!", core::Point::new(map_w_display + 20, y_offset + 10), imgproc::FONT_HERSHEY_SIMPLEX, 0.8, core::Scalar::new(0.0, 0.0, 255.0, 0.0), 2, imgproc::LINE_AA, false);
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
                imgproc::rectangle(&mut canvas, core::Rect::new(map_w_display + 10, y_offset - 35, 260, 80), core::Scalar::new(0.0, 50.0, 0.0, 0.0), -1, imgproc::LINE_8, 0).unwrap();
                imgproc::rectangle(&mut canvas, core::Rect::new(map_w_display + 10, y_offset - 35, 260, 80), core::Scalar::new(0.0, 255.0, 0.0, 0.0), 2, imgproc::LINE_8, 0).unwrap();
                
                let mode_str = if st.is_fixed_angle { "[定角打法]" } else { "[变角打法]" };
                let title = format!("{}  风力: {:.1}  X距: {:.1}  Y高: {:.1}", mode_str, st.wind, dx.abs(), dy);
                let _ = imgproc::put_text(&mut canvas, &title, core::Point::new(map_w_display + 15, y_offset - 10), imgproc::FONT_HERSHEY_SIMPLEX, 0.45, core::Scalar::new(200.0, 200.0, 200.0, 0.0), 1, imgproc::LINE_AA, false);
                
                let res_txt = if st.is_fixed_angle {
                    format!("锁定角度: {:.0}°   力度: {:.1}", final_angle, force)
                } else {
                    format!("推荐角度: {:.0}°   力度: {:.1}", final_angle, force)
                };
                let _ = imgproc::put_text(&mut canvas, &res_txt, core::Point::new(map_w_display + 15, y_offset + 15), imgproc::FONT_HERSHEY_SIMPLEX, 0.6, core::Scalar::new(0.0, 255.0, 0.0, 0.0), 2, imgproc::LINE_AA, false);
                
                y_offset += 100;
                
                let pt_p = core::Point::new(p.x, p.y);
                let pt_e = core::Point::new(e.x, e.y);
                let pt_corner = core::Point::new(e.x, p.y);
                // Draw horizontal line (X distance)
                let _ = imgproc::line(&mut canvas, pt_p, pt_corner, core::Scalar::new(0.0, 255.0, 255.0, 0.0), 1, imgproc::LINE_AA, 0);
                // Draw vertical line (Y distance)
                let _ = imgproc::line(&mut canvas, pt_corner, pt_e, core::Scalar::new(255.0, 100.0, 0.0, 0.0), 1, imgproc::LINE_AA, 0);
            };

            if let (Some(p), Some(e)) = (p1, e1) { draw_result(p, e); }
        } else if full_img.is_none() {
            imgproc::put_text(&mut canvas, "Error reading background screen!", core::Point::new(50, 50), imgproc::FONT_HERSHEY_SIMPLEX, 0.8, core::Scalar::new(0.0, 0.0, 255.0, 0.0), 2, imgproc::LINE_AA, false)?;
        } else {
            let warn_text = format!("Waiting for minimap... (score: {:.2})", max_val);
            imgproc::put_text(&mut canvas, &warn_text, core::Point::new(50, 50), imgproc::FONT_HERSHEY_SIMPLEX, 0.8, core::Scalar::new(0.0, 0.0, 255.0, 0.0), 2, imgproc::LINE_AA, false)?;
        }

        draw_btn(&mut canvas, btn_p1, "我方 (My)", st.edit_mode == EditMode::P1)?;
        draw_btn(&mut canvas, btn_e1, "敌方 (Enemy)", st.edit_mode == EditMode::E1)?;
        
        let lock_lbl = if st.locked_px_per_unit.is_some() { "[已锁定] 解锁尺子" } else { "[未锁定] 锁定距离尺" };
        draw_btn(&mut canvas, btn_lock_ruler, lock_lbl, st.locked_px_per_unit.is_some())?;

        let ruler_lbl = if st.edit_mode == EditMode::DrawRuler1 { "[步骤1] 点击左上角" } else if st.edit_mode == EditMode::DrawRuler2 { "[步骤2] 点击右下角" } else { "手动框选 1屏幕宽" };
        draw_btn(&mut canvas, btn_draw_ruler, ruler_lbl, st.edit_mode == EditMode::DrawRuler1 || st.edit_mode == EditMode::DrawRuler2)?;

        let auto_lbl = if st.auto_detect { "自动识别: 开" } else { "自动识别: 关" };
        draw_btn(&mut canvas, btn_auto_detect, auto_lbl, st.auto_detect)?;
        
        let map_lbl = if st.map_locked { "地图区域: 锁定" } else { "地图区域: 追踪" };
        draw_btn(&mut canvas, btn_lock_map, map_lbl, st.map_locked)?;

        draw_btn(&mut canvas, btn_clear, "清空手动标记", false)?;
        // Draw preset angle buttons
        draw_btn(&mut canvas, btn_a20, "20°", (st.current_angle - 20.0).abs() < 0.5)?;
        draw_btn(&mut canvas, btn_a30, "30°", (st.current_angle - 30.0).abs() < 0.5)?;
        draw_btn(&mut canvas, btn_a45, "45°", (st.current_angle - 45.0).abs() < 0.5)?;
        draw_btn(&mut canvas, btn_a50, "50°", (st.current_angle - 50.0).abs() < 0.5)?;
        draw_btn(&mut canvas, btn_a60, "60°", (st.current_angle - 60.0).abs() < 0.5)?;
        draw_btn(&mut canvas, btn_a65, "65°", (st.current_angle - 65.0).abs() < 0.5)?;
        draw_btn(&mut canvas, btn_a70, "70°", (st.current_angle - 70.0).abs() < 0.5)?;
        draw_btn(&mut canvas, btn_a75, "75°", (st.current_angle - 75.0).abs() < 0.5)?;

        draw_btn(&mut canvas, btn_ang_m5, "-5", false)?;
        draw_btn(&mut canvas, btn_ang_minus, "-1", false)?;
        draw_btn(&mut canvas, rect_ang_text, &format!("{:.0}°", st.current_angle), true)?;
        draw_btn(&mut canvas, btn_ang_plus, "+1", false)?;
        draw_btn(&mut canvas, btn_ang_p5, "+5", false)?;
        
        draw_btn(&mut canvas, btn_wind_m1, "-1.0", false)?;
        draw_btn(&mut canvas, btn_wind_m01, "-0.1", false)?;
        let wind_str = format!("风: {:.1}", st.wind);
        let _ = imgproc::put_text(&mut canvas, &wind_str, core::Point::new(rect_wind_text.x, rect_wind_text.y + 20), imgproc::FONT_HERSHEY_SIMPLEX, 0.4, core::Scalar::new(0.0, 255.0, 255.0, 0.0), 1, imgproc::LINE_AA, false);
        draw_btn(&mut canvas, btn_wind_p01, "+0.1", false)?;
        draw_btn(&mut canvas, btn_wind_p1, "+1.0", false)?;

        let hint_txt = "提示: 键盘数字或方向键可直接调角度";
        let _ = imgproc::put_text(&mut canvas, hint_txt, core::Point::new(map_w_display + 20, 370), imgproc::FONT_HERSHEY_SIMPLEX, 0.4, core::Scalar::new(180.0, 180.0, 180.0, 0.0), 1, imgproc::LINE_AA, false);

        highgui::imshow(window_name, &canvas)?;
        if first_show {
            highgui::set_window_property(window_name, highgui::WND_PROP_TOPMOST, 1.0)?;
            first_show = false;
        }

        let key = highgui::wait_key(100)?;
        if key == 27 || key == 'q' as i32 {
            break;
        } else if key == 'm' as i32 || key == 'M' as i32 {
            let mut st = app_state.lock().unwrap();
            st.is_fixed_angle = !st.is_fixed_angle;
        } else if key == 'w' as i32 {
            let mut st = app_state.lock().unwrap();
            if let Some(p) = st.manual_p1.as_mut() { p.y -= 1; }
            if let Some(e) = st.manual_e1.as_mut() { e.y -= 1; }
        } else if key == 's' as i32 {
            let mut st = app_state.lock().unwrap();
            if let Some(p) = st.manual_p1.as_mut() { p.y += 1; }
            if let Some(e) = st.manual_e1.as_mut() { e.y += 1; }
        } else if key == 'a' as i32 {
            let mut st = app_state.lock().unwrap();
            if let Some(p) = st.manual_p1.as_mut() { p.x -= 1; }
            if let Some(e) = st.manual_e1.as_mut() { e.x -= 1; }
        } else if key == 'd' as i32 {
            let mut st = app_state.lock().unwrap();
            if let Some(p) = st.manual_p1.as_mut() { p.x += 1; }
            if let Some(e) = st.manual_e1.as_mut() { e.x += 1; }
        } else if (48..=57).contains(&key) {
            let digit = key - 48;
            if let Ok(mut m_state) = app_state.lock() {
                let next_val = m_state.current_angle * 10.0 + digit as f64;
                if m_state.current_angle == 45.0 || next_val > 180.0 {
                    m_state.current_angle = digit as f64;
                } else {
                    m_state.current_angle = next_val;
                }
            }
        } else if key == 65362 || key == 0x260000 || key == 82 { // Up arrow
            if let Ok(mut m_state) = app_state.lock() { m_state.current_angle = (m_state.current_angle + 1.0).min(180.0); }
        } else if key == 65364 || key == 0x280000 || key == 84 { // Down arrow
            if let Ok(mut m_state) = app_state.lock() { m_state.current_angle = (m_state.current_angle - 1.0).max(0.0); }
        }
    }

    is_running.store(false, std::sync::atomic::Ordering::Relaxed);
    highgui::destroy_all_windows()?;
    Ok(())
}
