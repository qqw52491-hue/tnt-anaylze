use opencv::{core, imgcodecs, imgproc, prelude::*};
use std::io::{self, Write};
use std::process::Command;
use std::thread;
use std::time::Duration;

// ====== 寻找红蓝点 ======
fn find_dots_by_color_and_edge(
    minimap: &core::Mat,
    is_red: bool,
) -> opencv::Result<Vec<core::Point>> {
    let mut hsv = core::Mat::default();
    imgproc::cvt_color(
        minimap,
        &mut hsv,
        imgproc::COLOR_BGR2HSV,
        0,
        core::AlgorithmHint::ALGO_HINT_DEFAULT,
    )?;

    let mut mask = core::Mat::default();
    if is_red {
        let mut mask1 = core::Mat::default();
        core::in_range(
            &hsv,
            &core::Scalar::new(0.0, 50.0, 50.0, 0.0),
            &core::Scalar::new(5.0, 255.0, 255.0, 0.0),
            &mut mask1,
        )?;
        let mut mask2 = core::Mat::default();
        core::in_range(
            &hsv,
            &core::Scalar::new(175.0, 50.0, 50.0, 0.0),
            &core::Scalar::new(180.0, 255.0, 255.0, 0.0),
            &mut mask2,
        )?;
        core::add(&mask1, &mask2, &mut mask, &core::no_array(), -1)?;
    } else {
        core::in_range(
            &hsv,
            &core::Scalar::new(80.0, 50.0, 50.0, 0.0),
            &core::Scalar::new(140.0, 255.0, 255.0, 0.0),
            &mut mask,
        )?;
    }

    let mut contours = core::Vector::<core::Vector<core::Point>>::new();
    imgproc::find_contours(
        &mask,
        &mut contours,
        imgproc::RETR_EXTERNAL,
        imgproc::CHAIN_APPROX_SIMPLE,
        core::Point::new(0, 0),
    )?;

    let minimap_data = minimap.data_bytes()?;
    let cols = minimap.cols() as usize;

    let mut valid_points = Vec::new();
    for i in 0..contours.len() {
        let contour = contours.get(i)?;
        let rect = opencv::geometry::bounding_rect(&contour)?;
        let area = rect.width * rect.height;

        if area >= 15 && area <= 150 {
            let aspect = rect.width as f64 / rect.height as f64;
            if aspect > 0.7 && aspect < 1.4 {
                let mut dark_pixels = 0;
                let mut total_edge_pixels = 0;
                let margin = 1;
                let start_x = (rect.x - margin).max(0) as usize;
                let start_y = (rect.y - margin).max(0) as usize;
                let end_x = (rect.x + rect.width + margin).min(minimap.cols() - 1) as usize;
                let end_y = (rect.y + rect.height + margin).min(minimap.rows() - 1) as usize;

                let rx1 = rect.x as usize;
                let rx2 = (rect.x + rect.width) as usize;
                let ry1 = rect.y as usize;
                let ry2 = (rect.y + rect.height) as usize;

                for y in start_y..=end_y {
                    for x in start_x..=end_x {
                        if x < rx1 || x >= rx2 || y < ry1 || y >= ry2 {
                            total_edge_pixels += 1;
                            let idx = (y * cols + x) * 3;
                            let b = minimap_data[idx] as i32;
                            let g = minimap_data[idx + 1] as i32;
                            let r = minimap_data[idx + 2] as i32;
                            if b < 100 && g < 100 && r < 100 {
                                dark_pixels += 1;
                            }
                        }
                    }
                }

                let dark_ratio = if total_edge_pixels > 0 {
                    dark_pixels as f64 / total_edge_pixels as f64
                } else {
                    0.0
                };
                if dark_ratio >= 0.15 {
                    valid_points.push(core::Point::new(
                        rect.x + rect.width / 2,
                        rect.y + rect.height / 2,
                    ));
                }
            }
        }
    }

    Ok(valid_points)
}

// ====== 自动检测视野框 ======
fn detect_camera_frame(minimap: &core::Mat) -> opencv::Result<core::Rect> {
    let rows = minimap.rows() as usize;
    let cols = minimap.cols() as usize;

    let mut gray = core::Mat::default();
    imgproc::cvt_color(
        minimap,
        &mut gray,
        imgproc::COLOR_BGR2GRAY,
        0,
        core::AlgorithmHint::ALGO_HINT_DEFAULT,
    )?;
    let mut clahe = imgproc::create_clahe(4.0, core::Size::new(8, 8))?;
    let mut enhanced_gray = core::Mat::default();
    clahe.apply(&gray, &mut enhanced_gray)?;

    let gray_data = enhanced_gray.data_bytes()?;

    let mut col_scores: Vec<f64> = vec![0.0; cols];
    for x in 0..cols - 1 {
        let mut total = 0.0;
        for y in 0..rows {
            let p1 = gray_data[y * cols + x];
            let p2 = gray_data[y * cols + x + 1];
            total += (p1 as f64 - p2 as f64).abs();
        }
        col_scores[x] = total;
    }

    let mut row_scores: Vec<f64> = vec![0.0; rows];
    for y in 0..rows - 1 {
        let mut total = 0.0;
        for x in 0..cols {
            let p1 = gray_data[y * cols + x];
            let p2 = gray_data[(y + 1) * cols + x];
            total += (p1 as f64 - p2 as f64).abs();
        }
        row_scores[y] = total;
    }

    let mut x_candidates = Vec::new();
    for x in 5..cols - 5 {
        if col_scores[x] > col_scores[x - 1]
            && col_scores[x] > col_scores[x + 1]
            && col_scores[x] > 1000.0
        {
            x_candidates.push((x, col_scores[x]));
        }
    }
    x_candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    let mut y_candidates = Vec::new();
    for y in 5..rows - 5 {
        if row_scores[y] > row_scores[y - 1]
            && row_scores[y] > row_scores[y + 1]
            && row_scores[y] > 1000.0
        {
            y_candidates.push((y, row_scores[y]));
        }
    }
    y_candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    let mut best_score = 0.0;
    let mut best_rect = core::Rect::new(0, 0, 0, 0);

    for i in 0..x_candidates.len() {
        for j in (i + 1)..x_candidates.len() {
            let mut x1 = x_candidates[i].0;
            let mut x2 = x_candidates[j].0;
            if x1 > x2 {
                std::mem::swap(&mut x1, &mut x2);
            }

            let w = x2 - x1;
            if w < 80 || w > 110 {
                continue;
            }

            for k in 0..y_candidates.len() {
                for l in (k + 1)..y_candidates.len() {
                    let mut y1 = y_candidates[k].0;
                    let mut y2 = y_candidates[l].0;
                    if y1 > y2 {
                        std::mem::swap(&mut y1, &mut y2);
                    }

                    let h = y2 - y1;
                    if h < 55 || h > 75 {
                        continue;
                    }

                    let mut top_score = 0.0;
                    for x in x1..x2 {
                        let p1 = gray_data[y1 * cols + x];
                        let p2 = gray_data[(y1 + 1) * cols + x];
                        top_score += (p1 as f64 - p2 as f64).abs().min(30.0);
                    }
                    top_score /= w as f64;

                    let mut bottom_score = 0.0;
                    for x in x1..x2 {
                        let p1 = gray_data[y2 * cols + x];
                        let p2 = gray_data[(y2 + 1) * cols + x];
                        bottom_score += (p1 as f64 - p2 as f64).abs().min(30.0);
                    }
                    bottom_score /= w as f64;

                    let mut left_score = 0.0;
                    for y in y1..y2 {
                        let p1 = gray_data[y * cols + x1];
                        let p2 = gray_data[y * cols + x1 + 1];
                        left_score += (p1 as f64 - p2 as f64).abs().min(30.0);
                    }
                    left_score /= h as f64;

                    let mut right_score = 0.0;
                    for y in y1..y2 {
                        let p1 = gray_data[y * cols + x2];
                        let p2 = gray_data[y * cols + x2 + 1];
                        right_score += (p1 as f64 - p2 as f64).abs().min(30.0);
                    }
                    right_score /= h as f64;

                    let score = top_score + bottom_score + left_score + right_score;

                    if score > best_score {
                        best_score = score;
                        best_rect = core::Rect::new(x1 as i32, y1 as i32, w as i32, h as i32);
                    }
                }
            }
        }
    }

    Ok(best_rect)
}

use tnt_comput::physics::*;

fn calc_power_30deg(dx: f64, _dy: f64) -> f64 {
    let mut eff_dist = dx;
    if eff_dist < 0.0 { eff_dist = 0.0; }
    calc_power(30.0, eff_dist, 0.0)
}

fn calc_power_65deg(dx: f64, _dy: f64) -> f64 {
    let mut eff_dist = dx;
    if eff_dist < 0.0 { eff_dist = 0.0; }
    calc_power(65.0, eff_dist, 0.0)
}

fn main() -> opencv::Result<()> {
    println!("==================================================");
    println!(">>> 启动【像微信截图一样的交互选区 & 实时分析助手】");
    println!("==================================================\n");

    println!("👉 [步骤 1/2] 类似微信截图：请按下 Enter 键激活屏幕选区十字光标，用鼠标拖动框选游戏区域（或小地图区域）...");
    let mut stdin_input = String::new();
    let _ = io::stdin().read_line(&mut stdin_input);

    let crop_path = "/tmp/tnt_selected_area.png";

    // 调起 macOS 原生选区截图工具 (就像微信截图 Cmd+Shift+A 相同体验)
    let status = Command::new("screencapture")
        .arg("-i")
        .arg(crop_path)
        .status();

    if status.is_err() || !std::path::Path::new(crop_path).exists() {
        println!("⚠️ 未完成选区或取消了截图，程序退出。");
        return Ok(());
    }

    let initial_img = imgcodecs::imread(crop_path, imgcodecs::IMREAD_COLOR)?;
    if initial_img.empty() {
        println!("⚠️ 截图读取失败。");
        return Ok(());
    }

    println!("\n✅ [选区成功！] 截取区域尺寸: {}x{} 像素", initial_img.cols(), initial_img.rows());
    println!("\n👉 [步骤 2/2] 请按下 Enter 键【开始实时连续分析】(按 Ctrl+C 可随时停止)...");
    let _ = io::stdin().read_line(&mut stdin_input);

    println!("🚀 [实时分析启动中...] 正在以 300ms 间隔监控框选区域...\n");

    let template = initial_img.try_clone()?;
    let t_w = template.cols();
    let t_h = template.rows();

    println!("\n🚀 [实时分析已启动！]");
    println!(">>> 正在通过视觉引擎全屏动态锁定你框选的区域 (尺寸 {}x{})...", t_w, t_h);
    println!(">>> 调试预览图将每秒同步保存至 live_debug.png (按 Ctrl+C 可停止)\n");

    let mut last_players: Vec<core::Point> = Vec::new();
    let mut last_enemies: Vec<core::Point> = Vec::new();
    let mut last_valid_loc = core::Point::new(0, 0);
    let mut frame_count = 0usize;
    let full_path = "/tmp/tnt_fullscreen.png";

    loop {
        frame_count += 1;

        // 1. 截取全屏
        let _ = Command::new("screencapture")
            .arg("-x")
            .arg(full_path)
            .status();

        let full_img = match imgcodecs::imread(full_path, imgcodecs::IMREAD_COLOR) {
            Ok(m) if !m.empty() => m,
            _ => {
                thread::sleep(Duration::from_millis(300));
                continue;
            }
        };

        // 2. 利用模板匹配在全屏中精确定位用户框选的区域 (绝对不会偏到右上角/菜单栏)
        let mut match_result = core::Mat::default();
        imgproc::match_template(
            &full_img,
            &template,
            &mut match_result,
            imgproc::TM_CCOEFF_NORMED,
            &core::no_array(),
        )?;

        let mut max_val = 0.0;
        let mut max_loc = core::Point::new(0, 0);
        core::min_max_loc(
            &match_result,
            None,
            Some(&mut max_val),
            None,
            Some(&mut max_loc),
            &core::no_array(),
        )?;

        // 如果匹配度不够，很可能是小地图发生了爆炸动画导致画面改变，此时不要放弃，直接使用上一次的坐标！
        if max_val < 0.5 {
            if last_valid_loc.x == 0 && last_valid_loc.y == 0 {
                println!("⚠️ 全屏未找到框选区域 (匹配度 {:.2})，请确认游戏窗口未被完全遮挡...", max_val);
                thread::sleep(Duration::from_millis(500));
                continue;
            } else {
                // 回退使用上一次成功的坐标
                max_loc = last_valid_loc;
            }
        } else {
            // 更新最后一次成功匹配的坐标
            last_valid_loc = max_loc;
        }

        // 3. 准确裁剪出用户框选的画面区域
        let selected_roi = core::Rect::new(max_loc.x, max_loc.y, t_w, t_h);
        let img = core::Mat::roi(&full_img, selected_roi)?.try_clone()?;

        // 4. 判断用户框选的是小地图本身，还是包含了小地图的大窗口
        let minimap = if img.cols() > 250 && img.rows() > 180 {
            let roi = core::Rect::new(0, 0, 226.min(img.cols()), 136.min(img.rows()));
            core::Mat::roi(&img, roi)?.try_clone()?
        } else {
            img.try_clone()?
        };

        let camera_rect = detect_camera_frame(&minimap)?;
        let camera_width = if camera_rect.width > 0 {
            camera_rect.width as f64
        } else {
            170.0
        };

        let px_per_dist = camera_width / 12.0;

        let search_roi = core::Rect::new(0, 0, 250.min(img.cols()), 150.min(img.rows()));
        let search_map = core::Mat::roi(&img, search_roi)?.try_clone()?;

        let mut player_pts = find_dots_by_color_and_edge(&search_map, false)?;
        let mut enemy_pts = find_dots_by_color_and_edge(&search_map, true)?;

        player_pts.retain(|p| p.x <= 226 && p.y <= 136);
        enemy_pts.retain(|p| p.x <= 226 && p.y <= 136);

        // 位置变动去重检测
        let mut changed = false;
        if player_pts.len() != last_players.len() || enemy_pts.len() != last_enemies.len() {
            changed = true;
        } else {
            for (p1, p2) in player_pts.iter().zip(last_players.iter()) {
                if (p1.x - p2.x).abs() > 2 || (p1.y - p2.y).abs() > 2 {
                    changed = true;
                    break;
                }
            }
        }

        if changed || frame_count % 3 == 0 {
            last_players = player_pts.clone();
            last_enemies = enemy_pts.clone();

            // 生成可视化调试标记图
            let mut debug_map = minimap.try_clone()?;
            if camera_rect.width > 0 {
                let _ = imgproc::rectangle(
                    &mut debug_map,
                    camera_rect,
                    core::Scalar::new(0.0, 255.0, 0.0, 0.0),
                    2,
                    imgproc::LINE_8,
                    0,
                );
            }

            for pt in &player_pts {
                let _ = imgproc::circle(
                    &mut debug_map,
                    *pt,
                    4,
                    core::Scalar::new(255.0, 200.0, 0.0, 0.0),
                    -1,
                    imgproc::LINE_8,
                    0,
                );
            }

            for pt in &enemy_pts {
                let _ = imgproc::circle(
                    &mut debug_map,
                    *pt,
                    4,
                    core::Scalar::new(0.0, 0.0, 255.0, 0.0),
                    -1,
                    imgproc::LINE_8,
                    0,
                );
            }

            // 保存到 live_debug.png 供用户查看效果
            let debug_path = "live_debug.png";
            let _ = imgcodecs::imwrite(debug_path, &debug_map, &core::Vector::new());

            println!("==================================================");
            println!("⚡️ [实时更新 {:02}秒] 已保存调试预览图 -> live_debug.png", frame_count / 3);
            println!("=> 捕获我方(蓝点) {} 人, 敌方(红点) {} 人 (比例尺: 12距 = {:.1}px)", player_pts.len(), enemy_pts.len(), camera_width);

            if player_pts.is_empty() {
                println!("⚠️ 未在 ROI 内检测到蓝点 (请确认框选的是小地图区域)");
            }
            if enemy_pts.is_empty() {
                println!("⚠️ 未在 ROI 内检测到红点 (已把当前分析过程保存至 live_debug.png)");
            }

            for (p_idx, p_pt) in player_pts.iter().enumerate() {
                for (e_idx, e_pt) in enemy_pts.iter().enumerate() {
                    let dx_px = (e_pt.x - p_pt.x).abs() as f64;
                    let dx_dist = dx_px / px_per_dist;

                    let dy_px = (p_pt.y - e_pt.y) as f64;
                    let dy_dist = dy_px / px_per_dist;

                    let height_desc = if dy_dist.abs() < 0.2 {
                        "齐平".to_string()
                    } else if dy_dist > 0.0 {
                        format!("高 {:.1} 距 ⬆️", dy_dist)
                    } else {
                        format!("低 {:.1} 距 ⬇️", dy_dist.abs())
                    };

                    let power_30 = calc_power_30deg(dx_dist, dy_dist);
                    let power_65 = calc_power_65deg(dx_dist, dy_dist);

                    println!(
                        "🎯 [玩家 {} -> 敌人 {}]",
                        p_idx + 1,
                        e_idx + 1
                    );
                    println!(
                        "   ├─ 水平距离: {:.2} 距, 地形高低: {}",
                        dx_dist, height_desc
                    );
                    println!(
                        "   └─ 推荐开火力度: [30度平抛] {:.1} 力  |  [65度高抛] {:.1} 力",
                        power_30.max(0.0),
                        power_65.max(0.0)
                    );
                }
            }
            println!("==================================================\n");
        }

        thread::sleep(Duration::from_millis(300));
    }
}
