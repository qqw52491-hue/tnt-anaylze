use opencv::{core, imgcodecs, imgproc, prelude::*, videoio};
use std::env;
use std::time::Instant;

// ====== 色彩与几何黑边算法找红蓝点 ======
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

// ====== 视野框检测 ======
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

fn calc_power_30deg(dx: f64, dy: f64) -> f64 {
    let base = (dx * 30.0).sqrt() * 1.5 + 15.0;
    // dy > 0 表示敌人比玩家低，需要扣减力度；dy < 0 表示敌人高，需要增加力度
    base - dy * 0.8
}

fn calc_power_65deg(dx: f64, dy: f64) -> f64 {
    let base = (dx * 65.0).sqrt() * 1.0 + 25.0;
    base - dy * 1.0
}

fn main() -> opencv::Result<()> {
    let args: Vec<String> = env::args().collect();
    let video_path = if args.len() > 1 {
        args[1].clone()
    } else {
        println!("用法: cargo run --bin video_analyzer -- <视频文件路径.mp4/mov>");
        println!("提示: 如果未传参，程序将展示模拟流程\n");
        "sample_video.mp4".to_string()
    };

    let mut cap = match videoio::VideoCapture::from_file(&video_path, videoio::CAP_ANY) {
        Ok(c) => c,
        Err(_) => {
            println!("无法打开视频文件或文件不存在: {}", video_path);
            println!("💡 请提供有效视频路径，如: cargo run --bin video_analyzer -- ~/Desktop/gameplay.mp4");
            return Ok(());
        }
    };

    if !cap.is_opened()? {
        println!("VideoCapture 无法读取视频内容: {}", video_path);
        return Ok(());
    }

    let fps = cap.get(videoio::CAP_PROP_FPS)?;
    let frame_count = cap.get(videoio::CAP_PROP_FRAME_COUNT)?;
    println!(">>> 成功加载游戏视频文件: {}", video_path);
    println!(">>> 视频帧率: {:.1} FPS, 总帧数: {}\n", fps, frame_count);

    let mut frame_idx = 0;
    let mut frame = core::Mat::default();

    let sample_interval = (fps.max(1.0) * 1.0) as usize; // 每隔1.0秒抽查1帧分析

    let mut last_players: Vec<core::Point> = Vec::new();
    let mut last_enemies: Vec<core::Point> = Vec::new();

    while cap.read(&mut frame)? {
        frame_idx += 1;

        if frame.empty() {
            break;
        }

        // 定时抽帧分析（避免每一帧重复计算）
        if frame_idx % sample_interval != 0 {
            continue;
        }

        let timestamp_sec = frame_idx as f64 / fps.max(1.0);

        let roi = core::Rect::new(0, 0, 226.min(frame.cols()), 136.min(frame.rows()));
        let minimap_ref = core::Mat::roi(&frame, roi)?;
        let minimap = minimap_ref.try_clone()?;

        let camera_rect = detect_camera_frame(&minimap)?;
        let camera_width = if camera_rect.width > 0 {
            camera_rect.width as f64
        } else {
            170.0
        };

        // 12 距对应的像素比例尺
        let px_per_dist = camera_width / 12.0;

        let search_roi = core::Rect::new(0, 0, 250.min(frame.cols()), 150.min(frame.rows()));
        let search_map = core::Mat::roi(&frame, search_roi)?.try_clone()?;

        let mut player_pts = find_dots_by_color_and_edge(&search_map, false)?;
        let mut enemy_pts = find_dots_by_color_and_edge(&search_map, true)?;

        player_pts.retain(|p| p.x <= 226 && p.y <= 136);
        enemy_pts.retain(|p| p.x <= 226 && p.y <= 136);

        // 如果坐标变化不大，跳过重复打印
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

        if !changed {
            continue;
        }

        last_players = player_pts.clone();
        last_enemies = enemy_pts.clone();

        let minutes = (timestamp_sec / 60.0) as u32;
        let seconds = (timestamp_sec % 60.0) as u32;

        println!("==================================================");
        println!("⏱️ [视频时间戳 {:02}:{:02}] 发现战场位置更新！", minutes, seconds);
        println!("=> 检测到我方(蓝点) {} 人, 敌方(红点) {} 人", player_pts.len(), enemy_pts.len());
        println!("=> 当前地图比例尺: 12距 = {:.1} 像素\n", camera_width);

        for (p_idx, p_pt) in player_pts.iter().enumerate() {
            for (e_idx, e_pt) in enemy_pts.iter().enumerate() {
                // 水平物理距离 dx (距)
                let dx_px = (e_pt.x - p_pt.x).abs() as f64;
                let dx_dist = dx_px / px_per_dist;

                // 垂直高低差 dy (距)  (Y坐标在图像中向下增大，所以 p_pt.y - e_pt.y > 0 表示敌人更高)
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
                    "   └─ 建议开火力度: [30度平抛] {:.1} 力  |  [65度高抛] {:.1} 力",
                    power_30.max(0.0),
                    power_65.max(0.0)
                );
            }
        }
        println!("==================================================\n");
    }

    println!(">>> 视频分析完成！");
    Ok(())
}
