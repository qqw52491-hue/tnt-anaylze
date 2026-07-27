use opencv::{core, imgcodecs, imgproc, prelude::*};
use std::time::Instant;

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
            &core::Scalar::new(0.0, 70.0, 70.0, 0.0),
            &core::Scalar::new(5.0, 255.0, 255.0, 0.0),
            &mut mask1,
        )?;
        let mut mask2 = core::Mat::default();
        core::in_range(
            &hsv,
            &core::Scalar::new(175.0, 70.0, 70.0, 0.0),
            &core::Scalar::new(180.0, 255.0, 255.0, 0.0),
            &mut mask2,
        )?;
        core::add(&mask1, &mask2, &mut mask, &core::no_array(), -1)?;
    } else {
        core::in_range(
            &hsv,
            &core::Scalar::new(80.0, 70.0, 70.0, 0.0),
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

fn analyze_frame(img: &core::Mat) -> opencv::Result<(usize, usize)> {
    let roi = core::Rect::new(0, 0, 226, 136);
    let minimap_ref = core::Mat::roi(img, roi)?;
    let minimap = minimap_ref.try_clone()?;

    let _camera_rect = detect_camera_frame(&minimap)?;

    let search_roi = core::Rect::new(0, 0, 250.min(img.cols()), 150.min(img.rows()));
    let search_map = core::Mat::roi(img, search_roi)?.try_clone()?;

    let mut player_pts = find_dots_by_color_and_edge(&search_map, false)?;
    let mut enemy_pts = find_dots_by_color_and_edge(&search_map, true)?;

    player_pts.retain(|p| p.x <= 226 && p.y <= 136);
    enemy_pts.retain(|p| p.x <= 226 && p.y <= 136);

    Ok((player_pts.len(), enemy_pts.len()))
}

fn main() -> opencv::Result<()> {
    let img_path = "截屏2026-07-26 13.40.25.png";
    let img = imgcodecs::imread(img_path, imgcodecs::IMREAD_COLOR)?;
    if img.empty() {
        eprintln!("无法加载测试图片: {}", img_path);
        return Ok(());
    }

    println!(">>> 启动【实时连续分析 - FPS 性能模拟测试】...");
    println!(">>> 将模拟真实游戏中的“连续内存分析”模式，运行 100 帧循环...\n");

    let iterations = 100;
    let start_total = Instant::now();

    let mut player_count = 0;
    let mut enemy_count = 0;

    for _i in 0..iterations {
        let (p_cnt, e_cnt) = analyze_frame(&img)?;
        player_count = p_cnt;
        enemy_count = e_cnt;
    }

    let elapsed_total = start_total.elapsed();
    let avg_ms = elapsed_total.as_secs_f64() * 1000.0 / (iterations as f64);
    let fps = 1000.0 / avg_ms;

    println!("================ 性能测试统计报告 ================");
    println!("测试帧数: {} 帧", iterations);
    println!("总共耗时: {:.2?} ms", elapsed_total.as_millis());
    println!("单帧平均处理时间: {:.2} ms", avg_ms);
    println!("估算极限吞吐帧率: {:.1} FPS", fps);
    println!("单帧结果: 找到 {} 个玩家点，{} 个敌人点", player_count, enemy_count);
    println!("==================================================");

    if avg_ms < 100.0 {
        println!("\n✅ [结论] 性能完全达标！单帧处理时间远远小于 100ms。");
        println!("这意味着如果接上“屏幕画面抓取”，你可以无压力实现每秒 10~30 次的实时游戏画面检测！");
    } else {
        println!("\n⚠️ [结论] 处理时间大于 100ms，需进一步微调加速。");
    }

    Ok(())
}
