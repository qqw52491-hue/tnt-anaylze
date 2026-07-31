use opencv::{core, core::ToInputArray, geometry::bounding_rect, imgcodecs, imgproc, prelude::*};
use std::env;

struct EnemyTarget {
    pos: core::Point,
    game_distance: f64,
    power_30deg: f64,
    power_45deg: f64,
    power_65deg: f64,
}

fn calc_power_30deg(d: f64) -> f64 {
    (d * 30.0).sqrt() * 1.5 + 15.0
}
fn calc_power_45deg(d: f64) -> f64 {
    (d * 45.0).sqrt() * 1.2 + 20.0
}
fn calc_power_65deg(d: f64) -> f64 {
    (d * 65.0).sqrt() * 1.0 + 25.0
}

// 非极大值抑制 / 距离聚类：将挨得很近的匹配点合并为一个真正的物理圆点
fn cluster_points(points: &core::Mat, threshold: i32) -> opencv::Result<Vec<core::Point>> {
    let mut clusters: Vec<Vec<core::Point>> = Vec::new();

    // find_non_zero 返回的是 Nx1 的 Mat，每个元素是一个 Point
    let count = points.total();
    for i in 0..count {
        let p: core::Point = *points.at::<core::Point>(i as i32)?;

        let mut added = false;
        for cluster in &mut clusters {
            let mut close = false;
            for &cp in cluster.iter() {
                let dx = (cp.x - p.x).abs();
                let dy = (cp.y - p.y).abs();
                if dx <= threshold && dy <= threshold {
                    close = true;
                    break;
                }
            }
            if close {
                cluster.push(p);
                added = true;
                break;
            }
        }
        if !added {
            clusters.push(vec![p]);
        }
    }

    // 计算每个聚类的质心
    let mut centroids = Vec::new();
    for cluster in clusters {
        let sum_x: i32 = cluster.iter().map(|p| p.x).sum();
        let sum_y: i32 = cluster.iter().map(|p| p.y).sum();
        let len = cluster.len() as i32;
        centroids.push(core::Point::new(sum_x / len, sum_y / len));
    }

    Ok(centroids)
}

fn find_dots_by_color_and_edge(
    img_input: &impl core::ToInputArray,
    is_red: bool,
) -> opencv::Result<Vec<core::Point>> {
    let minimap = img_input.input_array()?.get_mat(-1)?;
    let mut hsv = core::Mat::default();
    imgproc::cvt_color(
        &minimap,
        &mut hsv,
        imgproc::COLOR_BGR2HSV,
        0,
        core::AlgorithmHint::ALGO_HINT_DEFAULT,
    )?;

    let mut mask = core::Mat::default();
    if is_red {
        // 红色高亮区间 (放宽 S 和 V 捕捉极其微弱的红点)
        let mut mask1 = core::Mat::default();
        core::in_range(
            &hsv,
            &core::Scalar::new(0.0, 50.0, 50.0, 0.0),
            &core::Scalar::new(8.0, 255.0, 255.0, 0.0),
            &mut mask1,
        )?;
        let mut mask2 = core::Mat::default();
        core::in_range(
            &hsv,
            &core::Scalar::new(170.0, 50.0, 50.0, 0.0),
            &core::Scalar::new(180.0, 255.0, 255.0, 0.0),
            &mut mask2,
        )?;
        core::add(&mask1, &mask2, &mut mask, &core::no_array(), -1)?;
    } else {
        // 蓝色/青色高亮区间 (扩展 H 范围到 80，放宽 S 和 V)
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

    let mut valid_points = Vec::new();
    for i in 0..contours.len() {
        let contour = contours.get(i)?;
        let rect = opencv::geometry::bounding_rect(&contour)?;
        let area = rect.width * rect.height;
        let is_color = if is_red { "红" } else { "蓝" };
        if area >= 10 {
            println!("{}点候选区域面积: {}, 宽高: {}x{}", is_color, area, rect.width, rect.height);
        }

        // 过滤掉太大或太小的噪点 (真正的红蓝点面积一般在 16~36 左右)
        // 宽度高度不超过 9 像素，过滤掉长条形的地图色块
        if area >= 12 && area <= 64 && rect.width <= 9 && rect.height <= 9 {
            let aspect = rect.width as f64 / rect.height as f64;
            // 像素非常少时，长宽比可能达到 0.66 (4x6) 或 1.5 (6x4)，适当放宽
            if aspect > 0.5 && aspect < 2.0 {
                // 黑边检测逻辑
                let mut dark_pixels = 0;
                let mut total_edge_pixels = 0;
                let margin = 1;
                let start_x = (rect.x - margin).max(0);
                let start_y = (rect.y - margin).max(0);
                let end_x = (rect.x + rect.width + margin).min(minimap.cols() - 1);
                let end_y = (rect.y + rect.height + margin).min(minimap.rows() - 1);

                for y in start_y..=end_y {
                    for x in start_x..=end_x {
                        if x < rect.x
                            || x >= rect.x + rect.width
                            || y < rect.y
                            || y >= rect.y + rect.height
                        {
                            total_edge_pixels += 1;
                            let p = minimap.at_2d::<core::Vec3b>(y, x)?;
                            let b = p[0] as i32;
                            let g = p[1] as i32;
                            let r = p[2] as i32;
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
                
                // 排除紧贴在小地图边缘的噪点（通常是边框或 UI 按钮，不是真正的玩家）
                let is_near_border = rect.x <= 3
                    || rect.y <= 3
                    || rect.x + rect.width >= minimap.cols() - 3
                    || rect.y + rect.height >= minimap.rows() - 3;

                // 只要边缘有 5% 以上是暗色，并且不紧贴边缘，就承认它是真正的目标
                if dark_ratio >= 0.05 && !is_near_border {
                    valid_points.push(core::Point::new(
                        rect.x + rect.width / 2,
                        rect.y + rect.height / 2,
                    ));
                }
            }
        }
    }

    // NMS去重
    let mut final_points: Vec<core::Point> = Vec::new();
    for p in valid_points {
        let mut is_duplicate = false;
        for existing in &mut final_points {
            let dx = existing.x - p.x;
            let dy = existing.y - p.y;
            let dist = ((dx * dx + dy * dy) as f64).sqrt();
            if dist < 10.0 {
                is_duplicate = true;
                break;
            }
        }
        if !is_duplicate {
            final_points.push(p);
        }
    }
    Ok(final_points)
}

// ====== 自动检测视野框 ======
// 算法原理：使用 HSV 颜色掩码过滤游戏 UI 素材固定的视野框边框，配合 boundingRect 取最大矩形轮廓。
fn detect_camera_frame(minimap: &core::Mat) -> opencv::Result<core::Rect> {
    let mut hsv = core::Mat::default();
    imgproc::cvt_color(
        minimap,
        &mut hsv,
        imgproc::COLOR_BGR2HSV,
        0,
        core::AlgorithmHint::ALGO_HINT_DEFAULT,
    )?;

    // 视野框素材 HSV 范围 (黄色/绿色边框)
    let lower_yellow_green = core::Scalar::new(15.0, 60.0, 60.0, 0.0);
    let upper_yellow_green = core::Scalar::new(85.0, 255.0, 255.0, 0.0);

    let mut mask = core::Mat::default();
    core::in_range(&hsv, &lower_yellow_green, &upper_yellow_green, &mut mask)?;

    let mut contours = core::Vector::<core::Vector<core::Point>>::new();
    imgproc::find_contours(
        &mask,
        &mut contours,
        imgproc::RETR_EXTERNAL,
        imgproc::CHAIN_APPROX_SIMPLE,
        core::Point::new(0, 0),
    )?;

    let mut best_rect = core::Rect::new(0, 0, 90, 60);
    let mut max_area = 0;

    for i in 0..contours.len() {
        let contour = contours.get(i)?;
        let rect = opencv::geometry::bounding_rect(&contour)?;
        let area = rect.width * rect.height;
        if rect.width >= 40 && rect.width <= 226 && area > max_area {
            max_area = area;
            best_rect = rect;
        }
    }

    Ok(best_rect)
}

fn main() -> opencv::Result<()> {
    println!(">>> 启动 cv_full 完全体引擎...");
    let args: Vec<String> = std::env::args().collect();
    let img_path = if args.len() > 1 {
        args[1].clone()
    } else {
        "截屏2026-07-26 13.40.25.png".to_string()
    };

    let img = imgcodecs::imread(&img_path, imgcodecs::IMREAD_COLOR)?;
    if img.empty() {
        eprintln!("无法加载图片: {}", img_path);
        return Ok(());
    }

    println!("正在分析图像...");

    // 0. 严格约束在左上角小地图内，防止大图中其他 UI 干扰
    let roi = core::Rect::new(0, 0, 226, 136);
    let minimap_ref = core::Mat::roi(&img, roi)?;
    let minimap = minimap_ref.try_clone()?;
    let mut debug_map = minimap.try_clone()?;

    // 1. 动态获取视野框并计算占比12距的宽度
    let camera_rect = detect_camera_frame(&minimap)?;
    let mut camera_width = 170.0; // 默认值 fallback

    if camera_rect.width > 0 {
        camera_width = camera_rect.width as f64;
        println!(
            "=> [核心参数] 成功捕获视野框！占比12距对应的像素宽度为: {}",
            camera_width
        );
        imgproc::rectangle(
            &mut debug_map,
            camera_rect,
            core::Scalar::new(0.0, 255.0, 0.0, 0.0),
            1,
            imgproc::LINE_8,
            0,
        )?;
        let cam_center = core::Point::new(
            camera_rect.x + camera_rect.width / 2,
            camera_rect.y + camera_rect.height / 2,
        );
        // 画十字准星
        imgproc::line(
            &mut debug_map,
            core::Point::new(cam_center.x - 5, cam_center.y),
            core::Point::new(cam_center.x + 5, cam_center.y),
            core::Scalar::new(0.0, 255.0, 0.0, 0.0),
            1,
            imgproc::LINE_8,
            0,
        )?;
        imgproc::line(
            &mut debug_map,
            core::Point::new(cam_center.x, cam_center.y - 5),
            core::Point::new(cam_center.x, cam_center.y + 5),
            core::Scalar::new(0.0, 255.0, 0.0, 0.0),
            1,
            imgproc::LINE_8,
            0,
        )?;
    } else {
        println!("=> [警告] 未能检测到视野框，使用默认宽度 170.0");
    }

    // 不要过早裁剪！小地图边缘的点(比如 223, 134)如果紧贴 226x136 的边缘，会被裁剪掉半个黑边导致识别失败
    // 所以我们在一个稍大一点的区域 (比如 250x150) 进行搜索，搜完之后再过滤掉超出 226x136 的点
    let search_roi = core::Rect::new(0, 0, 250.min(img.cols()), 150.min(img.rows()));
    let search_map = core::Mat::roi(&img, search_roi)?.try_clone()?;

    let mut player_pts = find_dots_by_color_and_edge(&search_map, false)?;
    let mut enemy_pts = find_dots_by_color_and_edge(&search_map, true)?;

    // 过滤掉真正小地图区域之外的误判点
    player_pts.retain(|p| p.x <= 226 && p.y <= 136);
    enemy_pts.retain(|p| p.x <= 226 && p.y <= 136);

    println!("=> 找到 {} 个蓝点 (玩家)！", player_pts.len());
    for (i, pt) in player_pts.iter().enumerate() {
        println!("   玩家 {} 坐标: ({}, {})", i + 1, pt.x, pt.y);
        imgproc::circle(
            &mut debug_map,
            *pt,
            3,
            core::Scalar::new(255.0, 200.0, 0.0, 0.0),
            -1,
            imgproc::LINE_8,
            0,
        )?;
    }

    println!("=> 找到 {} 个红点 (敌人)！", enemy_pts.len());
    for (i, pt) in enemy_pts.iter().enumerate() {
        println!("   敌人 {} 坐标: ({}, {})", i + 1, pt.x, pt.y);
        imgproc::circle(
            &mut debug_map,
            *pt,
            3,
            core::Scalar::new(0.0, 200.0, 255.0, 0.0),
            -1,
            imgproc::LINE_8,
            0,
        )?;
    }

    // 放大调试图
    let mut big = core::Mat::default();
    imgproc::resize(
        &debug_map,
        &mut big,
        core::Size::new(226 * 4, 136 * 4),
        0.0,
        0.0,
        imgproc::INTER_NEAREST,
    )?;
    imgcodecs::imwrite("output_debug.png", &big, &core::Vector::new())?;
    println!("已生成调试图像 output_debug.png");

    // 3. 计算物理参数
    if player_pts.is_empty() {
        println!("未找到玩家，无法计算相对距离。");
        return Ok(());
    }
    let p1 = player_pts[0]; // 以第一个蓝点为基准

    println!("\n================ 力度计算结果 ================");
    for (i, e) in enemy_pts.iter().enumerate() {
        let dx_pixels = (p1.x - e.x).abs() as f64;
        let game_distance = (dx_pixels / camera_width) * 12.0;

        let p30 = calc_power_30deg(game_distance);
        let p45 = calc_power_45deg(game_distance);
        let p65 = calc_power_65deg(game_distance);

        println!("[敌人 {}] 距离玩家1: {:.2} 距", i + 1, game_distance);
        println!("    -> 推荐力度 [30度]: {:.1} 力", p30);
        println!("    -> 推荐力度 [45度]: {:.1} 力", p45);
        println!("    -> 推荐力度 [65度]: {:.1} 力", p65);
        println!("------------------------------------------------");
    }

    // 4. 绘制可视化调试图并保存
    let mut debug_img = img.clone();

    let mut output_img = minimap.try_clone()?;

    // 画出视野框
    if camera_rect.width > 0 {
        imgproc::rectangle(
            &mut output_img,
            camera_rect,
            core::Scalar::new(0.0, 255.0, 0.0, 0.0), // 绿色
            1,
            imgproc::LINE_8,
            0,
        )?;
    }

    for (i, pt) in player_pts.iter().enumerate() {
        imgproc::circle(
            &mut output_img,
            *pt,
            3,
            core::Scalar::new(255.0, 0.0, 0.0, 0.0),
            -1,
            imgproc::LINE_8,
            0,
        )?;
        imgproc::put_text(
            &mut output_img,
            &format!("P{}", i + 1),
            core::Point::new(pt.x + 5, pt.y),
            imgproc::FONT_HERSHEY_SIMPLEX,
            0.4,
            core::Scalar::new(255.0, 0.0, 0.0, 0.0),
            1,
            imgproc::LINE_8,
            false,
        )?;
    }

    for pt in &enemy_pts {
        imgproc::circle(
            &mut output_img,
            *pt,
            8,
            core::Scalar::new(0.0, 0.0, 255.0, 0.0),
            2,
            imgproc::LINE_8,
            0,
        )?;
    }

    imgcodecs::imwrite("output_debug.png", &output_img, &core::Vector::new())?;
    println!("已生成可视化标注调试图: output_debug.png");

    Ok(())
}
