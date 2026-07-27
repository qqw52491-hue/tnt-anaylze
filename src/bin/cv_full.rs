use opencv::{
    core, core::ToInputArray, imgcodecs, imgproc,
    prelude::*,
    geometry::bounding_rect
};
use std::env;

struct EnemyTarget {
    pos: core::Point,
    game_distance: f64,
    power_30deg: f64,
    power_45deg: f64,
    power_65deg: f64,
}

fn calc_power_30deg(d: f64) -> f64 { (d * 30.0).sqrt() * 1.5 + 15.0 }
fn calc_power_45deg(d: f64) -> f64 { (d * 45.0).sqrt() * 1.2 + 20.0 }
fn calc_power_65deg(d: f64) -> f64 { (d * 65.0).sqrt() * 1.0 + 25.0 }

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

fn find_all_matches(img: &impl ToInputArray, tmpl: &core::Mat, match_threshold: f64) -> opencv::Result<Vec<core::Point>> {
    let mut result = core::Mat::default();
    imgproc::match_template(img, tmpl, &mut result, imgproc::TM_CCOEFF_NORMED, &core::no_array())?;
    
    // 阈值过滤
    let mut thresholded = core::Mat::default();
    imgproc::threshold(&result, &mut thresholded, match_threshold, 1.0, imgproc::THRESH_BINARY)?;
    
    // 转换为8位无符号，才能找非零像素
    let mut thresholded_8u = core::Mat::default();
    thresholded.convert_to(&mut thresholded_8u, core::CV_8U, 255.0, 0.0)?;
    
    let mut non_zero = core::Mat::default();
    core::find_non_zero(&thresholded_8u, &mut non_zero)?;
    
    if non_zero.empty() {
        return Ok(Vec::new());
    }
    
    // 聚类 (将多个重叠像素点合并为1个实体点)
    let mut centroids = cluster_points(&non_zero, 8)?;
    
    // 补偿模板的偏移量，让坐标落在圆心
    let offset_x = tmpl.cols() / 2;
    let offset_y = tmpl.rows() / 2;
    for pt in &mut centroids {
        pt.x += offset_x;
        pt.y += offset_y;
    }
    
    Ok(centroids)
}

fn main() -> opencv::Result<()> {
    println!(">>> 启动 cv_full 完全体引擎...");
    let args: Vec<String> = env::args().collect();
    let img_path = if args.len() > 1 { args[1].clone() } else { "截屏2026-07-26 11.27.46.png".to_string() };

    let img = imgcodecs::imread(&img_path, imgcodecs::IMREAD_COLOR)?;
    let template_red = imgcodecs::imread("template_red_dot.png", imgcodecs::IMREAD_COLOR)?;
    let template_blue = imgcodecs::imread("template_blue_dot.png", imgcodecs::IMREAD_COLOR)?;

    if img.empty() {
        eprintln!("无法加载图片: {}", img_path);
        return Ok(());
    }

    println!("正在分析图像...");

    // 0. 截取小地图区域 (假设固定在左上角 226x136)
    // 这样可以避免在大图中识别到绿色的草地或其他干扰
    let roi = core::Rect::new(0, 0, 226, 136);
    let minimap = core::Mat::roi(&img, roi)?;

    // 1. 动态获取绿框 (视野框) 并计算占比12距的宽度
    let mut hsv = core::Mat::default();
    imgproc::cvt_color(&minimap, &mut hsv, imgproc::COLOR_BGR2HSV, 0, core::AlgorithmHint::ALGO_HINT_DEFAULT)?;
    
    // 专门针对“荧光绿”的细线，收紧 HSV 范围
    let lower_green = core::Scalar::new(35.0, 100.0, 100.0, 0.0);
    let upper_green = core::Scalar::new(85.0, 255.0, 255.0, 0.0);
    let mut mask_green = core::Mat::default();
    core::in_range(&hsv, &lower_green, &upper_green, &mut mask_green)?;
    
    let mut contours_green = core::Vector::<core::Vector<core::Point>>::new();
    imgproc::find_contours(&mask_green, &mut contours_green, imgproc::RETR_EXTERNAL, imgproc::CHAIN_APPROX_SIMPLE, core::Point::new(0, 0))?;
    
    let mut camera_width = 170.0; // 默认值 fallback
    for i in 0..contours_green.len() {
        let rect = bounding_rect(&contours_green.get(i)?)?;
        // 绿框的宽度通常在 100~226 之间，高度很小 (细线)
        if rect.width > 50 && rect.width <= 226 { 
            camera_width = rect.width as f64;
            // 选最大的那个
        }
    }
    println!("=> [核心参数] 成功捕获绿色视野框！占比12距对应的像素宽度为: {}", camera_width);

    // 2. 地毯式搜索红蓝点
    let player_pts = find_all_matches(&minimap, &template_blue, 0.80)?;
    let enemy_pts = find_all_matches(&minimap, &template_red, 0.80)?;

    println!("=> 找到 {} 个蓝点 (玩家)！", player_pts.len());
    for (i, pt) in player_pts.iter().enumerate() {
        println!("   玩家 {} 坐标: ({}, {})", i+1, pt.x, pt.y);
    }
    
    println!("=> 找到 {} 个红点 (敌人)！", enemy_pts.len());
    for (i, pt) in enemy_pts.iter().enumerate() {
        println!("   敌人 {} 坐标: ({}, {})", i+1, pt.x, pt.y);
    }
    
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
        
        println!("[敌人 {}] 距离玩家1: {:.2} 距", i+1, game_distance);
        println!("    -> 推荐力度 [30度]: {:.1} 力", p30);
        println!("    -> 推荐力度 [45度]: {:.1} 力", p45);
        println!("    -> 推荐力度 [65度]: {:.1} 力", p65);
        println!("------------------------------------------------");
    }

    Ok(())
}
