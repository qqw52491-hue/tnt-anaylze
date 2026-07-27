use opencv::{
    core, imgcodecs, imgproc,
    prelude::*,
    geometry::bounding_rect
};

fn main() -> opencv::Result<()> {
    let img = imgcodecs::imread("截屏2026-07-26 11.27.46.png", imgcodecs::IMREAD_COLOR)?;
    let roi = core::Rect::new(0, 0, 226, 136);
    let minimap = core::Mat::roi(&img, roi)?;
    
    let mut hsv = core::Mat::default();
    imgproc::cvt_color(&minimap, &mut hsv, imgproc::COLOR_BGR2HSV, 0, core::AlgorithmHint::ALGO_HINT_DEFAULT)?;
    
    // 红色 HSV 区间 (扩大包容度)
    let lower_red1 = core::Scalar::new(0.0, 100.0, 100.0, 0.0);
    let upper_red1 = core::Scalar::new(10.0, 255.0, 255.0, 0.0);
    let mut mask_red1 = core::Mat::default();
    core::in_range(&hsv, &lower_red1, &upper_red1, &mut mask_red1)?;
    
    let lower_red2 = core::Scalar::new(160.0, 100.0, 100.0, 0.0);
    let upper_red2 = core::Scalar::new(180.0, 255.0, 255.0, 0.0);
    let mut mask_red2 = core::Mat::default();
    core::in_range(&hsv, &lower_red2, &upper_red2, &mut mask_red2)?;
    
    let mut mask_red = core::Mat::default();
    core::add(&mask_red1, &mask_red2, &mut mask_red, &core::no_array(), -1)?;
    
    let mut contours_red = core::Vector::<core::Vector<core::Point>>::new();
    imgproc::find_contours(&mask_red, &mut contours_red, imgproc::RETR_EXTERNAL, imgproc::CHAIN_APPROX_SIMPLE, core::Point::new(0, 0))?;
    
    println!("=== 开始严格形状过滤 (只找圆形红点) ===");
    for i in 0..contours_red.len() {
        let contour = contours_red.get(i)?;
        let area = opencv::geometry::contour_area(&contour, false)?;
        let rect = bounding_rect(&contour)?;
        let perimeter = opencv::geometry::arc_length(&contour, true)?;
        
        if area > 10.0 && area < 400.0 { // 过滤掉太小或太大的噪点
            // 圆度计算公式: 4 * pi * Area / Perimeter^2 (越接近1越圆)
            let circularity = if perimeter > 0.0 { 4.0 * std::f64::consts::PI * area / (perimeter * perimeter) } else { 0.0 };
            
            // 宽高比 (越接近1越圆)
            let aspect_ratio = rect.width as f64 / rect.height as f64;
            
            println!("候选点 [坐标 ({}, {})] - 面积: {:.1}, 宽高比: {:.2}, 圆度: {:.2}", 
                     rect.x + rect.width/2, rect.y + rect.height/2, area, aspect_ratio, circularity);
                     
            if circularity > 0.6 && aspect_ratio > 0.6 && aspect_ratio < 1.4 {
                println!(">>> 🏆 确认为真正红点！坐标 ({}, {})", rect.x + rect.width/2, rect.y + rect.height/2);
            }
        }
    }
    
    Ok(())
}
