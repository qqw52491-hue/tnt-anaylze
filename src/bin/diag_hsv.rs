use opencv::{
    core, imgcodecs, imgproc,
    prelude::*,
    geometry::bounding_rect
};

fn main() -> opencv::Result<()> {
    let img = imgcodecs::imread("截屏2026-07-26 13.40.25.png", imgcodecs::IMREAD_COLOR)?;
    if img.empty() {
        println!("图片没找到！");
        return Ok(());
    }
    
    let roi = core::Rect::new(0, 0, 226, 136);
    let minimap = core::Mat::roi(&img, roi)?;
    
    let mut hsv = core::Mat::default();
    imgproc::cvt_color(&minimap, &mut hsv, imgproc::COLOR_BGR2HSV, 0, core::AlgorithmHint::ALGO_HINT_DEFAULT)?;
    
    // ==========================================
    // 严格的高亮高饱和度红色 (过滤暗色/低饱和度的棕色泥土)
    // 红色 H=0~10 或 170~180。S>150, V>150
    // ==========================================
    let lower_red1 = core::Scalar::new(0.0, 150.0, 150.0, 0.0);
    let upper_red1 = core::Scalar::new(10.0, 255.0, 255.0, 0.0);
    let mut mask_red1 = core::Mat::default();
    core::in_range(&hsv, &lower_red1, &upper_red1, &mut mask_red1)?;
    
    let lower_red2 = core::Scalar::new(170.0, 150.0, 150.0, 0.0);
    let upper_red2 = core::Scalar::new(180.0, 255.0, 255.0, 0.0);
    let mut mask_red2 = core::Mat::default();
    core::in_range(&hsv, &lower_red2, &upper_red2, &mut mask_red2)?;
    
    let mut mask_red = core::Mat::default();
    core::add(&mask_red1, &mask_red2, &mut mask_red, &core::no_array(), -1)?;
    
    let mut contours_red = core::Vector::<core::Vector<core::Point>>::new();
    imgproc::find_contours(&mask_red, &mut contours_red, imgproc::RETR_EXTERNAL, imgproc::CHAIN_APPROX_SIMPLE, core::Point::new(0, 0))?;
    
    println!("=== 测试极其严格的 HSV 红色过滤 (免疫背景) ===");
    for i in 0..contours_red.len() {
        let contour = contours_red.get(i)?;
        let rect = bounding_rect(&contour)?;
        let area = rect.width * rect.height;
        let aspect = rect.width as f64 / rect.height as f64;
        
        if area > 10 && area < 400 && aspect > 0.5 && aspect < 2.0 {
            println!("🎯 找到纯正红点: 坐标 ({}, {}), 大小: {}x{}", rect.x + rect.width/2, rect.y + rect.height/2, rect.width, rect.height);
        }
    }
    
    // ==========================================
    // 严格的高亮高饱和度蓝色
    // ==========================================
    let lower_blue = core::Scalar::new(100.0, 150.0, 100.0, 0.0);
    let upper_blue = core::Scalar::new(140.0, 255.0, 255.0, 0.0);
    let mut mask_blue = core::Mat::default();
    core::in_range(&hsv, &lower_blue, &upper_blue, &mut mask_blue)?;
    
    let mut contours_blue = core::Vector::<core::Vector<core::Point>>::new();
    imgproc::find_contours(&mask_blue, &mut contours_blue, imgproc::RETR_EXTERNAL, imgproc::CHAIN_APPROX_SIMPLE, core::Point::new(0, 0))?;
    
    println!("=== 测试极其严格的 HSV 蓝色过滤 (免疫背景) ===");
    for i in 0..contours_blue.len() {
        let contour = contours_blue.get(i)?;
        let rect = bounding_rect(&contour)?;
        let area = rect.width * rect.height;
        let aspect = rect.width as f64 / rect.height as f64;
        
        if area > 10 && area < 400 && aspect > 0.5 && aspect < 2.0 {
            println!("🔵 找到纯正蓝点: 坐标 ({}, {}), 大小: {}x{}", rect.x + rect.width/2, rect.y + rect.height/2, rect.width, rect.height);
        }
    }
    
    Ok(())
}
