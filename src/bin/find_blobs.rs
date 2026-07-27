use opencv::{
    core, imgcodecs, imgproc,
    prelude::*,
    geometry::bounding_rect
};

fn main() -> opencv::Result<()> {
    let img = imgcodecs::imread("截屏2026-07-26 11.27.46.png", imgcodecs::IMREAD_COLOR)?;
    let roi = core::Rect::new(0, 0, 250, 180);
    let minimap = core::Mat::roi(&img, roi)?;
    
    let mut hsv = core::Mat::default();
    imgproc::cvt_color(&minimap, &mut hsv, imgproc::COLOR_BGR2HSV, 0, core::AlgorithmHint::ALGO_HINT_DEFAULT)?;
    
    // 红色有两个区间 (H: 0-10 和 170-180)
    let lower_red1 = core::Scalar::new(0.0, 100.0, 100.0, 0.0);
    let upper_red1 = core::Scalar::new(10.0, 255.0, 255.0, 0.0);
    let mut mask_red1 = core::Mat::default();
    core::in_range(&hsv, &lower_red1, &upper_red1, &mut mask_red1)?;
    
    let lower_red2 = core::Scalar::new(170.0, 100.0, 100.0, 0.0);
    let upper_red2 = core::Scalar::new(180.0, 255.0, 255.0, 0.0);
    let mut mask_red2 = core::Mat::default();
    core::in_range(&hsv, &lower_red2, &upper_red2, &mut mask_red2)?;
    
    let mut mask_red = core::Mat::default();
    core::add(&mask_red1, &mask_red2, &mut mask_red, &core::no_array(), -1)?;
    
    // 蓝色区间
    let lower_blue = core::Scalar::new(100.0, 100.0, 100.0, 0.0);
    let upper_blue = core::Scalar::new(140.0, 255.0, 255.0, 0.0);
    let mut mask_blue = core::Mat::default();
    core::in_range(&hsv, &lower_blue, &upper_blue, &mut mask_blue)?;
    
    // 找红点轮廓
    let mut contours_red = core::Vector::<core::Vector<core::Point>>::new();
    imgproc::find_contours(&mask_red, &mut contours_red, imgproc::RETR_EXTERNAL, imgproc::CHAIN_APPROX_SIMPLE, core::Point::new(0, 0))?;
    
    let mut red_count = 0;
    for i in 0..contours_red.len() {
        let rect = bounding_rect(&contours_red.get(i)?)?;
        if rect.width >= 3 && rect.width <= 20 && rect.height >= 3 && rect.height <= 20 {
            println!("找到合适的红点 (长宽 {}x{}): ({}, {})", rect.width, rect.height, rect.x + rect.width/2, rect.y + rect.height/2);
            red_count += 1;
        }
    }
    println!("共找到 {} 个真正的红点。", red_count);
    
    // 找蓝点轮廓
    let mut contours_blue = core::Vector::<core::Vector<core::Point>>::new();
    imgproc::find_contours(&mask_blue, &mut contours_blue, imgproc::RETR_EXTERNAL, imgproc::CHAIN_APPROX_SIMPLE, core::Point::new(0, 0))?;
    
    let mut blue_count = 0;
    for i in 0..contours_blue.len() {
        let rect = bounding_rect(&contours_blue.get(i)?)?;
        if rect.width >= 3 && rect.width <= 20 && rect.height >= 3 && rect.height <= 20 {
            println!("找到合适的蓝点 (长宽 {}x{}): ({}, {})", rect.width, rect.height, rect.x + rect.width/2, rect.y + rect.height/2);
            blue_count += 1;
        }
    }
    println!("共找到 {} 个真正的蓝点。", blue_count);
    
    Ok(())
}
