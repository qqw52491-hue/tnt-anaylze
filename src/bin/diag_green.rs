use opencv::{
    core, imgcodecs, imgproc,
    prelude::*,
};

fn main() -> opencv::Result<()> {
    let img_path = "截屏2026-07-26 13.40.25.png";
    let img = imgcodecs::imread(img_path, imgcodecs::IMREAD_COLOR)?;
    let roi = core::Rect::new(0, 0, 226, 136);
    let minimap = core::Mat::roi(&img, roi)?;
    
    let mut hsv = core::Mat::default();
    imgproc::cvt_color(&minimap, &mut hsv, imgproc::COLOR_BGR2HSV, 0, core::AlgorithmHint::ALGO_HINT_DEFAULT)?;
    
    // 尝试放宽 HSV 范围
    let lower_green = core::Scalar::new(30.0, 40.0, 40.0, 0.0);
    let upper_green = core::Scalar::new(90.0, 255.0, 255.0, 0.0);
    let mut mask_green = core::Mat::default();
    core::in_range(&hsv, &lower_green, &upper_green, &mut mask_green)?;
    
    imgcodecs::imwrite("mask_green.png", &mask_green, &core::Vector::new())?;
    
    // 我们也打印一下纯绿色点的数量，看看是不是因为太淡没提出来
    let non_zero = core::count_non_zero(&mask_green)?;
    println!("绿框像素数: {}", non_zero);
    
    let mut contours_green = core::Vector::<core::Vector<core::Point>>::new();
    imgproc::find_contours(&mask_green, &mut contours_green, imgproc::RETR_EXTERNAL, imgproc::CHAIN_APPROX_SIMPLE, core::Point::new(0, 0))?;
    
    let mut min_x = 9999;
    let mut min_y = 9999;
    let mut max_x = 0;
    let mut max_y = 0;
    
    for i in 0..contours_green.len() {
        let rect = opencv::geometry::bounding_rect(&contours_green.get(i)?)?;
        if rect.x < min_x { min_x = rect.x; }
        if rect.y < min_y { min_y = rect.y; }
        if rect.x + rect.width > max_x { max_x = rect.x + rect.width; }
        if rect.y + rect.height > max_y { max_y = rect.y + rect.height; }
    }
    
    if max_x > min_x {
        let global_rect = core::Rect::new(min_x, min_y, max_x - min_x, max_y - min_y);
        println!("所有绿色的总包围盒尺寸: {}x{}", global_rect.width, global_rect.height);
    }
    
    Ok(())
}
