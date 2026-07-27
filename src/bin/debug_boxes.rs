use opencv::{core, imgcodecs, imgproc, geometry::bounding_rect};

fn main() -> opencv::Result<()> {
    let img = imgcodecs::imread("截屏2026-07-26 11.27.46.png", imgcodecs::IMREAD_COLOR)?;
    
    // 找绿色框 (视野框)
    let mut hsv = core::Mat::default();
    imgproc::cvt_color(&img, &mut hsv, imgproc::COLOR_BGR2HSV, 0, core::AlgorithmHint::ALGO_HINT_DEFAULT)?;
    
    // 绿色HSV范围
    let lower_green = core::Scalar::new(35.0, 50.0, 50.0, 0.0);
    let upper_green = core::Scalar::new(85.0, 255.0, 255.0, 0.0);
    let mut mask_green = core::Mat::default();
    core::in_range(&hsv, &lower_green, &upper_green, &mut mask_green)?;
    
    let mut contours_green = core::Vector::<core::Vector<core::Point>>::new();
    imgproc::find_contours(&mask_green, &mut contours_green, imgproc::RETR_EXTERNAL, imgproc::CHAIN_APPROX_SIMPLE, core::Point::new(0, 0))?;
    
    for i in 0..contours_green.len() {
        let rect = bounding_rect(&contours_green.get(i)?)?;
        if rect.width > 50 {
            println!("Green box: {:?}", rect);
        }
    }
    
    // 找黑色外框
    let lower_black = core::Scalar::new(0.0, 0.0, 0.0, 0.0);
    let upper_black = core::Scalar::new(180.0, 255.0, 50.0, 0.0);
    let mut mask_black = core::Mat::default();
    core::in_range(&hsv, &lower_black, &upper_black, &mut mask_black)?;
    
    let mut contours_black = core::Vector::<core::Vector<core::Point>>::new();
    imgproc::find_contours(&mask_black, &mut contours_black, imgproc::RETR_EXTERNAL, imgproc::CHAIN_APPROX_SIMPLE, core::Point::new(0, 0))?;
    for i in 0..contours_black.len() {
        let rect = bounding_rect(&contours_black.get(i)?)?;
        if rect.width > 150 && rect.height > 100 {
            println!("Black box: {:?}", rect);
        }
    }

    Ok(())
}
