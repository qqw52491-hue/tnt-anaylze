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
    
    // 从放大图来看，那个框是品红色/粉红色的！
    // 品红色 HSV: H = 140~180 (偏紫红), S > 50, V > 50
    let ranges = vec![
        ("品红/粉红 H140-180 S50+ V50+", 140.0, 180.0, 50.0, 50.0),
        ("紫色 H120-160 S30+ V30+", 120.0, 160.0, 30.0, 30.0),
        ("粉红宽松 H150-180 S30+ V100+", 150.0, 180.0, 30.0, 100.0),
        ("全紫红 H130-180 S20+ V20+", 130.0, 180.0, 20.0, 20.0),
    ];
    
    for (name, h_low, h_high, s_low, v_low) in &ranges {
        let mut mask = core::Mat::default();
        core::in_range(&hsv, &core::Scalar::new(*h_low, *s_low, *v_low, 0.0), &core::Scalar::new(*h_high, 255.0, 255.0, 0.0), &mut mask)?;
        let non_zero = core::count_non_zero(&mask)?;
        println!("{}: {} 个像素", name, non_zero);
        
        if non_zero > 10 {
            // 找轮廓
            let mut contours = core::Vector::<core::Vector<core::Point>>::new();
            imgproc::find_contours(&mask, &mut contours, imgproc::RETR_EXTERNAL, imgproc::CHAIN_APPROX_SIMPLE, core::Point::new(0, 0))?;
            
            for i in 0..contours.len() {
                let contour = contours.get(i)?;
                let rect = opencv::geometry::bounding_rect(&contour)?;
                let area = rect.width * rect.height;
                if area > 100 {
                    println!("  -> 找到大色块! 位置({}, {}), 尺寸: {}x{}", rect.x, rect.y, rect.width, rect.height);
                }
            }
        }
    }
    
    // 直接检查我们之前在 Canny 里找到的那个矩形 (65, 36, 50x32) 的像素
    println!("\n=== 检查 Canny 找到的矩形区域 (65,36) 50x32 的边缘像素颜色 ===");
    // 上边缘
    for x in 65..115 {
        let p = minimap.at_2d::<core::Vec3b>(36, x)?;
        let h = hsv.at_2d::<core::Vec3b>(36, x)?;
        if h[1] > 30 { // 只看有一定饱和度的
            println!("上边 ({}, 36): BGR=({},{},{}), HSV=({},{},{})", x, p[0], p[1], p[2], h[0], h[1], h[2]);
        }
    }
    // 左边缘
    for y in 36..68 {
        let p = minimap.at_2d::<core::Vec3b>(y, 65)?;
        let h = hsv.at_2d::<core::Vec3b>(y, 65)?;
        if h[1] > 30 {
            println!("左边 (65, {}): BGR=({},{},{}), HSV=({},{},{})", y, p[0], p[1], p[2], h[0], h[1], h[2]);
        }
    }
    
    Ok(())
}
