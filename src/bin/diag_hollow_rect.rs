use opencv::{
    core, imgcodecs, imgproc,
    prelude::*,
};

fn main() -> opencv::Result<()> {
    let img_path = "截屏2026-07-26 13.40.25.png";
    println!("==== 测试图片: {} ====", img_path);
    let img = imgcodecs::imread(img_path, imgcodecs::IMREAD_COLOR)?;
    let roi = core::Rect::new(0, 0, 226, 136);
    let minimap = core::Mat::roi(&img, roi)?;
    
    let mut hsv = core::Mat::default();
    imgproc::cvt_color(&minimap, &mut hsv, imgproc::COLOR_BGR2HSV, 0, core::AlgorithmHint::ALGO_HINT_DEFAULT)?;
    
    let colors = vec![("Blue", false), ("Red", true)];
    for (color_name, is_red) in colors {
        println!("-- 搜索 {} --", color_name);
        let mut mask = core::Mat::default();
        if is_red {
            let mut mask1 = core::Mat::default();
            core::in_range(&hsv, &core::Scalar::new(0.0, 100.0, 100.0, 0.0), &core::Scalar::new(15.0, 255.0, 255.0, 0.0), &mut mask1)?;
            let mut mask2 = core::Mat::default();
            core::in_range(&hsv, &core::Scalar::new(165.0, 100.0, 100.0, 0.0), &core::Scalar::new(180.0, 255.0, 255.0, 0.0), &mut mask2)?;
            core::add(&mask1, &mask2, &mut mask, &core::no_array(), -1)?;
        } else {
            core::in_range(&hsv, &core::Scalar::new(90.0, 80.0, 80.0, 0.0), &core::Scalar::new(140.0, 255.0, 255.0, 0.0), &mut mask)?;
        }
        
        let mut contours = core::Vector::<core::Vector<core::Point>>::new();
        imgproc::find_contours(&mask, &mut contours, imgproc::RETR_EXTERNAL, imgproc::CHAIN_APPROX_SIMPLE, core::Point::new(0, 0))?;
        
        for i in 0..contours.len() {
            let contour = contours.get(i)?;
            let rect = opencv::geometry::bounding_rect(&contour)?;
            let area = rect.width * rect.height;
            let aspect = rect.width as f64 / rect.height as f64;
            
            if area >= 4 && area <= 100 && aspect > 0.5 && aspect < 2.0 {
                // 检查边框像素是否连续（是否是一个矩形框）
                let mut border_pixels = 0;
                let mut colored_border_pixels = 0;
                
                for y in rect.y..rect.y+rect.height {
                    for x in rect.x..rect.x+rect.width {
                        // 如果是边框像素（上、下、左、右）
                        if x == rect.x || x == rect.x + rect.width - 1 || y == rect.y || y == rect.y + rect.height - 1 {
                            border_pixels += 1;
                            let p = mask.at_2d::<u8>(y, x)?;
                            if *p > 0 {
                                colored_border_pixels += 1;
                            }
                        }
                    }
                }
                
                let solid_border_ratio = colored_border_pixels as f64 / border_pixels as f64;
                
                println!("候选框 坐标({}, {}), 尺寸: {}x{}, 边框像素: {}/{}, 边框完整度: {:.1}%", 
                    rect.x, rect.y, rect.width, rect.height, colored_border_pixels, border_pixels, solid_border_ratio * 100.0);
            }
        }
    }
    Ok(())
}
