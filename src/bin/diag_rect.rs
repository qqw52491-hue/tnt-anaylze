use opencv::{
    core, imgcodecs, imgproc,
    prelude::*,
};

fn main() -> opencv::Result<()> {
    // 载入新旧两张图测试
    let imgs = vec!["截屏2026-07-26 11.27.46.png", "截屏2026-07-26 13.40.25.png"];
    
    for img_path in imgs {
        println!("==== 测试图片: {} ====", img_path);
        let img = imgcodecs::imread(img_path, imgcodecs::IMREAD_COLOR)?;
        if img.empty() {
            println!("图片加载失败");
            continue;
        }
        
        let roi = core::Rect::new(0, 0, 226, 136);
        let minimap = core::Mat::roi(&img, roi)?;
        
        let mut hsv = core::Mat::default();
        imgproc::cvt_color(&minimap, &mut hsv, imgproc::COLOR_BGR2HSV, 0, core::AlgorithmHint::ALGO_HINT_DEFAULT)?;
        
        // 分别测试蓝色和红色
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
                
                if area >= 2 && area <= 200 {
                    // 计算 mask 中在这个 rect 内的纯色像素真实数量
                    let rect_mask = core::Mat::roi(&mask, rect)?;
                    let non_zero = core::count_non_zero(&rect_mask)?;
                    
                    println!("发现色块: 坐标({}, {}), 尺寸: {}x{}, Bbox面积: {}, 真实高亮像素数: {}", 
                        rect.x, rect.y, rect.width, rect.height, area, non_zero);
                }
            }
        }
    }
    
    Ok(())
}
