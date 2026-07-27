use opencv::{
    core, imgcodecs, imgproc,
    prelude::*, types,
};

fn main() -> opencv::Result<()> {
    let img_path = "截屏2026-07-26 14.33.50.png";
    let img = imgcodecs::imread(img_path, imgcodecs::IMREAD_COLOR)?;
    let roi = core::Rect::new(0, 0, 226, 136);
    let minimap_ref = core::Mat::roi(&img, roi)?;
    let minimap = minimap_ref.try_clone()?;
    
    // 1. 转灰度，寻找所有“带黑边的圆点”
    let mut gray = core::Mat::default();
    imgproc::cvt_color(&minimap, &mut gray, imgproc::COLOR_BGR2GRAY, 0, core::AlgorithmHint::ALGO_HINT_DEFAULT)?;
    
    // 二值化：提取黑边（假设黑边像素比较暗，灰度 < 60）
    let mut black_mask = core::Mat::default();
    imgproc::threshold(&gray, &mut black_mask, 60.0, 255.0, imgproc::THRESH_BINARY_INV)?;
    
    // 形态学闭运算：把残缺的黑边连成完整的圆环
    let mut closed = core::Mat::default();
    let kernel = imgproc::get_structuring_element(imgproc::MORPH_ELLIPSE, core::Size::new(3, 3), core::Point::new(-1, -1))?;
    imgproc::morphology_ex(&black_mask, &mut closed, imgproc::MORPH_CLOSE, &kernel, core::Point::new(-1, -1), 1, core::BORDER_CONSTANT, core::Scalar::default())?;
    
    // 寻找轮廓
    let mut contours = core::Vector::<core::Mat>::new();
    imgproc::find_contours(&closed, &mut contours, imgproc::RETR_EXTERNAL, imgproc::CHAIN_APPROX_SIMPLE, core::Point::new(0, 0))?;
    
    let mut debug_map = minimap.try_clone()?;
    
    println!(">>> 正在基于形状提取小地图上的所有目标点...");
    for i in 0..contours.len() {
        let contour = contours.get(i)?;
        let area = opencv::geometry::contour_area(&contour, false)?;
        
        // 游戏里的点很小，面积大概在 10 ~ 80 像素之间
        // if area < 10.0 || area > 80.0 { continue; }
        
        // 获取外接圆
        let mut center = core::Point2f::default();
        let mut radius = 0.0;
        opencv::geometry::min_enclosing_circle(&contour, &mut center, &mut radius)?;
        
        // 检查圆度（外接圆面积与实际面积的比例不应该差太多）
        let circle_area = std::f32::consts::PI * radius * radius;
        // if circle_area / (area as f32) > 2.5 { continue; } // 太不圆的过滤掉
        
        let cx = center.x as i32;
        let cy = center.y as i32;
        
        // 读取中心点的颜色
        let p = minimap.at_2d::<core::Vec3b>(cy, cx)?;
        let b = p[0]; let g = p[1]; let r = p[2];
        
        // 打印出找到的点的颜色
        println!("发现轮廓! 坐标: ({}, {}), 面积: {:.1}, 半径: {:.1}, 圆度: {:.2}, 颜色: B={}, G={}, R={}", cx, cy, area, radius, circle_area / (area as f32), b, g, r);
        
        imgproc::circle(&mut debug_map, core::Point::new(cx, cy), radius as i32 + 2, core::Scalar::new(0.0, 255.0, 0.0, 0.0), 1, imgproc::LINE_8, 0)?;
    }
    
    // 放大查看
    let mut big = core::Mat::default();
    imgproc::resize(&debug_map, &mut big, core::Size::new(226*4, 136*4), 0.0, 0.0, imgproc::INTER_NEAREST)?;
    imgcodecs::imwrite("dots_debug.png", &big, &core::Vector::new())?;
    
    Ok(())
}
