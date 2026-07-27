use opencv::{
    core, imgcodecs, imgproc,
    prelude::*,
};

fn main() -> opencv::Result<()> {
    let img_path = "截屏2026-07-26 13.40.25.png";
    let img = imgcodecs::imread(img_path, imgcodecs::IMREAD_COLOR)?;
    let roi = core::Rect::new(0, 0, 226, 136);
    let minimap = core::Mat::roi(&img, roi)?;
    
    // 用 Canny 边缘检测，找出所有的直线/矩形边缘
    let mut gray = core::Mat::default();
    imgproc::cvt_color(&minimap, &mut gray, imgproc::COLOR_BGR2GRAY, 0, core::AlgorithmHint::ALGO_HINT_DEFAULT)?;
    
    let mut edges = core::Mat::default();
    imgproc::canny(&gray, &mut edges, 30.0, 90.0, 3, false)?;
    imgcodecs::imwrite("edges_minimap.png", &edges, &core::Vector::new())?;
    
    // 找所有轮廓
    let mut contours = core::Vector::<core::Vector<core::Point>>::new();
    imgproc::find_contours(&edges, &mut contours, imgproc::RETR_LIST, imgproc::CHAIN_APPROX_SIMPLE, core::Point::new(0, 0))?;
    
    println!("共找到 {} 个轮廓", contours.len());
    
    // 用多边形逼近，看哪些轮廓接近4个顶点（矩形）
    for i in 0..contours.len() {
        let contour = contours.get(i)?;
        let rect = opencv::geometry::bounding_rect(&contour)?;
        let area = rect.width * rect.height;
        
        // 只看中等大小的轮廓（太小是噪点，太大是整个小地图边缘）
        if area < 100 || area > 20000 { continue; }
        
        // 多边形近似
        let mut approx = core::Vector::<core::Point>::new();
        let peri = opencv::geometry::arc_length(&contour, true)?;
        opencv::geometry::approx_poly_dp(&contour, &mut approx, 0.02 * peri, true)?;
        
        let vertices = approx.len();
        if vertices == 4 {
            let aspect = rect.width as f64 / rect.height as f64;
            println!("找到四边形! 位置({}, {}), 尺寸: {}x{}, 面积: {}, 长宽比: {:.2}", 
                rect.x, rect.y, rect.width, rect.height, area, aspect);
        }
    }
    
    // 同时，让我们看一下小地图的所有独特颜色
    // 用HSV分析，扫描所有可能的"半透明叠加层"
    let mut hsv = core::Mat::default();
    imgproc::cvt_color(&minimap, &mut hsv, imgproc::COLOR_BGR2HSV, 0, core::AlgorithmHint::ALGO_HINT_DEFAULT)?;
    
    // 保存放大版本方便查看
    let mut big_minimap = core::Mat::default();
    imgproc::resize(&minimap, &mut big_minimap, core::Size::new(226*4, 136*4), 0.0, 0.0, imgproc::INTER_NEAREST)?;
    imgcodecs::imwrite("minimap_4x.png", &big_minimap, &core::Vector::new())?;
    println!("已保存 4x 放大小地图: minimap_4x.png");
    
    Ok(())
}
