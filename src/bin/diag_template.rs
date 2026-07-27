use opencv::{
    core, imgcodecs, imgproc,
    prelude::*,
};

fn find_dots_by_template(minimap: &core::Mat, template: &core::Mat, threshold: f32) -> opencv::Result<Vec<core::Point>> {
    let mut result = core::Mat::default();
    imgproc::match_template(minimap, template, &mut result, imgproc::TM_CCOEFF_NORMED, &core::Mat::default())?;
    
    let mut points = Vec::new();
    let tw = template.cols();
    let th = template.rows();
    
    loop {
        let mut min_val = 0.0;
        let mut max_val = 0.0;
        let mut min_loc = core::Point::default();
        let mut max_loc = core::Point::default();
        
        core::min_max_loc(&result, Some(&mut min_val), Some(&mut max_val), Some(&mut min_loc), Some(&mut max_loc), &core::Mat::default())?;
        
        if max_val >= threshold as f64 {
            // 找到一个匹配点 (中心坐标)
            let center = core::Point::new(max_loc.x + tw / 2, max_loc.y + th / 2);
            points.push(center);
            
            // 将这个区域涂黑，避免重复找到
            let rect = core::Rect::new(max_loc.x - tw/2, max_loc.y - th/2, tw*2, th*2);
            imgproc::rectangle(&mut result, rect, core::Scalar::all(0.0), -1, imgproc::LINE_8, 0)?;
        } else {
            break;
        }
    }
    
    Ok(points)
}
