use opencv::{
    core::{self, Point, Size},
    imgcodecs, imgproc, prelude::*,
};

fn main() -> opencv::Result<()> {
    let img = imgcodecs::imread("src/pic/QQ_1785400893053.png", imgcodecs::IMREAD_COLOR)?;

    // 缩放到标准高度 24 (按比例放大)
    let scale = 24.0 / img.rows() as f64;
    let target_w = (img.cols() as f64 * scale).round() as i32;
    let mut resized_img = core::Mat::default();
    imgproc::resize(&img, &mut resized_img, Size::new(target_w, 24), 0.0, 0.0, imgproc::INTER_CUBIC)?;

    println!("原始尺寸: {}x{} => 规范化尺寸: {}x{}", img.cols(), img.rows(), resized_img.cols(), resized_img.rows());

    let mut bgr = core::Vector::<core::Mat>::new();
    core::split(&resized_img, &mut bgr)?;
    let r = bgr.get(2)?;
    let mut mask = core::Mat::default();
    imgproc::threshold(&r, &mut mask, 100.0, 255.0, imgproc::THRESH_BINARY)?;

    let mut contours = core::Vector::<core::Vector<Point>>::new();
    imgproc::find_contours(&mask, &mut contours, imgproc::RETR_EXTERNAL, imgproc::CHAIN_APPROX_SIMPLE, Point::new(0, 0))?;

    let mut rects = Vec::new();
    for i in 0..contours.len() {
        let c = contours.get(i)?;
        let rect = opencv::geometry::bounding_rect(&c)?;
        if rect.width >= 2 && rect.height >= 5 {
            rects.push(rect);
        }
    }
    rects.sort_by_key(|r| r.x);

    println!("切割出 {} 个轮廓:", rects.len());
    for (idx, r) in rects.iter().enumerate() {
        let roi = core::Mat::roi(&mask, *r)?;
        println!("轮廓 {}: w={}, h={}", idx, r.width, r.height);
        for row in 0..roi.rows() {
            let mut line = String::new();
            for col in 0..roi.cols() {
                let p = *roi.at_2d::<u8>(row, col)?;
                line.push_str(if p > 0 { "██" } else { "░░" });
            }
            println!("  {}", line);
        }
    }

    Ok(())
}
