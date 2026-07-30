use opencv::{
    core::{self, Point},
    imgcodecs, imgproc, prelude::*,
};

fn to_binary_r_channel(img: &core::Mat, thresh: f64) -> opencv::Result<core::Mat> {
    let mut bgr = core::Vector::<core::Mat>::new();
    core::split(img, &mut bgr)?;
    let r = bgr.get(2)?;
    let mut binary = core::Mat::default();
    imgproc::threshold(&r, &mut binary, thresh, 255.0, imgproc::THRESH_BINARY)?;
    Ok(binary)
}

fn main() -> opencv::Result<()> {
    let img = imgcodecs::imread("src/pic/QQ_1785400893053.png", imgcodecs::IMREAD_COLOR)?;
    println!("图片尺寸: {}x{}", img.cols(), img.rows());

    for thresh in [60.0, 80.0, 100.0, 120.0] {
        let mask = to_binary_r_channel(&img, thresh)?;
        let mut contours = core::Vector::<core::Vector<Point>>::new();
        imgproc::find_contours(&mask, &mut contours, imgproc::RETR_EXTERNAL,
            imgproc::CHAIN_APPROX_SIMPLE, Point::new(0, 0))?;

        let mut rects = Vec::new();
        for i in 0..contours.len() {
            let c = contours.get(i)?;
            let rect = opencv::geometry::bounding_rect(&c)?;
            if rect.width >= 2 && rect.height >= 5 {
                rects.push(rect);
            }
        }
        rects.sort_by_key(|r| r.x);

        println!("\n阈值 {}: 切出 {} 个轮廓", thresh, rects.len());
        for (idx, rect) in rects.iter().enumerate() {
            println!("  轮廓 {}: x={}, y={}, w={}, h={}", idx, rect.x, rect.y, rect.width, rect.height);
            let roi = core::Mat::roi(&mask, *rect)?;
            println!("  ASCII 结构:");
            for row in 0..roi.rows() {
                let mut line = String::new();
                for col in 0..roi.cols() {
                    let p = *roi.at_2d::<u8>(row, col)?;
                    line.push_str(if p > 0 { "██" } else { "░░" });
                }
                println!("    {}", line);
            }
        }
    }

    Ok(())
}
