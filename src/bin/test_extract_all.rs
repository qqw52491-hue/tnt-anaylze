use opencv::{
    core::{self, Point, Scalar, Size},
    imgcodecs, imgproc, prelude::*,
};

use std::fs;

fn process_angle(img_path: &str, global_count: &mut i32) -> opencv::Result<()> {
    let img = imgcodecs::imread(img_path, imgcodecs::IMREAD_COLOR)?;
    if img.empty() {
        println!("Could not read {}", img_path);
        return Ok(());
    }

    let mut gray = core::Mat::default();
    imgproc::cvt_color(&img, &mut gray, imgproc::COLOR_BGR2GRAY, 0, core::AlgorithmHint::ALGO_HINT_DEFAULT)?;

    let mut binary = core::Mat::default();
    // 再次降低阈值，因为 4 的左边那一撇可能非常暗（半透明）
    imgproc::threshold(&gray, &mut binary, 80.0, 255.0, imgproc::THRESH_BINARY)?;

    // 膨胀操作（闭运算）：关键修改！只做“垂直方向”的粘合，不做水平方向！
    // 这样上下断裂的 4 会被接上，但左右相邻的 4 和 9 不会被蓝线连在一起！
    let kernel = imgproc::get_structuring_element(imgproc::MORPH_RECT, Size::new(1, 3), Point::new(-1, -1))?;
    let mut closed = core::Mat::default();
    imgproc::morphology_ex(&binary, &mut closed, imgproc::MORPH_CLOSE, &kernel, Point::new(-1, -1), 1, core::BORDER_CONSTANT, imgproc::morphology_default_border_value()?)?;

    let mut contours = core::Vector::<core::Vector<Point>>::new();
    imgproc::find_contours(&closed, &mut contours, imgproc::RETR_EXTERNAL, imgproc::CHAIN_APPROX_SIMPLE, Point::new(0, 0))?;

    let mut rects = Vec::new();
    for i in 0..contours.len() {
        let contour = contours.get(i)?;
        let rect = opencv::geometry::bounding_rect(&contour)?;
        if rect.width > 1 && rect.height > 2 {
            rects.push(rect);
        }
    }
    rects.sort_by_key(|r| r.x);

    let mut local_count = 0;
    for rect in rects {
        let digit_roi = core::Mat::roi(&binary, rect)?;
        let filename = format!("src/temp_crops/crop_{}.png", *global_count);
        imgcodecs::imwrite(&filename, &digit_roi, &core::Vector::new())?;
        *global_count += 1;
        local_count += 1;
    }
    println!("Processed {}: found {} digits", img_path, local_count);
    Ok(())
}

fn main() {
    let mut global_count = 0;
    if let Ok(entries) = fs::read_dir("src/pic") {
        for entry in entries {
            if let Ok(entry) = entry {
                let path = entry.path();
                if let Some(ext) = path.extension() {
                    if ext == "png" || ext == "jpg" {
                        let _ = process_angle(path.to_str().unwrap(), &mut global_count);
                    }
                }
            }
        }
    }
    println!("Total digits extracted: {}", global_count);
}
