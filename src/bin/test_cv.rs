use opencv::{
    core::{self, Point, Rect, Scalar, Size},
    highgui, imgcodecs, imgproc, prelude::*,
};
use std::env;

fn process_angle(img_path: &str) -> opencv::Result<()> {
    let img = imgcodecs::imread(img_path, imgcodecs::IMREAD_COLOR)?;
    if img.empty() {
        println!("Could not read {}", img_path);
        return Ok(());
    }

    let mut gray = core::Mat::default();
    imgproc::cvt_color(&img, &mut gray, imgproc::COLOR_BGR2GRAY, 0, core::AlgorithmHint::ALGO_HINT_DEFAULT)?;

    let mut binary = core::Mat::default();
    imgproc::threshold(&gray, &mut binary, 150.0, 255.0, imgproc::THRESH_BINARY)?;
    imgcodecs::imwrite("angle_binary.png", &binary, &core::Vector::new())?;

    let mut contours = core::Vector::<core::Vector<Point>>::new();
    imgproc::find_contours(&binary, &mut contours, imgproc::RETR_EXTERNAL, imgproc::CHAIN_APPROX_SIMPLE, Point::new(0, 0))?;

    let mut debug_img = img.clone();
    let mut count = 0;
    
    let mut rects = Vec::new();
    for i in 0..contours.len() {
        let contour = contours.get(i)?;
        let rect = opencv::geometry::bounding_rect(&contour)?;
        // 极致缩小的过滤条件，连小数点的 2x2 像素都不放过
        if rect.width > 1 && rect.height > 2 {
            rects.push(rect);
        }
    }
    rects.sort_by_key(|r| r.x);

    for rect in rects {
        imgproc::rectangle(&mut debug_img, rect, Scalar::new(0.0, 255.0, 0.0, 0.0), 1, imgproc::LINE_8, 0)?;
        let digit_roi = core::Mat::roi(&binary, rect)?;
        
        // 1. 放大 10 倍
        let mut resized = core::Mat::default();
        imgproc::resize(&digit_roi, &mut resized, Size::new(0, 0), 10.0, 10.0, imgproc::INTER_NEAREST)?;
        
        // 2. 反色 (变成白底黑字，这是 OCR 最喜欢的格式)
        let mut inverted = core::Mat::default();
        core::bitwise_not(&resized, &mut inverted, &core::no_array())?;
        
        // 3. 加一圈白色边框 (白底边距)
        let mut padded = core::Mat::default();
        core::copy_make_border(&inverted, &mut padded, 30, 30, 30, 30, core::BORDER_CONSTANT, Scalar::new(255.0, 255.0, 255.0, 0.0))?;
        
        let filename = format!("angle_digit_{}_proc.png", count);
        imgcodecs::imwrite(&filename, &padded, &core::Vector::new())?;
        
        // 用 Tesseract 识别单个字符
        if let Ok(output) = std::process::Command::new("tesseract")
            .args(&[&filename, "stdout", "--psm", "10", "-c", "tessedit_char_whitelist=0123456789."])
            .output() 
        {
            let text = String::from_utf8_lossy(&output.stdout).trim().to_string();
            print!("{}", text);
        }
        count += 1;
    }
    println!();
    imgcodecs::imwrite("angle_debug.png", &debug_img, &core::Vector::new())?;
    Ok(())
}

fn process_wind(img_path: &str) -> opencv::Result<()> {
    let img = imgcodecs::imread(img_path, imgcodecs::IMREAD_COLOR)?;
    if img.empty() {
        println!("Could not read {}", img_path);
        return Ok(());
    }

    let mut hsv = core::Mat::default();
    imgproc::cvt_color(&img, &mut hsv, imgproc::COLOR_BGR2HSV, 0, core::AlgorithmHint::ALGO_HINT_DEFAULT)?;

    let mut binary = core::Mat::default();
    let lower = Scalar::new(0.0, 0.0, 180.0, 0.0);
    let upper = Scalar::new(180.0, 60.0, 255.0, 0.0);
    core::in_range(&hsv, &lower, &upper, &mut binary)?;

    imgcodecs::imwrite("wind_binary.png", &binary, &core::Vector::new())?;

    let mut contours = core::Vector::<core::Vector<Point>>::new();
    imgproc::find_contours(&binary, &mut contours, imgproc::RETR_EXTERNAL, imgproc::CHAIN_APPROX_SIMPLE, Point::new(0, 0))?;

    let mut debug_img = img.clone();
    let mut count = 0;
    
    let mut rects = Vec::new();
    for i in 0..contours.len() {
        let contour = contours.get(i)?;
        let rect = opencv::geometry::bounding_rect(&contour)?;
        // 提取最小 1x1 的方块（能捕捉小数点）
        if rect.width >= 1 && rect.height >= 1 {
            rects.push(rect);
        }
    }
    rects.sort_by_key(|r| r.x);

    for rect in rects {
        imgproc::rectangle(&mut debug_img, rect, Scalar::new(0.0, 255.0, 0.0, 0.0), 1, imgproc::LINE_8, 0)?;
        let digit_roi = core::Mat::roi(&binary, rect)?;
        let filename = format!("wind_digit_{}.png", count);
        imgcodecs::imwrite(&filename, &digit_roi, &core::Vector::new())?;
        count += 1;
        
        // 用 Tesseract 识别单个字符
        if let Ok(output) = std::process::Command::new("tesseract")
            .args(&[&filename, "stdout", "--psm", "10", "-c", "tessedit_char_whitelist=0123456789."])
            .output() 
        {
            let text = String::from_utf8_lossy(&output.stdout).trim().to_string();
            print!("{}", text);
        }
    }
    println!();
    imgcodecs::imwrite("wind_debug.png", &debug_img, &core::Vector::new())?;
    Ok(())
}

fn main() {
    // 之前的老图片
    // let angle_img = "src/QQ_1785391461914.png";
    // let wind_img = "src/QQ_1785391469112.png";
    
    let new_img_1 = "src/QQ_1785394137344.png";
    let new_img_2 = "src/QQ_1785394145423.png";
    
    let _ = process_angle(new_img_1);
    let _ = process_wind(new_img_2);
}
