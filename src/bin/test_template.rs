use opencv::{
    core::{self, Point, Scalar},
    imgcodecs, imgproc, prelude::*,
};

fn main() -> opencv::Result<()> {
    // 1. 读取原图和二值化
    let img_path = "src/QQ_1785394137344.png"; // 角度 80
    let img = imgcodecs::imread(img_path, imgcodecs::IMREAD_COLOR)?;
    
    let mut gray = core::Mat::default();
    imgproc::cvt_color(&img, &mut gray, imgproc::COLOR_BGR2GRAY, 0, core::AlgorithmHint::ALGO_HINT_DEFAULT)?;
    let mut binary = core::Mat::default();
    imgproc::threshold(&gray, &mut binary, 150.0, 255.0, imgproc::THRESH_BINARY)?;

    // 我们先把之前的切割结果读取进来，假装它们是我们提前保存在字模库里的 "8" 和 "0"
    let template_8 = imgcodecs::imread("angle_digit_0.png", imgcodecs::IMREAD_GRAYSCALE)?;
    let template_0 = imgcodecs::imread("angle_digit_1.png", imgcodecs::IMREAD_GRAYSCALE)?;

    // 2. 在全屏二值化图像上，去寻找 "8" 这个字模
    let mut result_8 = core::Mat::default();
    imgproc::match_template(&binary, &template_8, &mut result_8, imgproc::TM_CCOEFF_NORMED, &core::no_array())?;
    
    let mut min_val = 0.0;
    let mut max_val_8 = 0.0;
    let mut min_loc = Point::default();
    let mut max_loc_8 = Point::default();
    core::min_max_loc(&result_8, Some(&mut min_val), Some(&mut max_val_8), Some(&mut min_loc), Some(&mut max_loc_8), &core::no_array())?;

    println!("--- Template Matching Results ---");
    println!("搜索数字 '8': 匹配度 {:.2}% (位置: x={}, y={})", max_val_8 * 100.0, max_loc_8.x, max_loc_8.y);

    // 3. 在全屏二值化图像上，去寻找 "0" 这个字模
    let mut result_0 = core::Mat::default();
    imgproc::match_template(&binary, &template_0, &mut result_0, imgproc::TM_CCOEFF_NORMED, &core::no_array())?;
    
    let mut max_val_0 = 0.0;
    let mut max_loc_0 = Point::default();
    core::min_max_loc(&result_0, None, Some(&mut max_val_0), None, Some(&mut max_loc_0), &core::no_array())?;

    println!("搜索数字 '0': 匹配度 {:.2}% (位置: x={}, y={})", max_val_0 * 100.0, max_loc_0.x, max_loc_0.y);

    // 根据坐标排序输出结果
    let mut found = vec![
        (max_loc_8.x, "8", max_val_8),
        (max_loc_0.x, "0", max_val_0)
    ];
    found.sort_by_key(|f| f.0);
    
    print!("最终算法识别出的字符串: ");
    for (_, digit, conf) in found {
        if conf > 0.95 { // 置信度大于95%
            print!("{}", digit);
        }
    }
    println!("\n---------------------------------");
    
    Ok(())
}
