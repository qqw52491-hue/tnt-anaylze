use opencv::{
    core, imgcodecs, imgproc,
    prelude::*,
};

fn main() -> opencv::Result<()> {
    // 读取原始未偏色的大图
    let img = imgcodecs::imread("截屏2026-07-26 11.27.46.png", imgcodecs::IMREAD_COLOR)?;
    let mut img_gray = core::Mat::default();
    imgproc::cvt_color(&img, &mut img_gray, imgproc::COLOR_BGR2GRAY, 0, core::AlgorithmHint::ALGO_HINT_DEFAULT)?;

    // ==========================================
    // 1. 处理红点：读取用户截的图 my_red.png
    // ==========================================
    let user_red = imgcodecs::imread("my_red.png", imgcodecs::IMREAD_COLOR)?;
    if !user_red.empty() {
        let mut user_red_gray = core::Mat::default();
        imgproc::cvt_color(&user_red, &mut user_red_gray, imgproc::COLOR_BGR2GRAY, 0, core::AlgorithmHint::ALGO_HINT_DEFAULT)?;
        
        let mut res_red = core::Mat::default();
        // 用灰度匹配（免疫 Mac 截图带来的微小偏色）
        imgproc::match_template(&img_gray, &user_red_gray, &mut res_red, imgproc::TM_CCOEFF_NORMED, &core::no_array())?;
        
        let mut min_val = 0.0;
        let mut max_val = 0.0;
        let mut min_loc = core::Point::default();
        let mut max_loc = core::Point::default();
        core::min_max_loc(&res_red, Some(&mut min_val), Some(&mut max_val), Some(&mut min_loc), Some(&mut max_loc), &core::no_array())?;
        
        // max_loc 就是用户截图在原大图里的左上角坐标
        // 我们要以此为中心，在没有偏色的原大图里抠出最核心的 16x16
        let center_x = max_loc.x + user_red.cols() / 2;
        let center_y = max_loc.y + user_red.rows() / 2;
        
        let crop_rect = core::Rect::new(center_x - 8, center_y - 8, 16, 16);
        let pure_red_template = core::Mat::roi(&img, crop_rect)?.try_clone()?;
        imgcodecs::imwrite("template_red_dot.png", &pure_red_template, &core::Vector::new())?;
        println!("红点定位成功！原图坐标 ({}, {})。已提取完美红点模板！", center_x, center_y);
    } else {
        println!("找不到 my_red.png");
    }

    // ==========================================
    // 2. 处理蓝点：读取用户截的图 my_blue.png
    // ==========================================
    let user_blue = imgcodecs::imread("my_blue.png", imgcodecs::IMREAD_COLOR)?;
    if !user_blue.empty() {
        let mut user_blue_gray = core::Mat::default();
        imgproc::cvt_color(&user_blue, &mut user_blue_gray, imgproc::COLOR_BGR2GRAY, 0, core::AlgorithmHint::ALGO_HINT_DEFAULT)?;
        
        let mut res_blue = core::Mat::default();
        imgproc::match_template(&img_gray, &user_blue_gray, &mut res_blue, imgproc::TM_CCOEFF_NORMED, &core::no_array())?;
        
        let mut min_val = 0.0;
        let mut max_val = 0.0;
        let mut min_loc = core::Point::default();
        let mut max_loc = core::Point::default();
        core::min_max_loc(&res_blue, Some(&mut min_val), Some(&mut max_val), Some(&mut min_loc), Some(&mut max_loc), &core::no_array())?;
        
        let center_x = max_loc.x + user_blue.cols() / 2;
        let center_y = max_loc.y + user_blue.rows() / 2;
        
        let crop_rect = core::Rect::new(center_x - 8, center_y - 8, 16, 16);
        let pure_blue_template = core::Mat::roi(&img, crop_rect)?.try_clone()?;
        imgcodecs::imwrite("template_blue_dot.png", &pure_blue_template, &core::Vector::new())?;
        println!("蓝点定位成功！原图坐标 ({}, {})。已提取完美蓝点模板！", center_x, center_y);
    } else {
        println!("找不到 my_blue.png");
    }

    Ok(())
}
