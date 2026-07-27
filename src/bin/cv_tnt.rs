use opencv::{
    core,
    imgcodecs,
    imgproc,
    prelude::*,
};
use std::env;

struct EnemyTarget {
    pos: (u32, u32),
    game_distance: f64,
    power_30deg: f64,
}

// 提取物理计算公式
fn calc_power_30deg(distance: f64) -> f64 {
    (distance * 30.0).sqrt() * 1.5 + 15.0
}

fn main() -> opencv::Result<()> {
    println!(">>> cv_tnt 开始执行！");
    let args: Vec<String> = env::args().collect();
    let img_path = if args.len() > 1 {
        args[1].clone()
    } else {
        "截屏2026-07-26 11.27.46.png".to_string()
    };

    println!("正在用 OpenCV 识别图片: {}", img_path);

    // 1. 加载大图和模板
    let img = imgcodecs::imread(&img_path, imgcodecs::IMREAD_COLOR)?;
    let template_red = imgcodecs::imread("template_red_dot.png", imgcodecs::IMREAD_COLOR)?;
    let template_blue = imgcodecs::imread("template_blue_dot.png", imgcodecs::IMREAD_COLOR)?;

    if img.empty() || template_red.empty() || template_blue.empty() {
        eprintln!("无法加载图片或模板");
        return Ok(());
    }

    // 假设小地图区域 (226x136)
    let roi = core::Rect::new(0, 0, 226, 136);
    let minimap = core::Mat::roi(&img, roi)?;
    
    // 2. 匹配蓝点 (玩家)
    let mut result_blue = core::Mat::default();
    imgproc::match_template(&minimap, &template_blue, &mut result_blue, imgproc::TM_CCOEFF_NORMED, &core::no_array())?;
    
    let mut player_pos = (0, 0);
    let threshold = 0.85; // 匹配阈值
    
    // 寻找蓝点坐标 (这里简化为找最大匹配)
    let mut min_val = 0.0;
    let mut max_val = 0.0;
    let mut min_loc = core::Point::default();
    let mut max_loc = core::Point::default();
    core::min_max_loc(&result_blue, Some(&mut min_val), Some(&mut max_val), Some(&mut min_loc), Some(&mut max_loc), &core::no_array())?;
    
    if max_val >= threshold {
        // 加上模板一半的宽度使其居中
        player_pos = ((max_loc.x + template_blue.cols() / 2) as u32, (max_loc.y + template_blue.rows() / 2) as u32);
        println!("玩家(蓝点)位置: {:?}", player_pos);
    }

    // 3. 匹配红点 (敌人)
    let mut result_red = core::Mat::default();
    imgproc::match_template(&minimap, &template_red, &mut result_red, imgproc::TM_CCOEFF_NORMED, &core::no_array())?;
    
    // 我们可能需要遍历结果矩阵来找到所有超过阈值的红点
    // 由于 Rust cv 遍历矩阵比较麻烦，我们这里先用阈值过滤，或者寻找最大的前 N 个。
    // 为了演示，直接找最大值：
    core::min_max_loc(&result_red, Some(&mut min_val), Some(&mut max_val), Some(&mut min_loc), Some(&mut max_loc), &core::no_array())?;
    
    if max_val >= threshold {
        let enemy_pos = ((max_loc.x + template_red.cols() / 2) as u32, (max_loc.y + template_red.rows() / 2) as u32);
        println!("敌人(红点)位置: {:?}", enemy_pos);
        
        let dx_pixels = (player_pos.0 as i64 - enemy_pos.0 as i64).abs() as f64;
        // 假设通过找黑框或者历史记录得知 width = 170
        let camera_width = 170.0; 
        let game_distance = (dx_pixels / camera_width) * 12.0;
        println!("距离: {:.2} 距，推荐力度: {:.1}", game_distance, calc_power_30deg(game_distance));
    } else {
        println!("未找到敌人点！(相似度: {:.2})", max_val);
    }
    
    // (轮廓框代码先省略，验证红蓝点要紧)
    Ok(())
}
