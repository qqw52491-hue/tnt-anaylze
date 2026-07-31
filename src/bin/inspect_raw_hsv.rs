use opencv::{
    core::{self, Vector},
    imgcodecs, imgproc,
    prelude::*,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let raw = imgcodecs::imread("src/pic/QQ_1785395679500.png", imgcodecs::IMREAD_COLOR)?;
    println!("图像尺寸: {}x{}", raw.cols(), raw.rows());

    let mut hsv = core::Mat::default();
    imgproc::cvt_color(&raw, &mut hsv, imgproc::COLOR_BGR2HSV, 0, core::AlgorithmHint::ALGO_HINT_DEFAULT)?;
    let mut ch = Vector::<core::Mat>::new();
    core::split(&hsv, &mut ch)?;
    let s = ch.get(1)?;
    let v = ch.get(2)?;

    println!("\n=== 原图像素 (Value 亮度通道 0-255) ===");
    for y in 0..v.rows() {
        for x in 0..v.cols() {
            let val = *v.at_2d::<u8>(y, x)?;
            let sat = *s.at_2d::<u8>(y, x)?;
            if sat > 100 {
                print!("  "); // 被 S > 100 抹掉的彩色部分
            } else if val > 180 {
                print!("██");
            } else if val > 100 {
                print!("▒▒");
            } else if val > 40 {
                print!("░░");
            } else {
                print!("  ");
            }
        }
        println!();
    }

    Ok(())
}
