use opencv::{core, imgcodecs, prelude::*};
use std::env;
use tnt_comput::ui::UiRecognizer;

// 风速 OCR 调试: test_wind <风速区截图>
// 打印识别出的带符号绝对风速(正=向右吹)。
fn main() -> opencv::Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        println!("Usage: test_wind <image_path>");
        return Ok(());
    }

    let img_path = &args[1];
    let img = imgcodecs::imread(img_path, imgcodecs::IMREAD_COLOR)?;
    if img.empty() {
        println!("❌ 无法读取图像: {}", img_path);
        return Ok(());
    }

    let recognizer = UiRecognizer::new("src/templates")?;

    // 落盘中间产物方便调 mask
    let (mask, gray, _combo) = recognizer.binarize_and_clean(&img)?;
    let _ = std::fs::create_dir_all("/tmp/tnt_dbg");
    imgcodecs::imwrite("/tmp/tnt_dbg/wind_mask.png", &mask, &core::Vector::new())?;
    imgcodecs::imwrite("/tmp/tnt_dbg/wind_gray.png", &gray, &core::Vector::new())?;

    match recognizer.recognize_wind(&img)? {
        Some(v) => println!("✅ 风速 = {:+.1} (正=向右吹, 负=向左吹)", v),
        None => println!("❌ 未识别"),
    }
    Ok(())
}
