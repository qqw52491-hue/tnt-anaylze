use opencv::{core, imgcodecs, prelude::*};
use std::env;
use tnt_comput::ui::UiRecognizer;

fn main() -> opencv::Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        println!("Usage: cli_recognize <image_path>");
        return Ok(());
    }
    
    let img_path = &args[1];
    let img = imgcodecs::imread(img_path, imgcodecs::IMREAD_COLOR)?;
    if img.empty() {
        println!("❌ 无法读取图像: {}", img_path);
        return Ok(());
    }

    let recognizer = UiRecognizer::new("src/templates")?;

    // Save intermediate images
    let (mask, gray) = recognizer.binarize_and_clean(&img)?;
    let artifact_dir = "/Users/wx/.gemini/antigravity-ide/brain/a3a38ad1-4ad9-4ad3-960c-735a7583dcfd";
    imgcodecs::imwrite(&format!("{}/mask.png", artifact_dir), &mask, &core::Vector::new())?;
    imgcodecs::imwrite(&format!("{}/gray_cleaned.png", artifact_dir), &gray, &core::Vector::new())?;

    let digit_mats = recognizer.extract_individual_digits(&mask, &gray, 1.55)?;
    for (i, dmat) in digit_mats.iter().enumerate() {
        let tmpl = recognizer.to_template_40(dmat)?;
        imgcodecs::imwrite(&format!("{}/digit_crop_{}.png", artifact_dir, i), &tmpl, &core::Vector::new())?;
    }

    match recognizer.recognize_angle_digit(&img) {
        Ok(Some(angle)) => {
            println!("============== {} 识别结果 ==============", img_path);
            println!("✅【识别出的数字】: {}", angle);
        },
        Ok(None) => {
            println!("❌ 未能在该切片中识别出任何数字。");
        },
        Err(e) => {
            println!("⚠️ 发生错误: {:?}", e);
        }
    }
    
    Ok(())
}
