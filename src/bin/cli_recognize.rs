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
    let (mask, gray, gray_combo) = recognizer.binarize_and_clean(&img)?;
    let artifact_dir = "/tmp/tnt_dbg";
    let _ = std::fs::create_dir_all(artifact_dir);
    imgcodecs::imwrite(&format!("{}/mask.png", artifact_dir), &mask, &core::Vector::new())?;
    imgcodecs::imwrite(&format!("{}/gray_cleaned.png", artifact_dir), &gray, &core::Vector::new())?;
    imgcodecs::imwrite(&format!("{}/gray_combo.png", artifact_dir), &gray_combo, &core::Vector::new())?;

    for (pass, allow) in [(1, false), (2, true)] {
        let digit_mats = recognizer.extract_individual_digits(&mask, &gray, &gray_combo, 1.55, allow)?;
        println!("第{}遍: 切出 {} 个数字组件:", pass, digit_mats.len());
        for (i, (dmat, absorbed)) in digit_mats.iter().enumerate() {
            let tmpl = recognizer.to_template_40(dmat)?;
            if pass == 2 {
                imgcodecs::imwrite(&format!("{}/digit_crop_{}.png", artifact_dir, i), &tmpl, &core::Vector::new())?;
            }
            let scores = recognizer.score_digits_debug(&tmpl)?;
            let top: Vec<String> = scores.iter().take(4)
                .map(|(d, adj, raw)| format!("{}={:.3}/{:.3}", d, adj, raw)).collect();
            println!("  位[{}]{} 前四名(修正分/原始分): {}", i, if *absorbed { "(拼)" } else { "" }, top.join("  "));
        }
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
