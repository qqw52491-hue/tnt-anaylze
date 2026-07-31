use std::fs;
use opencv::{core, imgcodecs};
use tnt_comput::ui::UiRecognizer;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ground_truth = vec![
        ("QQ_1785395550263.png", 46),
        ("QQ_1785395577212.png", 45),
        ("QQ_1785395607716.png", 49),
        ("QQ_1785395679500.png", 76),
        ("QQ_1785395719748.png", 61),
        ("QQ_1785395741836.png", 62),
        ("QQ_1785400893053.png", 59),
        ("QQ_1785401461142.png", 80),
        ("QQ_1785457855758.png", 45),
        ("QQ_1785457866731.png", 45),
        ("QQ_1785460158243.png", 49),
        ("QQ_1785460167434.png", 80),
        ("QQ_1785460784542.png", 35),
        ("QQ_1785460794325.png", 49),
        ("QQ_1785460804565.png", 36),
        ("QQ_1785461834029.png", 74),
        ("QQ_1785476343893.png", 79),
    ];

    println!("==================== UiRecognizer 综合准确率评估 ====================");
    
    // 初始化识别器（和 live_gui 使用的完全一样）
    let recognizer = UiRecognizer::new("src/templates")?;
    println!("成功加载 src/templates 中的物理模板库。");

    let mut correct = 0;
    let mut total = 0;

    for (fname, expected) in &ground_truth {
        let path = format!("src/pic/{}", fname);
        if !std::path::Path::new(&path).exists() {
            continue;
        }

        total += 1;
        // 加载原始未清洗的 ROI
        let raw_roi = imgcodecs::imread(&path, imgcodecs::IMREAD_COLOR)?;
        
        // 直接调用实机 API 识别角度
        let result = recognizer.recognize_angle_digit(&raw_roi)?;
        
        match result {
            Some(pred) if pred == *expected => {
                println!("✅ {} \t真值: {} \t| 预测: {} \t[匹配成功]", fname, expected, pred);
                correct += 1;
            },
            Some(pred) => {
                println!("❌ {} \t真值: {} \t| 预测: {} \t[识别错误]", fname, expected, pred);
            },
            None => {
                println!("❌ {} \t真值: {} \t| 预测: 无 (被 0.85 阈值过滤) \t[识别失败]", fname, expected);
            }
        }
    }

    println!("\n==================== 评估总结 ====================");
    println!("测试图片数: {}", total);
    println!("正确识别数: {}", correct);
    if total > 0 {
        let acc = (correct as f64 / total as f64) * 100.0;
        println!("整体准确率: {:.2}%", acc);
    }
    
    Ok(())
}
