use burn::tensor::{activation::softmax, Tensor};
use tnt_comput::{burn_model::Model, burn_state, mnist};

fn predict(model: &Model, img: &image::GrayImage, device: &burn::prelude::Device) -> (usize, f32) {
    let img28 = mnist::to_mnist_28(img, 1, 0.6);
    let data = mnist::to_mnist_tensor_data(&img28);
    let input = Tensor::<1>::from_floats(data.as_slice(), device).reshape([1, 28, 28]);

    let probs = softmax(model.forward(input), 1);
    let v: Vec<f32> = probs.into_data().to_vec().unwrap();

    let (digit, &conf) = v
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap();

    (digit, conf)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let (model, device) = burn_state::build_and_load_model();

    // 标注的真实真值表 (Ground Truth)
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
    ];

    println!("==================== src/pic 测试集批量准确率评估报告 ====================");
    println!("模型架构: Burn CNN (26万参数, MNIST 预训练)");

    let mut correct_count = 0;
    let mut total_count = 0;

    for (fname, expected_val) in &ground_truth {
        let path = format!("src/pic/{}", fname);
        if !std::path::Path::new(&path).exists() {
            continue;
        }

        total_count += 1;

        // 识别单字
        let mut predicted_digits = Vec::new();
        for i in 0..2 {
            let single_path = format!("src/pic_cleaned/{}_{}.png", fname.trim_end_matches(".png"), i);
            if let Ok(img) = image::open(&single_path) {
                let (d, _conf) = predict(&model, &img.to_luma8(), &device);
                predicted_digits.push(d.to_string());
            }
        }

        let pred_str = predicted_digits.join("");
        let pred_val = pred_str.parse::<i32>().unwrap_or(-1);

        let is_correct = pred_val == *expected_val;
        if is_correct {
            correct_count += 1;
            println!("✅ {:<24} 真值: {:<4} | 预测: {:<4}  [匹配成功]", fname, expected_val, pred_val);
        } else {
            println!("❌ {:<24} 真值: {:<4} | 预测: {:<4}  [识别不匹配]", fname, expected_val, pred_val);
        }
    }

    let accuracy = if total_count > 0 {
        (correct_count as f64 / total_count as f64) * 100.0
    } else {
        0.0
    };

    println!("\n==================== 评估总结 ====================");
    println!("测试图片数: {}", total_count);
    println!("正确识别数: {}", correct_count);
    println!("整体准确率: {:.2}%", accuracy);

    Ok(())
}
