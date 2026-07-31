use burn::tensor::{activation::softmax, Tensor};
use std::fs;
use tnt_comput::{burn_model::Model, burn_state, mnist};

fn predict(model: &Model, img: &image::GrayImage, device: &burn::prelude::Device) -> (usize, f32) {
    // 预处理流水线：判极性 -> 裁边 -> 膨胀加粗 -> 20x20等比 -> 质心28x28 -> 0.6边缘柔化
    let img28 = mnist::to_mnist_28(img, 1, 0.6);
    let data = mnist::to_mnist_tensor_data(&img28);

    // 输入维度 [1, 28, 28] (3维 Tensor, channel 维度由模型 forward 内部处理)
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
    println!("==================== Burn 官方 MNIST 深度学习模型推理测试 ====================");
    println!("加载嵌入式权重文件: model.bpk (包含26万参数 Conv+BatchNorm+Gelu 神经网络)");

    let (model, device) = burn_state::build_and_load_model();
    println!("模型加载成功！设备: {:?}", device);

    let input_dir = "src/pic_cleaned";
    let entries = fs::read_dir(input_dir)?;
    let mut files: Vec<_> = entries.filter_map(Result::ok).collect();
    files.sort_by_key(|a| a.path());

    println!("\n对 src/pic_cleaned 下的切图进行实时 Burn CNN 推理预测:\n");

    for entry in files {
        let path = entry.path();
        if path.extension().unwrap_or_default() != "png" {
            continue;
        }

        let fname = path.file_name().unwrap().to_str().unwrap();
        let img = image::open(&path)?.to_luma8();

        let (digit, conf) = predict(&model, &img, &device);

        println!("📷 {:<32} => 🤖 预测数字: {}  (置信度: {:.2}%)", fname, digit, conf * 100.0);
    }

    Ok(())
}
