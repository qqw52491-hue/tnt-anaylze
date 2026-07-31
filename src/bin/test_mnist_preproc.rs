use std::fs;
use tnt_comput::mnist;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let input_dir = "src/pic_cleaned";
    let output_dir = "src/pic_mnist_preproc_debug";
    fs::create_dir_all(output_dir)?;

    let entries = fs::read_dir(input_dir)?;
    let mut files: Vec<_> = entries.filter_map(Result::ok).collect();
    files.sort_by_key(|a| a.path());

    println!("==================== MNIST 完整预处理流水线测试 ====================");
    println!("测试流程: 判极性(黑底白字) -> 裁黑边 -> 笔画膨胀 -> 20x20等比缩放 -> 质心贴合28x28 -> 边缘柔化 -> 归一化 Tensor");

    let configs = [
        ("raw", 0, 0.0),
        ("dilate1_noblur", 1, 0.0),
        ("dilate1_blur06", 1, 0.6),
        ("dilate2_blur06", 2, 0.6),
    ];

    for entry in files {
        let path = entry.path();
        if path.extension().unwrap_or_default() != "png" {
            continue;
        }

        let fname_stem = path.file_stem().unwrap().to_str().unwrap();
        let img = image::open(&path)?.to_luma8();

        for &(cfg_name, dilate_times, blur_sigma) in &configs {
            let img28 = mnist::to_mnist_28(&img, dilate_times, blur_sigma);
            let out_fname = format!("{}/{}_{}.png", output_dir, fname_stem, cfg_name);
            img28.save(&out_fname)?;

            // 提取归一化 Float 数组
            let tensor_data = mnist::to_mnist_tensor_data(&img28);
            assert_eq!(tensor_data.len(), 28 * 28);
        }

        println!("成功生成全预处理组合图像: {}*", fname_stem);
    }

    println!("\n所有预处理测试已完成！输出对比图片已保存至: {}", output_dir);
    Ok(())
}
