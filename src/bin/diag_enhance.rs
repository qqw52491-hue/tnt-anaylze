use opencv::{
    core, imgcodecs, imgproc,
    prelude::*,
};

fn main() -> opencv::Result<()> {
    let img = imgcodecs::imread("截屏2026-07-26 14.33.50.png", imgcodecs::IMREAD_COLOR)?;
    let roi = core::Rect::new(0, 0, 226, 136);
    let minimap = core::Mat::roi(&img, roi)?.try_clone()?;

    // 1. 转灰度
    let mut gray = core::Mat::default();
    imgproc::cvt_color(&minimap, &mut gray, imgproc::COLOR_BGR2GRAY, 0, core::AlgorithmHint::ALGO_HINT_DEFAULT)?;

    // 2. 自适应直方图均衡化 (CLAHE) 极大增强局部对比度
    let mut clahe = imgproc::create_clahe(4.0, core::Size::new(8, 8))?;
    let mut enhanced_gray = core::Mat::default();
    clahe.apply(&gray, &mut enhanced_gray)?;

    // 3. 计算 Sobel 梯度或者 Canny 边缘
    let mut edges = core::Mat::default();
    imgproc::canny(&enhanced_gray, &mut edges, 20.0, 60.0, 3, false)?;

    // 放大查看结果
    let mut big = core::Mat::default();
    imgproc::resize(&edges, &mut big, core::Size::new(226*4, 136*4), 0.0, 0.0, imgproc::INTER_NEAREST)?;
    imgcodecs::imwrite("enhanced_edges.png", &big, &core::Vector::new())?;

    let mut big_clahe = core::Mat::default();
    imgproc::resize(&enhanced_gray, &mut big_clahe, core::Size::new(226*4, 136*4), 0.0, 0.0, imgproc::INTER_NEAREST)?;
    imgcodecs::imwrite("enhanced_gray.png", &big_clahe, &core::Vector::new())?;

    println!("生成增强对比度图像！");
    Ok(())
}
