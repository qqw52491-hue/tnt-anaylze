use image::{imageops, GrayImage, Luma};

/// 1. 自动判定极性：保证白字黑底 (MNIST 格式)
pub fn ensure_white_on_black(img: &GrayImage) -> GrayImage {
    let (w, h) = img.dimensions();
    if w == 0 || h == 0 {
        return img.clone();
    }
    let (mut sum, mut n) = (0u64, 0u64);

    // 采样最外圈一圈像素
    for x in 0..w {
        sum += img.get_pixel(x, 0).0[0] as u64;
        sum += img.get_pixel(x, h - 1).0[0] as u64;
        n += 2;
    }
    for y in 0..h {
        sum += img.get_pixel(0, y).0[0] as u64;
        sum += img.get_pixel(w - 1, y).0[0] as u64;
        n += 2;
    }

    let border_mean = sum as f32 / n as f32;

    if border_mean > 127.0 {
        // 边缘是亮的 → 说明背景是白的 → 反相为黑底白字
        let mut out = img.clone();
        for p in out.pixels_mut() {
            p.0[0] = 255 - p.0[0];
        }
        out
    } else {
        img.clone()
    }
}

/// 2. 形态学膨胀 (3x3 最大值滤波)：加粗笔画并填补微断笔
pub fn dilate(img: &GrayImage, times: u32) -> GrayImage {
    let mut current = img.clone();
    for _ in 0..times {
        let (w, h) = current.dimensions();
        let mut out = GrayImage::new(w, h);
        for y in 0..h {
            for x in 0..w {
                let mut max_val = 0u8;
                for dy in -1i32..=1 {
                    for dx in -1i32..=1 {
                        let nx = x as i32 + dx;
                        let ny = y as i32 + dy;
                        if nx >= 0 && ny >= 0 && (nx as u32) < w && (ny as u32) < h {
                            max_val = max_val.max(current.get_pixel(nx as u32, ny as u32).0[0]);
                        }
                    }
                }
                out.put_pixel(x, y, Luma([max_val]));
            }
        }
        current = out;
    }
    current
}

/// 3. 核心预处理流水线：将任意单字切图转换为 MNIST 规范 28x28 质心对齐图像
pub fn to_mnist_28(src: &GrayImage, dilate_times: u32, blur_sigma: f32) -> GrayImage {
    // 1) 判极性 (白字黑底)
    let inverted = ensure_white_on_black(src);

    // 2) 寻找前景外接框，裁掉多余黑边
    let (w, h) = inverted.dimensions();
    let (mut x0, mut y0, mut x1, mut y1) = (w, h, 0u32, 0u32);
    for y in 0..h {
        for x in 0..w {
            if inverted.get_pixel(x, y).0[0] > 50 {
                x0 = x0.min(x);
                y0 = y0.min(y);
                x1 = x1.max(x);
                y1 = y1.max(y);
            }
        }
    }

    if x1 < x0 || y1 < y0 {
        return GrayImage::new(28, 28); // 全黑图
    }

    let cropped = imageops::crop_imm(&inverted, x0, y0, x1 - x0 + 1, y1 - y0 + 1).to_image();

    // 3) 笔画膨胀加粗 (解决笔画太细与断笔问题)
    let dilated = dilate(&cropped, dilate_times);

    // 4) 等比缩放，让长边 = 20
    let (cw, ch) = dilated.dimensions();
    let scale = 20.0 / cw.max(ch) as f32;
    let nw = ((cw as f32 * scale).round() as u32).max(1);
    let nh = ((ch as f32 * scale).round() as u32).max(1);
    let resized = imageops::resize(&dilated, nw, nh, imageops::FilterType::Triangle);

    // 5) 计算像素质心 Center of Mass
    let (mut sum, mut sx, mut sy) = (0f32, 0f32, 0f32);
    for (x, y, p) in resized.enumerate_pixels() {
        let v = p.0[0] as f32;
        sum += v;
        sx += v * x as f32;
        sy += v * y as f32;
    }

    let (cx, cy) = if sum > 0.0 {
        (sx / sum, sy / sum)
    } else {
        (nw as f32 / 2.0, nh as f32 / 2.0)
    };

    // 6) 按质心贴到 28x28 画布中心 (14.0, 14.0)
    let mut canvas = GrayImage::new(28, 28);
    let offset_x = (14.0 - cx).round() as i64;
    let offset_y = (14.0 - cy).round() as i64;

    imageops::overlay(&mut canvas, &resized, offset_x, offset_y);

    // 7) 柔化边缘 (模拟 MNIST 的抗锯齿渐变)
    if blur_sigma > 0.0 {
        imageops::blur(&canvas, blur_sigma)
    } else {
        canvas
    }
}

/// 4. 提取 1x28x28 MNIST 标准化浮点数组 ((x/255 - 0.1307) / 0.3081)
pub fn to_mnist_tensor_data(img28: &GrayImage) -> Vec<f32> {
    img28
        .pixels()
        .map(|p| ((p.0[0] as f32 / 255.0) - 0.1307) / 0.3081)
        .collect()
}
