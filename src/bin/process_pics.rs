use opencv::{
    core::{self, Point, Rect, Scalar, Size},
    imgcodecs, imgproc,
    prelude::*,
};
use std::fs;

fn binarize_and_clean(roi: &core::Mat) -> opencv::Result<core::Mat> {
    const UPSCALE: i32 = 3;
    
    let gray = if roi.channels() == 3 {
        let mut hsv = core::Mat::default();
        imgproc::cvt_color(
            roi,
            &mut hsv,
            imgproc::COLOR_BGR2HSV,
            0,
            core::AlgorithmHint::ALGO_HINT_DEFAULT,
        )?;
        let mut ch = core::Vector::<core::Mat>::new();
        core::split(&hsv, &mut ch)?;
        let s = ch.get(1)?;
        let v = ch.get(2)?;
        
        // 1. 饱和度 > 100 且 亮度 < 180 判定为纯彩色干扰线/背景，防止将与干扰线重叠的白字切断
        let mut s_mask = core::Mat::default();
        imgproc::threshold(&s, &mut s_mask, 100.0, 255.0, imgproc::THRESH_BINARY)?;
        
        let mut v_bright = core::Mat::default();
        imgproc::threshold(&v, &mut v_bright, 180.0, 255.0, imgproc::THRESH_BINARY_INV)?;

        let mut erase_mask = core::Mat::default();
        core::bitwise_and(&s_mask, &v_bright, &mut erase_mask, &core::no_array())?;

        let mut gray = v.clone();
        gray.set_to(&core::Scalar::all(0.0), &erase_mask)?;
        gray
    } else {
        roi.clone()
    };

    // 2. 放大3倍提升像素精度
    let mut up = core::Mat::default();
    imgproc::resize(
        &gray,
        &mut up,
        Size::new(gray.cols() * UPSCALE, gray.rows() * UPSCALE),
        0.0,
        0.0,
        imgproc::INTER_CUBIC,
    )?;

    // 3. Top-Hat 提取前景亮字
    let k = imgproc::get_structuring_element(
        imgproc::MORPH_ELLIPSE,
        Size::new(15, 15),
        Point::new(-1, -1),
    )?;
    let mut tophat = core::Mat::default();
    imgproc::morphology_ex(
        &up,
        &mut tophat,
        imgproc::MORPH_TOPHAT,
        &k,
        Point::new(-1, -1),
        1,
        core::BORDER_REPLICATE,
        imgproc::morphology_default_border_value()?,
    )?;

    // 4. Otsu 自适应二值化 (设下限 30)
    let mut mask = core::Mat::default();
    let otsu_thresh = imgproc::threshold(
        &tophat,
        &mut mask,
        0.0,
        255.0,
        imgproc::THRESH_BINARY | imgproc::THRESH_OTSU,
    )?;
    
    if otsu_thresh < 30.0 {
        imgproc::threshold(
            &tophat,
            &mut mask,
            30.0,
            255.0,
            imgproc::THRESH_BINARY,
        )?;
    }

    // 5. 形态学开运算去除微小斑点
    let k_open = imgproc::get_structuring_element(
        imgproc::MORPH_RECT,
        Size::new(3, 3),
        Point::new(-1, -1),
    )?;
    let mut cleaned = core::Mat::default();
    imgproc::morphology_ex(
        &mask,
        &mut cleaned,
        imgproc::MORPH_OPEN,
        &k_open,
        Point::new(-1, -1),
        1,
        core::BORDER_CONSTANT,
        imgproc::morphology_default_border_value()?,
    )?;

    // 6. 连通域二次去噪
    let mut labels = core::Mat::default();
    let mut stats = core::Mat::default();
    let mut centroids = core::Mat::default();
    let num_labels = imgproc::connected_components_with_stats(
        &cleaned,
        &mut labels,
        &mut stats,
        &mut centroids,
        8,
        core::CV_32S,
    )?;

    let mut final_mask = core::Mat::new_rows_cols_with_default(
        cleaned.rows(),
        cleaned.cols(),
        core::CV_8UC1,
        Scalar::all(0.0),
    )?;

    for i in 1..num_labels {
        let w = *stats.at_2d::<i32>(i, imgproc::CC_STAT_WIDTH)?;
        let h = *stats.at_2d::<i32>(i, imgproc::CC_STAT_HEIGHT)?;
        let area = *stats.at_2d::<i32>(i, imgproc::CC_STAT_AREA)?;
        let aspect_ratio = (w as f64) / (h as f64);

        if area >= 100 && h >= 18 && aspect_ratio <= 3.0 {
            let mut comp_mask = core::Mat::default();
            core::compare(&labels, &Scalar::all(i as f64), &mut comp_mask, core::CMP_EQ)?;
            final_mask.set_to(&Scalar::all(255.0), &comp_mask)?;
        }
    }

    Ok(final_mask)
}

fn extract_individual_digits(mask: &core::Mat, split_ratio: f64) -> opencv::Result<Vec<core::Mat>> {
    let mut labels = core::Mat::default();
    let mut stats = core::Mat::default();
    let mut centroids = core::Mat::default();
    let num_labels = imgproc::connected_components_with_stats(
        mask,
        &mut labels,
        &mut stats,
        &mut centroids,
        8,
        core::CV_32S,
    )?;

    let min_area = 100;
    let min_h = 18;

    let mut valid_rects = Vec::new();
    for i in 1..num_labels {
        let x = *stats.at_2d::<i32>(i, imgproc::CC_STAT_LEFT)?;
        let y = *stats.at_2d::<i32>(i, imgproc::CC_STAT_TOP)?;
        let w = *stats.at_2d::<i32>(i, imgproc::CC_STAT_WIDTH)?;
        let h = *stats.at_2d::<i32>(i, imgproc::CC_STAT_HEIGHT)?;
        let area = *stats.at_2d::<i32>(i, imgproc::CC_STAT_AREA)?;
        let aspect_ratio = (w as f64) / (h as f64);

        if area >= min_area && h >= min_h && aspect_ratio <= 3.0 {
            valid_rects.push(Rect::new(x, y, w, h));
        }
    }

    // 按 X 坐标从左到右排序
    valid_rects.sort_by_key(|r| r.x);

    let mut final_rects = Vec::new();
    for r in valid_rects {
        let expect_w = ((r.height as f64) * 0.62).round() as i32;
        let expect_w = expect_w.max(6);
        // 根据传入的 split_ratio 决定是否强行拆分连体字
        if r.width > (expect_w as f64 * split_ratio) as i32 {
            let num_parts = ((r.width as f64) / (expect_w as f64)).round().max(2.0) as i32;
            let part_w = r.width / num_parts;
            for p in 0..num_parts {
                let rx = r.x + p * part_w;
                let rw = if p == num_parts - 1 { r.x + r.width - rx } else { part_w };
                final_rects.push(Rect::new(rx, r.y, rw, r.height));
            }
        } else {
            final_rects.push(r);
        }
    }

    let mut digit_mats = Vec::new();
    for rect in final_rects {
        let digit_roi = core::Mat::roi(mask, rect)?;
        digit_mats.push(digit_roi.try_clone()?);
    }

    Ok(digit_mats)
}

/// 按照 MNIST 规范平滑抗锯齿与断笔，并将单字转换为 28x28 质心对齐图像
fn to_mnist_28(src: &core::Mat) -> opencv::Result<core::Mat> {
    let cols = src.cols();
    let rows = src.rows();
    if cols == 0 || rows == 0 {
        return core::Mat::new_rows_cols_with_default(28, 28, core::CV_8UC1, Scalar::all(0.0));
    }

    // 0. 轻微膨胀平滑 (修复断笔与锯齿，线条更丝滑)
    let k_smooth = imgproc::get_structuring_element(
        imgproc::MORPH_ELLIPSE,
        Size::new(2, 2),
        Point::new(-1, -1),
    )?;
    let mut src_smooth = core::Mat::default();
    imgproc::dilate(
        src,
        &mut src_smooth,
        &k_smooth,
        Point::new(-1, -1),
        1,
        core::BORDER_CONSTANT,
        imgproc::morphology_default_border_value()?,
    )?;

    // 1. 找前景外接框 (裁掉多余黑边)
    let mut min_x = cols;
    let mut max_x = 0i32;
    let mut min_y = rows;
    let mut max_y = 0i32;
    let mut has_fg = false;

    for y in 0..rows {
        for x in 0..cols {
            let val = *src_smooth.at_2d::<u8>(y, x)?;
            if val > 50 {
                if x < min_x { min_x = x; }
                if x > max_x { max_x = x; }
                if y < min_y { min_y = y; }
                if y > max_y { max_y = y; }
                has_fg = true;
            }
        }
    }

    if !has_fg || max_x < min_x || max_y < min_y {
        return core::Mat::new_rows_cols_with_default(28, 28, core::CV_8UC1, Scalar::all(0.0));
    }

    let crop_w = max_x - min_x + 1;
    let crop_h = max_y - min_y + 1;
    let cropped = core::Mat::roi(&src_smooth, Rect::new(min_x, min_y, crop_w, crop_h))?;

    // 2. 等比缩放，让长边 = 20
    let scale = 20.0 / (crop_w.max(crop_h) as f64);
    let nw = ((crop_w as f64 * scale).round() as i32).max(1);
    let nh = ((crop_h as f64 * scale).round() as i32).max(1);

    let mut resized = core::Mat::default();
    imgproc::resize(&cropped, &mut resized, Size::new(nw, nh), 0.0, 0.0, imgproc::INTER_AREA)?;

    // 3. 计算像素质心 Center of Mass
    let mut sum_v = 0.0f64;
    let mut sum_x = 0.0f64;
    let mut sum_y = 0.0f64;

    for y in 0..nh {
        for x in 0..nw {
            let v = *resized.at_2d::<u8>(y, x)? as f64;
            sum_v += v;
            sum_x += v * (x as f64);
            sum_y += v * (y as f64);
        }
    }

    let (cx, cy) = if sum_v > 0.0 {
        (sum_x / sum_v, sum_y / sum_v)
    } else {
        (nw as f64 / 2.0, nh as f64 / 2.0)
    };

    // 4. 按质心贴到 28x28 画布中心 (14.0, 14.0)
    let mut canvas = core::Mat::new_rows_cols_with_default(28, 28, core::CV_8UC1, Scalar::all(0.0))?;
    let offset_x = (14.0 - cx).round() as i32;
    let offset_y = (14.0 - cy).round() as i32;

    for y in 0..nh {
        for x in 0..nw {
            let tx = x + offset_x;
            let ty = y + offset_y;
            if tx >= 0 && tx < 28 && ty >= 0 && ty < 28 {
                let v = *resized.at_2d::<u8>(y, x)?;
                *canvas.at_2d_mut::<u8>(ty, tx)? = v;
            }
        }
    }

    Ok(canvas)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let split_ratio: f64 = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(1.55);

    let out_dir = "src/pic_cleaned";
    let mnist_dir = "src/pic_mnist28";
    fs::create_dir_all(out_dir)?;
    fs::create_dir_all(mnist_dir)?;

    let entries = fs::read_dir("src/pic")?;
    let mut files: Vec<_> = entries.filter_map(Result::ok).collect();
    files.sort_by_key(|a| a.path());

    println!("开始处理 src/pic 下的图片 (使用拆分倍数 split_ratio = {})...", split_ratio);

    for entry in files {
        let path = entry.path();
        if path.extension().unwrap_or_default() != "png" {
            continue;
        }

        let fname_stem = path.file_stem().unwrap().to_str().unwrap();
        let raw_img = imgcodecs::imread(path.to_str().unwrap(), imgcodecs::IMREAD_COLOR)?;
        if raw_img.empty() {
            continue;
        }

        let clean_mask = binarize_and_clean(&raw_img)?;
        let digit_mats = extract_individual_digits(&clean_mask, split_ratio)?;

        for (i, digit_mat) in digit_mats.iter().enumerate() {
            let out_path = format!("{}/{}_{}.png", out_dir, fname_stem, i);
            imgcodecs::imwrite(&out_path, digit_mat, &core::Vector::new())?;

            // 转换为 MNIST 28x28 质心对齐标准小图并保存
            let mnist_img = to_mnist_28(digit_mat)?;
            let mnist_path = format!("{}/{}_{}_28.png", mnist_dir, fname_stem, i);
            imgcodecs::imwrite(&mnist_path, &mnist_img, &core::Vector::new())?;

            println!("导出独立单字: {} | MNIST 规范: {}", out_path, mnist_path);
        }
    }

    println!("所有图片处理完成！");
    println!("1. 原始裁剪单字保存至: {}", out_dir);
    println!("2. 丝滑平滑 + 28x28 质心对齐单字保存至: {}", mnist_dir);
    Ok(())
}
