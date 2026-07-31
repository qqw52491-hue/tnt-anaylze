use opencv::{
    core::{self, Point, Rect, Scalar, Size},
    imgcodecs, imgproc,
    prelude::*,
};
use std::fs;

fn binarize_and_clean(roi: &core::Mat) -> opencv::Result<(core::Mat, core::Mat)> {
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

    let mut up = core::Mat::default();
    imgproc::resize(
        &gray,
        &mut up,
        Size::new(gray.cols() * UPSCALE, gray.rows() * UPSCALE),
        0.0,
        0.0,
        imgproc::INTER_CUBIC,
    )?;

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

    let mut gray_out = core::Mat::new_rows_cols_with_default(
        cleaned.rows(),
        cleaned.cols(),
        core::CV_8UC1,
        Scalar::all(0.0),
    )?;
    tophat.copy_to_masked(&mut gray_out, &final_mask)?;

    Ok((final_mask, gray_out))
}

fn split_by_valley(
    mask: &core::Mat,
    r: Rect,
    num_parts: i32,
) -> opencv::Result<Vec<Rect>> {
    let sub = core::Mat::roi(mask, r)?;

    let mut col_sum = vec![0i32; r.width as usize];
    for y in 0..r.height {
        for x in 0..r.width {
            if *sub.at_2d::<u8>(y, x)? > 0 {
                col_sum[x as usize] += 1;
            }
        }
    }

    let mut cuts = vec![0i32];
    for p in 1..num_parts {
        let center = r.width * p / num_parts;
        let span = ((r.width / num_parts) as f64 * 0.30).round().max(2.0) as i32;
        let lo = (center - span).max(1);
        let hi = (center + span).min(r.width - 1);

        let mut best = center;
        let mut best_v = i32::MAX;
        for x in lo..hi {
            let v = col_sum[x as usize];
            if v < best_v || (v == best_v && (x - center).abs() < (best - center).abs()) {
                best_v = v;
                best = x;
            }
        }

        if (best_v as f64) > (r.height as f64 * 0.60) {
            return Ok(vec![r]);
        }

        cuts.push(best);
    }
    cuts.push(r.width);

    Ok(cuts
        .windows(2)
        .filter(|w| w[1] - w[0] >= 3)
        .map(|w| Rect::new(r.x + w[0], r.y, w[1] - w[0], r.height))
        .collect())
}

fn extract_individual_digits(
    mask: &core::Mat,
    gray: &core::Mat,
    split_ratio: f64,
) -> opencv::Result<Vec<core::Mat>> {
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

    valid_rects.sort_by_key(|r| r.x);

    // 为了更精准地切分粘连字符（如 36），我们专门为 split_by_valley 准备一个腐蚀过的 mask
    // 这样 3 和 6 之间细微的粘连会被断开，投影曲线在中间会有真正的谷底
    let k_erode = imgproc::get_structuring_element(
        imgproc::MORPH_RECT,
        core::Size::new(2, 2),
        core::Point::new(-1, -1),
    )?;
    let mut eroded_mask = core::Mat::default();
    imgproc::erode(
        mask,
        &mut eroded_mask,
        &k_erode,
        core::Point::new(-1, -1),
        1,
        core::BORDER_CONSTANT,
        imgproc::morphology_default_border_value()?,
    )?;

    let mut final_rects = Vec::new();
    for r in valid_rects {
        let expect_w = (((r.height as f64) * 0.62).round() as i32).max(6);
        if r.width > (expect_w as f64 * split_ratio) as i32 {
            let num_parts = ((r.width as f64) / (expect_w as f64)).round().max(2.0) as i32;
            // 用瘦身后的 eroded_mask 寻找切割点，切分更加准确
            final_rects.extend(split_by_valley(&eroded_mask, r, num_parts)?);
        } else {
            final_rects.push(r);
        }
    }

    let mut digit_mats = Vec::new();
    for rect in final_rects {
        let digit_roi = core::Mat::roi(gray, rect)?;
        digit_mats.push(digit_roi.try_clone()?);
    }

    Ok(digit_mats)
}

fn to_template_40(src: &core::Mat) -> opencv::Result<core::Mat> {
    let cols = src.cols();
    let rows = src.rows();
    if cols == 0 || rows == 0 {
        return core::Mat::new_rows_cols_with_default(40, 40, core::CV_8UC1, Scalar::all(0.0));
    }

    let mut min_x = cols;
    let mut max_x = 0i32;
    let mut min_y = rows;
    let mut max_y = 0i32;
    let mut has_fg = false;

    for y in 0..rows {
        for x in 0..cols {
            let val = *src.at_2d::<u8>(y, x)?;
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
        return core::Mat::new_rows_cols_with_default(40, 40, core::CV_8UC1, Scalar::all(0.0));
    }

    let crop_w = max_x - min_x + 1;
    let crop_h = max_y - min_y + 1;
    let cropped = core::Mat::roi(src, Rect::new(min_x, min_y, crop_w, crop_h))?;

    let scale = 36.0 / (crop_w.max(crop_h) as f64);
    let nw = ((crop_w as f64 * scale).round() as i32).max(1);
    let nh = ((crop_h as f64 * scale).round() as i32).max(1);

    let mut resized = core::Mat::default();
    imgproc::resize(&cropped, &mut resized, Size::new(nw, nh), 0.0, 0.0, imgproc::INTER_AREA)?;

    // 对于固定字体，直接用几何中心对齐即可，最稳
    let mut canvas = core::Mat::new_rows_cols_with_default(40, 40, core::CV_8UC1, Scalar::all(0.0))?;
    let offset_x = (40 - nw) / 2;
    let offset_y = (40 - nh) / 2;

    for y in 0..nh {
        for x in 0..nw {
            let tx = x + offset_x;
            let ty = y + offset_y;
            if tx >= 0 && tx < 40 && ty >= 0 && ty < 40 {
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
    let tmpl_dir = "src/pic_template40";
    fs::create_dir_all(out_dir)?;
    fs::create_dir_all(tmpl_dir)?;

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

        let (clean_mask, clean_gray) = binarize_and_clean(&raw_img)?;
        let digit_mats = extract_individual_digits(&clean_mask, &clean_gray, split_ratio)?;

        for (i, digit_mat) in digit_mats.iter().enumerate() {
            let out_path = format!("{}/{}_{}.png", out_dir, fname_stem, i);
            imgcodecs::imwrite(&out_path, digit_mat, &core::Vector::new())?;

            let tmpl_img = to_template_40(digit_mat)?;
            let tmpl_path = format!("{}/{}_{}_40.png", tmpl_dir, fname_stem, i);
            imgcodecs::imwrite(&tmpl_path, &tmpl_img, &core::Vector::new())?;

            println!("导出独立单字: {} | 40x40 模板: {}", out_path, tmpl_path);
        }
    }

    println!("所有图片处理完成！");
    println!("1. 原始裁剪单字保存至: {}", out_dir);
    println!("2. 40x40 高清去模糊单字保存至: {}", tmpl_dir);
    Ok(())
}
