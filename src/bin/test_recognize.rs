use opencv::{
    core::{self, Point, Size},
    imgcodecs, imgproc,
    prelude::*,
};
use std::fs;

fn to_binary_r_channel(img: &core::Mat, thresh: f64) -> opencv::Result<core::Mat> {
    let mut bgr = core::Vector::<core::Mat>::new();
    core::split(img, &mut bgr)?;
    let r = bgr.get(2)?;
    let mut binary = core::Mat::default();
    imgproc::threshold(&r, &mut binary, thresh, 255.0, imgproc::THRESH_BINARY)?;
    Ok(binary)
}

fn main() -> opencv::Result<()> {
    let thresh = 90.0;
    let target_size = Size::new(8, 12);

    // 1. 加载所有 0-9 多变体字模库（支持多样本模式）
    let mut templates: Vec<Vec<core::Mat>> = vec![vec![]; 10];

    // 加载 src/templates/ 下所有文件
    if let Ok(entries) = fs::read_dir("src/templates") {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().unwrap_or_default() != "png" {
                continue;
            }
            let name = path.file_stem().unwrap().to_str().unwrap();
            // 获取数字类别 (例如 "5", "5_var2" -> 5)
            if let Some(digit_char) = name.chars().next() {
                if let Some(digit) = digit_char.to_digit(10) {
                    let tpl =
                        imgcodecs::imread(path.to_str().unwrap(), imgcodecs::IMREAD_GRAYSCALE)?;
                    if !tpl.empty() {
                        let mut resized_tpl = core::Mat::default();
                        imgproc::resize(
                            &tpl,
                            &mut resized_tpl,
                            target_size,
                            0.0,
                            0.0,
                            imgproc::INTER_NEAREST,
                        )?;
                        let mut binary = core::Mat::default();
                        imgproc::threshold(
                            &resized_tpl,
                            &mut binary,
                            127.0,
                            255.0,
                            imgproc::THRESH_BINARY,
                        )?;
                        templates[digit as usize].push(binary);
                    }
                }
            }
        }
    }

    let entries = fs::read_dir("src/pic").unwrap();
    let mut files: Vec<_> = entries.filter_map(Result::ok).collect();
    files.sort_by_key(|a| a.path());

    println!("--- 开始【多样本 8x12 矢量比对】测试 ---");

    for entry in files {
        let path = entry.path();
        if path.extension().unwrap_or_default() != "png" {
            continue;
        }

        let raw_img = imgcodecs::imread(path.to_str().unwrap(), imgcodecs::IMREAD_COLOR)?;
        if raw_img.empty() {
            continue;
        }

        let mask = to_binary_r_channel(&raw_img, thresh)?;

        let mut contours = core::Vector::<core::Vector<Point>>::new();
        imgproc::find_contours(
            &mask,
            &mut contours,
            imgproc::RETR_EXTERNAL,
            imgproc::CHAIN_APPROX_SIMPLE,
            Point::new(0, 0),
        )?;

        let mut rects = Vec::new();
        for i in 0..contours.len() {
            let c = contours.get(i)?;
            let rect = opencv::geometry::bounding_rect(&c)?;
            if rect.width >= 2 && rect.height >= 4 {
                rects.push(rect);
            }
        }
        rects.sort_by_key(|r| r.x);

        let mut final_digits = Vec::new();
        let mut debug_scores = Vec::new();

        for rect in rects {
            let roi = core::Mat::roi(&mask, rect)?;

            let mut roi_resized = core::Mat::default();
            imgproc::resize(
                &roi,
                &mut roi_resized,
                target_size,
                0.0,
                0.0,
                imgproc::INTER_NEAREST,
            )?;
            let mut roi_bin = core::Mat::default();
            imgproc::threshold(
                &roi_resized,
                &mut roi_bin,
                127.0,
                255.0,
                imgproc::THRESH_BINARY,
            )?;

            let mut best_score = i32::MAX;
            let mut best_digit = -1i32;
            let mut all_scores = Vec::new();

            for i in 0..=9usize {
                let vars = &templates[i];
                if vars.is_empty() {
                    continue;
                }

                // 取该数字所有样本变体中的最小差异分
                let mut min_var_score = i32::MAX;
                for tpl in vars {
                    let mut diff = core::Mat::default();
                    core::bitwise_xor(&roi_bin, tpl, &mut diff, &core::no_array())?;
                    let score = core::count_non_zero(&diff)?;
                    if score < min_var_score {
                        min_var_score = score;
                    }
                }

                all_scores.push((i, min_var_score));
                if min_var_score < best_score {
                    best_score = min_var_score;
                    best_digit = i as i32;
                }
            }

            all_scores.sort_by_key(|a| a.1);
            let top3: Vec<String> = all_scores
                .iter()
                .take(3)
                .map(|(d, s)| format!("{}({})", d, s))
                .collect();

            final_digits.push(best_digit.to_string());
            debug_scores.push(format!("[{}]", top3.join(",")));
        }

        println!(
            "{:<25} => {} | {}",
            path.file_name().unwrap().to_str().unwrap(),
            final_digits.join(""),
            debug_scores.join(" ")
        );
    }
    println!("--- 完成 ---");
    Ok(())
}
