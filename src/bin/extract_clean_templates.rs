use opencv::{
    core::{self, Point, Size},
    imgcodecs, imgproc, prelude::*,
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

fn print_ascii_mat(title: &str, mat: &core::Mat) -> opencv::Result<()> {
    println!("\n--- {} ({}x{}) ---", title, mat.cols(), mat.rows());
    for r in 0..mat.rows() {
        let mut line = String::new();
        for c in 0..mat.cols() {
            let p = *mat.at_2d::<u8>(r, c)?;
            if p > 128 {
                line.push_str("██");
            } else {
                line.push_str("░░");
            }
        }
        println!("{}", line);
    }
    Ok(())
}

fn main() -> opencv::Result<()> {
    // 已知各图片的真实数字
    let ground_truth: Vec<(&str, Vec<(usize, usize)>)> = vec![
        ("src/pic/QQ_1785395550263.png", vec![(0, 4), (1, 6)]),
        ("src/pic/QQ_1785395577212.png", vec![(0, 4), (1, 5)]),
        ("src/pic/QQ_1785395607716.png", vec![(0, 4), (1, 9)]),
        ("src/pic/QQ_1785395679500.png", vec![(0, 7), (1, 6)]),
        ("src/pic/QQ_1785395719748.png", vec![(0, 6), (1, 1)]),
        ("src/pic/QQ_1785395741836.png", vec![(0, 6), (1, 2)]),
        ("src/pic/QQ_1785395795605.png", vec![(0, 3), (1, 6)]),
    ];

    let _ = fs::create_dir_all("src/templates_clean");

    for thresh in [80.0, 100.0, 120.0] {
        println!("\n==========================================");
        println!("  测试阈值: R_channel > {}", thresh);
        println!("==========================================");

        for (img_path, labels) in &ground_truth {
            let img = imgcodecs::imread(img_path, imgcodecs::IMREAD_COLOR)?;
            let mask = to_binary_r_channel(&img, thresh)?;

            // 找轮廓（直接在 raw mask 上找，不加 MORPH_CLOSE 填充！）
            let mut contours = core::Vector::<core::Vector<Point>>::new();
            imgproc::find_contours(&mask, &mut contours, imgproc::RETR_EXTERNAL,
                imgproc::CHAIN_APPROX_SIMPLE, Point::new(0, 0))?;

            let mut rects = Vec::new();
            for i in 0..contours.len() {
                let c = contours.get(i)?;
                let rect = opencv::geometry::bounding_rect(&c)?;
                if rect.width >= 2 && rect.height >= 5 {
                    rects.push(rect);
                }
            }
            rects.sort_by_key(|r| r.x);

            for &(roi_idx, digit) in labels {
                if roi_idx < rects.len() {
                    let r = rects[roi_idx];
                    let roi = core::Mat::roi(&mask, r)?;
                    let roi_mat = roi.try_clone()?;
                    let title = format!("数字 {} (来自 {} 阈值 {})", digit, img_path.split('/').last().unwrap(), thresh);
                    print_ascii_mat(&title, &roi_mat)?;

                    if (thresh - 100.0).abs() < 1.0 {
                        let out_path = format!("src/templates_clean/{}.png", digit);
                        imgcodecs::imwrite(&out_path, &roi_mat, &core::Vector::new())?;
                    }
                }
            }
        }
    }

    Ok(())
}
