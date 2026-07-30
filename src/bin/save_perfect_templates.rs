use opencv::{
    core::{self, Point, Size},
    imgcodecs, imgproc,
    prelude::*,
};

fn to_binary_r_channel(img: &core::Mat, thresh: f64) -> opencv::Result<core::Mat> {
    let mut bgr = core::Vector::<core::Mat>::new();
    core::split(img, &mut bgr)?;
    let r = bgr.get(2)?;
    let mut binary = core::Mat::default();
    imgproc::threshold(&r, &mut binary, thresh, 255.0, imgproc::THRESH_BINARY)?;
    Ok(binary)
}

fn normalize_to_8x12(img: &core::Mat) -> opencv::Result<core::Mat> {
    let mut resized = core::Mat::default();
    imgproc::resize(
        img,
        &mut resized,
        Size::new(8, 12),
        0.0,
        0.0,
        imgproc::INTER_NEAREST,
    )?;
    let mut bin = core::Mat::default();
    imgproc::threshold(&resized, &mut bin, 127.0, 255.0, imgproc::THRESH_BINARY)?;
    Ok(bin)
}

fn main() -> opencv::Result<()> {
    let ground_truth: Vec<(&str, Vec<(usize, usize)>)> = vec![
        ("src/pic/QQ_1785395550263.png", vec![(0, 4), (1, 6)]),
        ("src/pic/QQ_1785395577212.png", vec![(0, 4), (1, 5)]),
        ("src/pic/QQ_1785395607716.png", vec![(0, 4), (1, 9)]),
        ("src/pic/QQ_1785395679500.png", vec![(0, 7), (1, 6)]),
        ("src/pic/QQ_1785395719748.png", vec![(0, 6), (1, 1)]),
        ("src/pic/QQ_1785395741836.png", vec![(0, 6), (1, 2)]),
        ("src/pic/QQ_1785400893053.png", vec![(0, 5), (1, 9)]),
    ];

    let thresh = 90.0;
    let mut count = vec![0; 10];

    for (img_path, labels) in &ground_truth {
        let raw_img = imgcodecs::imread(img_path, imgcodecs::IMREAD_COLOR)?;
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

        for &(roi_idx, digit) in labels {
            if roi_idx < rects.len() {
                let r = rects[roi_idx];
                let roi = core::Mat::roi(&mask, r)?;
                let roi_mat = roi.try_clone()?;
                let normalized_tpl = normalize_to_8x12(&roi_mat)?;

                let out_path = format!("src/templates/{}_{}.png", digit, count[digit]);
                imgcodecs::imwrite(&out_path, &normalized_tpl, &core::Vector::new())?;

                // 同时保存为主字模
                let main_path = format!("src/templates/{}.png", digit);
                imgcodecs::imwrite(&main_path, &normalized_tpl, &core::Vector::new())?;

                println!(
                    "保存数字 {} 样本 #{}: 来自 {}",
                    digit, count[digit], img_path
                );
                count[digit] += 1;
            }
        }
    }

    Ok(())
}
