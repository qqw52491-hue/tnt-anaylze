use opencv::{
    core::{self, Point, Rect, Scalar},
    imgcodecs, imgproc,
    prelude::*,
};
use std::path::Path;

use crate::detect;

/// 5大 ROI 识别结果
#[derive(Debug, Clone)]
pub struct RecognizerResult {
    /// 1. 小地图视野框
    pub camera_rect: Option<Rect>,
    pub camera_width: f64,
    /// 2. 小地图点点: (蓝点/己方, 红点/敌方)
    pub player_dots: Vec<Point>,
    pub enemy_dots: Vec<Point>,
    /// 3. 角度数字 (例如 45)
    pub angle: Option<i32>,
    /// 4. 力度百分比 (0.0 ~ 100.0%)
    pub power_percent: f64,
    /// 5. 风力值/偏移量 (-10.0 ~ +10.0)
    pub wind_value: f64,
}

/// 模板匹配的判定阈值。用的是 ZNCC（零均值归一化互相关），取值 -1..=1。
const SIM_MIN: f64 = 0.58;
/// 最佳候选需要领先"第二名的其他数字"的幅度。
///
/// 卡得太死会适得其反：3、8、9 和 7、1、9 在 ZNCC 下本来就很像，
/// 一帧稍糊就可能是 0.63 vs 0.62——冠军是对的，却因为领先不够被否决。
const SIM_MARGIN: f64 = 0.005;

/// 角度上限。反抛物线可以打过头顶，屏幕上会显示 91..=180。
const ANGLE_MAX: i32 = 180;

pub struct UiRecognizer {
    /// (数字, 去均值后的 40x40 展平像素, 去均值后的 L2 范数)
    /// 预展平 + 预去均值，是为了让热循环彻底不碰 OpenCV 的 at_2d。
    templates: Vec<(u8, Vec<f64>, f64)>,
    /// 失败样本落盘开关，由环境变量 TNT_DUMP_FAIL 控制。
    dump_fail: bool,
}

impl UiRecognizer {
    /// 把一张 40x40 灰度图展平并去均值，返回 (展平像素, L2 范数)。
    fn flatten_centered(img: &core::Mat) -> opencv::Result<(Vec<f64>, f64)> {
        let mut flat = vec![0f64; 40 * 40];
        let mut sum = 0f64;
        for y in 0..40 {
            for x in 0..40 {
                let v = *img.at_2d::<u8>(y, x)? as f64;
                flat[(y * 40 + x) as usize] = v;
                sum += v;
            }
        }
        let mean = sum / (40.0 * 40.0);
        let mut norm_sq = 0f64;
        for v in flat.iter_mut() {
            *v -= mean;
            norm_sq += *v * *v;
        }
        Ok((flat, norm_sq.sqrt()))
    }

    /// 初始化识别器
    pub fn new<P: AsRef<Path>>(template_dir: P) -> opencv::Result<Self> {
        let mut templates = Vec::new();

        // 笔画粗细扩增用的核。
        let k = imgproc::get_structuring_element(
            imgproc::MORPH_RECT,
            core::Size::new(2, 2),
            Point::new(-1, -1),
        )?;

        if let Ok(entries) = std::fs::read_dir(template_dir) {
            for entry in entries.filter_map(Result::ok) {
                let path = entry.path();
                if path.extension().unwrap_or_default() != "png" {
                    continue;
                }
                let fname = path.file_name().unwrap().to_string_lossy().to_string();
                let Some(first_char) = fname.chars().next() else { continue };
                let Some(digit) = first_char.to_digit(10) else { continue };

                let img = imgcodecs::imread(path.to_str().unwrap(), imgcodecs::IMREAD_GRAYSCALE)?;
                // 模板必须是 40x40，否则展平就对不上了
                if img.empty() || img.rows() != 40 || img.cols() != 40 {
                    continue;
                }

                // 每张模板额外生成一张变细、一张变粗的。
                // 同一个数字在画面里会因为抗锯齿、UPSCALE=3 插值、tophat 响应强弱
                // 而笔画胖一圈或瘦一圈，这正是样本少的数字盖不住的维度。
                let mut thin = core::Mat::default();
                imgproc::erode(
                    &img,
                    &mut thin,
                    &k,
                    Point::new(-1, -1),
                    1,
                    core::BORDER_CONSTANT,
                    imgproc::morphology_default_border_value()?,
                )?;
                let mut thick = core::Mat::default();
                imgproc::dilate(
                    &img,
                    &mut thick,
                    &k,
                    Point::new(-1, -1),
                    1,
                    core::BORDER_CONSTANT,
                    imgproc::morphology_default_border_value()?,
                )?;

                for variant in [&img, &thin, &thick] {
                    let (flat, norm) = Self::flatten_centered(variant)?;
                    if norm > 1e-6 {
                        templates.push((digit as u8, flat, norm));
                    }
                }
            }
        }

        let dump_fail = std::env::var("TNT_DUMP_FAIL").is_ok();
        if dump_fail {
            eprintln!(
                "[ui] 失败样本落盘已开启 -> /tmp/tnt_fail_*.png（已加载 {} 个模板变体）",
                templates.len()
            );
        }

        Ok(Self { templates, dump_fail })
    }

    /// 1. 小地图·视野框识别
    ///
    /// 委托给 crate::detect，避免 ui.rs / main.rs / live_gui.rs 各维护一套阈值。
    pub fn detect_minimap_fov(&self, minimap: &core::Mat) -> opencv::Result<(Option<Rect>, f64)> {
        let rect = detect::detect_camera_frame(minimap)?;
        let camera_width = rect.map(|r| r.width as f64).unwrap_or(170.0);
        Ok((rect, camera_width))
    }

    /// 2. 小地图·点点识别: (己方蓝, 敌方红)
    ///
    /// 委托给 crate::detect：连通域 + 填充率 + 黑描边 + 圆度打分，
    /// 阈值按小地图实际宽度自动缩放。返回值按置信度降序。
    pub fn detect_minimap_dots(
        &self,
        minimap: &core::Mat,
    ) -> opencv::Result<(Vec<Point>, Vec<Point>)> {
        let params = detect::DotParams::for_width(minimap.cols());
        let blue = detect::detect_dots(minimap, false, &params)?;
        let red = detect::detect_dots(minimap, true, &params)?;
        Ok((
            blue.iter().map(|d| d.pt).collect(),
            red.iter().map(|d| d.pt).collect(),
        ))
    }

    /// 4. 力度条识别: 底部力度条填充像素长度 ÷ 总长度 (不用 OCR)
    pub fn measure_power_bar(&self, power_bar_roi: &core::Mat) -> opencv::Result<f64> {
        let cols = power_bar_roi.cols();
        let rows = power_bar_roi.rows();
        if cols == 0 || rows == 0 {
            return Ok(0.0);
        }

        let mut hsv = core::Mat::default();
        imgproc::cvt_color(
            power_bar_roi,
            &mut hsv,
            imgproc::COLOR_BGR2HSV,
            0,
            core::AlgorithmHint::ALGO_HINT_DEFAULT,
        )?;

        let lower_red1 = Scalar::new(0.0, 100.0, 100.0, 0.0);
        let upper_red1 = Scalar::new(10.0, 255.0, 255.0, 0.0);
        let lower_red2 = Scalar::new(160.0, 100.0, 100.0, 0.0);
        // FIX: 原来写的是 (180, 100, 100)，S/V 上界等于下界，mask2 恒为空，
        // 导致 H>=160 那一段红色永远匹配不到。
        let upper_red2 = Scalar::new(180.0, 255.0, 255.0, 0.0);

        let mut mask1 = core::Mat::default();
        let mut mask2 = core::Mat::default();
        let mut mask = core::Mat::default();
        core::in_range(&hsv, &lower_red1, &upper_red1, &mut mask1)?;
        core::in_range(&hsv, &lower_red2, &upper_red2, &mut mask2)?;
        core::bitwise_or(&mask1, &mask2, &mut mask, &core::no_array())?;

        // FIX: 原来一列只要有 1 个像素就算填充，任何噪点都能把读数顶到 100%。
        // 现在要求该列至少 30% 的行被填充。
        let need = ((rows as f64) * 0.3).ceil() as i32;
        let mut filled_end_x: i32 = -1;
        for x in 0..cols {
            let mut cnt = 0;
            for y in 0..rows {
                if *mask.at_2d::<u8>(y, x)? > 0 {
                    cnt += 1;
                }
            }
            if cnt >= need {
                filled_end_x = x;
            }
        }

        if filled_end_x < 0 {
            return Ok(0.0);
        }

        let percent = (filled_end_x as f64 / (cols - 1).max(1) as f64) * 100.0;
        Ok(percent.clamp(0.0, 100.0))
    }

    /// 5. 风力指示器识别: 顶部指示条箭头中心相对中线的偏移量 (不用 OCR)
    pub fn measure_wind_indicator(&self, wind_roi: &core::Mat) -> opencv::Result<f64> {
        let cols = wind_roi.cols();
        let rows = wind_roi.rows();
        if cols == 0 || rows == 0 {
            return Ok(0.0);
        }

        let center_x = cols as f64 / 2.0;

        let mut hsv = core::Mat::default();
        imgproc::cvt_color(
            wind_roi,
            &mut hsv,
            imgproc::COLOR_BGR2HSV,
            0,
            core::AlgorithmHint::ALGO_HINT_DEFAULT,
        )?;

        // FIX: 白色 = 高明度 且 低饱和度。原来 S 上界放到 255，
        // 等于把所有亮色（黄条、红字、UI 高光）全算成箭头了。
        let lower = Scalar::new(0.0, 0.0, 200.0, 0.0);
        let upper = Scalar::new(180.0, 60.0, 255.0, 0.0);
        let mut mask = core::Mat::default();
        core::in_range(&hsv, &lower, &upper, &mut mask)?;

        let mut contours = core::Vector::<core::Vector<Point>>::new();
        imgproc::find_contours(
            &mask,
            &mut contours,
            imgproc::RETR_EXTERNAL,
            imgproc::CHAIN_APPROX_SIMPLE,
            Point::new(0, 0),
        )?;

        let mut best_arrow_x: Option<f64> = None;
        let mut max_area = 0;

        for i in 0..contours.len() {
            let contour = contours.get(i)?;
            let rect = opencv::geometry::bounding_rect(&contour)?;
            let area = rect.width * rect.height;

            // FIX: 原来没有任何形状约束，直接取最大白色轮廓。
            if area < 6 || rect.height < 3 {
                continue; // 太小 = 噪点
            }
            if rect.width > cols / 3 {
                continue; // 太宽 = UI 长条，不是箭头
            }
            let aspect = rect.width as f64 / (rect.height.max(1)) as f64;
            if !(0.3..=3.0).contains(&aspect) {
                continue;
            }
            if area > max_area {
                max_area = area;
                best_arrow_x = Some(rect.x as f64 + rect.width as f64 / 2.0);
            }
        }

        // FIX: 找不到箭头就老实返回 0，而不是拿中线当结果假装识别成功。
        let Some(arrow_x) = best_arrow_x else {
            return Ok(0.0);
        };

        let offset_px = arrow_x - center_x;
        let wind_val = (offset_px / (cols as f64 / 2.0)) * 10.0;
        Ok(wind_val)
    }

    /// 连通域是不是一个合法的数字笔画。
    ///
    /// FIX: 原来是写死的 `area >= 100`。这个绝对面积门槛对 1/3/7 这种
    /// 笔画稀疏的数字非常不友好。现在改成跟字高挂钩的相对门槛：
    /// 高度已经卡住 h >= 18，噪点根本长不到这么高。
    fn is_digit_component(w: i32, h: i32, area: i32) -> bool {
        if h < 18 {
            return false;
        }
        let aspect = (w as f64) / (h as f64);
        if aspect > 3.0 {
            return false;
        }
        let min_area = (((h as f64) * 1.2).max(40.0)) as i32;
        area >= min_area
    }

    pub fn binarize_and_clean(&self, roi: &core::Mat) -> opencv::Result<(core::Mat, core::Mat)> {
        const UPSCALE: i32 = 3;

        let gray = if roi.channels() == 3 {
            let mut hsv = core::Mat::default();
            imgproc::cvt_color(roi, &mut hsv, imgproc::COLOR_BGR2HSV, 0, core::AlgorithmHint::ALGO_HINT_DEFAULT)?;
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
        imgproc::resize(&gray, &mut up, core::Size::new(gray.cols() * UPSCALE, gray.rows() * UPSCALE), 0.0, 0.0, imgproc::INTER_CUBIC)?;

        let k = imgproc::get_structuring_element(imgproc::MORPH_ELLIPSE, core::Size::new(15, 15), Point::new(-1, -1))?;
        let mut tophat = core::Mat::default();
        imgproc::morphology_ex(&up, &mut tophat, imgproc::MORPH_TOPHAT, &k, Point::new(-1, -1), 1, core::BORDER_REPLICATE, imgproc::morphology_default_border_value()?)?;

        let mut mask = core::Mat::default();
        let otsu_thresh = imgproc::threshold(&tophat, &mut mask, 0.0, 255.0, imgproc::THRESH_BINARY | imgproc::THRESH_OTSU)?;
        if otsu_thresh < 30.0 {
            imgproc::threshold(&tophat, &mut mask, 30.0, 255.0, imgproc::THRESH_BINARY)?;
        }

        let k_open = imgproc::get_structuring_element(imgproc::MORPH_RECT, core::Size::new(3, 3), Point::new(-1, -1))?;
        let mut cleaned = core::Mat::default();
        imgproc::morphology_ex(&mask, &mut cleaned, imgproc::MORPH_OPEN, &k_open, Point::new(-1, -1), 1, core::BORDER_CONSTANT, imgproc::morphology_default_border_value()?)?;

        let mut labels = core::Mat::default();
        let mut stats = core::Mat::default();
        let mut centroids = core::Mat::default();
        let num_labels = imgproc::connected_components_with_stats(&cleaned, &mut labels, &mut stats, &mut centroids, 8, core::CV_32S)?;

        let mut final_mask = core::Mat::new_rows_cols_with_default(cleaned.rows(), cleaned.cols(), core::CV_8UC1, Scalar::all(0.0))?;

        for i in 1..num_labels {
            let w = *stats.at_2d::<i32>(i, imgproc::CC_STAT_WIDTH)?;
            let h = *stats.at_2d::<i32>(i, imgproc::CC_STAT_HEIGHT)?;
            let area = *stats.at_2d::<i32>(i, imgproc::CC_STAT_AREA)?;

            if Self::is_digit_component(w, h, area) {
                let mut comp_mask = core::Mat::default();
                core::compare(&labels, &Scalar::all(i as f64), &mut comp_mask, core::CMP_EQ)?;
                final_mask.set_to(&Scalar::all(255.0), &comp_mask)?;
            }
        }

        let mut gray_out = core::Mat::new_rows_cols_with_default(cleaned.rows(), cleaned.cols(), core::CV_8UC1, Scalar::all(0.0))?;
        tophat.copy_to_masked(&mut gray_out, &final_mask)?;

        Ok((final_mask, gray_out))
    }

    fn split_by_valley(&self, mask: &core::Mat, r: Rect, num_parts: i32) -> opencv::Result<Vec<Rect>> {
        let sub = core::Mat::roi(mask, r)?;
        let mut col_sum = vec![0i32; r.width as usize];
        for y in 0..r.height {
            for x in 0..r.width {
                if *sub.at_2d::<u8>(y, x)? > 0 { col_sum[x as usize] += 1; }
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
            if (best_v as f64) > (r.height as f64 * 0.60) { return Ok(vec![r]); }
            cuts.push(best);
        }
        cuts.push(r.width);
        Ok(cuts.windows(2).filter(|w| w[1] - w[0] >= 3).map(|w| Rect::new(r.x + w[0], r.y, w[1] - w[0], r.height)).collect())
    }

    pub fn extract_individual_digits(&self, mask: &core::Mat, gray: &core::Mat, split_ratio: f64) -> opencv::Result<Vec<core::Mat>> {
        let mut labels = core::Mat::default();
        let mut stats = core::Mat::default();
        let mut centroids = core::Mat::default();
        let num_labels = imgproc::connected_components_with_stats(mask, &mut labels, &mut stats, &mut centroids, 8, core::CV_32S)?;

        let mut valid_rects = Vec::new();
        for i in 1..num_labels {
            let x = *stats.at_2d::<i32>(i, imgproc::CC_STAT_LEFT)?;
            let y = *stats.at_2d::<i32>(i, imgproc::CC_STAT_TOP)?;
            let w = *stats.at_2d::<i32>(i, imgproc::CC_STAT_WIDTH)?;
            let h = *stats.at_2d::<i32>(i, imgproc::CC_STAT_HEIGHT)?;
            let area = *stats.at_2d::<i32>(i, imgproc::CC_STAT_AREA)?;
            if Self::is_digit_component(w, h, area) {
                valid_rects.push(Rect::new(x, y, w, h));
            }
        }
        valid_rects.sort_by_key(|r| r.x);

        let k_erode = imgproc::get_structuring_element(imgproc::MORPH_RECT, core::Size::new(2, 2), core::Point::new(-1, -1))?;
        let mut eroded_mask = core::Mat::default();
        imgproc::erode(mask, &mut eroded_mask, &k_erode, core::Point::new(-1, -1), 1, core::BORDER_CONSTANT, imgproc::morphology_default_border_value()?)?;

        let mut final_rects = Vec::new();
        for r in valid_rects {
            let expect_w = (((r.height as f64) * 0.62).round() as i32).max(6);
            if r.width > (expect_w as f64 * split_ratio) as i32 {
                let num_parts = ((r.width as f64) / (expect_w as f64)).round().max(2.0) as i32;
                final_rects.extend(self.split_by_valley(&eroded_mask, r, num_parts)?);
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

    pub fn to_template_40(&self, src: &core::Mat) -> opencv::Result<core::Mat> {
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

        // FIX: 前景判定从 >50 降到 >30，与 binarize_and_clean 里 Otsu 的兜底阈值一致。
        // 原来 3 和 7 的细笔画末端（tophat 响应本来就弱）会被排除在外接框之外，
        // 导致居中和缩放都偏掉。
        for y in 0..rows {
            for x in 0..cols {
                let val = *src.at_2d::<u8>(y, x)?;
                if val > 30 {
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
        imgproc::resize(&cropped, &mut resized, core::Size::new(nw, nh), 0.0, 0.0, imgproc::INTER_AREA)?;

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

    /// 算出 0..9 每个数字的最佳 ZNCC 得分。
    ///
    /// 用 ZNCC（先减各自均值）而不是裸余弦：灰度像素恒为非负，
    /// 裸余弦会严重依赖前景像素的多少，稀疏的 3、7 分数被系统性压低。
    /// 去均值后相似度只反映形状，笔画密度不再影响判定。
    ///
    /// PERF: 模板已在 new() 里预展平并预去均值，热循环完全不碰 at_2d。
    fn score_digits(&self, target: &core::Mat) -> opencv::Result<[f64; 10]> {
        let mut per_digit = [-2.0f64; 10];
        if self.templates.is_empty() {
            return Ok(per_digit);
        }

        let mut raw = vec![0f64; 40 * 40];
        for y in 0..40 {
            for x in 0..40 {
                raw[(y * 40 + x) as usize] = *target.at_2d::<u8>(y, x)? as f64;
            }
        }

        let mut shifted = vec![0f64; 40 * 40];

        for dy in -1i32..=1 {
            for dx in -1i32..=1 {
                let mut sum = 0f64;
                for y in 0..40i32 {
                    for x in 0..40i32 {
                        let sy = y + dy;
                        let sx = x + dx;
                        let v = if sx >= 0 && sx < 40 && sy >= 0 && sy < 40 {
                            raw[(sy * 40 + sx) as usize]
                        } else {
                            0.0
                        };
                        shifted[(y * 40 + x) as usize] = v;
                        sum += v;
                    }
                }

                let mean = sum / (40.0 * 40.0);
                let mut norm_sq = 0f64;
                for v in shifted.iter_mut() {
                    *v -= mean;
                    norm_sq += *v * *v;
                }
                let norm = norm_sq.sqrt();
                if norm <= 1e-6 {
                    continue; // 全空白
                }

                for (digit, tmpl, tmpl_norm) in &self.templates {
                    let mut dot = 0f64;
                    for i in 0..(40 * 40) {
                        dot += shifted[i] * tmpl[i];
                    }
                    let sim = dot / (norm * tmpl_norm);
                    let slot = &mut per_digit[*digit as usize];
                    if sim > *slot {
                        *slot = sim;
                    }
                }
            }
        }

        Ok(per_digit)
    }

    /// 从得分表里选出 (冠军数字, 冠军分, 亚军分)。
    fn pick_best(per_digit: &[f64; 10]) -> (usize, f64, f64) {
        let mut best_idx = 0usize;
        let mut best_sim = -2.0f64;
        for (i, v) in per_digit.iter().enumerate() {
            if *v > best_sim {
                best_sim = *v;
                best_idx = i;
            }
        }
        let mut second_sim = -2.0f64;
        for (i, v) in per_digit.iter().enumerate() {
            if i != best_idx && *v > second_sim {
                second_sim = *v;
            }
        }
        (best_idx, best_sim, second_sim)
    }

    /// 模板匹配单个数字，返回 (数字, ZNCC 相似度)。
    fn match_digit(&self, target: &core::Mat) -> opencv::Result<(Option<u8>, f64)> {
        let per_digit = self.score_digits(target)?;
        let (best_idx, best_sim, second_sim) = Self::pick_best(&per_digit);

        if best_sim >= SIM_MIN && (best_sim - second_sim) >= SIM_MARGIN {
            Ok((Some(best_idx as u8), best_sim))
        } else {
            Ok((None, best_sim.max(0.0)))
        }
    }

    /// 匹配失败时把归一化后的 40x40 存盘，并打印前三名得分。
    ///
    /// 存下来的图尺寸、格式跟正式模板完全一致，直接改名成
    /// `<正确数字>_fix_<任意>_0.png` 丢进 src/templates/ 就能用。
    fn dump_failure(&self, tmpl40: &core::Mat, per_digit: &[f64; 10]) {
        let ts = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis())
            .unwrap_or(0);

        let mut ranked: Vec<(usize, f64)> =
            per_digit.iter().copied().enumerate().collect();
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let top: Vec<String> = ranked
            .iter()
            .take(3)
            .map(|(d, s)| format!("{}={:.3}", d, s))
            .collect();

        let path = format!("/tmp/tnt_fail_{}.png", ts);
        let _ = imgcodecs::imwrite(&path, tmpl40, &core::Vector::new());
        eprintln!("[ui] 匹配失败 -> {}  前三名: {}", path, top.join("  "));
    }

    /// 角度识别 (基于模板匹配)，返回 (角度, 最低一位的置信度)。
    ///
    /// 屏幕上的角度可以到三位数：反抛物线往身后打时会显示 91..=180。
    /// 这里统一折回 0..=90 再交给上层，102 -> 78、109 -> 71，
    /// 这样 tnt.rs / physics.rs 的角度上限不用动。
    pub fn recognize_angle_digit_conf(
        &self,
        angle_roi: &core::Mat,
    ) -> opencv::Result<(Option<i32>, f64)> {
        let (mask, gray) = self.binarize_and_clean(angle_roi)?;
        let digit_mats = self.extract_individual_digits(&mask, &gray, 1.55)?;

        if digit_mats.is_empty() {
            return Ok((None, 0.0));
        }

        // 角度最多三位。切出 4 块以上说明分割本身就错了（噪点或过分割）。
        if digit_mats.len() > 3 {
            return Ok((None, 0.0));
        }

        let mut digits: Vec<u8> = Vec::with_capacity(digit_mats.len());
        let mut worst_conf = 1.0f64;

        for mat in digit_mats {
            let tmpl40 = self.to_template_40(&mat)?;
            let per_digit = self.score_digits(&tmpl40)?;
            let (best_idx, best_sim, second_sim) = Self::pick_best(&per_digit);

            if best_sim < SIM_MIN || (best_sim - second_sim) < SIM_MARGIN {
                // 只要有一位没认出来，整帧作废，交给上层沿用上一帧。
                // 旧版是静默跳过这一位，剩下一位照样拼成角度返回，
                // 73 会变成 7 且能通过范围校验，变成一个"看起来正常"的错角度。
                if self.dump_fail {
                    self.dump_failure(&tmpl40, &per_digit);
                }
                return Ok((None, best_sim.max(0.0)));
            }

            digits.push(best_idx as u8);
            if best_sim < worst_conf {
                worst_conf = best_sim;
            }
        }

        // 三位数只可能是 1xx（角度封顶 180），首位不是 1 就是把两位数过分割了。
        // 放开位数限制之后，这条是挡住 92 被切成 9/2/杂点读成 923 的主要防线。
        if digits.len() == 3 && digits[0] != 1 {
            return Ok((None, worst_conf));
        }

        let val = digits.iter().fold(0i32, |acc, &d| acc * 10 + d as i32);

        // 超出 0..=180 就是识别错了，宁可返回 None。
        if !(0..=ANGLE_MAX).contains(&val) {
            return Ok((None, worst_conf));
        }

        // 反抛物线折返：屏幕显示 95 -> 实际按 85 算，102 -> 78。
        let val = if val > 90 { ANGLE_MAX - val } else { val };

        // 置信度取所有位里最低的那一位，避免"一位很准 + 一位勉强"被平均掩盖。
        Ok((Some(val), worst_conf))
    }

    /// 角度识别 (兼容旧签名)
    pub fn recognize_angle_digit(&self, angle_roi: &core::Mat) -> opencv::Result<Option<i32>> {
        Ok(self.recognize_angle_digit_conf(angle_roi)?.0)
    }

    /// 把 ROI 夹到画面范围内，防止分辨率异常时 Mat::roi 直接 panic。
    fn clamp_rect(r: Rect, cols: i32, rows: i32) -> Rect {
        let x = r.x.clamp(0, (cols - 1).max(0));
        let y = r.y.clamp(0, (rows - 1).max(0));
        let w = r.width.clamp(1, (cols - x).max(1));
        let h = r.height.clamp(1, (rows - y).max(1));
        Rect::new(x, y, w, h)
    }

    /// 一键解析截屏图像
    pub fn process_frame(&self, full_frame: &core::Mat, save_debug: bool) -> opencv::Result<RecognizerResult> {
        let cols = full_frame.cols();
        let rows = full_frame.rows();

        let minimap_rect = Self::clamp_rect(
            Rect::new(0, 0, (cols as f64 * 0.22) as i32, (rows as f64 * 0.22) as i32),
            cols,
            rows,
        );
        let angle_rect = Self::clamp_rect(
            Rect::new(
                (cols as f64 * 0.02) as i32,
                (rows as f64 * 0.85) as i32,
                (cols as f64 * 0.12) as i32,
                (rows as f64 * 0.12) as i32,
            ),
            cols,
            rows,
        );
        let power_rect = Self::clamp_rect(
            Rect::new(
                (cols as f64 * 0.15) as i32,
                (rows as f64 * 0.92) as i32,
                (cols as f64 * 0.70) as i32,
                (rows as f64 * 0.05) as i32,
            ),
            cols,
            rows,
        );
        let wind_rect = Self::clamp_rect(
            Rect::new(
                (cols as f64 * 0.40) as i32,
                (rows as f64 * 0.01) as i32,
                (cols as f64 * 0.20) as i32,
                (rows as f64 * 0.06) as i32,
            ),
            cols,
            rows,
        );

        let minimap = core::Mat::roi(full_frame, minimap_rect)?.try_clone()?;
        let angle_roi = core::Mat::roi(full_frame, angle_rect)?.try_clone()?;
        let power_roi = core::Mat::roi(full_frame, power_rect)?.try_clone()?;
        let wind_roi = core::Mat::roi(full_frame, wind_rect)?.try_clone()?;

        let (camera_rect, camera_width) = self.detect_minimap_fov(&minimap)?;
        let (player_dots, enemy_dots) = self.detect_minimap_dots(&minimap)?;
        let angle = self.recognize_angle_digit(&angle_roi)?;
        let power_percent = self.measure_power_bar(&power_roi)?;
        let wind_value = self.measure_wind_indicator(&wind_roi)?;

        if save_debug {
            let _ = imgcodecs::imwrite("debug_roi_1_minimap.png", &minimap, &core::Vector::new());
            let _ = imgcodecs::imwrite("debug_roi_3_angle.png", &angle_roi, &core::Vector::new());
            let _ = imgcodecs::imwrite("debug_roi_4_power.png", &power_roi, &core::Vector::new());
            let _ = imgcodecs::imwrite("debug_roi_5_wind.png", &wind_roi, &core::Vector::new());
        }

        Ok(RecognizerResult {
            camera_rect,
            camera_width,
            player_dots,
            enemy_dots,
            angle,
            power_percent,
            wind_value,
        })
    }
}
