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

/// 模板匹配的判定阈值。
///
/// 注意：这里用的是 ZNCC（零均值归一化互相关），取值范围 -1..=1，
/// 和旧版"未去均值的余弦相似度"完全不是一个量纲。
/// 旧版因为像素值全为非负，任意两个数字的相似度都在 0.8 以上，
/// 0.85 这个阈值实际上只是在卡"笔画密度"，稀疏的 3 和 7 天然吃亏。
const SIM_MIN: f64 = 0.60;
/// 最佳候选必须比"第二名的其他数字"高出这么多，否则视为不可信。
const SIM_MARGIN: f64 = 0.02;

pub struct UiRecognizer {
    /// (数字, 去均值后的 40x40 展平像素, 去均值后的 L2 范数)
    /// 预展平 + 预去均值，是为了让 match_digit 的热循环彻底不碰 OpenCV 的 at_2d。
    templates: Vec<(u8, Vec<f64>, f64)>,
}

impl UiRecognizer {
    /// 初始化识别器
    pub fn new<P: AsRef<Path>>(template_dir: P) -> opencv::Result<Self> {
        let mut templates = Vec::new();
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
                // 模板必须是 40x40，否则下面的展平就对不上了
                if img.empty() || img.rows() != 40 || img.cols() != 40 {
                    continue;
                }

                let mut flat = vec![0f64; 40 * 40];
                let mut sum = 0f64;
                for y in 0..40 {
                    for x in 0..40 {
                        let v = *img.at_2d::<u8>(y, x)? as f64;
                        flat[(y * 40 + x) as usize] = v;
                        sum += v;
                    }
                }
                // 去均值：这样相似度只看"形状"，不看"有多少前景像素"。
                let mean = sum / (40.0 * 40.0);
                let mut norm_sq = 0f64;
                for v in flat.iter_mut() {
                    *v -= mean;
                    norm_sq += *v * *v;
                }
                templates.push((digit as u8, flat, norm_sq.sqrt()));
            }
        }
        Ok(Self { templates })
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
    /// 笔画稀疏的数字非常不友好——同样的字高，7 只有两笔、3 是三段弧，
    /// 前景像素数可能只有 8 的一半，抗锯齿再吃掉一点就直接被丢了，
    /// 表现就是"两位数少识别一位"。
    /// 现在改成跟字高挂钩的相对门槛：高度已经卡住 h >= 18 了，
    /// 噪点根本长不到这么高，面积门槛只需要排掉细长的划痕。
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

        // FIX: 前景判定从 >50 降到 >30，与 binarize_and_clean 里 Otsu 的
        // 兜底阈值保持一致。原来 3 和 7 的细笔画末端（tophat 响应本来就弱）
        // 会被排除在外接框之外，导致居中和缩放都偏掉。
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

    /// 模板匹配单个数字，返回 (数字, ZNCC 相似度)。
    ///
    /// FIX（3/7 漏识别的主因）：旧版算的是"未去均值的余弦相似度"。
    /// 因为灰度像素恒为非负，这个量对任意两张图都偏高，而且严重依赖
    /// 前景像素的多少：稀疏的目标（3、7）跟稠密的模板（8、0）比，
    /// 分子只算重叠、分母却是全图范数，分数被系统性压低，
    /// 于是同一个 0.85 阈值对 8 很宽松、对 7 几乎不可能通过。
    /// 改成 ZNCC（先减各自均值）后，相似度只反映形状，笔画密度不再影响判定。
    ///
    /// PERF: 模板已在 new() 里预展平并预去均值，热循环完全不碰 at_2d。
    /// to_template_40 已按外接框居中，平移搜索保持 ±1。
    fn match_digit(&self, target: &core::Mat) -> opencv::Result<(Option<u8>, f64)> {
        if self.templates.is_empty() {
            return Ok((None, 0.0));
        }

        let mut raw = vec![0f64; 40 * 40];
        for y in 0..40 {
            for x in 0..40 {
                raw[(y * 40 + x) as usize] = *target.at_2d::<u8>(y, x)? as f64;
            }
        }

        // 每个数字各自的最佳得分，便于后面算"和第二名的差距"。
        let mut per_digit = [-2.0f64; 10];
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
                    if *tmpl_norm <= 1e-6 {
                        continue;
                    }
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

        if best_sim >= SIM_MIN && (best_sim - second_sim) >= SIM_MARGIN {
            Ok((Some(best_idx as u8), best_sim))
        } else {
            Ok((None, best_sim.max(0.0)))
        }
    }

    /// 角度识别 (基于模板匹配)，返回 (角度, 平均置信度)。
    pub fn recognize_angle_digit_conf(
        &self,
        angle_roi: &core::Mat,
    ) -> opencv::Result<(Option<i32>, f64)> {
        let (mask, gray) = self.binarize_and_clean(angle_roi)?;
        let digit_mats = self.extract_individual_digits(&mask, &gray, 1.55)?;

        if digit_mats.is_empty() {
            return Ok((None, 0.0));
        }

        // FIX: 角度最多两位。切出 3 块以上说明分割本身就错了（噪点或过分割），
        // 旧版是 truncate(2) 硬取前两块，等于用错误的碎片拼一个看似合法的角度。
        if digit_mats.len() > 2 {
            return Ok((None, 0.0));
        }

        let expected = digit_mats.len();
        let mut digits: Vec<u8> = Vec::with_capacity(expected);
        let mut conf_sum = 0.0f64;
        let mut worst_conf = 1.0f64;

        for mat in digit_mats {
            let tmpl40 = self.to_template_40(&mat)?;
            let (d, c) = self.match_digit(&tmpl40)?;
            match d {
                Some(d) => {
                    digits.push(d);
                    conf_sum += c;
                    if c < worst_conf {
                        worst_conf = c;
                    }
                }
                None => {
                    // FIX（"少识别一个数字"的直接原因）：旧版是 `if let Some(d)`，
                    // 匹配失败的那一位被静默跳过，剩下一位照样拼成角度返回。
                    // 73 里的 3 没匹配上就返回 7，而 7 落在 0..=90 内，
                    // 范围校验也拦不住，结果就是一个"看起来正常"的错误角度。
                    // 现在只要有一位没认出来，整帧作废，交给上层沿用上一帧。
                    return Ok((None, c));
                }
            }
        }

        let val = digits.iter().fold(0i32, |acc, &d| acc * 10 + d as i32);
        let conf = conf_sum / digits.len() as f64;

        // 角度物理上只可能落在 0..=90，超出就是识别错了，宁可返回 None。
        if !(0..=90).contains(&val) {
            return Ok((None, conf));
        }
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
