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

/// 每多一个对不上的环（洞）扣多少分。
///
/// 拓扑是 ZNCC 看不到的维度：3 和 8 的像素分布重合很高，
/// 但"有几个封闭的环"跟字体、笔画粗细、模糊程度都无关：
/// 3/7/1/2/5 零个环，0/6/9 一个，8 两个。
const HOLE_PENALTY: f64 = 0.15;

/// 宽高比差异的扣分系数。专治 1↔7：1 又窄又高（约 0.3），7 明显宽（约 0.6）。
const ASPECT_PENALTY: f64 = 0.50;

/// "吸收过碎片"的数字的原始分门槛。
/// 拼出来的数字字形天然残一些，但残得不一样：被 ROI 裁掉再拼回去的
/// 窄 9 看着像个 0，原始分能到 0.7——比正常门限高得多。所以拼过的
/// 数字要拿更高的原始分才放行，认不出就整帧作废。
const SIM_MIN_ABSORBED: f64 = 0.75;

pub struct UiRecognizer {
    /// (数字, 去均值后的 40x40 展平像素, L2 范数, 洞数, 宽高比)
    /// 预展平 + 预去均值，是为了让热循环彻底不碰 OpenCV 的 at_2d。
    templates: Vec<(u8, Vec<f64>, f64, i32, f64)>,
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

    /// 字形的结构特征：(洞数, 墨迹宽高比)。
    ///
    /// 洞 = 反色后不接触图像边界的背景连通域。背景用 4 连通（前景 8 连通），
    /// 否则斜向缝隙会把环"漏"到外面去。面积小于 8 像素的不算环，挡噪点。
    fn glyph_stats(img: &core::Mat) -> opencv::Result<(i32, f64)> {
        if img.empty() || img.channels() != 1 {
            return Ok((0, 1.0));
        }

        let rows = img.rows();
        let cols = img.cols();

        let mut bin = core::Mat::default();
        imgproc::threshold(img, &mut bin, 30.0, 255.0, imgproc::THRESH_BINARY)?;

        let mut min_x = cols;
        let mut max_x = -1i32;
        let mut min_y = rows;
        let mut max_y = -1i32;
        for y in 0..rows {
            for x in 0..cols {
                if *bin.at_2d::<u8>(y, x)? > 0 {
                    if x < min_x { min_x = x; }
                    if x > max_x { max_x = x; }
                    if y < min_y { min_y = y; }
                    if y > max_y { max_y = y; }
                }
            }
        }
        if max_x < min_x || max_y < min_y {
            return Ok((0, 1.0));
        }
        let aspect = ((max_x - min_x + 1) as f64) / (((max_y - min_y + 1).max(1)) as f64);

        let mut inv = core::Mat::default();
        imgproc::threshold(&bin, &mut inv, 127.0, 255.0, imgproc::THRESH_BINARY_INV)?;

        let mut labels = core::Mat::default();
        let mut stats = core::Mat::default();
        let mut centroids = core::Mat::default();
        let n = imgproc::connected_components_with_stats(
            &inv,
            &mut labels,
            &mut stats,
            &mut centroids,
            4,
            core::CV_32S,
        )?;

        let mut holes = 0i32;
        for i in 1..n {
            let x = *stats.at_2d::<i32>(i, imgproc::CC_STAT_LEFT)?;
            let y = *stats.at_2d::<i32>(i, imgproc::CC_STAT_TOP)?;
            let w = *stats.at_2d::<i32>(i, imgproc::CC_STAT_WIDTH)?;
            let h = *stats.at_2d::<i32>(i, imgproc::CC_STAT_HEIGHT)?;
            let a = *stats.at_2d::<i32>(i, imgproc::CC_STAT_AREA)?;
            // 摸到边界的是外部背景，不是环。
            if x == 0 || y == 0 || x + w == cols || y + h == rows {
                continue;
            }
            if a < 8 {
                continue;
            }
            holes += 1;
        }

        Ok((holes, aspect))
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
                    let (holes, aspect) = Self::glyph_stats(variant)?;
                    if norm > 1e-6 {
                        templates.push((digit as u8, flat, norm, holes, aspect));
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

    /// 返回 (组件mask, tophat灰度, combo灰度)。
    /// combo = max(tophat, blackhat) ∩ 膨胀mask,只用于"吸收过碎片"的断笔数字。
    pub fn binarize_and_clean(&self, roi: &core::Mat) -> opencv::Result<(core::Mat, core::Mat, core::Mat)> {
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

        // 黑帽(亮底上的暗笔画)。数字压到白色圆盘/亮地形上时笔画是"暗"的,
        // tophat 看不见 -> 7 的斜笔在亮斑区域整段消失。识别用灰度取
        // max(tophat, blackhat),亮底暗笔画也能捞回来。
        // 注意:blackhat 只进灰度图,不进组件 mask——组件仍按亮笔画定位,
        // 否则暗色背景结构会炸出一堆假连通域。
        let mut blackhat = core::Mat::default();
        imgproc::morphology_ex(&up, &mut blackhat, imgproc::MORPH_BLACKHAT, &k, Point::new(-1, -1), 1, core::BORDER_REPLICATE, imgproc::morphology_default_border_value()?)?;
        let mut combo = core::Mat::default();
        core::max(&tophat, &blackhat, &mut combo)?;

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

            // 数字形连通域之外,把"疑似数字碎片"也保留下来:瞄准线/描边会把数字
            // 切成两半(实测 7 的横杠整条断开),碎片若在 final_mask 里被抹掉,
            // 后面 extract_individual_digits 的碎片吸收就没有素材可用。
            // 碎片本身不会被当成数字,只作为拼接候选参与行内校验。
            if Self::is_digit_component(w, h, area) || (area >= 30 && w <= 90 && h <= 45) {
                let mut comp_mask = core::Mat::default();
                core::compare(&labels, &Scalar::all(i as f64), &mut comp_mask, core::CMP_EQ)?;
                final_mask.set_to(&Scalar::all(255.0), &comp_mask)?;
            }
        }

        // 两份灰度:
        // - gray_out:tophat ∩ final_mask,给正常数字用,笔画干净利落
        // - combo_out:max(tophat,blackhat) ∩ 膨胀的 final_mask,只给"吸收过碎片"
        //   的断笔数字用——膨胀把连通域之间的断口(如被瞄准线切断的 7 斜笔上段、
        //   亮斑上的暗色笔段)圈进裁剪域。combo 不能给正常数字用:黑帽描边会让
        //   笔画变糊(实测干净的 9 会因此认成 5)。
        let mut gray_out = core::Mat::new_rows_cols_with_default(cleaned.rows(), cleaned.cols(), core::CV_8UC1, Scalar::all(0.0))?;
        tophat.copy_to_masked(&mut gray_out, &final_mask)?;

        let k_dil = imgproc::get_structuring_element(imgproc::MORPH_ELLIPSE, core::Size::new(11, 11), Point::new(-1, -1))?;
        let mut mask_dil = core::Mat::default();
        imgproc::dilate(&final_mask, &mut mask_dil, &k_dil, Point::new(-1, -1), 1, core::BORDER_CONSTANT, imgproc::morphology_default_border_value()?)?;
        let mut combo_out = core::Mat::new_rows_cols_with_default(cleaned.rows(), cleaned.cols(), core::CV_8UC1, Scalar::all(0.0))?;
        combo.copy_to_masked(&mut combo_out, &mask_dil)?;

        Ok((final_mask, gray_out, combo_out))
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

    /// 返回每位数字的灰度块和"是否吸收过碎片"标记。
    /// 吸收过碎片的数字是拼出来的,上层要用更严的置信度门限。
    pub fn extract_individual_digits(&self, mask: &core::Mat, gray: &core::Mat, gray_combo: &core::Mat, split_ratio: f64, allow_absorb: bool) -> opencv::Result<Vec<(core::Mat, bool)>> {
        let mut labels = core::Mat::default();
        let mut stats = core::Mat::default();
        let mut centroids = core::Mat::default();
        let num_labels = imgproc::connected_components_with_stats(mask, &mut labels, &mut stats, &mut centroids, 8, core::CV_32S)?;

        let mut valid_rects: Vec<(Rect, bool)> = Vec::new();
        let mut frag_rects = Vec::new();
        for i in 1..num_labels {
            let x = *stats.at_2d::<i32>(i, imgproc::CC_STAT_LEFT)?;
            let y = *stats.at_2d::<i32>(i, imgproc::CC_STAT_TOP)?;
            let w = *stats.at_2d::<i32>(i, imgproc::CC_STAT_WIDTH)?;
            let h = *stats.at_2d::<i32>(i, imgproc::CC_STAT_HEIGHT)?;
            let area = *stats.at_2d::<i32>(i, imgproc::CC_STAT_AREA)?;
            if Self::is_digit_component(w, h, area) {
                valid_rects.push((Rect::new(x, y, w, h), false));
            } else if allow_absorb && area >= 30 && w <= 90 && h <= 45 {
                // 碎片只在第二遍收集:第一遍保持和旧版一致的组件集合,
                // 否则同一行里的碎块会触发"残片否决"把好帧也毙掉。
                frag_rects.push(Rect::new(x, y, w, h));
            }
        }

        // 吸回数字碎片(仅 allow_absorb 时):瞄准线/描边会把一个数字切成
        // "数字形连通域 + 短条碎块"(实测 7 的横杠就是这么和斜笔断开的,
        // 横杠高不够直接被上面过滤丢弃)。碎块与某数字横向重叠过半、纵向间距
        // 在数字高度 45% 以内,就并回该数字的外接框。
        // 贴边碎片不吸——分不清是边缘杂斑还是被裁掉的数字残肢。
        // 只在第二遍启用:好端端的数字吸上杂块反而把字形搞脏。
        if allow_absorb {
            let cols = mask.cols();
            let rows = mask.rows();
            let at_edge = |r: &Rect| {
                r.x <= 1
                    || r.y <= 1
                    || r.x + r.width >= cols - 1
                    || r.y + r.height >= rows - 1
            };
            let mut used = vec![false; frag_rects.len()];
            for (fi, f) in frag_rects.iter().enumerate() {
                if at_edge(f) {
                    continue;
                }
                let mut best: Option<usize> = None;
                let mut best_ov = 0i32;
                for (i, (d, _)) in valid_rects.iter().enumerate() {
                    let x_ov = (d.x + d.width).min(f.x + f.width) - d.x.max(f.x);
                    let gap = (f.y - (d.y + d.height)).max(d.y - (f.y + f.height));
                    if x_ov * 2 > f.width
                        && x_ov > best_ov
                        && gap <= (d.height as f64 * 0.45) as i32
                        && f.width <= d.width * 2
                        && f.height <= d.height
                    {
                        best = Some(i);
                        best_ov = x_ov;
                    }
                }
                if let Some(i) = best {
                    let d = valid_rects[i].0;
                    let nx = d.x.min(f.x);
                    let ny = d.y.min(f.y);
                    valid_rects[i] = (
                        Rect::new(
                            nx,
                            ny,
                            (d.x + d.width).max(f.x + f.width) - nx,
                            (d.y + d.height).max(f.y + f.height) - ny,
                        ),
                        true,
                    );
                    used[fi] = true;
                }
            }
            frag_rects = frag_rects
                .iter()
                .zip(used.iter())
                .filter(|&(_, u)| !u)
                .map(|(r, _)| *r)
                .collect();
        }

        // 同一串角度数字:高度一致、纵向位置同一行。杂斑(圆盘装饰/箭头/徽章碎片)
        // 高矮和中线都对不上,按中位高度 + 中线偏移把离群的剔掉,再排序。
        // 杂斑比数字多时中位数落在杂斑上 -> 剔光 -> 空 -> 上层沿用上一帧,不读错。
        if !valid_rects.is_empty() {
            let mut hs: Vec<i32> = valid_rects.iter().map(|r| r.0.height).collect();
            hs.sort();
            let med_h = hs[hs.len() / 2] as f64;
            let mut cys: Vec<f64> = valid_rects
                .iter()
                .map(|r| r.0.y as f64 + r.0.height as f64 / 2.0)
                .collect();
            cys.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let med_cy = cys[cys.len() / 2];
            let (kept, dropped): (Vec<(Rect, bool)>, Vec<(Rect, bool)>) =
                valid_rects.into_iter().partition(|r| {
                    let h = r.0.height as f64;
                    let cy = r.0.y as f64 + h / 2.0;
                    h >= med_h * 0.7
                        && h <= med_h * 1.4
                        && (cy - med_cy).abs() <= med_h * 0.35
                });

            if !kept.is_empty() {
                let band_top = kept.iter().map(|r| r.0.y).min().unwrap();
                let band_bot = kept.iter().map(|r| r.0.y + r.0.height).max().unwrap();

                // FIX: 两道防"裁半个字"的闸。
                // 1) 留下的数字里有贴边的 -> 多半被框选裁掉一半(实测 "4" 剩半边
                //    认成 "1"、"9" 缺角认成 "0"),读出来也是能过校验的错角度。
                // 2) 被剔掉的东西若还挤在数字行里(纵向重叠超自身一半),多半
                //    是裁残的碎片——亮核不一定贴边,闸 1 抓不到。
                // 两种情况都宁可整帧作废沿用上一帧稳定值,也不读半个字。
                // 注意只能查"同一行":别的行的贴边杂斑(比如左上徽章角)
                // 已被聚类剔掉,不该连累整帧。
                let cols = mask.cols();
                let rows = mask.rows();
                if kept.iter().any(|r| {
                    r.0.x <= 1
                        || r.0.y <= 1
                        || r.0.x + r.0.width >= cols - 1
                        || r.0.y + r.0.height >= rows - 1
                }) || dropped
                    .iter()
                    .map(|r| &r.0)
                    .chain(frag_rects.iter())
                    .any(|r| {
                        let ov = (r.y + r.height).min(band_bot) - r.y.max(band_top);
                        ov * 2 > r.height
                    })
                {
                    return Ok(Vec::new());
                }
            }
            valid_rects = kept;
        }

        valid_rects.sort_by_key(|r| r.0.x);

        let k_erode = imgproc::get_structuring_element(imgproc::MORPH_RECT, core::Size::new(2, 2), core::Point::new(-1, -1))?;
        let mut eroded_mask = core::Mat::default();
        imgproc::erode(mask, &mut eroded_mask, &k_erode, core::Point::new(-1, -1), 1, core::BORDER_CONSTANT, imgproc::morphology_default_border_value()?)?;

        let mut final_rects: Vec<(Rect, bool)> = Vec::new();
        for (r, absorbed) in valid_rects {
            let expect_w = (((r.height as f64) * 0.62).round() as i32).max(6);
            if r.width > (expect_w as f64 * split_ratio) as i32 {
                let num_parts = ((r.width as f64) / (expect_w as f64)).round().max(2.0) as i32;
                final_rects.extend(
                    self.split_by_valley(&eroded_mask, r, num_parts)?
                        .into_iter()
                        .map(|sr| (sr, absorbed)),
                );
            } else {
                final_rects.push((r, absorbed));
            }
        }

        let mut digit_mats = Vec::new();
        for (rect, absorbed) in final_rects {
            // 拼过的数字从 combo 灰度里取——黑帽能把亮底上的暗笔段捞回来;
            // 正常数字保持 tophat 灰度,避免描边晕墨把字形弄糊。
            let src = if absorbed { gray_combo } else { gray };
            let digit_roi = core::Mat::roi(src, rect)?;
            digit_mats.push((digit_roi.try_clone()?, absorbed));
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

    /// 调试入口：返回每个数字的 (修正分, 原始 ZNCC) 供 cli_recognize 打印前三名。
    pub fn score_digits_debug(
        &self,
        target: &core::Mat,
    ) -> opencv::Result<Vec<(u8, f64, f64)>> {
        let (adj, raw) = self.score_digits(target)?;
        let mut v: Vec<(u8, f64, f64)> =
            (0..10u8).map(|d| (d, adj[d as usize], raw[d as usize])).collect();
        v.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        Ok(v)
    }

    /// 算出 0..9 每个数字的得分，返回 (修正后得分, 对应的原始 ZNCC 得分)。
    ///
    /// 用 ZNCC（先减各自均值）而不是裸余弦：灰度像素恒为非负，
    /// 裸余弦会严重依赖前景像素的多少，稀疏的 3、7 分数被系统性压低。
    ///
    /// 修正项（拓扑 + 宽高比）只用来排名次，不用来判生死：
    /// SIM_MIN 那道门槛看的是原始分，否则扣分会把本来能认出来的字扣成废帧。
    ///
    /// 洞数只扣单向（模板比目标多才扣）：真实帧里 3 的笔画太粗可能糊出一个假环，
    /// 反向也扣的话假环会反咬 3 一口。
    ///
    /// PERF: 模板已在 new() 里预展平、预去均值、预算结构特征。
    fn score_digits(&self, target: &core::Mat) -> opencv::Result<([f64; 10], [f64; 10])> {
        let mut adj = [-2.0f64; 10];
        let mut raw_at_best = [0.0f64; 10];
        if self.templates.is_empty() {
            return Ok((adj, raw_at_best));
        }

        let (t_holes, t_aspect) = Self::glyph_stats(target)?;

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

                for (digit, tmpl, tmpl_norm, tmpl_holes, tmpl_aspect) in &self.templates {
                    let mut dot = 0f64;
                    for i in 0..(40 * 40) {
                        dot += shifted[i] * tmpl[i];
                    }
                    let sim = dot / (norm * tmpl_norm);

                    let hole_gap = (*tmpl_holes - t_holes).max(0) as f64;
                    let penalty = HOLE_PENALTY * hole_gap
                        + ASPECT_PENALTY * (t_aspect - *tmpl_aspect).abs();
                    let score = sim - penalty;

                    let d = *digit as usize;
                    if score > adj[d] {
                        adj[d] = score;
                        raw_at_best[d] = sim;
                    }
                }
            }
        }

        Ok((adj, raw_at_best))
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
        let (adj, raw_at_best) = self.score_digits(target)?;
        let (best_idx, best_adj, second_adj) = Self::pick_best(&adj);
        let best_raw = raw_at_best[best_idx];

        if best_raw >= SIM_MIN && (best_adj - second_adj) >= SIM_MARGIN {
            Ok((Some(best_idx as u8), best_raw))
        } else {
            Ok((None, best_raw.max(0.0)))
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
        let (mask, gray, gray_combo) = self.binarize_and_clean(angle_roi)?;

        // 第一遍:不做碎片吸收,组件集合与判分门限都和旧版一致。
        // 好端端的帧不会受到碎片逻辑的任何影响。
        let mats = self.extract_individual_digits(&mask, &gray, &gray_combo, 1.55, false)?;
        let first = self.score_digit_mats(mats)?;
        if first.0.is_some() {
            return Ok(first);
        }

        // 第二遍:碎片吸收修复断笔(瞄准线/亮斑把数字切断的场景)。
        // 拼过的位要走更严的 SIM_MIN_ABSORBED 门限,防"裁残的 9 拼回去读成 0"。
        let mats2 = self.extract_individual_digits(&mask, &gray, &gray_combo, 1.55, true)?;
        let second = self.score_digit_mats(mats2)?;
        if second.0.is_some() {
            return Ok(second);
        }
        Ok((None, first.1.max(second.1)))
    }

    /// 对一组数字灰度块逐个模板匹配,拼成角度值。
    /// 只要有一位认不出来整帧作废——旧版静默跳过该位,73 会变成 7 还合法。
    fn score_digit_mats(
        &self,
        digit_mats: Vec<(core::Mat, bool)>,
    ) -> opencv::Result<(Option<i32>, f64)> {
        if digit_mats.is_empty() {
            return Ok((None, 0.0));
        }

        // 角度最多三位。切出 4 块以上说明分割本身就错了（噪点或过分割）。
        if digit_mats.len() > 3 {
            return Ok((None, 0.0));
        }

        let mut digits: Vec<u8> = Vec::with_capacity(digit_mats.len());
        let mut worst_conf = 1.0f64;

        for (mat, absorbed) in digit_mats {
            let tmpl40 = self.to_template_40(&mat)?;
            let (adj, raw_at_best) = self.score_digits(&tmpl40)?;
            let (best_idx, best_adj, second_adj) = Self::pick_best(&adj);
            let best_raw = raw_at_best[best_idx];

            let min_raw = if absorbed { SIM_MIN_ABSORBED } else { SIM_MIN };
            if best_raw < min_raw || (best_adj - second_adj) < SIM_MARGIN {
                // 只要有一位没认出来，整帧作废，交给上层沿用上一帧。
                // 旧版是静默跳过这一位，剩下一位照样拼成角度返回，
                // 73 会变成 7 且能通过范围校验，变成一个"看起来正常"的错角度。
                if self.dump_fail {
                    self.dump_failure(&tmpl40, &adj);
                }
                return Ok((None, best_raw.max(0.0)));
            }

            digits.push(best_idx as u8);
            if best_raw < worst_conf {
                worst_conf = best_raw;
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

        // 反抛物线折返：大于 90 才折，95 -> 85，102 -> 78；小于等于 90 原样不动。
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
