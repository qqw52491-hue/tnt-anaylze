use opencv::{
    core::{self, Point, Rect, Scalar},
    imgcodecs, imgproc,
    prelude::*,
};
use std::path::Path;

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

pub struct UiRecognizer;

impl UiRecognizer {
    /// 初始化识别器
    pub fn new<P: AsRef<Path>>(_template_dir: P) -> opencv::Result<Self> {
        Ok(Self)
    }

    /// 1. 小地图·视野框识别: HSV 黄色掩码 + boundingRect 取最大矩形
    pub fn detect_minimap_fov(&self, minimap: &core::Mat) -> opencv::Result<(Option<Rect>, f64)> {
        let mut hsv = core::Mat::default();
        imgproc::cvt_color(minimap, &mut hsv, imgproc::COLOR_BGR2HSV, 0, core::AlgorithmHint::ALGO_HINT_DEFAULT)?;

        let lower_yellow_green = Scalar::new(15.0, 60.0, 60.0, 0.0);
        let upper_yellow_green = Scalar::new(85.0, 255.0, 255.0, 0.0);

        let mut mask = core::Mat::default();
        core::in_range(&hsv, &lower_yellow_green, &upper_yellow_green, &mut mask)?;

        let mut contours = core::Vector::<core::Vector<Point>>::new();
        imgproc::find_contours(&mask, &mut contours, imgproc::RETR_EXTERNAL, imgproc::CHAIN_APPROX_SIMPLE, Point::new(0, 0))?;

        let mut best_rect: Option<Rect> = None;
        let mut max_area = 0;

        for i in 0..contours.len() {
            let contour = contours.get(i)?;
            let rect = opencv::geometry::bounding_rect(&contour)?;
            let area = rect.width * rect.height;
            if rect.width >= 40 && rect.width <= 226 && area > max_area {
                max_area = area;
                best_rect = Some(rect);
            }
        }

        let camera_width = best_rect.map(|r| r.width as f64).unwrap_or(170.0);
        Ok((best_rect, camera_width))
    }

    /// 2. 小地图·点点识别: HSV 按色相分类 (己方蓝 / 敌方红) + 面积 3~60px 过滤
    pub fn detect_minimap_dots(&self, minimap: &core::Mat) -> opencv::Result<(Vec<Point>, Vec<Point>)> {
        let mut hsv = core::Mat::default();
        imgproc::cvt_color(minimap, &mut hsv, imgproc::COLOR_BGR2HSV, 0, core::AlgorithmHint::ALGO_HINT_DEFAULT)?;

        let lower_blue = Scalar::new(80.0, 50.0, 50.0, 0.0);
        let upper_blue = Scalar::new(140.0, 255.0, 255.0, 0.0);
        let mut mask_blue = core::Mat::default();
        core::in_range(&hsv, &lower_blue, &upper_blue, &mut mask_blue)?;

        let lower_red1 = Scalar::new(0.0, 50.0, 50.0, 0.0);
        let upper_red1 = Scalar::new(10.0, 255.0, 255.0, 0.0);
        let lower_red2 = Scalar::new(160.0, 50.0, 50.0, 0.0);
        let upper_red2 = Scalar::new(180.0, 255.0, 255.0, 0.0);
        let mut mask_red1 = core::Mat::default();
        let mut mask_red2 = core::Mat::default();
        let mut mask_red = core::Mat::default();
        core::in_range(&hsv, &lower_red1, &upper_red1, &mut mask_red1)?;
        core::in_range(&hsv, &lower_red2, &upper_red2, &mut mask_red2)?;
        core::bitwise_or(&mask_red1, &mask_red2, &mut mask_red, &core::no_array())?;

        let parse_dots = |mask: &core::Mat| -> opencv::Result<Vec<Point>> {
            let mut contours = core::Vector::<core::Vector<Point>>::new();
            imgproc::find_contours(mask, &mut contours, imgproc::RETR_EXTERNAL, imgproc::CHAIN_APPROX_SIMPLE, Point::new(0, 0))?;
            let mut pts = Vec::new();
            for i in 0..contours.len() {
                let contour = contours.get(i)?;
                let rect = opencv::geometry::bounding_rect(&contour)?;
                let area = rect.width * rect.height;
                if area >= 3 && area <= 60 && rect.width <= 15 && rect.height <= 15 {
                    pts.push(Point::new(rect.x + rect.width / 2, rect.y + rect.height / 2));
                }
            }
            pts.sort_by_key(|p| p.x);
            Ok(pts)
        };

        let player_pts = parse_dots(&mask_blue)?;
        let enemy_pts = parse_dots(&mask_red)?;
        Ok((player_pts, enemy_pts))
    }

    /// 4. 力度条识别: 底部力度条填充像素长度 ÷ 总长度 (不用 OCR)
    pub fn measure_power_bar(&self, power_bar_roi: &core::Mat) -> opencv::Result<f64> {
        let cols = power_bar_roi.cols();
        let rows = power_bar_roi.rows();
        if cols == 0 || rows == 0 {
            return Ok(0.0);
        }

        let mut hsv = core::Mat::default();
        imgproc::cvt_color(power_bar_roi, &mut hsv, imgproc::COLOR_BGR2HSV, 0, core::AlgorithmHint::ALGO_HINT_DEFAULT)?;

        let lower_red1 = Scalar::new(0.0, 100.0, 100.0, 0.0);
        let upper_red1 = Scalar::new(10.0, 255.0, 255.0, 0.0);
        let lower_red2 = Scalar::new(160.0, 100.0, 100.0, 0.0);
        let upper_red2 = Scalar::new(180.0, 100.0, 100.0, 0.0);

        let mut mask1 = core::Mat::default();
        let mut mask2 = core::Mat::default();
        let mut mask = core::Mat::default();
        core::in_range(&hsv, &lower_red1, &upper_red1, &mut mask1)?;
        core::in_range(&hsv, &lower_red2, &upper_red2, &mut mask2)?;
        core::bitwise_or(&mask1, &mask2, &mut mask, &core::no_array())?;

        let mut filled_end_x = 0;
        for x in 0..cols {
            let mut col_has_filled = false;
            for y in 0..rows {
                if *mask.at_2d::<u8>(y, x)? > 0 {
                    col_has_filled = true;
                    break;
                }
            }
            if col_has_filled {
                filled_end_x = x;
            }
        }

        let percent = (filled_end_x as f64 / cols as f64) * 100.0;
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
        imgproc::cvt_color(wind_roi, &mut hsv, imgproc::COLOR_BGR2HSV, 0, core::AlgorithmHint::ALGO_HINT_DEFAULT)?;

        let lower = Scalar::new(0.0, 0.0, 180.0, 0.0);
        let upper = Scalar::new(180.0, 255.0, 255.0, 0.0);
        let mut mask = core::Mat::default();
        core::in_range(&hsv, &lower, &upper, &mut mask)?;

        let mut contours = core::Vector::<core::Vector<Point>>::new();
        imgproc::find_contours(&mask, &mut contours, imgproc::RETR_EXTERNAL, imgproc::CHAIN_APPROX_SIMPLE, Point::new(0, 0))?;

        let mut best_arrow_x = center_x;
        let mut max_area = 0;

        for i in 0..contours.len() {
            let contour = contours.get(i)?;
            let rect = opencv::geometry::bounding_rect(&contour)?;
            let area = rect.width * rect.height;
            if area > max_area {
                max_area = area;
                best_arrow_x = rect.x as f64 + rect.width as f64 / 2.0;
            }
        }

        let offset_px = best_arrow_x - center_x;
        let wind_val = (offset_px / (cols as f64 / 2.0)) * 10.0;
        Ok(wind_val)
    }

    /// 角度识别 (已停用 OCR)
    pub fn recognize_angle_digit(&self, _angle_roi: &core::Mat) -> opencv::Result<Option<i32>> {
        Ok(None)
    }

    /// 一键解析截屏图像
    pub fn process_frame(&self, full_frame: &core::Mat, save_debug: bool) -> opencv::Result<RecognizerResult> {
        let cols = full_frame.cols();
        let rows = full_frame.rows();

        let minimap_rect = Rect::new(0, 0, (cols as f64 * 0.22) as i32, (rows as f64 * 0.22) as i32);
        let angle_rect = Rect::new(
            (cols as f64 * 0.02) as i32,
            (rows as f64 * 0.85) as i32,
            (cols as f64 * 0.12) as i32,
            (rows as f64 * 0.12) as i32,
        );
        let power_rect = Rect::new(
            (cols as f64 * 0.15) as i32,
            (rows as f64 * 0.92) as i32,
            (cols as f64 * 0.70) as i32,
            (rows as f64 * 0.05) as i32,
        );
        let wind_rect = Rect::new(
            (cols as f64 * 0.40) as i32,
            (rows as f64 * 0.01) as i32,
            (cols as f64 * 0.20) as i32,
            (rows as f64 * 0.06) as i32,
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
