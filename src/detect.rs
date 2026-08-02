//! 统一识别层：小地图红/蓝点检测 + 视野框 + 多帧跟踪
//!
//! ⚠️ 本文件不含任何弹道 / 风力 / 力度公式。physics.rs、tnt.rs 未改动一行。
//!
//! 解决的问题：
//!   1. main.rs / ui.rs / live_gui.rs 三套不一致的阈值 -> 统一到这里一份
//!   2. 只用 boundingRect 面积判定 -> 连通域 + 填充率 + 黑边 + 圆度加权打分
//!   3. 逐帧独立识别导致闪烁/跳点 -> Tracker 最近邻关联 + EMA 平滑 + 丢失容忍
//!   4. 帧差(呼吸灯)从「唯一依据」降级为「置信度加成」，避免爆炸特效抢点

use opencv::{
    core::{self, Point, Rect, Scalar},
    imgproc,
    prelude::*,
};

#[derive(Clone, Copy, Debug)]
pub struct Dot {
    pub pt: Point,
    pub score: f64,
    pub area: i32,
}

#[derive(Clone, Copy, Debug)]
pub struct DotParams {
    pub min_area: i32,
    pub max_area: i32,
    pub max_side: i32,
    pub min_aspect: f64,
    pub max_aspect: f64,
    pub min_fill: f64,
    pub min_dark_ratio: f64,
    pub border_margin: i32,
    pub nms_dist: f64,
    pub min_sat: f64,
    pub min_val: f64,
}

impl Default for DotParams {
    fn default() -> Self {
        Self {
            min_area: 6,
            max_area: 90,
            max_side: 12,
            min_aspect: 0.45,
            max_aspect: 2.2,
            min_fill: 0.45,
            min_dark_ratio: 0.04,
            border_margin: 2,
            nms_dist: 6.0,
            min_sat: 70.0,
            min_val: 70.0,
        }
    }
}

impl DotParams {
    /// 按小地图实际宽度自动缩放阈值（基线 226px）。
    /// 换分辨率 / 换显示器时不用再手改魔法数字。
    pub fn for_width(width: i32) -> Self {
        let s = (width as f64 / 226.0).max(0.35);
        let a = s * s;
        let mut p = Self::default();
        p.min_area = ((p.min_area as f64) * a).round().max(3.0) as i32;
        p.max_area = ((p.max_area as f64) * a).round().max(20.0) as i32;
        p.max_side = ((p.max_side as f64) * s).round().max(5.0) as i32;
        p.nms_dist = (p.nms_dist * s).max(3.0);
        p.border_margin = ((p.border_margin as f64) * s).round().max(1.0) as i32;
        p
    }
}

/// HSV 颜色掩码。红色跨 0/180，分两段再 or。
pub fn color_mask(minimap: &core::Mat, is_red: bool, p: &DotParams) -> opencv::Result<core::Mat> {
    let mut hsv = core::Mat::default();
    imgproc::cvt_color(
        minimap,
        &mut hsv,
        imgproc::COLOR_BGR2HSV,
        0,
        core::AlgorithmHint::ALGO_HINT_DEFAULT,
    )?;

    let mut mask = core::Mat::default();
    if is_red {
        let mut m1 = core::Mat::default();
        let mut m2 = core::Mat::default();
        core::in_range(
            &hsv,
            &Scalar::new(0.0, p.min_sat, p.min_val, 0.0),
            &Scalar::new(10.0, 255.0, 255.0, 0.0),
            &mut m1,
        )?;
        core::in_range(
            &hsv,
            &Scalar::new(165.0, p.min_sat, p.min_val, 0.0),
            &Scalar::new(180.0, 255.0, 255.0, 0.0),
            &mut m2,
        )?;
        core::bitwise_or(&m1, &m2, &mut mask, &core::no_array())?;
    } else {
        core::in_range(
            &hsv,
            &Scalar::new(85.0, p.min_sat, p.min_val, 0.0),
            &Scalar::new(140.0, 255.0, 255.0, 0.0),
            &mut mask,
        )?;
    }

    // 闭运算：把被黑色描边切开的点重新连成一块
    let k = imgproc::get_structuring_element(
        imgproc::MORPH_ELLIPSE,
        core::Size::new(3, 3),
        Point::new(-1, -1),
    )?;
    let mut closed = core::Mat::default();
    imgproc::morphology_ex(
        &mask,
        &mut closed,
        imgproc::MORPH_CLOSE,
        &k,
        Point::new(-1, -1),
        1,
        core::BORDER_CONSTANT,
        imgproc::morphology_default_border_value()?,
    )?;
    Ok(closed)
}

/// 目标外框一圈的暗色像素占比。游戏里的玩家点都有黑描边，地图色块没有。
fn dark_ratio(minimap: &core::Mat, r: Rect, margin: i32) -> opencv::Result<f64> {
    if minimap.channels() != 3 {
        return Ok(1.0);
    }
    let sx = (r.x - margin).max(0);
    let sy = (r.y - margin).max(0);
    let ex = (r.x + r.width + margin).min(minimap.cols() - 1);
    let ey = (r.y + r.height + margin).min(minimap.rows() - 1);

    let mut dark = 0i32;
    let mut total = 0i32;
    for y in sy..=ey {
        for x in sx..=ex {
            if x >= r.x && x < r.x + r.width && y >= r.y && y < r.y + r.height {
                continue;
            }
            total += 1;
            let px = minimap.at_2d::<core::Vec3b>(y, x)?;
            if (px[0] as i32) < 110 && (px[1] as i32) < 110 && (px[2] as i32) < 110 {
                dark += 1;
            }
        }
    }
    Ok(if total > 0 { dark as f64 / total as f64 } else { 0.0 })
}

fn nms(dots: &mut Vec<Dot>, min_dist: f64) {
    dots.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
    let mut kept: Vec<Dot> = Vec::new();
    for d in dots.iter() {
        let mut dup = false;
        for k in kept.iter() {
            let dx = (k.pt.x - d.pt.x) as f64;
            let dy = (k.pt.y - d.pt.y) as f64;
            if (dx * dx + dy * dy).sqrt() < min_dist {
                dup = true;
                break;
            }
        }
        if !dup {
            kept.push(*d);
        }
    }
    *dots = kept;
}

/// 从掩码里提点。返回按 score 降序排列。
pub fn dots_from_mask(
    minimap: &core::Mat,
    mask: &core::Mat,
    p: &DotParams,
) -> opencv::Result<Vec<Dot>> {
    let mut labels = core::Mat::default();
    let mut stats = core::Mat::default();
    let mut centroids = core::Mat::default();
    let n = imgproc::connected_components_with_stats(
        mask,
        &mut labels,
        &mut stats,
        &mut centroids,
        8,
        core::CV_32S,
    )?;

    let mut out: Vec<Dot> = Vec::new();
    for i in 1..n {
        let x = *stats.at_2d::<i32>(i, imgproc::CC_STAT_LEFT)?;
        let y = *stats.at_2d::<i32>(i, imgproc::CC_STAT_TOP)?;
        let w = *stats.at_2d::<i32>(i, imgproc::CC_STAT_WIDTH)?;
        let h = *stats.at_2d::<i32>(i, imgproc::CC_STAT_HEIGHT)?;
        let px = *stats.at_2d::<i32>(i, imgproc::CC_STAT_AREA)?;

        if w > p.max_side || h > p.max_side {
            continue;
        }
        if px < p.min_area || px > p.max_area {
            continue;
        }
        let aspect = w as f64 / (h.max(1)) as f64;
        if aspect < p.min_aspect || aspect > p.max_aspect {
            continue;
        }
        // 填充率：真圆点接近 0.7+，长条 UI / 地图色块很低
        let fill = px as f64 / ((w * h).max(1)) as f64;
        if fill < p.min_fill {
            continue;
        }
        // 贴边的一律不要（小地图边框、UI 按钮）
        if x <= p.border_margin
            || y <= p.border_margin
            || x + w >= minimap.cols() - p.border_margin
            || y + h >= minimap.rows() - p.border_margin
        {
            continue;
        }
        let dr = dark_ratio(minimap, Rect::new(x, y, w, h), 1)?;
        if dr < p.min_dark_ratio {
            continue;
        }

        let cx = *centroids.at_2d::<f64>(i, 0)?;
        let cy = *centroids.at_2d::<f64>(i, 1)?;
        let roundness = 1.0 - (aspect - 1.0).abs().min(1.0);
        let score = 0.45 * fill + 0.35 * (dr.min(0.5) / 0.5) + 0.20 * roundness;

        out.push(Dot {
            pt: Point::new(cx.round() as i32, cy.round() as i32),
            score,
            area: px,
        });
    }

    nms(&mut out, p.nms_dist);
    Ok(out)
}

/// 一步到位：颜色 -> 掩码 -> 点，按置信度降序。
pub fn detect_dots(minimap: &core::Mat, is_red: bool, p: &DotParams) -> opencv::Result<Vec<Dot>> {
    let mask = color_mask(minimap, is_red, p)?;
    dots_from_mask(minimap, &mask, p)
}

/// 呼吸灯帧差掩码。只作为加分项，不作为唯一依据。
pub fn breathing_mask(prev: &core::Mat, curr: &core::Mat, thresh: f64) -> opencv::Result<core::Mat> {
    if prev.empty() || curr.empty() || prev.size()? != curr.size()? {
        return Ok(core::Mat::default());
    }
    let mut diff = core::Mat::default();
    core::absdiff(prev, curr, &mut diff)?;

    let mut gray = core::Mat::default();
    imgproc::cvt_color(
        &diff,
        &mut gray,
        imgproc::COLOR_BGR2GRAY,
        0,
        core::AlgorithmHint::ALGO_HINT_DEFAULT,
    )?;

    let mut mask = core::Mat::default();
    imgproc::threshold(&gray, &mut mask, thresh, 255.0, imgproc::THRESH_BINARY)?;

    let k = imgproc::get_structuring_element(
        imgproc::MORPH_ELLIPSE,
        core::Size::new(3, 3),
        Point::new(-1, -1),
    )?;
    let mut dil = core::Mat::default();
    imgproc::dilate(
        &mask,
        &mut dil,
        &k,
        Point::new(-1, -1),
        1,
        core::BORDER_CONSTANT,
        imgproc::morphology_default_border_value()?,
    )?;
    Ok(dil)
}

/// 在帧差掩码里闪烁的点加分 —— 我方点会呼吸，敌方点通常不会。
pub fn boost_breathing(
    dots: &mut Vec<Dot>,
    bmask: &core::Mat,
    radius: i32,
    bonus: f64,
) -> opencv::Result<()> {
    if bmask.empty() {
        return Ok(());
    }
    for d in dots.iter_mut() {
        let sx = (d.pt.x - radius).max(0);
        let sy = (d.pt.y - radius).max(0);
        let ex = (d.pt.x + radius).min(bmask.cols() - 1);
        let ey = (d.pt.y + radius).min(bmask.rows() - 1);
        let mut hit = false;
        'outer: for y in sy..=ey {
            for x in sx..=ex {
                if *bmask.at_2d::<u8>(y, x)? > 0 {
                    hit = true;
                    break 'outer;
                }
            }
        }
        if hit {
            d.score += bonus;
        }
    }
    dots.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
    Ok(())
}

/// 视野框（距离标尺）。加了宽高比和占比校验，避免抓到地图上的黄色地形。
pub fn detect_camera_frame(minimap: &core::Mat) -> opencv::Result<Option<Rect>> {
    let mut hsv = core::Mat::default();
    imgproc::cvt_color(
        minimap,
        &mut hsv,
        imgproc::COLOR_BGR2HSV,
        0,
        core::AlgorithmHint::ALGO_HINT_DEFAULT,
    )?;

    let mut mask = core::Mat::default();
    core::in_range(
        &hsv,
        &Scalar::new(15.0, 60.0, 60.0, 0.0),
        &Scalar::new(85.0, 255.0, 255.0, 0.0),
        &mut mask,
    )?;

    let mut contours = core::Vector::<core::Vector<Point>>::new();
    imgproc::find_contours(
        &mask,
        &mut contours,
        imgproc::RETR_EXTERNAL,
        imgproc::CHAIN_APPROX_SIMPLE,
        Point::new(0, 0),
    )?;

    let min_w = (minimap.cols() as f64 * 0.18) as i32;
    let mut best: Option<Rect> = None;
    let mut max_area = 0;
    for i in 0..contours.len() {
        let c = contours.get(i)?;
        let r = opencv::geometry::bounding_rect(&c)?;
        let area = r.width * r.height;
        if r.width < min_w || r.width > minimap.cols() || r.height < 8 {
            continue;
        }
        let ar = r.width as f64 / (r.height.max(1)) as f64;
        if !(1.0..=4.0).contains(&ar) {
            // 视野框是横向矩形，太方 / 太扁的都不是
            continue;
        }
        if area > max_area {
            max_area = area;
            best = Some(r);
        }
    }
    Ok(best)
}

/// 多帧跟踪器：最近邻关联 + EMA 平滑 + 丢失容忍。
/// 替代原来 live_gui 里「位移 < 20px 就永远保持旧点」的写法（那个逻辑会把点卡死）。
pub struct Tracker {
    last: Option<(f64, f64)>,
    miss: i32,
    pub max_miss: i32,
    pub gate: f64,
    pub alpha: f64,
}

impl Tracker {
    pub fn new(max_miss: i32, gate: f64) -> Self {
        Self {
            last: None,
            miss: 0,
            max_miss,
            gate,
            alpha: 0.5,
        }
    }

    pub fn reset(&mut self) {
        self.last = None;
        self.miss = 0;
    }

    pub fn last_point(&self) -> Option<Point> {
        self.last
            .map(|(x, y)| Point::new(x.round() as i32, y.round() as i32))
    }

    /// `cands` 必须按 score 降序（detect_dots / boost_breathing 已保证）。
    pub fn update(&mut self, cands: &[Dot]) -> Option<Point> {
        if cands.is_empty() {
            self.miss += 1;
            if self.miss > self.max_miss {
                self.last = None;
            }
            return self.last_point();
        }

        let pick = match self.last {
            Some((lx, ly)) => {
                let mut best: Option<Dot> = None;
                let mut best_cost = f64::MAX;
                for d in cands {
                    let dx = d.pt.x as f64 - lx;
                    let dy = d.pt.y as f64 - ly;
                    let dist = (dx * dx + dy * dy).sqrt();
                    if dist <= self.gate {
                        let cost = dist - d.score * self.gate * 0.5;
                        if cost < best_cost {
                            best_cost = cost;
                            best = Some(*d);
                        }
                    }
                }
                best.unwrap_or(cands[0])
            }
            None => cands[0],
        };

        self.miss = 0;
        let np = match self.last {
            Some((lx, ly)) => {
                let dx = pick.pt.x as f64 - lx;
                let dy = pick.pt.y as f64 - ly;
                if (dx * dx + dy * dy).sqrt() > self.gate {
                    // 大跳变 = 换目标，硬切
                    (pick.pt.x as f64, pick.pt.y as f64)
                } else {
                    (lx + self.alpha * dx, ly + self.alpha * dy)
                }
            }
            None => (pick.pt.x as f64, pick.pt.y as f64),
        };
        self.last = Some(np);
        self.last_point()
    }
}

/// 多帧投票：连续 N 帧里出现过半才认。抗单帧误识别。
pub struct ValueVoter {
    buf: Vec<i32>,
    cap: usize,
}

impl ValueVoter {
    pub fn new(cap: usize) -> Self {
        Self {
            buf: Vec::new(),
            cap: cap.max(1),
        }
    }

    pub fn push(&mut self, v: i32) -> Option<i32> {
        self.buf.push(v);
        if self.buf.len() > self.cap {
            self.buf.remove(0);
        }
        let need = self.cap / 2 + 1;
        for &c in self.buf.iter() {
            if self.buf.iter().filter(|&&x| x == c).count() >= need {
                return Some(c);
            }
        }
        None
    }

    pub fn clear(&mut self) {
        self.buf.clear();
    }
}
