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

/// 视野框（距离标尺）。比例尺只需要框的宽度 -> 只要找左右两条竖边。
/// 两种框都覆盖：亮线框（边是亮色细线）和罩色框（软边界,同 hue）。
/// 做法：提垂直边缘 -> 每列算"亮边最长连续段"和"同 hue 最长连续段" -> 分别配对。
/// 证据不足返回 None（错框会带歪力度比例，宁可走默认宽度）。
#[derive(Clone, Copy)]
struct LineFeat {
    pos: i32,
    hue: i32, // -1 = 亮线类(不查 hue)
    run_s: i32,
    run_e: i32,
    run_cnt: i32,
}

fn hue_dist(a: i32, b: i32) -> i32 {
    let d = (a - b).abs();
    d.min(180 - d)
}

/// 最长连续段（允许 ≤GAP 缺口）。
fn longest_run<F: Fn(usize) -> bool>(n: usize, pred: F) -> (i32, i32, i32) {
    const GAP: usize = 8;
    let (mut bs, mut be, mut bc) = (0i32, -1i32, 0i32);
    let (mut cs, mut cc, mut gap) = (0usize, 0i32, 0usize);
    for y in 0..n {
        if pred(y) {
            if cc == 0 {
                cs = y;
            }
            cc += 1;
            gap = 0;
        } else {
            gap += 1;
            if gap > GAP {
                if cc > 0 && (y - gap - cs) as i32 > be - bs {
                    bs = cs as i32;
                    be = (y - gap) as i32 - 1;
                    bc = cc;
                }
                cc = 0;
            }
        }
    }
    if cc > 0 && (n - cs) as i32 > be - bs {
        bs = cs as i32;
        be = n as i32 - 1;
        bc = cc;
    }
    (bs, be, bc)
}

/// 列扫描结果：亮度/hue 图 + 两类候选竖边。
struct ColScan {
    w: i32,
    h: i32,
    ecol: Vec<Vec<bool>>,
    huemap: Vec<Vec<u8>>,
    satmap: Vec<Vec<u8>>,
    bright: Vec<LineFeat>,
    soft: Vec<LineFeat>,
}

/// 提垂直边缘 + 扫描每列的亮线边/同hue软边特征。
fn scan_columns(minimap: &core::Mat) -> opencv::Result<Option<ColScan>> {
    let w = minimap.cols();
    let h = minimap.rows();
    if w < 30 || h < 30 {
        return Ok(None);
    }

    let mut gray = core::Mat::default();
    imgproc::cvt_color(
        minimap,
        &mut gray,
        imgproc::COLOR_BGR2GRAY,
        0,
        core::AlgorithmHint::ALGO_HINT_DEFAULT,
    )?;
    let mut blurred = core::Mat::default();
    imgproc::gaussian_blur(
        &gray,
        &mut blurred,
        core::Size::new(3, 3),
        0.0,
        0.0,
        core::BORDER_DEFAULT,
        core::AlgorithmHint::ALGO_HINT_DEFAULT,
    )?;
    let mut hsv = core::Mat::default();
    imgproc::cvt_color(
        minimap,
        &mut hsv,
        imgproc::COLOR_BGR2HSV,
        0,
        core::AlgorithmHint::ALGO_HINT_DEFAULT,
    )?;

    let (wu, hu) = (w as usize, h as usize);
    let mut g = vec![vec![0u8; wu]; hu];
    let mut huemap = vec![vec![0u8; wu]; hu];
    let mut satmap = vec![vec![0u8; wu]; hu];
    let mut valmap = vec![vec![0u8; wu]; hu];
    for y in 0..h {
        for x in 0..w {
            g[y as usize][x as usize] = *blurred.at_2d::<u8>(y, x)?;
            let px = hsv.at_2d::<core::Vec3b>(y, x)?;
            huemap[y as usize][x as usize] = px[0];
            satmap[y as usize][x as usize] = px[1];
            valmap[y as usize][x as usize] = px[2];
        }
    }

    // 垂直边缘：相邻两列灰度跳变（框线是软边，阈值放低）
    const EDGE_T: i32 = 7;
    let mut ecol = vec![vec![false; wu]; hu];
    for y in 0..hu {
        for x in 0..wu.saturating_sub(2) {
            ecol[y][x] = (g[y][x + 2] as i32 - g[y][x] as i32).abs() > EDGE_T;
        }
    }
    // 边缘带：软框线的边缘像素散在 ±1 列里,按 3 列的 OR 算
    let mut eband = vec![vec![false; wu]; hu];
    for y in 0..hu {
        for x in 0..wu {
            let xa = x.saturating_sub(1);
            let xb = (x + 1).min(wu - 1);
            eband[y][x] = (xa..=xb).any(|xx| ecol[y][xx]);
        }
    }

    // 每列两类特征：亮线边(val>168 的边缘点成段) / 同hue软边
    let min_run = (h * 10 / 100).max(12);
    let mut bright_cols: Vec<LineFeat> = Vec::new();
    let mut soft_cols: Vec<LineFeat> = Vec::new();
    for x in 0..wu {
        // 亮线类：边缘带内该列像素够亮
        let (bs, be, bc) = longest_run(hu, |y| eband[y][x] && valmap[y][x] > 168);
        if be - bs + 1 >= min_run && bc * 100 >= (be - bs + 1) * 40 {
            bright_cols.push(LineFeat {
                pos: x as i32,
                hue: -1,
                run_s: bs,
                run_e: be,
                run_cnt: bc,
            });
        }
        // 软边类：边缘段 + 主导 hue 一致
        let mut hist = [0i32; 36];
        let mut n_on = 0i32;
        let mut n_sat = 0i32;
        for y in 0..hu {
            if eband[y][x] {
                n_on += 1;
                if satmap[y][x] > 70 {
                    n_sat += 1;
                    hist[(huemap[y][x] as usize / 5).min(35)] += 1;
                }
            }
        }
        let (ss, se, sc) = longest_run(hu, |y| eband[y][x]);
        if se - ss + 1 < min_run || sc * 100 < (se - ss + 1) * 18 || n_sat < n_on / 2 {
            continue;
        }
        let (dom, &mx) = hist
            .iter()
            .enumerate()
            .max_by_key(|(_, v)| *v)
            .map(|(k, v)| (k, v))
            .unwrap();
        if (mx as f64) < (n_sat as f64) * 0.45 {
            continue;
        }
        soft_cols.push(LineFeat {
            pos: x as i32,
            hue: dom as i32 * 5,
            run_s: ss,
            run_e: se,
            run_cnt: sc,
        });
    }

    // 相邻合并（同类内）
    let dedupe = |feats: Vec<LineFeat>| -> Vec<LineFeat> {
        let mut out: Vec<LineFeat> = Vec::new();
        for f in feats {
            match out.last() {
                Some(l)
                    if f.pos - l.pos <= 6
                        && (f.hue < 0 || l.hue < 0 || hue_dist(f.hue, l.hue) <= 20) =>
                {
                    if f.run_cnt > l.run_cnt {
                        *out.last_mut().unwrap() = f;
                    }
                }
                _ => out.push(f),
            }
        }
        out
    };

    Ok(Some(ColScan {
        w,
        h,
        ecol,
        huemap,
        satmap,
        bright: dedupe(bright_cols),
        soft: dedupe(soft_cols),
    }))
}

impl ColScan {
    /// 列 x 在 [y1,y2] 段内的边缘覆盖率。
    fn cov(&self, x: i32, y1: i32, y2: i32) -> f64 {
        let wu = self.w as usize;
        let (xa, xb) = (
            (x - 1).max(0) as usize,
            ((x + 1).max(0) as usize).min(wu - 1),
        );
        let mut n = 0i32;
        for y in y1..=y2 {
            if (xa..=xb).any(|xx| self.ecol[y as usize][xx]) {
                n += 1;
            }
        }
        n as f64 / (y2 - y1 + 1) as f64
    }

    /// [xa,xb] 贴条内饱和像素里 hue≈he 的比例。
    fn frac_match(&self, xa: i32, xb: i32, y1: i32, y2: i32, he: i32) -> f64 {
        let wu = self.w;
        let (xa, xb) = (xa.max(0), xb.min(wu - 1));
        if xa >= xb {
            return 0.0;
        }
        let (mut n, mut m) = (0i32, 0i32);
        for y in y1..=y2 {
            for x in xa..=xb {
                if self.satmap[y as usize][x as usize] > 70 {
                    n += 1;
                    if hue_dist(self.huemap[y as usize][x as usize] as i32, he) <= 12 {
                        m += 1;
                    }
                }
            }
        }
        if n == 0 {
            0.0
        } else {
            m as f64 / n as f64
        }
    }
}

/// 视野框检测（全图扫描）。返回 (左,右) 竖边张成的矩形。
pub fn detect_camera_frame(minimap: &core::Mat) -> opencv::Result<Option<Rect>> {
    let Some(scan) = scan_columns(minimap)? else {
        return Ok(None);
    };
    let w = scan.w;

    // 配对：竖向区间重叠多 + 宽度合理 + (软边类)同 hue + 罩色台阶
    // wd > 80% 基本是整图内框/地图边框的签名;真框真占满时退回默认宽度也不亏
    let mut best: Option<(f64, i32, i32, i32, i32)> = None;
    for cols in [&scan.bright, &scan.soft] {
        for i in 0..cols.len() {
            for j in (i + 1)..cols.len() {
                let (a, b) = (cols[i], cols[j]);
                let wd = b.pos - a.pos;
                if wd < w * 15 / 100 || wd > w * 80 / 100 {
                    continue;
                }
                if a.hue >= 0 && b.hue >= 0 && hue_dist(a.hue, b.hue) > 18 {
                    continue;
                }
                let ov_s = a.run_s.max(b.run_s);
                let ov_e = a.run_e.min(b.run_e);
                let ov = ov_e - ov_s + 1;
                if ov * 100 < (a.run_e - a.run_s).min(b.run_e - b.run_s) * 40 {
                    continue;
                }
                let sup = scan.cov(a.pos, ov_s, ov_e).min(scan.cov(b.pos, ov_s, ov_e));
                if sup < 0.22 {
                    continue;
                }
                // 罩色框才有意义:边外侧素净、内侧被染成边色。
                // 左边要求"右侧贴条比左侧贴条更靠框色",右边反之。
                if a.hue >= 0 {
                    let he = a.hue;
                    let left_step = scan.frac_match(a.pos + 2, a.pos + 9, ov_s, ov_e, he)
                        - scan.frac_match(a.pos - 9, a.pos - 2, ov_s, ov_e, he);
                    let right_step = scan.frac_match(b.pos - 9, b.pos - 2, ov_s, ov_e, he)
                        - scan.frac_match(b.pos + 2, b.pos + 9, ov_s, ov_e, he);
                    if left_step < 0.08 || right_step < 0.08 {
                        continue;
                    }
                }
                let score = wd as f64 * ov as f64 * sup * sup;
                if best.map(|b| score > b.0).unwrap_or(true) {
                    best = Some((score, a.pos, ov_s, b.pos, ov_e));
                }
            }
        }
    }

    Ok(best.map(|(_, x1, y1, x2, y2)| Rect::new(x1, y1, x2 - x1, y2 - y1)))
}

/// 贴边吸附：用户手画大致视野框 roi,左右两边各自在 ±40% 宽范围内
/// 找"roi 纵向窗口内边缘密度最高"的列并吸过去。哪边找不到就保留手画位置。
/// 只吸附宽度——上下沿始终用手画的。
pub fn snap_camera_frame(minimap: &core::Mat, roi: &Rect) -> opencv::Result<Option<Rect>> {
    let Some(scan) = scan_columns(minimap)? else {
        return Ok(None);
    };
    let (w, h) = (scan.w, scan.h);
    let (rw, rh) = (roi.width.max(1), roi.height.max(1));
    // 竖边应纵贯手画框的高度;窗口放宽 30% 容错
    let (y0, y1) = (
        (roi.y - rh * 30 / 100).max(0),
        (roi.y + roi.height + rh * 30 / 100).min(h - 1),
    );
    if y1 - y0 < 10 {
        return Ok(None);
    }
    let span = rw * 40 / 100;

    // 贴条内饱和像素 hue≈he 的比例
    let frac_match = |xa: i32, xb: i32, he: f64| -> f64 {
        let xa = xa.max(0);
        let xb = xb.min(w - 1);
        if xa > xb {
            return 0.0;
        }
        let (mut n, mut m) = (0i32, 0i32);
        for y in y0..=y1 {
            for x in xa..=xb {
                if scan.satmap[y as usize][x as usize] > 70 {
                    n += 1;
                    let h = scan.huemap[y as usize][x as usize] as f64;
                    let d = (h - he).abs() % 180.0;
                    if d.min(180.0 - d) <= 14.0 {
                        m += 1;
                    }
                }
            }
        }
        if n < 10 {
            0.0
        } else {
            m as f64 / n as f64
        }
    };

    // 贴条饱和像素的环形均值 hue
    let mean_hue = |xa: i32, xb: i32| -> Option<f64> {
        let xa = xa.max(0);
        let xb = xb.min(w - 1);
        if xa > xb {
            return None;
        }
        let (mut s, mut c, mut n) = (0f64, 0f64, 0i32);
        for y in y0..=y1 {
            for x in xa..=xb {
                if scan.satmap[y as usize][x as usize] > 70 {
                    let r = (scan.huemap[y as usize][x as usize] as f64 * 2.0).to_radians();
                    s += r.sin();
                    c += r.cos();
                    n += 1;
                }
            }
        }
        (n >= 10).then(|| (s.atan2(c).to_degrees() / 2.0 + 180.0) % 180.0)
    };

    // 单边吸附：边缘密度 × (1 + 罩色台阶权重)。
    // 真框边是"罩色区"的边界——两侧 hue 均值差大;纯地形边两侧同色。
    let debug = std::env::var("TNT_SNAP_DEBUG").is_ok();
    // 距离衰减:离手画边越远权重越低,到搜索半径处归零——防止密度满分
    // 的图边框把内圈真框边挤掉
    let decay = |x: i32, side: i32| -> f64 {
        let t = ((x - side).abs() as f64 / span as f64).min(1.0);
        (1.0 - t) * (1.0 - t)
    };
    let snap_side = |side: i32| -> Option<i32> {
        let lo = (side - span).max(0);
        let hi = (side + span).min(w - 1);
        let mut scored: Vec<(f64, f64, f64, i32)> = Vec::new();
        for x in lo..=hi {
            let xa = (x - 1).max(0) as usize;
            let xb = ((x + 1) as usize).min(scan.ecol[0].len() - 1);
            let mut n = 0i32;
            for y in y0..=y1 {
                if (xa..=xb).any(|xx| scan.ecol[y as usize][xx]) {
                    n += 1;
                }
            }
            let den = n as f64 / (y1 - y0 + 1) as f64;
            let step = match (mean_hue(x - 9, x - 2), mean_hue(x + 2, x + 9)) {
                (Some(a), Some(b)) => {
                    let d = (a - b).abs() % 180.0;
                    d.min(180.0 - d)
                }
                _ => 0.0,
            };
            let score = den * (0.5 + step.min(40.0) / 40.0) * decay(x, side);
            scored.push((score, den, step, x));
        }
        if debug {
            let mut top = scored.clone();
            top.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
            eprintln!("snap side={} top:", side);
            for (s, d, st, x) in top.iter().take(6) {
                eprintln!("  x{}: score={:.3} den={:.2} step={:.0}", x, s, d, st);
            }
        }
        let (_, den, _, x) = scored
            .into_iter()
            .max_by(|a, b| a.0.partial_cmp(&b.0).unwrap())?;
        (den >= 0.20).then_some(x)
    };

    let left = snap_side(roi.x).unwrap_or(roi.x);
    // 左边定了,罩色 hue 就能取出来——右边要求"内侧贴条像罩色、外侧不像"的
    // 台阶(drop)。无边框罩色时所有 drop≈0,退回普通密度打分。
    let right = {
        let in_hi = (rw / 3).min(70).max(10);
        let h_in = mean_hue(left + 8, left + in_hi);
        let lo = (roi.x + roi.width - span).max(left + w * 10 / 100);
        let hi = (roi.x + roi.width + span).min(w - 1);
        let mut scored: Vec<(f64, f64, f64, i32)> = Vec::new();
        for x in lo..=hi {
            let xa = (x - 1).max(0) as usize;
            let xb = ((x + 1) as usize).min(scan.ecol[0].len() - 1);
            let mut n = 0i32;
            for y in y0..=y1 {
                if (xa..=xb).any(|xx| scan.ecol[y as usize][xx]) {
                    n += 1;
                }
            }
            let den = n as f64 / (y1 - y0 + 1) as f64;
            let drop = match h_in {
                Some(he) => frac_match(x - 70, x - 8, he) - frac_match(x + 8, x + 70, he),
                None => 0.0,
            };
            let dc = decay(x, roi.x + roi.width);
            let score = if drop >= 0.15 && den >= 0.20 {
                den * (0.2 + drop) + 10.0 // 过闸的优先一大截(罩色台阶已够特异,不再衰减)
            } else {
                den * 0.5 * dc
            };
            scored.push((score, den, drop, x));
        }
        if debug {
            let mut top = scored.clone();
            top.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
            eprintln!("snap side=R(drop) h_in={:?} top:", h_in);
            for (s, d, dr, x) in top.iter().take(6) {
                eprintln!("  x{}: score={:.3} den={:.2} drop={:.2}", x, s, d, dr);
            }
        }
        scored
            .into_iter()
            .max_by(|a, b| a.0.partial_cmp(&b.0).unwrap())
            .and_then(|(_, den, _, x)| (den >= 0.20).then_some(x))
            .unwrap_or(roi.x + roi.width)
    };
    if left == roi.x && right == roi.x + roi.width {
        return Ok(None); // 两边都没吸到
    }
    if right - left < w * 10 / 100 {
        return Ok(None); // 吸完窄得离谱,不可信
    }
    Ok(Some(Rect::new(left, roi.y, right - left, roi.height)))
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
