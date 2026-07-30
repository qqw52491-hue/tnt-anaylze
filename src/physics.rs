use std::f64::consts::PI;

pub const K: f64 = 0.983638; 
pub const G: f64 = 0.039367; 
pub const C: f64 = 0.014945; 
pub const P0: f64 = 1.9176; 
pub const H0: f64 = 0.0335; 
pub const WIND_SCALE: f64 = 0.016220;

pub const MIN_ANGLE: f64 = 15.0;
pub const MAX_ANGLE: f64 = 89.0;
const MAX_TICKS: usize = 100_000;

#[derive(Clone, Copy, Debug)]
struct St {
    x: f64,
    y: f64,
    vx: f64,
    vy: f64,
}

#[inline]
fn start(angle: f64, power: f64) -> St {
    let v0 = C * (power + P0);
    let r = angle * PI / 180.0;
    St {
        x: 0.0,
        y: H0,
        vx: v0 * r.cos(),
        vy: v0 * r.sin(),
    }
}

#[inline]
fn tick(s: &mut St, wind_accel: f64) {
    s.x += s.vx;
    s.y += s.vy;
    s.vy = (s.vy - G) * K;
    s.vx = (s.vx - wind_accel) * K + wind_accel;
}

pub fn height_at(angle: f64, power: f64, wind: f64, target_x: f64) -> Option<f64> {
    if target_x <= 0.0 {
        return Some(H0);
    }
    let w = wind * WIND_SCALE;
    let mut s = start(angle, power);
    if s.vx <= 0.0 && target_x > 0.0 {
        return None;
    }
    for _ in 0..MAX_TICKS {
        let p = s;
        tick(&mut s, w);
        if s.x >= target_x {
            let t = if s.x > p.x { (target_x - p.x) / (s.x - p.x) } else { 0.0 };
            return Some(p.y + t * (s.y - p.y));
        }
        if s.vy < 0.0 && s.x <= p.x {
            return None; // 落回去了且没达到 target_x
        }
    }
    None
}

/// 固定角度，指定 x(dx) 和 y(dy)，求唯一需要的力度
pub fn power_for_angle(angle: f64, dx: f64, dy: f64, wind: f64) -> Option<f64> {
    if dx <= 0.0 { return None; }
    let a = angle.clamp(MIN_ANGLE, MAX_ANGLE);
    if dy >= dx * (a * PI / 180.0).tan() + H0 {
        return None; // 目标高于初始切线，任何力度都打不到
    }
    let mut lo = 0.0f64;
    let mut hi = 150.0f64;
    let mut ok = false;
    for _ in 0..50 {
        match height_at(a, hi, wind, dx) {
            Some(y) if y >= dy => { ok = true; break; }
            _ => hi *= 2.0, // 加大力度继续找上限
        }
    }
    if !ok { return None; }
    
    // 二分查找精确力度
    for _ in 0..50 {
        let mid = (lo + hi) / 2.0;
        match height_at(a, mid, wind, dx) {
            Some(y) => {
                if y < dy { lo = mid; } else { hi = mid; }
            }
            None => { lo = mid; }
        }
    }
    Some((lo + hi) / 2.0)
}

// 兼容老接口
pub fn calc_power(angle: f64, distance: f64, wind: f64) -> f64 {
    power_for_angle(angle, distance, 0.0, wind).unwrap_or(100.0)
}

pub fn calc_angle(distance: f64, dy: f64, power: f64, wind: f64, hint_angle: f64) -> f64 {
    let mut best_angle = hint_angle;
    let mut min_diff = f64::MAX;
    
    let mut test_angle = (hint_angle - 25.0).max(15.0);
    let max_test = (hint_angle + 25.0).min(89.0);
    
    while test_angle <= max_test {
        if let Some(y) = height_at(test_angle, power, wind, distance) {
            let diff = (y - dy).abs();
            if diff < min_diff {
                min_diff = diff;
                best_angle = test_angle;
            }
        }
        test_angle += 0.1;
    }
    best_angle
}
