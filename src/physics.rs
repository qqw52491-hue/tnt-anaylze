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
        // 炮弹正在下落且被风吹回（vx 已反向），不可能再到达 target_x
        if s.vy < 0.0 && s.x <= p.x {
            return None;
        }
    }
    None
}

pub fn simulate_path(angle: f64, power: f64, wind: f64) -> Vec<(f64, f64)> {
    let mut path = Vec::new();
    let w = wind * WIND_SCALE;
    let mut s = start(angle, power);
    path.push((s.x, s.y));
    for _ in 0..MAX_TICKS {
        tick(&mut s, w);
        path.push((s.x, s.y));
        if s.y < -50.0 {
            break;
        }
    }
    path
}

/// 固定角度，指定 x(dx) 和 y(dy)，求唯一需要的力度
/// 两阶段求解：先粗扫（0.5 步长遍历 1~150），再二分精修
pub fn power_for_angle(angle: f64, dx: f64, dy: f64, wind: f64) -> Option<f64> {
    if dx <= 0.0 {
        return None;
    }
    let a = angle.clamp(MIN_ANGLE, MAX_ANGLE);

    // ====== Phase 1: 暴力粗扫 ======
    // 力度 1~150，每 0.5 试一次（300 次 height_at，< 1ms）
    let mut best_power = 0.0f64;
    let mut best_diff = f64::MAX;

    let mut power = 1.0;
    while power <= 150.0 {
        if let Some(y) = height_at(a, power, wind, dx) {
            let diff = (y - dy).abs();
            if diff < best_diff {
                best_diff = diff;
                best_power = power;
            }
        }
        power += 0.5;
    }

    // 粗扫最优解误差 > 1.0 → 该角度+风力组合不可达
    if best_diff > 1.0 {
        return None;
    }

    // ====== Phase 2: 二分精修 ======
    // 在 best_power ± 0.5 的邻域内二分，精度到小数点后 10 位
    let mut lo = (best_power - 0.5f64).max(1.0);
    let mut hi = (best_power + 0.5f64).min(150.0);
    for _ in 0..40 {
        let mid = (lo + hi) / 2.0;
        match height_at(a, mid, wind, dx) {
            Some(y) => {
                if y < dy {
                    lo = mid;
                } else {
                    hi = mid;
                }
            }
            None => {
                lo = mid;
            }
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
