use std::f64::consts::PI;

// Frozen on first four recorded matches; no new-match refit.
// World coordinates: x right, y down. Public HUD API: y up and HUD distance units.
// One tick is a server trajectory sample interval, not a claimed frame or second.
pub const K: f64 = 0.9792415487505262;
const GRAVITY_WORLD: f64 = 4.202064201553961;
const WIND_ACCEL_WORLD: f64 = 0.02668859228049719;
const SPEED_SLOPE_WORLD: f64 = 1.4830798630263653;
const SPEED_INTERCEPT_WORLD: f64 = 0.06399350205524457;
const INITIAL_WIND_WORLD: f64 = 0.013021726383376776;
const INITIAL_DOWN_WORLD: f64 = 2.109151970666793;
const H0: f64 = 0.0; // origin is the projectile start; frontend origin acquisition remains unchanged.

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
fn start(angle: f64, power: f64, wind: f64) -> St {
    let scale = super::world_units_per_hud_unit();
    let speed = SPEED_SLOPE_WORLD * power + SPEED_INTERCEPT_WORLD;
    let r = angle * PI / 180.0;
    St {
        x: 0.0,
        y: H0,
        vx: (speed * r.cos() + INITIAL_WIND_WORLD * wind) / scale,
        vy: (speed * r.sin() - INITIAL_DOWN_WORLD) / scale,
    }
}

#[inline]
fn tick(s: &mut St, wind_accel: f64) {
    // Position-before-velocity update, exactly as the validated candidate.
    s.x += s.vx;
    s.y += s.vy;
    s.vy = K * s.vy - GRAVITY_WORLD / super::world_units_per_hud_unit();
    s.vx = K * s.vx + wind_accel;
}

pub fn height_at(angle: f64, power: f64, wind: f64, target_x: f64) -> Option<f64> {
    if target_x <= 0.0 {
        return Some(H0);
    }
    let w = wind * WIND_ACCEL_WORLD / super::world_units_per_hud_unit();
    let mut s = start(angle, power, wind);
    if s.vx <= 0.0 && target_x > 0.0 {
        return None;
    }
    for _ in 0..MAX_TICKS {
        let p = s;
        tick(&mut s, w);
        if s.x >= target_x {
            let t = if s.x > p.x {
                (target_x - p.x) / (s.x - p.x)
            } else {
                0.0
            };
            return Some(p.y + t * (s.y - p.y));
        }
        // 炮弹正在下落且被风吹回（vx 已反向），不可能再到达 target_x
        if s.vy < 0.0 && s.x <= p.x {
            return None;
        }
    }
    None
}

/// 炮弹下落时经过 `target_y` 的横坐标。
///
/// 大逆风下横向速度可能先向前、后反向；用下降交点求解可以覆盖
/// “第一次经过目标 x 时太高，但下落时实际能够命中”的轨迹。
fn descending_x_at_height(angle: f64, power: f64, wind: f64, target_y: f64) -> Option<f64> {
    let w = wind * WIND_ACCEL_WORLD / super::world_units_per_hud_unit();
    let mut s = start(angle, power, wind);

    for _ in 0..MAX_TICKS {
        let p = s;
        tick(&mut s, w);

        if p.y >= target_y && s.y <= target_y && s.y < p.y {
            let t = (p.y - target_y) / (p.y - s.y);
            return Some(p.x + t * (s.x - p.x));
        }

        // 已经在目标高度下方并继续下落，不可能再产生下降交点。
        if p.y < target_y && s.y < target_y && s.vy < 0.0 {
            return None;
        }
    }

    None
}

pub fn simulate_path(angle: f64, power: f64, wind: f64) -> Vec<(f64, f64)> {
    let mut path = Vec::new();
    let w = wind * WIND_ACCEL_WORLD / super::world_units_per_hud_unit();
    let mut s = start(angle, power, wind);
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

fn refine_descending_power(
    angle: f64,
    dx: f64,
    dy: f64,
    wind: f64,
    mut lo: f64,
    mut hi: f64,
    mut lo_error: f64,
) -> Option<f64> {
    const POSITION_EPSILON: f64 = 1e-6;

    for _ in 0..60 {
        let mid = (lo + hi) / 2.0;
        let mid_error = descending_x_at_height(angle, mid, wind, dy)? - dx;
        if (lo_error > 0.0) == (mid_error > 0.0) {
            lo = mid;
            lo_error = mid_error;
        } else {
            hi = mid;
        }
    }

    let power = (lo + hi) / 2.0;
    let x = descending_x_at_height(angle, power, wind, dy)?;
    ((x - dx).abs() <= POSITION_EPSILON).then_some(power)
}

/// 当第一次经过目标 x 的分支无解时，搜索下落分支。
/// 下降交点关于力度可能不是全局单调的，因此先用小区间找变号，再局部二分。
fn power_for_descending_hit(angle: f64, dx: f64, dy: f64, wind: f64) -> Option<f64> {
    const MIN_POWER: f64 = 0.0;
    const MAX_POWER: f64 = 150.0;
    const POWER_STEP: f64 = 0.25;
    const POSITION_EPSILON: f64 = 1e-6;

    let sample_count = ((MAX_POWER - MIN_POWER) / POWER_STEP) as usize;
    let mut previous: Option<(f64, f64)> = None;
    let mut previous_sample_power = MIN_POWER;

    for i in 0..=sample_count {
        let power = MIN_POWER + i as f64 * POWER_STEP;
        let Some(x) = descending_x_at_height(angle, power, wind, dy) else {
            previous = None;
            previous_sample_power = power;
            continue;
        };
        let error = x - dx;

        if error.abs() <= POSITION_EPSILON {
            return Some(power);
        }

        // 如果下降交点刚刚从 None 变为 Some，先二分找到这个连续分支的起点，
        // 防止根恰好落在第一个 0.25 采样点之前。
        if previous.is_none() && power > MIN_POWER {
            let mut unreachable = previous_sample_power;
            let mut reachable = power;
            if descending_x_at_height(angle, unreachable, wind, dy).is_none() {
                for _ in 0..60 {
                    let mid = (unreachable + reachable) / 2.0;
                    if descending_x_at_height(angle, mid, wind, dy).is_some() {
                        reachable = mid;
                    } else {
                        unreachable = mid;
                    }
                }

                if let Some(boundary_x) = descending_x_at_height(angle, reachable, wind, dy) {
                    let boundary_error = boundary_x - dx;
                    if boundary_error.abs() <= POSITION_EPSILON {
                        return Some(reachable);
                    }
                    previous = Some((reachable, boundary_error));
                }
            }
        }

        if let Some((previous_power, previous_error)) = previous {
            if (previous_error > 0.0) != (error > 0.0) {
                return refine_descending_power(
                    angle,
                    dx,
                    dy,
                    wind,
                    previous_power,
                    power,
                    previous_error,
                );
            }
        }

        previous = Some((power, error));
        previous_sample_power = power;
    }

    None
}

/// 固定角度，指定 x(dx) 和 y(dy)，求 0~150 范围内需要的力度。
///
/// 优先对第一次经过目标 x 的单调分支直接二分；若大逆风使该分支从
/// “到不了”直接跳到“已经高于目标”，再搜索完整轨迹的下落分支。
pub fn power_for_angle(angle: f64, dx: f64, dy: f64, wind: f64) -> Option<f64> {
    const MIN_POWER: f64 = 0.0;
    const MAX_POWER: f64 = 150.0;
    const POSITION_EPSILON: f64 = 1e-6;

    if dx <= 0.0 {
        return None;
    }
    let a = angle.clamp(MIN_ANGLE, MAX_ANGLE);

    if let Some(max_height) = height_at(a, MAX_POWER, wind, dx) {
        if max_height >= dy {
            let mut lo = MIN_POWER;
            let mut hi = MAX_POWER;
            for _ in 0..60 {
                let mid = (lo + hi) / 2.0;
                match height_at(a, mid, wind, dx) {
                    Some(y) if y >= dy => hi = mid,
                    _ => lo = mid,
                }
            }

            let power = (lo + hi) / 2.0;
            if let Some(final_height) = height_at(a, power, wind, dx) {
                if (final_height - dy).abs() <= POSITION_EPSILON {
                    return Some(power);
                }
            }
        }
    }

    power_for_descending_hit(a, dx, dy, wind)
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

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn recurrence_matches_independent_world_coordinate_golden_cases() {
        let path = simulate_path(117.0, 60.7, -5.1);
        for (i, x_world, y_down_world) in [(1, -40.96502625439154, -78.15890037179726), (10, -379.2247984141482, -533.4803171851452), (35, -1091.5946246650533, 54.80352817616575)] {
            let scale = super::super::world_units_per_hud_unit();
            assert!((path[i].0 * scale - x_world).abs() < 1e-9);
            assert!((-path[i].1 * scale - y_down_world).abs() < 1e-9);
        }
        let path = simulate_path(52.0, 74.88, -5.8);
        for (i, x_world, y_down_world) in [(1, 68.33493854172745, -85.45224969062802), (10, 616.3302046785013, -599.9646429951821), (35, 1637.9887430090303, -127.93285898416315)] {
            let scale = super::super::world_units_per_hud_unit();
            assert!((path[i].0 * scale - x_world).abs() < 1e-9);
            assert!((-path[i].1 * scale - y_down_world).abs() < 1e-9);
        }
        let path = simulate_path(154.0, 84.12, -31.5);
        for (i, x_world, y_down_world) in [(1, -112.59826085209399, -52.60862897276743), (10, -1062.228357706687, -300.5705121053173), (35, -3223.924199047161, 694.9709029061463)] {
            let scale = super::super::world_units_per_hud_unit();
            assert!((path[i].0 * scale - x_world).abs() < 1e-9);
            assert!((-path[i].1 * scale - y_down_world).abs() < 1e-9);
        }
    }
    #[test]
    fn mirrored_angle_and_wind_mirror_only_x() {
        let right = simulate_path(52.0, 74.88, -5.8);
        let left = simulate_path(128.0, 74.88, 5.8);
        for (r, l) in right.iter().zip(&left) {
            assert!((r.0 + l.0).abs() < 1e-10);
            assert!((r.1 - l.1).abs() < 1e-10);
        }
    }
    #[test]
    fn inverse_power_roundtrip_no_wind() {
        let target_y = height_at(45.0, 60.0, 0.0, 12.0).unwrap();
        let p = power_for_angle(45.0, 12.0, target_y, 0.0).unwrap();
        assert!((p - 60.0).abs() < 1e-6);
    }
    #[test]
    fn strong_wind_and_height_solutions_really_hit() {
        for (angle, dx, dy, wind) in [
            (50.0, 3.0, -8.0, -20.0),
            (60.0, 6.0, -8.0, -20.0),
            (70.0, 12.0, 0.0, -20.0),
            (55.0, 10.0, -3.0, 31.5),
            (65.0, 9.0, 2.0, -31.5),
        ] {
            let p = power_for_angle(angle, dx, dy, wind).expect("reachable reference case");
            let first = height_at(angle, p, wind, dx)
                .is_some_and(|y| (y - dy).abs() < 1e-6);
            let descending = descending_x_at_height(angle, p, wind, dy)
                .is_some_and(|x| (x - dx).abs() < 1e-6);
            assert!(first || descending, "angle={angle}, wind={wind}, power={p}");
        }
    }
    #[test]
    fn origin_is_not_legacy_height_offset() {
        assert_eq!(simulate_path(45.0, 50.0, 0.0)[0], (0.0, 0.0));
    }
}
