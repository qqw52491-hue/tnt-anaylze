// tnt.rs  --  rustc -O tnt.rs -o tnt && ./tnt
//
// Complete ballistics solver. No external crates.
//
// Replaces the old 36-parameter (a, b, c per angle) lookup fit with 6 physical
// constants, and adds the two things the old formula structurally could not do:
//   * height difference  (dy)
//   * wind that works at every angle and every distance without patches
//
// WHAT IT CAN ANSWER
//   given (dx, dy, wind) and a fixed angle  -> what power?      power_for_angle
//   given (dx, dy, wind) and a fixed power  -> what angle(s)?   angles_for_power
//   given (dx, dy, wind), high-arc style    -> angle at ~95     high_arc
//   given (dx, dy, wind)                    -> pick a shot      best_shot
//   given a shot                            -> full trajectory  trajectory
//
// SIGN CONVENTIONS
//   dx   > 0  horizontal distance to the target, in screen units (1 screen = 12)
//   dy   > 0  target is HIGHER than you, same unit as dx
//   wind > 0  tailwind (pushing toward the target), < 0 headwind
//
// PROVENANCE OF THE CONSTANTS
//   K, G, C, P0, H0 : fitted to the 270-cell no-wind angle/distance/power
//                     table.  RMS 0.48 power units, which is the table's own
//                     quantisation floor.
//   WIND_SCALE      : derived from the community high-arc rule 2.5 wind = 1 deg.
//
//   Independent checks the fit was NOT trained on:
//     * along angle = 90 - dist the model predicts power 94.3..96.3, i.e. the
//       community's "high arc = fixed 95 power" rule, at angles 74-86 deg that
//       were outside the fitted range entirely.
//     * community says 2.5 wind per degree at 80 deg, model says 2.51.
//       community says 2.0 at 70 deg, model says 2.34.
//     * model max-range angle 39.6 deg matches the table's power minimum at 40.

use std::f64::consts::PI;

// ------------------------------------------------------------------ constants

pub const K: f64 = 0.983638; // per-tick velocity retention (drag)
pub const G: f64 = 0.039367; // per-tick gravity
pub const C: f64 = 0.014945; // power -> launch speed
pub const P0: f64 = 1.9176; // power offset: v0 = C * (power + P0)
pub const H0: f64 = 0.0335; // launch height above your own ground level
pub const WIND_SCALE: f64 = 0.016220; // game wind number -> game speed units

pub const MIN_ANGLE: f64 = 20.0;
pub const MAX_ANGLE: f64 = 86.0; // validated up to ~86 via the high-arc rule
pub const MAX_POWER: f64 = 100.0;
pub const HIGH_ARC_POWER: f64 = 95.0;

const MAX_TICKS: usize = 100_000;

// ------------------------------------------------------------------ core sim

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
    St { x: 0.0, y: H0, vx: v0 * r.cos(), vy: v0 * r.sin() }
}

#[inline]
fn tick(s: &mut St, w: f64) {
    s.x += s.vx;
    s.y += s.vy;
    s.vy = (s.vy - G) * K;
    s.vx = (s.vx - w) * K + w;
}

/// Height of the shot when it has travelled `dx` horizontally.
/// `None` if it never gets that far (too little power, or a headwind stops it).
pub fn height_at(angle: f64, power: f64, wind: f64, dx: f64) -> Option<f64> {
    if dx <= 0.0 {
        return Some(H0);
    }
    let w = wind * WIND_SCALE;
    let mut s = start(angle, power);
    if s.vx <= 0.0 {
        return None;
    }
    for _ in 0..MAX_TICKS {
        let p = s;
        tick(&mut s, w);
        if s.x >= dx {
            let t = if s.x > p.x { (dx - p.x) / (s.x - p.x) } else { 0.0 };
            return Some(p.y + t * (s.y - p.y));
        }
        if s.x <= p.x {
            return None; // stalled or blown back before reaching the target
        }
    }
    None
}

/// Where the shot crosses height `dy` on its way DOWN. `dy = 0` is flat ground.
pub fn range_to(angle: f64, power: f64, wind: f64, dy: f64) -> Option<f64> {
    let w = wind * WIND_SCALE;
    let mut s = start(angle, power);
    for _ in 0..MAX_TICKS {
        let p = s;
        tick(&mut s, w);
        if s.y <= dy && p.y > dy {
            let t = (p.y - dy) / (p.y - s.y);
            return Some(p.x + t * (s.x - p.x));
        }
        if s.vy < 0.0 && s.x <= p.x {
            return None;
        }
    }
    None
}

/// Full trajectory, one point per tick, until it drops to height `dy`.
/// Use this to draw the arc or to compare against the game frame by frame.
pub fn trajectory(angle: f64, power: f64, wind: f64, dy: f64) -> Vec<(f64, f64)> {
    let w = wind * WIND_SCALE;
    let mut s = start(angle, power);
    let mut out = vec![(s.x, s.y)];
    for _ in 0..MAX_TICKS {
        let p = s;
        tick(&mut s, w);
        if s.y <= dy && p.y > dy {
            let t = (p.y - dy) / (p.y - s.y);
            out.push((p.x + t * (s.x - p.x), dy));
            break;
        }
        out.push((s.x, s.y));
        if s.vy < 0.0 && s.x <= p.x {
            break;
        }
    }
    out
}

// ------------------------------------------------------------------ inverses

/// THE MAIN ONE. Power needed to hit a target `dx` away and `dy` higher,
/// at a fixed angle, with the given wind.
///
/// Returns None if the target is unreachable at that angle. Note the hard
/// ceiling: at any power the shot can never be steeper than the launch line,
/// so dy must be below dx * tan(angle).
pub fn power_for_angle(angle: f64, dx: f64, dy: f64, wind: f64) -> Option<f64> {
    if dx <= 0.0 {
        return None;
    }
    let a = angle.clamp(MIN_ANGLE, MAX_ANGLE);
    if dy >= dx * (a * PI / 180.0).tan() + H0 {
        return None; // above the launch line, impossible at any power
    }
    // height at dx rises monotonically with power -> plain bisection
    let mut lo = 0.0f64;
    let mut hi = 20.0f64;
    let mut ok = false;
    for _ in 0..40 {
        match height_at(a, hi, wind, dx) {
            Some(h) if h >= dy => {
                ok = true;
                break;
            }
            _ => hi *= 2.0,
        }
    }
    if !ok {
        return None;
    }
    for _ in 0..70 {
        let m = 0.5 * (lo + hi);
        match height_at(a, m, wind, dx) {
            Some(h) if h >= dy => hi = m,
            _ => lo = m,
        }
    }
    Some(0.5 * (lo + hi))
}

/// All angles that hit the target at a fixed power. Usually two -- a flat one
/// and a lofted one -- sorted ascending. Empty if out of reach.
///
/// The old calc_angle scanned for minimum error and returned one value, which
/// meant it silently jumped between these two branches. Pick deliberately.
pub fn angles_for_power(dx: f64, dy: f64, power: f64, wind: f64) -> Vec<f64> {
    let err = |a: f64| height_at(a, power, wind, dx).map(|h| h - dy);
    let mut out: Vec<f64> = Vec::new();
    let step = 0.25f64;
    let mut prev = MIN_ANGLE;
    let mut fprev = err(prev);
    let mut a = MIN_ANGLE + step;
    while a <= MAX_ANGLE + 1e-9 {
        let fa = err(a);
        if let (Some(p), Some(c)) = (fprev, fa) {
            if p * c < 0.0 {
                let mut lo = prev;
                let mut hi = a;
                let mut flo = p;
                for _ in 0..60 {
                    let m = 0.5 * (lo + hi);
                    match err(m) {
                        Some(fm) => {
                            if (flo > 0.0) == (fm > 0.0) {
                                lo = m;
                                flo = fm;
                            } else {
                                hi = m;
                            }
                        }
                        None => lo = m,
                    }
                }
                out.push(0.5 * (lo + hi));
            }
        }
        prev = a;
        fprev = fa;
        a += step;
    }
    out.sort_by(|p, q| p.partial_cmp(q).unwrap());
    out
}

/// High-arc solver: the playstyle you said is very accurate in practice.
/// Holds power fixed (default 95) and finds the steep-branch angle.
/// Generalises the community rule "angle = 90 - screendist" to any dy and wind.
pub fn high_arc(dx: f64, dy: f64, wind: f64, power: f64) -> Option<f64> {
    let all = angles_for_power(dx, dy, power, wind);
    all.into_iter().filter(|a| *a >= 45.0).next_back()
}

#[derive(Clone, Copy, Debug)]
pub struct Shot {
    pub angle: f64,
    pub power: f64,
    pub high_arc: bool,
}

/// Convenience: prefer the high arc at ~95 power. If that cannot reach, fall
/// back to the flattest angle that can, staying within the power cap.
pub fn best_shot(dx: f64, dy: f64, wind: f64) -> Option<Shot> {
    if let Some(a) = high_arc(dx, dy, wind, HIGH_ARC_POWER) {
        return Some(Shot { angle: a, power: HIGH_ARC_POWER, high_arc: true });
    }
    // sweep angles, keep the one needing the least power
    let mut best: Option<Shot> = None;
    let mut a = MIN_ANGLE;
    while a <= MAX_ANGLE {
        if let Some(p) = power_for_angle(a, dx, dy, wind) {
            if p <= MAX_POWER {
                let better = match best {
                    None => true,
                    Some(b) => p < b.power,
                };
                if better {
                    best = Some(Shot { angle: a, power: p, high_arc: a >= 45.0 });
                }
            }
        }
        a += 0.25;
    }
    best
}

/// Furthest you can reach at a given power, and the angle that does it.
/// Call this first to answer "can I even hit that?"
pub fn max_range(power: f64, wind: f64, dy: f64) -> (f64, f64) {
    let mut best = (0.0f64, MIN_ANGLE);
    let mut a = MIN_ANGLE;
    while a <= MAX_ANGLE {
        if let Some(d) = range_to(a, power, wind, dy) {
            if d > best.0 {
                best = (d, a);
            }
        }
        a += 0.25;
    }
    let lo = (best.1 - 0.5).max(MIN_ANGLE);
    let hi = (best.1 + 0.5).min(MAX_ANGLE);
    let mut a = lo;
    while a <= hi {
        if let Some(d) = range_to(a, power, wind, dy) {
            if d > best.0 {
                best = (d, a);
            }
        }
        a += 0.01;
    }
    best
}

/// Does a tailwind mean you should ADD angle or REDUCE it?
/// Above the max-range angle, add. Below it, reduce -- the community's blanket
/// "tailwind means more angle" is backwards down there.
pub fn tailwind_means_more_angle(angle: f64, power: f64) -> bool {
    let h = 0.05;
    let a = range_to(angle + h, power, 0.0, 0.0);
    let b = range_to(angle - h, power, 0.0, 0.0);
    match (a, b) {
        (Some(x), Some(y)) => (x - y) < 0.0,
        _ => true,
    }
}

// ------------------------------------------------------------------ self test

const ANGLES: [f64; 11] = [20., 25., 30., 35., 40., 45., 50., 55., 60., 65., 70.];
const TABLE: [[f64; 11]; 25] = [
    [13.0, 11.0, 11.0, 10.0, 11.0, 11.0, 11.0, 10.0, 12.0, 13.0, 14.0],
    [20.0, 18.0, 18.0, 17.0, 16.0, 17.0, 17.0, 17.0, 18.0, 20.0, 22.0],
    [26.5, 24.0, 23.0, 22.0, 21.0, 22.0, 22.0, 22.0, 24.0, 26.0, 29.0],
    [30.5, 28.0, 27.0, 26.0, 25.0, 26.0, 26.0, 27.0, 29.0, 31.0, 34.0],
    [35.0, 32.0, 31.0, 30.0, 30.0, 29.5, 30.0, 31.0, 32.5, 36.0, 40.0],
    [39.0, 36.0, 34.0, 33.0, 33.0, 33.0, 33.5, 34.0, 36.5, 40.0, 45.0],
    [42.0, 39.0, 37.0, 36.0, 36.0, 36.0, 37.0, 38.0, 40.5, 44.0, 49.0],
    [46.0, 43.0, 40.0, 39.0, 39.0, 39.0, 39.0, 41.0, 44.0, 48.0, 53.0],
    [49.0, 46.0, 43.0, 42.0, 42.0, 41.5, 42.5, 44.0, 47.0, 51.0, 57.0],
    [52.0, 49.0, 45.0, 45.0, 45.0, 45.5, 45.5, 47.0, 50.0, 55.0, 62.0],
    [54.0, 51.0, 48.5, 47.0, 48.0, 47.0, 48.5, 50.0, 53.5, 58.0, 66.0],
    [58.0, 54.0, 51.0, 50.0, 50.0, 50.0, 51.0, 53.0, 56.0, 62.0, 69.0],
    [60.0, 57.0, 52.5, 52.0, 52.0, 52.0, 52.5, 56.0, 59.0, 65.5, 73.0],
    [63.0, 59.0, 55.5, 55.0, 54.0, 54.0, 55.0, 58.0, 62.0, 67.5, 77.0],
    [65.0, 62.0, 58.0, 57.0, 56.0, 57.0, 59.0, 61.0, 65.0, 70.0, 80.0],
    [68.0, 64.0, 60.0, 59.0, 58.0, 59.0, 60.0, 63.0, 67.0, 73.0, 83.0],
    [70.0, 66.0, 62.5, 62.0, 60.0, 61.0, 63.0, 66.0, 70.0, 77.0, 87.0],
    [72.5, 68.0, 65.0, 64.0, 62.0, 63.5, 65.0, 68.0, 73.0, 80.0, 90.5],
    [74.0, 71.0, 66.5, 66.0, 65.0, 66.0, 67.0, 71.0, 75.0, 83.0, 94.5],
    [77.0, 73.0, 68.5, 68.0, 68.0, 68.0, 70.0, 73.0, 78.0, 86.0, 97.0],
    [79.5, 75.0, 71.0, 70.0, 70.0, 70.0, 72.0, 75.0, 81.0, 89.0, f64::NAN],
    [81.5, 77.0, 73.0, 72.0, 72.0, 72.0, 74.0, 78.0, 83.0, 91.0, f64::NAN],
    [84.0, 79.0, 75.0, 74.0, 74.0, 74.5, 76.0, 80.0, 85.5, 94.0, f64::NAN],
    [86.0, 81.0, 77.0, 76.0, 76.0, 76.0, 78.0, 82.0, 88.0, 97.0, f64::NAN],
    [88.0, 83.0, 79.5, 78.0, 78.0, 78.0, 80.0, 84.0, 90.5, 99.0, f64::NAN],
];

fn main() {
    // 1. reproduce the no-wind table
    let mut sse = 0.0f64;
    let mut n = 0usize;
    let mut worst = (0.0f64, 0.0f64, 0.0f64);
    for (ri, row) in TABLE.iter().enumerate() {
        let dist = (ri + 1) as f64;
        for (ci, &p) in row.iter().enumerate() {
            if !p.is_finite() {
                continue;
            }
            if let Some(pred) = power_for_angle(ANGLES[ci], dist, 0.0, 0.0) {
                let e = pred - p;
                sse += e * e;
                n += 1;
                if e.abs() > worst.0.abs() {
                    worst = (e, ANGLES[ci], dist);
                }
            }
        }
    }
    let denom = n as f64;
    println!("=== self test: original no-wind table ===");
    println!("n={}  RMS={:.4}  (expect ~0.48)", n, (sse / denom).sqrt());
    println!("worst {:+.3} at angle {} dist {}\n", worst.0, worst.1, worst.2);

    // 2. out-of-sample: the community high-arc rule
    println!("=== community high-arc rule: angle = 90 - dist, power should be ~95 ===");
    for d in [6.0f64, 8.0, 10.0, 12.0, 14.0] {
        let a = 90.0 - d;
        match power_for_angle(a, d, 0.0, 0.0) {
            Some(p) => println!("  dist {:>4}  angle {:>4}  -> power {:6.2}", d, a, p),
            None => println!("  dist {:>4}  angle {:>4}  -> unreachable", d, a),
        }
    }

    // 3. wind, cross-checked against the guides
    println!("\n=== wind check (community: angle = 90 - dist +/- wind/2.5) ===");
    for (d, w, base, want) in [
        (12.0f64, -5.0f64, 78.0f64, 76.0f64),
        (12.0, 5.0, 78.0, 80.0),
        (7.0, 2.2, 83.0, 84.0),
        (10.0, -4.0, 80.0, 78.4),
    ] {
        let pw = power_for_angle(base, d, 0.0, 0.0).unwrap_or(HIGH_ARC_POWER);
        match high_arc(d, 0.0, w, pw) {
            Some(a) => println!(
                "  dist {:>4} wind {:>5} -> model {:6.2}   guides {:5.1}",
                d, w, a, want
            ),
            None => println!("  dist {:>4} wind {:>5} -> no solution", d, w),
        }
    }

    // 4. THE NEW CAPABILITY: height difference
    println!("\n=== power needed at a fixed angle, with height difference ===");
    println!("    target 12 units away, no wind");
    println!("      dy    45deg    60deg    70deg    78deg");
    for dy in [-8.0f64, -4.0, -2.0, 0.0, 2.0, 4.0, 8.0] {
        print!("  {:+6.1}", dy);
        for a in [45.0f64, 60.0, 70.0, 78.0] {
            match power_for_angle(a, 12.0, dy, 0.0) {
                Some(p) => print!("  {:7.2}", p),
                None => print!("      n/a"),
            }
        }
        println!();
    }

    println!("\n=== high-arc angle at power 95, with height difference and wind ===");
    println!("    dist    dy   wind  ->  angle");
    for (d, dy, w) in [
        (12.0f64, 0.0f64, 0.0f64),
        (12.0, 4.0, 0.0),
        (12.0, -4.0, 0.0),
        (12.0, 4.0, 5.0),
        (12.0, -4.0, -5.0),
        (8.0, 3.0, 3.0),
    ] {
        match high_arc(d, dy, w, HIGH_ARC_POWER) {
            Some(a) => println!("  {:6.1} {:+5.1} {:+6.1}  ->  {:6.2}", d, dy, w, a),
            None => println!("  {:6.1} {:+5.1} {:+6.1}  ->  unreachable at 95", d, dy, w),
        }
    }

    // 5. both branches
    println!("\n=== both solutions at power 60, target 12 away ===");
    for dy in [-6.0f64, -3.0, 0.0, 3.0] {
        let v = angles_for_power(12.0, dy, 60.0, 0.0);
        let s: Vec<String> = v.iter().map(|a| format!("{:.2}", a)).collect();
        println!("  dy {:+5.1} -> [{}]", dy, s.join(", "));
    }

    // 6. best_shot convenience
    println!("\n=== best_shot ===");
    for (d, dy, w) in [
        (5.0f64, 0.0f64, 0.0f64),
        (12.0, 0.0, 0.0),
        (12.0, 5.0, -3.0),
        (20.0, -5.0, 2.0),
        (25.0, 0.0, 0.0),
    ] {
        match best_shot(d, dy, w) {
            Some(s) => println!(
                "  dist {:>5} dy {:+5.1} wind {:+5.1} -> angle {:6.2} power {:6.2} {}",
                d, dy, w, s.angle, s.power,
                if s.high_arc { "(high arc)" } else { "(flat)" }
            ),
            None => println!("  dist {:>5} dy {:+5.1} wind {:+5.1} -> out of reach", d, dy, w),
        }
    }

    // 7. direction of the wind correction
    println!("\n=== tailwind: add angle or reduce it? ===");
    for a in [25.0f64, 30.0, 35.0, 40.0, 45.0, 60.0, 78.0] {
        let dir = if tailwind_means_more_angle(a, 60.0) { "ADD" } else { "REDUCE" };
        println!("  {:5.0} deg -> {}", a, dir);
    }
    let (r, a) = max_range(60.0, 0.0, 0.0);
    println!("  (max-range angle at power 60 is {:.2} deg, reach {:.2})", a, r);

    // 8. a trajectory you can compare against the game
    println!("\n=== trajectory: 78 deg, power 95, no wind (every 20th tick) ===");
    let t = trajectory(78.0, 95.0, 0.0, 0.0);
    println!("  flight time: {} ticks", t.len() - 1);
    for (i, (x, y)) in t.iter().enumerate() {
        if i % 20 == 0 || i == t.len() - 1 {
            println!("    tick {:>4}  x {:8.4}  y {:8.4}", i, x, y);
        }
    }
}
