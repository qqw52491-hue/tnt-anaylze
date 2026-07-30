// fit.rs  --  compile & run:   rustc -O fit.rs -o fit && ./fit
// No external crates needed.
//
// Model (continuous linear drag, time unit chosen so g == 1):
//   v0 = c * (Power + P0)
//   y(x) = ((vy0 + g/b)/vx0)*x + (g/b^2)*ln(1 - b*x/vx0)
// Fits 3 params (b, c, P0) against the whole angle/distance/power table.

use std::f64::consts::PI;
use std::f64::NAN;

const ANGLES: [f64; 12] = [20., 25., 30., 35., 40., 45., 50., 55., 60., 65., 70., 80.];
const G: f64 = 1.0;

// rows = 屏距 1..=25 ; columns = ANGLES ; NAN = no data
const TABLE: [[f64; 12]; 25] = [
    [13.0, 11.0, 11.0, 10.0, 11.0, 11.0, 11.0, 10.0, 12.0, 13.0, 14.0, 20.0],
    [20.0, 18.0, 18.0, 17.0, 16.0, 17.0, 17.0, 17.0, 18.0, 20.0, 22.0, 31.0],
    [26.5, 24.0, 23.0, 22.0, 21.0, 22.0, 22.0, 22.0, 24.0, 26.0, 29.0, 42.0],
    [30.5, 28.0, 27.0, 26.0, 25.0, 26.0, 26.0, 27.0, 29.0, 31.0, 34.0, 51.0],
    [35.0, 32.0, 31.0, 30.0, 30.0, 29.5, 30.0, 31.0, 32.5, 36.0, 40.0, 60.0],
    [39.0, 36.0, 34.0, 33.0, 33.0, 33.0, 33.5, 34.0, 36.5, 40.0, 45.0, 67.0],
    [42.0, 39.0, 37.0, 36.0, 36.0, 36.0, 37.0, 38.0, 40.5, 44.0, 49.0, 74.0],
    [46.0, 43.0, 40.0, 39.0, 39.0, 39.0, 39.0, 41.0, 44.0, 48.0, 53.0, 80.0],
    [49.0, 46.0, 43.0, 42.0, 42.0, 41.5, 42.5, 44.0, 47.0, 51.0, 57.0, 88.0],
    [52.0, 49.0, 45.0, 45.0, 45.0, 45.5, 45.5, 47.0, 50.0, 55.0, 62.0, 95.0],
    [54.0, 51.0, 48.5, 47.0, 48.0, 47.0, 48.5, 50.0, 53.5, 58.0, 66.0, 100.0],
    [58.0, 54.0, 51.0, 50.0, 50.0, 50.0, 51.0, 53.0, 56.0, 62.0, 69.0, NAN],
    [60.0, 57.0, 52.5, 52.0, 52.0, 52.0, 52.5, 56.0, 59.0, 65.5, 73.0, NAN],
    [63.0, 59.0, 55.5, 55.0, 54.0, 54.0, 55.0, 58.0, 62.0, 67.5, 77.0, NAN],
    [65.0, 62.0, 58.0, 57.0, 56.0, 57.0, 59.0, 61.0, 65.0, 70.0, 80.0, NAN],
    [68.0, 64.0, 60.0, 59.0, 58.0, 59.0, 60.0, 63.0, 67.0, 73.0, 83.0, NAN],
    [70.0, 66.0, 62.5, 62.0, 60.0, 61.0, 63.0, 66.0, 70.0, 77.0, 87.0, NAN],
    [72.5, 68.0, 65.0, 64.0, 62.0, 63.5, 65.0, 68.0, 73.0, 80.0, 90.5, NAN],
    [74.0, 71.0, 66.5, 66.0, 65.0, 66.0, 67.0, 71.0, 75.0, 83.0, 94.5, NAN],
    [77.0, 73.0, 68.5, 68.0, 68.0, 68.0, 70.0, 73.0, 78.0, 86.0, 97.0, NAN],
    [79.5, 75.0, 71.0, 70.0, 70.0, 70.0, 72.0, 75.0, 81.0, 89.0, NAN, NAN],
    [81.5, 77.0, 73.0, 72.0, 72.0, 72.0, 74.0, 78.0, 83.0, 91.0, NAN, NAN],
    [84.0, 79.0, 75.0, 74.0, 74.0, 74.5, 76.0, 80.0, 85.5, 94.0, NAN, NAN],
    [86.0, 81.0, 77.0, 76.0, 76.0, 76.0, 78.0, 82.0, 88.0, 97.0, NAN, NAN],
    [88.0, 83.0, 79.5, 78.0, 78.0, 78.0, 80.0, 84.0, 90.5, 99.0, NAN, NAN],
];

#[derive(Clone, Copy)]
struct Pt {
    angle: f64,
    dist: f64,
    power: f64,
}

fn load() -> Vec<Pt> {
    let mut v = Vec::new();
    for (ri, row) in TABLE.iter().enumerate() {
        let dist = (ri + 1) as f64;
        for (ci, &p) in row.iter().enumerate() {
            if p.is_finite() {
                v.push(Pt { angle: ANGLES[ci], dist, power: p });
            }
        }
    }
    v
}

/// Height at horizontal distance x (closed form, linear drag).
fn height_at(angle: f64, power: f64, b: f64, c: f64, p0: f64, x: f64) -> f64 {
    let v0 = c * (power + p0);
    if v0 <= 0.0 {
        return f64::NEG_INFINITY;
    }
    let r = angle * PI / 180.0;
    let (vx, vy) = (v0 * r.cos(), v0 * r.sin());
    if vx <= 1e-12 {
        return f64::NEG_INFINITY;
    }
    let xa = vx / b; // horizontal asymptote
    if x >= xa {
        return f64::NEG_INFINITY;
    }
    let aa = (vy + G / b) / vx;
    let bb = G / (b * b);
    aa * x + bb * (-x / xa).ln_1p()
}

/// Flat-ground range: solve height_at == 0 for x > 0.
fn range_of(angle: f64, power: f64, b: f64, c: f64, p0: f64) -> f64 {
    let v0 = c * (power + p0);
    if v0 <= 0.0 {
        return 0.0;
    }
    let r = angle * PI / 180.0;
    let vx = v0 * r.cos();
    if vx <= 1e-12 {
        return 0.0;
    }
    let xa = vx / b;
    let f = |x: f64| height_at(angle, power, b, c, p0, x);

    let mut hi = xa * (1.0 - 1e-12);
    if f(hi) >= 0.0 {
        return hi; // never comes back down inside the domain
    }
    // find a small x with f > 0
    let mut lo = xa * 1e-9;
    let mut guard = 0;
    while f(lo) <= 0.0 {
        lo *= 10.0;
        guard += 1;
        if guard > 8 || lo >= hi {
            return 0.0;
        }
    }
    for _ in 0..50 {
        let mid = 0.5 * (lo + hi);
        if f(mid) > 0.0 {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    0.5 * (lo + hi)
}

/// Invert range(power) -- monotonically increasing in power.
fn power_for(angle: f64, dist: f64, b: f64, c: f64, p0: f64) -> f64 {
    let f = |p: f64| range_of(angle, p, b, c, p0) - dist;
    let mut lo = 1e-9;
    let mut hi = 20.0;
    let mut guard = 0;
    while f(hi) < 0.0 {
        hi *= 2.0;
        guard += 1;
        if guard > 50 {
            return NAN;
        }
    }
    if f(lo) > 0.0 {
        return lo;
    }
    for _ in 0..60 {
        let mid = 0.5 * (lo + hi);
        if f(mid) < 0.0 {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    0.5 * (lo + hi)
}

fn in_bounds(p: &[f64; 3]) -> bool {
    p[0] > 1e-5 && p[0] < 5.0 && p[1] > 1e-7 && p[1] < 100.0 && p[2] > -20.0 && p[2] < 40.0
}

/// Fast stage: residuals in DISTANCE units (one level of root-finding only).
fn cost_dist(data: &[Pt], p: &[f64; 3]) -> f64 {
    if !in_bounds(p) {
        return 1e12;
    }
    let mut s = 0.0;
    for q in data {
        let d = range_of(q.angle, q.power, p[0], p[1], p[2]);
        if !d.is_finite() {
            return 1e12;
        }
        let e = d - q.dist;
        s += e * e;
    }
    s
}

/// Polish stage: residuals in POWER units (matches how the table was measured).
fn cost_power(data: &[Pt], p: &[f64; 3]) -> f64 {
    if !in_bounds(p) {
        return 1e12;
    }
    let mut s = 0.0;
    for q in data {
        let pr = power_for(q.angle, q.dist, p[0], p[1], p[2]);
        if !pr.is_finite() {
            return 1e12;
        }
        let e = pr - q.power;
        s += e * e;
    }
    s
}

fn nelder_mead<F: FnMut(&[f64; 3]) -> f64>(
    mut f: F,
    x0: [f64; 3],
    step: [f64; 3],
    iters: usize,
) -> ([f64; 3], f64) {
    const N: usize = 3;
    let mut pts: Vec<[f64; 3]> = Vec::with_capacity(N + 1);
    pts.push(x0);
    for i in 0..N {
        let mut p = x0;
        p[i] += step[i];
        pts.push(p);
    }
    let mut vals: Vec<f64> = pts.iter().map(|p| f(p)).collect();

    for _ in 0..iters {
        let mut idx: Vec<usize> = (0..=N).collect();
        idx.sort_by(|&a, &b| vals[a].partial_cmp(&vals[b]).unwrap());
        pts = idx.iter().map(|&i| pts[i]).collect();
        vals = idx.iter().map(|&i| vals[i]).collect();

        if (vals[N] - vals[0]).abs() <= 1e-14 * (vals[0].abs() + 1e-14) {
            break;
        }
        let mut cen = [0.0f64; 3];
        for i in 0..N {
            for k in 0..3 {
                cen[k] += pts[i][k] / N as f64;
            }
        }
        let worst = pts[N];
        let mut xr = [0.0f64; 3];
        for k in 0..3 {
            xr[k] = cen[k] + (cen[k] - worst[k]);
        }
        let fr = f(&xr);

        if fr < vals[0] {
            let mut xe = [0.0f64; 3];
            for k in 0..3 {
                xe[k] = cen[k] + 2.0 * (cen[k] - worst[k]);
            }
            let fe = f(&xe);
            if fe < fr {
                pts[N] = xe;
                vals[N] = fe;
            } else {
                pts[N] = xr;
                vals[N] = fr;
            }
        } else if fr < vals[N - 1] {
            pts[N] = xr;
            vals[N] = fr;
        } else {
            let mut xc = [0.0f64; 3];
            for k in 0..3 {
                xc[k] = cen[k] + 0.5 * (worst[k] - cen[k]);
            }
            let fc = f(&xc);
            if fc < vals[N] {
                pts[N] = xc;
                vals[N] = fc;
            } else {
                let p0 = pts[0];
                for i in 1..=N {
                    for k in 0..3 {
                        pts[i][k] = p0[k] + 0.5 * (pts[i][k] - p0[k]);
                    }
                    vals[i] = f(&pts[i]);
                }
            }
        }
    }
    let mut bi = 0;
    for i in 1..=N {
        if vals[i] < vals[bi] {
            bi = i;
        }
    }
    (pts[bi], vals[bi])
}

fn main() {
    let data = load();
    println!("data points: {}", data.len());

    // ---------- stage 1: multi-start on distance residuals ----------
    let mut best = [0.1f64, 0.15, 0.0];
    let mut bestc = f64::INFINITY;
    for &b0 in &[0.03f64, 0.06, 0.10, 0.20, 0.40] {
        for &c0 in &[0.05f64, 0.15, 0.4, 1.0] {
            for &p00 in &[0.0f64, 4.0] {
                let (x, v) = nelder_mead(
                    |p| cost_dist(&data, p),
                    [b0, c0, p00],
                    [b0 * 0.3, c0 * 0.3, 2.0],
                    1500,
                );
                if v < bestc {
                    best = x;
                    bestc = v;
                }
            }
        }
    }
    println!("stage1 (distance residuals) SSE = {:.6}", bestc);

    // ---------- stage 2: polish on power residuals ----------
    let mut bp = best;
    let mut bv = cost_power(&data, &bp);
    for _ in 0..6 {
        let (x, v) = nelder_mead(
            |p| cost_power(&data, p),
            bp,
            [bp[0] * 0.05, bp[1] * 0.05, 0.5],
            600,
        );
        if v < bv {
            bp = x;
            bv = v;
        }
    }

    let (b, c, p0) = (bp[0], bp[1], bp[2]);
    println!("\n=== FIT (3 params, g fixed = 1 by choice of time unit) ===");
    println!("b  (drag rate)    = {:.6}", b);
    println!("c  (power->speed) = {:.6}", c);
    println!("P0 (power offset) = {:.4}", p0);

    // residual stats in power units
    let mut errs: Vec<(f64, f64, f64)> = Vec::new(); // angle, dist, err
    let mut sse = 0.0;
    let mut mx: f64 = 0.0;
    let mut sabs = 0.0;
    for q in &data {
        let pr = power_for(q.angle, q.dist, b, c, p0);
        let e = pr - q.power;
        errs.push((q.angle, q.dist, e));
        sse += e * e;
        sabs += e.abs();
        mx = mx.max(e.abs());
    }
    let n = data.len() as f64;
    println!("RMS power error   = {:.4}", (sse / n).sqrt());
    println!("mean |error|      = {:.4}", sabs / n);
    println!("max  |error|      = {:.4}", mx);

    let inv = b / G.sqrt();
    let mc = 0.0202027_f64 / 0.04_f64.sqrt(); // Minecraft k=0.98, g=0.04
    println!("\ndimensionless b/sqrt(g) = {:.5}   (Minecraft k=.98,g=.04 -> {:.5})", inv, mc);
    let mu = (1.0f64 / 0.04).sqrt();
    println!(
        "if we adopt g=0.04/tick -> b={:.6}, k=exp(-b)={:.5}",
        b / mu,
        (-b / mu).exp()
    );

    // ---------- residual structure ----------
    println!("\n=== residual by angle (pred - actual, power units) ===");
    println!("angle    n     mean      std      min      max");
    for &a in ANGLES.iter() {
        let v: Vec<f64> = errs.iter().filter(|e| e.0 == a).map(|e| e.2).collect();
        if v.is_empty() {
            continue;
        }
        let m = v.iter().sum::<f64>() / v.len() as f64;
        let sd = (v.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / v.len() as f64).sqrt();
        let lo = v.iter().cloned().fold(f64::INFINITY, f64::min);
        let hi = v.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        println!("{:5.0} {:4}  {:8.3} {:8.3} {:8.3} {:8.3}", a, v.len(), m, sd, lo, hi);
    }

    println!("\n=== residual by distance band ===");
    println!("band       n     mean      std");
    for &(l, h) in &[(0.5, 5.5), (5.5, 10.5), (10.5, 15.5), (15.5, 20.5), (20.5, 25.5)] {
        let v: Vec<f64> = errs
            .iter()
            .filter(|e| e.1 >= l && e.1 < h)
            .map(|e| e.2)
            .collect();
        if v.is_empty() {
            continue;
        }
        let m = v.iter().sum::<f64>() / v.len() as f64;
        let sd = (v.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / v.len() as f64).sqrt();
        println!("{:4.0}-{:<4.0} {:4}  {:8.3} {:8.3}", l + 0.5, h - 0.5, v.len(), m, sd);
    }

    // ---------- full residual dump (CSV to stdout) ----------
    println!("\n=== CSV: angle,dist,actual,pred,err ===");
    for (i, q) in data.iter().enumerate() {
        println!(
            "{},{},{},{:.3},{:.3}",
            q.angle,
            q.dist,
            q.power,
            q.power + errs[i].2,
            errs[i].2
        );
    }

    // ---------- payoff demo: solve angle WITH height difference ----------
    println!("\n=== demo: angle needed for target at distance 15, power 60 ===");
    for &dy in &[-6.0f64, -3.0, 0.0, 3.0, 6.0] {
        let f = |a: f64| height_at(a, 60.0, b, c, p0, 15.0) - dy;
        let mut sols: Vec<f64> = Vec::new();
        let mut prev = 20.0f64;
        let mut fp = f(prev);
        let mut a = 20.5f64;
        while a <= 85.0 {
            let fa = f(a);
            if fp.is_finite() && fa.is_finite() && fp * fa < 0.0 {
                let (mut lo, mut hi) = (prev, a);
                for _ in 0..60 {
                    let mid = 0.5 * (lo + hi);
                    if f(lo) * f(mid) <= 0.0 {
                        hi = mid;
                    } else {
                        lo = mid;
                    }
                }
                sols.push(0.5 * (lo + hi));
            }
            prev = a;
            fp = fa;
            a += 0.5;
        }
        let s: Vec<String> = sols.iter().map(|x| format!("{:.2}", x)).collect();
        println!("  dy = {:+5.1}  ->  angles: [{}]", dy, s.join(", "));
    }
}
