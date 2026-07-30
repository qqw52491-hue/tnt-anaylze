// fit2.rs  --  rustc -O fit2.rs -o fit2 && ./fit2
// Discrete per-tick simulation (matches how the game actually computes it)
// + launch height h0, + optional separate horizontal/vertical drag.
// Compares several model variants side by side so you can see which one
// removes the "smile" in the residual-by-angle column.

use std::f64::consts::PI;
use std::f64::NAN;

const ANGLES: [f64; 12] = [20., 25., 30., 35., 40., 45., 50., 55., 60., 65., 70., 80.];

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
struct Pt { angle: f64, dist: f64, power: f64 }

fn load(skip80: bool) -> Vec<Pt> {
    let mut v = Vec::new();
    for (ri, row) in TABLE.iter().enumerate() {
        let dist = (ri + 1) as f64;
        for (ci, &p) in row.iter().enumerate() {
            if p.is_finite() {
                if skip80 && ANGLES[ci] >= 80.0 { continue; }
                v.push(Pt { angle: ANGLES[ci], dist, power: p });
            }
        }
    }
    v
}

// ---------------- model ----------------
// params vector layout is described by ModelSpec
#[derive(Clone, Copy)]
struct Model {
    variant: u8,      // tick update order
    free_k: bool,     // fit kh (and kv)
    split_k: bool,    // separate horizontal / vertical drag
    free_g: bool,     // fit g
    free_h0: bool,    // fit launch height
    fix_k: f64,       // used when !free_k
    fix_g: f64,       // used when !free_g
}

struct P { kh: f64, kv: f64, g: f64, c: f64, p0: f64, h0: f64 }

impl Model {
    fn nparams(&self) -> usize {
        let mut n = 2; // c, p0
        if self.free_k { n += if self.split_k { 2 } else { 1 } }
        if self.free_g { n += 1 }
        if self.free_h0 { n += 1 }
        n
    }
    fn unpack(&self, x: &[f64]) -> P {
        let mut i = 0;
        let (kh, kv);
        if self.free_k {
            kh = x[i]; i += 1;
            if self.split_k { kv = x[i]; i += 1 } else { kv = kh }
        } else { kh = self.fix_k; kv = self.fix_k }
        let g = if self.free_g { let v = x[i]; i += 1; v } else { self.fix_g };
        let c = x[i]; i += 1;
        let p0 = x[i]; i += 1;
        let h0 = if self.free_h0 { x[i] } else { 0.0 };
        P { kh, kv, g, c, p0, h0 }
    }
    fn start(&self) -> (Vec<f64>, Vec<f64>) {
        let mut x = Vec::new(); let mut s = Vec::new();
        if self.free_k { x.push(0.98); s.push(0.01);
            if self.split_k { x.push(0.98); s.push(0.01) } }
        if self.free_g { x.push(0.04); s.push(0.01) }
        x.push(0.08); s.push(0.02);   // c
        x.push(2.7);  s.push(1.0);    // p0
        if self.free_h0 { x.push(1.5); s.push(0.5) }
        (x, s)
    }
    fn ok(&self, p: &P) -> bool {
        p.kh > 0.80 && p.kh < 1.0 && p.kv > 0.80 && p.kv < 1.0
            && p.g > 1e-4 && p.g < 2.0
            && p.c > 1e-6 && p.c < 100.0
            && p.p0 > -20.0 && p.p0 < 40.0
            && p.h0 >= 0.0 && p.h0 < 30.0
    }
}

/// Discrete per-tick trajectory. Returns horizontal distance where y crosses 0
/// coming down (sub-tick linear interpolation).
fn sim_range(angle: f64, power: f64, m: &Model, p: &P) -> f64 {
    let v0 = p.c * (power + p.p0);
    if v0 <= 0.0 { return 0.0 }
    let r = angle * PI / 180.0;
    let mut vx = v0 * r.cos();
    let mut vy = v0 * r.sin();
    let mut x = 0.0f64;
    let mut y = p.h0;
    for _ in 0..200_000 {
        let (px, py) = (x, y);
        match m.variant {
            0 => { vy = (vy - p.g) * p.kv; vx *= p.kh; x += vx; y += vy; }
            1 => { vy = vy * p.kv - p.g;   vx *= p.kh; x += vx; y += vy; }
            _ => { x += vx; y += vy; vy = (vy - p.g) * p.kv; vx *= p.kh; }
        }
        if y <= 0.0 && py > 0.0 {
            let t = py / (py - y);
            return px + t * (x - px);
        }
        if vx < 1e-14 && vy < 0.0 { return x }
    }
    x
}

/// Height at horizontal distance xt (for solving with height difference).
fn sim_height_at(angle: f64, power: f64, m: &Model, p: &P, xt: f64) -> f64 {
    let v0 = p.c * (power + p.p0);
    if v0 <= 0.0 { return f64::NEG_INFINITY }
    let r = angle * PI / 180.0;
    let mut vx = v0 * r.cos();
    let mut vy = v0 * r.sin();
    let mut x = 0.0f64;
    let mut y = p.h0;
    if xt <= 0.0 { return y }
    for _ in 0..200_000 {
        let (px, py) = (x, y);
        match m.variant {
            0 => { vy = (vy - p.g) * p.kv; vx *= p.kh; x += vx; y += vy; }
            1 => { vy = vy * p.kv - p.g;   vx *= p.kh; x += vx; y += vy; }
            _ => { x += vx; y += vy; vy = (vy - p.g) * p.kv; vx *= p.kh; }
        }
        if x >= xt {
            let t = if x > px { (xt - px) / (x - px) } else { 0.0 };
            return py + t * (y - py);
        }
        if vx < 1e-14 { return f64::NEG_INFINITY }
    }
    f64::NEG_INFINITY
}

fn power_for(angle: f64, dist: f64, m: &Model, p: &P) -> f64 {
    let f = |pw: f64| {
        let mut q = P { kh: p.kh, kv: p.kv, g: p.g, c: p.c, p0: p.p0, h0: p.h0 };
        let _ = &mut q;
        sim_range(angle, pw, m, p) - dist
    };
    let mut lo = -p.p0 + 1e-9;
    if lo < 0.0 { lo = 0.0 }
    let mut hi = 20.0f64.max(lo + 1.0);
    let mut guard = 0;
    while f(hi) < 0.0 { hi *= 2.0; guard += 1; if guard > 40 { return NAN } }
    if f(lo) > 0.0 { return lo }
    for _ in 0..50 {
        let mid = 0.5 * (lo + hi);
        if f(mid) < 0.0 { lo = mid } else { hi = mid }
    }
    0.5 * (lo + hi)
}

fn cost_dist(data: &[Pt], m: &Model, x: &[f64]) -> f64 {
    let p = m.unpack(x);
    if !m.ok(&p) { return 1e12 }
    let mut s = 0.0;
    for q in data {
        let d = sim_range(q.angle, q.power, m, &p);
        if !d.is_finite() { return 1e12 }
        let e = d - q.dist; s += e * e;
    }
    s
}

fn cost_power(data: &[Pt], m: &Model, x: &[f64]) -> f64 {
    let p = m.unpack(x);
    if !m.ok(&p) { return 1e12 }
    let mut s = 0.0;
    for q in data {
        let pr = power_for(q.angle, q.dist, m, &p);
        if !pr.is_finite() { return 1e12 }
        let e = pr - q.power; s += e * e;
    }
    s
}

fn nelder_mead<F: FnMut(&[f64]) -> f64>(
    mut f: F, x0: &[f64], step: &[f64], iters: usize,
) -> (Vec<f64>, f64) {
    let n = x0.len();
    let mut pts: Vec<Vec<f64>> = vec![x0.to_vec()];
    for i in 0..n {
        let mut p = x0.to_vec(); p[i] += step[i]; pts.push(p);
    }
    let mut vals: Vec<f64> = pts.iter().map(|p| f(p)).collect();
    for _ in 0..iters {
        let mut idx: Vec<usize> = (0..=n).collect();
        idx.sort_by(|&a, &b| vals[a].partial_cmp(&vals[b]).unwrap());
        pts = idx.iter().map(|&i| pts[i].clone()).collect();
        vals = idx.iter().map(|&i| vals[i]).collect();
        if (vals[n] - vals[0]).abs() <= 1e-13 * (vals[0].abs() + 1e-13) { break }
        let mut cen = vec![0.0; n];
        for i in 0..n { for k in 0..n { cen[k] += pts[i][k] / n as f64 } }
        let worst = pts[n].clone();
        let xr: Vec<f64> = (0..n).map(|k| cen[k] + (cen[k] - worst[k])).collect();
        let fr = f(&xr);
        if fr < vals[0] {
            let xe: Vec<f64> = (0..n).map(|k| cen[k] + 2.0 * (cen[k] - worst[k])).collect();
            let fe = f(&xe);
            if fe < fr { pts[n] = xe; vals[n] = fe } else { pts[n] = xr; vals[n] = fr }
        } else if fr < vals[n - 1] {
            pts[n] = xr; vals[n] = fr;
        } else {
            let xc: Vec<f64> = (0..n).map(|k| cen[k] + 0.5 * (worst[k] - cen[k])).collect();
            let fc = f(&xc);
            if fc < vals[n] { pts[n] = xc; vals[n] = fc }
            else {
                let p0 = pts[0].clone();
                for i in 1..=n {
                    for k in 0..n { pts[i][k] = p0[k] + 0.5 * (pts[i][k] - p0[k]) }
                    vals[i] = f(&pts[i]);
                }
            }
        }
    }
    let mut bi = 0;
    for i in 1..=n { if vals[i] < vals[bi] { bi = i } }
    (pts[bi].clone(), vals[bi])
}

fn fit(name: &str, m: Model, data: &[Pt]) -> (Vec<f64>, f64) {
    let (x0, s0) = m.start();
    // stage 1: distance residuals, several restarts
    let mut best = x0.clone();
    let mut bv = f64::INFINITY;
    for mult in [1.0f64, 0.5, 2.0, 4.0] {
        let s: Vec<f64> = s0.iter().map(|v| v * mult).collect();
        let (x, v) = nelder_mead(|x| cost_dist(data, &m, x), &best, &s, 3000);
        if v < bv { best = x; bv = v }
    }
    // stage 2: power residuals
    let mut bp = best.clone();
    let mut bpv = cost_power(data, &m, &bp);
    for _ in 0..8 {
        let s: Vec<f64> = bp.iter().enumerate()
            .map(|(i, v)| (v.abs() * 0.03).max(s0[i] * 0.1)).collect();
        let (x, v) = nelder_mead(|x| cost_power(data, &m, x), &bp, &s, 1200);
        if v < bpv { bp = x; bpv = v }
    }

    let p = m.unpack(&bp);
    let n = data.len() as f64;
    let dof = n - m.nparams() as f64;
    println!("\n================ {} ================", name);
    println!("kh={:.6}  kv={:.6}  g={:.6}  c={:.6}  P0={:.4}  h0={:.4}",
             p.kh, p.kv, p.g, p.c, p.p0, p.h0);
    println!("params={}  n={}  RMS={:.4}   RMS(adj,dof)={:.4}",
             m.nparams(), data.len(), (bpv / n).sqrt(), (bpv / dof).sqrt());

    // residual by angle -- THE diagnostic
    let mut worst = (0.0f64, 0.0, 0.0);
    print!("  mean err by angle: ");
    for &a in ANGLES.iter() {
        let v: Vec<f64> = data.iter().filter(|q| q.angle == a)
            .map(|q| power_for(q.angle, q.dist, &m, &p) - q.power).collect();
        if v.is_empty() { continue }
        let mm = v.iter().sum::<f64>() / v.len() as f64;
        print!("{:.0}:{:+.2} ", a, mm);
    }
    println!();
    for q in data {
        let e = power_for(q.angle, q.dist, &m, &p) - q.power;
        if e.abs() > worst.0.abs() { worst = (e, q.angle, q.dist) }
    }
    println!("  worst point: err={:+.3} at angle={} dist={}", worst.0, worst.1, worst.2);
    (bp, bpv)
}

fn main() {
    for &skip80 in &[false, true] {
        let data = load(skip80);
        println!("\n##################################################");
        println!("###   {}   (n = {})",
                 if skip80 { "EXCLUDING 80-degree column" } else { "ALL DATA" },
                 data.len());
        println!("##################################################");

        let base = Model { variant: 0, free_k: true, split_k: false, free_g: true,
                           free_h0: false, fix_k: 0.98, fix_g: 0.04 };

        fit("M1  discrete, no launch height", base, &data);

        let mut m = base; m.free_h0 = true;
        let (bx, _) = fit("M2  discrete + launch height h0", m, &data);

        let mut m3 = m; m3.split_k = true;
        fit("M3  + separate horizontal/vertical drag", m3, &data);

        let mut m4 = m; m4.variant = 1;
        fit("M4  h0, tick order: vy = vy*k - g", m4, &data);

        let mut m5 = m; m5.variant = 2;
        fit("M5  h0, tick order: move first", m5, &data);

        // the big test: are the pretty numbers exactly right?
        let mut m6 = m; m6.free_k = false; m6.free_g = false;
        m6.fix_k = 0.98; m6.fix_g = 0.04;
        fit("M6  k=0.98, g=0.04 FIXED (only c,P0,h0 free)", m6, &data);

        let mut m7 = m6; m7.fix_k = 0.99;
        fit("M7  k=0.99, g=0.04 FIXED", m7, &data);

        let mut m8 = m6; m8.fix_g = 0.05;
        fit("M8  k=0.98, g=0.05 FIXED", m8, &data);

        if !skip80 {
            // demo with the best free model
            let p = m.unpack(&bx);
            println!("\n=== demo (M2): dist 15, power 60, varying height diff ===");
            for &dy in &[-6.0f64, -3.0, 0.0, 3.0, 6.0] {
                let f = |a: f64| sim_height_at(a, 60.0, &m, &p, 15.0) - dy;
                let mut sols = Vec::new();
                let mut prev = 20.0f64; let mut fp = f(prev);
                let mut a = 20.5f64;
                while a <= 85.0 {
                    let fa = f(a);
                    if fp.is_finite() && fa.is_finite() && fp * fa < 0.0 {
                        let (mut lo, mut hi) = (prev, a);
                        for _ in 0..50 {
                            let mid = 0.5 * (lo + hi);
                            if f(lo) * f(mid) <= 0.0 { hi = mid } else { lo = mid }
                        }
                        sols.push(0.5 * (lo + hi));
                    }
                    prev = a; fp = fa; a += 0.5;
                }
                let s: Vec<String> = sols.iter().map(|v| format!("{:.2}", v)).collect();
                println!("  dy={:+5.1} -> [{}]", dy, s.join(", "));
            }
        }
    }
}
