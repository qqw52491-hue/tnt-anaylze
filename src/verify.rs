// verify.rs  --  rustc -O verify.rs -o verify && ./verify
//
// Two things, both requiring NO new data and NO definition of a height unit:
//
//  PART A: unit-free trajectory shape.  Everything is printed as a RATIO
//          (height / range), so you can check it against a screenshot with
//          a ruler and never define a vertical unit at all.
//
//  PART B: cross-check your empirically-validated wind formula
//              shift = wind * (angle/160) * (2/3)
//          against the physical wind model, and solve for the single
//          WIND_SCALE that makes them agree. If one scale works across all
//          angles, both models are right and the scale is free.

use std::f64::consts::PI;

// M5 constants
const K: f64 = 0.983638;
const G: f64 = 0.039367;
const C: f64 = 0.014945;
const P0: f64 = 1.9176;
const H0: f64 = 0.0335;

const ANGLES: [f64; 11] = [20., 25., 30., 35., 40., 45., 50., 55., 60., 65., 70.];
const MAX_TICKS: usize = 200_000;

#[derive(Clone, Copy)]
struct S { x: f64, y: f64, vx: f64, vy: f64 }

fn launch(angle: f64, power: f64) -> S {
    let v0 = C * (power + P0);
    let r = angle * PI / 180.0;
    S { x: 0.0, y: H0, vx: v0 * r.cos(), vy: v0 * r.sin() }
}

#[inline]
fn step(s: &mut S, w: f64) {
    s.x += s.vx;
    s.y += s.vy;
    s.vy = (s.vy - G) * K;
    s.vx = (s.vx - w) * K + w;
}

/// Whole flight until it comes back down to y = 0.
fn flight(angle: f64, power: f64, w: f64) -> Vec<(f64, f64)> {
    let mut s = launch(angle, power);
    let mut out = vec![(s.x, s.y)];
    for _ in 0..MAX_TICKS {
        let p = s;
        step(&mut s, w);
        if s.y <= 0.0 && p.y > 0.0 {
            let t = p.y / (p.y - s.y);
            out.push((p.x + t * (s.x - p.x), 0.0));
            break;
        }
        out.push((s.x, s.y));
        if s.vy < 0.0 && (s.x - p.x).abs() < 1e-13 { break }
    }
    out
}

fn range_of(angle: f64, power: f64, w: f64) -> f64 {
    let f = flight(angle, power, w);
    f.last().map(|v| v.0).unwrap_or(0.0)
}

/// Height at horizontal position xt, by interpolating the flight path.
fn height_at_path(path: &[(f64, f64)], xt: f64) -> f64 {
    if xt <= path[0].0 { return path[0].1 }
    for i in 1..path.len() {
        if path[i].0 >= xt {
            let (x0, y0) = path[i - 1];
            let (x1, y1) = path[i];
            let t = if x1 > x0 { (xt - x0) / (x1 - x0) } else { 0.0 };
            return y0 + t * (y1 - y0);
        }
    }
    path[path.len() - 1].1
}

/// Power that lands exactly at `dist` on flat ground.
fn power_for(angle: f64, dist: f64, w: f64) -> Option<f64> {
    let f = |p: f64| range_of(angle, p, w) - dist;
    let lo = 1e-9f64;
    let mut hi = 20.0f64;
    let mut g = 0;
    while f(hi) < 0.0 { hi *= 2.0; g += 1; if g > 40 { return None } }
    let (mut a, mut b) = (lo, hi);
    for _ in 0..60 {
        let m = 0.5 * (a + b);
        if f(m) < 0.0 { a = m } else { b = m }
    }
    Some(0.5 * (a + b))
}

fn main() {
    // ================= PART A: unit-free shape =================
    println!("###########################################################");
    println!("### PART A  unit-free trajectory shape");
    println!("###   every number is a RATIO, so no height unit needed.");
    println!("###   measure the same ratios on a screenshot to check.");
    println!("###########################################################");

    println!("\n--- apex height / range  (the single easiest thing to check) ---");
    println!("angle  dist=8            dist=15           dist=22");
    for &a in ANGLES.iter() {
        print!("{:5.0}  ", a);
        for &d in &[8.0f64, 15.0, 22.0] {
            match power_for(a, d, 0.0) {
                Some(pw) => {
                    let path = flight(a, pw, 0.0);
                    let r = path.last().unwrap().0;
                    let apex = path.iter().fold(0.0f64, |m, v| m.max(v.1));
                    let apex_x = path.iter().fold((0.0f64, 0.0f64), |m, v| if v.1 > m.1 { (v.0, v.1) } else { m }).0;
                    print!("h/R={:.4} xa/R={:.3}   ", apex / r, apex_x / r);
                }
                None => print!("  unreachable        "),
            }
        }
        println!();
    }

    println!("\n--- normalised path:  y/R at x/R = 0.1 .. 1.0 ---");
    println!("(overlay this on a screenshot; R = the landing distance you see)");
    for &d in &[8.0f64, 15.0, 22.0] {
        println!("\n  target distance {} units:", d);
        print!("  angle |");
        for i in 1..=10 { print!("  {:.1} ", i as f64 * 0.1) }
        println!();
        for &a in ANGLES.iter() {
            if let Some(pw) = power_for(a, d, 0.0) {
                let path = flight(a, pw, 0.0);
                let r = path.last().unwrap().0;
                print!("  {:5.0} |", a);
                for i in 1..=10 {
                    let x = r * i as f64 * 0.1;
                    print!(" {:.3}", height_at_path(&path, x) / r);
                }
                println!("   (power {:.1})", pw);
            }
        }
    }

    println!("\n--- asymmetry check: rise distance vs fall distance ---");
    println!("(a drag-free parabola is perfectly symmetric -> ratio exactly 0.5;");
    println!(" the further this is BELOW 0.5, the more drag there is.");
    println!(" this is measurable on ONE screenshot and pins the drag down.)");
    println!("angle   xa/R at dist=8   dist=15   dist=22");
    for &a in ANGLES.iter() {
        print!("{:5.0}   ", a);
        for &d in &[8.0f64, 15.0, 22.0] {
            match power_for(a, d, 0.0) {
                Some(pw) => {
                    let path = flight(a, pw, 0.0);
                    let r = path.last().unwrap().0;
                    let ax = path.iter().fold((0.0f64, 0.0f64), |m, v| if v.1 > m.1 { (v.0, v.1) } else { m }).0;
                    print!("{:11.4}", ax / r);
                }
                None => print!("          -"),
            }
        }
        println!();
    }

    // ================= PART B: wind cross-check =================
    println!("\n\n###########################################################");
    println!("### PART B  does the physical wind model reproduce YOUR");
    println!("###         validated formula  shift = w*(angle/160)*(2/3) ?");
    println!("###########################################################");

    let winds = [2.0f64, 4.0, 6.0, 8.0];
    let dists = [8.0f64, 15.0, 22.0];

    // least squares for one global WIND_SCALE
    let mut best = (f64::INFINITY, 0.0f64);
    let mut scale = 0.0005f64;
    while scale <= 0.05 {
        let mut sse = 0.0;
        let mut n = 0;
        for &d in dists.iter() {
            for &a in ANGLES.iter() {
                if let Some(pw) = power_for(a, d, 0.0) {
                    let base = range_of(a, pw, 0.0);
                    for &wv in winds.iter() {
                        let phys = range_of(a, pw, wv * scale) - base;
                        let yours = wv * (a / 160.0) * (2.0 / 3.0);
                        let e = phys - yours;
                        sse += e * e;
                        n += 1;
                    }
                }
            }
        }
        if n > 0 && sse / (n as f64) < best.0 {
            best = (sse / n as f64, scale);
        }
        scale *= 1.02;
    }
    let ws = best.1;
    println!("\nbest-fit WIND_SCALE = {:.6}   (RMS mismatch = {:.4} distance units)",
             ws, best.0.sqrt());
    println!("\nshift comparison at distance 15  (yours vs physical):");
    println!("angle    w=2            w=4            w=6            w=8");
    for &a in ANGLES.iter() {
        if let Some(pw) = power_for(a, 15.0, 0.0) {
            let base = range_of(a, pw, 0.0);
            print!("{:5.0}  ", a);
            for &wv in winds.iter() {
                let phys = range_of(a, pw, wv * ws) - base;
                let yours = wv * (a / 160.0) * (2.0 / 3.0);
                print!("{:5.2}/{:5.2}   ", yours, phys);
            }
            println!();
        }
    }
    println!("\nHOW TO READ THIS:");
    println!(" - if the two columns track each other closely at EVERY angle,");
    println!("   your formula is the linearised version of real drag, and");
    println!("   WIND_SCALE above is calibrated for free. Keep using yours.");
    println!(" - if they diverge at the extremes (20 and 70), your formula is");
    println!("   a local fit that happens to work in the range you tested.");
    println!("   It will drift once height difference changes the flight time.");
    println!(" - note the physical shift also depends on DISTANCE, while yours");
    println!("   depends only on angle. Compare across dist=8/15/22 above.");

    println!("\n--- does the physical wind shift depend on distance? ---");
    println!("(yours says no. if physics says yes, that is a testable difference)");
    println!("angle    shift at w=6, dist=8 / 15 / 22");
    for &a in ANGLES.iter() {
        print!("{:5.0}   ", a);
        for &d in dists.iter() {
            match power_for(a, d, 0.0) {
                Some(pw) => {
                    let s = range_of(a, pw, 6.0 * ws) - range_of(a, pw, 0.0);
                    print!("{:8.3}", s);
                }
                None => print!("       -"),
            }
        }
        let yours = 6.0 * (a / 160.0) * (2.0 / 3.0);
        println!("     (yours: {:.3} for all)", yours);
    }
}
