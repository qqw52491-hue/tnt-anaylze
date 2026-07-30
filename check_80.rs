use std::f64::consts::PI;

const K: f64 = 0.983638;
const G: f64 = 0.039367;
const C: f64 = 0.014945;
const P0: f64 = 1.9176;
const H0: f64 = 0.0335;

fn sim(angle: f64, power: f64) -> f64 {
    let v0 = C * (power + P0);
    if v0 <= 0.0 { return 0.0; }
    let r = angle * PI / 180.0;
    let mut vx = v0 * r.cos();
    let mut vy = v0 * r.sin();
    let mut x = 0.0;
    let mut y = H0;
    for _ in 0..200000 {
        let (px, py) = (x, y);
        x += vx;
        y += vy;
        if y <= 0.0 && py > 0.0 {
            let t = py / (py - y);
            return px + t * (x - px);
        }
        vy = (vy - G) * K;
        vx *= K;
        if vx < 1e-14 && vy < 0.0 { return x; }
    }
    x
}

fn find_power(angle: f64, dist: f64) -> f64 {
    let mut lo = 0.0;
    let mut hi = 150.0;
    for _ in 0..50 {
        let mid = (lo + hi) / 2.0;
        if sim(angle, mid) < dist { lo = mid; } else { hi = mid; }
    }
    (lo + hi) / 2.0
}

fn main() {
    let actual = vec![20.0, 31.0, 42.0, 51.0, 60.0, 67.0, 74.0, 80.0, 88.0, 95.0, 100.0];
    let mut total_err = 0.0;
    for (i, &p) in actual.iter().enumerate() {
        let dist = (i + 1) as f64;
        let pred = find_power(80.0, dist);
        let err = pred - p;
        total_err += err.abs();
        println!("dist: {:>2}, actual: {:>5}, pred: {:>6.2}, err: {:>6.2}", dist, p, pred, err);
    }
    println!("Mean absolute error: {:.2}", total_err / actual.len() as f64);
}
