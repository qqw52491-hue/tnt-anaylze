use std::env;
use std::f64::consts::PI;

const K: f64 = 0.983638;
const G: f64 = 0.039367;
const C: f64 = 0.014945;
const P0: f64 = 1.9176;
const H0: f64 = 0.0335;
const WIND_SCALE: f64 = 0.016220;

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 4 {
        eprintln!("Usage: {} <angle> <power> <wind>", args[0]);
        std::process::exit(1);
    }
    let angle: f64 = args[1].parse().expect("Invalid angle");
    let power: f64 = args[2].parse().expect("Invalid power");
    let wind: f64 = args[3].parse().expect("Invalid wind");

    let v0 = C * (power + P0);
    let r = angle * PI / 180.0;
    let mut vx = v0 * r.cos();
    let mut vy = v0 * r.sin();

    // 风力转化为物理引擎中每一帧的水平加速度
    let ax = wind * WIND_SCALE;

    let mut x = 0.0;
    let mut y = H0;

    let mut pts = vec![(x, y)];

    for _ in 0..1000 {
        let (px, py) = *pts.last().unwrap();
        x += vx;
        y += vy;

        if y < 0.0 {
            // 插值计算落地点的精确坐标
            let t = py / (py - y);
            pts.push((px + t * (x - px), 0.0));
            break;
        }
        pts.push((x, y));

        // 游戏引擎每帧的物理更新法则
        vy = (vy - G) * K;
        vx = (vx - ax) * K + ax; // 正确的风力计算公式
    }

    // 统计最高点和最远点（逆风太大会导致 x 往后退产生负数）
    let final_x = pts.last().unwrap().0;
    let mut max_y = 0.0;
    let mut max_y_x = 0.0;
    let mut min_x = 0.0_f64;
    let mut max_x = 0.0_f64;
    for p in &pts {
        if p.1 > max_y {
            max_y = p.1;
            max_y_x = p.0;
        }
        if p.0 > max_x {
            max_x = p.0;
        }
        if p.0 < min_x {
            min_x = p.0;
        }
    }

    println!(
        "=== 弹道分析: 角度 {}° / 力度 {} / 风力 {} ===",
        angle, power, wind
    );
    println!("总飞行帧数: {} ticks", pts.len() - 1);
    println!(
        "最高点高度: {:.2} 格 (出现在水平距离 {:.2} 格处)",
        max_y, max_y_x
    );
    println!("最终落点距: {:.2} 格", final_x);

    // 绘制简易字符画
    let width = 60;
    let height = 15;
    let mut grid = vec![vec![' '; width]; height];

    let x_range = (max_x - min_x).max(1.0);

    // 如果 X 有负数区间，我们画一条纵向虚线代表发射起点 X=0
    let zero_col = if min_x < 0.0 {
        (((-min_x) / x_range) * (width as f64 - 1.0)).round() as usize
    } else {
        0
    };
    if zero_col > 0 && zero_col < width {
        for r in 0..height {
            grid[r][zero_col] = '|'; // 起点轴
        }
    }

    for p in &pts {
        let col = (((p.0 - min_x) / x_range) * (width as f64 - 1.0)).round() as usize;
        let row = ((1.0 - p.1 / max_y) * (height as f64 - 1.0)).round() as usize;
        if col < width && row < height {
            grid[row][col] = '*';
        }
    }

    println!("\n=== 抛物线截面图 (自适应缩放) ===");
    println!("^{}", "-".repeat(width));
    for r in 0..height {
        print!("|");
        for c in 0..width {
            print!("{}", grid[r][c]);
        }
        println!();
    }
    println!("+{}", "-".repeat(width));

    println!("\n=== 核心坐标数据 (每 10 帧采样) ===");
    for (i, p) in pts.iter().enumerate() {
        if i % 10 == 0 || i == pts.len() - 1 {
            println!(
                "Tick {:>3}:  X(水平) = {:>6.2},  Y(高度) = {:>6.2}",
                i, p.0, p.1
            );
        }
    }
}
