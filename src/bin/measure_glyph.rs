use image::{GrayImage, Luma};
use std::fs;

/// 用水平游程的中位数估计平均笔画宽度（竖笔给短游程，对粗细最敏感）
fn stroke_width(img: &GrayImage, thresh: u8) -> f32 {
    let (w, h) = img.dimensions();
    let mut runs: Vec<u32> = Vec::new();
    for y in 0..h {
        let mut run = 0u32;
        for x in 0..w {
            if img.get_pixel(x, y).0[0] > thresh {
                run += 1;
            } else if run > 0 {
                runs.push(run);
                run = 0;
            }
        }
        if run > 0 { runs.push(run); }
    }
    if runs.is_empty() { return 0.0; }
    runs.sort_unstable();
    runs[runs.len() / 2] as f32
}

/// 从四边洪水填充背景；填不到的背景像素 = 被笔画包围的内孔
fn counter_area(img: &GrayImage, thresh: u8) -> u32 {
    let (w, h) = (img.width() as i32, img.height() as i32);
    let idx = |x: i32, y: i32| (y * w + x) as usize;
    let is_bg = |x: i32, y: i32| img.get_pixel(x as u32, y as u32).0[0] <= thresh;

    let mut seen = vec![false; (w * h) as usize];
    let mut stack: Vec<(i32, i32)> = Vec::new();
    for x in 0..w { stack.push((x, 0)); stack.push((x, h - 1)); }
    for y in 0..h { stack.push((0, y)); stack.push((w - 1, y)); }

    while let Some((x, y)) = stack.pop() {
        if x < 0 || y < 0 || x >= w || y >= h { continue; }
        if seen[idx(x, y)] || !is_bg(x, y) { continue; }
        seen[idx(x, y)] = true;
        stack.push((x + 1, y));
        stack.push((x - 1, y));
        stack.push((x, y + 1));
        stack.push((x, y - 1));
    }

    let mut holes = 0;
    for y in 0..h {
        for x in 0..w {
            if !seen[idx(x, y)] && is_bg(x, y) { holes += 1; }
        }
    }
    holes
}

/// 前景外接框高度
fn glyph_height(img: &GrayImage, thresh: u8) -> u32 {
    let (w, h) = img.dimensions();
    let (mut y0, mut y1) = (h, 0u32);
    for y in 0..h {
        for x in 0..w {
            if img.get_pixel(x, y).0[0] > thresh {
                y0 = y0.min(y);
                y1 = y1.max(y);
            }
        }
    }
    if y1 >= y0 { y1 - y0 + 1 } else { 0 }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 阈值 0 = "任何非纯黑" 都算墨水，因为 MNIST 归一化后任何非零像素都是实质特征
    const T: u8 = 0;

    let mut files: Vec<_> = fs::read_dir("src/pic_mnist28")?
        .filter_map(Result::ok)
        .map(|e| e.path())
        .filter(|p| p.extension().map_or(false, |e| e == "png"))
        .collect();
    files.sort();

    println!("{:<34} {:>6} {:>7} {:>8} {:>7}", "file", "height", "stroke", "stroke%", "hole");
    println!("{}", "-".repeat(66));

    for p in files {
        let img = image::open(&p)?.to_luma8();
        let h = glyph_height(&img, T);
        let sw = stroke_width(&img, T);
        let pct = if h > 0 { sw / h as f32 * 100.0 } else { 0.0 };
        let hole = counter_area(&img, T);
        println!(
            "{:<34} {:>6} {:>7.1} {:>7.1}% {:>7}",
            p.file_name().unwrap().to_string_lossy(), h, sw, pct, hole
        );
    }
    Ok(())
}
