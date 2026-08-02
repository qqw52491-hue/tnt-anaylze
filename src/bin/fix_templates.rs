// 一次性工具：清洗 3 的两张模板（顶部混入了隔壁字符的残块）
// 用法（在仓库根目录）: cargo run --bin fix_templates
use opencv::{core, imgcodecs, imgproc, prelude::*};
use std::path::Path;

// (文件, 需要从顶部抹掉的行数)
const JOBS: [(&str, i32); 2] = [
    ("src/templates/3_QQ_1785460784542_0.png", 10),
    ("src/templates/3_QQ_1785460804565_0.png", 7),
];

fn norm40(src: &core::Mat) -> opencv::Result<core::Mat> {
    let mut x0 = i32::MAX;
    let mut y0 = i32::MAX;
    let mut x1 = -1i32;
    let mut y1 = -1i32;
    for y in 0..src.rows() {
        for x in 0..src.cols() {
            if *src.at_2d::<u8>(y, x)? > 30 {
                if x < x0 {
                    x0 = x;
                }
                if x > x1 {
                    x1 = x;
                }
                if y < y0 {
                    y0 = y;
                }
                if y > y1 {
                    y1 = y;
                }
            }
        }
    }
    if x1 < 0 {
        return src.try_clone();
    }
    let w = x1 - x0 + 1;
    let h = y1 - y0 + 1;
    println!("    外接框 y={}..{} x={}..{}  w={} h={}", y0, y1, x0, x1, w, h);
    let crop = core::Mat::roi(src, core::Rect::new(x0, y0, w, h))?;
    let s = 36.0 / (w.max(h) as f64);
    let nw = ((w as f64 * s).round() as i32).max(1);
    let nh = ((h as f64 * s).round() as i32).max(1);
    let mut r = core::Mat::default();
    imgproc::resize(
        &crop,
        &mut r,
        core::Size::new(nw, nh),
        0.0,
        0.0,
        imgproc::INTER_AREA,
    )?;
    let mut out =
        core::Mat::new_rows_cols_with_default(40, 40, core::CV_8UC1, core::Scalar::all(0.0))?;
    let ox = (40 - nw) / 2;
    let oy = (40 - nh) / 2;
    for y in 0..nh {
        for x in 0..nw {
            *out.at_2d_mut::<u8>(oy + y, ox + x)? = *r.at_2d::<u8>(y, x)?;
        }
    }
    Ok(out)
}

fn main() -> opencv::Result<()> {
    if Path::new("src/templates/.cleaned3").exists() {
        println!("已经洗过了，跳过。要重来：git checkout src/templates && rm src/templates/.cleaned3");
        return Ok(());
    }
    for (path, cut) in JOBS {
        let src = imgcodecs::imread(path, imgcodecs::IMREAD_GRAYSCALE)?;
        if src.empty() {
            println!("读不到 {}，跳过", path);
            continue;
        }
        println!("处理 {}", path);
        println!("  清洗前:");
        let _ = norm40(&src)?;
        let mut fixed = src.try_clone()?;
        let n = cut.min(fixed.rows());
        for y in 0..n {
            for x in 0..fixed.cols() {
                *fixed.at_2d_mut::<u8>(y, x)? = 0;
            }
        }
        println!("  清洗后:");
        let out = norm40(&fixed)?;
        imgcodecs::imwrite(path, &out, &core::Vector::<i32>::new())?;
        println!("  已写回");
    }
    std::fs::write("src/templates/.cleaned3", b"ok").ok();
    println!("完成，直接重跑 live_gui");
    Ok(())
}
