// 视野框检测调试:
//   test_frame <image.png>              全图自动检测 detect_camera_frame
//   test_frame <image.png> snap x y w h 模拟手画 roi,跑 snap_camera_frame 吸附
// 结果存 /tmp/frame_dbg_<name> (吸附时同时画出手画框=蓝、吸附结果=红)
use opencv::{core, imgcodecs, imgproc, prelude::*};

fn main() -> opencv::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("usage: test_frame <image> [snap x y w h]");
        std::process::exit(1);
    }
    let path = &args[1];
    let img = imgcodecs::imread(path, imgcodecs::IMREAD_COLOR)?;
    if img.empty() {
        eprintln!("cannot read {}", path);
        std::process::exit(1);
    }

    let (r, drawn) = if args.len() >= 7 && args[2] == "snap" {
        let roi = core::Rect::new(
            args[3].parse().unwrap(),
            args[4].parse().unwrap(),
            args[5].parse().unwrap(),
            args[6].parse().unwrap(),
        );
        let snapped = tnt_comput::detect::snap_camera_frame(&img, &roi)?;
        println!("{} snap {:?} -> {:?}", path, roi, snapped);
        (snapped, Some(roi))
    } else {
        let r = tnt_comput::detect::detect_camera_frame(&img)?;
        println!("{} -> {:?}", path, r);
        (r, None)
    };

    let mut vis = img.clone();
    if let Some(roi) = drawn {
        imgproc::rectangle(
            &mut vis,
            roi,
            core::Scalar::new(255.0, 120.0, 0.0, 0.0),
            1,
            imgproc::LINE_8,
            0,
        )?;
    }
    if let Some(r) = r {
        imgproc::rectangle(
            &mut vis,
            r,
            core::Scalar::new(0.0, 0.0, 255.0, 0.0),
            1,
            imgproc::LINE_8,
            0,
        )?;
    }
    let name = std::path::Path::new(path)
        .file_name()
        .unwrap()
        .to_string_lossy()
        .to_string();
    let out = format!("/tmp/frame_dbg_{}", name);
    imgcodecs::imwrite(&out, &vis, &core::Vector::new())?;
    println!("saved {}", out);
    Ok(())
}
