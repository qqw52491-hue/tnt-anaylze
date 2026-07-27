use opencv::{
    core, imgcodecs, imgproc,
    prelude::*,
};

fn main() -> opencv::Result<()> {
    let img = imgcodecs::imread("截屏2026-07-26 11.27.46.png", imgcodecs::IMREAD_COLOR)?;
    let roi = core::Rect::new(0, 0, 226, 136);
    let minimap = core::Mat::roi(&img, roi)?;
    
    // The known true blue dot in the old image is at (134, 30).
    // Let's inspect a 10x10 region around it to see the black border.
    let x = 134;
    let y = 30;
    
    println!("=== 观察 (134, 30) 附近的像素 (BGR格式) ===");
    for dy in -5..=5 {
        for dx in -5..=5 {
            let p = minimap.at_2d::<core::Vec3b>(y + dy, x + dx)?;
            // If the pixel is very dark (black edge), print it as 'B'
            // If it's blue, print 'O'
            // Otherwise print '.'
            let b = p[0] as i32;
            let g = p[1] as i32;
            let r = p[2] as i32;
            
            print!("{:02X}{:02X}{:02X} ", r, g, b); // Print RGB as HEX
        }
        println!();
    }
    
    Ok(())
}
