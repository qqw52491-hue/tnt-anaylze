use image::GenericImageView;
use std::env;

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        return;
    }
    let img = image::open(&args[1]).unwrap();
    let rgba = img.to_rgba8();

    let mut dark_count = 0;
    println!("Scanning for dark pixels...");
    for y in 0..300 {
        for x in 0..300 {
            let p = rgba.get_pixel(x, y);
            let sum = p[0] as u32 + p[1] as u32 + p[2] as u32;
            if sum < 50 { // Very dark pixel
                dark_count += 1;
                if dark_count % 100 == 0 {
                    println!("Dark pixel at ({}, {}): R={}, G={}, B={}", x, y, p[0], p[1], p[2]);
                }
            }
        }
    }
    println!("Found {} dark pixels", dark_count);
}
