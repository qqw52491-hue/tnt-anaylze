use opencv::{imgcodecs, prelude::*};
fn main() {
    let rec = tnt_comput::ui::UiRecognizer::new("src/templates").unwrap();
    for (p, expect) in [("/tmp/dial1.png",43),("/tmp/dial2.png",61),
                        ("/tmp/d3_6.png",6),("/tmp/d4_23.png",23),("/tmp/d5_98.png",98)] {
        let m = imgcodecs::imread(p, imgcodecs::IMREAD_COLOR).unwrap();
        match rec.recognize_angle_dial(&m) {
            Ok(v) => println!("{} expect={} -> {:?}", p, expect, v),
            Err(e) => println!("{} err: {}", p, e),
        }
    }
}
