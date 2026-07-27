use opencv::{core, imgcodecs, imgproc, prelude::*};

fn main() -> opencv::Result<()> {
    let img = imgcodecs::imread("截屏2026-07-26 11.27.46.png", imgcodecs::IMREAD_COLOR)?;
    let template_red = imgcodecs::imread("template_red_dot.png", imgcodecs::IMREAD_COLOR)?;
    
    let mut result_red = core::Mat::default();
    imgproc::match_template(&img, &template_red, &mut result_red, imgproc::TM_CCOEFF_NORMED, &core::no_array())?;
    
    // threshold
    let mut thresholded = core::Mat::default();
    imgproc::threshold(&result_red, &mut thresholded, 0.8, 1.0, imgproc::THRESH_BINARY)?;
    
    // convert to 8-bit to find contours or non-zero
    let mut thresholded_8u = core::Mat::default();
    thresholded.convert_to(&mut thresholded_8u, core::CV_8U, 255.0, 0.0)?;
    
    let mut non_zero = core::Mat::default();
    core::find_non_zero(&thresholded_8u, &mut non_zero)?;
    
    println!("Found {} pixels above threshold for red dot.", non_zero.total());
    Ok(())
}
