//! Lightweight native HUD drawing primitives. No capture or gameplay behavior.
use opencv::{core, imgproc};

pub type Color = [f64; 3]; // BGR
pub const BACKGROUND: Color = [25.0, 21.0, 18.0];
pub const SURFACE: Color = [37.0, 31.0, 26.0];
pub const RAISED: Color = [49.0, 41.0, 34.0];
pub const BORDER: Color = [62.0, 53.0, 45.0];
pub const TEXT: Color = [243.0, 237.0, 227.0];
pub const MUTED: Color = [159.0, 144.0, 128.0];
pub const ACCENT: Color = [192.0, 216.0, 79.0];
pub const ACCENT_DARK: Color = [61.0, 65.0, 33.0];
pub const POWER: Color = [181.0, 235.0, 152.0];

pub fn color(c: Color) -> core::Scalar { core::Scalar::new(c[0],c[1],c[2],0.0) }

pub fn rounded(canvas: &mut core::Mat, r: core::Rect, radius: i32, c: Color) -> opencv::Result<()> {
    let radius = radius.min(r.width/2).min(r.height/2).max(0);
    if radius == 0 { return imgproc::rectangle(canvas,r,color(c),-1,imgproc::LINE_AA,0); }
    imgproc::rectangle(canvas,core::Rect::new(r.x+radius,r.y,r.width-2*radius,r.height),color(c),-1,imgproc::LINE_AA,0)?;
    imgproc::rectangle(canvas,core::Rect::new(r.x,r.y+radius,r.width,r.height-2*radius),color(c),-1,imgproc::LINE_AA,0)?;
    for (x,y) in [(r.x+radius,r.y+radius),(r.x+r.width-radius-1,r.y+radius),
                  (r.x+radius,r.y+r.height-radius-1),(r.x+r.width-radius-1,r.y+r.height-radius-1)] {
        imgproc::circle(canvas,core::Point::new(x,y),radius,color(c),-1,imgproc::LINE_AA,0)?;
    }
    Ok(())
}

pub fn card(canvas: &mut core::Mat, r: core::Rect) -> opencv::Result<()> {
    rounded(canvas,r,12,BORDER)?;
    rounded(canvas,core::Rect::new(r.x+1,r.y+1,r.width-2,r.height-2),11,SURFACE)
}

pub fn fit(text: &str, width: i32, scale: f64, thickness: i32) -> opencv::Result<String> {
    let chars: Vec<_> = text.chars().map(|c| if c.is_control() {' '} else {c}).collect();
    let clean: String = chars.iter().collect();
    let mut baseline = 0;
    let fits = |s: &str, baseline: &mut i32| -> opencv::Result<bool> {
        Ok(imgproc::get_text_size(s,imgproc::FONT_HERSHEY_SIMPLEX,scale,thickness,baseline)?.width <= width)
    };
    if fits(&clean,&mut baseline)? { return Ok(clean); }
    let (mut lo,mut hi) = (0,chars.len());
    while lo < hi {
        let mid = (lo+hi+1)/2;
        let candidate = chars[..mid].iter().collect::<String>() + "...";
        if fits(&candidate,&mut baseline)? { lo=mid; } else { hi=mid-1; }
    }
    let shortened=chars[..lo].iter().collect::<String>()+"...";
    Ok(if fits(&shortened,&mut baseline)? {shortened} else {String::new()})
}

pub fn text(canvas: &mut core::Mat, text: &str, x: i32, y: i32, width: i32, scale: f64, c: Color) -> opencv::Result<()> {
    let text=fit(text,width,scale,1)?;
    imgproc::put_text(canvas,&text,core::Point::new(x,y),imgproc::FONT_HERSHEY_SIMPLEX,
        scale,color(c),1,imgproc::LINE_AA,false)
}

pub fn button(canvas: &mut core::Mat, r: core::Rect, label: &str, active: bool) -> opencv::Result<()> {
    rounded(canvas,r,6,if active {ACCENT} else {BORDER})?;
    rounded(canvas,core::Rect::new(r.x+1,r.y+1,r.width-2,r.height-2),5,
        if active {ACCENT_DARK} else {RAISED})?;
    let scale=if r.height<=24 {0.36} else {0.43};
    let label=fit(label,r.width-if r.width<55 {6} else {16},scale,1)?;
    let mut baseline=0;
    let size=imgproc::get_text_size(&label,imgproc::FONT_HERSHEY_SIMPLEX,scale,1,&mut baseline)?;
    text(canvas,&label,r.x+(r.width-size.width)/2,r.y+(r.height+size.height)/2,
        r.width-4,scale,if active {ACCENT} else {TEXT})
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn text_fit_respects_pixel_width_and_unicode_boundaries() -> opencv::Result<()> {
        let fitted=fit("很长的角色名字🙂\n重复名字",100,0.43,1)?;
        let mut baseline=0;
        assert!(!fitted.contains('\n'));
        assert!(imgproc::get_text_size(&fitted,imgproc::FONT_HERSHEY_SIMPLEX,0.43,1,&mut baseline)?.width<=100);
        Ok(())
    }
}
