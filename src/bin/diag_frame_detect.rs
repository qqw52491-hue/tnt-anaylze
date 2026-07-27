use opencv::{
    core, imgcodecs, imgproc,
    prelude::*,
};

fn main() -> opencv::Result<()> {
    let imgs = vec![
        ("截屏2026-07-26 13.40.25.png", "frame_v9_1340.png"),
        ("截屏2026-07-26 11.27.46.png", "frame_v9_1127.png"),
    ];
    
    for (img_path, out_name) in imgs {
        println!("==== {} ====", img_path);
        let img = imgcodecs::imread(img_path, imgcodecs::IMREAD_COLOR)?;
        if img.empty() { continue; }
        let roi = core::Rect::new(0, 0, 226, 136);
        let minimap = core::Mat::roi(&img, roi)?;
        
        let rows = minimap.rows() as usize;
        let cols = minimap.cols() as usize;
        
        // 1. 计算每一列和每一行的“直线边缘强度”
        let mut col_scores: Vec<f64> = vec![0.0; cols];
        for x in 0..cols-1 {
            let mut total = 0.0;
            for y in 0..rows {
                let p1 = minimap.at_2d::<core::Vec3b>(y as i32, x as i32)?;
                let p2 = minimap.at_2d::<core::Vec3b>(y as i32, (x+1) as i32)?;
                total += (p1[0] as f64 - p2[0] as f64).abs()
                       + (p1[1] as f64 - p2[1] as f64).abs()
                       + (p1[2] as f64 - p2[2] as f64).abs();
            }
            col_scores[x] = total;
        }
        
        let mut row_scores: Vec<f64> = vec![0.0; rows];
        for y in 0..rows-1 {
            let mut total = 0.0;
            for x in 0..cols {
                let p1 = minimap.at_2d::<core::Vec3b>(y as i32, x as i32)?;
                let p2 = minimap.at_2d::<core::Vec3b>((y+1) as i32, x as i32)?;
                total += (p1[0] as f64 - p2[0] as f64).abs()
                       + (p1[1] as f64 - p2[1] as f64).abs()
                       + (p1[2] as f64 - p2[2] as f64).abs();
            }
            row_scores[y] = total;
        }
        
        // 2. 找到最强的候选线（利用局部最大值，防止相邻线互相干扰）
        let mut x_candidates = Vec::new();
        for x in 5..cols-5 {
            if col_scores[x] > col_scores[x-1] && col_scores[x] > col_scores[x+1] && col_scores[x] > 1000.0 {
                x_candidates.push((x, col_scores[x]));
            }
        }
        x_candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        
        let mut y_candidates = Vec::new();
        for y in 5..rows-5 {
            if row_scores[y] > row_scores[y-1] && row_scores[y] > row_scores[y+1] && row_scores[y] > 1000.0 {
                y_candidates.push((y, row_scores[y]));
            }
        }
        y_candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        
        // 3. 在候选线中寻找能组成合理“视野框尺寸”的组合
        // 已知视野框大小约：宽 80~105，高 55~75
        let mut best_score = 0.0;
        let mut best_rect = core::Rect::default();
        
        for i in 0..x_candidates.len() {
            for j in (i+1)..x_candidates.len() {
                let mut x1 = x_candidates[i].0;
                let mut x2 = x_candidates[j].0;
                if x1 > x2 { std::mem::swap(&mut x1, &mut x2); }
                
                let w = x2 - x1;
                // 限制宽度
                if w < 80 || w > 110 { continue; }
                
                for k in 0..y_candidates.len() {
                    for l in (k+1)..y_candidates.len() {
                        let mut y1 = y_candidates[k].0;
                        let mut y2 = y_candidates[l].0;
                        if y1 > y2 { std::mem::swap(&mut y1, &mut y2); }
                        
                        let h = y2 - y1;
                        // 限制高度
                        if h < 55 || h > 75 { continue; }
                        
                        // 计算矩形四条线段上的平均梯度
                        let mut top_score = 0.0;
                        for x in x1..x2 {
                            let p1 = minimap.at_2d::<core::Vec3b>(y1 as i32, x as i32)?;
                            let p2 = minimap.at_2d::<core::Vec3b>((y1+1) as i32, x as i32)?;
                            top_score += (p1[0] as f64 - p2[0] as f64).abs() + (p1[1] as f64 - p2[1] as f64).abs() + (p1[2] as f64 - p2[2] as f64).abs();
                        }
                        top_score /= w as f64;
                        
                        let mut bottom_score = 0.0;
                        for x in x1..x2 {
                            let p1 = minimap.at_2d::<core::Vec3b>(y2 as i32, x as i32)?;
                            let p2 = minimap.at_2d::<core::Vec3b>((y2+1) as i32, x as i32)?;
                            bottom_score += (p1[0] as f64 - p2[0] as f64).abs() + (p1[1] as f64 - p2[1] as f64).abs() + (p1[2] as f64 - p2[2] as f64).abs();
                        }
                        bottom_score /= w as f64;
                        
                        let mut left_score = 0.0;
                        for y in y1..y2 {
                            let p1 = minimap.at_2d::<core::Vec3b>(y as i32, x1 as i32)?;
                            let p2 = minimap.at_2d::<core::Vec3b>(y as i32, (x1+1) as i32)?;
                            left_score += (p1[0] as f64 - p2[0] as f64).abs() + (p1[1] as f64 - p2[1] as f64).abs() + (p1[2] as f64 - p2[2] as f64).abs();
                        }
                        left_score /= h as f64;
                        
                        let mut right_score = 0.0;
                        for y in y1..y2 {
                            let p1 = minimap.at_2d::<core::Vec3b>(y as i32, x2 as i32)?;
                            let p2 = minimap.at_2d::<core::Vec3b>(y as i32, (x2+1) as i32)?;
                            right_score += (p1[0] as f64 - p2[0] as f64).abs() + (p1[1] as f64 - p2[1] as f64).abs() + (p1[2] as f64 - p2[2] as f64).abs();
                        }
                        right_score /= h as f64;
                        
                        let score = top_score + bottom_score + left_score + right_score;
                        
                        if score > best_score {
                            best_score = score;
                            best_rect = core::Rect::new(x1 as i32, y1 as i32, w as i32, h as i32);
                        }
                    }
                }
            }
        }
        
        println!(">>> 最终检测视野框: ({}, {}), 尺寸: {}x{}, 边缘总分: {:.0}", best_rect.x, best_rect.y, best_rect.width, best_rect.height, best_score);
        
        // 画出来
        let mut big = core::Mat::default();
        imgproc::resize(&minimap, &mut big, core::Size::new(226*4, 136*4), 0.0, 0.0, imgproc::INTER_NEAREST)?;
        
        if best_score > 0.0 {
            let big_rect = core::Rect::new(best_rect.x * 4, best_rect.y * 4, best_rect.width * 4, best_rect.height * 4);
            imgproc::rectangle(&mut big, big_rect, core::Scalar::new(0.0, 255.0, 0.0, 0.0), 3, imgproc::LINE_8, 0)?;
            imgproc::put_text(&mut big, &format!("CAMERA {}x{}", best_rect.width, best_rect.height), 
                core::Point::new(big_rect.x + 10, big_rect.y - 10),
                imgproc::FONT_HERSHEY_SIMPLEX, 0.7, core::Scalar::new(0.0, 255.0, 0.0, 0.0), 2, imgproc::LINE_8, false)?;
        }
        
        imgcodecs::imwrite(out_name, &big, &core::Vector::new())?;
    }
    
    Ok(())
}
