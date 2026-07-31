use std::fs;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ground_truth = vec![
        ("QQ_1785395550263.png", 46),
        ("QQ_1785395577212.png", 45),
        ("QQ_1785395607716.png", 49),
        ("QQ_1785395679500.png", 76),
        ("QQ_1785395719748.png", 61),
        ("QQ_1785395741836.png", 62),
        ("QQ_1785400893053.png", 59),
        ("QQ_1785401461142.png", 80),
        ("QQ_1785457855758.png", 45),
        ("QQ_1785457866731.png", 45),
        ("QQ_1785460158243.png", 49),
        ("QQ_1785460167434.png", 80),
        ("QQ_1785460784542.png", 35),
        ("QQ_1785460794325.png", 49),
        ("QQ_1785460804565.png", 36),
        ("QQ_1785461834029.png", 74),
        ("QQ_1785476343893.png", 79),
    ];

    fs::create_dir_all("src/templates")?;

    for (fname, expected_val) in &ground_truth {
        let digits = [expected_val / 10, expected_val % 10];

        for i in 0..2 {
            let src_path = format!(
                "src/pic_template40/{}_{}_40.png",
                fname.trim_end_matches(".png"),
                i
            );
            if std::path::Path::new(&src_path).exists() {
                let out_path = format!(
                    "src/templates/{}_{}_{}.png",
                    digits[i],
                    fname.trim_end_matches(".png"),
                    i
                );
                fs::copy(&src_path, &out_path)?;
                println!("Saved template {}", out_path);
            }
        }
    }
    Ok(())
}
