pub fn get_abc_for_angle(target_angle: f64) -> (f64, f64, f64) {
    let points = [
        (20.0, 16.836, 0.275, -4.069),
        (25.0, 16.119, 0.303, -5.326),
        (30.0, 13.774, 0.482, -2.685),
        (35.0, 13.546, 0.538, -3.464),
        (40.0, 14.424, 0.339, -4.487),
        (45.0, 13.565, 0.499, -3.039),
        (50.0, 14.734, 0.385, -4.422),
        (55.0, 14.057, 0.727, -4.472),
        (60.0, 13.837, 0.938, -2.911),
        (65.0, 14.514, 1.159, -2.625),
        (70.0, 15.882, 1.463, -3.232),
        (80.0, 23.450, 2.661, -6.573),
    ];
    for i in 0..points.len() - 1 {
        let (x0, a0, b0, c0) = points[i];
        let (x1, a1, b1, c1) = points[i + 1];
        if target_angle >= x0 && target_angle <= x1 {
            if target_angle == x1 { return (a1, b1, c1); }
            if target_angle == x0 { return (a0, b0, c0); }
            if x0 == 70.0 && x1 == 80.0 {
                let ratio = (target_angle - x0) / (x1 - x0);
                let a_interpolated = a0 + ratio * (a1 - a0) - (ratio * (1.0 - ratio) * 6.0);
                return (a_interpolated, b0 + ratio * (b1 - b0), c0 + ratio * (c1 - c0));
            }
            let ratio = (target_angle - x0) / (x1 - x0);
            return (a0 + ratio * (a1 - a0), b0 + ratio * (b1 - b0), c0 + ratio * (c1 - c0));
        }
    }
    let a_80 = 23.450;
    let b_80 = 2.661;
    let c_80 = -6.573;
    let sin_160 = (160.0_f64.to_radians()).sin();
    let sin_2x = ((target_angle * 2.0).to_radians()).sin();
    let a_extrapolated = if sin_2x > 0.0 { a_80 * (sin_160 / sin_2x).sqrt() } else { 999.0 };
    (a_extrapolated, b_80, c_80)
}

pub fn get_wind_distance_impact(angle: f64, wind: f64) -> f64 {
    // 按照反馈，风力影响过大（现在的4相当于之前的6），将风力影响等比例缩小到原来的三分之二 (2/3)
    wind * (angle / 160.0) * (2.0 / 3.0)
}

pub fn calc_power(angle: f64, distance: f64, wind: f64) -> f64 {
    let (a, b, c) = get_abc_for_angle(angle);
    let wind_shift = get_wind_distance_impact(angle, wind);
    let d_eff = distance - wind_shift;
    let d_eff = if d_eff < 0.0 { 0.0 } else { d_eff };
    a * d_eff.sqrt() + b * d_eff + c
}

pub fn calc_angle(distance: f64, power: f64, wind: f64, hint_angle: f64) -> f64 {
    let mut best_angle = hint_angle;
    let mut min_diff = f64::MAX;
    
    let mut test_angle = (hint_angle - 15.0).max(15.0);
    let max_test = (hint_angle + 15.0).min(89.0);
    
    while test_angle <= max_test {
        let p = calc_power(test_angle, distance, wind);
        let diff = (p - power).abs();
        if diff < min_diff {
            min_diff = diff;
            best_angle = test_angle;
        }
        test_angle += 0.1;
    }
    best_angle
}

pub fn calc_wind_power_difference(angle: f64, distance: f64, wind: f64) -> f64 {
    let power_no_wind = calc_power(angle, distance, 0.0);
    let power_with_wind = calc_power(angle, distance, wind);
    power_with_wind - power_no_wind
}
