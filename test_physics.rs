fn compute_force(dx_units: f64, dy_units: f64, angle_deg: f64, wind: f64) -> f64 {
    let mut eff_angle = angle_deg;
    let is_reverse = eff_angle > 90.0;
    if is_reverse { eff_angle = 180.0 - eff_angle; }
    
    // Empirical Tables
    let tables = [
        (20.0, [10.0, 19.0, 25.0, 30.0, 36.0, 40.0, 44.0, 48.0, 51.0, 54.0, 57.0, 60.0, 63.0, 66.0, 69.0, 72.0, 74.0, 76.0, 78.0, 80.0]),
        (30.0, [14.0, 20.0, 24.7, 28.7, 32.3, 35.7, 38.8, 41.8, 44.7, 47.5, 50.2, 52.8, 55.3, 57.9, 60.3, 62.7, 65.7, 67.5, 69.8, 72.1]),
        (45.0, [13.0, 16.0, 20.0, 25.0, 30.0, 33.0, 35.0, 38.0, 41.0, 45.0, 48.0, 51.0, 53.0, 55.0, 57.0, 59.0, 60.0, 63.0, 66.0, 68.0]),
        (50.0, [14.1, 20.1, 24.8, 28.8, 32.5, 35.9, 39.0, 42.0, 44.9, 48.3, 50.5, 53.0, 55.5, 58.0, 60.5, 63.0, 65.5, 68.0, 70.0, 72.5]),
        (65.0, [13.0, 21.0, 26.0, 32.0, 37.0, 41.0, 44.0, 49.0, 53.0, 56.0, 58.0, 61.0, 64.0, 67.0, 70.0, 73.0, 76.0, 79.0, 82.0, 85.0]),
        (70.0, [10.0, 20.0, 30.0, 40.0, 45.0, 49.0, 55.0, 60.0, 64.0, 68.0, 73.0, 78.0, 83.0, 86.0, 90.0, 93.0, 95.0, 97.0, 99.0, 100.0]),
    ];

    let mut dist = dx_units.abs();
    
    // Wind compensation: Game typically treats 1.0 wind as a distance offset depending on angle
    // Standard rule: 1 wind = angle compensation, or distance compensation
    // We will adjust the "effective distance" based on wind to leverage the tables.
    // For 30 degree, 1 wind is approx 0.5 dist (or angle).
    // Let's implement angle compensation for wind:
    // In TNT, the rule is usually "change angle, keep force" or "change force, keep angle".
    // If the user wants to KEEP the angle and adjust force, we can adjust the "effective distance".
    let wind_effect = wind * (if is_reverse { -1.0 } else { 1.0 });
    // Empirical: 1 wind +/- reduces/adds about 0.5 to 1.0 distance unit requirement
    let wind_dist_compensation = wind_effect * 0.8; 
    
    // Height compensation: High place -> needs less force, Low place -> needs more force
    // Empirical: 1 unit of Y difference = 1 unit of X distance
    let height_dist_compensation = dy_units * 1.0; 
    
    let mut eff_dist = dist - wind_dist_compensation + height_dist_compensation;
    
    if eff_dist < 1.0 { eff_dist = 1.0; }
    if eff_dist >= 20.0 { eff_dist = 19.999; }
    
    let dist_idx = (eff_dist as usize) - 1;
    let dist_frac = eff_dist - eff_dist.floor();
    
    let mut dist_forces = [0.0; 6];
    for (i, &(_, ref forces)) in tables.iter().enumerate() {
        let f1 = forces[dist_idx];
        let f2 = forces[dist_idx + 1];
        dist_forces[i] = f1 + (f2 - f1) * dist_frac;
    }
    
    let mut base_force = 0.0;
    if eff_angle <= tables[0].0 {
        base_force = dist_forces[0];
    } else if eff_angle >= tables[5].0 {
        base_force = dist_forces[5];
    } else {
        for i in 0..5 {
            if eff_angle >= tables[i].0 && eff_angle <= tables[i+1].0 {
                let a1 = tables[i].0;
                let a2 = tables[i+1].0;
                let f1 = dist_forces[i];
                let f2 = dist_forces[i+1];
                base_force = f1 + (f2 - f1) * (eff_angle - a1) / (a2 - a1);
                break;
            }
        }
    }
    
    base_force.clamp(1.0, 100.0)
}

fn main() {
    println!("{}", compute_force(10.0, 0.0, 37.0, 0.0));
    println!("{}", compute_force(15.0, 0.0, 37.0, 0.0));
}
