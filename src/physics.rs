//! Selectable HUD physics. Default: frozen recorded-trajectory candidate.
//! TNT_PHYSICS=legacy restores the preserved pre-change implementation.
#[path = "physics/legacy.rs"]
pub mod legacy;
#[path = "physics/recorded.rs"]
mod recorded;
use std::sync::OnceLock;

pub const MIN_ANGLE: f64 = 15.0;
pub const MAX_ANGLE: f64 = 89.0;

fn use_legacy() -> bool {
    static LEGACY: OnceLock<bool> = OnceLock::new();
    *LEGACY.get_or_init(|| match std::env::var("TNT_PHYSICS").as_deref() {
        Ok("legacy") => true,
        Ok("new") | Ok("recorded") | Err(_) => false,
        Ok(other) => {
            eprintln!("Unknown TNT_PHYSICS={other:?}; using recorded-frozen-v1");
            false
        }
    })
}
pub fn model_name() -> &'static str {
    if use_legacy() { "legacy-before-recorded-v1" } else { "recorded-frozen-v1" }
}
pub fn is_recorded_model() -> bool { !use_legacy() }
pub fn world_units_per_hud_unit() -> f64 {
    static SCALE: OnceLock<f64> = OnceLock::new();
    *SCALE.get_or_init(|| std::env::var("TNT_WPU").ok()
        .and_then(|s| s.parse::<f64>().ok())
        .filter(|v| v.is_finite() && *v > 0.0)
        .unwrap_or(80.0))
}
pub fn height_at(angle: f64, power: f64, wind: f64, target_x: f64) -> Option<f64> {
    if ![angle, power, wind, target_x].iter().all(|v| v.is_finite()) { return None; }
    if use_legacy() { legacy::height_at(angle, power, wind, target_x) }
    else { recorded::height_at(angle, power, wind, target_x) }
}
pub fn simulate_path(angle: f64, power: f64, wind: f64) -> Vec<(f64, f64)> {
    if ![angle, power, wind].iter().all(|v| v.is_finite()) { return Vec::new(); }
    if use_legacy() { legacy::simulate_path(angle, power, wind) }
    else { recorded::simulate_path(angle, power, wind) }
}
pub fn power_for_angle(angle: f64, dx: f64, dy: f64, wind: f64) -> Option<f64> {
    if ![angle, dx, dy, wind].iter().all(|v| v.is_finite()) { return None; }
    if use_legacy() { legacy::power_for_angle(angle, dx, dy, wind) }
    else { recorded::power_for_angle(angle, dx, dy, wind) }
}
pub fn calc_power(angle: f64, distance: f64, wind: f64) -> f64 {
    if use_legacy() { legacy::calc_power(angle, distance, wind) }
    else { recorded::calc_power(angle, distance, wind) }
}
pub fn calc_angle(distance: f64, dy: f64, power: f64, wind: f64, hint_angle: f64) -> f64 {
    if use_legacy() { legacy::calc_angle(distance, dy, power, wind, hint_angle) }
    else { recorded::calc_angle(distance, dy, power, wind, hint_angle) }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn default_is_recorded_candidate() {
        assert!(is_recorded_model());
        assert_eq!(model_name(), "recorded-frozen-v1");
    }
    #[test]
    fn rejects_nonfinite_inputs() {
        assert!(height_at(45.0, f64::NAN, 0.0, 10.0).is_none());
        assert!(power_for_angle(45.0, 10.0, f64::INFINITY, 0.0).is_none());
        assert!(simulate_path(45.0, 50.0, f64::NAN).is_empty());
    }
}
