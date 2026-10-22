//! Selectable HUD physics. Default: calibrated ordinary-track candidate v2.
//! TNT_PHYSICS=frozen-v1 restores v1; legacy restores the earlier table model.
#[path = "physics/legacy.rs"]
pub mod legacy;
#[path = "physics/recorded.rs"]
mod recorded;
#[path = "physics/recorded_frozen.rs"]
mod recorded_frozen;
use std::sync::OnceLock;

pub const MIN_ANGLE: f64 = 15.0;
pub const MAX_ANGLE: f64 = 89.0;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Model { Legacy, FrozenV1, CalibratedV2 }

fn select_model(value: Option<&str>) -> Model {
    match value {
        Some("legacy") => Model::Legacy,
        Some("frozen-v1" | "recorded-frozen-v1") => Model::FrozenV1,
        Some("new" | "recorded" | "calibrated" | "recorded-calibrated-v2") | None => Model::CalibratedV2,
        Some(other) => {
            eprintln!("Unknown TNT_PHYSICS={other:?}; using recorded-calibrated-v2");
            Model::CalibratedV2
        }
    }
}
fn model() -> Model {
    static MODEL: OnceLock<Model> = OnceLock::new();
    *MODEL.get_or_init(|| select_model(std::env::var("TNT_PHYSICS").ok().as_deref()))
}
pub fn model_name() -> &'static str {
    match model() {
        Model::Legacy => "legacy-before-recorded-v1",
        Model::FrozenV1 => "recorded-frozen-v1",
        Model::CalibratedV2 => "recorded-calibrated-v2",
    }
}
pub fn is_recorded_model() -> bool { model() != Model::Legacy }
pub fn world_units_per_hud_unit() -> f64 {
    static SCALE: OnceLock<f64> = OnceLock::new();
    *SCALE.get_or_init(|| std::env::var("TNT_WPU").ok()
        .and_then(|s| s.parse::<f64>().ok())
        .filter(|v| v.is_finite() && *v > 0.0)
        .unwrap_or(80.0))
}
pub fn height_at(angle: f64, power: f64, wind: f64, target_x: f64) -> Option<f64> {
    if ![angle, power, wind, target_x].iter().all(|v| v.is_finite()) { return None; }
    if model() == Model::Legacy { legacy::height_at(angle, power, wind, target_x) }
    else if model() == Model::FrozenV1 { recorded_frozen::height_at(angle, power, wind, target_x) }
    else { recorded::height_at(angle, power, wind, target_x) }
}
pub fn simulate_path(angle: f64, power: f64, wind: f64) -> Vec<(f64, f64)> {
    if ![angle, power, wind].iter().all(|v| v.is_finite()) { return Vec::new(); }
    if model() == Model::Legacy { legacy::simulate_path(angle, power, wind) }
    else if model() == Model::FrozenV1 { recorded_frozen::simulate_path(angle, power, wind) }
    else { recorded::simulate_path(angle, power, wind) }
}
pub fn power_for_angle(angle: f64, dx: f64, dy: f64, wind: f64) -> Option<f64> {
    if ![angle, dx, dy, wind].iter().all(|v| v.is_finite()) { return None; }
    if model() == Model::Legacy { legacy::power_for_angle(angle, dx, dy, wind) }
    else if model() == Model::FrozenV1 { recorded_frozen::power_for_angle(angle, dx, dy, wind) }
    else { recorded::power_for_angle(angle, dx, dy, wind) }
}
pub fn calc_power(angle: f64, distance: f64, wind: f64) -> f64 {
    if model() == Model::Legacy { legacy::calc_power(angle, distance, wind) }
    else if model() == Model::FrozenV1 { recorded_frozen::calc_power(angle, distance, wind) }
    else { recorded::calc_power(angle, distance, wind) }
}
pub fn calc_angle(distance: f64, dy: f64, power: f64, wind: f64, hint_angle: f64) -> f64 {
    if model() == Model::Legacy { legacy::calc_angle(distance, dy, power, wind, hint_angle) }
    else if model() == Model::FrozenV1 { recorded_frozen::calc_angle(distance, dy, power, wind, hint_angle) }
    else { recorded::calc_angle(distance, dy, power, wind, hint_angle) }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn default_is_recorded_candidate() {
        assert!(is_recorded_model());
        assert_eq!(model_name(), "recorded-calibrated-v2");
    }
    #[test]
    fn preserves_explicit_model_selection() {
        assert_eq!(select_model(None), Model::CalibratedV2);
        assert_eq!(select_model(Some("new")), Model::CalibratedV2);
        assert_eq!(select_model(Some("frozen-v1")), Model::FrozenV1);
        assert_eq!(select_model(Some("recorded-frozen-v1")), Model::FrozenV1);
        assert_eq!(select_model(Some("legacy")), Model::Legacy);
    }
    #[test]
    fn rejects_nonfinite_inputs() {
        assert!(height_at(45.0, f64::NAN, 0.0, 10.0).is_none());
        assert!(power_for_angle(45.0, 10.0, f64::INFINITY, 0.0).is_none());
        assert!(simulate_path(45.0, 50.0, f64::NAN).is_empty());
    }
}
