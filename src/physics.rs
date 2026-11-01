//! Selectable HUD physics. Default: calibrated ordinary-track candidate v2.
//! TNT_PHYSICS=frozen-v1 restores v1; legacy restores the earlier table model.
#[path = "physics/legacy.rs"]
pub mod legacy;
#[path = "physics/recorded.rs"]
mod recorded;
#[path = "physics/recorded_frozen.rs"]
mod recorded_frozen;
#[path = "physics/expanded.rs"]
mod expanded;
use std::sync::OnceLock;

pub const MIN_ANGLE: f64 = 15.0;
pub const MAX_ANGLE: f64 = 89.0;

#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum ShotScale {
    #[default]
    Normal,
    Expanded,
}
impl ShotScale {
    pub fn from_factor(value: f64) -> Option<Self> {
        if (value-1.0).abs()<=1e-4 {Some(Self::Normal)}
        else if (value-2.0).abs()<=1e-4 {Some(Self::Expanded)} else {None}
    }
    pub fn factor(self) -> f64 { if self==Self::Expanded {2.0} else {1.0} }
}

pub fn model_name_for_scale(scale: ShotScale) -> &'static str {
    if scale==ShotScale::Expanded && is_recorded_model() {"recorded-expanded-v1"} else {model_name()}
}
pub fn simulate_path_for_scale(scale: ShotScale, angle: f64, power: f64, wind: f64) -> Vec<(f64,f64)> {
    if ![angle,power,wind].iter().all(|v|v.is_finite()) {return Vec::new();}
    if scale==ShotScale::Expanded && is_recorded_model() {expanded::simulate_path(angle,power,wind)}
    else {simulate_path(angle,power,wind)}
}
pub fn power_for_angle_for_scale(scale: ShotScale, angle: f64, dx: f64, dy: f64, wind: f64) -> Option<f64> {
    if ![angle,dx,dy,wind].iter().all(|v|v.is_finite()) {return None;}
    if scale==ShotScale::Expanded && is_recorded_model() {expanded::power_for_angle(angle,dx,dy,wind)}
    else {power_for_angle(angle,dx,dy,wind)}
}
pub fn calc_angle_for_scale(scale: ShotScale, distance: f64, dy: f64, power: f64, wind: f64, hint: f64) -> f64 {
    if scale==ShotScale::Expanded && is_recorded_model() {expanded::calc_angle(distance,dy,power,wind,hint)}
    else {calc_angle(distance,dy,power,wind,hint)}
}

/// World firing angle (0 right, 180 left), world wind, signed HUD-unit target.
/// Ordinary shots retain the existing solver. A shot initially flying away
/// can still return in wind, so search its descending branch rather than
/// mirroring it toward the target or checking the character's facing flag.
pub fn power_for_world_angle_for_scale(scale: ShotScale, angle: f64, dx: f64, dy: f64, wind: f64) -> Option<f64> {
    if ![angle,dx,dy,wind].iter().all(|v|v.is_finite()) || !(0.0..=180.0).contains(&angle) {return None;}
    let left=angle>90.0;
    let elevation=if left {180.0-angle} else {angle};
    let forward_dx=if left {-dx} else {dx};
    if forward_dx>0.0 && (MIN_ANGLE..=MAX_ANGLE).contains(&elevation) {
        return power_for_angle_for_scale(scale,elevation,forward_dx,dy,if left {-wind} else {wind});
    }
    let error=|power| {
        let path=simulate_path_for_scale(scale,angle,power,wind);
        path.windows(2).find_map(|p| {
            let (a,b)=(p[0],p[1]);
            if a.1>=dy && b.1<=dy && b.1<a.1 {
                Some(a.0+(b.0-a.0)*(a.1-dy)/(a.1-b.1)-dx)
            } else {None}
        })
    };
    let mut previous=None;
    for i in 0..=600 {
        let power=i as f64*0.25;
        let Some(e)=error(power) else {previous=None;continue;};
        if e.abs()<=1e-6 {return Some(power);}
        if let Some((mut lo,lo_error))=previous {
            if (lo_error>0.0)!=(e>0.0) {
                let mut hi=power;
                for _ in 0..50 {
                    let mid=(lo+hi)*0.5;
                    let em=error(mid)?;
                    if (em>0.0)==(lo_error>0.0) {lo=mid;} else {hi=mid;}
                }
                let result=(lo+hi)*0.5;
                return (error(result)?.abs()<=1e-6).then_some(result);
            }
        }
        previous=Some((power,e));
    }
    None
}

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
    fn descending_x(path: &[(f64,f64)], y: f64) -> f64 {
        path.windows(2).find_map(|p| {
            let (a,b)=(p[0],p[1]);
            (a.1>=y && b.1<=y && b.1<a.1).then(||a.0+(b.0-a.0)*(a.1-y)/(a.1-b.1))
        }).expect("reference trajectory crosses target height")
    }

    #[test]
    fn world_angle_solver_preserves_left_right_and_wind_return_shots() {
        for scale in [ShotScale::Normal,ShotScale::Expanded] {
            let right=power_for_world_angle_for_scale(scale,60.0,10.0,-2.0,-7.7).unwrap();
            let left=power_for_world_angle_for_scale(scale,120.0,-10.0,-2.0,7.7).unwrap();
            assert!((right-left).abs()<1e-9);
            let reference=simulate_path_for_scale(scale,80.0,60.0,-31.5);
            let x=descending_x(&reference,0.0);
            assert!(x<0.0,"reference returns behind launch direction: {x}");
            let force=power_for_world_angle_for_scale(scale,80.0,x,0.0,-31.5).unwrap();
            let hit=descending_x(&simulate_path_for_scale(scale,80.0,force,-31.5),0.0);
            assert!((hit-x).abs()<1e-6);
            let mirrored=power_for_world_angle_for_scale(scale,100.0,-x,0.0,31.5).unwrap();
            assert!((force-mirrored).abs()<1e-6);
            // Initially travelling away, no wind to bring it back: no invented mirror shot.
            assert!(power_for_world_angle_for_scale(scale,60.0,-10.0,0.0,0.0).is_none());
        }
    }
    #[test]
    fn expanded_profile_dispatches_solver_and_path_together() {
        assert_eq!(ShotScale::from_factor(2.0),Some(ShotScale::Expanded));
        for value in [1.5,0.0,f64::NAN,f64::INFINITY] {assert_eq!(ShotScale::from_factor(value),None);}
        assert_eq!(model_name_for_scale(ShotScale::Expanded),"recorded-expanded-v1");
        let path=simulate_path_for_scale(ShotScale::Expanded,57.0,50.36,-52.7);
        assert_eq!(path,expanded::simulate_path(57.0,50.36,-52.7));
        assert_eq!(simulate_path_for_scale(ShotScale::Normal,57.0,50.36,-52.7),simulate_path(57.0,50.36,-52.7));
        assert_ne!(path[40],simulate_path(57.0,50.36,-52.7)[40]);
        let power=power_for_angle_for_scale(ShotScale::Expanded,57.0,10.0,-3.0,0.0).unwrap();
        assert!((expanded::height_at(57.0,power,0.0,10.0).unwrap()+3.0).abs()<1e-6);
        let normal_power=power_for_angle_for_scale(ShotScale::Normal,57.0,10.0,-3.0,0.0).unwrap();
        assert!((power-normal_power).abs()>5.0);
    }
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
