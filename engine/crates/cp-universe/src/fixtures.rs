//! Test fixtures shared by the unit tests: a small system built in code and the same system as
//! a system file.

use cp_core::ids::BodyId;

use crate::body::{BodyDef, BodyKind, BodyShape, CubeDef, Rotation, radians};
use crate::orbit::OrbitElements;
use crate::system::SystemDef;

pub(crate) fn sun() -> BodyDef {
    BodyDef {
        key: "test:sun".to_owned(),
        display_name: "Sun".to_owned(),
        kind: BodyKind::Star,
        parent: None,
        shape: BodyShape::Sphere { radius: 70_000.0 },
        surface_gravity: 147.5,
        rotation: Rotation::Uniform {
            period: 43_200.0,
            phase: 0.0,
        },
        axial_tilt: 0.0,
        orbit: None,
    }
}

pub(crate) fn cube(edge: u32, grid_offset: i32, atmosphere_height: f64) -> CubeDef {
    CubeDef {
        edge,
        sea_level: 0.0,
        grid_offset,
        atmosphere_height,
        generator: "test:hills".to_owned(),
        generator_seed: 7,
    }
}

pub(crate) fn elements(a: f64, e: f64, i: f64, node: f64, peri: f64, m0: f64) -> OrbitElements {
    OrbitElements {
        semi_major_axis: a,
        eccentricity: e,
        inclination: radians(i),
        ascending_node: radians(node),
        periapsis_argument: radians(peri),
        mean_anomaly_at_epoch: radians(m0),
    }
}

pub(crate) fn home() -> BodyDef {
    BodyDef {
        key: "test:home".to_owned(),
        display_name: "Home".to_owned(),
        kind: BodyKind::Planet,
        parent: Some(BodyId(0)),
        shape: BodyShape::Cube(cube(40_960, 1365, 8_000.0)),
        surface_gravity: 9.81,
        rotation: Rotation::Uniform {
            period: 6_912.0,
            phase: radians(135.0),
        },
        axial_tilt: radians(23.5),
        orbit: Some(elements(8_192_000.0, 0.0, 0.0, 0.0, 0.0, 270.0)),
    }
}

pub(crate) fn moon() -> BodyDef {
    BodyDef {
        key: "test:moon".to_owned(),
        display_name: "Moon".to_owned(),
        kind: BodyKind::Moon,
        parent: Some(BodyId(1)),
        shape: BodyShape::Cube(cube(5_500, 0, 0.0)),
        surface_gravity: 1.62,
        rotation: Rotation::Synchronous,
        // From the orbit normal: the pole passes the system's +Y by 1.5°, like the Moon's.
        axial_tilt: radians(7.5),
        orbit: Some(elements(120_000.0, 0.02, 6.0, 90.0, 0.0, 270.0)),
    }
}

pub(crate) fn outer() -> BodyDef {
    BodyDef {
        key: "test:outer".to_owned(),
        display_name: "Outer".to_owned(),
        kind: BodyKind::Planet,
        parent: Some(BodyId(0)),
        shape: BodyShape::Cube(cube(23_000, 0, 3_500.0)),
        surface_gravity: 7.0,
        rotation: Rotation::Uniform {
            period: 10_800.0,
            phase: 0.0,
        },
        axial_tilt: radians(35.0),
        orbit: Some(elements(11_000_000.0, 0.03, 2.2, 110.0, 200.0, 320.0)),
    }
}

pub(crate) fn bodies() -> Vec<BodyDef> {
    vec![sun(), home(), moon(), outer()]
}

pub(crate) fn system() -> SystemDef {
    SystemDef::new("test:system", "Test system", bodies()).expect("the fixture is valid")
}

/// [`system`] as a system file.
pub(crate) const TOML: &str = r#"# A test system.
[system]
key = "test:system"
display_name = "Test system"

[[body]]
key = "test:sun"
display_name = "Sun"
kind = "star"
radius_m = 70_000.0
surface_gravity_m_s2 = 147.5
rotation_period_s = 43_200
axial_tilt_deg = 0

[[body]]
key = "test:home"
display_name = "Home"
kind = "planet"
parent = "test:sun"
edge_m = 40_960
surface_gravity_m_s2 = 9.81
sea_level_m = 0.0
grid_offset = 1365
atmosphere_height_m = 8000.0
generator = "test:hills"
generator_seed = 7
rotation_period_s = 6912.0
rotation_phase_deg = 135.0
axial_tilt_deg = 23.5

[body.orbit]
semi_major_axis_m = 8_192_000.0
eccentricity = 0.0
inclination_deg = 0.0
ascending_node_deg = 0.0
periapsis_arg_deg = 0.0
mean_anomaly_deg = 270.0

[[body]]
key = "test:moon"
display_name = "Moon"
kind = "moon"
parent = "test:home"
edge_m = 5500
surface_gravity_m_s2 = 1.62
sea_level_m = 0
grid_offset = 0
atmosphere_height_m = 0
generator = "test:hills"
generator_seed = 7
tidally_locked = true
axial_tilt_deg = 7.5
orbit = { semi_major_axis_m = 120_000.0, eccentricity = 0.02, inclination_deg = 6.0, ascending_node_deg = 90.0, periapsis_arg_deg = 0.0, mean_anomaly_deg = 270.0 }

[[body]]
key = "test:outer"
display_name = "Outer"
kind = "planet"
parent = "test:sun"
edge_m = 23_000
surface_gravity_m_s2 = 7.0
sea_level_m = 0.0
grid_offset = 0
atmosphere_height_m = 3500.0
generator = "test:hills"
generator_seed = 7
rotation_period_s = 10800.0
axial_tilt_deg = 35.0

[body.orbit]
semi_major_axis_m = 11_000_000.0
eccentricity = 0.03
inclination_deg = 2.2
ascending_node_deg = 110.0
periapsis_arg_deg = 200.0
mean_anomaly_deg = 320.0
"#;
