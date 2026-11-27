//! Tests of systems: hierarchy, every validation rule, the ephemeris and the spheres of
//! influence.

use core::f64::consts::PI;

use super::*;
use crate::body::{CubeDef, radians};
use crate::fixtures::{bodies, system};
use cp_core::hash::Rng;

const SUN: BodyId = BodyId(0);
const HOME: BodyId = BodyId(1);
const MOON: BodyId = BodyId(2);
const OUTER: BodyId = BodyId(3);

fn close(a: DVec3, b: DVec3, tolerance: f64) -> bool {
    (a - b).length() <= tolerance
}

fn cube_of(body: &mut BodyDef) -> &mut CubeDef {
    match &mut body.shape {
        BodyShape::Cube(cube) => cube,
        BodyShape::Sphere { .. } => panic!("not a cube"),
    }
}

#[test]
fn hierarchy_and_lookup() {
    let system = system();
    assert_eq!(system.key(), "test:system");
    assert_eq!(system.display_name(), "Test system");
    assert_eq!(system.body_count(), 4);
    assert_eq!(system.bodies().len(), 4);
    assert_eq!(system.root(), SUN);
    assert_eq!(system.children(SUN), &[HOME, OUTER]);
    assert_eq!(system.children(HOME), &[MOON]);
    assert!(system.children(MOON).is_empty());
    assert_eq!(system.find("test:moon"), Some(MOON));
    assert_eq!(system.find("test:nothing"), None);
    assert_eq!(system.get(BodyId(9)), None);
    assert_eq!(system.body(HOME).key, "test:home");
    assert_eq!(system.ids().collect::<Vec<_>>(), [SUN, HOME, MOON, OUTER]);
    assert_eq!(system.frame_id(HOME), FrameId::Body(HOME));
    assert!(system.orbit(SUN).is_none());
    assert_eq!(system.sphere_of_influence(SUN), f64::INFINITY);
    assert_eq!(system.gravity(HOME).mu(), system.body(HOME).mu());
    assert!(matches!(system.gravity(SUN), GravityField::Sphere(_)));
}

#[test]
fn orbits_use_the_two_body_gravitational_parameter() {
    let system = system();
    let home = system.body(HOME);
    let orbit = system.orbit(HOME).unwrap();
    assert_eq!(orbit.mu(), system.body(SUN).mu() + home.mu());
    let moon_orbit = system.orbit(MOON).unwrap();
    assert_eq!(moon_orbit.mu(), home.mu() + system.body(MOON).mu());
    let period = TAU * (det::powi(8_192_000.0_f64, 3) / orbit.mu()).sqrt();
    assert!((orbit.period() - period).abs() < 1e-6);
}

#[test]
fn frames_compose_with_the_parent_frames() {
    let system = system();
    let mut rng = Rng::new(51, 1);
    for _ in 0..300 {
        let t = rng.range_f64(-1.0e7, 1.0e7);
        for id in system.ids() {
            let direct = system.body_frame(id, t);
            let local = system.frame_in_parent(id, t);
            let composed = match system.body(id).parent {
                Some(parent) => BodyFrame::compose(&system.body_frame(parent, t), &local),
                None => local,
            };
            assert!(
                close(composed.position, direct.position, 1e-6),
                "{id} at {t}"
            );
            assert!(
                close(composed.velocity, direct.velocity, 1e-9),
                "{id} at {t}"
            );
            assert!(
                composed.orientation.abs_diff_eq(direct.orientation, 1e-14),
                "{id} at {t}"
            );
            assert!(close(
                composed.angular_velocity,
                direct.angular_velocity,
                1e-15
            ));
        }
        // `frames` is `body_frame`, bit for bit.
        let frames = system.frames(t);
        for id in system.ids() {
            assert_eq!(frames[id.index()], system.body_frame(id, t));
        }
    }
    // The root's frame in its parent is its frame in the system: at rest at the origin.
    let sun = system.frame_in_parent(SUN, 5.0);
    assert_eq!(sun, system.body_frame(SUN, 5.0));
    assert_eq!((sun.position, sun.velocity), (DVec3::ZERO, DVec3::ZERO));
}

#[test]
fn velocities_are_the_time_derivatives_of_positions() {
    let system = system();
    let h = 1e-2;
    for t in [0.0, 1234.5, 9.87e5, -3.3e6] {
        for id in system.ids() {
            let numeric = (system.body_frame(id, t + h).position
                - system.body_frame(id, t - h).position)
                / (2.0 * h);
            let analytic = system.body_frame(id, t).velocity;
            assert!(
                close(numeric, analytic, 1e-6 * analytic.length() + 1e-6),
                "{id} at {t}: {numeric:?} vs {analytic:?}"
            );
            // Relative to the parent's rotating frame as well.
            let numeric = (system.frame_in_parent(id, t + h).position
                - system.frame_in_parent(id, t - h).position)
                / (2.0 * h);
            let analytic = system.frame_in_parent(id, t).velocity;
            assert!(
                close(numeric, analytic, 1e-6 * analytic.length() + 1e-6),
                "{id} in its parent at {t}: {numeric:?} vs {analytic:?}"
            );
        }
    }
}

#[test]
fn angular_velocities_turn_the_bodies() {
    let system = system();
    let h = 1e-3;
    let point = DVec3::new(20_480.0, 3_000.0, -7_000.0);
    for id in system.ids() {
        for t in [0.0, 5_000.0, 2.0e6] {
            let frame = system.body_frame(id, t);
            let numeric = (system.body_frame(id, t + h).orientation * point
                - system.body_frame(id, t - h).orientation * point)
                / (2.0 * h);
            let analytic = frame.angular_velocity.cross(frame.orientation * point);
            assert!(
                close(numeric, analytic, 1e-6 * analytic.length() + 1e-9),
                "{id} at {t}"
            );
            // And relative to the parent's rotating frame.
            let local = system.frame_in_parent(id, t);
            let numeric = (system.frame_in_parent(id, t + h).orientation * point
                - system.frame_in_parent(id, t - h).orientation * point)
                / (2.0 * h);
            let analytic = local.angular_velocity.cross(local.orientation * point);
            assert!(
                close(numeric, analytic, 1e-6 * analytic.length() + 1e-9),
                "{id} in its parent at {t}"
            );
        }
    }
}

#[test]
fn spin_follows_the_documented_convention() {
    let system = system();
    let spin = system.spin(HOME);
    assert!((spin.period() - 6912.0).abs() < 1e-9);
    let tilt = radians(23.5);
    // The north pole leans towards -X.
    assert!(close(
        spin.axis(),
        DVec3::new(-det::sin(tilt), det::cos(tilt), 0.0),
        1e-15
    ));
    assert!(close(
        spin.angular_velocity(),
        spin.axis() * (TAU / 6912.0),
        1e-18
    ));
    // At the epoch the +Z face normal is R_Z(tilt) · R_Y(135°) · Z.
    let normal = system.body_frame(HOME, 0.0).orientation * DVec3::Z;
    let (sin, cos) = det::sin_cos(radians(135.0));
    let expected = det::dquat_from_axis_angle(DVec3::Z, tilt) * DVec3::new(sin, 0.0, cos);
    assert!(close(normal, expected, 1e-14));
    // A positive rotation turns +X towards -Z: prograde, counter-clockwise from the north.
    let untilted = Spin {
        rate: 1.0,
        phase: 0.0,
        tilt: DQuat::IDENTITY,
    };
    assert!(close(
        untilted.orientation(0.5 * PI) * DVec3::X,
        DVec3::NEG_Z,
        1e-15
    ));
    assert!((0.0..TAU).contains(&untilted.angle(-100.0)));
    assert!((system.spin(SUN).period() - 43_200.0).abs() < 1e-9);
}

/// Angle between two directions in radians, accurate for small angles too.
fn angle_between(a: DVec3, b: DVec3) -> f64 {
    det::atan2(a.cross(b).length(), a.dot(b))
}

/// The fixture with one change to the moon.
fn with_moon(change: impl FnOnce(&mut BodyDef)) -> Result<SystemDef, DefinitionError> {
    let mut list = bodies();
    change(&mut list[2]);
    SystemDef::new("test:system", "Test", list)
}

/// Largest angle between the moon's `-X` face normal and the direction to Home, sampled over
/// one orbit.
fn worst_near_side_angle(system: &SystemDef, samples: u32) -> f64 {
    let period = system.orbit(MOON).unwrap().period();
    let mut worst = 0.0_f64;
    for k in 0..samples {
        let t = f64::from(k) * period / f64::from(samples) + 3.0e6;
        let moon = system.body_frame(MOON, t);
        let home = system.body_frame(HOME, t);
        let near_side = moon.orientation * DVec3::NEG_X;
        worst = det::max(
            worst,
            angle_between(near_side, home.position - moon.position),
        );
    }
    worst
}

#[test]
fn tidally_locked_moon_keeps_its_near_side_towards_its_planet() {
    let system = system();
    let orbit = *system.orbit(MOON).unwrap();
    let spin = system.spin(MOON);
    assert_eq!(spin.rate, orbit.mean_motion());
    let elements = orbit.elements();
    assert_eq!(
        spin.phase,
        (elements.periapsis_argument + elements.mean_anomaly_at_epoch).rem_euclid(TAU)
    );
    // Libration: up to the tilt (7.5° from the orbit normal) in latitude, up to the equation
    // of the centre (2.3° at e = 0.02) in longitude.
    let bound = radians(7.5) + max_equation_of_centre(0.02);
    let worst = worst_near_side_angle(&system, 400);
    assert!(worst <= bound + 1e-10, "{worst} > {bound}");
    assert!(
        worst > radians(7.4),
        "the tilt librates in latitude: {worst}"
    );
}

#[test]
fn locked_rotation_follows_orbits_of_any_inclination() {
    // Prograde, polar and retrograde orbits with different nodes: an untilted lock on a
    // circular orbit points -X exactly at the planet, and otherwise the near side librates by
    // at most the tilt plus the largest equation of the centre. (The former construction in
    // the reference plane missed by the inclination, and turned against retrograde orbits.)
    for inclination in [0.0, 6.0, 60.0, 90.0, 120.0, 150.0, 174.0, 180.0] {
        for node in [0.0, 90.0, 200.0] {
            for (eccentricity, tilt) in [(0.0, 0.0), (0.0, 20.0), (0.02, 1.5), (0.2, 10.0)] {
                let case = format!("i {inclination} node {node} e {eccentricity} tilt {tilt}");
                let system = with_moon(|moon| {
                    moon.axial_tilt = radians(tilt);
                    let orbit = moon.orbit.as_mut().unwrap();
                    orbit.inclination = radians(inclination);
                    orbit.ascending_node = radians(node);
                    orbit.eccentricity = eccentricity;
                    orbit.periapsis_argument = radians(130.0);
                })
                .unwrap_or_else(|error| panic!("{case}: {error}"));
                // (Rounding of the angles n·t, some 1e-12 rad three million seconds after the
                // epoch, is all that separates the lock from exact.)
                let worst = worst_near_side_angle(&system, 360);
                let bound = radians(tilt) + max_equation_of_centre(eccentricity);
                assert!(worst <= bound + 1e-10, "{case}: {worst} > {bound}");
                if tilt == 0.0 && eccentricity == 0.0 {
                    assert!(worst < 1e-10, "{case}: {worst}");
                }
                // The pole is the orbit normal tilted towards the system's +Y in the plane of
                // both, i.e. about the line of nodes.
                let axis = system.spin(MOON).axis();
                let orbit = system.orbit(MOON).unwrap();
                let normal = orbit.normal();
                assert!(
                    (angle_between(axis, normal) - radians(tilt)).abs() < 1e-12,
                    "{case}"
                );
                assert!(
                    (angle_between(axis, DVec3::Y) - radians(inclination - tilt).abs()).abs()
                        < 1e-12,
                    "{case}"
                );
                // The rotation runs with the orbit, also on retrograde ones.
                assert!(
                    system.spin(MOON).angular_velocity().dot(normal) > 0.0,
                    "{case}"
                );
                assert!((system.spin(MOON).period() - orbit.period()).abs() < 1e-9);
            }
        }
    }
}

#[test]
fn locks_that_cannot_face_their_parent_are_rejected() {
    // The limit is the tilt plus the largest equation of the centre: 90°.
    let eccentricity = 0.5;
    let longitude = max_equation_of_centre(eccentricity);
    let below = (FRAC_PI_2 - longitude) * (1.0 - 1e-9);
    let above = (FRAC_PI_2 - longitude) * (1.0 + 1e-9);
    let eccentric = |tilt: f64| {
        with_moon(|moon| {
            moon.axial_tilt = tilt;
            moon.orbit.as_mut().unwrap().eccentricity = eccentricity;
        })
    };
    let system = eccentric(below).expect("a lock inside the limit is valid");
    assert!(worst_near_side_angle(&system, 720) < FRAC_PI_2);
    let error = eccentric(above).unwrap_err();
    assert_eq!((error.body, error.field), (Some(2), Some("tidally_locked")));
    assert!(error.message.contains("below 90 degrees"), "{error}");
    // A lock tilted by 90° or more would turn against its orbit.
    let error = with_moon(|moon| moon.axial_tilt = FRAC_PI_2).unwrap_err();
    assert_eq!(error.field, Some("tidally_locked"));
    // Uniform rotations have no such limit.
    assert!(
        with_moon(|moon| {
            moon.axial_tilt = radians(170.0);
            moon.rotation = Rotation::Uniform {
                period: 5_000.0,
                phase: 0.0,
            };
        })
        .is_ok()
    );
}

#[test]
fn mean_solar_days() {
    let system = system();
    let year = system.orbit(HOME).unwrap().period();
    let day = system.mean_solar_day(HOME).unwrap();
    assert!((day - 1.0 / (1.0 / 6912.0 - 1.0 / year)).abs() < 1e-9 * day);
    assert_eq!(system.mean_solar_day(SUN), None);
    // A locked moon's day is the synodic month against the planet's year; on a retrograde
    // orbit the moon turns against the year, which shortens its day.
    let month = system.orbit(MOON).unwrap().period();
    let moon_day = system.mean_solar_day(MOON).unwrap();
    assert!((moon_day - 1.0 / (1.0 / month - 1.0 / year)).abs() < 1e-9 * moon_day);
    let retrograde = with_moon(|moon| moon.orbit.as_mut().unwrap().inclination = radians(150.0))
        .expect("a retrograde lock is valid");
    let moon_day = retrograde.mean_solar_day(MOON).unwrap();
    assert!((moon_day - 1.0 / (1.0 / month + 1.0 / year)).abs() < 1e-9 * moon_day);
    // A planet locked to its star has no solar day; a retrograde spin a shorter one.
    let mut changed = bodies();
    changed[3].rotation = Rotation::Synchronous;
    let locked = SystemDef::new("test:system", "Test", changed).unwrap();
    assert_eq!(locked.mean_solar_day(OUTER), None);
    let mut changed = bodies();
    changed[1].axial_tilt = PI;
    let retrograde = SystemDef::new("test:system", "Test", changed).unwrap();
    let day = retrograde.mean_solar_day(HOME).unwrap();
    assert!((day - 1.0 / (1.0 / 6912.0 + 1.0 / year)).abs() < 1e-9 * day);
}

#[test]
fn spheres_of_influence_follow_laplace() {
    let system = system();
    let soi = system.sphere_of_influence(HOME);
    let expected = 8_192_000.0 * det::powf(system.body(HOME).mu() / system.body(SUN).mu(), 0.4);
    assert_eq!(soi, expected);
    assert_eq!(
        soi,
        sphere_of_influence(8_192_000.0, system.body(HOME).mu(), system.body(SUN).mu())
    );
    assert!((1.0e6..1.1e6).contains(&soi), "{soi}");
    assert!(system.sphere_of_influence(MOON) > system.body(MOON).bounding_radius());
}

#[test]
fn dominant_body_descends_through_the_spheres_of_influence() {
    let system = system();
    let t = 12_345.0;
    let frames = system.frames(t);
    let home = frames[HOME.index()].position;
    let moon = frames[MOON.index()].position;
    assert_eq!(system.dominant_body(home, t), HOME);
    assert_eq!(system.dominant_body(moon, t), MOON);
    assert_eq!(
        system.dominant_body_in(moon + DVec3::X * 5_000.0, &frames),
        MOON
    );
    // On the line from the moon to Home, just inside and outside the moon's sphere.
    let soi_moon = system.sphere_of_influence(MOON);
    let towards_home = (home - moon).normalize();
    assert_eq!(
        system.dominant_body_in(moon + towards_home * (soi_moon * 0.99), &frames),
        MOON
    );
    assert_eq!(
        system.dominant_body_in(moon + towards_home * (soi_moon * 1.01), &frames),
        HOME
    );
    // At the edge of Home's sphere, away from the star.
    let soi_home = system.sphere_of_influence(HOME);
    let outward = home.normalize();
    assert_eq!(
        system.dominant_body_in(home + outward * (soi_home * 0.999), &frames),
        HOME
    );
    assert_eq!(
        system.dominant_body_in(home + outward * (soi_home * 1.001), &frames),
        SUN
    );
    assert_eq!(system.dominant_body_in(DVec3::ZERO, &frames), SUN);
    assert_eq!(
        system.dominant_body_in(frames[OUTER.index()].position, &frames),
        OUTER
    );
}

#[test]
fn ephemeris_is_pinned() {
    // Deterministic maths (contract 2.5): these bits must be the same on every machine.
    let system = system();
    let frame = system.body_frame(MOON, 1.0e6 + 0.25);
    let local = system.frame_in_parent(MOON, 1.0e6 + 0.25);
    let bits = [
        frame.position.x,
        frame.position.y,
        frame.position.z,
        frame.velocity.x,
        frame.orientation.w,
        local.position.x,
        local.velocity.z,
    ]
    .map(f64::to_bits);
    assert_eq!(bits, PINNED_EPHEMERIS, "{bits:#018x?}");
}

/// Pinned 2026-11-26 on x86_64 Windows. Re-pinned the same day for the orientation only
/// (`orientation.w`, 25 ulps): the locked moon's rotation is now composed in its orbit's frame,
/// `R_Y(90°) · R_X(6° - 7.5°)`, which for this node equals the former `R_Z(1.5°) · R_Y(90°)` up
/// to rounding. The positions and velocities kept every bit.
const PINNED_EPHEMERIS: [u64; 7] = [
    0xc15d_f7de_bf20_5fe1,
    0xc0c7_74ed_fc46_9973,
    0x413d_733b_3c91_0d9a,
    0x405e_67d3_7894_fc66,
    0xbfef_b329_430b_74f8,
    0x40f4_556a_8199_3f45,
    0xc053_5680_9089_5ba0,
];

/// A change to the fixture's bodies.
type Change = fn(&mut Vec<BodyDef>);

/// A broken rule: the change, then the body index, field and message part of the error.
type Case = (Change, Option<usize>, &'static str, &'static str);

/// Builds the fixture with one change and expects the named rule to fail.
fn assert_rejected(change: Change, body: Option<usize>, field: &str, message: &str) {
    let mut list = bodies();
    change(&mut list);
    let error = SystemDef::new("test:system", "Test", list)
        .expect_err(&format!("{field}: the change must be rejected"));
    assert_eq!(error.body, body, "{error}");
    assert_eq!(error.field, Some(field), "{error}");
    assert!(
        error.message.contains(message),
        "expected {message:?} in {error}"
    );
}

#[test]
fn every_rule_is_checked() {
    let cases: &[Case] = &[
        (
            |b| b[1].key = "Home".into(),
            Some(1),
            "key",
            "not a valid key",
        ),
        (
            |b| b[3].key = "test:home".into(),
            Some(3),
            "key",
            "defined twice (first as body #1)",
        ),
        (
            |b| b[2].display_name = " ".into(),
            Some(2),
            "display_name",
            "must not be empty",
        ),
        (
            |b| b[1].surface_gravity = 0.0,
            Some(1),
            "surface_gravity_m_s2",
            "must be positive",
        ),
        (
            |b| b[1].axial_tilt = 4.0,
            Some(1),
            "axial_tilt_deg",
            "between 0 and 180",
        ),
        (
            |b| {
                b[1].rotation = Rotation::Uniform {
                    period: -1.0,
                    phase: 0.0,
                };
            },
            Some(1),
            "rotation_period_s",
            "must be positive",
        ),
        (
            |b| {
                b[1].rotation = Rotation::Uniform {
                    period: 100.0,
                    phase: f64::NAN,
                };
            },
            Some(1),
            "rotation_phase_deg",
            "finite",
        ),
        (
            |b| b[0].rotation = Rotation::Synchronous,
            Some(0),
            "tidally_locked",
            "only a body that orbits",
        ),
        (
            |b| b[2].axial_tilt = radians(89.0),
            Some(2),
            "tidally_locked",
            "stays below 90 degrees",
        ),
        (
            |b| {
                let orbit = b[2].orbit.as_mut().unwrap();
                orbit.semi_major_axis = 300_000.0;
                orbit.eccentricity = 0.75;
                b[2].axial_tilt = 0.0;
            },
            Some(2),
            "tidally_locked",
            "at eccentricity 0.75",
        ),
        (
            |b| b[1].shape = BodyShape::Sphere { radius: 1000.0 },
            Some(1),
            "kind",
            "stars are spheres",
        ),
        (
            |b| b[0].shape = BodyShape::Sphere { radius: 0.0 },
            Some(0),
            "radius_m",
            "must be positive",
        ),
        (
            |b| cube_of(&mut b[1]).edge = 40_961,
            Some(1),
            "edge_m",
            "even",
        ),
        (
            |b| cube_of(&mut b[1]).edge = u32::MAX - 1,
            Some(1),
            "edge_m",
            "too large for the octree",
        ),
        (
            |b| cube_of(&mut b[1]).sea_level = 30_000.0,
            Some(1),
            "sea_level_m",
            "closer",
        ),
        (
            |b| cube_of(&mut b[1]).atmosphere_height = -1.0,
            Some(1),
            "atmosphere_height_m",
            "at least 0",
        ),
        (
            |b| cube_of(&mut b[1]).generator = "Hills".into(),
            Some(1),
            "generator",
            "not a valid key",
        ),
        (
            |b| cube_of(&mut b[1]).grid_offset = 20_000,
            Some(1),
            "grid_offset",
            "does not fit into its root chunks",
        ),
        (
            |b| cube_of(&mut b[1]).grid_offset = 0,
            Some(1),
            "grid_offset",
            "1365 keeps it 33.3 % away",
        ),
        (
            |b| {
                b[3].kind = BodyKind::Star;
                b[3].parent = None;
                b[3].shape = BodyShape::Sphere { radius: 7.0e4 };
                b[3].orbit = None;
            },
            Some(3),
            "parent",
            "`test:sun` already is the root",
        ),
        (
            |b| b[0].parent = Some(BodyId(1)),
            Some(0),
            "parent",
            "a star orbits nothing",
        ),
        (|b| b[1].parent = None, Some(1), "parent", "missing"),
        (
            |b| b[2].parent = Some(BodyId(3)),
            Some(2),
            "parent",
            "`test:outer` must be defined before `test:moon`",
        ),
        (
            |b| b[1].parent = Some(BodyId(1)),
            Some(1),
            "parent",
            "cannot orbit itself",
        ),
        (
            |b| b[1].parent = Some(BodyId(17)),
            Some(1),
            "parent",
            "no body #17",
        ),
        (
            |b| b[2].parent = Some(BodyId(0)),
            Some(2),
            "parent",
            "a moon orbits a planet, but `test:sun` is a star",
        ),
        (
            |b| b[3].parent = Some(BodyId(1)),
            Some(3),
            "parent",
            "a planet orbits a star, but `test:home` is a planet",
        ),
        (|b| b[1].orbit = None, Some(1), "orbit", "missing"),
        (
            |b| b[0].orbit = b[1].orbit,
            Some(0),
            "orbit",
            "orbits nothing",
        ),
        (
            |b| b[1].orbit.as_mut().unwrap().semi_major_axis = 0.0,
            Some(1),
            "orbit.semi_major_axis_m",
            "must be positive",
        ),
        (
            |b| b[1].orbit.as_mut().unwrap().eccentricity = 1.0,
            Some(1),
            "orbit.eccentricity",
            "below 1",
        ),
        (
            |b| b[1].orbit.as_mut().unwrap().inclination = -0.1,
            Some(1),
            "orbit.inclination_deg",
            "between 0 and 180",
        ),
        (
            |b| b[1].orbit.as_mut().unwrap().ascending_node = f64::INFINITY,
            Some(1),
            "orbit.ascending_node_deg",
            "finite",
        ),
        (
            |b| b[2].orbit.as_mut().unwrap().semi_major_axis = 40_000.0,
            Some(2),
            "orbit.semi_major_axis_m",
            "too close",
        ),
        (
            |b| b[2].surface_gravity = 0.001,
            Some(2),
            "orbit.semi_major_axis_m",
            "does not fit into its own sphere of influence",
        ),
        (
            |b| b[2].orbit.as_mut().unwrap().semi_major_axis = 1.0e6,
            Some(2),
            "orbit.semi_major_axis_m",
            "leaves the sphere of influence of `test:home`",
        ),
        (
            |b| b[3].orbit.as_mut().unwrap().semi_major_axis = 8.5e6,
            Some(3),
            "orbit",
            "overlaps that of `test:home`",
        ),
    ];
    for &(change, body, field, message) in cases {
        assert_rejected(change, body, field, message);
    }
    // The system itself.
    let error = SystemDef::new("bad", "Test", bodies()).unwrap_err();
    assert_eq!((error.body, error.field), (None, Some("key")));
    let error = SystemDef::new("test:x", "", bodies()).unwrap_err();
    assert_eq!(error.field, Some("display_name"));
    let error = SystemDef::new("test:x", "X", Vec::new()).unwrap_err();
    assert_eq!(error.field, Some("body"));
    // The fixture itself is valid.
    assert!(SystemDef::new("test:x", "X", bodies()).is_ok());
}
