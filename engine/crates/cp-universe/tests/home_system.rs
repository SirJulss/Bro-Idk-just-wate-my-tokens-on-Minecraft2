//! The hand-made home system of the base pack (`content/base/systems/home.toml`): it loads,
//! and the numbers its comments promise hold.

use std::path::Path;

use cp_core::det;
use cp_core::ids::BodyId;
use cp_core::math::DVec3;
use cp_universe::{BodyKind, Rotation, SystemDef, load_systems};

fn home_system() -> SystemDef {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../content/base/systems");
    let mut systems = load_systems(&dir).unwrap_or_else(|error| panic!("{error}"));
    assert_eq!(systems.len(), 1, "the base pack defines one system");
    systems.remove(0)
}

fn id(system: &SystemDef, key: &str) -> BodyId {
    system
        .find(&format!("cubeplanet:{key}"))
        .unwrap_or_else(|| panic!("no body {key}"))
}

fn degrees(radians: f64) -> f64 {
    radians * 180.0 / core::f64::consts::PI
}

fn angle(a: DVec3, b: DVec3) -> f64 {
    degrees(det::acos(det::clamp(
        a.normalize().dot(b.normalize()),
        -1.0,
        1.0,
    )))
}

#[test]
fn bodies_and_hierarchy() {
    let system = home_system();
    assert_eq!(system.key(), "cubeplanet:home");
    assert_eq!(system.display_name(), "Heimatsystem");
    let expected = [
        ("sonne", "Sonne", BodyKind::Star, None),
        ("heimat", "Heimat", BodyKind::Planet, Some("sonne")),
        ("kiesel", "Kiesel", BodyKind::Moon, Some("heimat")),
        ("cinder", "Cinder", BodyKind::Planet, Some("sonne")),
        ("ignis", "Ignis", BodyKind::Planet, Some("sonne")),
        ("thalassa", "Thalassa", BodyKind::Planet, Some("sonne")),
        ("glacia", "Glacia", BodyKind::Planet, Some("sonne")),
        ("zephyr", "Zephyr", BodyKind::Planet, Some("sonne")),
        ("relikt", "Relikt", BodyKind::Planet, Some("sonne")),
    ];
    assert_eq!(system.body_count(), expected.len());
    for (index, (key, name, kind, parent)) in expected.into_iter().enumerate() {
        let body = &system.bodies()[index];
        assert_eq!(body.key, format!("cubeplanet:{key}"));
        assert_eq!(body.display_name, name);
        assert_eq!(body.kind, kind);
        assert_eq!(body.parent, parent.map(|parent| id(&system, parent)));
    }
    // Eight cube worlds between 4 and 41 km (contract 1).
    for body in system
        .bodies()
        .iter()
        .filter(|body| body.kind != BodyKind::Star)
    {
        let edge = body.cube().unwrap().edge;
        assert!((4_000..=41_000).contains(&edge), "{}: {edge}", body.key);
    }
}

#[test]
fn heimat_keeps_the_prototype_day_and_has_a_24_day_year() {
    let system = home_system();
    let heimat = id(&system, "heimat");
    let body = system.body(heimat);
    let cube = body.cube().unwrap();
    assert_eq!(cube.edge, 40_960);
    assert_eq!(body.surface_gravity, 9.81);
    assert_eq!(body.root_level(), Some(10));
    assert_eq!(cube.grid_offset, 1365);
    assert!((cube.surface_clearance() - 1.0 / 3.0).abs() < 1e-3);
    assert!(cube.headroom() - cube.atmosphere_height > 2_900.0);
    let year = system.orbit(heimat).unwrap().period();
    assert!((year - 172_800.0).abs() < 1e-6, "{year}");
    let day = system.mean_solar_day(heimat).unwrap();
    assert!((day - 7_200.0).abs() < 1e-6, "{day}");
    assert_eq!((year / day).round(), 24.0);
    assert!((system.sphere_of_influence(heimat) / 1000.0 - 1036.4).abs() < 0.1);
    // Gravity at a face centre and the escape speed from there.
    let gravity = system.gravity(heimat);
    let face = gravity.field(DVec3::new(0.0, 0.0, 20_480.0));
    assert!((face.acceleration - DVec3::new(0.0, 0.0, -9.81)).length() < 1e-12);
    let escape = (-2.0 * face.potential).sqrt();
    assert!((escape - 642.0).abs() < 1.0, "{escape}");
}

#[test]
fn kiesel_is_a_close_locked_moon() {
    let system = home_system();
    let heimat = id(&system, "heimat");
    let kiesel = id(&system, "kiesel");
    let body = system.body(kiesel);
    assert_eq!(body.rotation, Rotation::Synchronous);
    assert_eq!(body.atmosphere_height(), 0.0);
    let orbit = system.orbit(kiesel).unwrap();
    assert_eq!(orbit.elements().semi_major_axis, 120_000.0);
    assert!(
        (orbit.period() - 4_065.76).abs() < 0.01,
        "{}",
        orbit.period()
    );
    let month = system.mean_solar_day(kiesel).unwrap();
    assert!((month - 4_163.7).abs() < 0.1, "{month}");
    let soi = system.sphere_of_influence(kiesel);
    assert!((soi - 11_716.0).abs() < 10.0, "{soi}");
    // Apparent sizes: Kiesel in Heimat's sky, Heimat in Kiesel's.
    assert!((degrees(5_500.0 / 120_000.0) - 2.63).abs() < 0.01);
    assert!((degrees(40_960.0 / 120_000.0) - 19.56).abs() < 0.01);
    // 29.5° between Kiesel's orbit and Heimat's equator.
    let tilt = angle(orbit.normal(), system.spin(heimat).axis());
    assert!((tilt - 29.5).abs() < 0.1, "{tilt}");
    // Kiesel's pole: 7.5° from its orbit's normal, 1.5° from the ecliptic's pole on the far
    // side, towards -X (the orbit's normal leans towards +X).
    let pole = system.spin(kiesel).axis();
    assert!((angle(pole, orbit.normal()) - 7.5).abs() < 1e-9);
    assert!((angle(pole, DVec3::Y) - 1.5).abs() < 1e-9);
    assert!(pole.x < 0.0 && orbit.normal().x > 0.0);
    // The near side faces Heimat within the libration: 7.5° in latitude, 2.3° in longitude.
    let mut worst = 0.0_f64;
    for k in 0..2_000 {
        let t = f64::from(k) * orbit.period() / 2_000.0;
        let moon = system.body_frame(kiesel, t);
        let planet = system.body_frame(heimat, t);
        let off = angle(
            moon.orientation * DVec3::NEG_X,
            planet.position - moon.position,
        );
        worst = det::max(worst, off);
    }
    assert!((7.4..9.8).contains(&worst), "{worst}");
    let escape = (-2.0 * system.gravity(kiesel).field(DVec3::Y * 2_750.0).potential).sqrt();
    assert!((escape - 95.0).abs() < 2.0, "{escape}");
}

#[test]
fn periods_spheres_and_days_are_as_documented() {
    let system = home_system();
    // (key, year in s, sphere of influence in km, solar day in s), from the file's comments.
    let documented = [
        ("cinder", 38_401.0, 89.8, 115_193.0),
        ("ignis", 68_154.0, 347.9, 18_257.6),
        ("thalassa", 108_379.0, 690.7, 5_683.2),
        ("glacia", 269_465.0, 766.3, 11_250.9),
        ("zephyr", 407_976.0, 693.1, 3_632.0),
        ("relikt", 636_206.0, 1049.8, 18_524.1),
    ];
    for (key, year, soi, day) in documented {
        let body = id(&system, key);
        let period = system.orbit(body).unwrap().period();
        assert!((period - year).abs() < 1.0, "{key}: year {period}");
        let radius = system.sphere_of_influence(body) / 1000.0;
        assert!((radius - soi).abs() < 0.1, "{key}: sphere {radius}");
        let solar = system.mean_solar_day(body).unwrap();
        assert!((solar - day).abs() < 1.0, "{key}: day {solar}");
    }
    // The neighbours appear as discs at their closest approach (circular estimate).
    for (key, size) in [("thalassa", 0.99), ("glacia", 0.47), ("ignis", 0.33)] {
        let body = id(&system, key);
        let edge = f64::from(system.body(body).cube().unwrap().edge);
        let a = system.orbit(body).unwrap().elements().semi_major_axis;
        let degrees = degrees(edge / (a - 8_192_000.0).abs());
        assert!((degrees - size).abs() < 0.01, "{key}: {degrees}");
    }
    assert!((degrees(140_000.0 / 8_192_000.0) - 0.98).abs() < 0.01);
}

/// The +Z face centre of Heimat in system coordinates and its outward normal.
fn start_face(system: &SystemDef, t: f64) -> (DVec3, DVec3) {
    let frame = system.body_frame(id(system, "heimat"), t);
    (
        frame.point_to_parent(DVec3::new(0.0, 0.0, 20_480.0)),
        frame.direction_to_parent(DVec3::Z),
    )
}

#[test]
fn the_game_starts_on_a_spring_morning() {
    let system = home_system();
    let heimat = id(&system, "heimat");
    let (centre, normal) = start_face(&system, 0.0);
    let elevation = |t: f64| {
        let (centre, normal) = start_face(&system, t);
        90.0 - angle(normal, -centre)
    };
    let sun = elevation(0.0);
    assert!((sun - 45.0).abs() < 1.0, "{sun}");
    assert!(elevation(60.0) > sun, "the sun rises");
    // Spring equinox: the sun stands over the equator; a quarter year later over 23.5° north.
    let subsolar = |t: f64| {
        let frame = system.body_frame(heimat, t);
        90.0 - angle(system.spin(heimat).axis(), -frame.position)
    };
    assert!(subsolar(0.0).abs() < 0.1, "{}", subsolar(0.0));
    assert!(
        (subsolar(43_200.0) - 23.5).abs() < 0.1,
        "{}",
        subsolar(43_200.0)
    );
    // Kiesel is up, about half lit and far from the sun.
    let kiesel = system.body_frame(id(&system, "kiesel"), 0.0).position;
    let moon_elevation = 90.0 - angle(normal, kiesel - centre);
    assert!((moon_elevation - 27.0).abs() < 1.0, "{moon_elevation}");
    let from_sun = angle(kiesel - centre, -centre);
    assert!((from_sun - 100.0).abs() < 1.5, "{from_sun}");
    // Glacia near opposition, Thalassa near its greatest elongation, Ignis at 33°.
    let home = system.body_frame(heimat, 0.0).position;
    let elongation = |key: &str| {
        let planet = system.body_frame(id(&system, key), 0.0).position;
        angle(planet - home, -home)
    };
    assert!(elongation("glacia") > 165.0, "{}", elongation("glacia"));
    assert!((elongation("thalassa") - 48.0).abs() < 1.0);
    assert!((elongation("ignis") - 33.5).abs() < 1.0);
    assert!((elongation("cinder") - 16.0).abs() < 1.0);
}

#[test]
fn dominant_bodies_around_heimat() {
    let system = home_system();
    let frames = system.frames(1_000.0);
    let heimat = id(&system, "heimat");
    let kiesel = id(&system, "kiesel");
    let surface = frames[heimat.index()].point_to_parent(DVec3::new(0.0, 20_500.0, 0.0));
    assert_eq!(system.dominant_body_in(surface, &frames), heimat);
    assert_eq!(
        system.dominant_body_in(frames[kiesel.index()].position, &frames),
        kiesel
    );
    assert_eq!(system.dominant_body_in(DVec3::ZERO, &frames), system.root());
}

/// Kick-drift-kick leapfrog in a body's frame (its rotation ignored); returns the lowest
/// distance from the centre.
fn lowest_point(field: &cp_universe::GravityField, start: DVec3, along: DVec3, orbits: f64) -> f64 {
    let radius = start.length();
    let speed = (field.mu() / radius).sqrt();
    let period = core::f64::consts::TAU * radius / speed;
    let dt = period / 20_000.0;
    let (mut p, mut v) = (start, along.normalize() * speed);
    let mut a = field.field(p).acceleration;
    let mut lowest = radius;
    for _ in 0..(orbits * 20_000.0) as u64 {
        v += a * (0.5 * dt);
        p += v * dt;
        a = field.field(p).acceleration;
        v += a * (0.5 * dt);
        lowest = det::min(lowest, p.length());
    }
    lowest
}

#[test]
fn lowest_practical_orbits() {
    let system = home_system();
    let corner = DVec3::ONE.normalize();
    let over_corners = DVec3::new(1.0, -1.0, 0.0).normalize().cross(corner);
    // Heimat: 40 km over the edges survives; 36 km is pulled into the planet.
    let heimat = system.gravity(id(&system, "heimat"));
    assert!(lowest_point(heimat, DVec3::X * 40_000.0, DVec3::NEG_Z, 5.0) > 39_000.0);
    assert!(lowest_point(heimat, DVec3::X * 36_000.0, DVec3::NEG_Z, 5.0) < 20_480.0);
    // Kiesel: over the corners from about 6.5 km.
    let kiesel = system.gravity(id(&system, "kiesel"));
    assert!(lowest_point(kiesel, corner * 6_600.0, over_corners, 5.0) > 6_400.0);
    assert!(lowest_point(kiesel, corner * 5_600.0, over_corners, 5.0) < 2_750.0 * 3.0_f64.sqrt());
}

#[test]
fn kiesel_overtakes_heimats_sky() {
    let system = home_system();
    let heimat = id(&system, "heimat");
    let kiesel = id(&system, "kiesel");
    // In Heimat's turning frame Kiesel moves prograde and the sun retrograde: Kiesel rises
    // where the sun sets.
    let sense = |position: &dyn Fn(f64) -> DVec3| {
        let (now, later) = (position(0.0), position(10.0));
        now.cross(later).dot(DVec3::Y)
    };
    assert!(sense(&|t| system.frame_in_parent(kiesel, t).position) > 0.0);
    assert!(sense(&|t| system.body_frame(heimat, t).point_to_body(DVec3::ZERO)) < 0.0);
    // It stays above the start face for about 75 minutes at a time.
    let above = |t: f64| {
        let (centre, normal) = start_face(&system, t);
        let moon = system.body_frame(kiesel, t).position;
        normal.dot(moon - centre) > 0.0
    };
    let mut changes = Vec::new();
    let mut previous = above(0.0);
    let mut t = 0.0;
    while t < 4.0 * 3600.0 {
        t += 5.0;
        let now = above(t);
        if now != previous {
            changes.push((t, now));
        }
        previous = now;
    }
    let rise = changes.iter().find(|(_, up)| *up).expect("Kiesel rises").0;
    let set = changes
        .iter()
        .find(|(when, up)| !*up && *when > rise)
        .expect("and sets again")
        .0;
    let minutes = (set - rise) / 60.0;
    assert!((65.0..85.0).contains(&minutes), "{minutes}");
}

#[test]
fn kiesel_crosses_heimats_shadow_at_every_full_moon() {
    // Heimat stands in as its inscribed sphere (20 480 m); its real silhouette is larger.
    let system = home_system();
    let heimat = id(&system, "heimat");
    let kiesel = id(&system, "kiesel");
    let month = system.mean_solar_day(kiesel).unwrap();
    let step = 2.0;
    let mut passages: Vec<f64> = Vec::new();
    let mut since: Option<f64> = None;
    let mut t = 0.0;
    while t < 30.0 * month {
        let frames = system.frames(t);
        let planet = frames[heimat.index()].position;
        let away_from_sun = planet.normalize();
        let relative = frames[kiesel.index()].position - planet;
        let along = relative.dot(away_from_sun);
        let across = (relative - away_from_sun * along).length();
        let umbra = 20_480.0 - along * (70_000.0 - 20_480.0) / planet.length();
        let dark = along > 0.0 && across < umbra;
        match (dark, since) {
            (true, None) => since = Some(t),
            (false, Some(start)) => {
                passages.push(t - start);
                since = None;
            }
            _ => {}
        }
        t += step;
    }
    assert!(passages.len() >= 29, "{} passages", passages.len());
    for seconds in passages {
        assert!(
            (120.0..300.0).contains(&seconds),
            "{seconds} s in the shadow"
        );
    }
}
