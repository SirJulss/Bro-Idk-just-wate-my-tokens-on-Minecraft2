//! Tests of the gravity fields: conservative, continuous, exact at the face centres and far
//! away, symmetric, and energy-conserving under a symplectic integrator.

use super::*;
use cp_core::hash::Rng;
use cp_core::time::TICK_DT;

/// Home: edge 40 960 m, 9.81 m/s².
fn home() -> CubeGravity {
    CubeGravity::new(20_480.0, 9.81, CubeGravityShape::for_half_edge(20_480.0)).unwrap()
}

/// Kiesel-sized: a small, light cube.
fn moon() -> CubeGravity {
    CubeGravity::new(2_750.0, 1.62, CubeGravityShape::for_half_edge(2_750.0)).unwrap()
}

/// A shape with narrow blend bands, as a stress case for continuity and the gradient.
fn narrow() -> CubeGravity {
    let shape = CubeGravityShape {
        edge_rounding: 20.0,
        outer_start: 50.0,
        outer_width: 100.0,
        inner_start: 30.0,
        inner_width: 60.0,
    };
    CubeGravity::new(1_000.0, 12.0, shape).unwrap()
}

/// Central-difference gradient of the potential, step `h` metres.
fn numeric_gradient(field: &CubeGravity, p: DVec3, h: f64) -> DVec3 {
    let axis = |unit: DVec3| {
        (field.field(p + unit * h).potential - field.field(p - unit * h).potential) / (2.0 * h)
    };
    DVec3::new(axis(DVec3::X), axis(DVec3::Y), axis(DVec3::Z))
}

/// Points of every region: face centres at several heights, edges, corners, the deep
/// interior and far space, scaled to the body.
fn landmark_points(field: &CubeGravity) -> Vec<DVec3> {
    let r = field.half_edge();
    let shape = field.shape();
    let heights = [
        -0.9 * r,
        -0.5 * r,
        -(shape.inner_start + 0.5 * shape.inner_width),
        -0.5 * shape.inner_start,
        0.0,
        0.5 * shape.outer_start,
        shape.outer_start + 0.3 * shape.outer_width,
        shape.outer_start + 0.8 * shape.outer_width,
        2.0 * r,
        30.0 * r,
    ];
    let mut points = Vec::new();
    for height in heights {
        let level = r + height;
        // Face centres (slightly off-centre too), edge midpoints, corners.
        points.push(DVec3::new(0.0, level, 0.0));
        points.push(DVec3::new(0.013 * r, -level, -0.021 * r));
        points.push(DVec3::new(level, level, 0.0));
        points.push(DVec3::new(-level, 0.4 * level, level));
        points.push(DVec3::new(level, level, level));
        points.push(DVec3::new(-level, level * 0.999, -level * 1.001));
    }
    // The deep interior and the centre region.
    for fraction in [0.01, 0.1, 0.3, 0.6] {
        points.push(DVec3::new(0.7, 0.2, -0.4).normalize() * (fraction * r));
    }
    points
}

fn assert_conservative(field: &CubeGravity, points: &[DVec3], h: f64) {
    let g = field.surface_gravity();
    for &p in points {
        let analytic = field.field(p).acceleration;
        let numeric = -numeric_gradient(field, p, h);
        // Central differences are second-order accurate where the potential is smooth; where
        // its second derivative jumps (rounded edges, the core boundary) the error is of the
        // order h · g / edge_rounding.
        let tolerance = 1e-6 * analytic.length() + 2e-6 * g;
        assert!(
            (analytic - numeric).length() <= tolerance,
            "at {p:?}: analytic {analytic:?}, numeric {numeric:?}, difference {:e}",
            (analytic - numeric).length()
        );
    }
}

#[test]
fn acceleration_is_the_negative_gradient_of_the_potential() {
    for field in [home(), moon(), narrow()] {
        let r = field.half_edge();
        // 1 mm on Home, proportionally less on smaller bodies.
        let h = 1e-3 * r / 20_480.0;
        let mut points = landmark_points(&field);
        // Random points everywhere from the core to far space.
        let mut rng = Rng::new(31, 7);
        for _ in 0..4000 {
            let scale = r * [0.5, 1.0, 1.2, 2.0, 4.0][rng.range_u32(0, 5) as usize];
            points.push(DVec3::new(
                rng.range_f64(-scale, scale),
                rng.range_f64(-scale, scale),
                rng.range_f64(-scale, scale),
            ));
        }
        assert_conservative(&field, &points, h);
    }
}

#[test]
fn face_centres_have_the_configured_surface_gravity() {
    for field in [home(), moon(), narrow()] {
        let r = field.half_edge();
        let shape = *field.shape();
        let g = field.surface_gravity();
        let axes = [
            DVec3::X,
            DVec3::NEG_X,
            DVec3::Y,
            DVec3::NEG_Y,
            DVec3::Z,
            DVec3::NEG_Z,
        ];
        // From just below the inner blend to the start of the outer blend.
        let mut height = -shape.inner_start;
        while height <= shape.outer_start {
            for axis in axes {
                let sample = field.field(axis * (r + height));
                assert!(
                    (sample.acceleration - axis * -g).length() <= 1e-12 * g,
                    "height {height}: {:?}",
                    sample.acceleration
                );
            }
            height += 0.37 * (shape.inner_start + shape.outer_start) / 50.0;
        }
        // Off the centre too, as long as the edge rounding is far.
        let off = DVec3::new(0.3 * r, r, -0.2 * r);
        assert!((field.field(off).acceleration - DVec3::new(0.0, -g, 0.0)).length() <= 1e-12 * g);
    }
    // The configured value is what the face centre of Home feels.
    let sample = home().field(DVec3::new(0.0, 0.0, 20_480.0));
    assert_eq!(sample.acceleration, DVec3::new(0.0, 0.0, -9.81));
}

#[test]
fn far_field_is_keplers() {
    for field in [home(), moon(), narrow()] {
        let mu = field.mu();
        assert_eq!(
            mu,
            field.surface_gravity() * field.half_edge() * field.half_edge()
        );
        let mut rng = Rng::new(32, 1);
        for _ in 0..2000 {
            let direction = DVec3::new(
                rng.range_f64(-1.0, 1.0),
                rng.range_f64(-1.0, 1.0),
                rng.range_f64(-1.0, 1.0),
            )
            .normalize();
            let r = field.kepler_radius() * rng.range_f64(1.0, 1000.0);
            let p = direction * r;
            let sample = field.field(p);
            let potential = -mu / r;
            let acceleration = p * (-mu / (r * r * r));
            assert!((sample.potential - potential).abs() <= 1e-14 * potential.abs());
            assert!((sample.acceleration - acceleration).length() <= 1e-13 * acceleration.length());
        }
        // Above a face centre Kepler starts right after the outer band.
        let shape = field.shape();
        let r = field.half_edge() + shape.outer_start + shape.outer_width + 1.0;
        let sample = field.field(DVec3::new(0.0, 0.0, -r));
        assert!((sample.potential + mu / r).abs() <= 1e-14 * mu / r);
    }
}

#[test]
fn acceleration_points_inwards_and_stays_bounded() {
    for (field, default_shape) in [(home(), true), (moon(), true), (narrow(), false)] {
        let r = field.half_edge();
        let g = field.surface_gravity();
        let mut rng = Rng::new(33, 1);
        for _ in 0..20_000 {
            let scale = r * rng.range_f64(0.001, 3.0);
            let p = DVec3::new(
                rng.range_f64(-scale, scale),
                rng.range_f64(-scale, scale),
                rng.range_f64(-scale, scale),
            );
            let a = field.field(p).acceleration;
            assert!(a.dot(p) < 0.0, "outward force at {p:?}: {a:?}");
            // The blends steepen the field (prototype model, measured 2026-11-26 for Home):
            // up to 2.92 g in the inner band below the face centres (about 9 km deep) and about
            // 2.7 g in the outer band above edges and corners (1 to 8 km above the surface).
            // Narrow bands steepen it much more (the stress shape reaches about 5 g).
            if default_shape {
                assert!(a.length() < 3.0 * g, "too strong at {p:?}: {a:?}");
            }
        }
        assert_eq!(field.field(DVec3::ZERO).acceleration, DVec3::ZERO);
    }
}

#[test]
fn the_surface_shell_never_pulls_harder_than_the_surface_gravity() {
    // From `inner_start` below to `outer_start` above the reference surface the potential is
    // the cube distance itself: gravity is at most g everywhere (exactly g, perpendicular to
    // the face, away from the rounded edges), so walking and building see Home's 9.81 m/s².
    for field in [home(), moon(), narrow()] {
        let r = field.half_edge();
        let shape = *field.shape();
        let g = field.surface_gravity();
        let mut rng = Rng::new(36, 1);
        let mut checked = 0;
        while checked < 20_000 {
            let p = DVec3::new(
                rng.range_f64(-1.1 * r, 1.1 * r),
                rng.range_f64(-1.1 * r, 1.1 * r),
                rng.range_f64(-1.1 * r, 1.1 * r),
            );
            let height = field.cube_distance(p).value - r;
            if !(-shape.inner_start..=shape.outer_start).contains(&height) {
                continue;
            }
            checked += 1;
            let a = field.field(p).acceleration;
            assert!(a.length() <= g * (1.0 + 1e-12), "{p:?}: {a:?}");
        }
    }
}

#[test]
fn field_is_symmetric_under_the_cube_symmetries() {
    let field = home();
    let mut rng = Rng::new(34, 1);
    let permutations = [
        [0, 1, 2],
        [1, 2, 0],
        [2, 0, 1],
        [1, 0, 2],
        [0, 2, 1],
        [2, 1, 0],
    ];
    for _ in 0..300 {
        let p = DVec3::new(
            rng.range_f64(-45_000.0, 45_000.0),
            rng.range_f64(-45_000.0, 45_000.0),
            rng.range_f64(-45_000.0, 45_000.0),
        );
        let base = field.field(p);
        for permutation in permutations {
            for signs in 0..8 {
                let map = |v: DVec3| {
                    let flip = |axis: usize| if signs & (1 << axis) != 0 { -1.0 } else { 1.0 };
                    DVec3::new(
                        v[permutation[0]] * flip(0),
                        v[permutation[1]] * flip(1),
                        v[permutation[2]] * flip(2),
                    )
                };
                let image = field.field(map(p));
                assert!((image.potential - base.potential).abs() <= 1e-12 * base.potential.abs());
                assert!(
                    (image.acceleration - map(base.acceleration)).length()
                        <= 1e-12 * base.acceleration.length() + 1e-15
                );
            }
        }
    }
}

/// The ray parameter where the rounded cube distance along `direction` equals `target`.
fn along_ray_to_cube_distance(field: &CubeGravity, direction: DVec3, target: f64) -> f64 {
    let (mut lo, mut hi) = (0.0, 10.0 * target);
    for _ in 0..200 {
        let mid = 0.5 * (lo + hi);
        if field.cube_distance(direction * mid).value < target {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    0.5 * (lo + hi)
}

/// Potential and acceleration agree on two points `distance` apart up to what their slope and
/// curvature allow: a jump of either would exceed these bounds by orders of magnitude at the
/// small distances used.
fn assert_continuous(field: &CubeGravity, a: DVec3, b: DVec3, what: &str) {
    let g = field.surface_gravity();
    let curvature = g / field.shape().edge_rounding;
    let distance = (a - b).length();
    let (first, second) = (field.field(a), field.field(b));
    let slope = first.acceleration.length() + second.acceleration.length();
    assert!(
        (first.potential - second.potential).abs()
            <= slope * distance + 1e-12 * first.potential.abs(),
        "{what}: potential jumps between {a:?} and {b:?}: {} vs {}",
        first.potential,
        second.potential
    );
    assert!(
        (first.acceleration - second.acceleration).length()
            <= 4.0 * curvature * distance + 1e-9 * g,
        "{what}: acceleration jumps between {a:?} and {b:?}: {:?} vs {:?}",
        first.acceleration,
        second.acceleration
    );
}

#[test]
fn field_is_continuous_across_the_blend_bands() {
    for field in [home(), moon(), narrow()] {
        let r = field.half_edge();
        let shape = *field.shape();
        let boundaries = [
            r + shape.outer_start,
            r + shape.outer_start + shape.outer_width,
            r - shape.inner_start,
            r - shape.inner_start - shape.inner_width,
        ];
        let directions = [
            DVec3::Y,
            DVec3::new(1.0, 1.0, 0.0).normalize(),
            DVec3::ONE.normalize(),
            DVec3::new(0.3, -0.8, 0.52).normalize(),
            DVec3::new(-0.9, 0.05, 0.43).normalize(),
        ];
        for direction in directions {
            for target in boundaries {
                let s = along_ray_to_cube_distance(&field, direction, target);
                for delta in [1e-9 * r, 1e-6 * r] {
                    assert_continuous(
                        &field,
                        direction * (s - delta),
                        direction * (s + delta),
                        &format!("blend boundary {target} along {direction:?}"),
                    );
                }
            }
        }
        // Across the edge rounding: two coordinates that differ by exactly the rounding width,
        // on the surface and inside the outer blend.
        let w = shape.edge_rounding;
        for level in [r, r + shape.outer_start + 0.4 * shape.outer_width] {
            for delta in [1e-9 * r, 1e-6 * r] {
                assert_continuous(
                    &field,
                    DVec3::new(level, level - w - delta, 0.1 * r),
                    DVec3::new(level, level - w + delta, 0.1 * r),
                    "edge rounding",
                );
            }
        }
        // The breakpoints of the radial profile: the core boundary, the start and the end of
        // the outer band (in ρ), crossed above a face centre and on a corner ray.
        for (direction, radius) in [
            (DVec3::Y, field.core),
            (DVec3::ONE.normalize(), field.core),
            (DVec3::Z, field.anchor),
            (DVec3::NEG_X, field.end),
        ] {
            for delta in [1e-9 * r, 1e-6 * r] {
                assert_continuous(
                    &field,
                    direction * (radius - delta),
                    direction * (radius + delta),
                    &format!("profile breakpoint {radius} along {direction:?}"),
                );
            }
        }
    }
}

#[test]
fn invalid_parameters_are_rejected() {
    let good = CubeGravityShape::for_half_edge(1000.0);
    assert!(CubeGravity::new(1000.0, 9.81, good).is_ok());
    assert!(CubeGravity::new(0.0, 9.81, good).is_err());
    assert!(CubeGravity::new(1000.0, -1.0, good).is_err());
    assert!(CubeGravity::new(f64::NAN, 9.81, good).is_err());
    for broken in [
        CubeGravityShape {
            edge_rounding: 0.0,
            ..good
        },
        CubeGravityShape {
            outer_width: -1.0,
            ..good
        },
        CubeGravityShape {
            inner_start: -1.0,
            ..good
        },
        CubeGravityShape {
            inner_width: 900.0,
            ..good
        },
        CubeGravityShape {
            outer_start: f64::INFINITY,
            ..good
        },
    ] {
        let error = CubeGravity::new(1000.0, 9.81, broken).unwrap_err();
        assert!(
            error
                .to_string()
                .starts_with("invalid cube gravity parameters")
        );
    }
    assert!(SphereGravity::new(0.0, 1.0).is_err());
    assert!(SphereGravity::new(1.0, 0.0).is_err());
    assert!(SphereGravity::new(f64::INFINITY, 1.0).is_err());
}

#[test]
fn shape_fractions_reproduce_the_prototype_for_home() {
    let shape = CubeGravityShape::for_half_edge(20_480.0);
    assert_eq!(
        shape,
        CubeGravityShape {
            edge_rounding: 240.0,
            outer_start: 1024.0,
            outer_width: 6656.0,
            inner_start: 2560.0,
            inner_width: 12_800.0,
        }
    );
    let field = home();
    assert_eq!(field.kepler_radius(), 3.0_f64.sqrt() * 28_160.0);
}

#[test]
fn uniform_sphere() {
    let star = SphereGravity::new(70_000.0, 147.5).unwrap();
    let mu = star.mu();
    assert_eq!(mu, 147.5 * 70_000.0 * 70_000.0);
    // Continuous at the surface, with the surface gravity there.
    let inside = star.field(DVec3::new(0.0, 70_000.0 - 1e-6, 0.0));
    let outside = star.field(DVec3::new(0.0, 70_000.0 + 1e-6, 0.0));
    assert!((inside.potential - outside.potential).abs() < 1e-3);
    assert!((inside.acceleration.length() - 147.5).abs() < 1e-6);
    assert!((outside.acceleration.length() - 147.5).abs() < 1e-6);
    assert_eq!(star.field(DVec3::ZERO).acceleration, DVec3::ZERO);
    // Conservative inside and outside.
    let mut rng = Rng::new(35, 1);
    for _ in 0..500 {
        let p = DVec3::new(
            rng.range_f64(-2.0e5, 2.0e5),
            rng.range_f64(-2.0e5, 2.0e5),
            rng.range_f64(-2.0e5, 2.0e5),
        );
        let h = 1e-2;
        let gradient = DVec3::new(
            star.field(p + DVec3::X * h).potential - star.field(p - DVec3::X * h).potential,
            star.field(p + DVec3::Y * h).potential - star.field(p - DVec3::Y * h).potential,
            star.field(p + DVec3::Z * h).potential - star.field(p - DVec3::Z * h).potential,
        ) / (2.0 * h);
        let a = star.field(p).acceleration;
        assert!((a + gradient).length() <= 1e-6 * a.length() + 1e-4, "{p:?}");
    }
    let field = GravityField::Sphere(star);
    assert_eq!(field.mu(), mu);
    assert_eq!(field.field(DVec3::X * 1.0e6), star.field(DVec3::X * 1.0e6));
}

/// Kick-drift-kick leapfrog; returns the largest relative deviation of the energy from its
/// start value and the smallest distance from the centre on the way.
fn leapfrog_energy_drift(
    field: &CubeGravity,
    position: DVec3,
    velocity: DVec3,
    dt: f64,
    steps: u64,
) -> (f64, f64) {
    let (mut p, mut v) = (position, velocity);
    let mut sample = field.field(p);
    let start = 0.5 * v.length_squared() + sample.potential;
    let mut worst = 0.0_f64;
    let mut lowest = f64::INFINITY;
    for _ in 0..steps {
        v += sample.acceleration * (0.5 * dt);
        p += v * dt;
        sample = field.field(p);
        v += sample.acceleration * (0.5 * dt);
        let energy = 0.5 * v.length_squared() + sample.potential;
        worst = det::max(worst, ((energy - start) / start).abs());
        lowest = det::min(lowest, p.length());
    }
    (worst, lowest)
}

#[test]
fn leapfrog_keeps_the_energy_of_a_low_circular_orbit_around_home() {
    // Contract task: a low circular orbit around Home, 100 orbits at the simulation tick.
    // 40 km from the centre in the equatorial plane (19.5 km above the face centres, 4.5 km
    // above the edges), where the field is exactly Kepler's.
    let field = home();
    let radius = 40_000.0;
    let speed = (field.mu() / radius).sqrt();
    let period = core::f64::consts::TAU * radius / speed;
    let steps = (100.0 * period / TICK_DT).round() as u64;
    let (drift, lowest) = leapfrog_energy_drift(
        &field,
        DVec3::new(radius, 0.0, 0.0),
        DVec3::new(0.0, 0.0, -speed),
        TICK_DT,
        steps,
    );
    // Measured 2026-11-26: 3.67e-13 over 4.7 million steps; pinned with a margin.
    assert!(drift < 2e-12, "relative energy drift {drift:e}");
    assert!(lowest > 0.999 * radius, "{lowest}");
}

#[test]
fn leapfrog_keeps_the_energy_of_an_orbit_through_the_blend_band() {
    // 48 km from the centre in a plane through two corners and the ±Z face centres: over the
    // corners the orbit runs through the end of the outer blend band (the field is Kepler's
    // only beyond 48.8 km there), elsewhere through the Kepler region. The energy stays
    // bounded although the orbit is no Kepler ellipse. (Lower orbits over edges or corners are
    // captured by the steepened band field and fall into the planet: below about 39 km over
    // the edges and 48 km over the corners for the prototype's shape, measured 2026-11-26.)
    let field = home();
    let radius = 48_000.0;
    let corner = DVec3::ONE.normalize();
    let normal = DVec3::new(1.0, -1.0, 0.0).normalize();
    let speed = (field.mu() / radius).sqrt();
    let period = core::f64::consts::TAU * radius / speed;
    let steps = (20.0 * period / TICK_DT).round() as u64;
    let (drift, lowest) = leapfrog_energy_drift(
        &field,
        corner * radius,
        normal.cross(corner) * speed,
        TICK_DT,
        steps,
    );
    // Measured 2026-11-26: 3.14e-7 relative, lowest point 47 667 m.
    assert!(drift < 1e-6, "relative energy drift {drift:e}");
    assert!(lowest > 47_000.0, "{lowest}");
}
