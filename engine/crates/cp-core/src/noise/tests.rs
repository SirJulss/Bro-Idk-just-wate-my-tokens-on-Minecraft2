//! Tests of the public noise API: range, determinism, tileability, continuity and quality.

use super::*;
use crate::det;
use crate::hash::{Rng, hash_bytes};

const SEED: u64 = 0x00C0_FFEE;
const SQRT_2: f64 = core::f64::consts::SQRT_2;

fn random2(rng: &mut Rng, extent: f64) -> DVec2 {
    DVec2::new(
        rng.range_f64(-extent, extent),
        rng.range_f64(-extent, extent),
    )
}

fn random3(rng: &mut Rng, extent: f64) -> DVec3 {
    DVec3::new(
        rng.range_f64(-extent, extent),
        rng.range_f64(-extent, extent),
        rng.range_f64(-extent, extent),
    )
}

/// Sample points for the pinned grids: a block around the origin that crosses many cells in
/// both signs, and a small block far away where coordinates are large.
fn grid2() -> Vec<DVec2> {
    let mut points = Vec::new();
    for iy in -32..32 {
        for ix in -32..32 {
            points.push(DVec2::new(
                f64::from(ix) * 0.173 + 0.05,
                f64::from(iy) * 0.211 - 0.03,
            ));
        }
    }
    for iy in 0..16 {
        for ix in 0..16 {
            points.push(DVec2::new(
                123_456.7 + f64::from(ix) * 0.31,
                -7_654_321.1 + f64::from(iy) * 0.29,
            ));
        }
    }
    points
}

fn grid3() -> Vec<DVec3> {
    let mut points = Vec::new();
    for iz in -10..10 {
        for iy in -10..10 {
            for ix in -10..10 {
                points.push(DVec3::new(
                    f64::from(ix) * 0.29 + 0.05,
                    f64::from(iy) * 0.31 - 0.03,
                    f64::from(iz) * 0.37 + 0.11,
                ));
            }
        }
    }
    for iz in 0..8 {
        for iy in 0..8 {
            for ix in 0..8 {
                points.push(DVec3::new(
                    123_456.7 + f64::from(ix) * 0.31,
                    -7_654_321.1 + f64::from(iy) * 0.29,
                    33_554_431.9 + f64::from(iz) * 0.27,
                ));
            }
        }
    }
    points
}

/// FNV-1a over the little-endian bit patterns of the samples (contract §2.5, rule 4).
fn hash_samples(samples: impl IntoIterator<Item = f64>) -> u64 {
    let mut bytes = Vec::new();
    for sample in samples {
        bytes.extend_from_slice(&sample.to_bits().to_le_bytes());
    }
    hash_bytes(&bytes)
}

fn pin2(f: impl Fn(DVec2) -> f64) -> u64 {
    hash_samples(grid2().into_iter().map(f))
}

fn pin3(f: impl Fn(DVec3) -> f64) -> u64 {
    hash_samples(grid3().into_iter().map(f))
}

/// Hashes of sample grids of every public noise function. These pin the exact bits of the
/// output: generated worlds and textures depend on them, on every platform. A mismatch means
/// the noise changed and is a breaking change of saved worlds, not a number to update lightly.
const PINNED: &[(&str, u64)] = &[
    ("value2", 0x006d9f082ed7574f),
    ("value3", 0x348373a49ee72194),
    ("gradient2", 0xe6cc1cb29be4581d),
    ("gradient3", 0x56e5d27f5e1f5efd),
    ("worley2", 0xc109b2137f5deabd),
    ("worley3", 0x5bf4b9b95988e440),
    ("value2_tiled", 0xc361f97caa73b76e),
    ("value3_tiled", 0x746ccadec4fde1d2),
    ("gradient2_tiled", 0x06c16f125b49368c),
    ("gradient3_tiled", 0x8ee1e8c7abca6a0f),
    ("worley2_tiled", 0x9a096ad61a82e885),
    ("worley3_tiled", 0x6f1dca5d022dd8d2),
    ("fbm.sample2", 0xc15998df5fcce484),
    ("fbm.sample3", 0x056674e09153507a),
    ("fbm.ridged2", 0x4e45da30962ed4ee),
    ("fbm.ridged3", 0x759438987ae0b09d),
    ("fbm.billow2", 0xe4ed4030ceaba770),
    ("fbm.billow3", 0x04ce5bac273f5ea0),
    ("fbm.sample2_tiled", 0x33c594db93d4e726),
    ("fbm.sample3_tiled", 0x66c9f13e5ecf0d82),
    ("fbm.ridged2_tiled", 0xeaa3f8a3535a9da4),
    ("fbm.ridged3_tiled", 0xa054c03ef85d370a),
    ("fbm.billow2_tiled", 0xc97ba82cee8454af),
    ("fbm.billow3_tiled", 0x9052c1ec4dd79be3),
    ("fbm.domain_warp2", 0x7ddedf86e38c8463),
    ("fbm.domain_warp3", 0x20f0d7808695a39b),
    ("odd.sample2", 0xf20f8994556d1f2e),
    ("odd.sample3", 0x044ded6e938193ad),
    ("odd.sample2_tiled", 0x60da1102dc04f59a),
    ("seed0.gradient3", 0x873babf138510640),
    ("seedmax.gradient3", 0xb699b9bead4da388),
];

/// A few plain values, for anyone porting the functions (to a shader, say) who wants a number
/// to compare with before hashing whole grids: `value2`, `gradient2` at `(0.3, 0.7)`;
/// `value3`, `gradient3`, `worley3` (`f1`, `f2`) at `(0.3, 0.7, -1.6)`; five-octave
/// `Fbm::sample2` and `Fbm::sample3` at the same points. Seed [`SEED`].
///
/// These values and several of the hashes above are reproduced bit for bit by an independent
/// port of the algorithms to Python (`tests/reference_port.py`), so they are a property of the
/// algorithms and not of this compiler.
const PINNED_SAMPLES: [f64; 8] = [
    0.267306142564215,
    -0.33976159198059913,
    -0.011498072430835185,
    -0.14642775927369894,
    0.7163968408275815,
    0.8393281908231011,
    0.37846472210099635,
    -0.36237597355757356,
];

#[test]
fn outputs_are_pinned() {
    let fbm = Fbm::new(5, 2.0, 0.5);
    let odd = Fbm::new(3, 2.7, 0.65);
    let pair2 = |f: fn(u64, DVec2) -> (f64, f64)| {
        hash_samples(grid2().into_iter().flat_map(|p| {
            let (f1, f2) = f(SEED, p);
            [f1, f2]
        }))
    };
    let pair3 = |f: fn(u64, DVec3) -> (f64, f64)| {
        hash_samples(grid3().into_iter().flat_map(|p| {
            let (f1, f2) = f(SEED, p);
            [f1, f2]
        }))
    };
    let warp2 = hash_samples(grid2().into_iter().flat_map(|p| {
        let q = fbm.domain_warp2(SEED, p, 0.7, 1.3);
        [q.x, q.y]
    }));
    let warp3 = hash_samples(grid3().into_iter().flat_map(|p| {
        let q = fbm.domain_warp3(SEED, p, 0.7, 1.3);
        [q.x, q.y, q.z]
    }));
    let actual = [
        ("value2", pin2(|p| value2(SEED, p))),
        ("value3", pin3(|p| value3(SEED, p))),
        ("gradient2", pin2(|p| gradient2(SEED, p))),
        ("gradient3", pin3(|p| gradient3(SEED, p))),
        ("worley2", pair2(worley2)),
        ("worley3", pair3(worley3)),
        ("value2_tiled", pin2(|p| value2_tiled(SEED, p, 4))),
        ("value3_tiled", pin3(|p| value3_tiled(SEED, p, 4))),
        ("gradient2_tiled", pin2(|p| gradient2_tiled(SEED, p, 4))),
        ("gradient3_tiled", pin3(|p| gradient3_tiled(SEED, p, 4))),
        ("worley2_tiled", pair2(|seed, p| worley2_tiled(seed, p, 3))),
        ("worley3_tiled", pair3(|seed, p| worley3_tiled(seed, p, 3))),
        ("fbm.sample2", pin2(|p| fbm.sample2(SEED, p))),
        ("fbm.sample3", pin3(|p| fbm.sample3(SEED, p))),
        ("fbm.ridged2", pin2(|p| fbm.ridged2(SEED, p))),
        ("fbm.ridged3", pin3(|p| fbm.ridged3(SEED, p))),
        ("fbm.billow2", pin2(|p| fbm.billow2(SEED, p))),
        ("fbm.billow3", pin3(|p| fbm.billow3(SEED, p))),
        ("fbm.sample2_tiled", pin2(|p| fbm.sample2_tiled(SEED, p, 4))),
        ("fbm.sample3_tiled", pin3(|p| fbm.sample3_tiled(SEED, p, 4))),
        ("fbm.ridged2_tiled", pin2(|p| fbm.ridged2_tiled(SEED, p, 4))),
        ("fbm.ridged3_tiled", pin3(|p| fbm.ridged3_tiled(SEED, p, 4))),
        ("fbm.billow2_tiled", pin2(|p| fbm.billow2_tiled(SEED, p, 4))),
        ("fbm.billow3_tiled", pin3(|p| fbm.billow3_tiled(SEED, p, 4))),
        ("fbm.domain_warp2", warp2),
        ("fbm.domain_warp3", warp3),
        ("odd.sample2", pin2(|p| odd.sample2(SEED, p))),
        ("odd.sample3", pin3(|p| odd.sample3(SEED, p))),
        ("odd.sample2_tiled", pin2(|p| odd.sample2_tiled(SEED, p, 5))),
        ("seed0.gradient3", pin3(|p| gradient3(0, p))),
        ("seedmax.gradient3", pin3(|p| gradient3(u64::MAX, p))),
    ];
    let listing: String = actual
        .iter()
        .map(|(name, hash)| format!("    (\"{name}\", {hash:#018x}),\n"))
        .collect();
    assert!(
        actual.as_slice() == PINNED,
        "noise output differs from the pinned hashes:\n{listing}"
    );

    let p2 = DVec2::new(0.3, 0.7);
    let p3 = DVec3::new(0.3, 0.7, -1.6);
    let (f1, f2) = worley3(SEED, p3);
    let samples = [
        value2(SEED, p2),
        gradient2(SEED, p2),
        value3(SEED, p3),
        gradient3(SEED, p3),
        f1,
        f2,
        fbm.sample2(SEED, p2),
        fbm.sample3(SEED, p3),
    ];
    assert!(
        samples == PINNED_SAMPLES,
        "sample values changed: {samples:?}"
    );
}

#[test]
fn same_input_same_output_and_seeds_differ() {
    let p2 = DVec2::new(3.7, -1.2);
    let p3 = DVec3::new(3.7, -1.2, 9.4);
    assert_eq!(gradient3(1, p3).to_bits(), gradient3(1, p3).to_bits());
    assert_ne!(gradient3(1, p3), gradient3(2, p3));
    assert_ne!(gradient2(1, p2), gradient2(2, p2));
    assert_ne!(value3(1, p3), value3(2, p3));
    assert_ne!(worley3(1, p3), worley3(2, p3));
    // The families are independent for the same seed: their correlation over many samples is nil.
    let mut rng = Rng::new(1, 9);
    let (mut vg, mut vv, mut gg) = (0.0, 0.0, 0.0);
    let n = 100_000;
    for _ in 0..n {
        let p = random3(&mut rng, 50.0);
        let (value, gradient) = (value3(SEED, p), gradient3(SEED, p));
        vg += value * gradient;
        vv += value * value;
        gg += gradient * gradient;
    }
    let correlation = vg / (vv * gg).sqrt();
    assert!(
        correlation.abs() < 0.02,
        "value/gradient correlation {correlation}"
    );
    // Neighbouring seeds are uncorrelated too.
    let mut rng = Rng::new(2, 9);
    let (mut ab, mut aa, mut bb) = (0.0, 0.0, 0.0);
    for _ in 0..n {
        let p = random3(&mut rng, 50.0);
        let (a, b) = (gradient3(7, p), gradient3(8, p));
        ab += a * b;
        aa += a * a;
        bb += b * b;
    }
    let correlation = ab / (aa * bb).sqrt();
    assert!(
        correlation.abs() < 0.02,
        "seed 7/8 correlation {correlation}"
    );
}

/// Running statistics of a sample stream.
#[derive(Default)]
struct Stats {
    n: f64,
    sum: f64,
    sum_squares: f64,
    min: f64,
    max: f64,
    /// Samples exactly at -1 or +1: the signature of a clamp that actually cut something.
    at_limit: u32,
}

impl Stats {
    fn new() -> Self {
        Self {
            min: f64::INFINITY,
            max: f64::NEG_INFINITY,
            ..Self::default()
        }
    }

    fn add(&mut self, value: f64) {
        assert!(
            (-1.0..=1.0).contains(&value),
            "sample {value} outside [-1, 1]"
        );
        self.n += 1.0;
        self.sum += value;
        self.sum_squares += value * value;
        self.min = det::min(self.min, value);
        self.max = det::max(self.max, value);
        self.at_limit += u32::from(value.abs() == 1.0);
    }

    fn mean(&self) -> f64 {
        self.sum / self.n
    }

    fn deviation(&self) -> f64 {
        (self.sum_squares / self.n - self.mean() * self.mean()).sqrt()
    }
}

#[test]
fn basis_noise_stays_in_range_with_the_documented_spread() {
    // Uniformly random points near the origin and far out, for the statistics.
    let mut rng = Rng::new(100, 1);
    let (mut v2, mut v3, mut g2, mut g3) = (Stats::new(), Stats::new(), Stats::new(), Stats::new());
    let (mut v2t, mut v3t, mut g2t, mut g3t) =
        (Stats::new(), Stats::new(), Stats::new(), Stats::new());
    for i in 0..1_000_000_u32 {
        // Wide extents, so that the samples fall into (mostly) different cells and the
        // statistics are those of the ensemble, not of a few hundred lattice points.
        let extent = [600.0, 3.0e6][(i % 2) as usize];
        let (p2, p3) = (random2(&mut rng, extent), random3(&mut rng, extent));
        let seed = SEED + u64::from(i % 7);
        v2.add(value2(seed, p2));
        v3.add(value3(seed, p3));
        g2.add(gradient2(seed, p2));
        g3.add(gradient3(seed, p3));
        // A tiled lattice has only period^n distinct cells, so the ensemble comes from the seeds.
        let (period, seed) = (2 + i % 8, SEED + u64::from(i));
        v2t.add(value2_tiled(seed, p2, period));
        v3t.add(value3_tiled(seed, p3, period));
        g2t.add(gradient2_tiled(seed, p2, period));
        g3t.add(gradient3_tiled(seed, p3, period));
    }
    // Cell centres, where gradient noise attains its extremes: range only.
    let (mut g2_centres, mut g3_centres) = (Stats::new(), Stats::new());
    for i in 0..3_000_000_u32 {
        let seed = SEED + u64::from(i % 5);
        g2_centres.add(gradient2(seed, random2(&mut rng, 4000.0).floor() + 0.5));
        g3_centres.add(gradient3(seed, random3(&mut rng, 4000.0).floor() + 0.5));
    }

    let report = format!(
        "deviation v2 {:.4} v3 {:.4} g2 {:.4} g3 {:.4} | \
         extremes v2 [{:.4}, {:.4}] v3 [{:.4}, {:.4}] g2 [{:.4}, {:.4}] g3 [{:.4}, {:.4}] | \
         at centres g2 [{:.4}, {:.4}] g3 [{:.4}, {:.4}]",
        v2.deviation(),
        v3.deviation(),
        g2.deviation(),
        g3.deviation(),
        v2.min,
        v2.max,
        v3.min,
        v3.max,
        g2.min,
        g2.max,
        g3.min,
        g3.max,
        g2_centres.min,
        g2_centres.max,
        g3_centres.min,
        g3_centres.max,
    );
    for (name, stats, deviation) in [
        ("value2", &v2, DEVIATION_VALUE2),
        ("value3", &v3, DEVIATION_VALUE3),
        ("gradient2", &g2, DEVIATION_GRADIENT2),
        ("gradient3", &g3, DEVIATION_GRADIENT3),
    ] {
        assert!(
            stats.mean().abs() < 0.004,
            "{name}: mean {} ({report})",
            stats.mean()
        );
        let measured = stats.deviation();
        assert!(
            (measured / deviation - 1.0).abs() < 0.02,
            "{name}: deviation {measured} ({report})"
        );
        // The distribution is symmetric about zero.
        assert!(
            (stats.max + stats.min).abs() < 0.1,
            "{name} is lopsided ({report})"
        );
    }
    // The scale factors are the mathematical bounds, not fudge factors: large samples use
    // most of the range, and the final clamp never has to cut a value.
    assert!(
        g2_centres.max > EXTREME_GRADIENT2 && g2_centres.min < -EXTREME_GRADIENT2,
        "{report}"
    );
    assert!(
        g3_centres.max > EXTREME_GRADIENT3 && g3_centres.min < -EXTREME_GRADIENT3,
        "{report}"
    );
    let clamped = [&g2, &g3, &g2t, &g3t, &g2_centres, &g3_centres].map(|stats| stats.at_limit);
    assert_eq!(clamped, [0; 6], "{report}");
    assert!(
        v2.max > 0.99 && v2.min < -0.99 && v3.max > 0.95 && v3.min < -0.95,
        "{report}"
    );
    // Tiling changes which lattice values are used, not how they are distributed.
    for (tiled, plain) in [(&v2t, &v2), (&v3t, &v3), (&g2t, &g2), (&g3t, &g3)] {
        assert!(tiled.mean().abs() < 0.01, "{report}");
        assert!(
            (tiled.deviation() / plain.deviation() - 1.0).abs() < 0.05,
            "{report}"
        );
    }
}

// Standard deviations quoted in the module documentation.
const DEVIATION_VALUE2: f64 = 0.45;
const DEVIATION_VALUE3: f64 = 0.40;
const DEVIATION_GRADIENT2: f64 = 0.305;
const DEVIATION_GRADIENT3: f64 = 0.22;
// Magnitudes that three million cell centres are certain to exceed.
const EXTREME_GRADIENT2: f64 = 0.99;
const EXTREME_GRADIENT3: f64 = 0.8;

#[test]
fn value_noise_interpolates_its_lattice_values() {
    let mut rng = Rng::new(101, 1);
    for _ in 0..2000 {
        let cell = random3(&mut rng, 1000.0).floor();
        let corner = |dx: f64, dy: f64, dz: f64| value3(SEED, cell + DVec3::new(dx, dy, dz));
        // Halfway along an edge, on a face centre and in the cell centre the fade weights are
        // exactly one half each, so the value is the plain average of the surrounding corners.
        let edge = value3(SEED, cell + DVec3::new(0.5, 0.0, 0.0));
        assert!((edge - 0.5 * (corner(0.0, 0.0, 0.0) + corner(1.0, 0.0, 0.0))).abs() < 1e-15);
        let mut sum = 0.0;
        for dz in [0.0, 1.0] {
            for dy in [0.0, 1.0] {
                for dx in [0.0, 1.0] {
                    sum += corner(dx, dy, dz);
                }
            }
        }
        assert!((value3(SEED, cell + 0.5) - sum / 8.0).abs() < 1e-15);
        // Inside a cell the value never leaves the range spanned by its corners.
        let (mut low, mut high) = (f64::INFINITY, f64::NEG_INFINITY);
        for dz in [0.0, 1.0] {
            for dy in [0.0, 1.0] {
                for dx in [0.0, 1.0] {
                    low = det::min(low, corner(dx, dy, dz));
                    high = det::max(high, corner(dx, dy, dz));
                }
            }
        }
        let inside = value3(SEED, cell + random3(&mut rng, 0.5) + 0.5);
        assert!(inside >= low - 1e-15 && inside <= high + 1e-15);

        let cell2 = random2(&mut rng, 1000.0).floor();
        let corner2 = |dx: f64, dy: f64| value2(SEED, cell2 + DVec2::new(dx, dy));
        let centre = value2(SEED, cell2 + 0.5);
        let average =
            0.25 * (corner2(0.0, 0.0) + corner2(1.0, 0.0) + corner2(0.0, 1.0) + corner2(1.0, 1.0));
        assert!((centre - average).abs() < 1e-15);
    }
}

#[test]
fn gradient_noise_is_zero_on_the_lattice_and_follows_its_gradients() {
    let mut rng = Rng::new(102, 1);
    for _ in 0..2000 {
        let corner3 = random3(&mut rng, 1000.0).floor();
        assert_eq!(gradient3(SEED, corner3), 0.0);
        let corner2 = random2(&mut rng, 1000.0).floor();
        assert_eq!(gradient2(SEED, corner2), 0.0);
        // Next to a lattice point the noise is the plane defined by that point's unit
        // gradient: the slope there has the length of the scale factor.
        let h = 1.0e-5;
        let slope = DVec3::new(
            gradient3(SEED, corner3 + DVec3::X * h) - gradient3(SEED, corner3 - DVec3::X * h),
            gradient3(SEED, corner3 + DVec3::Y * h) - gradient3(SEED, corner3 - DVec3::Y * h),
            gradient3(SEED, corner3 + DVec3::Z * h) - gradient3(SEED, corner3 - DVec3::Z * h),
        ) / (2.0 * h);
        assert!(
            (slope.length() - 2.0 / 3.0_f64.sqrt()).abs() < 1e-6,
            "{slope:?}"
        );
        let slope = DVec2::new(
            gradient2(SEED, corner2 + DVec2::X * h) - gradient2(SEED, corner2 - DVec2::X * h),
            gradient2(SEED, corner2 + DVec2::Y * h) - gradient2(SEED, corner2 - DVec2::Y * h),
        ) / (2.0 * h);
        assert!((slope.length() - SQRT_2).abs() < 1e-6, "{slope:?}");
    }
}

/// Second difference quotient of `f` at `x` with step `h` (approximates the second derivative).
fn second_difference(f: impl Fn(f64) -> f64, x: f64, h: f64) -> f64 {
    (f(x - h) - 2.0 * f(x) + f(x + h)) / (h * h)
}

#[test]
fn value_and_gradient_noise_are_c2_across_cell_boundaries() {
    // Walk along random lines through lattice planes. One-sided estimates of the first and
    // second derivative taken just before and just after the plane must agree; for a function
    // that is only C1 (cubic fade) the second derivative jumps by an amount of order 1.
    let mut rng = Rng::new(103, 1);
    // Steps for the slope and for the curvature estimates: small enough that the legitimate
    // change of a C2 function over them is far below the tolerances, large enough that
    // rounding in the difference quotients is too.
    let h = 1.0e-4;
    let h2 = 2.0e-5;
    let mut largest_curvature: f64 = 0.0;
    for case in 0..4000 {
        let origin = random3(&mut rng, 40.0);
        let direction = random3(&mut rng, 1.0).normalize();
        // The parameter t at which the line crosses the next lattice plane of one axis.
        let axis = case % 3;
        let plane = origin[axis].floor() + 1.0;
        if direction[axis].abs() < 0.2 {
            continue;
        }
        let t0 = (plane - origin[axis]) / direction[axis];
        let line = |t: f64| origin + direction * t;
        // (function along the line, whether it is two-dimensional). The last two are the raw
        // octave of `Fbm` on the unrotated lattice.
        let functions: [(&dyn Fn(f64) -> f64, bool); 6] = [
            (&|t| gradient3(SEED, line(t)), false),
            (&|t| value3(SEED, line(t)), false),
            (&|t| gradient2(SEED, line(t).truncate()), true),
            (&|t| value2(SEED, line(t).truncate()), true),
            (
                &|t| basis::value_gradient3_on(SEED, line(t), lattice::Unbounded),
                false,
            ),
            (
                &|t| basis::value_gradient2_on(SEED, line(t).truncate(), lattice::Unbounded),
                true,
            ),
        ];
        for (index, (f, two_dimensional)) in functions.iter().enumerate() {
            // The 2D functions only see the x and y planes.
            if *two_dimensional && axis == 2 {
                continue;
            }
            // Continuity of the value: no jump at the plane.
            assert!((f(t0 + 1e-9) - f(t0 - 1e-9)).abs() < 1e-7);
            // First derivative: one-sided differences on both sides.
            let before = (f(t0 - h) - f(t0 - 2.0 * h)) / h;
            let after = (f(t0 + 2.0 * h) - f(t0 + h)) / h;
            assert!(
                (before - after).abs() < 0.02,
                "case {case} f{index}: slope {before} vs {after}"
            );
            // Second derivative, centred two steps before and after the plane.
            let before = second_difference(f, t0 - 2.0 * h2, h2);
            let after = second_difference(f, t0 + 2.0 * h2, h2);
            largest_curvature = det::max(largest_curvature, before.abs());
            assert!(
                (before - after).abs() < 0.05,
                "case {case} f{index}: curvature {before} vs {after}"
            );
        }
    }
    // The test has teeth: curvatures of order 10 occur, so a jump of that order (which is
    // what a fade without zero second derivative produces) could not hide.
    assert!(largest_curvature > 5.0, "{largest_curvature}");
}

/// Largest change of the second difference quotient between neighbouring steps along a line,
/// and the largest second difference itself. For a C2 function the first is of the order of
/// `h` times its third derivative; a jump of the second derivative anywhere on the line shows
/// up with at least half its size (the stencil that straddles the jump averages both sides).
fn curvature_scan(f: impl Fn(f64) -> f64, steps: usize, h: f64) -> (f64, f64) {
    let values: Vec<f64> = (0..steps).map(|i| f(i as f64 * h)).collect();
    let curvatures: Vec<f64> = values
        .windows(3)
        .map(|w| (w[0] - 2.0 * w[1] + w[2]) / (h * h))
        .collect();
    let largest_jump = curvatures
        .windows(2)
        .map(|w| (w[1] - w[0]).abs())
        .fold(0.0, det::max);
    let largest = curvatures.iter().map(|c| c.abs()).fold(0.0, det::max);
    (largest_jump, largest)
}

#[test]
fn fractal_octaves_are_c2_everywhere() {
    // The octaves of `Fbm` live on rotated, shifted lattices, so their cell boundaries are not
    // where the test could aim at them. Scan random lines in small steps instead: wherever a
    // boundary is crossed, the curvature must not jump.
    let single = Fbm::new(1, 2.0, 0.5);
    let h = 2.0e-4;
    let steps = 20_000; // four lattice units per line
    // The scan can tell: x|x| is C1 with a curvature jump of 4 at its kink.
    let (jump, _) = curvature_scan(|t| (t - 1.0) * (t - 1.0).abs(), steps, h);
    assert!(jump > 1.99, "{jump}");

    let mut rng = Rng::new(113, 1);
    let (mut worst_jump, mut largest_curvature) = (0.0_f64, 0.0_f64);
    for _ in 0..40 {
        let origin = random3(&mut rng, 50.0);
        let direction = random3(&mut rng, 1.0).normalize();
        let line = |t: f64| origin + direction * t;
        let functions: [&dyn Fn(f64) -> f64; 4] = [
            &|t| single.sample3(SEED, line(t)),
            &|t| single.sample2(SEED, line(t).truncate()),
            &|t| single.sample3_tiled(SEED, line(t), 3),
            &|t| single.sample2_tiled(SEED, line(t).truncate(), 3),
        ];
        for f in functions {
            let (jump, largest) = curvature_scan(f, steps, h);
            worst_jump = det::max(worst_jump, jump);
            largest_curvature = det::max(largest_curvature, largest);
        }
    }
    // Curvatures above ten occur along the lines; the largest step-to-step change stays a
    // small fraction of that, as it must for continuous curvature.
    let report = format!("largest curvature {largest_curvature}, largest jump {worst_jump}");
    assert!(largest_curvature > 10.0, "{report}");
    assert!(worst_jump < 0.3, "{report}");
}

#[test]
fn derivatives_have_no_preferred_direction() {
    // Mean squared directional derivative of gradient noise along the axes, face diagonals
    // and space diagonals. For an unbiased gradient set these are all equal.
    let mut rng = Rng::new(104, 1);
    let h = 1.0e-4;
    let s = 1.0 / SQRT_2;
    let t = 1.0 / 3.0_f64.sqrt();
    let directions3 = [
        DVec3::X,
        DVec3::Y,
        DVec3::Z,
        DVec3::new(s, s, 0.0),
        DVec3::new(s, -s, 0.0),
        DVec3::new(0.0, s, s),
        DVec3::new(s, 0.0, -s),
        DVec3::new(t, t, t),
        DVec3::new(t, -t, t),
    ];
    let directions2 = [
        DVec2::X,
        DVec2::Y,
        DVec2::new(s, s),
        DVec2::new(s, -s),
        DVec2::new(0.6, 0.8),
    ];
    let mut energy3 = [0.0; 9];
    let mut energy2 = [0.0; 5];
    let n = 150_000;
    for _ in 0..n {
        let p = random3(&mut rng, 200.0);
        for (direction, energy) in directions3.iter().zip(&mut energy3) {
            let slope = (gradient3(SEED, p + *direction * h) - gradient3(SEED, p - *direction * h))
                / (2.0 * h);
            *energy += slope * slope / f64::from(n);
        }
        let p = p.truncate();
        for (direction, energy) in directions2.iter().zip(&mut energy2) {
            let slope = (gradient2(SEED, p + *direction * h) - gradient2(SEED, p - *direction * h))
                / (2.0 * h);
            *energy += slope * slope / f64::from(n);
        }
    }
    let mean3 = energy3.iter().sum::<f64>() / 9.0;
    for (direction, energy) in directions3.iter().zip(energy3) {
        assert!(
            (energy / mean3 - 1.0).abs() < 0.03,
            "3D {direction:?}: {energy} vs mean {mean3}"
        );
    }
    let mean2 = energy2.iter().sum::<f64>() / 5.0;
    for (direction, energy) in directions2.iter().zip(energy2) {
        assert!(
            (energy / mean2 - 1.0).abs() < 0.03,
            "2D {direction:?}: {energy} vs mean {mean2}"
        );
    }
}

/// Points whose coordinates are multiples of 1/64: adding an integer period to them is exact,
/// so a tileable function must return the identical bits.
fn dyadic2(rng: &mut Rng) -> DVec2 {
    DVec2::new(
        f64::from(rng.range_u32(0, 2048)),
        f64::from(rng.range_u32(0, 2048)),
    ) / 64.0
        - 16.0
}

fn dyadic3(rng: &mut Rng) -> DVec3 {
    DVec3::new(
        f64::from(rng.range_u32(0, 2048)),
        f64::from(rng.range_u32(0, 2048)),
        f64::from(rng.range_u32(0, 2048)),
    ) / 64.0
        - 16.0
}

#[test]
fn tiled_variants_repeat_exactly() {
    let fbm = Fbm::new(4, 2.0, 0.5);
    let triple = Fbm::new(3, 3.0, 0.6);
    let mut rng = Rng::new(105, 1);
    for case in 0..6000 {
        let period = [1_u32, 2, 3, 4, 7, 16][case % 6];
        let shift = f64::from(period);
        let p2 = dyadic2(&mut rng);
        let p3 = dyadic3(&mut rng);
        // Shift by whole periods along every axis and every combination of signs.
        let k = DVec3::new(
            f64::from(rng.range_u32(0, 7)) - 3.0,
            f64::from(rng.range_u32(0, 7)) - 3.0,
            f64::from(rng.range_u32(0, 7)) - 3.0,
        );
        let (q2, q3) = (p2 + k.truncate() * shift, p3 + k * shift);

        let scalar2: [&dyn Fn(DVec2) -> f64; 7] = [
            &|p| value2_tiled(SEED, p, period),
            &|p| gradient2_tiled(SEED, p, period),
            &|p| fbm.sample2_tiled(SEED, p, period),
            &|p| fbm.ridged2_tiled(SEED, p, period),
            &|p| fbm.billow2_tiled(SEED, p, period),
            &|p| triple.sample2_tiled(SEED, p, period),
            &|p| worley2_tiled(SEED, p, period).0 + 2.0 * worley2_tiled(SEED, p, period).1,
        ];
        for (index, f) in scalar2.iter().enumerate() {
            assert_eq!(
                f(p2).to_bits(),
                f(q2).to_bits(),
                "2D function {index}, period {period}"
            );
        }
        let scalar3: [&dyn Fn(DVec3) -> f64; 7] = [
            &|p| value3_tiled(SEED, p, period),
            &|p| gradient3_tiled(SEED, p, period),
            &|p| fbm.sample3_tiled(SEED, p, period),
            &|p| fbm.ridged3_tiled(SEED, p, period),
            &|p| fbm.billow3_tiled(SEED, p, period),
            &|p| triple.sample3_tiled(SEED, p, period),
            &|p| worley3_tiled(SEED, p, period).0 + 2.0 * worley3_tiled(SEED, p, period).1,
        ];
        for (index, f) in scalar3.iter().enumerate() {
            assert_eq!(
                f(p3).to_bits(),
                f(q3).to_bits(),
                "3D function {index}, period {period}"
            );
        }

        // Arbitrary (non-dyadic) points repeat up to the rounding of the shifted coordinate.
        let r3 = random3(&mut rng, 20.0);
        for f in &scalar3 {
            assert!((f(r3) - f(r3 + k * shift)).abs() < 1e-9);
        }
        for f in &scalar2 {
            assert!((f(r3.truncate()) - f(r3.truncate() + k.truncate() * shift)).abs() < 1e-9);
        }
    }
}

#[test]
fn tiled_variants_match_the_infinite_noise_away_from_the_seam() {
    // Inside the first period, wherever no lattice coordinate wraps, tiled and infinite
    // noise are the same function; and the period really is the period, not a divisor of it.
    let mut rng = Rng::new(106, 1);
    let period = 8_u32;
    let mut differs_from_half_period = false;
    for _ in 0..5000 {
        let p3 = DVec3::new(
            rng.range_f64(0.0, 7.0),
            rng.range_f64(0.0, 7.0),
            rng.range_f64(0.0, 7.0),
        );
        let p2 = p3.truncate();
        assert_eq!(value2_tiled(SEED, p2, period), value2(SEED, p2));
        assert_eq!(value3_tiled(SEED, p3, period), value3(SEED, p3));
        assert_eq!(gradient2_tiled(SEED, p2, period), gradient2(SEED, p2));
        assert_eq!(gradient3_tiled(SEED, p3, period), gradient3(SEED, p3));
        differs_from_half_period |=
            gradient3_tiled(SEED, p3, period) != gradient3_tiled(SEED, p3 + 4.0, period);
        // Worley looks two cells around, so the agreement holds a little further inside.
        let inner = p3 * (3.0 / 7.0) + 2.0;
        assert_eq!(
            worley2_tiled(SEED, inner.truncate(), period),
            worley2(SEED, inner.truncate())
        );
        assert_eq!(worley3_tiled(SEED, inner, period), worley3(SEED, inner));
    }
    assert!(differs_from_half_period);
    // A period of zero is treated as one.
    let p = DVec3::new(0.3, 0.6, 0.9);
    assert_eq!(gradient3_tiled(SEED, p, 0), gradient3_tiled(SEED, p, 1));
    assert_eq!(
        value2_tiled(SEED, p.truncate(), 0),
        value2_tiled(SEED, p.truncate(), 1)
    );
    assert_eq!(worley3_tiled(SEED, p, 0), worley3_tiled(SEED, p, 1));
}

#[test]
fn tiled_noise_is_seamless() {
    // Crossing the seam at the period boundary is as smooth as crossing any other cell
    // boundary: compare values and slopes a hair on either side.
    let fbm = Fbm::new(4, 2.0, 0.5);
    let mut rng = Rng::new(107, 1);
    let period = 4_u32;
    let e = 1.0e-7;
    for _ in 0..3000 {
        let y = rng.range_f64(-10.0, 10.0);
        let z = rng.range_f64(-10.0, 10.0);
        let edge = f64::from(period) * f64::from(rng.range_u32(0, 5)) - 8.0;
        let functions: [&dyn Fn(f64) -> f64; 6] = [
            &|x| gradient3_tiled(SEED, DVec3::new(x, y, z), period),
            &|x| value3_tiled(SEED, DVec3::new(x, y, z), period),
            &|x| fbm.sample3_tiled(SEED, DVec3::new(x, y, z), period),
            &|x| fbm.ridged2_tiled(SEED, DVec2::new(y, x), period),
            &|x| fbm.billow2_tiled(SEED, DVec2::new(x, z), period),
            &|x| worley3_tiled(SEED, DVec3::new(y, x, z), period).0,
        ];
        for (index, f) in functions.iter().enumerate() {
            let jump = (f(edge + e) - f(edge - e)).abs();
            assert!(jump < 1.0e-4, "function {index}: jump {jump} at the seam");
        }
    }
}

#[test]
fn fbm_stays_in_range_for_all_shapes_and_parameters() {
    let mut rng = Rng::new(108, 1);
    let parameter_sets = [
        Fbm::default(),
        Fbm::new(1, 2.0, 0.5),
        Fbm::new(8, 2.0, 0.5),
        Fbm::new(6, 2.0, 1.0),
        Fbm::new(4, 3.7, 0.9),
        Fbm::new(5, 1.0, 0.5),
        Fbm::new(3, 2.0, 2.5),
        Fbm::new(3, 2.0, -0.7),
        Fbm::new(32, 2.0, 0.8),
        Fbm::new(1000, 1.9, 0.6),
        Fbm::new(4, 2.0, 0.0),
    ];
    for fbm in parameter_sets {
        let mut all = Stats::new();
        for i in 0..6000 {
            let extent = [5.0, 300.0, 2.0e6][i % 3];
            let p3 = random3(&mut rng, extent);
            let p2 = p3.truncate();
            let period = 1 + (i % 5) as u32;
            for value in [
                fbm.sample2(SEED, p2),
                fbm.sample3(SEED, p3),
                fbm.ridged2(SEED, p2),
                fbm.ridged3(SEED, p3),
                fbm.billow2(SEED, p2),
                fbm.billow3(SEED, p3),
                fbm.sample2_tiled(SEED, p2, period),
                fbm.sample3_tiled(SEED, p3, period),
                fbm.ridged2_tiled(SEED, p2, period),
                fbm.ridged3_tiled(SEED, p3, period),
                fbm.billow2_tiled(SEED, p2, period),
                fbm.billow3_tiled(SEED, p3, period),
            ] {
                all.add(value);
            }
        }
        assert!(all.max - all.min > 0.5, "{fbm:?} is suspiciously flat");
    }
    // No octaves: silence.
    let silent = Fbm::new(0, 2.0, 0.5);
    let p = DVec3::new(1.5, 2.5, 3.5);
    assert_eq!(silent.sample3(SEED, p), 0.0);
    assert_eq!(silent.ridged2(SEED, p.truncate()), 0.0);
    assert_eq!(silent.billow3_tiled(SEED, p, 4), 0.0);
    assert_eq!(silent.domain_warp3(SEED, p, 1.0, 5.0), p);
    assert_eq!(Fbm::default(), Fbm::new(5, 2.0, 0.5));
    // A gain of zero leaves only the first octave.
    let first_only = Fbm::new(6, 2.0, 0.0);
    let single = Fbm::new(1, 2.0, 0.5);
    assert_eq!(first_only.sample3(SEED, p), single.sample3(SEED, p));
    // More than 32 octaves are ignored.
    assert_eq!(
        Fbm::new(1000, 2.0, 0.5).sample2(SEED, p.truncate()),
        Fbm::new(32, 2.0, 0.5).sample2(SEED, p.truncate())
    );
}

#[test]
fn fbm_shapes_have_their_character() {
    let fbm = Fbm::new(5, 2.0, 0.5);
    let single = Fbm::new(1, 2.0, 0.5);
    let mut rng = Rng::new(109, 1);
    // [plain, ridged, billow] in 3D and 2D, and single octaves in 3D and 2D.
    let mut stats3 = [Stats::new(), Stats::new(), Stats::new()];
    let mut stats2 = [Stats::new(), Stats::new(), Stats::new()];
    let (mut octave3, mut octave2) = (Stats::new(), Stats::new());
    for i in 0..400_000_u32 {
        // Many seeds and far-apart points: an ensemble, not one patch of one field.
        let seed = SEED + u64::from(i % 64);
        let p = random3(&mut rng, 5000.0);
        let q = p.truncate();
        stats3[0].add(fbm.sample3(seed, p));
        stats3[1].add(fbm.ridged3(seed, p));
        stats3[2].add(fbm.billow3(seed, p));
        stats2[0].add(fbm.sample2(seed, q));
        stats2[1].add(fbm.ridged2(seed, q));
        stats2[2].add(fbm.billow2(seed, q));
        octave3.add(single.sample3(seed, p));
        octave2.add(single.sample2(seed, q));
    }
    let describe = |stats: &Stats| {
        format!(
            "{:.3}±{:.3} [{:.3}, {:.3}]",
            stats.mean(),
            stats.deviation(),
            stats.min,
            stats.max
        )
    };
    let report = format!(
        "3D plain {} ridged {} billow {} | 2D plain {} ridged {} billow {} | octave 3D {} 2D {}",
        describe(&stats3[0]),
        describe(&stats3[1]),
        describe(&stats3[2]),
        describe(&stats2[0]),
        describe(&stats2[1]),
        describe(&stats2[2]),
        describe(&octave3),
        describe(&octave2),
    );
    // The documented spread of a single octave.
    assert!(
        (octave2.deviation() - OCTAVE_DEVIATION2).abs() < 0.005,
        "{report}"
    );
    assert!(
        (octave3.deviation() - OCTAVE_DEVIATION3).abs() < 0.005,
        "{report}"
    );
    assert!(
        octave2.mean().abs() < 0.005 && octave3.mean().abs() < 0.005,
        "{report}"
    );
    assert_eq!(octave2.at_limit + octave3.at_limit, 0, "{report}");
    for [plain, ridged, billow] in [&stats3, &stats2] {
        // Plain fBm is centred, symmetric and uses a good part of the range.
        assert!(plain.mean().abs() < 0.005, "{report}");
        assert!((plain.max + plain.min).abs() < 0.12, "{report}");
        assert!(
            plain.deviation() > 0.22 && plain.deviation() < 0.30,
            "{report}"
        );
        assert!(plain.max > 0.7 && plain.max < 1.0, "{report}");
        // Ridged spans crests near +1 and valleys near -1.
        assert!(ridged.max > 0.9 && ridged.min < -0.9, "{report}");
        assert!(ridged.deviation() > 0.3, "{report}");
        // Billow reaches from its creases near -1 well into the positive half.
        assert!(billow.min < -0.9 && billow.max > 0.6, "{report}");
        assert!(billow.mean() < -0.1 && billow.deviation() > 0.2, "{report}");
    }

    // Single octaves are the textbook formulas of the underlying noise n:
    // billow = 2|n| - 1, ridged = 2(1 - |n|)² - 1.
    for _ in 0..2000 {
        let p = random3(&mut rng, 100.0);
        let n = single.sample3(SEED, p);
        assert!((single.billow3(SEED, p) - (2.0 * n.abs() - 1.0)).abs() < 1e-15);
        assert!(
            (single.ridged3(SEED, p) - (2.0 * det::powi(1.0 - n.abs(), 2) - 1.0)).abs() < 1e-15
        );
        let p = p.truncate();
        let n = single.sample2(SEED, p);
        assert!((single.billow2(SEED, p) - (2.0 * n.abs() - 1.0)).abs() < 1e-15);
        assert!(
            (single.ridged2(SEED, p) - (2.0 * det::powi(1.0 - n.abs(), 2) - 1.0)).abs() < 1e-15
        );
    }
}

// Standard deviations of a single `Fbm` octave, as quoted in its documentation.
const OCTAVE_DEVIATION2: f64 = 0.437;
const OCTAVE_DEVIATION3: f64 = 0.424;

#[test]
fn fbm_has_no_lattice_artefacts() {
    // Single-octave gradient noise is exactly zero at every lattice point. If the octaves of a
    // fractal sum shared their lattice points (lacunarity 2 without offsets), the sum would be
    // pinned to zero on a regular grid, a visible artefact. Compare the energy on the integer
    // grid with the energy at random places.
    let fbm = Fbm::new(5, 2.0, 0.5);
    let mut rng = Rng::new(110, 1);
    let n = 40_000;
    let mut on_grid = [0.0_f64; 4];
    let mut off_grid = [0.0_f64; 4];
    for _ in 0..n {
        let p = random3(&mut rng, 300.0);
        let g = p.floor();
        let samples = |q: DVec3| {
            [
                fbm.sample3(SEED, q),
                fbm.sample2(SEED, q.truncate()),
                fbm.sample3_tiled(SEED, q, 16),
                fbm.sample2_tiled(SEED, q.truncate(), 16),
            ]
        };
        for (slot, value) in on_grid.iter_mut().zip(samples(g)) {
            *slot += value * value;
        }
        for (slot, value) in off_grid.iter_mut().zip(samples(p)) {
            *slot += value * value;
        }
    }
    for (index, (on, off)) in on_grid.iter().zip(off_grid).enumerate() {
        let ratio = (on / off).sqrt();
        if index < 2 {
            // Rotated octaves: the integer grid is like any other set of points.
            assert!(
                (0.93..1.07).contains(&ratio),
                "variant {index}: grid/off-grid RMS {ratio}"
            );
        } else {
            // Tileable octaves stay axis-aligned, so every grid point meets each octave at the
            // same place inside its cell (the octave's offset). The energy there differs from
            // the average, but it must not collapse.
            assert!(
                (0.5..1.6).contains(&ratio),
                "variant {index}: grid/off-grid RMS {ratio}"
            );
        }
    }
    // The origin is not special either: across seeds the value there varies like anywhere.
    let mut at_origin = Stats::new();
    for seed in 0..2000 {
        at_origin.add(fbm.sample3(seed, DVec3::ZERO));
        at_origin.add(fbm.sample2(seed, DVec2::ZERO));
        at_origin.add(fbm.sample2_tiled(seed, DVec2::ZERO, 8));
    }
    assert!(at_origin.deviation() > 0.1, "{}", at_origin.deviation());
}

#[test]
fn fbm_is_isotropic_along_world_directions() {
    // Mean squared directional derivative of 3D fBm along axes and diagonals. The octave
    // rotations spread every lattice's anisotropy over different directions, so no world
    // direction stands out.
    let fbm = Fbm::new(5, 2.0, 0.5);
    let mut rng = Rng::new(111, 1);
    let h = 1.0e-4;
    let s = 1.0 / SQRT_2;
    let t = 1.0 / 3.0_f64.sqrt();
    let directions = [
        DVec3::X,
        DVec3::Y,
        DVec3::Z,
        DVec3::new(s, s, 0.0),
        DVec3::new(0.0, s, -s),
        DVec3::new(s, 0.0, s),
        DVec3::new(t, t, t),
        DVec3::new(-t, t, t),
    ];
    let mut energy = [0.0; 8];
    let n = 60_000;
    for _ in 0..n {
        let p = random3(&mut rng, 200.0);
        for (direction, slot) in directions.iter().zip(&mut energy) {
            let slope = (fbm.sample3(SEED, p + *direction * h)
                - fbm.sample3(SEED, p - *direction * h))
                / (2.0 * h);
            *slot += slope * slope / f64::from(n);
        }
    }
    let mean = energy.iter().sum::<f64>() / 8.0;
    for (direction, value) in directions.iter().zip(energy) {
        assert!(
            (value / mean - 1.0).abs() < 0.04,
            "{direction:?}: {value} vs mean {mean}"
        );
    }
}

#[test]
fn domain_warp_displaces_by_at_most_the_amplitude() {
    let fbm = Fbm::new(3, 2.0, 0.5);
    let mut rng = Rng::new(112, 1);
    let (mut largest, mut sum) = (0.0_f64, DVec3::ZERO);
    let n = 20_000;
    for _ in 0..n {
        let p = random3(&mut rng, 1000.0);
        let amplitude = rng.range_f64(0.0, 50.0);
        let frequency = rng.range_f64(0.001, 2.0);
        let q = fbm.domain_warp3(SEED, p, frequency, amplitude);
        let d = (q - p) / det::max(amplitude, 1e-12);
        assert!(d.abs().max_element() <= 1.0 + 1e-9, "{d:?}");
        largest = det::max(largest, d.abs().max_element());
        sum += d;
        let q2 = fbm.domain_warp2(SEED, p.truncate(), frequency, amplitude);
        let d2 = (q2 - p.truncate()) / det::max(amplitude, 1e-12);
        assert!(d2.abs().max_element() <= 1.0 + 1e-9, "{d2:?}");
    }
    // The displacement is not degenerate and has no drift.
    assert!(largest > 0.4, "{largest}");
    assert!((sum / f64::from(n)).abs().max_element() < 0.02, "{sum:?}");
    // Zero amplitude is the identity; the channels are independent, not one noise repeated.
    let p = DVec3::new(12.3, -4.5, 6.7);
    assert_eq!(fbm.domain_warp3(SEED, p, 0.3, 0.0), p);
    assert_eq!(fbm.domain_warp2(SEED, p.truncate(), 0.3, 0.0), p.truncate());
    let d = fbm.domain_warp3(SEED, p, 0.3, 1.0) - p;
    assert!(d.x != d.y && d.y != d.z && d.x != d.z, "{d:?}");
    let d2 = fbm.domain_warp2(SEED, p.truncate(), 0.3, 1.0) - p.truncate();
    assert!(d2.x != d2.y, "{d2:?}");
    // The warp field has the requested frequency: it is the same field as with frequency 1 on
    // scaled coordinates.
    let a = fbm.domain_warp3(SEED, p, 0.25, 1.0) - p;
    let b = fbm.domain_warp3(SEED, p * 0.25, 1.0, 1.0) - p * 0.25;
    assert!((a - b).length() < 1e-12);
}

#[test]
fn extreme_inputs_never_panic() {
    let fbm = Fbm::default();
    let weird = [
        0.0,
        -0.0,
        1.0e-300,
        -1.0e-300,
        4.5e15,
        -4.5e15,
        9.0e18,
        -9.3e18,
        1.0e300,
        -1.0e300,
        f64::MAX,
        f64::MIN,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NAN,
    ];
    for &x in &weird {
        for &y in &weird {
            let p2 = DVec2::new(x, y);
            let p3 = DVec3::new(x, y, x);
            let finite = x.is_finite() && y.is_finite() && x.abs() < 4.0e18 && y.abs() < 4.0e18;
            let values = [
                value2(SEED, p2),
                value3(SEED, p3),
                gradient2(SEED, p2),
                gradient3(SEED, p3),
                value2_tiled(SEED, p2, 4),
                gradient3_tiled(SEED, p3, 4),
                fbm.sample2(SEED, p2),
                fbm.ridged3(SEED, p3),
                fbm.billow2_tiled(SEED, p2, 4),
            ];
            let (f1, f2) = worley2(SEED, p2);
            let (g1, g2) = worley3_tiled(SEED, p3, 4);
            if finite {
                // Within the documented domain the documented ranges hold.
                for value in values {
                    assert!((-1.0..=1.0).contains(&value), "{value} at {p3:?}");
                }
                assert!(
                    f1 >= 0.0 && f2 >= f1 && f2 <= 2.5_f64.sqrt(),
                    "{f1} {f2} at {p2:?}"
                );
                assert!(
                    g1 >= 0.0 && g2 >= g1 && g2 <= 3.0_f64.sqrt(),
                    "{g1} {g2} at {p3:?}"
                );
            }
        }
    }
}
