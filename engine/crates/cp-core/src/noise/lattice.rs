//! Building blocks shared by all noise functions: lattice hashing, coordinate splitting,
//! interpolation weights and the gradient tables.

use crate::hash::mix64;

/// How lattice coordinates are mapped before hashing: unchanged for infinite noise, wrapped
/// for tileable noise.
pub(super) trait Lattice: Copy {
    fn wrap(self, coordinate: i64) -> i64;
}

/// The infinite lattice.
#[derive(Clone, Copy)]
pub(super) struct Unbounded;

impl Lattice for Unbounded {
    #[inline(always)]
    fn wrap(self, coordinate: i64) -> i64 {
        coordinate
    }
}

/// A lattice that repeats every `period` cells on every axis.
#[derive(Clone, Copy)]
pub(super) struct Tiled(i64);

impl Tiled {
    /// `period` is clamped to at least 1.
    #[inline]
    pub(super) fn new(period: u64) -> Self {
        Self(i64::try_from(period.max(1)).unwrap_or(i64::MAX))
    }
}

impl Lattice for Tiled {
    #[inline(always)]
    fn wrap(self, coordinate: i64) -> i64 {
        coordinate.rem_euclid(self.0)
    }
}

/// Salts that make the noise families independent of each other for the same user seed.
pub(super) const SALT_VALUE: u64 = 0x5641_4C55_45A7_3D19;
pub(super) const SALT_GRADIENT: u64 = 0x4752_4144_9C0B_62E5;
pub(super) const SALT_VALUE_GRADIENT: u64 = 0x5647_5241_7B36_D40F;
pub(super) const SALT_WORLEY: u64 = 0x574F_524C_E1F3_58B7;
pub(super) const SALT_OCTAVE: u64 = 0x4F43_5441_2D8E_A6C3;

/// Scrambles the user seed once per sample so that nearby seeds give unrelated lattices.
#[inline(always)]
pub(super) fn seed_state(seed: u64, salt: u64) -> u64 {
    mix64(seed ^ salt)
}

/// Absorbs one lattice coordinate into a hash state. A bijection in `state` and in
/// `coordinate`, so distinct lattice points can only collide by chance; unlike the common
/// `x * P1 ^ y * P2 ^ z * P3` scheme it has no translation that maps the lattice onto itself.
#[inline(always)]
pub(super) fn fold(state: u64, coordinate: i64) -> u64 {
    let x = (state ^ coordinate as u64).wrapping_mul(0xD6E8_FEB8_6659_FD93);
    x ^ (x >> 32)
}

/// Final avalanche of a folded state: every output bit depends on every coordinate bit.
#[inline(always)]
pub(super) fn finish(state: u64) -> u64 {
    mix64(state)
}

/// Splits `x` into `floor(x)` and the fraction in `[0, 1)`.
///
/// Uses a truncating cast instead of `f64::floor`, which is a library call on baseline x86-64.
/// Exact for all finite `|x| < 2^63`; beyond that the cast saturates and the fraction is
/// meaningless (the public functions clamp their results, so the range still holds).
#[inline(always)]
pub(super) fn split(x: f64) -> (i64, f64) {
    let truncated = x as i64;
    let floor = truncated.wrapping_sub(i64::from(x < truncated as f64));
    (floor, x - floor as f64)
}

/// Quintic fade `6t^5 - 15t^4 + 10t^3` (Perlin 2002): zero first and second derivative at
/// 0 and 1, which makes the interpolated noise C2-continuous across cell boundaries.
#[inline(always)]
pub(super) fn fade(t: f64) -> f64 {
    t * t * t * (t * (t * 6.0 - 15.0) + 10.0)
}

#[inline(always)]
pub(super) fn lerp(a: f64, b: f64, t: f64) -> f64 {
    a + (b - a) * t
}

/// Maps a hash to a lattice value in `[-1, 1)` using its top 53 bits.
#[inline(always)]
pub(super) fn signed_unit(hash: u64) -> f64 {
    (hash >> 11) as f64 * (1.0 / 4_503_599_627_370_496.0) - 1.0
}

// ---------------------------------------------------------------------------------------------
// Gradient tables
// ---------------------------------------------------------------------------------------------
//
// The tables are computed at compile time with the functions below, which use only `+ - * /`
// on `f64`. Compile-time float arithmetic is IEEE-exact and independent of the host, so the
// tables have the same bits everywhere (contract: "no tables with platform-dependent contents").

/// Number of gradient directions; a power of two so that hash bits index it directly.
pub(super) const GRADIENT_COUNT: usize = 256;

/// Square root by Newton's iteration, for `0 < x <= 16`. Accurate to an ulp.
pub(super) const fn const_sqrt(x: f64) -> f64 {
    let mut y = if x > 1.0 { x } else { 1.0 };
    let mut i = 0;
    while i < 48 {
        y = 0.5 * (y + x / y);
        i += 1;
    }
    y
}

/// `(sin, cos)` of an angle given in turns (`0 <= turns < 1`), by Taylor series on the first
/// quadrant.
pub(super) const fn const_sin_cos_turns(turns: f64) -> (f64, f64) {
    let quarter_turns = turns * 4.0;
    let quadrant = quarter_turns as u32;
    let r = (quarter_turns - quadrant as f64) * core::f64::consts::FRAC_PI_2;
    let r2 = r * r;
    let (mut sin, mut cos) = (r, 1.0);
    let (mut sin_term, mut cos_term) = (r, 1.0);
    let mut k = 1;
    while k <= 14 {
        let n = (2 * k) as f64;
        cos_term = -cos_term * r2 / ((n - 1.0) * n);
        sin_term = -sin_term * r2 / (n * (n + 1.0));
        cos += cos_term;
        sin += sin_term;
        k += 1;
    }
    match quadrant & 3 {
        0 => (sin, cos),
        1 => (cos, -sin),
        2 => (-sin, -cos),
        _ => (-cos, sin),
    }
}

/// 256 unit vectors at evenly spaced angles `(i + 0.5) * 360° / 256`.
///
/// Evenly spaced directions make the gradient distribution isotropic (no preferred axis or
/// diagonal, unlike the 4 or 8 directions of classic Perlin noise). Entry `i + 128` is the
/// exact negative of entry `i`, so the mean gradient is exactly zero.
const fn build_gradients2() -> [[f64; 2]; GRADIENT_COUNT] {
    let mut table = [[0.0; 2]; GRADIENT_COUNT];
    let half = GRADIENT_COUNT / 2;
    let mut i = 0;
    while i < half {
        let (sin, cos) = const_sin_cos_turns((i as f64 + 0.5) / GRADIENT_COUNT as f64);
        let length = const_sqrt(sin * sin + cos * cos);
        let (x, y) = (cos / length, sin / length);
        table[i] = [x, y];
        table[i + half] = [-x, -y];
        i += 1;
    }
    table
}

/// 256 unit vectors spread evenly over the sphere: a Fibonacci spiral of 128 points on the
/// upper hemisphere plus their antipodes.
///
/// The spiral covers the hemisphere with nearly equal area per point, so no axis or diagonal is
/// preferred (classic Perlin noise uses the 12 cube-edge directions and shows it). Entries
/// `2i` and `2i + 1` are exact negatives, so the mean gradient is exactly zero.
const fn build_gradients3() -> [[f64; 3]; GRADIENT_COUNT] {
    /// 1 / golden ratio: successive azimuths advance by this fraction of a turn.
    const GOLDEN_TURN: f64 = 0.618_033_988_749_894_9;
    let mut table = [[0.0; 3]; GRADIENT_COUNT];
    let half = GRADIENT_COUNT / 2;
    let mut i = 0;
    while i < half {
        // Equal steps in z give equal areas on the sphere (Archimedes).
        let z = (i as f64 + 0.5) / half as f64;
        let radius = const_sqrt(1.0 - z * z);
        let turns = i as f64 * GOLDEN_TURN;
        let turns = turns - (turns as u32) as f64;
        let (sin, cos) = const_sin_cos_turns(turns);
        let (x, y) = (radius * cos, radius * sin);
        let length = const_sqrt(x * x + y * y + z * z);
        let (x, y, z) = (x / length, y / length, z / length);
        table[2 * i] = [x, y, z];
        table[2 * i + 1] = [-x, -y, -z];
        i += 1;
    }
    table
}

pub(super) static GRADIENTS2: [[f64; 2]; GRADIENT_COUNT] = build_gradients2();
pub(super) static GRADIENTS3: [[f64; 3]; GRADIENT_COUNT] = build_gradients3();

/// Classic gradient noise with unit gradients reaches at most `sqrt(n) / 2` in `n` dimensions
/// (at a cell centre, when every corner gradient points at it). These factors scale that bound
/// to 1.
pub(super) const GRADIENT2_SCALE: f64 = core::f64::consts::SQRT_2;
pub(super) const GRADIENT3_SCALE: f64 = 2.0 / const_sqrt(3.0);

/// Index of the gradient for a corner hash: its top eight bits.
#[inline(always)]
pub(super) fn gradient_index(hash: u64) -> usize {
    (hash >> 56) as usize
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::det;

    #[test]
    fn const_math_is_accurate() {
        for x in [1.0e-4, 0.0078, 0.25, 0.5, 1.0, 2.0, 3.0, 4.0] {
            let error = (const_sqrt(x) - f64::sqrt(x)).abs();
            assert!(error <= f64::EPSILON * f64::sqrt(x), "sqrt({x}): {error:e}");
        }
        for i in 0..1000 {
            let turns = f64::from(i) / 1000.0;
            let (sin, cos) = const_sin_cos_turns(turns);
            // The reference rounds its argument `turns * TAU` once, which alone is worth a
            // few 1e-16; the comparison cannot be tighter than that.
            let angle = turns * core::f64::consts::TAU;
            assert!(
                (sin - det::sin(angle)).abs() < 1.0e-15,
                "sin({turns} turns)"
            );
            assert!(
                (cos - det::cos(angle)).abs() < 1.0e-15,
                "cos({turns} turns)"
            );
            // (The table builders renormalise, so the entries are unit length to the last bit.)
            assert!(
                (sin * sin + cos * cos - 1.0).abs() < 1.0e-15,
                "unit length at {turns} turns"
            );
        }
        assert!((GRADIENT3_SCALE - 2.0 / f64::sqrt(3.0)).abs() <= f64::EPSILON);
        assert_eq!(GRADIENT2_SCALE * GRADIENT2_SCALE, 2.000_000_000_000_000_4);
    }

    #[test]
    fn split_is_floor_and_fraction() {
        for x in [
            0.0, 0.5, 1.0, 1.25, -0.25, -1.0, -1.5, 7.999_999, -7.999_999, 1.0e15, -1.0e15,
        ] {
            let (floor, fraction) = split(x);
            assert_eq!(floor as f64, x.floor(), "{x}");
            assert_eq!(fraction, x - x.floor(), "{x}");
            assert!((0.0..1.0).contains(&fraction), "{x}");
        }
        assert_eq!(split(-0.0), (0, 0.0));
        // Just below an integer: the fraction rounds to at most the largest double below 1.
        let (floor, fraction) = split(-1.0e-300);
        assert_eq!(floor, -1);
        assert!(fraction <= 1.0);
        // Out-of-range and non-finite inputs never panic.
        for x in [
            1.0e300,
            -1.0e300,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::NAN,
        ] {
            let _ = split(x);
        }
    }

    #[test]
    fn fade_is_c2_at_the_ends() {
        assert_eq!(fade(0.0), 0.0);
        assert_eq!(fade(1.0), 1.0);
        assert_eq!(fade(0.5), 0.5);
        let h = 1.0e-3;
        // Value, slope and curvature vanish at 0 and mirror at 1: fade(h) = 10 h^3 + O(h^4).
        assert!((fade(h) - 10.0 * h * h * h).abs() < 2.0e-11);
        assert!(((1.0 - fade(1.0 - h)) - 10.0 * h * h * h).abs() < 2.0e-11);
        let mut previous = 0.0;
        for i in 0..=1000 {
            let t = f64::from(i) / 1000.0;
            let value = fade(t);
            assert!(value >= previous && (0.0..=1.0).contains(&value));
            // Point symmetry about (1/2, 1/2), up to a few ulps of rounding in the polynomial.
            assert!((value + fade(1.0 - t) - 1.0).abs() < 4.0e-15);
            previous = value;
        }
    }

    #[test]
    fn signed_unit_covers_the_interval() {
        assert_eq!(signed_unit(0), -1.0);
        assert_eq!(signed_unit(u64::MAX), 1.0 - f64::EPSILON);
        assert_eq!(signed_unit(1 << 63), 0.0);
    }

    #[test]
    fn tiled_lattice_wraps_with_floor_semantics() {
        let lattice = Tiled::new(4);
        assert_eq!(
            [-5, -4, -1, 0, 3, 4, 9].map(|c| lattice.wrap(c)),
            [3, 0, 3, 0, 3, 0, 1]
        );
        assert_eq!(Tiled::new(0).wrap(17), 0);
        assert_eq!(Tiled::new(u64::MAX).wrap(-1), i64::MAX - 1);
        assert_eq!(Unbounded.wrap(-5), -5);
    }

    #[test]
    fn lattice_hash_is_well_distributed() {
        // No collisions on a block of lattice points, unbiased bits, and gradient indices
        // (top byte) that are uniform and uncorrelated between neighbours.
        let state = seed_state(1, SALT_GRADIENT);
        let mut seen = std::collections::HashSet::new();
        let mut ones = [0_u32; 64];
        let mut histogram = [0_u32; GRADIENT_COUNT];
        let mut equal_neighbours = 0_u32;
        let mut count = 0_u32;
        for x in -24..24_i64 {
            for y in -24..24_i64 {
                for z in -24..24_i64 {
                    let hash = finish(fold(fold(fold(state, x), y), z));
                    assert!(seen.insert(hash));
                    for (bit, slot) in ones.iter_mut().enumerate() {
                        *slot += ((hash >> bit) & 1) as u32;
                    }
                    histogram[gradient_index(hash)] += 1;
                    let neighbour = finish(fold(fold(fold(state, x + 1), y), z));
                    equal_neighbours +=
                        u32::from(gradient_index(hash) == gradient_index(neighbour));
                    count += 1;
                }
            }
        }
        for (bit, &n) in ones.iter().enumerate() {
            let p = f64::from(n) / f64::from(count);
            assert!((p - 0.5).abs() < 0.006, "bit {bit}: {p}");
        }
        // 110592 samples in 256 bins: mean 432, standard deviation about 21.
        for (index, &n) in histogram.iter().enumerate() {
            assert!((330..540).contains(&n), "gradient {index} chosen {n} times");
        }
        let expected = f64::from(count) / 256.0;
        assert!((f64::from(equal_neighbours) - expected).abs() < 5.0 * expected.sqrt());
        // Different seeds and different salts give different lattices.
        let other_seed = seed_state(2, SALT_GRADIENT);
        let other_salt = seed_state(1, SALT_VALUE);
        let h = |s: u64| finish(fold(fold(s, 3), -7));
        assert_ne!(h(state), h(other_seed));
        assert_ne!(h(state), h(other_salt));
        // Order of coordinates matters.
        assert_ne!(
            finish(fold(fold(state, 3), -7)),
            finish(fold(fold(state, -7), 3))
        );
    }

    #[test]
    fn gradient_tables_are_unit_balanced_and_isotropic() {
        // 2D: unit length, exact antipodes, evenly spaced angles.
        let mut sum = [0.0_f64; 2];
        let mut moment = [[0.0_f64; 2]; 2];
        for (i, g) in GRADIENTS2.iter().enumerate() {
            assert!((g[0] * g[0] + g[1] * g[1] - 1.0).abs() < 4.0e-16);
            let opposite = GRADIENTS2[(i + GRADIENT_COUNT / 2) % GRADIENT_COUNT];
            assert_eq!([-g[0], -g[1]], opposite);
            let angle = (i as f64 + 0.5) * core::f64::consts::TAU / GRADIENT_COUNT as f64;
            assert!(
                (g[0] - det::cos(angle)).abs() < 1.0e-15
                    && (g[1] - det::sin(angle)).abs() < 1.0e-15
            );
            for a in 0..2 {
                sum[a] += g[a];
                for b in 0..2 {
                    moment[a][b] += g[a] * g[b] / GRADIENT_COUNT as f64;
                }
            }
        }
        // Exact antipodes make the true sum zero; the running sum only carries rounding noise.
        assert!(sum[0].abs() < 1.0e-13 && sum[1].abs() < 1.0e-13, "{sum:?}");
        for (a, row) in moment.iter().enumerate() {
            for (b, entry) in row.iter().enumerate() {
                let expected = if a == b { 0.5 } else { 0.0 };
                assert!((entry - expected).abs() < 1.0e-14, "2D moment {a}{b}");
            }
        }

        // 3D: unit length, exact antipodes, zero mean, second moment I/3 (isotropy), and no two
        // directions closer than a good covering of the sphere allows.
        let mut sum = [0.0_f64; 3];
        let mut moment = [[0.0_f64; 3]; 3];
        for (i, g) in GRADIENTS3.iter().enumerate() {
            assert!((g[0] * g[0] + g[1] * g[1] + g[2] * g[2] - 1.0).abs() < 4.0e-16);
            let opposite = GRADIENTS3[i ^ 1];
            assert_eq!([-g[0], -g[1], -g[2]], opposite);
            for a in 0..3 {
                sum[a] += g[a];
                for b in 0..3 {
                    moment[a][b] += g[a] * g[b] / GRADIENT_COUNT as f64;
                }
            }
        }
        // Antipodes are adjacent entries, so the running sum returns to exactly zero each time.
        assert_eq!(sum, [0.0, 0.0, 0.0]);
        for (a, row) in moment.iter().enumerate() {
            for (b, entry) in row.iter().enumerate() {
                let expected = if a == b { 1.0 / 3.0 } else { 0.0 };
                // Measured: within 0.002 of isotropic, under 1 % of the diagonal.
                assert!(
                    (entry - expected).abs() < 0.004,
                    "3D moment {a}{b}: {entry}"
                );
            }
        }
        // Fourth moments tell an even spread from a merely balanced one: for directions uniform
        // on the sphere the mean of (g · d)^4 is 1/5 for every unit vector d, on the circle 3/8.
        let s = core::f64::consts::FRAC_1_SQRT_2;
        let t = 1.0 / f64::sqrt(3.0);
        let probes = [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [s, s, 0.0],
            [s, 0.0, -s],
            [0.0, s, s],
            [t, t, t],
            [t, -t, t],
            [0.36, 0.48, 0.8],
        ];
        let fourth_moment = |set: &[[f64; 3]], d: &[f64; 3]| {
            set.iter()
                .map(|g| det::powi(g[0] * d[0] + g[1] * d[1] + g[2] * d[2], 4))
                .sum::<f64>()
                / set.len() as f64
        };
        for d in &probes {
            let moment3 = fourth_moment(&GRADIENTS3, d);
            assert!(
                (moment3 / 0.2 - 1.0).abs() < 0.02,
                "3D fourth moment along {d:?}: {moment3}"
            );
            let moment2 = GRADIENTS2
                .iter()
                .map(|g| det::powi(g[0] * d[0] + g[1] * d[1], 4))
                .sum::<f64>()
                / GRADIENT_COUNT as f64;
            let expected = 0.375 * det::powi(d[0] * d[0] + d[1] * d[1], 2);
            assert!(
                (moment2 - expected).abs() < 1.0e-14,
                "2D fourth moment along {d:?}"
            );
        }
        // The same measure exposes the 12 cube-edge gradients of classic Perlin noise: 17 % too
        // little along the axes, 11 % too much along the space diagonals.
        let classic: Vec<[f64; 3]> = [
            [1.0, 1.0, 0.0],
            [1.0, -1.0, 0.0],
            [1.0, 0.0, 1.0],
            [1.0, 0.0, -1.0],
            [0.0, 1.0, 1.0],
            [0.0, 1.0, -1.0],
        ]
        .iter()
        .flat_map(|g| {
            [
                [g[0] * s, g[1] * s, g[2] * s],
                [-g[0] * s, -g[1] * s, -g[2] * s],
            ]
        })
        .collect();
        assert!((fourth_moment(&classic, &probes[0]) / 0.2 - 1.0).abs() > 0.15);
        assert!((fourth_moment(&classic, &probes[6]) / 0.2 - 1.0).abs() > 0.10);

        // 256 points on a sphere of area 4π: an ideal packing has neighbours about
        // sqrt(4π / 256) ≈ 0.22 apart. No pair may be much closer, no point isolated.
        for (i, a) in GRADIENTS3.iter().enumerate() {
            let mut nearest = f64::INFINITY;
            for (j, b) in GRADIENTS3.iter().enumerate() {
                if i != j {
                    let d2 = det::powi(a[0] - b[0], 2)
                        + det::powi(a[1] - b[1], 2)
                        + det::powi(a[2] - b[2], 2);
                    nearest = det::min(nearest, d2.sqrt());
                }
            }
            assert!(
                (0.13..0.23).contains(&nearest),
                "direction {i}: nearest neighbour {nearest}"
            );
        }
    }
}
