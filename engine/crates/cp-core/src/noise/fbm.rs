//! Fractal sums of value-gradient noise.

use serde::{Deserialize, Serialize};

use super::basis::{value_gradient2_on, value_gradient3_on};
use super::lattice::{SALT_OCTAVE, Tiled, Unbounded, const_sin_cos_turns, const_sqrt};
use crate::det;
use crate::hash::{hash_combine, mix64};
use crate::math::{DVec2, DVec3};

/// More octaves than this add nothing visible (at the usual gain of 0.5 the 33rd octave would
/// have an amplitude of 2^-32), so the count is capped to bound the cost of bad parameters.
const MAX_OCTAVES: u32 = 32;

/// `2 - φ ≈ 0.382`: the fraction of a turn whose multiples are spread most evenly around the
/// circle (the golden angle is this fraction of 360°).
const GOLDEN_FRACTION: f64 = 1.0 - 0.618_033_988_749_894_9;

/// Per-octave rotation in 2D: the golden fraction of a quarter turn, about 34.4°.
///
/// A square lattice looks the same after a quarter turn, so what matters is the angle modulo
/// 90°. Golden-ratio steps keep the orientations of successive octaves as far apart from each
/// other and from the axes as possible; no two octaves ever share lattice directions.
const ROTATION2: [[f64; 2]; 2] = {
    let (sin, cos) = const_sin_cos_turns(0.25 * GOLDEN_FRACTION);
    [[cos, -sin], [sin, cos]]
};

/// Per-octave rotation in 3D: the golden angle (about 137.5°) about the axis
/// `(1, 2, 3) / sqrt(14)`.
///
/// The axis is well away from every symmetry axis of the cube, so no power of the rotation
/// maps the lattice onto itself, and golden-angle steps keep successive octaves far from each
/// other's orientation.
const ROTATION3: [[f64; 3]; 3] = {
    let length = const_sqrt(14.0);
    let (x, y, z) = (1.0 / length, 2.0 / length, 3.0 / length);
    let (sin, cos) = const_sin_cos_turns(GOLDEN_FRACTION);
    let k = 1.0 - cos;
    // Rodrigues' formula: cos·I + (1 - cos)·a·aᵀ + sin·[a]×.
    [
        [cos + k * x * x, k * x * y - sin * z, k * x * z + sin * y],
        [k * y * x + sin * z, cos + k * y * y, k * y * z - sin * x],
        [k * z * x - sin * y, k * z * y + sin * x, cos + k * z * z],
    ]
};

/// How the octaves are shaped before they are summed.
#[derive(Clone, Copy)]
enum Shape {
    /// `n`: classic fractal Brownian motion.
    Plain,
    /// `2|n| - 1`: creases at the zero crossings point down (puffy clouds, dunes).
    Billow,
    /// `(1 - |n|)²`, weighted by the octave before: sharp ridges with detail on the ridges and
    /// smooth valleys (Musgrave's ridged multifractal).
    Ridged,
}

/// Fractal noise: a sum of `octaves` layers of noise, each `lacunarity` times finer and `gain`
/// times weaker than the one before.
///
/// All sampling methods return values in `[-1, 1]` (the sum is divided by the total amplitude)
/// and take coordinates in lattice units of the first octave. With `octaves == 0` they
/// return 0.
///
/// # Construction
/// Each octave is *value-gradient noise*: every lattice point carries a random value and a
/// random gradient, blended with the quintic fade. Unlike plain gradient noise it is not
/// pinned to zero on the lattice, so thresholds, ridges and creases do not follow a grid, and
/// its spread varies little across a cell (by about 12 % in 2D and 30 % in 3D, where gradient
/// noise goes from zero to its maximum).
///
/// The raw octave rarely gets far from zero (it only reaches ±1 when all corners of a cell
/// conspire), which would leave most of the output range unused and the ridged and billow
/// shapes flat. Each octave therefore passes through a fixed smooth contrast curve, an odd
/// polynomial that maps `[-1, 1]` onto itself with slope 1.9 (2D) or 2.5 (3D) at zero and
/// slope zero at the ends. Nothing is clipped and the octave stays C2; afterwards it has a
/// standard deviation of about 0.44 in 2D and 0.42 in 3D, and a five-octave sum with the
/// default parameters one of about 0.26.
///
/// Every octave has its own seed, a fixed rotation (the golden angle in 3D, the golden fraction
/// of a quarter turn in 2D) and a pseudo-random offset, so the lattices of different octaves
/// share neither their directions nor their points, and none is aligned with the world axes.
/// The tileable variants cannot rotate (a rotated lattice does not tile) and use the offset
/// only.
#[derive(Clone, Copy, PartialEq, Debug, Serialize, Deserialize)]
pub struct Fbm {
    /// Number of layers, at most 32 (more are ignored). Cost is proportional to it.
    pub octaves: u32,
    /// Frequency ratio between successive octaves, usually 2. The tileable variants round it
    /// to the nearest integer `>= 1`, because only integer ratios tile.
    pub lacunarity: f64,
    /// Amplitude ratio between successive octaves, usually 0.5. Smaller is smoother.
    pub gain: f64,
}

impl Default for Fbm {
    /// Five octaves, lacunarity 2, gain 0.5.
    fn default() -> Self {
        Self::new(5, 2.0, 0.5)
    }
}

impl Fbm {
    /// Fractal noise with the given number of octaves, frequency ratio and amplitude ratio.
    #[inline]
    #[must_use]
    pub const fn new(octaves: u32, lacunarity: f64, gain: f64) -> Self {
        Self {
            octaves,
            lacunarity,
            gain,
        }
    }

    /// 2D fractal Brownian motion in `[-1, 1]`: centred on zero and symmetric. With the default
    /// parameters the standard deviation is about 0.26 and values beyond ±0.8 are rare.
    #[inline]
    #[must_use]
    pub fn sample2(&self, seed: u64, p: DVec2) -> f64 {
        self.sum2(seed, p, Shape::Plain)
    }

    /// 3D fractal Brownian motion in `[-1, 1]`, see [`Fbm::sample2`].
    #[inline]
    #[must_use]
    pub fn sample3(&self, seed: u64, p: DVec3) -> f64 {
        self.sum3(seed, p, Shape::Plain)
    }

    /// 2D ridged multifractal in `[-1, 1]`: sharp crests near +1 along the zero crossings of
    /// the underlying noise, smooth valleys towards -1. Mountain ranges, cracks, veins.
    ///
    /// Every octave contributes `2 w (1 - |n|)² - 1`, where `n` is the octave and the weight `w`
    /// is `min(1, 2 (1 - |n|)²)` of the octave before (1 for the first), so detail gathers on
    /// the ridges (Musgrave). With the default parameters the mean is about -0.2 and the
    /// standard deviation 0.4; the whole range is used.
    #[inline]
    #[must_use]
    pub fn ridged2(&self, seed: u64, p: DVec2) -> f64 {
        self.sum2(seed, p, Shape::Ridged)
    }

    /// 3D ridged multifractal in `[-1, 1]`, see [`Fbm::ridged2`].
    #[inline]
    #[must_use]
    pub fn ridged3(&self, seed: u64, p: DVec3) -> f64 {
        self.sum3(seed, p, Shape::Ridged)
    }

    /// 2D billow noise in `[-1, 1]`: rounded bumps separated by sharp creases near -1.
    ///
    /// Every octave contributes `2|n| - 1`. With the default parameters the mean is about -0.3
    /// and the standard deviation 0.28; values rarely exceed +0.7.
    #[inline]
    #[must_use]
    pub fn billow2(&self, seed: u64, p: DVec2) -> f64 {
        self.sum2(seed, p, Shape::Billow)
    }

    /// 3D billow noise in `[-1, 1]`, see [`Fbm::billow2`].
    #[inline]
    #[must_use]
    pub fn billow3(&self, seed: u64, p: DVec3) -> f64 {
        self.sum3(seed, p, Shape::Billow)
    }

    /// [`Fbm::sample2`] that repeats every `period` lattice units on both axes
    /// (`period >= 1`, zero is treated as 1).
    #[inline]
    #[must_use]
    pub fn sample2_tiled(&self, seed: u64, p: DVec2, period: u32) -> f64 {
        self.sum2_tiled(seed, p, period, Shape::Plain)
    }

    /// [`Fbm::sample3`] that repeats every `period` lattice units on all axes.
    #[inline]
    #[must_use]
    pub fn sample3_tiled(&self, seed: u64, p: DVec3, period: u32) -> f64 {
        self.sum3_tiled(seed, p, period, Shape::Plain)
    }

    /// [`Fbm::ridged2`] that repeats every `period` lattice units on both axes.
    #[inline]
    #[must_use]
    pub fn ridged2_tiled(&self, seed: u64, p: DVec2, period: u32) -> f64 {
        self.sum2_tiled(seed, p, period, Shape::Ridged)
    }

    /// [`Fbm::ridged3`] that repeats every `period` lattice units on all axes.
    #[inline]
    #[must_use]
    pub fn ridged3_tiled(&self, seed: u64, p: DVec3, period: u32) -> f64 {
        self.sum3_tiled(seed, p, period, Shape::Ridged)
    }

    /// [`Fbm::billow2`] that repeats every `period` lattice units on both axes.
    #[inline]
    #[must_use]
    pub fn billow2_tiled(&self, seed: u64, p: DVec2, period: u32) -> f64 {
        self.sum2_tiled(seed, p, period, Shape::Billow)
    }

    /// [`Fbm::billow3`] that repeats every `period` lattice units on all axes.
    #[inline]
    #[must_use]
    pub fn billow3_tiled(&self, seed: u64, p: DVec3, period: u32) -> f64 {
        self.sum3_tiled(seed, p, period, Shape::Billow)
    }

    /// Domain warp: returns `p` displaced by a 2D vector of fractal noise,
    /// `p + amplitude * (n1, n2)(p * frequency)`.
    ///
    /// Sample another noise at the returned position to bend its features. `frequency` sets the
    /// size of the bends (in the units of `p`), `amplitude` their strength: each component
    /// moves by at most `amplitude`. The two noise channels are independent of each other and
    /// of [`Fbm::sample2`] with the same seed.
    #[inline]
    #[must_use]
    pub fn domain_warp2(&self, seed: u64, p: DVec2, frequency: f64, amplitude: f64) -> DVec2 {
        let q = p * frequency;
        let displacement = DVec2::new(
            self.sample2(hash_combine(seed, WARP_CHANNEL_X), q),
            self.sample2(hash_combine(seed, WARP_CHANNEL_Y), q),
        );
        p + displacement * amplitude
    }

    /// Domain warp in 3D, see [`Fbm::domain_warp2`].
    #[inline]
    #[must_use]
    pub fn domain_warp3(&self, seed: u64, p: DVec3, frequency: f64, amplitude: f64) -> DVec3 {
        let q = p * frequency;
        let displacement = DVec3::new(
            self.sample3(hash_combine(seed, WARP_CHANNEL_X), q),
            self.sample3(hash_combine(seed, WARP_CHANNEL_Y), q),
            self.sample3(hash_combine(seed, WARP_CHANNEL_Z), q),
        );
        p + displacement * amplitude
    }

    /// Shapes and sums the octaves produced by `octave_noise(index)`.
    #[inline(always)]
    fn accumulate(&self, shape: Shape, mut octave_noise: impl FnMut(u32) -> f64) -> f64 {
        let mut sum = 0.0;
        let mut total_amplitude = 0.0;
        let mut amplitude = 1.0;
        // Ridged only: how much of the next octave is let through.
        let mut weight = 1.0;
        for octave in 0..self.octaves.min(MAX_OCTAVES) {
            let n = octave_noise(octave);
            let shaped = match shape {
                Shape::Plain => n,
                Shape::Billow => 2.0 * n.abs() - 1.0,
                Shape::Ridged => {
                    let ridge = 1.0 - n.abs();
                    let ridge = ridge * ridge;
                    let shaped = 2.0 * weight * ridge - 1.0;
                    // Valleys (small `ridge`) mute the finer octaves, ridges keep them.
                    weight = det::min(2.0 * ridge, 1.0);
                    shaped
                }
            };
            sum += amplitude * shaped;
            total_amplitude += amplitude.abs();
            amplitude *= self.gain;
        }
        if total_amplitude > 0.0 {
            det::clamp(sum / total_amplitude, -1.0, 1.0)
        } else {
            0.0
        }
    }

    #[inline(always)]
    fn sum2(&self, seed: u64, p: DVec2, shape: Shape) -> f64 {
        let mut q = p;
        self.accumulate(shape, |octave| {
            let octave_seed = octave_seed(seed, octave);
            q = rotate2(q);
            let n = expand2(value_gradient2_on(
                octave_seed,
                q + offset2(octave_seed),
                Unbounded,
            ));
            q *= self.lacunarity;
            n
        })
    }

    #[inline(always)]
    fn sum3(&self, seed: u64, p: DVec3, shape: Shape) -> f64 {
        let mut q = p;
        self.accumulate(shape, |octave| {
            let octave_seed = octave_seed(seed, octave);
            q = rotate3(q);
            let n = expand3(value_gradient3_on(
                octave_seed,
                q + offset3(octave_seed),
                Unbounded,
            ));
            q *= self.lacunarity;
            n
        })
    }

    #[inline(always)]
    fn sum2_tiled(&self, seed: u64, p: DVec2, period: u32, shape: Shape) -> f64 {
        let ratio = self.integer_lacunarity();
        let mut q = p;
        let mut octave_period = u64::from(period.max(1));
        self.accumulate(shape, |octave| {
            let octave_seed = octave_seed(seed, octave);
            let lattice = Tiled::new(octave_period);
            let n = expand2(value_gradient2_on(
                octave_seed,
                q + offset2(octave_seed),
                lattice,
            ));
            q *= ratio as f64;
            octave_period = octave_period.saturating_mul(ratio);
            n
        })
    }

    #[inline(always)]
    fn sum3_tiled(&self, seed: u64, p: DVec3, period: u32, shape: Shape) -> f64 {
        let ratio = self.integer_lacunarity();
        let mut q = p;
        let mut octave_period = u64::from(period.max(1));
        self.accumulate(shape, |octave| {
            let octave_seed = octave_seed(seed, octave);
            let lattice = Tiled::new(octave_period);
            let n = expand3(value_gradient3_on(
                octave_seed,
                q + offset3(octave_seed),
                lattice,
            ));
            q *= ratio as f64;
            octave_period = octave_period.saturating_mul(ratio);
            n
        })
    }

    /// The lacunarity as the integer frequency ratio used by the tileable variants.
    #[inline]
    fn integer_lacunarity(&self) -> u64 {
        // The cast saturates and maps NaN to 0; the clamp bounds it to a sane ratio.
        (self.lacunarity.round() as u64).clamp(1, 1 << 16)
    }
}

const WARP_CHANNEL_X: u64 = 0x5741_5250_0000_0001;
const WARP_CHANNEL_Y: u64 = 0x5741_5250_0000_0002;
const WARP_CHANNEL_Z: u64 = 0x5741_5250_0000_0003;

/// Seed of one octave. The noise functions scramble their seed again, so a simple odd
/// multiple is enough to separate the octaves.
#[inline(always)]
fn octave_seed(seed: u64, octave: u32) -> u64 {
    seed.wrapping_add((u64::from(octave) + 1).wrapping_mul(0xA24B_AED4_963E_E407))
}

/// Offset of an octave's lattice, each component in `[0, 1)`.
#[inline(always)]
fn offset2(octave_seed: u64) -> DVec2 {
    let hash = mix64(octave_seed ^ SALT_OCTAVE);
    const SCALE: f64 = 1.0 / 4_294_967_296.0;
    DVec2::new(
        (hash >> 32) as f64 * SCALE,
        (hash & 0xFFFF_FFFF) as f64 * SCALE,
    )
}

/// Offset of an octave's lattice, each component in `[0, 1)`.
#[inline(always)]
fn offset3(octave_seed: u64) -> DVec3 {
    let hash = mix64(octave_seed ^ SALT_OCTAVE);
    const SCALE: f64 = 1.0 / 2_097_152.0;
    DVec3::new(
        (hash >> 43) as f64 * SCALE,
        ((hash >> 22) & 0x1F_FFFF) as f64 * SCALE,
        ((hash >> 1) & 0x1F_FFFF) as f64 * SCALE,
    )
}

/// Contrast curve of a 2D octave: `x (15 - 10x^2 + 3x^4) / 8`, the antiderivative of
/// `(1 - x^2)^2` scaled to map `[-1, 1]` onto itself. Odd, strictly increasing, slope 15/8 at
/// zero, first and second derivative zero at the ends.
#[inline(always)]
fn expand2(n: f64) -> f64 {
    let n2 = n * n;
    n * (15.0 + n2 * (3.0 * n2 - 10.0)) * (1.0 / 8.0)
}

/// Contrast curve of a 3D octave: `x (315 - 420x^2 + 378x^4 - 180x^6 + 35x^8) / 128`, the
/// antiderivative of `(1 - x^2)^4` scaled to map `[-1, 1]` onto itself. Slope 315/128 at zero;
/// the 3D octave is weaker than the 2D one and needs the stronger curve to end up with the
/// same spread.
#[inline(always)]
fn expand3(n: f64) -> f64 {
    let n2 = n * n;
    n * (315.0 + n2 * (n2 * (378.0 + n2 * (35.0 * n2 - 180.0)) - 420.0)) * (1.0 / 128.0)
}

#[inline(always)]
fn rotate2(p: DVec2) -> DVec2 {
    let m = &ROTATION2;
    DVec2::new(m[0][0] * p.x + m[0][1] * p.y, m[1][0] * p.x + m[1][1] * p.y)
}

#[inline(always)]
fn rotate3(p: DVec3) -> DVec3 {
    let m = &ROTATION3;
    DVec3::new(
        m[0][0] * p.x + m[0][1] * p.y + m[0][2] * p.z,
        m[1][0] * p.x + m[1][1] * p.y + m[1][2] * p.z,
        m[2][0] * p.x + m[2][1] * p.y + m[2][2] * p.z,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn octave_rotations_are_proper_and_never_axis_aligned() {
        let m = ROTATION2;
        assert!((m[0][0] * m[1][1] - m[0][1] * m[1][0] - 1.0).abs() < 1.0e-15);
        assert!((m[0][0] * m[0][0] + m[1][0] * m[1][0] - 1.0).abs() < 1.0e-15);
        assert!((m[0][0] * m[0][1] + m[1][0] * m[1][1]).abs() < 1.0e-15);

        let m = ROTATION3;
        for i in 0..3 {
            for j in 0..3 {
                let dot: f64 = (0..3).map(|k| m[i][k] * m[j][k]).sum();
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!((dot - expected).abs() < 1.0e-15, "rows {i} and {j}");
            }
        }
        let determinant = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
            - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
            + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
        assert!((determinant - 1.0).abs() < 1.0e-15);

        // The angles are the documented ones.
        let quarter = core::f64::consts::FRAC_PI_2;
        let angle2 = crate::det::atan2(ROTATION2[1][0], ROTATION2[0][0]);
        assert!((angle2 - quarter * GOLDEN_FRACTION).abs() < 1.0e-15);
        let trace = m[0][0] + m[1][1] + m[2][2];
        let golden_angle = core::f64::consts::TAU * GOLDEN_FRACTION;
        assert!((crate::det::acos((trace - 1.0) / 2.0) - golden_angle).abs() < 1.0e-14);
        let axis = DVec3::new(1.0, 2.0, 3.0) / 14.0_f64.sqrt();
        assert!((rotate3(axis) - axis).length() < 1.0e-15);

        // Repeated rotation stays a rotation and never brings an axis back onto an axis: after
        // any number of octaves the images of the unit vectors keep all components away from
        // the values 0 and ±1 that an axis-aligned lattice would have. The octaves that carry
        // visible amplitude (the first eight) stay at least 5° away from the axes in 2D.
        let mut p2 = DVec2::X;
        let mut p3 = [DVec3::X, DVec3::Y, DVec3::Z];
        for octave in 1..=32 {
            p2 = rotate2(p2);
            assert!((p2.length() - 1.0).abs() < 1.0e-13);
            let off_axis = crate::det::min(p2.x.abs(), p2.y.abs());
            assert!(off_axis > 0.02, "2D octave {octave}: {p2:?}");
            assert!(octave > 8 || off_axis > 0.085, "2D octave {octave}: {p2:?}");
            for axis in &mut p3 {
                *axis = rotate3(*axis);
                assert!((axis.length() - 1.0).abs() < 1.0e-13);
                assert!(
                    axis.abs().max_element() < 0.9995,
                    "3D octave {octave}: {axis:?}"
                );
                // The first twelve octaves keep every lattice axis about 10° or more away from
                // every world axis.
                assert!(
                    octave > 12 || axis.abs().max_element() < 0.985,
                    "3D octave {octave}"
                );
            }
        }
    }

    #[test]
    fn contrast_curves_map_the_unit_interval_onto_itself() {
        type Curve = fn(f64) -> f64;
        let curves: [(Curve, f64, &str); 2] =
            [(expand2, 15.0 / 8.0, "2D"), (expand3, 315.0 / 128.0, "3D")];
        for (curve, slope, name) in curves {
            assert_eq!(curve(0.0), 0.0, "{name}");
            assert_eq!(curve(1.0), 1.0, "{name}");
            assert_eq!(curve(-1.0), -1.0, "{name}");
            // Odd and strictly increasing: a bijection of [-1, 1], so nothing is ever clipped.
            let mut previous = -1.0;
            for i in -999..=1000 {
                let x = f64::from(i) / 1000.0;
                let y = curve(x);
                assert!(y > previous && y <= 1.0, "{name} at {x}");
                assert_eq!(curve(-x), -y, "{name} at {x}");
                previous = y;
            }
            // The documented slope at zero, and a flat, C2 arrival at the ends.
            let h = 1.0e-4;
            assert!((curve(h) / h - slope).abs() < 1.0e-6, "{name}");
            assert!(1.0 - curve(1.0 - h) < 1.0e-11, "{name}");
            let curvature = (curve(1.0 - 2.0 * h) - 2.0 * curve(1.0 - h) + curve(1.0)) / (h * h);
            assert!(curvature.abs() < 1.0e-2, "{name}: {curvature}");
        }
        // They are the antiderivatives of (1 - x^2)^k: check the derivative in the interior.
        let h = 1.0e-5;
        for x in [-0.8, -0.3, 0.1, 0.5, 0.9] {
            let bell = 1.0 - x * x;
            let slope2 = (expand2(x + h) - expand2(x - h)) / (2.0 * h);
            let slope3 = (expand3(x + h) - expand3(x - h)) / (2.0 * h);
            assert!((slope2 - 15.0 / 8.0 * crate::det::powi(bell, 2)).abs() < 1.0e-8);
            assert!((slope3 - 315.0 / 128.0 * crate::det::powi(bell, 4)).abs() < 1.0e-8);
        }
    }

    #[test]
    fn octave_offsets_are_inside_the_unit_cell_and_distinct() {
        let mut seen = Vec::new();
        for octave in 0..32 {
            let seed = octave_seed(42, octave);
            let o2 = offset2(seed);
            let o3 = offset3(seed);
            assert!(o2.cmpge(DVec2::ZERO).all() && o2.cmplt(DVec2::ONE).all());
            assert!(o3.cmpge(DVec3::ZERO).all() && o3.cmplt(DVec3::ONE).all());
            assert!(!seen.contains(&seed));
            seen.push(seed);
        }
        assert_ne!(octave_seed(42, 0), 42);
        assert_ne!(offset2(octave_seed(1, 0)), offset2(octave_seed(2, 0)));
    }

    #[test]
    fn integer_lacunarity_is_sane() {
        let ratio = |lacunarity| Fbm::new(1, lacunarity, 0.5).integer_lacunarity();
        assert_eq!(ratio(2.0), 2);
        assert_eq!(ratio(2.4), 2);
        assert_eq!(ratio(2.5), 3);
        assert_eq!(ratio(0.2), 1);
        assert_eq!(ratio(-3.0), 1);
        assert_eq!(ratio(f64::NAN), 1);
        assert_eq!(ratio(f64::INFINITY), 1 << 16);
    }
}
