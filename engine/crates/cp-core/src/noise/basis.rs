//! The interpolated lattice noises: value, gradient and value-gradient noise.
//!
//! All three share one skeleton: hash the corners of the lattice cell around the sample, turn
//! each hash into a corner contribution, and blend the contributions with the quintic fade.
//! They differ only in the contribution:
//!
//! - value noise: a random value,
//! - gradient noise: a random gradient times the offset from the corner,
//! - value-gradient noise: both, i.e. a random first-order Taylor polynomial per corner.

use super::lattice::{
    GRADIENT2_SCALE, GRADIENT3_SCALE, GRADIENTS2, GRADIENTS3, Lattice, SALT_GRADIENT, SALT_VALUE,
    SALT_VALUE_GRADIENT, const_sqrt, fade, finish, fold, gradient_index, lerp, seed_state,
    signed_unit, split,
};
use crate::det;
use crate::math::{DVec2, DVec3};

/// Blends `corner(hash, dx, dy)` over the four corners of the cell containing `p`, where
/// `(dx, dy)` is the offset from the corner to `p`.
#[inline(always)]
fn blend2<L: Lattice>(
    state: u64,
    p: DVec2,
    lattice: L,
    corner: impl Fn(u64, f64, f64) -> f64,
) -> f64 {
    let (ix, fx) = split(p.x);
    let (iy, fy) = split(p.y);
    let (x0, x1) = (
        fold(state, lattice.wrap(ix)),
        fold(state, lattice.wrap(ix.wrapping_add(1))),
    );
    let (y0, y1) = (lattice.wrap(iy), lattice.wrap(iy.wrapping_add(1)));
    let at = |x: u64, y: i64, dx: f64, dy: f64| corner(finish(fold(x, y)), dx, dy);
    let (gx, gy) = (fx - 1.0, fy - 1.0);

    let u = fade(fx);
    let low = lerp(at(x0, y0, fx, fy), at(x1, y0, gx, fy), u);
    let high = lerp(at(x0, y1, fx, gy), at(x1, y1, gx, gy), u);
    lerp(low, high, fade(fy))
}

/// Blends `corner(hash, dx, dy, dz)` over the eight corners of the cell containing `p`.
#[inline(always)]
fn blend3<L: Lattice>(
    state: u64,
    p: DVec3,
    lattice: L,
    corner: impl Fn(u64, f64, f64, f64) -> f64,
) -> f64 {
    let (ix, fx) = split(p.x);
    let (iy, fy) = split(p.y);
    let (iz, fz) = split(p.z);
    let (x0, x1) = (
        fold(state, lattice.wrap(ix)),
        fold(state, lattice.wrap(ix.wrapping_add(1))),
    );
    let (y0, y1) = (lattice.wrap(iy), lattice.wrap(iy.wrapping_add(1)));
    let (z0, z1) = (lattice.wrap(iz), lattice.wrap(iz.wrapping_add(1)));
    let (x0y0, x1y0, x0y1, x1y1) = (fold(x0, y0), fold(x1, y0), fold(x0, y1), fold(x1, y1));
    let at = |xy: u64, z: i64, dx: f64, dy: f64, dz: f64| corner(finish(fold(xy, z)), dx, dy, dz);
    let (gx, gy, gz) = (fx - 1.0, fy - 1.0, fz - 1.0);

    let u = fade(fx);
    let v = fade(fy);
    let near = lerp(
        lerp(at(x0y0, z0, fx, fy, fz), at(x1y0, z0, gx, fy, fz), u),
        lerp(at(x0y1, z0, fx, gy, fz), at(x1y1, z0, gx, gy, fz), u),
        v,
    );
    let far = lerp(
        lerp(at(x0y0, z1, fx, fy, gz), at(x1y0, z1, gx, fy, gz), u),
        lerp(at(x0y1, z1, fx, gy, gz), at(x1y1, z1, gx, gy, gz), u),
        v,
    );
    lerp(near, far, fade(fz))
}

/// Dot product of the corner's gradient with the offset from the corner.
#[inline(always)]
fn slope2(hash: u64, dx: f64, dy: f64) -> f64 {
    let gradient = &GRADIENTS2[gradient_index(hash)];
    gradient[0] * dx + gradient[1] * dy
}

#[inline(always)]
fn slope3(hash: u64, dx: f64, dy: f64, dz: f64) -> f64 {
    let gradient = &GRADIENTS3[gradient_index(hash)];
    gradient[0] * dx + gradient[1] * dy + gradient[2] * dz
}

/// The corner's random value in `[-1, 1)`, from hash bits that [`gradient_index`] does not
/// use, so that value and gradient of a corner are independent.
#[inline(always)]
fn height(hash: u64) -> f64 {
    signed_unit(hash << 8)
}

#[inline(always)]
pub(super) fn value2_on<L: Lattice>(seed: u64, p: DVec2, lattice: L) -> f64 {
    let state = seed_state(seed, SALT_VALUE);
    let sum = blend2(state, p, lattice, |hash, _, _| signed_unit(hash));
    det::clamp(sum, -1.0, 1.0)
}

#[inline(always)]
pub(super) fn value3_on<L: Lattice>(seed: u64, p: DVec3, lattice: L) -> f64 {
    let state = seed_state(seed, SALT_VALUE);
    let sum = blend3(state, p, lattice, |hash, _, _, _| signed_unit(hash));
    det::clamp(sum, -1.0, 1.0)
}

#[inline(always)]
pub(super) fn gradient2_on<L: Lattice>(seed: u64, p: DVec2, lattice: L) -> f64 {
    let state = seed_state(seed, SALT_GRADIENT);
    det::clamp(
        blend2(state, p, lattice, slope2) * GRADIENT2_SCALE,
        -1.0,
        1.0,
    )
}

#[inline(always)]
pub(super) fn gradient3_on<L: Lattice>(seed: u64, p: DVec3, lattice: L) -> f64 {
    let state = seed_state(seed, SALT_GRADIENT);
    det::clamp(
        blend3(state, p, lattice, slope3) * GRADIENT3_SCALE,
        -1.0,
        1.0,
    )
}

// Value-gradient noise mixes a value part (weight `b`) and a gradient part (weight `a`,
// `a + b = 1`, so the sum stays in [-1, 1]).
//
// Gradient noise alone is zero at every lattice point, which threads its zero set, and with
// it every ridge, crease and coastline derived from it, through a regular grid. Value noise
// alone has zero slope at every lattice point and looks blocky. Giving each corner a value
// *and* a gradient removes both artefacts.
//
// The weights are chosen so that the variance of the noise is the same at the lattice points
// (where only the value part contributes: b²/3) and at the cell centres (a²·2⁻ⁿ/n from the
// gradients plus b²·2⁻ⁿ/3 from the values, in n dimensions).
// That gives (b/a)² = 3 / (n (2ⁿ - 1)): 1/2 in 2D and 1/7 in 3D. Those two places are the
// minima; the standard deviation elsewhere in the cell is at most about 12 % (2D) or 30 % (3D)
// higher, instead of swinging between zero and its maximum as it does for gradient noise.

/// `b` in 2D: `sqrt(2) - 1`.
pub(super) const VALUE_WEIGHT2: f64 = core::f64::consts::SQRT_2 - 1.0;
/// `a` in 2D (`2 - sqrt(2)`) times the gradient scale.
const GRADIENT_WEIGHT2: f64 = (1.0 - VALUE_WEIGHT2) * GRADIENT2_SCALE;
/// `b` in 3D: `1 / (sqrt(7) + 1)`.
pub(super) const VALUE_WEIGHT3: f64 = 1.0 / (const_sqrt(7.0) + 1.0);
/// `a` in 3D (`sqrt(7) / (sqrt(7) + 1)`) times the gradient scale.
const GRADIENT_WEIGHT3: f64 = (1.0 - VALUE_WEIGHT3) * GRADIENT3_SCALE;

/// 2D value-gradient noise in `[-1, 1]`, the basis of [`super::Fbm`].
#[inline(always)]
pub(super) fn value_gradient2_on<L: Lattice>(seed: u64, p: DVec2, lattice: L) -> f64 {
    let state = seed_state(seed, SALT_VALUE_GRADIENT);
    let sum = blend2(state, p, lattice, |hash, dx, dy| {
        VALUE_WEIGHT2 * height(hash) + GRADIENT_WEIGHT2 * slope2(hash, dx, dy)
    });
    det::clamp(sum, -1.0, 1.0)
}

/// 3D value-gradient noise in `[-1, 1]`, the basis of [`super::Fbm`].
#[inline(always)]
pub(super) fn value_gradient3_on<L: Lattice>(seed: u64, p: DVec3, lattice: L) -> f64 {
    let state = seed_state(seed, SALT_VALUE_GRADIENT);
    let sum = blend3(state, p, lattice, |hash, dx, dy, dz| {
        VALUE_WEIGHT3 * height(hash) + GRADIENT_WEIGHT3 * slope3(hash, dx, dy, dz)
    });
    det::clamp(sum, -1.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::super::lattice::Unbounded;
    use super::*;
    use crate::hash::Rng;

    fn random3(rng: &mut Rng, extent: f64) -> DVec3 {
        DVec3::new(
            rng.range_f64(-extent, extent),
            rng.range_f64(-extent, extent),
            rng.range_f64(-extent, extent),
        )
    }

    #[test]
    fn weights_sum_to_one_and_follow_the_variance_rule() {
        let (b2, a2) = (VALUE_WEIGHT2, 1.0 - VALUE_WEIGHT2);
        let (b3, a3) = (VALUE_WEIGHT3, 1.0 - VALUE_WEIGHT3);
        assert!((b2 * b2 / (a2 * a2) - 0.5).abs() < 1e-15);
        assert!((b3 * b3 / (a3 * a3) - 1.0 / 7.0).abs() < 1e-15);
        assert!((GRADIENT_WEIGHT2 - a2 * core::f64::consts::SQRT_2).abs() < 1e-15);
        assert!((GRADIENT_WEIGHT3 - a3 * 2.0 / 3.0_f64.sqrt()).abs() < 1e-15);
    }

    #[test]
    fn value_gradient_noise_is_not_pinned_to_the_lattice() {
        // The design goal of the weights, measured: the spread of the noise at lattice points
        // equals its spread at cell centres, and both are close to the overall spread.
        let mut rng = Rng::new(300, 1);
        let n = 400_000;
        // [lattice points, cell centres, anywhere] for 2D and 3D.
        let (mut squares2, mut squares3) = ([0.0_f64; 3], [0.0_f64; 3]);
        let (mut sums2, mut sums3) = ([0.0_f64; 3], [0.0_f64; 3]);
        let mut clamped = 0;
        for i in 0..n {
            let seed = u64::from(i % 5_u32);
            let anywhere = random3(&mut rng, 3000.0);
            let node = anywhere.floor();
            for (slot, p) in [node, node + 0.5, anywhere].into_iter().enumerate() {
                let v3 = value_gradient3_on(seed, p, Unbounded);
                let v2 = value_gradient2_on(seed, p.truncate(), Unbounded);
                assert!((-1.0..=1.0).contains(&v2) && (-1.0..=1.0).contains(&v3));
                clamped += u32::from(v2.abs() == 1.0 || v3.abs() == 1.0);
                squares2[slot] += v2 * v2;
                squares3[slot] += v3 * v3;
                sums2[slot] += v2;
                sums3[slot] += v3;
            }
        }
        assert_eq!(clamped, 0);
        let n = f64::from(n);
        let deviation = |squares: f64, sum: f64| (squares / n - (sum / n) * (sum / n)).sqrt();
        let d2: Vec<f64> = (0..3).map(|i| deviation(squares2[i], sums2[i])).collect();
        let d3: Vec<f64> = (0..3).map(|i| deviation(squares3[i], sums3[i])).collect();
        let report = format!("2D {d2:.4?}, 3D {d3:.4?}");
        for sums in [sums2, sums3] {
            assert!(sums.iter().all(|sum| (sum / n).abs() < 0.003), "{report}");
        }
        // At lattice points only the values contribute: b / sqrt(3) exactly.
        assert!(
            (d2[0] / (VALUE_WEIGHT2 / 3.0_f64.sqrt()) - 1.0).abs() < 0.01,
            "{report}"
        );
        assert!(
            (d3[0] / (VALUE_WEIGHT3 / 3.0_f64.sqrt()) - 1.0).abs() < 0.01,
            "{report}"
        );
        // Nodes and centres match; the average over the cell is somewhat higher (edge and face
        // midpoints carry more), but nowhere near the zero-to-maximum swing of gradient noise.
        assert!((d2[1] / d2[0] - 1.0).abs() < 0.015, "{report}");
        assert!((d3[1] / d3[0] - 1.0).abs() < 0.015, "{report}");
        assert!(d2[2] / d2[0] > 1.0 && d2[2] / d2[0] < 1.12, "{report}");
        assert!(d3[2] / d3[0] > 1.0 && d3[2] / d3[0] < 1.26, "{report}");
        assert!(
            (d2[2] - DEVIATION2).abs() < 0.003 && (d3[2] - DEVIATION3).abs() < 0.003,
            "{report}"
        );
    }

    // Standard deviations of single octaves, as quoted in the documentation of `Fbm`.
    const DEVIATION2: f64 = 0.26;
    const DEVIATION3: f64 = 0.194;

    #[test]
    fn value_and_gradient_of_a_corner_are_independent() {
        // The two parts read disjoint bits of the corner hash; their correlation is nil.
        let mut rng = Rng::new(301, 1);
        let (mut hs, mut hh, mut ss) = (0.0, 0.0, 0.0);
        for _ in 0..200_000 {
            let hash = rng.next_u64();
            let h = height(hash);
            let s = slope3(hash, 0.3, -0.4, 0.5);
            hs += h * s;
            hh += h * h;
            ss += s * s;
            assert!((-1.0..1.0).contains(&h));
        }
        let correlation = hs / (hh * ss).sqrt();
        assert!(correlation.abs() < 0.01, "{correlation}");
        assert_eq!(height(0xFF00_0000_0000_0000), -1.0);
        assert_eq!(gradient_index(0x00FF_FFFF_FFFF_FFFF), 0);
    }
}
