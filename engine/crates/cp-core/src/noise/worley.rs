//! Worley (cellular) noise: distances to the nearest feature points.
//!
//! Every lattice cell holds exactly one feature point at a hashed position inside the cell.
//! The functions return the Euclidean distances `(f1, f2)` to the nearest and second-nearest
//! feature point, in lattice units.
//!
//! The result is **exact**. A fixed 3×3(×3) neighbourhood, as commonly used, misses closer
//! points in rare configurations, which shows as hairline discontinuities, most visibly in
//! `f2 - f1`. Here the neighbourhood is searched first and the surrounding ring of the 5×5(×5)
//! block only when the second distance found so far is large enough for the ring to matter.
//! That block always suffices: with one point per cell, `f2` never exceeds `sqrt(2.5)` in 2D and
//! `sqrt(3)` in 3D, while everything outside the block is at least 2 away.
//!
//! Cells are visited nearest first, and a cell, row or column of cells is skipped when even
//! its closest corner is farther away than the second distance found so far. Skipping never
//! changes a result bit (the tests compare against an exhaustive search); it only saves work.

use super::lattice::{Lattice, SALT_WORLEY, Tiled, Unbounded, finish, fold, seed_state, split};
use crate::det;
use crate::math::{DVec2, DVec3};

/// Cells whose nearest possible point is farther away than the second distance found so far
/// cannot matter. The factor keeps that comparison safe against rounding in the distance
/// computations, so that skipping never changes a result bit.
const SAFETY: f64 = 1.0 - 1.0e-9;

/// Order in which the cells of the 3x3(x3) block are visited along each axis: own cell first.
/// Early small distances let more of the remaining cells be skipped.
const NEAR_FIRST: [i64; 3] = [0, -1, 1];

/// 2D Worley noise: `(f1, f2)`, the distances to the nearest and second-nearest feature point.
///
/// `0 <= f1 <= f2`, `f1 <= sqrt(2)`, `f2 <= sqrt(2.5)`; the averages are `f1 ≈ 0.43` and
/// `f2 ≈ 0.70`. Both are continuous (1-Lipschitz). `f1` gives cells, `f2 - f1` their borders.
#[inline]
#[must_use]
pub fn worley2(seed: u64, p: DVec2) -> (f64, f64) {
    worley2_on(seed, p, Unbounded)
}

/// [`worley2`] that repeats every `period` lattice units on both axes (`period >= 1`, zero is
/// treated as 1). The repeated images of a feature point count as separate points.
#[inline]
#[must_use]
pub fn worley2_tiled(seed: u64, p: DVec2, period: u32) -> (f64, f64) {
    worley2_on(seed, p, Tiled::new(u64::from(period)))
}

/// 3D Worley noise: `(f1, f2)`, the distances to the nearest and second-nearest feature point.
///
/// `0 <= f1 <= f2 <= sqrt(3)`; the averages are `f1 ≈ 0.52` and `f2 ≈ 0.72`. Both are
/// continuous (1-Lipschitz).
#[inline]
#[must_use]
pub fn worley3(seed: u64, p: DVec3) -> (f64, f64) {
    worley3_on(seed, p, Unbounded)
}

/// [`worley3`] that repeats every `period` lattice units on all axes (`period >= 1`, zero is
/// treated as 1). The repeated images of a feature point count as separate points.
#[inline]
#[must_use]
pub fn worley3_tiled(seed: u64, p: DVec3, period: u32) -> (f64, f64) {
    worley3_on(seed, p, Tiled::new(u64::from(period)))
}

/// The two smallest squared distances seen so far.
#[derive(Clone, Copy)]
struct Nearest {
    first: f64,
    second: f64,
}

impl Nearest {
    const NONE: Self = Self {
        first: f64::INFINITY,
        second: f64::INFINITY,
    };

    #[inline(always)]
    fn add(&mut self, squared_distance: f64) {
        if squared_distance < self.first {
            self.second = self.first;
            self.first = squared_distance;
        } else if squared_distance < self.second {
            self.second = squared_distance;
        }
    }

    /// Whether points at least `reach` away could still change the result.
    #[inline(always)]
    fn may_improve_beyond(&self, reach: f64) -> bool {
        self.second >= reach * reach * SAFETY
    }

    /// True when no point at a squared distance of `squared_gap` or more can change the
    /// result.
    #[inline(always)]
    fn excludes(&self, squared_gap: f64) -> bool {
        squared_gap * SAFETY >= self.second
    }

    #[inline(always)]
    fn distances(&self) -> (f64, f64) {
        (self.first.sqrt(), self.second.sqrt())
    }
}

/// Distance from a fraction in `[0, 1]` to the nearer end of its cell.
#[inline(always)]
fn margin(fraction: f64) -> f64 {
    det::min(fraction, 1.0 - fraction)
}

/// Squared distance, along one axis, from a sample at `fraction` inside its cell to the
/// nearest edge of the cell `offset` cells away (zero for the sample's own cell). Every point
/// of that cell is at least this far away along the axis.
#[inline(always)]
fn squared_gap(offset: i64, fraction: f64) -> f64 {
    let gap = if offset > 0 {
        offset as f64 - fraction
    } else if offset < 0 {
        fraction - (offset + 1) as f64
    } else {
        0.0
    };
    gap * gap
}

const TWO_POW_MINUS_32: f64 = 1.0 / 4_294_967_296.0;
const TWO_POW_MINUS_21: f64 = 1.0 / 2_097_152.0;

/// Offset of a cell's feature point from the cell's minimum corner, each component in
/// `[0, 1)`, from 32 hash bits per axis.
#[inline(always)]
fn feature2(hash: u64) -> (f64, f64) {
    (
        (hash >> 32) as f64 * TWO_POW_MINUS_32,
        (hash & 0xFFFF_FFFF) as f64 * TWO_POW_MINUS_32,
    )
}

/// Like [`feature2`] with 21 hash bits per axis.
#[inline(always)]
fn feature3(hash: u64) -> (f64, f64, f64) {
    (
        (hash >> 43) as f64 * TWO_POW_MINUS_21,
        ((hash >> 22) & 0x1F_FFFF) as f64 * TWO_POW_MINUS_21,
        ((hash >> 1) & 0x1F_FFFF) as f64 * TWO_POW_MINUS_21,
    )
}

#[inline(always)]
fn worley2_on<L: Lattice>(seed: u64, p: DVec2, lattice: L) -> (f64, f64) {
    let state = seed_state(seed, SALT_WORLEY);
    let (cx, fx) = split(p.x);
    let (cy, fy) = split(p.y);
    let mut nearest = Nearest::NONE;
    // Squared distance from the sample to the feature point of the cell at offset (dx, dy);
    // `column` is the hash state after the cell's x coordinate.
    let cell = |column: u64, dx: i64, dy: i64| {
        let (jx, jy) = feature2(finish(fold(column, lattice.wrap(cy.wrapping_add(dy)))));
        let (ox, oy) = ((dx as f64 + jx) - fx, (dy as f64 + jy) - fy);
        ox * ox + oy * oy
    };

    for dx in NEAR_FIRST {
        let gap_x = squared_gap(dx, fx);
        if nearest.excludes(gap_x) {
            continue;
        }
        let column = fold(state, lattice.wrap(cx.wrapping_add(dx)));
        for dy in NEAR_FIRST {
            if !nearest.excludes(gap_x + squared_gap(dy, fy)) {
                nearest.add(cell(column, dx, dy));
            }
        }
    }
    // Everything outside the 3x3 block is at least one cell plus the margin away.
    if nearest.may_improve_beyond(1.0 + det::min(margin(fx), margin(fy))) {
        for dx in -2..=2_i64 {
            let gap_x = squared_gap(dx, fx);
            if nearest.excludes(gap_x) {
                continue;
            }
            let column = fold(state, lattice.wrap(cx.wrapping_add(dx)));
            for dy in -2..=2_i64 {
                let in_ring = dx.abs() == 2 || dy.abs() == 2;
                if in_ring && !nearest.excludes(gap_x + squared_gap(dy, fy)) {
                    nearest.add(cell(column, dx, dy));
                }
            }
        }
    }
    nearest.distances()
}

#[inline(always)]
fn worley3_on<L: Lattice>(seed: u64, p: DVec3, lattice: L) -> (f64, f64) {
    let state = seed_state(seed, SALT_WORLEY);
    let (cx, fx) = split(p.x);
    let (cy, fy) = split(p.y);
    let (cz, fz) = split(p.z);
    let mut nearest = Nearest::NONE;
    // Squared distance from the sample to the feature point of the cell at offset
    // (dx, dy, dz); `row` is the hash state after the cell's x and y coordinates.
    let cell = |row: u64, dx: i64, dy: i64, dz: i64| {
        let (jx, jy, jz) = feature3(finish(fold(row, lattice.wrap(cz.wrapping_add(dz)))));
        let (ox, oy, oz) = (
            (dx as f64 + jx) - fx,
            (dy as f64 + jy) - fy,
            (dz as f64 + jz) - fz,
        );
        ox * ox + oy * oy + oz * oz
    };

    for dx in NEAR_FIRST {
        let gap_x = squared_gap(dx, fx);
        if nearest.excludes(gap_x) {
            continue;
        }
        let column = fold(state, lattice.wrap(cx.wrapping_add(dx)));
        for dy in NEAR_FIRST {
            let gap_xy = gap_x + squared_gap(dy, fy);
            if nearest.excludes(gap_xy) {
                continue;
            }
            let row = fold(column, lattice.wrap(cy.wrapping_add(dy)));
            for dz in NEAR_FIRST {
                if !nearest.excludes(gap_xy + squared_gap(dz, fz)) {
                    nearest.add(cell(row, dx, dy, dz));
                }
            }
        }
    }
    // Everything outside the 3x3x3 block is at least one cell plus the margin away.
    let nearest_margin = det::min(det::min(margin(fx), margin(fy)), margin(fz));
    if nearest.may_improve_beyond(1.0 + nearest_margin) {
        for dx in -2..=2_i64 {
            let gap_x = squared_gap(dx, fx);
            if nearest.excludes(gap_x) {
                continue;
            }
            let column = fold(state, lattice.wrap(cx.wrapping_add(dx)));
            for dy in -2..=2_i64 {
                let gap_xy = gap_x + squared_gap(dy, fy);
                if nearest.excludes(gap_xy) {
                    continue;
                }
                let row = fold(column, lattice.wrap(cy.wrapping_add(dy)));
                for dz in -2..=2_i64 {
                    let in_ring = dx.abs() == 2 || dy.abs() == 2 || dz.abs() == 2;
                    if in_ring && !nearest.excludes(gap_xy + squared_gap(dz, fz)) {
                        nearest.add(cell(row, dx, dy, dz));
                    }
                }
            }
        }
    }
    nearest.distances()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hash::Rng;

    /// Reference implementation: every cell within `radius` cells of the sample, no shortcuts,
    /// then sort. It shares only the definition of a cell's feature point with the fast path.
    fn reference2<L: Lattice>(seed: u64, p: DVec2, lattice: L, radius: i64) -> (f64, f64) {
        let state = seed_state(seed, SALT_WORLEY);
        let (cx, fx) = split(p.x);
        let (cy, fy) = split(p.y);
        let mut squared = Vec::new();
        for dx in -radius..=radius {
            for dy in -radius..=radius {
                let column = fold(state, lattice.wrap(cx + dx));
                let (jx, jy) = feature2(finish(fold(column, lattice.wrap(cy + dy))));
                let (ox, oy) = ((dx as f64 + jx) - fx, (dy as f64 + jy) - fy);
                squared.push(ox * ox + oy * oy);
            }
        }
        squared.sort_by(f64::total_cmp);
        (squared[0].sqrt(), squared[1].sqrt())
    }

    fn reference3<L: Lattice>(seed: u64, p: DVec3, lattice: L, radius: i64) -> (f64, f64) {
        let state = seed_state(seed, SALT_WORLEY);
        let (cx, fx) = split(p.x);
        let (cy, fy) = split(p.y);
        let (cz, fz) = split(p.z);
        let mut squared = Vec::new();
        for dx in -radius..=radius {
            for dy in -radius..=radius {
                for dz in -radius..=radius {
                    let column = fold(state, lattice.wrap(cx + dx));
                    let row = fold(column, lattice.wrap(cy + dy));
                    let (jx, jy, jz) = feature3(finish(fold(row, lattice.wrap(cz + dz))));
                    let (ox, oy, oz) = (
                        (dx as f64 + jx) - fx,
                        (dy as f64 + jy) - fy,
                        (dz as f64 + jz) - fz,
                    );
                    squared.push(ox * ox + oy * oy + oz * oz);
                }
            }
        }
        squared.sort_by(f64::total_cmp);
        (squared[0].sqrt(), squared[1].sqrt())
    }

    fn random3(rng: &mut Rng, extent: f64) -> DVec3 {
        DVec3::new(
            rng.range_f64(-extent, extent),
            rng.range_f64(-extent, extent),
            rng.range_f64(-extent, extent),
        )
    }

    #[test]
    fn feature_points_lie_inside_their_cell() {
        let mut rng = Rng::new(200, 1);
        for _ in 0..100_000 {
            let hash = rng.next_u64();
            let (x, y) = feature2(hash);
            assert!((0.0..1.0).contains(&x) && (0.0..1.0).contains(&y));
            let (x, y, z) = feature3(hash);
            assert!((0.0..1.0).contains(&x) && (0.0..1.0).contains(&y) && (0.0..1.0).contains(&z));
        }
        assert_eq!(feature2(0), (0.0, 0.0));
        assert_eq!(feature3(0), (0.0, 0.0, 0.0));
        let top = 1.0 - TWO_POW_MINUS_32;
        assert_eq!(feature2(u64::MAX), (top, top));
        let top = 1.0 - TWO_POW_MINUS_21;
        assert_eq!(feature3(u64::MAX), (top, top, top));
        // The three 21-bit fields do not overlap.
        assert_eq!(feature3(1 << 43), (TWO_POW_MINUS_21, 0.0, 0.0));
        assert_eq!(feature3(1 << 22), (0.0, TWO_POW_MINUS_21, 0.0));
        assert_eq!(feature3(1 << 1), (0.0, 0.0, TWO_POW_MINUS_21));
    }

    /// Sample points for the comparison with the exhaustive search. Two thirds are uniformly
    /// random (near the origin and far out); one third sits close to lattice corners, where the
    /// sample is as far as it can be from the middle of the searched block and the distances
    /// are largest: the cases in which a fixed 3x3(x3) search goes wrong.
    fn search_case(rng: &mut Rng, i: u32) -> (DVec3, bool) {
        match i % 3 {
            0 => (random3(rng, 30.0), true),
            1 => (random3(rng, 1.0e6), true),
            _ => (random3(rng, 1000.0).round() + random3(rng, 0.08), false),
        }
    }

    /// Running sums for the mean distances of the uniformly random samples.
    #[derive(Default)]
    struct Means {
        n: f64,
        f1: f64,
        f2: f64,
    }

    impl Means {
        fn add(&mut self, f1: f64, f2: f64) {
            self.n += 1.0;
            self.f1 += f1;
            self.f2 += f2;
        }

        fn check(&self, expected_f1: f64, expected_f2: f64, context: &str) {
            let (f1, f2) = (self.f1 / self.n, self.f2 / self.n);
            let ok = (f1 - expected_f1).abs() < 0.005 && (f2 - expected_f2).abs() < 0.005;
            assert!(ok, "mean f1 {f1:.4}, mean f2 {f2:.4} ({context})");
        }
    }

    #[test]
    fn worley2_equals_the_exhaustive_search() {
        let mut rng = Rng::new(201, 1);
        let mut naive_errors = 0;
        let (mut largest_f1, mut largest_f2) = (0.0_f64, 0.0_f64);
        let mut means = Means::default();
        for i in 0..450_000_u32 {
            let seed = u64::from(i % 11);
            let (p, uniform) = search_case(&mut rng, i);
            let p = p.truncate();
            let (f1, f2) = worley2(seed, p);
            // Bit-exact agreement with a 9x9 search.
            assert_eq!((f1, f2), reference2(seed, p, Unbounded, 4), "at {p:?}");
            // The plain 3x3 search is not always right; count how often it would have erred.
            naive_errors += u32::from((f1, f2) != reference2(seed, p, Unbounded, 1));
            assert!(f1 >= 0.0 && f2 >= f1);
            largest_f1 = det::max(largest_f1, f1);
            largest_f2 = det::max(largest_f2, f2);
            if uniform {
                means.add(f1, f2);
            }

            let period = 1 + i % 6;
            let tiled = Tiled::new(u64::from(period));
            assert_eq!(
                worley2_tiled(seed, p, period),
                reference2(seed, p, tiled, 4)
            );
        }
        let context =
            format!("largest {largest_f1:.4} / {largest_f2:.4}, 3x3 errors {naive_errors}");
        // Documented bounds, with real configurations coming close to them.
        assert!(
            largest_f1 <= 2.0_f64.sqrt() && largest_f2 <= 2.5_f64.sqrt(),
            "{context}"
        );
        assert!(largest_f1 > 1.1 && largest_f2 > 1.3, "{context}");
        // The second search ring is needed for real inputs.
        assert!(
            naive_errors > 20,
            "the ring path is hardly exercised ({context})"
        );
        means.check(MEAN_F1_2D, MEAN_F2_2D, &context);
    }

    #[test]
    fn worley3_equals_the_exhaustive_search() {
        let mut rng = Rng::new(202, 1);
        let mut naive_errors = 0;
        let (mut largest_f1, mut largest_f2) = (0.0_f64, 0.0_f64);
        let mut means = Means::default();
        for i in 0..450_000_u32 {
            let seed = u64::from(i % 11);
            let (p, uniform) = search_case(&mut rng, i);
            let (f1, f2) = worley3(seed, p);
            // Bit-exact agreement with a 7x7x7 search.
            assert_eq!((f1, f2), reference3(seed, p, Unbounded, 3), "at {p:?}");
            naive_errors += u32::from((f1, f2) != reference3(seed, p, Unbounded, 1));
            assert!(f1 >= 0.0 && f2 >= f1);
            largest_f1 = det::max(largest_f1, f1);
            largest_f2 = det::max(largest_f2, f2);
            if uniform {
                means.add(f1, f2);
            }

            if i % 4 == 0 {
                let period = 1 + (i / 4) % 6;
                let tiled = Tiled::new(u64::from(period));
                assert_eq!(
                    worley3_tiled(seed, p, period),
                    reference3(seed, p, tiled, 3)
                );
            }
        }
        let context =
            format!("largest {largest_f1:.4} / {largest_f2:.4}, 3x3x3 errors {naive_errors}");
        assert!(
            largest_f1 <= 3.0_f64.sqrt() && largest_f2 <= 3.0_f64.sqrt(),
            "{context}"
        );
        assert!(largest_f1 > 1.1 && largest_f2 > 1.2, "{context}");
        // In 3D the fixed search errs rarely, but it does.
        assert!(
            naive_errors > 0,
            "the ring path was never needed ({context})"
        );
        means.check(MEAN_F1_3D, MEAN_F2_3D, &context);
    }

    // Typical values quoted in the documentation.
    const MEAN_F1_2D: f64 = 0.429;
    const MEAN_F2_2D: f64 = 0.699;
    const MEAN_F1_3D: f64 = 0.52;
    const MEAN_F2_3D: f64 = 0.717;

    #[test]
    fn distances_are_one_lipschitz() {
        // The distance to the k-th nearest point of a fixed set can change by at most the
        // distance moved. A wrong search range shows up as a jump that violates this.
        let mut rng = Rng::new(203, 1);
        for i in 0..200_000 {
            let step = [1.0e-6, 1.0e-3, 0.05, 0.7][i % 4];
            let p = random3(&mut rng, 40.0);
            let q = p + random3(&mut rng, step);
            let moved = (q - p).length() + 1.0e-12;
            let (a1, a2) = worley3(7, p);
            let (b1, b2) = worley3(7, q);
            assert!(
                (a1 - b1).abs() <= moved && (a2 - b2).abs() <= moved,
                "3D {p:?} -> {q:?}"
            );
            let (a1, a2) = worley3_tiled(7, p, 3);
            let (b1, b2) = worley3_tiled(7, q, 3);
            assert!(
                (a1 - b1).abs() <= moved && (a2 - b2).abs() <= moved,
                "3D tiled {p:?}"
            );

            let (p, q) = (p.truncate(), q.truncate());
            let moved = (q - p).length() + 1.0e-12;
            let (a1, a2) = worley2(7, p);
            let (b1, b2) = worley2(7, q);
            assert!(
                (a1 - b1).abs() <= moved && (a2 - b2).abs() <= moved,
                "2D {p:?} -> {q:?}"
            );
            let (a1, a2) = worley2_tiled(7, p, 2);
            let (b1, b2) = worley2_tiled(7, q, 2);
            assert!(
                (a1 - b1).abs() <= moved && (a2 - b2).abs() <= moved,
                "2D tiled {p:?}"
            );
        }
    }

    #[test]
    fn f1_is_the_distance_to_a_point() {
        // Using the public function only: walking from any sample against the gradient of f1
        // by exactly f1 arrives at a feature point, where f1 is zero. This holds for a true
        // Euclidean distance field and for nothing else.
        let mut rng = Rng::new(204, 1);
        let h = 1.0e-6;
        let mut checked = 0;
        for _ in 0..20_000 {
            let p = random3(&mut rng, 50.0);
            let (f1, f2) = worley3(3, p);
            // Close to the border between two cells of the diagram the gradient flips.
            if f2 - f1 > 0.02 && f1 > 0.02 {
                let gradient = DVec3::new(
                    worley3(3, p + DVec3::X * h).0 - worley3(3, p - DVec3::X * h).0,
                    worley3(3, p + DVec3::Y * h).0 - worley3(3, p - DVec3::Y * h).0,
                    worley3(3, p + DVec3::Z * h).0 - worley3(3, p - DVec3::Z * h).0,
                ) / (2.0 * h);
                assert!(
                    (gradient.length() - 1.0).abs() < 1.0e-6,
                    "{}",
                    gradient.length()
                );
                let feature = p - gradient * f1;
                assert!(
                    worley3(3, feature).0 < 1.0e-5,
                    "no feature point at {feature:?}"
                );
                checked += 1;
            }

            let p = p.truncate();
            let (f1, f2) = worley2(3, p);
            if f2 - f1 > 0.02 && f1 > 0.02 {
                let gradient = DVec2::new(
                    worley2(3, p + DVec2::X * h).0 - worley2(3, p - DVec2::X * h).0,
                    worley2(3, p + DVec2::Y * h).0 - worley2(3, p - DVec2::Y * h).0,
                ) / (2.0 * h);
                assert!(
                    (gradient.length() - 1.0).abs() < 1.0e-6,
                    "{}",
                    gradient.length()
                );
                let feature = p - gradient * f1;
                assert!(
                    worley2(3, feature).0 < 1.0e-5,
                    "no feature point at {feature:?}"
                );
                checked += 1;
            }
        }
        assert!(checked > 25_000, "{checked}");
    }

    #[test]
    fn f1_vanishes_exactly_at_each_cell_feature_point() {
        // Each cell has one feature point at a hashed position inside it. Standing on it, the
        // nearest distance is exactly zero and the second distance is not.
        for seed in [0_u64, 5, u64::MAX] {
            let state = seed_state(seed, SALT_WORLEY);
            for cx in -6..6_i64 {
                for cy in -6..6_i64 {
                    let (jx, jy) = feature2(finish(fold(fold(state, cx), cy)));
                    let point = DVec2::new(cx as f64 + jx, cy as f64 + jy);
                    assert_eq!(point.floor(), DVec2::new(cx as f64, cy as f64));
                    let (f1, f2) = worley2(seed, point);
                    assert_eq!(f1, 0.0, "cell ({cx}, {cy})");
                    assert!(f2 > 0.0);
                    // The tiled lattice puts the same point into every repetition of the cell.
                    let tiled = Tiled::new(4);
                    let wrapped = fold(fold(state, tiled.wrap(cx)), tiled.wrap(cy));
                    let (jx, jy) = feature2(finish(wrapped));
                    let point = DVec2::new(cx as f64 + jx, cy as f64 + jy);
                    assert_eq!(worley2_tiled(seed, point, 4).0, 0.0);
                    for cz in -3..3_i64 {
                        let (jx, jy, jz) = feature3(finish(fold(fold(fold(state, cx), cy), cz)));
                        let point = DVec3::new(cx as f64 + jx, cy as f64 + jy, cz as f64 + jz);
                        let (f1, f2) = worley3(seed, point);
                        assert_eq!(f1, 0.0, "cell ({cx}, {cy}, {cz})");
                        assert!(f2 > 0.0);
                    }
                }
            }
        }
    }
}
