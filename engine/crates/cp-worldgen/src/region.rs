//! Cell geometry shared by the generators (contract §2.2, §5.1).
//!
//! A cell of level `L` with coordinate `c` covers the voxels `[c * 2^L, (c + 1) * 2^L)` and
//! has its centre at `(c + 0.5) * 2^L` m. Coordinates are `i64` here, so halos, products and
//! doubled coordinates never overflow for valid keys. Every cell decision compares a cell
//! centre (exact in `f64`) with a field value; the helpers below turn such comparisons into
//! index thresholds without changing a single decision.

use cp_core::coords::MAX_LOD_LEVEL;
use cp_core::det;
use cp_core::ids::BlockStateId;
use cp_core::math::IAabb;

/// Field values are clamped to this magnitude (m) before thresholds are derived from them, so
/// that a corrupt value can neither overflow an index nor keep a search loop running. Valid
/// worlds never come near it: voxel coordinates stay below 2^30 m.
const VALUE_GUARD: f64 = 1.0e12;

/// A half-open box of cells of one level: `min <= c < max` on every axis.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Region {
    pub(crate) min: [i64; 3],
    pub(crate) max: [i64; 3],
}

impl Region {
    pub(crate) const fn new(min: [i64; 3], max: [i64; 3]) -> Self {
        Self { min, max }
    }

    pub(crate) fn from_aabb(aabb: IAabb) -> Self {
        Self {
            min: [aabb.min.x.into(), aabb.min.y.into(), aabb.min.z.into()],
            max: [aabb.max.x.into(), aabb.max.y.into(), aabb.max.z.into()],
        }
    }

    /// Number of cells along `axis` (0 when the box is empty on that axis).
    pub(crate) fn extent(&self, axis: usize) -> usize {
        usize::try_from(self.max[axis] - self.min[axis]).unwrap_or(0)
    }

    pub(crate) fn volume(&self) -> usize {
        self.extent(0) * self.extent(1) * self.extent(2)
    }

    pub(crate) fn is_empty(&self) -> bool {
        (0..3).any(|axis| self.max[axis] <= self.min[axis])
    }

    pub(crate) fn contains(&self, cell: [i64; 3]) -> bool {
        (0..3).all(|axis| self.min[axis] <= cell[axis] && cell[axis] < self.max[axis])
    }

    /// Index of `(x, y, z)` in the `fill` layout: x fastest, then z, then y.
    #[inline]
    pub(crate) fn index(&self, x: i64, y: i64, z: i64) -> usize {
        debug_assert!(self.contains([x, y, z]), "cell outside the region");
        let nx = self.max[0] - self.min[0];
        let nz = self.max[2] - self.min[2];
        // Non-negative and below the region volume, which fits in usize.
        ((x - self.min[0]) + (z - self.min[2]) * nx + (y - self.min[1]) * nx * nz) as usize
    }

    /// The cells common to both boxes, or `None` when there are none.
    pub(crate) fn intersection(&self, other: &Self) -> Option<Self> {
        let min = core::array::from_fn(|axis| self.min[axis].max(other.min[axis]));
        let max = core::array::from_fn(|axis| self.max[axis].min(other.max[axis]));
        let region = Self { min, max };
        (!region.is_empty()).then_some(region)
    }

    pub(crate) fn intersects(&self, other: &Self) -> bool {
        self.intersection(other).is_some()
    }

    /// Smallest box containing both (neither may be empty).
    pub(crate) fn union(&self, other: &Self) -> Self {
        Self {
            min: core::array::from_fn(|axis| self.min[axis].min(other.min[axis])),
            max: core::array::from_fn(|axis| self.max[axis].max(other.max[axis])),
        }
    }

    /// The level-0 voxels covered by this box of level-`level` cells.
    pub(crate) fn voxels(&self, level: u8) -> Self {
        let size = 1_i64 << level;
        Self {
            min: self.min.map(|c| c * size),
            max: self.max.map(|c| c * size),
        }
    }
}

/// Checks a `fill` request and converts its region.
///
/// # Panics
/// If `level > MAX_LOD_LEVEL` or `out.len()` differs from the region's cell count; both are
/// violated preconditions of `BaseSource::fill` (programming errors of the caller).
pub(crate) fn checked_request(level: u8, region: IAabb, out: &[BlockStateId]) -> Region {
    assert!(
        level <= MAX_LOD_LEVEL,
        "fill: level {level} is above MAX_LOD_LEVEL"
    );
    let region = Region::from_aabb(region);
    assert_eq!(
        out.len(),
        region.volume(),
        "fill: the output holds {} cells, the region {region:?} has {}",
        out.len(),
        region.volume()
    );
    region
}

/// Edge of a level-`level` cell in metres, `2^level` (exact).
#[inline]
pub(crate) fn cell_size(level: u8) -> f64 {
    f64::from(1_u32 << level)
}

/// Centre of cell `c` of `level` along one axis in metres, `(c + 0.5) * 2^level`. Exact for
/// `|c| < 2^52`.
#[inline]
pub(crate) fn centre(level: u8, c: i64) -> f64 {
    (c as f64 + 0.5) * cell_size(level)
}

fn guard(value: f64) -> f64 {
    if value.is_nan() {
        0.0
    } else {
        det::clamp(value, -VALUE_GUARD, VALUE_GUARD)
    }
}

/// The largest cell index of `level` whose centre lies strictly below `value`.
pub(crate) fn last_centre_below(level: u8, value: f64) -> i64 {
    let value = guard(value);
    // A first guess from floating point, then corrected against the exact predicate, so the
    // result agrees with `centre(level, c) < value` for every `c` however the guess rounded.
    let mut c = (value / cell_size(level) - 0.5).floor() as i64;
    while centre(level, c + 1) < value {
        c += 1;
    }
    while centre(level, c) >= value {
        c -= 1;
    }
    c
}

/// The smallest cell index of `level` whose centre lies strictly above `value`.
pub(crate) fn first_centre_above(level: u8, value: f64) -> i64 {
    let value = guard(value);
    let mut c = (value / cell_size(level) - 0.5).floor() as i64 + 1;
    while centre(level, c - 1) > value {
        c -= 1;
    }
    while centre(level, c) <= value {
        c += 1;
    }
    c
}

/// The cells of `level` that overlap the voxels `[lo, hi)` along one axis, as a half-open
/// index range (`hi > lo`).
#[inline]
pub(crate) fn cells_overlapping(level: u8, lo: i64, hi: i64) -> (i64, i64) {
    (lo >> level, ((hi - 1) >> level) + 1)
}

/// `floor(a / b)` for `b > 0`.
#[inline]
pub(crate) fn floor_div(a: i64, b: i64) -> i64 {
    a.div_euclid(b)
}

/// `ceil(a / b)` for `b > 0`.
#[inline]
pub(crate) fn ceil_div(a: i64, b: i64) -> i64 {
    -(-a).div_euclid(b)
}

#[cfg(test)]
mod tests {
    use super::*;
    use cp_core::hash::Rng;
    use cp_core::math::IVec3;

    #[test]
    fn index_layout_is_x_then_z_then_y() {
        let region = Region::from_aabb(IAabb::new(IVec3::new(-3, 5, 7), IVec3::new(1, 8, 9)));
        assert_eq!(
            [region.extent(0), region.extent(1), region.extent(2)],
            [4, 3, 2]
        );
        assert_eq!(region.volume(), 24);
        let mut expected = 0;
        for y in 5..8 {
            for z in 7..9 {
                for x in -3..1 {
                    assert_eq!(region.index(x, y, z), expected);
                    expected += 1;
                }
            }
        }
        // Agrees with the layout helper of the trait's crate.
        let aabb = IAabb::new(IVec3::new(-3, 5, 7), IVec3::new(1, 8, 9));
        let cell = IVec3::new(0, 6, 8);
        assert_eq!(region.index(0, 6, 8), cp_voxel::region_index(aabb, cell));
    }

    #[test]
    fn boxes_intersect_and_unite() {
        let a = Region::new([0, 0, 0], [4, 4, 4]);
        let b = Region::new([2, -1, 3], [9, 2, 5]);
        assert_eq!(a.intersection(&b), Some(Region::new([2, 0, 3], [4, 2, 4])));
        assert!(a.intersects(&b));
        assert_eq!(a.union(&b), Region::new([0, -1, 0], [9, 4, 5]));
        let far = Region::new([4, 0, 0], [5, 1, 1]);
        assert_eq!(a.intersection(&far), None);
        assert!(Region::new([0, 0, 0], [0, 1, 1]).is_empty());
        assert_eq!(
            Region::new([-1, 0, 2], [1, 1, 3]).voxels(3),
            Region::new([-8, 0, 16], [8, 8, 24])
        );
    }

    #[test]
    fn centre_thresholds_match_the_predicate() {
        let mut rng = Rng::new(17, 4);
        let mut values = vec![
            0.0, -0.0, 0.5, -0.5, 1.0, -1.0, 3.0, 2.5, -2.5, 1.0e-300, -1.0e-300,
        ];
        for _ in 0..4000 {
            values.push(rng.range_f64(-5000.0, 5000.0));
        }
        for _ in 0..500 {
            // Values exactly on cell centres and boundaries of some level.
            let level = rng.range_u32(0, 8) as u8;
            let c = i64::from(rng.range_u32(0, 2000)) - 1000;
            values.push(centre(level, c));
            values.push(c as f64 * cell_size(level));
        }
        for &value in &values {
            for level in [0_u8, 1, 2, 3, 5, 9, 20] {
                let below = last_centre_below(level, value);
                assert!(centre(level, below) < value, "{level} {value}");
                assert!(centre(level, below + 1) >= value, "{level} {value}");
                let above = first_centre_above(level, value);
                assert!(centre(level, above) > value, "{level} {value}");
                assert!(centre(level, above - 1) <= value, "{level} {value}");
            }
        }
        // Corrupt values neither hang nor overflow.
        for value in [
            f64::NAN,
            f64::INFINITY,
            f64::NEG_INFINITY,
            1.0e300,
            -1.0e300,
        ] {
            let _ = (last_centre_below(4, value), first_centre_above(4, value));
        }
    }

    #[test]
    fn overlap_and_division_helpers() {
        assert_eq!(cells_overlapping(0, -3, 2), (-3, 2));
        assert_eq!(cells_overlapping(2, -3, 2), (-1, 1));
        assert_eq!(cells_overlapping(3, 8, 9), (1, 2));
        assert_eq!(cells_overlapping(3, 7, 9), (0, 2));
        assert_eq!(cells_overlapping(3, -1, 0), (-1, 0));
        for a in -20..20 {
            for b in [1, 2, 3, 8] {
                let exact = f64::from(a) / f64::from(b);
                assert_eq!(floor_div(a.into(), b.into()), exact.floor() as i64);
                assert_eq!(ceil_div(a.into(), b.into()), exact.ceil() as i64);
            }
        }
    }
}
