//! Tileable building blocks for the recipes: noise wrappers, cells and scattered points.
//!
//! Positions are in **design units**: the tile spans 16 units on both axes, one unit per texel
//! of the native 16 px resolution. Every function here is periodic with period 16 on both
//! axes, *exactly*: positions one or more tiles apart give bit-identical results, as long as
//! the position is a texel centre (a multiple of a small power of two). The wrappers reduce
//! coordinates with exact arithmetic before anything is rounded.

use cp_core::det;
use cp_core::hash::{Rng, hash_combine};
use cp_core::math::DVec2;
use cp_core::noise::Fbm;

/// Design units per tile.
pub(crate) const TILE: f64 = 16.0;
/// Edge length of the design grid that stamps (hand-placed pixels) are drawn on.
pub(crate) const DESIGN: usize = 16;

/// Maps a hash to `[0, 1)` using its top 53 bits.
pub(crate) fn unit(hash: u64) -> f64 {
    (hash >> 11) as f64 * (1.0 / 9_007_199_254_740_992.0)
}

/// Wraps a design coordinate into `[0, 16)`. Exact.
pub(crate) fn wrap(v: f64) -> f64 {
    v.rem_euclid(TILE)
}

/// Lattice position for a noise with `period` cells per tile, reduced to `[0, period)`.
fn lattice(p: DVec2, period: u32) -> DVec2 {
    let cells = f64::from(period);
    // Multiplying by an integer and dividing by 16 are exact for texel centres, and so is the
    // remainder; the noise therefore sees identical input in every repetition of the tile.
    DVec2::new(
        (p.x * cells / TILE).rem_euclid(cells),
        (p.y * cells / TILE).rem_euclid(cells),
    )
}

/// Isotropic fractal noise in `[-1, 1]` with `period` lattice cells per tile, i.e. features of
/// about `16 / period` units.
pub(crate) fn fbm(seed: u64, p: DVec2, period: u32, octaves: u32) -> f64 {
    let period = period.max(1);
    Fbm::new(octaves, 2.0, 0.5).sample2_tiled(seed, lattice(p, period), period)
}

/// Ridged fractal noise in `[-1, 1]`: thin bright veins on a dark ground.
pub(crate) fn ridged(seed: u64, p: DVec2, period: u32, octaves: u32) -> f64 {
    let period = period.max(1);
    Fbm::new(octaves, 2.0, 0.5).ridged2_tiled(seed, lattice(p, period), period)
}

fn fade(t: f64) -> f64 {
    t * t * (3.0 - 2.0 * t)
}

/// A shift of a pattern's lattice that depends on the seed, in `[0, 16)` on both axes.
///
/// Lattice patterns have structure along their lattice lines (value noise is flat there, cell
/// borders gather near them). Without the shift one of those lines would always lie on the
/// tile's border and mark it; with it the border is just another place. The shift is a
/// multiple of 1/64, so shifted texel centres stay exactly representable.
fn drift(seed: u64) -> DVec2 {
    let hash = hash_combine(seed, 0x6472_6966_7400);
    DVec2::new(
        (hash & 0x3FF) as f64 / 64.0,
        ((hash >> 10) & 0x3FF) as f64 / 64.0,
    )
}

/// Value noise in `[-1, 1]` with a separate number of lattice cells per tile on each axis.
///
/// `cp_core`'s tileable noise has one period for both axes; stretched features (wood grain,
/// strata, streaks) need two.
pub(crate) fn value(seed: u64, p: DVec2, periods: (u32, u32)) -> f64 {
    let p = p + drift(seed);
    let (px, py) = (i64::from(periods.0.max(1)), i64::from(periods.1.max(1)));
    let (qx, qy) = (p.x * px as f64 / TILE, p.y * py as f64 / TILE);
    let (ix, iy) = (qx.floor(), qy.floor());
    let (fx, fy) = (qx - ix, qy - iy);
    let corner = |dx: i64, dy: i64| {
        let cx = (ix as i64 + dx).rem_euclid(px) as u64;
        let cy = (iy as i64 + dy).rem_euclid(py) as u64;
        unit(hash_combine(hash_combine(seed, cx), cy)) * 2.0 - 1.0
    };
    let (u, v) = (fade(fx), fade(fy));
    let top = corner(0, 0) + (corner(1, 0) - corner(0, 0)) * u;
    let bottom = corner(0, 1) + (corner(1, 1) - corner(0, 1)) * u;
    top + (bottom - top) * v
}

/// Sum of `octaves` layers of [`value`] noise, each twice as fine and half as strong as the
/// one before, normalised to `[-1, 1]`.
pub(crate) fn value_fbm(seed: u64, p: DVec2, periods: (u32, u32), octaves: u32) -> f64 {
    let (mut sum, mut total, mut amplitude) = (0.0, 0.0, 1.0);
    let mut periods = periods;
    for octave in 0..octaves {
        sum += amplitude * value(hash_combine(seed, u64::from(octave)), p, periods);
        total += amplitude;
        amplitude *= 0.5;
        periods = (periods.0.saturating_mul(2), periods.1.saturating_mul(2));
    }
    if total > 0.0 { sum / total } else { 0.0 }
}

/// What [`cells`] knows about the cell a position lies in.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Cell {
    /// Hash of the cell, the same in every repetition of the tile.
    pub hash: u64,
    /// Distance to the nearest cell border in design units; zero on the border.
    pub edge: f64,
    /// Unit vector from the position towards that border (image space: x right, y down).
    pub outward: DVec2,
}

/// Cellular (Voronoi) pattern with one feature point in every cell of a `counts.0 × counts.1`
/// grid over the tile. `jitter` in `[0, 1]` moves the points away from the cell centres:
/// 0 gives a regular grid, 1 fully irregular cells.
///
/// Unlike `cp_core::noise::worley2` this reports which cell a position belongs to and the true
/// distance to the cell border, which stones, scales and leaf clumps need for their outlines.
pub(crate) fn cells(seed: u64, p: DVec2, counts: (u32, u32), jitter: f64) -> Cell {
    cells_on(seed, p, counts, jitter, false)
}

/// Like [`cells`], with every other row of feature points moved half a cell sideways: without
/// jitter the cells are hexagons. The number of rows is rounded up to an even number, since
/// the offset rows have to alternate across the tile's edge as well.
pub(crate) fn honeycomb(seed: u64, p: DVec2, counts: (u32, u32), jitter: f64) -> Cell {
    cells_on(seed, p, (counts.0, counts.1 + counts.1 % 2), jitter, true)
}

fn cells_on(seed: u64, p: DVec2, counts: (u32, u32), jitter: f64, stagger: bool) -> Cell {
    let p = p + drift(seed);
    let (nx, ny) = (i64::from(counts.0.max(1)), i64::from(counts.1.max(1)));
    let (sx, sy) = (TILE / nx as f64, TILE / ny as f64);
    let (qx, qy) = (p.x * nx as f64 / TILE, p.y * ny as f64 / TILE);
    let (ix, iy) = (qx.floor(), qy.floor());
    let (fx, fy) = (qx - ix, qy - iy);

    // Offsets from the position to the feature points of the surrounding 5×5 cells.
    let mut points = [(DVec2::ZERO, 0_u64); 25];
    let mut nearest = 0;
    let mut nearest_distance = f64::INFINITY;
    for (slot, point) in points.iter_mut().enumerate() {
        let (dx, dy) = ((slot % 5) as i64 - 2, (slot / 5) as i64 - 2);
        let cx = (ix as i64 + dx).rem_euclid(nx) as u64;
        let cy = (iy as i64 + dy).rem_euclid(ny) as u64;
        let hash = hash_combine(hash_combine(seed, cx), cy);
        let jx = 0.5 + jitter * (unit(hash) - 0.5);
        let jy = 0.5 + jitter * (unit(hash_combine(hash, 1)) - 0.5);
        // The row's parity is the same in every repetition because the row count is even.
        let shift = if stagger && (iy as i64 + dy).rem_euclid(2) == 1 {
            0.5
        } else {
            0.0
        };
        let offset = DVec2::new(
            (dx as f64 + shift + jx - fx) * sx,
            (dy as f64 + jy - fy) * sy,
        );
        let distance = offset.length_squared();
        if distance < nearest_distance {
            nearest_distance = distance;
            nearest = slot;
        }
        *point = (offset, hash);
    }

    let (own, hash) = points[nearest];
    let mut edge = f64::INFINITY;
    let mut outward = DVec2::ZERO;
    for (slot, &(other, _)) in points.iter().enumerate() {
        let between = other - own;
        let length = between.length();
        if slot == nearest || length <= 0.0 {
            continue;
        }
        // Distance to the line of points equally far from both feature points.
        let direction = between / length;
        let distance = (own + other).dot(direction) * 0.5;
        if distance < edge {
            edge = distance;
            outward = direction;
        }
    }
    Cell {
        hash,
        edge: det::max(edge, 0.0),
        outward,
    }
}

/// Distance between two coordinates on the design grid's ring of 16.
fn ring_distance(a: i64, b: i64) -> i64 {
    let d = (a - b).rem_euclid(DESIGN as i64);
    d.min(DESIGN as i64 - d)
}

/// Up to `count` positions on the 16×16 design grid that keep at least `spacing` units from
/// each other across the tile's edges. Dart throwing: evenly spread without looking regular.
pub(crate) fn scatter(rng: &mut Rng, count: usize, spacing: f64) -> Vec<(i64, i64)> {
    let mut points: Vec<(i64, i64)> = Vec::with_capacity(count);
    let limit = spacing * spacing;
    for _ in 0..count.saturating_mul(24) {
        if points.len() >= count {
            break;
        }
        let candidate = (
            i64::from(rng.range_u32(0, DESIGN as u32)),
            i64::from(rng.range_u32(0, DESIGN as u32)),
        );
        let free = points.iter().all(|&other| {
            let dx = ring_distance(candidate.0, other.0);
            let dy = ring_distance(candidate.1, other.1);
            (dx * dx + dy * dy) as f64 >= limit
        });
        if free {
            points.push(candidate);
        }
    }
    points
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Texel centres of a 32 px texture plus whole-tile shifts: where recipes sample.
    fn sample_points() -> Vec<DVec2> {
        let mut points = Vec::new();
        for y in 0..32 {
            for x in 0..32 {
                points.push(DVec2::new(x as f64 + 0.5, y as f64 + 0.5) * 0.5);
            }
        }
        points
    }

    const SHIFTS: [DVec2; 4] = [
        DVec2::new(16.0, 0.0),
        DVec2::new(0.0, -16.0),
        DVec2::new(-48.0, 32.0),
        DVec2::new(16.0 * 37.0, 16.0 * 5.0),
    ];

    #[test]
    fn noise_wrappers_are_exactly_periodic_and_in_range() {
        for p in sample_points() {
            for period in [1, 2, 3, 5, 6, 8] {
                let a = fbm(7, p, period, 2);
                let b = ridged(7, p, period, 2);
                let c = value(7, p, (period, 3));
                let d = value_fbm(7, p, (2, period), 3);
                for n in [a, b, c, d] {
                    assert!((-1.0..=1.0).contains(&n), "{n} at {p:?}");
                }
                for shift in SHIFTS {
                    let q = p + shift;
                    assert_eq!(fbm(7, q, period, 2).to_bits(), a.to_bits());
                    assert_eq!(ridged(7, q, period, 2).to_bits(), b.to_bits());
                    assert_eq!(value(7, q, (period, 3)).to_bits(), c.to_bits());
                    assert_eq!(value_fbm(7, q, (2, period), 3).to_bits(), d.to_bits());
                }
            }
        }
    }

    #[test]
    fn noise_depends_on_seed_and_position() {
        let p = DVec2::new(3.5, 9.5);
        assert_ne!(fbm(1, p, 4, 2), fbm(2, p, 4, 2));
        assert_ne!(value(1, p, (4, 4)), value(2, p, (4, 4)));
        assert_ne!(value(1, p, (4, 4)), value(1, p + DVec2::X, (4, 4)));
        assert_eq!(value_fbm(1, p, (4, 4), 0), 0.0);
        // Not only texel centres: any position that is a multiple of 1/64 repeats exactly.
        let corner = value(9, DVec2::new(4.0, 8.015_625), (4, 2));
        assert!((-1.0..=1.0).contains(&corner));
        assert_eq!(corner, value(9, DVec2::new(20.0, -7.984_375), (4, 2)));
    }

    #[test]
    fn lattices_are_shifted_by_a_seed_dependent_offset() {
        let mut seen = std::collections::BTreeSet::new();
        for seed in 0..200 {
            let shift = drift(seed);
            assert!((0.0..16.0).contains(&shift.x) && (0.0..16.0).contains(&shift.y));
            // Multiples of 1/64: adding them to a texel centre is exact.
            assert_eq!((shift * 64.0).fract(), DVec2::ZERO);
            seen.insert(((shift.x * 64.0) as u32, (shift.y * 64.0) as u32));
        }
        assert!(seen.len() > 190, "{}", seen.len());
        // Value noise takes its lattice values exactly at the shifted lattice points: there
        // its slope vanishes, so stepping off by a little changes almost nothing.
        let corner = DVec2::new(4.0, 8.0) - drift(9);
        let at_corner = value(9, corner, (4, 2));
        let beside = value(9, corner + DVec2::new(0.015_625, 0.0), (4, 2));
        assert!((at_corner - beside).abs() < 2.0e-4, "{at_corner} {beside}");
        let midway = value(9, corner + DVec2::new(2.0, 0.0), (4, 2));
        assert!((at_corner - midway).abs() > 2.0e-4, "{at_corner} {midway}");
    }

    #[test]
    fn stretched_noise_varies_faster_along_the_finer_axis() {
        // Mean absolute difference between neighbouring texels along x and along y.
        let (mut along_x, mut along_y) = (0.0, 0.0);
        for p in sample_points() {
            let here = value(3, p, (8, 2));
            along_x += (value(3, p + DVec2::X * 0.5, (8, 2)) - here).abs();
            along_y += (value(3, p + DVec2::Y * 0.5, (8, 2)) - here).abs();
        }
        assert!(along_x > 2.5 * along_y, "{along_x} vs {along_y}");
    }

    #[test]
    fn cells_are_exactly_periodic() {
        for p in sample_points() {
            let cell = cells(11, p, (3, 4), 0.8);
            assert!(cell.edge >= 0.0 && cell.edge.is_finite());
            assert!((cell.outward.length() - 1.0).abs() < 1.0e-12);
            for shift in SHIFTS {
                let other = cells(11, p + shift, (3, 4), 0.8);
                assert_eq!(other.hash, cell.hash);
                assert_eq!(other.edge.to_bits(), cell.edge.to_bits());
                assert_eq!(other.outward, cell.outward);
            }
        }
    }

    #[test]
    fn cell_borders_separate_cells() {
        // Walking along a row, the cell hash changes exactly where the edge distance has a
        // local minimum close to zero, and the count of distinct cells matches the grid.
        let mut hashes = std::collections::BTreeSet::new();
        let step = 1.0 / 16.0;
        for row in 0..64 {
            let y = row as f64 * 0.25 + 0.125;
            let mut previous = cells(5, DVec2::new(0.5 * step, y), (4, 4), 0.9);
            for i in 1..256 {
                let cell = cells(5, DVec2::new((i as f64 + 0.5) * step, y), (4, 4), 0.9);
                if cell.hash != previous.hash {
                    assert!(
                        cell.edge < step && previous.edge < step,
                        "border not at a cell change: {} / {}",
                        previous.edge,
                        cell.edge
                    );
                }
                hashes.insert(cell.hash);
                previous = cell;
            }
        }
        assert_eq!(hashes.len(), 16);
    }

    #[test]
    fn the_border_normal_points_away_from_the_cell() {
        // Stepping along `outward` by a little less than `edge` stays inside the cell, a little
        // more crosses into the neighbour.
        for p in sample_points().into_iter().step_by(7) {
            let cell = cells(21, p, (3, 3), 0.85);
            if cell.edge < 0.05 {
                continue;
            }
            let inside = cells(21, p + cell.outward * (cell.edge - 0.02), (3, 3), 0.85);
            let outside = cells(21, p + cell.outward * (cell.edge + 0.02), (3, 3), 0.85);
            assert_eq!(inside.hash, cell.hash, "at {p:?}");
            assert_ne!(outside.hash, cell.hash, "at {p:?}");
        }
    }

    #[test]
    fn a_regular_grid_has_square_cells() {
        // Without jitter the feature points sit in the centres of the (shifted) grid cells: the
        // border is the grid.
        let centre = DVec2::new(2.0, 2.0) - drift(1);
        let cell = cells(1, centre, (4, 4), 0.0);
        assert!((cell.edge - 2.0).abs() < 1.0e-12);
        let cell = cells(1, centre + DVec2::new(1.5, 0.0), (4, 4), 0.0);
        assert!((cell.edge - 0.5).abs() < 1.0e-12);
        assert!((cell.outward - DVec2::X).length() < 1.0e-12);
    }

    #[test]
    fn a_regular_honeycomb_has_six_sided_cells() {
        // Feature points: rows 4 units apart, columns 4 units apart, odd rows shifted by 2.
        // The cell around a feature point is a hexagon whose corners are 2.5 units from its
        // centre, so a circle of radius 2.6 runs through its six neighbours and nothing else.
        let centre = DVec2::new(2.0, 2.0) - drift(8);
        let own = honeycomb(8, centre, (4, 4), 0.0);
        assert!((own.edge - 2.0).abs() < 1.0e-12);
        let mut neighbours = std::collections::BTreeSet::new();
        for step in 0..360 {
            let (sin, cos) = det::sin_cos(f64::from(step).to_radians());
            let p = centre + 2.6 * DVec2::new(cos, sin);
            let cell = honeycomb(8, p, (4, 4), 0.0);
            assert_ne!(cell.hash, own.hash, "step {step}");
            neighbours.insert(cell.hash);
        }
        assert_eq!(neighbours.len(), 6);
        // The same walk on the square grid meets all eight neighbours.
        let mut neighbours = std::collections::BTreeSet::new();
        for step in 0..360 {
            let (sin, cos) = det::sin_cos(f64::from(step).to_radians());
            let p = centre + 2.9 * DVec2::new(cos, sin);
            neighbours.insert(cells(8, p, (4, 4), 0.0).hash);
        }
        assert_eq!(neighbours.len(), 8);
    }

    #[test]
    fn honeycombs_are_exactly_periodic_even_for_odd_row_counts() {
        for p in sample_points().into_iter().step_by(3) {
            for counts in [(3, 4), (3, 3), (5, 1)] {
                let cell = honeycomb(13, p, counts, 0.3);
                for shift in SHIFTS {
                    let other = honeycomb(13, p + shift, counts, 0.3);
                    assert_eq!(other.hash, cell.hash);
                    assert_eq!(other.edge.to_bits(), cell.edge.to_bits());
                    assert_eq!(other.outward, cell.outward);
                }
            }
        }
    }

    #[test]
    fn scattered_points_keep_their_distance_across_the_edges() {
        let mut rng = Rng::new(3, 1);
        let points = scatter(&mut rng, 12, 3.5);
        assert!(points.len() >= 8, "{}", points.len());
        for (i, a) in points.iter().enumerate() {
            assert!((0..16).contains(&a.0) && (0..16).contains(&a.1));
            for b in &points[..i] {
                let (dx, dy) = (ring_distance(a.0, b.0), ring_distance(a.1, b.1));
                assert!((dx * dx + dy * dy) as f64 >= 3.5 * 3.5, "{a:?} {b:?}");
            }
        }
        // More points than fit: the function gives up instead of looping forever.
        assert!(scatter(&mut rng, 400, 3.0).len() < 40);
        assert!(scatter(&mut rng, 0, 3.0).is_empty());
        assert_eq!(ring_distance(1, 15), 2);
        assert_eq!(ring_distance(15, 1), 2);
        assert_eq!(ring_distance(4, 12), 8);
    }

    #[test]
    fn unit_and_wrap_stay_in_range() {
        assert_eq!(unit(0), 0.0);
        assert!(unit(u64::MAX) < 1.0);
        assert_eq!(wrap(16.0), 0.0);
        assert_eq!(wrap(-0.5), 15.5);
        assert_eq!(wrap(35.25), 3.25);
    }
}
