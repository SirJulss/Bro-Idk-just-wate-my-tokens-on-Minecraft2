//! Square grids that wrap around: the canvas every recipe paints on.
//!
//! Every neighbourhood operation here treats the grid as a torus and depends only on the
//! relative arrangement of the cells, never on where the grid's first row or column happens to
//! be. That is what makes the finished textures tile without a seam (the tests shift the window
//! and expect the rolled image, bit for bit).

use cp_core::det;

/// A `size × size` grid in row-major order, row 0 at the top.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Grid<T> {
    size: usize,
    cells: Vec<T>,
}

impl<T: Copy> Grid<T> {
    pub(crate) fn filled(size: usize, value: T) -> Self {
        Self {
            size,
            cells: vec![value; size * size],
        }
    }

    /// Builds a grid by calling `f(x, y)` for every cell, row by row.
    pub(crate) fn from_fn(size: usize, mut f: impl FnMut(usize, usize) -> T) -> Self {
        let mut cells = Vec::with_capacity(size * size);
        for y in 0..size {
            for x in 0..size {
                cells.push(f(x, y));
            }
        }
        Self { size, cells }
    }

    pub(crate) fn size(&self) -> usize {
        self.size
    }

    pub(crate) fn cells(&self) -> &[T] {
        &self.cells
    }

    /// Cell at `(x, y)`; both must be below `size`.
    pub(crate) fn get(&self, x: usize, y: usize) -> T {
        self.cells[y * self.size + x]
    }

    /// Cell at any integer position: the grid repeats in both directions.
    pub(crate) fn wrapped(&self, x: i64, y: i64) -> T {
        self.cells[self.wrapped_index(x, y)]
    }

    /// Cell `(dx, dy)` away from `(x, y)`, wrapping around the edges.
    pub(crate) fn near(&self, x: usize, y: usize, dx: i64, dy: i64) -> T {
        self.wrapped(x as i64 + dx, y as i64 + dy)
    }

    pub(crate) fn set_wrapped(&mut self, x: i64, y: i64, value: T) {
        let index = self.wrapped_index(x, y);
        self.cells[index] = value;
    }

    fn wrapped_index(&self, x: i64, y: i64) -> usize {
        let n = self.size as i64;
        (y.rem_euclid(n) * n + x.rem_euclid(n)) as usize
    }

    pub(crate) fn map<U: Copy>(&self, f: impl FnMut(T) -> U) -> Grid<U> {
        Grid {
            size: self.size,
            cells: self.cells.iter().copied().map(f).collect(),
        }
    }
}

impl<A: Copy, B: Copy> Grid<(A, B)> {
    /// Splits a grid of pairs into two grids.
    pub(crate) fn unzip(&self) -> (Grid<A>, Grid<B>) {
        (self.map(|pair| pair.0), self.map(|pair| pair.1))
    }
}

impl Grid<f64> {
    /// Smallest and largest cell. Independent of the order of the cells.
    pub(crate) fn min_max(&self) -> (f64, f64) {
        self.cells
            .iter()
            .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), &v| {
                (det::min(lo, v), det::max(hi, v))
            })
    }

    /// Rescales the cells linearly so that they span `[0, 1]`; a constant grid becomes 0.5.
    pub(crate) fn normalized(&self) -> Self {
        let (lo, hi) = self.min_max();
        let range = hi - lo;
        if range > 0.0 {
            self.map(|v| (v - lo) / range)
        } else {
            self.map(|_| 0.5)
        }
    }

    /// 3×3 binomial blur (weights 1-2-1 on both axes) on the torus.
    pub(crate) fn blurred(&self) -> Self {
        Grid::from_fn(self.size, |x, y| {
            let at = |dx: i64, dy: i64| self.near(x, y, dx, dy);
            let corners = at(-1, -1) + at(1, -1) + at(-1, 1) + at(1, 1);
            let edges = at(0, -1) + at(-1, 0) + at(1, 0) + at(0, 1);
            (corners + 2.0 * edges + 4.0 * at(0, 0)) / 16.0
        })
    }

    /// Separable blur on the torus with tent weights `reach - |k|` for `|k| < reach`. A reach of
    /// 2 is the 1-2-1 kernel of [`Grid::blurred`], up to rounding.
    pub(crate) fn tent_blurred(&self, reach: usize) -> Self {
        let reach = reach.max(1) as i64;
        let weights: Vec<f64> = (1 - reach..reach)
            .map(|k| (reach - k.abs()) as f64)
            .collect();
        self.separable(&weights)
    }

    /// Mean over the `2 * reach - 1` cells around each cell on both axes (separable, on the
    /// torus). A reach of 1 leaves the grid as it is.
    pub(crate) fn box_blurred(&self, reach: usize) -> Self {
        if reach <= 1 {
            return self.clone();
        }
        self.separable(&vec![1.0; 2 * reach - 1])
    }

    /// Convolution with `weights` (odd length, centred, normalised here) along the rows, then
    /// along the columns.
    fn separable(&self, weights: &[f64]) -> Self {
        let total: f64 = weights.iter().sum();
        let reach = (weights.len() / 2) as i64;
        let pass = |grid: &Self, along_rows: bool| {
            Grid::from_fn(grid.size, |x, y| {
                let mut sum = 0.0;
                for (i, weight) in weights.iter().enumerate() {
                    let k = i as i64 - reach;
                    let (dx, dy) = if along_rows { (k, 0) } else { (0, k) };
                    sum += weight * grid.near(x, y, dx, dy);
                }
                sum / total
            })
        };
        pass(&pass(self, true), false)
    }

    /// The value `t` for which the highest `fraction` of the cells satisfy `cell >= t`.
    ///
    /// Thresholding by rank instead of by value keeps the share of every tone the same for all
    /// seeds. `fraction <= 0` returns infinity (no cell qualifies), `fraction >= 1` the minimum.
    /// Cells with equal values fall on the same side, so the share is exact only for distinct
    /// values.
    pub(crate) fn threshold_above(&self, fraction: f64) -> f64 {
        let mut sorted = self.cells.clone();
        sorted.sort_by(f64::total_cmp);
        let count = sorted.len();
        let wanted = (fraction * count as f64).round();
        if wanted < 1.0 || count == 0 {
            f64::INFINITY
        } else if wanted >= count as f64 {
            sorted[0]
        } else {
            sorted[count - wanted as usize]
        }
    }

    /// Splits the cells into `shares.len()` classes of increasing value whose sizes are
    /// proportional to `shares`; class 0 holds the lowest values.
    pub(crate) fn quantize(&self, shares: &[f64]) -> Grid<u8> {
        let total: f64 = shares.iter().sum();
        let mut thresholds = Vec::with_capacity(shares.len());
        let mut above = total;
        for share in shares.iter().take(shares.len().saturating_sub(1)) {
            above -= share;
            thresholds.push(self.threshold_above(above / total));
        }
        self.map(|v| thresholds.iter().filter(|&&t| v >= t).count() as u8)
    }
}

impl Grid<u8> {
    /// Removes single stray texels: a cell none of whose four edge neighbours has its class
    /// takes the most common class among them. Ties go to the class closest to the cell's own,
    /// then to the lower one. Cells for which `keep` returns true are left alone.
    ///
    /// Stray texels are what makes thresholded noise look like static; hand-pixelled art has
    /// none that are not deliberate.
    pub(crate) fn despeckled(&self, keep: impl Fn(u8) -> bool) -> Self {
        Grid::from_fn(self.size, |x, y| {
            let own = self.get(x, y);
            let around = [
                self.near(x, y, -1, 0),
                self.near(x, y, 1, 0),
                self.near(x, y, 0, -1),
                self.near(x, y, 0, 1),
            ];
            if keep(own) || around.contains(&own) {
                return own;
            }
            let mut best = around[0];
            let mut best_key = (0_usize, u8::MAX, u8::MAX);
            for candidate in around {
                let votes = around.iter().filter(|&&c| c == candidate).count();
                // More votes first, then the smaller tone distance, then the lower class.
                let key = (4 - votes, candidate.abs_diff(own), candidate);
                if key < best_key {
                    best = candidate;
                    best_key = key;
                }
            }
            best
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ramp(size: usize) -> Grid<f64> {
        // Distinct values in a scrambled order.
        Grid::from_fn(size, |x, y| {
            ((x * 7 + y * 13) % (size * size)) as f64 + 0.25 * x as f64
        })
    }

    #[test]
    fn indexing_wraps_in_both_directions() {
        let grid = Grid::from_fn(4, |x, y| (x + 10 * y) as i32);
        assert_eq!(grid.get(3, 2), 23);
        assert_eq!(grid.wrapped(-1, 0), 3);
        assert_eq!(grid.wrapped(4, 5), 10);
        assert_eq!(grid.wrapped(-5, -1), 33);
        assert_eq!(grid.near(0, 0, -1, -1), 33);
        let mut grid = grid;
        grid.set_wrapped(-1, 4, 99);
        assert_eq!(grid.get(3, 0), 99);
        assert_eq!(grid.size(), 4);
        assert_eq!(grid.cells().len(), 16);
    }

    #[test]
    fn normalizing_spans_the_unit_interval() {
        let grid = ramp(8).normalized();
        assert_eq!(grid.min_max(), (0.0, 1.0));
        assert!(
            Grid::filled(4, 3.5)
                .normalized()
                .cells()
                .iter()
                .all(|&v| v == 0.5)
        );
    }

    #[test]
    fn blur_keeps_the_mean_and_flattens() {
        let grid = ramp(8);
        let blurred = grid.blurred();
        let mean = |g: &Grid<f64>| g.cells().iter().sum::<f64>() / 64.0;
        assert!((mean(&grid) - mean(&blurred)).abs() < 1.0e-9);
        let (lo, hi) = grid.min_max();
        let (blo, bhi) = blurred.min_max();
        assert!(blo > lo && bhi < hi);
        assert_eq!(Grid::filled(4, 2.0).blurred(), Grid::filled(4, 2.0));
    }

    #[test]
    fn wide_blurs_keep_the_mean_and_generalise_the_binomial_one() {
        let grid = ramp(16);
        let mean = |g: &Grid<f64>| g.cells().iter().sum::<f64>() / 256.0;
        // A tent of reach 2 is the 1-2-1 blur, computed separably.
        let tent = grid.tent_blurred(2);
        for (a, b) in tent.cells().iter().zip(grid.blurred().cells()) {
            assert!((a - b).abs() < 1.0e-9, "{a} {b}");
        }
        assert_eq!(grid.box_blurred(1), grid);
        for blurred in [grid.tent_blurred(5), grid.box_blurred(4)] {
            assert!((mean(&blurred) - mean(&grid)).abs() < 1.0e-9);
            let (lo, hi) = blurred.min_max();
            assert!(lo > grid.min_max().0 && hi < grid.min_max().1);
        }
        // A box of 2 * 3 - 1 cells turns a step into a ramp of that many cells.
        let step = Grid::from_fn(16, |x, _| if x >= 8 { 1.0 } else { 0.0 });
        let smooth = step.box_blurred(3);
        let row: Vec<f64> = (4..12).map(|x| smooth.get(x, 5)).collect();
        let expected = [0.0, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.0];
        for (got, want) in row.iter().zip(expected) {
            assert!((got - want).abs() < 1.0e-12, "{row:?}");
        }
        // Both stay shift-invariant on the torus.
        let roll = |g: &Grid<f64>| Grid::from_fn(16, |x, y| g.wrapped(x as i64 + 5, y as i64 - 3));
        assert_eq!(roll(&grid).tent_blurred(6), roll(&grid.tent_blurred(6)));
        assert_eq!(roll(&grid).box_blurred(3), roll(&grid.box_blurred(3)));
    }

    #[test]
    fn rank_thresholds_select_exact_shares() {
        let grid = ramp(8);
        for (fraction, expected) in [(0.0, 0), (0.25, 16), (0.5, 32), (0.9, 58), (1.0, 64)] {
            let threshold = grid.threshold_above(fraction);
            let count = grid.cells().iter().filter(|&&v| v >= threshold).count();
            assert_eq!(count, expected, "fraction {fraction}");
        }
    }

    #[test]
    fn quantizing_follows_the_shares() {
        let grid = ramp(8);
        let classes = grid.quantize(&[1.0, 2.0, 1.0]);
        let count = |class: u8| classes.cells().iter().filter(|&&c| c == class).count();
        assert_eq!((count(0), count(1), count(2)), (16, 32, 16));
        // Classes are ordered by value.
        for (value, class) in grid.cells().iter().zip(classes.cells()) {
            for (other, other_class) in grid.cells().iter().zip(classes.cells()) {
                assert!(value <= other || class >= other_class);
            }
        }
        // A share of zero leaves its class empty.
        let classes = grid.quantize(&[0.0, 1.0]);
        assert!(classes.cells().iter().all(|&c| c == 1));
    }

    #[test]
    fn despeckling_removes_only_stray_cells() {
        let mut grid = Grid::filled(6, 2_u8);
        grid.set_wrapped(2, 2, 5); // stray
        grid.set_wrapped(4, 4, 7); // pair, stays
        grid.set_wrapped(5, 4, 7);
        grid.set_wrapped(0, 0, 9); // stray, but protected
        let cleaned = grid.despeckled(|class| class == 9);
        assert_eq!(cleaned.get(2, 2), 2);
        assert_eq!((cleaned.get(4, 4), cleaned.get(5, 4)), (7, 7));
        assert_eq!(cleaned.get(0, 0), 9);

        // Ties: two neighbours each of classes 1 and 4 around a stray 3 -> the closer class 4.
        let mut grid = Grid::filled(5, 1_u8);
        grid.set_wrapped(2, 2, 3);
        grid.set_wrapped(1, 2, 4);
        grid.set_wrapped(0, 2, 4);
        grid.set_wrapped(2, 1, 4);
        grid.set_wrapped(2, 0, 4);
        assert_eq!(grid.despeckled(|_| false).get(2, 2), 4);
    }

    #[test]
    fn operations_commute_with_cyclic_shifts() {
        let grid = ramp(8);
        let shifted = Grid::from_fn(8, |x, y| grid.wrapped(x as i64 + 3, y as i64 + 6));
        let roll = |g: &Grid<f64>| Grid::from_fn(8, |x, y| g.wrapped(x as i64 + 3, y as i64 + 6));
        assert_eq!(shifted.blurred(), roll(&grid.blurred()));
        assert_eq!(shifted.normalized(), roll(&grid.normalized()));
        assert_eq!(shifted.threshold_above(0.3), grid.threshold_above(0.3));
        let classes = grid.quantize(&[1.0, 1.0, 2.0]);
        let rolled = Grid::from_fn(8, |x, y| classes.wrapped(x as i64 + 3, y as i64 + 6));
        assert_eq!(shifted.quantize(&[1.0, 1.0, 2.0]), rolled);
        assert_eq!(
            shifted.quantize(&[1.0, 1.0, 2.0]).despeckled(|_| false),
            Grid::from_fn(8, |x, y| classes
                .despeckled(|_| false)
                .wrapped(x as i64 + 3, y as i64 + 6))
        );
    }
}
