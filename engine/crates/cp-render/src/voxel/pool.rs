//! The range allocator of the quad pool (contract §5.3 "Quad pool").
//!
//! The pool is one storage buffer of packed quads, created once at start-up and never
//! reallocated. It is divided into units of [`POOL_UNIT_QUADS`] quads; every chunk owns one
//! contiguous range of whole units that holds all its layers. The allocator is best fit with
//! the lowest address among equally good ranges, so its decisions depend only on the sequence
//! of requests, and freed ranges merge with free neighbours at once.

use std::collections::{BTreeMap, BTreeSet};

use super::layout::POOL_UNIT_QUADS;

/// A range of pool units, `start .. start + len`; quad index `start * 256` is its first quad.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct PoolRange {
    /// First unit.
    pub start: u32,
    /// Units, at least 1.
    pub len: u32,
}

impl PoolRange {
    /// One past the last unit.
    #[inline]
    #[must_use]
    pub const fn end(self) -> u32 {
        self.start + self.len
    }

    /// Pool index of the first quad.
    #[inline]
    #[must_use]
    pub const fn first_quad(self) -> u32 {
        self.start * POOL_UNIT_QUADS
    }

    /// Quads the range can hold.
    #[inline]
    #[must_use]
    pub const fn quad_capacity(self) -> u64 {
        self.len as u64 * POOL_UNIT_QUADS as u64
    }
}

/// Units needed for `quads` quads (`quads > 0`).
#[inline]
#[must_use]
pub const fn units_for(quads: u64) -> u64 {
    quads.div_ceil(POOL_UNIT_QUADS as u64)
}

/// Best-fit allocator of unit ranges with immediate coalescing.
#[derive(Clone, Debug)]
pub struct RangeAllocator {
    capacity: u32,
    /// Free ranges by start.
    by_start: BTreeMap<u32, u32>,
    /// Free ranges as `(len, start)`, for best fit.
    by_size: BTreeSet<(u32, u32)>,
    free: u32,
}

impl RangeAllocator {
    /// An allocator over `capacity` units, all free.
    #[must_use]
    pub fn new(capacity: u32) -> Self {
        let mut allocator = Self {
            capacity,
            by_start: BTreeMap::new(),
            by_size: BTreeSet::new(),
            free: 0,
        };
        if capacity > 0 {
            allocator.insert_free(0, capacity);
        }
        allocator
    }

    /// Units in the pool.
    #[must_use]
    pub fn capacity(&self) -> u32 {
        self.capacity
    }

    /// Units not allocated.
    #[must_use]
    pub fn free_units(&self) -> u32 {
        self.free
    }

    /// The largest range that can be allocated now, in units.
    #[must_use]
    pub fn largest_free(&self) -> u32 {
        self.by_size.last().map_or(0, |&(len, _)| len)
    }

    /// Number of separate free ranges (1 for an empty pool).
    #[must_use]
    pub fn free_ranges(&self) -> usize {
        self.by_start.len()
    }

    /// The smallest free range of at least `len` units (`len >= 1`), lowest start first, or
    /// `None` when no free range is large enough.
    pub fn allocate(&mut self, len: u32) -> Option<PoolRange> {
        if len == 0 {
            return None;
        }
        let &(free_len, start) = self.by_size.range((len, 0)..).next()?;
        self.remove_free(start, free_len);
        if free_len > len {
            self.insert_free(start + len, free_len - len);
        }
        Some(PoolRange { start, len })
    }

    /// Returns `range` to the pool, merging it with free neighbours.
    ///
    /// # Panics
    /// In debug builds, when `range` overlaps a free range or leaves the pool: a double free
    /// is a bug of the caller.
    pub fn free(&mut self, range: PoolRange) {
        debug_assert!(range.len > 0 && range.end() <= self.capacity, "{range:?}");
        let mut start = range.start;
        let mut len = range.len;
        if let Some((&before, &before_len)) = self.by_start.range(..start).next_back() {
            debug_assert!(before + before_len <= start, "double free of {range:?}");
            if before + before_len == start {
                self.remove_free(before, before_len);
                start = before;
                len += before_len;
            }
        }
        if let Some(&after_len) = self.by_start.get(&range.end()) {
            self.remove_free(range.end(), after_len);
            len += after_len;
        }
        debug_assert!(
            self.by_start.range(start..start + len).next().is_none(),
            "double free of {range:?}"
        );
        self.insert_free(start, len);
    }

    fn insert_free(&mut self, start: u32, len: u32) {
        self.by_start.insert(start, len);
        self.by_size.insert((len, start));
        self.free += len;
    }

    fn remove_free(&mut self, start: u32, len: u32) {
        self.by_start.remove(&start);
        self.by_size.remove(&(len, start));
        self.free -= len;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The obviously correct model: one flag per unit.
    struct Reference {
        used: Vec<bool>,
    }

    impl Reference {
        /// The best-fit choice by scanning every free run.
        fn best_fit(&self, len: u32) -> Option<u32> {
            let mut best: Option<(u32, u32)> = None;
            let mut unit = 0;
            while unit < self.used.len() {
                if self.used[unit] {
                    unit += 1;
                    continue;
                }
                let start = unit;
                while unit < self.used.len() && !self.used[unit] {
                    unit += 1;
                }
                let run = (unit - start) as u32;
                if run >= len && best.is_none_or(|(best_len, _)| run < best_len) {
                    best = Some((run, start as u32));
                }
            }
            best.map(|(_, start)| start)
        }

        fn set(&mut self, range: PoolRange, used: bool) {
            for unit in range.start..range.end() {
                assert_ne!(self.used[unit as usize], used, "unit {unit}");
                self.used[unit as usize] = used;
            }
        }
    }

    #[test]
    fn allocations_follow_best_fit_and_coalesce_like_the_reference() {
        let capacity = 512;
        let mut allocator = RangeAllocator::new(capacity);
        let mut reference = Reference {
            used: vec![false; capacity as usize],
        };
        let mut live: Vec<PoolRange> = Vec::new();
        let mut state = 0x2545_F491_u32;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            state
        };
        for step in 0..20_000 {
            let roll = next();
            if roll % 3 != 0 || live.is_empty() {
                let len = 1 + next() % 24;
                let expected = reference.best_fit(len);
                let got = allocator.allocate(len);
                assert_eq!(got.map(|range| range.start), expected, "step {step}");
                if let Some(range) = got {
                    assert_eq!(range.len, len);
                    reference.set(range, true);
                    live.push(range);
                }
            } else {
                let range = live.swap_remove(next() as usize % live.len());
                allocator.free(range);
                reference.set(range, false);
            }
            let free = reference.used.iter().filter(|&&used| !used).count();
            assert_eq!(allocator.free_units() as usize, free, "step {step}");
            // Coalescing: the allocator's free ranges are exactly the maximal free runs.
            let runs = reference
                .used
                .iter()
                .enumerate()
                .filter(|&(unit, &used)| !used && (unit == 0 || reference.used[unit - 1]))
                .count();
            assert_eq!(allocator.free_ranges(), runs, "step {step}");
        }
        for range in live.drain(..) {
            allocator.free(range);
        }
        assert_eq!(allocator.free_units(), capacity);
        assert_eq!(allocator.free_ranges(), 1);
        assert_eq!(allocator.largest_free(), capacity);
    }

    #[test]
    fn exhaustion_and_edge_cases() {
        let mut allocator = RangeAllocator::new(10);
        assert_eq!(allocator.allocate(0), None);
        assert_eq!(allocator.allocate(11), None);
        let a = allocator.allocate(4).unwrap();
        let b = allocator.allocate(6).unwrap();
        assert_eq!((a.start, b.start), (0, 4));
        assert_eq!(allocator.free_units(), 0);
        assert_eq!(allocator.allocate(1), None);
        allocator.free(a);
        // Best fit prefers the exact hole over a larger one.
        let c = allocator.allocate(4).unwrap();
        assert_eq!(c, a);
        allocator.free(b);
        allocator.free(c);
        assert_eq!(allocator.largest_free(), 10);
        assert_eq!(RangeAllocator::new(0).allocate(1), None);

        assert_eq!(units_for(1), 1);
        assert_eq!(units_for(256), 1);
        assert_eq!(units_for(257), 2);
        let range = PoolRange { start: 3, len: 2 };
        assert_eq!(range.first_quad(), 768);
        assert_eq!(range.quad_capacity(), 512);
        assert_eq!(range.end(), 5);
    }
}
