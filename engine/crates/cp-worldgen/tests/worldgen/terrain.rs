//! The look of the hills' ground on level 0 (beauty review E2, round 2): no single pits, humps
//! or notches of one cell and no small plateaus or hollows, which read from afar as dark
//! "shark fins" and dotted contour lines.

use cp_core::hash::Rng;

use crate::common::hills;

/// What [`blobs`] counted.
#[derive(Debug, Default)]
struct Blobs {
    /// Columns whose top at least three of the four neighbours overtop, or at least three
    /// undercut.
    isolated: usize,
    /// Columns of plateaus and hollows of at most four columns: connected columns of one top
    /// that their whole rim overtops or undercuts.
    small: usize,
    /// Columns looked at.
    columns: usize,
}

/// Counts isolated columns and small blobs in `side` × `side` squares at `count` random places.
fn blobs(seed: u64, count: usize, side: i32) -> Blobs {
    let source = hills(seed);
    let mut rng = Rng::new(seed, 17);
    let mut found = Blobs::default();
    // Two columns of margin, so that every blob inside is seen with its rim.
    let n = (side + 4) as usize;
    for _ in 0..count {
        let x0 = rng.range_u32(0, 16_000) as i32 - 8000;
        let z0 = rng.range_u32(0, 16_000) as i32 - 8000;
        let tops: Vec<i32> = (0..n * n)
            .map(|i| source.ground_top(x0 + (i % n) as i32 - 2, z0 + (i / n) as i32 - 2))
            .collect();
        let neighbours = |i: usize, j: usize| [(i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)];
        let mut seen = vec![false; n * n];
        for j in 2..n - 2 {
            for i in 2..n - 2 {
                let top = tops[j * n + i];
                let around = neighbours(i, j).map(|(x, z)| tops[z * n + x]);
                let higher = around.iter().filter(|&&t| t > top).count();
                let lower = around.iter().filter(|&&t| t < top).count();
                found.isolated += usize::from(higher >= 3 || lower >= 3);
                found.columns += 1;
                if seen[j * n + i] {
                    continue;
                }
                // The connected columns of this top, up to five of them.
                let mut blob = vec![(i, j)];
                let mut frontier = vec![(i, j)];
                seen[j * n + i] = true;
                while let Some((x, z)) = frontier.pop() {
                    for (a, b) in neighbours(x, z) {
                        if (1..n - 1).contains(&a)
                            && (1..n - 1).contains(&b)
                            && !seen[b * n + a]
                            && tops[b * n + a] == top
                            && blob.len() < 5
                        {
                            seen[b * n + a] = true;
                            blob.push((a, b));
                            frontier.push((a, b));
                        }
                    }
                }
                if blob.len() > 4 {
                    continue;
                }
                let rim: Vec<i32> = blob
                    .iter()
                    .flat_map(|&(x, z)| neighbours(x, z))
                    .filter(|cell| !blob.contains(cell))
                    .map(|(x, z)| tops[z * n + x])
                    .collect();
                if rim.iter().all(|&t| t > top) || rim.iter().all(|&t| t < top) {
                    found.small += blob.len();
                }
            }
        }
    }
    found
}

#[test]
fn the_ground_has_no_single_pits_humps_or_small_plateaus() {
    // 64 squares of one chunk each, on two seeds: below half a percent of the columns.
    for seed in [42, 0x00C0_FFEE_2026] {
        let found = blobs(seed, 64, 32);
        let share = |count: usize| count as f64 / found.columns as f64;
        assert!(share(found.isolated) < 0.005, "seed {seed:#x}: {found:?}");
        assert!(share(found.small) < 0.005, "seed {seed:#x}: {found:?}");
    }
}
