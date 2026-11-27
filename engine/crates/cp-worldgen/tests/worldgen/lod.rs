//! LOD consistency (contract §5.1): for keys of levels 1–3, `fill(parent)` against the parent
//! downsampled from its eight generated children by the normative rule
//! (`cp_voxel::downsample_chunk`).
//!
//! Measured per parent chunk:
//! - **surface offset**: per column, the difference between the highest ground cell of the
//!   generated and of the downsampled parent, in coarse cells (and, without trees, the same
//!   for the highest non-air cell, which includes water). Must be at most 1 everywhere.
//! - **differing cells**: the share of the 32³ cells whose state differs. Pinned below
//!   [`MAX_DIFFERING`] per chunk.
//! - **class mismatch**: the share whose LOD class (empty, fluid, solid) differs, i.e. the
//!   geometric part of the difference. Pinned below [`MAX_CLASS_MISMATCH`] per chunk and
//!   [`MAX_MEAN_CLASS_MISMATCH`] on average.
//!
//! Where the two differ and why: the downsample rule makes a parent solid with four solid
//! children of eight, which rounds the surface up by a quarter of a coarse cell against the
//! unbiased point rule, so about a quarter of the surface columns carry one extra cell; its
//! tie rule (smallest id) turns some top cells from grass into dirt where the generator keeps
//! the surface layer on top; and it keeps tree trunks that the canopy stand-in leaves out.
//! One full row of a chunk is 1/32 = 3.1 % of its cells; flat ground at an unlucky height
//! differs in two rows (one extra solid row, one row of tie materials), the worst case of
//! any height field.
//!
//! Measured when the limits were pinned (seed 42, 14 parents per level):
//!
//! | source | differing (worst) | class mismatch (worst / mean) |
//! |---|---|---|
//! | hills without trees, levels 1–3 | 3.9 % | 0.99 % / 0.55 % |
//! | hills with trees, levels 1–3 | 4.3 % | 1.7 % / 0.72 % |
//! | flat ground (surface at 6 m) | 6.25 % | 3.1 % / 0.69 % |
//! | structure field | 3.1 % | 0.01 % / 0.00 % |

use cp_core::block::LodClass;
use cp_core::coords::{LodKey, cell_index};
use cp_core::hash::Rng;
use cp_core::ids::BlockStateId;
use cp_core::math::IVec3;
use cp_voxel::BaseSource;
use cp_worldgen::HillsSource;

use crate::common::{
    chunk, class, downsampled, flat, hills, hills_without_trees, is_ground, structures,
};

/// Largest share of differing cells in one parent chunk (two full rows are 6.25 %).
const MAX_DIFFERING: f64 = 0.07;
/// Largest share of class mismatches in one parent chunk (one full row is 3.125 %).
const MAX_CLASS_MISMATCH: f64 = 0.035;
/// Largest mean share of class mismatches over the tested parents of one source.
const MAX_MEAN_CLASS_MISMATCH: f64 = 0.012;

#[derive(Clone, Copy, Debug, Default)]
struct Comparison {
    differing: f64,
    class_mismatch: f64,
    ground_offset: i32,
    top_offset: i32,
}

/// Highest cell of every column `(x, z)` that satisfies `keep`, or -1.
fn tops(cells: &[BlockStateId], keep: impl Fn(BlockStateId) -> bool) -> Vec<i32> {
    let mut tops = vec![-1; 32 * 32];
    for z in 0..32_u8 {
        for x in 0..32_u8 {
            for y in (0..32_u8).rev() {
                if keep(cells[cell_index(x, y, z)]) {
                    tops[usize::from(x) + 32 * usize::from(z)] = i32::from(y);
                    break;
                }
            }
        }
    }
    tops
}

fn max_offset(a: &[i32], b: &[i32]) -> i32 {
    a.iter()
        .zip(b)
        .map(|(a, b)| (a - b).abs())
        .max()
        .unwrap_or(0)
}

fn compare(source: &dyn BaseSource, key: LodKey) -> Comparison {
    let generated = chunk(source, key);
    let down = downsampled(source, key);
    let total = generated.len() as f64;
    let differing = generated.iter().zip(&down).filter(|(g, d)| g != d).count();
    let classes = generated
        .iter()
        .zip(&down)
        .filter(|&(&g, &d)| class(g) != class(d))
        .count();
    let ground = |state| is_ground(state);
    let filled = |state| class(state) != LodClass::Empty;
    Comparison {
        differing: differing as f64 / total,
        class_mismatch: classes as f64 / total,
        ground_offset: max_offset(&tops(&generated, ground), &tops(&down, ground)),
        top_offset: max_offset(&tops(&generated, filled), &tops(&down, filled)),
    }
}

/// Parent keys of levels 1–3 whose chunk contains the level-0 surface at a random column,
/// plus columns on a lake shore, a ridge and in a forest.
fn surface_keys(source: &HillsSource, per_level: usize) -> Vec<LodKey> {
    let mut rng = Rng::new(2026, 11);
    let mut columns: Vec<(i32, i32)> = (0..per_level)
        .map(|_| {
            (
                rng.range_u32(0, 16_384) as i32 - 8192,
                rng.range_u32(0, 16_384) as i32 - 8192,
            )
        })
        .collect();
    // Lake shore, coast, snowy ridge (seed 42; see the golden keys) and a forest.
    columns.extend([(384, -640), (256, 512), (128, 1024), (-40, -60)]);
    let mut keys = Vec::new();
    for level in 1..=3_u8 {
        for &(x, z) in &columns {
            let y = source.ground_top(x, z);
            keys.push(LodKey::containing(level, IVec3::new(x, y, z)));
        }
    }
    keys
}

/// Checks every key against the pinned limits and returns the worst values and the mean
/// class mismatch.
fn check(source: &dyn BaseSource, keys: &[LodKey], strict_top: bool) -> (Comparison, f64) {
    let mut worst = Comparison::default();
    let mut sum = 0.0;
    for &key in keys {
        let c = compare(source, key);
        let report = format!("{key:?}: {c:?}");
        assert!(c.ground_offset <= 1, "{report}");
        if strict_top {
            assert!(c.top_offset <= 1, "{report}");
        }
        assert!(c.differing <= MAX_DIFFERING, "{report}");
        assert!(c.class_mismatch <= MAX_CLASS_MISMATCH, "{report}");
        worst.differing = cp_core::det::max(worst.differing, c.differing);
        worst.class_mismatch = cp_core::det::max(worst.class_mismatch, c.class_mismatch);
        worst.ground_offset = worst.ground_offset.max(c.ground_offset);
        worst.top_offset = worst.top_offset.max(c.top_offset);
        sum += c.class_mismatch;
    }
    let mean = sum / keys.len() as f64;
    assert!(
        mean <= MAX_MEAN_CLASS_MISMATCH,
        "mean {mean}, worst {worst:?}"
    );
    (worst, mean)
}

#[test]
fn hills_coarse_levels_agree_with_downsampled_children() {
    let source = hills_without_trees(42);
    check(&source, &surface_keys(&source, 10), true);
}

#[test]
fn forests_agree_with_downsampled_children() {
    // Trees stand on the ground, so the ground surface is still within one cell; the top of
    // the canopy may differ more where the downsample keeps trunks or drops crown edges.
    let source = hills(42);
    check(&source, &surface_keys(&source, 10), false);
}

#[test]
fn flat_and_structures_agree_with_downsampled_children() {
    let keys: Vec<LodKey> = (1..=3)
        .flat_map(|level| {
            [IVec3::ZERO, IVec3::new(1, 0, 0), IVec3::new(0, -1, 1)]
                .map(|pos| LodKey::new(level, pos))
        })
        .collect();
    check(&flat(), &keys, true);
    check(&structures(), &keys, false);
}
