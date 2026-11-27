//! Features are pull-based (contract §5.1): a tree that crosses a chunk border looks the same
//! whichever neighbour is generated first, on whichever thread, and whatever region is asked
//! for.

use cp_core::coords::LodKey;
use cp_core::ids::BlockStateId;
use cp_core::math::{IAabb, IVec3};
use cp_voxel::{BaseSource, chunk_cells, region_index, region_volume};
use cp_worldgen::{FeatureKind, Species, TREE_MAX_RADIUS, TreeInfo};

use crate::common::{ALL_LEAVES, ALL_LOGS, chunk, hills};

/// A broad tree whose crown crosses the border between two level-0 chunks along x, near the
/// origin.
fn border_tree() -> TreeInfo {
    let source = hills(42);
    source
        .trees_in(IAabb::new(IVec3::splat(-512), IVec3::splat(512)))
        .into_iter()
        .find(|tree| {
            let local = tree.base.x.rem_euclid(32);
            let broad = matches!(tree.kind, FeatureKind::Tree(Species::Oak | Species::Cherry));
            // The crown (radius >= 2.5 m) reaches over the border at x = 32k.
            broad && tree.crown_radius_halves >= 5 && (local <= 1 || local >= 30)
        })
        .expect("no tree on a chunk border near the origin")
}

fn fill_region(source: &dyn BaseSource, level: u8, region: IAabb) -> Vec<BlockStateId> {
    let mut cells = vec![BlockStateId::UNCHANGED; region_volume(region)];
    source
        .fill(level, region, &mut cells, &|| false)
        .expect("never cancelled");
    cells
}

#[test]
fn trees_do_not_depend_on_the_generation_order() {
    let tree = border_tree();
    let side = if tree.base.x.rem_euclid(32) <= 1 {
        -1
    } else {
        1
    };
    for level in 0..=3_u8 {
        let a = LodKey::containing(level, tree.base);
        let b = LodKey::new(level, a.pos + IVec3::new(side, 0, 0));
        // Two fresh sources, opposite orders.
        let first = hills(42);
        let (a1, b1) = (chunk(&first, a), chunk(&first, b));
        let second = hills(42);
        let (b2, a2) = (chunk(&second, b), chunk(&second, a));
        assert_eq!(a1, a2, "level {level}");
        assert_eq!(b1, b2, "level {level}");
        // One fill of both chunks together equals the two separate fills.
        let (ca, cb) = (chunk_cells(a), chunk_cells(b));
        let union = IAabb::new(ca.min.min(cb.min), ca.max.max(cb.max));
        let joint = fill_region(&first, level, union);
        for (cells, own) in [(&a1, ca), (&b1, cb)] {
            for y in own.min.y..own.max.y {
                for z in own.min.z..own.max.z {
                    for x in own.min.x..own.max.x {
                        let cell = IVec3::new(x, y, z);
                        assert_eq!(
                            joint[region_index(union, cell)],
                            cells[region_index(own, cell)],
                            "level {level} {cell}"
                        );
                    }
                }
            }
        }
        if level == 0 {
            // The tree really is on both sides of the border.
            let leaves =
                |cells: &[BlockStateId]| cells.iter().filter(|s| ALL_LEAVES.contains(s)).count();
            assert!(leaves(&a1) > 0 && leaves(&b1) > 0);
            assert!(a1.iter().any(|s| ALL_LOGS.contains(s)));
        }
    }
}

#[test]
fn single_cells_equal_whole_chunks() {
    // The output of a cell never depends on the region that was asked for: fill single
    // columns and single cells around a border tree and compare with the chunk.
    let source = hills(42);
    let tree = border_tree();
    for level in 0..=2_u8 {
        let key = LodKey::containing(level, tree.base);
        let cells = chunk_cells(key);
        let whole = chunk(&source, key);
        let radius = TREE_MAX_RADIUS + 2;
        let around = IVec3::new(
            tree.base.x >> level,
            tree.base.y >> level,
            tree.base.z >> level,
        );
        for dz in -radius..=radius {
            for dx in -radius..=radius {
                let column = around + IVec3::new(dx, 0, dz);
                if !cells.contains(column.with_y(cells.min.y)) {
                    continue;
                }
                let region = IAabb::new(
                    column.with_y(cells.min.y),
                    column.with_y(cells.max.y) + IVec3::new(1, 0, 1),
                );
                let single = fill_region(&source, level, region);
                for (i, state) in single.iter().enumerate() {
                    let cell = column.with_y(cells.min.y + i as i32);
                    assert_eq!(
                        *state,
                        whole[region_index(cells, cell)],
                        "level {level} {cell}"
                    );
                }
            }
        }
    }
}

#[test]
fn threads_and_orders_give_the_same_chunks() {
    let source = hills(42);
    let centre = LodKey::containing(0, border_tree().base).pos;
    // The 3 × 3 chunks around the border tree.
    let keys: Vec<LodKey> = (0..9)
        .map(|i| LodKey::new(0, centre + IVec3::new(i % 3 - 1, 0, i / 3 - 1)))
        .collect();
    let reference: Vec<Vec<BlockStateId>> = keys.iter().map(|&key| chunk(&source, key)).collect();
    std::thread::scope(|scope| {
        let handles: Vec<_> = (0..4_usize)
            .map(|thread| {
                let (source, keys) = (&source, &keys);
                scope.spawn(move || {
                    // Each thread walks the keys in its own order: rotated, every other one
                    // reversed.
                    let count = keys.len();
                    let mut out = vec![Vec::new(); count];
                    for step in 0..count {
                        let rotated = (step + 2 * thread) % count;
                        let index = if thread % 2 == 1 {
                            count - 1 - rotated
                        } else {
                            rotated
                        };
                        out[index] = chunk(source, keys[index]);
                    }
                    out
                })
            })
            .collect();
        for handle in handles {
            let chunks = handle.join().unwrap();
            assert_eq!(chunks, reference);
        }
    });
}
