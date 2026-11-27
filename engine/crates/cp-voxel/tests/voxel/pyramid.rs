//! The edit pyramid against a from-scratch recomputation; reverting; receipts.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;
use std::time::Duration;

use cp_core::coords::{CHUNK_VOLUME, LodKey};
use cp_core::face::Face;
use cp_core::ids::BlockStateId;
use cp_core::math::IVec3;
use cp_voxel::{
    BaseSource, CellEdit, EditReceipt, NodeClass, VoxelChunk, VoxelSnapshot, VoxelTables,
    VoxelWorld, WorldConfig, downsample_cell,
};

use crate::common::{
    AIR, CountingBase, ExactClassify, FLOWER, GLASS, HillsBase, LEAVES, LOG, PLANK, STONE, WATER,
    base_cell, base_chunk, global_cell, range_i32, rng, tables,
};

const TOP_LEVEL: u8 = 5;

type Cell = [i32; 3];

/// Overrides and authored cells of one level, keyed by global cell of that level.
#[derive(Debug, Default, PartialEq, Eq)]
struct Level {
    overrides: BTreeMap<Cell, BlockStateId>,
    authored: BTreeSet<Cell>,
}

/// The pyramid computed from scratch from the level-0 overrides, cell by cell, straight from
/// the contract's definition.
fn reference_pyramid(
    level0: &BTreeMap<Cell, BlockStateId>,
    base: &dyn BaseSource,
    tables: &VoxelTables,
) -> Vec<Level> {
    let mut levels = vec![Level {
        overrides: level0.clone(),
        authored: level0.keys().copied().collect(),
    }];
    for level in 0..TOP_LEVEL {
        let below = &levels[usize::from(level)];
        let parents: BTreeSet<Cell> = below
            .overrides
            .keys()
            .chain(below.authored.iter())
            .map(|&cell| (IVec3::from_array(cell) >> 1_u32).to_array())
            .collect();
        let mut next = Level::default();
        for parent in parents {
            let parent_vec = IVec3::from_array(parent);
            let mut children = [AIR; 8];
            let mut authored = 0_u8;
            let mut any_override = false;
            for (child, slot) in children.iter_mut().enumerate() {
                let cell = (parent_vec * 2 + LodKey::octant_offset(child)).to_array();
                *slot = match below.overrides.get(&cell) {
                    Some(&state) => {
                        any_override = true;
                        state
                    }
                    None => base_cell(base, level, IVec3::from_array(cell)),
                };
                if below.authored.contains(&cell) {
                    authored |= 1 << child;
                }
            }
            if authored != 0 {
                next.authored.insert(parent);
            }
            if any_override {
                let down =
                    downsample_cell(&children, authored, level + 1, &tables.flags, &tables.lod);
                if down.state != base_cell(base, level + 1, parent_vec) {
                    next.overrides.insert(parent, down.state);
                }
            }
        }
        levels.push(next);
    }
    levels
}

/// The overrides the world stores, in the same form.
fn stored_pyramid(snapshot: &VoxelSnapshot) -> Vec<Level> {
    let mut levels: Vec<Level> = (0..=TOP_LEVEL).map(|_| Level::default()).collect();
    for (key, layer) in snapshot.overrides() {
        let level = &mut levels[usize::from(key.level)];
        let mut empty = true;
        for index in 0..CHUNK_VOLUME {
            let cell = global_cell(key, index).to_array();
            let state = layer.get(index);
            if !state.is_unchanged() {
                level.overrides.insert(cell, state);
                empty = false;
            }
            if layer.is_authored(index) {
                level.authored.insert(cell);
                empty = false;
            }
        }
        assert!(!empty, "an empty layer is kept for {key:?}");
    }
    levels
}

/// Random clustered and scattered edits over a region with negative coordinates.
fn random_batch(
    rng: &mut cp_core::hash::Rng,
    model: &mut BTreeMap<Cell, BlockStateId>,
) -> Vec<CellEdit> {
    let states = [
        AIR,
        STONE,
        PLANK,
        LOG,
        LEAVES,
        WATER,
        GLASS,
        FLOWER,
        BlockStateId::UNCHANGED,
    ];
    let mut edits = Vec::new();
    for _ in 0..3 {
        let min = IVec3::new(
            range_i32(rng, -70, 60),
            range_i32(rng, -12, 30),
            range_i32(rng, -70, 60),
        );
        let size = IVec3::new(
            range_i32(rng, 1, 11),
            range_i32(rng, 1, 11),
            range_i32(rng, 1, 11),
        );
        let state = states[rng.range_u32(0, states.len() as u32) as usize];
        for y in 0..size.y {
            for z in 0..size.z {
                for x in 0..size.x {
                    edits.push(CellEdit {
                        cell: min + IVec3::new(x, y, z),
                        state,
                    });
                }
            }
        }
    }
    for _ in 0..60 {
        let cell = IVec3::new(
            range_i32(rng, -80, 80),
            range_i32(rng, -16, 40),
            range_i32(rng, -80, 80),
        );
        let state = states[rng.range_u32(0, states.len() as u32) as usize];
        edits.push(CellEdit { cell, state });
    }
    // Revert some previously edited cells.
    let edited: Vec<Cell> = model.keys().copied().collect();
    for _ in 0..40.min(edited.len()) {
        let cell = edited[rng.range_u32(0, edited.len() as u32) as usize];
        edits.push(CellEdit {
            cell: IVec3::from_array(cell),
            state: BlockStateId::UNCHANGED,
        });
    }
    for edit in &edits {
        if edit.state.is_unchanged() {
            model.remove(&edit.cell.to_array());
        } else {
            model.insert(edit.cell.to_array(), edit.state);
        }
    }
    edits
}

fn world() -> VoxelWorld {
    world_on(Arc::new(HillsBase::new()))
}

fn world_on(base: Arc<dyn BaseSource>) -> VoxelWorld {
    let config = WorldConfig {
        top_level: TOP_LEVEL,
        ..WorldConfig::default()
    };
    VoxelWorld::new(base, tables(), config)
}

/// Every key the receipt must list: chunks whose composite or authored mask changed, and
/// neighbours whose face halo changed, checked by comparing two snapshots.
fn check_receipt(before: &VoxelSnapshot, after: &VoxelSnapshot, receipt: &EditReceipt) {
    let dirty: BTreeSet<LodKey> = receipt.dirty.iter().copied().collect();
    assert_eq!(dirty.len(), receipt.dirty.len(), "dirty keys are unique");
    assert!(
        receipt.dirty.windows(2).all(|pair| pair[0] < pair[1]),
        "sorted"
    );
    // A padded chunk depends only on the base and the layers of the chunk and its face
    // neighbours, so the keys that can have changed are the ones whose layer changed (not the
    // same `Arc` any more) and their face neighbours.
    let layers = |snapshot: &VoxelSnapshot| -> BTreeMap<LodKey, Arc<VoxelChunk>> {
        snapshot
            .overrides()
            .into_iter()
            .map(|(key, layer)| (key, Arc::clone(layer)))
            .collect()
    };
    let (old_layers, new_layers) = (layers(before), layers(after));
    let mut candidates: BTreeSet<LodKey> = BTreeSet::new();
    for key in old_layers.keys().chain(new_layers.keys()) {
        let same = match (old_layers.get(key), new_layers.get(key)) {
            (Some(old), Some(new)) => Arc::ptr_eq(old, new),
            _ => false,
        };
        if !same {
            candidates.insert(*key);
            for face in Face::ALL {
                candidates.insert(LodKey::new(key.level, key.pos + face.normal()));
            }
        }
    }
    // Every dirty key must be a candidate: nothing else can have changed.
    for key in &dirty {
        assert!(
            candidates.contains(key),
            "{key:?} dirty without a changed layer nearby"
        );
    }
    for &key in &candidates {
        let old = before.sample_padded(key);
        let new = after.sample_padded(key);
        let mut changed = old.authored() != new.authored();
        for y in -1..=32 {
            for z in -1..=32 {
                for x in -1..=32 {
                    let outside = [x, y, z].iter().filter(|&&c| c == -1 || c == 32).count();
                    if outside <= 1 && old.get(x, y, z) != new.get(x, y, z) {
                        changed = true;
                    }
                }
            }
        }
        if changed {
            assert!(dirty.contains(&key), "{key:?} changed but is not dirty");
        } else if key.level > 0 {
            // Exact on pyramid levels: an authored bit alone never dirties a neighbour, and a
            // pyramid override differs from the base by construction. (Level 0 may list the
            // neighbours of an override that equals its base cell.)
            assert!(!dirty.contains(&key), "{key:?} dirty without a change");
        }
        // Revisions change exactly for the dirty keys.
        if dirty.contains(&key) {
            assert_eq!(after.revision(key), receipt.revision);
        } else {
            assert_eq!(after.revision(key), before.revision(key), "{key:?}");
        }
    }
    // Class changes: exactly the keys whose classification differs.
    let reported: BTreeMap<LodKey, _> = receipt
        .class_changes
        .iter()
        .map(|change| (change.key, (change.before, change.after)))
        .collect();
    assert!(
        receipt
            .class_changes
            .windows(2)
            .all(|pair| pair[0].key < pair[1].key)
    );
    for &key in &candidates {
        let classes = (before.classify(key), after.classify(key));
        if classes.0 == classes.1 {
            assert!(
                !reported.contains_key(&key),
                "{key:?} reported without change"
            );
        } else {
            assert_eq!(reported.get(&key), Some(&classes), "{key:?}");
        }
    }
}

#[test]
fn pyramid_after_random_edits_equals_a_recomputation() {
    let mut world = world();
    let tables = tables();
    let mut model: BTreeMap<Cell, BlockStateId> = BTreeMap::new();
    let mut rng = rng(3);
    for batch in 0..24 {
        let before = world.snapshot();
        let edits = random_batch(&mut rng, &mut model);
        let receipt = world.apply(&edits);
        assert_eq!(receipt.rejected, 0);
        let after = world.snapshot();
        if batch % 12 == 0 {
            check_receipt(&before, &after, &receipt);
        }
        if batch % 8 == 7 {
            let reference = reference_pyramid(&model, world.base().as_ref(), &tables);
            assert_eq!(stored_pyramid(&after), reference, "after batch {batch}");
        }
    }
    let snapshot = world.snapshot();
    let reference = reference_pyramid(&model, world.base().as_ref(), &tables);
    let stored = stored_pyramid(&snapshot);
    for level in 0..=usize::from(TOP_LEVEL) {
        assert_eq!(stored[level], reference[level], "level {level}");
    }
    assert!(
        !reference[usize::from(TOP_LEVEL)].authored.is_empty(),
        "edits reach the top level"
    );
    assert!(
        !reference[3].overrides.is_empty(),
        "the pyramid carries state overrides at level 3"
    );
    // Composites at every level are the overrides over the base.
    for (key, _) in snapshot.overrides().into_iter().take(40) {
        let sampled = snapshot.sample(key);
        let base = base_chunk(world.base().as_ref(), key);
        let level = &reference[usize::from(key.level)];
        for index in (0..CHUNK_VOLUME).step_by(7) {
            let cell = global_cell(key, index).to_array();
            let expected = level
                .overrides
                .get(&cell)
                .copied()
                .unwrap_or_else(|| base.get(index));
            assert_eq!(sampled.get(index), expected);
            assert_eq!(sampled.is_authored(index), level.authored.contains(&cell));
        }
    }
}

#[test]
fn reverting_every_edit_restores_the_base() {
    let mut world = world();
    let mut model: BTreeMap<Cell, BlockStateId> = BTreeMap::new();
    let mut rng = rng(4);
    let mut touched: BTreeSet<LodKey> = BTreeSet::new();
    for _ in 0..10 {
        let edits = random_batch(&mut rng, &mut model);
        let receipt = world.apply(&edits);
        touched.extend(receipt.dirty.iter().copied());
    }
    assert!(!world.edits().is_empty());
    let before_revert = world.snapshot();
    // Revert in two batches, in an order unrelated to the edit order.
    let cells: Vec<Cell> = model.keys().rev().copied().collect();
    let (first, second) = cells.split_at(cells.len() / 2);
    for half in [first, second] {
        let edits: Vec<CellEdit> = half
            .iter()
            .map(|&cell| CellEdit {
                cell: IVec3::from_array(cell),
                state: BlockStateId::UNCHANGED,
            })
            .collect();
        world.apply(&edits);
    }
    assert!(world.edits().is_empty(), "all override layers are gone");
    assert_eq!(world.edits().layer_count(), 0);
    let snapshot = world.snapshot();
    assert!(snapshot.overrides().is_empty());
    for &key in &touched {
        let sampled = snapshot.sample(key);
        assert_eq!(*sampled, base_chunk(world.base().as_ref(), key), "{key:?}");
        assert!(!sampled.has_authored());
        assert_eq!(snapshot.classify(key), world.base().classify(key));
        // A key that ever changed keeps a non-zero revision.
        assert!(snapshot.revision(key) > 0);
    }
    // The snapshot taken before the revert still has every override.
    assert!(!before_revert.overrides().is_empty());
}

#[test]
fn receipts_list_halo_neighbours_and_pyramid_chunks() {
    let mut world = world();
    let before = world.snapshot();
    // A cell at a chunk corner: three face neighbours see it in their halo.
    let cell = IVec3::new(-33, 31, 64);
    let receipt = world.apply(&[CellEdit { cell, state: PLANK }]);
    let after = world.snapshot();
    let own = LodKey::lod0(IVec3::new(-2, 0, 2));
    for key in [
        own,
        LodKey::lod0(IVec3::new(-1, 0, 2)),
        LodKey::lod0(IVec3::new(-2, 1, 2)),
        LodKey::lod0(IVec3::new(-2, 0, 1)),
    ] {
        assert!(receipt.dirty.contains(&key), "{key:?}");
    }
    // Edge and corner neighbours are not dirty (their halo cells are unspecified).
    assert!(!receipt.dirty.contains(&LodKey::lod0(IVec3::new(-1, 1, 2))));
    // Every pyramid level has the ancestor chunk dirty (at least the authored bit changed).
    for level in 1..=TOP_LEVEL {
        let ancestor = own.ancestor_at(level).expect("valid level");
        assert!(receipt.dirty.contains(&ancestor), "level {level}");
        assert!(after.override_layer(ancestor).is_some());
    }
    assert_eq!(receipt.changed_cells, 1);
    check_receipt(&before, &after, &receipt);
    // Re-applying the same edit changes nothing.
    let again = world.apply(&[CellEdit { cell, state: PLANK }]);
    assert!(again.dirty.is_empty() && again.changed_cells == 0);
    // Out-of-range cells and unknown states are rejected, not applied.
    let rejected = world.apply(&[
        CellEdit {
            cell: IVec3::new(1 << 30, 0, 0),
            state: STONE,
        },
        CellEdit {
            cell: IVec3::ZERO,
            state: BlockStateId(9999),
        },
    ]);
    assert_eq!(rejected.rejected, 2);
    assert!(rejected.dirty.is_empty());
    assert_eq!(
        world.snapshot().cell_state(IVec3::ZERO),
        HillsBase::cell(0, IVec3::ZERO)
    );
}

#[test]
fn receipts_are_exact_over_a_base_with_exact_classes() {
    // A base that answers Empty and Opaque wherever true, so that the class bookkeeping has
    // to get every transition near the surface right (not just "Mixed stays Mixed").
    let mut world = world_on(Arc::new(ExactClassify(HillsBase::new())));
    let mut model: BTreeMap<Cell, BlockStateId> = BTreeMap::new();
    let mut rng = rng(13);
    let mut class_changes = 0;
    for _ in 0..6 {
        let before = world.snapshot();
        let edits = random_batch(&mut rng, &mut model);
        let receipt = world.apply(&edits);
        class_changes += receipt.class_changes.len();
        check_receipt(&before, &world.snapshot(), &receipt);
    }
    assert!(
        class_changes > 20,
        "the edits change classes ({class_changes})"
    );
}

#[test]
fn class_changes_ask_the_base_once_per_chunk_and_never_for_authored_bits() {
    let base = Arc::new(CountingBase::new(HillsBase::new(), Duration::ZERO));
    let config = WorldConfig {
        top_level: 10,
        ..WorldConfig::default()
    };
    let mut world = VoxelWorld::new(base.clone(), tables(), config);
    let ancestors = |cell: IVec3| -> Vec<LodKey> {
        (0..=10)
            .map(|level| LodKey::containing(level, cell))
            .collect()
    };
    // A block in the sky, at local x = z = 0 on every level. An authored solid keeps its
    // parent solid up to level 2, so levels 0-2 gain a state override; levels 3-10 gain the
    // authored bit only.
    let cell = IVec3::new(0, 5000, 0);
    let keys = ancestors(cell);
    let before = world.snapshot();
    let receipt = world.apply(&[CellEdit { cell, state: PLANK }]);
    let after = world.snapshot();
    // Probes around the block see air only, which cannot tell Empty from Mixed: one classify
    // on each of levels 0-2. The authored-only ancestors ask nothing.
    assert_eq!(base.classifies(), 3);
    let changes: Vec<(LodKey, NodeClass, NodeClass)> = receipt
        .class_changes
        .iter()
        .map(|change| (change.key, change.before, change.after))
        .collect();
    let expected: Vec<(LodKey, NodeClass, NodeClass)> = keys[..3]
        .iter()
        .map(|&key| (key, NodeClass::Empty, NodeClass::Mixed))
        .collect();
    assert_eq!(changes, expected);
    // The halo neighbours across -X and -Z see the new state on levels 0-2; on levels 3-10
    // only the authored bit changed, which no halo carries.
    let mut dirty = BTreeSet::new();
    for (level, &key) in keys.iter().enumerate() {
        dirty.insert(key);
        if level <= 2 {
            dirty.insert(LodKey::new(key.level, key.pos - IVec3::X));
            dirty.insert(LodKey::new(key.level, key.pos - IVec3::Z));
        }
    }
    assert_eq!(receipt.dirty, dirty.into_iter().collect::<Vec<_>>());
    let level5_neighbour = LodKey::new(5, keys[5].pos - IVec3::X);
    assert_eq!(after.revision(level5_neighbour), 0);

    // More edits in the same chunks, then removing everything: the remembered classes answer.
    world.apply(&[CellEdit {
        cell: cell + IVec3::Y,
        state: PLANK,
    }]);
    let removal = world.apply(&[
        CellEdit {
            cell,
            state: BlockStateId::UNCHANGED,
        },
        CellEdit {
            cell: cell + IVec3::Y,
            state: BlockStateId::UNCHANGED,
        },
    ]);
    assert_eq!(base.classifies(), 3, "no chunk is asked twice");
    assert!(world.edits().is_empty());
    let back: Vec<(LodKey, NodeClass, NodeClass)> = removal
        .class_changes
        .iter()
        .map(|change| (change.key, change.before, change.after))
        .collect();
    let expected: Vec<(LodKey, NodeClass, NodeClass)> = keys[..3]
        .iter()
        .map(|&key| (key, NodeClass::Mixed, NodeClass::Empty))
        .collect();
    assert_eq!(back, expected);
    // Snapshots use the remembered classes as well: only levels 3-10 still ask the base.
    // (From level 8 up the ancestor chunks reach down to the terrain and are Mixed.)
    let snapshot = world.snapshot();
    for &key in &keys {
        assert_eq!(snapshot.classify(key), HillsBase::new().classify(key));
    }
    assert_eq!(base.classifies(), 3 + 8);
    // The first receipt against the snapshots (its classify calls are not counted above).
    check_receipt(&before, &after, &receipt);
}

#[test]
fn surface_edits_in_fresh_areas_are_classified_by_a_few_base_cells() {
    // classify of this base evaluates every column of the chunk, like cp-worldgen's hills.
    let base = Arc::new(CountingBase::new(
        HillsBase::column_classified(),
        Duration::ZERO,
    ));
    let config = WorldConfig {
        top_level: 10,
        ..WorldConfig::default()
    };
    let mut world = VoxelWorld::new(base.clone(), tables(), config);
    let mut rng = rng(17);
    let edits = 60;
    let mut apply_classifies = 0;
    for i in 0..edits {
        let x = range_i32(&mut rng, -4000, 4000);
        let z = range_i32(&mut rng, -4000, 4000);
        let top = HillsBase::ground_top(x, z);
        let (cell, state) = if i % 2 == 0 {
            (IVec3::new(x, top, z), AIR)
        } else {
            (IVec3::new(x, top + 1, z), PLANK)
        };
        let before = world.snapshot();
        let calls = base.classifies();
        let receipt = world.apply(&[CellEdit { cell, state }]);
        apply_classifies += base.classifies() - calls;
        if i % 10 == 0 {
            check_receipt(&before, &world.snapshot(), &receipt);
        }
    }
    // Before the class memo and the probes, every such edit asked classify on about ten
    // levels (once per level whose chunk gained a layer, a state or an authored bit; 9.8
    // calls per edit in `bench_apply_class_bookkeeping`). Now only chunks whose probe box
    // sees one kind of cell ask, mostly where the surface runs along a chunk border: 5 calls
    // for these 60 edits when this was written.
    assert!(
        apply_classifies <= edits / 6,
        "{apply_classifies} classify calls for {edits} surface edits"
    );
}
