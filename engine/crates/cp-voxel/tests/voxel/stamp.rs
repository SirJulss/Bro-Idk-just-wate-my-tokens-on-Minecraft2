//! The persistent apron stamp of a chunk's override data (contract §6 mesh cache, §2.5).

use std::collections::BTreeSet;
use std::sync::Arc;

use cp_core::block::{BlockFlags, LodInfo};
use cp_core::coords::{LodKey, cell_index};
use cp_core::face::Face;
use cp_core::hash::hash_bytes;
use cp_core::ids::BlockStateId;
use cp_core::math::IVec3;
use cp_voxel::{
    APRON_STAMP_VERSION, BaseSource, CellEdit, UNCHANGED_KEY, VoxelSnapshot, VoxelTables,
    VoxelWorld, WorldConfig,
};

use crate::common::{
    AIR, FnBase, HillsBase, KEYS, PLANK, STATE_COUNT, STONE, key_of, range_i32, rng, tables,
};

/// Persistent key of a test state (never called with UNCHANGED by the stamp).
fn persistent(state: BlockStateId) -> &'static str {
    KEYS[state.index()]
}

fn world(base: Arc<dyn BaseSource>, tables: VoxelTables, top_level: u8) -> VoxelWorld {
    let config = WorldConfig {
        top_level,
        ..WorldConfig::default()
    };
    VoxelWorld::new(base, tables, config)
}

fn air_base() -> Arc<dyn BaseSource> {
    Arc::new(FnBase::new("test:air", |_, _| AIR))
}

/// Every key that has a layer, with its face neighbours.
fn keys_near_layers(snapshot: &VoxelSnapshot) -> BTreeSet<LodKey> {
    let mut keys = BTreeSet::new();
    for (key, _) in snapshot.overrides() {
        keys.insert(key);
        for face in Face::ALL {
            keys.insert(LodKey::new(key.level, key.pos + face.normal()));
        }
    }
    keys
}

#[test]
fn apron_stamp_follows_the_documented_recipe() {
    let mut world = world(air_base(), tables(), 0);
    let unedited = world
        .snapshot()
        .apron_stamp(LodKey::lod0(IVec3::ZERO), persistent);
    // One block inside chunk (0, 0, 0) and one on the face of (1, 0, 0) that touches it.
    world.apply(&[
        CellEdit {
            cell: IVec3::new(5, 6, 7),
            state: PLANK,
        },
        CellEdit {
            cell: IVec3::new(32, 6, 7),
            state: STONE,
        },
    ]);
    let snapshot = world.snapshot();
    let key = LodKey::lod0(IVec3::ZERO);
    let stamp = snapshot.apron_stamp(key, persistent);

    let mut bytes = b"cp-voxel:apron".to_vec();
    bytes.extend_from_slice(&APRON_STAMP_VERSION.to_le_bytes());
    let mut no_data = bytes.clone();
    no_data.extend_from_slice(&[0; 7]);
    assert_eq!(unedited, hash_bytes(&no_data), "no override data anywhere");
    // The own layer: its canonical encoding and its authored mask.
    let layer = snapshot.override_layer(key).expect("edited");
    bytes.push(1);
    layer.blocks.write_canonical(&mut bytes, key_of);
    assert_eq!(UNCHANGED_KEY, key_of(BlockStateId::UNCHANGED));
    bytes.push(1);
    for word in layer.authored_words().expect("authored") {
        bytes.extend_from_slice(&word.to_le_bytes());
    }
    // +X: cell (a, b) = (y, z) of the neighbour's x = 0 slice; one stone at (6, 7).
    bytes.push(1);
    bytes.extend_from_slice(&2_u32.to_le_bytes());
    for key in [UNCHANGED_KEY, persistent(STONE)] {
        bytes.extend_from_slice(&(key.len() as u32).to_le_bytes());
        bytes.extend_from_slice(key.as_bytes());
    }
    for index in 0..1024_u16 {
        let entry: u16 = u16::from(index == 6 * 32 + 7);
        bytes.extend_from_slice(&entry.to_le_bytes());
    }
    // -X, +Y, -Y, +Z, -Z: no neighbour layer.
    bytes.extend_from_slice(&[0; 5]);
    assert_eq!(stamp, hash_bytes(&bytes));
    assert_ne!(stamp, unedited);
}

#[test]
fn apron_stamps_are_equal_for_equal_overrides_whatever_the_history() {
    let edits: Vec<CellEdit> = (0..300)
        .map(|i| {
            let mut rng = rng(500 + i);
            CellEdit {
                cell: IVec3::new(
                    range_i32(&mut rng, -70, 70),
                    range_i32(&mut rng, -10, 30),
                    range_i32(&mut rng, -70, 70),
                ),
                state: if i % 3 == 0 { AIR } else { PLANK },
            }
        })
        .collect();
    // Session 1: one apply.
    let mut first = world(Arc::new(HillsBase::new()), tables(), 4);
    first.apply(&edits);
    // Session 2: unrelated edits (above the edited box, sharing its pyramid chunks) that are
    // reverted again, then the final state of every edited cell in two batches, the later
    // half first, and one batch twice.
    let mut second = world(Arc::new(HillsBase::new()), tables(), 4);
    let junk: Vec<CellEdit> = (0..50)
        .map(|i| CellEdit {
            cell: IVec3::new(i - 25, 35, 40),
            state: STONE,
        })
        .collect();
    second.apply(&junk);
    let unique: BTreeSet<[i32; 3]> = edits.iter().map(|edit| edit.cell.to_array()).collect();
    let last_wins: Vec<CellEdit> = unique
        .iter()
        .map(|&cell| {
            *edits
                .iter()
                .rev()
                .find(|edit| edit.cell.to_array() == cell)
                .expect("cell from the edits")
        })
        .collect();
    let (front, back) = last_wins.split_at(last_wins.len() / 2);
    second.apply(back);
    second.apply(
        &junk
            .iter()
            .map(|edit| CellEdit {
                cell: edit.cell,
                state: BlockStateId::UNCHANGED,
            })
            .collect::<Vec<_>>(),
    );
    second.apply(front);
    second.apply(front);
    let (a, b) = (first.snapshot(), second.snapshot());
    let keys = keys_near_layers(&a);
    assert_eq!(keys, keys_near_layers(&b), "the same layers");
    let mut different_revisions = 0;
    for &key in &keys {
        assert_eq!(
            a.apron_stamp(key, persistent),
            b.apron_stamp(key, persistent),
            "{key:?}"
        );
        if a.revision(key) != b.revision(key) {
            different_revisions += 1;
        }
    }
    assert!(keys.len() > 100);
    assert!(different_revisions > 0, "revisions depend on the history");
}

#[test]
fn apron_stamp_covers_the_chunk_and_the_halo_slices_only() {
    let mut world = world(air_base(), tables(), 5);
    let key = LodKey::lod0(IVec3::new(2, 0, -3));
    let origin = key.origin_voxel();
    let stamp = |world: &VoxelWorld, key: LodKey| world.snapshot().apron_stamp(key, persistent);
    let mut last = stamp(&world, key);
    let mut expect_change = |world: &mut VoxelWorld, cell: IVec3, changes: bool| {
        world.apply(&[CellEdit { cell, state: PLANK }]);
        let now = stamp(&*world, key);
        assert_eq!(now != last, changes, "edit at {cell:?}");
        last = now;
    };
    // Inside the chunk, at its border too.
    expect_change(&mut world, origin + IVec3::new(9, 9, 9), true);
    expect_change(&mut world, origin + IVec3::new(0, 31, 4), true);
    // A neighbour's interior and its far side: not in the halo.
    expect_change(&mut world, origin + IVec3::new(32 + 1, 4, 4), false);
    expect_change(&mut world, origin + IVec3::new(4, 4, -32), false);
    // Edge and corner neighbours: their cells are no face halo.
    expect_change(&mut world, origin + IVec3::new(32, 32, 4), false);
    expect_change(&mut world, origin + IVec3::new(-1, -1, -1), false);
    // The slices that touch the chunk, on every side.
    for cell in [
        IVec3::new(32, 4, 4),
        IVec3::new(-1, 4, 4),
        IVec3::new(4, 32, 4),
        IVec3::new(4, -1, 4),
        IVec3::new(4, 4, 32),
        IVec3::new(4, 4, -1),
    ] {
        expect_change(&mut world, origin + cell, true);
    }

    // On a pyramid level: a layer with authored bits only is override data of its chunk but
    // not of the neighbours' halos.
    let cell = IVec3::new(0, 3000, 0);
    let coarse = LodKey::containing(5, cell);
    let neighbour = LodKey::new(5, coarse.pos - IVec3::X);
    let (own_before, neighbour_before) = (stamp(&world, coarse), stamp(&world, neighbour));
    world.apply(&[CellEdit { cell, state: PLANK }]);
    let snapshot = world.snapshot();
    let layer = snapshot.override_layer(coarse).expect("authored ancestor");
    assert_eq!(layer.blocks.uniform_state(), Some(BlockStateId::UNCHANGED));
    // Level-5 cell y = 3000 >> 5 = 93, local 93 & 31 = 29; x = z = 0 lie on the border.
    assert!(layer.is_authored(cell_index(0, 29, 0)));
    assert_ne!(stamp(&world, coarse), own_before);
    assert_eq!(stamp(&world, neighbour), neighbour_before);
}

#[test]
fn apron_stamp_uses_persistent_keys_not_session_ids() {
    // A second registry that numbers grass and plank the other way round.
    let swap = |state: BlockStateId| match state.0 {
        3 => BlockStateId(8),
        8 => BlockStateId(3),
        _ => state,
    };
    let reference = tables();
    let flags: Vec<BlockFlags> = (0..STATE_COUNT as u32)
        .map(|id| reference.flags[swap(BlockStateId(id)).index()])
        .collect();
    let lod: Vec<LodInfo> = (0..STATE_COUNT as u32)
        .map(|id| {
            let info = reference.lod[swap(BlockStateId(id)).index()];
            LodInfo {
                class: info.class,
                proxy: swap(info.proxy),
            }
        })
        .collect();
    let swapped = VoxelTables::new(Arc::from(flags), Arc::from(lod));
    let swapped_key = move |state: BlockStateId| KEYS[swap(state).index()];
    assert_eq!(swapped_key(swap(PLANK)), persistent(PLANK));

    // Planks and air only, so that no downsample tie depends on the id order.
    let edits: Vec<CellEdit> = (0..400)
        .map(|i| CellEdit {
            cell: IVec3::new(i % 37 - 18, (i / 37) % 11 - 2, i % 23 - 11),
            state: if i % 4 == 0 { AIR } else { PLANK },
        })
        .collect();
    let mut a = world(air_base(), tables(), 3);
    a.apply(&edits);
    let mut b = world(air_base(), swapped, 3);
    b.apply(
        &edits
            .iter()
            .map(|edit| CellEdit {
                cell: edit.cell,
                state: swap(edit.state),
            })
            .collect::<Vec<_>>(),
    );
    let (a, b) = (a.snapshot(), b.snapshot());
    let keys = keys_near_layers(&a);
    assert!(keys.iter().any(|key| key.level == 3));
    for &key in &keys {
        assert_eq!(
            a.apron_stamp(key, persistent),
            b.apron_stamp(key, swapped_key),
            "{key:?}"
        );
    }
    // The keys do enter the stamp.
    let key = LodKey::lod0(IVec3::ZERO);
    let renamed = a.apron_stamp(key, |state| {
        if state == PLANK {
            "test:renamed_plank"
        } else {
            persistent(state)
        }
    });
    assert_ne!(renamed, a.apron_stamp(key, persistent));
}
