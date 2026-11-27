//! Snapshot isolation, copy on write, padded halos and the single-flight cache.

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Barrier};
use std::time::{Duration, Instant};

use cp_core::coords::{CHUNK_VOLUME, LodKey, cell_index, index_to_cell};
use cp_core::face::Face;
use cp_core::ids::BlockStateId;
use cp_core::math::{IAabb, IVec3};
use cp_voxel::{
    BaseSource, Cancelled, CellEdit, GeneratorIdentity, NodeClass, VoxelWorld, WorldConfig,
};

use crate::common::{
    AIR, CountingBase, FnBase, HillsBase, LEAVES, LOG, PLANK, STONE, WATER, base_chunk, identity,
    range_i32, rng, tables,
};

fn hills_world(top_level: u8) -> VoxelWorld {
    let config = WorldConfig {
        top_level,
        ..WorldConfig::default()
    };
    VoxelWorld::new(Arc::new(HillsBase::new()), tables(), config)
}

#[test]
fn snapshots_are_isolated_and_share_untouched_chunks() {
    let mut world = hills_world(4);
    // Edits in several chunks and regions, including negative ones.
    let mut edits = Vec::new();
    for (i, cell) in [
        IVec3::new(3, 5, 3),
        IVec3::new(-40, 2, -7),
        IVec3::new(600, 0, -600),
        IVec3::new(100, 9, 100),
    ]
    .into_iter()
    .enumerate()
    {
        for d in 0..5 {
            edits.push(CellEdit {
                cell: cell + IVec3::new(d, i as i32, 0),
                state: PLANK,
            });
        }
    }
    world.apply(&edits);
    let first = world.snapshot();
    let first_sample = first.sample(LodKey::lod0(IVec3::ZERO));
    // Change one chunk after the snapshot.
    let receipt = world.apply(&[CellEdit {
        cell: IVec3::new(4, 6, 4),
        state: STONE,
    }]);
    let second = world.snapshot();
    let key = LodKey::lod0(IVec3::ZERO);
    // The old snapshot still sees the old state, the new one the edit.
    assert_eq!(
        first.sample(key).get(cell_index(4, 6, 4)),
        first_sample.get(cell_index(4, 6, 4))
    );
    assert_eq!(second.sample(key).get(cell_index(4, 6, 4)), STONE);
    assert_eq!(
        first.cell_state(IVec3::new(4, 6, 4)),
        HillsBase::cell(0, IVec3::new(4, 6, 4))
    );
    assert_eq!(second.cell_state(IVec3::new(4, 6, 4)), STONE);
    assert!(second.revision(key) > first.revision(key));
    assert_eq!(first.sequence() + 1, second.sequence());
    // Layers the edit did not touch are the same allocation in both snapshots; the written
    // one was copied, so the old snapshot keeps its version.
    let mut shared = 0;
    for (layer_key, old) in first.overrides() {
        let new = second
            .override_layer(layer_key)
            .expect("no layer was removed");
        if !receipt.dirty.contains(&layer_key) {
            assert!(Arc::ptr_eq(old, new), "{layer_key:?} was copied");
            shared += 1;
        }
    }
    assert!(shared > 10, "most layers stay shared ({shared})");
    assert!(!Arc::ptr_eq(
        first.override_layer(key).expect("edited before"),
        second.override_layer(key).expect("edited after")
    ));
    // Base chunks come from the shared cache: the same allocation in both snapshots.
    let unedited = LodKey::lod0(IVec3::new(-5, 0, 9));
    assert!(Arc::ptr_eq(
        &first.sample(unedited),
        &second.sample(unedited)
    ));
    // Snapshots move to other threads.
    let handle = std::thread::spawn(move || first.sample(key).get(cell_index(4, 6, 4)));
    assert_eq!(
        handle.join().unwrap(),
        first_sample.get(cell_index(4, 6, 4))
    );
}

#[test]
fn sample_padded_halo_equals_the_neighbour_cells() {
    let mut world = hills_world(3);
    let mut rng = rng(8);
    // Edits concentrated on chunk borders (local coordinate 0 or 31) plus clusters.
    let mut edits = Vec::new();
    for _ in 0..3000 {
        let mut cell = IVec3::new(
            range_i32(&mut rng, -70, 70),
            range_i32(&mut rng, -20, 40),
            range_i32(&mut rng, -70, 70),
        );
        let axis = rng.range_u32(0, 3) as usize;
        cell[axis] = (cell[axis] >> 5) * 32 + if rng.range_u32(0, 2) == 0 { 0 } else { 31 };
        let states = [AIR, STONE, PLANK, LOG, LEAVES, WATER];
        let state = states[rng.range_u32(0, states.len() as u32) as usize];
        edits.push(CellEdit { cell, state });
    }
    for x in -8..8 {
        for y in 0..16 {
            edits.push(CellEdit {
                cell: IVec3::new(x, y, 31),
                state: STONE,
            });
        }
    }
    world.apply(&edits);
    let snapshot = world.snapshot();
    let mut keys: Vec<LodKey> = Vec::new();
    for level in 0..=3_u8 {
        let span = if level == 0 { 3 } else { 1 };
        for x in -span..span {
            for y in -1..2 {
                for z in -span..span {
                    keys.push(LodKey::new(level, IVec3::new(x, y, z)));
                }
            }
        }
    }
    let mut edited_halos = 0;
    for key in keys {
        let padded = snapshot.sample_padded(key);
        assert_eq!(padded.key(), key);
        let own = snapshot.sample(key);
        for index in 0..CHUNK_VOLUME {
            let [x, y, z] = index_to_cell(index).map(i32::from);
            assert_eq!(padded.get(x, y, z), own.get(index), "{key:?} interior");
        }
        assert_eq!(padded.authored(), own.authored_words(), "{key:?} authored");
        for face in Face::ALL {
            let neighbour_key = LodKey::new(key.level, key.pos + face.normal());
            let neighbour = snapshot.sample(neighbour_key);
            if snapshot.override_layer(neighbour_key).is_some() {
                edited_halos += 1;
            }
            let axis = face.axis();
            for a in 0..32 {
                for b in 0..32 {
                    let mut ours = [0_i32; 3];
                    let mut theirs = [0_u8; 3];
                    ours[axis] = if face.is_negative() { -1 } else { 32 };
                    theirs[axis] = if face.is_negative() { 31 } else { 0 };
                    ours[(axis + 1) % 3] = a;
                    ours[(axis + 2) % 3] = b;
                    theirs[(axis + 1) % 3] = a as u8;
                    theirs[(axis + 2) % 3] = b as u8;
                    assert_eq!(
                        padded.get(ours[0], ours[1], ours[2]),
                        neighbour.get(cell_index(theirs[0], theirs[1], theirs[2])),
                        "{key:?} halo {face:?}"
                    );
                }
            }
        }
    }
    assert!(
        edited_halos > 20,
        "the test reaches edited neighbours ({edited_halos})"
    );
    // The fast path (no override near the chunk) equals the base fill.
    let far = LodKey::lod0(IVec3::new(40, 0, 40));
    let padded = snapshot.sample_padded(far);
    let base = base_chunk(world.base().as_ref(), far);
    assert_eq!(padded.get(5, 6, 7), base.get(cell_index(5, 6, 7)));
    assert!(padded.authored().is_none());
}

#[test]
fn concurrent_requests_generate_once() {
    let base = Arc::new(CountingBase::new(
        HillsBase::new(),
        Duration::from_millis(60),
    ));
    let world = VoxelWorld::new(base.clone(), tables(), WorldConfig::default());
    let snapshot = world.snapshot();
    let key = LodKey::lod0(IVec3::new(1, 0, -1));
    let threads = 8;
    let barrier = Barrier::new(threads);
    let results: Vec<_> = std::thread::scope(|scope| {
        let handles: Vec<_> = (0..threads)
            .map(|_| {
                scope.spawn(|| {
                    barrier.wait();
                    snapshot.sample(key)
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });
    assert_eq!(
        base.fills(),
        1,
        "one generation for {threads} concurrent requests"
    );
    for result in &results {
        assert!(Arc::ptr_eq(result, &results[0]));
    }
    assert_eq!(*results[0], base_chunk(&HillsBase::new(), key));
    // Later requests are cache hits.
    let again = snapshot.sample(key);
    assert!(Arc::ptr_eq(&again, &results[0]));
    assert_eq!(base.fills(), 1);
    // A second snapshot shares the cache.
    assert!(Arc::ptr_eq(&world.snapshot().sample(key), &results[0]));
    assert_eq!(base.fills(), 1);
    assert_eq!(world.cache_usage().chunks, 1);
}

/// A base whose first `fill` waits, then panics.
struct PanicOnce {
    identity: GeneratorIdentity,
    fills: AtomicUsize,
}

impl BaseSource for PanicOnce {
    fn identity(&self) -> &GeneratorIdentity {
        &self.identity
    }

    fn fill(
        &self,
        level: u8,
        region: IAabb,
        out: &mut [BlockStateId],
        cancel: &dyn Fn() -> bool,
    ) -> Result<(), Cancelled> {
        if self.fills.fetch_add(1, Ordering::SeqCst) == 0 {
            std::thread::sleep(Duration::from_millis(80));
            panic!("first generation fails");
        }
        crate::common::fill_with(level, region, out, cancel, |_, _| STONE)
    }

    fn classify(&self, _key: LodKey) -> NodeClass {
        NodeClass::Mixed
    }

    fn subtree_empty(&self, _key: LodKey) -> bool {
        false
    }
}

#[test]
fn a_failed_generation_is_retried_by_a_waiter() {
    let base = Arc::new(PanicOnce {
        identity: identity("test:panic_once"),
        fills: AtomicUsize::new(0),
    });
    let world = VoxelWorld::new(base.clone(), tables(), WorldConfig::default());
    let snapshot = world.snapshot();
    let key = LodKey::lod0(IVec3::ZERO);
    std::thread::scope(|scope| {
        let leader = scope.spawn(|| snapshot.sample(key));
        std::thread::sleep(Duration::from_millis(20));
        let waiter = scope.spawn(|| snapshot.sample(key));
        assert!(
            leader.join().is_err(),
            "the leader's panic propagates to its caller"
        );
        let chunk = waiter
            .join()
            .expect("the waiter generates after the failure");
        assert_eq!(chunk.blocks.uniform_state(), Some(STONE));
    });
    assert_eq!(base.fills.load(Ordering::SeqCst), 2);
}

#[test]
fn cancellation_stops_generation_and_waiting() {
    // A cancelled generation returns Cancelled and leaves nothing behind.
    let base = Arc::new(FnBase::new("test:stone", |_, _| STONE));
    let world = VoxelWorld::new(base, tables(), WorldConfig::default());
    let snapshot = world.snapshot();
    let key = LodKey::lod0(IVec3::new(0, 0, 3));
    assert_eq!(snapshot.try_sample(key, &|| true).unwrap_err(), Cancelled);
    assert_eq!(
        snapshot.try_sample_padded(key, &|| true).unwrap_err(),
        Cancelled
    );
    assert_eq!(world.cache_usage().chunks, 0);
    assert_eq!(snapshot.sample(key).blocks.uniform_state(), Some(STONE));

    // A waiter whose own cancel fires returns at once while the leader keeps going.
    let slow = Arc::new(CountingBase::new(
        FnBase::new("test:slow", |_, _| STONE),
        Duration::from_millis(300),
    ));
    let world = VoxelWorld::new(slow.clone(), tables(), WorldConfig::default());
    let snapshot = world.snapshot();
    let key = LodKey::lod0(IVec3::new(2, 2, 2));
    let stop = AtomicBool::new(false);
    std::thread::scope(|scope| {
        let leader = scope.spawn(|| snapshot.sample(key));
        std::thread::sleep(Duration::from_millis(30));
        let started = Instant::now();
        let waiter = scope.spawn(|| snapshot.try_sample(key, &|| stop.load(Ordering::SeqCst)));
        std::thread::sleep(Duration::from_millis(30));
        stop.store(true, Ordering::SeqCst);
        assert_eq!(waiter.join().unwrap().unwrap_err(), Cancelled);
        assert!(
            started.elapsed() < Duration::from_millis(250),
            "waiter returned early"
        );
        assert_eq!(leader.join().unwrap().blocks.uniform_state(), Some(STONE));
    });
    assert_eq!(slow.fills(), 1);
}

#[test]
fn classify_combines_base_and_overrides() {
    let mut world = hills_world(2);
    let sky = LodKey::lod0(IVec3::new(0, 2, 0));
    let deep = LodKey::lod0(IVec3::new(0, -2, 0));
    let snapshot = world.snapshot();
    assert_eq!(snapshot.classify(sky), NodeClass::Empty);
    assert_eq!(snapshot.classify(deep), NodeClass::Opaque);
    // An empty sky chunk is stored as one uniform state.
    assert_eq!(snapshot.sample(sky).blocks.uniform_state(), Some(AIR));
    // Placing a block in the sky and digging underground make both chunks Mixed.
    let receipt = world.apply(&[
        CellEdit {
            cell: IVec3::new(3, 70, 3),
            state: PLANK,
        },
        CellEdit {
            cell: IVec3::new(3, -60, 3),
            state: AIR,
        },
    ]);
    let after = world.snapshot();
    assert_eq!(after.classify(sky), NodeClass::Mixed);
    assert_eq!(after.classify(deep), NodeClass::Mixed);
    let changes: Vec<LodKey> = receipt
        .class_changes
        .iter()
        .map(|change| change.key)
        .collect();
    assert!(changes.contains(&sky) && changes.contains(&deep));
    // Opaque stays opaque when an opaque block replaces another one.
    let receipt = world.apply(&[CellEdit {
        cell: IVec3::new(5, -60, 5),
        state: PLANK,
    }]);
    assert!(
        !receipt
            .class_changes
            .iter()
            .any(|change| change.key == deep && change.after == NodeClass::Opaque)
    );
    world.apply(&[CellEdit {
        cell: IVec3::new(3, -60, 3),
        state: BlockStateId::UNCHANGED,
    }]);
    assert_eq!(world.snapshot().classify(deep), NodeClass::Opaque);
}

#[test]
fn generating_a_base_chunk_is_one_fill_and_no_classify() {
    // classify of this base evaluates every column of the chunk: as costly as the fill of a
    // surface chunk, so a cache miss must not ask it first.
    let base = Arc::new(CountingBase::new(
        HillsBase::column_classified(),
        Duration::ZERO,
    ));
    let world = VoxelWorld::new(base.clone(), tables(), WorldConfig::default());
    let snapshot = world.snapshot();
    for key in [
        LodKey::lod0(IVec3::new(3, 0, -2)),
        LodKey::lod0(IVec3::new(0, 9, 0)),
        LodKey::new(4, IVec3::new(-1, 0, 1)),
    ] {
        let (fills, classifies) = (base.fills(), base.classifies());
        let chunk = snapshot.sample(key);
        assert_eq!(base.fills(), fills + 1, "{key:?}");
        assert_eq!(base.classifies(), classifies, "{key:?}");
        assert_eq!(*chunk, base_chunk(&HillsBase::new(), key));
        if key.level == 0 {
            // Queries read the same cached chunk.
            let _ = snapshot.cell_state(key.origin_voxel());
            assert_eq!(base.fills(), fills + 1);
        }
    }
    assert_eq!(base.classifies(), 0);
    // The sky chunk took the base's all-air fast path and is uniform.
    let sky = snapshot.sample(LodKey::lod0(IVec3::new(0, 9, 0)));
    assert_eq!(sky.blocks.uniform_state(), Some(AIR));
}

#[test]
fn subtree_empty_sees_player_builds() {
    // Pyramid up to level 2; keys above it are answered from the level-2 layers.
    let mut world = hills_world(2);
    let platform = IVec3::new(-300, 5000, 700);
    let keys = [0_u8, 1, 2, 3, 6].map(|level| LodKey::containing(level, platform));
    let elsewhere = LodKey::containing(3, platform + IVec3::new(4096, 0, 0));
    let snapshot = world.snapshot();
    for key in keys {
        assert!(snapshot.subtree_empty(key), "{key:?}");
    }
    world.apply(&[CellEdit {
        cell: platform,
        state: PLANK,
    }]);
    let built = world.snapshot();
    for key in keys {
        assert!(!built.subtree_empty(key), "{key:?} hides the platform");
    }
    assert!(built.subtree_empty(elsewhere));
    // Ground is never empty.
    assert!(!built.subtree_empty(LodKey::new(5, IVec3::ZERO)));
    world.apply(&[CellEdit {
        cell: platform,
        state: BlockStateId::UNCHANGED,
    }]);
    let reverted = world.snapshot();
    for key in keys {
        assert!(reverted.subtree_empty(key), "{key:?} after the revert");
    }
}
