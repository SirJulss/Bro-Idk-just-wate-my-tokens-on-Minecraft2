//! Measurements (contract §2.7, §10.8). Ignored by default; run in release:
//! `cargo test -p cp-voxel --release --test voxel -- --ignored --nocapture bench_`.
//! Timings taken while other builds run on the machine are not reliable.

// The benchmarks report their numbers on stdout.
#![allow(clippy::print_stdout)]

use std::sync::Arc;
use std::time::{Duration, Instant};

use cp_core::coords::{CHUNK_VOLUME, LodKey};
use cp_core::ids::BlockStateId;
use cp_core::math::IVec3;
use cp_voxel::{
    CellEdit, ChunkData, VoxelChunk, VoxelWorld, WorldConfig, padded_cells, region_volume,
};

use crate::common::{AIR, CountingBase, FnBase, HillsBase, PLANK, STONE, range_i32, rng, tables};

fn stats(name: &str, samples: &mut [Duration]) {
    samples.sort_unstable();
    let total: Duration = samples.iter().sum();
    let mean = total / samples.len() as u32;
    let p50 = samples[samples.len() / 2];
    let p99 = samples[samples.len() * 99 / 100];
    let max = samples[samples.len() - 1];
    println!(
        "{name}: n={} mean={:.1} us p50={:.1} us p99={:.1} us max={:.1} us",
        samples.len(),
        mean.as_secs_f64() * 1e6,
        p50.as_secs_f64() * 1e6,
        p99.as_secs_f64() * 1e6,
        max.as_secs_f64() * 1e6
    );
}

#[test]
#[ignore = "benchmark; run in release with --ignored --nocapture"]
fn bench_sample_padded() {
    let world = VoxelWorld::new(Arc::new(HillsBase::new()), tables(), WorldConfig::default());
    let snapshot = world.snapshot();
    for level in [0_u8, 3] {
        let mut samples = Vec::new();
        for i in 0..300 {
            // Surface chunks, each one new (the padded path always fills 34³ cells).
            let key = LodKey::new(level, IVec3::new(i % 20 - 10, 0, i / 20 - 7));
            let start = Instant::now();
            let padded = snapshot.sample_padded(key);
            samples.push(start.elapsed());
            assert_eq!(padded.cells().len(), region_volume(padded_cells(key)));
        }
        stats(&format!("sample_padded hills level {level}"), &mut samples);
    }
    // The same with a trivial base: the cost of the path without generation work.
    let trivial = VoxelWorld::new(
        Arc::new(FnBase::new("test:constant", |_, _| STONE)),
        tables(),
        WorldConfig::default(),
    );
    let snapshot = trivial.snapshot();
    let mut samples = Vec::new();
    for i in 0..300 {
        let key = LodKey::lod0(IVec3::new(i, 0, 0));
        let start = Instant::now();
        let _ = snapshot.sample_padded(key);
        samples.push(start.elapsed());
    }
    stats("sample_padded constant base level 0", &mut samples);
    // sample() of a generated chunk: cache miss (generation) and cache hit.
    let snapshot = world.snapshot();
    let mut misses = Vec::new();
    let mut hits = Vec::new();
    for i in 0..300 {
        let key = LodKey::lod0(IVec3::new(i % 20 - 10, 0, i / 20 + 20));
        let start = Instant::now();
        let first = snapshot.sample(key);
        misses.push(start.elapsed());
        let start = Instant::now();
        let again = snapshot.sample(key);
        hits.push(start.elapsed());
        assert!(Arc::ptr_eq(&first, &again));
    }
    stats("sample hills level 0 (generate)", &mut misses);
    stats("sample hills level 0 (cache hit)", &mut hits);
}

#[test]
#[ignore = "benchmark; run in release with --ignored --nocapture"]
fn bench_apply_one_cell() {
    // Home has root level 10 (contract §2.2): the pyramid is updated up to level 10.
    let config = WorldConfig {
        top_level: 10,
        ..WorldConfig::default()
    };
    let mut world = VoxelWorld::new(Arc::new(HillsBase::new()), tables(), config);
    let mut rng = rng(21);
    let mut samples = Vec::new();
    let mut dirty = 0;
    for i in 0..2000 {
        let cell = IVec3::new(
            range_i32(&mut rng, -300, 300),
            range_i32(&mut rng, -2, 20),
            range_i32(&mut rng, -300, 300),
        );
        let state = if i % 2 == 0 { PLANK } else { AIR };
        // Keep a snapshot alive, as the renderer does, so every write pays copy on write.
        let _snapshot = world.snapshot();
        let start = Instant::now();
        let receipt = world.apply(&[CellEdit { cell, state }]);
        samples.push(start.elapsed());
        dirty += receipt.dirty.len();
    }
    stats(
        "apply 1 cell (pyramid to level 10, snapshot held)",
        &mut samples,
    );
    println!(
        "  mean dirty keys per edit: {:.1}; override layers: {}",
        dirty as f64 / 2000.0,
        world.edits().layer_count()
    );
    let start = Instant::now();
    let snapshot = world.snapshot();
    println!("snapshot(): {:.2} us", start.elapsed().as_secs_f64() * 1e6);
    drop(snapshot);

    // Worst case for the pyramid: every edit lands in a level-10 cell (1024 m) no edit has
    // reached before, so all eleven levels change.
    let mut world = VoxelWorld::new(Arc::new(HillsBase::new()), tables(), config);
    let mut samples = Vec::new();
    let mut dirty = 0;
    for i in 0..225 {
        let cell = IVec3::new((i % 15 - 7) * 1024 + 300, 12, (i / 15 - 7) * 1024 + 300);
        let _snapshot = world.snapshot();
        let start = Instant::now();
        let receipt = world.apply(&[CellEdit { cell, state: PLANK }]);
        samples.push(start.elapsed());
        dirty += receipt.dirty.len();
    }
    stats(
        "apply 1 cell in an untouched area (all 11 levels change, snapshot held)",
        &mut samples,
    );
    println!(
        "  mean dirty keys per edit: {:.1}",
        dirty as f64 / samples.len() as f64
    );
}

/// Base calls and time of `apply` with a base whose `classify` evaluates every column of the
/// chunk (exact-conservative, like cp-worldgen's hills): the class bookkeeping of the receipt
/// must not turn every edit into a series of such evaluations.
#[test]
#[ignore = "benchmark; run in release with --ignored --nocapture"]
fn bench_apply_class_bookkeeping() {
    let config = WorldConfig {
        top_level: 10,
        ..WorldConfig::default()
    };
    let column_world = || {
        let base = Arc::new(CountingBase::new(
            HillsBase::column_classified(),
            Duration::ZERO,
        ));
        (VoxelWorld::new(base.clone(), tables(), config), base)
    };
    // What one classify of a surface chunk costs here, for scale.
    let (_, base) = column_world();
    let mut samples = Vec::new();
    for i in 0..300 {
        let key = LodKey::lod0(IVec3::new(i % 20 - 10, 0, i / 20));
        let start = Instant::now();
        let _ = cp_voxel::BaseSource::classify(&*base, key);
        samples.push(start.elapsed());
    }
    stats("column classify of a level-0 surface chunk", &mut samples);

    let report = |name: &str, samples: &mut Vec<Duration>, base: &CountingBase<HillsBase>| {
        let n = samples.len();
        stats(name, samples);
        println!(
            "  per apply: classify calls {:.2}, fill calls {:.2}",
            base.classifies() as f64 / n as f64,
            base.fills() as f64 / n as f64
        );
    };

    // Single blocks at the surface of areas no edit has reached (all levels get new layers).
    let (mut world, base) = column_world();
    let mut rng = rng(41);
    let mut samples = Vec::new();
    for i in 0..300 {
        let x = range_i32(&mut rng, -4000, 4000);
        let z = range_i32(&mut rng, -4000, 4000);
        let top = HillsBase::ground_top(x, z);
        let (cell, state) = if i % 2 == 0 {
            (IVec3::new(x, top, z), AIR)
        } else {
            (IVec3::new(x, top + 1, z), PLANK)
        };
        let _snapshot = world.snapshot();
        let start = Instant::now();
        world.apply(&[CellEdit { cell, state }]);
        samples.push(start.elapsed());
    }
    report(
        "apply 1 surface cell in a fresh area (pyramid to level 10, snapshot held)",
        &mut samples,
        &base,
    );

    // A tunnel: digs next to each other below the surface, in chunks edited before.
    let (mut world, base) = column_world();
    let origin = IVec3::new(100, HillsBase::ground_top(100, 100) - 3, 100);
    let mut samples = Vec::new();
    for i in 0..200 {
        let cell = origin + IVec3::new(i % 20, -(i / 20), 0);
        let _snapshot = world.snapshot();
        let start = Instant::now();
        world.apply(&[CellEdit { cell, state: AIR }]);
        samples.push(start.elapsed());
    }
    report("apply 1 tunnel cell (snapshot held)", &mut samples, &base);

    // A blueprint paste: one apply of a 64 x 6 x 64 slab of planks on fresh terrain.
    let (mut world, base) = column_world();
    let mut samples = Vec::new();
    for i in 0..10 {
        let corner = IVec3::new(i * 4096 - 20_000, 0, 9000);
        let mut edits = Vec::new();
        for z in 0..64 {
            for x in 0..64 {
                let top = HillsBase::ground_top(corner.x + x, corner.z + z);
                for y in 1..=6 {
                    edits.push(CellEdit {
                        cell: IVec3::new(corner.x + x, top + y, corner.z + z),
                        state: PLANK,
                    });
                }
            }
        }
        let _snapshot = world.snapshot();
        let start = Instant::now();
        world.apply(&edits);
        samples.push(start.elapsed());
    }
    report(
        "apply a 64 x 6 x 64 paste on fresh terrain (24576 cells)",
        &mut samples,
        &base,
    );
}

#[test]
#[ignore = "benchmark; run in release with --ignored --nocapture"]
fn bench_memory_per_chunk() {
    let report = |name: &str, chunk: &VoxelChunk| {
        println!(
            "memory {name}: {} bytes (bits {})",
            chunk.memory_bytes(),
            chunk.blocks.bits()
        );
    };
    report("uniform", &VoxelChunk::uniform(STONE));
    for states in [2_u32, 4, 16, 256, 4096] {
        let blocks = ChunkData::from_fn(|index| BlockStateId(index as u32 % states));
        report(
            &format!("paletted {states} states"),
            &VoxelChunk::new(blocks),
        );
    }
    let hills = HillsBase::new();
    let surface = crate::common::base_chunk(&hills, LodKey::lod0(IVec3::ZERO));
    report("hills surface chunk", &surface);
    let mut authored = surface.clone();
    authored.set_authored(0, true);
    report("hills surface chunk with authored mask", &authored);
    let mut layer = VoxelChunk::uniform(BlockStateId::UNCHANGED);
    layer.blocks.set(100, PLANK);
    layer.set_authored(100, true);
    report("one-cell override layer", &layer);
    println!(
        "size_of: ChunkData {} VoxelChunk {} (cells per chunk {CHUNK_VOLUME})",
        size_of::<ChunkData>(),
        size_of::<VoxelChunk>()
    );
}
