//! Release benchmark on terrain-like chunks (contract §5.2: at most 0.2 ms per chunk on
//! typical terrain).
//!
//! `cargo test -p cp-mesh --release --lib -- --ignored --nocapture bench_terrain`
//!
//! Scenes:
//! - `hills`: the fixture palette's rolling hills with trees and a sea, level 0 (the first
//!   version of this benchmark).
//! - `caves`: the base pack's states and ids; hills with worm caves, ore clusters and gravel
//!   pockets, level 0.
//! - `mountains`: base-pack ids; a ridged mountain surface with caves on level 3 (8 m cells),
//!   where the surface climbs several cells per column.
//! - `canopy`: base-pack ids; a dense 16-cell forest canopy with a quarter of its cells open.
//! - the last three again with every state id multiplied by 64 (`spread`), which puts every
//!   state into the same slot of any id-keyed direct-mapped table of up to 64 entries. The
//!   mesher's per-state lookups must not care: the summary prints the time ratio.

use std::time::{Duration, Instant};

use cp_core::ids::BlockStateId;

use super::fixtures::{Palette, base_pack, input, palette, spread, spread_id};
use super::terrain::{Style, Terrain, bench_chunks};
use crate::{ChunkMesh, Mesher};

/// Timed runs per chunk; the report gives the first run and the fastest one.
const RUNS: usize = 5;

/// Multiplier of the state ids in the spread scenes.
const SPREAD: u32 = 64;

/// Measurements of one chunk.
struct Sample {
    /// First timed run (after one warm-up pass over all chunks of the scene, so the chunk's
    /// cells are not in the caches any more).
    first: Duration,
    /// Fastest of [`RUNS`] runs in a row (cells and scratch buffers in the caches).
    fastest: Duration,
    regular: usize,
    walls: usize,
}

/// One chunk of a scene.
struct Chunk {
    cells: Vec<BlockStateId>,
    level: u8,
    /// Where it comes from, for the report of the slowest chunks.
    label: String,
}

/// Chunks meshed with one palette.
struct Scene {
    name: String,
    palette: Palette,
    chunks: Vec<Chunk>,
}

impl Scene {
    /// The same chunks with every state id multiplied by [`SPREAD`].
    fn spread(&self) -> Self {
        Self {
            name: format!("{} spread", self.name),
            palette: spread(&self.palette, SPREAD),
            chunks: self
                .chunks
                .iter()
                .map(|chunk| Chunk {
                    cells: chunk
                        .cells
                        .iter()
                        .map(|&state| spread_id(state, SPREAD))
                        .collect(),
                    level: chunk.level,
                    label: chunk.label.clone(),
                })
                .collect(),
        }
    }
}

/// Mean and maximum in microseconds.
fn mean_max_us(values: impl Iterator<Item = Duration> + Clone) -> (f64, f64) {
    let count = values.clone().count().max(1) as f64;
    let total: Duration = values.clone().sum();
    let max = values.max().unwrap_or_default();
    (total.as_secs_f64() * 1e6 / count, max.as_secs_f64() * 1e6)
}

fn micros(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1e6
}

/// The scenes listed in the module documentation.
fn scenes() -> Vec<Scene> {
    let terrain = Terrain::new(5);
    let (fixture, blocks) = palette();
    let hills = Scene {
        name: "hills L0".to_owned(),
        palette: fixture,
        chunks: bench_chunks()
            .into_iter()
            .map(|chunk| Chunk {
                cells: terrain.padded_chunk(&blocks, chunk),
                level: 0,
                label: format!("{chunk:?}"),
            })
            .collect(),
    };
    let (pack, states) = base_pack();
    let columns = || (0..8).flat_map(|cz| (0..8).map(move |cx| [cx, cz]));
    let layered = |style: Style, level: u8, heights: &[i32]| -> Vec<Chunk> {
        let mut chunks = Vec::new();
        for &cy in heights {
            for [cx, cz] in columns() {
                let chunk = [cx, cy, cz];
                chunks.push(Chunk {
                    cells: terrain.pack_chunk(&states, style, level, chunk),
                    level,
                    label: format!("L{level} {chunk:?}"),
                });
            }
        }
        chunks
    };
    let caves = Scene {
        name: "caves L0".to_owned(),
        palette: pack.clone(),
        chunks: layered(
            Style {
                rough: false,
                caves: true,
            },
            0,
            &[-3, -2, -1, 0],
        ),
    };
    let mountains = Scene {
        name: "mountains L3".to_owned(),
        palette: pack.clone(),
        chunks: layered(
            Style {
                rough: true,
                caves: true,
            },
            3,
            &[-1, 0, 1],
        ),
    };
    let canopy = Scene {
        name: "canopy L0".to_owned(),
        palette: pack,
        chunks: columns()
            .map(|column| Chunk {
                cells: terrain.canopy_chunk(&states, column),
                level: 0,
                label: format!("canopy {column:?}"),
            })
            .collect(),
    };
    let spread: Vec<Scene> = [&caves, &mountains, &canopy]
        .into_iter()
        .map(Scene::spread)
        .collect();
    let mut all = vec![hills, caves, mountains, canopy];
    all.extend(spread);
    all
}

/// Meshes every chunk of `scene` and prints the report; returns the mean of the fastest runs
/// in microseconds and the number of chunks with regular quads.
// The benchmark reports to the test output.
#[allow(clippy::print_stdout)]
fn run(scene: &Scene, mesher: &mut Mesher, mesh: &mut ChunkMesh) -> (f64, usize) {
    let inputs: Vec<_> = scene
        .chunks
        .iter()
        .map(|chunk| input(&scene.palette, &chunk.cells, None, chunk.level))
        .collect();
    // Warm-up: every chunk once.
    for input in &inputs {
        mesher.mesh_into(input, mesh).expect("valid scene");
    }
    let mut samples = Vec::new();
    for input in &inputs {
        let mut times = [Duration::ZERO; RUNS];
        for time in &mut times {
            let start = Instant::now();
            mesher.mesh_into(input, mesh).expect("valid scene");
            *time = start.elapsed();
        }
        samples.push(Sample {
            first: times[0],
            fastest: times.iter().min().copied().unwrap_or_default(),
            regular: mesh.regular_len(),
            walls: mesh.wall_len(),
        });
    }
    // Phase split from the profiler, one more pass over all chunks.
    cp_core::profile::set_enabled(true);
    let _ = cp_core::profile::collect();
    for input in &inputs {
        mesher.mesh_into(input, mesh).expect("valid scene");
    }
    cp_core::profile::set_enabled(false);
    let phases = cp_core::profile::collect();

    println!(
        "scene {}: {} chunks, {RUNS} timed runs each",
        scene.name,
        samples.len()
    );
    type Filter = fn(&Sample) -> bool;
    let categories: [(&str, Filter); 4] = [
        ("all", |_| true),
        ("surface", |s| s.regular > 0),
        ("buried", |s| s.regular == 0 && s.walls > 0),
        ("empty", |s| s.regular == 0 && s.walls == 0),
    ];
    for (name, filter) in categories {
        let chosen: Vec<&Sample> = samples.iter().filter(|s| filter(s)).collect();
        if chosen.is_empty() {
            continue;
        }
        let (first_mean, first_max) = mean_max_us(chosen.iter().map(|s| s.first));
        let (best_mean, best_max) = mean_max_us(chosen.iter().map(|s| s.fastest));
        let count = chosen.len() as f64;
        let regular = chosen.iter().map(|s| s.regular).sum::<usize>() as f64 / count;
        let walls = chosen.iter().map(|s| s.walls).sum::<usize>() as f64 / count;
        let max_quads = chosen
            .iter()
            .map(|s| s.regular + s.walls)
            .max()
            .unwrap_or(0);
        println!(
            "  {name:7} {:4} chunks | first run: mean {first_mean:6.1} us, max {first_max:6.1} us \
             | fastest of {RUNS}: mean {best_mean:6.1} us, max {best_max:6.1} us \
             | quads: {regular:6.1} regular + {walls:5.1} walls (max {max_quads})",
            chosen.len()
        );
    }
    // The slowest chunk, to tell real outliers from interference (a chunk that is slow in
    // every run is real; one slow first run is not).
    let slowest = (0..samples.len()).max_by_key(|&index| samples[index].fastest);
    if let Some(index) = slowest {
        let s = &samples[index];
        println!(
            "  slowest: {}: first {:6.1} us, fastest {:6.1} us, {} quads",
            scene.chunks[index].label,
            micros(s.first),
            micros(s.fastest),
            s.regular + s.walls
        );
    }
    // `faces` runs only for chunks with renderable cells, `scatter` and `greedy` once per
    // direction inside it; all are averaged over every chunk of the scene.
    for phase in &phases {
        println!(
            "  phase {:17} {:6.1} us per chunk ({} calls, profiled pass)",
            phase.name,
            phase.total_ns as f64 / 1e3 / inputs.len().max(1) as f64,
            phase.calls
        );
    }
    let surface = samples.iter().filter(|s| s.regular > 0).count();
    (mean_max_us(samples.iter().map(|s| s.fastest)).0, surface)
}

#[test]
#[ignore = "benchmark: run in release with --ignored --nocapture"]
// The benchmark reports to the test output.
#[allow(clippy::print_stdout)]
fn bench_terrain() {
    let scenes = scenes();
    let mut mesher = Mesher::new();
    let mut mesh = ChunkMesh::new();
    let mut means: Vec<(String, f64)> = Vec::new();
    for scene in &scenes {
        let (mean, surface) = run(scene, &mut mesher, &mut mesh);
        // Every scene must be mostly about visible faces, not empty or buried chunks.
        let needed = if means.is_empty() {
            200
        } else {
            scene.chunks.len() / 4
        };
        assert!(
            surface >= needed,
            "{}: {surface} of {} chunks have regular quads, {needed} needed",
            scene.name,
            scene.chunks.len()
        );
        means.push((scene.name.clone(), mean));
    }
    println!("summary (mean of the fastest runs):");
    for (name, mean) in &means {
        let spread = means
            .iter()
            .find(|(other, _)| *other == format!("{name} spread"));
        match spread {
            Some((_, spread)) => println!(
                "  {name:14} {mean:6.1} us, ids x{SPREAD}: {spread:6.1} us ({:.2}x)",
                spread / mean
            ),
            None if !name.ends_with("spread") => println!("  {name:14} {mean:6.1} us"),
            None => {}
        }
    }
}
