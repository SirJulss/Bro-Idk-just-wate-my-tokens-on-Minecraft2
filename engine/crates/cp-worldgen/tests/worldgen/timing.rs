//! Release benchmark: generation time per padded chunk (34³ cells, what the mesher reads) at
//! levels 0–3, for surface chunks and for chunks deep below the ground, and the cost of
//! `classify` (contract §9 target: generation <= 3 ms per chunk).
//!
//! Not part of the normal test run; start it explicitly in a release build:
//! `cargo test -p cp-worldgen --release --test worldgen -- --ignored --nocapture timing`.
//! Timings taken while other builds run on the machine are not reliable.

use std::hint::black_box;
use std::time::Instant;

use cp_core::coords::LodKey;
use cp_core::ids::BlockStateId;
use cp_core::math::IVec3;
use cp_voxel::{BaseSource, padded_cells, region_volume};

use crate::common::{hills, structures};

/// Surface chunks of `level` along a line through hills, forests, lake shores and ridges.
fn surface_keys(source: &cp_worldgen::HillsSource, level: u8) -> Vec<LodKey> {
    (0..64)
        .map(|i| {
            let (x, z) = (i * 97 - 3000, (i * 53) % 2000 - 1000);
            LodKey::containing(level, IVec3::new(x, source.ground_top(x, z), z))
        })
        .collect()
}

/// The same keys moved by `chunks` chunks along y.
fn shifted(keys: &[LodKey], chunks: i32) -> Vec<LodKey> {
    keys.iter()
        .map(|key| LodKey::new(key.level, key.pos + IVec3::Y * chunks))
        .collect()
}

/// Mean milliseconds per call of `work` over `keys`; the best of five passes.
fn best_mean_ms(keys: &[LodKey], mut work: impl FnMut(LodKey)) -> f64 {
    let mut best = f64::INFINITY;
    for _ in 0..5 {
        let start = Instant::now();
        for &key in keys {
            work(key);
        }
        let mean = start.elapsed().as_secs_f64() * 1.0e3 / keys.len() as f64;
        best = cp_core::det::min(best, mean);
    }
    best
}

#[test]
#[ignore = "benchmark: run in release with -- --ignored --nocapture timing"]
// The report is the point of a benchmark; nothing else in the crate prints.
#[allow(clippy::print_stdout)]
fn timing_padded_chunks() {
    let source = hills(42);
    let mut out = vec![BlockStateId::AIR; region_volume(padded_cells(LodKey::lod0(IVec3::ZERO)))];
    println!("cp-worldgen timing (release build expected), hills seed 42, 64 chunks per row:");
    for level in 0..=3_u8 {
        let keys = surface_keys(&source, level);
        let mut fill = |keys: &[LodKey]| {
            best_mean_ms(keys, |key| {
                source
                    .fill(level, padded_cells(key), &mut out, &|| false)
                    .unwrap();
                black_box(&out);
            })
        };
        // Surface chunks, and chunks four chunks below them (deep stone).
        let (surface_fill, deep_fill) = (fill(&keys), fill(&shifted(&keys, -4)));
        let classify = |keys: &[LodKey]| {
            best_mean_ms(keys, |key| {
                black_box(source.classify(key));
            })
        };
        // The surface chunks, the chunks right above them (empty unless a tree or the
        // neighbouring ground reaches in) and the sky four chunks up.
        let surface = classify(&keys);
        let above = classify(&shifted(&keys, 1));
        let sky = classify(&shifted(&keys, 4));
        println!(
            "  level {level}: fill {surface_fill:.3} ms per padded chunk (surface) / \
             {deep_fill:.3} ms (deep); classify {surface:.4} ms (surface) / {above:.4} ms \
             (one chunk up) / {sky:.4} ms (sky)"
        );
    }
    let field = structures();
    let keys: Vec<LodKey> = (0..4).map(|x| LodKey::lod0(IVec3::new(x, 0, 0))).collect();
    let fill = best_mean_ms(&keys, |key| {
        field
            .fill(0, padded_cells(key), &mut out, &|| false)
            .unwrap();
        black_box(&out);
    });
    println!("  structure field level 0: fill {fill:.3} ms per padded chunk");
}
