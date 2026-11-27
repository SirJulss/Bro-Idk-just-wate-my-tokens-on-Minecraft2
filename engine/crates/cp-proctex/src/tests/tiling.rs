//! Seamless tiling, tested twice: exactly, by painting other windows of the tiled plane, and
//! by measurement, by comparing the step across the wrapped edge with the steps inside.

use std::collections::BTreeSet;
use std::fmt::Write as _;

use cp_core::det;

use super::{Entry, SEED, default, luminance, pack, paint, tones};
use crate::{Params, PbrTexture, generate, generate_window};

/// The texture as a window shows it that starts `origin` texels further along both axes.
fn rolled(texture: &PbrTexture, origin: (i64, i64)) -> PbrTexture {
    let n = i64::from(texture.size);
    let mut out = texture.clone();
    for y in 0..n {
        for x in 0..n {
            let to = (y * n + x) as usize;
            let from = ((y + origin.1).rem_euclid(n) * n + (x + origin.0).rem_euclid(n)) as usize;
            out.albedo[to] = texture.albedo[from];
            out.normal[to] = texture.normal[from];
            out.roughness[to] = texture.roughness[from];
            out.emissive[to] = texture.emissive[from];
            out.height[to] = texture.height[from];
        }
    }
    out
}

/// The exact criterion for a seamless texture: it is a window onto an image that repeats on
/// the whole plane, and every other window onto that image shows the same texels, rolled.
///
/// Painting a shifted window evaluates every pattern at other positions, some of them one or
/// more tiles away, and runs every neighbourhood operation across a different border. If any
/// step knew where the tile ends (a filter that clamps at the edge, a pattern with the wrong
/// period, a clean-up pass that depends on scan order) the shifted image would differ.
#[test]
fn every_window_onto_the_tiled_plane_shows_the_same_image() {
    let windows: [(u32, &[(i64, i64)]); 3] = [
        (
            16,
            &[(1, 0), (0, 1), (5, 11), (-7, 3), (16, -32), (37, 250)],
        ),
        (32, &[(3, -9), (32, 64), (-33, 17)]),
        (64, &[(21, 40)]),
    ];
    for entry in pack() {
        for (size, origins) in windows {
            let base = paint(&entry, SEED, size);
            for &origin in origins {
                let window = generate_window(entry.recipe, &entry.params, SEED, size, origin)
                    .expect("valid request");
                assert!(
                    window == rolled(&base, origin),
                    "{} at {size} px is not the same image in the window at {origin:?}",
                    entry.label
                );
            }
        }
    }
    // The helper itself: rolling by whole tiles changes nothing, rolling by one texel does.
    let stone = default("stone", SEED);
    assert_eq!(rolled(&stone, (16, -48)), stone);
    assert_ne!(rolled(&stone, (1, 0)), stone);
    assert_eq!(rolled(&rolled(&stone, (3, 5)), (-3, -5)), stone);
}

/// How the tile's border relates to the design, along one axis.
#[derive(Clone, Copy, PartialEq, Debug)]
pub(super) enum Border {
    /// The design does not know the border: the step across it has to be a step like any
    /// other.
    Free,
    /// A joint of the design lies on the border, and the same joint occurs inside the tile
    /// (the gaps between the boards of planks, the bark around the end of a log). The step
    /// across the border may be strong, but no stronger than those inside.
    Joint,
    /// Opposite edges differ by design: the lit and the shaded bevel of a metal plate, the
    /// grass above and the soil below the side of a grass block.
    Edge,
}

/// The declared relation of `recipe`'s design to the border between its columns (left and
/// right edge) or between its rows (top and bottom edge).
pub(super) fn border(recipe: &str, columns: bool) -> Border {
    match (recipe, columns) {
        ("metal", _) | ("grass_side", false) => Border::Edge,
        ("planks", false) | ("log_top", _) => Border::Joint,
        _ => Border::Free,
    }
}

/// Mean difference between each line of texels and the next one, summed over colour, alpha
/// and height. Lines are columns or rows; the last entry is the step from the last line
/// across the tile's edge to the first.
fn line_steps(texture: &PbrTexture, columns: bool) -> Vec<f64> {
    let n = texture.size as usize;
    let difference = |a: usize, b: usize| {
        let (ta, tb) = (texture.albedo[a], texture.albedo[b]);
        let color: u32 = (0..4).map(|c| u32::from(ta[c].abs_diff(tb[c]))).sum();
        f64::from(color + u32::from(texture.height[a].abs_diff(texture.height[b])))
    };
    (0..n)
        .map(|line| {
            let next = (line + 1) % n;
            let total: f64 = (0..n)
                .map(|i| {
                    if columns {
                        difference(i * n + line, i * n + next)
                    } else {
                        difference(line * n + i, next * n + i)
                    }
                })
                .sum();
            total / n as f64
        })
        .collect()
}

/// Number of seeds the seam statistics are averaged over.
const SEAM_SEEDS: u64 = 64;

/// How much stronger than an average step inside the tile the step across a free border may
/// be, averaged over the seeds. The average of 64 seeds scatters by a few percent; a pattern
/// that jumps at the border is off by far more (see `the_seam_measure_detects_seams`).
const SEAM_TOLERANCE: f64 = 1.2;

/// Averages over many seeds of: the step across the wrapped edge, the mean step between
/// neighbouring lines inside the tile, and the largest step inside the tile.
pub(super) fn edge_steps(entry: &Entry, columns: bool) -> (f64, f64, f64) {
    let (mut across, mut inside_mean, mut inside_max) = (0.0, 0.0, 0.0);
    for seed in 0..SEAM_SEEDS {
        let steps = line_steps(&paint(entry, seed, 16), columns);
        let (edge, inside) = steps.split_last().expect("16 lines");
        across += edge;
        inside_mean += inside.iter().sum::<f64>() / inside.len() as f64;
        inside_max += inside.iter().copied().fold(0.0, det::max);
    }
    let seeds = SEAM_SEEDS as f64;
    (across / seeds, inside_mean / seeds, inside_max / seeds)
}

/// The measured criterion: across many seeds, the step over the wrapped edge is as large as
/// the steps between neighbouring lines inside the tile, not larger. A pattern that is
/// periodic but jumps at the border (a sawtooth, a layout anchored to the tile) passes the
/// window test above and fails here.
#[test]
fn the_step_across_the_wrapped_edge_is_a_step_like_any_other() {
    let mut failures = String::new();
    for entry in pack() {
        for columns in [true, false] {
            let kind = border(entry.recipe, columns);
            if kind == Border::Edge {
                continue;
            }
            let (across, inside_mean, inside_max) = edge_steps(&entry, columns);
            // Never the strongest line of the tile ...
            let hidden = across <= inside_max;
            // ... and, where the design does not put a joint on the border, an average one.
            let ordinary = kind == Border::Joint || across <= SEAM_TOLERANCE * inside_mean;
            if !hidden || !ordinary {
                let axis = if columns { "columns" } else { "rows" };
                let _ = writeln!(
                    failures,
                    "{} ({axis}): across the edge {across:.1}, inside mean {inside_mean:.1}, \
                     inside max {inside_max:.1}",
                    entry.label
                );
            }
        }
    }
    assert!(failures.is_empty(), "seams:\n{failures}");
}

#[test]
fn the_seam_measure_detects_seams() {
    // Windows of a 32 px texture, which repeats after 32 texels and not after 16: their
    // opposite edges do not match.
    let mut jumps = 0;
    for seed in 0..SEAM_SEEDS {
        let stone = generate("stone", &Params::new(), seed, 32).expect("stone");
        let mut cut = default("stone", seed);
        for y in 0..16 {
            for x in 0..16 {
                let (from, to) = ((y + 9) * 32 + x + 5, y * 16 + x);
                cut.albedo[to] = stone.albedo[from];
                cut.height[to] = stone.height[from];
            }
        }
        for columns in [true, false] {
            let steps = line_steps(&cut, columns);
            let (across, inside) = steps.split_last().expect("16 lines");
            let mean = inside.iter().sum::<f64>() / inside.len() as f64;
            jumps += u32::from(*across > SEAM_TOLERANCE * mean);
        }
    }
    // Nearly every cut-out has a seam on both axes; the real textures average out to none.
    assert!(
        u64::from(jumps) > SEAM_SEEDS * 3 / 2,
        "only {jumps} seams found"
    );
    // Designed borders are declared, not overlooked.
    assert_eq!(border("metal", true), Border::Edge);
    assert_eq!(border("grass_side", true), Border::Free);
    assert_eq!(border("grass_side", false), Border::Edge);
    assert_eq!(border("planks", true), Border::Free);
    assert_eq!(border("planks", false), Border::Joint);
    assert_eq!(border("stone", false), Border::Free);
}

#[test]
fn designed_borders_are_what_they_claim_to_be() {
    for seed in 0..8 {
        // Metal: the left and top edges are the lit bevel, the right and bottom the shaded.
        let metal = default("metal", seed);
        let at = |x: usize, y: usize| {
            let [r, g, b, _] = metal.albedo[y * 16 + x];
            luminance([r, g, b])
        };
        for i in 1..15 {
            assert!(at(0, i) > at(15, i) && at(i, 0) > at(i, 15), "seed {seed}");
        }
        // Planks: every fourth row is a joint, on the border as inside the tile.
        let planks = default("planks", seed);
        let darkest = tones(&planks)[0];
        for row in [3, 7, 11, 15] {
            for x in 0..16 {
                assert_eq!(planks.albedo[row * 16 + x][..3], darkest, "seed {seed}");
            }
        }
        // Log end: bark on all four edges, wood in the middle.
        let log = default("log_top", seed);
        let bark: BTreeSet<[u8; 3]> = (0..16)
            .flat_map(|i| [i, 15 * 16 + i, i * 16, i * 16 + 15])
            .map(|texel| {
                let [r, g, b, _] = log.albedo[texel];
                [r, g, b]
            })
            .collect();
        assert!(bark.len() <= 2, "seed {seed}: {bark:?}");
        let [r, g, b, _] = log.albedo[8 * 16 + 8];
        assert!(!bark.contains(&[r, g, b]), "seed {seed}");
    }
}
