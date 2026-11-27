//! Cutout textures and mip chains: coverage that is believable at the base level and stays
//! the same further down.

use super::{SEED, coverage, pack, paint, tones};
use crate::{AlphaMode, average_color, mip_chain, recipe_info};

#[test]
fn leaves_keep_a_believable_coverage() {
    let cutouts: Vec<_> = pack().into_iter().filter(|entry| entry.cutout).collect();
    assert_eq!(cutouts.len(), 5, "leaves and four species");
    for entry in &cutouts {
        let mut total = 0.0;
        for seed in 0..32 {
            let texture = paint(entry, seed, 16);
            let covered = coverage(&texture);
            // Dense enough to read as a crown, open enough to see through.
            assert!(
                (0.66..=0.9).contains(&covered),
                "{} seed {seed}: {covered}",
                entry.label
            );
            total += covered;
            // Holes and leaves come in clumps: no texel stands alone.
            let open = |x: usize, y: usize| texture.albedo[(y % 16) * 16 + x % 16][3] == 0;
            for y in 0..16 {
                for x in 0..16 {
                    let here = open(x, y);
                    let alone = [(x + 1, y), (x + 15, y), (x, y + 1), (x, y + 15)]
                        .iter()
                        .all(|&(nx, ny)| open(nx, ny) != here);
                    assert!(
                        !alone,
                        "{} seed {seed}: lone texel at {x}, {y}",
                        entry.label
                    );
                }
            }
            // The colour hidden in the holes is the darkest tone of the leaves.
            let darkest = tones(&texture)[0];
            for texel in texture.albedo.iter().filter(|texel| texel[3] == 0) {
                assert_eq!(texel[..3], darkest, "{} seed {seed}", entry.label);
            }
        }
        let mean = total / 32.0;
        assert!(
            (0.72..=0.86).contains(&mean),
            "{}: mean coverage {mean}",
            entry.label
        );
    }
    // Everything else is solid.
    for entry in pack().iter().filter(|entry| !entry.cutout) {
        assert_eq!(coverage(&paint(entry, SEED, 16)), 1.0, "{}", entry.label);
    }
}

#[test]
fn mips_of_cutout_textures_keep_the_coverage_of_the_base_level() {
    for entry in pack().iter().filter(|entry| entry.cutout) {
        for seed in 0..32 {
            for size in [16, 64] {
                let base = paint(entry, seed, size);
                let wanted = coverage(&base);
                let chain = mip_chain(&base, AlphaMode::Coverage);
                assert_eq!(chain.len(), size.ilog2() as usize + 1);
                for level in &chain[1..] {
                    // Exact to the resolution of the level: at most half a texel off.
                    let tolerance = 0.5 / level.albedo.len() as f64 + 1.0e-12;
                    let error = (coverage(level) - wanted).abs();
                    assert!(
                        error <= tolerance,
                        "{} seed {seed}, {} px level: coverage {} instead of {wanted}",
                        entry.label,
                        level.size,
                        coverage(level)
                    );
                }
            }
        }
    }
}

#[test]
fn mip_chains_keep_colour_and_solidity() {
    for entry in pack() {
        let base = paint(&entry, SEED, 16);
        let mode = recipe_info(entry.recipe)
            .expect("listed recipe")
            .alpha_mode();
        let chain = mip_chain(&base, mode);
        let sizes: Vec<u32> = chain.iter().map(|level| level.size).collect();
        assert_eq!(sizes, [16, 8, 4, 2, 1], "{}", entry.label);
        assert_eq!(chain[0], base);
        let average = average_color(&base);
        for level in &chain[1..] {
            if !entry.cutout {
                assert!(
                    level.albedo.iter().all(|texel| texel[3] == 255),
                    "{}",
                    entry.label
                );
            }
            // Every level has the colour of the base level. Cutout levels weigh their texels
            // by the rescaled alpha, hence the wider tolerance.
            let tolerance = if entry.cutout { 0.06 } else { 0.012 };
            let color = average_color(level);
            for channel in 0..3 {
                assert!(
                    (color[channel] - average[channel]).abs() < tolerance,
                    "{} at {} px: {color:?} instead of {average:?}",
                    entry.label,
                    level.size
                );
            }
            // Emission survives the averaging: what glows up close glows from afar.
            let glows = |texture: &crate::PbrTexture| texture.emissive.iter().any(|&e| e > 0);
            assert_eq!(glows(level), glows(&base), "{}", entry.label);
        }
    }
}
