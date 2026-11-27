//! Palettes and value ranges: few colours, soft ones, and channels that stay usable.

use cp_core::det;

use super::{SEED, luminance, pack, paint, palette};
use crate::color::rgb8_to_oklab;
use crate::{ALPHA_CUTOFF, ParamKind, Params, decode_normal, generate, recipe_info, recipes};

#[test]
fn palettes_have_four_to_six_tones() {
    for entry in pack() {
        for seed in 0..24 {
            for size in [16, 32] {
                let count = palette(&paint(&entry, seed, size)).len();
                assert!(
                    (4..=6).contains(&count),
                    "{} (seed {seed}, {size} px) has {count} colours",
                    entry.label
                );
            }
        }
    }
}

#[test]
fn palettes_do_not_depend_on_seed_or_size() {
    // The colours belong to the material: other seeds and sizes only rearrange them.
    for entry in pack() {
        let reference = palette(&paint(&entry, SEED, 16));
        for (seed, size) in [(SEED, 64), (7, 16), (99, 32)] {
            let other = palette(&paint(&entry, seed, size));
            assert!(
                other.is_subset(&reference) || reference.is_subset(&other),
                "{}: {reference:?} vs {other:?}",
                entry.label
            );
        }
    }
}

#[test]
fn no_palette_is_a_grey_ramp_and_none_is_neon() {
    for entry in pack() {
        let colors = palette(&paint(&entry, SEED, 16));
        // Colour temperature in every ramp: at most one tone may be a pure grey.
        let greys = colors
            .iter()
            .filter(|c| c[0] == c[1] && c[1] == c[2])
            .count();
        assert!(greys <= 1, "{}: {colors:?}", entry.label);
        // Soft colours: chroma stays well below that of the sRGB primaries (0.26 to 0.32).
        // Only what glows may be vivid.
        let limit = if entry.recipe == "lava" { 0.2 } else { 0.14 };
        for &color in &colors {
            let lab = rgb8_to_oklab(color);
            let chroma = (lab.a * lab.a + lab.b * lab.b).sqrt();
            assert!(
                chroma <= limit,
                "{}: {color:?} has chroma {chroma:.3}",
                entry.label
            );
        }
        // Readable but not harsh: the tones span a clear range of lightness without reaching
        // black or white.
        let lightness: Vec<f64> = colors.iter().map(|&c| rgb8_to_oklab(c).l).collect();
        let darkest = lightness.iter().copied().fold(f64::INFINITY, det::min);
        let lightest = lightness.iter().copied().fold(0.0, det::max);
        assert!(
            (0.1..=0.65).contains(&(lightest - darkest)),
            "{}: lightness {darkest:.2} to {lightest:.2}",
            entry.label
        );
        assert!(
            darkest > 0.15 && lightest < 0.99,
            "{}: {lightness:?}",
            entry.label
        );
    }
}

/// The requests the value ranges are checked on: many seeds at the native size, a few at the
/// larger ones, where relief would turn steep if it were measured in texels.
const RANGE_CHECKS: [(u64, u32); 13] = [
    (0, 16),
    (1, 16),
    (2, 16),
    (3, 16),
    (4, 16),
    (5, 16),
    (6, 16),
    (7, 16),
    (0, 32),
    (1, 32),
    (2, 64),
    (3, 64),
    (1, 256),
];

#[test]
fn channel_values_stay_in_range() {
    for entry in pack() {
        for (seed, size) in RANGE_CHECKS {
            let texture = paint(&entry, seed, size);
            let label = &format!("{} (seed {seed}, {size} px)", entry.label);
            for (texel, albedo) in texture.albedo.iter().enumerate() {
                // Alpha is binary at the base level, and only cutout recipes have holes.
                assert!(albedo[3] == 0 || albedo[3] == 255, "{label}");
                assert!(entry.cutout || albedo[3] == 255, "{label}");
                // Never black, never white: lighting needs room in both directions.
                let light = luminance([albedo[0], albedo[1], albedo[2]]);
                assert!((0.004..=0.95).contains(&light), "{label}: {albedo:?}");
                // Normals are unit vectors that point out of the surface, none of them steeper
                // than about 70 degrees.
                let [nx, ny, nz] = decode_normal(texture.normal[texel]);
                assert!(texture.normal[texel][0] >= 1 && texture.normal[texel][1] >= 1);
                assert!(
                    (nx * nx + ny * ny + nz * nz - 1.0).abs() < 1.0e-4,
                    "{label}"
                );
                assert!(nz >= 0.34, "{label}: {nx} {ny} {nz}");
                // Roughness never reaches the mirror, where specular highlights degenerate.
                assert!((12..=250).contains(&texture.roughness[texel]), "{label}");
                // Only lava emits light.
                assert!(
                    entry.recipe == "lava" || texture.emissive[texel] == 0,
                    "{label}"
                );
            }
            // The height map uses its range: a flat map would make the normals pointless.
            let lowest = texture.height.iter().copied().min().unwrap_or(0);
            let highest = texture.height.iter().copied().max().unwrap_or(0);
            assert!(
                highest - lowest >= 40,
                "{label}: height {lowest} to {highest}"
            );
            // And the normal map is not flat either.
            assert!(texture.normal.iter().any(|&n| n != [128, 128]), "{label}");
        }
    }
}

/// Colours at the edges of what a pack may pass in: black, white, the primaries at full
/// strength, a dark saturated blue and an obsidian black.
const EXTREME_COLOURS: [[u8; 3]; 10] = [
    [0, 0, 0],
    [255, 255, 255],
    [255, 0, 0],
    [0, 255, 0],
    [0, 0, 255],
    [255, 255, 0],
    [0, 255, 255],
    [0, 0, 40],
    [20, 20, 24],
    [250, 250, 235],
];

#[test]
fn caller_colours_cannot_make_black_or_white() {
    // A caller's colour goes through the same ramps as the defaults: whatever a pack passes
    // in, no texel is black or white.
    for name in recipes() {
        let info = recipe_info(name).expect("listed recipe");
        let styles: Vec<Option<&str>> = match info.params.iter().find_map(|p| match p.kind {
            ParamKind::Variant(names) => Some(names),
            ParamKind::Color => None,
        }) {
            Some(names) => names.iter().map(|&name| Some(name)).collect(),
            None => vec![None],
        };
        for param in info.params.iter().filter(|p| p.kind == ParamKind::Color) {
            for style in &styles {
                for color in EXTREME_COLOURS {
                    let mut params = Params::new().with_color(param.key, color);
                    if let Some(style) = style {
                        params = params.with_variant("style", style);
                    }
                    let texture = generate(name, &params, SEED, 16).expect("valid request");
                    for texel in texture.albedo.iter().filter(|t| t[3] >= ALPHA_CUTOFF) {
                        let light = luminance([texel[0], texel[1], texel[2]]);
                        assert!(
                            (0.004..=0.95).contains(&light),
                            "{name}.{} = {color:?} ({style:?}): {texel:?} has luminance {light}",
                            param.key
                        );
                    }
                }
            }
        }
    }
}
