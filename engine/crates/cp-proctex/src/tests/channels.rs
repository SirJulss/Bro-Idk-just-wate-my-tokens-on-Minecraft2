//! Consistency of the channels: normals, height, roughness and emission describe the surface
//! the albedo shows.

use std::collections::BTreeMap;
use std::fmt::Write as _;

use cp_core::hash::Rng;

use super::{SEED, default, luminance, pack, paint, tones};
use crate::{Params, PbrTexture, decode_normal, generate};

#[test]
fn normals_are_the_slope_of_the_height_map() {
    for entry in pack() {
        for size in [16_usize, 32] {
            let texture = paint(&entry, SEED, size as u32);
            let height =
                |x: usize, y: usize| i32::from(texture.height[(y % size) * size + x % size]);
            let mut tilted = 0;
            for y in 0..size {
                for x in 0..size {
                    // Rise towards image right and towards image up.
                    let rise_u = height(x + 1, y) - height(x + size - 1, y);
                    let rise_v = height(x, y + size - 1) - height(x, y + 1);
                    let [nx, ny] = texture.normal[y * size + x].map(|byte| i32::from(byte) - 128);
                    // The normal leans away from higher ground and is upright on level ground.
                    assert!(
                        nx == 0 || nx.signum() == -rise_u.signum(),
                        "{}",
                        entry.label
                    );
                    assert!(
                        ny == 0 || ny.signum() == -rise_v.signum(),
                        "{}",
                        entry.label
                    );
                    assert!(rise_u != 0 || nx == 0, "{}", entry.label);
                    assert!(rise_v != 0 || ny == 0, "{}", entry.label);
                    tilted += u32::from(nx != 0 || ny != 0);
                }
            }
            assert!(
                tilted as usize > size * size / 10,
                "{}: only {tilted} tilted normals",
                entry.label
            );
        }
    }
}

/// Share of the height map's variance that the albedo explains: 1 when every colour has
/// exactly one height, 0 when knowing the colour of a texel says nothing about its height.
pub(super) fn height_explained_by_colour(texture: &PbrTexture) -> f64 {
    // Per colour: count, sum and sum of squares of the heights.
    let mut groups: BTreeMap<[u8; 4], (f64, f64, f64)> = BTreeMap::new();
    for (albedo, &height) in texture.albedo.iter().zip(&texture.height) {
        let group = groups.entry(*albedo).or_insert((0.0, 0.0, 0.0));
        let h = f64::from(height);
        *group = (group.0 + 1.0, group.1 + h, group.2 + h * h);
    }
    let (count, sum, squares) = groups
        .values()
        .fold((0.0, 0.0, 0.0), |a, g| (a.0 + g.0, a.1 + g.1, a.2 + g.2));
    let total = squares - sum * sum / count;
    let within: f64 = groups.values().map(|g| g.2 - g.1 * g.1 / g.0).sum();
    if total > 0.0 {
        1.0 - within / total
    } else {
        1.0
    }
}

/// Number of seeds [`height_explained_by_colour`] is averaged over.
pub(super) const RELIEF_SEEDS: u64 = 16;

#[test]
fn the_height_map_follows_the_albedo() {
    // The relief and the painted tones describe the same shapes: in every texture the tones
    // account for a good part of the variation in height. (The rest is slope: textures that
    // are shaded like a relief are lighter where they face the light, at any height.)
    let mut failures = String::new();
    for entry in pack() {
        let explained: f64 = (0..RELIEF_SEEDS)
            .map(|seed| height_explained_by_colour(&paint(&entry, seed, 16)))
            .sum::<f64>()
            / RELIEF_SEEDS as f64;
        if explained < 0.45 {
            let _ = writeln!(failures, "{}: {explained:.2}", entry.label);
        }
    }
    assert!(
        failures.is_empty(),
        "height not explained by colour:\n{failures}"
    );

    // The measure itself: a height map unrelated to the colours explains nothing.
    let mut scrambled = default("cobble", SEED);
    let honest = height_explained_by_colour(&scrambled);
    let mut rng = Rng::new(5, 5);
    for height in &mut scrambled.height {
        *height = rng.range_u32(0, 256) as u8;
    }
    let random = height_explained_by_colour(&scrambled);
    assert!(honest > 0.7 && random < 0.15, "{honest} {random}");
}

#[test]
fn roughness_and_emission_belong_to_the_colour() {
    for entry in pack() {
        for seed in 0..4 {
            let texture = paint(&entry, seed, 16);
            let mut surface: BTreeMap<[u8; 3], (u8, u8)> = BTreeMap::new();
            for (texel, &[r, g, b, _]) in texture.albedo.iter().enumerate() {
                let properties = (texture.roughness[texel], texture.emissive[texel]);
                let known = *surface.entry([r, g, b]).or_insert(properties);
                assert_eq!(known, properties, "{}: {:?}", entry.label, [r, g, b]);
            }
        }
    }
}

#[test]
fn roughness_suits_the_material() {
    let mean = |recipe: &str| {
        let texture = default(recipe, SEED);
        texture.roughness.iter().map(|&r| f64::from(r)).sum::<f64>() / (255.0 * 256.0)
    };
    for glossy in ["water", "ice"] {
        assert!(mean(glossy) < 0.25, "{glossy}: {}", mean(glossy));
    }
    assert!((0.25..0.5).contains(&mean("metal")), "{}", mean("metal"));
    for rough in [
        "stone",
        "cobble",
        "gravel",
        "dirt",
        "grass_top",
        "sand",
        "log_side",
    ] {
        assert!(mean(rough) > 0.8, "{rough}: {}", mean(rough));
    }
    // Within a texture: ore is smoother than the rock around it.
    let ore = default("ore", SEED);
    let smoothest = ore.roughness.iter().copied().min().unwrap_or(0);
    let roughest = ore.roughness.iter().copied().max().unwrap_or(0);
    assert!(smoothest < 100 && roughest > 200, "{smoothest} {roughest}");
    let brightest = (0..256)
        .max_by(|&a, &b| {
            let light = |texel: usize| {
                let [r, g, b, _] = ore.albedo[texel];
                luminance([r, g, b])
            };
            light(a).total_cmp(&light(b))
        })
        .expect("256 texels");
    assert_eq!(ore.roughness[brightest], smoothest);
}

#[test]
fn lava_glows_where_it_is_bright() {
    for seed in 0..8 {
        let lava = default("lava", seed);
        // Per texel: luminance, emission, roughness.
        let mut tones: Vec<(f64, u8, u8)> = lava
            .albedo
            .iter()
            .enumerate()
            .map(|(texel, &[r, g, b, _])| {
                (
                    luminance([r, g, b]),
                    lava.emissive[texel],
                    lava.roughness[texel],
                )
            })
            .collect();
        tones.sort_by(|a, b| a.0.total_cmp(&b.0));
        tones.dedup_by_key(|tone| (tone.1, tone.2));
        assert_eq!(tones.len(), 5, "seed {seed}");
        // Brighter tones glow more and are smoother: melt against crust.
        for pair in tones.windows(2) {
            assert!(pair[1].1 > pair[0].1 && pair[1].2 < pair[0].2, "{tones:?}");
        }
        assert!(tones[0].1 < 40 && tones[4].1 == 255, "{tones:?}");
        // The crust floats on the melt: what glows lies lower.
        let mean_height = |glowing: bool| {
            let heights: Vec<f64> = (0..256)
                .filter(|&texel| (lava.emissive[texel] > 128) == glowing)
                .map(|texel| f64::from(lava.height[texel]))
                .collect();
            heights.iter().sum::<f64>() / heights.len() as f64
        };
        assert!(mean_height(false) > mean_height(true) + 20.0, "seed {seed}");
    }
}

/// The pack's light in tangent space (x image right, y image up, z out of the surface): from
/// the upper left and above, `(-1, 1, 1) / sqrt(3)`.
const LIGHT: [f64; 3] = [
    -0.577_350_269_189_625_8,
    0.577_350_269_189_625_8,
    0.577_350_269_189_625_8,
];

/// How much more light than a flat texel the texels painted in `color` catch on average:
/// the mean of `N.L - L.z`.
fn extra_light(texture: &PbrTexture, color: [u8; 3]) -> f64 {
    let (mut sum, mut count) = (0.0, 0.0);
    for (albedo, &normal) in texture.albedo.iter().zip(&texture.normal) {
        if albedo[..3] == color {
            let n = decode_normal(normal).map(f64::from);
            sum += n[0] * LIGHT[0] + n[1] * LIGHT[1] + n[2] * LIGHT[2] - LIGHT[2];
            count += 1.0;
        }
    }
    sum / count
}

#[test]
fn painted_light_and_relief_agree() {
    // Where the albedo paints light on a raised detail (the lit rims and rivet heads of metal,
    // the ledges of slate, the dashes of smooth bark, the pebbles of dirt) the normals face
    // the pack's light, and the shadow painted beside it faces away. Heights taken from the
    // tones got this backwards: a shadow became a pit that tilted its highlight away.
    let smooth = Params::new().with_variant("style", "smooth");
    let cases = [
        ("metal", Params::new(), true),
        ("slate", Params::new(), true),
        ("log_side", smooth, true),
        // The darkest tone of dirt is mostly the soil's pits, not the pebbles' shadows.
        ("dirt", Params::new(), false),
    ];
    for (recipe, params, check_shadow) in cases {
        for size in [16, 64] {
            let (mut lit, mut shaded) = (0.0, 0.0);
            for seed in 0..RELIEF_SEEDS {
                let texture = generate(recipe, &params, seed, size).expect("valid request");
                let ramp = tones(&texture);
                lit += extra_light(&texture, ramp[ramp.len() - 1]);
                shaded += extra_light(&texture, ramp[0]);
            }
            let (lit, shaded) = (lit / RELIEF_SEEDS as f64, shaded / RELIEF_SEEDS as f64);
            assert!(lit > 0.01, "{recipe}, {size} px: lightest tone {lit:+.3}");
            assert!(
                !check_shadow || shaded < -0.01,
                "{recipe}, {size} px: darkest tone {shaded:+.3}"
            );
        }
    }
}

/// Mean steepness of a normal map, `1 - n.z`: grows with the square of the slopes, so it
/// shows relief that turned into cliffs even where the mean slope stays the same.
fn steepness(texture: &PbrTexture) -> f64 {
    let sum: f64 = (texture.normal.iter())
        .map(|&normal| 1.0 - f64::from(decode_normal(normal)[2]))
        .sum();
    sum / texture.normal.len() as f64
}

#[test]
fn larger_sizes_show_the_same_relief() {
    // A larger texture samples the 16 px design more finely; its relief is no steeper.
    let mut failures = String::new();
    for entry in pack() {
        let native = steepness(&paint(&entry, SEED, 16));
        for size in [32, 64, 256] {
            let ratio = steepness(&paint(&entry, SEED, size)) / native;
            if !(0.5..=2.0).contains(&ratio) {
                let _ = writeln!(failures, "{} at {size} px: {ratio:.2} x", entry.label);
            }
        }
    }
    assert!(failures.is_empty(), "relief changes with size:\n{failures}");
}
