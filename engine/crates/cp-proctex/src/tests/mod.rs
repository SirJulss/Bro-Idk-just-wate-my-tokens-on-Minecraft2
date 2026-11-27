//! Tests of the pack as a whole: every recipe and preset against the guarantees of the crate.
//!
//! The modules follow those guarantees: valid requests and determinism ([`requests`]),
//! seamless tiling ([`tiling`]), small palettes and sane value ranges ([`palettes`]),
//! consistent channels ([`channels`]), cutout coverage ([`cutout`]), the tint convention
//! ([`tinting`]) and the relations between textures and their parameters ([`families`]).
//! [`review`] writes the images and numbers the art was judged by.

mod channels;
mod cutout;
mod families;
mod palettes;
mod requests;
mod review;
mod tiling;
mod tinting;

use std::collections::BTreeSet;

use crate::color::rgb8_to_linear;
use crate::{
    ALPHA_CUTOFF, Params, PbrTexture, Tint, generate, presets, recipe_info, recipes, tint_weight,
};

/// Seed of the pinned textures and of the tests that look at a single seed.
const SEED: u64 = 1;

/// One texture of the pack: a recipe with default parameters, or a preset.
struct Entry {
    label: String,
    recipe: &'static str,
    params: Params,
    tint: Tint,
    cutout: bool,
}

/// Every recipe with default parameters, then every preset.
fn pack() -> Vec<Entry> {
    let mut entries = Vec::new();
    for name in recipes() {
        let info = recipe_info(name).expect("listed recipe");
        entries.push(Entry {
            label: (*name).to_owned(),
            recipe: name,
            params: Params::new(),
            tint: info.tint,
            cutout: info.cutout,
        });
    }
    for preset in presets() {
        let info = recipe_info(preset.recipe).expect("preset of a listed recipe");
        entries.push(Entry {
            label: preset.name.to_owned(),
            recipe: preset.recipe,
            params: preset.to_params(),
            tint: preset.tint,
            cutout: info.cutout,
        });
    }
    entries
}

fn paint(entry: &Entry, seed: u64, size: u32) -> PbrTexture {
    generate(entry.recipe, &entry.params, seed, size)
        .unwrap_or_else(|error| panic!("{}: {error}", entry.label))
}

/// A recipe with default parameters.
fn default(recipe: &str, seed: u64) -> PbrTexture {
    generate(recipe, &Params::new(), seed, 16).unwrap_or_else(|error| panic!("{recipe}: {error}"))
}

/// Colours of the texels that are not cut out.
fn visible(texture: &PbrTexture) -> impl Iterator<Item = [u8; 3]> + '_ {
    (texture.albedo.iter())
        .filter(|texel| texel[3] >= ALPHA_CUTOFF)
        .map(|&[r, g, b, _]| [r, g, b])
}

fn palette(texture: &PbrTexture) -> BTreeSet<[u8; 3]> {
    visible(texture).collect()
}

/// Relative luminance (Rec. 709) of an sRGB colour.
fn luminance(rgb: [u8; 3]) -> f64 {
    let [r, g, b] = rgb8_to_linear(rgb);
    0.2126 * r + 0.7152 * g + 0.0722 * b
}

/// The visible colours of a texture from dark to light.
fn tones(texture: &PbrTexture) -> Vec<[u8; 3]> {
    let mut colors: Vec<[u8; 3]> = palette(texture).into_iter().collect();
    colors.sort_by(|a, b| luminance(*a).total_cmp(&luminance(*b)));
    colors
}

/// Share of the texels that are not cut out.
fn coverage(texture: &PbrTexture) -> f64 {
    visible(texture).count() as f64 / texture.albedo.len() as f64
}

/// How much of a tint an sRGB colour takes.
fn weight(rgb: [u8; 3]) -> f32 {
    tint_weight(rgb8_to_linear(rgb).map(|c| c as f32))
}

#[test]
fn the_pack_is_every_recipe_and_every_preset() {
    let entries = pack();
    assert_eq!(entries.len(), recipes().len() + presets().len());
    assert_eq!(entries[0].label, "stone");
    assert!(
        entries
            .iter()
            .all(|entry| recipes().contains(&entry.recipe))
    );
    // The helpers agree with each other on a texture with holes.
    let leaves = default("leaves", SEED);
    assert!(coverage(&leaves) < 1.0);
    assert_eq!(tones(&leaves).len(), palette(&leaves).len());
    assert!(luminance([255, 255, 255]) > 0.999 && luminance([0, 0, 0]) == 0.0);
    assert_eq!(weight([128, 128, 128]), 1.0);
}
