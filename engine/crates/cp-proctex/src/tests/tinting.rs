//! The tint convention: tinted textures are neutral exactly where the tint belongs.

use cp_core::hash::Rng;

use super::{SEED, default, pack, paint, palette, visible, weight};
use crate::tint::takes_full_tint;
use crate::{Params, Tint, generate};

#[test]
fn tinted_textures_are_fully_neutral_where_they_are_tinted() {
    for entry in pack().iter().filter(|entry| entry.tint != Tint::None) {
        for seed in 0..8 {
            let texture = paint(entry, seed, 16);
            for color in visible(&texture) {
                let w = weight(color);
                // No texel is half tinted ...
                assert!(
                    w == 0.0 || w == 1.0,
                    "{}: {color:?} has weight {w}",
                    entry.label
                );
                // ... and only the side of a grass block and the ground covers (flowers,
                // fallen leaves, straw in the turf) have texels that keep their colour.
                let cover = entry.recipe == "grass_top"
                    && entry
                        .params
                        .variant("style")
                        .is_some_and(|style| style != "turf");
                assert!(
                    w == 1.0 || entry.recipe == "grass_side" || cover,
                    "{}: {color:?} is not neutral",
                    entry.label
                );
            }
        }
    }
    // The pack says which textures these are.
    let tinted: Vec<String> = pack()
        .into_iter()
        .filter(|entry| entry.tint != Tint::None)
        .map(|entry| entry.label)
        .collect();
    assert_eq!(
        tinted,
        [
            "grass_top",
            "grass_side",
            "leaves",
            "water",
            "oak_leaves",
            "birch_leaves",
            "spruce_leaves",
            "flowers_grass_top",
            "blossoms_grass_top",
            "litter_grass_top",
            "straw_grass_top"
        ]
    );
    // A baked colour is not neutral: cherry leaves would come out wrong if they were tinted,
    // which is why their preset says they are not.
    let cherry = pack()
        .into_iter()
        .find(|entry| entry.label == "cherry_leaves")
        .expect("preset");
    assert_eq!(cherry.tint, Tint::None);
    assert!(visible(&paint(&cherry, SEED, 16)).any(|color| weight(color) < 1.0));
}

#[test]
fn any_near_neutral_colour_gives_a_fully_tintable_texture() {
    // What the leaf presets rely on: a near-neutral leaf, grass or water colour (lighter,
    // darker, warmer or cooler than the default) still takes the whole tint in every tone.
    let mut rng = Rng::new(77, 1);
    let mut tried = 0;
    while tried < 60 {
        let level = rng.range_u32(110, 236);
        let mut channel = || (level + rng.range_u32(0, 31) - 15) as u8;
        let color = [channel(), channel(), channel()];
        if !takes_full_tint(color) {
            continue;
        }
        tried += 1;
        for (recipe, key) in [
            ("grass_top", "grass"),
            ("grass_side", "grass"),
            ("leaves", "leaf"),
            ("water", "color"),
        ] {
            let params = Params::new().with_color(key, color);
            let texture = generate(recipe, &params, SEED, 16).expect("valid request");
            for tone in palette(&texture) {
                let w = weight(tone);
                assert!(
                    w == 1.0 || (recipe == "grass_side" && w == 0.0),
                    "{recipe} from {color:?}: {tone:?} has weight {w}"
                );
            }
        }
    }
}

#[test]
fn the_side_of_a_grass_block_is_tinted_in_its_fringe_only() {
    for seed in 0..24 {
        let side = default("grass_side", seed);
        let tinted = |x: usize, y: usize| {
            let [r, g, b, _] = side.albedo[y * 16 + x];
            weight([r, g, b]) == 1.0
        };
        let mut fringe = 0;
        for x in 0..16 {
            // Grass on top, soil below, and no soil above grass within a column.
            assert!(tinted(x, 0) && tinted(x, 1), "seed {seed}");
            assert!((8..16).all(|y| !tinted(x, y)), "seed {seed}");
            let depth = (0..16).take_while(|&y| tinted(x, y)).count();
            assert!(
                (depth..16).all(|y| !tinted(x, y)),
                "seed {seed}, column {x}"
            );
            fringe += depth;
        }
        // Three to six rows on average: deep enough that the steps of a hillside stay green
        // from above (beauty review E2, round 1), still a fringe over the soil.
        assert!(
            (48..=96).contains(&fringe),
            "seed {seed}: {fringe} grass texels"
        );
        // The fringe is painted with tones of the top of the block, so that the two meet in
        // one colour whatever the tint.
        let top = palette(&default("grass_top", seed));
        for color in visible(&side).filter(|&color| weight(color) == 1.0) {
            assert!(top.contains(&color), "seed {seed}: {color:?}");
        }
    }
    // With a baked grass colour nothing is neutral any more.
    let baked = Params::new().with_color("grass", [96, 152, 60]);
    let side = generate("grass_side", &baked, SEED, 16).expect("grass_side");
    assert!(visible(&side).all(|color| weight(color) == 0.0));
}
