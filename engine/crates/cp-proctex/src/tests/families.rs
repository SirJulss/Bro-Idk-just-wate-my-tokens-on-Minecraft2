//! Relations between textures: recipes that continue each other, parameters and what they
//! change, presets of one species, and the same design at other sizes.

use std::collections::BTreeSet;

use cp_core::det;

use super::{SEED, default, pack, paint, palette, tones};
use crate::color::rgb8_to_oklab;
use crate::{ParamKind, Params, Tint, average_color, generate, presets, recipe_info, recipes};

#[test]
fn the_soil_of_a_grass_block_continues_into_dirt() {
    let cases = [
        (0, Params::new()),
        (1, Params::new()),
        (2, Params::new()),
        (3, Params::new().with_color("dirt", [96, 84, 110])),
    ];
    for (seed, params) in cases {
        let side = generate("grass_side", &params, seed, 16).expect("grass_side");
        let soil = match params.color("dirt") {
            Some(color) => Params::new().with_color("color", color),
            None => Params::new(),
        };
        let dirt = generate("dirt", &soil, seed, 16).expect("dirt");
        let ramp = tones(&dirt);
        assert_eq!(ramp.len(), 5);
        // Below the fringe the side shows the dirt texture of the same seed, with its two
        // extreme tones folded onto their neighbours.
        for y in 8..16 {
            for x in 0..16 {
                let [r, g, b, _] = dirt.albedo[y * 16 + x];
                let tone = ramp
                    .iter()
                    .position(|&c| c == [r, g, b])
                    .expect("dirt tone");
                let expected = ramp[tone.clamp(1, 3)];
                assert_eq!(
                    side.albedo[y * 16 + x][..3],
                    expected,
                    "seed {seed} at {x}, {y}"
                );
            }
        }
    }
}

#[test]
fn ore_sits_in_the_stone_of_the_same_seed() {
    for entry in pack().iter().filter(|entry| entry.recipe == "ore") {
        for seed in 0..6 {
            let ore = paint(entry, seed, 16);
            let stone = default("stone", seed);
            let ramp = tones(&stone);
            assert_eq!(ramp.len(), 5);
            let mut rock = 0;
            for (texel, &[r, g, b, _]) in ore.albedo.iter().enumerate() {
                let [sr, sg, sb, _] = stone.albedo[texel];
                let tone = ramp
                    .iter()
                    .position(|&c| c == [sr, sg, sb])
                    .expect("stone tone");
                // Rock keeps the stone's tone (extremes folded inwards) or lies in the shadow
                // of a piece of ore; everything else is ore.
                if [r, g, b] == ramp[tone.clamp(1, 3)] {
                    rock += 1;
                } else if ramp.contains(&[r, g, b]) {
                    assert_eq!([r, g, b], ramp[1], "{} seed {seed}", entry.label);
                }
            }
            assert!(
                (150..=235).contains(&rock),
                "{} seed {seed}: {rock} texels of rock",
                entry.label
            );
            // The ore itself: a few pieces in up to three tones of their own.
            let own: BTreeSet<[u8; 3]> = palette(&ore)
                .into_iter()
                .filter(|c| !ramp.contains(c))
                .collect();
            assert!((1..=3).contains(&own.len()), "{}: {own:?}", entry.label);
        }
    }
}

/// Distance of two colours in OKLab.
fn oklab_distance(a: [u8; 3], b: [u8; 3]) -> f64 {
    let (a, b) = (rgb8_to_oklab(a), rgb8_to_oklab(b));
    let (dl, da, db) = (a.l - b.l, a.a - b.a, a.b - b.b);
    (dl * dl + da * da + db * db).sqrt()
}

/// Contract 4.3: every ore of the base game can be told from every other at a glance. Each
/// pair differs in where and how its pieces lie, and each ore of a pair shows a colour that
/// is far from every colour of the other.
#[test]
fn every_ore_can_be_told_from_every_other() {
    let ores = ["copper_ore", "tin_ore", "iron_ore", "coal_ore"];
    let entries: Vec<_> = (pack().into_iter())
        .filter(|entry| ores.contains(&entry.label.as_str()))
        .collect();
    assert_eq!(entries.len(), ores.len());
    for seed in 0..6 {
        let rock = tones(&default("stone", seed));
        // Per ore: the texels that show ore or its stain, and the colours of its own.
        let looks: Vec<(Vec<bool>, Vec<[u8; 3]>)> = (entries.iter())
            .map(|entry| {
                let texture = paint(entry, seed, 16);
                let mask = (texture.albedo.iter())
                    .map(|&[r, g, b, _]| !rock.contains(&[r, g, b]))
                    .collect();
                let own = (palette(&texture).into_iter())
                    .filter(|color| !rock.contains(color))
                    .collect();
                (mask, own)
            })
            .collect();
        for (i, (mask_a, own_a)) in looks.iter().enumerate() {
            for (j, (mask_b, own_b)) in looks.iter().enumerate().skip(i + 1) {
                let pair = format!("{} / {} seed {seed}", entries[i].label, entries[j].label);
                // Shape: less than a third of the texels covered by either is covered by both.
                let both = (mask_a.iter().zip(mask_b))
                    .filter(|&(a, b)| *a && *b)
                    .count();
                let either = (mask_a.iter().zip(mask_b))
                    .filter(|&(a, b)| *a || *b)
                    .count();
                assert!(
                    3 * both < either,
                    "{pair}: {both} of {either} texels shared"
                );
                // Colour: each shows a colour that is far from all colours of the other.
                for (mine, theirs) in [(own_a, own_b), (own_b, own_a)] {
                    let apart = (mine.iter())
                        .map(|&color| {
                            (theirs.iter())
                                .map(|&other| oklab_distance(color, other))
                                .fold(f64::INFINITY, det::min)
                        })
                        .fold(0.0, det::max);
                    assert!(apart > 0.12, "{pair}: colours only {apart:.3} apart");
                }
            }
        }
    }
}

/// Parameters under which `key` of `recipe` has an effect at all.
fn context(recipe: &str, key: &str) -> Params {
    match (recipe, key) {
        // Marks exist on two of the four bark styles only.
        ("log_side", "marks") => Params::new().with_variant("style", "birch"),
        _ => Params::new(),
    }
}

#[test]
fn every_parameter_changes_the_texture() {
    for name in recipes() {
        let info = recipe_info(name).expect("listed recipe");
        for param in info.params {
            let base = context(name, param.key);
            let reference = generate(name, &base, SEED, 16).expect("valid request");
            match param.kind {
                ParamKind::Color => {
                    let color = [60, 150, 170];
                    let params = base.clone().with_color(param.key, color);
                    let changed = generate(name, &params, SEED, 16).expect("valid request");
                    assert_ne!(changed.albedo, reference.albedo, "{name}.{}", param.key);
                    // The colour shows up: some tone is close to it.
                    let wanted = rgb8_to_oklab(color);
                    let nearest = palette(&changed)
                        .into_iter()
                        .map(|tone| {
                            let lab = rgb8_to_oklab(tone);
                            let (dl, da, db) =
                                (lab.l - wanted.l, lab.a - wanted.a, lab.b - wanted.b);
                            (dl * dl + da * da + db * db).sqrt()
                        })
                        .fold(f64::INFINITY, det::min);
                    assert!(
                        nearest < 0.09,
                        "{name}.{}: nearest tone {nearest:.3}",
                        param.key
                    );
                    // A colour changes colours, not the relief.
                    assert_eq!(changed.height, reference.height, "{name}.{}", param.key);
                }
                ParamKind::Variant(names) => {
                    // The first variant is the default.
                    let first = base.clone().with_variant(param.key, names[0]);
                    assert_eq!(
                        generate(name, &first, SEED, 16).expect("default"),
                        reference
                    );
                    let mut seen = vec![reference.albedo.clone()];
                    for variant in &names[1..] {
                        let params = base.clone().with_variant(param.key, variant);
                        let texture = generate(name, &params, SEED, 16).expect("valid request");
                        assert!(
                            !seen.contains(&texture.albedo),
                            "{name}.{}={variant}",
                            param.key
                        );
                        seen.push(texture.albedo);
                    }
                }
            }
        }
    }
    // Marks have no effect on the two bark styles without marks, as documented.
    let plain = default("log_side", SEED);
    let marked = Params::new().with_color("marks", [60, 150, 170]);
    assert_eq!(
        generate("log_side", &marked, SEED, 16).expect("valid"),
        plain
    );
}

#[test]
fn presets_of_one_species_share_their_colours() {
    // The bark around a log's end is the bark of its side; planks are the colour of the wood
    // inside the log.
    for species in ["oak", "birch", "spruce", "cherry"] {
        let preset = |part: &str| {
            let name = format!("{species}_{part}");
            let preset = presets().iter().find(|p| p.name == name).expect("preset");
            preset.to_params()
        };
        assert_eq!(
            preset("log_side").color("bark"),
            preset("log_top").color("bark")
        );
        assert_eq!(
            preset("planks").color("wood"),
            preset("log_top").color("wood")
        );
    }
    // The species differ from each other in every part.
    for part in ["log_side", "log_top", "planks", "leaves"] {
        let mut seen = Vec::new();
        for species in ["oak", "birch", "spruce", "cherry"] {
            let name = format!("{species}_{part}");
            let entry = pack().into_iter().find(|entry| entry.label == name);
            let texture = paint(&entry.expect("preset"), SEED, 16);
            assert!(!seen.contains(&texture.albedo), "{name}");
            seen.push(texture.albedo);
        }
    }
}

#[test]
fn larger_sizes_show_the_same_design_more_finely() {
    for entry in pack() {
        let native = paint(&entry, SEED, 16);
        let fine = paint(&entry, SEED, 128);
        // Same material, same balance of tones: the average colour barely moves.
        let (a, b) = (average_color(&native), average_color(&fine));
        for channel in 0..3 {
            assert!(
                (a[channel] - b[channel]).abs() < 0.03,
                "{}: {a:?} vs {b:?}",
                entry.label
            );
        }
        // Wherever the fine texture is uniform across a whole design unit, the native texture
        // mostly shows that colour in its one texel there.
        let (mut agree, mut uniform) = (0, 0);
        for y in 0..16 {
            for x in 0..16 {
                let corner = fine.albedo[(y * 8) * 128 + x * 8];
                let same =
                    |dx: usize, dy: usize| fine.albedo[(y * 8 + dy) * 128 + x * 8 + dx] == corner;
                if (0..8).all(|dy| (0..8).all(|dx| same(dx, dy))) {
                    uniform += 1;
                    agree += u32::from(native.albedo[y * 16 + x] == corner);
                }
            }
        }
        assert!(
            uniform > 0 && agree * 10 >= uniform * 8,
            "{}: {agree} of {uniform}",
            entry.label
        );
    }
}

/// The row of the crate documentation's table that starts with `` `name` ``, as its cells.
/// `skip` selects among several tables that have a row for the name.
fn documented_row(name: &str, skip: usize) -> Vec<&'static str> {
    let prefix = format!("//! | `{name}` |");
    include_str!("../lib.rs")
        .lines()
        .filter(|line| line.starts_with(&prefix))
        .nth(skip)
        .unwrap_or_else(|| panic!("the crate documentation has no row {skip} for {name}"))
        .split('|')
        .map(str::trim)
        .collect()
}

#[test]
fn the_crate_documentation_lists_every_recipe_and_parameter() {
    for name in recipes() {
        let info = recipe_info(name).expect("listed recipe");
        // Cells: comment marker, recipe, what it shows, parameters, (end).
        let cells = documented_row(name, 0);
        assert_eq!(cells.len(), 5, "{name}: {cells:?}");
        let mut expected: Vec<String> = Vec::new();
        for param in info.params {
            expected.push(format!("`{}`", param.key));
            if let ParamKind::Variant(names) = param.kind {
                expected.extend(names.iter().map(|variant| format!("`{variant}`")));
            }
        }
        for word in &expected {
            assert!(cells[3].contains(word.as_str()), "{name}: {word} missing");
        }
        // Nothing else is listed: every quoted word is a key or a variant.
        assert_eq!(cells[3].matches('`').count(), 2 * expected.len(), "{name}");
        assert_eq!(cells[2].contains("(cutout)"), info.cutout, "{name}");
        assert_eq!(cells[2].contains("(emissive)"), *name == "lava", "{name}");
    }
    // The table of tinted recipes: class and the parameter that bakes a colour in.
    for name in recipes() {
        let info = recipe_info(name).expect("listed recipe");
        let class = match info.tint {
            Tint::None => continue,
            Tint::Grass => "grass",
            Tint::Foliage => "foliage",
            Tint::Water => "water",
        };
        let cells = documented_row(name, 1);
        assert_eq!(cells[2], class, "{name}");
        let key = cells[4].trim_matches('`');
        let param = info.params.iter().find(|param| param.key == key);
        assert_eq!(
            param.map(|param| param.kind),
            Some(ParamKind::Color),
            "{name}"
        );
        // Baking that colour in makes the texture's tinted part coloured.
        let baked = Params::new().with_color(key, [70, 140, 90]);
        let texture = generate(name, &baked, SEED, 16).expect("valid request");
        assert_ne!(palette(&texture), palette(&default(name, SEED)), "{name}");
    }
}
