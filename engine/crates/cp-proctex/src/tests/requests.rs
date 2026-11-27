//! Requests: what is accepted, what is rejected, and that the same request always gives the
//! same texture.

use std::fmt::Write as _;

use cp_core::hash::hash_bytes;

use super::{SEED, default, pack, paint};
use crate::{
    MAX_SIZE, MIN_SIZE, Params, PbrTexture, ProcTexError, contact_sheet, generate, recipes,
};

#[test]
fn every_texture_has_complete_channels() {
    let check = |texture: &PbrTexture, size: u32, label: &str| {
        let texels = (size * size) as usize;
        assert_eq!(texture.size, size, "{label}");
        assert_eq!(texture.albedo.len(), texels, "{label}");
        assert_eq!(texture.normal.len(), texels, "{label}");
        assert_eq!(texture.roughness.len(), texels, "{label}");
        assert_eq!(texture.emissive.len(), texels, "{label}");
        assert_eq!(texture.height.len(), texels, "{label}");
    };
    for entry in pack() {
        for size in [16, 32, 64] {
            check(&paint(&entry, SEED, size), size, &entry.label);
        }
    }
    // The largest size, for every recipe.
    for name in recipes() {
        let texture = generate(name, &Params::new(), SEED, MAX_SIZE).expect("supported size");
        check(&texture, MAX_SIZE, name);
    }
}

#[test]
fn unknown_recipes_are_errors() {
    for name in ["granite", "", "Stone", "stone ", "grass", "log"] {
        assert_eq!(
            generate(name, &Params::new(), SEED, 16),
            Err(ProcTexError::UnknownRecipe {
                recipe: name.to_owned()
            })
        );
    }
    // The recipe is checked first: a request that is wrong in every way names the recipe.
    let nonsense = Params::new().with_color("nope", [1, 2, 3]);
    assert_eq!(
        generate("granite", &nonsense, SEED, 7),
        Err(ProcTexError::UnknownRecipe {
            recipe: "granite".to_owned()
        })
    );
}

#[test]
fn only_powers_of_two_in_range_are_sizes() {
    assert_eq!((MIN_SIZE, MAX_SIZE), (16, 256));
    let unsupported = [
        0,
        1,
        2,
        4,
        8,
        12,
        15,
        17,
        24,
        48,
        100,
        255,
        257,
        512,
        1 << 20,
        u32::MAX,
    ];
    for size in unsupported {
        assert_eq!(
            generate("stone", &Params::new(), SEED, size),
            Err(ProcTexError::UnsupportedSize { size }),
        );
    }
    for size in [16, 32, 64, 128, 256] {
        let texture = generate("stone", &Params::new(), SEED, size).expect("supported size");
        assert_eq!(texture.albedo.len(), (size * size) as usize);
    }
}

#[test]
fn parameters_that_do_not_fit_are_errors() {
    let unknown = Params::new().with_color("colour", [1, 2, 3]);
    assert_eq!(
        generate("stone", &unknown, SEED, 16),
        Err(ProcTexError::UnknownParam {
            recipe: "stone".to_owned(),
            key: "colour".to_owned()
        })
    );
    let variant = Params::new().with_variant("style", "willow");
    assert_eq!(
        generate("log_side", &variant, SEED, 16),
        Err(ProcTexError::UnknownVariant {
            recipe: "log_side".to_owned(),
            key: "style".to_owned(),
            value: "willow".to_owned()
        })
    );
    let kind = Params::new().with_variant("bark", "brown");
    assert_eq!(
        generate("log_side", &kind, SEED, 16),
        Err(ProcTexError::WrongParamKind {
            recipe: "log_side".to_owned(),
            key: "bark".to_owned(),
            expected: "colour"
        })
    );
    // A parameter of another recipe is unknown here.
    let foreign = Params::new().with_color("ore", [1, 2, 3]);
    assert!(generate("stone", &foreign, SEED, 16).is_err());
    assert!(generate("ore", &foreign, SEED, 16).is_ok());
    // The size is checked before the parameters.
    assert_eq!(
        generate("stone", &unknown, SEED, 20),
        Err(ProcTexError::UnsupportedSize { size: 20 })
    );
}

#[test]
fn the_same_request_gives_the_same_texture() {
    for entry in pack() {
        let first = paint(&entry, SEED, 16);
        assert_eq!(first, paint(&entry, SEED, 16), "{}", entry.label);
        // Other seeds give other textures of the same material.
        let other = paint(&entry, SEED + 1, 16);
        assert_ne!(first.albedo, other.albedo, "{}", entry.label);
    }
    // Different recipes do not share their noise: same seed, different layout.
    assert_ne!(
        default("stone", SEED).height,
        default("limestone", SEED).height
    );
    // The whole 64-bit seed counts.
    let low = generate("stone", &Params::new(), 5, 16).expect("stone");
    let high = generate("stone", &Params::new(), 5 + (1 << 40), 16).expect("stone");
    assert_ne!(low, high);
}

/// FNV-1a over all channels in a fixed order.
fn fingerprint(texture: &PbrTexture) -> u64 {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&texture.size.to_le_bytes());
    bytes.extend(texture.albedo.iter().flatten());
    bytes.extend(texture.normal.iter().flatten());
    bytes.extend_from_slice(&texture.roughness);
    bytes.extend_from_slice(&texture.emissive);
    bytes.extend_from_slice(&texture.height);
    hash_bytes(&bytes)
}

/// Bit-exact results for one seed. Nothing that is saved depends on these textures, but
/// "the same on every machine" is only a claim until something fails when it breaks. A change
/// to the art changes these numbers on purpose; the failure message prints the new table.
#[test]
fn textures_are_pinned() {
    let actual: Vec<(String, u64)> = pack()
        .iter()
        .map(|entry| (entry.label.clone(), fingerprint(&paint(entry, SEED, 16))))
        .collect();
    let unchanged = actual.len() == PINNED.len()
        && (actual.iter().zip(PINNED))
            .all(|((label, hash), (pinned_label, pinned))| label == pinned_label && hash == pinned);
    let mut table = String::new();
    for (label, hash) in &actual {
        let _ = writeln!(table, "    (\"{label}\", {hash:#018x}),");
    }
    assert!(
        unchanged,
        "textures changed; if that is intended, PINNED becomes:\n{table}"
    );

    // The other resolutions, all in one number.
    let mut combined = Vec::new();
    for entry in pack() {
        for size in [32, 64] {
            combined.extend_from_slice(&fingerprint(&paint(&entry, SEED, size)).to_le_bytes());
        }
    }
    let larger = hash_bytes(&combined);
    assert!(
        larger == PINNED_LARGER,
        "32 and 64 px textures changed: PINNED_LARGER becomes {larger:#018x}"
    );

    // And the contact sheet with its layout and lettering.
    let (width, height, pixels) = contact_sheet(16, 2);
    let sheet = hash_bytes(&pixels);
    assert!(
        (width, height, sheet) == PINNED_SHEET,
        "contact sheet changed: PINNED_SHEET becomes ({width}, {height}, {sheet:#018x})"
    );
}

// Re-pinned in the beauty review of E2, round 2: the grass side lost the dark tone of its
// fringe, and the four ground-cover presets (flowers, blossoms, litter, straw) joined the pack,
// which also adds a row to the contact sheet. Re-pinned in round 3: the fallen leaves turned
// darker and duller and the straw lighter and calmer (litter and straw rows and the sheet).
const PINNED: &[(&str, u64)] = &[
    ("stone", 0xbe4a69e0335004c3),
    ("cobble", 0x676aeeed5536e4ec),
    ("gravel", 0xf897617e95728ab1),
    ("dirt", 0x71c99f61e351df9c),
    ("grass_top", 0x8071ec86c0920dc3),
    ("grass_side", 0x33e6bbba03db2eda),
    ("sand", 0x7be6efb548d64d11),
    ("sandstone", 0xc6e2a9f90c2914b3),
    ("snow", 0x1c8b0ceed5424465),
    ("ice", 0x95ab8d4155029603),
    ("clay", 0xda3d382c649faf50),
    ("basalt", 0xf50029e888986eed),
    ("limestone", 0xf1dde365de8c7324),
    ("slate", 0x42be8b206842c907),
    ("log_side", 0x5ae1b7cac7063fdb),
    ("log_top", 0x9aeb436ff52f4610),
    ("planks", 0x1e5ff25249153ffc),
    ("leaves", 0xc43e829677a81e3a),
    ("water", 0x6d65981817b3f5b1),
    ("lava", 0xc9ab5df7989cc0ea),
    ("ore", 0xf450c8d6cea0c925),
    ("metal", 0xd7ab504b67c52e06),
    ("oak_log_side", 0x5ae1b7cac7063fdb),
    ("oak_log_top", 0x9aeb436ff52f4610),
    ("oak_planks", 0x1e5ff25249153ffc),
    ("oak_leaves", 0xc43e829677a81e3a),
    ("birch_log_side", 0x0dfddd49446aa343),
    ("birch_log_top", 0x91a9d3cf19763eb5),
    ("birch_planks", 0x718f69907aa488b9),
    ("birch_leaves", 0x68347fb8244ac9f7),
    ("spruce_log_side", 0x3dc87e7691fdaa4d),
    ("spruce_log_top", 0xafe555cda35185ef),
    ("spruce_planks", 0x715fb7c94ee01a32),
    ("spruce_leaves", 0xa6057015c726cf48),
    ("cherry_log_side", 0xb457e61a6405f7d4),
    ("cherry_log_top", 0xd535014ecd7a5e41),
    ("cherry_planks", 0xb1521fb2b7e59981),
    ("cherry_leaves", 0x5961f94a3915702a),
    ("copper_ore", 0x4c4bad0b45ff0e1a),
    ("tin_ore", 0xf7b3b268865bb22e),
    ("iron_ore", 0x7f4efdfd6494cd8c),
    ("coal_ore", 0x38f7a649919bb179),
    ("flowers_grass_top", 0x74d0fbe39946e20f),
    ("blossoms_grass_top", 0x7131251b53eb06e7),
    ("litter_grass_top", 0x5d4e17e0b07482dc),
    ("straw_grass_top", 0x1d76cd4d2907f1fc),
];
const PINNED_LARGER: u64 = 0x1e6de3123b44028b;
const PINNED_SHEET: (u32, u32, u64) = (520, 535, 0x67d02230cd8565d2);
