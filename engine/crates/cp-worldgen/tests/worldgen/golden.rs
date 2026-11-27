//! Golden hashes over fixed keys at every level 0–`MAX_LOD_LEVEL` (contract §2.5 rule 6, §6):
//! they pin the generators' output. A failing comparison means the output changed: if the
//! change is intended, bump the source's `VERSION` and re-pin the hashes, stating why.
//!
//! The hashes are FNV-1a over the canonical encoding (`ChunkData::write_canonical`) with the
//! persistent keys of the test registry, so they do not depend on session-local ids.
//!
//! Levels 0–3 use several keys each (forests, coasts, ridges, the structure copies). From
//! level 4 on there is at least one key per level and source, chosen so that the chunk is not
//! uniform: while the terrain still varies, the chunk with the ground surface and the open
//! cells above it (a canopy stand-in would show there) and the chunk below it; on the coarsest
//! levels the chunk just below the surface (`pos.y = -1`), whose top row holds the surface
//! layer over stone. Rules gated on coarse levels (the canopy cut-off, the conservative
//! structure rasterisation, the octave sets) are therefore watched as well.

use cp_core::coords::{LodKey, MAX_LOD_LEVEL};
use cp_core::math::IVec3;
use cp_voxel::BaseSource;
use cp_worldgen::{FlatConfig, FlatLayer, FlatSource, HillsSource, PaletteRole, StructureField};

use crate::common::{canonical_hash, chunk, flags, flat, hills, palette, structures};

/// Flat ground with a deeper stack than the grassland, so that every level shows layers.
fn layered_flat() -> FlatSource {
    let layer = |role, thickness| FlatLayer { role, thickness };
    let config = FlatConfig {
        surface_y: 37,
        layers: vec![
            layer(PaletteRole::Grass, 1),
            layer(PaletteRole::Dirt, 3),
            layer(PaletteRole::Sand, 12),
            layer(PaletteRole::Gravel, 20),
        ],
        bottom: PaletteRole::Stone,
    };
    FlatSource::new(0, &config, &palette(), &flags()).expect("valid layered flat ground")
}

fn key(level: u8, x: i32, y: i32, z: i32) -> LodKey {
    LodKey::new(level, IVec3::new(x, y, z))
}

/// Surface with trees, ground, a lake shore, a snowy ridge (level 0); forest edges, the lake
/// and ridge regions on levels 1–3. Levels 4–6: the chunk with the ground surface, snow caps
/// and the open cells above it (a canopy stand-in would show there), and the chunk below with
/// sea, beaches, gravel and grass over stone. Level 7 keeps snow and sand, from level 8 on the
/// band-limited field leaves grass over stone; levels 19 and 20 sit at the voxel limits.
const HILLS_KEYS: &[(u8, [i32; 3])] = &[
    (0, [0, 1, 0]),
    (0, [1, 0, -1]),
    (0, [12, -1, -20]),
    (0, [4, 1, 32]),
    (1, [0, 0, 0]),
    (1, [1, 0, 0]),
    (1, [6, -1, -10]),
    (2, [0, 0, 0]),
    (2, [-3, 0, 1]),
    (2, [1, 0, 8]),
    (3, [0, 0, 0]),
    (3, [2, 0, -2]),
    (4, [-3, 0, -5]),
    (4, [-6, -1, -5]),
    (5, [1, 0, -6]),
    (5, [-6, -1, -6]),
    (6, [-5, 0, -6]),
    (6, [2, -1, -6]),
    (7, [-5, -1, -6]),
    (8, [-6, -1, -6]),
    (9, [3, -1, -2]),
    (10, [-1, -1, 0]),
    (11, [0, -1, 0]),
    (12, [2, -1, 1]),
    (13, [0, -1, -1]),
    (14, [-2, -1, 0]),
    (15, [0, -1, 0]),
    (16, [1, -1, -1]),
    (17, [0, -1, 0]),
    (18, [-1, -1, 1]),
    (19, [63, -1, -64]),
    (20, [-32, -1, 31]),
];

/// Keys of [`layered_flat`]: the surface at 37 m and the layers below it on every level; from
/// level 7 on the surface cell is the top row of the chunk below `y = 0`.
const FLAT_KEYS: &[(u8, [i32; 3])] = &[
    (0, [0, 1, 0]),
    (0, [0, 0, 0]),
    (1, [0, 0, 0]),
    (2, [-1, 0, 3]),
    (3, [0, 0, 0]),
    (4, [0, 0, 0]),
    (5, [0, 0, 0]),
    (6, [-1, 0, 1]),
    (7, [0, -1, 0]),
    (8, [0, -1, 0]),
    (9, [1, -1, 0]),
    (10, [0, -1, 0]),
    (11, [0, -1, -1]),
    (12, [0, -1, 0]),
    (13, [0, -1, 0]),
    (14, [-1, -1, 0]),
    (15, [0, -1, 0]),
    (16, [0, -1, 0]),
    (17, [0, -1, 0]),
    (18, [0, -1, 1]),
    (19, [0, -1, 0]),
    (20, [31, -1, -32]),
];

/// Both layout copies on level 0, the whole layout (kept conservatively) on the coarser
/// levels, and the ground below it on level 1.
const STRUCTURE_KEYS: &[(u8, [i32; 3])] = &[
    (0, [0, 0, 0]),
    (0, [1, 0, 0]),
    (0, [3, 0, 0]),
    (0, [1, 0, 1]),
    (1, [0, 0, 0]),
    (1, [1, 0, 0]),
    (2, [0, 0, 0]),
    (3, [0, 0, 0]),
    (1, [0, -1, 0]),
    (4, [0, 0, 0]),
    (5, [0, 0, 0]),
    (6, [0, 0, 0]),
    (7, [0, 0, 0]),
    (8, [0, 0, 0]),
    (9, [0, 0, 0]),
    (10, [0, 0, 0]),
    (11, [0, 0, 0]),
    (12, [0, 0, 0]),
    (13, [0, 0, 0]),
    (14, [0, 0, 0]),
    (15, [0, 0, 0]),
    (16, [0, 0, 0]),
    (17, [0, 0, 0]),
    (18, [0, 0, 0]),
    (19, [0, 0, 0]),
    (20, [0, 0, 0]),
];

/// Version 3 (beauty review E2, round 2): settled columns (a 3 × 3 median over the column
/// heights) and damped fine octaves, warped ridges, beaches as strips along the water, the
/// shallows as water, dunes in the driest heath, spruce cones that hide their trunks, copses,
/// boulder fields, bushes of several mounds, and crowns drawn for the distance from level 1.
/// Version 4 (beauty review E2, round 3): micro relief (billow noise of 64 and 32 m) that
/// breaks the long one-metre terraces of smooth slopes, warped hills, beaches decided by
/// whole steps and narrower on most shores, clumped forests with trees anywhere in their grid
/// cell, more varied sizes, young trees and veterans. Re-pinned because the generator's output
/// changed on purpose (version 3 → 4); the chunks of levels 8–20 are unchanged.
const PINNED_HILLS: &[u64] = &[
    0x9fb2_7500_9099_4b2e,
    0xdfea_5de4_b20e_01fc,
    0x3f07_92aa_316a_ba4d,
    0xc27f_b762_0ca4_b7a4,
    0xa903_859d_3194_cf52,
    0xf2a6_efe0_0c5f_2531,
    0xf39e_b1d7_db9b_36f3,
    0x0327_4f8f_56de_437d,
    0x826a_699c_001c_b17f,
    0x8524_299a_4e8a_22de,
    0xd923_d2d9_65ed_5388,
    0x671b_556e_b2ed_189a,
    // Levels 4–20.
    0x419a_192a_5394_c3e7,
    0x9f46_e9b6_010c_1819,
    0xedc6_5e93_1c4e_8192,
    0x779e_d420_2173_edcb,
    0xaf1f_2176_6790_f1ba,
    0x4f4d_383b_a2d7_c8d5,
    0xba37_99f1_a2e0_a69e,
    0xc556_b130_e20e_547b,
    0xc556_b130_e20e_547b,
    0xc556_b130_e20e_547b,
    0xc556_b130_e20e_547b,
    0xc556_b130_e20e_547b,
    0xc556_b130_e20e_547b,
    0xc556_b130_e20e_547b,
    0xc556_b130_e20e_547b,
    0xc556_b130_e20e_547b,
    0xc556_b130_e20e_547b,
    0xc556_b130_e20e_547b,
    0xc556_b130_e20e_547b,
    0xc556_b130_e20e_547b,
];
const PINNED_FLAT: &[u64] = &[
    0xe37a_daa7_4ff2_acaa,
    0xf5b3_5cc9_7add_1cf6,
    0xe622_e546_1c1d_e37e,
    0x8d79_e329_3da1_ad40,
    0x10f0_00a2_1cf4_2b40,
    // Levels 4–20.
    0x5dfd_41aa_7d54_4cf4,
    0xd14c_cdce_e0aa_b50a,
    0xd14c_cdce_e0aa_b50a,
    0xc556_b130_e20e_547b,
    0xc556_b130_e20e_547b,
    0xc556_b130_e20e_547b,
    0xc556_b130_e20e_547b,
    0xc556_b130_e20e_547b,
    0xc556_b130_e20e_547b,
    0xc556_b130_e20e_547b,
    0xc556_b130_e20e_547b,
    0xc556_b130_e20e_547b,
    0xc556_b130_e20e_547b,
    0xc556_b130_e20e_547b,
    0xc556_b130_e20e_547b,
    0xc556_b130_e20e_547b,
    0xc556_b130_e20e_547b,
];
const PINNED_STRUCTURES: &[u64] = &[
    0x8a2c_930d_5708_99c8,
    0x5646_23f5_bc27_ac18,
    0xe84d_4396_c927_c071,
    0x552a_ee3f_ed79_1f38,
    0x4e38_d5ab_001c_2a88,
    0xed15_a042_728b_d142,
    0xb562_0f27_e675_d8c2,
    0x1d29_44f7_a7d6_92c1,
    // The ground on level 1, then levels 4–20.
    0xdbbb_7929_3501_ad8b,
    0x1a6e_2e8b_b387_dfad,
    0xf87e_d630_1b8f_bf48,
    0x8c32_bffd_6350_c167,
    0x105b_5080_3a21_c163,
    0x07dd_8b60_3f0a_4161,
    0x07dd_8b60_3f0a_4161,
    0x07dd_8b60_3f0a_4161,
    0x07dd_8b60_3f0a_4161,
    0x07dd_8b60_3f0a_4161,
    0x07dd_8b60_3f0a_4161,
    0x07dd_8b60_3f0a_4161,
    0x07dd_8b60_3f0a_4161,
    0x07dd_8b60_3f0a_4161,
    0x07dd_8b60_3f0a_4161,
    0x07dd_8b60_3f0a_4161,
    0x07dd_8b60_3f0a_4161,
    0x07dd_8b60_3f0a_4161,
];

/// The key set has at least one key on every level `0..=MAX_LOD_LEVEL`, and every key is
/// valid (inside the voxel limits).
fn assert_covers_every_level(keys: &[(u8, [i32; 3])]) {
    for level in 0..=MAX_LOD_LEVEL {
        assert!(
            keys.iter().any(|&(l, _)| l == level),
            "no golden key on level {level}"
        );
    }
    for &(level, [x, y, z]) in keys {
        assert!(
            key(level, x, y, z).is_valid(),
            "golden key {level} {x} {y} {z} is invalid"
        );
    }
}

fn hashes(source: &dyn BaseSource, keys: &[(u8, [i32; 3])]) -> Vec<u64> {
    assert_covers_every_level(keys);
    keys.iter()
        .map(|&(level, [x, y, z])| {
            let cells = chunk(source, key(level, x, y, z));
            // Every golden chunk shows real content, not a uniform block.
            assert!(
                cells.iter().any(|&state| state != cells[0]),
                "golden chunk {level} {x} {y} {z} is uniform"
            );
            canonical_hash(&cells)
        })
        .collect()
}

#[test]
fn hills_output_is_pinned() {
    let source = hills(42);
    assert_eq!(
        (source.identity().key.as_str(), source.identity().version),
        (HillsSource::KEY, 4),
        "re-pin the hashes together with the version"
    );
    let actual = hashes(&source, HILLS_KEYS);
    assert!(
        actual == PINNED_HILLS,
        "hills output changed; bump HillsSource::VERSION and re-pin: {actual:#018x?}"
    );
}

#[test]
fn flat_output_is_pinned() {
    let source = layered_flat();
    assert_eq!(source.identity().version, 1);
    assert_eq!(source.identity().key, FlatSource::KEY);
    let actual = hashes(&source, FLAT_KEYS);
    assert!(
        actual == PINNED_FLAT,
        "flat output changed; bump FlatSource::VERSION and re-pin: {actual:#018x?}"
    );
}

#[test]
fn structure_field_output_is_pinned() {
    let source = structures();
    assert_eq!(source.identity().version, 1);
    assert_eq!(source.identity().key, StructureField::KEY);
    let actual = hashes(&source, STRUCTURE_KEYS);
    assert!(
        actual == PINNED_STRUCTURES,
        "structure field output changed; bump StructureField::VERSION and re-pin: \
         {actual:#018x?}"
    );
}

#[test]
fn identities_are_pinned() {
    // The identity hash (cache keys, saves) covers the parameter hash; pin both for the
    // default configurations of the test registry.
    let actual = [
        hills(42).identity().macro_hash,
        flat().identity().macro_hash,
        structures().identity().macro_hash,
        hills(42).identity().stable_hash(),
        layered_flat().identity().macro_hash,
        flat().identity().stable_hash(),
    ];
    // The flat values were re-pinned when the flat identity started to cover the layer roles
    // (its cells did not change, so `FlatSource::VERSION` stayed 1). All six were re-pinned
    // when the palette gained the roles of birch, spruce, cherry, clay, slate and limestone
    // (the key hash covers every role; the flat and structure cells did not change) and the
    // hills became version 2 (beauty review E2, round 1). The hills' stable hash was re-pinned
    // when they became version 3 (beauty review E2, round 2; the version enters it, the
    // parameter hash did not change), and again when they became version 4 (round 3).
    let pinned = [
        0x0dd5_eaed_5871_509a,
        0x77d4_a2ab_6d5e_4240,
        0x3823_ef77_eff8_607b,
        0x0c82_d9a6_4243_2933,
        0xb0dd_8661_62fd_6a50,
        0x42ab_49d3_16fd_f1ed,
    ];
    assert!(actual == pinned, "identities changed: {actual:#018x?}");
}

#[test]
fn flat_identities_tell_the_layer_states_apart() {
    // Regression: the flat identity once hashed the thicknesses and the palette's key hash
    // only, so these two worlds of the same palette shared an identity.
    let palette = palette();
    let grassland = flat();
    let layer = |role, thickness| FlatLayer { role, thickness };
    let beach = FlatConfig {
        surface_y: 6,
        layers: vec![layer(PaletteRole::Sand, 1), layer(PaletteRole::Gravel, 3)],
        bottom: PaletteRole::Snow,
    };
    let beach = FlatSource::new(0, &beach, &palette, &flags()).expect("valid beach");
    let key = LodKey::lod0(IVec3::ZERO);
    assert_ne!(chunk(&grassland, key), chunk(&beach, key));
    assert_ne!(grassland.identity(), beach.identity());
    assert_ne!(
        grassland.identity().stable_hash(),
        beach.identity().stable_hash()
    );
}
