//! Golden pin of the mesher output (contract §2.7). The mesh cache keys stored meshes by
//! `MESH_FORMAT_VERSION` and `MESHER_VERSION` (contract §6), so an output change without a
//! version bump would let it serve stale meshes for every unchanged chunk.
//!
//! The test meshes a fixed set of inputs and pins, per input and next to both versions, one
//! FNV-1a hash over the quad words and one over the layer ranges and bounds. When it fails:
//! - quads changed (another merge order, a face-rule fix, another greedy): bump
//!   [`MESHER_VERSION`]; if the cause is a new layout of the packed quad words (cp-core), bump
//!   [`MESH_FORMAT_VERSION`] instead;
//! - only the ranges changed (another range layout, another bounds rule): bump
//!   [`MESH_FORMAT_VERSION`];
//!
//! then re-pin the hashes with the new versions and state the reason in the report. Never
//! re-pin without a version bump.
//!
//! The inputs are frozen: they are built here from the pinned `cp_core::hash` functions and
//! integer arithmetic only, so changing other fixtures or the noise functions cannot move the
//! pins. Add a new case instead of editing an existing one.

use cp_core::block::{BlockFlags, BlockRender, FaceAppearance, RenderClass};
use cp_core::hash::{Rng, hash_bytes, hash_ivec3};
use cp_core::ids::{BlockStateId, MaterialId};
use cp_core::math::IVec3;

use super::fixtures::{Palette, padded_from_fn};
use crate::{
    CellBounds, ChunkMesh, LayerRanges, MESH_FORMAT_VERSION, MESHER_VERSION, MeshInput, Mesher,
    PADDED_VOLUME, padded_index,
};

/// `(MESH_FORMAT_VERSION, MESHER_VERSION)` the hashes below belong to.
///
/// History: (2, 1) first pin. (3, 1) `LayerRanges::bounds` became the bounds of all quads of
/// a layer and `regular_bounds` was added: range hashes re-pinned, quad hashes unchanged.
const PINNED_VERSIONS: (u32, u32) = (3, 1);

/// Per case: name, hash of the quads, hash of the layer ranges (over all its levels).
const PINNED: [(&str, u64, u64); 9] = [
    ("random 30 %", 0xc111_b82b_18c8_0c10, 0x57da_c32d_fb60_ec4e),
    ("random 70 %", 0x878d_e570_c1db_8f19, 0x7b9f_e308_7705_248b),
    (
        "random opaque 95 %",
        0xe0dc_7f57_f03c_7e45,
        0x7ed8_3aec_418c_d7d3,
    ),
    ("boxes", 0xb4db_d81e_a24d_3d12, 0x7491_c1ba_b10b_f9fa),
    ("terrain", 0x9379_3c9e_716d_7d56, 0xf5e8_e242_d0b2_4cbb),
    (
        "terrain authored",
        0x0a3e_755e_0a04_fd25,
        0x7956_3f74_00c2_564d,
    ),
    ("mixed halos", 0xec53_cce6_5914_efa4, 0xf22b_0da4_f31b_60ad),
    ("group halos", 0x5eb8_8eaa_8a5a_6eb1, 0x8299_1b7c_e90d_155a),
    (
        "flat authored rows",
        0x7e0d_2341_ac5d_e097,
        0x936f_ed04_d36f_8dfe,
    ),
];

const OPAQUE: BlockFlags = BlockFlags::SOLID.union(BlockFlags::OPAQUE);
const CUTOUT: BlockFlags = BlockFlags::SOLID.union(BlockFlags::CUTOUT);

/// One state of the golden palette: flags, class, cull group, `(material, uv_rot)` per world
/// face in `Face::index` order.
type GoldenState = (BlockFlags, RenderClass, u16, [(u16, u8); 6]);

/// `(material, uv_rot)` of the golden states whose faces differ, by `Face::index`.
const GRASS_FACES: [(u16, u8); 6] = [(4, 0), (4, 1), (3, 0), (2, 0), (4, 2), (4, 3)];
const LOG_FACES: [(u16, u8); 6] = [(6, 0), (6, 0), (7, 1), (7, 1), (7, 3), (7, 3)];
const FURNACE_FACES: [(u16, u8); 6] = [(20, 0), (21, 1), (22, 2), (23, 3), (24, 0), (25, 1)];
/// The furnace's materials with other rotations.
const TURNED_FACES: [(u16, u8); 6] = [(20, 1), (21, 2), (22, 3), (23, 0), (24, 1), (25, 2)];
const LAVA_FACES: [(u16, u8); 6] = [(19, 2), (19, 2), (18, 0), (19, 0), (19, 2), (19, 2)];

/// Every render class; cull groups within and across classes, and two groups that share
/// their low bits; two states with one merge key; equal materials with other rotations;
/// per-face materials and rotations; the largest material id.
const STATES: [GoldenState; 21] = {
    use RenderClass::{Cube, Cutout, Fluid, Model, None};
    const fn all(material: u16) -> [(u16, u8); 6] {
        [(material, 0); 6]
    }
    let empty = BlockFlags::empty();
    let solid = BlockFlags::SOLID;
    let fluid = BlockFlags::FLUID;
    [
        (empty, None, 0, all(0)),              // 0 air
        (OPAQUE, Cube, 0, all(1)),             // 1 stone
        (OPAQUE, Cube, 0, all(2)),             // 2 dirt
        (OPAQUE, Cube, 0, GRASS_FACES),        // 3 grass
        (OPAQUE, Cube, 0, LOG_FACES),          // 4 log
        (OPAQUE, Cube, 0, all(8)),             // 5 ore
        (OPAQUE, Cube, 0, all(1)),             // 6 stone twin: stone's merge key
        (OPAQUE, Cube, 0, FURNACE_FACES),      // 7 furnace
        (OPAQUE, Cube, 0, TURNED_FACES),       // 8 turned furnace
        (OPAQUE, Cube, 0, [(u16::MAX, 3); 6]), // 9 widest material
        (CUTOUT, Cutout, 2, all(11)),          // 10 leaves
        (CUTOUT, Cutout, 0, all(12)),          // 11 grate
        (solid, Cube, 1, all(13)),             // 12 glass
        (CUTOUT, Cutout, 1, all(14)),          // 13 pane, glass's group
        (solid, Cube, 0, all(15)),             // 14 clear
        (solid, Cube, 0x8001, all(16)),        // 15 glass's low bits
        (fluid, Fluid, 3, all(17)),            // 16 water
        (fluid, Fluid, 4, LAVA_FACES),         // 17 lava
        (OPAQUE, Model, 0, all(0)),            // 18 machine
        (empty, Model, 0, all(0)),             // 19 lever
        (OPAQUE, None, 0, all(0)),             // 20 barrier
    ]
};

const AIR: BlockStateId = BlockStateId(0);
const STONE: BlockStateId = BlockStateId(1);
const DIRT: BlockStateId = BlockStateId(2);
const GRASS: BlockStateId = BlockStateId(3);
const LOG: BlockStateId = BlockStateId(4);
const ORE: BlockStateId = BlockStateId(5);
const STONE_TWIN: BlockStateId = BlockStateId(6);
const LEAVES: BlockStateId = BlockStateId(10);
const CLEAR: BlockStateId = BlockStateId(14);
const WATER: BlockStateId = BlockStateId(16);
const BARRIER: BlockStateId = BlockStateId(20);
/// Sand stand-in on beaches: the turned furnace (per-face materials and rotations).
const BEACH: BlockStateId = BlockStateId(8);

fn golden_palette() -> Palette {
    let mut palette = Palette::default();
    for (flags, class, group, faces) in STATES {
        palette.flags.push(flags);
        palette.render.push(BlockRender {
            class,
            faces: faces.map(|(material, uv_rot)| FaceAppearance {
                material: MaterialId(material),
                uv_rot,
            }),
        });
        palette.cull_group.push(group);
    }
    palette
}

/// A hash of the lattice point `(x, y, z)` under `seed`.
fn at(seed: u64, x: i32, y: i32, z: i32) -> u64 {
    hash_ivec3(seed, IVec3::new(x, y, z))
}

/// The authored mask of `kind`: none, sparse (1 in 7 cells), half (cells with odd z), full,
/// or the box `[3, 20) x [5, 30) x [0, 17)`.
fn authored(kind: u32, seed: u64) -> Option<Box<[u64; 512]>> {
    if kind == 0 {
        return None;
    }
    let mut mask = Box::new([0_u64; 512]);
    for y in 0..32 {
        for z in 0..32 {
            for x in 0..32 {
                let set = match kind {
                    1 => at(seed ^ 0xA0, x, y, z).is_multiple_of(7),
                    2 => z % 2 == 1,
                    3 => true,
                    _ => (3..20).contains(&x) && (5..30).contains(&y) && z < 17,
                };
                if set {
                    let index = (x + 32 * z + 1024 * y) as usize;
                    mask[index / 64] |= 1 << (index % 64);
                }
            }
        }
    }
    Some(mask)
}

/// Every readable cell from `mix` with probability `percent` / 100, else air.
fn random(seed: u64, percent: u64, mix: &[BlockStateId]) -> Vec<BlockStateId> {
    padded_from_fn(|x, y, z| {
        let hash = at(seed, x, y, z);
        if hash % 100 < percent {
            mix[(hash >> 32) as usize % mix.len()]
        } else {
            AIR
        }
    })
}

/// 40 overlapping boxes of random states (air included), drawn from `Rng::new(seed, 1)`.
fn boxes(seed: u64) -> Vec<BlockStateId> {
    let mut rng = Rng::new(seed, 1);
    let mut volume = vec![AIR; PADDED_VOLUME];
    for _ in 0..40 {
        let state = BlockStateId(rng.range_u32(0, STATES.len() as u32));
        let [rx, ry, rz] = [0, 1, 2].map(|_| {
            let low = rng.range_u32(0, 34) as i32 - 1;
            let size = rng.range_u32(1, 24) as i32;
            low..(low + size).min(33)
        });
        for y in ry {
            for z in rz.clone() {
                for x in rx.clone() {
                    volume[padded_index(x, y, z)] = state;
                }
            }
        }
    }
    padded_from_fn(|x, y, z| volume[padded_index(x, y, z)])
}

/// Height of the integer heightfield at column `(x, z)`: two octaves of bilinear value noise
/// on lattices of 16 and 4 cells.
fn height(seed: u64, x: i32, z: i32) -> i32 {
    let octave = |cell: i32, amplitude: i64, salt: u64| {
        let value = |gx: i32, gz: i32| {
            (at(seed ^ salt, gx, 0, gz) % (2 * amplitude as u64 + 1)) as i64 - amplitude
        };
        let (gx, fx) = (x.div_euclid(cell), i64::from(x.rem_euclid(cell)));
        let (gz, fz) = (z.div_euclid(cell), i64::from(z.rem_euclid(cell)));
        let c = i64::from(cell);
        let near = value(gx, gz) * (c - fx) + value(gx + 1, gz) * fx;
        let far = value(gx, gz + 1) * (c - fx) + value(gx + 1, gz + 1) * fx;
        (near * (c - fz) + far * fz).div_euclid(c * c)
    };
    (octave(16, 9, 1) + octave(4, 2, 2)) as i32 + 12
}

/// Layered terrain with a sea at height 6, beaches, ores, blob caves and trees.
fn terrain(seed: u64) -> Vec<BlockStateId> {
    // Heights of the padded columns and a margin of 2 for the crowns.
    let span = 38;
    let heights: Vec<i32> = (0..span * span)
        .map(|i| height(seed, i % span - 3, i / span - 3))
        .collect();
    let h = |x: i32, z: i32| heights[((z + 3) * span + x + 3) as usize];
    // Columns with a tree: a trunk of 4 logs, not on beaches.
    let tree = |x: i32, z: i32| at(seed ^ 0x77, x, 0, z).is_multiple_of(31) && h(x, z) > 7;
    padded_from_fn(|x, y, z| {
        let h0 = h(x, z);
        let beach = h0 <= 7;
        if y > h0 && y <= h0 + 4 && tree(x, z) {
            return LOG;
        }
        if y > h0 {
            // A diamond crown of radius 2 around each trunk top.
            let crown = (-2..=2).any(|dz: i32| {
                (-2..=2).any(|dx: i32| {
                    let top = h(x + dx, z + dz) + 4;
                    dx.abs() + dz.abs() + (y - top).abs() <= 2 && tree(x + dx, z + dz)
                })
            });
            return if crown {
                LEAVES
            } else if y <= 6 {
                WATER
            } else {
                AIR
            };
        }
        if y == h0 {
            return if beach { BEACH } else { GRASS };
        }
        if y > h0 - 4 {
            return if beach { BEACH } else { DIRT };
        }
        // Blob caves on a 4-cell lattice, ores and twin stone by cell hash.
        if at(seed ^ 0xCA, x >> 2, y >> 2, z >> 2).is_multiple_of(9) {
            return AIR;
        }
        match at(seed ^ 0x0E, x, y, z) % 20 {
            0 => ORE,
            1 => STONE_TWIN,
            _ => STONE,
        }
    })
}

/// A random interior with a different halo on each side: stone, leaves, water, clear glass,
/// air and an opaque non-rendered state.
fn mixed_halos(seed: u64) -> Vec<BlockStateId> {
    let mix: Vec<BlockStateId> = (0..STATES.len() as u32).map(BlockStateId).collect();
    let halos = [STONE, LEAVES, WATER, CLEAR, AIR, BARRIER];
    let interior = random(seed, 60, &mix);
    padded_from_fn(|x, y, z| {
        let side = [x, y, z]
            .iter()
            .enumerate()
            .find(|(_, c)| !(0..32).contains(*c))
            .map(|(axis, &c)| 2 * axis + usize::from(c < 0));
        match side {
            Some(face) => halos[face],
            None => interior[padded_index(x, y, z)],
        }
    })
}

/// Bodies of leaves, water and glass whose halo continues them: walls of their own groups.
fn group_halos(seed: u64) -> Vec<BlockStateId> {
    padded_from_fn(|x, y, z| match (y.div_euclid(11), at(seed, x, y, z) % 10) {
        (_, 0) => AIR,
        (0, _) => WATER,
        (1, _) => LEAVES,
        _ => BlockStateId(12),
    })
}

/// Flat ground with stone, dirt and grass layers; the authored masks cut its uniform rows.
fn flat() -> Vec<BlockStateId> {
    padded_from_fn(|_, y, _| match y {
        ..=11 => STONE,
        12..=14 => DIRT,
        15 => GRASS,
        _ => AIR,
    })
}

/// Appends a cell box to `bytes`: a tag, then min and max.
fn push_bounds(bytes: &mut Vec<u8>, bounds: Option<CellBounds>) {
    match bounds {
        None => bytes.push(0),
        Some(CellBounds { min, max }) => {
            bytes.push(1);
            bytes.extend_from_slice(&min);
            bytes.extend_from_slice(&max);
        }
    }
}

/// Appends the encodings of `mesh` to the quad and range byte strings.
fn encode(mesh: &ChunkMesh, quads: &mut Vec<u8>, ranges: &mut Vec<u8>) {
    quads.extend_from_slice(&(mesh.quads.len() as u32).to_le_bytes());
    for quad in &mesh.quads {
        quads.extend_from_slice(&quad.w0.to_le_bytes());
        quads.extend_from_slice(&quad.w1.to_le_bytes());
    }
    for layer in &mesh.layers {
        // Exhaustive: a new field must be added to the hash (and the format version bumped).
        let LayerRanges {
            first,
            counts,
            bounds,
            regular_bounds,
            wall_bounds,
        } = *layer;
        ranges.extend_from_slice(&first.to_le_bytes());
        for count in counts {
            ranges.extend_from_slice(&count.to_le_bytes());
        }
        push_bounds(ranges, bounds);
        push_bounds(ranges, regular_bounds);
        push_bounds(ranges, wall_bounds);
    }
}

/// The golden cases: name, cells, authored mask kind, levels.
fn cases() -> Vec<(&'static str, Vec<BlockStateId>, u32, &'static [u8])> {
    let all: Vec<BlockStateId> = (0..STATES.len() as u32).map(BlockStateId).collect();
    let opaque: Vec<BlockStateId> = (1..10).map(BlockStateId).collect();
    vec![
        ("random 30 %", random(1, 30, &all), 1, &[0, 1, 2]),
        ("random 70 %", random(2, 70, &all), 4, &[0, 1, 2]),
        ("random opaque 95 %", random(3, 95, &opaque), 2, &[0, 2]),
        ("boxes", boxes(4), 4, &[0, 1, 2]),
        ("terrain", terrain(5), 0, &[0, 1, 2]),
        ("terrain authored", terrain(6), 1, &[0, 3]),
        ("mixed halos", mixed_halos(7), 0, &[0, 1, 2]),
        ("group halos", group_halos(8), 2, &[0, 2]),
        ("flat authored rows", flat(), 2, &[0]),
    ]
}

#[test]
fn mesher_output_is_pinned() {
    let palette = golden_palette();
    let mut mesher = Mesher::new();
    let mut actual = Vec::new();
    for (name, cells, kind, levels) in cases() {
        let mask = authored(kind, 9);
        let (mut quads, mut ranges) = (Vec::new(), Vec::new());
        for &level in levels {
            let input = MeshInput {
                cells: &cells,
                authored: mask.as_deref(),
                level,
                tables: palette.tables(),
            };
            let mesh = mesher.mesh(&input).expect("valid golden input");
            assert!(!mesh.is_empty(), "{name} level {level} is empty");
            encode(&mesh, &mut quads, &mut ranges);
        }
        actual.push((name, hash_bytes(&quads), hash_bytes(&ranges)));
    }
    let report: String = actual
        .iter()
        .map(|(name, quads, ranges)| format!("    ({name:?}, {quads:#018x}, {ranges:#018x}),\n"))
        .collect();
    let changed: Vec<String> = actual
        .iter()
        .zip(PINNED)
        .filter(|(a, p)| **a != *p)
        .map(|(a, p)| {
            let what = if a.1 != p.1 {
                "quads (bump MESHER_VERSION, or MESH_FORMAT_VERSION for a new quad layout)"
            } else {
                "ranges (bump MESH_FORMAT_VERSION)"
            };
            format!("{}: {what}", a.0)
        })
        .collect();
    assert!(
        changed.is_empty() && (MESH_FORMAT_VERSION, MESHER_VERSION) == PINNED_VERSIONS,
        "mesher output or versions changed: {changed:?}; versions {:?}, pinned {:?}.\n\
         Bump the version(s), then re-pin:\n{report}",
        (MESH_FORMAT_VERSION, MESHER_VERSION),
        PINNED_VERSIONS
    );
}
