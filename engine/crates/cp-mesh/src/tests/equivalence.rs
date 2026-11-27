//! The greedy mesh covers exactly the faces of the naive reference, regular and walls, for
//! random and structured chunks (contract §5.2, §9 E1).

use cp_core::face::Face;
use cp_core::hash::Rng;
use cp_core::ids::BlockStateId;

use super::fixtures::{
    Blocks, Palette, input, padded_from_fn, palette, random_authored, random_boxes, random_cells,
};
use super::reference::{
    assert_same_faces, assert_same_mesh, check_invariants, mesh_faces, reference_faces,
    reference_mesh,
};
use crate::{CellBounds, ChunkMesh, Layer, Mesher};

/// Meshes `cells` at `level`, checks the invariants, the face set and the exact quads against
/// the references, returns the mesh.
fn mesh_and_check(
    mesher: &mut Mesher,
    palette: &Palette,
    cells: &[BlockStateId],
    authored: Option<&[u64; 512]>,
    level: u8,
    context: &str,
) -> ChunkMesh {
    let input = input(palette, cells, authored, level);
    let mesh = mesher.mesh(&input).expect("valid input");
    check_invariants(&mesh, context);
    assert_same_faces(&mesh_faces(&mesh), &reference_faces(&input), context);
    assert_same_mesh(&mesh, &reference_mesh(&input), context);
    mesh
}

/// The mixes of the random tests, by name.
fn mixes(blocks: &Blocks) -> Vec<(&'static str, Vec<BlockStateId>)> {
    vec![
        ("opaque", blocks.opaque_mix()),
        ("cutout", blocks.cutout_mix()),
        ("fluid", blocks.fluid_mix()),
        ("glass", blocks.glass_mix()),
        ("models", blocks.model_mix()),
        ("everything", blocks.everything()),
    ]
}

#[test]
fn random_cells_match_the_reference() {
    let (palette, blocks) = palette();
    let mut mesher = Mesher::new();
    let mut checked = 0;
    let mut walls = 0;
    for (mix_index, (name, mix)) in mixes(&blocks).into_iter().enumerate() {
        for (density_index, density) in [0.05, 0.3, 0.6, 0.95].into_iter().enumerate() {
            for seed in 0..SEEDS {
                let stream = (mix_index * 16 + density_index) as u64;
                let mut rng = Rng::new(seed, stream);
                let cells = random_cells(&mut rng, &mix, density);
                let authored = random_authored(&mut rng, (seed as u32) + density_index as u32);
                for level in [0, 1, 2] {
                    let context = format!("{name} density {density} seed {seed} level {level}");
                    let mesh = mesh_and_check(
                        &mut mesher,
                        &palette,
                        &cells,
                        authored.as_deref(),
                        level,
                        &context,
                    );
                    walls += mesh.wall_len();
                    checked += 1;
                }
            }
        }
    }
    assert_eq!(checked, 6 * 4 * SEEDS * 3);
    assert!(walls > 0, "the random chunks must exercise the walls");
}

/// Seeds per mix and density of the random tests (boxes use more, see below).
const SEEDS: u64 = 5;

#[test]
fn random_boxes_match_the_reference() {
    let (palette, blocks) = palette();
    let mut mesher = Mesher::new();
    let mut merged = 0;
    for (mix_index, (name, mix)) in mixes(&blocks).into_iter().enumerate() {
        for seed in 0..2 * SEEDS {
            let mut rng = Rng::new(seed, 100 + mix_index as u64);
            let cells = random_boxes(&mut rng, &mix, 12 + 8 * seed as u32);
            let authored = random_authored(&mut rng, seed as u32);
            for level in [0, 1, 2] {
                let context = format!("boxes of {name} seed {seed} level {level}");
                let mesh = mesh_and_check(
                    &mut mesher,
                    &palette,
                    &cells,
                    authored.as_deref(),
                    level,
                    &context,
                );
                merged += mesh
                    .quads
                    .iter()
                    .filter(|quad| quad.width() > 1 || quad.height() > 1)
                    .count();
            }
        }
    }
    assert!(
        merged > 1000,
        "the boxes must produce merged quads ({merged})"
    );
}

#[test]
fn authored_masks_on_structured_chunks_match_the_reference() {
    // Rows of one state take a shortcut (one key per row); it must honour authored cells
    // whether the row's faces are all authored, none or some of them.
    let (palette, blocks) = palette();
    let grass = blocks.grass[Face::PosY.index()];
    let flat = padded_from_fn(|_, y, _| match y {
        ..=11 => blocks.stone,
        12..=14 => blocks.dirt,
        15 => grass,
        _ => blocks.air,
    });
    let stairs = padded_from_fn(|x, y, _| {
        if y <= x.div_euclid(2) {
            blocks.stone
        } else {
            blocks.air
        }
    });
    let pillars = padded_from_fn(|x, y, z| {
        if (x / 4 + z / 4) % 2 == 0 && y < 20 {
            blocks.logs[1]
        } else if y < 3 {
            blocks.water
        } else {
            blocks.air
        }
    });
    let buried = vec![blocks.stone; crate::PADDED_VOLUME];
    // Authored masks: everything, a slab of whole rows, alternate rows, sparse cells.
    let mask_where = |f: &dyn Fn(usize, usize, usize) -> bool| {
        let mut mask = Box::new([0_u64; 512]);
        for y in 0..32 {
            for z in 0..32 {
                for x in 0..32 {
                    if f(x, y, z) {
                        let index = x + 32 * z + 1024 * y;
                        mask[index / 64] |= 1 << (index % 64);
                    }
                }
            }
        }
        mask
    };
    let masks = [
        mask_where(&|_, _, _| true),
        mask_where(&|_, y, _| (5..13).contains(&y)),
        mask_where(&|_, _, z| z % 2 == 1),
        mask_where(&|x, y, z| (x * 7 + y * 13 + z * 3) % 11 == 0),
    ];
    let mut mesher = Mesher::new();
    let mut authored_quads = 0;
    for (name, cells) in [
        ("flat", &flat),
        ("stairs", &stairs),
        ("pillars", &pillars),
        ("buried", &buried),
    ] {
        for (index, mask) in masks.iter().enumerate() {
            let context = format!("{name} with authored mask {index}");
            let mesh = mesh_and_check(&mut mesher, &palette, cells, Some(mask), 0, &context);
            authored_quads += mesh.quads.iter().filter(|q| q.is_authored()).count();
        }
    }
    assert!(authored_quads > 100, "{authored_quads}");
}

#[test]
fn flat_ground() {
    let (palette, blocks) = palette();
    let grass = blocks.grass[Face::PosY.index()];
    let cells = padded_from_fn(|_, y, _| match y {
        ..=11 => blocks.stone,
        12..=14 => blocks.dirt,
        15 => grass,
        _ => blocks.air,
    });
    let mesh = mesh_and_check(&mut Mesher::new(), &palette, &cells, None, 0, "flat");
    // One 32 x 32 grass top; everything else is hidden by neighbours or the halo.
    assert_eq!(mesh.regular_len(), 1);
    let top = mesh.regular_quads(Layer::Opaque, Face::PosY)[0];
    assert_eq!(
        (top.cell(), top.width(), top.height()),
        ([0, 15, 0], 32, 32)
    );
    // Walls: three bands (stone, dirt, grass side) on each side and one stone floor.
    for face in [Face::PosX, Face::NegX, Face::PosZ, Face::NegZ] {
        assert_eq!(mesh.wall_quads(Layer::Opaque, face).len(), 3, "{face:?}");
    }
    assert_eq!(mesh.wall_quads(Layer::Opaque, Face::NegY).len(), 1);
    assert_eq!(mesh.wall_len(), 13);
    let opaque = mesh.layer(Layer::Opaque);
    let regular = opaque.regular_bounds.expect("has quads");
    assert_eq!((regular.min, regular.max), ([0, 15, 0], [31, 15, 31]));
    let walls = opaque.wall_bounds.expect("has walls");
    assert_eq!((walls.min, walls.max), ([0, 0, 0], [31, 15, 31]));
    // The layer's bounds reach down to the walls, far below the regular quads.
    let all = opaque.bounds.expect("has quads");
    assert_eq!((all.min, all.max), ([0, 0, 0], [31, 15, 31]));
}

#[test]
fn bounds_of_a_layer_with_walls_only_cover_the_walls() {
    // Contract §5.2: a layer records the tight bounds of all its quads. A buried chunk has no
    // regular quad, but its walls close the seam to a coarser neighbour whose seam cells are
    // air (an overhang that only appears after downsampling), so a draw culled with the
    // layer's bounds must keep them.
    let (palette, blocks) = palette();
    let mut mesher = Mesher::new();
    let buried = padded_from_fn(|_, _, _| blocks.stone);
    let mesh = mesh_and_check(&mut mesher, &palette, &buried, None, 4, "buried");
    let opaque = mesh.layer(Layer::Opaque);
    assert_eq!((opaque.regular_len(), opaque.wall_len()), (0, 6));
    let full = CellBounds {
        min: [0; 3],
        max: [31; 3],
    };
    assert_eq!(
        (opaque.bounds, opaque.regular_bounds, opaque.wall_bounds),
        (Some(full), None, Some(full))
    );
    // Stone below y = 16 under an opaque cover that is not drawn: walls on five sides only.
    let covered = padded_from_fn(|_, y, _| if y < 16 { blocks.stone } else { blocks.barrier });
    let mesh = mesh_and_check(&mut mesher, &palette, &covered, None, 4, "covered");
    let opaque = mesh.layer(Layer::Opaque);
    assert_eq!((opaque.regular_len(), opaque.wall_len()), (0, 5));
    let lower = CellBounds {
        min: [0; 3],
        max: [31, 15, 31],
    };
    assert_eq!(
        (opaque.bounds, opaque.regular_bounds, opaque.wall_bounds),
        (Some(lower), None, Some(lower))
    );
}

#[test]
fn checkerboard_worst_case() {
    let (palette, blocks) = palette();
    let cells = padded_from_fn(|x, y, z| {
        if (x + y + z).rem_euclid(2) == 0 {
            blocks.stone
        } else {
            blocks.air
        }
    });
    let mesh = mesh_and_check(
        &mut Mesher::new(),
        &palette,
        &cells,
        None,
        0,
        "checkerboard",
    );
    // Nothing can merge: six 1 x 1 quads per solid cell, no walls.
    assert_eq!(mesh.regular_len(), 16_384 * 6);
    assert_eq!(mesh.wall_len(), 0);
    assert!(mesh.quads.iter().all(|q| q.width() == 1 && q.height() == 1));
}

#[test]
fn hollow_box() {
    let (palette, blocks) = palette();
    let shell = |c: i32| (6..=25).contains(&c);
    let inside = |c: i32| (7..=24).contains(&c);
    let cells = padded_from_fn(|x, y, z| {
        if shell(x) && shell(y) && shell(z) && !(inside(x) && inside(y) && inside(z)) {
            blocks.stone
        } else {
            blocks.air
        }
    });
    let mesh = mesh_and_check(&mut Mesher::new(), &palette, &cells, None, 0, "hollow box");
    // One 20 x 20 outer and one 18 x 18 inner quad per face.
    assert_eq!(mesh.regular_len(), 12);
    assert_eq!(mesh.wall_len(), 0);
    for face in Face::ALL {
        let quads = mesh.regular_quads(Layer::Opaque, face);
        let mut sizes: Vec<_> = quads.iter().map(|q| (q.width(), q.height())).collect();
        sizes.sort_unstable();
        assert_eq!(sizes, [(18, 18), (20, 20)], "{face:?}");
    }
}

#[test]
fn staircase() {
    let (palette, blocks) = palette();
    // Two cells wide steps rising along +x; solid up to height x / 2 (floor).
    let cells = padded_from_fn(|x, y, _| {
        if y <= x.div_euclid(2) {
            blocks.stone
        } else {
            blocks.air
        }
    });
    let mesh = mesh_and_check(&mut Mesher::new(), &palette, &cells, None, 0, "staircase");
    // 16 treads (2 x 32) and 16 risers (1 x 32).
    assert_eq!(mesh.regular_quads(Layer::Opaque, Face::PosY).len(), 16);
    assert_eq!(mesh.regular_quads(Layer::Opaque, Face::NegX).len(), 16);
    assert_eq!(mesh.regular_len(), 32);
    // Walls: the floor, the +X side and 16 strips on each Z side.
    assert_eq!(mesh.wall_quads(Layer::Opaque, Face::NegY).len(), 1);
    assert_eq!(mesh.wall_quads(Layer::Opaque, Face::PosX).len(), 1);
    assert_eq!(mesh.wall_quads(Layer::Opaque, Face::PosZ).len(), 16);
    assert_eq!(mesh.wall_quads(Layer::Opaque, Face::NegZ).len(), 16);
    assert_eq!(mesh.wall_len(), 34);
}

#[test]
fn every_orientation_carries_its_materials() {
    // Isolated furnaces in all 24 orientations: every world face shows the material and
    // uv_rot of the BlockRender table, as the reference reads it.
    let (palette, blocks) = palette();
    let cells = padded_from_fn(|x, y, z| {
        if x % 4 == 1 && y % 4 == 1 && z % 4 == 1 {
            blocks.furnace[((x / 4) + 8 * (y / 4) + 3 * (z / 4)) as usize % 24]
        } else {
            blocks.air
        }
    });
    let mesh = mesh_and_check(&mut Mesher::new(), &palette, &cells, None, 0, "furnaces");
    assert_eq!(mesh.regular_len(), 8 * 8 * 8 * 6);
    let rotations: std::collections::BTreeSet<u8> =
        mesh.quads.iter().map(|quad| quad.uv_rot()).collect();
    assert_eq!(rotations.len(), 4, "all four texture rotations occur");
}

#[test]
fn cull_groups_and_hiding() {
    let (palette, blocks) = palette();
    let mut mesher = Mesher::new();
    // Two cells next to each other along x at (10, 10, 10) and (11, 10, 10).
    let pair = |a: BlockStateId, b: BlockStateId| {
        padded_from_fn(move |x, y, z| match (x, y, z) {
            (10, 10, 10) => a,
            (11, 10, 10) => b,
            _ => blocks.air,
        })
    };
    // Number of regular faces between the two cells: (a's +X face, b's -X face).
    let between = |mesher: &mut Mesher, a, b| {
        let cells = pair(a, b);
        let mesh = mesh_and_check(mesher, &palette, &cells, None, 0, "pair");
        let faces = mesh_faces(&mesh);
        let has = |cell: [u8; 3], face| faces.iter().any(|f| f.cell == cell && f.face == face);
        (has([10, 10, 10], Face::PosX), has([11, 10, 10], Face::NegX))
    };
    let b = &blocks;
    assert_eq!(between(&mut mesher, b.leaves, b.leaves), (false, false));
    assert_eq!(between(&mut mesher, b.glass, b.pane), (false, false));
    assert_eq!(between(&mut mesher, b.water, b.water), (false, false));
    assert_eq!(between(&mut mesher, b.water, b.lava), (true, true));
    assert_eq!(between(&mut mesher, b.glass, b.odd), (true, true));
    assert_eq!(between(&mut mesher, b.clear, b.clear), (true, true));
    assert_eq!(between(&mut mesher, b.grate, b.grate), (true, true));
    assert_eq!(between(&mut mesher, b.stone, b.leaves), (true, false));
    assert_eq!(between(&mut mesher, b.water, b.stone), (false, true));
    assert_eq!(between(&mut mesher, b.stone, b.machine), (false, false));
    assert_eq!(between(&mut mesher, b.stone, b.lever), (true, false));
    assert_eq!(between(&mut mesher, b.stone, b.barrier), (false, false));
}

#[test]
fn cutout_moves_to_the_opaque_layer_from_level_two() {
    let (palette, blocks) = palette();
    let cells = padded_from_fn(|x, y, z| {
        if (4..12).contains(&x) && (4..12).contains(&y) && (4..12).contains(&z) {
            blocks.leaves
        } else {
            blocks.air
        }
    });
    let mut mesher = Mesher::new();
    for level in [0, 1] {
        let mesh = mesh_and_check(&mut mesher, &palette, &cells, None, level, "leaves");
        assert_eq!(mesh.layer(Layer::Cutout).len(), 6, "level {level}");
        assert!(mesh.layer(Layer::Opaque).is_empty());
    }
    for level in [2, 3, 20] {
        let mesh = mesh_and_check(&mut mesher, &palette, &cells, None, level, "leaves");
        assert!(mesh.layer(Layer::Cutout).is_empty(), "level {level}");
        assert_eq!(mesh.layer(Layer::Opaque).len(), 6);
    }
}

#[test]
fn walls_appear_where_the_halo_hides_faces() {
    let (palette, blocks) = palette();
    let mut mesher = Mesher::new();
    // A stone chunk in air: six regular 32 x 32 faces, no walls.
    let alone = padded_from_fn(|x, y, z| {
        if [x, y, z].iter().all(|c| (0..32).contains(c)) {
            blocks.stone
        } else {
            blocks.air
        }
    });
    let mesh = mesh_and_check(&mut mesher, &palette, &alone, None, 0, "stone in air");
    assert_eq!((mesh.regular_len(), mesh.wall_len()), (6, 0));
    // Leaves surrounded by leaves (same cull group) and water by water: walls in their layers.
    for (state, layer) in [(blocks.leaves, Layer::Cutout), (blocks.water, Layer::Fluid)] {
        let cells = padded_from_fn(|_, _, _| state);
        let mesh = mesh_and_check(&mut mesher, &palette, &cells, None, 0, "group walls");
        assert_eq!(mesh.regular_len(), 0);
        assert_eq!(mesh.layer(layer).wall_len(), 6);
        assert_eq!(mesh.wall_len(), 6);
    }
    // Clear glass (no group) hides nothing: inside the chunk every cell shows all six faces,
    // one 32 x 32 quad per slice and direction. A stone halo turns the 6 boundary slices into
    // walls; a clear halo leaves them regular.
    let boxed = |halo: BlockStateId| {
        padded_from_fn(move |x, y, z| {
            if [x, y, z].iter().all(|c| (0..32).contains(c)) {
                blocks.clear
            } else {
                halo
            }
        })
    };
    let mesh = mesh_and_check(
        &mut mesher,
        &palette,
        &boxed(blocks.stone),
        None,
        0,
        "clear",
    );
    assert_eq!((mesh.regular_len(), mesh.wall_len()), (6 * 31, 6));
    let mesh = mesh_and_check(
        &mut mesher,
        &palette,
        &boxed(blocks.clear),
        None,
        0,
        "clear",
    );
    assert_eq!((mesh.regular_len(), mesh.wall_len()), (6 * 32, 0));
}
