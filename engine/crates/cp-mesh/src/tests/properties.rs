//! Canonical output, empty and buried chunks, unread halo cells, input errors.

use cp_core::block::{BlockFlags, BlockRender, FaceAppearance, RenderClass};
use cp_core::face::Face;
use cp_core::hash::Rng;
use cp_core::ids::{BlockStateId, MaterialId};

use super::fixtures::{
    Palette, input, is_readable, padded_from_fn, palette, random_authored, random_boxes,
    random_cells,
};
use super::reference::{
    assert_same_faces, assert_same_mesh, check_invariants, mesh_faces, reference_faces,
    reference_mesh,
};
use crate::{
    ChunkMesh, Layer, MeshError, MeshInput, MeshTables, Mesher, PADDED_VOLUME, padded_index,
};

/// Cells, authored mask and level of one test chunk.
type Chunk = (Vec<BlockStateId>, Option<Box<[u64; 512]>>, u8);

#[test]
fn output_is_canonical_whatever_was_meshed_before() {
    let (palette, blocks) = palette();
    let mut rng = Rng::new(7, 7);
    let everything = blocks.everything();
    let chunks: Vec<Chunk> = (0..6)
        .map(|i| {
            let cells = if i % 2 == 0 {
                random_cells(&mut rng, &everything, 0.2 + 0.15 * f64::from(i))
            } else {
                random_boxes(&mut rng, &everything, 20 + 5 * i)
            };
            (cells, random_authored(&mut rng, i), (i % 3) as u8)
        })
        .collect();
    let inputs: Vec<MeshInput<'_>> = chunks
        .iter()
        .map(|(cells, authored, level)| input(&palette, cells, authored.as_deref(), *level))
        .collect();
    // Each chunk meshed by a fresh mesher into a fresh mesh.
    let fresh: Vec<ChunkMesh> = inputs
        .iter()
        .map(|input| Mesher::new().mesh(input).expect("valid input"))
        .collect();
    // One mesher and one output buffer, chunks in a scrambled order with repeats.
    let mut mesher = Mesher::new();
    let mut out = ChunkMesh::new();
    for &index in &[3, 0, 5, 5, 1, 4, 2, 0, 3, 1, 2, 4] {
        mesher
            .mesh_into(&inputs[index], &mut out)
            .expect("valid input");
        assert_eq!(out, fresh[index], "chunk {index}");
        let bytes: &[u8] = bytemuck::cast_slice(&out.quads);
        assert_eq!(bytes, bytemuck::cast_slice::<_, u8>(&fresh[index].quads));
    }
    // An error in between leaves no trace either.
    let broken = vec![BlockStateId(9_999); PADDED_VOLUME];
    assert!(
        mesher
            .mesh_into(&input(&palette, &broken, None, 0), &mut out)
            .is_err()
    );
    assert!(out.is_empty());
    mesher.mesh_into(&inputs[2], &mut out).expect("valid input");
    assert_eq!(out, fresh[2]);
}

#[test]
fn tables_may_change_between_calls() {
    // A mesher keeps per-state data for one call only: tables of the same length but other
    // content (a hot reload of materials, another registry) must not see anything of the
    // previous call.
    let (palette, blocks) = palette();
    let mut rng = Rng::new(21, 0);
    let cells = random_boxes(&mut rng, &blocks.everything(), 30);
    let mut reloaded = palette.clone();
    for render in &mut reloaded.render {
        for face in &mut render.faces {
            face.material.0 += 100;
            face.uv_rot = (face.uv_rot + 1) % 4;
        }
    }
    // Stone stops hiding its neighbours, dirt joins a cull group, leaves leave theirs.
    reloaded.flags[blocks.stone.index()] = BlockFlags::SOLID;
    reloaded.cull_group[blocks.dirt.index()] = 7;
    reloaded.cull_group[blocks.leaves.index()] = 0;
    let mut mesher = Mesher::new();
    for (round, tables) in [&palette, &reloaded, &palette, &reloaded]
        .into_iter()
        .enumerate()
    {
        for level in [0, 2] {
            let input = input(tables, &cells, None, level);
            let mesh = mesher.mesh(&input).expect("valid input");
            let context = format!("round {round} level {level}");
            assert_same_mesh(&mesh, &reference_mesh(&input), &context);
            assert_eq!(
                mesh,
                Mesher::new().mesh(&input).expect("valid"),
                "{context}"
            );
        }
    }
}

#[test]
fn state_ids_beyond_sixteen_bits_work() {
    // 120 000 states, of which the chunk uses a few far apart (ids are u32 everywhere): all
    // beyond u16 and equal modulo every power of two up to 1024.
    let count = 120_000;
    let (base, blocks) = palette();
    let mut palette = Palette {
        flags: vec![BlockFlags::empty(); count],
        render: vec![BlockRender::NONE; count],
        cull_group: vec![0; count],
    };
    let far = |state: BlockStateId| match state.0 {
        0 => 0,
        id => 65_536 + 1_024 * id,
    };
    let mix = [
        blocks.stone,
        blocks.dirt,
        blocks.leaves,
        blocks.water,
        blocks.glass,
        blocks.grass[2],
    ];
    for state in mix.into_iter().chain([blocks.air]) {
        let id = far(state) as usize;
        palette.flags[id] = base.flags[state.index()];
        palette.render[id] = base.render[state.index()];
        palette.cull_group[id] = base.cull_group[state.index()];
    }
    let far_mix: Vec<BlockStateId> = mix.iter().map(|&state| BlockStateId(far(state))).collect();
    assert!(
        far_mix
            .iter()
            .all(|state| state.0 >= 65_536 && state.index() < count)
    );
    let mut rng = Rng::new(5, 5);
    let mut mesher = Mesher::new();
    for cells in [
        random_cells(&mut rng, &far_mix, 0.6),
        random_boxes(&mut rng, &far_mix, 40),
    ] {
        let input = input(&palette, &cells, None, 0);
        let mesh = mesher.mesh(&input).expect("valid input");
        check_invariants(&mesh, "far ids");
        assert_same_mesh(&mesh, &reference_mesh(&input), "far ids");
        assert!(mesh.regular_len() > 100);
    }
}

#[test]
fn air_and_buried_chunks_have_no_regular_quads() {
    let (palette, blocks) = palette();
    let mut mesher = Mesher::new();
    // Uniform air: nothing at all.
    let air = vec![BlockStateId::AIR; PADDED_VOLUME];
    let mesh = mesher.mesh(&input(&palette, &air, None, 0)).expect("valid");
    assert!(mesh.is_empty());
    assert_eq!(mesh.layers, ChunkMesh::new().layers);
    // Uniform stone, halo included: no regular quad, one 32 x 32 wall per side.
    let stone = vec![blocks.stone; PADDED_VOLUME];
    let mesh = mesher
        .mesh(&input(&palette, &stone, None, 0))
        .expect("valid");
    assert_eq!(mesh.regular_len(), 0);
    assert_eq!(mesh.wall_len(), 6);
    for face in Face::ALL {
        let walls = mesh.wall_quads(Layer::Opaque, face);
        assert_eq!((walls[0].width(), walls[0].height()), (32, 32), "{face:?}");
    }
    // Random opaque cells of many materials and orientations in an opaque halo: still no
    // regular quad, and the walls cover the six boundary layers exactly.
    let mix = blocks.opaque_mix();
    let mut rng = Rng::new(3, 3);
    for level in [0, 2] {
        let cells = padded_from_fn(|x, y, z| {
            if [x, y, z].iter().all(|c| (0..32).contains(c)) {
                mix[rng.range_u32(0, mix.len() as u32) as usize]
            } else {
                blocks.stone
            }
        });
        let input = input(&palette, &cells, None, level);
        let mesh = mesher.mesh(&input).expect("valid");
        check_invariants(&mesh, "buried");
        assert_same_faces(&mesh_faces(&mesh), &reference_faces(&input), "buried");
        assert_same_mesh(&mesh, &reference_mesh(&input), "buried");
        assert_eq!(mesh.regular_len(), 0);
        let wall_faces: usize = mesh
            .quads
            .iter()
            .map(|q| usize::from(q.width()) * usize::from(q.height()))
            .sum();
        assert_eq!(wall_faces, 6 * 32 * 32);
    }
}

#[test]
fn edge_and_corner_halo_cells_are_never_read() {
    let (palette, blocks) = palette();
    let mut rng = Rng::new(11, 0);
    let cells = random_boxes(&mut rng, &blocks.everything(), 40);
    let mut mesher = Mesher::new();
    let expected = mesher
        .mesh(&input(&palette, &cells, None, 0))
        .expect("valid");
    // The fixtures already hold UNCHANGED there; any other garbage gives the same mesh.
    let mut garbage = cells.clone();
    for y in -1..=32 {
        for z in -1..=32 {
            for x in -1..=32 {
                if !is_readable(x, y, z) {
                    garbage[padded_index(x, y, z)] = BlockStateId(rng.next_u32());
                }
            }
        }
    }
    let mesh = mesher
        .mesh(&input(&palette, &garbage, None, 0))
        .expect("unread cells");
    assert_eq!(mesh, expected);
}

/// The palette's tables with one row changed.
fn tables_with(
    palette: &Palette,
    f: impl FnOnce(&mut Vec<BlockFlags>, &mut Vec<BlockRender>, &mut Vec<u16>),
) -> Palette {
    let mut changed = Palette {
        flags: palette.flags.clone(),
        render: palette.render.clone(),
        cull_group: palette.cull_group.clone(),
    };
    f(
        &mut changed.flags,
        &mut changed.render,
        &mut changed.cull_group,
    );
    changed
}

#[test]
fn bad_input_is_an_error() {
    let (palette, blocks) = palette();
    let mut mesher = Mesher::new();
    let mut out = ChunkMesh::new();
    let good = padded_from_fn(|_, y, _| if y < 16 { blocks.stone } else { blocks.air });
    mesher
        .mesh_into(&input(&palette, &good, None, 0), &mut out)
        .expect("valid");
    assert!(!out.is_empty());

    let short = &good[..PADDED_VOLUME - 1];
    assert_eq!(
        mesher.mesh_into(&input(&palette, short, None, 0), &mut out),
        Err(MeshError::CellCount {
            len: PADDED_VOLUME - 1
        })
    );
    assert!(out.is_empty(), "a failed call leaves the output empty");

    let empty = MeshTables {
        flags: &[],
        render: &[],
        cull_group: &[],
    };
    let unequal = MeshTables {
        cull_group: &palette.cull_group[1..],
        ..palette.tables()
    };
    for tables in [empty, unequal] {
        let input = MeshInput {
            tables,
            ..input(&palette, &good, None, 0)
        };
        assert!(matches!(
            mesher.mesh(&input),
            Err(MeshError::TableLengths { .. })
        ));
    }

    let opaque_air = tables_with(&palette, |flags, _, _| flags[0] = BlockFlags::OPAQUE);
    let grouped_air = tables_with(&palette, |_, _, groups| groups[0] = 5);
    for changed in [&opaque_air, &grouped_air] {
        assert_eq!(
            mesher.mesh(&input(changed, &good, None, 0)),
            Err(MeshError::AirHidesFaces)
        );
    }

    // Unknown states: in a chunk cell, in a face-halo cell, the edit sentinel.
    let unknown = BlockStateId(palette.flags.len() as u32);
    for (cell, state) in [
        ([5, 6, 7], unknown),
        ([31, 31, 31], unknown),
        ([-1, 4, 9], unknown),
        ([12, 32, 3], BlockStateId::UNCHANGED),
        ([0, 0, -1], BlockStateId::UNCHANGED),
    ] {
        let mut cells = good.clone();
        cells[padded_index(cell[0], cell[1], cell[2])] = state;
        assert_eq!(
            mesher.mesh(&input(&palette, &cells, None, 0)),
            Err(MeshError::UnknownState {
                state,
                cell: cell.map(|c| c as i8)
            }),
            "{cell:?}"
        );
    }

    // An invalid uv_rot is an error where the face is shown, as a regular or a wall quad, and
    // ignored where the face is never shown.
    let bad_top = tables_with(&palette, |_, render, _| {
        render[blocks.stone.index()].faces[Face::PosY.index()] = FaceAppearance {
            material: MaterialId(1),
            uv_rot: 4,
        };
    });
    assert_eq!(
        mesher.mesh(&input(&bad_top, &good, None, 0)),
        Err(MeshError::InvalidUvRot {
            state: blocks.stone,
            face: Face::PosY,
            uv_rot: 4
        })
    );
    let bad_bottom = tables_with(&palette, |_, render, _| {
        render[blocks.stone.index()].faces[Face::NegY.index()].uv_rot = 9;
    });
    let layered = |below: BlockStateId| {
        padded_from_fn(move |_, y, _| match y {
            ..=4 => below,
            5..=9 => blocks.stone,
            _ => blocks.air,
        })
    };
    // Stone over air: its bottom faces are regular quads.
    assert_eq!(
        mesher.mesh(&input(&bad_bottom, &layered(blocks.air), None, 0)),
        Err(MeshError::InvalidUvRot {
            state: blocks.stone,
            face: Face::NegY,
            uv_rot: 9
        })
    );
    // Stone down to the boundary over a stone halo (`good`): its bottom faces are walls.
    assert!(matches!(
        mesher.mesh(&input(&bad_bottom, &good, None, 0)),
        Err(MeshError::InvalidUvRot { .. })
    ));
    // Stone resting on dirt: its bottom faces are never shown.
    let mesh = mesher
        .mesh(&input(&bad_bottom, &layered(blocks.dirt), None, 0))
        .expect("the broken face is never shown");
    assert!(!mesh.is_empty());
    assert!(
        mesh.quads
            .iter()
            .all(|q| q.face() != Face::NegY || q.material() != MaterialId(1))
    );

    // Errors are readable.
    let message = MeshError::UnknownState {
        state: unknown,
        cell: [1, 2, 3],
    }
    .to_string();
    assert!(message.contains("[1, 2, 3]"), "{message}");
}

#[test]
fn wide_palette_matches_the_reference() {
    // 300 states with their own materials and rotations per face, every class, opaque and
    // non-opaque models and eight cull groups: many keys per slice and many group planes.
    let count = 300;
    let palette = Palette {
        flags: (0..count)
            .map(|i| match i {
                0 => BlockFlags::empty(),
                _ if i % 3 == 0 => BlockFlags::SOLID | BlockFlags::OPAQUE,
                _ => BlockFlags::SOLID,
            })
            .collect(),
        render: (0..count)
            .map(|i| BlockRender {
                class: match i {
                    0 => RenderClass::None,
                    _ if i % 7 == 0 => RenderClass::Fluid,
                    _ if i % 5 == 0 => RenderClass::Cutout,
                    _ if i % 11 == 0 => RenderClass::Model,
                    _ => RenderClass::Cube,
                },
                faces: Face::ALL.map(|face| FaceAppearance {
                    material: MaterialId(i as u16 * 6 + face.index() as u16),
                    uv_rot: ((i + face.index()) % 4) as u8,
                }),
            })
            .collect(),
        cull_group: (0..count)
            .map(|i| if i % 4 == 1 { (i % 9) as u16 } else { 0 })
            .collect(),
    };
    let mut mesher = Mesher::new();
    let mix: Vec<BlockStateId> = (0..count as u32).map(BlockStateId).collect();
    for seed in 0..4 {
        let mut rng = Rng::new(seed, 99);
        let cells = if seed % 2 == 0 {
            random_cells(&mut rng, &mix, 0.5)
        } else {
            random_boxes(&mut rng, &mix, 60)
        };
        for level in [0, 2] {
            let input = input(&palette, &cells, None, level);
            let mesh = mesher.mesh(&input).expect("valid");
            check_invariants(&mesh, "wide palette");
            assert_same_faces(&mesh_faces(&mesh), &reference_faces(&input), "wide palette");
            assert_same_mesh(&mesh, &reference_mesh(&input), "wide palette");
        }
    }
}
