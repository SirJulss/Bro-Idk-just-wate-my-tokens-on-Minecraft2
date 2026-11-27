//! The wire contracts of contract §2.2–2.3 (v2) in one place, checked through the public API
//! only.
//!
//! Shaders, the mesher, the content pipeline and saved data rely on every rule in this file.
//! The tests here do what those consumers do: decode packed words with nothing but the
//! documented bit layout, place vertices with nothing but the documented geometry and drawing
//! rules, and map textures with nothing but the documented formula, then compare with
//! `cp-core`.

use cp_core::block::{BlockRender, RenderClass};
use cp_core::coords::{
    CHUNK_BITS, CHUNK_SIZE, CHUNK_VOLUME, LodKey, MAX_LOD_LEVEL, VOXEL_LIMIT, cell_index,
    index_to_cell, voxel_to_chunk,
};
use cp_core::face::{Face, Orientation, UV_TO_TEX};
use cp_core::ids::{
    BlockStateId, BodyId, CellAddress, ChunkKey, EntityId, FrameId, GridId, MaterialId,
};
use cp_core::math::IVec3;
use cp_core::quad::{
    CORNER_MAP, CORNER_MAP_NEGATIVE, CORNER_MAP_POSITIVE, PackedQuad, QUAD_INDEX_PATTERN,
    QUAD_INDICES, QUAD_VERTICES,
};

const AXES: [IVec3; 3] = [IVec3::X, IVec3::Y, IVec3::Z];

#[test]
fn chunks_cells_and_lod_keys() {
    assert_eq!(
        (
            CHUNK_BITS,
            CHUNK_SIZE,
            CHUNK_VOLUME,
            MAX_LOD_LEVEL,
            VOXEL_LIMIT
        ),
        (5, 32, 32_768, 20, 1 << 30)
    );
    // index = x + z * 32 + y * 1024
    assert_eq!(cell_index(3, 5, 7), 3 + 7 * 32 + 5 * 1024);
    assert_eq!(index_to_cell(3 + 7 * 32 + 5 * 1024), [3, 5, 7]);
    // chunk = voxel >> 5, local = voxel & 31
    assert_eq!(
        voxel_to_chunk(IVec3::new(-1, 32, 70)),
        (IVec3::new(-1, 1, 2), [31, 0, 6])
    );
    // Chunk min corner at pos * (32 << level) voxels, cell size 2^level.
    for level in 0..=MAX_LOD_LEVEL {
        let key = LodKey {
            level,
            pos: IVec3::new(-3, 2, 1),
        };
        assert_eq!(key.cell_size(), 1 << level);
        assert_eq!(key.chunk_extent(), 32 << level);
        assert_eq!(key.origin_voxel(), IVec3::new(-3, 2, 1) * (32 << level));
        assert_eq!(key.aabb_voxels().size(), IVec3::splat(32 << level));
    }
    // Children in octant order dx + 2 * dz + 4 * dy.
    let children = LodKey::new(4, IVec3::new(1, -1, 0)).children();
    assert_eq!(children[0], LodKey::new(3, IVec3::new(2, -2, 0)));
    assert_eq!(children[1].pos, IVec3::new(3, -2, 0));
    assert_eq!(children[2].pos, IVec3::new(2, -2, 1));
    assert_eq!(children[4].pos, IVec3::new(2, -1, 0));
    assert_eq!(children[7].pos, IVec3::new(3, -1, 1));
    // The edit sentinel.
    assert_eq!(BlockStateId::UNCHANGED.0, u32::MAX);
}

#[test]
fn grid_keys_are_named_where_the_contract_lists_them() {
    // Contract §4.1 lists the grid, chunk and cell keys under `ids`; `coords` defines the chunk
    // and cell keys next to `LodKey`. Both paths name the same types.
    let grid = GridId::Construct(EntityId::new(3, 1));
    let cell: CellAddress = cp_core::coords::CellAddress {
        grid,
        cell: IVec3::new(-1, 40, 5),
    };
    let (chunk, local): (cp_core::coords::ChunkKey, [u8; 3]) = cell.to_chunk();
    let chunk: ChunkKey = chunk;
    assert_eq!(
        chunk,
        ChunkKey {
            grid,
            lod: LodKey::lod0(IVec3::new(-1, 1, 0)),
        }
    );
    assert_eq!(local, [31, 8, 5]);
    // A grid lives in the frame of its owner.
    assert_eq!(grid.frame(), FrameId::Construct(EntityId::new(3, 1)));
    assert_eq!(GridId::from(BodyId(2)).frame(), FrameId::Body(BodyId(2)));
}

#[test]
fn faces_and_tangent_axes() {
    // 0:+X 1:-X 2:+Y 3:-Y 4:+Z 5:-Z; axis = index >> 1; negative = index & 1.
    let normals = [
        IVec3::new(1, 0, 0),
        IVec3::new(-1, 0, 0),
        IVec3::new(0, 1, 0),
        IVec3::new(0, -1, 0),
        IVec3::new(0, 0, 1),
        IVec3::new(0, 0, -1),
    ];
    for (index, face) in Face::ALL.into_iter().enumerate() {
        assert_eq!(face.index(), index);
        assert_eq!(face.normal(), normals[index]);
        assert_eq!(
            (face.axis(), face.is_negative()),
            (index >> 1, index & 1 == 1)
        );
        // u_axis = (n + 1) % 3, v_axis = (n + 2) % 3, u x v = +n on both faces of an axis.
        assert_eq!(
            (face.u_axis(), face.v_axis()),
            ((face.axis() + 1) % 3, (face.axis() + 2) % 3)
        );
        assert_eq!(
            AXES[face.u_axis()].cross(AXES[face.v_axis()]),
            AXES[face.axis()]
        );
    }
}

/// `R(up_world, turn × 90°) ∘ T(up)` exactly as contract §2.3 words it.
fn contract_rotation(up: usize, turn: u8, v: IVec3) -> IVec3 {
    let IVec3 { x, y, z } = v;
    let t = match up {
        2 => IVec3::new(x, y, z),
        3 => IVec3::new(x, -y, -z),
        0 => IVec3::new(y, -x, z),
        1 => IVec3::new(-y, x, z),
        4 => IVec3::new(x, -z, y),
        _ => IVec3::new(x, z, -y),
    };
    // A quarter turn counter-clockwise seen from the tip of n: v -> n x v + n (n . v).
    let n = Face::ALL[up].normal();
    (0..turn).fold(t, |w, _| n.cross(w) + n * n.dot(w))
}

#[test]
fn orientation_rotations() {
    assert_eq!(Orientation::ALL.len(), 24);
    let v = IVec3::new(2, 3, 5);
    for (index, orientation) in Orientation::ALL.into_iter().enumerate() {
        // index = up.index() + 6 * turn
        let (up, turn) = (index % 6, (index / 6) as u8);
        assert_eq!((orientation.up().index(), orientation.turn()), (up, turn));
        assert_eq!(orientation.index(), index);
        assert_eq!(orientation.rotate_vec(v), contract_rotation(up, turn, v));
        assert_eq!(orientation.inverse_rotate_vec(orientation.rotate_vec(v)), v);
        assert_eq!(orientation.rotate_vec(IVec3::Y), Face::ALL[up].normal());
        // The front is local +Z.
        assert_eq!(
            orientation.front().normal(),
            orientation.rotate_vec(IVec3::Z)
        );
        // Proper rotation: right-handed triple stays right-handed.
        let [x, y, z] = AXES.map(|axis| orientation.rotate_vec(axis));
        assert_eq!(x.cross(y), z);
    }
    assert_eq!(Orientation::IDENTITY.index(), 2);
}

/// Decodes a quad with nothing but the documented bit layout.
struct Decoded {
    cell: IVec3,
    w: i32,
    h: i32,
    face: usize,
    material: u32,
    uv_rot: u32,
    authored: bool,
    payload: u32,
}

fn decode(w0: u32, w1: u32) -> Decoded {
    assert_eq!(w0 >> 28, 0, "w0 bits 28-31 are reserved");
    assert_eq!(w1 >> 31, 0, "w1 bit 31 is reserved");
    Decoded {
        cell: IVec3::new(
            (w0 & 31) as i32,
            ((w0 >> 5) & 31) as i32,
            ((w0 >> 10) & 31) as i32,
        ),
        w: ((w0 >> 15) & 31) as i32 + 1,
        h: ((w0 >> 20) & 31) as i32 + 1,
        face: ((w0 >> 25) & 7) as usize,
        material: w1 & 0xFFFF,
        uv_rot: (w1 >> 16) & 3,
        authored: (w1 >> 18) & 1 == 1,
        payload: (w1 >> 19) & 0xFFF,
    }
}

/// What the vertex shader does for `vertex_index` (counting vertices of all quads): position
/// of the vertex in cells.
fn pulled_vertex(quad: &Decoded, vertex_index: u32) -> IVec3 {
    let (axis, negative) = (quad.face >> 1, quad.face & 1 == 1);
    let (u_axis, v_axis) = ((axis + 1) % 3, (axis + 2) % 3);
    // corner = CORNER_MAP[face sign][vertex_index & 3], maps [0,1,2,3] and [0,3,2,1].
    let corner = [[0, 1, 2, 3], [0, 3, 2, 1]][usize::from(negative)][(vertex_index & 3) as usize];
    // c0 = (0, 0), c1 = (w, 0), c2 = (w, h), c3 = (0, h) in (u, v).
    let (cu, cv) = [(0, 0), (quad.w, 0), (quad.w, quad.h), (0, quad.h)][corner];
    // The plane is at cell_n + 1 for positive faces and at cell_n for negative faces.
    let mut position = quad.cell;
    position[axis] += i32::from(!negative);
    position[u_axis] += cu;
    position[v_axis] += cv;
    position
}

#[test]
fn packed_quads_decode_and_wind_as_documented() {
    assert_eq!(QUAD_INDEX_PATTERN, [0, 1, 2, 0, 2, 3]);
    assert_eq!((QUAD_VERTICES, QUAD_INDICES), (4, 6));
    assert_eq!(CORNER_MAP_POSITIVE, [0, 1, 2, 3]);
    assert_eq!(CORNER_MAP_NEGATIVE, [0, 3, 2, 1]);
    assert_eq!(CORNER_MAP, [CORNER_MAP_POSITIVE, CORNER_MAP_NEGATIVE]);
    assert_eq!(size_of::<PackedQuad>(), 8);

    // A golden quad, word for word.
    let golden = PackedQuad::new([1, 2, 3], 4, 5, Face::NegY, MaterialId(0x1234), 2, true)
        .unwrap()
        .with_payload(0x7FF)
        .unwrap();
    assert_eq!((golden.w0, golden.w1), (0x0641_8C41, 0x3FFE_1234));
    assert_eq!(
        gpu_bytes(&golden),
        [0x41, 0x8C, 0x41, 0x06, 0x34, 0x12, 0xFE, 0x3F]
    );

    let mut checked = 0;
    for face in Face::ALL {
        for cell in [[0_u8, 0, 0], [31, 31, 31], [7, 0, 19], [16, 30, 2]] {
            for (w, h) in [(1_u8, 1_u8), (2, 1), (1, 3), (5, 9), (32, 32)] {
                for (uv_rot, authored, material, payload) in
                    [(0_u8, false, 0_u16, 0_u16), (3, true, 0xFFFF, 0xFFF)]
                {
                    let material = MaterialId(material);
                    let Ok(quad) = PackedQuad::new(cell, w, h, face, material, uv_rot, authored)
                    else {
                        // The rectangle would leave the chunk; that is rejected, not packed.
                        let u_end = u32::from(cell[face.u_axis()]) + u32::from(w);
                        let v_end = u32::from(cell[face.v_axis()]) + u32::from(h);
                        assert!(u_end > 32 || v_end > 32);
                        continue;
                    };
                    let quad = quad.with_payload(payload).unwrap();
                    let decoded = decode(quad.w0, quad.w1);
                    assert_eq!(decoded.cell.to_array().map(|c| c as u8), cell);
                    assert_eq!((decoded.w, decoded.h), (i32::from(w), i32::from(h)));
                    assert_eq!(decoded.face, face.index());
                    assert_eq!(decoded.material, u32::from(material.0));
                    assert_eq!(decoded.uv_rot, u32::from(uv_rot));
                    assert_eq!(decoded.authored, authored);
                    assert_eq!(decoded.payload, u32::from(payload));

                    // The quad is quad number 7 of its range: vertices 28..32, indices 42..48.
                    let base = 7 * QUAD_VERTICES;
                    let vertices = quad.vertices();
                    for (k, expected) in vertices.iter().enumerate() {
                        assert_eq!(pulled_vertex(&decoded, base + k as u32), *expected);
                    }
                    // Front faces are counter-clockwise seen from outside.
                    for triangle in QUAD_INDEX_PATTERN.chunks(3) {
                        let [a, b, c] =
                            [0, 1, 2].map(|k| pulled_vertex(&decoded, base + triangle[k]));
                        assert!((b - a).cross(c - a).dot(face.normal()) > 0);
                    }
                    checked += 1;
                }
            }
        }
    }
    assert!(checked > 120, "{checked}");
}

/// Byte image of a quad as the GPU buffer holds it (little-endian words).
fn gpu_bytes(quad: &PackedQuad) -> [u8; 8] {
    let mut bytes = [0; 8];
    bytes[..4].copy_from_slice(&quad.w0.to_le_bytes());
    bytes[4..].copy_from_slice(&quad.w1.to_le_bytes());
    bytes
}

/// What the fragment shader does: texture coordinates from the quad-plane coordinates
/// `(u, v)` along the face's tangent axes, the face index and `uv_rot`. Contract §2.3 (v2):
/// texturing frame `(a, b) = (u, v)` on positive and `(v, u)` on negative faces, then the
/// rotation list.
fn shader_tex(face: usize, uv_rot: u8, u: i32, v: i32) -> [i32; 2] {
    let (a, b) = if face & 1 == 1 { (v, u) } else { (u, v) };
    match uv_rot {
        0 => [a, b],
        1 => [b, -a],
        2 => [-a, -b],
        _ => [-b, a],
    }
}

/// The other way a shader can do it: look the mapping up in a constant table generated from
/// `cp_core::face::UV_TO_TEX`.
fn shader_tex_from_table(face: usize, uv_rot: u8, u: i32, v: i32) -> [i32; 2] {
    let [[su, sv], [tu, tv]] = UV_TO_TEX[face][usize::from(uv_rot)].map(|row| row.map(i32::from));
    [su * u + sv * v, tu * u + tv * v]
}

#[test]
fn texture_rule_table_and_formula_agree_on_all_faces_and_rotations() {
    // The comparison every shader implementation of the rule has to pass: 6 faces x 4 rotations
    // against the normative `Face::uv_to_tex`.
    assert_eq!((UV_TO_TEX.len(), UV_TO_TEX[0].len()), (6, 4));
    for face in Face::ALL {
        for uv_rot in 0..4_u8 {
            for (u, v) in [(1, 0), (0, 1), (2, -3), (-7, 4)] {
                let expected = face.uv_to_tex(uv_rot, u, v);
                assert_eq!(shader_tex(face.index(), uv_rot, u, v), expected);
                assert_eq!(shader_tex_from_table(face.index(), uv_rot, u, v), expected);
            }
        }
    }
    // Rotating (u, v) without the swap, as contract v1 printed it, is the rule on positive
    // faces only; on every negative face it is a different (mirrored) mapping.
    let unswapped = |uv_rot: u8, u: i32, v: i32| match uv_rot {
        0 => [u, v],
        1 => [v, -u],
        2 => [-u, -v],
        _ => [-v, u],
    };
    for face in Face::ALL {
        for uv_rot in 0..4_u8 {
            let same = face.uv_to_tex(uv_rot, 2, -3) == unswapped(uv_rot, 2, -3);
            assert_eq!(same, !face.is_negative(), "{face:?} uv_rot {uv_rot}");
        }
    }
}

#[test]
fn textures_follow_the_block_through_every_orientation() {
    // A block with six different materials, placed in all 24 orientations. For every world
    // face, moving across the face must move through the texture exactly as it does on the
    // face of the unrotated block that ended up there.
    let local_materials = [10, 11, 20, 30, 12, 13].map(MaterialId);
    for orientation in Orientation::ALL {
        let render = BlockRender::oriented(RenderClass::Cube, local_materials, orientation);
        for world in Face::ALL {
            let appearance = render.face(world);
            let local = orientation.local_face(world);
            assert_eq!(appearance.material, local_materials[local.index()]);
            assert_eq!(appearance.uv_rot, orientation.uv_rot(world));
            assert_eq!(orientation.world_face(local), world);

            // Texture axes of the local face (image right, image up), from contract §2.3.
            let (tex_u, tex_v) = match local {
                Face::PosY => (IVec3::X, IVec3::NEG_Z),
                Face::NegY => (IVec3::X, IVec3::Z),
                side => (IVec3::Y.cross(side.normal()), IVec3::Y),
            };
            assert_eq!(tex_u.cross(tex_v), local.normal());

            for (u, v) in [(1, 0), (0, 1), (2, -3)] {
                // A displacement in the world face's plane...
                let step = AXES[world.u_axis()] * u + AXES[world.v_axis()] * v;
                // ...seen from the block...
                let local_step = orientation.inverse_rotate_vec(step);
                // ...moves through the image by this much:
                let expected = [local_step.dot(tex_u), local_step.dot(tex_v)];
                assert_eq!(shader_tex(world.index(), appearance.uv_rot, u, v), expected);
                assert_eq!(world.uv_to_tex(appearance.uv_rot, u, v), expected);
            }
        }
    }
    // The values for the unrotated block.
    let identity = Face::ALL.map(|face| Orientation::IDENTITY.uv_rot(face));
    assert_eq!(identity, [3, 0, 1, 0, 0, 3]);
}
