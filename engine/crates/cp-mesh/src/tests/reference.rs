//! The obviously-correct reference: one face per cell, straight from the wording of contract
//! §5.2; the greedy cover of those faces by the rule the crate documents, written out the slow
//! way; and the checks every mesh has to pass. Shares no code with the mesher.

use std::collections::BTreeMap;

use cp_core::block::{BlockFlags, RenderClass};
use cp_core::coords::cell_index;
use cp_core::face::Face;
use cp_core::ids::{BlockStateId, MaterialId};
use cp_core::quad::PackedQuad;

use crate::{
    CellBounds, ChunkMesh, FACE_STORAGE_ORDER, Layer, LayerRanges, MeshInput, RANGES_PER_LAYER,
    WALL_RANGE_OFFSET, padded_index,
};

/// One cell face as the reference emits it and as a quad covers it.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub(crate) struct FaceRecord {
    pub layer: Layer,
    pub wall: bool,
    pub face: Face,
    pub cell: [u8; 3],
    pub material: MaterialId,
    pub uv_rot: u8,
    pub authored: bool,
}

/// Every face of the chunk, sorted. Expects valid input.
pub(crate) fn reference_faces(input: &MeshInput<'_>) -> Vec<FaceRecord> {
    let tables = &input.tables;
    let state = |x: i32, y: i32, z: i32| input.cells[padded_index(x, y, z)];
    // "B hides A iff B is OPAQUE, or cull_group(A) != 0 && cull_group(A) == cull_group(B)."
    let hides = |b: BlockStateId, a: BlockStateId| {
        let group = tables.cull_group[a.index()];
        tables.flags[b.index()].contains(BlockFlags::OPAQUE)
            || (group != 0 && group == tables.cull_group[b.index()])
    };
    let mut faces = Vec::new();
    for y in 0..32 {
        for z in 0..32 {
            for x in 0..32 {
                let a = state(x, y, z);
                let render = tables.render[a.index()];
                let layer = match render.class {
                    RenderClass::Cube => Layer::Opaque,
                    RenderClass::Cutout if input.level < 2 => Layer::Cutout,
                    RenderClass::Cutout => Layer::Opaque,
                    RenderClass::Fluid => Layer::Fluid,
                    RenderClass::None | RenderClass::Model => continue,
                };
                let cell = [x as u8, y as u8, z as u8];
                let index = cell_index(cell[0], cell[1], cell[2]);
                let authored = input
                    .authored
                    .is_some_and(|mask| (mask[index / 64] >> (index % 64)) & 1 == 1);
                for face in Face::ALL {
                    let n = face.normal();
                    let (nx, ny, nz) = (x + n.x, y + n.y, z + n.z);
                    let on_boundary = [nx, ny, nz].iter().any(|c| !(0..32).contains(c));
                    let wall = if !hides(state(nx, ny, nz), a) {
                        false
                    } else if on_boundary && !hides(BlockStateId::AIR, a) {
                        // Exists if the halo were air, but the real halo culls it.
                        true
                    } else {
                        continue;
                    };
                    let appearance = render.faces[face.index()];
                    faces.push(FaceRecord {
                        layer,
                        wall,
                        face,
                        cell,
                        material: appearance.material,
                        uv_rot: appearance.uv_rot,
                        authored,
                    });
                }
            }
        }
    }
    faces.sort_unstable();
    faces
}

/// The whole mesh the documented rules make from [`reference_faces`], quad for quad (crate
/// documentation, "Greedy merge" and "Output"): the faces of one range, slice and merge key
/// form a 32 × 32 matrix (rows along the face's `v` axis, columns along `u`); scanning it in
/// `(v, u)` order, every face not yet covered starts a quad, which takes the uncovered run
/// along `u` and grows along `v` while the next row holds the whole run. Each range is then
/// sorted by `(plane, v, u)` of the min cells, and the bounds are the tight boxes of the
/// covered cells.
pub(crate) fn reference_mesh(input: &MeshInput<'_>) -> ChunkMesh {
    type MergeKey = (MaterialId, u8, bool);
    // (layer, range slot, slice, merge key) -> faces[v][u].
    let mut matrices: BTreeMap<(Layer, usize, u8, MergeKey), [[bool; 32]; 32]> = BTreeMap::new();
    for record in reference_faces(input) {
        let face = record.face;
        let slot = FACE_STORAGE_ORDER
            .iter()
            .position(|&stored| stored == face)
            .expect("every face has a slot")
            + if record.wall { WALL_RANGE_OFFSET } else { 0 };
        let key = (record.material, record.uv_rot, record.authored);
        let slice = record.cell[face.axis()];
        let faces = matrices
            .entry((record.layer, slot, slice, key))
            .or_insert([[false; 32]; 32]);
        faces[usize::from(record.cell[face.v_axis()])][usize::from(record.cell[face.u_axis()])] =
            true;
    }
    // (layer, slot) -> quads with their in-range order (plane, v, u).
    type Ordered = ((u8, u8, u8), PackedQuad);
    let mut ranges: BTreeMap<(Layer, usize), Vec<Ordered>> = BTreeMap::new();
    for ((layer, slot, slice, (material, uv_rot, authored)), mut faces) in matrices {
        let face = FACE_STORAGE_ORDER[slot % 6];
        for v in 0..32 {
            for u in 0..32 {
                if !faces[v][u] {
                    continue;
                }
                let mut w = 1;
                while u + w < 32 && faces[v][u + w] {
                    w += 1;
                }
                let mut h = 1;
                while v + h < 32 && faces[v + h][u..u + w].iter().all(|&f| f) {
                    h += 1;
                }
                for row in &mut faces[v..v + h] {
                    row[u..u + w].fill(false);
                }
                let mut cell = [0_u8; 3];
                cell[face.axis()] = slice;
                cell[face.u_axis()] = u as u8;
                cell[face.v_axis()] = v as u8;
                let quad =
                    PackedQuad::new(cell, w as u8, h as u8, face, material, uv_rot, authored)
                        .expect("a rectangle of chunk faces");
                let plane = slice + u8::from(!face.is_negative());
                ranges
                    .entry((layer, slot))
                    .or_default()
                    .push(((plane, v as u8, u as u8), quad));
            }
        }
    }
    let mut mesh = ChunkMesh::new();
    for layer in Layer::ALL {
        let first = mesh.quads.len() as u32;
        let mut counts = [0; RANGES_PER_LAYER];
        // Regular and wall quads.
        let mut bounds: [Option<CellBounds>; 2] = [None, None];
        for (slot, count) in counts.iter_mut().enumerate() {
            let mut quads = ranges.remove(&(layer, slot)).unwrap_or_default();
            quads.sort_unstable_by_key(|&(order, _)| order);
            *count = quads.len() as u32;
            for (_, quad) in quads {
                for cell in quad.covered_cells() {
                    let entry = &mut bounds[usize::from(slot >= WALL_RANGE_OFFSET)];
                    let single = CellBounds::cell(cell);
                    *entry = Some(entry.map_or(single, |known| known.union(single)));
                }
                mesh.quads.push(quad);
            }
        }
        let all = match bounds {
            [Some(regular), Some(walls)] => Some(regular.union(walls)),
            [regular, walls] => regular.or(walls),
        };
        mesh.layers[layer.index()] = LayerRanges {
            first,
            counts,
            bounds: all,
            regular_bounds: bounds[0],
            wall_bounds: bounds[1],
        };
    }
    mesh
}

/// Panics with the first difference if `actual` is not exactly `expected`.
pub(crate) fn assert_same_mesh(actual: &ChunkMesh, expected: &ChunkMesh, context: &str) {
    if actual == expected {
        return;
    }
    for layer in Layer::ALL {
        assert_eq!(
            actual.layer(layer),
            expected.layer(layer),
            "{context}: ranges of {layer:?}"
        );
    }
    let first = actual
        .quads
        .iter()
        .zip(&expected.quads)
        .position(|(a, e)| a != e)
        .unwrap_or(actual.quads.len().min(expected.quads.len()));
    let describe = |quad: Option<&PackedQuad>| {
        quad.map(|q| {
            format!(
                "{:?} {:?} {}x{} material {} uv_rot {} authored {}",
                q.face(),
                q.cell(),
                q.width(),
                q.height(),
                q.material(),
                q.uv_rot(),
                q.is_authored()
            )
        })
    };
    panic!(
        "{context}: {} quads, reference {}; first difference at {first}: {:?}, reference {:?}",
        actual.quads.len(),
        expected.quads.len(),
        describe(actual.quads.get(first)),
        describe(expected.quads.get(first)),
    );
}

/// The faces covered by the quads of `mesh`, one record per covered cell face, sorted. Layer
/// and wall flag come from the range each quad is stored in.
pub(crate) fn mesh_faces(mesh: &ChunkMesh) -> Vec<FaceRecord> {
    let mut faces = Vec::new();
    for layer in Layer::ALL {
        let ranges = mesh.layer(layer);
        for slot in 0..RANGES_PER_LAYER {
            for quad in &mesh.quads[ranges.slot(slot)] {
                faces.extend(quad.covered_cells().map(|cell| FaceRecord {
                    layer,
                    wall: slot >= WALL_RANGE_OFFSET,
                    face: quad.face(),
                    cell,
                    material: quad.material(),
                    uv_rot: quad.uv_rot(),
                    authored: quad.is_authored(),
                }));
            }
        }
    }
    faces.sort_unstable();
    faces
}

/// Panics with the first difference if the two sorted face lists differ.
pub(crate) fn assert_same_faces(actual: &[FaceRecord], expected: &[FaceRecord], context: &str) {
    if actual == expected {
        return;
    }
    let first = actual
        .iter()
        .zip(expected)
        .position(|(a, e)| a != e)
        .unwrap_or(actual.len().min(expected.len()));
    panic!(
        "{context}: mesh covers {} faces, reference has {}; first difference at {first}: \
         mesh {:?}, reference {:?}",
        actual.len(),
        expected.len(),
        actual.get(first),
        expected.get(first),
    );
}

/// The checks of the mesh invariants (contract §5.2, [`ChunkMesh`]): layout of the layers and
/// ranges, valid quads on the face of their range with payload 0, walls on the boundary, the
/// in-range sort order, no two quads covering the same cell face, tight bounds.
pub(crate) fn check_invariants(mesh: &ChunkMesh, context: &str) {
    let mut next = 0;
    let mut covered = vec![false; 6 * 32 * 32 * 32];
    for layer in Layer::ALL {
        let ranges = mesh.layer(layer);
        assert_eq!(
            ranges.first as usize, next,
            "{context}: {layer:?} starts after a gap"
        );
        next = ranges.range().end;
        // Regular, wall and all quads of the layer.
        let mut bounds: [Option<CellBounds>; 3] = [None; 3];
        for slot in 0..RANGES_PER_LAYER {
            let wall = slot >= WALL_RANGE_OFFSET;
            let face = FACE_STORAGE_ORDER[slot % 6];
            let (n, u, v) = (face.axis(), face.u_axis(), face.v_axis());
            let mut previous = None;
            for quad in &mesh.quads[ranges.slot(slot)] {
                assert_eq!(quad.validate(), Ok(()), "{context}: invalid quad {quad:?}");
                assert_eq!(quad.payload(), 0, "{context}: payload must be 0");
                assert_eq!(
                    quad.face(),
                    face,
                    "{context}: quad in the range of {face:?}"
                );
                let cell = quad.cell();
                if wall {
                    let boundary = if face.is_negative() { 0 } else { 31 };
                    assert_eq!(cell[n], boundary, "{context}: wall quad off the boundary");
                }
                let plane = cell[n] + u8::from(!face.is_negative());
                let order = (plane, cell[v], cell[u]);
                assert!(
                    previous < Some(order),
                    "{context}: {layer:?} slot {slot}: {order:?} after {previous:?}"
                );
                previous = Some(order);
                for covered_cell in quad.covered_cells() {
                    let index = face.index() * 32_768
                        + cell_index(covered_cell[0], covered_cell[1], covered_cell[2]);
                    assert!(
                        !covered[index],
                        "{context}: face {face:?} of cell {covered_cell:?} covered twice"
                    );
                    covered[index] = true;
                    let single = CellBounds::cell(covered_cell);
                    for entry in [usize::from(wall), 2] {
                        let entry = &mut bounds[entry];
                        *entry = Some(entry.map_or(single, |known| known.union(single)));
                    }
                }
            }
        }
        assert_eq!(
            ranges.regular_bounds, bounds[0],
            "{context}: {layer:?} regular bounds"
        );
        assert_eq!(
            ranges.wall_bounds, bounds[1],
            "{context}: {layer:?} wall bounds"
        );
        // Contract §5.2: the layer's bounds are tight over all its quads, walls included.
        assert_eq!(ranges.bounds, bounds[2], "{context}: {layer:?} bounds");
        assert_eq!(ranges.bounds.is_none(), ranges.is_empty(), "{context}");
    }
    assert_eq!(
        next,
        mesh.quads.len(),
        "{context}: layers do not cover the quads"
    );
}
