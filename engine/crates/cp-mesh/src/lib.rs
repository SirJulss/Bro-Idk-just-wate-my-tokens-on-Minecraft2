//! `cp-mesh`: binary greedy meshing of padded voxel chunks into packed quads.
//!
//! Binding contract: `engine/docs/ARCHITEKTUR.md`, sections 2.3, 4.1 and 5.2.
//!
//! [`Mesher::mesh_into`] turns one padded chunk ([`MeshInput`]) into a [`ChunkMesh`]: a list of
//! [`PackedQuad`](cp_core::quad::PackedQuad)s in three layers of twelve ranges. A [`Mesher`]
//! owns its scratch buffers and is reused for many chunks; it reads nothing but its input, so
//! one mesher per worker thread can mesh chunks in any order. The same code meshes every LOD
//! level and every grid (bodies and constructs).
//!
//! ```
//! use cp_core::block::{BlockFlags, BlockRender, RenderClass};
//! use cp_core::face::{Face, Orientation};
//! use cp_core::ids::{BlockStateId, MaterialId};
//! use cp_mesh::{Layer, MeshInput, MeshTables, Mesher, PADDED_VOLUME, padded_index};
//!
//! // State 0 is air, state 1 a stone cube of material 7 (normally the registry's tables).
//! let flags = [BlockFlags::empty(), BlockFlags::SOLID | BlockFlags::OPAQUE];
//! let stone = BlockRender::oriented(RenderClass::Cube, [MaterialId(7); 6], Orientation::IDENTITY);
//! let render = [BlockRender::NONE, stone];
//! let cull_group = [0, 0];
//! let tables = MeshTables { flags: &flags, render: &render, cull_group: &cull_group };
//!
//! // One stone cell at (3, 4, 5) in an air chunk with an air halo.
//! let mut cells = vec![BlockStateId::AIR; PADDED_VOLUME];
//! cells[padded_index(3, 4, 5)] = BlockStateId(1);
//! let input = MeshInput { cells: &cells, authored: None, level: 0, tables };
//!
//! let mut mesher = Mesher::new();
//! let mesh = mesher.mesh(&input)?;
//! assert_eq!(mesh.layer(Layer::Opaque).len(), 6); // one quad per face, no walls
//! let top = mesh.regular_quads(Layer::Opaque, Face::PosY)[0];
//! assert_eq!((top.cell(), top.width(), top.height()), ([3, 4, 5], 1, 1));
//! assert_eq!(top.material(), MaterialId(7));
//! # Ok::<(), cp_mesh::MeshError>(())
//! ```
//!
//! # Input
//! - `cells`: the [`PADDED_VOLUME`] states of the padded chunk, its 32³ cells plus a one-cell
//!   halo. Cell `(x, y, z)`, each coordinate in `-1..=32`, is at [`padded_index`]`(x, y, z)` =
//!   `(x+1) + (z+1)·34 + (y+1)·1156`. Face-halo cells (exactly one coordinate outside
//!   `0..32`) are the neighbour chunks' cells at the same level; edge and corner halo cells are
//!   never read, so their contents do not matter.
//! - `authored`: optional mask of the chunk's cells that carry a player edit (bit `i` of word
//!   `i / 64` for the cell index `i = x + 32·z + 1024·y`).
//! - `level`: the chunk's LOD level; it only decides the layer of cutout states.
//! - [`MeshTables`]: per-state flags, render descriptions and cull groups.
//!
//! A `cp_voxel::PaddedChunk` maps onto [`MeshInput`] field by field: `cells: padded.cells()`,
//! `authored: padded.authored()`, `level: padded.level()`; the tables are the registry's
//! `BlockTables` (contract §4.4) as slices.
//!
//! # Face rule
//! The face of cell A towards its neighbour B is emitted iff A is renderable (render class
//! `Cube`, `Cutout` or `Fluid`) and B does not hide it. B hides A iff B is `OPAQUE`, or A has a
//! cull group (`!= 0`) and B has the same one. `None` and `Model` cells emit nothing (the
//! instanced model path is reserved, contract §1). Cube faces go to [`Layer::Opaque`], fluid
//! faces to [`Layer::Fluid`], cutout faces to [`Layer::Cutout`] on levels 0 and 1 and to
//! [`Layer::Opaque`] from level 2 on (the renderer draws them with their material colour).
//! Every face takes material and `uv_rot` from the state's `BlockRender` entry for the world
//! face it lies on.
//!
//! # Walls
//! For each of the six chunk boundaries, the boundary faces that would exist if the halo were
//! air, but are hidden by the real halo, are emitted as wall quads in ranges of their own. Air
//! (state 0) hides nothing (checked), so these are the faces of renderable boundary cells
//! whose halo neighbour hides them. The renderer draws a wall range only where the neighbour
//! chunk is not drawn at the same level (contract §5.3).
//!
//! # Greedy merge
//! Faces of one direction and one plane merge into rectangles when their merge key
//! `(material, uv_rot, authored)` is equal (and they belong to the same layer and the same kind
//! of range). Visibility is computed with 64-bit row masks (one bit per cell along x), the
//! faces of each `(plane, key)` are collected into a 32 × 32 bit matrix (rows along the face's
//! `v` axis, bits along its `u` axis) and each matrix is covered greedily: the lowest set bit
//! of the lowest row starts a quad, which takes the run of set bits along `u` and grows along
//! `v` while the next row holds the whole run. The class payload of every quad is 0. This
//! rule is part of the output (the tests compare every mesh quad for quad with a slow
//! implementation of it), so changing it means bumping [`MESHER_VERSION`].
//!
//! # Output
//! [`ChunkMesh::quads`] holds the layers in the order opaque, cutout, fluid. Inside a layer
//! come the six regular face ranges in [`FACE_STORAGE_ORDER`] (`+X, +Y, +Z, -X, -Y, -Z`), then
//! the six wall ranges in the same order. Quads inside a range are sorted by the coordinate of
//! their plane on the normal axis, then by the min cell's `v`, then by its `u` coordinate (all
//! ascending). [`LayerRanges`] records the ranges of each layer and its tight cell bounds:
//! [`LayerRanges::bounds`] over all its quads, walls included (the box of contract §5.2), and
//! for draws without walls the smaller [`LayerRanges::regular_bounds`] (with
//! [`LayerRanges::wall_bounds`] the box of the walls alone).
//! The output is a pure function of the input: identical input gives identical quads in
//! identical order, whatever the mesher meshed before or which state ids the tables use.
//! A golden test pins it next to [`MESH_FORMAT_VERSION`] and [`MESHER_VERSION`].
//!
//! # Deviations from contract §5.2
//! - **Plain slices instead of cp-voxel types.** §5.2 names a `PaddedChunk` and the chunk's
//!   authored mask as the input. [`MeshInput`] takes the same data as slices (see *Input*), so
//!   cp-mesh does not depend on cp-voxel (§3.1 would allow it) and its tests build chunks
//!   directly.
//! - **No `lod` table.** §5.2 lists `MeshTables { flags, render, cull_group, lod }`;
//!   [`MeshTables`] has no `lod` field. The face rule reads the `OPAQUE` flag, the render
//!   class, the face appearances and the cull groups, nothing else, and LOD only enters
//!   through `level` (the cutout layer). The `lod` table drives the downsampling in cp-voxel,
//!   which happens before a chunk is meshed. A field the mesher ignored would suggest that
//!   coarse levels mesh differently, for example that cutout cells of `LodClass::Solid` hide
//!   their neighbours from level 2 on; they do not, on any level.

mod mesh;
mod mesher;

#[cfg(test)]
mod tests;

pub use mesh::{
    CUTOUT_MAX_LEVEL, CellBounds, ChunkMesh, FACE_STORAGE_ORDER, Layer, LayerRanges,
    RANGES_PER_LAYER, WALL_RANGE_OFFSET, storage_slot,
};
pub use mesher::{MeshError, MeshInput, MeshTables, Mesher};

/// Version of the mesh format: the [`PackedQuad`](cp_core::quad::PackedQuad) words and the
/// layer and range layout of [`ChunkMesh`], bounds included. Part of the mesh-cache key
/// (contract §6); bump it on every change of either. A golden test pins it together with
/// hashes of the output.
///
/// - 2: packed quads v2 (contract §4.1), three layers of six regular and six wall ranges.
/// - 3: [`LayerRanges::bounds`] covers all quads of a layer, walls included (contract §5.2);
///   [`LayerRanges::regular_bounds`] holds the box of the regular ranges. Quads unchanged.
pub const MESH_FORMAT_VERSION: u32 = 3;

/// Version of the meshing algorithm. Part of the mesh-cache key (contract §6); bump it on every
/// change that alters the output for some input (another merge order, another face rule). A
/// golden test pins it together with hashes of the output, so such a change fails until the
/// version is bumped and the hashes are re-pinned.
pub const MESHER_VERSION: u32 = 1;

/// Edge of a padded chunk in cells: 32 cells and a one-cell halo on each side.
pub const PADDED_SIZE: usize = 34;

/// Number of cells of a padded chunk, `34³ = 39304`.
pub const PADDED_VOLUME: usize = PADDED_SIZE * PADDED_SIZE * PADDED_SIZE;

/// Index of the cell `(x, y, z)` of a padded chunk, each coordinate in `-1..=32`:
/// `(x+1) + (z+1)·34 + (y+1)·1156` (x fastest, then z, then y, like chunk cells).
///
/// # Panics
/// In debug builds, if a coordinate is outside `-1..=32`.
#[inline]
#[must_use]
pub const fn padded_index(x: i32, y: i32, z: i32) -> usize {
    debug_assert!(
        -1 <= x && x <= 32 && -1 <= y && y <= 32 && -1 <= z && z <= 32,
        "padded cell coordinate outside -1..=32"
    );
    (x + 1) as usize + (z + 1) as usize * PADDED_SIZE + (y + 1) as usize * PADDED_SIZE * PADDED_SIZE
}
