//! The voxel scene on the GPU: the quad pool, the draw item table and the object table
//! (contract §5.3 "Data on the GPU"), and the types its callers use.
//!
//! cp-app (and later cp-stream) turns every chunk mesh of cp-mesh into a [`ChunkUpload`] and
//! hands it to [`VoxelScene::insert_chunk`]. The upload is checked and queued; the renderer
//! copies queued meshes into the pool at the start of each frame within the upload budget, and
//! a chunk is drawn from the frame its mesh arrived in. Replacing a mesh keeps the old one
//! drawn until the new one is in the pool, so no chunk disappears for a frame.
//!
//! The checks cost about 10 ns per quad. Streaming moves them off the frame thread: the meshing
//! worker turns its mesh into a [`PreparedChunk`] (which owns the quads and is `Send`), and the
//! frame thread hands that to [`VoxelScene::insert_prepared`] or
//! [`VoxelScene::replace_prepared`], which only reserve space and queue it, without a copy.

use std::ops::Range;

use cp_core::face::Face;
use cp_core::math::{IAabb, IVec3};
use cp_core::quad::{PackedQuad, QuadError};
use thiserror::Error;

use super::camera::{FrameCamera, GridPose, compose_object};
use super::layout::{
    Fade, GpuDrawItem, GpuObject, MESH_LAYERS, POOL_UNIT_QUADS, QUAD_BYTES, RANGES_PER_LAYER,
    WallMask,
};
use super::state::{PreparedMesh, QuadWrite, SceneState, prepare_mesh};
use crate::error::RenderError;
use crate::gpu::GpuContext;
use crate::scope::capture;

/// A grid that is drawn (a body or a construct), by its slot in the object table.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ObjectId(pub(crate) u32);

impl ObjectId {
    /// Index in the object table (the `object` field of the draw items).
    #[must_use]
    pub fn index(self) -> u32 {
        self.0
    }
}

/// A chunk of the scene. Ids of removed chunks are never valid again (the slot's generation
/// changes).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ChunkId {
    pub(crate) index: u32,
    pub(crate) generation: u32,
}

impl ChunkId {
    /// Slot of the chunk.
    #[must_use]
    pub fn index(self) -> u32 {
        self.index
    }

    /// Generation of the slot when the chunk was inserted.
    #[must_use]
    pub fn generation(self) -> u32 {
        self.generation
    }
}

/// A grid to draw: its voxel bounds (half-open, grid-local, inside the voxel limit; they
/// decide the anchor and should contain every chunk of the grid) and its pose.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ObjectDesc {
    /// Voxels the grid can contain.
    pub bounds: IAabb,
    /// Where the grid lies in the camera's reference frame.
    pub pose: GridPose,
}

/// Quads per range slot of one mesh layer: the six regular face ranges and the six wall ranges,
/// each in the storage order `+X, +Y, +Z, -X, -Y, -Z` (contract §5.2).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct LayerCounts {
    /// Regular quads per face, storage order.
    pub regular: [u32; 6],
    /// Wall quads per face, storage order.
    pub walls: [u32; 6],
}

impl LayerCounts {
    /// A layer without quads.
    pub const EMPTY: Self = Self {
        regular: [0; 6],
        walls: [0; 6],
    };

    /// The counts of the twelve slots of cp-mesh's `LayerRanges::counts` (regular slots
    /// `0..6`, wall slots `6..12`).
    #[must_use]
    pub fn from_slots(slots: [u32; RANGES_PER_LAYER]) -> Self {
        let mut counts = Self::EMPTY;
        counts.regular.copy_from_slice(&slots[..6]);
        counts.walls.copy_from_slice(&slots[6..]);
        counts
    }

    /// The twelve slots.
    #[must_use]
    pub fn slots(&self) -> [u32; RANGES_PER_LAYER] {
        let mut slots = [0; RANGES_PER_LAYER];
        slots[..6].copy_from_slice(&self.regular);
        slots[6..].copy_from_slice(&self.walls);
        slots
    }

    /// Quads of the layer.
    #[must_use]
    pub fn total(&self) -> u64 {
        self.regular
            .iter()
            .chain(&self.walls)
            .map(|&count| u64::from(count))
            .sum()
    }
}

/// The mesh of one chunk in the layout of cp-mesh (contract §5.2), with its place.
///
/// `quads` holds the layers in storage order (opaque, cutout, fluid) without gaps, each layer
/// its twelve ranges in slot order without gaps, so `layers` alone locates every range. Quads
/// inside a range lie on the range's face. `origin` is the chunk's min corner in level-0
/// voxels of the grid, a multiple of `32 << level` per axis.
#[derive(Clone, Copy, Debug)]
pub struct ChunkUpload<'a> {
    /// Every quad of the chunk.
    pub quads: &'a [PackedQuad],
    /// Range counts per layer, indexed by [`MeshLayer::index`](super::MeshLayer::index).
    pub layers: [LayerCounts; MESH_LAYERS],
    /// Min corner, level-0 voxels.
    pub origin: IVec3,
    /// LOD level, `0..=20`; cells are `2^level` m.
    pub level: u8,
    /// The grid the chunk belongs to.
    pub object: ObjectId,
}

impl ChunkUpload<'_> {
    /// Checks the upload and copies it into a [`PreparedChunk`]; runs on any thread.
    ///
    /// # Errors
    /// Those of [`PreparedChunk::from_vec`].
    pub fn prepare(&self) -> Result<PreparedChunk, VoxelError> {
        let mesh = prepare_mesh(self.quads, &self.layers, self.origin, self.level)?;
        Ok(PreparedChunk {
            quads: self.quads.to_vec(),
            mesh: Box::new(mesh),
            origin: self.origin,
            level: self.level,
            object: self.object,
        })
    }
}

/// A chunk mesh that passed every check that does not depend on the scene (LOD level, origin,
/// range counts, quad words and faces), with its quads owned and its layers located and
/// bounded. It is `Send`: the meshing worker builds it ([`PreparedChunk::from_vec`],
/// [`ChunkUpload::prepare`]) and the frame thread hands it to
/// [`VoxelScene::insert_prepared`] or [`VoxelScene::replace_prepared`], which then only check
/// the object, reserve pool and item space and queue the quads without copying them.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PreparedChunk {
    pub(crate) quads: Vec<PackedQuad>,
    /// Boxed so that a [`RefusedChunk`] stays small.
    pub(crate) mesh: Box<PreparedMesh>,
    pub(crate) origin: IVec3,
    pub(crate) level: u8,
    pub(crate) object: ObjectId,
}

impl PreparedChunk {
    /// Checks a chunk mesh given in the layout of [`ChunkUpload`] (layers in storage order,
    /// twelve ranges each, every quad on its range's face; origin a multiple of `32 << level`
    /// level-0 voxels per axis) and keeps `quads` without copying them.
    ///
    /// # Errors
    /// [`VoxelError::InvalidLevel`], [`VoxelError::MisalignedOrigin`],
    /// [`VoxelError::OriginOutOfRange`], [`VoxelError::RangeTooLarge`],
    /// [`VoxelError::QuadCountMismatch`], [`VoxelError::InvalidQuad`],
    /// [`VoxelError::QuadOnWrongFace`].
    pub fn from_vec(
        quads: Vec<PackedQuad>,
        layers: [LayerCounts; MESH_LAYERS],
        origin: IVec3,
        level: u8,
        object: ObjectId,
    ) -> Result<Self, VoxelError> {
        let mesh = prepare_mesh(&quads, &layers, origin, level)?;
        Ok(Self {
            quads,
            mesh: Box::new(mesh),
            origin,
            level,
            object,
        })
    }

    /// The grid the chunk belongs to.
    #[must_use]
    pub fn object(&self) -> ObjectId {
        self.object
    }

    /// Min corner, level-0 voxels.
    #[must_use]
    pub fn origin(&self) -> IVec3 {
        self.origin
    }

    /// LOD level, `0..=20`.
    #[must_use]
    pub fn level(&self) -> u8 {
        self.level
    }

    /// Every quad of the chunk, in upload order.
    #[must_use]
    pub fn quads(&self) -> &[PackedQuad] {
        &self.quads
    }
}

/// A [`PreparedChunk`] the scene refused, given back unchanged so that it can be tried again.
#[derive(Clone, Debug, Error, PartialEq, Eq)]
#[error("{error}")]
pub struct RefusedChunk {
    /// Why it was refused.
    pub error: VoxelError,
    /// The chunk.
    pub chunk: PreparedChunk,
}

/// Limits of a scene, fixed at creation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SceneLimits {
    /// Quads of the pool, a positive multiple of [`POOL_UNIT_QUADS`].
    pub pool_quads: u64,
    /// Draw items (chunk layers) at once.
    pub max_draw_items: u32,
    /// Objects at once.
    pub max_objects: u32,
}

/// What a scene holds.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SceneStats {
    /// Live objects.
    pub objects: u32,
    /// Live chunks.
    pub chunks: u32,
    /// Chunks whose mesh is in the pool (drawn).
    pub resident_chunks: u32,
    /// Draw items (non-empty layers of resident chunks).
    pub draw_items: u32,
    /// Quads the pool holds.
    pub pool_quads: u64,
    /// Quads not allocated.
    pub pool_free_quads: u64,
    /// Quads of the largest free range.
    pub pool_largest_free_quads: u64,
    /// Meshes waiting for upload.
    pub pending_chunks: u32,
    /// Bytes waiting for upload.
    pub pending_bytes: u64,
}

/// The uploads of one call of [`VoxelScene::flush_uploads`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct UploadReport {
    /// Meshes copied into the pool.
    pub chunks: u32,
    /// Bytes of quads copied.
    pub bytes: u64,
    /// Meshes still waiting.
    pub pending_chunks: u32,
    /// Bytes still waiting.
    pub pending_bytes: u64,
}

/// Why a scene operation was refused. Nothing changes when an operation fails, except that a
/// [`VoxelScene::replace_chunk`] or [`VoxelScene::replace_prepared`] refused for lack of space
/// also drops a replacement queued earlier for the chunk.
#[derive(Clone, Copy, Debug, Error, PartialEq, Eq)]
pub enum VoxelError {
    /// The object id does not name a live object.
    #[error("object {object} does not exist")]
    UnknownObject {
        /// The index of the id.
        object: u32,
    },
    /// The chunk id does not name a live chunk.
    #[error("chunk {index} (generation {generation}) does not exist")]
    UnknownChunk {
        /// Slot of the id.
        index: u32,
        /// Generation of the id.
        generation: u32,
    },
    /// An object still has chunks.
    #[error("object {object} still has {chunks} chunks")]
    ObjectInUse {
        /// The object.
        object: u32,
        /// Its chunks.
        chunks: u32,
    },
    /// Object bounds that are empty or leave the voxel limit.
    #[error("object bounds {min:?}..{max:?} are empty or leave the voxel limit")]
    InvalidBounds {
        /// Min corner.
        min: [i32; 3],
        /// Max corner (exclusive).
        max: [i32; 3],
    },
    /// Every object slot is taken.
    #[error("all {limit} object slots are in use")]
    TooManyObjects {
        /// `max_objects`.
        limit: u32,
    },
    /// The draw item table cannot take the layers of the mesh.
    #[error("the draw item table is full ({limit} items)")]
    TooManyDrawItems {
        /// `max_draw_items`.
        limit: u32,
    },
    /// No free pool range is large enough.
    #[error(
        "the quad pool cannot fit {quads} quads ({free_quads} quads free, the largest free range \
         holds {largest_quads})"
    )]
    PoolFull {
        /// Quads of the mesh.
        quads: u64,
        /// Free quads in total.
        free_quads: u64,
        /// Quads of the largest free range.
        largest_quads: u64,
    },
    /// A LOD level above 20.
    #[error("LOD level {level} is above the maximum of 20")]
    InvalidLevel {
        /// The level.
        level: u8,
    },
    /// An origin that is not a multiple of the chunk extent.
    #[error("chunk origin {origin:?} is not a multiple of the level-{level} chunk extent")]
    MisalignedOrigin {
        /// The origin.
        origin: [i32; 3],
        /// The level.
        level: u8,
    },
    /// A chunk that reaches outside the voxel limit.
    #[error("the level-{level} chunk at {origin:?} leaves the voxel limit")]
    OriginOutOfRange {
        /// The origin.
        origin: [i32; 3],
        /// The level.
        level: u8,
    },
    /// More quads in a range than a chunk can have.
    #[error("layer {layer} range slot {slot} has {count} quads, more than the {limit} of a chunk")]
    RangeTooLarge {
        /// Layer index.
        layer: u8,
        /// Range slot `0..12`.
        slot: u8,
        /// Its count.
        count: u32,
        /// The most a chunk can have there.
        limit: u32,
    },
    /// The counts do not add up to the quads given.
    #[error("the range counts add up to {counted} quads, but {given} quads were given")]
    QuadCountMismatch {
        /// Sum of the counts.
        counted: u64,
        /// Length of `quads`.
        given: u64,
    },
    /// A quad whose words are not a valid packed quad.
    #[error("quad {index} is invalid: {error}")]
    InvalidQuad {
        /// Index in `quads`.
        index: u32,
        /// What is wrong.
        error: QuadError,
    },
    /// A quad in the range of another face.
    #[error("quad {index} lies on face {face:?} but is in a range of face {expected:?}")]
    QuadOnWrongFace {
        /// Index in `quads`.
        index: u32,
        /// Its face.
        face: Face,
        /// The range's face.
        expected: Face,
    },
    /// A replacement that would move the chunk to another object, origin or level.
    #[error(
        "chunk {index} (generation {generation}): a replacement mesh must keep the object, origin \
         and level"
    )]
    ReplaceChangesPlacement {
        /// Slot of the chunk.
        index: u32,
        /// Its generation.
        generation: u32,
    },
    /// A renderer configuration the device or the format cannot hold.
    #[error("invalid voxel renderer configuration: {reason}")]
    InvalidConfig {
        /// What is wrong.
        reason: &'static str,
    },
    /// A ground map outside its documented ranges.
    #[error("invalid ground map: {reason}")]
    InvalidGroundMap {
        /// What is wrong.
        reason: &'static str,
    },
}

/// The scene's buffers on the GPU and its bookkeeping. Owned by the
/// [`VoxelRenderer`](super::VoxelRenderer), which binds the buffers once.
#[derive(Debug)]
pub struct VoxelScene {
    state: SceneState,
    pool: wgpu::Buffer,
    items: wgpu::Buffer,
    objects: wgpu::Buffer,
    objects_cpu: Vec<GpuObject>,
    upload_budget: u64,
    writes: Vec<QuadWrite>,
}

impl VoxelScene {
    /// Creates the buffers for `limits`, which the renderer has checked against the device.
    pub(crate) fn new(
        gpu: &GpuContext,
        limits: SceneLimits,
        upload_budget: u64,
    ) -> Result<Self, RenderError> {
        let (pool, items, objects) = capture(gpu, "voxel scene buffers", |device| {
            let storage = |label: &str, size: u64| {
                device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(label),
                    size,
                    usage: wgpu::BufferUsages::STORAGE
                        | wgpu::BufferUsages::COPY_DST
                        | wgpu::BufferUsages::COPY_SRC,
                    mapped_at_creation: false,
                })
            };
            (
                storage("voxel quad pool", limits.pool_quads * QUAD_BYTES),
                storage(
                    "voxel draw items",
                    u64::from(limits.max_draw_items) * size_of::<GpuDrawItem>() as u64,
                ),
                storage(
                    "voxel objects",
                    u64::from(limits.max_objects) * size_of::<GpuObject>() as u64,
                ),
            )
        })?;
        Ok(Self {
            state: SceneState::new(limits),
            pool,
            items,
            objects,
            objects_cpu: Vec::new(),
            upload_budget,
            writes: Vec::new(),
        })
    }

    /// The limits of the scene.
    pub fn limits(&self) -> SceneLimits {
        *self.state.limits()
    }

    /// Adds a grid.
    ///
    /// # Errors
    /// [`VoxelError::InvalidBounds`], [`VoxelError::TooManyObjects`].
    pub fn add_object(&mut self, desc: ObjectDesc) -> Result<ObjectId, VoxelError> {
        self.state.add_object(desc)
    }

    /// Moves a grid (a construct, or every body when the reference frame moves); takes effect
    /// in the next frame, and the frame after it uses this pose for its previous matrix.
    ///
    /// # Errors
    /// [`VoxelError::UnknownObject`].
    pub fn set_object_pose(&mut self, id: ObjectId, pose: GridPose) -> Result<(), VoxelError> {
        self.state.set_object_pose(id, pose)
    }

    /// The description of a grid.
    ///
    /// # Errors
    /// [`VoxelError::UnknownObject`].
    pub fn object(&self, id: ObjectId) -> Result<ObjectDesc, VoxelError> {
        Ok(self.state.object(id)?.desc)
    }

    /// Removes a grid that has no chunks left.
    ///
    /// # Errors
    /// [`VoxelError::UnknownObject`], [`VoxelError::ObjectInUse`].
    pub fn remove_object(&mut self, id: ObjectId) -> Result<(), VoxelError> {
        self.state.remove_object(id)
    }

    /// Checks a chunk mesh, reserves its pool range and item slots (one per non-empty layer)
    /// and queues a copy of its quads. The chunk starts without walls and without fade; it is
    /// drawn once its mesh is uploaded (at once for an empty mesh). The checks run on the
    /// calling thread; [`Self::insert_prepared`] takes meshes checked elsewhere.
    ///
    /// # Errors
    /// The checks of [`VoxelError`]: unknown object, level, origin, counts, quads, and the
    /// limits of the pool and the item table.
    pub fn insert_chunk(&mut self, upload: &ChunkUpload<'_>) -> Result<ChunkId, VoxelError> {
        self.state.insert_chunk(upload)
    }

    /// [`Self::insert_chunk`] for a mesh that was checked when it was prepared: only the object
    /// is checked and space reserved, and the quads move into the queue.
    ///
    /// # Errors
    /// [`RefusedChunk`] with [`VoxelError::UnknownObject`], [`VoxelError::TooManyDrawItems`]
    /// or [`VoxelError::PoolFull`]; the scene is unchanged and the chunk comes back.
    pub fn insert_prepared(&mut self, chunk: PreparedChunk) -> Result<ChunkId, RefusedChunk> {
        self.state.insert_prepared(chunk)
    }

    /// Queues a new mesh for a chunk; the old one stays drawn until the new one is uploaded.
    /// The upload must keep the chunk's object, origin and level. Until then both meshes hold
    /// pool space, while the item table only needs room for the layers the new mesh adds.
    ///
    /// # Errors
    /// Those of [`Self::insert_chunk`], [`VoxelError::UnknownChunk`] and
    /// [`VoxelError::ReplaceChangesPlacement`].
    pub fn replace_chunk(
        &mut self,
        id: ChunkId,
        upload: &ChunkUpload<'_>,
    ) -> Result<(), VoxelError> {
        self.state.replace_chunk(id, upload)
    }

    /// [`Self::replace_chunk`] for a mesh that was checked when it was prepared.
    ///
    /// # Errors
    /// [`RefusedChunk`] with [`VoxelError::UnknownChunk`],
    /// [`VoxelError::ReplaceChangesPlacement`], [`VoxelError::TooManyDrawItems`] or
    /// [`VoxelError::PoolFull`]; the chunk comes back. Like a failed [`Self::replace_chunk`],
    /// a refusal for lack of space drops a replacement queued earlier for the chunk.
    pub fn replace_prepared(
        &mut self,
        id: ChunkId,
        chunk: PreparedChunk,
    ) -> Result<(), RefusedChunk> {
        self.state.replace_prepared(id, chunk)
    }

    /// Removes a chunk at once, with any upload still queued for it.
    ///
    /// # Errors
    /// [`VoxelError::UnknownChunk`].
    pub fn remove_chunk(&mut self, id: ChunkId) -> Result<(), VoxelError> {
        self.state.remove_chunk(id)
    }

    /// Sets the faces whose wall ranges are drawn (LOD borders, edited neighbours).
    ///
    /// # Errors
    /// [`VoxelError::UnknownChunk`].
    pub fn set_wall_mask(&mut self, id: ChunkId, walls: WallMask) -> Result<(), VoxelError> {
        self.state.set_wall_mask(id, walls)
    }

    /// Sets the cross-fade of a chunk; [`Fade::FULL`] ends it.
    ///
    /// # Errors
    /// [`VoxelError::UnknownChunk`].
    pub fn set_fade(&mut self, id: ChunkId, fade: Fade) -> Result<(), VoxelError> {
        self.state.set_fade(id, fade)
    }

    /// Whether the chunk's mesh is in the pool, so that it is drawn.
    ///
    /// # Errors
    /// [`VoxelError::UnknownChunk`].
    pub fn is_resident(&self, id: ChunkId) -> Result<bool, VoxelError> {
        self.state.is_resident(id)
    }

    /// Whether a mesh for the chunk waits in the upload queue (a new chunk that is not drawn
    /// yet, or a replacement while the old mesh is still drawn).
    ///
    /// # Errors
    /// [`VoxelError::UnknownChunk`].
    pub fn has_pending_upload(&self, id: ChunkId) -> Result<bool, VoxelError> {
        self.state.has_pending_upload(id)
    }

    /// What the scene holds.
    pub fn stats(&self) -> SceneStats {
        self.state.stats()
    }

    /// Bytes of quads copied into the pool per frame at most (at least one mesh per frame
    /// always goes).
    pub fn upload_budget(&self) -> u64 {
        self.upload_budget
    }

    /// Sets the per-frame upload budget in bytes.
    pub fn set_upload_budget(&mut self, bytes: u64) {
        self.upload_budget = bytes;
    }

    /// Copies queued meshes into the pool within the upload budget, oldest first, and writes
    /// the changed draw items. The renderer calls it at the start of every frame.
    pub fn flush_uploads(&mut self, gpu: &GpuContext) -> UploadReport {
        self.flush(gpu, self.upload_budget)
    }

    /// Copies every queued mesh into the pool, whatever the budget (loading screens, tests).
    pub fn flush_all_uploads(&mut self, gpu: &GpuContext) -> UploadReport {
        self.flush(gpu, u64::MAX)
    }

    fn flush(&mut self, gpu: &GpuContext, budget: u64) -> UploadReport {
        self.writes.clear();
        let report = self.state.take_uploads(budget, &mut self.writes);
        for write in self.writes.drain(..) {
            gpu.queue.write_buffer(
                &self.pool,
                u64::from(write.first_quad) * QUAD_BYTES,
                bytemuck::cast_slice(&write.quads),
            );
        }
        self.write_items(gpu);
        report
    }

    /// Writes the draw items changed since the last call.
    pub(crate) fn write_items(&mut self, gpu: &GpuContext) {
        let runs: Vec<Range<usize>> = self.state.take_dirty_items();
        let items = self.state.items();
        for run in runs {
            gpu.queue.write_buffer(
                &self.items,
                (run.start * size_of::<GpuDrawItem>()) as u64,
                bytemuck::cast_slice(&items[run]),
            );
        }
    }

    /// Composes and writes the object table for a frame seen by `now`, with `prev` and the
    /// previous poses for the previous matrices.
    pub(crate) fn update_objects(
        &mut self,
        gpu: &GpuContext,
        now: &FrameCamera,
        prev: &FrameCamera,
    ) {
        self.objects_cpu.clear();
        for slot in self.state.object_slots() {
            self.objects_cpu.push(match slot {
                Some(entry) => compose_object(
                    &entry.desc.bounds,
                    &entry.desc.pose,
                    &entry.prev_pose,
                    now,
                    prev,
                ),
                None => GpuObject::default(),
            });
        }
        if !self.objects_cpu.is_empty() {
            gpu.queue
                .write_buffer(&self.objects, 0, bytemuck::cast_slice(&self.objects_cpu));
        }
    }

    /// Makes the current object poses the previous ones.
    pub(crate) fn commit_poses(&mut self) {
        self.state.commit_poses();
    }

    pub(crate) fn items_cpu(&self) -> &[GpuDrawItem] {
        self.state.items()
    }

    /// Per draw class, the most draws the items can produce in any frame.
    pub(crate) fn draw_bounds(&self) -> [u64; 2] {
        self.state.draw_bounds()
    }

    pub(crate) fn objects_cpu(&self) -> &[GpuObject] {
        &self.objects_cpu
    }

    pub(crate) fn pool_buffer(&self) -> &wgpu::Buffer {
        &self.pool
    }

    pub(crate) fn items_buffer(&self) -> &wgpu::Buffer {
        &self.items
    }

    pub(crate) fn objects_buffer(&self) -> &wgpu::Buffer {
        &self.objects
    }
}

/// Quads the pool of `pool_bytes` holds, rounded down to whole units.
pub(crate) fn pool_quads_for_bytes(pool_bytes: u64) -> u64 {
    let unit = u64::from(POOL_UNIT_QUADS);
    pool_bytes / QUAD_BYTES / unit * unit
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layer_counts_round_trip_the_slots_of_cp_mesh() {
        let slots = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
        let counts = LayerCounts::from_slots(slots);
        assert_eq!(counts.regular, [1, 2, 3, 4, 5, 6]);
        assert_eq!(counts.walls, [7, 8, 9, 10, 11, 12]);
        assert_eq!(counts.slots(), slots);
        assert_eq!(counts.total(), 78);
        assert_eq!(LayerCounts::EMPTY.total(), 0);
        assert_eq!(pool_quads_for_bytes(128 << 20), 16 << 20);
        assert_eq!(pool_quads_for_bytes(2047), 0);
        assert_eq!(pool_quads_for_bytes(2048 * 3 + 5), 768);
    }

    #[test]
    fn errors_stay_small_and_readable() {
        assert!(size_of::<VoxelError>() <= 64, "{}", size_of::<VoxelError>());
        let error = VoxelError::PoolFull {
            quads: 10,
            free_quads: 5,
            largest_quads: 2,
        };
        assert_eq!(
            error.to_string(),
            "the quad pool cannot fit 10 quads (5 quads free, the largest free range holds 2)"
        );
    }
}
