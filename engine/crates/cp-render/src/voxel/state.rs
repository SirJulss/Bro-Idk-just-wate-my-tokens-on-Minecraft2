//! The CPU side of the voxel scene: objects, chunks, their draw items, the pool allocator and
//! the upload queue. Pure bookkeeping without a device, so that every rule is tested without a
//! GPU; [`VoxelScene`](super::VoxelScene) applies the results to the buffers.

use std::collections::VecDeque;
use std::ops::Range;

use cp_core::coords::{CHUNK_SIZE, MAX_LOD_LEVEL, VOXEL_LIMIT};
use cp_core::math::IVec3;
use cp_core::quad::PackedQuad;

use super::camera::GridPose;
use super::layout::{
    CellBounds, DRAW_CLASSES, FACE_STORAGE_ORDER, Fade, GpuDrawItem, MAX_ITEM_DRAWS,
    MAX_REGULAR_RANGE_QUADS, MAX_WALL_RANGE_QUADS, MESH_LAYERS, MeshLayer, POOL_UNIT_QUADS,
    QUAD_BYTES, RANGES_PER_LAYER, WallMask,
};
use super::pool::{PoolRange, RangeAllocator, units_for};
use super::scene::{
    ChunkId, ChunkUpload, LayerCounts, ObjectDesc, ObjectId, PreparedChunk, RefusedChunk,
    SceneLimits, SceneStats, UploadReport, VoxelError,
};

/// One non-empty layer of an uploaded mesh.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct LayerMesh {
    /// First quad of the layer relative to the chunk's first quad.
    offset: u32,
    counts: LayerCounts,
    /// Cells of the regular quads.
    bounds: Option<CellBounds>,
    /// Cells of the wall quads.
    wall_bounds: Option<CellBounds>,
}

/// The mesh of a chunk that the GPU holds and draws.
#[derive(Clone, Debug)]
struct ResidentMesh {
    range: Option<PoolRange>,
    layers: [Option<LayerMesh>; MESH_LAYERS],
    items: [Option<u32>; MESH_LAYERS],
}

impl ResidentMesh {
    /// Draw items the mesh holds: its non-empty layers.
    fn item_count(&self) -> u32 {
        self.items.iter().flatten().count() as u32
    }
}

/// A queued upload's entry in its chunk.
#[derive(Clone, Copy, Debug)]
struct PendingRef {
    serial: u64,
    range: Option<PoolRange>,
    /// Item slots reserved for the upload: what it adds to the table when it completes (its
    /// layers minus the items of the resident mesh it replaces, at least 0).
    items: u32,
    bytes: u64,
}

#[derive(Clone, Debug)]
struct Chunk {
    object: u32,
    origin: IVec3,
    level: u8,
    walls: WallMask,
    fade: Fade,
    resident: Option<ResidentMesh>,
    pending: Option<PendingRef>,
}

#[derive(Clone, Debug, Default)]
struct ChunkSlot {
    generation: u32,
    chunk: Option<Chunk>,
}

#[derive(Debug)]
struct PendingUpload {
    chunk: ChunkId,
    serial: u64,
    range: Option<PoolRange>,
    quads: Vec<PackedQuad>,
    layers: [Option<LayerMesh>; MESH_LAYERS],
}

/// A grid of the scene.
#[derive(Clone, Copy, Debug)]
pub(crate) struct ObjectEntry {
    pub(crate) desc: ObjectDesc,
    /// The pose of the last rendered frame.
    pub(crate) prev_pose: GridPose,
    chunks: u32,
}

/// Quads to copy into the pool: `quads` from pool index `first_quad`.
#[derive(Debug)]
pub(crate) struct QuadWrite {
    pub(crate) first_quad: u32,
    pub(crate) quads: Vec<PackedQuad>,
}

/// The layout of a checked mesh: place, counts and cell bounds of each non-empty layer.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct PreparedMesh {
    layers: [Option<LayerMesh>; MESH_LAYERS],
    /// Non-empty layers: the draw items the mesh needs.
    layer_count: u32,
}

/// Checks a chunk mesh against every rule that does not depend on a scene (level, origin,
/// range counts, quad words and faces) and lays out its layers. Runs on any thread.
pub(crate) fn prepare_mesh(
    quads: &[PackedQuad],
    layers: &[LayerCounts; MESH_LAYERS],
    origin: IVec3,
    level: u8,
) -> Result<PreparedMesh, VoxelError> {
    if level > MAX_LOD_LEVEL {
        return Err(VoxelError::InvalidLevel { level });
    }
    let extent = CHUNK_SIZE << level;
    let origin = origin.to_array();
    if origin.iter().any(|&v| v.rem_euclid(extent) != 0) {
        return Err(VoxelError::MisalignedOrigin { origin, level });
    }
    let limit = i64::from(VOXEL_LIMIT);
    if origin
        .iter()
        .any(|&v| i64::from(v) < -limit || i64::from(v) + i64::from(extent) > limit)
    {
        return Err(VoxelError::OriginOutOfRange { origin, level });
    }
    let mut counted = 0_u64;
    let mut layer_count = 0;
    for (layer, counts) in layers.iter().enumerate() {
        for (slot, &count) in counts.slots().iter().enumerate() {
            let limit = if slot < 6 {
                MAX_REGULAR_RANGE_QUADS
            } else {
                MAX_WALL_RANGE_QUADS
            };
            if count > limit {
                return Err(VoxelError::RangeTooLarge {
                    // Below 3 and 12.
                    layer: layer as u8,
                    slot: slot as u8,
                    count,
                    limit,
                });
            }
            counted += u64::from(count);
        }
        if counts.total() > 0 {
            layer_count += 1;
        }
    }
    if counted != quads.len() as u64 {
        return Err(VoxelError::QuadCountMismatch {
            counted,
            given: quads.len() as u64,
        });
    }
    let mut laid_out = [None; MESH_LAYERS];
    let mut index = 0_usize;
    for (layer, counts) in layers.iter().enumerate() {
        let offset = index as u32;
        let mut bounds = [None::<CellBounds>; 2];
        for (slot, &count) in counts.slots().iter().enumerate() {
            let expected = FACE_STORAGE_ORDER[slot % 6];
            for &quad in &quads[index..index + count as usize] {
                // The counts are checked, so every index fits u32.
                let at = index as u32;
                quad.validate()
                    .map_err(|error| VoxelError::InvalidQuad { index: at, error })?;
                if quad.face() != expected {
                    return Err(VoxelError::QuadOnWrongFace {
                        index: at,
                        face: quad.face(),
                        expected,
                    });
                }
                let cells = CellBounds::of_quad(quad);
                let group = &mut bounds[usize::from(slot >= 6)];
                *group = Some(group.map_or(cells, |b| b.union(cells)));
                index += 1;
            }
        }
        if counts.total() > 0 {
            laid_out[layer] = Some(LayerMesh {
                offset,
                counts: *counts,
                bounds: bounds[0],
                wall_bounds: bounds[1],
            });
        }
    }
    Ok(PreparedMesh {
        layers: laid_out,
        layer_count,
    })
}

impl PreparedMesh {
    /// Draw items the mesh needs: its non-empty layers.
    pub(crate) fn layer_count(&self) -> u32 {
        self.layer_count
    }
}

/// The bookkeeping of a voxel scene.
#[derive(Debug)]
pub(crate) struct SceneState {
    limits: SceneLimits,
    allocator: RangeAllocator,
    objects: Vec<Option<ObjectEntry>>,
    live_objects: u32,
    chunks: Vec<ChunkSlot>,
    free_chunks: Vec<u32>,
    live_chunks: u32,
    resident_chunks: u32,
    items: Vec<GpuDrawItem>,
    /// Per item: chunk slot and layer.
    owners: Vec<(u32, u8)>,
    /// Items changed since the last [`Self::take_dirty_items`], one bit per item index.
    dirty: Vec<u64>,
    /// Per draw class, the sum of [`item_draw_bound`] over its items: the most draws the class
    /// can have in any frame.
    draw_bounds: [u64; DRAW_CLASSES as usize],
    pending: VecDeque<PendingUpload>,
    pending_chunks: u32,
    pending_bytes: u64,
    /// Item slots reserved by queued uploads. `items.len() + pending_items` never exceeds
    /// `max_draw_items`, so every queued upload finds its slots when it completes.
    pending_items: u32,
    next_serial: u64,
}

/// The most draws `item` can produce from any camera position: runs of visible slots are
/// separated by non-empty invisible ones, so `n` non-empty slots give at most `⌈n / 2⌉` runs.
pub(crate) fn item_draw_bound(item: &GpuDrawItem) -> u32 {
    let nonempty = (0..RANGES_PER_LAYER)
        .filter(|&slot| item.slot_count(slot) > 0)
        .count() as u32;
    nonempty.div_ceil(2).min(MAX_ITEM_DRAWS)
}

impl SceneState {
    /// An empty scene; `limits` must have been checked (pool size a multiple of the unit).
    pub(crate) fn new(limits: SceneLimits) -> Self {
        let units =
            u32::try_from(limits.pool_quads / u64::from(POOL_UNIT_QUADS)).unwrap_or(u32::MAX);
        Self {
            limits,
            allocator: RangeAllocator::new(units),
            objects: Vec::new(),
            live_objects: 0,
            chunks: Vec::new(),
            free_chunks: Vec::new(),
            live_chunks: 0,
            resident_chunks: 0,
            items: Vec::new(),
            owners: Vec::new(),
            dirty: vec![0; (limits.max_draw_items as usize).div_ceil(64)],
            draw_bounds: [0; DRAW_CLASSES as usize],
            pending: VecDeque::new(),
            pending_chunks: 0,
            pending_bytes: 0,
            pending_items: 0,
            next_serial: 0,
        }
    }

    pub(crate) fn limits(&self) -> &SceneLimits {
        &self.limits
    }

    /// Per draw class, the most draws the current items can produce: the `max_count` of the
    /// count draws of path A, which costs GPU time per slot on some drivers.
    pub(crate) fn draw_bounds(&self) -> [u64; DRAW_CLASSES as usize] {
        self.draw_bounds
    }

    fn count_item(&mut self, item: &GpuDrawItem, add: bool) {
        let bound = &mut self.draw_bounds[item.draw_class() as usize];
        if add {
            *bound += u64::from(item_draw_bound(item));
        } else {
            *bound -= u64::from(item_draw_bound(item));
        }
    }

    // --- objects -------------------------------------------------------------------------

    pub(crate) fn add_object(&mut self, desc: ObjectDesc) -> Result<ObjectId, VoxelError> {
        let limit = i64::from(VOXEL_LIMIT);
        let (min, max) = (desc.bounds.min, desc.bounds.max);
        let inside = |v: i32| (-limit..=limit).contains(&i64::from(v));
        if desc.bounds.is_empty() || !min.to_array().into_iter().chain(max.to_array()).all(inside) {
            return Err(VoxelError::InvalidBounds {
                min: min.to_array(),
                max: max.to_array(),
            });
        }
        let slot = match self.objects.iter().position(Option::is_none) {
            Some(slot) => slot,
            None if self.objects.len() < self.limits.max_objects as usize => {
                self.objects.push(None);
                self.objects.len() - 1
            }
            None => {
                return Err(VoxelError::TooManyObjects {
                    limit: self.limits.max_objects,
                });
            }
        };
        self.objects[slot] = Some(ObjectEntry {
            desc,
            prev_pose: desc.pose,
            chunks: 0,
        });
        self.live_objects += 1;
        // Below max_objects, which is a u32.
        Ok(ObjectId(slot as u32))
    }

    fn object_mut(&mut self, id: ObjectId) -> Result<&mut ObjectEntry, VoxelError> {
        self.objects
            .get_mut(id.0 as usize)
            .and_then(Option::as_mut)
            .ok_or(VoxelError::UnknownObject { object: id.0 })
    }

    pub(crate) fn object(&self, id: ObjectId) -> Result<&ObjectEntry, VoxelError> {
        self.objects
            .get(id.0 as usize)
            .and_then(Option::as_ref)
            .ok_or(VoxelError::UnknownObject { object: id.0 })
    }

    pub(crate) fn set_object_pose(
        &mut self,
        id: ObjectId,
        pose: GridPose,
    ) -> Result<(), VoxelError> {
        self.object_mut(id)?.desc.pose = pose;
        Ok(())
    }

    pub(crate) fn remove_object(&mut self, id: ObjectId) -> Result<(), VoxelError> {
        let chunks = self.object(id)?.chunks;
        if chunks > 0 {
            return Err(VoxelError::ObjectInUse {
                object: id.0,
                chunks,
            });
        }
        self.objects[id.0 as usize] = None;
        self.live_objects -= 1;
        Ok(())
    }

    /// Object slots up to the last live one, `None` for free slots.
    pub(crate) fn object_slots(&self) -> &[Option<ObjectEntry>] {
        let len = self
            .objects
            .iter()
            .rposition(Option::is_some)
            .map_or(0, |last| last + 1);
        &self.objects[..len]
    }

    /// Makes the current poses the previous ones (after a frame was rendered).
    pub(crate) fn commit_poses(&mut self) {
        for entry in self.objects.iter_mut().flatten() {
            entry.prev_pose = entry.desc.pose;
        }
    }

    // --- chunks --------------------------------------------------------------------------

    fn chunk(&self, id: ChunkId) -> Result<&Chunk, VoxelError> {
        self.chunks
            .get(id.index as usize)
            .filter(|slot| slot.generation == id.generation)
            .and_then(|slot| slot.chunk.as_ref())
            .ok_or(VoxelError::UnknownChunk {
                index: id.index,
                generation: id.generation,
            })
    }

    fn chunk_mut(&mut self, id: ChunkId) -> Result<&mut Chunk, VoxelError> {
        self.chunks
            .get_mut(id.index as usize)
            .filter(|slot| slot.generation == id.generation)
            .and_then(|slot| slot.chunk.as_mut())
            .ok_or(VoxelError::UnknownChunk {
                index: id.index,
                generation: id.generation,
            })
    }

    /// Reserves `items` draw item slots and pool space for `quads` quads; nothing changes on an
    /// error.
    fn reserve(&mut self, quads: u64, items: u32) -> Result<Option<PoolRange>, VoxelError> {
        let used = self.items.len() as u64 + u64::from(self.pending_items);
        if used + u64::from(items) > u64::from(self.limits.max_draw_items) {
            return Err(VoxelError::TooManyDrawItems {
                limit: self.limits.max_draw_items,
            });
        }
        if quads == 0 {
            return Ok(None);
        }
        let range = u32::try_from(units_for(quads))
            .ok()
            .and_then(|units| self.allocator.allocate(units));
        match range {
            Some(range) => Ok(Some(range)),
            None => Err(VoxelError::PoolFull {
                quads,
                free_quads: u64::from(self.allocator.free_units()) * u64::from(POOL_UNIT_QUADS),
                largest_quads: u64::from(self.allocator.largest_free())
                    * u64::from(POOL_UNIT_QUADS),
            }),
        }
    }

    pub(crate) fn insert_chunk(&mut self, upload: &ChunkUpload<'_>) -> Result<ChunkId, VoxelError> {
        self.object(upload.object)?;
        let chunk = upload.prepare()?;
        self.insert_prepared(chunk).map_err(|refused| refused.error)
    }

    pub(crate) fn insert_prepared(
        &mut self,
        chunk: PreparedChunk,
    ) -> Result<ChunkId, RefusedChunk> {
        if let Err(error) = self.object(chunk.object) {
            return Err(RefusedChunk { error, chunk });
        }
        let items = chunk.mesh.layer_count();
        let range = match self.reserve(chunk.quads.len() as u64, items) {
            Ok(range) => range,
            Err(error) => return Err(RefusedChunk { error, chunk }),
        };
        let index = match self.free_chunks.pop() {
            Some(index) => index,
            None => {
                self.chunks.push(ChunkSlot::default());
                // Chunks are bounded by the pool and the item table, far below 2^32.
                (self.chunks.len() - 1) as u32
            }
        };
        let id = ChunkId {
            index,
            generation: self.chunks[index as usize].generation,
        };
        self.chunks[index as usize].chunk = Some(Chunk {
            object: chunk.object.0,
            origin: chunk.origin,
            level: chunk.level,
            walls: WallMask::NONE,
            fade: Fade::FULL,
            resident: None,
            pending: None,
        });
        self.live_chunks += 1;
        if let Some(Some(object)) = self.objects.get_mut(chunk.object.0 as usize) {
            object.chunks += 1;
        }
        self.queue(id, chunk, range, items);
        Ok(id)
    }

    /// The draw items of the chunk's resident mesh, once the chunk exists and a replacement
    /// keeps its object, origin and level.
    fn replaced_items(
        &self,
        id: ChunkId,
        object: ObjectId,
        origin: IVec3,
        level: u8,
    ) -> Result<u32, VoxelError> {
        let chunk = self.chunk(id)?;
        if object.0 != chunk.object || origin != chunk.origin || level != chunk.level {
            return Err(VoxelError::ReplaceChangesPlacement {
                index: id.index,
                generation: id.generation,
            });
        }
        Ok(chunk.resident.as_ref().map_or(0, ResidentMesh::item_count))
    }

    pub(crate) fn replace_chunk(
        &mut self,
        id: ChunkId,
        upload: &ChunkUpload<'_>,
    ) -> Result<(), VoxelError> {
        self.replaced_items(id, upload.object, upload.origin, upload.level)?;
        let chunk = upload.prepare()?;
        self.replace_prepared(id, chunk)
            .map_err(|refused| refused.error)
    }

    pub(crate) fn replace_prepared(
        &mut self,
        id: ChunkId,
        chunk: PreparedChunk,
    ) -> Result<(), RefusedChunk> {
        let resident = match self.replaced_items(id, chunk.object, chunk.origin, chunk.level) {
            Ok(items) => items,
            Err(error) => return Err(RefusedChunk { error, chunk }),
        };
        // The superseded upload gives its reservation back before the new one is made.
        self.cancel_pending(id);
        // The upload replaces the resident items when it completes, so only what it adds
        // needs a slot; the old and the new pool ranges coexist until then.
        let items = chunk.mesh.layer_count().saturating_sub(resident);
        let range = match self.reserve(chunk.quads.len() as u64, items) {
            Ok(range) => range,
            Err(error) => return Err(RefusedChunk { error, chunk }),
        };
        self.queue(id, chunk, range, items);
        Ok(())
    }

    pub(crate) fn remove_chunk(&mut self, id: ChunkId) -> Result<(), VoxelError> {
        self.chunk(id)?;
        self.cancel_pending(id);
        let slot = &mut self.chunks[id.index as usize];
        let chunk = slot.chunk.take().expect("checked above");
        slot.generation = slot.generation.wrapping_add(1);
        self.free_chunks.push(id.index);
        self.live_chunks -= 1;
        if let Some(mesh) = chunk.resident {
            self.drop_mesh(&mesh);
            self.resident_chunks -= 1;
        }
        if let Some(Some(object)) = self.objects.get_mut(chunk.object as usize) {
            object.chunks -= 1;
        }
        Ok(())
    }

    pub(crate) fn set_wall_mask(&mut self, id: ChunkId, walls: WallMask) -> Result<(), VoxelError> {
        self.chunk_mut(id)?.walls = walls;
        self.rebuild_items(id);
        Ok(())
    }

    pub(crate) fn set_fade(&mut self, id: ChunkId, fade: Fade) -> Result<(), VoxelError> {
        self.chunk_mut(id)?.fade = fade;
        self.rebuild_items(id);
        Ok(())
    }

    pub(crate) fn is_resident(&self, id: ChunkId) -> Result<bool, VoxelError> {
        Ok(self.chunk(id)?.resident.is_some())
    }

    pub(crate) fn has_pending_upload(&self, id: ChunkId) -> Result<bool, VoxelError> {
        Ok(self.chunk(id)?.pending.is_some())
    }

    /// Queues a mesh whose pool range and `items` item slots are reserved; an empty mesh
    /// becomes resident at once. The quads move into the queue without a copy.
    fn queue(&mut self, id: ChunkId, chunk: PreparedChunk, range: Option<PoolRange>, items: u32) {
        let serial = self.next_serial;
        self.next_serial += 1;
        let upload = PendingUpload {
            chunk: id,
            serial,
            range,
            quads: chunk.quads,
            layers: chunk.mesh.layers,
        };
        if upload.quads.is_empty() {
            self.complete(upload);
            return;
        }
        let bytes = upload.quads.len() as u64 * QUAD_BYTES;
        if let Ok(chunk) = self.chunk_mut(id) {
            chunk.pending = Some(PendingRef {
                serial,
                range,
                items,
                bytes,
            });
        }
        self.pending_chunks += 1;
        self.pending_bytes += bytes;
        self.pending_items += items;
        self.pending.push_back(upload);
    }

    /// Drops the queued upload of `id`, if any: its pool range is freed at once, its entry in
    /// the queue is skipped when it comes up.
    fn cancel_pending(&mut self, id: ChunkId) {
        let Ok(chunk) = self.chunk_mut(id) else {
            return;
        };
        let Some(pending) = chunk.pending.take() else {
            return;
        };
        if let Some(range) = pending.range {
            self.allocator.free(range);
        }
        self.pending_chunks -= 1;
        self.pending_bytes -= pending.bytes;
        self.pending_items -= pending.items;
    }

    /// Completes queued uploads in order while `budget_bytes` allows, at least one per call,
    /// and pushes their pool writes to `writes`.
    pub(crate) fn take_uploads(
        &mut self,
        budget_bytes: u64,
        writes: &mut Vec<QuadWrite>,
    ) -> UploadReport {
        let mut report = UploadReport::default();
        while let Some(front) = self.pending.front() {
            let current = self
                .chunk(front.chunk)
                .ok()
                .and_then(|chunk| chunk.pending)
                .is_some_and(|pending| pending.serial == front.serial);
            if !current {
                self.pending.pop_front();
                continue;
            }
            let bytes = front.quads.len() as u64 * QUAD_BYTES;
            if report.chunks > 0 && report.bytes + bytes > budget_bytes {
                break;
            }
            let Some(upload) = self.pending.pop_front() else {
                break;
            };
            let chunk = self.chunk_mut(upload.chunk).expect("checked above");
            let pending = chunk.pending.take().expect("checked above");
            self.pending_chunks -= 1;
            self.pending_bytes -= pending.bytes;
            self.pending_items -= pending.items;
            report.chunks += 1;
            report.bytes += bytes;
            if let Some(write) = self.complete(upload) {
                writes.push(write);
            }
        }
        report.pending_chunks = self.pending_chunks;
        report.pending_bytes = self.pending_bytes;
        report
    }

    /// Makes an upload the chunk's resident mesh, replacing the old one.
    fn complete(&mut self, upload: PendingUpload) -> Option<QuadWrite> {
        let Ok(chunk) = self.chunk_mut(upload.chunk) else {
            if let Some(range) = upload.range {
                self.allocator.free(range);
            }
            return None;
        };
        let old = chunk.resident.take();
        if let Some(old) = &old {
            self.drop_mesh(old);
        } else {
            self.resident_chunks += 1;
        }
        let chunk = self.chunk(upload.chunk).expect("checked above").clone();
        let first_quad = upload.range.map_or(0, PoolRange::first_quad);
        let mut items = [None; MESH_LAYERS];
        for (layer, mesh) in upload.layers.iter().enumerate() {
            if let Some(mesh) = mesh {
                let item = build_item(&chunk, MeshLayer::ALL[layer], mesh, first_quad);
                // Below max_draw_items, a u32.
                items[layer] = Some(self.items.len() as u32);
                self.mark_dirty(self.items.len());
                self.count_item(&item, true);
                self.items.push(item);
                self.owners.push((upload.chunk.index, layer as u8));
            }
        }
        let resident = ResidentMesh {
            range: upload.range,
            layers: upload.layers,
            items,
        };
        self.chunk_mut(upload.chunk)
            .expect("checked above")
            .resident = Some(resident);
        (!upload.quads.is_empty()).then_some(QuadWrite {
            first_quad,
            quads: upload.quads,
        })
    }

    /// Removes the items of `mesh` and frees its pool range.
    fn drop_mesh(&mut self, mesh: &ResidentMesh) {
        let mut indices: Vec<u32> = mesh.items.iter().flatten().copied().collect();
        // Highest first: the item moved into a hole is then never one still to be removed.
        indices.sort_unstable_by(|a, b| b.cmp(a));
        for index in indices {
            self.remove_item(index as usize);
        }
        if let Some(range) = mesh.range {
            self.allocator.free(range);
        }
    }

    /// Swap-removes item `index`; the moved item's owner learns its new index.
    fn remove_item(&mut self, index: usize) {
        let removed = self.items[index];
        self.count_item(&removed, false);
        let last = self.items.len() - 1;
        if index != last {
            self.items[index] = self.items[last];
            self.owners[index] = self.owners[last];
            let (chunk, layer) = self.owners[index];
            if let Some(mesh) = self.chunks[chunk as usize]
                .chunk
                .as_mut()
                .and_then(|chunk| chunk.resident.as_mut())
            {
                // Below max_draw_items.
                mesh.items[usize::from(layer)] = Some(index as u32);
            }
            self.mark_dirty(index);
        }
        self.items.pop();
        self.owners.pop();
    }

    /// Rebuilds the items of a resident chunk after its walls or fade changed.
    fn rebuild_items(&mut self, id: ChunkId) {
        let Ok(chunk) = self.chunk(id) else {
            return;
        };
        let Some(mesh) = &chunk.resident else {
            return;
        };
        let first_quad = mesh.range.map_or(0, PoolRange::first_quad);
        let mut rebuilt = Vec::new();
        for layer in 0..MESH_LAYERS {
            if let (Some(index), Some(layer_mesh)) = (mesh.items[layer], mesh.layers[layer]) {
                let item = build_item(chunk, MeshLayer::ALL[layer], &layer_mesh, first_quad);
                rebuilt.push((index as usize, item));
            }
        }
        for (index, item) in rebuilt {
            // A fade moves an item between the draw classes.
            let old = self.items[index];
            self.count_item(&old, false);
            self.count_item(&item, true);
            self.items[index] = item;
            self.mark_dirty(index);
        }
    }

    fn mark_dirty(&mut self, index: usize) {
        self.dirty[index / 64] |= 1 << (index % 64);
    }

    /// The ranges of item indices changed since the last call, coalesced, below the current
    /// item count; clears the record.
    pub(crate) fn take_dirty_items(&mut self) -> Vec<Range<usize>> {
        let mut runs: Vec<Range<usize>> = Vec::new();
        let len = self.items.len();
        for (word_index, word) in self.dirty.iter_mut().enumerate() {
            let mut bits = std::mem::take(word);
            while bits != 0 {
                let bit = bits.trailing_zeros() as usize;
                bits &= bits - 1;
                let index = word_index * 64 + bit;
                if index >= len {
                    continue;
                }
                match runs.last_mut() {
                    Some(run) if run.end == index => run.end += 1,
                    _ => runs.push(index..index + 1),
                }
            }
        }
        runs
    }

    pub(crate) fn items(&self) -> &[GpuDrawItem] {
        &self.items
    }

    pub(crate) fn stats(&self) -> SceneStats {
        let unit = u64::from(POOL_UNIT_QUADS);
        SceneStats {
            objects: self.live_objects,
            chunks: self.live_chunks,
            resident_chunks: self.resident_chunks,
            draw_items: self.items.len() as u32,
            pool_quads: u64::from(self.allocator.capacity()) * unit,
            pool_free_quads: u64::from(self.allocator.free_units()) * unit,
            pool_largest_free_quads: u64::from(self.allocator.largest_free()) * unit,
            pending_chunks: self.pending_chunks,
            pending_bytes: self.pending_bytes,
        }
    }
}

/// The draw item of one layer of a resident chunk.
fn build_item(chunk: &Chunk, layer: MeshLayer, mesh: &LayerMesh, first_quad: u32) -> GpuDrawItem {
    let cells = match (mesh.bounds, mesh.wall_bounds) {
        (Some(regular), Some(walls)) if !chunk.walls.is_empty() => regular.union(walls),
        (Some(regular), _) => regular,
        (None, Some(walls)) => walls,
        (None, None) => CellBounds::CHUNK,
    };
    GpuDrawItem {
        origin: chunk.origin.to_array(),
        lod_flags: GpuDrawItem::pack_lod_flags(chunk.level, layer, chunk.walls),
        cell_bounds: cells.pack(),
        quad_base: first_quad + mesh.offset,
        counts: [0, 1, 2, 3, 4, 5]
            .map(|slot| mesh.counts.regular[slot] | mesh.counts.walls[slot] << 16),
        object: chunk.object,
        fade: chunk.fade.pack(),
        reserved: [0; 2],
    }
}

#[cfg(test)]
mod tests {
    use cp_core::face::Face;
    use cp_core::ids::MaterialId;
    use cp_core::math::IAabb;

    use super::*;
    use crate::voxel::layout::storage_slot;

    fn limits() -> SceneLimits {
        SceneLimits {
            pool_quads: 64 * u64::from(POOL_UNIT_QUADS),
            max_draw_items: 16,
            max_objects: 2,
        }
    }

    fn object(state: &mut SceneState) -> ObjectId {
        state
            .add_object(ObjectDesc {
                bounds: IAabb::new(IVec3::splat(-4096), IVec3::splat(4096)),
                pose: GridPose::IDENTITY,
            })
            .unwrap()
    }

    /// A mesh with `per_slot[layer][slot]` single-cell quads on the slot's face.
    fn mesh(per_slot: &[[u32; 12]; 3]) -> (Vec<PackedQuad>, [LayerCounts; 3]) {
        let mut quads = Vec::new();
        let mut layers = [LayerCounts::EMPTY; 3];
        for (layer, slots) in per_slot.iter().enumerate() {
            for (slot, &count) in slots.iter().enumerate() {
                let face = FACE_STORAGE_ORDER[slot % 6];
                for k in 0..count {
                    let cell = [(k % 32) as u8, (k / 32 % 32) as u8, (slot as u8) * 2];
                    quads.push(PackedQuad::new(cell, 1, 1, face, MaterialId(0), 0, false).unwrap());
                }
            }
            layers[layer] = LayerCounts::from_slots(*slots);
        }
        (quads, layers)
    }

    fn upload<'a>(
        object: ObjectId,
        quads: &'a [PackedQuad],
        layers: [LayerCounts; 3],
        origin: IVec3,
    ) -> ChunkUpload<'a> {
        ChunkUpload {
            quads,
            layers,
            origin,
            level: 0,
            object,
        }
    }

    fn one_slot(layer: usize, slot: usize, count: u32) -> [[u32; 12]; 3] {
        let mut per_slot = [[0; 12]; 3];
        per_slot[layer][slot] = count;
        per_slot
    }

    #[test]
    fn uploads_become_resident_within_the_budget_and_in_order() {
        let mut state = SceneState::new(limits());
        let grid = object(&mut state);
        let (quads, layers) = mesh(&one_slot(0, 1, 300));
        let a = state
            .insert_chunk(&upload(grid, &quads, layers, IVec3::ZERO))
            .unwrap();
        let b = state
            .insert_chunk(&upload(grid, &quads, layers, IVec3::new(32, 0, 0)))
            .unwrap();
        assert!(!state.is_resident(a).unwrap());
        assert_eq!(state.stats().pending_chunks, 2);
        assert_eq!(state.stats().pending_bytes, 2 * 300 * 8);
        // 300 quads need two units of 256.
        assert_eq!(state.stats().pool_free_quads, 60 * 256);
        let mut writes = Vec::new();
        // A budget smaller than one chunk still uploads one chunk per call.
        let report = state.take_uploads(100, &mut writes);
        assert_eq!(
            (report.chunks, report.bytes, report.pending_chunks),
            (1, 2400, 1)
        );
        assert!(state.is_resident(a).unwrap() && !state.is_resident(b).unwrap());
        assert_eq!(writes.len(), 1);
        assert_eq!(writes[0].first_quad, 0);
        assert_eq!(writes[0].quads, quads);
        assert_eq!(state.items().len(), 1);
        let report = state.take_uploads(u64::MAX, &mut writes);
        assert_eq!((report.chunks, report.pending_chunks), (1, 0));
        assert_eq!(writes[1].first_quad, 512);
        assert_eq!(state.items()[1].quad_base, 512);
        assert_eq!(state.items()[1].origin, [32, 0, 0]);
        assert_eq!(state.items()[1].slot_count(storage_slot(Face::PosY)), 300);
        assert_eq!(state.take_dirty_items(), vec![Range { start: 0, end: 2 }]);
        assert!(state.take_dirty_items().is_empty());
    }

    #[test]
    fn layers_become_items_with_their_offsets_and_bounds() {
        let mut state = SceneState::new(limits());
        let grid = object(&mut state);
        let mut per_slot = [[0; 12]; 3];
        per_slot[0][0] = 3;
        per_slot[0][6] = 2; // walls of +X
        per_slot[2][4] = 5;
        let (quads, layers) = mesh(&per_slot);
        let id = state
            .insert_chunk(&upload(grid, &quads, layers, IVec3::new(0, -32, 64)))
            .unwrap();
        state.take_uploads(u64::MAX, &mut Vec::new());
        let items = state.items();
        assert_eq!(items.len(), 2, "the empty cutout layer has no item");
        assert_eq!(items[0].layer_index(), 0);
        assert_eq!(items[0].quad_base, 0);
        assert_eq!(items[1].layer_index(), 2);
        assert_eq!(items[1].quad_base, 5);
        assert_eq!(items[0].quad_count(), 5);
        // Without walls the bounds cover the regular quads only (cells z = 0 for slot 0).
        let regular = CellBounds::unpack(items[0].cell_bounds);
        assert_eq!(
            regular,
            CellBounds {
                min: [0, 0, 0],
                max: [2, 0, 0]
            }
        );
        state
            .set_wall_mask(id, WallMask::from_faces(&[Face::PosX]))
            .unwrap();
        let with_walls = CellBounds::unpack(state.items()[0].cell_bounds);
        // Wall quads of slot 6 sit at z = 12.
        assert_eq!(
            with_walls,
            CellBounds {
                min: [0, 0, 0],
                max: [2, 0, 12]
            }
        );
        assert!(state.items()[0].walls().contains(Face::PosX));
        state.set_fade(id, Fade::new(0.5, true)).unwrap();
        assert_eq!(state.items()[1].draw_class(), 1);
        assert_eq!(state.take_dirty_items(), vec![Range { start: 0, end: 2 }]);
    }

    #[test]
    fn removal_keeps_the_item_table_dense_and_consistent() {
        let mut state = SceneState::new(limits());
        let grid = object(&mut state);
        let mut ids = Vec::new();
        for k in 0..5 {
            let mut per_slot = [[0; 12]; 3];
            per_slot[0][1] = 1 + k;
            per_slot[1][2] = 2;
            let (quads, layers) = mesh(&per_slot);
            let id = state
                .insert_chunk(&upload(
                    grid,
                    &quads,
                    layers,
                    IVec3::new(32 * k as i32, 0, 0),
                ))
                .unwrap();
            ids.push(id);
        }
        state.take_uploads(u64::MAX, &mut Vec::new());
        assert_eq!(state.items().len(), 10);
        // One non-empty range per item: one draw at most, five per class.
        assert_eq!(state.draw_bounds(), [5, 5]);
        assert_eq!(state.draw_bounds(), recomputed_bounds(&state));
        state.take_dirty_items();
        state.remove_chunk(ids[1]).unwrap();
        state.remove_chunk(ids[4]).unwrap();
        assert_eq!(state.items().len(), 6);
        assert_eq!(state.draw_bounds(), [3, 3]);
        state.set_fade(ids[0], Fade::new(0.25, false)).unwrap();
        assert_eq!(
            state.draw_bounds(),
            [2, 4],
            "the fading opaque item moved to class 1"
        );
        assert_eq!(state.draw_bounds(), recomputed_bounds(&state));
        // Every remaining item belongs to a live chunk and its chunk points back at it.
        let mut origins: Vec<i32> = state.items().iter().map(|item| item.origin[0]).collect();
        origins.sort_unstable();
        assert_eq!(origins, [0, 0, 64, 64, 96, 96]);
        for (index, &(chunk, layer)) in state.owners.iter().enumerate() {
            let mesh = state.chunks[chunk as usize]
                .chunk
                .as_ref()
                .unwrap()
                .resident
                .as_ref()
                .unwrap();
            assert_eq!(mesh.items[usize::from(layer)], Some(index as u32));
        }
        assert!(state.take_dirty_items().iter().all(|run| run.end <= 6));
        // Stale ids are rejected; the slot is reused with a new generation.
        assert!(matches!(
            state.remove_chunk(ids[1]),
            Err(VoxelError::UnknownChunk { .. })
        ));
        let (quads, layers) = mesh(&one_slot(0, 0, 1));
        let again = state
            .insert_chunk(&upload(grid, &quads, layers, IVec3::ZERO))
            .unwrap();
        assert_eq!(again.index(), ids[4].index());
        assert_ne!(again.generation(), ids[4].generation());
        // The pool is fully free again once everything is gone.
        for id in [ids[0], ids[2], ids[3], again] {
            state.remove_chunk(id).unwrap();
        }
        assert_eq!(state.stats().pool_free_quads, state.stats().pool_quads);
        assert_eq!(state.items().len(), 0);
        assert_eq!(state.draw_bounds(), [0, 0]);
        state.remove_object(grid).unwrap();
        assert_eq!(state.stats().objects, 0);
    }

    fn recomputed_bounds(state: &SceneState) -> [u64; 2] {
        let mut bounds = [0; 2];
        for item in state.items() {
            bounds[item.draw_class() as usize] += u64::from(item_draw_bound(item));
        }
        bounds
    }

    #[test]
    fn the_draw_bound_is_the_most_runs_of_any_visibility() {
        use crate::voxel::cull::count_runs;
        for nonempty in 0..1_u32 << 12 {
            let mut item = GpuDrawItem::default();
            for slot in 0..12 {
                if nonempty >> slot & 1 == 1 {
                    item.counts[slot % 6] |= if slot < 6 { 1 } else { 1 << 16 };
                }
            }
            // Every subset of the non-empty slots as the visible ones.
            let mut most = 0;
            let mut visible = nonempty;
            loop {
                most = most.max(count_runs(visible, nonempty));
                if visible == 0 {
                    break;
                }
                visible = (visible - 1) & nonempty;
            }
            assert_eq!(item_draw_bound(&item), most, "{nonempty:012b}");
        }
    }

    #[test]
    fn replacement_keeps_the_old_mesh_until_the_new_one_is_uploaded() {
        let mut state = SceneState::new(limits());
        let grid = object(&mut state);
        let (old_quads, old_layers) = mesh(&one_slot(0, 1, 10));
        let id = state
            .insert_chunk(&upload(grid, &old_quads, old_layers, IVec3::ZERO))
            .unwrap();
        state.take_uploads(u64::MAX, &mut Vec::new());
        let (new_quads, new_layers) = mesh(&one_slot(0, 2, 20));
        state
            .replace_chunk(id, &upload(grid, &new_quads, new_layers, IVec3::ZERO))
            .unwrap();
        assert!(state.has_pending_upload(id).unwrap());
        assert_eq!(state.items()[0].quad_count(), 10, "old mesh still drawn");
        // A second replacement supersedes the first queued one.
        let (newest, newest_layers) = mesh(&one_slot(0, 3, 30));
        state
            .replace_chunk(id, &upload(grid, &newest, newest_layers, IVec3::ZERO))
            .unwrap();
        assert_eq!(state.stats().pending_chunks, 1);
        let mut writes = Vec::new();
        let report = state.take_uploads(u64::MAX, &mut writes);
        assert_eq!(report.chunks, 1);
        assert_eq!(writes.len(), 1);
        assert_eq!(writes[0].quads, newest);
        assert_eq!(state.items().len(), 1);
        assert_eq!(state.items()[0].quad_count(), 30);
        // Only the newest mesh holds pool space.
        assert_eq!(
            state.stats().pool_free_quads,
            state.stats().pool_quads - 256
        );
        // Moving a chunk is not a replacement.
        assert!(matches!(
            state.replace_chunk(
                id,
                &upload(grid, &newest, newest_layers, IVec3::new(32, 0, 0))
            ),
            Err(VoxelError::ReplaceChangesPlacement { .. })
        ));
        // An empty replacement takes effect at once.
        state
            .replace_chunk(id, &upload(grid, &[], [LayerCounts::EMPTY; 3], IVec3::ZERO))
            .unwrap();
        assert!(state.items().is_empty());
        assert!(state.is_resident(id).unwrap());
        assert_eq!(state.stats().pool_free_quads, state.stats().pool_quads);
    }

    /// A mesh with one single-cell quad in each of the first `layers` layers.
    fn layered(layers: usize) -> (Vec<PackedQuad>, [LayerCounts; 3]) {
        let mut per_slot = [[0; 12]; 3];
        for slots in &mut per_slot[..layers] {
            slots[0] = 1;
        }
        mesh(&per_slot)
    }

    #[test]
    fn replacements_reserve_only_the_items_they_add() {
        // A table of four items, filled by two chunks of two layers.
        let mut state = SceneState::new(SceneLimits {
            max_draw_items: 4,
            ..limits()
        });
        let grid = object(&mut state);
        let (two, two_layers) = layered(2);
        let a = state
            .insert_chunk(&upload(grid, &two, two_layers, IVec3::ZERO))
            .unwrap();
        let b = state
            .insert_chunk(&upload(grid, &two, two_layers, IVec3::new(32, 0, 0)))
            .unwrap();
        state.take_uploads(u64::MAX, &mut Vec::new());
        assert_eq!(state.items().len(), 4);
        // An edit of a full table: the new mesh of `a` has as many layers as the old one.
        state
            .replace_chunk(a, &upload(grid, &two, two_layers, IVec3::ZERO))
            .unwrap();
        // A mesh with a third layer would need a fifth item.
        let (three, three_layers) = layered(3);
        assert_eq!(
            state.replace_chunk(b, &upload(grid, &three, three_layers, IVec3::new(32, 0, 0))),
            Err(VoxelError::TooManyDrawItems { limit: 4 })
        );
        // A mesh with fewer layers fits, but frees its item only once it is resident.
        let (one, one_layer) = layered(1);
        state
            .replace_chunk(b, &upload(grid, &one, one_layer, IVec3::new(32, 0, 0)))
            .unwrap();
        let new_chunk = upload(grid, &one, one_layer, IVec3::new(64, 0, 0));
        assert_eq!(
            state.insert_chunk(&new_chunk),
            Err(VoxelError::TooManyDrawItems { limit: 4 })
        );
        let report = state.take_uploads(u64::MAX, &mut Vec::new());
        assert_eq!((report.chunks, report.pending_chunks), (2, 0));
        assert_eq!(state.items().len(), 3);
        assert_eq!(state.pending_items, 0);
        let c = state.insert_chunk(&new_chunk).unwrap();
        state.take_uploads(u64::MAX, &mut Vec::new());
        assert_eq!(state.items().len(), 4);
        // Superseded and cancelled replacements give their reservations back.
        state
            .replace_chunk(c, &upload(grid, &one, one_layer, IVec3::new(64, 0, 0)))
            .unwrap();
        state
            .replace_chunk(c, &upload(grid, &one, one_layer, IVec3::new(64, 0, 0)))
            .unwrap();
        state.remove_chunk(c).unwrap();
        assert_eq!((state.pending_items, state.items().len()), (0, 3));
        state
            .insert_chunk(&upload(grid, &one, one_layer, IVec3::new(96, 0, 0)))
            .unwrap();
    }

    #[test]
    fn prepared_chunks_are_checked_elsewhere_and_queued_without_a_copy() {
        fn send_and_sync<T: Send + Sync>() {}
        send_and_sync::<PreparedChunk>();
        send_and_sync::<RefusedChunk>();
        let mut state = SceneState::new(SceneLimits {
            max_draw_items: 2,
            ..limits()
        });
        let grid = object(&mut state);
        let (quads, layers) = mesh(&one_slot(0, 1, 40));
        // Checked on a worker thread.
        let prepared = std::thread::spawn(move || {
            PreparedChunk::from_vec(quads, layers, IVec3::new(0, 32, 0), 0, grid)
        })
        .join()
        .unwrap()
        .unwrap();
        assert_eq!(
            (prepared.object(), prepared.origin(), prepared.level()),
            (grid, IVec3::new(0, 32, 0), 0)
        );
        let address = prepared.quads().as_ptr();
        let id = state.insert_prepared(prepared).unwrap();
        let mut writes = Vec::new();
        state.take_uploads(u64::MAX, &mut writes);
        assert!(state.is_resident(id).unwrap());
        assert_eq!(
            writes[0].quads.as_ptr(),
            address,
            "the quads were moved, not copied"
        );
        assert_eq!(state.items()[0].slot_count(1), 40);
        // The same mesh through a borrowed upload gives the same item.
        let (quads, layers) = mesh(&one_slot(0, 1, 40));
        let copied = upload(grid, &quads, layers, IVec3::new(32, 32, 0))
            .prepare()
            .unwrap();
        assert_eq!(copied.quads(), quads.as_slice());
        let other = state.insert_prepared(copied).unwrap();
        state.take_uploads(u64::MAX, &mut Vec::new());
        let [first, second] = [0, 1].map(|index| state.items()[index]);
        assert_eq!(first.counts, second.counts);
        assert_eq!(first.cell_bounds, second.cell_bounds);
        // A refused chunk comes back unchanged, and nothing in the scene changed.
        let (extra, extra_layers) = mesh(&one_slot(0, 2, 3));
        let prepared =
            PreparedChunk::from_vec(extra.clone(), extra_layers, IVec3::ZERO, 0, grid).unwrap();
        let free = state.stats().pool_free_quads;
        let refused = state.insert_prepared(prepared.clone()).unwrap_err();
        assert_eq!(refused.error, VoxelError::TooManyDrawItems { limit: 2 });
        assert_eq!(refused.chunk, prepared);
        assert_eq!(state.stats().pool_free_quads, free);
        assert_eq!(state.stats().chunks, 2);
        let moved =
            PreparedChunk::from_vec(extra, extra_layers, IVec3::new(32, 0, 0), 0, grid).unwrap();
        let refused = state.replace_prepared(other, moved).unwrap_err();
        assert!(matches!(
            refused.error,
            VoxelError::ReplaceChangesPlacement { .. }
        ));
        assert_eq!(refused.chunk.origin(), IVec3::new(32, 0, 0));
        // Once there is room, the refused chunk goes in as it is.
        state.remove_chunk(other).unwrap();
        let id = state.insert_prepared(refused.chunk).unwrap();
        assert!(state.has_pending_upload(id).unwrap());
        // The same mesh as a replacement of a resident chunk with as many layers.
        let again = PreparedChunk::from_vec(
            mesh(&one_slot(0, 1, 5)).0,
            mesh(&one_slot(0, 1, 5)).1,
            IVec3::new(0, 32, 0),
            0,
            grid,
        )
        .unwrap();
        state.replace_prepared(state_id(&state, 0), again).unwrap();
        // Errors of the scene-independent checks come from preparing.
        let (bad, bad_layers) = mesh(&one_slot(0, 0, 2));
        assert!(matches!(
            PreparedChunk::from_vec(bad.clone(), bad_layers, IVec3::new(16, 0, 0), 0, grid),
            Err(VoxelError::MisalignedOrigin { .. })
        ));
        assert!(matches!(
            PreparedChunk::from_vec(bad, bad_layers, IVec3::ZERO, 21, grid),
            Err(VoxelError::InvalidLevel { level: 21 })
        ));
        // A refusal displays as its error.
        assert_eq!(
            RefusedChunk {
                error: VoxelError::TooManyDrawItems { limit: 2 },
                chunk: prepared,
            }
            .to_string(),
            "the draw item table is full (2 items)"
        );
    }

    /// The id of the live chunk whose slot is `index`.
    fn state_id(state: &SceneState, index: u32) -> ChunkId {
        ChunkId {
            index,
            generation: state.chunks[index as usize].generation,
        }
    }

    #[test]
    fn bad_uploads_are_errors() {
        let mut state = SceneState::new(limits());
        let grid = object(&mut state);
        let (quads, layers) = mesh(&one_slot(0, 0, 2));
        let check = |state: &mut SceneState, upload: ChunkUpload<'_>| state.insert_chunk(&upload);
        let mut bad = upload(grid, &quads, layers, IVec3::new(16, 0, 0));
        assert!(matches!(
            check(&mut state, bad),
            Err(VoxelError::MisalignedOrigin { .. })
        ));
        bad = upload(grid, &quads, layers, IVec3::new(VOXEL_LIMIT, 0, 0));
        assert!(matches!(
            check(&mut state, bad),
            Err(VoxelError::OriginOutOfRange { .. })
        ));
        bad = upload(grid, &quads[..1], layers, IVec3::ZERO);
        assert!(matches!(
            check(&mut state, bad),
            Err(VoxelError::QuadCountMismatch { .. })
        ));
        bad = upload(grid, &quads, layers, IVec3::ZERO);
        bad.level = 21;
        assert!(matches!(
            check(&mut state, bad),
            Err(VoxelError::InvalidLevel { level: 21 })
        ));
        bad = upload(ObjectId(1), &quads, layers, IVec3::ZERO);
        assert!(matches!(
            check(&mut state, bad),
            Err(VoxelError::UnknownObject { object: 1 })
        ));
        // A quad in the range of another face.
        let mut wrong = layers;
        wrong[0] = LayerCounts::from_slots([0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
        bad = upload(grid, &quads, wrong, IVec3::ZERO);
        assert!(matches!(
            check(&mut state, bad),
            Err(VoxelError::QuadOnWrongFace {
                index: 0,
                face: Face::PosX,
                expected: Face::PosY
            })
        ));
        // Corrupt words.
        let corrupt = [PackedQuad { w0: 7 << 25, w1: 0 }, quads[1]];
        bad = upload(grid, &corrupt, layers, IVec3::ZERO);
        assert!(matches!(
            check(&mut state, bad),
            Err(VoxelError::InvalidQuad { index: 0, .. })
        ));
        let mut huge = layers;
        huge[1].walls[0] = MAX_WALL_RANGE_QUADS + 1;
        bad = upload(grid, &quads, huge, IVec3::ZERO);
        assert!(matches!(
            check(&mut state, bad),
            Err(VoxelError::RangeTooLarge {
                layer: 1,
                slot: 6,
                ..
            })
        ));
        // Pool and item limits.
        let mut pool_full = SceneState::new(SceneLimits {
            max_draw_items: 64,
            ..limits()
        });
        let big_grid = object(&mut pool_full);
        let (big, big_layers) = mesh(&one_slot(0, 1, 1024));
        for k in 0..16 {
            let placed = upload(big_grid, &big, big_layers, IVec3::new(32 * k, 0, 0));
            let result = pool_full.insert_chunk(&placed);
            assert!(result.is_ok(), "{k}: {result:?}");
        }
        let full =
            pool_full.insert_chunk(&upload(big_grid, &big, big_layers, IVec3::new(0, 32, 0)));
        assert!(
            matches!(
                full,
                Err(VoxelError::PoolFull {
                    quads: 1024,
                    free_quads: 0,
                    ..
                })
            ),
            "{full:?}"
        );
        assert_eq!(pool_full.stats().chunks, 16);
        assert!(matches!(
            pool_full.remove_object(big_grid),
            Err(VoxelError::ObjectInUse { chunks: 16, .. })
        ));
        let mut items_full = SceneState::new(SceneLimits {
            max_draw_items: 1,
            ..limits()
        });
        let grid = object(&mut items_full);
        let mut two_layers = [[0; 12]; 3];
        two_layers[0][0] = 1;
        two_layers[2][0] = 1;
        let (quads, layers) = mesh(&two_layers);
        assert!(matches!(
            items_full.insert_chunk(&upload(grid, &quads, layers, IVec3::ZERO)),
            Err(VoxelError::TooManyDrawItems { limit: 1 })
        ));
        // Objects.
        assert!(
            state
                .add_object(ObjectDesc {
                    bounds: IAabb::EMPTY,
                    pose: GridPose::IDENTITY
                })
                .is_err()
        );
        object(&mut state);
        assert!(matches!(
            state.add_object(ObjectDesc {
                bounds: IAabb::new(IVec3::ZERO, IVec3::ONE),
                pose: GridPose::IDENTITY,
            }),
            Err(VoxelError::TooManyObjects { limit: 2 })
        ));
    }
}
