//! Level-0 queries on a snapshot (contract §5.1): cell reads, raycasts, box overlaps.

use std::sync::Arc;

use cp_core::block::BlockFlags;
use cp_core::coords::{CHUNK_BITS, CHUNK_MASK, LodKey, VOXEL_LIMIT, cell_index};
use cp_core::face::Face;
use cp_core::ids::BlockStateId;
use cp_core::math::{DAabb, DVec3, IVec3};

use crate::chunk::VoxelChunk;
use crate::snapshot::VoxelSnapshot;

fn never() -> bool {
    false
}

/// The first cell a ray hits.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RayHit {
    /// The hit voxel.
    pub cell: IVec3,
    /// The face of `cell` through which the ray entered it. For a hit in the start cell
    /// (`distance == 0`) the face that points most against the ray direction.
    pub face: Face,
    /// The state of `cell`.
    pub state: BlockStateId,
    /// Distance in metres from the ray origin to the entry point, along the normalised
    /// direction.
    pub distance: f64,
}

/// Level-0 composite cells of one chunk: its override layer over its base chunk.
struct ChunkView {
    layer: Option<Arc<VoxelChunk>>,
    /// `None` when the layer leaves no cell to the base.
    base: Option<Arc<VoxelChunk>>,
}

impl ChunkView {
    #[inline]
    fn get(&self, index: usize) -> BlockStateId {
        if let Some(layer) = &self.layer {
            let state = layer.get(index);
            if !state.is_unchanged() {
                return state;
            }
        }
        self.base
            .as_ref()
            .map_or(BlockStateId::AIR, |base| base.get(index))
    }

    /// The state of every cell if the chunk is known to be uniform without a cell scan.
    fn uniform(&self) -> Option<BlockStateId> {
        match (&self.layer, &self.base) {
            (None, Some(base)) => base.blocks.uniform_state(),
            (Some(layer), _) => layer
                .blocks
                .uniform_state()
                .filter(|state| !state.is_unchanged()),
            (None, None) => None,
        }
    }
}

/// Reads level-0 composite cells, keeping the chunk last read.
struct Reader<'a> {
    snapshot: &'a VoxelSnapshot,
    current: Option<(IVec3, ChunkView)>,
}

impl<'a> Reader<'a> {
    fn new(snapshot: &'a VoxelSnapshot) -> Self {
        Self {
            snapshot,
            current: None,
        }
    }

    fn view(&mut self, chunk: IVec3) -> &ChunkView {
        if self.current.as_ref().is_none_or(|(pos, _)| *pos != chunk) {
            let key = LodKey::lod0(chunk);
            let layer = self.snapshot.layers.layer(key).cloned();
            let needs_base = layer
                .as_ref()
                .is_none_or(|layer| layer.blocks.contains(BlockStateId::UNCHANGED));
            let base = needs_base.then(|| {
                self.snapshot
                    .base_chunk(key, &never)
                    .expect("generation without a cancel request cannot be cancelled")
            });
            self.current = Some((chunk, ChunkView { layer, base }));
        }
        let (_, view) = self.current.as_ref().expect("view was just loaded");
        view
    }

    fn get(&mut self, cell: IVec3) -> BlockStateId {
        let chunk = cell >> CHUNK_BITS;
        let local = cell & CHUNK_MASK;
        let index = cell_index(local.x as u8, local.y as u8, local.z as u8);
        self.view(chunk).get(index)
    }
}

/// True when every component of `cell` lies in `[-VOXEL_LIMIT, VOXEL_LIMIT)`.
#[inline]
fn inside_limits(cell: IVec3) -> bool {
    cell.cmpge(IVec3::splat(-VOXEL_LIMIT)).all() && cell.cmplt(IVec3::splat(VOXEL_LIMIT)).all()
}

impl VoxelSnapshot {
    /// The level-0 composite state of one voxel. Voxels outside `[-VOXEL_LIMIT, VOXEL_LIMIT)`
    /// read as air.
    #[must_use]
    pub fn cell_state(&self, cell: IVec3) -> BlockStateId {
        if !inside_limits(cell) {
            return BlockStateId::AIR;
        }
        Reader::new(self).get(cell)
    }

    /// The first non-air level-0 cell on the ray from `origin` (grid metres) along `dir`
    /// within `max_dist` metres; [`VoxelSnapshot::raycast_with`] with "any state but air".
    #[must_use]
    pub fn raycast(&self, origin: DVec3, dir: DVec3, max_dist: f64) -> Option<RayHit> {
        self.raycast_with(origin, dir, max_dist, |state| !state.is_air())
    }

    /// The first level-0 cell on the ray for which `is_hit(state)` holds.
    ///
    /// Voxel DDA (Amanatides–Woo) in `f64`: the time to every cell boundary is computed
    /// directly from the integer boundary (`(boundary - origin) / dir`), never accumulated, so
    /// chunk borders and negative coordinates are exact. The cells visited are those whose
    /// interior the ray passes through, in order; where the ray passes exactly through an
    /// edge or corner, the axes are stepped one at a time in the order x, y, z, so a cell
    /// touched only at that edge or corner is visited as well. The start cell is the cell the
    /// ray enters from `origin` (for an origin on a boundary, the cell on the side `dir`
    /// points to) and is tested too (`distance` 0).
    ///
    /// `dir` need not be normalised; distances are in metres. Returns `None` for a zero or
    /// non-finite `dir`, a non-finite origin, a negative or NaN `max_dist`, or when the ray
    /// leaves `[-VOXEL_LIMIT, VOXEL_LIMIT)`. Cost: proportional to the cells traversed.
    pub fn raycast_with(
        &self,
        origin: DVec3,
        dir: DVec3,
        max_dist: f64,
        mut is_hit: impl FnMut(BlockStateId) -> bool,
    ) -> Option<RayHit> {
        if !origin.is_finite() || !dir.is_finite() || max_dist.is_nan() || max_dist < 0.0 {
            return None;
        }
        let length = dir.length();
        if length <= 0.0 {
            return None;
        }
        let limit = f64::from(VOXEL_LIMIT);
        if origin.cmplt(DVec3::splat(-limit)).any() || origin.cmpgt(DVec3::splat(limit)).any() {
            return None;
        }
        let dir = dir / length;
        let mut cell = IVec3::ZERO;
        let mut step = IVec3::ZERO;
        for axis in 0..3 {
            let floor = origin[axis].floor();
            // An origin on a boundary starts in the cell the ray moves into.
            let start = if dir[axis] < 0.0 && floor == origin[axis] {
                floor - 1.0
            } else {
                floor
            };
            cell[axis] = start as i32;
            step[axis] = if dir[axis] > 0.0 {
                1
            } else if dir[axis] < 0.0 {
                -1
            } else {
                0
            };
        }
        if !inside_limits(cell) {
            return None;
        }
        let boundary_time = |axis: usize, coordinate: i32| -> f64 {
            match step[axis] {
                1 => (f64::from(coordinate + 1) - origin[axis]) / dir[axis],
                -1 => (f64::from(coordinate) - origin[axis]) / dir[axis],
                _ => f64::INFINITY,
            }
        };

        let mut reader = Reader::new(self);
        let state = reader.get(cell);
        if is_hit(state) {
            return Some(RayHit {
                cell,
                face: Face::from_dominant_axis(-dir),
                state,
                distance: 0.0,
            });
        }
        let mut next = [
            boundary_time(0, cell.x),
            boundary_time(1, cell.y),
            boundary_time(2, cell.z),
        ];
        loop {
            let axis = if next[0] <= next[1] && next[0] <= next[2] {
                0
            } else if next[1] <= next[2] {
                1
            } else {
                2
            };
            let time = next[axis];
            if time > max_dist {
                return None;
            }
            cell[axis] += step[axis];
            if !inside_limits(cell) {
                return None;
            }
            next[axis] = boundary_time(axis, cell[axis]);
            let state = reader.get(cell);
            if is_hit(state) {
                return Some(RayHit {
                    cell,
                    face: Face::from_axis(axis, step[axis] > 0),
                    state,
                    distance: time,
                });
            }
        }
    }

    /// True when a level-0 cell that overlaps `aabb` (grid metres) has all of `flags`.
    ///
    /// A cell overlaps the box when they share interior points: the cells
    /// `floor(min) ..= ceil(max) - 1` on every axis, so a box face lying exactly on a cell
    /// boundary does not reach into the next cell (a body standing on the ground does not
    /// overlap it). With empty `flags` every cell matches. An empty or NaN box overlaps
    /// nothing; the box is clipped to `[-VOXEL_LIMIT, VOXEL_LIMIT)`.
    #[must_use]
    pub fn overlaps(&self, aabb: DAabb, flags: BlockFlags) -> bool {
        if aabb.is_empty() {
            return false;
        }
        let limit = f64::from(VOXEL_LIMIT);
        let clip = |value: f64| cp_core::det::clamp(value, -limit, limit) as i32;
        let lo = IVec3::new(
            clip(aabb.min.x.floor()),
            clip(aabb.min.y.floor()),
            clip(aabb.min.z.floor()),
        );
        let hi = IVec3::new(
            clip(aabb.max.x.ceil()),
            clip(aabb.max.y.ceil()),
            clip(aabb.max.z.ceil()),
        );
        if lo.cmpge(hi).any() {
            return false;
        }
        let matches = |state: BlockStateId| self.tables.flags_of(state).contains(flags);
        let first_chunk = lo >> CHUNK_BITS;
        let last_chunk = (hi - 1) >> CHUNK_BITS;
        let mut reader = Reader::new(self);
        for cy in first_chunk.y..=last_chunk.y {
            for cz in first_chunk.z..=last_chunk.z {
                for cx in first_chunk.x..=last_chunk.x {
                    let chunk = IVec3::new(cx, cy, cz);
                    let view = reader.view(chunk);
                    if let Some(state) = view.uniform() {
                        if matches(state) {
                            return true;
                        }
                        continue;
                    }
                    let origin = chunk << CHUNK_BITS;
                    let from = (lo - origin).max(IVec3::ZERO);
                    let to = (hi - origin).min(IVec3::splat(32));
                    for y in from.y..to.y {
                        for z in from.z..to.z {
                            for x in from.x..to.x {
                                let index = cell_index(x as u8, y as u8, z as u8);
                                if matches(view.get(index)) {
                                    return true;
                                }
                            }
                        }
                    }
                }
            }
        }
        false
    }
}
