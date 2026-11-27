//! A synthetic calibration scene for the renderer benchmarks (contract §9, E2: G-buffer time
//! at 1.0 M visible triangles, cull time, nanoseconds per indirect draw).
//!
//! `resident_chunks` chunks with `quads_per_chunk` quads each are inserted into one grid;
//! `visible_chunks` of them lie on a square grid below a camera that looks straight down and
//! sees all of them, the rest lie above the camera, where the frustum test rejects them. Every
//! quad is an up face (`+Y`) of a varied height, so each visible chunk is exactly one draw and
//! the counts of a frame are known in advance ([`CalibrationCounts`]).

use cp_core::coords::{CHUNK_SIZE, MAX_LOD_LEVEL};
use cp_core::det;
use cp_core::face::Face;
use cp_core::ids::MaterialId;
use cp_core::math::{DVec3, IAabb, IVec3};
use cp_core::quad::PackedQuad;

use super::camera::{Camera, GridPose};
use super::layout::storage_slot;
use super::scene::{
    ChunkId, ChunkUpload, LayerCounts, ObjectDesc, ObjectId, VoxelError, VoxelScene,
};

/// What the calibration scene contains.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CalibrationParams {
    /// Chunks in the scene (draw items the cull looks at).
    pub resident_chunks: u32,
    /// Chunks in view, `<= resident_chunks`; each is one draw.
    pub visible_chunks: u32,
    /// Quads per chunk, `1..=1024`; they tile the chunk's top in a square grid.
    pub quads_per_chunk: u32,
    /// LOD level of every chunk.
    pub level: u8,
    /// Material ids used, `0..materials` (at least 1).
    pub materials: u16,
    /// Vertical field of view of the camera, radians, `0 < fov_y < 3`.
    pub fov_y: f32,
    /// Aspect ratio (width / height) of the output.
    pub aspect: f32,
}

impl Default for CalibrationParams {
    /// 1.0 M visible triangles in 5,000 draws at 16:9.
    fn default() -> Self {
        Self {
            resident_chunks: 5_000,
            visible_chunks: 5_000,
            quads_per_chunk: 100,
            level: 0,
            materials: 1,
            fov_y: 60.0_f32.to_radians(),
            aspect: 16.0 / 9.0,
        }
    }
}

/// The per-frame counts the calibration scene produces.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CalibrationCounts {
    /// Draw items (one per chunk).
    pub draw_items: u32,
    /// Visible draws (one per visible chunk).
    pub draws: u32,
    /// Quads in the visible draws.
    pub quads: u64,
    /// Triangles of the visible draws.
    pub triangles: u64,
}

/// A calibration scene in a [`VoxelScene`].
#[derive(Clone, Debug)]
pub struct CalibrationScene {
    /// The grid of every chunk.
    pub object: ObjectId,
    /// Every chunk, the visible ones first.
    pub chunks: Vec<ChunkId>,
    /// The camera that sees exactly the visible chunks.
    pub camera: Camera,
    /// What a frame of this camera draws.
    pub expected: CalibrationCounts,
}

/// Where everything of a calibration scene goes.
struct Layout {
    camera: Camera,
    origins: Vec<IVec3>,
    bounds: IAabb,
}

impl Layout {
    fn new(params: &CalibrationParams) -> Result<Self, VoxelError> {
        let invalid = |reason| VoxelError::InvalidConfig { reason };
        if params.visible_chunks > params.resident_chunks {
            return Err(invalid("calibration: more visible than resident chunks"));
        }
        if !(1..=1024).contains(&params.quads_per_chunk) {
            return Err(invalid("calibration: quads per chunk must be in 1..=1024"));
        }
        if params.level > MAX_LOD_LEVEL || params.materials == 0 {
            return Err(invalid("calibration: level above 20 or no material"));
        }
        if !(params.fov_y > 0.0 && params.fov_y < 3.0 && params.aspect > 0.0) {
            return Err(invalid("calibration: field of view or aspect out of range"));
        }
        if params.resident_chunks > 1 << 20 {
            return Err(invalid("calibration: at most 2^20 chunks"));
        }
        let extent = CHUNK_SIZE << params.level;
        let columns = ceil_sqrt(params.visible_chunks.max(1));
        let rows = params.visible_chunks.max(1).div_ceil(columns);
        // At most 2^20 chunks: the grid indices stay far inside i32.
        let place = |index: u32, y: i32| {
            let x = (index % columns) as i32 - (columns / 2) as i32;
            let z = (index / columns) as i32 - (rows / 2) as i32;
            IVec3::new(x * extent, y, z * extent)
        };
        let lo = place(0, 0).as_dvec3();
        let hi = place(columns * rows - 1, 0).as_dvec3() + DVec3::splat(f64::from(extent));
        let center = (lo + hi) * 0.5;
        let half = (hi - lo) * 0.5;
        let tan_y = f64::from(det::tan(0.5 * params.fov_y));
        let tan_x = tan_y * f64::from(params.aspect);
        // 10 % margin on the frustum sides; the tops of the chunks are the nearest points.
        let distance = det::max(half.x / (0.9 * tan_x), half.z / (0.9 * tan_y));
        let height = f64::from(extent) + distance + 1.0;
        let position = DVec3::new(center.x + 0.25, height + 0.5, center.z + 0.25);
        let camera = Camera::look_at(position, position - DVec3::Y, DVec3::NEG_Z, params.fov_y);
        // Hidden chunks lie above the camera, behind its near plane. The height is a few
        // kilometres at most for 2^20 chunks.
        let hidden_y = ((height / f64::from(extent)).ceil() as i32 + 2) * extent;
        let mut bounds = IAabb::from_min_size(IVec3::ZERO, IVec3::splat(extent));
        let origins = (0..params.resident_chunks)
            .map(|chunk| {
                let origin = if chunk < params.visible_chunks {
                    place(chunk, 0)
                } else {
                    place(chunk - params.visible_chunks, hidden_y)
                };
                bounds = bounds.union(&IAabb::from_min_size(origin, IVec3::splat(extent)));
                origin
            })
            .collect();
        Ok(Self {
            camera,
            origins,
            bounds,
        })
    }
}

impl CalibrationScene {
    /// Inserts the scene: one object and its chunks, whose uploads are queued (flush them
    /// with [`VoxelScene::flush_all_uploads`] before measuring).
    ///
    /// # Errors
    /// [`VoxelError::InvalidConfig`] for parameters out of range, and the errors of
    /// [`VoxelScene::insert_chunk`] (pool or item table too small); chunks inserted before
    /// the error stay.
    pub fn build(scene: &mut VoxelScene, params: &CalibrationParams) -> Result<Self, VoxelError> {
        let layout = Layout::new(params)?;
        let object = scene.add_object(ObjectDesc {
            bounds: layout.bounds,
            pose: GridPose::IDENTITY,
        })?;
        let mut chunks = Vec::with_capacity(layout.origins.len());
        let mut quads = Vec::with_capacity(params.quads_per_chunk as usize);
        for (chunk, &origin) in (0..).zip(&layout.origins) {
            chunk_quads(chunk, params, &mut quads);
            chunks.push(scene.insert_chunk(&ChunkUpload {
                quads: &quads,
                layers: layers(params),
                origin,
                level: params.level,
                object,
            })?);
        }
        Ok(Self {
            object,
            chunks,
            camera: layout.camera,
            expected: expected(params),
        })
    }

    /// Removes the scene's chunks and its object.
    ///
    /// # Errors
    /// [`VoxelError::UnknownChunk`] or [`VoxelError::UnknownObject`] when parts of it were
    /// removed already.
    pub fn remove(self, scene: &mut VoxelScene) -> Result<(), VoxelError> {
        for chunk in self.chunks {
            scene.remove_chunk(chunk)?;
        }
        scene.remove_object(self.object)
    }
}

/// The counts a frame of the scene's camera produces.
fn expected(params: &CalibrationParams) -> CalibrationCounts {
    let quads = u64::from(params.visible_chunks) * u64::from(params.quads_per_chunk);
    CalibrationCounts {
        draw_items: params.resident_chunks,
        draws: params.visible_chunks,
        quads,
        triangles: 2 * quads,
    }
}

/// Every quad in the opaque layer's `+Y` range.
fn layers(params: &CalibrationParams) -> [LayerCounts; 3] {
    let mut layers = [LayerCounts::EMPTY; 3];
    layers[0].regular[storage_slot(Face::PosY)] = params.quads_per_chunk;
    layers
}

/// Smallest `n` with `n * n >= value` (`value >= 1`).
fn ceil_sqrt(value: u32) -> u32 {
    let mut root = f64::from(value).sqrt() as u32;
    while root * root < value {
        root += 1;
    }
    while root > 1 && (root - 1) * (root - 1) >= value {
        root -= 1;
    }
    root.max(1)
}

/// The quads of chunk `chunk`: a square grid of up faces, heights from a hash.
fn chunk_quads(chunk: u32, params: &CalibrationParams, quads: &mut Vec<PackedQuad>) {
    quads.clear();
    let per_side = ceil_sqrt(params.quads_per_chunk);
    // per_side <= 32, so side >= 1 and per_side * side <= 32.
    let side = (32 / per_side) as u8;
    for quad in 0..params.quads_per_chunk {
        let hash = (chunk.wrapping_mul(0x9E37_79B1) ^ quad.wrapping_mul(0x85EB_CA77))
            .wrapping_mul(0xC2B2_AE3D);
        let x = (quad % per_side) as u8 * side;
        let z = (quad / per_side) as u8 * side;
        let y = (hash >> 28) as u8;
        let material = (chunk.wrapping_mul(31).wrapping_add(quad)) % u32::from(params.materials);
        // Inside the chunk by construction: x + side <= 32, z + side <= 32, y < 16.
        let packed = PackedQuad::new(
            [x, y, z],
            side,
            side,
            Face::PosY,
            MaterialId(material as u16),
            0,
            false,
        )
        .expect("calibration quads lie inside the chunk");
        quads.push(packed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::voxel::camera::{FrameCamera, compose_object};
    use crate::voxel::cull::DrawList;
    use crate::voxel::layout::MAX_ITEM_DRAWS;
    use crate::voxel::scene::SceneLimits;
    use crate::voxel::state::SceneState;

    #[test]
    fn square_roots_round_up() {
        for value in 1..5000 {
            let root = ceil_sqrt(value);
            assert!(
                root * root >= value && (root - 1) * (root - 1) < value,
                "{value}"
            );
        }
    }

    #[test]
    fn the_camera_sees_exactly_the_visible_chunks() {
        for (resident, visible, quads, level) in [
            (10, 7, 100, 0),
            (500, 300, 25, 1),
            (1, 0, 1, 0),
            (64, 64, 1024, 3),
            (2000, 2000, 1, 0),
        ] {
            let params = CalibrationParams {
                resident_chunks: resident,
                visible_chunks: visible,
                quads_per_chunk: quads,
                level,
                materials: 3,
                ..CalibrationParams::default()
            };
            let layout = Layout::new(&params).unwrap();
            let mut state = SceneState::new(SceneLimits {
                pool_quads: 1 << 24,
                max_draw_items: 1 << 16,
                max_objects: 1,
            });
            let object = state
                .add_object(ObjectDesc {
                    bounds: layout.bounds,
                    pose: GridPose::IDENTITY,
                })
                .unwrap();
            let mut chunk_quad_list = Vec::new();
            for (chunk, &origin) in (0..).zip(&layout.origins) {
                chunk_quads(chunk, &params, &mut chunk_quad_list);
                state
                    .insert_chunk(&ChunkUpload {
                        quads: &chunk_quad_list,
                        layers: layers(&params),
                        origin,
                        level,
                        object,
                    })
                    .unwrap();
            }
            state.take_uploads(u64::MAX, &mut Vec::new());
            let frame = FrameCamera {
                camera: layout.camera,
                aspect: params.aspect,
                jitter: [0.0, 0.0],
            };
            let pose = GridPose::IDENTITY;
            let gpu_object = compose_object(&layout.bounds, &pose, &pose, &frame, &frame);
            let items = state.items();
            let mut list = DrawList::default();
            list.build(items, &[gpu_object], MAX_ITEM_DRAWS * items.len() as u32);
            let counts = CalibrationCounts {
                draw_items: items.len() as u32,
                draws: list.draws(0) + list.draws(1),
                quads: list.visible_quads,
                triangles: 2 * list.visible_quads,
            };
            assert_eq!(counts, expected(&params), "{params:?}");
        }
    }

    #[test]
    fn bad_parameters_are_refused() {
        let base = CalibrationParams::default();
        for params in [
            CalibrationParams {
                visible_chunks: base.resident_chunks + 1,
                ..base
            },
            CalibrationParams {
                quads_per_chunk: 0,
                ..base
            },
            CalibrationParams {
                quads_per_chunk: 1025,
                ..base
            },
            CalibrationParams {
                materials: 0,
                ..base
            },
            CalibrationParams { fov_y: 0.0, ..base },
        ] {
            assert!(matches!(
                Layout::new(&params),
                Err(VoxelError::InvalidConfig { .. })
            ));
        }
    }
}
