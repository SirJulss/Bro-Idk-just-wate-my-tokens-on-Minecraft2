//! The structures scene: cp-worldgen's [`StructureField`] (pillar, one-thick wall, beam, hollow
//! cube and tree, twice) generated at LOD levels 0, 1, 2 and 3 and set out in four rows side by
//! side, so that the conservative rasterisation of every shape on the coarse levels can be
//! compared with level 0 (contract §5.1, §9 E1).
//!
//! The ground is one object under all rows: the field's own ground (grass on dirt and stone,
//! flat at `y = 0` on every level), level 0 within 320 m of the rows and coarser rings out to
//! 6 km, so that it reaches the haze of the horizon; being flat, the levels meet without a gap.
//! Only the shapes differ between the rows. Row `L` is an object of its own holding the field's
//! level-`L` chunks above the ground, moved `ROW_SPACING * L` metres along `+Z`. The field holds
//! no shape there at level 0, so the ground's grass lies under every moved row.

use std::sync::Arc;

use cp_core::coords::{CHUNK_SIZE, LodKey};
use cp_core::math::{DVec3, IAabb, IVec3};
use cp_render::voxel::{GridPose, ObjectDesc, ObjectId, VoxelScene, WallMask};
use cp_voxel::{BaseSource, VoxelSnapshot, VoxelWorld, WorldConfig};
use cp_worldgen::StructureField;

use super::SceneCamera;
use super::build::ChunkRequest;
use super::lod::{LodParams, plan_columns};
use super::sky::Sun;
use crate::Failure;
use crate::assets::Assets;

/// Levels shown, one row each.
pub const LEVELS: [u8; 4] = [0, 1, 2, 3];

/// Distance between the rows along `+Z` (m): the field's shapes lie within `z` 4..56.
pub const ROW_SPACING: f64 = 72.0;

/// The field's shapes lie within `x` 8..134 and `z` 4..56 (m).
const FIELD_EXTENT: [i32; 2] = [140, 64];

/// Number of cameras.
pub const CAMERA_COUNT: usize = 3;

/// The structure field and its cameras.
pub struct StructureScene {
    field: Arc<StructureField>,
    _world: VoxelWorld,
    snapshot: Arc<VoxelSnapshot>,
}

impl std::fmt::Debug for StructureScene {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StructureScene")
            .field("identity", self.field.identity())
            .finish_non_exhaustive()
    }
}

/// The objects of the scene in a renderer: the ground and one row per level.
#[derive(Clone, Debug)]
pub struct StructureObjects {
    /// The level-0 ground under every row.
    pub ground: ObjectId,
    /// Row `i` shows level `LEVELS[i]`.
    pub rows: Vec<ObjectId>,
}

impl StructureScene {
    /// The field with the base pack's states.
    ///
    /// # Errors
    /// A [`Failure`] when the palette or the field cannot be made.
    pub fn new(assets: &Assets) -> Result<Self, Failure> {
        let palette = assets.palette()?;
        let field = StructureField::new(1, palette, &assets.tables().flags)
            .map_err(|error| Failure::new(format!("structure field: {error}")))?;
        let field = Arc::new(field);
        let base: Arc<dyn BaseSource> = field.clone();
        let world = VoxelWorld::new(base, assets.voxel_tables(), WorldConfig::default());
        let snapshot = Arc::new(world.snapshot());
        Ok(Self {
            field,
            _world: world,
            snapshot,
        })
    }

    /// The snapshot the chunk jobs read.
    pub fn snapshot(&self) -> &Arc<VoxelSnapshot> {
        &self.snapshot
    }

    /// Adds the ground and the rows to `scene` and returns the chunks to build.
    ///
    /// # Errors
    /// A [`Failure`] when the scene refuses an object.
    pub fn install(
        &self,
        scene: &mut VoxelScene,
    ) -> Result<(StructureObjects, Vec<ChunkRequest>), Failure> {
        let add = |scene: &mut VoxelScene, bounds: IAabb, translation: DVec3| {
            scene
                .add_object(ObjectDesc {
                    bounds,
                    pose: GridPose {
                        rotation: cp_core::math::DQuat::IDENTITY,
                        translation,
                    },
                })
                .map_err(|error| Failure::new(format!("structures scene: {error}")))
        };
        // The ground: the chunk below y = 0 of every column of a LOD plan around the rows.
        let rows_z = ROW_SPACING * (LEVELS.len() - 1) as f64 + f64::from(FIELD_EXTENT[1]);
        let plan = LodParams {
            center: DVec3::new(f64::from(FIELD_EXTENT[0]) * 0.5, 0.0, rows_z * 0.5),
            level0_radius: 320.0,
            growth: 2.0,
            max_radius: 6144.0,
            top_level: 6,
            edge_walls: false,
        };
        let ground_keys: Vec<LodKey> = plan_columns(&plan, |_| (-1.0, 1.0))
            .iter()
            .map(|column| LodKey::new(column.key.level, IVec3::new(column.key.x, -1, column.key.z)))
            .collect();
        let ground_bounds = union(&ground_keys);
        let ground = add(scene, ground_bounds, DVec3::ZERO)?;
        let mut requests: Vec<ChunkRequest> = ground_keys
            .iter()
            .map(|&key| ChunkRequest {
                key,
                walls: WallMask::NONE,
                object: ground,
                distance: 0.0,
            })
            .collect();
        let mut rows = Vec::new();
        for (row, &level) in LEVELS.iter().enumerate() {
            let extent = CHUNK_SIZE << level;
            let keys: Vec<LodKey> = (0..=(FIELD_EXTENT[0] - 1) / extent)
                .flat_map(|x| {
                    (0..=(FIELD_EXTENT[1] - 1) / extent)
                        .map(move |z| LodKey::new(level, IVec3::new(x, 0, z)))
                })
                .collect();
            let object = add(
                scene,
                union(&keys),
                DVec3::new(0.0, 0.0, ROW_SPACING * row as f64),
            )?;
            requests.extend(keys.iter().map(|&key| ChunkRequest {
                key,
                walls: WallMask::NONE,
                object,
                distance: 1.0 + row as f64,
            }));
            rows.push(object);
        }
        Ok((StructureObjects { ground, rows }, requests))
    }

    /// Camera `index`: 0 an oblique overview of all rows from their level-0 end, so that the
    /// near shapes show their textures and shadows and the coarser rows recede behind them; 1
    /// rows 0 and 1, 2 rows 2 and 3.
    /// The sun stands behind the cameras and to their right, so the shapes are lit from the
    /// front and cast their shadows away from the eye (from the camera's side, the scene was
    /// backlit and lost its contrast).
    pub fn camera(index: usize) -> Option<SceneCamera> {
        // Looking south-west from the north-east, down at the rows.
        let make = |name, position: DVec3, target: DVec3, fov_deg: f64| {
            let view = target - position;
            let yaw = cp_core::det::atan2(view.x, -view.z);
            let pitch = cp_core::det::atan2(view.y, (view.x * view.x + view.z * view.z).sqrt());
            SceneCamera {
                name,
                position,
                yaw,
                pitch,
                fov_y: fov_deg.to_radians(),
                sun: Sun {
                    elevation_deg: 40.0,
                    azimuth_deg: 330.0,
                },
            }
        };
        let middle = |rows: f64| ROW_SPACING * rows + 30.0;
        match index {
            0 => Some(make(
                "overview",
                DVec3::new(190.0, 75.0, -100.0),
                DVec3::new(70.0, 3.0, middle(1.5)),
                40.0,
            )),
            1 => Some(make(
                "levels_0_1",
                DVec3::new(150.0, 75.0, -70.0),
                DVec3::new(70.0, 3.0, middle(0.5)),
                40.0,
            )),
            2 => Some(make(
                "levels_2_3",
                DVec3::new(150.0, 85.0, ROW_SPACING * 2.0 - 70.0),
                DVec3::new(70.0, 5.0, middle(2.5)),
                40.0,
            )),
            _ => None,
        }
    }

    /// Where row `row`'s label belongs in the world: west of its first shape, at eye level.
    pub fn label_anchor(row: usize) -> DVec3 {
        DVec3::new(-6.0, 14.0, ROW_SPACING * row as f64 + 8.0)
    }
}

/// The voxel box of a set of chunks.
fn union(keys: &[LodKey]) -> IAabb {
    keys.iter()
        .map(LodKey::aabb_voxels)
        .reduce(|a, b| a.union(&b))
        .unwrap_or_else(|| LodKey::lod0(IVec3::ZERO).aabb_voxels())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rows_do_not_overlap_the_field_of_level_zero() {
        // The shapes of the unmoved field lie within z < 64; the moved rows start at 72.
        assert!(ROW_SPACING >= f64::from(FIELD_EXTENT[1]));
        for index in 0..CAMERA_COUNT {
            let camera = StructureScene::camera(index).expect("a camera");
            assert!(camera.pitch < 0.0, "{index} looks down");
        }
        assert!(StructureScene::camera(CAMERA_COUNT).is_none());
    }
}
