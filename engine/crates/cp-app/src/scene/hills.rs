//! The hills scene: cp-worldgen's [`HillsSource`] seen from eight cameras that a deterministic
//! search places in the landscape, plus a flight path over it.
//!
//! | camera | what it shows |
//! |---|---|
//! | 0 `valley` | eye height on a hill, looking across a valley with water towards a ridge, with the evening sun beside the view |
//! | 1 `altitude` | 120 m above the ground behind camera 0, looking at the horizon |
//! | 2 `shore` | close-up of a slope with trees, grass, sand and the water's edge, preferably with the evening sun beside the view |
//! | 3 `ridge` | on the crest of a pronounced ridge below the snow, looking along the ridge line into an open view, preferably to snowy heights |
//! | 4 `forest` | inside the densest forest near camera 0, the nearest trunk at a third of the frame |
//! | 5 `coast` | on a beach, looking out over the widest water with the shoreline in view |
//! | 6 `vista_high` | eye height on high ground at least 2 km from camera 0, a deep, far view, preferably to snowy heights |
//! | 7 `forest_edge` | eye height at least 2 km from cameras 0 and 6: an open meadow before the edge of a wood, no lake |
//!
//! Cameras 6 and 7 exist to check that distant places do not look the same; they are chosen
//! by what they see, with different preferences, from a lattice of candidate places.
//!
//! The landscape is cp-worldgen's hills with taller, snow-capped ridges ([`landscape`]). The
//! search runs on a height grid of level 4 of it (24 m spacing, the octaves shorter than 64 m
//! left out), sampled once on the job system's threads. Sight lines march over that grid; the
//! final eye heights come from level 0. Everything is a pure function of the seed, so every
//! run places the cameras identically.
//!
//! Each camera's world is generated around it ([`HillsScene::plan`]): level 0 near the camera
//! and coarser rings out to several kilometres ([`super::lod`]).

use std::collections::BTreeMap;
use std::sync::{Arc, OnceLock};

use cp_core::coords::LodKey;
use cp_core::det;
use cp_core::math::{DVec2, DVec3, IAabb, IVec3};
use cp_jobs::JobSystem;
use cp_render::voxel::{GroundMap, ObjectId};
use cp_voxel::{BaseSource, VoxelSnapshot, VoxelWorld, WorldConfig};
use cp_worldgen::{
    FeatureKind, HillsConfig, HillsSource, Species, TREE_CANOPY_MAX_LEVEL, TREE_GRID,
    TREE_MAX_HEIGHT, TreeInfo,
};

use super::SceneCamera;
use super::build::ChunkRequest;
use super::lod::{LodParams, PlannedChunk, plan_chunks, plan_columns};
use super::sky::Sun;
use crate::Failure;
use crate::assets::Assets;

/// World seed of the hills scene.
pub const HILLS_SEED: u64 = 0x00C0_FFEE_2026;

/// The landscape of the hills scene: cp-worldgen's defaults with taller ridges and a higher
/// snow line, so that the ridge ranges rise well above the rolling hills (up to about 130 m)
/// and carry snow, while the lowland keeps its lakes, beaches and forests.
pub fn landscape() -> HillsConfig {
    HillsConfig {
        hill_amplitude: 26.0,
        ridge_height: 96.0,
        snow_line: 64.0,
        ..HillsConfig::default()
    }
}

/// Ground at or above this height (m) counts as high ground for the camera search: the upper
/// slopes of the ridges, below the snow line of [`landscape`].
const HIGH_GROUND: f64 = 46.0;

/// Number of fixed cameras.
pub const CAMERA_COUNT: usize = 8;

/// Vertical field of view of the flight path (60 degrees, the contract's reference view).
pub const FOV_Y: f64 = std::f64::consts::FRAC_PI_3;

/// Vertical field of view of the eye-level landscape views (degrees).
const VISTA_FOV_DEG: f64 = 50.0;
/// Vertical field of view of the forest camera (degrees).
const FOREST_FOV_DEG: f64 = 65.0;

/// Eye height above the ground for cameras at eye level (contract: 1.7 m).
pub const EYE_HEIGHT: f64 = 1.7;

/// Sun azimuth of a camera's default sun relative to its view: behind the camera and well to
/// its right, so that faces towards the camera are lit, the two visible sides of a block
/// differ, and slopes facing away from the sun read darker.
const SUN_BEHIND_DEG: f64 = 120.0;

/// Compass azimuth (degrees) of the low sun of the evening shots. The valley and shore cameras
/// prefer to look [`EVENING_SIDE_DEG`] beside it ([`evening_side`]): the sun stands just
/// outside the frame, its glow and the warm band of the horizon below it fill the outer third
/// of the sky, crowns get a rim of light and long shadows run diagonally towards the eye, while
/// the land is lit from the side (warm faces towards the sun, cool ones away from it) instead
/// of from behind.
const EVENING_SUN_DEG: f64 = 250.0;
const EVENING_SIDE_DEG: std::ops::RangeInclusive<f64> = 35.0..=55.0;

/// How well a view towards the compass direction `az` (radians) sees the evening sun from the
/// side, as a weight: 1 when the sun stands [`EVENING_SIDE_DEG`] off the middle of the view
/// (either side), falling to 0 within 15 degrees more or less.
fn evening_side(az: f64) -> f64 {
    let off = ((az.to_degrees() - EVENING_SUN_DEG + 540.0).rem_euclid(360.0) - 180.0).abs();
    let (near, far) = (*EVENING_SIDE_DEG.start(), *EVENING_SIDE_DEG.end());
    smooth(near - 15.0, near, off) * (1.0 - smooth(far, far + 15.0, off))
}

/// Where the horizon of an eye-level landscape view sits, as a share of the frame from the
/// top: the land takes the larger part, the sky still a good third.
const HORIZON_FROM_TOP: f64 = 0.4;

/// How far (m) a tree that anchors the foreground of a landscape view stands from the eye: near
/// enough to give the view depth, far enough that its crown fits the frame (at 18 m a crown of
/// 9 m reaches 21 degrees above the eye) and does not hide the view.
const ANCHOR_DISTANCE: std::ops::RangeInclusive<f64> = 18.0..=40.0;

/// A slope that rises more steeply than this (degrees above the eye's horizon) within 300 m
/// of a landscape camera fills its frame like a wall ([`Search::wall_ahead`]).
const WALL_ANGLE_DEG: f64 = 7.0;

/// A landscape view is free when [`Search::trees_in_view`] stays below this: less than a tree
/// 20 m ahead beside the middle of the view, or three at 60 m. The clumped woods by the lakes
/// leave a few trees in most views from their slopes, which give the view depth as long as
/// they are few.
const FREE_VIEW_TREES: f64 = 2.0;
/// The same bound for the valley camera, which looks from a wooded slope down to a lake: the
/// young trees on the slope below the eye frame its foreground.
const VALLEY_TREES: f64 = 5.0;

/// The pitch (radians) that puts the horizon `from_top` of the way down a frame of the
/// vertical field of view `fov_deg`.
fn horizon_pitch(fov_deg: f64, from_top: f64) -> f64 {
    -det::atan((1.0 - 2.0 * from_top) * det::tan(0.5 * fov_deg.to_radians()))
}

/// Spacing (m) and level of the search grid.
const GRID_STEP: f64 = 24.0;
const GRID_LEVEL: u8 = 4;
/// The grid covers `±GRID_HALF` m around the origin.
const GRID_HALF: f64 = 6144.0;

/// How far around a camera its world reaches.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Reach {
    /// Level 0 within this distance (m).
    pub level0_radius: f64,
    /// Nothing beyond this distance (m).
    pub max_radius: f64,
    /// Coarsest level.
    pub top_level: u8,
}

impl Reach {
    /// The distances (m) over which the world fades completely into the haze, so that its
    /// edge never shows: from 72 to 95 percent of its radius.
    pub fn horizon_fade(&self) -> [f32; 2] {
        [
            (0.72 * self.max_radius) as f32,
            (0.95 * self.max_radius) as f32,
        ]
    }

    /// The world of a shot: level 0 to 512 m, coarser rings to 8 km.
    pub const SHOT: Self = Self {
        level0_radius: 512.0,
        max_radius: 8192.0,
        top_level: 6,
    };
    /// The world of the high camera: more rings, to 16 km.
    pub const HIGH: Self = Self {
        level0_radius: 512.0,
        max_radius: 16_384.0,
        top_level: 7,
    };
    /// The world of `cpe run`: level 0 to 1.15 km (at least 4,000 non-empty level-0 chunks
    /// with a margin, contract §9 E2; camera 0 stands by a lake, whose water fills fewer
    /// chunks than hills), coarser rings to 6 km.
    pub const RUN: Self = Self {
        level0_radius: 1152.0,
        max_radius: 6144.0,
        top_level: 6,
    };
}

/// Texels per side of the ground map ([`HillsScene::ground_map`]): 16 m texels over the
/// world of a shot, 32 m over the high camera's.
pub const GROUND_MAP_SIZE: u32 = 1024;
/// Level of the height field the ground map reads (8 m cells: smooth enough for 16 m texels).
const GROUND_MAP_LEVEL: u8 = 3;
/// Ground this far below the snow line (m) starts to look alpine, fully from this far.
const ALPINE_FROM: f64 = 30.0;
const ALPINE_FULL: f64 = 8.0;

/// The centre of the 64 m square that contains `value` (m): where the plan of a camera's
/// world and its ground map are centred.
fn snap_to_plan(value: f64) -> f64 {
    (value / 64.0).floor() * 64.0 + 32.0
}

/// The landscape on a regular grid, for sight lines.
#[derive(Clone, Debug)]
struct HeightGrid {
    min: DVec2,
    step: f64,
    side: usize,
    heights: Vec<f32>,
}

impl HeightGrid {
    fn build(jobs: &JobSystem, source: &HillsSource) -> Self {
        let side = (2.0 * GRID_HALF / GRID_STEP) as usize + 1;
        let min = DVec2::splat(-GRID_HALF);
        let rows: Vec<OnceLock<Vec<f32>>> = (0..side).map(|_| OnceLock::new()).collect();
        jobs.par_for(side, 4, |row| {
            let z = min.y + row as f64 * GRID_STEP;
            let values = (0..side)
                .map(|column| {
                    let x = min.x + column as f64 * GRID_STEP;
                    source.height(GRID_LEVEL, x, z) as f32
                })
                .collect();
            // Each row is computed exactly once.
            let _ = rows[row].set(values);
        });
        let heights = rows
            .into_iter()
            .flat_map(|row| row.into_inner().expect("par_for computes every row"))
            .collect();
        Self {
            min,
            step: GRID_STEP,
            side,
            heights,
        }
    }

    /// Bilinear height at `(x, z)`, clamped to the grid.
    fn at(&self, x: f64, z: f64) -> f64 {
        let last = (self.side - 1) as f64;
        let gx = det::clamp((x - self.min.x) / self.step, 0.0, last);
        let gz = det::clamp((z - self.min.y) / self.step, 0.0, last);
        let (ix, iz) = (
            (gx as usize).min(self.side - 2),
            (gz as usize).min(self.side - 2),
        );
        let (fx, fz) = (gx - ix as f64, gz - iz as f64);
        let h = |i: usize, j: usize| f64::from(self.heights[j * self.side + i]);
        let top = h(ix, iz) * (1.0 - fx) + h(ix + 1, iz) * fx;
        let bottom = h(ix, iz + 1) * (1.0 - fx) + h(ix + 1, iz + 1) * fx;
        top * (1.0 - fz) + bottom * fz
    }

    /// True when `(x, z)` is higher than or as high as the grid at the eight points `spacing`
    /// metres around it.
    fn is_local_max(&self, x: f64, z: f64, spacing: f64) -> bool {
        let h = self.at(x, z);
        (-1..=1).all(|i| {
            (-1..=1).all(|j| {
                (i == 0 && j == 0)
                    || self.at(x + f64::from(i) * spacing, z + f64::from(j) * spacing) <= h
            })
        })
    }
}

/// The features (trees, bushes, boulders) of the search area, placed once on the job system's
/// threads and bucketed by 64 m squares: the search asks for the trees near thousands of
/// places, and placing a tree evaluates the settled ground under it.
#[derive(Clone, Debug)]
struct TreeIndex {
    min: DVec2,
    side: usize,
    /// Per bucket (z rows, x fastest) its features in the order of `HillsSource::trees_in`.
    buckets: Vec<Vec<TreeInfo>>,
}

impl TreeIndex {
    /// Edge of a bucket (m), a multiple of the placement grid.
    const BUCKET: f64 = 64.0;

    fn build(jobs: &JobSystem, source: &HillsSource, half: f64) -> Self {
        let side = (2.0 * half / Self::BUCKET) as usize;
        let min = DVec2::splat(-half);
        let rows: Vec<OnceLock<Vec<Vec<TreeInfo>>>> = (0..side).map(|_| OnceLock::new()).collect();
        jobs.par_for(side, 1, |row| {
            let z = (min.y + row as f64 * Self::BUCKET) as i32;
            let band = IAabb::new(
                IVec3::new(min.x as i32, 0, z),
                IVec3::new((min.x + 2.0 * half) as i32, 1, z + Self::BUCKET as i32),
            );
            let mut buckets = vec![Vec::new(); side];
            for tree in source.trees_in(band) {
                let column = ((f64::from(tree.base.x) - min.x) / Self::BUCKET) as usize;
                buckets[column.min(side - 1)].push(tree);
            }
            // Each row is computed exactly once.
            let _ = rows[row].set(buckets);
        });
        let buckets = rows
            .into_iter()
            .flat_map(|row| row.into_inner().expect("par_for computes every row"))
            .collect();
        Self { min, side, buckets }
    }

    /// The features whose trunk column lies in `[x0, x1) × [z0, z1)` (m), in the order of
    /// `HillsSource::trees_in` (grid order: z, then x); only those inside the indexed area.
    fn trees_in(&self, [x0, x1]: [f64; 2], [z0, z1]: [f64; 2]) -> Vec<TreeInfo> {
        let bucket = |value: f64, origin: f64| {
            det::clamp(
                ((value - origin) / Self::BUCKET).floor(),
                0.0,
                (self.side - 1) as f64,
            ) as usize
        };
        let mut found: Vec<TreeInfo> = Vec::new();
        for row in bucket(z0, self.min.y)..=bucket(z1, self.min.y) {
            for column in bucket(x0, self.min.x)..=bucket(x1, self.min.x) {
                found.extend(
                    self.buckets[row * self.side + column]
                        .iter()
                        .filter(|tree| {
                            let (x, z) = (f64::from(tree.base.x), f64::from(tree.base.z));
                            x >= x0 && x < x1 && z >= z0 && z < z1
                        }),
                );
            }
        }
        // One feature per grid cell: sorting by the grid cell restores the grid order.
        found.sort_by_key(|tree| {
            (
                tree.base.z.div_euclid(TREE_GRID),
                tree.base.x.div_euclid(TREE_GRID),
            )
        });
        found
    }
}

/// One sample of a sight line.
#[derive(Clone, Copy, Debug)]
struct Sample {
    distance: f64,
    height: f64,
    visible: bool,
}

/// Walks from `eye` along the compass direction `azimuth` (radians) over the grid, from `start`
/// to `end` metres in steps of `step`, and tells for each sample whether the surface (water at
/// the sea level) can be seen from the eye.
fn sight_line(
    grid: &HeightGrid,
    sea: f64,
    eye: DVec3,
    azimuth: f64,
    [start, end, step]: [f64; 3],
    mut visit: impl FnMut(Sample),
) {
    let (sin, cos) = det::sin_cos(azimuth);
    let mut horizon = f64::NEG_INFINITY;
    let mut distance = start;
    while distance <= end {
        let height = grid.at(eye.x + sin * distance, eye.z - cos * distance);
        let angle = (det::max(height, sea) - eye.y) / distance;
        let visible = angle >= horizon;
        horizon = det::max(horizon, angle);
        visit(Sample {
            distance,
            height,
            visible,
        });
        distance += step;
    }
}

/// `smoothstep(edge0, edge1, x)` in `f64`.
fn smooth(edge0: f64, edge1: f64, x: f64) -> f64 {
    let t = det::clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// The compass azimuth `index` of `count` around the circle, radians.
fn azimuth(index: usize, count: usize) -> f64 {
    std::f64::consts::TAU * index as f64 / count as f64
}

/// Lattice points `spacing` apart within `half` metres of `center`, row by row.
fn lattice(center: DVec2, half: f64, spacing: f64) -> impl Iterator<Item = DVec2> {
    let steps = (half / spacing) as i32;
    (-steps..=steps).flat_map(move |j| {
        (-steps..=steps)
            .map(move |i| center + DVec2::new(f64::from(i) * spacing, f64::from(j) * spacing))
    })
}

/// The best-scoring item; on equal scores the first one (the iteration order is fixed).
fn best<T>(items: impl Iterator<Item = (f64, T)>) -> Option<(f64, T)> {
    items.fold(None, |best: Option<(f64, T)>, (score, item)| match best {
        Some((top, _)) if top >= score => best,
        _ => Some((score, item)),
    })
}

/// The items from the best score down; equal scores keep their order, so the first item is
/// the one [`best`] picks.
fn ranked<T>(items: impl Iterator<Item = (f64, T)>) -> Vec<T> {
    let mut items: Vec<(f64, T)> = items.collect();
    items.sort_by(|a, b| b.0.total_cmp(&a.0));
    items.into_iter().map(|(_, item)| item).collect()
}

/// The hills landscape and its camera script.
pub struct HillsScene {
    source: Arc<HillsSource>,
    _world: VoxelWorld,
    snapshot: Arc<VoxelSnapshot>,
    cameras: Vec<SceneCamera>,
    flight_center: DVec2,
}

impl std::fmt::Debug for HillsScene {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HillsScene")
            .field("identity", self.source.identity())
            .field("cameras", &self.cameras)
            .finish_non_exhaustive()
    }
}

impl HillsScene {
    /// The landscape of [`HILLS_SEED`] with the base pack's states, and its cameras.
    ///
    /// # Errors
    /// A [`Failure`] when the palette or the generator cannot be made, or the search finds no
    /// place for a camera.
    pub fn new(assets: &Assets, jobs: &JobSystem) -> Result<Self, Failure> {
        let palette = assets.palette()?;
        let source = HillsSource::new(HILLS_SEED, landscape(), palette, &assets.tables().flags)
            .map_err(|error| Failure::new(format!("hills generator: {error}")))?;
        let source = Arc::new(source);
        let base: Arc<dyn BaseSource> = source.clone();
        let world = VoxelWorld::new(base, assets.voxel_tables(), WorldConfig::default());
        let snapshot = Arc::new(world.snapshot());
        let grid = HeightGrid::build(jobs, &source);
        let trees = TreeIndex::build(jobs, &source, GRID_HALF);
        let cameras = Search {
            source: &source,
            grid: &grid,
            trees: &trees,
            sea: f64::from(source.config().sea_level),
        }
        .cameras()?;
        for (index, camera) in cameras.iter().enumerate() {
            log::info!(
                "hills camera {index} `{}`: eye ({:.1}, {:.1}, {:.1}), yaw {:.1} deg, pitch {:.1} \
                 deg",
                camera.name,
                camera.position.x,
                camera.position.y,
                camera.position.z,
                camera.yaw.to_degrees(),
                camera.pitch.to_degrees()
            );
        }
        let flight_center = DVec2::new(cameras[0].position.x, cameras[0].position.z);
        Ok(Self {
            source,
            _world: world,
            snapshot,
            cameras,
            flight_center,
        })
    }

    /// The generator.
    pub fn source(&self) -> &Arc<HillsSource> {
        &self.source
    }

    /// The snapshot the chunk jobs read.
    pub fn snapshot(&self) -> &Arc<VoxelSnapshot> {
        &self.snapshot
    }

    /// The fixed cameras.
    pub fn cameras(&self) -> &[SceneCamera] {
        &self.cameras
    }

    /// The reach of the world around camera `index`.
    pub fn reach(index: usize) -> Reach {
        if index == 1 { Reach::HIGH } else { Reach::SHOT }
    }

    /// Ground height under `(x, z)` at level 0 (top of the highest ground voxel), metres; the
    /// water surface where the ground lies below it.
    pub fn ground(&self, x: f64, z: f64) -> f64 {
        let top = self.source.ground_top(x.floor() as i32, z.floor() as i32);
        det::max(
            f64::from(top + 1),
            f64::from(self.source.config().sea_level),
        )
    }

    /// The flight path: a loop of about 420 m radius around camera 0, about 38 m above the
    /// smoothed ground, period 150 s, looking ahead and a little down. `t` in seconds.
    pub fn flight(&self, t: f64) -> SceneCamera {
        const PERIOD: f64 = 150.0;
        const RADIUS: f64 = 420.0;
        let point = |time: f64| {
            let angle = std::f64::consts::TAU * time / PERIOD;
            let radius = RADIUS * (1.0 + 0.18 * det::sin(3.0 * angle));
            let (sin, cos) = det::sin_cos(angle);
            let x = self.flight_center.x + radius * sin;
            let z = self.flight_center.y - radius * cos;
            let sea = f64::from(self.source.config().sea_level);
            let smooth = det::max(self.source.height(5, x, z), sea);
            DVec3::new(x, smooth + 38.0, z)
        };
        let position = point(t);
        let ahead = point(t + 0.5) - point(t - 0.5);
        let yaw = det::atan2(ahead.x, -ahead.z);
        let climb = det::atan2(ahead.y, DVec2::new(ahead.x, ahead.z).length());
        SceneCamera {
            name: "flight",
            position,
            yaw,
            pitch: climb - 0.16,
            fov_y: FOV_Y,
            sun: Sun::default_from((yaw.to_degrees() + SUN_BEHIND_DEG).rem_euclid(360.0)),
        }
    }

    /// The height range `(low, high)` in metres that chunks of `level` must cover: from below
    /// the lowest possible surface (deeper chunks are buried stone that shows nothing) to above
    /// the highest ground, tree and water cell.
    pub fn vertical(&self, level: u8) -> (f64, f64) {
        let (low, high) = self.source.height_bounds(level);
        let cell = f64::from(1_u32 << level);
        let sea = f64::from(self.source.config().sea_level);
        let trees = if level <= TREE_CANOPY_MAX_LEVEL {
            f64::from(TREE_MAX_HEIGHT) + 2.0 * cell
        } else {
            0.0
        };
        (
            det::min(low, sea) - 2.0 * cell - 4.0,
            det::max(high, sea) + trees + cell,
        )
    }

    /// The ground map of the world around `eye` (cp-render's [`GroundMap`]): a square of
    /// [`GROUND_MAP_SIZE`] texels over the reach of the world, centred like [`Self::plan`],
    /// holding at each texel centre the dryness of the region, the wood density of the trees
    /// and how close the ground comes to the snow line. Made on the job system's threads.
    pub fn ground_map(&self, jobs: &JobSystem, eye: DVec3, reach: Reach) -> GroundMap {
        let size = GROUND_MAP_SIZE;
        let texel = 2.0 * reach.max_radius / f64::from(size);
        let centre = DVec2::new(snap_to_plan(eye.x), snap_to_plan(eye.z));
        let origin = centre - DVec2::splat(reach.max_radius);
        let snow = self.source.snow_line();
        let rows: Vec<OnceLock<Vec<[u8; 4]>>> = (0..size).map(|_| OnceLock::new()).collect();
        jobs.par_for(size as usize, 4, |row| {
            let z = origin.y + (row as f64 + 0.5) * texel;
            let texels = (0..size)
                .map(|column| {
                    let x = origin.x + (f64::from(column) + 0.5) * texel;
                    let byte = |value: f64| (det::clamp(value, 0.0, 1.0) * 255.0).round() as u8;
                    let dryness = self.source.dryness(x, z);
                    let height = self.source.height(GROUND_MAP_LEVEL, x, z);
                    let alpine = smooth(snow - ALPINE_FROM, snow - ALPINE_FULL, height);
                    [
                        byte(0.5 + 0.5 * dryness),
                        byte(self.source.wood(x, z)),
                        byte(alpine),
                        255,
                    ]
                })
                .collect();
            // Each row is computed exactly once.
            let _ = rows[row].set(texels);
        });
        GroundMap {
            origin: [origin.x, origin.y],
            texel_metres: texel,
            size,
            texels: rows
                .into_iter()
                .flat_map(|row| row.into_inner().expect("par_for computes every row"))
                .collect(),
        }
    }

    /// The chunks of the world around `eye` and the voxel bounds of the object that holds
    /// them. The plan is centred on `eye` moved to the centre of its 64 m square, so that
    /// cameras a frame apart (motion pairs) usually share it.
    pub fn plan(&self, eye: DVec3, reach: Reach) -> (IAabb, Vec<PlannedChunk>) {
        let snap = snap_to_plan;
        let params = LodParams {
            center: DVec3::new(snap(eye.x), eye.y, snap(eye.z)),
            level0_radius: reach.level0_radius,
            growth: 2.0,
            max_radius: reach.max_radius,
            top_level: reach.top_level,
            edge_walls: false,
        };
        let vertical = |level: u8| self.vertical(level);
        let columns = plan_columns(&params, vertical);
        let chunks = plan_chunks(&columns, vertical);
        let bounds = chunks
            .iter()
            .map(|chunk| chunk.key.aabb_voxels())
            .reduce(|a, b| a.union(&b))
            .unwrap_or_else(|| LodKey::lod0(IVec3::ZERO).aabb_voxels());
        (bounds, chunks)
    }

    /// The chunk requests of a plan for `object`.
    pub fn requests(planned: &[PlannedChunk], object: ObjectId) -> Vec<ChunkRequest> {
        planned
            .iter()
            .map(|chunk| ChunkRequest {
                key: chunk.key,
                walls: chunk.walls,
                object,
                distance: chunk.distance,
            })
            .collect()
    }
}

/// The camera search over one landscape.
struct Search<'a> {
    source: &'a HillsSource,
    trees: &'a TreeIndex,
    grid: &'a HeightGrid,
    sea: f64,
}

/// A feature (tree, bush, boulder) near a camera place, as the search sees it.
#[derive(Clone, Copy, Debug)]
struct Feature {
    /// Centre of the trunk column (x, z), m.
    trunk: DVec2,
    /// Height of its top (m), about: the base, the trunk and half the crown's width above it.
    top: f64,
    kind: FeatureKind,
}

/// What [`Search::valley`] found: the eye and where it looks.
struct Valley {
    eye: DVec3,
    yaw: f64,
    pitch: f64,
}

/// What a sight line sees ([`Search::view`]).
#[derive(Clone, Copy, Debug, Default)]
struct View {
    /// How far the ground falls away in front of the eye (m, from the drops at 35 and 70 m):
    /// a view from the edge of a slope instead of across a flat hill top.
    drop: f64,
    /// The ground within 90 m rises to the eye: the view is blocked.
    blocked: bool,
    /// Metres of visible water within 1.5 km.
    water: f64,
    /// Distance of the first visible water.
    first_water: Option<f64>,
    /// Metres of visible high ground beyond 300 m.
    high: f64,
    /// The highest visible point beyond 400 m: height and distance.
    ridge: Option<(f64, f64)>,
    /// The farthest visible sample.
    far: f64,
}

impl Search<'_> {
    fn cameras(&self) -> Result<Vec<SceneCamera>, Failure> {
        let missing = |what: &str| {
            Failure::new(format!(
                "the hills of seed {HILLS_SEED:#x} have no place for the {what} camera"
            ))
        };
        let valley = self.valley().ok_or_else(|| missing("valley"))?;
        // Landscape views at eye level use a longer lens than the close-ups, so that distant
        // hills and ridges keep their size in the frame.
        let camera = |name, position: DVec3, yaw: f64, pitch: f64, fov_deg: f64| SceneCamera {
            name,
            position,
            yaw,
            pitch,
            fov_y: fov_deg.to_radians(),
            sun: Sun::default_from((yaw.to_degrees() + SUN_BEHIND_DEG).rem_euclid(360.0)),
        };
        let c0 = camera(
            "valley",
            valley.eye,
            valley.yaw,
            valley.pitch,
            VISTA_FOV_DEG,
        );
        let c1 = self.altitude(&c0);
        let c1 = camera("altitude", c1.0, valley.yaw, c1.1, 60.0);
        let (eye, yaw, pitch) = self.shore().ok_or_else(|| missing("shore"))?;
        let c2 = camera("shore", eye, yaw, pitch, 60.0);
        let (eye, yaw, pitch) = self.ridge().ok_or_else(|| missing("ridge"))?;
        let c3 = camera("ridge", eye, yaw, pitch, VISTA_FOV_DEG);
        let (eye, yaw, pitch) = self
            .forest(DVec2::new(c0.position.x, c0.position.z))
            .ok_or_else(|| missing("forest"))?;
        let c4 = camera("forest", eye, yaw, pitch, FOREST_FOV_DEG);
        let (eye, yaw, pitch) = self.coast().ok_or_else(|| missing("coast"))?;
        let c5 = camera("coast", eye, yaw, pitch, VISTA_FOV_DEG);
        let p0 = DVec2::new(c0.position.x, c0.position.z);
        let p3 = DVec2::new(c3.position.x, c3.position.z);
        let (eye, yaw, pitch) = self
            .vista(&[p0], &[p3], VistaTaste::High)
            .ok_or_else(|| missing("high vista"))?;
        let c6 = camera("vista_high", eye, yaw, pitch, VISTA_FOV_DEG);
        let p6 = DVec2::new(c6.position.x, c6.position.z);
        let (eye, yaw, pitch) = self
            .forest_edge(&[p0, p6])
            .ok_or_else(|| missing("forest edge"))?;
        let c7 = camera("forest_edge", eye, yaw, pitch, VISTA_FOV_DEG);
        Ok(vec![c0, c1, c2, c3, c4, c5, c6, c7])
    }

    /// Ground at level 0 under `(x, z)` (top face of the highest ground voxel, at least the
    /// water surface).
    fn ground(&self, x: f64, z: f64) -> f64 {
        let top = self.source.ground_top(x.floor() as i32, z.floor() as i32);
        det::max(f64::from(top + 1), self.sea)
    }

    /// An eye `EYE_HEIGHT` above the level-0 ground at `(x, z)`, at the centre of its column.
    fn eye_at(&self, point: DVec2) -> DVec3 {
        let (x, z) = (point.x.floor() + 0.5, point.y.floor() + 0.5);
        DVec3::new(x, self.ground(x, z) + EYE_HEIGHT, z)
    }

    /// The eye ([`Self::eye_at`]) on the highest level-0 column within `reach` metres of
    /// `point` (sampled every 2 m; the first of equal heights).
    fn high_eye(&self, point: DVec2, reach: f64) -> DVec3 {
        best(lattice(point, reach, 2.0).map(|p| (self.ground(p.x, p.y), p)))
            .map_or_else(|| self.eye_at(point), |(_, top)| self.eye_at(top))
    }

    /// What can be seen from `eye` towards the compass direction `az` (radians), up to 3 km.
    fn view(&self, eye: DVec3, az: f64) -> View {
        let (sin, cos) = det::sin_cos(az);
        let ground = self.grid.at(eye.x, eye.z);
        let below = |distance: f64| {
            ground
                - det::max(
                    self.grid.at(eye.x + sin * distance, eye.z - cos * distance),
                    self.sea,
                )
        };
        let mut view = View {
            drop: det::min(below(70.0), 2.0 * below(35.0)).clamp(-20.0, 30.0),
            ..View::default()
        };
        sight_line(self.grid, self.sea, eye, az, [12.0, 3000.0, 12.0], |s| {
            if s.distance <= 90.0 && det::max(s.height, self.sea) > eye.y - 0.3 {
                view.blocked = true;
            }
            if !s.visible {
                return;
            }
            view.far = s.distance;
            if s.height < self.sea - 0.8 && s.distance <= 1500.0 {
                view.water += 12.0;
                view.first_water.get_or_insert(s.distance);
            }
            if s.height > HIGH_GROUND && s.distance >= 300.0 {
                view.high += 12.0;
            }
            if s.distance >= 400.0 && view.ridge.is_none_or(|(h, _)| s.height > h) {
                view.ridge = Some((s.height, s.distance));
            }
        });
        view
    }

    /// The brow of the hill at `top` towards `az`: the farthest point up to 120 m ahead to
    /// which the ground stays within 2.5 m of the top, where the view opens up.
    fn brow(&self, top: DVec2, az: f64) -> DVec2 {
        let (sin, cos) = det::sin_cos(az);
        let height = self.grid.at(top.x, top.y);
        let mut brow = top;
        for step in 1..=10 {
            let point = top + DVec2::new(sin, -cos) * (12.0 * f64::from(step));
            if self.grid.at(point.x, point.y) < height - 2.5 {
                break;
            }
            brow = point;
        }
        brow
    }

    /// Trunk positions of the trees within 170 m of `point` (the brow of a hill and 45 m
    /// around it).
    fn trunks_near(&self, point: DVec2) -> Vec<DVec2> {
        self.features_near(point)
            .into_iter()
            .map(|feature| feature.trunk)
            .collect()
    }

    /// The features within 170 m of `point`.
    fn features_near(&self, point: DVec2) -> Vec<Feature> {
        let reach = 170.0;
        self.trees
            .trees_in(
                [point.x - reach, point.x + reach],
                [point.y - reach, point.y + reach],
            )
            .iter()
            .map(|tree| Feature {
                trunk: DVec2::new(f64::from(tree.base.x) + 0.5, f64::from(tree.base.z) + 0.5),
                top: f64::from(tree.base.y + tree.trunk_height)
                    + 0.5 * f64::from(tree.crown_radius_halves),
                kind: tree.kind,
            })
            .collect()
    }

    /// Of `features`, the broadleaf trees (oak, birch, cherry): the trees that may anchor the
    /// foreground of a view ([`Self::anchors_in_view`]). A spruce near the eye is a dark wall of
    /// needles, and bushes and boulders are no trees.
    fn anchor_trees(features: &[Feature]) -> Vec<Feature> {
        features
            .iter()
            .filter(|feature| {
                matches!(feature.kind, FeatureKind::Tree(species) if species != Species::Spruce)
            })
            .copied()
            .collect()
    }

    /// Of `features`, the trunks of `kind`.
    fn trunks_of(features: &[Feature], kind: FeatureKind) -> Vec<DVec2> {
        features
            .iter()
            .filter(|feature| feature.kind == kind)
            .map(|feature| feature.trunk)
            .collect()
    }

    /// Trees that would stand in a view from `eye` towards `az`: trunks up to 120 m ahead
    /// inside the 40 degrees to each side that a landscape view shows, the near ones counting
    /// much more (a trunk at 5 m fills the frame, one at 100 m is a detail), and those in the
    /// middle of the view three times as much (a lone tree in the centre blocks the view; one
    /// at the side frames it). Trees whose crowns stay well below the eye (more than 4.6
    /// degrees, a tenth of their distance less a little) do not count: from a hill the eye looks
    /// over the woods below, which are the view and do not block it.
    fn trees_in_view(features: &[Feature], eye: DVec3, az: f64) -> f64 {
        let point = DVec2::new(eye.x, eye.z);
        let (sin, cos) = det::sin_cos(az);
        let look = DVec2::new(sin, -cos);
        features
            .iter()
            .filter_map(|feature| {
                let offset = feature.trunk - point;
                let distance = offset.length();
                let along = offset.dot(look);
                // A tree at the edge of the foreground (25 degrees or more off the middle) frames
                // the view instead of blocking it (see `anchors_in_view`).
                let anchor = ANCHOR_DISTANCE.contains(&distance) && along < 0.9 * distance;
                let below = eye.y - feature.top > 0.08 * distance;
                (distance <= 120.0 && along > 0.766 * distance && !anchor && !below).then(|| {
                    let centre = smooth(0.966, 0.995, along / det::max(distance, 1e-9));
                    40.0 / det::max(distance, 4.0) * (1.0 + 2.0 * centre)
                })
            })
            .sum()
    }

    /// Trees that anchor the foreground of a view from `eye` towards `az` with the vertical
    /// field of view `fov_deg` (16:9, the horizon [`HORIZON_FROM_TOP`] down the frame): trees
    /// [`ANCHOR_DISTANCE`] away whose trunk stands in the outer part of the frame (from 60 to
    /// 90 percent of the way out from the middle) and whose crown stays inside it (its top at
    /// least two degrees below the frame's top edge), where they frame the view instead of
    /// hiding it or being cut off; the nearer the more (a tree at 18 m counts 1). `trees` are
    /// the candidates ([`Self::anchor_trees`]).
    fn anchors_in_view(trees: &[Feature], eye: DVec3, az: f64, fov_deg: f64) -> f64 {
        let point = DVec2::new(eye.x, eye.z);
        let (sin, cos) = det::sin_cos(az);
        let look = DVec2::new(sin, -cos);
        let right = DVec2::new(cos, sin);
        let half_width = det::tan(0.5 * fov_deg.to_radians()) * 16.0 / 9.0;
        let frame_top = 0.5 * fov_deg.to_radians() + horizon_pitch(fov_deg, HORIZON_FROM_TOP);
        let rise_limit = det::tan(frame_top - 2.0_f64.to_radians());
        trees
            .iter()
            .filter_map(|tree| {
                let offset = tree.trunk - point;
                let (along, across) = (offset.dot(look), offset.dot(right));
                let distance = offset.length();
                if !ANCHOR_DISTANCE.contains(&distance) || along <= 0.0 {
                    return None;
                }
                // Horizontal position in the frame, -1 left to 1 right.
                let x = across / (along * half_width);
                let inside = (tree.top - eye.y) / along < rise_limit;
                ((0.6..=0.9).contains(&x.abs()) && inside)
                    .then(|| ANCHOR_DISTANCE.start() / distance)
            })
            .sum()
    }

    /// Places on the far side of a lake from a ridge: for every ridge top of the search area
    /// and every direction away from it, the slope that rises out of the first lake on that
    /// line, with the compass direction back towards the ridge. Hills that face a ridge across
    /// water are slopes more often than hill tops.
    fn across_lakes(&self) -> Vec<(DVec2, f64)> {
        let ridges: Vec<DVec2> = lattice(DVec2::ZERO, 3200.0, 64.0)
            .filter(|p| {
                self.grid.at(p.x, p.y) >= HIGH_GROUND + 10.0
                    && self.grid.is_local_max(p.x, p.y, 64.0)
            })
            .collect();
        let mut places = Vec::new();
        for ridge in ridges {
            for k in 0..48 {
                let away = azimuth(k, 48);
                let (sin, cos) = det::sin_cos(away);
                let step = DVec2::new(sin, -cos) * 20.0;
                // March out to the first lake, across it, and up the far shore.
                let mut distance = 0;
                let mut in_lake = false;
                let mut far_shore = None;
                while distance < 2400 / 20 {
                    distance += 1;
                    let point = ridge + step * f64::from(distance);
                    let h = self.grid.at(point.x, point.y);
                    if !in_lake && h < self.sea - 1.0 && distance * 20 >= 300 {
                        in_lake = true;
                    } else if in_lake && h > self.sea + 0.5 {
                        far_shore = Some(distance);
                        break;
                    }
                }
                let Some(shore) = far_shore else {
                    continue;
                };
                // Up the far shore to 10 to 45 m above the water, at most 600 m from it.
                let back = (away + std::f64::consts::PI).rem_euclid(std::f64::consts::TAU);
                for climb in 1..=30 {
                    let point = ridge + step * f64::from(shore + climb);
                    let h = self.grid.at(point.x, point.y);
                    if (self.sea + 10.0..=45.0).contains(&h) {
                        places.push((point, back));
                    }
                }
            }
        }
        places
    }

    /// Whether the level-0 ground in front of `eye` towards `az` is open, across the width of
    /// a 16:9 frame with the vertical field of view `fov_deg` (plus a margin): up to 24 m away
    /// it stays below the ground the eye stands on, and up to 16 m it never rises along a line
    /// of sight (a dip shows the riser on its far side). A block rising beside or before the
    /// camera fills a corner of the frame; the height grid is too coarse to see such steps.
    /// Towards the sides of the frame (the outer third of each half) a hummock may rise a
    /// metre higher from 8 m on: it frames the view there instead of blocking it.
    fn open_foreground(&self, eye: DVec3, az: f64, fov_deg: f64) -> bool {
        let floor = eye.y - EYE_HEIGHT + 0.5;
        let half = det::atan(det::tan(0.5 * fov_deg.to_radians()) * 16.0 / 9.0) + 0.1;
        (0..=36).all(|k| {
            let (sin, cos) = det::sin_cos(az - half + half * f64::from(k) / 18.0);
            let side = (k - 18_i32).abs() > 12;
            let mut lowest = floor;
            (1..=24).all(|metres| {
                let d = f64::from(metres);
                let height = self.ground(eye.x + sin * d, eye.z - cos * d);
                lowest = det::min(lowest, height);
                let limit = if side && metres >= 8 {
                    floor + 1.0
                } else {
                    floor
                };
                let rise = if side && metres >= 8 { 1.5 } else { 0.5 };
                height <= limit && (metres > 16 || height <= lowest + rise)
            })
        })
    }

    /// Whether the level-0 ground stays more than 1.2 m below the eye for `reach` metres in
    /// the middle of a view towards `az` (within 12 degrees to each side).
    fn clear_ahead(&self, eye: DVec3, az: f64, reach: f64) -> bool {
        let limit = eye.y - 1.2;
        (-3..=3).all(|k| {
            let (sin, cos) = det::sin_cos(az + f64::from(k) * 4.0_f64.to_radians());
            let mut d = 2.0;
            while d <= reach {
                if self.ground(eye.x + sin * d, eye.z - cos * d) > limit {
                    return false;
                }
                d += 1.0;
            }
            true
        })
    }

    /// Whether a landscape view from `eye` towards `az` (vertical field of view `fov_deg`) is
    /// free: no feature within 10 m and no spruce within 20 m inside 75 degrees of the view (a
    /// crown would hang into the frame; the cone of a spruce is a dark wall), few trees ahead
    /// ([`Self::trees_in_view`] below `crowd`, [`FREE_VIEW_TREES`] for most views), and an open
    /// foreground.
    fn free_view(&self, eye: DVec3, az: f64, fov_deg: f64, crowd: f64) -> bool {
        let point = DVec2::new(eye.x, eye.z);
        let features = self.features_near(point);
        let trunks: Vec<DVec2> = features.iter().map(|feature| feature.trunk).collect();
        let spruces = Self::trunks_of(&features, FeatureKind::Tree(Species::Spruce));
        let (sin, cos) = det::sin_cos(az);
        let look = DVec2::new(sin, -cos);
        let near = |trunks: &[DVec2], reach: f64| {
            trunks.iter().any(|trunk| {
                let offset = *trunk - point;
                let distance = offset.length();
                distance <= reach && offset.dot(look) > 0.26 * distance
            })
        };
        !near(&trunks, 10.0)
            && !near(&spruces, 20.0)
            && Self::trees_in_view(&features, eye, az) < crowd
            && self.open_foreground(eye, az, fov_deg)
    }

    /// Whether the ground 30 to 300 m in front of `eye` towards `az` (within 10 degrees to each
    /// side) rises more than [`WALL_ANGLE_DEG`] above the eye's horizon: a slope that would
    /// fill the frame like a wall.
    fn wall_ahead(&self, eye: DVec3, az: f64) -> bool {
        let limit = det::tan(WALL_ANGLE_DEG.to_radians());
        (-2..=2).any(|k| {
            let (sin, cos) = det::sin_cos(az + f64::from(k) * 5.0_f64.to_radians());
            let mut distance = 30.0;
            while distance <= 300.0 {
                let height = self.grid.at(eye.x + sin * distance, eye.z - cos * distance);
                if height - eye.y > limit * distance {
                    return true;
                }
                distance += 12.0;
            }
            false
        })
    }

    /// The first of the `limit` best `candidates` whose view from `eye(candidate)` is free (see
    /// [`Self::free_view`]), or the best one when none is.
    fn first_free<T: Copy>(
        &self,
        candidates: &[T],
        limit: usize,
        fov_deg: f64,
        crowd: f64,
        view: impl Fn(&T) -> (DVec3, f64),
    ) -> Option<T> {
        candidates
            .iter()
            .take(limit)
            .find(|candidate| {
                let (eye, az) = view(candidate);
                self.free_view(eye, az, fov_deg, crowd)
            })
            .or_else(|| candidates.first())
            .copied()
    }

    /// Camera 0: a hill well above a lake, the lake below in the near distance and higher
    /// ground behind it, with no tree in the middle of the view and an open foreground.
    /// Candidates: hill tops (each looking every way from its brow) and the slopes across a
    /// lake from a ridge.
    fn valley(&self) -> Option<Valley> {
        let tops = lattice(DVec2::ZERO, 2600.0, 48.0).filter(|p| {
            let h = self.grid.at(p.x, p.y);
            (self.sea + 10.0..=45.0).contains(&h) && self.grid.is_local_max(p.x, p.y, 48.0)
        });
        let from_tops = tops.flat_map(|top| {
            (0..72).map(move |k| {
                let az = azimuth(k, 72);
                (self.brow(top, az), az)
            })
        });
        let across = self.across_lakes().into_iter().flat_map(|(point, toward)| {
            // Around the direction to the ridge, so the lake and the ridge share the view.
            (-3..=3).map(move |turn| (point, toward + f64::from(turn) * 0.087))
        });
        let candidates = ranked(from_tops.chain(across).filter_map(|(point, az)| {
            let eye = DVec3::new(point.x, self.grid.at(point.x, point.y) + 2.0, point.y);
            let view = self.view(eye, az);
            let first = view.first_water?;
            let (ridge_height, ridge_distance) = view.ridge?;
            let rise = ridge_height - eye.y;
            if view.blocked
                || view.water < 150.0
                || !(60.0..=500.0).contains(&first)
                || rise < 4.0
                || eye.y < self.sea + 12.0
            {
                return None;
            }
            let features = self.features_near(point);
            let trees = Self::trees_in_view(&features, eye, az);
            let anchors =
                Self::anchors_in_view(&Self::anchor_trees(&features), eye, az, VISTA_FOV_DEG);
            // The ridge has to rise clearly above the eye across the water. The lake should lie
            // well below the eye (the angle down to its near shore, `dip`): the deeper the
            // near shore lies in the frame, the less of it is empty meadow between the eye and
            // the water, and a view from the brow of a slope down to it is better than one
            // across a flat hill top. A tree in a lower corner anchors the foreground. Above
            // all the view has the evening sun beside it, which lights its dusk shot (the
            // shots of the review fix the sun).
            let dip = det::atan2(eye.y - self.sea, first).to_degrees();
            let score = view.water.min(600.0)
                + 30.0 * rise.min(45.0)
                + 150.0 * dip.min(14.0)
                + 20.0 * view.drop.clamp(0.0, 25.0)
                - 60.0 * trees
                + 300.0 * anchors.min(2.0)
                + 2000.0 * evening_side(az);
            Some((score, (point, az, first, ridge_height, ridge_distance)))
        }));
        // The level-0 checks cost about a thousand ground lookups: from the top down only. The
        // eye stands on the highest column within 4 m of the candidate's (the grid does not see
        // the hummocks of the ground, and a hummock just ahead blocks the view).
        let (point, yaw, first, ridge_height, ridge_distance) = self.first_free(
            &candidates,
            400,
            VISTA_FOV_DEG,
            VALLEY_TREES,
            |&(point, az, ..)| (self.high_eye(point, 4.0), az),
        )?;
        let high = self.high_eye(point, 4.0);
        let point = self.anchored(DVec2::new(high.x, high.z), yaw, VISTA_FOV_DEG, VALLEY_TREES);
        let eye = self.eye_at(point);
        // The horizon 40 percent down the frame (the land below it is the subject); the top
        // of the ridge no higher than a quarter from the top, and the near shore of the lake
        // well inside the frame.
        let half = 0.5 * VISTA_FOV_DEG.to_radians();
        let to_water = det::atan2(self.sea - eye.y, first);
        let to_ridge = det::atan2(ridge_height - eye.y, ridge_distance);
        let pitch = det::max(
            horizon_pitch(VISTA_FOV_DEG, HORIZON_FROM_TOP),
            to_ridge - det::atan(0.5 * det::tan(half)),
        );
        let pitch = det::min(pitch, to_water + 0.7 * half);
        Some(Valley {
            eye,
            yaw,
            pitch: det::clamp(pitch, -0.25, 0.12),
        })
    }

    /// Camera 1: 120 m above the ground 150 m behind camera 0, looking the same way and a
    /// little down at the horizon.
    fn altitude(&self, valley: &SceneCamera) -> (DVec3, f64) {
        let (sin, cos) = det::sin_cos(valley.yaw);
        let x = valley.position.x - 150.0 * sin;
        let z = valley.position.z + 150.0 * cos;
        (DVec3::new(x, self.ground(x, z) + 120.0, z), -0.14)
    }

    /// Camera 2: over shallow water, looking at a shore whose sand rises to grass and trees.
    fn shore(&self) -> Option<(DVec3, f64, f64)> {
        let shore_points = lattice(DVec2::ZERO, 2000.0, 16.0).filter(|p| {
            let h = self.grid.at(p.x, p.y);
            (self.sea - 1.5..=self.sea - 0.2).contains(&h)
                && [(16.0, 0.0), (-16.0, 0.0), (0.0, 16.0), (0.0, -16.0)]
                    .iter()
                    .any(|&(dx, dz)| self.grid.at(p.x + dx, p.y + dz) >= self.sea + 0.6)
        });
        let found = best(shore_points.filter_map(|point| {
            let gradient = DVec2::new(
                self.grid.at(point.x + 16.0, point.y) - self.grid.at(point.x - 16.0, point.y),
                self.grid.at(point.x, point.y + 16.0) - self.grid.at(point.x, point.y - 16.0),
            );
            let land = gradient.try_normalize()?;
            let trees: Vec<TreeInfo> = self
                .trees
                .trees_in(
                    [point.x - 40.0, point.x + 40.0],
                    [point.y - 40.0, point.y + 40.0],
                )
                .into_iter()
                .filter(|tree| {
                    let offset = DVec2::new(
                        f64::from(tree.base.x) - point.x,
                        f64::from(tree.base.z) - point.y,
                    );
                    let distance = offset.length();
                    (10.0..=40.0).contains(&distance)
                        && offset.dot(land) > 0.5 * distance
                        && (2.0..=25.0).contains(&f64::from(tree.base.y))
                })
                .collect();
            if trees.len() < 2 {
                return None;
            }
            let rise = self
                .grid
                .at(point.x + land.x * 25.0, point.y + land.y * 25.0)
                - self.sea;
            // The view up the shore, preferably with the evening sun beside it.
            let side = evening_side(det::atan2(land.x, -land.y));
            let score =
                trees.len().min(10) as f64 * 10.0 + rise.clamp(0.0, 12.0) * 2.0 + 40.0 * side;
            Some((score, (point, land)))
        }))?;
        let (point, land) = found.1;
        // Twelve metres out over the water (less where the lake is narrow), 2.5 m above it.
        let mut out = 12.0;
        while out > 3.0 && self.grid.at(point.x - land.x * out, point.y - land.y * out) > self.sea {
            out -= 3.0;
        }
        let eye_xz = point - land * out;
        let eye = DVec3::new(eye_xz.x, self.sea + 2.5, eye_xz.y);
        let target_xz = point + land * 20.0;
        let target = DVec3::new(
            target_xz.x,
            self.ground(target_xz.x, target_xz.y) + 2.0,
            target_xz.y,
        );
        let view = target - eye;
        let yaw = det::atan2(view.x, -view.z);
        let pitch = det::atan2(view.y, DVec2::new(view.x, view.z).length());
        Some((eye, yaw, pitch))
    }

    /// Camera 3: on the crest of a pronounced ridge of the search area, looking along the ridge
    /// line: high ground continues ahead while the slopes fall away to both sides. Of the best
    /// crests, the view has to open up (no slope rises like a wall within 300 m,
    /// [`Self::wall_ahead`]; standing at the foot of a higher ridge, the eye saw nothing but its
    /// face) and is ranked again by what it shows: snow on the heights beyond 400 m and the
    /// ground falling away in front.
    fn ridge(&self) -> Option<(DVec3, f64, f64)> {
        let mean_along = |p: DVec2, az: f64, distances: &[f64]| {
            let (sin, cos) = det::sin_cos(az);
            distances
                .iter()
                .map(|&d| self.grid.at(p.x + sin * d, p.y - cos * d))
                .sum::<f64>()
                / distances.len() as f64
        };
        let ahead = [80.0, 160.0, 240.0, 320.0, 400.0];
        let side = [60.0, 120.0];
        // High, but below the snow: a crest of grass, scree and rock that looks along the
        // ridge to the snow and down to the woods (a crest in the snow is a white frame).
        let snow = self.source.snow_line();
        let crests = ranked(
            lattice(DVec2::ZERO, 3000.0, 32.0)
                .filter(|p| (HIGH_GROUND..snow - 6.0).contains(&self.grid.at(p.x, p.y)))
                .flat_map(|p| {
                    (0..36).map(move |k| {
                        let az = azimuth(k, 36);
                        let quarter = std::f64::consts::FRAC_PI_2;
                        let along = mean_along(p, az, &ahead);
                        let across = 0.5
                            * (mean_along(p, az + quarter, &side)
                                + mean_along(p, az - quarter, &side));
                        // A crest: the line ahead stays high, the sides fall away, and the eye
                        // stands near the line's height.
                        let here = self.grid.at(p.x, p.y);
                        let score = (along - across) - 0.5 * (here - along).abs() + 0.2 * here;
                        (score, (score, p, az))
                    })
                }),
        );
        let candidates = ranked(crests.iter().take(6000).filter_map(|&(crest, p, az)| {
            let eye = DVec3::new(p.x, self.grid.at(p.x, p.y) + EYE_HEIGHT, p.y);
            if self.wall_ahead(eye, az) {
                return None;
            }
            let view = self.view(eye, az);
            if view.blocked {
                return None;
            }
            let snowy = view.ridge.is_some_and(|(height, _)| height >= snow + 8.0);
            let score = crest + if snowy { 25.0 } else { 0.0 } + view.drop.clamp(0.0, 20.0);
            Some((score, (p, az)))
        }));
        // Without an open crest, the best crests as they are.
        let candidates = if candidates.is_empty() {
            crests.iter().map(|&(_, p, az)| (p, az)).collect()
        } else {
            candidates
        };
        // The eye on the highest level-0 column within 8 m of the grid's crest; the best crest
        // with a free view (ridges are often wooded), standing on grass where one does (scree
        // and rock fill the foreground with large, plain blocks).
        let top_eye = |crest: DVec2| self.high_eye(crest, 8.0);
        let grass = self.source.palette().grass;
        let on_grass: Vec<(DVec2, f64)> = candidates
            .iter()
            .take(300)
            .copied()
            .filter(|&(crest, _)| {
                let eye = top_eye(crest);
                self.source
                    .ground_state(eye.x.floor() as i32, eye.z.floor() as i32)
                    == grass
            })
            .collect();
        let candidates = if on_grass.is_empty() {
            candidates
        } else {
            on_grass
        };
        let (crest, yaw) = self.first_free(
            &candidates,
            300,
            VISTA_FOV_DEG,
            FREE_VIEW_TREES,
            |&(crest, az)| (top_eye(crest), az),
        )?;
        let eye = top_eye(crest);
        let along = mean_along(DVec2::new(eye.x, eye.z), yaw, &ahead);
        // Down along the ridge, the horizon at least 40 percent down the frame.
        let pitch = det::min(
            det::clamp(det::atan2(along - eye.y, 240.0), -0.2, 0.0) - 0.06,
            horizon_pitch(VISTA_FOV_DEG, HORIZON_FROM_TOP),
        );
        Some((eye, yaw, pitch))
    }

    /// Camera 4: between the trunks of the densest forest within 1.5 km of camera 0.
    fn forest(&self, near: DVec2) -> Option<(DVec3, f64, f64)> {
        const BIN: f64 = 20.0;
        let reach = 1500.0;
        let trees = self.trees.trees_in(
            [near.x - reach, near.x + reach],
            [near.y - reach, near.y + reach],
        );
        let mut bins: BTreeMap<(i32, i32), Vec<TreeInfo>> = BTreeMap::new();
        for tree in &trees {
            let key = (
                (f64::from(tree.base.x) / BIN).floor() as i32,
                (f64::from(tree.base.z) / BIN).floor() as i32,
            );
            bins.entry(key).or_default().push(*tree);
        }
        let around = |(x, z): (i32, i32)| {
            (-1..=1)
                .flat_map(|i| (-1..=1).map(move |j| (x + i, z + j)))
                .map(|key| bins.get(&key).map_or(0, Vec::len))
                .sum::<usize>()
        };
        let (_, bin) = best(
            bins.keys()
                .filter(|&&key| bins[&key].len() >= 2)
                .map(|&key| (around(key) as f64, key)),
        )?;
        let nearby: Vec<DVec2> = trees
            .iter()
            .map(|tree| DVec2::new(f64::from(tree.base.x) + 0.5, f64::from(tree.base.z) + 0.5))
            .filter(|trunk| {
                let center = DVec2::new(
                    (f64::from(bin.0) + 0.5) * BIN,
                    (f64::from(bin.1) + 0.5) * BIN,
                );
                trunk.distance(center) < 60.0
            })
            .collect();
        // A clearing between trunks: at least 4.5 m from every trunk (a nearer one fills a
        // third of the frame), with many trunks within 5 to 16 m.
        let origin = DVec2::new(f64::from(bin.0) * BIN, f64::from(bin.1) * BIN);
        let spot = best(
            (0..20)
                .flat_map(|j| (0..20).map(move |i| (i, j)))
                .filter_map(|(i, j)| {
                    let p = origin + DVec2::new(f64::from(i) + 0.5, f64::from(j) + 0.5);
                    let closest = nearby
                        .iter()
                        .map(|trunk| trunk.distance(p))
                        .fold(f64::INFINITY, det::min);
                    if closest < 4.5 {
                        return None;
                    }
                    let around = nearby
                        .iter()
                        .filter(|trunk| (5.0..=16.0).contains(&trunk.distance(p)))
                        .count();
                    Some((around as f64 - 0.01 * closest, p))
                }),
        )?
        .1;
        let eye = self.eye_at(spot);
        let (_, yaw) = best((0..72).map(|k| {
            let az = azimuth(k, 72);
            let (sin, cos) = det::sin_cos(az);
            let look = DVec2::new(sin, -cos);
            let seen = nearby
                .iter()
                .filter(|trunk| {
                    let offset = **trunk - DVec2::new(eye.x, eye.z);
                    let distance = offset.length();
                    (5.0..=35.0).contains(&distance) && offset.dot(look) > 0.9 * distance
                })
                .count();
            (seen as f64, az)
        }))?;
        // The nearest trunk ahead a third of the way across the frame, not in its middle: the
        // view turns so that it stands at the left third (or the right one, if it stood right
        // of the middle), and the forest opens beside it.
        let here = DVec2::new(eye.x, eye.z);
        let (sin, cos) = det::sin_cos(yaw);
        let look = DVec2::new(sin, -cos);
        let nearest = nearby
            .iter()
            .filter(|trunk| {
                let offset = **trunk - here;
                offset.dot(look) > 0.7 * offset.length()
            })
            .map(|trunk| (trunk.distance(here), *trunk))
            .fold(
                None,
                |best: Option<(f64, DVec2)>, (distance, trunk)| match best {
                    Some((top, _)) if top <= distance => best,
                    _ => Some((distance, trunk)),
                },
            );
        let third = det::atan(det::tan(0.5 * FOREST_FOV_DEG.to_radians()) * 16.0 / 9.0 / 3.0);
        let yaw = nearest.map_or(yaw, |(_, trunk)| {
            let offset = trunk - here;
            let toward = det::atan2(offset.x, -offset.y);
            let side = (toward - yaw + std::f64::consts::PI).rem_euclid(std::f64::consts::TAU)
                - std::f64::consts::PI;
            if side < 0.0 {
                toward + third
            } else {
                toward - third
            }
        });
        Some((eye, yaw, 0.07))
    }

    /// Camera 5: on a beach, looking out over the widest water, turned so that the shoreline
    /// runs through the view.
    fn coast(&self) -> Option<(DVec3, f64, f64)> {
        let beaches = lattice(DVec2::ZERO, 2500.0, 24.0).filter(|p| {
            let h = self.grid.at(p.x, p.y);
            (self.sea + 0.3..=self.sea + 2.4).contains(&h)
        });
        let water_run = |eye: DVec3, az: f64| {
            let mut run = 0.0;
            let mut open = true;
            sight_line(self.grid, self.sea, eye, az, [20.0, 2500.0, 20.0], |s| {
                if open && s.height < self.sea - 0.3 {
                    run = s.distance;
                } else if s.distance > 60.0 {
                    open = false;
                }
            });
            run
        };
        let found = best(beaches.flat_map(|point| {
            let eye = DVec3::new(point.x, self.sea + 2.0, point.y);
            (0..36).filter_map(move |k| {
                let az = azimuth(k, 36);
                let run = water_run(eye, az);
                (run >= 600.0).then_some((run, (point, az)))
            })
        }))?;
        let (point, az) = found.1;
        // Down the shore to the last column above the water, so the camera stands at the
        // water's edge with the beach (or the shore) under it, not in the meadow behind.
        let (sin, cos) = det::sin_cos(az);
        let toward = DVec2::new(sin, -cos);
        let mut edge = point;
        for step in 1..=40 {
            let next = point + toward * f64::from(step);
            if self
                .source
                .ground_top(next.x.floor() as i32, next.y.floor() as i32)
                < 0
            {
                break;
            }
            edge = next;
        }
        let eye = self.eye_at(edge);
        // Turn 25 degrees towards the side where land is nearer, so the shore is in view.
        let side = 35.0_f64.to_radians();
        let left = water_run(eye, az - side);
        let right = water_run(eye, az + side);
        let turn = 25.0_f64.to_radians();
        let yaw = if left < right { az - turn } else { az + turn };
        // The horizon 40 percent down: the water and the shore below it are the subject.
        Some((eye, yaw, horizon_pitch(VISTA_FOV_DEG, HORIZON_FROM_TOP)))
    }

    /// `point`, or a place up to 30 m beside or behind it with the same view towards `az`,
    /// from which a tree stands in a lower corner of the frame ([`Self::anchors_in_view`]): a
    /// near tree at the side gives an open landscape a foreground and depth. Only places with
    /// a free view qualify; without such a place `point` stays.
    fn anchored(&self, point: DVec2, az: f64, fov_deg: f64, crowd: f64) -> DVec2 {
        let (sin, cos) = det::sin_cos(az);
        let (look, right) = (DVec2::new(sin, -cos), DVec2::new(cos, sin));
        let trees = Self::anchor_trees(&self.features_near(point));
        let offsets = (-10..=10).flat_map(|i| (0..=6).map(move |j| (i, j)));
        let found = best(offsets.filter_map(|(i, j)| {
            let place = point + right * (3.0 * f64::from(i)) - look * (5.0 * f64::from(j));
            let anchors = Self::anchors_in_view(&trees, self.eye_at(place), az, fov_deg).min(2.0);
            if anchors < 0.4 {
                return None;
            }
            let moved = place.distance(point);
            Some((anchors - 0.01 * moved, place))
        }));
        // From the best down: the first that keeps a free view.
        let mut ranked: Vec<(f64, DVec2)> = found.into_iter().collect();
        ranked.sort_by(|a, b| b.0.total_cmp(&a.0));
        ranked
            .into_iter()
            .map(|(_, place)| place)
            .find(|&place| self.free_view(self.eye_at(place), az, fov_deg, crowd))
            .unwrap_or(point)
    }

    /// Camera 7: an open meadow before the edge of a wood, at least 2 km from every place in
    /// `away`, in a region whose character differs from the first place's: the trees 25 to
    /// 90 m away fill the middle of the frame, none stands nearer, and no lake is in view (the
    /// valley and coast cameras have the lakes). Dry regions are preferred, where the edge is
    /// a grove of birches over straw-coloured grass.
    fn forest_edge(&self, away: &[DVec2]) -> Option<(DVec3, f64, f64)> {
        let home = away.first().map_or(0.0, |p| self.source.dryness(p.x, p.y));
        let grass = self.source.palette().grass;
        let half = det::atan(det::tan(0.5 * VISTA_FOV_DEG.to_radians()) * 16.0 / 9.0);
        let candidates = ranked(
            lattice(DVec2::ZERO, 3600.0, 48.0)
                .filter(|p| {
                    let h = self.grid.at(p.x, p.y);
                    h >= self.sea + 4.0
                        && away.iter().all(|other| other.distance(*p) >= 2000.0)
                        && self.source.wood(p.x, p.y) < 0.1
                })
                .flat_map(|point| {
                    let trunks = self.trunks_near(point);
                    let dryness = self.source.dryness(point.x, point.y);
                    (0..36).filter_map(move |k| {
                        let az = azimuth(k, 36);
                        let (sin, cos) = det::sin_cos(az);
                        let look = DVec2::new(sin, -cos);
                        let (mut edge, mut near) = (0.0_f64, 0);
                        for trunk in &trunks {
                            let offset = *trunk - point;
                            let distance = offset.length();
                            let angle = det::acos(det::clamp(
                                offset.dot(look) / det::max(distance, 1e-9),
                                -1.0,
                                1.0,
                            ));
                            if distance < 25.0 && angle < half + 0.2 {
                                near += 1;
                            } else if (25.0..=90.0).contains(&distance) && angle < 0.8 * half {
                                edge += 1.0;
                            }
                        }
                        if near > 0 || edge < 12.0 {
                            return None;
                        }
                        let eye =
                            DVec3::new(point.x, self.grid.at(point.x, point.y) + 2.0, point.y);
                        let view = self.view(eye, az);
                        if view.blocked || view.water > 0.0 {
                            return None;
                        }
                        let score =
                            edge.min(60.0) + 120.0 * (dryness - home).abs() + 60.0 * dryness;
                        Some((score, (point, az)))
                    })
                }),
        );
        let (point, az) = candidates.iter().copied().find(|&(point, az)| {
            self.source
                .ground_state(point.x.floor() as i32, point.y.floor() as i32)
                == grass
                && self.open_foreground(self.eye_at(point), az, VISTA_FOV_DEG)
        })?;
        let eye = self.eye_at(point);
        Some((eye, az, horizon_pitch(VISTA_FOV_DEG, HORIZON_FROM_TOP)))
    }

    /// Cameras 6 and 7: the best open view from a hill at least 2 km from every place in
    /// `away` and 800 m from every place in `avoid` (other cameras whose views it should not
    /// repeat), by `taste`, in a region whose character differs from the first place's (the
    /// dryness of the land, [`HillsSource::dryness`]); no tree in the middle of the view, no
    /// slope ahead that rises like a wall ([`Self::wall_ahead`]), and standing on grass (not on
    /// snow, scree or rock, which fill the foreground with large blocks). Snow on the far
    /// heights counts much: the high vista is the view to the mountains. The horizon sits 40
    /// percent down the frame, and a broadleaf tree in a lower corner is welcome.
    fn vista(
        &self,
        away: &[DVec2],
        avoid: &[DVec2],
        taste: VistaTaste,
    ) -> Option<(DVec3, f64, f64)> {
        let home = away.first().map_or(0.0, |p| self.source.dryness(p.x, p.y));
        let grass = self.source.palette().grass;
        let candidates = lattice(DVec2::ZERO, 3600.0, 64.0).filter(|p| {
            let h = self.grid.at(p.x, p.y);
            h >= self.sea + 6.0
                && away.iter().all(|other| other.distance(*p) >= 2000.0)
                && avoid.iter().all(|other| other.distance(*p) >= 800.0)
                && self.grid.is_local_max(p.x, p.y, 64.0)
        });
        let snow = self.source.snow_line();
        let candidates = ranked(candidates.flat_map(|top| {
            let features = self.features_near(top);
            let anchor_trees = Self::anchor_trees(&features);
            let dryness = self.source.dryness(top.x, top.y);
            let character = 150.0 * (dryness - home).abs();
            (0..36).filter_map(move |k| {
                let az = azimuth(k, 36);
                let point = self.brow(top, az);
                let eye = DVec3::new(point.x, self.grid.at(point.x, point.y) + 2.0, point.y);
                if self
                    .source
                    .ground_state(point.x.floor() as i32, point.y.floor() as i32)
                    != grass
                {
                    return None;
                }
                let view = self.view(eye, az);
                if view.blocked || self.wall_ahead(eye, az) {
                    return None;
                }
                let (water, high) = (view.water.min(800.0), view.high.min(800.0));
                let rise = view
                    .ridge
                    .map_or(0.0, |(height, _)| (height - eye.y).clamp(0.0, 40.0));
                // Snow on the heights beyond 400 m: a view to the mountains.
                let snowy = view.ridge.is_some_and(|(height, _)| height >= snow + 8.0);
                let trees = Self::trees_in_view(&features, eye, az);
                let drop = view.drop.clamp(0.0, 25.0);
                let score = match taste {
                    // High ground with a deep, far view over the land below.
                    VistaTaste::High => {
                        if eye.y < HIGH_GROUND {
                            return None;
                        }
                        14.0 * drop
                            + 0.4 * high
                            + 6.0 * rise
                            + 0.2 * water
                            + 0.1 * view.far
                            + if snowy { 200.0 } else { 0.0 }
                    }
                } - 80.0 * trees
                    + 60.0 * Self::anchors_in_view(&anchor_trees, eye, az, VISTA_FOV_DEG).min(2.0)
                    + character;
                Some((score, (point, az)))
            })
        }));
        // Only views whose middle stays clear of rocks and steps at eye height for 64 m (the
        // height grid is too coarse to see a wall of rock in front of the camera).
        let clear: Vec<(DVec2, f64)> = candidates
            .iter()
            .take(600)
            .filter(|&&(point, az)| self.clear_ahead(self.eye_at(point), az, 64.0))
            .copied()
            .collect();
        let pool = if clear.is_empty() {
            &candidates
        } else {
            &clear
        };
        let (point, az) =
            self.first_free(pool, 300, VISTA_FOV_DEG, FREE_VIEW_TREES, |&(point, az)| {
                (self.eye_at(point), az)
            })?;
        let point = self.anchored(point, az, VISTA_FOV_DEG, FREE_VIEW_TREES);
        let eye = self.eye_at(point);
        // The horizon 40 percent down the frame: the land is the subject.
        Some((eye, az, horizon_pitch(VISTA_FOV_DEG, HORIZON_FROM_TOP)))
    }
}

/// What a vista camera looks for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum VistaTaste {
    /// Ridges and high ground.
    High,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn best_keeps_the_first_of_equal_scores() {
        let items = [(1.0, 'a'), (3.0, 'b'), (3.0, 'c'), (2.0, 'd')];
        assert_eq!(best(items.into_iter()), Some((3.0, 'b')));
        assert_eq!(best(std::iter::empty::<(f64, char)>()), None);
    }

    #[test]
    fn ranked_starts_with_the_pick_of_best_and_keeps_ties_in_order() {
        let items = [(1.0, 'a'), (3.0, 'b'), (-2.0, 'e'), (3.0, 'c'), (2.0, 'd')];
        assert_eq!(ranked(items.into_iter()), ['b', 'c', 'd', 'a', 'e']);
        assert_eq!(
            ranked(items.into_iter()).first().copied(),
            best(items.into_iter()).map(|(_, item)| item)
        );
    }

    #[test]
    fn the_evening_sun_stands_beside_the_view() {
        let at = |degrees: f64| evening_side(degrees.to_radians());
        // The sun 45 degrees to either side: the best views; straight ahead, behind or at a
        // right angle: none.
        assert_eq!(at(EVENING_SUN_DEG - 45.0), 1.0);
        assert_eq!(at(EVENING_SUN_DEG + 45.0), 1.0);
        assert_eq!(at(EVENING_SUN_DEG + 45.0 - 360.0), 1.0);
        assert_eq!(at(EVENING_SUN_DEG), 0.0);
        assert_eq!(at(EVENING_SUN_DEG + 90.0), 0.0);
        assert_eq!(at(EVENING_SUN_DEG + 180.0), 0.0);
        assert!((0.0..1.0).contains(&at(EVENING_SUN_DEG + 25.0)));
    }

    #[test]
    fn the_lattice_is_centred_and_complete() {
        let points: Vec<DVec2> = lattice(DVec2::new(10.0, 20.0), 32.0, 16.0).collect();
        assert_eq!(points.len(), 25);
        assert_eq!(points[0], DVec2::new(-22.0, -12.0));
        assert_eq!(points[12], DVec2::new(10.0, 20.0));
    }

    #[test]
    fn sight_lines_hide_what_lies_behind_a_hill() {
        let grid = HeightGrid {
            min: DVec2::splat(-100.0),
            step: 10.0,
            side: 21,
            // A wall 20 m high at x = 20 (column 12), flat ground elsewhere.
            heights: (0..21 * 21)
                .map(|i| if i % 21 == 12 { 20.0 } else { 0.0 })
                .collect(),
        };
        let mut seen = Vec::new();
        // East along +x from the origin, eye 2 m above the ground.
        sight_line(
            &grid,
            -100.0,
            DVec3::new(0.0, 2.0, 0.0),
            std::f64::consts::FRAC_PI_2,
            [10.0, 60.0, 10.0],
            |sample| seen.push((sample.distance, sample.visible)),
        );
        assert_eq!(
            seen,
            [
                (10.0, true),
                (20.0, true),
                (30.0, false),
                (40.0, false),
                (50.0, false),
                (60.0, false)
            ]
        );
    }
}
