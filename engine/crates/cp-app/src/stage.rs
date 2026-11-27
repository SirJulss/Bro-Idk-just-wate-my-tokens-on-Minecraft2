//! What `shot`, `bench` and `run` share: the engine's CPU side (content, job system), the
//! renderer over the base pack, and a scene loaded into a renderer for one camera.

use std::sync::Arc;

use cp_content::BlockTables;
use cp_core::math::{DMat4, DVec3, DVec4};
use cp_jobs::JobSystem;
use cp_render::gpu::{GpuContext, GpuOptions};
use cp_render::overlay::{Overlay, OverlayColor};
use cp_render::voxel::{
    CalibrationScene, Camera, ChunkId, GridPose, GroundMap, ObjectDesc, ObjectId, VoxelRenderer,
    VoxelRendererConfig,
};
use cp_render::wgpu;

use crate::Failure;
use crate::assets::Assets;
use crate::scene::build::{BuildReport, build_blocking};
use crate::scene::calib::CalibSubScene;
use crate::scene::hills::{HillsScene, Reach};
use crate::scene::structures::{LEVELS, StructureScene};

/// Draw items of the renderers of cp-app: twice the default, because the hills put up to
/// about 40,000 chunk layers into one scene.
pub const MAX_DRAW_ITEMS: u32 = 65_536;

/// The CPU side of the engine: the content and the one job system (contract §4.2).
pub struct Engine {
    /// Content, textures, tables.
    pub assets: Assets,
    /// The engine's job system ([`JobSystem::standard`]).
    pub jobs: JobSystem,
    /// The registry's per-state tables, shared with the chunk jobs.
    pub tables: Arc<BlockTables>,
}

impl std::fmt::Debug for Engine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Engine")
            .field("assets", &self.assets)
            .field("workers", &self.jobs.threads())
            .finish_non_exhaustive()
    }
}

impl Engine {
    /// Loads the base pack and starts the job system.
    ///
    /// # Errors
    /// Those of [`Assets::load`].
    pub fn new() -> Result<Self, Failure> {
        let assets = Assets::load(&[])?;
        let tables = Arc::new(assets.tables().clone());
        let jobs = JobSystem::standard("engine");
        log::info!(
            "job system: {} workers, at most {} on bulk jobs",
            jobs.threads(),
            jobs.bulk_limit()
        );
        Ok(Self {
            assets,
            jobs,
            tables,
        })
    }
}

/// Opens the GPU without a window.
///
/// # Errors
/// A [`Failure`] when no device can be opened.
pub fn open_headless() -> Result<GpuContext, Failure> {
    GpuContext::new_blocking(&GpuOptions::default(), None)
        .map_err(|error| Failure::new(format!("cannot open a GPU device: {error}")))
}

/// The renderer configuration of cp-app for `format`: the device's quality tier with
/// [`MAX_DRAW_ITEMS`].
pub fn renderer_config(gpu: &GpuContext, format: wgpu::TextureFormat) -> VoxelRendererConfig {
    let mut config = VoxelRendererConfig::for_tier(gpu.caps.tier, format);
    config.max_draw_items = MAX_DRAW_ITEMS;
    config
}

/// A renderer with the base pack's materials and textures.
///
/// # Errors
/// A [`Failure`] when the GPU refuses a resource or pipeline.
pub fn make_renderer(
    gpu: &GpuContext,
    assets: &Assets,
    config: &VoxelRendererConfig,
) -> Result<VoxelRenderer, Failure> {
    let materials = assets.material_table(gpu)?;
    let albedo = assets.albedo_array(gpu)?;
    let renderer = VoxelRenderer::new(gpu, config, materials, albedo)
        .map_err(|error| Failure::new(format!("cannot create the voxel renderer: {error}")))?;
    log::info!(
        "renderer: {} pipelines in {:.1} ms, draw path {} (available: {})",
        renderer.pipeline_count(),
        renderer.startup_ms(),
        renderer.draw_mode().id(),
        renderer
            .available_draw_modes()
            .iter()
            .map(|mode| mode.id())
            .collect::<Vec<_>>()
            .join(", ")
    );
    Ok(renderer)
}

/// A text label at a point of the world.
#[derive(Clone, Debug, PartialEq)]
pub struct Label {
    /// Where it belongs (metres).
    pub anchor: DVec3,
    /// What it says.
    pub text: String,
}

/// A scene loaded into a renderer.
#[derive(Debug, Default)]
pub struct Stage {
    /// The objects it added.
    pub objects: Vec<ObjectId>,
    /// The chunks it added.
    pub chunks: Vec<ChunkId>,
    /// The build of its chunks, if it built any.
    pub report: Option<BuildReport>,
    /// The calibration scene, for the calibration sub-scenes.
    pub calib: Option<CalibrationScene>,
    /// Labels to draw over the image.
    pub labels: Vec<Label>,
}

impl Stage {
    /// The hills world around `eye`, generated and meshed on the job system.
    ///
    /// # Errors
    /// A [`Failure`] when a chunk cannot be built or the renderer refuses it.
    pub fn hills(
        renderer: &mut VoxelRenderer,
        gpu: &GpuContext,
        engine: &Engine,
        hills: &HillsScene,
        eye: DVec3,
        reach: Reach,
    ) -> Result<Self, Failure> {
        let (bounds, planned) = hills.plan(eye, reach);
        let object = renderer
            .scene_mut()
            .add_object(ObjectDesc {
                bounds,
                pose: GridPose::IDENTITY,
            })
            .map_err(|error| Failure::new(format!("hills object: {error}")))?;
        let requests = HillsScene::requests(&planned, object);
        let mut stage = Self {
            objects: vec![object],
            ..Self::default()
        };
        let report = build_blocking(
            &engine.jobs,
            hills.snapshot(),
            &engine.tables,
            &requests,
            renderer.scene_mut(),
            "hills",
            &mut stage.chunks,
        )?;
        renderer.scene_mut().flush_all_uploads(gpu);
        stage.report = Some(report);
        set_ground_map(renderer, gpu, &hills.ground_map(&engine.jobs, eye, reach))?;
        Ok(stage)
    }

    /// The structure field at every level of [`LEVELS`], labelled.
    ///
    /// # Errors
    /// A [`Failure`] when a chunk cannot be built or the renderer refuses it.
    pub fn structures(
        renderer: &mut VoxelRenderer,
        gpu: &GpuContext,
        engine: &Engine,
        scene: &StructureScene,
    ) -> Result<Self, Failure> {
        let (objects, requests) = scene.install(renderer.scene_mut())?;
        let mut stage = Self {
            objects: std::iter::once(objects.ground)
                .chain(objects.rows.iter().copied())
                .collect(),
            ..Self::default()
        };
        let report = build_blocking(
            &engine.jobs,
            scene.snapshot(),
            &engine.tables,
            &requests,
            renderer.scene_mut(),
            "structures",
            &mut stage.chunks,
        )?;
        renderer.scene_mut().flush_all_uploads(gpu);
        stage.report = Some(report);
        set_ground_map(renderer, gpu, &GroundMap::empty())?;
        stage.labels = LEVELS
            .iter()
            .enumerate()
            .map(|(row, level)| Label {
                anchor: StructureScene::label_anchor(row),
                text: format!("level {level} ({} m cells)", 1_u32 << level),
            })
            .collect();
        Ok(stage)
    }

    /// A calibration sub-scene for an output of `aspect` (width / height).
    ///
    /// # Errors
    /// A [`Failure`] when the renderer cannot hold the sub-scene.
    pub fn calib(
        renderer: &mut VoxelRenderer,
        gpu: &GpuContext,
        sub: &CalibSubScene,
        aspect: f32,
    ) -> Result<Self, Failure> {
        let mut params = sub.params;
        params.aspect = aspect;
        let calib = CalibrationScene::build(renderer.scene_mut(), &params)
            .map_err(|error| Failure::new(format!("calibration `{}`: {error}", sub.name)))?;
        renderer.scene_mut().flush_all_uploads(gpu);
        set_ground_map(renderer, gpu, &GroundMap::empty())?;
        Ok(Self {
            calib: Some(calib),
            ..Self::default()
        })
    }

    /// Removes everything the stage added from the renderer.
    ///
    /// # Errors
    /// A [`Failure`] when the renderer does not know a chunk or object of the stage.
    pub fn clear(self, renderer: &mut VoxelRenderer, gpu: &GpuContext) -> Result<(), Failure> {
        let scene = renderer.scene_mut();
        let fail = |error: cp_render::voxel::VoxelError| {
            Failure::new(format!("cannot remove the stage: {error}"))
        };
        for chunk in self.chunks {
            scene.remove_chunk(chunk).map_err(fail)?;
        }
        for object in self.objects {
            scene.remove_object(object).map_err(fail)?;
        }
        if let Some(calib) = self.calib {
            calib.remove(scene).map_err(fail)?;
        }
        scene.flush_all_uploads(gpu);
        Ok(())
    }

    /// Draws the stage's labels into `overlay` for `camera` and an output of `size` pixels;
    /// a label whose anchor lies off to a side is pushed back inside the image.
    pub fn draw_labels(&self, overlay: &mut Overlay, camera: &Camera, size: (u32, u32)) {
        let scale = (size.1 / 540).max(1);
        let to_clip = camera.frame_to_clip(size.0 as f32 / size.1.max(1) as f32);
        for label in &self.labels {
            if let Some((x, y)) = project(&to_clip, label.anchor, size) {
                // The panel's margin is two font pixels on each side.
                let extent = Overlay::measure(&label.text, scale);
                let width = i32::try_from(extent.width + 4 * scale).unwrap_or(i32::MAX);
                let height = i32::try_from(extent.height + 4 * scale).unwrap_or(i32::MAX);
                let max_x = i32::try_from(size.0).unwrap_or(i32::MAX) - width;
                let max_y = i32::try_from(size.1).unwrap_or(i32::MAX) - height;
                if y < 0 || y > max_y {
                    continue;
                }
                overlay.text_panel(
                    x.clamp(0, max_x.max(0)),
                    y,
                    scale,
                    &[label.text.as_str()],
                    OverlayColor::WHITE,
                    OverlayColor::PANEL,
                );
            }
        }
    }
}

/// Gives the renderer the ground map of a stage (an empty one for worlds without).
fn set_ground_map(
    renderer: &mut VoxelRenderer,
    gpu: &GpuContext,
    map: &GroundMap,
) -> Result<(), Failure> {
    renderer
        .set_ground_map(gpu, map)
        .map_err(|error| Failure::new(format!("cannot set the ground map: {error}")))
}

/// The pixel of `point` for `to_clip` and an output of `size` pixels; `None` behind the
/// camera.
pub fn project(to_clip: &DMat4, point: DVec3, size: (u32, u32)) -> Option<(i32, i32)> {
    let clip = *to_clip * DVec4::new(point.x, point.y, point.z, 1.0);
    if clip.w <= 1.0e-6 {
        return None;
    }
    let (x, y) = (clip.x / clip.w, clip.y / clip.w);
    let px = (x + 1.0) * 0.5 * f64::from(size.0);
    let py = (1.0 - y) * 0.5 * f64::from(size.1);
    let limit = f64::from(i32::MAX / 2);
    (px.abs() < limit && py.abs() < limit).then_some((px as i32, py as i32))
}
