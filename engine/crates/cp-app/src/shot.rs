//! `cpe shot`: one headless frame of a scene camera as a PNG file (contract §5.4).
//!
//! The scene's world is built around the camera first and every mesh is uploaded before the
//! frame is drawn, so a shot never shows a world that is still loading. The renderer draws a
//! history of frames before the one that is saved: temporal anti-aliasing accumulates them
//! (a fixed camera repeats itself for two cycles of the jitter; a camera on the flight path
//! flies the 0.8 s before its time at 30 frames per second, so that two shots 1/30 s apart are
//! consecutive frames of one flight), path A's counters arrive a frame late. The last frame is
//! read back from an sRGB target and written with the PNG `sRGB` chunk.

use std::fmt;
use std::path::PathBuf;

use cp_render::headless::HeadlessTarget;
use cp_render::overlay::Overlay;
use cp_render::voxel::{Camera, DrawMode, FrameInput, VisibleCounts, VoxelRenderer};
use cp_render::wgpu;

use crate::cli::ShotArgs;
use crate::metrics::ImageMetrics;
use crate::scene::SceneName;
use crate::scene::calib::sub_scenes;
use crate::scene::hills::{HillsScene, Reach};
use crate::scene::sky::{Sun, daylight};
use crate::scene::structures::StructureScene;
use crate::stage::{Engine, Stage, make_renderer, open_headless, renderer_config};
use crate::{Failure, image, write_file};

/// Format of the shot target: sRGB, so the PNG holds display-encoded pixels.
pub const SHOT_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8UnormSrgb;

/// Frames drawn before the one that is saved by a fixed camera: two cycles of the jitter of
/// temporal anti-aliasing (and enough for path A's late counters).
const WARM_UP_FRAMES: usize = 2 * cp_render::voxel::JITTER_PHASES as usize;

/// Frames drawn before the one that is saved by a camera on the flight path, at
/// [`FLIGHT_FRAME_RATE`]: the history of temporal anti-aliasing is then the flight's own.
const FLIGHT_HISTORY_FRAMES: usize = 24;

/// Frame rate of the flight history (frames per second); the beauty review's motion pairs are
/// 1/30 s apart.
pub const FLIGHT_FRAME_RATE: f64 = 30.0;

/// What a shot wrote.
#[derive(Clone, Debug, PartialEq)]
pub struct ShotInfo {
    /// The PNG file.
    pub path: PathBuf,
    /// Size in pixels.
    pub size: (u32, u32),
    /// Scene and camera, as a label.
    pub view: String,
    /// The draw path.
    pub mode: DrawMode,
    /// What the last frame drew.
    pub visible: Option<VisibleCounts>,
    /// Chunks in the scene.
    pub chunks: u32,
    /// GPU time of the saved frame, milliseconds.
    pub gpu_ms: Option<f64>,
    /// Measured criteria of the saved image for the beauty review.
    pub metrics: ImageMetrics,
}

impl fmt::Display for ShotInfo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "wrote {}: {} x {} px, {}, draw path {}, {} chunks",
            self.path.display(),
            self.size.0,
            self.size.1,
            self.view,
            self.mode.id(),
            self.chunks
        )?;
        if let Some(visible) = self.visible {
            write!(
                f,
                ", {} draws, {:.2} M triangles",
                visible.total_draws(),
                visible.triangles() as f64 / 1.0e6
            )?;
        }
        if let Some(ms) = self.gpu_ms {
            write!(f, ", GPU {ms:.2} ms")?;
        }
        write!(f, "\nimage: {}", self.metrics)
    }
}

/// A scene camera ready to render: the renderer's camera, the default sun, the cameras of the
/// frames before it (oldest first) and the distances over which the world fades into the haze.
#[derive(Clone, Debug)]
pub(crate) struct ShotCamera {
    pub(crate) camera: Camera,
    pub(crate) sun: Sun,
    pub(crate) history: Vec<Camera>,
    pub(crate) horizon_fade: [f32; 2],
}

impl ShotCamera {
    /// A camera that stands still: its history is itself.
    fn fixed(camera: Camera, sun: Sun, horizon_fade: [f32; 2]) -> Self {
        Self {
            camera,
            sun,
            history: vec![camera; WARM_UP_FRAMES],
            horizon_fade,
        }
    }
}

/// Renders the shot `args` asks for and writes the PNG.
///
/// # Errors
/// A [`Failure`] for an unknown camera, a GPU problem, or a file that cannot be written.
pub fn run(args: &ShotArgs) -> Result<ShotInfo, Failure> {
    let engine = Engine::new()?;
    let gpu = open_headless()?;
    let config = renderer_config(&gpu, SHOT_FORMAT);
    let mut renderer = make_renderer(&gpu, &engine.assets, &config)?;
    if let Some(mode) = args.path {
        renderer
            .set_draw_mode(mode)
            .map_err(|error| Failure::new(error.to_string()))?;
    }
    renderer
        .debug_view_mut()
        .select(args.view)
        .map_err(|error| Failure::new(error.to_string()))?;
    let aspect = args.width as f32 / args.height as f32;
    let (stage, shot, view) = load(&mut renderer, &gpu, &engine, args, aspect)?;
    let sun = args.sun.unwrap_or(shot.sun);
    let mut lighting = daylight(sun);
    lighting.horizon_fade = shot.horizon_fade;
    let target = HeadlessTarget::new(&gpu, args.width, args.height, SHOT_FORMAT)
        .map_err(|error| Failure::new(format!("cannot create the shot target: {error}")))?;
    let mut overlay = Overlay::new();
    stage.draw_labels(&mut overlay, &shot.camera, target.size());
    let mut stats = None;
    for camera in shot.history.iter().chain([&shot.camera]) {
        let input = FrameInput {
            camera,
            lighting: &lighting,
            target: target.view(),
            target_size: target.size(),
            overlay: (!overlay.is_empty()).then_some(&overlay),
        };
        stats = Some(
            renderer
                .render(&gpu, &input)
                .map_err(|error| Failure::new(format!("rendering failed: {error}")))?,
        );
        renderer
            .wait_for_gpu(&gpu)
            .map_err(|error| Failure::new(format!("rendering failed: {error}")))?;
    }
    let stats = stats.expect("at least one frame was rendered");
    let image = target
        .capture_image(&gpu)
        .map_err(|error| Failure::new(format!("cannot read the shot back: {error}")))?;
    let png = image::encode_rgba_with(image.width, image.height, &image.pixels, image.srgb)
        .map_err(|reason| Failure::new(format!("cannot encode the shot: {reason}")))?;
    write_file(&args.out, &png)?;
    let visible = match stats.mode {
        DrawMode::A => renderer.latest_gpu_counts(),
        DrawMode::B | DrawMode::Direct => stats.visible,
    };
    let gpu_ms = renderer
        .latest_timings()
        .filter(|timings| timings.frame == stats.frame)
        .map(cp_render::frame_graph::FrameTimings::gpu_total_ms);
    Ok(ShotInfo {
        path: args.out.clone(),
        size: target.size(),
        view: format!(
            "{view}, sun {:.0}/{:.0} deg, view {}",
            sun.elevation_deg,
            sun.azimuth_deg,
            args.view.id()
        ),
        mode: stats.mode,
        visible,
        chunks: stats.chunks,
        gpu_ms,
        metrics: ImageMetrics::measure(image.width, image.height, &image.pixels),
    })
}

/// Loads the scene of `args` into `renderer` and returns it with the camera and a label.
fn load(
    renderer: &mut VoxelRenderer,
    gpu: &cp_render::gpu::GpuContext,
    engine: &Engine,
    args: &ShotArgs,
    aspect: f32,
) -> Result<(Stage, ShotCamera, String), Failure> {
    let unknown = |count: usize| {
        Failure::new(format!(
            "scene `{}` has no camera {} (cameras 0 to {})",
            args.scene,
            args.camera,
            count.saturating_sub(1)
        ))
    };
    match args.scene {
        SceneName::Hills => {
            let hills = HillsScene::new(&engine.assets, &engine.jobs)?;
            let (camera, history, reach, label) = match args.time {
                Some(time) => {
                    // The frames before `time` on the flight path, oldest first.
                    let history = (1..=FLIGHT_HISTORY_FRAMES)
                        .rev()
                        .map(|back| {
                            let t = time - back as f64 / FLIGHT_FRAME_RATE;
                            hills.flight(t).render_camera()
                        })
                        .collect();
                    (
                        hills.flight(time),
                        Some(history),
                        Reach::SHOT,
                        format!("hills flight t = {time} s"),
                    )
                }
                None => {
                    let camera = *hills
                        .cameras()
                        .get(args.camera)
                        .ok_or_else(|| unknown(hills.cameras().len()))?;
                    let label = format!("hills camera {} `{}`", args.camera, camera.name);
                    (camera, None, HillsScene::reach(args.camera), label)
                }
            };
            let stage = Stage::hills(renderer, gpu, engine, &hills, camera.position, reach)?;
            let mut shot =
                ShotCamera::fixed(camera.render_camera(), camera.sun, reach.horizon_fade());
            if let Some(history) = history {
                shot.history = history;
            }
            Ok((stage, shot, label))
        }
        SceneName::Structures => {
            let camera = StructureScene::camera(args.camera)
                .ok_or_else(|| unknown(crate::scene::structures::CAMERA_COUNT))?;
            let scene = StructureScene::new(&engine.assets)?;
            let stage = Stage::structures(renderer, gpu, engine, &scene)?;
            Ok((
                stage,
                ShotCamera::fixed(camera.render_camera(), camera.sun, [0.0, 0.0]),
                format!("structures camera {} `{}`", args.camera, camera.name),
            ))
        }
        SceneName::Calib => {
            let scenes = sub_scenes();
            let sub = scenes
                .get(args.camera)
                .ok_or_else(|| unknown(scenes.len()))?;
            let stage = Stage::calib(renderer, gpu, sub, aspect)?;
            let camera = stage
                .calib
                .as_ref()
                .map(|calib| calib.camera)
                .expect("a calibration stage holds its scene");
            Ok((
                stage,
                ShotCamera::fixed(camera, Sun::default_from(135.0), [0.0, 0.0]),
                format!("calibration `{}`: {}", sub.name, sub.description),
            ))
        }
    }
}
