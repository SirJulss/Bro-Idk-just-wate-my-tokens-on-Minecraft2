//! `cpe run`: a window over a scene with a free-fly camera (contract §5.4).
//!
//! The window opens at once; the hills world is generated and meshed on the job system in the
//! background and appears chunk by chunk, with a progress line in the log and in the HUD. The
//! frame thread never waits for a chunk: it only takes the results that have arrived.
//!
//! Controls: mouse and WASD fly, Space up, C down, Shift fast (the speed grows with the
//! height above the ground); F1 HUD; F2 debug view; F3 draw path; F4 next camera of the scene
//! (calib: next sub-scene); F5 the scene's flight path (hills). A click captures the mouse,
//! Esc releases it, Esc again quits.
//!
//! A minimised window renders nothing until it comes back; a resize reconfigures the surface;
//! a lost or outdated surface is recreated; a lost device ends the run with an error instead
//! of a panic.

use std::collections::BTreeSet;
use std::sync::Arc;
use std::time::{Duration, Instant};

use cp_core::det;
use cp_core::math::DVec3;
use cp_jobs::JobClass;
use cp_render::gpu::{GpuContext, GpuOptions};
use cp_render::overlay::{Overlay, OverlayColor};
use cp_render::voxel::{FrameInput, FrameStats, GridPose, ObjectDesc, VoxelRenderer};
use cp_render::wgpu;
use winit::application::ApplicationHandler;
use winit::dpi::PhysicalSize;
use winit::event::{DeviceEvent, DeviceId, ElementState, MouseButton, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{CursorGrabMode, Window, WindowId};

use crate::Failure;
use crate::cli::RunArgs;
use crate::scene::build::{BuildBatch, BuildReport, BuiltChunk, insert};
use crate::scene::calib::sub_scenes;
use crate::scene::hills::{HillsScene, Reach};
use crate::scene::sky::{Sun, daylight, view_direction};
use crate::scene::structures::{self, StructureScene};
use crate::scene::{MAX_PITCH, SceneCamera, SceneName};
use crate::stage::{Engine, Stage, make_renderer, renderer_config};

/// Mouse look in radians per pixel of mouse motion.
const MOUSE_RADIANS: f64 = 0.0025;
/// Flying speed at the ground (m/s); it grows by this much per 10 m of height.
const BASE_SPEED: f64 = 10.0;
/// Speed factor with Shift held.
const FAST: f64 = 6.0;
/// Longest frame step the camera integrates (s), so a stall does not throw it far away.
const MAX_STEP: f64 = 0.1;

/// Runs the window until it is closed.
///
/// # Errors
/// A [`Failure`] when the window system, the GPU or the scene cannot be started, or the device
/// is lost while running.
pub fn run(args: &RunArgs) -> Result<String, Failure> {
    let engine = Engine::new()?;
    let event_loop = EventLoop::new()
        .map_err(|error| Failure::new(format!("cannot start the window system: {error}")))?;
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut app = App {
        scene: args.scene,
        engine,
        live: None,
        failure: None,
        frames: 0,
        loaded_frames: 0,
        quit_after: args.frames,
        started: Instant::now(),
    };
    event_loop
        .run_app(&mut app)
        .map_err(|error| Failure::new(format!("the window system failed: {error}")))?;
    if let Some(failure) = app.failure {
        return Err(failure);
    }
    Ok(format!(
        "{} frames in {:.1} s ({} after the world had loaded)",
        app.frames,
        app.started.elapsed().as_secs_f64(),
        app.loaded_frames
    ))
}

struct App {
    scene: SceneName,
    engine: Engine,
    live: Option<Live>,
    failure: Option<Failure>,
    /// Frames drawn.
    frames: u64,
    /// Frames drawn after the world had loaded.
    loaded_frames: u32,
    /// `--frames`: quit after this many loaded frames.
    quit_after: Option<u32>,
    started: Instant,
}

/// Everything that exists while the window is open.
struct Live {
    window: Arc<Window>,
    gpu: GpuContext,
    surface: wgpu::Surface<'static>,
    config: wgpu::SurfaceConfiguration,
    renderer: VoxelRenderer,
    world: World,
    fly: Fly,
    keys: BTreeSet<KeyCode>,
    mouse: (f64, f64),
    captured: bool,
    hud: bool,
    minimized: bool,
    last_frame: Instant,
    frame_ms: f64,
    last_stats: Option<FrameStats>,
}

/// The scene content of a window.
enum World {
    Hills(Box<HillsWorld>),
    Structures { stage: Stage, camera: usize },
    Calib { stage: Stage, index: usize },
}

/// The hills while they load and after.
struct HillsWorld {
    scene: HillsScene,
    batch: BuildBatch,
    report: BuildReport,
    arrived: Vec<BuiltChunk>,
    last_line: Instant,
    camera: usize,
    flight: Option<f64>,
}

/// The free-fly camera.
#[derive(Clone, Copy, Debug)]
struct Fly {
    position: DVec3,
    yaw: f64,
    pitch: f64,
    fov_y: f64,
    sun: Sun,
}

impl Fly {
    fn from_camera(camera: &SceneCamera) -> Self {
        Self {
            position: camera.position,
            yaw: camera.yaw,
            pitch: camera.pitch,
            fov_y: camera.fov_y,
            sun: camera.sun,
        }
    }

    fn scene_camera(&self) -> SceneCamera {
        SceneCamera {
            name: "free",
            position: self.position,
            yaw: self.yaw,
            pitch: self.pitch,
            fov_y: self.fov_y,
            sun: self.sun,
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.live.is_some() {
            return;
        }
        match Live::open(event_loop, self.scene, &self.engine) {
            Ok(live) => self.live = Some(live),
            Err(failure) => {
                self.failure = Some(failure);
                event_loop.exit();
            }
        }
    }

    fn device_event(&mut self, _: &ActiveEventLoop, _: DeviceId, event: DeviceEvent) {
        if let (Some(live), DeviceEvent::MouseMotion { delta }) = (&mut self.live, event)
            && live.captured
        {
            live.mouse.0 += delta.0;
            live.mouse.1 += delta.1;
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _: WindowId, event: WindowEvent) {
        let Some(live) = &mut self.live else {
            return;
        };
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => live.resize(size),
            WindowEvent::Focused(false) => {
                live.capture(false);
                live.keys.clear();
            }
            WindowEvent::MouseInput {
                state: ElementState::Pressed,
                button: MouseButton::Left,
                ..
            } => live.capture(true),
            WindowEvent::KeyboardInput { event, .. } => {
                let PhysicalKey::Code(code) = event.physical_key else {
                    return;
                };
                if event.state == ElementState::Released {
                    live.keys.remove(&code);
                    return;
                }
                live.keys.insert(code);
                if event.repeat {
                    return;
                }
                if code == KeyCode::Escape {
                    if live.captured {
                        live.capture(false);
                    } else {
                        event_loop.exit();
                    }
                    return;
                }
                if let Err(failure) = live.key(code) {
                    self.failure = Some(failure);
                    event_loop.exit();
                }
            }
            WindowEvent::RedrawRequested => match live.frame() {
                Ok(drawn) => {
                    self.frames += u64::from(drawn);
                    if drawn && live.world.progress().is_none() {
                        self.loaded_frames += 1;
                        if self
                            .quit_after
                            .is_some_and(|frames| self.loaded_frames >= frames)
                        {
                            event_loop.exit();
                        }
                    }
                }
                Err(failure) => {
                    self.failure = Some(failure);
                    event_loop.exit();
                }
            },
            _ => {}
        }
    }

    fn about_to_wait(&mut self, _: &ActiveEventLoop) {
        if let Some(live) = &self.live {
            live.window.request_redraw();
        }
    }
}

impl Live {
    fn open(
        event_loop: &ActiveEventLoop,
        scene: SceneName,
        engine: &Engine,
    ) -> Result<Self, Failure> {
        let window = event_loop
            .create_window(
                Window::default_attributes()
                    .with_title(format!("cpe run - {scene}"))
                    .with_inner_size(PhysicalSize::new(1600, 900))
                    .with_min_inner_size(PhysicalSize::new(320, 200)),
            )
            .map_err(|error| Failure::new(format!("cannot open a window: {error}")))?;
        let window = Arc::new(window);
        let surface_window = Arc::clone(&window);
        let (gpu, surface) =
            GpuContext::with_surface_blocking(&GpuOptions::default(), |instance| {
                instance.create_surface(surface_window)
            })
            .map_err(|error| Failure::new(format!("cannot open a GPU device: {error}")))?;
        let size = window.inner_size();
        let mut config = surface
            .get_default_config(&gpu.adapter, size.width.max(1), size.height.max(1))
            .ok_or_else(|| Failure::new("the window's surface does not fit the GPU"))?;
        let capabilities = surface.get_capabilities(&gpu.adapter);
        if let Some(format) = capabilities.formats.iter().copied().find(|f| f.is_srgb()) {
            config.format = format;
        }
        config.present_mode = wgpu::PresentMode::AutoVsync;
        surface.configure(&gpu.device, &config);
        let render_config = renderer_config(&gpu, config.format);
        let mut renderer = make_renderer(&gpu, &engine.assets, &render_config)?;
        let (world, fly) = World::start(&mut renderer, &gpu, scene, engine)?;
        let mut live = Self {
            window,
            gpu,
            surface,
            config,
            renderer,
            world,
            fly,
            keys: BTreeSet::new(),
            mouse: (0.0, 0.0),
            captured: false,
            hud: true,
            minimized: size.width == 0 || size.height == 0,
            last_frame: Instant::now(),
            frame_ms: 0.0,
            last_stats: None,
        };
        if live.window.has_focus() {
            live.capture(true);
        }
        Ok(live)
    }

    fn capture(&mut self, capture: bool) {
        if capture {
            let grabbed = [CursorGrabMode::Locked, CursorGrabMode::Confined]
                .into_iter()
                .any(|mode| self.window.set_cursor_grab(mode).is_ok());
            if !grabbed {
                log::warn!("cannot capture the mouse");
            }
        } else if let Err(error) = self.window.set_cursor_grab(CursorGrabMode::None) {
            log::debug!("cannot release the mouse: {error}");
        }
        self.window.set_cursor_visible(!capture);
        self.captured = capture;
        self.mouse = (0.0, 0.0);
    }

    fn resize(&mut self, size: PhysicalSize<u32>) {
        self.minimized = size.width == 0 || size.height == 0;
        if !self.minimized {
            self.config.width = size.width;
            self.config.height = size.height;
            self.surface.configure(&self.gpu.device, &self.config);
        }
    }

    fn key(&mut self, code: KeyCode) -> Result<(), Failure> {
        match code {
            KeyCode::F1 => self.hud = !self.hud,
            KeyCode::F2 => {
                let view = self.renderer.debug_view_mut().cycle();
                log::info!("debug view: {view} ({})", view.description());
            }
            KeyCode::F3 => {
                let modes = self.renderer.available_draw_modes().to_vec();
                let current = self.renderer.draw_mode();
                let next = modes
                    .iter()
                    .position(|&mode| mode == current)
                    .map_or(modes[0], |at| modes[(at + 1) % modes.len()]);
                self.renderer
                    .set_draw_mode(next)
                    .map_err(|error| Failure::new(error.to_string()))?;
                log::info!("draw path: {}", next.id());
            }
            KeyCode::F4 => self.next_camera()?,
            KeyCode::F5 => {
                if let World::Hills(hills) = &mut self.world {
                    hills.flight = match hills.flight {
                        Some(_) => None,
                        None => Some(0.0),
                    };
                }
            }
            _ => {}
        }
        Ok(())
    }

    fn next_camera(&mut self) -> Result<(), Failure> {
        match &mut self.world {
            World::Hills(hills) => {
                hills.camera = (hills.camera + 1) % hills.scene.cameras().len();
                hills.flight = None;
                let camera = hills.scene.cameras()[hills.camera];
                log::info!(
                    "camera {} `{}` (the world was generated around camera 0)",
                    hills.camera,
                    camera.name
                );
                self.fly = Fly::from_camera(&camera);
            }
            World::Structures { camera, .. } => {
                *camera = (*camera + 1) % structures::CAMERA_COUNT;
                if let Some(scene_camera) = StructureScene::camera(*camera) {
                    self.fly = Fly::from_camera(&scene_camera);
                }
            }
            World::Calib { stage, index } => {
                let scenes = sub_scenes();
                *index = (*index + 1) % scenes.len();
                let old = std::mem::take(stage);
                old.clear(&mut self.renderer, &self.gpu)?;
                let aspect = self.config.width as f32 / self.config.height.max(1) as f32;
                *stage = Stage::calib(&mut self.renderer, &self.gpu, &scenes[*index], aspect)?;
                log::info!(
                    "calibration `{}`: {}",
                    scenes[*index].name,
                    scenes[*index].description
                );
                self.fly = calib_fly(stage);
            }
        }
        Ok(())
    }

    /// Takes the chunks that arrived, moves the camera and draws one frame. Returns whether a
    /// frame was drawn.
    fn frame(&mut self) -> Result<bool, Failure> {
        let now = Instant::now();
        let dt = det::min(now.duration_since(self.last_frame).as_secs_f64(), MAX_STEP);
        let frame_ms = now.duration_since(self.last_frame).as_secs_f64() * 1.0e3;
        self.last_frame = now;
        self.frame_ms = if self.frame_ms == 0.0 {
            frame_ms
        } else {
            self.frame_ms * 0.9 + frame_ms * 0.1
        };
        self.world.receive(&mut self.renderer)?;
        self.steer(dt);
        if self.minimized {
            return Ok(false);
        }
        if self.gpu.is_lost() {
            return Err(Failure::new(format!(
                "the GPU device was lost: {}",
                self.gpu
                    .ensure_alive()
                    .err()
                    .map_or_else(String::new, |error| error.to_string())
            )));
        }
        let texture = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(texture)
            | wgpu::CurrentSurfaceTexture::Suboptimal(texture) => texture,
            wgpu::CurrentSurfaceTexture::Timeout | wgpu::CurrentSurfaceTexture::Occluded => {
                return Ok(false);
            }
            wgpu::CurrentSurfaceTexture::Outdated => {
                self.surface.configure(&self.gpu.device, &self.config);
                return Ok(false);
            }
            wgpu::CurrentSurfaceTexture::Lost => {
                self.surface = self
                    .gpu
                    .instance
                    .create_surface(Arc::clone(&self.window))
                    .map_err(|error| {
                        Failure::new(format!("cannot recreate the surface: {error}"))
                    })?;
                self.surface.configure(&self.gpu.device, &self.config);
                return Ok(false);
            }
            wgpu::CurrentSurfaceTexture::Validation => {
                return Err(Failure::new("the window's surface failed validation"));
            }
        };
        let view = texture.texture.create_view(&wgpu::TextureViewDescriptor {
            format: Some(self.config.format),
            ..wgpu::TextureViewDescriptor::default()
        });
        let camera = self.fly.scene_camera().render_camera();
        let mut lighting = daylight(self.fly.sun);
        if matches!(self.world, World::Hills(_)) {
            lighting.horizon_fade = Reach::RUN.horizon_fade();
        }
        let mut overlay = Overlay::new();
        if self.hud {
            self.draw_hud(&mut overlay);
        }
        if let World::Structures { stage, .. } = &self.world {
            stage.draw_labels(
                &mut overlay,
                &camera,
                (self.config.width, self.config.height),
            );
        }
        let input = FrameInput {
            camera: &camera,
            lighting: &lighting,
            target: &view,
            target_size: (self.config.width, self.config.height),
            overlay: (!overlay.is_empty()).then_some(&overlay),
        };
        let stats = self
            .renderer
            .render(&self.gpu, &input)
            .map_err(|error| Failure::new(format!("rendering failed: {error}")))?;
        self.last_stats = Some(stats);
        self.window.pre_present_notify();
        self.gpu.queue.present(texture);
        Ok(true)
    }

    /// Moves the camera by the input of the last `dt` seconds.
    fn steer(&mut self, dt: f64) {
        if let World::Hills(hills) = &mut self.world
            && let Some(time) = &mut hills.flight
        {
            *time += dt;
            self.fly = Fly::from_camera(&hills.scene.flight(*time));
            self.mouse = (0.0, 0.0);
            return;
        }
        self.fly.yaw =
            (self.fly.yaw + self.mouse.0 * MOUSE_RADIANS).rem_euclid(std::f64::consts::TAU);
        self.fly.pitch =
            (self.fly.pitch - self.mouse.1 * MOUSE_RADIANS).clamp(-MAX_PITCH, MAX_PITCH);
        self.mouse = (0.0, 0.0);
        let pressed = |code| self.keys.contains(&code);
        let forward = view_direction(self.fly.yaw, self.fly.pitch);
        let (sin, cos) = det::sin_cos(self.fly.yaw);
        let right = DVec3::new(cos, 0.0, sin);
        let mut motion = DVec3::ZERO;
        if pressed(KeyCode::KeyW) {
            motion += forward;
        }
        if pressed(KeyCode::KeyS) {
            motion -= forward;
        }
        if pressed(KeyCode::KeyD) {
            motion += right;
        }
        if pressed(KeyCode::KeyA) {
            motion -= right;
        }
        if pressed(KeyCode::Space) {
            motion += DVec3::Y;
        }
        if pressed(KeyCode::KeyC) || pressed(KeyCode::ControlLeft) {
            motion -= DVec3::Y;
        }
        let Some(direction) = motion.try_normalize() else {
            return;
        };
        let height = self.fly.position.y - self.world.ground(self.fly.position);
        let mut speed = BASE_SPEED * (1.0 + height.max(0.0) / 10.0);
        if pressed(KeyCode::ShiftLeft) || pressed(KeyCode::ShiftRight) {
            speed *= FAST;
        }
        self.fly.position += direction * speed * dt;
    }

    fn draw_hud(&self, overlay: &mut Overlay) {
        let mut lines = vec![format!(
            "cpe run - {}   F1 HUD  F2 view  F3 path  F4 camera  F5 flight  Esc mouse/quit",
            self.world.name()
        )];
        if let Some(progress) = self.world.progress() {
            lines.push(progress);
        }
        lines.push(format!(
            "frame {:.2} ms ({:.0} fps)",
            self.frame_ms,
            1000.0 / self.frame_ms.max(1.0e-3)
        ));
        if let Some(stats) = &self.last_stats {
            let cull = stats
                .cpu_cull_ms
                .map_or_else(String::new, |ms| format!(", cpu cull {ms:.2} ms"));
            lines.push(format!(
                "cpu render {:.2} ms{cull}, uploads {} chunks ({} pending)",
                stats.cpu_frame_ms, stats.uploads.chunks, stats.uploads.pending_chunks
            ));
            if let Some(timings) = self.renderer.latest_timings() {
                let passes: Vec<String> = timings
                    .passes
                    .iter()
                    .map(|pass| format!("{} {:.2}", pass.name, pass.gpu_ms))
                    .collect();
                lines.push(format!(
                    "gpu {:.2} ms: {}",
                    timings.gpu_total_ms(),
                    passes.join("  ")
                ));
            }
            let visible = match stats.mode {
                cp_render::voxel::DrawMode::A => self.renderer.latest_gpu_counts(),
                _ => stats.visible,
            };
            let scene = self.renderer.scene().stats();
            lines.push(format!(
                "chunks {}  draw items {}  draws {}  triangles {:.2} M  pool {:.1}/{:.1} M quads",
                stats.chunks,
                stats.draw_items,
                visible.map_or(0, |v| v.total_draws()),
                visible.map_or(0.0, |v| v.triangles() as f64 / 1.0e6),
                (scene.pool_quads - scene.pool_free_quads) as f64 / 1.0e6,
                scene.pool_quads as f64 / 1.0e6
            ));
        }
        let view = self.renderer.debug_view().current();
        lines.push(format!(
            "view {} ({})  path {}",
            view,
            view.description(),
            self.renderer.draw_mode().id()
        ));
        let p = self.fly.position;
        lines.push(format!(
            "position {:.1} {:.1} {:.1}  yaw {:.0}  pitch {:.0}  height {:.1} m",
            p.x,
            p.y,
            p.z,
            self.fly.yaw.to_degrees(),
            self.fly.pitch.to_degrees(),
            p.y - self.world.ground(p)
        ));
        let refs: Vec<&str> = lines.iter().map(String::as_str).collect();
        let scale = (self.config.height / 540).max(1);
        overlay.text_panel(8, 8, scale, &refs, OverlayColor::WHITE, OverlayColor::PANEL);
    }
}

/// The camera of a calibration stage.
fn calib_fly(stage: &Stage) -> Fly {
    let camera = stage
        .calib
        .as_ref()
        .map(|calib| calib.camera)
        .expect("a calibration stage holds its scene");
    let forward = camera.orientation * DVec3::NEG_Z;
    Fly {
        position: camera.position,
        yaw: det::atan2(forward.x, -forward.z),
        pitch: det::asin(forward.y.clamp(-1.0, 1.0)).clamp(-MAX_PITCH, MAX_PITCH),
        fov_y: f64::from(camera.fov_y),
        sun: Sun::default_from(135.0),
    }
}

impl World {
    fn start(
        renderer: &mut VoxelRenderer,
        gpu: &GpuContext,
        scene: SceneName,
        engine: &Engine,
    ) -> Result<(Self, Fly), Failure> {
        match scene {
            SceneName::Hills => {
                let hills = HillsScene::new(&engine.assets, &engine.jobs)?;
                let camera = hills.cameras()[0];
                let (bounds, planned) = hills.plan(camera.position, Reach::RUN);
                let object = renderer
                    .scene_mut()
                    .add_object(ObjectDesc {
                        bounds,
                        pose: GridPose::IDENTITY,
                    })
                    .map_err(|error| Failure::new(format!("hills object: {error}")))?;
                let requests = HillsScene::requests(&planned, object);
                log::info!(
                    "hills: generating and meshing {} planned chunks around camera 0 on {} \
                     workers",
                    requests.len(),
                    engine.jobs.threads()
                );
                // The window shows the world while it loads; the player waits for it, so all
                // workers build (the bulk limit of contract §4.2 is for streaming in play).
                let batch = BuildBatch::spawn(
                    &engine.jobs,
                    JobClass::Interactive,
                    hills.snapshot(),
                    &engine.tables,
                    &requests,
                );
                let fly = Fly::from_camera(&camera);
                Ok((
                    Self::Hills(Box::new(HillsWorld {
                        scene: hills,
                        batch,
                        report: BuildReport::default(),
                        arrived: Vec::new(),
                        last_line: Instant::now(),
                        camera: 0,
                        flight: None,
                    })),
                    fly,
                ))
            }
            SceneName::Structures => {
                let field = StructureScene::new(&engine.assets)?;
                let stage = Stage::structures(renderer, gpu, engine, &field)?;
                let camera = StructureScene::camera(0).expect("the structures have camera 0");
                Ok((
                    Self::Structures { stage, camera: 0 },
                    Fly::from_camera(&camera),
                ))
            }
            SceneName::Calib => {
                let scenes = sub_scenes();
                let stage = Stage::calib(renderer, gpu, &scenes[0], 16.0 / 9.0)?;
                let fly = calib_fly(&stage);
                Ok((Self::Calib { stage, index: 0 }, fly))
            }
        }
    }

    fn name(&self) -> String {
        match self {
            Self::Hills(hills) => format!("hills, camera {}", hills.camera),
            Self::Structures { camera, .. } => format!("structures, camera {camera}"),
            Self::Calib { index, .. } => {
                format!("calib `{}`", sub_scenes()[*index].name)
            }
        }
    }

    /// Ground height under `position` (m), for the flying speed.
    fn ground(&self, position: DVec3) -> f64 {
        match self {
            Self::Hills(hills) => hills.scene.ground(position.x, position.z),
            Self::Structures { .. } | Self::Calib { .. } => 0.0,
        }
    }

    /// The loading line, while the world is still loading.
    fn progress(&self) -> Option<String> {
        match self {
            Self::Hills(hills) if !hills.batch.is_done() => Some(format!(
                "loading: {}/{} chunks ({:.0} %), {} drawn, {:.1} s",
                hills.batch.received(),
                hills.batch.total(),
                100.0 * hills.batch.received() as f64 / hills.batch.total().max(1) as f64,
                hills.report.drawn,
                hills.batch.elapsed().as_secs_f64()
            )),
            _ => None,
        }
    }

    /// Adds the chunks that have arrived to the renderer's scene.
    fn receive(&mut self, renderer: &mut VoxelRenderer) -> Result<(), Failure> {
        let Self::Hills(hills) = self else {
            return Ok(());
        };
        if hills.batch.is_done() {
            return Ok(());
        }
        hills.batch.poll(&mut hills.arrived)?;
        for built in hills.arrived.drain(..) {
            hills.report.add(&built);
            insert(renderer.scene_mut(), built)?;
        }
        let done = hills.batch.is_done();
        if done || hills.last_line.elapsed() >= Duration::from_millis(500) {
            hills.last_line = Instant::now();
            log::info!(
                "hills: {}/{} chunks ({:.0} %), {} drawn, {:.1} s",
                hills.batch.received(),
                hills.batch.total(),
                100.0 * hills.batch.received() as f64 / hills.batch.total().max(1) as f64,
                hills.report.drawn,
                hills.batch.elapsed().as_secs_f64()
            );
        }
        if done {
            hills.report.wall_seconds = hills.batch.elapsed().as_secs_f64();
            log::info!("hills: {}", hills.report.line());
            log::info!(
                "hills: {} non-empty level-0 chunks generated and meshed at start-up",
                hills.report.drawn_level0()
            );
        }
        Ok(())
    }
}
