//! `cpe bench`: renderer measurements per sub-scene as JSON (contract §5.4, §9 E2).
//!
//! Every measured frame is rendered alone: the benchmark waits for the GPU after each frame,
//! so the timestamp queries of the frame graph give the GPU time of every pass of that frame
//! without overlap. CPU frame time is the time of the renderer's `render` call. Each
//! sub-scene first renders warm-up frames that are not measured.
//!
//! Sub-scenes: the calibration scenes of [`crate::scene::calib`]; the eight cameras of the
//! hills, each in its own world, and a flight along the flight path through the world of
//! `cpe run`; the cameras of the structures scene. For generated worlds the report also holds
//! the start-up phase: generation and meshing time per chunk, chunks per level, wall clock.
//!
//! Benchmarks measure release builds only: a debug build refuses to run.

use std::collections::BTreeMap;
use std::path::PathBuf;

use cp_render::gpu::GpuContext;
use cp_render::headless::HeadlessTarget;
use cp_render::voxel::{Camera, DrawMode, FrameInput, Lighting, VoxelRenderer};

use crate::cli::BenchArgs;
use crate::json::Json;
use crate::scene::SceneName;
use crate::scene::build::BuildReport;
use crate::scene::calib::{BASELINE, sub_scenes};
use crate::scene::hills::{HillsScene, Reach};
use crate::scene::sky::daylight;
use crate::scene::structures::{self, StructureScene};
use crate::shot::SHOT_FORMAT;
use crate::stage::{Engine, Stage, make_renderer, open_headless, renderer_config};
use crate::stats::Summary;
use crate::{Failure, write_file};

/// Frames rendered before the measured ones of every sub-scene.
pub const WARM_UP_FRAMES: u32 = 30;

/// Version of the report layout.
pub const REPORT_SCHEMA: u32 = 1;

/// What a benchmark wrote.
#[derive(Clone, Debug, PartialEq)]
pub struct BenchInfo {
    /// The JSON file.
    pub path: PathBuf,
    /// One line per sub-scene.
    pub lines: Vec<String>,
}

impl std::fmt::Display for BenchInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for line in &self.lines {
            writeln!(f, "{line}")?;
        }
        write!(f, "wrote {}", self.path.display())
    }
}

/// The measurements of one sub-scene.
#[derive(Clone, Debug, Default)]
struct Measured {
    /// GPU milliseconds per pass name.
    passes: BTreeMap<String, Vec<f64>>,
    /// GPU milliseconds of every pass together.
    gpu_total: Vec<f64>,
    /// CPU milliseconds of `render`.
    cpu_frame: Vec<f64>,
    /// CPU milliseconds of the cull of paths B and direct.
    cpu_cull: Vec<f64>,
    /// Visible draws per frame.
    draws: Vec<f64>,
    /// Visible triangles per frame.
    triangles: Vec<f64>,
    /// Frames whose timestamps did not arrive.
    missing_timings: u32,
    /// Draw items and chunks of the last frame.
    draw_items: u32,
    chunks: u32,
    mode: Option<DrawMode>,
}

/// Runs the benchmark of `args`.
///
/// # Errors
/// A [`Failure`] in a debug build, and for GPU, scene or file problems.
pub fn run(args: &BenchArgs) -> Result<BenchInfo, Failure> {
    if cfg!(debug_assertions) {
        return Err(Failure::new(
            "benchmarks measure release builds only: run `cargo run -p cp-app --release -- \
             bench ...`",
        ));
    }
    let engine = Engine::new()?;
    let gpu = open_headless()?;
    let config = renderer_config(&gpu, SHOT_FORMAT);
    let mut renderer = make_renderer(&gpu, &engine.assets, &config)?;
    if let Some(mode) = args.path {
        renderer
            .set_draw_mode(mode)
            .map_err(|error| Failure::new(error.to_string()))?;
    }
    let target = HeadlessTarget::new(&gpu, args.width, args.height, SHOT_FORMAT)
        .map_err(|error| Failure::new(format!("cannot create the benchmark target: {error}")))?;
    let mut bench = Bench {
        renderer: &mut renderer,
        gpu: &gpu,
        target: &target,
        frames: args.frames,
        sub_scenes: Vec::new(),
        startup: Vec::new(),
        lines: Vec::new(),
    };
    match args.scene {
        SceneName::Calib => bench.calib()?,
        SceneName::Hills => bench.hills(&engine)?,
        SceneName::Structures => bench.structures(&engine)?,
    }
    let Bench {
        sub_scenes,
        startup,
        lines,
        ..
    } = bench;
    let adapter = &gpu.caps.adapter;
    let report = Json::object()
        .with("schema", REPORT_SCHEMA)
        .with("tool", "cpe bench")
        .with("scene", args.scene.id())
        .with("build", "release")
        .with(
            "adapter",
            Json::object()
                .with("name", adapter.name.as_str())
                .with("backend", adapter.backend.to_string())
                .with("device_type", format!("{:?}", adapter.device_type))
                .with("driver", adapter.driver.as_str())
                .with("driver_info", adapter.driver_info.as_str()),
        )
        .with("quality_tier", gpu.caps.tier.id())
        .with("draw_path", renderer.draw_mode().id())
        .with(
            "resolution",
            vec![Json::from(args.width), Json::from(args.height)],
        )
        .with("frames_per_sub_scene", args.frames)
        .with("warm_up_frames", WARM_UP_FRAMES)
        .with(
            "method",
            "every frame is rendered alone (the CPU waits for the GPU after each frame); GPU \
             times are timestamp queries per pass; statistics are mean and nearest-rank \
             percentiles over the measured frames",
        )
        .with("sub_scenes", sub_scenes)
        .with("startup", startup);
    write_file(&args.out, report.render().as_bytes())?;
    Ok(BenchInfo {
        path: args.out.clone(),
        lines,
    })
}

/// The state of one benchmark run.
struct Bench<'a> {
    renderer: &'a mut VoxelRenderer,
    gpu: &'a GpuContext,
    target: &'a HeadlessTarget,
    frames: u32,
    sub_scenes: Vec<Json>,
    startup: Vec<Json>,
    lines: Vec<String>,
}

impl Bench<'_> {
    fn calib(&mut self) -> Result<(), Failure> {
        let size = self.target.size();
        let aspect = size.0 as f32 / size.1 as f32;
        let mut gbuffer_means = Vec::new();
        let scenes = sub_scenes();
        for sub in &scenes {
            let stage = Stage::calib(self.renderer, self.gpu, sub, aspect)?;
            let calib = stage
                .calib
                .as_ref()
                .expect("a calibration stage holds its scene");
            let camera = calib.camera;
            let expected = calib.expected;
            let lighting = daylight(crate::scene::sky::Sun::default_from(135.0));
            let measured = self.measure(&|_| camera, &lighting)?;
            let triangles = Summary::of(&measured.triangles);
            let draws = Summary::of(&measured.draws);
            if triangles.max != triangles.min || triangles.max as u64 != expected.triangles {
                log::warn!(
                    "calibration `{}`: {} triangles drawn, {} expected",
                    sub.name,
                    triangles.max,
                    expected.triangles
                );
            }
            let gbuffer = Summary::of(
                measured
                    .passes
                    .get("gbuffer")
                    .map_or(&[][..], Vec::as_slice),
            );
            gbuffer_means.push(gbuffer.mean);
            let mut json = measured.json(sub.name, sub.description);
            json = json
                .with("expected_triangles", expected.triangles)
                .with("expected_draws", expected.draws)
                .with("resident_chunks", sub.params.resident_chunks)
                .with(
                    "resident_quads",
                    u64::from(sub.params.resident_chunks) * u64::from(sub.params.quads_per_chunk),
                );
            self.lines.push(format!(
                "{:<14} {:>8} draws {:>6.2} M triangles: G-buffer {:.3} ms mean / {:.3} p95, cull \
                 {}, CPU frame {:.3} ms",
                sub.name,
                draws.max,
                triangles.max / 1.0e6,
                gbuffer.mean,
                gbuffer.p95,
                measured.passes.get("cull").map_or_else(
                    || "-".to_owned(),
                    |cull| format!("{:.4} ms", Summary::of(cull).mean)
                ),
                Summary::of(&measured.cpu_frame).mean
            ));
            self.sub_scenes.push(json);
            stage.clear(self.renderer, self.gpu)?;
        }
        // Nanoseconds per indirect draw: the one-quad scenes against the one-draw baseline.
        let baseline = gbuffer_means[BASELINE];
        for (index, sub) in scenes.iter().enumerate() {
            if index <= BASELINE || sub.params.quads_per_chunk != 1 {
                continue;
            }
            let draws = f64::from(sub.params.visible_chunks - 1);
            let per_draw = (gbuffer_means[index] - baseline) * 1.0e6 / draws;
            let json = std::mem::replace(&mut self.sub_scenes[index], Json::Null);
            self.sub_scenes[index] = json.with("ns_per_draw", per_draw).with(
                "ns_per_draw_method",
                "(G-buffer mean - one-draw baseline mean) / (draws - 1)",
            );
            self.lines.push(format!(
                "{}: {:.1} ns per indirect draw",
                sub.name, per_draw
            ));
        }
        Ok(())
    }

    fn hills(&mut self, engine: &Engine) -> Result<(), Failure> {
        let hills = HillsScene::new(&engine.assets, &engine.jobs)?;
        for (index, camera) in hills.cameras().iter().enumerate() {
            let stage = Stage::hills(
                self.renderer,
                self.gpu,
                engine,
                &hills,
                camera.position,
                HillsScene::reach(index),
            )?;
            let render_camera = camera.render_camera();
            let mut lighting = daylight(camera.sun);
            lighting.horizon_fade = HillsScene::reach(index).horizon_fade();
            let measured = self.measure(&|_| render_camera, &lighting)?;
            let name = format!("camera_{index}_{}", camera.name);
            self.finish(
                &name,
                "a fixed camera of the hills",
                measured,
                stage.report.as_ref(),
            );
            stage.clear(self.renderer, self.gpu)?;
        }
        // The flight path through the world of `cpe run`, spread over the measured frames.
        let start = hills.cameras()[0];
        let stage = Stage::hills(
            self.renderer,
            self.gpu,
            engine,
            &hills,
            start.position,
            Reach::RUN,
        )?;
        let frames = self.frames;
        let mut lighting = daylight(start.sun);
        lighting.horizon_fade = Reach::RUN.horizon_fade();
        let path = |frame: u32| {
            let time = 150.0 * f64::from(frame.saturating_sub(WARM_UP_FRAMES)) / f64::from(frames);
            hills.flight(time).render_camera()
        };
        let measured = self.measure(&path, &lighting)?;
        self.finish(
            "flight",
            "the flight path (one loop) through the world of `cpe run` (level 0 to 1 km)",
            measured,
            stage.report.as_ref(),
        );
        stage.clear(self.renderer, self.gpu)
    }

    fn structures(&mut self, engine: &Engine) -> Result<(), Failure> {
        let scene = StructureScene::new(&engine.assets)?;
        let stage = Stage::structures(self.renderer, self.gpu, engine, &scene)?;
        for index in 0..structures::CAMERA_COUNT {
            let camera = StructureScene::camera(index).expect("the structures have this camera");
            let render_camera = camera.render_camera();
            let lighting = daylight(camera.sun);
            let measured = self.measure(&|_| render_camera, &lighting)?;
            let report = (index == 0).then_some(stage.report.as_ref()).flatten();
            self.finish(
                &format!("camera_{index}_{}", camera.name),
                "a camera of the structure field",
                measured,
                report,
            );
        }
        stage.clear(self.renderer, self.gpu)
    }

    /// Records a measured sub-scene and, if given, the start-up of its world.
    fn finish(
        &mut self,
        name: &str,
        description: &str,
        measured: Measured,
        report: Option<&BuildReport>,
    ) {
        let gpu = Summary::of(&measured.gpu_total);
        let gbuffer = Summary::of(
            measured
                .passes
                .get("gbuffer")
                .map_or(&[][..], Vec::as_slice),
        );
        self.lines.push(format!(
            "{name:<22} {:>6.2} M triangles {:>7.0} draws: GPU {:.3} ms mean / {:.3} p99 (G-buffer \
             {:.3}), CPU frame {:.3} ms",
            Summary::of(&measured.triangles).mean / 1.0e6,
            Summary::of(&measured.draws).mean,
            gpu.mean,
            gpu.p99,
            gbuffer.mean,
            Summary::of(&measured.cpu_frame).mean
        ));
        self.sub_scenes.push(measured.json(name, description));
        if let Some(report) = report {
            self.lines
                .push(format!("{name}: start-up {}", report.line()));
            self.startup.push(startup_json(name, report));
        }
    }

    /// Renders the warm-up and the measured frames; `camera(frame)` places the camera.
    fn measure(
        &mut self,
        camera: &dyn Fn(u32) -> Camera,
        lighting: &Lighting,
    ) -> Result<Measured, Failure> {
        let mut measured = Measured::default();
        let fail =
            |error: cp_render::RenderError| Failure::new(format!("rendering failed: {error}"));
        for frame in 0..WARM_UP_FRAMES + self.frames {
            let camera = camera(frame);
            let input = FrameInput {
                camera: &camera,
                lighting,
                target: self.target.view(),
                target_size: self.target.size(),
                overlay: None,
            };
            let stats = self.renderer.render(self.gpu, &input).map_err(fail)?;
            self.renderer.wait_for_gpu(self.gpu).map_err(fail)?;
            if frame < WARM_UP_FRAMES {
                continue;
            }
            match self
                .renderer
                .latest_timings()
                .filter(|timings| timings.frame == stats.frame)
            {
                Some(timings) => {
                    for pass in &timings.passes {
                        measured
                            .passes
                            .entry(pass.name.clone())
                            .or_default()
                            .push(pass.gpu_ms);
                    }
                    measured.gpu_total.push(timings.gpu_total_ms());
                }
                None => measured.missing_timings += 1,
            }
            measured.cpu_frame.push(stats.cpu_frame_ms);
            if let Some(ms) = stats.cpu_cull_ms {
                measured.cpu_cull.push(ms);
            }
            let visible = match stats.mode {
                DrawMode::A => self.renderer.latest_gpu_counts(),
                DrawMode::B | DrawMode::Direct => stats.visible,
            };
            if let Some(visible) = visible {
                measured.draws.push(f64::from(visible.total_draws()));
                measured.triangles.push(visible.triangles() as f64);
            }
            measured.draw_items = stats.draw_items;
            measured.chunks = stats.chunks;
            measured.mode = Some(stats.mode);
        }
        if measured.missing_timings > 0 {
            log::warn!(
                "{} measured frames had no GPU timings",
                measured.missing_timings
            );
        }
        Ok(measured)
    }
}

impl Measured {
    fn json(&self, name: &str, description: &str) -> Json {
        let mut passes = Json::object();
        for (pass, values) in &self.passes {
            passes = passes.with(pass, summary_json(&Summary::of(values)));
        }
        passes = passes.with("total", summary_json(&Summary::of(&self.gpu_total)));
        Json::object()
            .with("name", name)
            .with("description", description)
            .with("draw_path", self.mode.map(DrawMode::id))
            .with("chunks", self.chunks)
            .with("draw_items", self.draw_items)
            .with("triangles", summary_json(&Summary::of(&self.triangles)))
            .with("draws", summary_json(&Summary::of(&self.draws)))
            .with("gpu_ms", passes)
            .with("cpu_frame_ms", summary_json(&Summary::of(&self.cpu_frame)))
            .with(
                "cpu_cull_ms",
                (!self.cpu_cull.is_empty()).then(|| summary_json(&Summary::of(&self.cpu_cull))),
            )
            .with("frames_without_timings", self.missing_timings)
    }
}

fn summary_json(summary: &Summary) -> Json {
    Json::object()
        .with("count", summary.count)
        .with("mean", summary.mean)
        .with("p50", summary.p50)
        .with("p95", summary.p95)
        .with("p99", summary.p99)
        .with("min", summary.min)
        .with("max", summary.max)
}

fn startup_json(name: &str, report: &BuildReport) -> Json {
    Json::object()
        .with("name", name)
        .with("chunks_planned", report.requested)
        .with("empty_by_class", report.empty_class)
        .with("empty_meshes", report.empty_mesh)
        .with("drawn", report.drawn)
        .with("drawn_level0", report.drawn_level0())
        .with(
            "drawn_per_level",
            report
                .drawn_per_level
                .iter()
                .map(|&count| Json::from(count))
                .collect::<Vec<_>>(),
        )
        .with("quads", report.quads)
        .with("generate_ms", summary_json(&report.generate_summary()))
        .with("mesh_ms", summary_json(&report.mesh_summary()))
        .with("wall_seconds", report.wall_seconds)
        .with(
            "chunks_per_second",
            report.requested as f64 / report.wall_seconds.max(1.0e-9),
        )
}
