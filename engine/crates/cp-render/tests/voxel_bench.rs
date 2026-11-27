//! Calibration benchmark of the voxel renderer (contract §9, E2): G-buffer time at 1.0 M
//! visible triangles, cull time and nanoseconds per indirect draw, measured with the
//! calibration scene of `cp_render::voxel` at 1920x1080.
//!
//! Ignored by default; run in release:
//! `cargo test -p cp-render --release --test voxel_bench -- --ignored --nocapture`.
//! GPU times come from the frame graph's timestamp queries; every frame waits for the GPU so
//! that each measured frame runs alone.

use std::sync::Once;

use cp_render::gpu::{GpuContext, GpuError, GpuOptions};
use cp_render::headless::HeadlessTarget;
use cp_render::material::{GpuMaterial, MaterialTable};
use cp_render::texture::{TextureArrayBuilder, TextureArrayKind};
use cp_render::voxel::{
    CalibrationParams, CalibrationScene, DrawMode, FrameInput, Lighting, PASS_CULL, PASS_GBUFFER,
    VoxelRenderer, VoxelRendererConfig,
};
use cp_render::wgpu;

struct BenchLogger;

static LOGGER: BenchLogger = BenchLogger;

impl log::Log for BenchLogger {
    fn enabled(&self, metadata: &log::Metadata<'_>) -> bool {
        metadata.level() <= log::Level::Error
            || (metadata.level() <= log::Level::Info
                && (metadata.target().starts_with("cp_render")
                    || metadata.target().starts_with("voxel_bench")))
    }

    // Libraries stay silent (contract §2.6); the benchmark binary prints its results.
    #[allow(clippy::print_stderr)]
    fn log(&self, record: &log::Record<'_>) {
        if self.enabled(record.metadata()) {
            eprintln!("[{} {}] {}", record.level(), record.target(), record.args());
        }
    }

    fn flush(&self) {}
}

fn init_logging() {
    static INIT: Once = Once::new();
    INIT.call_once(|| {
        log::set_logger(&LOGGER).expect("no other logger is installed in this test binary");
        log::set_max_level(log::LevelFilter::Info);
    });
}

/// Median of the measured values.
fn median(values: &mut [f64]) -> f64 {
    values.sort_by(f64::total_cmp);
    values[values.len() / 2]
}

#[derive(Debug)]
struct Measurement {
    gbuffer_ms: f64,
    cull_ms: Option<f64>,
    cpu_frame_ms: f64,
    cpu_cull_ms: Option<f64>,
}

/// Renders `frames` frames of `scene` with `mode` after a warm-up and returns the medians.
fn measure(
    renderer: &mut VoxelRenderer,
    gpu: &GpuContext,
    target: &HeadlessTarget,
    scene: &CalibrationScene,
    mode: DrawMode,
    frames: usize,
) -> Measurement {
    renderer.set_draw_mode(mode).expect("the mode is available");
    let lighting = Lighting::default();
    let input = FrameInput {
        camera: &scene.camera,
        lighting: &lighting,
        target: target.view(),
        target_size: target.size(),
        overlay: None,
    };
    let mut gbuffer = Vec::new();
    let mut cull = Vec::new();
    let mut cpu_frame = Vec::new();
    let mut cpu_cull = Vec::new();
    for frame in 0..frames + 10 {
        let stats = renderer
            .render(gpu, &input)
            .expect("render a calibration frame");
        renderer.wait_for_gpu(gpu).expect("wait for the GPU");
        // The counts prove that the frame drew what the scene promises.
        let visible = match mode {
            DrawMode::A => renderer
                .latest_gpu_counts()
                .expect("the counters arrive after a wait"),
            _ => stats.visible.expect("CPU paths count at once"),
        };
        assert_eq!(visible.total_draws(), scene.expected.draws);
        assert_eq!(visible.triangles(), scene.expected.triangles);
        if frame < 10 {
            continue;
        }
        let timings = renderer.latest_timings().expect("timestamps are enabled");
        assert_eq!(
            timings.frame, stats.frame,
            "the timings of this frame arrived"
        );
        gbuffer.push(
            timings
                .pass(PASS_GBUFFER)
                .expect("the G-buffer pass is timed")
                .gpu_ms,
        );
        if let Some(pass) = timings.pass(PASS_CULL) {
            cull.push(pass.gpu_ms);
        }
        cpu_frame.push(stats.cpu_frame_ms);
        if let Some(ms) = stats.cpu_cull_ms {
            cpu_cull.push(ms);
        }
    }
    Measurement {
        gbuffer_ms: median(&mut gbuffer),
        cull_ms: (!cull.is_empty()).then(|| median(&mut cull)),
        cpu_frame_ms: median(&mut cpu_frame),
        cpu_cull_ms: (!cpu_cull.is_empty()).then(|| median(&mut cpu_cull)),
    }
}

#[test]
#[ignore = "benchmark: run in release with --ignored --nocapture"]
fn calibration_bench() {
    init_logging();
    let gpu = match GpuContext::new_blocking(&GpuOptions::default(), None) {
        Ok(gpu) => gpu,
        Err(error @ (GpuError::NoAdapter { .. } | GpuError::BackendNotCompiled { .. })) => {
            log::warn!("calibration_bench: skipped: {error}");
            return;
        }
        Err(error) => panic!("GPU context creation failed: {error}"),
    };
    if cfg!(debug_assertions) {
        log::warn!("calibration_bench: debug build; benchmark figures need --release");
    }
    let format = wgpu::TextureFormat::Rgba8UnormSrgb;
    let mut config = VoxelRendererConfig::for_tier(gpu.caps.tier, format);
    config.upload_budget_bytes = u64::MAX;
    // Sixteen 16x16 layers of noise, the size of the base pack's textures.
    let mut builder = TextureArrayBuilder::new("bench", TextureArrayKind::Albedo, 16, 16).unwrap();
    for layer in 0..16_u32 {
        let mut levels = Vec::new();
        for level in 0..5_u32 {
            let side = 16 >> level;
            let texels: Vec<u8> = (0..side * side)
                .flat_map(|texel| {
                    let value = ((texel * 37 + layer * 11) % 64) as u8;
                    [96 + value, 120 + value / 2, 80 + value, 255]
                })
                .collect();
            levels.push(texels);
        }
        let slices: Vec<&[u8]> = levels.iter().map(Vec::as_slice).collect();
        builder.push_layer(&slices).unwrap();
    }
    let albedo = builder.build(&gpu).unwrap();
    let materials: Vec<GpuMaterial> = (0..16)
        .map(|layer| GpuMaterial {
            average_color: [0.3, 0.4, 0.25],
            roughness: 0.8,
            texture_layer: layer,
            ..GpuMaterial::default()
        })
        .collect();
    let materials = MaterialTable::new(&gpu, &materials).unwrap();
    let mut renderer = VoxelRenderer::new(&gpu, &config, materials, albedo).unwrap();
    let target = HeadlessTarget::new(&gpu, 1920, 1080, format).unwrap();
    let modes = renderer.available_draw_modes().to_vec();
    log::info!(
        "calibration_bench on {} ({}), 1920x1080, modes {:?}, startup {:.1} ms for {} pipelines",
        gpu.caps.adapter.name,
        gpu.caps.adapter.backend,
        modes,
        renderer.startup_ms(),
        renderer.pipeline_count()
    );
    let cases = [
        ("baseline: 1 one-quad draw", 1, 1, 1),
        ("1.0 M triangles, 5,000 draws", 5_000, 5_000, 100),
        ("1.0 M triangles, 20,000 draws", 20_000, 20_000, 25),
        ("5,000 one-quad draws", 5_000, 5_000, 1),
        ("20,000 one-quad draws", 20_000, 20_000, 1),
        ("20,000 items, none visible", 20_000, 0, 1),
    ];
    // G-buffer time of the one-draw scene per mode: the fixed cost of the pass (clears,
    // attachments), subtracted for the cost per draw.
    let mut baseline_ms = std::collections::BTreeMap::new();
    for (name, resident, visible, quads) in cases {
        let params = CalibrationParams {
            resident_chunks: resident,
            visible_chunks: visible,
            quads_per_chunk: quads,
            materials: 16,
            ..CalibrationParams::default()
        };
        let scene = CalibrationScene::build(renderer.scene_mut(), &params).unwrap();
        renderer.scene_mut().flush_all_uploads(&gpu);
        for &mode in &modes {
            let result = measure(&mut renderer, &gpu, &target, &scene, mode, 60);
            if visible == 1 {
                baseline_ms.insert(mode.id(), result.gbuffer_ms);
            }
            let per_draw = match baseline_ms.get(mode.id()) {
                Some(&base) if quads == 1 && visible > 1 => format!(
                    ", {:.1} ns per draw ((G-buffer - baseline) / draws), {:.1} ns per draw \
                     (G-buffer / draws)",
                    (result.gbuffer_ms - base) * 1.0e6 / f64::from(visible - 1),
                    result.gbuffer_ms * 1.0e6 / f64::from(visible)
                ),
                _ => String::new(),
            };
            log::info!(
                "{name} [{}]: G-buffer {:.3} ms, cull {} ms (GPU), CPU cull {} ms, CPU frame {:.3} \
                 ms{per_draw}",
                mode.id(),
                result.gbuffer_ms,
                result
                    .cull_ms
                    .map_or("-".to_owned(), |ms| format!("{ms:.4}")),
                result
                    .cpu_cull_ms
                    .map_or("-".to_owned(), |ms| format!("{ms:.3}")),
                result.cpu_frame_ms,
            );
        }
        scene.remove(renderer.scene_mut()).unwrap();
        renderer.scene_mut().flush_all_uploads(&gpu);
    }
    log::info!("calibration_bench: baseline G-buffer time per mode: {baseline_ms:?}");
}
