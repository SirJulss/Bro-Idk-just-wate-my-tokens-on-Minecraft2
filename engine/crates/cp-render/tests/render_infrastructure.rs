//! Tests of the renderer infrastructure against the real GPU, through the public API only:
//! headless rendering and read-back, the overlay, the frame graph with its pool and timings,
//! shader and pipeline creation on every backend, texture arrays and the material table.
//!
//! Every test skips with a log line when the machine has no GPU adapter (contract §2.7). Run
//! with `-- --nocapture` to see the measured pass times.

use std::sync::Once;

use cp_render::RenderError;
use cp_render::frame_graph::{
    FrameGraph, FrameGraphDesc, FrameGraphError, PassId, TransientDesc, TransientId, TransientSize,
};
use cp_render::gpu::{BACKEND_ENV, GpuContext, GpuError, GpuOptions};
use cp_render::headless::{HEADLESS_FORMATS, HeadlessError, HeadlessTarget};
use cp_render::material::{GpuMaterial, MaterialTable, TextureTurn};
use cp_render::overlay::{Overlay, OverlayColor, OverlayRenderer, font};
use cp_render::scope::capture;
use cp_render::shader::{ShaderLibrary, ShaderTarget, create_shader_module};
use cp_render::texture::{TextureArrayBuilder, TextureArrayKind, TextureError, mip_size};
use cp_render::{WgpuErrorKind, wgpu};

/// Writes the log of the crate under test and of these tests to stderr, where the harness
/// shows it for failed tests and with `--nocapture`.
struct TestLogger;

static LOGGER: TestLogger = TestLogger;

impl log::Log for TestLogger {
    fn enabled(&self, metadata: &log::Metadata<'_>) -> bool {
        let own = ["cp_render", "render_infrastructure"];
        metadata.level() <= log::Level::Error
            || (metadata.level() <= log::Level::Info
                && own
                    .iter()
                    .any(|prefix| metadata.target().starts_with(prefix)))
    }

    // The harness captures what the print macros write; contract §2.6 keeps libraries silent,
    // so the test binary is where the log is printed.
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

/// The context of the default adapter, or `None` after a log line when there is none.
fn context_or_skip(test: &str) -> Option<GpuContext> {
    init_logging();
    match GpuContext::new_blocking(&GpuOptions::default(), None) {
        Ok(gpu) => Some(gpu),
        Err(error @ (GpuError::NoAdapter { .. } | GpuError::BackendNotCompiled { .. })) => {
            log::warn!("{test}: skipped: {error}");
            None
        }
        Err(error) => panic!("{test}: GPU context creation failed: {error}"),
    }
}

/// One context per backend of the contract that has an adapter here (with `CP_BACKEND` set,
/// the one context of the forced backend).
fn contexts_per_backend(test: &str) -> Vec<GpuContext> {
    init_logging();
    if std::env::var_os(BACKEND_ENV).is_some() {
        return context_or_skip(test).into_iter().collect();
    }
    let mut contexts = Vec::new();
    for backends in [wgpu::Backends::VULKAN, wgpu::Backends::DX12] {
        let options = GpuOptions {
            backends: Some(backends),
            ..GpuOptions::default()
        };
        match GpuContext::new_blocking(&options, None) {
            Ok(gpu) => contexts.push(gpu),
            Err(error @ (GpuError::NoAdapter { .. } | GpuError::BackendNotCompiled { .. })) => {
                log::warn!("{test}: {backends:?} skipped: {error}");
            }
            Err(error) => panic!("{test}: GPU context creation on {backends:?} failed: {error}"),
        }
    }
    contexts
}

fn encoder(gpu: &GpuContext, label: &str) -> wgpu::CommandEncoder {
    gpu.device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some(label) })
}

/// A colour attachment that clears `view` to `color` and stores the result.
fn clear_attachment(
    view: &wgpu::TextureView,
    color: wgpu::Color,
) -> wgpu::RenderPassColorAttachment<'_> {
    wgpu::RenderPassColorAttachment {
        view,
        depth_slice: None,
        resolve_target: None,
        ops: wgpu::Operations {
            load: wgpu::LoadOp::Clear(color),
            store: wgpu::StoreOp::Store,
        },
    }
}

/// `bytes` as a clear colour of a unorm target (each byte over 255).
fn unorm_color(bytes: [u8; 4]) -> wgpu::Color {
    let [r, g, b, a] = bytes.map(|byte| f64::from(byte) / 255.0);
    wgpu::Color { r, g, b, a }
}

/// What the overlay must produce: `clear` everywhere, then the rectangle, then the ink of the
/// text, computed from the font data on the CPU.
struct OverlayScene {
    width: u32,
    height: u32,
    clear: [u8; 4],
    rect: (i32, i32, u32, u32, [u8; 4]),
    text: (i32, i32, u32, [u8; 4], &'static str),
}

impl OverlayScene {
    fn overlay(&self) -> Overlay {
        let mut overlay = Overlay::new();
        let (x, y, width, height, color) = self.rect;
        overlay.rect(x, y, width, height, OverlayColor(color));
        let (x, y, scale, color, text) = self.text;
        overlay.text(x, y, scale, OverlayColor(color), text);
        overlay
    }

    fn expected(&self) -> Vec<[u8; 4]> {
        let width = self.width as i32;
        let mut pixels = vec![self.clear; (self.width * self.height) as usize];
        let mut set = |x: i32, y: i32, color: [u8; 4]| {
            if (0..width).contains(&x) && (0..self.height as i32).contains(&y) {
                pixels[(y * width + x) as usize] = color;
            }
        };
        let (x0, y0, rect_width, rect_height, color) = self.rect;
        for y in y0..y0 + rect_height as i32 {
            for x in x0..x0 + rect_width as i32 {
                set(x, y, color);
            }
        }
        let (x0, y0, scale, color, text) = self.text;
        let scale = scale as i32;
        let advance = font::GLYPH_ADVANCE as i32 * scale;
        for (column, character) in (0..).zip(text.chars()) {
            let glyph = font::glyph_index(character);
            for gy in 0..font::GLYPH_HEIGHT {
                for gx in 0..font::GLYPH_WIDTH {
                    if !font::glyph_pixel(glyph, gx, gy) {
                        continue;
                    }
                    for sy in 0..scale {
                        for sx in 0..scale {
                            set(
                                x0 + column * advance + gx as i32 * scale + sx,
                                y0 + gy as i32 * scale + sy,
                                color,
                            );
                        }
                    }
                }
            }
        }
        pixels
    }
}

#[test]
fn a_clear_colour_and_overlay_text_are_read_back_headlessly() {
    let test = "a_clear_colour_and_overlay_text_are_read_back_headlessly";
    for gpu in contexts_per_backend(test) {
        for format in HEADLESS_FORMATS {
            check_overlay_scene(&gpu, format);
        }
        log::info!(
            "{test}: matches the CPU reference on {} for {HEADLESS_FORMATS:?} (exact for \
             unorm, within 1 for sRGB)",
            gpu.caps.adapter.backend
        );
    }
}

/// Renders the overlay test scene into a headless target of `format` and compares every
/// pixel with the CPU reference.
fn check_overlay_scene(gpu: &GpuContext, format: wgpu::TextureFormat) {
    let backend = gpu.caps.adapter.backend;
    let srgb = format.is_srgb();
    let scene = OverlayScene {
        width: 96,
        height: 40,
        // Black and white are exact in both encodings; mid tones only in unorm.
        clear: if srgb {
            [0, 0, 0, 255]
        } else {
            [10, 20, 30, 255]
        },
        // The rectangle reaches under the first glyphs: later draws are on top.
        rect: (2, 10, 20, 6, [200, 40, 90, 255]),
        text: (4, 12, 2, [250, 240, 5, 255], "Hi! g_"),
    };
    let target = HeadlessTarget::new(gpu, scene.width, scene.height, format)
        .unwrap_or_else(|error| panic!("{backend} {format:?}: {error}"));
    let mut renderer = OverlayRenderer::new(gpu, format)
        .unwrap_or_else(|error| panic!("{backend} {format:?}: {error}"));
    assert_eq!(renderer.target_format(), format);
    let overlay = scene.overlay();
    // One rectangle and the five visible glyphs of "Hi! g_".
    assert_eq!(overlay.len(), 6);
    renderer
        .prepare(gpu, &overlay, target.size())
        .expect("upload the overlay");

    let mut commands = encoder(gpu, "overlay test");
    {
        let mut pass = commands.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("overlay test"),
            color_attachments: &[Some(clear_attachment(
                target.view(),
                unorm_color(scene.clear),
            ))],
            ..Default::default()
        });
        renderer.draw(&mut pass);
    }
    target.encode_readback(&mut commands);
    gpu.queue.submit([commands.finish()]);
    let image = target.read_pixels(gpu).expect("read the target back");
    assert_eq!((image.width, image.height, image.srgb), (96, 40, srgb));
    assert_eq!(image.pixels.len(), 96 * 40 * 4);

    // sRGB targets go through a conversion to linear and back: one step of rounding.
    let tolerance = u8::from(srgb);
    let expected = scene.expected();
    let mut wrong = Vec::new();
    for y in 0..scene.height {
        for x in 0..scene.width {
            let got = image.pixel(x, y);
            let want = expected[(y * scene.width + x) as usize];
            let close = got
                .iter()
                .zip(want)
                .all(|(&got, want)| got.abs_diff(want) <= tolerance);
            if !close {
                wrong.push(format!("({x}, {y}): {got:?} instead of {want:?}"));
            }
        }
    }
    assert!(
        wrong.is_empty(),
        "{backend} {format:?}: {} pixels differ, first: {:?}",
        wrong.len(),
        &wrong[..wrong.len().min(8)]
    );
    // The scene really has all three colours.
    for color in [scene.clear, scene.rect.4, scene.text.3] {
        assert!(expected.contains(&color));
    }
}

#[test]
fn overlay_frames_can_grow_and_reuse_the_renderer() {
    let Some(gpu) = context_or_skip("overlay_frames_can_grow_and_reuse_the_renderer") else {
        return;
    };
    let format = wgpu::TextureFormat::Rgba8Unorm;
    let target = HeadlessTarget::new(&gpu, 64, 64, format).unwrap();
    let mut renderer = OverlayRenderer::new(&gpu, format).unwrap();
    for count in [0_u32, 3, 1000] {
        let mut overlay = Overlay::new();
        for index in 0..count {
            overlay.rect((index % 64) as i32, 0, 1, 1, OverlayColor::WHITE);
        }
        renderer.prepare(&gpu, &overlay, target.size()).unwrap();
        let mut commands = encoder(&gpu, "overlay growth");
        {
            let mut pass = commands.begin_render_pass(&wgpu::RenderPassDescriptor {
                color_attachments: &[Some(clear_attachment(target.view(), wgpu::Color::BLACK))],
                ..Default::default()
            });
            renderer.draw(&mut pass);
        }
        gpu.queue.submit([commands.finish()]);
        let image = target.capture_image(&gpu).unwrap();
        let lit = (0..64).filter(|&x| image.pixel(x, 0)[0] == 255).count();
        assert_eq!(lit, count.min(64) as usize, "{count} rectangles");
        assert_eq!(image.pixel(0, 1), [0, 0, 0, 255]);
    }
}

#[test]
fn headless_targets_reject_formats_and_sizes_they_cannot_read_back() {
    let Some(gpu) = context_or_skip("headless_targets_reject_formats_and_sizes") else {
        return;
    };
    for format in [
        wgpu::TextureFormat::Rgba16Float,
        wgpu::TextureFormat::R8Unorm,
        wgpu::TextureFormat::Depth32Float,
    ] {
        assert!(!HEADLESS_FORMATS.contains(&format));
        assert!(matches!(
            HeadlessTarget::new(&gpu, 4, 4, format),
            Err(RenderError::Headless(
                HeadlessError::UnsupportedFormat { .. }
            ))
        ));
    }
    let limit = gpu.caps.enabled_limits.max_texture_dimension_2d;
    for (width, height) in [(0, 4), (4, 0), (limit + 1, 1)] {
        assert!(matches!(
            HeadlessTarget::new(&gpu, width, height, wgpu::TextureFormat::Rgba8Unorm),
            Err(RenderError::Headless(HeadlessError::InvalidSize { .. }))
        ));
    }
    // A width whose rows need padding reads back unpadded.
    let target = HeadlessTarget::new(&gpu, 3, 2, wgpu::TextureFormat::Rgba8Unorm).unwrap();
    let mut commands = encoder(&gpu, "odd width");
    commands.begin_render_pass(&wgpu::RenderPassDescriptor {
        color_attachments: &[Some(clear_attachment(
            target.view(),
            unorm_color([1, 2, 3, 4]),
        ))],
        ..Default::default()
    });
    gpu.queue.submit([commands.finish()]);
    let image = target.capture_image(&gpu).unwrap();
    assert_eq!(image.pixels, [1, 2, 3, 4].repeat(6));
}

/// A graph of two render passes that only clear: `scene` (full size) and `bloom` (half size,
/// declared to read `scene`).
struct ClearingGraph {
    desc: FrameGraphDesc,
    scene: TransientId,
    bloom: TransientId,
    scene_pass: PassId,
    bloom_pass: PassId,
}

fn clearing_graph() -> ClearingGraph {
    let mut desc = FrameGraphDesc::new();
    let scene = desc.transient(
        TransientDesc::new(
            "scene",
            wgpu::TextureFormat::Rgba8Unorm,
            TransientSize::FULL,
        )
        .with_usage(wgpu::TextureUsages::COPY_SRC),
    );
    let bloom = desc.transient(TransientDesc::new(
        "bloom",
        wgpu::TextureFormat::Rgba16Float,
        TransientSize::HALF,
    ));
    let scene_pass = desc.render_pass("scene", &[], &[scene]);
    let bloom_pass = desc.render_pass("bloom", &[scene], &[bloom]);
    ClearingGraph {
        desc,
        scene,
        bloom,
        scene_pass,
        bloom_pass,
    }
}

/// Records and submits one frame of a [`ClearingGraph`]; returns the textures it used.
fn run_clearing_frame(
    graph: &mut FrameGraph,
    ids: &ClearingGraph,
    gpu: &GpuContext,
) -> [wgpu::Texture; 2] {
    let mut frame = graph.begin_frame(gpu).expect("begin the frame");
    let textures = [frame.texture(ids.scene), frame.texture(ids.bloom)];
    for (pass, transient) in [(ids.scene_pass, ids.scene), (ids.bloom_pass, ids.bloom)] {
        let view = frame.view(transient);
        let recorded = frame
            .begin_render_pass(
                pass,
                &wgpu::RenderPassDescriptor {
                    color_attachments: &[Some(clear_attachment(&view, wgpu::Color::RED))],
                    ..Default::default()
                },
            )
            .expect("passes in declaration order");
        drop(recorded);
    }
    frame.submit().expect("submit the frame");
    textures
}

#[test]
fn the_frame_graph_reuses_transients_and_recreates_them_on_resize() {
    let test = "the_frame_graph_reuses_transients_and_recreates_them_on_resize";
    let Some(gpu) = context_or_skip(test) else {
        return;
    };
    let ids = clearing_graph();
    let (scene, bloom) = (ids.scene, ids.bloom);
    let mut graph = FrameGraph::new(&gpu, ids.desc.clone(), (320, 200)).unwrap();
    assert_eq!(graph.pool_stats().textures, 0);
    let names: Vec<&str> = graph
        .desc()
        .passes()
        .iter()
        .map(|pass| pass.name.as_str())
        .collect();
    assert_eq!(names, ["scene", "bloom"]);

    let first = run_clearing_frame(&mut graph, &ids, &gpu);
    let stats = graph.pool_stats();
    assert_eq!((stats.textures, stats.created_total), (2, 2));
    assert_eq!(stats.bytes, 320 * 200 * 4 + 160 * 100 * 8);
    assert_eq!((first[0].width(), first[0].height()), (320, 200));
    assert_eq!((first[1].width(), first[1].height()), (160, 100));
    assert!(first[0].usage().contains(
        wgpu::TextureUsages::RENDER_ATTACHMENT
            | wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::COPY_SRC
    ));
    assert_eq!(
        graph.transient_usage(bloom),
        wgpu::TextureUsages::RENDER_ATTACHMENT
    );

    // The next frames get the same textures back from the pool.
    for _ in 0..3 {
        let again = run_clearing_frame(&mut graph, &ids, &gpu);
        assert_eq!(again, first);
    }
    assert_eq!(graph.pool_stats().created_total, 2);
    assert_eq!(graph.frames_begun(), 4);

    // A resize makes new textures of the new size and drops the old ones.
    graph.resize(641, 99);
    assert_eq!(graph.transient_size(scene), (641, 99));
    assert_eq!(graph.transient_size(bloom), (321, 50));
    let resized = run_clearing_frame(&mut graph, &ids, &gpu);
    assert_ne!(resized[0], first[0]);
    assert_ne!(resized[1], first[1]);
    assert_eq!((resized[0].width(), resized[0].height()), (641, 99));
    assert_eq!((resized[1].width(), resized[1].height()), (321, 50));
    let stats = graph.pool_stats();
    assert_eq!((stats.textures, stats.created_total), (2, 4));

    // Resizing to the same size changes nothing.
    graph.resize(641, 99);
    assert_eq!(run_clearing_frame(&mut graph, &ids, &gpu), resized);
    assert_eq!(graph.pool_stats().created_total, 4);
    assert!(graph.last_cpu_encode_ms().is_some_and(|ms| ms >= 0.0));
    gpu.poll(wgpu::PollType::wait_indefinitely()).unwrap();
}

#[test]
fn passes_must_be_begun_in_order_and_as_declared() {
    let Some(gpu) = context_or_skip("passes_must_be_begun_in_order_and_as_declared") else {
        return;
    };
    let ids = clearing_graph();
    let mut graph = FrameGraph::new(&gpu, ids.desc.clone(), (16, 16)).unwrap();
    // A pass of another, larger graph.
    let mut other = FrameGraphDesc::new();
    let foreign = (0..8)
        .map(|index| other.compute_pass(format!("other {index}"), &[], &[]))
        .last()
        .expect("eight passes");

    let mut frame = graph.begin_frame(&gpu).unwrap();
    let view = frame.view(ids.bloom);
    let attachments = [Some(clear_attachment(&view, wgpu::Color::BLACK))];
    let desc = wgpu::RenderPassDescriptor {
        color_attachments: &attachments,
        ..Default::default()
    };
    // The second pass may begin first (the first one is skipped) ...
    drop(frame.begin_render_pass(ids.bloom_pass, &desc).unwrap());
    // ... but nothing declared before it afterwards, and it cannot run twice.
    assert_eq!(
        frame
            .begin_render_pass(ids.scene_pass, &desc)
            .map(|_| ())
            .unwrap_err(),
        FrameGraphError::OutOfOrder {
            pass: "scene".to_owned(),
            previous: "bloom".to_owned()
        }
    );
    assert!(matches!(
        frame.begin_render_pass(ids.bloom_pass, &desc),
        Err(FrameGraphError::OutOfOrder { .. })
    ));
    assert!(matches!(
        frame.begin_compute_pass(ids.bloom_pass),
        Err(FrameGraphError::WrongPassKind { .. })
    ));
    assert_eq!(
        frame
            .begin_render_pass(foreign, &desc)
            .map(|_| ())
            .unwrap_err(),
        FrameGraphError::UnknownPass { index: 7 }
    );
    frame.submit().unwrap();
}

/// A compute shader with enough work per texel to take measurable GPU time.
const BUSY_SHADER: &str = "
@group(0) @binding(0) var output_texture: texture_storage_2d<rgba8unorm, write>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let size = textureDimensions(output_texture);
    if id.x >= size.x || id.y >= size.y {
        return;
    }
    var value = vec4<f32>(0.0);
    for (var i = 0u; i < 64u; i++) {
        value += vec4<f32>(sin(f32(id.x + i)), cos(f32(id.y + i)), 0.5, 1.0) / 64.0;
    }
    textureStore(output_texture, vec2<i32>(id.xy), value);
}
";

#[test]
fn the_frame_graph_times_every_pass_with_non_negative_values() {
    let test = "the_frame_graph_times_every_pass_with_non_negative_values";
    let Some(gpu) = context_or_skip(test) else {
        return;
    };
    let mut desc = FrameGraphDesc::new();
    let color = desc.transient(TransientDesc::new(
        "color",
        wgpu::TextureFormat::Rgba16Float,
        TransientSize::FULL,
    ));
    let busy = desc.transient(TransientDesc::new(
        "busy",
        wgpu::TextureFormat::Rgba8Unorm,
        TransientSize::FULL,
    ));
    let clear = desc.render_pass("clear", &[], &[color]);
    let compute = desc.compute_pass("busy", &[color], &[busy]);
    let mut graph = FrameGraph::new(&gpu, desc, (1024, 1024)).unwrap();

    let module = create_shader_module(&gpu, "busy", BUSY_SHADER).unwrap();
    let (pipeline, layout) = capture(&gpu, "busy pipeline", |device| {
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("busy"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::StorageTexture {
                    access: wgpu::StorageTextureAccess::WriteOnly,
                    format: wgpu::TextureFormat::Rgba8Unorm,
                    view_dimension: wgpu::TextureViewDimension::D2,
                },
                count: None,
            }],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("busy"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("busy"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        (pipeline, layout)
    })
    .unwrap();

    // Records one frame; `with_clear: false` skips the first pass.
    let run_frame = |graph: &mut FrameGraph, with_clear: bool| {
        let mut frame = graph.begin_frame(&gpu).unwrap();
        if with_clear {
            let view = frame.view(color);
            let attachments = [Some(clear_attachment(&view, wgpu::Color::BLUE))];
            drop(
                frame
                    .begin_render_pass(
                        clear,
                        &wgpu::RenderPassDescriptor {
                            color_attachments: &attachments,
                            ..Default::default()
                        },
                    )
                    .unwrap(),
            );
        }
        let target = frame.view(busy);
        let bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("busy"),
            layout: &layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&target),
            }],
        });
        let (width, height) = frame.transient_size(busy);
        {
            let mut pass = frame.begin_compute_pass(compute).unwrap();
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(width.div_ceil(8), height.div_ceil(8), 1);
        }
        let index = frame.index();
        frame.submit().unwrap();
        index
    };

    // Frame by frame, each frame's timings arrive once the GPU is done with it.
    for round in 0..6 {
        let with_clear = round % 3 != 2;
        let index = run_frame(&mut graph, with_clear);
        assert_eq!(index, round);
        let cpu = graph.last_cpu_encode_ms().expect("a frame was submitted");
        assert!(cpu.is_finite() && cpu >= 0.0, "{cpu}");
        gpu.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        graph.collect_timings();
        if !graph.gpu_timing_enabled() {
            assert!(graph.latest_timings().is_none());
            log::warn!("{test}: GPU part skipped, the device has no TIMESTAMP_QUERY");
            continue;
        }
        let timings = graph.latest_timings().expect("the timings of the frame");
        assert_eq!(timings.frame, index);
        let names: Vec<&str> = timings
            .passes
            .iter()
            .map(|pass| pass.name.as_str())
            .collect();
        let expected: &[&str] = if with_clear {
            &["clear", "busy"]
        } else {
            &["busy"]
        };
        assert_eq!(names, expected, "frame {index}");
        assert_eq!(timings.passes.last().map(|pass| pass.pass), Some(compute));
        for pass in &timings.passes {
            assert!(
                pass.gpu_ms.is_finite() && pass.gpu_ms >= 0.0,
                "frame {index}: {pass:?}"
            );
        }
        assert!(timings.cpu_encode_ms >= 0.0 && timings.gpu_total_ms() >= 0.0);
        if round == 4 {
            log::info!(
                "{test}: {} on {}: clear {:.3} ms, busy {:.3} ms GPU, encode {:.3} ms CPU \
                 (other builds may be running; not a benchmark)",
                gpu.caps.adapter.name,
                gpu.caps.adapter.backend,
                timings.passes[0].gpu_ms,
                timings.passes[1].gpu_ms,
                timings.cpu_encode_ms
            );
        }
    }

    // A burst larger than the read-back ring never blocks; frames beyond the ring go untimed
    // and the timings catch up once the GPU is done.
    let burst: Vec<u64> = (0..12).map(|_| run_frame(&mut graph, true)).collect();
    gpu.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    graph.collect_timings();
    if graph.gpu_timing_enabled() {
        let latest = graph.latest_timings().expect("timings of the burst");
        assert!(burst.contains(&latest.frame), "frame {}", latest.frame);
        assert!(latest.passes.iter().all(|pass| pass.gpu_ms >= 0.0));
        // The ring is free again: the next frame is timed.
        let index = run_frame(&mut graph, true);
        gpu.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        graph.collect_timings();
        assert_eq!(
            graph.latest_timings().map(|timings| timings.frame),
            Some(index)
        );
    }
}

#[test]
fn a_transient_the_device_cannot_create_is_an_error() {
    let Some(gpu) = context_or_skip("a_transient_the_device_cannot_create_is_an_error") else {
        return;
    };
    // A texture wider than the device allows is rejected on every device.
    let limit = gpu.caps.enabled_limits.max_texture_dimension_2d;
    let mut desc = FrameGraphDesc::new();
    let wide = desc.transient(TransientDesc::new(
        "too wide",
        wgpu::TextureFormat::Rgba8Unorm,
        TransientSize::Fixed {
            width: limit + 1,
            height: 1,
        },
    ));
    desc.render_pass("draw wide", &[], &[wide]);
    let mut graph = FrameGraph::new(&gpu, desc, (8, 8)).unwrap();
    match graph.begin_frame(&gpu) {
        Err(RenderError::Wgpu {
            kind, operation, ..
        }) => {
            assert_eq!(kind, WgpuErrorKind::Validation);
            assert_eq!(operation, "transient texture too wide");
        }
        other => panic!("expected a validation error, got {:?}", other.map(|_| ())),
    }
    assert!(!gpu.is_lost());
    assert_eq!(graph.pool_stats().textures, 0);
    // Declarations are checked before any GPU work.
    let mut desc = FrameGraphDesc::new();
    desc.transient(TransientDesc::new(
        "unused",
        wgpu::TextureFormat::Rgba8Unorm,
        TransientSize::FULL,
    ));
    assert!(matches!(
        FrameGraph::new(&gpu, desc, (8, 8)),
        Err(RenderError::FrameGraph(FrameGraphError::Unused { .. }))
    ));
}

#[test]
fn every_shader_and_pipeline_is_created_on_each_backend() {
    let test = "every_shader_and_pipeline_is_created_on_each_backend";
    let library = ShaderLibrary::builtin();
    for gpu in contexts_per_backend(test) {
        let backend = gpu.caps.adapter.backend;
        let target = ShaderTarget::of_device(&gpu.caps);
        for file in library.files() {
            if !target.features.contains(file.required_features) {
                log::warn!("{test}: {backend}: {} skipped, features missing", file.path);
                continue;
            }
            library
                .validate(file.path, &target)
                .unwrap_or_else(|error| panic!("{backend}: {error}"));
            library
                .create_module(&gpu, file.path)
                .unwrap_or_else(|error| panic!("{backend}: {}: {error}", file.path));
        }
        for format in HEADLESS_FORMATS
            .into_iter()
            .chain([wgpu::TextureFormat::Rgba16Float])
        {
            OverlayRenderer::new(&gpu, format)
                .unwrap_or_else(|error| panic!("{backend}: overlay for {format:?}: {error}"));
        }
        assert!(!gpu.is_lost(), "{backend}");
        log::info!("{test}: every shader and pipeline created on {backend}");
    }
}

#[test]
fn a_bad_shader_or_pipeline_is_an_error_not_a_panic() {
    let Some(gpu) = context_or_skip("a_bad_shader_or_pipeline_is_an_error_not_a_panic") else {
        return;
    };
    match create_shader_module(&gpu, "broken", "fn broken( {") {
        Err(RenderError::Wgpu {
            kind, operation, ..
        }) => {
            assert_eq!(kind, WgpuErrorKind::Validation);
            assert_eq!(operation, "shader module broken");
        }
        other => panic!("expected a validation error, got {:?}", other.map(|_| ())),
    }
    // A pipeline whose entry point does not exist.
    let module = create_shader_module(
        &gpu,
        "compute",
        "@compute @workgroup_size(1)\nfn main() {}\n",
    )
    .unwrap();
    let pipeline = capture(&gpu, "pipeline without entry point", |device| {
        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("missing entry"),
            layout: None,
            module: &module,
            entry_point: Some("does_not_exist"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        })
    });
    assert!(
        matches!(
            pipeline,
            Err(RenderError::Wgpu {
                kind: WgpuErrorKind::Validation,
                ..
            })
        ),
        "{pipeline:?}"
    );
    // The device is unharmed.
    assert!(!gpu.is_lost());
    assert!(create_shader_module(&gpu, "fine", "fn fine() {}\n").is_ok());
}

/// Reads mip level `level` of layer `layer` of an RGBA8 texture back.
fn read_texture_level(
    gpu: &GpuContext,
    texture: &wgpu::Texture,
    layer: u32,
    level: u32,
) -> Vec<u8> {
    let (width, height) = mip_size(texture.width(), texture.height(), level);
    let padded_row = (width * 4).next_multiple_of(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT);
    let buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("texture read-back"),
        size: u64::from(padded_row * height),
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut commands = encoder(gpu, "texture read-back");
    commands.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture,
            mip_level: level,
            origin: wgpu::Origin3d {
                x: 0,
                y: 0,
                z: layer,
            },
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &buffer,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(padded_row),
                rows_per_image: Some(height),
            },
        },
        wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
    );
    gpu.queue.submit([commands.finish()]);
    read_buffer(gpu, &buffer)
        .chunks(padded_row as usize)
        .flat_map(|row| row[..(width * 4) as usize].to_vec())
        .collect()
}

/// Maps `buffer` after everything submitted and returns its bytes.
fn read_buffer(gpu: &GpuContext, buffer: &wgpu::Buffer) -> Vec<u8> {
    buffer.map_async(wgpu::MapMode::Read, .., |result| {
        result.expect("map the read-back buffer");
    });
    gpu.poll(wgpu::PollType::wait_indefinitely())
        .expect("wait for the read-back");
    let bytes = buffer
        .get_mapped_range(..)
        .expect("the read-back buffer is mapped")
        .to_vec();
    buffer.unmap();
    bytes
}

/// The texels of mip level `level` of layer `layer` in the texture array test: every byte
/// tells layer, level and texel apart.
fn level_bytes(layer: u32, level: u32, width: u32, height: u32) -> Vec<u8> {
    (0..width * height * 4)
        .map(|byte| (layer * 64 + level * 16 + byte * 7) as u8)
        .collect()
}

#[test]
fn texture_arrays_upload_every_layer_and_mip_level() {
    let Some(gpu) = context_or_skip("texture_arrays_upload_every_layer_and_mip_level") else {
        return;
    };
    for kind in [TextureArrayKind::Linear, TextureArrayKind::Albedo] {
        let mut builder = TextureArrayBuilder::new("test array", kind, 8, 4).unwrap();
        let levels = builder.mip_level_count();
        assert_eq!(levels, 4);
        for layer in 0..3 {
            let mips: Vec<Vec<u8>> = (0..levels)
                .map(|level| {
                    let (width, height) = mip_size(8, 4, level);
                    level_bytes(layer, level, width, height)
                })
                .collect();
            let slices: Vec<&[u8]> = mips.iter().map(Vec::as_slice).collect();
            assert_eq!(builder.push_layer(&slices).unwrap(), layer);
        }
        let array = builder.build(&gpu).unwrap();
        assert_eq!(array.kind(), kind);
        assert_eq!(array.texture().format(), kind.format());
        assert_eq!(
            (array.size(), array.layer_count(), array.mip_level_count()),
            ((8, 4), 3, 4)
        );
        // Raw bytes come back unchanged; sRGB only matters when the GPU samples.
        for (layer, level) in [(0, 0), (2, 1), (1, 3), (2, 2)] {
            let (width, height) = mip_size(8, 4, level);
            assert_eq!(
                read_texture_level(&gpu, array.texture(), layer, level),
                level_bytes(layer, level, width, height),
                "{kind:?} layer {layer} level {level}"
            );
        }
    }

    // The device limits are checked before anything is created.
    let limit = gpu.caps.enabled_limits.max_texture_array_layers;
    let mut builder = TextureArrayBuilder::new("too deep", TextureArrayKind::Linear, 1, 1).unwrap();
    for _ in 0..=limit {
        builder.push_layer(&[&[0, 0, 0, 0]]).unwrap();
    }
    assert!(matches!(
        builder.build(&gpu),
        Err(RenderError::Texture(TextureError::TooManyLayers { .. }))
    ));
    let empty = TextureArrayBuilder::new("empty", TextureArrayKind::Linear, 4, 4).unwrap();
    assert!(matches!(
        empty.build(&gpu),
        Err(RenderError::Texture(TextureError::NoLayers { .. }))
    ));
}

#[test]
fn the_material_table_holds_and_updates_its_entries() {
    let Some(gpu) = context_or_skip("the_material_table_holds_and_updates_its_entries") else {
        return;
    };
    let material = |layer: u32, tint_class: u32| GpuMaterial {
        average_color: [0.25, 0.5, 0.75],
        roughness: 0.5,
        emissive: 0.0,
        metallic: 0.0,
        texture_layer: layer,
        tint_class,
        variation: GpuMaterial::variation_bits(TextureTurn::Free, tint_class == 2, layer),
        variant_layer: 40 + layer,
        reserved: [0, 0],
    };
    let mut materials = vec![material(0, 0), material(1, 1), material(2, 3)];
    let table = MaterialTable::new(&gpu, &materials).unwrap();
    assert_eq!(table.len(), 3);
    assert!(!table.is_empty());
    assert_eq!(table.buffer().size(), 3 * GpuMaterial::SIZE);

    materials[2] = material(9, 2);
    table.update(&gpu, 2, &materials[2..]).unwrap();
    assert!(matches!(
        table.update(&gpu, 2, &materials[1..]),
        Err(RenderError::Material(_))
    ));
    let mut bad = material(0, 0);
    bad.roughness = 2.0;
    assert!(matches!(
        table.update(&gpu, 0, &[bad]),
        Err(RenderError::Material(_))
    ));
    assert!(matches!(
        MaterialTable::new(&gpu, &[]),
        Err(RenderError::Material(_))
    ));

    let copy = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("material read-back"),
        size: table.buffer().size(),
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut commands = encoder(&gpu, "material read-back");
    commands.copy_buffer_to_buffer(table.buffer(), 0, &copy, 0, table.buffer().size());
    gpu.queue.submit([commands.finish()]);
    assert_eq!(
        read_buffer(&gpu, &copy),
        bytemuck::cast_slice::<_, u8>(&materials)
    );
}
