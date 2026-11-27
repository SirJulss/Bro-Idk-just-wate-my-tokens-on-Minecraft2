//! Tests of the GPU-driven voxel pipeline against the real GPU, through the public API only
//! (contract §5.3, §9 E2): identical images from draw path A, path B and direct draws; texture
//! orientation and winding on all 24 orientations and 6 faces; camera-relative precision far
//! from the origin; every pipeline on every backend; statistics and upload budgets.
//!
//! Every test skips with a log line when the machine has no GPU adapter (contract §2.7). The
//! images are written to `engine/target/render-tests/` for inspection.

#[path = "common/png.rs"]
mod png;

use std::path::PathBuf;
use std::sync::Once;

use cp_core::det;
use cp_core::face::{Face, Orientation};
use cp_core::ids::MaterialId;
use cp_core::math::{DQuat, DVec3, IAabb, IVec3};
use cp_core::quad::PackedQuad;
use cp_render::debug_view::DebugView;
use cp_render::gpu::{BACKEND_ENV, GpuContext, GpuError, GpuOptions};
use cp_render::headless::{HeadlessTarget, RgbaImage};
use cp_render::material::{GpuMaterial, MaterialTable};
use cp_render::texture::{TextureArray, TextureArrayBuilder, TextureArrayKind, mip_size};
use cp_render::voxel::{
    CLOUD_ANCHOR_PERIOD, CalibrationParams, CalibrationScene, Camera, ChunkId, ChunkUpload,
    DrawMode, Fade, FrameInput, FrameStats, GridPose, LayerCounts, Lighting, MeshLayer, ObjectDesc,
    ObjectId, PIPELINE_COUNT, VoxelRenderer, VoxelRendererConfig, WallMask, storage_slot,
};
use cp_render::wgpu;

// ---------------------------------------------------------------------------------------------
// Harness
// ---------------------------------------------------------------------------------------------

struct TestLogger;

static LOGGER: TestLogger = TestLogger;

impl log::Log for TestLogger {
    fn enabled(&self, metadata: &log::Metadata<'_>) -> bool {
        let own = ["cp_render", "voxel_pipeline"];
        metadata.level() <= log::Level::Error
            || (metadata.level() <= log::Level::Info
                && own
                    .iter()
                    .any(|prefix| metadata.target().starts_with(prefix)))
    }

    // Contract §2.6 keeps libraries silent; the test binary is where the log is printed.
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

/// One context per backend of the contract that has an adapter here.
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

fn image_dir() -> PathBuf {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../target/render-tests");
    std::fs::create_dir_all(&dir).expect("create target/render-tests");
    dir
}

fn save(name: &str, image: &RgbaImage) {
    let path = image_dir().join(name);
    let bytes = png::encode_rgba(image.width, image.height, &image.pixels, image.srgb);
    std::fs::write(&path, bytes).unwrap_or_else(|error| panic!("{}: {error}", path.display()));
}

fn assert_identical(reference: &RgbaImage, other: &RgbaImage, what: &str) {
    assert_eq!(
        (reference.width, reference.height),
        (other.width, other.height)
    );
    if reference.pixels == other.pixels {
        return;
    }
    let mut differing = 0;
    let mut first = None;
    for y in 0..reference.height {
        for x in 0..reference.width {
            if reference.pixel(x, y) != other.pixel(x, y) {
                differing += 1;
                first.get_or_insert((x, y, reference.pixel(x, y), other.pixel(x, y)));
            }
        }
    }
    panic!("{what}: {differing} pixels differ, the first at {first:?}");
}

fn srgb_to_linear(byte: u8) -> f32 {
    let encoded = f32::from(byte) / 255.0;
    if encoded <= 0.04045 {
        encoded / 12.92
    } else {
        det::powf((encoded + 0.055) / 1.055, 2.4)
    }
}

/// The full mip chain of a square RGBA8 image (box filter, alpha-weighted colour).
fn mip_chain(size: u32, level0: Vec<u8>) -> Vec<Vec<u8>> {
    let mut levels = vec![level0];
    let mut side = size;
    while side > 1 {
        let previous = levels.last().expect("the chain starts with level 0");
        let next_side = side / 2;
        let mut next = vec![0_u8; (next_side * next_side * 4) as usize];
        for y in 0..next_side {
            for x in 0..next_side {
                let mut sum = [0_u32; 4];
                for (dx, dy) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
                    let at = (((2 * y + dy) * side + 2 * x + dx) * 4) as usize;
                    for channel in 0..4 {
                        sum[channel] += u32::from(previous[at + channel]);
                    }
                }
                let at = ((y * next_side + x) * 4) as usize;
                for channel in 0..4 {
                    next[at + channel] = (sum[channel] / 4) as u8;
                }
            }
        }
        levels.push(next);
        side = next_side;
    }
    levels
}

fn texture_array(gpu: &GpuContext, layers: &[Vec<u8>]) -> TextureArray {
    let mut builder = TextureArrayBuilder::new("test blocks", TextureArrayKind::Albedo, 16, 16)
        .expect("16x16 layers");
    for layer in layers {
        let chain = mip_chain(16, layer.clone());
        for (level, data) in (0..).zip(&chain) {
            let (width, height) = mip_size(16, 16, level);
            assert_eq!(data.len(), (width * height * 4) as usize);
        }
        let slices: Vec<&[u8]> = chain.iter().map(Vec::as_slice).collect();
        builder.push_layer(&slices).expect("a full mip chain");
    }
    builder.build(gpu).expect("create the texture array")
}

fn material(srgb: [u8; 3], layer: u32) -> GpuMaterial {
    // No variation: the orientation tests rely on every texture being drawn as painted.
    GpuMaterial {
        average_color: srgb.map(srgb_to_linear),
        roughness: 0.85,
        emissive: 0.0,
        metallic: 0.0,
        texture_layer: layer,
        tint_class: 0,
        ..GpuMaterial::default()
    }
}

/// The renderer of these tests: single frames without temporal accumulation, so that images
/// of the same frame compare pixel for pixel (temporal anti-aliasing has tests of its own).
fn config(format: wgpu::TextureFormat) -> VoxelRendererConfig {
    VoxelRendererConfig {
        target_format: format,
        pool_bytes: 32 << 20,
        max_draw_items: 4096,
        max_objects: 8,
        upload_budget_bytes: u64::MAX,
        anisotropy: 16,
        temporal_aa: false,
    }
}

fn render(
    renderer: &mut VoxelRenderer,
    gpu: &GpuContext,
    target: &HeadlessTarget,
    camera: &Camera,
    lighting: &Lighting,
) -> (RgbaImage, FrameStats) {
    let stats = renderer
        .render(
            gpu,
            &FrameInput {
                camera,
                lighting,
                target: target.view(),
                target_size: target.size(),
                overlay: None,
            },
        )
        .expect("render a frame");
    (
        target.capture_image(gpu).expect("read the frame back"),
        stats,
    )
}

// ---------------------------------------------------------------------------------------------
// A small mesher for test chunks: one quad per exposed cell face.
// ---------------------------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq)]
struct Block {
    material: u16,
    layer: MeshLayer,
    uv_rot: u8,
}

/// The mesh of a 32³ chunk whose cells `cell(x, y, z)` returns (outside the chunk is air). A
/// face is emitted unless the neighbour is opaque or in the same layer; boundary faces of
/// `walls` go to the wall ranges instead of the regular ones.
fn naive_mesh(
    cell: impl Fn(i32, i32, i32) -> Option<Block>,
    walls: &[Face],
) -> (Vec<PackedQuad>, [LayerCounts; 3]) {
    let mut ranges: Vec<Vec<PackedQuad>> = vec![Vec::new(); 36];
    for y in 0..32 {
        for z in 0..32 {
            for x in 0..32 {
                let Some(block) = cell(x, y, z) else {
                    continue;
                };
                for face in Face::ALL {
                    let n = face.normal();
                    let (nx, ny, nz) = (x + n.x, y + n.y, z + n.z);
                    let inside =
                        (0..32).contains(&nx) && (0..32).contains(&ny) && (0..32).contains(&nz);
                    let neighbour = if inside { cell(nx, ny, nz) } else { None };
                    let hidden = neighbour.is_some_and(|other| {
                        other.layer == MeshLayer::Opaque || other.layer == block.layer
                    });
                    if hidden {
                        continue;
                    }
                    let wall = !inside && walls.contains(&face);
                    let slot = storage_slot(face) + if wall { 6 } else { 0 };
                    let quad = PackedQuad::new(
                        [x as u8, y as u8, z as u8],
                        1,
                        1,
                        face,
                        MaterialId(block.material),
                        block.uv_rot,
                        false,
                    )
                    .expect("test quads lie inside the chunk");
                    ranges[block.layer.index() * 12 + slot].push(quad);
                }
            }
        }
    }
    let mut quads = Vec::new();
    let mut layers = [LayerCounts::EMPTY; 3];
    for (layer, counts) in layers.iter_mut().enumerate() {
        let mut slots = [0_u32; 12];
        for (slot, count) in slots.iter_mut().enumerate() {
            let range = &ranges[layer * 12 + slot];
            *count = range.len() as u32;
            quads.extend_from_slice(range);
        }
        *counts = LayerCounts::from_slots(slots);
    }
    (quads, layers)
}

fn insert(
    renderer: &mut VoxelRenderer,
    object: ObjectId,
    origin: IVec3,
    level: u8,
    mesh: &(Vec<PackedQuad>, [LayerCounts; 3]),
) -> ChunkId {
    renderer
        .scene_mut()
        .insert_chunk(&ChunkUpload {
            quads: &mesh.0,
            layers: mesh.1,
            origin,
            level,
            object,
        })
        .expect("insert a test chunk")
}

fn hash(x: i32, y: i32, z: i32) -> u32 {
    let mut h = (x as u32).wrapping_mul(0x8DA6_B343)
        ^ (y as u32).wrapping_mul(0xD816_3841)
        ^ (z as u32).wrapping_mul(0xCB1A_B31F);
    h ^= h >> 15;
    h = h.wrapping_mul(0x2C1B_3C6D);
    h ^ (h >> 12)
}

// ---------------------------------------------------------------------------------------------
// The test world of the draw-path comparison
// ---------------------------------------------------------------------------------------------

const STONE: u16 = 0;
const GRASS: u16 = 1;
const LEAVES: u16 = 2;
const WATER: u16 = 3;
const METAL: u16 = 4;
const LAMP: u16 = 5;

/// Six 16x16 block textures: noisy stone and grass, leaves with holes, water, metal plates
/// and a lamp.
fn world_textures() -> Vec<Vec<u8>> {
    let mut layers = Vec::new();
    for layer in 0..6_i32 {
        let mut pixels = Vec::with_capacity(16 * 16 * 4);
        for y in 0..16 {
            for x in 0..16 {
                let noise = (hash(x, y, layer) % 32) as u8;
                let texel: [u8; 4] = match layer {
                    0 => [110 + noise, 108 + noise, 104 + noise, 255],
                    1 => [70 + noise, 140 + noise, 50 + noise / 2, 255],
                    2 => {
                        let hole = hash(x / 2, y / 2, 77).is_multiple_of(3);
                        [40 + noise, 110 + noise, 35, if hole { 0 } else { 255 }]
                    }
                    3 => [30, 80 + noise / 2, 170 + noise, 255],
                    4 => {
                        let seam = x % 8 == 0 || y % 8 == 0;
                        let shade = if seam { 90 } else { 170 + noise / 2 };
                        [shade, shade, shade + 10, 255]
                    }
                    _ => [250, 220 - noise, 120 + noise, 255],
                };
                pixels.extend_from_slice(&texel);
            }
        }
        layers.push(pixels);
    }
    layers
}

fn world_materials() -> Vec<GpuMaterial> {
    let mut materials = vec![
        material([125, 122, 118], 0),
        material([85, 155, 58], 1),
        material([55, 125, 35], 2),
        material([34, 92, 186], 3),
        material([165, 165, 175], 4),
        material([250, 205, 135], 5),
    ];
    materials[METAL as usize].metallic = 0.9;
    materials[METAL as usize].roughness = 0.35;
    materials[LAMP as usize].emissive = 3.0;
    materials[WATER as usize].roughness = 0.1;
    materials
}

/// Terrain height of level-0 column (x, z) in voxels of the grid.
fn terrain_height(x: i32, z: i32) -> i32 {
    let ridge = ((x - 20).abs() + (z + 10).abs()) / 6;
    8 + (hash(x / 4, 0, z / 4) % 4) as i32 - ridge.min(6)
}

/// Builds the comparison world into `renderer` and returns the camera that looks at it.
fn build_world(renderer: &mut VoxelRenderer) -> Camera {
    let terrain = renderer
        .scene_mut()
        .add_object(ObjectDesc {
            bounds: IAabb::new(IVec3::new(-512, -64, -512), IVec3::new(512, 512, 512)),
            pose: GridPose::IDENTITY,
        })
        .expect("add the terrain object");
    // Four level-0 terrain chunks with grass on stone, a pond and a leaf blob; the +X
    // boundary of the first one is emitted as walls (drawn: the mask has +X).
    for (index, (cx, cz)) in [(0, -1), (1, -1), (0, -2), (1, -2)].into_iter().enumerate() {
        let origin = IVec3::new(32 * cx, 0, 32 * cz);
        let cell = |x: i32, y: i32, z: i32| {
            let (wx, wz) = (origin.x + x, origin.z + z);
            let height = terrain_height(wx, wz);
            let pond = (wx - 24).pow(2) + (wz + 20).pow(2) < 30;
            if pond && y < height && y >= height - 2 {
                return Some(Block {
                    material: WATER,
                    layer: MeshLayer::Fluid,
                    uv_rot: 0,
                });
            }
            if y < height {
                let material = if y == height - 1 { GRASS } else { STONE };
                return Some(Block {
                    material,
                    layer: MeshLayer::Opaque,
                    uv_rot: (hash(wx, y, wz) % 4) as u8,
                });
            }
            let blob = (wx - 10).pow(2) + (y - 14).pow(2) + (wz + 40).pow(2) < 20;
            blob.then_some(Block {
                material: LEAVES,
                layer: MeshLayer::Cutout,
                uv_rot: 0,
            })
        };
        let walls: &[Face] = if index == 0 { &[Face::PosX] } else { &[] };
        let mesh = naive_mesh(cell, walls);
        let id = insert(renderer, terrain, origin, 0, &mesh);
        if index == 0 {
            renderer
                .scene_mut()
                .set_wall_mask(id, WallMask::from_faces(&[Face::PosX]))
                .expect("the chunk exists");
        }
        if index == 3 {
            // Half covered by the dither pattern: drawn by the cutout pipeline.
            renderer
                .scene_mut()
                .set_fade(id, Fade::new(0.5, false))
                .expect("the chunk exists");
        }
    }
    // A level-1 chunk (2 m cells) and a level-2 chunk (4 m cells, untextured).
    let hills = |level: i32| {
        move |x: i32, y: i32, z: i32| {
            let height = 2 + (hash(x / 3, level, z / 3) % 5) as i32;
            (y < height).then_some(Block {
                material: if y == height - 1 { GRASS } else { STONE },
                layer: MeshLayer::Opaque,
                uv_rot: 0,
            })
        }
    };
    insert(
        renderer,
        terrain,
        IVec3::new(64, 0, -128),
        1,
        &naive_mesh(hills(1), &[]),
    );
    insert(
        renderer,
        terrain,
        IVec3::new(-128, -128, -256),
        2,
        &naive_mesh(hills(2), &[]),
    );
    // A rotated construct with metal plates and a lamp.
    let construct = renderer
        .scene_mut()
        .add_object(ObjectDesc {
            bounds: IAabb::new(IVec3::ZERO, IVec3::splat(32)),
            pose: GridPose {
                rotation: DQuat::from_rotation_y(0.5),
                translation: DVec3::new(12.37, 13.25, -18.6),
            },
        })
        .expect("add the construct object");
    let structure = |x: i32, y: i32, z: i32| {
        let frame = x < 5 && y < 4 && z < 3;
        let lamp = x == 2 && y == 4 && z == 1;
        if lamp {
            Some(Block {
                material: LAMP,
                layer: MeshLayer::Opaque,
                uv_rot: 0,
            })
        } else {
            frame.then_some(Block {
                material: METAL,
                layer: MeshLayer::Opaque,
                uv_rot: 1,
            })
        }
    };
    insert(
        renderer,
        construct,
        IVec3::ZERO,
        0,
        &naive_mesh(structure, &[]),
    );
    Camera::look_at(
        DVec3::new(34.25, 26.5, 18.75),
        DVec3::new(14.0, 6.0, -30.0),
        DVec3::Y,
        70.0_f32.to_radians(),
    )
}

#[test]
fn draw_path_a_path_b_and_direct_draws_render_identical_images() {
    let test = "draw_path_a_path_b_and_direct_draws_render_identical_images";
    let Some(gpu) = context_or_skip(test) else {
        return;
    };
    let format = wgpu::TextureFormat::Rgba8UnormSrgb;
    let materials = MaterialTable::new(&gpu, &world_materials()).unwrap();
    let mut renderer = VoxelRenderer::new(
        &gpu,
        &config(format),
        materials,
        texture_array(&gpu, &world_textures()),
    )
    .unwrap();
    let camera = build_world(&mut renderer);
    let lighting = Lighting::default();
    let target = HeadlessTarget::new(&gpu, 480, 270, format).unwrap();
    let modes = renderer.available_draw_modes().to_vec();
    assert!(modes.contains(&DrawMode::Direct));
    log::info!(
        "{test}: comparing {:?} on {}",
        modes,
        gpu.caps.adapter.backend
    );
    // Every view; all but the draw-path view (which shows the path) must not depend on it.
    let views = [
        DebugView::Off,
        DebugView::Albedo,
        DebugView::Normal,
        DebugView::Depth,
        DebugView::LodLevel,
        DebugView::Overdraw,
        DebugView::ChunkBounds,
        DebugView::DrawPath,
    ];
    // The cross-fade dither rotates with the frame index (period 16): every mode renders 16
    // frames, the views twice, so the compared frames have equal indices modulo 16.
    let mut reference: Vec<Vec<Option<RgbaImage>>> = vec![vec![None; views.len()]; 2];
    let mut counts = Vec::new();
    for &mode in &modes {
        renderer.set_draw_mode(mode).unwrap();
        for (round, references) in reference.iter_mut().enumerate() {
            for (index, &view) in views.iter().enumerate() {
                renderer.debug_view_mut().select(view).unwrap();
                let (image, stats) = render(&mut renderer, &gpu, &target, &camera, &lighting);
                assert_eq!(stats.mode, mode);
                if round == 0 {
                    save(
                        &format!("draw_paths_{}_{}.png", view.id(), mode.id()),
                        &image,
                    );
                }
                if view == DebugView::DrawPath {
                    continue;
                }
                if view == DebugView::Off && round == 0 && mode != DrawMode::A {
                    counts.push(stats.visible.expect("CPU paths count at once"));
                }
                match &references[index] {
                    None => references[index] = Some(image),
                    Some(first) => assert_identical(
                        first,
                        &image,
                        &format!("{view} with path {} (round {round})", mode.id()),
                    ),
                }
            }
        }
    }
    let reference = &reference[0];
    // The GPU's own counts of path A agree with the CPU paths.
    if modes.contains(&DrawMode::A) {
        renderer.set_draw_mode(DrawMode::A).unwrap();
        renderer.debug_view_mut().select(DebugView::Off).unwrap();
        let (_, stats) = render(&mut renderer, &gpu, &target, &camera, &lighting);
        renderer.wait_for_gpu(&gpu).unwrap();
        let gpu_counts = renderer
            .latest_gpu_counts()
            .expect("the counters arrive after a wait");
        assert!(gpu_counts.frame <= stats.frame);
        assert_eq!(
            (gpu_counts.draws, gpu_counts.quads),
            (counts[0].draws, counts[0].quads)
        );
    }
    // Every comparison above was between images with content: the albedo view covers a good
    // part of the target with geometry, and the lit image differs from it.
    let albedo = reference[1].as_ref().unwrap();
    let covered = albedo
        .pixels
        .chunks(4)
        .filter(|pixel| pixel[..3] != [0, 0, 0])
        .count();
    let total = (albedo.width * albedo.height) as usize;
    assert!(
        covered > total / 3,
        "{covered} of {total} pixels show geometry"
    );
    assert_ne!(reference[0], reference[1]);
    for pair in counts.windows(2) {
        assert_eq!(
            (pair[0].draws, pair[0].quads),
            (pair[1].draws, pair[1].quads)
        );
    }
    let both_classes = counts[0].draws;
    assert!(
        both_classes[0] > 0 && both_classes[1] > 0,
        "{both_classes:?}"
    );
    log::info!(
        "{test}: {} draws ({:?}), {} triangles, identical in {} views x 2 rounds x {} paths \
         (tolerance 0)",
        counts[0].total_draws(),
        counts[0].draws,
        counts[0].triangles(),
        views.len() - 1,
        modes.len()
    );
}

// ---------------------------------------------------------------------------------------------
// Texture orientation and winding: 24 orientations x 6 faces with an asymmetric F
// ---------------------------------------------------------------------------------------------

/// Background colour of the texture of each block-local face (sRGB), by face index.
const FACE_COLORS: [[u8; 3]; 6] = [
    [200, 40, 40],
    [40, 180, 200],
    [60, 170, 60],
    [190, 60, 180],
    [50, 70, 200],
    [210, 190, 40],
];
const INK: [u8; 3] = [255, 255, 255];

/// The F glyph: image column `x` (right) and row `y` (down) of 16x16.
fn f_glyph(x: u32, y: u32) -> bool {
    let stem = (3..=5).contains(&x) && (2..=13).contains(&y);
    let top = (3..=12).contains(&x) && (2..=4).contains(&y);
    let middle = (3..=9).contains(&x) && (7..=9).contains(&y);
    stem || top || middle
}

/// Texel probes: centres of texels whose value tells every rotation and mirror apart.
const PROBES: [(u32, u32); 8] = [
    (4, 12),
    (11, 3),
    (8, 8),
    (11, 8),
    (11, 12),
    (1, 8),
    (8, 12),
    (4, 1),
];

fn f_textures() -> Vec<Vec<u8>> {
    FACE_COLORS
        .iter()
        .map(|background| {
            let mut pixels = Vec::new();
            for y in 0..16 {
                for x in 0..16 {
                    let color = if f_glyph(x, y) { INK } else { *background };
                    pixels.extend_from_slice(&[color[0], color[1], color[2], 255]);
                }
            }
            pixels
        })
        .collect()
}

/// The blocks of the orientation test: orientation `i` at cell (1 + 2 (i % 6), 1, 1 + 2 (i / 6)).
fn orientation_cell(index: usize) -> IVec3 {
    IVec3::new(1 + 2 * (index % 6) as i32, 1, 1 + 2 * (index / 6) as i32)
}

fn orientation_mesh() -> (Vec<PackedQuad>, [LayerCounts; 3]) {
    let mut ranges: Vec<Vec<PackedQuad>> = vec![Vec::new(); 6];
    for (index, orientation) in Orientation::ALL.into_iter().enumerate() {
        let cell = orientation_cell(index);
        for world in Face::ALL {
            let local = orientation.local_face(world);
            let quad = PackedQuad::new(
                [cell.x as u8, cell.y as u8, cell.z as u8],
                1,
                1,
                world,
                MaterialId(local.index() as u16),
                orientation.uv_rot(world),
                false,
            )
            .expect("orientation quads lie inside the chunk");
            ranges[storage_slot(world)].push(quad);
        }
    }
    let mut regular = [0; 6];
    let mut quads = Vec::new();
    for (slot, range) in ranges.iter().enumerate() {
        regular[slot] = range.len() as u32;
        quads.extend_from_slice(range);
    }
    let mut layers = [LayerCounts::EMPTY; 3];
    layers[0].regular = regular;
    (quads, layers)
}

/// The palette colour nearest to `pixel` and its distance (sum of channel differences).
fn classify(pixel: [u8; 4]) -> ([u8; 3], u32) {
    FACE_COLORS
        .iter()
        .chain([&INK])
        .map(|color| {
            let distance: u32 = (0..3).map(|c| u32::from(pixel[c].abs_diff(color[c]))).sum();
            (*color, distance)
        })
        .min_by_key(|&(_, distance)| distance)
        .expect("the palette is not empty")
}

#[test]
fn textures_keep_orientation_and_winding_on_all_24_orientations_and_6_faces() {
    let test = "textures_keep_orientation_and_winding_on_all_24_orientations_and_6_faces";
    let Some(gpu) = context_or_skip(test) else {
        return;
    };
    let format = wgpu::TextureFormat::Rgba8UnormSrgb;
    let materials: Vec<GpuMaterial> = (0..6)
        .map(|face| material(FACE_COLORS[face], face as u32))
        .collect();
    let mut renderer = VoxelRenderer::new(
        &gpu,
        &config(format),
        MaterialTable::new(&gpu, &materials).unwrap(),
        texture_array(&gpu, &f_textures()),
    )
    .unwrap();
    let grid = renderer
        .scene_mut()
        .add_object(ObjectDesc {
            bounds: IAabb::new(IVec3::ZERO, IVec3::splat(32)),
            pose: GridPose::IDENTITY,
        })
        .unwrap();
    insert(&mut renderer, grid, IVec3::ZERO, 0, &orientation_mesh());
    let (width, height) = (1280, 960);
    let target = HeadlessTarget::new(&gpu, width, height, format).unwrap();
    let lighting = Lighting::default();
    let center = DVec3::new(6.5, 1.5, 4.5);
    let aspect = width as f32 / height as f32;
    let mut checked = 0;
    let mut failures = Vec::new();
    for (name, direction, faces) in [
        ("positive", DVec3::ONE, [Face::PosX, Face::PosY, Face::PosZ]),
        (
            "negative",
            DVec3::NEG_ONE,
            [Face::NegX, Face::NegY, Face::NegZ],
        ),
    ] {
        let camera = Camera::look_at(
            center + direction.normalize() * 90.0,
            center,
            DVec3::Y,
            2.0 * det::atan(5.6_f32 / 90.0),
        );
        renderer.debug_view_mut().select(DebugView::Off).unwrap();
        let (lit, _) = render(&mut renderer, &gpu, &target, &camera, &lighting);
        save(&format!("orientation_{name}_lit.png"), &lit);
        renderer.debug_view_mut().select(DebugView::Albedo).unwrap();
        let (image, _) = render(&mut renderer, &gpu, &target, &camera, &lighting);
        save(&format!("orientation_{name}_albedo.png"), &image);
        let to_clip = camera.frame_to_clip(aspect);
        for (index, orientation) in Orientation::ALL.into_iter().enumerate() {
            let block_center = orientation_cell(index).as_dvec3() + DVec3::splat(0.5);
            for world in faces {
                let local = orientation.local_face(world);
                let (tex_u, tex_v) = local.local_tex_axes();
                let tex_u = orientation.rotate_vec(tex_u).as_dvec3();
                let tex_v = orientation.rotate_vec(tex_v).as_dvec3();
                let face_center = block_center + world.normal().as_dvec3() * 0.5;
                for (column, row) in PROBES {
                    // Texel centre in (s, t), image up = t.
                    let s = (f64::from(column) + 0.5) / 16.0;
                    let t = 1.0 - (f64::from(row) + 0.5) / 16.0;
                    let point = face_center + tex_u * (s - 0.5) + tex_v * (t - 0.5);
                    let clip = to_clip * point.extend(1.0);
                    let ndc = clip.truncate() / clip.w;
                    let x = ((ndc.x + 1.0) * 0.5 * f64::from(width)) as u32;
                    let y = ((1.0 - ndc.y) * 0.5 * f64::from(height)) as u32;
                    let expected = if f_glyph(column, row) {
                        INK
                    } else {
                        FACE_COLORS[local.index()]
                    };
                    let (seen, distance) = classify(image.pixel(x, y));
                    checked += 1;
                    if seen != expected || distance > 12 {
                        failures.push(format!(
                            "orientation {index} ({:?} turn {}) world {world:?} (local {local:?}) texel \
                             ({column}, {row}) at pixel ({x}, {y}): expected {expected:?}, saw {:?}",
                            orientation.up(),
                            orientation.turn(),
                            image.pixel(x, y)
                        ));
                    }
                }
            }
        }
    }
    assert!(
        failures.is_empty(),
        "{} of {checked} probes wrong:\n{}",
        failures.len(),
        failures.join("\n")
    );
    assert_eq!(checked, 24 * 6 * PROBES.len());
    log::info!("{test}: {checked} texel probes on 24 orientations x 6 faces match the contract");
}

// ---------------------------------------------------------------------------------------------
// Rectangular texture layers
// ---------------------------------------------------------------------------------------------

#[test]
fn rectangular_texture_layers_snap_to_their_own_rows_and_columns() {
    let test = "rectangular_texture_layers_snap_to_their_own_rows_and_columns";
    let Some(gpu) = context_or_skip(test) else {
        return;
    };
    let format = wgpu::TextureFormat::Rgba8UnormSrgb;
    // Twice as many rows as columns, in a checker: sampling a row between two of them mixes
    // red and blue.
    let (columns, rows) = (16_u32, 32_u32);
    const RED: [u8; 3] = [230, 30, 30];
    const BLUE: [u8; 3] = [30, 30, 230];
    let checker = |column: u32, row: u32| {
        if (column + row).is_multiple_of(2) {
            RED
        } else {
            BLUE
        }
    };
    let mut builder =
        TextureArrayBuilder::new("checker 16x32", TextureArrayKind::Albedo, columns, rows).unwrap();
    let levels: Vec<Vec<u8>> = (0..builder.mip_level_count())
        .map(|level| {
            let (width, height) = mip_size(columns, rows, level);
            let mut pixels = Vec::new();
            for row in 0..height {
                for column in 0..width {
                    // The smaller levels hold the checker's average.
                    let color = if level == 0 {
                        checker(column, row)
                    } else {
                        [168, 30, 168]
                    };
                    pixels.extend_from_slice(&[color[0], color[1], color[2], 255]);
                }
            }
            pixels
        })
        .collect();
    let slices: Vec<&[u8]> = levels.iter().map(Vec::as_slice).collect();
    builder.push_layer(&slices).unwrap();
    let albedo = builder.build(&gpu).unwrap();
    let materials = MaterialTable::new(&gpu, &[material([168, 30, 168], 0)]).unwrap();
    let mut renderer = VoxelRenderer::new(&gpu, &config(format), materials, albedo).unwrap();
    renderer.debug_view_mut().select(DebugView::Albedo).unwrap();
    let grid = renderer
        .scene_mut()
        .add_object(ObjectDesc {
            bounds: IAabb::new(IVec3::ZERO, IVec3::splat(32)),
            pose: GridPose::IDENTITY,
        })
        .unwrap();
    // The +Z face of cell (0, 0, 0) at z = 1: image right is +x, image up is +y.
    let quad = PackedQuad::new([0, 0, 0], 1, 1, Face::PosZ, MaterialId(0), 0, false).unwrap();
    let mut layers = [LayerCounts::EMPTY; 3];
    layers[0].regular[storage_slot(Face::PosZ)] = 1;
    insert(&mut renderer, grid, IVec3::ZERO, 0, &(vec![quad], layers));
    let (width, height) = (640, 720);
    let target = HeadlessTarget::new(&gpu, width, height, format).unwrap();
    // Straight in front of the face: a texel is about 41 x 20 pixels.
    let camera = Camera::look_at(
        DVec3::new(0.5, 0.5, 1.95),
        DVec3::new(0.5, 0.5, 1.0),
        DVec3::Y,
        60.0_f32.to_radians(),
    );
    let (image, _) = render(&mut renderer, &gpu, &target, &camera, &Lighting::default());
    save("rectangular_texels_albedo.png", &image);
    let to_clip = camera.frame_to_clip(width as f32 / height as f32);
    let mut failures = Vec::new();
    for row in 0..rows {
        for column in 0..columns {
            // Texel centres; image rows run down, so row r lies at height 1 - (r + 0.5) / rows.
            let point = DVec3::new(
                (f64::from(column) + 0.5) / f64::from(columns),
                1.0 - (f64::from(row) + 0.5) / f64::from(rows),
                1.0,
            );
            let clip = to_clip * point.extend(1.0);
            let ndc = clip.truncate() / clip.w;
            let x = ((ndc.x + 1.0) * 0.5 * f64::from(width)) as u32;
            let y = ((1.0 - ndc.y) * 0.5 * f64::from(height)) as u32;
            let seen = image.pixel(x, y);
            let expected = checker(column, row);
            let distance: u32 = (0..3)
                .map(|c| u32::from(seen[c].abs_diff(expected[c])))
                .sum();
            if distance > 12 {
                failures.push(format!(
                    "texel ({column}, {row}) at pixel ({x}, {y}): expected {expected:?}, saw \
                     {seen:?}"
                ));
            }
        }
    }
    assert!(
        failures.is_empty(),
        "{} of {} texels wrong:\n{}",
        failures.len(),
        columns * rows,
        failures.join("\n")
    );
    log::info!(
        "{test}: all {} texels of a 16x32 layer are crisp",
        columns * rows
    );
}

// ---------------------------------------------------------------------------------------------
// Precision far from the origin
// ---------------------------------------------------------------------------------------------

#[test]
fn a_chunk_3000_km_from_the_origin_renders_like_one_at_the_origin() {
    let test = "a_chunk_3000_km_from_the_origin_renders_like_one_at_the_origin";
    let Some(gpu) = context_or_skip(test) else {
        return;
    };
    let format = wgpu::TextureFormat::Rgba8UnormSrgb;
    let materials = MaterialTable::new(&gpu, &world_materials()).unwrap();
    let mut renderer = VoxelRenderer::new(
        &gpu,
        &config(format),
        materials,
        texture_array(&gpu, &world_textures()),
    )
    .unwrap();
    let target = HeadlessTarget::new(&gpu, 320, 240, format).unwrap();
    let lighting = Lighting::default();
    let structure = |x: i32, y: i32, z: i32| {
        let tower = (4..7).contains(&x) && y < 6 + x && (5..8).contains(&z);
        let floor = y == 0 && x < 20 && z < 16;
        (tower || floor).then_some(Block {
            material: if floor { GRASS } else { STONE },
            layer: MeshLayer::Opaque,
            uv_rot: ((x + z) % 4) as u8,
        })
    };
    let mesh = naive_mesh(structure, &[]);
    let offset = DVec3::new(15.25, 9.5, 21.75);
    let mut images = Vec::new();
    // 3,072 km out: a whole number of cloud anchor periods, so that the clouds (anchored to
    // the world) stand where they stand over the origin and only the geometry is compared.
    let far = 20 * CLOUD_ANCHOR_PERIOD as i32;
    assert!(far >= 3_000_000);
    for origin in [IVec3::ZERO, IVec3::new(far, 0, 0)] {
        let scene = renderer.scene_mut();
        let object = scene
            .add_object(ObjectDesc {
                bounds: IAabb::new(origin, origin + IVec3::splat(32)),
                pose: GridPose::IDENTITY,
            })
            .unwrap();
        let chunk = insert(&mut renderer, object, origin, 0, &mesh);
        let eye = origin.as_dvec3() + offset;
        let camera = Camera::look_at(
            eye,
            origin.as_dvec3() + DVec3::new(6.0, 3.0, 6.0),
            DVec3::Y,
            1.0,
        );
        let (image, _) = render(&mut renderer, &gpu, &target, &camera, &lighting);
        save(&format!("precision_origin_{}.png", origin.x), &image);
        images.push(image);
        renderer.scene_mut().remove_chunk(chunk).unwrap();
        renderer.scene_mut().remove_object(object).unwrap();
    }
    assert_identical(&images[0], &images[1], "the chunk 3,000 km out");
    // An image with content was compared: many distinct colours.
    let mut colours: Vec<[u8; 4]> = images[0]
        .pixels
        .chunks(4)
        .map(|p| [p[0], p[1], p[2], p[3]])
        .collect();
    colours.sort_unstable();
    colours.dedup();
    assert!(colours.len() > 100, "{} colours", colours.len());
}

// ---------------------------------------------------------------------------------------------
// Pipelines, statistics, uploads
// ---------------------------------------------------------------------------------------------

#[test]
fn every_voxel_pipeline_is_created_and_draws_on_each_backend() {
    let test = "every_voxel_pipeline_is_created_and_draws_on_each_backend";
    for gpu in contexts_per_backend(test) {
        let backend = gpu.caps.adapter.backend;
        for format in [
            wgpu::TextureFormat::Rgba8UnormSrgb,
            wgpu::TextureFormat::Bgra8Unorm,
        ] {
            let materials = MaterialTable::new(&gpu, &world_materials()).unwrap();
            let mut renderer = VoxelRenderer::new(
                &gpu,
                &config(format),
                materials,
                texture_array(&gpu, &world_textures()),
            )
            .unwrap_or_else(|error| panic!("{backend} {format:?}: {error}"));
            assert_eq!(renderer.pipeline_count(), PIPELINE_COUNT);
            let camera = build_world(&mut renderer);
            let target = HeadlessTarget::new(&gpu, 64, 48, format).unwrap();
            let lighting = Lighting::default();
            let modes = renderer.available_draw_modes().to_vec();
            let mut images = Vec::new();
            for &mode in &modes {
                renderer.set_draw_mode(mode).unwrap();
                // 16 frames per mode: the compared last frames share the rotation of the
                // cross-fade dither (period 16).
                let mut last = None;
                for _ in 0..16 {
                    let (image, stats) = render(&mut renderer, &gpu, &target, &camera, &lighting);
                    assert!(stats.draw_items > 0);
                    last = Some(image);
                }
                images.push(last.expect("16 frames were rendered"));
            }
            for image in &images[1..] {
                assert_identical(&images[0], image, &format!("{backend} {format:?}"));
            }
            log::info!(
                "{test}: {backend} {format:?}: {PIPELINE_COUNT} pipelines in {:.1} ms, modes {:?}",
                renderer.startup_ms(),
                modes.iter().map(|mode| mode.id()).collect::<Vec<_>>()
            );
        }
    }
}

/// Mean absolute difference of the colour channels of two images, in 8-bit steps.
fn mean_difference(a: &RgbaImage, b: &RgbaImage) -> f64 {
    let (mut sum, mut count) = (0_u64, 0_u64);
    for (pa, pb) in a.pixels.chunks(4).zip(b.pixels.chunks(4)) {
        for channel in 0..3 {
            sum += u64::from(pa[channel].abs_diff(pb[channel]));
            count += 1;
        }
    }
    sum as f64 / count as f64
}

#[test]
fn temporal_anti_aliasing_converges_and_is_the_same_on_every_path() {
    let test = "temporal_anti_aliasing_converges_and_is_the_same_on_every_path";
    let Some(gpu) = context_or_skip(test) else {
        return;
    };
    let format = wgpu::TextureFormat::Rgba8UnormSrgb;
    let make = |temporal_aa: bool| {
        let materials = MaterialTable::new(&gpu, &world_materials()).unwrap();
        let config = VoxelRendererConfig {
            temporal_aa,
            ..config(format)
        };
        VoxelRenderer::new(
            &gpu,
            &config,
            materials,
            texture_array(&gpu, &world_textures()),
        )
        .unwrap()
    };
    let mut renderer = make(true);
    assert!(renderer.temporal_aa());
    let camera = build_world(&mut renderer);
    let lighting = Lighting::default();
    let target = HeadlessTarget::new(&gpu, 240, 135, format).unwrap();
    // Two cycles of the jitter per path, each from a fresh history; the paths start at frame
    // indices that are multiples of the cycle, so their jitter sequences are the same.
    let frames = 2 * cp_render::voxel::JITTER_PHASES as usize;
    let mut finals = Vec::new();
    for mode in renderer.available_draw_modes().to_vec() {
        renderer.set_draw_mode(mode).unwrap();
        renderer.reset_temporal_history();
        let mut images = Vec::new();
        for _ in 0..frames {
            images.push(render(&mut renderer, &gpu, &target, &camera, &lighting).0);
        }
        let (first, before, last) = (&images[0], &images[frames - 2], &images[frames - 1]);
        // The accumulated image of a still camera is calm: consecutive frames hardly differ,
        // far less than the first frame (one jittered frame alone) differs from them.
        let settled = mean_difference(before, last);
        let start = mean_difference(first, last);
        log::info!(
            "{test}: path {}: consecutive frames differ by {settled:.3}, the first from the last \
             by {start:.3} (8-bit steps)",
            mode.id()
        );
        assert!(settled < 0.5, "path {}: {settled}", mode.id());
        assert!(start > settled, "path {}: {start} {settled}", mode.id());
        finals.push(last.clone());
    }
    save("taa_accumulated.png", &finals[0]);
    for image in &finals[1..] {
        assert_identical(&finals[0], image, "temporal anti-aliasing");
    }
    // Without jitter and history: the same picture, apart from the edges that the
    // accumulation smooths.
    let mut plain = make(false);
    let plain_camera = build_world(&mut plain);
    assert_eq!(plain_camera, camera);
    let (single, _) = render(&mut plain, &gpu, &target, &camera, &lighting);
    let apart = mean_difference(&single, &finals[0]);
    log::info!("{test}: the accumulated image differs from a single frame by {apart:.3}");
    assert!(apart < 6.0, "{apart}");
}

#[test]
fn statistics_count_items_draws_and_triangles_of_the_calibration_scene() {
    let test = "statistics_count_items_draws_and_triangles_of_the_calibration_scene";
    let Some(gpu) = context_or_skip(test) else {
        return;
    };
    let format = wgpu::TextureFormat::Rgba8UnormSrgb;
    let materials = MaterialTable::new(&gpu, &world_materials()).unwrap();
    let mut renderer = VoxelRenderer::new(
        &gpu,
        &config(format),
        materials,
        texture_array(&gpu, &world_textures()),
    )
    .unwrap();
    let params = CalibrationParams {
        resident_chunks: 900,
        visible_chunks: 600,
        quads_per_chunk: 49,
        materials: 6,
        aspect: 4.0 / 3.0,
        ..CalibrationParams::default()
    };
    let scene = CalibrationScene::build(renderer.scene_mut(), &params).unwrap();
    let report = renderer.scene_mut().flush_all_uploads(&gpu);
    assert_eq!(report.chunks, 900);
    let target = HeadlessTarget::new(&gpu, 320, 240, format).unwrap();
    let lighting = Lighting::default();
    for mode in renderer.available_draw_modes().to_vec() {
        renderer.set_draw_mode(mode).unwrap();
        let (image, stats) = render(&mut renderer, &gpu, &target, &scene.camera, &lighting);
        if mode == DrawMode::B {
            save("calibration_b.png", &image);
        }
        renderer.wait_for_gpu(&gpu).unwrap();
        let visible = match mode {
            DrawMode::A => renderer.latest_gpu_counts().unwrap(),
            _ => stats.visible.unwrap(),
        };
        assert_eq!(stats.draw_items, scene.expected.draw_items, "{mode:?}");
        assert_eq!(visible.total_draws(), scene.expected.draws, "{mode:?}");
        assert_eq!(visible.quads, scene.expected.quads, "{mode:?}");
        assert_eq!(visible.triangles(), scene.expected.triangles, "{mode:?}");
        assert_eq!(stats.chunks, 900);
    }
    scene.remove(renderer.scene_mut()).unwrap();
    assert_eq!(renderer.scene().stats().draw_items, 0);
    assert_eq!(
        renderer.scene().stats().pool_free_quads,
        renderer.scene().stats().pool_quads
    );
}

#[test]
fn uploads_respect_the_budget_and_replacements_never_leave_a_gap() {
    let test = "uploads_respect_the_budget_and_replacements_never_leave_a_gap";
    let Some(gpu) = context_or_skip(test) else {
        return;
    };
    let format = wgpu::TextureFormat::Rgba8UnormSrgb;
    let materials = MaterialTable::new(&gpu, &world_materials()).unwrap();
    let floor = |x: i32, y: i32, z: i32| {
        (y == 0 && x < 10 && z < 10).then_some(Block {
            material: GRASS,
            layer: MeshLayer::Opaque,
            uv_rot: 0,
        })
    };
    let mesh = naive_mesh(floor, &[]);
    let quads_per_chunk = mesh.0.len() as u64;
    let mut config = config(format);
    // Room for one chunk per frame, not for two.
    config.upload_budget_bytes = quads_per_chunk * 12;
    let mut renderer = VoxelRenderer::new(
        &gpu,
        &config,
        materials,
        texture_array(&gpu, &world_textures()),
    )
    .unwrap();
    let grid = renderer
        .scene_mut()
        .add_object(ObjectDesc {
            bounds: IAabb::new(IVec3::splat(-256), IVec3::splat(256)),
            pose: GridPose::IDENTITY,
        })
        .unwrap();
    let chunks: Vec<ChunkId> = (0..3)
        .map(|k| insert(&mut renderer, grid, IVec3::new(32 * k, 0, 0), 0, &mesh))
        .collect();
    let target = HeadlessTarget::new(&gpu, 64, 64, format).unwrap();
    let camera = Camera::look_at(
        DVec3::new(48.5, 60.0, 40.0),
        DVec3::new(48.0, 0.0, 5.0),
        DVec3::Y,
        1.2,
    );
    let lighting = Lighting::default();
    for frame in 1..=3_u32 {
        let (_, stats) = render(&mut renderer, &gpu, &target, &camera, &lighting);
        assert_eq!(stats.uploads.chunks, 1, "frame {frame}");
        assert_eq!(stats.uploads.pending_chunks, 3 - frame);
        let resident = chunks
            .iter()
            .filter(|&&chunk| renderer.scene().is_resident(chunk).unwrap())
            .count();
        assert_eq!(resident, frame as usize);
    }
    // A replacement keeps the old mesh drawn until its upload.
    let replacement = naive_mesh(
        |x, y, z| {
            (y < 2 && x < 10 && z < 10).then_some(Block {
                material: STONE,
                layer: MeshLayer::Opaque,
                uv_rot: 0,
            })
        },
        &[],
    );
    renderer.scene_mut().set_upload_budget(0);
    renderer
        .scene_mut()
        .replace_chunk(
            chunks[0],
            &ChunkUpload {
                quads: &replacement.0,
                layers: replacement.1,
                origin: IVec3::ZERO,
                level: 0,
                object: grid,
            },
        )
        .unwrap();
    assert!(renderer.scene().has_pending_upload(chunks[0]).unwrap());
    let (_, stats) = render(&mut renderer, &gpu, &target, &camera, &lighting);
    // One mesh per frame always goes, even with a zero budget.
    assert_eq!(stats.uploads.chunks, 1);
    assert!(!renderer.scene().has_pending_upload(chunks[0]).unwrap());
    let scene = renderer.scene().stats();
    assert_eq!(scene.resident_chunks, 3);
    assert_eq!(scene.draw_items, 3);
    assert_eq!(scene.pending_chunks, 0);
}
