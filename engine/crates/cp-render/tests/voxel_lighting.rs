//! Tests of the voxel shading against the real GPU, through the public API only: the per-texel
//! tint weight of tinted textures (the convention of `cp_proctex::tint_weight`), the
//! screen-space sun shadows and the ambient obscurance of the lighting pass.
//!
//! Every test skips with a log line when the machine has no GPU adapter (contract §2.7). The
//! images are written to `engine/target/render-tests/` for inspection.

#[path = "common/png.rs"]
mod png;

use std::path::PathBuf;
use std::sync::Once;

use cp_core::face::Face;
use cp_core::ids::MaterialId;
use cp_core::math::{DVec3, IAabb, IVec3, Vec3};
use cp_core::quad::PackedQuad;
use cp_render::debug_view::DebugView;
use cp_render::gpu::{GpuContext, GpuError, GpuOptions};
use cp_render::headless::{HeadlessTarget, RgbaImage};
use cp_render::material::{GpuMaterial, MaterialTable};
use cp_render::texture::{TextureArray, TextureArrayBuilder, TextureArrayKind};
use cp_render::voxel::{
    Camera, ChunkUpload, FrameInput, GridPose, LayerCounts, Lighting, ObjectDesc, VoxelRenderer,
    VoxelRendererConfig, storage_slot,
};
use cp_render::wgpu;

struct TestLogger;

static LOGGER: TestLogger = TestLogger;

impl log::Log for TestLogger {
    fn enabled(&self, metadata: &log::Metadata<'_>) -> bool {
        metadata.level() <= log::Level::Error
            || (metadata.level() <= log::Level::Info
                && ["cp_render", "voxel_lighting"]
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

fn context_or_skip(test: &str) -> Option<GpuContext> {
    static INIT: Once = Once::new();
    INIT.call_once(|| {
        log::set_logger(&LOGGER).expect("no other logger is installed in this test binary");
        log::set_max_level(log::LevelFilter::Info);
    });
    match GpuContext::new_blocking(&GpuOptions::default(), None) {
        Ok(gpu) => Some(gpu),
        Err(error @ (GpuError::NoAdapter { .. } | GpuError::BackendNotCompiled { .. })) => {
            log::warn!("{test}: skipped: {error}");
            None
        }
        Err(error) => panic!("{test}: GPU context creation failed: {error}"),
    }
}

fn save(name: &str, image: &RgbaImage) {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../target/render-tests");
    std::fs::create_dir_all(&dir).expect("create target/render-tests");
    let path = dir.join(name);
    let bytes = png::encode_rgba(image.width, image.height, &image.pixels, image.srgb);
    std::fs::write(&path, bytes).unwrap_or_else(|error| panic!("{}: {error}", path.display()));
}

const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8UnormSrgb;

/// A one-layer 16 x 16 albedo array of the texels `texel(x, y)`, with a box-filtered mip chain.
fn texture_array(gpu: &GpuContext, texel: impl Fn(u32, u32) -> [u8; 4]) -> TextureArray {
    let mut levels: Vec<Vec<u8>> = vec![(0..16 * 16).flat_map(|i| texel(i % 16, i / 16)).collect()];
    let mut side = 16_usize;
    while side > 1 {
        let previous = levels.last().expect("the chain starts with level 0");
        let half = side / 2;
        let mut next = vec![0_u8; half * half * 4];
        for y in 0..half {
            for x in 0..half {
                for channel in 0..4 {
                    let sum: u32 = [(0, 0), (1, 0), (0, 1), (1, 1)]
                        .iter()
                        .map(|&(dx, dy)| {
                            u32::from(previous[((2 * y + dy) * side + 2 * x + dx) * 4 + channel])
                        })
                        .sum();
                    next[(y * half + x) * 4 + channel] = (sum / 4) as u8;
                }
            }
        }
        levels.push(next);
        side = half;
    }
    let mut builder = TextureArrayBuilder::new("lighting test", TextureArrayKind::Albedo, 16, 16)
        .expect("16x16 layers");
    let slices: Vec<&[u8]> = levels.iter().map(Vec::as_slice).collect();
    builder.push_layer(&slices).expect("a full mip chain");
    builder.build(gpu).expect("create the texture array")
}

/// A renderer whose only material is `material`, textured with `albedo`.
fn renderer(gpu: &GpuContext, material: GpuMaterial, albedo: TextureArray) -> VoxelRenderer {
    // Single frames, no temporal accumulation: the probes read one frame.
    let config = VoxelRendererConfig {
        target_format: FORMAT,
        pool_bytes: 8 << 20,
        max_draw_items: 64,
        max_objects: 4,
        upload_budget_bytes: u64::MAX,
        anisotropy: 1,
        temporal_aa: false,
    };
    let table = MaterialTable::new(gpu, &[material]).expect("create the material table");
    VoxelRenderer::new(gpu, &config, table, albedo).expect("create the renderer")
}

/// Adds one level-0 chunk at the origin holding the cells where `solid(x, y, z)` holds: one
/// opaque quad of material 0 per exposed cell face, no walls.
fn add_chunk(renderer: &mut VoxelRenderer, solid: impl Fn(i32, i32, i32) -> bool) {
    let mut ranges: Vec<Vec<PackedQuad>> = vec![Vec::new(); 6];
    for y in 0..32 {
        for z in 0..32 {
            for x in 0..32 {
                if !solid(x, y, z) {
                    continue;
                }
                for face in Face::ALL {
                    let n = face.normal();
                    if solid(x + n.x, y + n.y, z + n.z) {
                        continue;
                    }
                    let quad = PackedQuad::new(
                        [x as u8, y as u8, z as u8],
                        1,
                        1,
                        face,
                        MaterialId(0),
                        0,
                        false,
                    )
                    .expect("test quads lie inside the chunk");
                    ranges[storage_slot(face)].push(quad);
                }
            }
        }
    }
    let mut quads = Vec::new();
    let mut slots = [0_u32; 12];
    for (slot, range) in ranges.iter().enumerate() {
        slots[slot] = range.len() as u32;
        quads.extend_from_slice(range);
    }
    let layers = [
        LayerCounts::from_slots(slots),
        LayerCounts::EMPTY,
        LayerCounts::EMPTY,
    ];
    let scene = renderer.scene_mut();
    let object = scene
        .add_object(ObjectDesc {
            bounds: IAabb::new(IVec3::ZERO, IVec3::splat(32)),
            pose: GridPose::IDENTITY,
        })
        .expect("add the test object");
    scene
        .insert_chunk(&ChunkUpload {
            quads: &quads,
            layers,
            origin: IVec3::ZERO,
            level: 0,
            object,
        })
        .expect("insert the test chunk");
}

fn render(
    renderer: &mut VoxelRenderer,
    gpu: &GpuContext,
    camera: &Camera,
    lighting: &Lighting,
    (width, height): (u32, u32),
) -> RgbaImage {
    let target = HeadlessTarget::new(gpu, width, height, FORMAT).expect("create the target");
    let input = FrameInput {
        camera,
        lighting,
        target: target.view(),
        target_size: target.size(),
        overlay: None,
    };
    renderer.render(gpu, &input).expect("render a frame");
    target.capture_image(gpu).expect("read the frame back")
}

/// The pixel of the image of `camera` that shows the frame point `point`.
fn pixel_at(image: &RgbaImage, camera: &Camera, point: DVec3) -> [u8; 4] {
    let (width, height) = (image.width, image.height);
    let clip = camera.frame_to_clip(width as f32 / height as f32) * point.extend(1.0);
    let ndc = clip.truncate() / clip.w;
    let x = ((ndc.x + 1.0) * 0.5 * f64::from(width)) as u32;
    let y = ((1.0 - ndc.y) * 0.5 * f64::from(height)) as u32;
    image.pixel(x, y)
}

/// Rec.709 luma of the encoded channels.
fn luma(pixel: [u8; 4]) -> f64 {
    0.2126 * f64::from(pixel[0]) + 0.7152 * f64::from(pixel[1]) + 0.0722 * f64::from(pixel[2])
}

#[test]
fn tinted_textures_keep_the_colour_of_their_saturated_texels() {
    let test = "tinted_textures_keep_the_colour_of_their_saturated_texels";
    let Some(gpu) = context_or_skip(test) else {
        return;
    };
    // Like grass_side: the left half of the layer is a near-neutral grey that takes the tint,
    // the right half the brown of the soil, which keeps its colour.
    const GREY: [u8; 4] = [200, 200, 200, 255];
    const SOIL: [u8; 4] = [134, 96, 66, 255];
    let albedo = texture_array(&gpu, |x, _| if x < 8 { GREY } else { SOIL });
    let material = GpuMaterial {
        average_color: [0.3, 0.3, 0.3],
        roughness: 1.0,
        texture_layer: 0,
        tint_class: 1,
        ..GpuMaterial::default()
    };
    let mut renderer = renderer(&gpu, material, albedo);
    renderer.debug_view_mut().select(DebugView::Albedo).unwrap();
    // The +Z face of cell (0, 0, 0) at z = 1: image right is +x.
    add_chunk(&mut renderer, |x, y, z| (x, y, z) == (0, 0, 0));
    let camera = Camera::look_at(
        DVec3::new(0.5, 0.5, 1.95),
        DVec3::new(0.5, 0.5, 1.0),
        DVec3::Y,
        60.0_f32.to_radians(),
    );
    let mut lighting = Lighting::default();
    lighting.tints[1] = Vec3::new(0.1, 0.9, 0.1);
    let image = render(&mut renderer, &gpu, &camera, &lighting, (256, 256));
    save("tint_weight_albedo.png", &image);
    let grey = pixel_at(&image, &camera, DVec3::new(0.25, 0.5, 1.0));
    let soil = pixel_at(&image, &camera, DVec3::new(0.75, 0.5, 1.0));
    let [r, g, b, _] = grey.map(u16::from);
    assert!(
        g > r + 60 && g > b + 60,
        "the neutral half takes the green tint: {grey:?}"
    );
    let distance: u32 = (0..3).map(|c| u32::from(soil[c].abs_diff(SOIL[c]))).sum();
    assert!(
        distance <= 12,
        "the soil keeps its colour {SOIL:?}: {soil:?}"
    );
    log::info!("{test}: tinted grey {grey:?}, untinted soil {soil:?}");
}

#[test]
fn a_pillar_shades_the_ground_behind_and_at_its_foot() {
    let test = "a_pillar_shades_the_ground_behind_and_at_its_foot";
    let Some(gpu) = context_or_skip(test) else {
        return;
    };
    let albedo = texture_array(&gpu, |_, _| [180, 180, 180, 255]);
    let material = GpuMaterial {
        average_color: [0.46, 0.46, 0.46],
        roughness: 1.0,
        texture_layer: 0,
        ..GpuMaterial::default()
    };
    let mut renderer = renderer(&gpu, material, albedo);
    // A floor one cell thick with its top at y = 1, and a pillar of six cells on it that
    // occupies x 16..17, y 1..7, z 16..17.
    add_chunk(&mut renderer, |x, y, z| {
        let floor = (0..32).contains(&x) && (0..32).contains(&z) && y == 0;
        floor || (x == 16 && z == 16 && (1..7).contains(&y))
    });
    // The sun 45 degrees above +X, so the shadow falls six metres along -X; no fog. The camera
    // looks from the south-west, where it sees the pillar's west and south faces.
    let lighting = Lighting {
        sun_direction: Vec3::new(1.0, 1.0, 0.0).normalize(),
        fog_density: 0.0,
        ..Lighting::default()
    };
    let camera = Camera::look_at(
        DVec3::new(6.0, 20.0, 30.0),
        DVec3::new(15.0, 1.0, 16.5),
        DVec3::Y,
        50.0_f32.to_radians(),
    );
    let image = render(&mut renderer, &gpu, &camera, &lighting, (640, 480));
    save("pillar_shadow_lit.png", &image);
    let floor = |x: f64, z: f64| luma(pixel_at(&image, &camera, DVec3::new(x, 1.0, z)));
    // In the sun: east of the pillar and far from it.
    let lit = floor(19.5, 16.5);
    let open = floor(10.5, 8.5);
    // In the shadow: three and three and a half metres west of the pillar, and at its foot.
    let shadowed = floor(13.5, 16.5);
    let along = floor(12.5, 16.5);
    let foot = floor(15.6, 16.5);
    log::info!(
        "{test}: lit {lit:.1}, open {open:.1}, shadow {shadowed:.1}, farther {along:.1}, \
         foot {foot:.1}"
    );
    assert!(
        (lit - open).abs() < 0.1 * open,
        "the open floor is evenly lit: {lit:.1} and {open:.1}"
    );
    assert!(
        shadowed < 0.75 * lit,
        "the pillar casts a shadow: {shadowed:.1} against {lit:.1} in the sun"
    );
    // The shadow has only ambient light, so its brightness is uniform away from the pillar...
    assert!(
        (shadowed - along).abs() < 0.05 * along,
        "the shadow is even: {shadowed:.1} and {along:.1}"
    );
    // ...and the ambient obscurance darkens it in the corner at the pillar's foot.
    assert!(
        foot < 0.95 * along,
        "the foot of the pillar is obscured: {foot:.1} against {along:.1}"
    );
}
