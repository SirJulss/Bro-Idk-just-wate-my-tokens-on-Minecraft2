//! Adapter-gated tests of the voxel pipeline that need the crate's internals: the GPU
//! compaction of draw path A produces exactly the CPU's draw lists, the prefix sum works over
//! more blocks than one workgroup holds, and greedy meshes render without cracks at their
//! T-junctions. They skip (log and return) without a GPU.

use cp_core::face::Face;
use cp_core::ids::MaterialId;
use cp_core::math::{DVec3, IAabb, IVec3};
use cp_core::quad::PackedQuad;

use super::*;
use crate::debug_view::DebugView;
use crate::gpu::{GpuContext, GpuError, GpuOptions};
use crate::headless::{HeadlessTarget, RgbaImage};
use crate::material::{GpuMaterial, MaterialTable};
use crate::texture::{TextureArray, TextureArrayBuilder, TextureArrayKind};

fn context_or_skip(test: &str) -> Option<GpuContext> {
    crate::gpu::testing::init_logging();
    match GpuContext::new_blocking(&GpuOptions::default(), None) {
        Ok(gpu) => Some(gpu),
        Err(error @ (GpuError::NoAdapter { .. } | GpuError::BackendNotCompiled { .. })) => {
            crate::gpu::testing::skip(test, &error);
            None
        }
        Err(error) => panic!("{test}: GPU context creation failed: {error}"),
    }
}

fn white_texture(gpu: &GpuContext) -> TextureArray {
    let mut builder =
        TextureArrayBuilder::new("test white", TextureArrayKind::Albedo, 1, 1).unwrap();
    builder.push_layer(&[&[255, 255, 255, 255]]).unwrap();
    builder.build(gpu).unwrap()
}

fn grey_materials(gpu: &GpuContext) -> MaterialTable {
    let material = GpuMaterial {
        average_color: [0.5, 0.5, 0.5],
        roughness: 0.8,
        ..GpuMaterial::default()
    };
    MaterialTable::new(gpu, &[material; 4]).unwrap()
}

fn renderer(gpu: &GpuContext, max_draw_items: u32) -> VoxelRenderer {
    renderer_for(gpu, max_draw_items, wgpu::TextureFormat::Rgba8UnormSrgb)
}

fn renderer_for(
    gpu: &GpuContext,
    max_draw_items: u32,
    target_format: wgpu::TextureFormat,
) -> VoxelRenderer {
    // Without temporal anti-aliasing: the tests compare single frames pixel by pixel.
    let config = VoxelRendererConfig {
        target_format,
        pool_bytes: 64 << 20,
        max_draw_items,
        max_objects: 4,
        upload_budget_bytes: u64::MAX,
        anisotropy: 1,
        temporal_aa: false,
    };
    VoxelRenderer::new(gpu, &config, grey_materials(gpu), white_texture(gpu)).unwrap()
}

struct Random(u64);

impl Random {
    fn below(&mut self, n: u64) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0 % n
    }
}

/// A mesh with `counts[layer][slot]` single-cell quads on each slot's face.
fn mesh(counts: &[[u32; 12]; 3]) -> (Vec<PackedQuad>, [LayerCounts; 3]) {
    let mut quads = Vec::new();
    for slots in counts {
        for (slot, &count) in slots.iter().enumerate() {
            let face = FACE_STORAGE_ORDER[slot % 6];
            for k in 0..count {
                let cell = [(k % 32) as u8, ((k / 32) % 32) as u8, (slot * 2) as u8];
                quads.push(
                    PackedQuad::new(cell, 1, 1, face, MaterialId((k % 4) as u16), 0, false)
                        .unwrap(),
                );
            }
        }
    }
    (quads, counts.map(LayerCounts::from_slots))
}

#[test]
fn gpu_compaction_produces_the_cpu_draw_lists() {
    let test = "gpu_compaction_produces_the_cpu_draw_lists";
    let Some(gpu) = context_or_skip(test) else {
        return;
    };
    let mut renderer = renderer(&gpu, 8192);
    if !renderer.available_draw_modes().contains(&DrawMode::A) {
        crate::gpu::testing::skip(test, &"the device has no draw path A");
        return;
    }
    let mut random = Random(0x5DEE_CE66_D1CE_4E5B);
    let scene = renderer.scene_mut();
    let grids: Vec<ObjectId> = (0..2)
        .map(|k| {
            scene
                .add_object(ObjectDesc {
                    bounds: IAabb::new(IVec3::splat(-8192), IVec3::splat(8192)),
                    pose: GridPose {
                        rotation: cp_core::det::dquat_from_axis_angle(DVec3::Y, 0.4 * f64::from(k)),
                        translation: DVec3::new(f64::from(k) * 37.5, 0.0, 0.0),
                    },
                })
                .unwrap()
        })
        .collect();
    let mut chunks = Vec::new();
    for index in 0..2500_u32 {
        let level = random.below(4) as u8;
        let extent = 32 << level;
        let origin = IVec3::new(
            (random.below(24) as i32 - 12) * extent,
            (random.below(8) as i32 - 4) * extent,
            (random.below(24) as i32 - 12) * extent,
        );
        let mut counts = [[0_u32; 12]; 3];
        for layer in &mut counts {
            if random.below(3) == 0 {
                continue;
            }
            for slot in layer.iter_mut() {
                *slot = if random.below(3) == 0 {
                    0
                } else {
                    random.below(20) as u32
                };
            }
        }
        let (quads, layers) = mesh(&counts);
        let grid = grids[(index % 2) as usize];
        // Several chunks may share an origin; the culling does not care.
        let id = scene
            .insert_chunk(&ChunkUpload {
                quads: &quads,
                layers,
                origin,
                level,
                object: grid,
            })
            .unwrap();
        chunks.push(id);
    }
    for (k, &id) in chunks.iter().enumerate() {
        scene
            .set_wall_mask(id, WallMask::from_bits((k * 7 % 64) as u8))
            .unwrap();
        if k % 9 == 0 {
            scene.set_fade(id, Fade::new(0.5, k % 2 == 0)).unwrap();
        }
    }
    scene.flush_all_uploads(&gpu);
    let items = renderer.scene().stats().draw_items;
    assert!(items > 2000, "{items}");

    let target = HeadlessTarget::new(&gpu, 96, 64, wgpu::TextureFormat::Rgba8UnormSrgb).unwrap();
    let lighting = Lighting::default();
    let mut compared_draws = 0;
    for (eye, look) in [
        (
            DVec3::new(3.25, 40.5, 2.75),
            DVec3::new(100.0, -20.0, -50.0),
        ),
        (DVec3::new(-200.0, 10.0, 150.0), DVec3::new(0.0, 0.0, 0.0)),
        (DVec3::new(0.5, -300.0, 0.5), DVec3::new(10.0, 0.0, 5.0)),
    ] {
        let camera = Camera::look_at(eye, look, DVec3::Y, 1.2);
        renderer.set_draw_mode(DrawMode::A).unwrap();
        let input = FrameInput {
            camera: &camera,
            lighting: &lighting,
            target: target.view(),
            target_size: target.size(),
            overlay: None,
        };
        renderer.render(&gpu, &input).unwrap();
        let lists = renderer.read_back_draw_lists(&gpu).unwrap();
        for class in 0..DRAW_CLASSES as usize {
            let (gpu_args, gpu_buckets) = &lists.gpu[class];
            let (cpu_args, cpu_buckets) = &lists.cpu[class];
            assert_eq!(
                gpu_args.len(),
                cpu_args.len(),
                "class {class} from {eye:?}: draw counts"
            );
            assert_eq!(
                gpu_args, cpu_args,
                "class {class} from {eye:?}: indirect arguments"
            );
            assert_eq!(
                gpu_buckets, cpu_buckets,
                "class {class} from {eye:?}: buckets"
            );
            compared_draws += gpu_args.len() / 5;
        }
        assert_eq!(lists.gpu_quads, lists.cpu_quads);
    }
    assert!(compared_draws > 1000, "{compared_draws}");
    log::info!("{test}: {compared_draws} draws of {items} items identical on GPU and CPU");
}

#[test]
fn the_prefix_sum_spans_many_workgroups() {
    let test = "the_prefix_sum_spans_many_workgroups";
    let Some(gpu) = context_or_skip(test) else {
        return;
    };
    // 700 blocks per segment: the scan loops three times over its 256 threads.
    let mut renderer = renderer(&gpu, 700 * CULL_WORKGROUP);
    let mut random = Random(99);
    let rows: Vec<Vec<u32>> = (0..SEGMENTS)
        .map(|segment| {
            (0..700)
                .map(|_| {
                    if segment % 5 == 0 {
                        0
                    } else {
                        random.below(1537) as u32
                    }
                })
                .collect()
        })
        .collect();
    let (scanned, totals) = renderer.run_scan(&gpu, &rows).unwrap();
    for (segment, row) in rows.iter().enumerate() {
        let mut sum = 0;
        let expected: Vec<u32> = row
            .iter()
            .map(|&value| {
                let before = sum;
                sum += value;
                before
            })
            .collect();
        assert_eq!(scanned[segment], expected, "segment {segment}");
        assert_eq!(totals[segment], sum, "segment {segment}");
    }
}

#[test]
fn every_pipeline_exists_and_a_frame_renders_in_every_mode() {
    let test = "every_pipeline_exists_and_a_frame_renders_in_every_mode";
    let Some(gpu) = context_or_skip(test) else {
        return;
    };
    // A unorm target: the composite pass encodes sRGB itself.
    let mut renderer = renderer_for(&gpu, 64, wgpu::TextureFormat::Rgba8Unorm);
    assert_eq!(renderer.pipeline_count(), PIPELINE_COUNT);
    let grid = renderer
        .scene_mut()
        .add_object(ObjectDesc {
            bounds: IAabb::new(IVec3::splat(-64), IVec3::splat(64)),
            pose: GridPose::IDENTITY,
        })
        .unwrap();
    let mut counts = [[0; 12]; 3];
    counts[0][1] = 5;
    counts[1][2] = 3;
    let (quads, layers) = mesh(&counts);
    renderer
        .scene_mut()
        .insert_chunk(&ChunkUpload {
            quads: &quads,
            layers,
            origin: IVec3::ZERO,
            level: 0,
            object: grid,
        })
        .unwrap();
    let target = HeadlessTarget::new(&gpu, 32, 32, wgpu::TextureFormat::Rgba8Unorm).unwrap();
    let camera = Camera::look_at(
        DVec3::new(16.5, 30.0, 40.0),
        DVec3::new(16.0, 1.0, 4.0),
        DVec3::Y,
        1.0,
    );
    let lighting = Lighting::default();
    let modes = renderer.available_draw_modes().to_vec();
    for mode in modes {
        renderer.set_draw_mode(mode).unwrap();
        for view in crate::debug_view::DebugView::ALL {
            renderer.debug_view_mut().select(view).unwrap();
            let stats = renderer
                .render(
                    &gpu,
                    &FrameInput {
                        camera: &camera,
                        lighting: &lighting,
                        target: target.view(),
                        target_size: (32, 32),
                        overlay: None,
                    },
                )
                .unwrap();
            assert_eq!(stats.mode, mode);
            assert_eq!(stats.draw_items, 2);
        }
    }
    renderer.wait_for_gpu(&gpu).unwrap();
    let timings = renderer.latest_timings();
    if renderer.frame_graph().gpu_timing_enabled() {
        let timings = timings.expect("timings arrive after a wait");
        assert!(timings.pass(PASS_GBUFFER).is_some());
    }
}

// ---------------------------------------------------------------------------------------------
// Seams of greedy meshes
// ---------------------------------------------------------------------------------------------

/// The opaque layer of a chunk mesh, regular ranges only.
type Mesh = (Vec<PackedQuad>, [LayerCounts; 3]);

/// Materials of the seam scenes: terrain tops (and the rows), terrain sides.
const TOP: u16 = 0;
const SIDE: u16 = 1;

/// Level-0 chunks per side of the seam terrain.
const SEAM_CHUNKS: i32 = 12;

fn seam_hash(x: i32, y: i32, z: i32) -> u32 {
    let mut h = (x as u32).wrapping_mul(0x8DA6_B343)
        ^ (y as u32).wrapping_mul(0xD816_3841)
        ^ (z as u32).wrapping_mul(0xCB1A_B31F);
    h ^= h >> 15;
    h = h.wrapping_mul(0x2C1B_3C6D);
    h ^ (h >> 12)
}

/// Column height of the seam terrain in cells (the top face lies at `y = height`): three
/// octaves of bilinear value noise inside the meshed square, `3..=25`; 0 outside it.
fn seam_height(x: i32, z: i32) -> i32 {
    let side = 32 * SEAM_CHUNKS;
    if !(0..side).contains(&x) || !(0..side).contains(&z) {
        return 0;
    }
    let octave = |cell: i32, amplitude: u32, seed: i32| {
        let (cx, cz) = (x.div_euclid(cell), z.div_euclid(cell));
        let (fx, fz) = (x.rem_euclid(cell), z.rem_euclid(cell));
        let corner = |i: i32, k: i32| (seam_hash(i, seed, k) % (amplitude + 1)) as i32;
        let near = corner(cx, cz) * (cell - fx) + corner(cx + 1, cz) * fx;
        let far = corner(cx, cz + 1) * (cell - fx) + corner(cx + 1, cz + 1) * fx;
        (near * (cell - fz) + far * fz) / (cell * cell)
    };
    3 + octave(48, 14, 1) + octave(11, 6, 2) + octave(3, 2, 3)
}

fn opaque_mesh(ranges: [Vec<PackedQuad>; 6]) -> Mesh {
    let mut slots = [0; 12];
    let mut quads = Vec::new();
    for (slot, range) in ranges.into_iter().enumerate() {
        slots[slot] = range.len() as u32;
        quads.extend(range);
    }
    let mut layers = [LayerCounts::EMPTY; 3];
    layers[MeshLayer::Opaque.index()] = LayerCounts::from_slots(slots);
    (quads, layers)
}

fn seam_quad(cell: [i32; 3], w: i32, h: i32, face: Face, material: u16) -> PackedQuad {
    let cell = cell.map(|v| u8::try_from(v).expect("a cell of the chunk"));
    PackedQuad::new(cell, w as u8, h as u8, face, MaterialId(material), 0, false)
        .expect("a quad inside the chunk")
}

/// The terrain chunk whose min column is `(ox, oz)`, merged the way a greedy mesher merges:
/// tops into rectangles of one height, the side faces of a column into one quad per side.
/// Corners of one quad end on the edges of others everywhere (inside the chunk, along its
/// borders, where sides meet tops), so the mesh is full of T-junctions.
fn greedy_terrain_chunk(ox: i32, oz: i32) -> Mesh {
    let height = |x: i32, z: i32| seam_height(ox + x, oz + z);
    let mut ranges: [Vec<PackedQuad>; 6] = Default::default();
    let mut used = [[false; 32]; 32];
    for x in 0..32 {
        for z in 0..32 {
            if used[x as usize][z as usize] {
                continue;
            }
            let top = height(x, z);
            let fits = |x: i32, z: i32, used: &[[bool; 32]; 32]| {
                !used[x as usize][z as usize] && height(x, z) == top
            };
            // +Y faces: u is z, v is x.
            let mut width = 1;
            while z + width < 32 && fits(x, z + width, &used) {
                width += 1;
            }
            let mut depth = 1;
            while x + depth < 32 && (z..z + width).all(|k| fits(x + depth, k, &used)) {
                depth += 1;
            }
            for row in &mut used[x as usize..(x + depth) as usize] {
                row[z as usize..(z + width) as usize].fill(true);
            }
            ranges[storage_slot(Face::PosY)].push(seam_quad(
                [x, top - 1, z],
                width,
                depth,
                Face::PosY,
                TOP,
            ));
        }
    }
    for x in 0..32 {
        for z in 0..32 {
            let top = height(x, z);
            for face in [Face::PosX, Face::NegX, Face::PosZ, Face::NegZ] {
                let n = face.normal();
                let low = height(x + n.x, z + n.z);
                if low >= top {
                    continue;
                }
                // The run is along u (y) on ±X faces and along v (y) on ±Z faces.
                let run = top - low;
                let (w, h) = if face.u_axis() == 1 {
                    (run, 1)
                } else {
                    (1, run)
                };
                ranges[storage_slot(face)].push(seam_quad([x, low, z], w, h, face, SIDE));
            }
        }
    }
    opaque_mesh(ranges)
}

/// +Y faces of the cells at y = 0 of a chunk in rows along x: even rows are one quad over the
/// whole row, odd rows 32 single-cell quads, so every corner of an odd row lies on an edge of
/// its neighbour rows (the worst case of greedy meshing).
fn greedy_rows_chunk() -> Mesh {
    let mut ranges: [Vec<PackedQuad>; 6] = Default::default();
    let slot = storage_slot(Face::PosY);
    for x in 0..32 {
        if x % 2 == 0 {
            ranges[slot].push(seam_quad([x, 0, 0], 32, 1, Face::PosY, TOP));
        } else {
            for z in 0..32 {
                ranges[slot].push(seam_quad([x, 0, z], 1, 1, Face::PosY, TOP));
            }
        }
    }
    opaque_mesh(ranges)
}

/// The same faces with one quad per cell face: neighbours share whole edges, corner on
/// corner, so the twin has no T-junction.
fn split_mesh(mesh: &Mesh) -> Mesh {
    let mut ranges: [Vec<PackedQuad>; 6] = Default::default();
    for &quad in &mesh.0 {
        let face = quad.face();
        for cell in quad.covered_cells() {
            let unit = PackedQuad::new(
                cell,
                1,
                1,
                face,
                quad.material(),
                quad.uv_rot(),
                quad.is_authored(),
            )
            .expect("a cell of a valid quad");
            ranges[storage_slot(face)].push(unit);
        }
    }
    opaque_mesh(ranges)
}

/// Cameras over the seam terrain with cells of `cell` metres: high and far, low over the
/// ground, looking across.
fn terrain_cameras(cell: f64) -> Vec<Camera> {
    let low = |x: f64, z: f64| DVec3::new(x, f64::from(seam_height(x as i32, z as i32)) + 1.7, z);
    [
        (DVec3::new(20.3, 40.0, 25.1), DVec3::new(300.0, 8.0, 260.0)),
        (DVec3::new(192.4, 34.2, 10.7), DVec3::new(190.0, 5.0, 300.0)),
        (DVec3::new(370.6, 36.0, 370.2), DVec3::new(50.0, 5.0, 80.0)),
        (low(100.3, 100.6), DVec3::new(300.0, 10.0, 250.0)),
        (low(300.2, 60.9), DVec3::new(60.0, 6.0, 200.0)),
        (DVec3::new(50.7, 70.0, 330.3), DVec3::new(250.0, 0.0, 100.0)),
        (
            low(200.5, 200.5) + DVec3::new(0.0, 10.0, 0.0),
            DVec3::new(230.0, 4.0, 120.0),
        ),
        (DVec3::new(10.2, 28.0, 190.0), DVec3::new(380.0, 6.0, 190.0)),
    ]
    .into_iter()
    .map(|(eye, target)| {
        Camera::look_at(eye * cell, target * cell, DVec3::Y, 60.0_f32.to_radians())
    })
    .collect()
}

/// Cameras just above the rows plane (y = 1), looking along it; the plane meets the view rays
/// at the centre of the image at 0.1° to 0.8°.
fn rows_cameras() -> Vec<Camera> {
    (0..8_u32)
        .map(|k| {
            let f = f64::from(k);
            // Camera 0 is 0.4 m above the plane.
            let height = if k == 0 { 1.4 } else { 2.0 + 0.37 * f };
            Camera::look_at(
                DVec3::new(3.37 + 0.613 * f, height, 5.11 + 0.271 * f),
                DVec3::new(200.0 + 7.0 * f, 1.0, 140.0 - 3.0 * f),
                DVec3::Y,
                60.0_f32.to_radians(),
            )
        })
        .collect()
}

fn chunk_upload(object: ObjectId, origin: IVec3, level: u8, mesh: &Mesh) -> ChunkUpload<'_> {
    ChunkUpload {
        quads: &mesh.0,
        layers: mesh.1,
        origin,
        level,
        object,
    }
}

fn is_background(pixel: &[u8]) -> bool {
    pixel[..3] == [0, 0, 0]
}

/// Pixels that show the background in `image` although `reference` covers them and their
/// eight neighbours: holes inside the surface, not differences along a silhouette.
fn cracks(image: &RgbaImage, reference: &RgbaImage) -> Vec<(u32, u32)> {
    let width = reference.width;
    let covered: Vec<bool> = reference
        .pixels
        .chunks(4)
        .map(|pixel| !is_background(pixel))
        .collect();
    let at = |x: u32, y: u32| covered[(y * width + x) as usize];
    let mut found = Vec::new();
    for y in 1..reference.height - 1 {
        for x in 1..width - 1 {
            let start = ((y * width + x) * 4) as usize;
            if !is_background(&image.pixels[start..start + 4]) {
                continue;
            }
            if (y - 1..=y + 1).all(|ny| (x - 1..=x + 1).all(|nx| at(nx, ny))) {
                found.push((x, y));
            }
        }
    }
    found
}

#[test]
fn greedy_meshes_render_without_cracks_at_their_t_junctions() {
    let test = "greedy_meshes_render_without_cracks_at_their_t_junctions";
    let Some(gpu) = context_or_skip(test) else {
        return;
    };
    let mut renderer = renderer(&gpu, 4096);
    renderer.debug_view_mut().select(DebugView::Albedo).unwrap();
    let grid = renderer
        .scene_mut()
        .add_object(ObjectDesc {
            bounds: IAabb::new(IVec3::splat(-32), IVec3::splat(4 * 32 * SEAM_CHUNKS + 32)),
            pose: GridPose::IDENTITY,
        })
        .unwrap();
    // The resolution of the measurements that found the cracks.
    let target =
        HeadlessTarget::new(&gpu, 1920, 1080, wgpu::TextureFormat::Rgba8UnormSrgb).unwrap();
    let lighting = Lighting::default();
    let terrain: Vec<(IVec3, Mesh)> = (0..SEAM_CHUNKS * SEAM_CHUNKS)
        .map(|index| {
            let origin = IVec3::new(32 * (index % SEAM_CHUNKS), 0, 32 * (index / SEAM_CHUNKS));
            (origin, greedy_terrain_chunk(origin.x, origin.z))
        })
        .collect();
    // The same chunks at level 2: 4 m cells, chunk origins 128 voxels apart.
    let coarse: Vec<(IVec3, Mesh)> = terrain
        .iter()
        .map(|(origin, mesh)| (*origin * 4, mesh.clone()))
        .collect();
    let rows: Vec<(IVec3, Mesh)> = (0..64)
        .map(|index| {
            let origin = IVec3::new(32 * (index % 8), 0, 32 * (index / 8));
            (origin, greedy_rows_chunk())
        })
        .collect();
    for (name, level, chunks, cameras) in [
        ("terrain", 0, terrain, terrain_cameras(1.0)),
        ("terrain at level 2", 2, coarse, terrain_cameras(4.0)),
        ("rows", 0, rows, rows_cameras()),
    ] {
        let split: Vec<Mesh> = chunks.iter().map(|(_, mesh)| split_mesh(mesh)).collect();
        let greedy_quads: usize = chunks.iter().map(|(_, mesh)| mesh.0.len()).sum();
        let split_quads: usize = split.iter().map(|mesh| mesh.0.len()).sum();
        assert!(
            greedy_quads * 3 < split_quads * 2,
            "{name}: the greedy mesh hardly merges ({greedy_quads} of {split_quads} quads)"
        );
        let ids: Vec<ChunkId> = chunks
            .iter()
            .map(|(origin, mesh)| {
                renderer
                    .scene_mut()
                    .insert_chunk(&chunk_upload(grid, *origin, level, mesh))
                    .unwrap()
            })
            .collect();
        renderer.scene_mut().flush_all_uploads(&gpu);
        let render_all = |renderer: &mut VoxelRenderer| -> Vec<RgbaImage> {
            cameras
                .iter()
                .map(|camera| {
                    let input = FrameInput {
                        camera,
                        lighting: &lighting,
                        target: target.view(),
                        target_size: target.size(),
                        overlay: None,
                    };
                    renderer.render(&gpu, &input).unwrap();
                    target.capture_image(&gpu).unwrap()
                })
                .collect()
        };
        // The control: without the edge overlap the T-junctions crack.
        renderer.set_edge_overlap_px(0.0);
        let bare = render_all(&mut renderer);
        renderer.set_edge_overlap_px(EDGE_OVERLAP_PX);
        let greedy = render_all(&mut renderer);
        for (&id, ((origin, _), mesh)) in ids.iter().zip(chunks.iter().zip(&split)) {
            renderer
                .scene_mut()
                .replace_chunk(id, &chunk_upload(grid, *origin, level, mesh))
                .unwrap();
        }
        renderer.scene_mut().flush_all_uploads(&gpu);
        let reference = render_all(&mut renderer);
        let mut total = 0;
        let mut control = 0;
        let mut first = None;
        for (k, ((image, bare), reference)) in greedy.iter().zip(&bare).zip(&reference).enumerate()
        {
            let covered = reference
                .pixels
                .chunks(4)
                .filter(|pixel| !is_background(pixel))
                .count();
            assert!(
                covered * 10 > reference.pixels.len() / 4,
                "{name} camera {k}: only {covered} pixels show the surface"
            );
            let found = cracks(image, reference);
            total += found.len();
            if let Some(&pixel) = found.first() {
                first.get_or_insert((k, pixel));
            }
            control += cracks(bare, reference).len();
        }
        log::info!(
            "{test}: {name}: {greedy_quads} greedy and {split_quads} split quads, {} cameras at \
             1920x1080: {control} crack pixels without the edge overlap, {total} with it",
            cameras.len()
        );
        if control == 0 {
            log::warn!(
                "{test}: {name}: no cracks even without the edge overlap on this device, so the \
                 check below shows nothing here"
            );
        }
        assert_eq!(
            total, 0,
            "{name}: background shows through {total} pixels inside the greedy surface; first at \
             camera and pixel {first:?}"
        );
        for id in ids {
            renderer.scene_mut().remove_chunk(id).unwrap();
        }
    }
}
