// Data and shared functions of the GPU-driven voxel pipeline (cp_render::voxel, contract §5.3).
//
// The Rust mirrors of the structs and constants live in cp_render::voxel::layout; tests hold
// both sides against each other. The functions only take values, so that every pass can
// include this file whatever it binds. cp_render::voxel::cull repeats the culling functions on
// the CPU operation for operation (draw path B and the direct path).

// One drawable layer of one chunk (64 bytes).
struct DrawItem {
    // Min corner of the chunk in level-0 voxels of its grid.
    origin: vec3<i32>,
    // Bits 0-4 LOD level, 5-6 mesh layer, 8-13 wall mask in storage-slot order.
    lod_flags: u32,
    // Inclusive cell bounds: min x, y, z in bits 0-4, 5-9, 10-14; max x, y, z in 15-29.
    cell_bounds: u32,
    // Pool index of the item's first quad; the twelve range slots follow without gaps.
    quad_base: u32,
    // Per storage slot (+X +Y +Z -X -Y -Z): regular count in bits 0-15, wall count in 16-31.
    counts: array<u32, 6>,
    // Index into the object table.
    object: u32,
    // Dithered cross-fade: coverage in bits 0-7 (255 = every pixel), bit 8 = complement.
    fade: u32,
    reserved: array<u32, 2>,
}

// One grid that is drawn (224 bytes). Local space: grid axes, origin at the anchor voxel.
struct Object {
    local_to_clip: mat4x4<f32>,
    prev_local_to_clip: mat4x4<f32>,
    local_to_render: mat4x4<f32>,
    anchor: vec3<i32>,
    flags: u32,
    // Camera position in local space, metres.
    camera_local: vec3<f32>,
    reserved: u32,
}

// The per-frame uniform of every voxel pass (464 bytes).
struct Frame {
    render_to_clip: mat4x4<f32>,
    clip_to_render: mat4x4<f32>,
    // xyz direction to the sun, w cosine of its angular radius.
    sun_direction: vec4<f32>,
    sun_color: vec4<f32>,
    sky_zenith: vec4<f32>,
    sky_horizon: vec4<f32>,
    ground: vec4<f32>,
    up: vec4<f32>,
    // Per tint class: none, grass, foliage, water.
    tints: array<vec4<f32>, 4>,
    // Width, height, 1 / width, 1 / height.
    viewport: vec4<f32>,
    // Exposure, ambient intensity, fog density (1/m), near plane (m).
    light: vec4<f32>,
    // Item count, block count, blocks per segment row, draw slots per class.
    cull: vec4<u32>,
    // Scratch offset of the block sums, scratch offset of the segment totals, frame index,
    // zero.
    scratch: vec4<u32>,
    // Debug view, draw mode (0 A, 1 B, 2 direct), output is sRGB, zero.
    mode: vec4<u32>,
    // Block texture width and height in texels, focal length in pixels (pixels per metre
    // across the view ray at 1 m depth), edge overlap of the quads in pixels.
    raster: vec4<f32>,
    // rgb: sky at the horizon below the sun.
    sky_sunset: vec4<f32>,
    // Fog falloff with height (1/m), camera altitude above the fog's reference level (m),
    // start and end of the fade into the sky (m).
    atmosphere: vec4<f32>,
    // xyz: camera position in its frame modulo the cloud anchor period (m); w: 1 when the
    // frame is accumulated by temporal anti-aliasing.
    anchor: vec4<f32>,
    // The ground map: grid x and z (m) of its corner, 1 / extent (1/m), side in texels.
    ground_map: vec4<f32>,
    // rgb: the sky's light on a surface facing up, before the ambient intensity; w: the gain
    // of the sky's own brightness (sky, clouds, fog, mirrored sky) against the land's.
    sky_light: vec4<f32>,
}

const LOD_LEVELS: u32 = 21u;
const DRAW_CLASSES: u32 = 2u;
const SEGMENTS: u32 = 42u;
const SLOTS: u32 = 12u;
const LAYER_CUTOUT: u32 = 1u;
const TEXTURED_LEVELS: u32 = 2u;
const CULL_GROUP: u32 = 256u;
const COUNTER_QUADS: u32 = 2u;
const CULL_MARGIN: f32 = 1e-4;
const FADE_FULL: u32 = 255u;
const OVERLAP_STRETCH_MAX: f32 = 256.0;
const OVERLAP_SLIDE_MAX_PX: f32 = 0.75;

fn item_level(item: DrawItem) -> u32 {
    return item.lod_flags & 31u;
}

fn item_layer(item: DrawItem) -> u32 {
    return (item.lod_flags >> 5u) & 3u;
}

fn item_walls(item: DrawItem) -> u32 {
    return (item.lod_flags >> 8u) & 63u;
}

fn item_cell_size(item: DrawItem) -> i32 {
    return 1i << item_level(item);
}

// 1 for the cutout pipeline (cutout layer and every fading item), 0 for the opaque one.
fn item_class(item: DrawItem) -> u32 {
    let fading = (item.fade & 255u) != FADE_FULL;
    return select(0u, 1u, item_layer(item) == LAYER_CUTOUT || fading);
}

fn item_segment(item: DrawItem) -> u32 {
    return item_class(item) * LOD_LEVELS + item_level(item);
}

// Quads of range slot `slot` (0-5 regular, 6-11 walls) from the item's count words.
fn slot_quads(counts: array<u32, 6>, slot: u32) -> u32 {
    var words = counts;
    let word = words[slot % 6u];
    return select(word & 0xffffu, word >> 16u, slot >= 6u);
}

// An axis-aligned box in local space, metres.
struct LocalBounds {
    lo: vec3<f32>,
    hi: vec3<f32>,
}

// The lattice box of the item's cells relative to `anchor`. Exact while every coordinate stays
// below 2^24 m.
fn item_bounds(item: DrawItem, anchor: vec3<i32>) -> LocalBounds {
    let b = item.cell_bounds;
    let lo_cells = vec3<i32>(vec3<u32>(b & 31u, (b >> 5u) & 31u, (b >> 10u) & 31u));
    let hi_cells = vec3<i32>(vec3<u32>((b >> 15u) & 31u, (b >> 20u) & 31u, (b >> 25u) & 31u));
    let size = item_cell_size(item);
    let base = item.origin - anchor;
    return LocalBounds(
        vec3<f32>(base + lo_cells * size),
        vec3<f32>(base + (hi_cells + vec3<i32>(1i)) * size),
    );
}

// True when the box lies outside `plane` (inside: dot(plane.xyz, p) + plane.w >= 0) by more
// than the margin.
fn outside_plane(plane: vec4<f32>, center: vec3<f32>, extent: vec3<f32>) -> bool {
    let distance = dot(plane.xyz, center) + plane.w;
    let radius = dot(abs(plane.xyz), extent);
    let scale = abs(plane.w) + dot(abs(plane.xyz), abs(center)) + radius;
    return distance + radius < -CULL_MARGIN * scale;
}

// Conservative frustum test of a local box against the planes of `local_to_clip`: left,
// right, bottom, top and near (the far plane is at infinity).
fn bounds_in_frustum(m: mat4x4<f32>, bounds: LocalBounds) -> bool {
    let center = (bounds.lo + bounds.hi) * 0.5;
    let extent = (bounds.hi - bounds.lo) * 0.5;
    let r0 = vec4<f32>(m[0].x, m[1].x, m[2].x, m[3].x);
    let r1 = vec4<f32>(m[0].y, m[1].y, m[2].y, m[3].y);
    let r2 = vec4<f32>(m[0].z, m[1].z, m[2].z, m[3].z);
    let r3 = vec4<f32>(m[0].w, m[1].w, m[2].w, m[3].w);
    let outside = outside_plane(r3 + r0, center, extent) || outside_plane(r3 - r0, center, extent)
        || outside_plane(r3 + r1, center, extent) || outside_plane(r3 - r1, center, extent)
        || outside_plane(r3 - r2, center, extent);
    return !outside;
}

fn axis_component(v: vec3<f32>, axis: u32) -> f32 {
    return select(select(v.x, v.y, axis == 1u), v.z, axis == 2u);
}

// The unit vector of `axis` (0-2). Shaders build vectors from it instead of writing a
// component through a dynamic index, which FXC (DX12) cannot compile.
fn axis_unit(axis: u32) -> vec3<i32> {
    return vec3<i32>(select(0i, 1i, axis == 0u), select(0i, 1i, axis == 1u), select(0i, 1i, axis == 2u));
}

// Column `axis` (0-2) of `m`: the image of that axis's unit vector, `m * vec4(axis_unit, 0)`.
fn axis_column(m: mat4x4<f32>, axis: u32) -> vec4<f32> {
    return select(select(m[0], m[1], axis == 1u), m[2], axis == 2u);
}

// Whether a face of storage slot `face_slot` (0-5) in `bounds` can face the camera: positive
// faces lie at cell max planes (the smallest is lo + cell size), negative faces at cell min
// planes (the largest is hi - cell size).
fn faces_camera(face_slot: u32, bounds: LocalBounds, cell_size: f32, eye: vec3<f32>) -> bool {
    let axis = face_slot % 3u;
    let camera = axis_component(eye, axis);
    if face_slot < 3u {
        return camera > axis_component(bounds.lo, axis) + cell_size;
    }
    return camera < axis_component(bounds.hi, axis) - cell_size;
}

// x: slots that hold quads facing the camera, of an item in the frustum (and, for wall slots,
// with the face in the wall mask); y: slots that hold quads at all. Items of the cutout layer
// are drawn from both sides (foliage), so all their slots count as facing the camera.
fn slot_masks(item: DrawItem, object: Object) -> vec2<u32> {
    var nonempty = 0u;
    for (var slot = 0u; slot < SLOTS; slot += 1u) {
        if slot_quads(item.counts, slot) > 0u {
            nonempty |= 1u << slot;
        }
    }
    let bounds = item_bounds(item, object.anchor);
    if !bounds_in_frustum(object.local_to_clip, bounds) {
        return vec2<u32>(0u, nonempty);
    }
    let cell_size = f32(item_cell_size(item));
    let walls = item_walls(item);
    let two_sided = item_layer(item) == LAYER_CUTOUT;
    var visible = 0u;
    for (var slot = 0u; slot < SLOTS; slot += 1u) {
        let face_slot = slot % 6u;
        let enabled = slot < 6u || ((walls >> face_slot) & 1u) == 1u;
        let filled = ((nonempty >> slot) & 1u) == 1u;
        let facing = two_sided || faces_camera(face_slot, bounds, cell_size, object.camera_local);
        if filled && enabled && facing {
            visible |= 1u << slot;
        }
    }
    return vec2<u32>(visible, nonempty);
}

// Draws of an item: maximal runs of visible slots that no non-empty invisible slot interrupts
// (empty slots hold no quads and join their neighbours).
fn count_runs(visible: u32, nonempty: u32) -> u32 {
    var runs = 0u;
    var open = false;
    for (var slot = 0u; slot < SLOTS; slot += 1u) {
        if ((visible >> slot) & 1u) == 1u {
            if !open {
                runs += 1u;
                open = true;
            }
        } else if ((nonempty >> slot) & 1u) == 1u {
            open = false;
        }
    }
    return runs;
}

// Quads in the visible slots.
fn visible_quads(item: DrawItem, visible: u32) -> u32 {
    var quads = 0u;
    for (var slot = 0u; slot < SLOTS; slot += 1u) {
        if ((visible >> slot) & 1u) == 1u {
            quads += slot_quads(item.counts, slot);
        }
    }
    return quads;
}

// Debug colour of a LOD level (cycles after eight levels).
fn lod_color(level: u32) -> vec3<f32> {
    var palette = array<vec3<f32>, 8>(
        vec3<f32>(0.90, 0.25, 0.20),
        vec3<f32>(0.95, 0.60, 0.15),
        vec3<f32>(0.90, 0.85, 0.20),
        vec3<f32>(0.35, 0.80, 0.30),
        vec3<f32>(0.20, 0.75, 0.80),
        vec3<f32>(0.25, 0.40, 0.90),
        vec3<f32>(0.60, 0.35, 0.85),
        vec3<f32>(0.85, 0.40, 0.65),
    );
    return palette[level % 8u];
}

// Debug colour of a draw mode: path A green, path B blue, direct draws orange.
fn draw_mode_color(mode: u32) -> vec3<f32> {
    if mode == 0u {
        return vec3<f32>(0.25, 0.80, 0.35);
    }
    if mode == 1u {
        return vec3<f32>(0.25, 0.45, 0.90);
    }
    return vec3<f32>(0.95, 0.55, 0.15);
}
