// The G-buffer pass of the voxel pipeline (contract §5.3): indexed four-vertex quads pulled
// from the quad pool, one shader set for draw path A, path B and direct draws.
//
// Vertex buffer 0 is the bucket (draw item, first quad) of the draw, an instance-step
// Uint32x2 attribute addressed by the draw's first_instance. The shader never reads
// @builtin(instance_index) and never relies on a base vertex: vertex_index is the index from
// the static index buffer, 4 * quad + corner.
//
// Corners are placed on their lattice points exactly and then moved outward by a fraction of a
// pixel along the quad's edges (edge_push), so that the T-junctions of greedy meshes leave no
// cracks.
//
// Targets: albedo (Rgba8UnormSrgb, linear values in), normal and roughness (Rgb10a2Unorm,
// octahedral render-space normal in rg, roughness in b), material (Rgba8Unorm: metallic in r,
// emissive e stored as e / (1 + e) in g, foliage in b). Depth32Float, reversed Z, compare
// Greater.

#include "voxel/common.wgsl"
#include "common/material.wgsl"
#include "common/noise.wgsl"
#include "common/octahedral.wgsl"
#include "common/debug_view.wgsl"

@group(0) @binding(0) var<uniform> frame: Frame;
@group(0) @binding(1) var<storage, read> items: array<DrawItem>;
@group(0) @binding(2) var<storage, read> objects: array<Object>;
@group(0) @binding(3) var<storage, read> quads: array<vec2<u32>>;
@group(0) @binding(4) var<storage, read> materials: array<Material>;
@group(1) @binding(0) var albedo_array: texture_2d_array<f32>;
@group(1) @binding(1) var albedo_sampler: sampler;
// The ground map (cp_render::voxel::GroundMap): r dryness, g wood, b alpine, a weight.
@group(1) @binding(2) var ground_map: texture_2d<f32>;
@group(1) @binding(3) var ground_sampler: sampler;

struct VoxelVertex {
    @builtin(position) position: vec4<f32>,
    // Texture coordinates in tiles (one tile per metre): x image right, y image down.
    @location(0) tex: vec2<f32>,
    // Render-space normal of the face.
    @location(1) @interpolate(flat) normal: vec3<f32>,
    // Textured quads: the tint; untextured ones: the material's average colour times the tint.
    @location(2) @interpolate(flat) color: vec3<f32>,
    // Roughness, metallic, emissive.
    @location(3) @interpolate(flat) surface: vec3<f32>,
    // Texture layer, flags (bit 0 textured, bit 1 two-sided: foliage of the cutout layer),
    // packed fade, tint class (0 in the LOD and draw-path views).
    @location(4) @interpolate(flat) info: vec4<u32>,
    // Position in the grid in metres (exact at the corners), for the tint variation and the
    // cell of a fragment.
    @location(5) grid: vec3<f32>,
    // Face index, the material's variation bits, its first variant layer, the LOD level.
    @location(6) @interpolate(flat) extra: vec4<u32>,
    // Distance from the eye along the view direction (m): clip w of the perspective projection.
    @location(7) depth: f32,
}

// The sides of grass cells (tint class grass, x and z faces) show more of their fringe with
// distance: from GRASS_SIDE_NEAR on the texture squeezes towards its top rows, and from
// GRASS_SIDE_FAR on the whole face shows the top GRASS_SIDE_FRINGE of the tile, rows that are
// grass in every column of grass_side (cp-proctex paints at least two of 16). The one-metre steps
// of a hillside then read as grass from afar instead of drawing brown contour lines, while the
// soil of the steps near the eye stays. The far LOD levels draw these faces in the fringe's
// colour (cp-app's material table).
const GRASS_SIDE_NEAR: f32 = 12.0;
const GRASS_SIDE_FAR: f32 = 80.0;
const GRASS_SIDE_FRINGE: f32 = 0.125;

// Edge overlap, a recorded deviation from the exact lattice positions of contract §5.3.
// Greedy meshing leaves T-junctions everywhere: corners of one quad that lie on an edge of
// another, inside a chunk, across chunk borders (each chunk is meshed alone) and against LOD
// walls. After the f32 transform and the rasteriser's subpixel snapping such a corner sits a
// few thousandths of a pixel off that edge, and pixel centres in the sliver between the two
// belong to neither quad: the background shows through, and the holes flicker as the camera
// moves. Every corner therefore moves outward along both edges of its quad, far enough that
// each edge of the quad moves frame.raster.w pixels away from its old line on screen.
// Neighbouring quads overlap by that much instead, which the depth test hides (a common plane,
// or a face behind its neighbour), and silhouettes grow by the same fraction of a pixel.
//
// Returns the moves in metres along u and v of a corner at `clip`, where `u_dir` and `v_dir`
// are the clip-space images of the unit vectors of u and v. A move along u shifts the edges
// along v sideways by |s_u x s_v| / |s_v| pixels per metre (s: pixels per metre on screen),
// which is small for a quad seen at a grazing angle even where s_u is not, because all its
// directions then project onto nearly one screen line. Rates below 1 / OVERLAP_STRETCH_MAX of
// the rate across the view ray count as that much, which bounds a move to OVERLAP_STRETCH_MAX
// overlaps at the corner's depth (overlap * depth / focal length metres). On top of that a
// corner never slides more than OVERLAP_SLIDE_MAX_PX pixels along an edge on screen: the
// sides of terraces seen obliquely from far above would otherwise grow into flags and spikes.
fn edge_push(clip: vec4<f32>, u_dir: vec4<f32>, v_dir: vec4<f32>) -> vec2<f32> {
    // Depth in metres. Corners in front of the near plane are clipped away; they only need a
    // finite push.
    let depth = max(clip.w, frame.light.w);
    // Pixels per metre along u and v: the derivatives of the pixel position.
    let to_pixels = frame.viewport.xy * (0.5 / (depth * depth));
    let su = (u_dir.xy * depth - clip.xy * u_dir.w) * to_pixels;
    let sv = (v_dir.xy * depth - clip.xy * v_dir.w) * to_pixels;
    // Screen area per square metre of the quad's plane, over the length of the edge direction
    // that each move pushes sideways.
    let area = abs(su.x * sv.y - su.y * sv.x);
    let rates = area / max(vec2<f32>(length(sv), length(su)), vec2<f32>(1e-30));
    let least = max(frame.raster.z / (depth * OVERLAP_STRETCH_MAX), 1e-30);
    let push = frame.raster.w / max(rates, vec2<f32>(least));
    // The slide of the corner along u is |s_u| pixels per metre of the move along u.
    let slide = OVERLAP_SLIDE_MAX_PX / max(vec2<f32>(length(su), length(sv)), vec2<f32>(1e-30));
    return min(push, slide);
}

@vertex
fn vs_voxel(@builtin(vertex_index) vertex: u32, @location(0) bucket: vec2<u32>) -> VoxelVertex {
    let item = items[bucket.x];
    let object = objects[item.object];
    let quad = quads[bucket.y + (vertex >> 2u)];

    // Corner of the quad: vertex k is corner CORNER_MAP[negative][k], [0,1,2,3] or [0,3,2,1].
    let face = (quad.x >> 25u) & 7u;
    let axis = face >> 1u;
    let negative = (face & 1u) == 1u;
    let k = vertex & 3u;
    let corner = select(k, (4u - k) & 3u, negative);
    let width = ((quad.x >> 15u) & 31u) + 1u;
    let height = ((quad.x >> 20u) & 31u) + 1u;
    let du = select(0u, width, corner == 1u || corner == 2u);
    let dv = select(0u, height, corner >= 2u);
    let cell = vec3<i32>(vec3<u32>(quad.x & 31u, (quad.x >> 5u) & 31u, (quad.x >> 10u) & 31u));
    let normal_axis = axis_unit(axis);
    let u_axis = (axis + 1u) % 3u;
    let v_axis = (axis + 2u) % 3u;
    let plane = select(normal_axis, vec3<i32>(0i), negative);
    let lattice = cell + plane + axis_unit(u_axis) * i32(du) + axis_unit(v_axis) * i32(dv);

    // Position: integer offset from the anchor first, then the matrix with the fraction.
    let cell_size = item_cell_size(item);
    let local = (item.origin - object.anchor) + lattice * cell_size;
    let clip = object.local_to_clip * vec4<f32>(vec3<f32>(local), 1.0);
    // Then outward along both edges by the edge overlap, in clip space so that no precision
    // of the integer position is lost: (outward sign along u, along v) * metres.
    let u_clip = axis_column(object.local_to_clip, u_axis);
    let v_clip = axis_column(object.local_to_clip, v_axis);
    let outward = vec2<f32>(select(-1.0, 1.0, du != 0u), select(-1.0, 1.0, dv != 0u));
    let push = outward * edge_push(clip, u_clip, v_clip);
    var out: VoxelVertex;
    out.position = clip + u_clip * push.x + v_clip * push.y;
    out.depth = out.position.w;

    let normal = vec3<f32>(normal_axis) * select(1.0, -1.0, negative);
    out.normal = normalize((object.local_to_render * vec4<f32>(normal, 0.0)).xyz);

    // Texturing frame (a, b) = (u, v) on positive faces, (v, u) on negative faces, rotated by
    // uv_rot (cp_core::face::Face::uv_to_tex); image y runs down, so it is -t. The coordinates
    // are those of the moved corner, so the texture stays fixed to the lattice.
    let u_metres = f32(du) * f32(cell_size) + push.x;
    let v_metres = f32(dv) * f32(cell_size) + push.y;
    let a = select(u_metres, v_metres, negative);
    let b = select(v_metres, u_metres, negative);
    var st = vec2<f32>(a, b);
    switch (quad.y >> 16u) & 3u {
        case 1u: {
            st = vec2<f32>(b, -a);
        }
        case 2u: {
            st = vec2<f32>(-a, -b);
        }
        case 3u: {
            st = vec2<f32>(-b, a);
        }
        default: {}
    }
    out.tex = vec2<f32>(st.x, -st.y);

    let material = materials[quad.y & 0xffffu];
    var tint_class = material.tint_class & 3u;
    let tint = frame.tints[tint_class].rgb;
    let level = item_level(item);
    var textured = level < TEXTURED_LEVELS;
    var color = select(material.average_color * tint, tint, textured);
    var variation = material.variation;
    let view = frame.mode.x;
    if view == DEBUG_VIEW_LOD_LEVEL {
        color = lod_color(level);
        textured = false;
        tint_class = 0u;
        variation = 0u;
    } else if view == DEBUG_VIEW_DRAW_PATH {
        color = draw_mode_color(frame.mode.y);
        textured = false;
        tint_class = 0u;
        variation = 0u;
    }
    out.color = color;
    out.surface = vec3<f32>(material.roughness, material.metallic, material.emissive);
    // The cutout layer is drawn from both sides (see fs_cutout).
    let two_sided = item_layer(item) == LAYER_CUTOUT;
    let flags = select(0u, 1u, textured) | select(0u, 2u, two_sided);
    out.info = vec4<u32>(material.texture_layer, flags, item.fade, tint_class);
    // Grid coordinates of the corner, exact in integers (f32 holds them exactly up to 16,000 km).
    let grid = item.origin + lattice * cell_size;
    out.grid = vec3<f32>(grid);
    out.extra = vec4<u32>(face, variation, material.variant_layer, level);
    return out;
}

// How much of a biome tint a texel takes: 1 for near-neutral texels, 0 for coloured ones, by
// the saturation of its linear colour (the convention of cp_proctex::tint_weight). Tinted
// textures paint their tintable parts in near-neutral tones and the rest (the soil under the
// grass fringe of grass_side) in its own colour, which must not take the tint.
fn tint_weight(color: vec3<f32>) -> f32 {
    let high = max(max(color.r, color.g), color.b);
    let low = min(min(color.r, color.g), color.b);
    if high <= 0.0 {
        return 1.0;
    }
    return 1.0 - smoothstep(0.45, 0.6, (high - low) / high);
}

// The land under a point of the grid plane, from the ground map: dryness in -1..1 (lush to
// dry), wood and alpine in 0..1, and the map's weight (0 where it holds no data).
struct Land {
    dryness: f32,
    wood: f32,
    alpine: f32,
    weight: f32,
}

fn land_at(position: vec2<f32>) -> Land {
    let uv = (position - frame.ground_map.xy) * frame.ground_map.z;
    let texel = textureSampleLevel(ground_map, ground_sampler, uv, 0.0);
    return Land(texel.r * 2.0 - 1.0, texel.g, texel.b, texel.a);
}

// How dry the meadow at `position` looks, 0 (lush) to 1 (dry): without a ground map value
// noise alone (a regional octave of 700 m and two of 160 m and 40 m) moves it; with one, the
// region's dryness from the generator does, fields of 150 to 300 m with fairly sharp edges
// move it by a quarter either way (an eye-level view shows two or three of them), and a 40 m
// octave softens their edges.
fn meadow_dryness(position: vec2<f32>, land: Land) -> f32 {
    let fine = value_noise(position * (1.0 / 40.0) + vec2<f32>(31.7, 17.3));
    // Inside the ground map (weight 1) only its branch is needed, outside it (0) only the
    // noise's; both only along the map's border texels.
    var noise = 0.0;
    if land.weight < 1.0 {
        let region = value_noise(position * (1.0 / 700.0) + vec2<f32>(11.3, 47.9));
        let broad = value_noise(position * (1.0 / 160.0));
        noise = smoothstep(0.32, 0.68, 0.45 * region + 0.35 * broad + 0.2 * fine);
    }
    var mapped = 0.0;
    if land.weight > 0.0 {
        let field =
            smoothstep(0.3, 0.7, value_noise(position * (1.0 / 240.0) + vec2<f32>(5.1, 27.3)));
        mapped = clamp(
            0.5 + 0.55 * land.dryness + 0.5 * (field - 0.5) + 0.12 * (fine - 0.5),
            0.0,
            1.0,
        );
    }
    return mix(noise, mapped, land.weight);
}

// The colours of the meadows, as multipliers of the scene's grass tint: dryness is said by the
// hue and saturation alone (no brown patches on top), from a lush, cool green (about 103
// degrees of hue when lit) over a meadow green (88 degrees) to a light, pale straw green (72
// degrees, a third less saturated). The high pastures turn to a grey sage.
const GRASS_LUSH: vec3<f32> = vec3<f32>(0.80, 0.96, 1.10);
const GRASS_MEADOW: vec3<f32> = vec3<f32>(1.22, 1.08, 1.24);
const GRASS_DRY: vec3<f32> = vec3<f32>(1.85, 1.2, 1.7);
const GRASS_SAGE: vec3<f32> = vec3<f32>(1.3, 0.97, 2.0);

// The meadow colour of `dryness` (0..1): lush to meadow over the first half, meadow to dry
// over the second.
fn meadow_colour(dryness: f32) -> vec3<f32> {
    let greener = mix(GRASS_LUSH, GRASS_MEADOW, smoothstep(0.0, 0.5, dryness));
    return mix(greener, GRASS_DRY, smoothstep(0.5, 1.0, dryness));
}

// The tint of grass (class 1) and foliage (class 2) varies across the land like the biomes of
// a real landscape, so that meadows and forests are not one flat colour: by the meadow's
// dryness ([`meadow_colour`]), a grey sage near the snow line, a darker, bluer green on the
// forest floor, and brightness patches of 23 m and tufts of 6 m (8 percent, a few degrees of
// hue) on top. Foliage follows the hue of its meadow at a third of its strength and adds a
// field of its own at the size of a crown (7 m), so neighbouring trees differ. The tint classes
// are a global table (contract §5.3); the land comes from the ground map, the rest is the
// shader's, smooth at the scale of chunks and the same on every LOD level.
fn tint_variation(position: vec2<f32>, tint_class: u32, land: Land) -> vec3<f32> {
    let dryness = meadow_dryness(position, land);
    var meadow = meadow_colour(dryness);
    // Sage on the high pastures, a shaded, bluer green under the trees.
    meadow = mix(meadow, GRASS_SAGE, 0.8 * land.alpine * land.weight);
    let forest = smoothstep(0.3, 0.8, land.wood) * land.weight;
    if tint_class == TINT_FOLIAGE {
        let crown = value_noise(position * (1.0 / 7.0) + vec2<f32>(3.3, 91.1));
        let own = mix(vec3<f32>(0.86, 0.98, 1.1), vec3<f32>(1.14, 1.03, 0.82), crown);
        let hue = meadow / max(meadow.g, 1e-3);
        return mix(vec3<f32>(1.0), hue, 0.35) * own * (0.9 + 0.2 * crown);
    }
    meadow = mix(meadow, meadow * vec3<f32>(0.8, 0.86, 0.92), forest);
    let brightness = 0.88 + 0.24 * value_noise(position * (1.0 / 23.0) + vec2<f32>(7.1, 3.9));
    // Clumps of a few metres: darker, bluer tufts and lighter, yellower ones, so that the
    // meadow at one's feet is not an even carpet.
    let clump = value_noise(position * (1.0 / 6.0) + vec2<f32>(53.1, 9.7));
    let tuft = mix(vec3<f32>(0.88, 0.94, 1.03), vec3<f32>(1.08, 1.05, 0.9), clump);
    return meadow * brightness * tuft;
}

// The colour shift of untinted foliage (birch, spruce, cherry keep their own colour): a field
// at the size of a crown makes neighbouring trees of one species a little lighter or darker,
// warmer or cooler, so that a stand is not one colour repeated.
fn foliage_shift(position: vec2<f32>) -> vec3<f32> {
    let crown = value_noise(position * (1.0 / 7.0) + vec2<f32>(41.3, 8.9));
    let tone = value_noise(position * (1.0 / 11.0) + vec2<f32>(2.7, 63.1));
    let hue = mix(vec3<f32>(0.94, 0.99, 1.06), vec3<f32>(1.06, 1.01, 0.92), tone);
    return hue * (0.88 + 0.24 * crown);
}

// The ground cover of the grass top texel that contains `position` (grid x and z, m): 0 none,
// else COVER_*. Each texel compares a threshold, three fifths a 0.75 m noise and two fifths a
// hash of the texel, with the shares of the covers at the place, so that covers lie in ragged
// patches of a metre or less that cross the cells' borders, sharp to the texel: fallen leaves
// under the canopy (their share follows an 8 m noise); straw blades in the driest meadows (at
// most three tenths); and patches of 20 to 60 m of flowers on some meadows (about one in
// seven), yellow and red ones or violet and pink ones by a field of a few hundred metres.
// `span` is the texels per pixel: where texels are smaller than a pixel the hash fades out
// (a pixel would pick a different texel's answer from frame to frame) and the patches follow
// the noise alone. Nothing without a ground map.
fn ground_cover(position: vec2<f32>, land: Land, span: f32) -> u32 {
    if land.weight < 0.5 {
        return 0u;
    }
    let texel = floor(position * 16.0);
    let p = (texel + vec2<f32>(0.5)) * (1.0 / 16.0);
    let cell = vec2<i32>(texel);
    let hash = f32(cell_hash(vec3<i32>(cell.x, 0, cell.y), 7u) & 1023u) / 1023.0;
    let grain = mix(hash, 0.5, smoothstep(0.5, 1.0, span));
    let pick = 0.6 * value_noise(p * (1.0 / 0.75) + vec2<f32>(13.7, 3.1)) + 0.4 * grain;
    let litter = smoothstep(0.3, 0.75, land.wood)
        * (0.3 + 0.6 * value_noise(p * (1.0 / 8.0) + vec2<f32>(17.9, 4.2)));
    if pick < litter {
        return COVER_LITTER;
    }
    let dryness = meadow_dryness(p, land);
    let straw = smoothstep(0.6, 0.95, dryness)
        * (0.1 + 0.2 * value_noise(p * (1.0 / 30.0) + vec2<f32>(5.5, 61.3)));
    if pick < litter + straw {
        return COVER_STRAW;
    }
    let flowery = smoothstep(0.66, 0.8, value_noise(p * (1.0 / 40.0) + vec2<f32>(91.7, 33.1)));
    let open = (1.0 - smoothstep(0.15, 0.4, land.wood)) * (1.0 - land.alpine)
        * (1.0 - smoothstep(0.55, 0.8, dryness));
    if pick < litter + straw + 0.4 * flowery * open {
        let kind = value_noise(p * (1.0 / 300.0) + vec2<f32>(2.9, 7.7));
        return select(COVER_FLOWERS, COVER_BLOSSOMS, kind > 0.5);
    }
    return 0u;
}

// How open the place at `position` is to the sky, for surfaces other than leaves: 1 in the
// open, down to 1 - CANOPY_SHADE under the closed canopy of a forest (the ground map's wood
// density), so that the floor of a wood lies in the shade of its crowns while the sun still
// falls through their gaps. Written to the G-buffer for the lighting pass.
const CANOPY_SHADE: f32 = 0.3;

fn sky_openness(land: Land) -> f32 {
    return 1.0 - CANOPY_SHADE * smoothstep(0.5, 0.95, land.wood) * land.weight;
}

// A 32-bit hash of an integer cell and a face, for the per-cell choices of the texture.
fn cell_hash(cell: vec3<i32>, face: u32) -> u32 {
    var h = bitcast<u32>(cell.x) * 0x8da6b343u ^ bitcast<u32>(cell.y) * 0xd8163841u
        ^ bitcast<u32>(cell.z) * 0xcb1ab31fu ^ face * 0x165667b1u;
    h = (h ^ (h >> 16u)) * 0x7feb352du;
    h = (h ^ (h >> 15u)) * 0x846ca68bu;
    return h ^ (h >> 16u);
}

// The transform of a cell's texture as two columns of a 2x2 matrix acting on coordinates
// centred in the tile (x right, y down): quarter turns and mirror images, as many as `turn`
// allows, picked by `hash`.
fn tile_transform(turn: u32, hash: u32) -> mat2x2<f32> {
    var pick = 0u;
    switch turn {
        case 1u: {
            pick = hash & 7u;
        }
        case 2u: {
            // Identity, mirror x (1), a half turn (4) and mirror y (mirror x, then a half
            // turn: 5).
            var axial = array<u32, 4>(0u, 1u, 4u, 5u);
            pick = axial[hash & 3u];
        }
        case 3u: {
            pick = hash & 1u;
        }
        default: {}
    }
    // Bit 0 mirrors x first, bits 1-2 then turn by quarter turns.
    let mirror = select(1.0, -1.0, (pick & 1u) == 1u);
    var m = mat2x2<f32>(vec2<f32>(mirror, 0.0), vec2<f32>(0.0, 1.0));
    switch (pick >> 1u) & 3u {
        case 1u: {
            m = mat2x2<f32>(vec2<f32>(0.0, 1.0), vec2<f32>(-1.0, 0.0)) * m;
        }
        case 2u: {
            m = mat2x2<f32>(vec2<f32>(-1.0, 0.0), vec2<f32>(0.0, -1.0)) * m;
        }
        case 3u: {
            m = mat2x2<f32>(vec2<f32>(0.0, -1.0), vec2<f32>(1.0, 0.0)) * m;
        }
        default: {}
    }
    return m;
}

// The unit normal of a face index in grid space.
fn face_normal(face: u32) -> vec3<f32> {
    return vec3<f32>(axis_unit(face >> 1u)) * select(1.0, -1.0, (face & 1u) == 1u);
}

// Albedo (linear) and alpha of the fragment. Pixel-art sampling: texel centres with an
// analytic edge ramp one pixel wide, sampled with the gradients of the unsnapped coordinates.
// The layers may be rectangular: x snaps to the texel columns, y to the texel rows. The tint
// (white for untinted materials; varied for grass and foliage) is multiplied in by each
// texel's tint weight.
//
// Each 1 m cell of a textured surface turns its tile by a transform its material allows and
// shows one of the material's texture variants, both picked by a hash of the cell, so that a
// meadow is not one tile repeated in one orientation. Untextured far levels vary the
// brightness of their cells a little instead, so that far meadows do not look like vinyl.
// Untinted foliage shifts its colour from crown to crown ([`foliage_shift`]).
//
// Also returns the openness of the place to the sky ([`sky_openness`], 1 for foliage), from
// the same sample of the ground map.
struct Surface {
    color: vec4<f32>,
    openness: f32,
}

fn surface_color(v: VoxelVertex) -> Surface {
    // Derivatives first, in uniform control flow.
    let size = frame.raster.xy;
    let raw_dx = dpdx(v.tex);
    let raw_dy = dpdy(v.tex);
    let face = v.extra.x;
    let variation = v.extra.y;
    let inside = v.grid - face_normal(face) * 0.5;
    let land = land_at(v.grid.xz);
    var tint = v.color;
    if v.info.w == 1u || v.info.w == 2u {
        tint *= tint_variation(v.grid.xz, v.info.w, land);
    }
    var shift = vec3<f32>(1.0);
    if (variation & VARIATION_FOLIAGE) != 0u && v.info.w == TINT_NONE {
        shift = foliage_shift(v.grid.xz);
    }
    var color = vec4<f32>(tint, 1.0);
    if (v.info.y & 1u) != 0u {
        let hash = cell_hash(vec3<i32>(floor(inside)), face);
        let turn = tile_transform(variation & 3u, hash);
        let tile = floor(v.tex);
        var tex = tile + vec2<f32>(0.5) + turn * (v.tex - tile - vec2<f32>(0.5));
        var dx = turn * raw_dx;
        var dy = turn * raw_dy;
        if v.info.w == TINT_GRASS && (face >> 1u) != 1u {
            // Image y runs down from the top of the tile (grass_side turns by mirror images
            // only, which keep it).
            let far = smoothstep(GRASS_SIDE_NEAR, GRASS_SIDE_FAR, v.depth);
            let squeeze = mix(1.0, GRASS_SIDE_FRINGE, far);
            tex.y = tile.y + (tex.y - tile.y) * squeeze;
            dx.y *= squeeze;
            dy.y *= squeeze;
        }
        let texels = tex * size;
        let span = (abs(dx) + abs(dy)) * size;
        let footprint = min(span, vec2<f32>(1.0));
        let seam = floor(texels + 0.5);
        let ramp = clamp(
            (texels - seam) / max(footprint, vec2<f32>(1e-6)),
            vec2<f32>(-0.5),
            vec2<f32>(0.5),
        );
        let uv = (seam + ramp) / size;
        let variants = (variation >> VARIANT_SHIFT) & 15u;
        var layer = v.info.x;
        if variants > 0u {
            let pick = (hash >> 8u) % (variants + 1u);
            layer = select(layer, v.extra.z + pick - 1u, pick > 0u);
        }
        // Ground cover on the tops of covered ground (+Y faces), texel by texel: its layers
        // follow the variants.
        if (variation & VARIATION_GROUND_COVER) != 0u && face == 2u {
            let cover = ground_cover(v.grid.xz, land, max(span.x, span.y));
            if cover > 0u {
                layer = v.extra.z + variants + cover - 1u;
            }
        }
        let texel = textureSampleGrad(albedo_array, albedo_sampler, uv, layer, dx, dy);
        let weighted = mix(vec3<f32>(1.0), tint, tint_weight(texel.rgb));
        color = vec4<f32>(texel.rgb * weighted * shift, texel.a);
    } else if variation != 0u {
        // One brightness per cell of the level, within 5 percent.
        let cell_metres = f32(1u << v.extra.w);
        let cell = vec3<i32>(floor(inside / cell_metres));
        let shade = f32(cell_hash(cell, face) & 255u) / 255.0;
        color = vec4<f32>(tint * shift * (0.95 + 0.1 * shade), 1.0);
    }
    let openness = select(1.0, sky_openness(land), translucency(v) < 0.9);
    return Surface(color, openness);
}

// 4x4 ordered dither threshold in 0..16 at `pixel`. Rotated by the frame index while temporal
// anti-aliasing accumulates the frames (the pattern then averages out); fixed otherwise, so
// that a cross-fade does not crawl.
fn dither_threshold(pixel: vec2<f32>) -> u32 {
    var bayer = array<u32, 16>(0u, 8u, 2u, 10u, 12u, 4u, 14u, 6u, 3u, 11u, 1u, 9u, 15u, 7u, 13u, 5u);
    let shift = select(0u, frame.scratch.z, frame.anchor.w > 0.5);
    let p = (vec2<u32>(pixel) + vec2<u32>(shift, shift >> 2u)) & vec2<u32>(3u);
    return bayer[p.y * 4u + p.x];
}

// Whether a fading item keeps the pixel: the item keeps threshold t < coverage, its
// complement keeps t >= 1 - coverage, with t = (bayer + 0.5) / 16.
fn fade_keeps(fade: u32, pixel: vec2<f32>) -> bool {
    let coverage = f32(fade & 255u) / 255.0;
    if (fade & 255u) == FADE_FULL {
        return true;
    }
    let t = (f32(dither_threshold(pixel)) + 0.5) / 16.0;
    if (fade & 256u) != 0u {
        return t >= 1.0 - coverage;
    }
    return t < coverage;
}

struct GBuffer {
    @location(0) albedo: vec4<f32>,
    @location(1) normal: vec4<f32>,
    @location(2) material: vec4<f32>,
}

// The G-buffer texels of a fragment. `inside` marks a back face of foliage: the inside of a
// crown seen through a gap in its near side; its normal is turned towards the eye and the
// material's alpha (0) tells the lighting pass that the sun and most of the sky do not reach
// it. Every other fragment stores its openness to the sky there (`openness`, see
// [`sky_openness`]: 1 for leaves, at least 1 - CANOPY_SHADE otherwise).
fn gbuffer(v: VoxelVertex, albedo: vec3<f32>, inside: bool, openness: f32) -> GBuffer {
    var out: GBuffer;
    out.albedo = vec4<f32>(albedo, 1.0);
    let normal = select(v.normal, -v.normal, inside);
    out.normal = vec4<f32>(octahedral_encode(normal), clamp(v.surface.x, 0.0, 1.0), 1.0);
    let emissive = max(v.surface.z, 0.0);
    out.material = vec4<f32>(
        clamp(v.surface.y, 0.0, 1.0),
        emissive / (1.0 + emissive),
        translucency(v),
        select(openness, 0.0, inside),
    );
    return out;
}

// How much light passes through the surface, 0..1 (material.b of the G-buffer; the lighting
// pass wraps the sun's light around such surfaces and gives them no sky reflection): 1 for
// foliage (leaves, opaque leaf blocks of the far levels included, everything tinted like
// foliage), 0.6 for grass (its blades stand up, so a low sun still lights a meadow), 0 for
// the rest.
fn translucency(v: VoxelVertex) -> f32 {
    if (v.extra.y & VARIATION_FOLIAGE) != 0u || v.info.w == TINT_FOLIAGE {
        return 1.0;
    }
    return select(0.0, 0.6, v.info.w == TINT_GRASS);
}

@fragment
fn fs_opaque(v: VoxelVertex) -> GBuffer {
    let surface = surface_color(v);
    return gbuffer(v, surface.color.rgb, false, surface.openness);
}

// The cutout pipeline rasterises both sides of its quads (a recorded deviation from "back
// faces are culled", contract §2.3). Leaves of one crown share a cull group, so a crown is a
// single shell of faces (contract §5.2); seen from one side only, every gap in its leaves
// showed the sky through the whole crown. Drawn from both sides, a gap in the near side shows
// the far side's leaves from within, which the lighting pass shades as the dark inside of the
// crown. Back faces of everything else the pipeline draws (fading opaque items) are discarded.
@fragment
fn fs_cutout(v: VoxelVertex, @builtin(front_facing) front: bool) -> GBuffer {
    if !front && (v.info.y & 2u) == 0u {
        discard;
    }
    let surface = surface_color(v);
    if surface.color.a < 0.5 || !fade_keeps(v.info.z, v.position.xy) {
        discard;
    }
    return gbuffer(v, surface.color.rgb, !front, surface.openness);
}

// Overdraw debug view: every fragment adds one (additive blending, no depth test).
@fragment
fn fs_overdraw(v: VoxelVertex) -> @location(0) vec4<f32> {
    return vec4<f32>(1.0, 0.0, 0.0, 1.0);
}
