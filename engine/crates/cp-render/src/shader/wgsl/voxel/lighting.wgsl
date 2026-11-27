// The lighting pass of E2 (contract §5.3, §7): the sun as a directional light with screen-space
// shadows, hemispheric sky ambient with screen-space ambient obscurance, a sky reflection on
// glossy surfaces, height fog and a gradient sky with a layer of clouds, written to the HDR
// target (Rgba16Float, scene-linear Rec.709). Everything happens in camera-relative render
// space.

#include "voxel/common.wgsl"
#include "common/noise.wgsl"
#include "common/octahedral.wgsl"
#include "common/fullscreen.wgsl"

@group(0) @binding(0) var<uniform> frame: Frame;
@group(0) @binding(1) var albedo_texture: texture_2d<f32>;
@group(0) @binding(2) var normal_texture: texture_2d<f32>;
@group(0) @binding(3) var material_texture: texture_2d<f32>;
@group(0) @binding(4) var depth_texture: texture_depth_2d;

const PI: f32 = 3.14159265358979;

// Share of a roof's sky light that a vertical face receives.
const WALL_SKY: f32 = 0.8;
// Under a low sun a face that does not look up is lit by the band of sky at the horizon in the
// direction it faces, with this share of that band's colourfulness (see the ambient term).
const SIDE_SKY_SATURATION: f32 = 0.6;
// The inside of a crown, seen through a gap of the leaves in front of it: the share of the
// sun that comes through the leaves, and of the ambient light that is left.
const CANOPY_SUN: f32 = 0.12;
const CANOPY_AMBIENT: f32 = 0.7;
// Wood and ground seen through a gap in a crown (trunks and branches inside it,
// [`behind_leaves`]) keep this share of the ambient light of the inside of a crown, and this
// share of the sun: the leaves all around them shade the sky and the sun as well, and the bark
// of a birch is so light that it would glow inside its crown otherwise.
const ENCLOSED_AMBIENT: f32 = 0.35;
const ENCLOSED_SUN: f32 = 0.05;
// A pixel counts as seen through a gap in a crown when at least ENCLOSED_TAPS of the eight
// pixels ENCLOSED_RING away show leaves between ENCLOSED_NEAREST and ENCLOSED_FARTHEST metres
// nearer than it; only pixels nearer than ENCLOSED_RANGE metres are tested (farther crowns are
// a few pixels wide, and the coarse levels draw no wood inside them).
const ENCLOSED_TAPS: u32 = 5u;
const ENCLOSED_RING: i32 = 3;
const ENCLOSED_NEAREST: f32 = 0.2;
const ENCLOSED_FARTHEST: f32 = 4.0;
const ENCLOSED_RANGE: f32 = 120.0;
// Far grass steps (see STEP_NEAR) facing the sun receive at most this share of the sunlight of
// a tread at the same place, so that a hillside reads as one surface and not as a bundle of
// bright lines.
const STEP_RISER_MAX: f32 = 0.92;
// Far grass steps. The sides of grass cells (G-buffer translucency 0.6) are lit more and more
// like the slope they are part of: from STEP_NEAR to STEP_FAR metres their shading normal tilts
// towards up by up to STEP_TILT (a mix weight), and their shadow march starts up to STEP_LIFT
// metres higher, above the cell whose side they are (a march from the side itself runs into that
// cell when the sun is behind it). The one-metre steps of a hillside then stop drawing dark
// contour lines from afar; the shadows of hills and trees stay.
const STEP_NEAR: f32 = 15.0;
const STEP_FAR: f32 = 110.0;
const STEP_TILT: f32 = 0.85;
const STEP_LIFT: f32 = 1.1;

@vertex
fn vs_fullscreen(@builtin(vertex_index) vertex: u32) -> FullscreenVertex {
    return FullscreenVertex(fullscreen_position(vertex));
}

// Normalised device coordinates of a pixel position (origin top left, y down).
fn pixel_ndc(pixel: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(
        pixel.x * frame.viewport.z * 2.0 - 1.0,
        1.0 - pixel.y * frame.viewport.w * 2.0,
    );
}

// The render-space point at `ndc` and reversed-Z depth `depth` (> 0).
fn unproject(ndc: vec2<f32>, depth: f32) -> vec3<f32> {
    let p = frame.clip_to_render * vec4<f32>(ndc, depth, 1.0);
    return p.xyz / p.w;
}

fn luminance(color: vec3<f32>) -> f32 {
    return dot(color, vec3<f32>(0.2126, 0.7152, 0.0722));
}

// Whether the frame is accumulated by temporal anti-aliasing: per-pixel patterns may then
// change from frame to frame and average out.
fn temporal() -> bool {
    return frame.anchor.w > 0.5;
}

// A value in 0..1 that changes by 5.588 (a step that spreads 16 values evenly over the unit
// interval) from frame to frame while the frames are accumulated, 0 otherwise. It repeats
// with the 16 phases of the jitter, like everything else that changes from frame to frame.
fn frame_offset() -> f32 {
    return select(0.0, fract(f32(frame.scratch.z & 15u) * 5.588238), temporal());
}

// Height (sine of the elevation) of the horizon band: the sky turns from the horizon colour
// to the zenith colour over about this much.
const HORIZON_BAND: f32 = 0.16;

// The sky at the horizon in the compass direction of `direction`: the colour away from the
// sun turns into the colour below the sun as the direction turns towards the sun's azimuth.
fn horizon_color(direction: vec3<f32>) -> vec3<f32> {
    let up = frame.up.xyz;
    let sun = frame.sun_direction.xyz;
    let flat_view = direction - up * dot(direction, up);
    let flat_sun = sun - up * dot(sun, up);
    let lengths = length(flat_view) * length(flat_sun);
    let toward = select(0.0, dot(flat_view, flat_sun) / max(lengths, 1e-6), lengths > 1e-6);
    let weight = pow(0.5 + 0.5 * toward, 3.0);
    return mix(frame.sky_horizon.rgb, frame.sky_sunset.rgb, weight);
}

// Gradient sky in the unit render-space `direction`; `with_sun` adds the sun disc and glow.
// Rayleigh scattering's phase function brightens the sky towards the sun and opposite to it
// and darkens it at right angles; forward scattering by haze adds a broad glow around the sun
// near the horizon, which shows even when the disc is out of the picture.
fn sky(direction: vec3<f32>, with_sun: bool) -> vec3<f32> {
    let height = dot(direction, frame.up.xyz);
    let above = 1.0 - exp(-max(height, 0.0) / HORIZON_BAND);
    let mu = dot(direction, frame.sun_direction.xyz);
    let rayleigh = 0.85 + 0.3 * mu * mu;
    let horizon = horizon_color(direction);
    var color = mix(horizon, frame.sky_zenith.rgb * rayleigh, above);
    color += frame.sun_color.rgb * (0.06 * pow(max(mu, 0.0), 8.0) * (1.0 - 0.7 * above));
    let below = clamp(-height * 3.0, 0.0, 1.0);
    color = mix(color, frame.ground.rgb * 0.6 + horizon * 0.25, below);
    if with_sun {
        color += sun_light(direction);
    }
    return color;
}

// The gain of the sky's own brightness against the land's (Lighting::sky_gain): the sky, its
// clouds, the fog and the mirrored sky are scaled by it.
fn sky_gain() -> f32 {
    return frame.sky_light.w;
}

// How low the sun stands, 0 from about 23 degrees up to 1 at the horizon: the light of dusk.
fn low_sun() -> f32 {
    return 1.0 - smoothstep(0.05, 0.4, dot(frame.sun_direction.xyz, frame.up.xyz));
}

// The colour of the fog in the unit render-space `direction`: the sky just above the horizon
// in that compass direction (with the glow around the sun), a little bluer, also where the
// direction points below the horizon.
fn fog_color(direction: vec3<f32>) -> vec3<f32> {
    let up = frame.up.xyz;
    let height = dot(direction, up);
    let level = normalize(direction - up * min(height, 0.0) + up * 0.02);
    return sky(level, false) * vec3<f32>(0.94, 0.98, 1.04) * sky_gain();
}

// The sun disc and its glow in the unit render-space `direction`.
fn sun_light(direction: vec3<f32>) -> vec3<f32> {
    let cosine = max(dot(direction, frame.sun_direction.xyz), 0.0);
    let glow = pow(cosine, 512.0) * 0.5 + pow(cosine, 12.0) * 0.06;
    let edge = frame.sun_direction.w;
    let disc = smoothstep(edge - 2e-6, edge + 2e-6, cosine);
    return frame.sun_color.rgb * (glow + disc * 25.0);
}

// A layer of fair-weather clouds CLOUD_HEIGHT metres above the camera, in the sky behind the
// scene and in its reflections; the fog keeps the plain gradient, so distant hills fade into
// haze, not into clouds. The layer is anchored to the world through the camera's position in
// its frame (frame.anchor), so it stays put while the camera flies. Its density is five
// octaves of value noise on the layer, each octave fading out where a pixel spans more than
// its period (no shimmer at the horizon); the cover is that density above CLOUD_COVER. Sunlit
// tops scatter CLOUD_SUN of the sun's light towards the eye (a white that the tone map shows
// at about 240, the brightest thing in a daylight frame after the sun), and glow near the
// sun; thick cores are a light grey from below, so the clouds keep their volume; the layer
// fades into the haze with distance. A low sun lights the clouds through the forward lobe of
// their phase function: towards the sun they glow warm at their edges, away from it they keep
// a fraction of the sunlight (CLOUD_BACK) and turn into the violet grey of the dusk sky.
const CLOUD_HEIGHT: f32 = 1600.0;
const CLOUD_PERIOD: f32 = 2400.0;
const CLOUD_COVER: f32 = 0.54;
const CLOUD_HAZE: f32 = 40000.0;
const CLOUD_SUN: f32 = 0.9;
const CLOUD_GLOW: f32 = 0.4;
const CLOUD_BACK: f32 = 0.3;
const CLOUD_DUSK: f32 = 0.6;

// The clouds in the unit render-space `direction`: colour and opacity.
fn clouds(direction: vec3<f32>) -> vec4<f32> {
    let up = frame.up.xyz;
    let height = dot(direction, up);
    if height <= 0.002 {
        return vec4<f32>(0.0);
    }
    // Two axes across the layer: x and z when up is +y.
    let side = select(vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(0.0, 0.0, 1.0), abs(up.x) > 0.9);
    let across = normalize(cross(side, up));
    let along = cross(up, across);
    let distance = CLOUD_HEIGHT / height;
    let hit = direction * distance;
    let anchor = frame.anchor.xyz;
    var p = (vec2<f32>(dot(hit, along), dot(hit, across))
        + vec2<f32>(dot(anchor, along), dot(anchor, across))) / CLOUD_PERIOD;
    // Size of a pixel on the layer in noise units: the distance over the focal length,
    // stretched by the grazing angle.
    let footprint = distance / (frame.raster.z * height * CLOUD_PERIOD);
    var density = 0.0;
    var weight = 0.5;
    var frequency = 1.0;
    for (var i = 0u; i < 5u; i += 1u) {
        let sharp = clamp(1.5 - 2.0 * footprint * frequency, 0.0, 1.0);
        density += weight * mix(0.5, value_noise(p), sharp);
        // Rotate by 37 degrees and double, so the lattices of the octaves do not line up.
        p = mat2x2<f32>(1.6, 1.2, -1.2, 1.6) * p + vec2<f32>(5.3, 1.7);
        weight *= 0.5;
        frequency *= 2.0;
    }
    density /= 0.96875;
    let cover = smoothstep(CLOUD_COVER, CLOUD_COVER + 0.16, density);
    let core = smoothstep(CLOUD_COVER + 0.05, CLOUD_COVER + 0.35, density);
    let sun = frame.sun_direction.xyz;
    let mu = dot(direction, sun);
    let low = low_sun();
    let forward = 0.5 + 0.5 * mu;
    // At dusk the thin layer is lit through the long path of the low sun: dimmer than by
    // day (CLOUD_DUSK), so that its warm colour is not bleached in the tone map's shoulder.
    let phase = mix(1.0, (CLOUD_BACK + (1.0 - CLOUD_BACK) * forward * forward) * CLOUD_DUSK, low);
    let lit = frame.sun_color.rgb * (CLOUD_SUN * phase) + frame.sky_horizon.rgb * 0.5
        + frame.sky_zenith.rgb * 0.1;
    let dusk_core = mix(frame.sky_horizon.rgb, frame.sky_zenith.rgb, 0.5) * 0.7;
    let shaded = mix(frame.sky_horizon.rgb * 0.62 + frame.ground.rgb * 0.15, dusk_core, low);
    var color = mix(lit, shaded, core * mix(0.5, 0.75, low));
    color += frame.sun_color.rgb
        * (CLOUD_GLOW * pow(max(mu, 0.0), 6.0) * (1.0 - core));
    return vec4<f32>(color, cover * exp(-distance / CLOUD_HAZE));
}

// The sky in `direction` with its clouds, and the sun behind them when `with_sun` is set,
// scaled by the sky's gain.
fn cloudy_sky(direction: vec3<f32>, with_sun: bool) -> vec3<f32> {
    let cloud = clouds(direction);
    var color = mix(sky(direction, false), cloud.rgb, cloud.a);
    if with_sun {
        color += sun_light(direction) * (1.0 - cloud.a);
    }
    return color * sky_gain();
}

// Screen-space ambient obscurance (after McGuire, Mara and Luebke 2012, "Scalable Ambient
// Obscurance"), at full resolution and without a blur, until the GTAO of contract §7 exists:
// AO_SAMPLES depth taps on a spiral of AO_RADIUS metres around the pixel, rotated per pixel by
// interleaved gradient noise (and from frame to frame while temporal anti-aliasing averages
// the frames). Each tap obscures by the cosine between the surface normal and the direction
// to the tapped point (less a bias against self-occlusion on flat ground), falling to zero at
// the radius, so a tap on a distant object behind an edge adds nothing.
const AO_SAMPLES: u32 = 16u;
const AO_RADIUS: f32 = 1.1;
const AO_BIAS: f32 = 0.05;
const AO_INTENSITY: f32 = 3.2;
const AO_SPIRAL_TURNS: f32 = 7.0;
const AO_MAX_PIXELS: f32 = 96.0;

// Interleaved gradient noise (Jimenez 2014): a value in 0..1 per pixel that differs as much
// as possible from its neighbours' without a visible period (a 4 x 4 ordered pattern left a
// fixed grain in soft shadows that the accumulation did not average out), turned by
// frame_offset from frame to frame.
fn interleaved_gradient_noise(pixel: vec2<f32>) -> f32 {
    let p = floor(pixel);
    return fract(52.9829189 * fract(dot(p, vec2<f32>(0.06711056, 0.00583715))) + frame_offset());
}

// Share of the ambient light that reaches the point `position` with `normal` at `pixel`, whose
// reversed-Z depth is `depth`: 1 in the open, less in creases, corners and under overhangs.
fn ambient_obscurance(pixel: vec2<i32>, depth: f32, position: vec3<f32>, normal: vec3<f32>) -> f32 {
    // Pixels per metre at the point's depth: the focal length over the view depth.
    let view_depth = frame.light.w / depth;
    let radius_px = min(AO_RADIUS * frame.raster.z / view_depth, AO_MAX_PIXELS);
    if radius_px < 1.0 {
        return 1.0;
    }
    let size = vec2<i32>(frame.viewport.xy);
    let turn = interleaved_gradient_noise(vec2<f32>(pixel)) * 6.2831853;
    var sum = 0.0;
    for (var i = 0u; i < AO_SAMPLES; i += 1u) {
        let along = (f32(i) + 0.5) / f32(AO_SAMPLES);
        let angle = along * AO_SPIRAL_TURNS * 6.2831853 + turn;
        let offset = vec2<f32>(cos(angle), sin(angle)) * (along * radius_px);
        let tap = pixel + vec2<i32>(round(offset));
        if any(tap < vec2<i32>(0)) || any(tap >= size) || all(tap == pixel) {
            continue;
        }
        let tap_depth = textureLoad(depth_texture, tap, 0);
        if tap_depth <= 0.0 {
            continue;
        }
        let toward = unproject(pixel_ndc(vec2<f32>(tap) + vec2<f32>(0.5)), tap_depth) - position;
        let squared = dot(toward, toward);
        let falloff = max(1.0 - squared / (AO_RADIUS * AO_RADIUS), 0.0);
        sum += falloff * max(dot(toward, normal) * inverseSqrt(squared + 1e-4) - AO_BIAS, 0.0);
    }
    return clamp(1.0 - AO_INTENSITY * sum / f32(AO_SAMPLES), 0.0, 1.0);
}

// Screen-space sun shadows (the contact shadows of contract §7, until shadow maps exist): a
// march of SHADOW_STEPS steps from the point towards the sun, up to SHADOW_DISTANCE metres,
// the steps growing with the square of their index so the near ones are fine. A step that
// lands behind the depth buffer, by less than the thickness an occluder is assumed to have, is
// inside that occluder; the point is then in shadow. The thickness is SHADOW_THICKNESS (less
// than the one-metre cells near the camera, so a step that passes behind a block, rather than
// through it, does not count) growing with the view depth like the LOD cells do, and at least
// the depth the march moved since the last step, so no step jumps over a surface. Occluders
// outside the image cast no shadow. Where the frames are accumulated each pixel marches twice,
// the second march half a step behind the first, and the steps start at a different offset
// each frame: the staircase of the steps becomes a soft edge with twice the levels.
//
// A crown is solid for the sun: a step that lands behind leaves (G-buffer translucency 1) by
// less than SHADOW_FOLIAGE_THICKNESS is inside the crown, so the trunk and the branches inside a
// crown are not lit through the front shell of its leaves, which is thinner than the crown.
// Leaves still let SHADOW_FOLIAGE_LIGHT of the sun through (light scattered through the
// leaves and through the small gaps between them that no step finds), so the floor of a wood
// lies in a green shade instead of a black one.
const SHADOW_STEPS: u32 = 32u;
const SHADOW_DISTANCE: f32 = 28.0;
const SHADOW_THICKNESS: f32 = 0.6;
const SHADOW_THICKNESS_PER_METRE: f32 = 0.004;
const SHADOW_FOLIAGE_THICKNESS: f32 = 3.5;
const SHADOW_FOLIAGE_LIGHT: f32 = 0.2;

// 1 when the march from `start` towards the sun with the step offset `jitter` (-0.5..1) leaves
// the image or its range without entering an occluder, 0 when it enters a solid one and
// SHADOW_FOLIAGE_LIGHT when it enters a crown.
fn shadow_march(start: vec3<f32>, jitter: f32) -> f32 {
    let sun = frame.sun_direction.xyz;
    let size = frame.viewport.xy;
    // View depths in metres: clip w of the projection.
    var last_depth = (frame.render_to_clip * vec4<f32>(start, 1.0)).w;
    for (var i = 1u; i <= SHADOW_STEPS; i += 1u) {
        let along = (f32(i) + jitter) / f32(SHADOW_STEPS);
        let point = start + sun * (SHADOW_DISTANCE * along * along);
        let clip = frame.render_to_clip * vec4<f32>(point, 1.0);
        if clip.w <= frame.light.w {
            break;
        }
        let ndc = clip.xyz / clip.w;
        let screen = vec2<f32>(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5) * size;
        if any(screen < vec2<f32>(0.0)) || any(screen >= size) {
            break;
        }
        let texel = vec2<i32>(screen);
        let scene = textureLoad(depth_texture, texel, 0);
        // Reversed Z: a larger depth is nearer.
        if scene > ndc.z {
            let behind = clip.w - frame.light.w / scene;
            var thickness = max(
                SHADOW_THICKNESS + SHADOW_THICKNESS_PER_METRE * clip.w,
                abs(clip.w - last_depth),
            );
            let leaves = textureLoad(material_texture, texel, 0).z > 0.9;
            if leaves {
                thickness = max(thickness, SHADOW_FOLIAGE_THICKNESS);
            }
            if behind < thickness {
                return select(0.0, SHADOW_FOLIAGE_LIGHT, leaves);
            }
        }
        last_depth = clip.w;
    }
    return 1.0;
}

// The share of the sun that reaches the point `position` with `normal`: 1 in the sun, 0 in
// shadow, a half on the soft edge of two marches.
fn sun_visibility(position: vec3<f32>, normal: vec3<f32>, pixel: vec2<f32>) -> f32 {
    // Off the surface by a little more than the depth error of a far point.
    let start = position + normal * (0.02 + 0.001 * length(position));
    if !temporal() {
        return shadow_march(start, 0.0);
    }
    let jitter = interleaved_gradient_noise(pixel) - 0.5;
    return 0.5 * (shadow_march(start, jitter) + shadow_march(start, jitter + 0.5));
}

// Whether the point at `pixel`, `view_depth` metres away, is seen through a gap in a crown:
// at least ENCLOSED_TAPS of the eight pixels ENCLOSED_RING away show leaves a little nearer
// (trunks and branches inside a crown; a bare trunk under a crown has leaves above it only).
fn behind_leaves(pixel: vec2<i32>, view_depth: f32) -> bool {
    let last = vec2<i32>(frame.viewport.xy) - vec2<i32>(1);
    let r = ENCLOSED_RING;
    let d = (2 * r) / 3;
    var ring = array<vec2<i32>, 8>(
        vec2<i32>(r, 0), vec2<i32>(d, d), vec2<i32>(0, r), vec2<i32>(-d, d),
        vec2<i32>(-r, 0), vec2<i32>(-d, -d), vec2<i32>(0, -r), vec2<i32>(d, -d),
    );
    var leaves = 0u;
    for (var k = 0u; k < 8u; k += 1u) {
        let tap = clamp(pixel + ring[k], vec2<i32>(0), last);
        let tap_depth = textureLoad(depth_texture, tap, 0);
        if tap_depth <= 0.0 {
            continue;
        }
        let nearer = view_depth - frame.light.w / tap_depth;
        if nearer > ENCLOSED_NEAREST && nearer < ENCLOSED_FARTHEST
            && textureLoad(material_texture, tap, 0).z > 0.9 {
            leaves += 1u;
        }
    }
    return leaves >= ENCLOSED_TAPS;
}

// The share of the light that the fog takes away over `distance` metres to the render-space
// point `position`: exponential height fog (density frame.light.z at the reference level,
// thinning with frame.atmosphere.x per metre of height; the camera at frame.atmosphere.y),
// integrated along the ray, and at the end of the world the fade into the sky
// (frame.atmosphere.zw).
fn fog_amount(position: vec3<f32>, distance: f32) -> f32 {
    let density = frame.light.z;
    let falloff = frame.atmosphere.x;
    var optical = density * distance;
    if falloff > 0.0 {
        let base = exp(-falloff * frame.atmosphere.y);
        let rise = falloff * dot(position, frame.up.xyz);
        // The mean of e^(-falloff * height) along the ray, relative to the camera's height.
        var shape = 1.0 - 0.5 * rise;
        if abs(rise) > 1e-3 {
            shape = (1.0 - exp(-rise)) / rise;
        }
        optical *= base * shape;
    }
    var fog = 1.0 - exp(-optical);
    let fade = frame.atmosphere.zw;
    if fade.y > fade.x {
        fog = max(fog, smoothstep(fade.x, fade.y, distance));
    }
    return fog;
}

@fragment
fn fs_lighting(v: FullscreenVertex) -> @location(0) vec4<f32> {
    let pixel = vec2<i32>(v.position.xy);
    let depth = textureLoad(depth_texture, pixel, 0);
    let ndc = pixel_ndc(v.position.xy);
    if depth <= 0.0 {
        return vec4<f32>(cloudy_sky(normalize(unproject(ndc, 1.0)), true), 1.0);
    }
    let albedo = textureLoad(albedo_texture, pixel, 0).rgb;
    let packed = textureLoad(normal_texture, pixel, 0);
    let material = textureLoad(material_texture, pixel, 0);
    let normal = octahedral_decode(packed.xy);
    let roughness = packed.z;
    let metallic = material.x;
    let emissive = material.y / max(1.0 - material.y, 1e-3);
    // Leaves (1) and grass (0.6) let light through: their diffuse light wraps around them.
    let translucency = material.z;
    // Alpha 0 where the inside of a canopy shows through a gap in the leaves in front of it
    // (the G-buffer pass writes the back faces of foliage that way); otherwise the openness
    // of the place to the sky, 1 in the open and less under a closed canopy.
    let foliage_inside = material.w < 0.25;
    let openness = select(material.w, 1.0, foliage_inside);

    let position = unproject(ndc, depth);
    let distance = length(position);
    let to_eye = -position / max(distance, 1e-6);
    let up = frame.up.xyz;
    let sun = frame.sun_direction.xyz;
    // Wood and ground seen through a gap in a crown are inside it as well.
    let view_depth = frame.light.w / depth;
    let enclosed = !foliage_inside && translucency < 0.9 && view_depth < ENCLOSED_RANGE
        && behind_leaves(pixel, view_depth);
    let inside = select(0.0, 1.0, foliage_inside || enclosed);
    // Far grass steps (see STEP_NEAR): 0 near the eye, on tops and on everything but grass.
    let grass = abs(translucency - 0.6) < 0.1;
    let far_step = select(
        0.0,
        smoothstep(STEP_NEAR, STEP_FAR, distance) * (1.0 - abs(dot(normal, up))),
        grass,
    );
    let shading = normalize(mix(normal, up, STEP_TILT * far_step));
    let n_dot_l = dot(shading, sun);
    let wrap = 0.4 * translucency;
    var lit = clamp((n_dot_l + wrap) / (1.0 + wrap), 0.0, 1.0);
    // A far step facing the sun gets no more light than the tread above it.
    let tread = clamp((dot(up, sun) + wrap) / (1.0 + wrap), 0.0, 1.0);
    lit = mix(lit, min(lit, STEP_RISER_MAX * tread), smoothstep(0.0, 0.3, far_step));
    var shadow = 0.0;
    if lit > 0.0 && inside < 1.0 {
        shadow = sun_visibility(position + up * (STEP_LIFT * far_step), normal, v.position.xy);
    }
    // The sun reaches the inside of a crown only through the leaves, and the sky only in part;
    // wood inside a crown is shaded by the leaves around it as well.
    shadow = mix(shadow, select(CANOPY_SUN, ENCLOSED_SUN, enclosed), inside);
    let canopy = mix(1.0, CANOPY_AMBIENT, inside) * select(1.0, ENCLOSED_AMBIENT, enclosed)
        * openness;

    // Glossy surfaces (water, ice) mirror the sky, rough ones do not: the gloss falls from 1
    // at roughness 0 to none at 0.5, so leaves, grass, sand and bark reflect no sky (seen at a
    // grazing angle they would otherwise turn into grey-blue panes). Fresnel's rise towards
    // grazing angles is capped by the roughness, and nothing mirrors what lies below the
    // horizon or is obscured.
    let n_dot_v = max(dot(normal, to_eye), 1e-4);
    let gloss = select(
        pow(clamp((0.5 - roughness) / 0.5, 0.0, 1.0), 2.0),
        0.0,
        translucency > 0.0,
    );
    let f0 = mix(vec3<f32>(0.04), albedo, metallic);
    let f_max = max(vec3<f32>(1.0 - roughness), f0);
    let fresnel = f0 + (f_max - f0) * pow(1.0 - n_dot_v, 5.0);
    let reflected = reflect(-to_eye, normal);
    let upward = clamp(1.0 + 3.0 * dot(reflected, up), 0.0, 1.0);
    let mirror = fresnel * gloss;

    // Lambert plus normalised Blinn-Phong with Schlick's F0 for rough surfaces; on glossy
    // ones the sun is a small, sharp glint: the disc itself in the mirror direction.
    let halfway = normalize(sun + to_eye);
    let shininess = exp2(11.0 * (1.0 - roughness) + 1.0);
    let lobe = pow(max(dot(normal, halfway), 0.0), shininess) * (shininess + 8.0) / (8.0 * PI);
    let diffuse = albedo * (1.0 - metallic) / PI;
    let direct = frame.sun_color.rgb * shadow
        * (diffuse * lit + f0 * lobe * (1.0 - gloss) * max(n_dot_l, 0.0));
    let glint = sun_light(reflected) * mirror * (shadow * upward);

    // Ambient: the sky's light from above and the sun's bounce off the ground from below,
    // obscured in creases and corners and under a canopy. The sky's light (frame.sky_light) is
    // mostly its brightness by day: a blue fill would turn every shadow cyan; at dusk it keeps
    // more of the sky's colour, so that shade turns cool against the warm sun. A wall sees the
    // bright band of sky above the horizon as well as the zenith, so it keeps WALL_SKY of the
    // sky light of a roof (not the half a uniform sky would give it) and half the bounce; a
    // face turned down gets all the bounce. Terrace risers in the shade therefore stay at about
    // a third of the light of a sunlit top instead of turning into ink lines.
    //
    // Under a low sun the zenith darkens while the band of sky at the horizon stays bright: a
    // face that does not look up then sees mostly that band in the compass direction it faces
    // (the glow below the sun, or the rose of the earth's shadow on the other side), so its
    // sky light turns towards that colour (desaturated like the day's, SIDE_SKY_SATURATION),
    // and walls in the shade of a dusk scene stay readable instead of sinking into black.
    let facing_up = dot(shading, up);
    let sky_light = frame.sky_light.rgb;
    let obscurance = ambient_obscurance(pixel, depth, position, normal);
    let level = shading - up * facing_up;
    let side = length(level);
    var side_light = sky_light;
    let low = low_sun();
    if low > 0.0 && side > 1e-3 {
        let band = horizon_color(level / side);
        let band_light = mix(vec3<f32>(luminance(band)), band, SIDE_SKY_SATURATION);
        side_light = mix(sky_light, band_light, low);
    }
    let up_share = clamp(facing_up, 0.0, 1.0);
    let wall_share = WALL_SKY + (1.0 - WALL_SKY) * min(facing_up, 0.0);
    let sky_term = mix(side_light * wall_share, sky_light, up_share);
    let ambient = (sky_term + frame.ground.rgb * (0.5 - 0.5 * facing_up))
        * (frame.light.y * obscurance * canopy);
    let mirrored = cloudy_sky(reflected, false) * mirror * (upward * obscurance);
    let kept = vec3<f32>(1.0) - mirror;
    var color = direct + kept * albedo * (1.0 - 0.5 * metallic) * ambient + mirrored + glint
        + albedo * emissive;

    color = mix(color, fog_color(-to_eye), fog_amount(position, distance));
    return vec4<f32>(color, 1.0);
}
