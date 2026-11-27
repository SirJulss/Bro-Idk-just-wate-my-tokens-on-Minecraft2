// Temporal anti-aliasing (contract §7 "Post: TAA", §11 "TAA instead of MSAA"): every frame is
// rendered with a sub-pixel jitter of the projection, and this pass blends it into the history
// of the earlier frames, reprojected to where each surface was in the last frame. Static
// geometry needs no velocity buffer: the motion comes from the depth and the two cameras
// (contract §5.3). The history is clipped to the colour range of the current neighbourhood in
// YCoCg (variance clipping), so that what the history saw but this frame does not (an
// uncovered background, a moved edge) leaves no ghost. Colours are blended in a compressed
// space (c / (1 + luma)) so that a single bright sample does not dominate the average.
//
// Output: the new history (Rgba16Float, scene-linear), which the composite pass tone-maps.

#include "voxel/common.wgsl"
#include "common/fullscreen.wgsl"

// The parameters of the pass (cp_render::voxel::taa::GpuTaa, 144 bytes).
struct Taa {
    // Clip space without this frame's jitter → render space.
    clip_to_render: mat4x4<f32>,
    // Render space of this frame → clip space of the last frame, without jitter.
    render_to_prev_clip: mat4x4<f32>,
    // x: the least weight of the current frame; y: 1 when the history is valid; z: frames in
    // the history (the weight is at least 1 / (z + 1)); w: the least weight of a pixel that
    // stands still.
    params: vec4<f32>,
}

@group(0) @binding(0) var<uniform> frame: Frame;
@group(0) @binding(1) var<uniform> taa: Taa;
@group(0) @binding(2) var current_texture: texture_2d<f32>;
@group(0) @binding(3) var depth_texture: texture_depth_2d;
@group(0) @binding(4) var history_texture: texture_2d<f32>;
@group(0) @binding(5) var history_sampler: sampler;

@vertex
fn vs_fullscreen(@builtin(vertex_index) vertex: u32) -> FullscreenVertex {
    return FullscreenVertex(fullscreen_position(vertex));
}

fn luma(color: vec3<f32>) -> f32 {
    return dot(color, vec3<f32>(0.2126, 0.7152, 0.0722));
}

// Scene-linear colour → the compressed space of the blend, and back.
fn compress(color: vec3<f32>) -> vec3<f32> {
    return color / (1.0 + luma(color));
}

fn expand(color: vec3<f32>) -> vec3<f32> {
    return color / max(1.0 - luma(color), 1e-4);
}

fn to_ycocg(c: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(
        0.25 * c.r + 0.5 * c.g + 0.25 * c.b,
        0.5 * c.r - 0.5 * c.b,
        -0.25 * c.r + 0.5 * c.g - 0.25 * c.b,
    );
}

fn from_ycocg(c: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(c.x + c.y - c.z, c.x + c.z, c.x - c.y - c.z);
}

// The history at `uv` with a Catmull-Rom filter in five bilinear taps (Jimenez 2016): sharper
// than one bilinear tap, so that the image does not soften frame by frame while the camera
// moves.
fn sample_history(uv: vec2<f32>) -> vec3<f32> {
    let size = frame.viewport.xy;
    let position = uv * size;
    let center = floor(position - 0.5) + 0.5;
    let f = position - center;
    let f2 = f * f;
    let f3 = f2 * f;
    let w0 = -0.5 * f3 + f2 - 0.5 * f;
    let w1 = 1.5 * f3 - 2.5 * f2 + vec2<f32>(1.0);
    let w2 = -1.5 * f3 + 2.0 * f2 + 0.5 * f;
    let w3 = 0.5 * f3 - 0.5 * f2;
    let w12 = w1 + w2;
    let tc12 = (center + w2 / w12) / size;
    let tc0 = (center - 1.0) / size;
    let tc3 = (center + 2.0) / size;
    let a = textureSampleLevel(history_texture, history_sampler, vec2<f32>(tc12.x, tc0.y), 0.0).rgb;
    let b = textureSampleLevel(history_texture, history_sampler, vec2<f32>(tc0.x, tc12.y), 0.0).rgb;
    let c = textureSampleLevel(history_texture, history_sampler, tc12, 0.0).rgb;
    let d = textureSampleLevel(history_texture, history_sampler, vec2<f32>(tc3.x, tc12.y), 0.0).rgb;
    let e = textureSampleLevel(history_texture, history_sampler, vec2<f32>(tc12.x, tc3.y), 0.0).rgb;
    let wa = w12.x * w0.y;
    let wb = w0.x * w12.y;
    let wc = w12.x * w12.y;
    let wd = w3.x * w12.y;
    let we = w12.x * w3.y;
    let color = (a * wa + b * wb + c * wc + d * wd + e * we) / (wa + wb + wc + wd + we);
    return max(color, vec3<f32>(0.0));
}

// `value` moved along the line to the centre of the box [lo, hi] until it lies inside.
fn clip_to_box(value: vec3<f32>, lo: vec3<f32>, hi: vec3<f32>) -> vec3<f32> {
    let center = 0.5 * (lo + hi);
    let extent = 0.5 * (hi - lo) + vec3<f32>(1e-5);
    let offset = value - center;
    let units = abs(offset / extent);
    let largest = max(units.x, max(units.y, units.z));
    if largest > 1.0 {
        return center + offset / largest;
    }
    return value;
}

@fragment
fn fs_taa(v: FullscreenVertex) -> @location(0) vec4<f32> {
    let pixel = vec2<i32>(v.position.xy);
    let last = vec2<i32>(frame.viewport.xy) - vec2<i32>(1);
    // The neighbourhood of this frame: its mean and spread, and the nearest depth (reversed Z:
    // the largest), whose motion is the pixel's, so that moving edges keep their foreground.
    var sum = vec3<f32>(0.0);
    var squares = vec3<f32>(0.0);
    var center = vec3<f32>(0.0);
    var nearest = 0.0;
    var luma_min = 1e9;
    var luma_max = -1e9;
    for (var dy = -1; dy <= 1; dy += 1) {
        for (var dx = -1; dx <= 1; dx += 1) {
            let p = clamp(pixel + vec2<i32>(dx, dy), vec2<i32>(0), last);
            let c = to_ycocg(compress(textureLoad(current_texture, p, 0).rgb));
            sum += c;
            squares += c * c;
            luma_min = min(luma_min, c.x);
            luma_max = max(luma_max, c.x);
            if dx == 0 && dy == 0 {
                center = c;
            }
            nearest = max(nearest, textureLoad(depth_texture, p, 0));
        }
    }
    let mean = sum / 9.0;
    let sigma = sqrt(max(squares / 9.0 - mean * mean, vec3<f32>(0.0)));
    let current = expand(from_ycocg(center));
    if taa.params.y < 0.5 {
        return vec4<f32>(current, 1.0);
    }
    // Where the surface of this pixel was in the last frame. The sky (depth 0) is a direction:
    // a point very far away moves with the rotation of the camera only.
    let ndc = vec2<f32>(
        v.position.x * frame.viewport.z * 2.0 - 1.0,
        1.0 - v.position.y * frame.viewport.w * 2.0,
    );
    let point = taa.clip_to_render * vec4<f32>(ndc, max(nearest, 1e-7), 1.0);
    let prev = taa.render_to_prev_clip * vec4<f32>(point.xyz / point.w, 1.0);
    let prev_ndc = prev.xy / prev.w;
    let uv = prev_ndc * vec2<f32>(0.5, -0.5) + vec2<f32>(0.5);
    if prev.w <= 0.0 || any(uv < vec2<f32>(0.0)) || any(uv > vec2<f32>(1.0)) {
        return vec4<f32>(current, 1.0);
    }
    let history = to_ycocg(compress(sample_history(uv)));
    // Motion of the pixel since the last frame, in pixels.
    let motion = length((uv - (v.position.xy * frame.viewport.zw)) * frame.viewport.xy);
    // The box is wider where the pixel stands still or creeps (GAMMA_STILL sigmas: sub-pixel
    // detail such as distant terrace edges and gaps in crowns jitters inside it instead of
    // flipping the box from frame to frame) and tighter in fast motion (GAMMA_FAST: no ghosts).
    let gamma = mix(GAMMA_STILL, GAMMA_FAST, clamp((motion - 1.0) / 7.0, 0.0, 1.0));
    let clipped = clip_to_box(history, mean - gamma * sigma, mean + gamma * sigma);
    // The faster the pixel moves, the more of the current frame (less blur in motion). A pixel
    // that stands still averages its whole history up to two cycles of the jitter, so that the
    // per-frame patterns of the shadows and the obscurance average out completely.
    let least = mix(taa.params.w, taa.params.x, smoothstep(0.02, 0.25, motion));
    var weight = max(
        max(least, 1.0 / (taa.params.z + 1.0)),
        mix(taa.params.x, 0.25, clamp(motion / 16.0, 0.0, 1.0)),
    );
    // Anti-flicker (after Lottes and Karis): a current sample far from the history, measured
    // against the contrast of its neighbourhood, is more likely aliasing than change, so it
    // gets less weight.
    let flicker = abs(clipped.x - center.x) / max(luma_max - luma_min, 0.02);
    weight = max(weight / (1.0 + 4.0 * flicker), min(least, 0.05));
    let blended = mix(clipped, center, weight);
    return vec4<f32>(expand(from_ycocg(blended)), 1.0);
}

// Width of the clipping box in standard deviations of the neighbourhood: for a pixel that
// moves less than a pixel per frame, and from 8 pixels per frame on.
const GAMMA_STILL: f32 = 1.25;
const GAMMA_FAST: f32 = 0.9;
