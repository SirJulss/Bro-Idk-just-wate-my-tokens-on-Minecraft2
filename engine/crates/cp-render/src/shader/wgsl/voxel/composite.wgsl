// The last full-screen pass of the voxel pipeline: tone mapping of the HDR target with AgX into
// the output, or one of the G-buffer debug views (cp_render::debug_view).
//
// The output is display-linear; an sRGB target encodes it, for a unorm target the shader
// applies the sRGB transfer function itself (frame.mode.z). Debug values that are meant as
// image bytes (normals, depth ramp, overdraw heat) are decoded first so that the stored bytes
// equal them.

#include "voxel/common.wgsl"
#include "common/color.wgsl"
#include "common/agx.wgsl"
#include "common/octahedral.wgsl"
#include "common/debug_view.wgsl"
#include "common/fullscreen.wgsl"

@group(0) @binding(0) var<uniform> frame: Frame;
@group(0) @binding(1) var hdr_texture: texture_2d<f32>;
@group(0) @binding(2) var albedo_texture: texture_2d<f32>;
@group(0) @binding(3) var normal_texture: texture_2d<f32>;
@group(0) @binding(4) var depth_texture: texture_depth_2d;
@group(0) @binding(5) var overdraw_texture: texture_2d<f32>;

@vertex
fn vs_fullscreen(@builtin(vertex_index) vertex: u32) -> FullscreenVertex {
    return FullscreenVertex(fullscreen_position(vertex));
}

// Heat ramp of a fragment count: 0 black, 1 blue, 2 cyan, 4 green, 8 yellow, 16 red, 32 white.
fn heat(count: f32) -> vec3<f32> {
    if count < 0.5 {
        return vec3<f32>(0.0);
    }
    var stops = array<vec3<f32>, 6>(
        vec3<f32>(0.10, 0.15, 0.85),
        vec3<f32>(0.10, 0.75, 0.90),
        vec3<f32>(0.15, 0.85, 0.20),
        vec3<f32>(0.95, 0.90, 0.15),
        vec3<f32>(0.95, 0.20, 0.10),
        vec3<f32>(1.00, 1.00, 1.00),
    );
    let position = clamp(log2(count), 0.0, 4.999);
    let index = u32(position);
    return mix(stops[index], stops[index + 1u], position - f32(index));
}

@fragment
fn fs_composite(v: FullscreenVertex) -> @location(0) vec4<f32> {
    let pixel = vec2<i32>(v.position.xy);
    var color = vec3<f32>(0.0);
    var encoded = false;
    switch frame.mode.x {
        case DEBUG_VIEW_ALBEDO: {
            color = textureLoad(albedo_texture, pixel, 0).rgb;
        }
        case DEBUG_VIEW_NORMAL: {
            let depth = textureLoad(depth_texture, pixel, 0);
            if depth > 0.0 {
                let normal = octahedral_decode(textureLoad(normal_texture, pixel, 0).xy);
                color = normal * 0.5 + vec3<f32>(0.5);
            }
            encoded = true;
        }
        case DEBUG_VIEW_DEPTH: {
            let depth = textureLoad(depth_texture, pixel, 0);
            if depth > 0.0 {
                // Distance on a log scale: white at the near plane, black at 2^17 m.
                let distance = frame.light.w / depth;
                color = vec3<f32>(1.0 - clamp(log2(distance) / 17.0, 0.0, 1.0));
            }
            encoded = true;
        }
        case DEBUG_VIEW_OVERDRAW: {
            color = heat(textureLoad(overdraw_texture, pixel, 0).r);
            encoded = true;
        }
        default: {
            // Off, LOD level, draw path and chunk bounds show the lit image (the G-buffer pass
            // replaced the albedo for the LOD and path views).
            color = agx_tonemap(textureLoad(hdr_texture, pixel, 0).rgb * frame.light.x);
        }
    }
    if encoded {
        color = srgb_to_linear(color);
    }
    if frame.mode.z == 0u {
        color = linear_to_srgb(clamp(color, vec3<f32>(0.0), vec3<f32>(1.0)));
    }
    return vec4<f32>(color, 1.0);
}
