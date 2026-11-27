// Chunk-bounds debug view: the twelve edges of every draw item in the frustum as lines over the
// final image, coloured by LOD level. A plain debug draw: instance i is item i (first instance
// 0 on every backend), 24 vertices per instance, no depth test.

#include "voxel/common.wgsl"
#include "common/color.wgsl"

@group(0) @binding(0) var<uniform> frame: Frame;
@group(0) @binding(1) var<storage, read> items: array<DrawItem>;
@group(0) @binding(2) var<storage, read> objects: array<Object>;

struct BoundsVertex {
    @builtin(position) position: vec4<f32>,
    @location(0) @interpolate(flat) color: vec3<f32>,
}

@vertex
fn vs_bounds(
    @builtin(vertex_index) vertex: u32,
    @builtin(instance_index) instance: u32,
) -> BoundsVertex {
    let item = items[instance];
    let object = objects[item.object];
    let bounds = item_bounds(item, object.anchor);
    var out: BoundsVertex;
    out.color = lod_color(item_level(item));
    if !bounds_in_frustum(object.local_to_clip, bounds) {
        // Both ends outside the right clip plane: the line is clipped away.
        out.position = vec4<f32>(2.0, 2.0, 0.5, 1.0);
        return out;
    }
    // Edge e (12 per box) runs along axis e / 4; the other two corner bits come from e % 4.
    let edge = vertex >> 1u;
    let axis = edge >> 2u;
    let k = edge & 3u;
    let corner = vec3<f32>(
        axis_unit((axis + 1u) % 3u) * i32(k & 1u) + axis_unit((axis + 2u) % 3u) * i32((k >> 1u) & 1u)
            + axis_unit(axis) * i32(vertex & 1u),
    );
    out.position = object.local_to_clip * vec4<f32>(mix(bounds.lo, bounds.hi, corner), 1.0);
    return out;
}

@fragment
fn fs_bounds(v: BoundsVertex) -> @location(0) vec4<f32> {
    var color = v.color;
    if frame.mode.z == 1u {
        color = srgb_to_linear(color);
    }
    return vec4<f32>(color, 1.0);
}
