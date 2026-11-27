// The fullscreen triangle of the screen-space passes: three vertices, no vertex buffer, the
// triangle (-1, -1), (3, -1), (-1, 3) in clip space covers the whole target.

struct FullscreenVertex {
    @builtin(position) position: vec4<f32>,
}

fn fullscreen_position(vertex: u32) -> vec4<f32> {
    let x = f32((vertex << 1u) & 2u) * 2.0 - 1.0;
    let y = f32(vertex & 2u) * 2.0 - 1.0;
    return vec4<f32>(x, y, 0.0, 1.0);
}
