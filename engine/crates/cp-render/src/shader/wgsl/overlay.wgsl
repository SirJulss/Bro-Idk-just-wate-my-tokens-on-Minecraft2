// Debug overlay of cp_render::overlay: solid rectangles and bitmap-font glyphs.
//
// One instance per rectangle, from vertex buffer 0 (step mode instance):
//   location 0  rect   target pixels (x0, y0, x1, y1), origin top left, y down
//   location 1  color  sRGB-encoded RGBA8 with straight alpha (Unorm8x4)
//   location 2  glyph  (glyph index into the font atlas or SOLID, integer pixel scale)
// Six vertices per instance make the two triangles of the rectangle.

#include "common/color.wgsl"

struct Screen {
    // Size of the target in pixels.
    size: vec2<f32>,
    // 1 when the target format is sRGB: colours are converted to linear before output, so
    // that the hardware encodes them back to the given values.
    srgb_target: u32,
    padding: u32,
}

@group(0) @binding(0) var<uniform> screen: Screen;
@group(0) @binding(1) var font: texture_2d<f32>;

// Keep in step with cp_render::overlay and cp_render::overlay::font.
const SOLID: u32 = 0xffffffffu;
const GLYPH_WIDTH: u32 = 5u;
const GLYPH_HEIGHT: u32 = 8u;
const ATLAS_COLUMNS: u32 = 16u;
const ATLAS_CELL: u32 = 8u;

struct Corner {
    @builtin(position) position: vec4<f32>,
    @location(0) @interpolate(flat) color: vec4<f32>,
    @location(1) @interpolate(flat) origin: vec2<f32>,
    @location(2) @interpolate(flat) glyph: vec2<u32>,
}

@vertex
fn vs_main(
    @builtin(vertex_index) vertex: u32,
    @location(0) rect: vec4<f32>,
    @location(1) color: vec4<f32>,
    @location(2) glyph: vec2<u32>,
) -> Corner {
    // Triangles (0, 0) (1, 0) (1, 1) and (0, 0) (1, 1) (0, 1) in units of the rectangle;
    // select keeps the corners exactly on the pixel edges.
    let right = vertex == 1u || vertex == 2u || vertex == 4u;
    let bottom = vertex == 2u || vertex == 4u || vertex == 5u;
    let pixel = vec2<f32>(select(rect.x, rect.z, right), select(rect.y, rect.w, bottom));
    let ndc = vec2<f32>(pixel.x / screen.size.x * 2.0 - 1.0, 1.0 - pixel.y / screen.size.y * 2.0);
    var corner: Corner;
    corner.position = vec4<f32>(ndc, 0.0, 1.0);
    corner.color = color;
    corner.origin = rect.xy;
    corner.glyph = glyph;
    return corner;
}

@fragment
fn fs_main(corner: Corner) -> @location(0) vec4<f32> {
    if corner.glyph.x != SOLID {
        let scale = f32(max(corner.glyph.y, 1u));
        // The fragment position is the pixel centre, so the glyph texel is exact.
        let texel = vec2<u32>(floor((corner.position.xy - corner.origin) / scale));
        if texel.x >= GLYPH_WIDTH || texel.y >= GLYPH_HEIGHT {
            discard;
        }
        let cell = vec2<u32>(corner.glyph.x % ATLAS_COLUMNS, corner.glyph.x / ATLAS_COLUMNS);
        if textureLoad(font, cell * ATLAS_CELL + texel, 0).r < 0.5 {
            discard;
        }
    }
    var rgb = corner.color.rgb;
    if screen.srgb_target != 0u {
        rgb = srgb_to_linear(rgb);
    }
    return vec4<f32>(rgb, corner.color.a);
}
