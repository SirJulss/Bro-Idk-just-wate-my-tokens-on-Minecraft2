// One entry of the material table, cp_render::material::GpuMaterial (48 bytes). A test checks
// the offsets, the size and the array stride against the Rust struct. The binding is declared
// by the shader that reads the table:
//
//     @group(G) @binding(B) var<storage, read> materials: array<Material>;

struct Material {
    // Linear average albedo of the material's texture, 0..1 per channel.
    average_color: vec3<f32>,
    // Perceptual roughness, 0..1.
    roughness: f32,
    // Emission factor, >= 0.
    emissive: f32,
    // Metalness, 0..1.
    metallic: f32,
    // Layer of the material's texture in the block texture arrays.
    texture_layer: u32,
    // Tint class: 0 none, 1 grass, 2 foliage, 3 water.
    tint_class: u32,
    // Bits 0-1: how the texture may turn from cell to cell (TURN_*); bit 2: foliage; bit 3:
    // ground cover (COVER_LAYERS more layers follow the variants); bits 8-11: number of extra
    // texture layers (variants).
    variation: u32,
    // First extra texture layer.
    variant_layer: u32,
    reserved: vec2<u32>,
}

const TINT_NONE: u32 = 0u;
const TINT_GRASS: u32 = 1u;
const TINT_FOLIAGE: u32 = 2u;
const TINT_WATER: u32 = 3u;

const TURN_FIXED: u32 = 0u;
const TURN_FREE: u32 = 1u;
const TURN_AXIAL: u32 = 2u;
const TURN_UPRIGHT: u32 = 3u;
const VARIATION_FOLIAGE: u32 = 4u;
const VARIATION_GROUND_COVER: u32 = 8u;
const VARIANT_SHIFT: u32 = 8u;
// Ground cover layers behind the variants, in this order: flowers, blossoms, litter, straw.
const COVER_LAYERS: u32 = 4u;
const COVER_FLOWERS: u32 = 1u;
const COVER_BLOSSOMS: u32 = 2u;
const COVER_LITTER: u32 = 3u;
const COVER_STRAW: u32 = 4u;
