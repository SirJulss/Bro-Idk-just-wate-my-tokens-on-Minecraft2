//! The GPU data of the voxel pipeline and how it is packed (contract §5.3, "Data on the GPU").
//!
//! Every `Gpu*` struct here is the byte image of a WGSL struct in `voxel/common.wgsl`; the
//! tests at the end of this file hold the member offsets and sizes of both sides against each
//! other with `wgpu::naga`. The constants are shared with the shaders the same way (a test
//! reads them out of the WGSL source).

use bytemuck::{Pod, Zeroable};
use cp_core::coords::{CHUNK_SIZE, CHUNK_VOLUME, MAX_LOD_LEVEL};
use cp_core::face::Face;
use cp_core::quad::PackedQuad;

/// Quads per allocation unit of the quad pool; a chunk gets whole units (contract §5.3: size
/// classes of 256 quads).
pub const POOL_UNIT_QUADS: u32 = 256;

/// Bytes of one [`PackedQuad`].
pub const QUAD_BYTES: u64 = 8;

/// Mesh layers of a chunk in their storage order (contract §5.2): opaque, cutout, fluid.
pub const MESH_LAYERS: usize = 3;

/// Range slots of a layer: six regular face ranges, then six wall ranges, each group in
/// [`FACE_STORAGE_ORDER`] (contract §5.2).
pub const RANGES_PER_LAYER: usize = 12;

/// The faces in the storage order of a layer's ranges: `+X, +Y, +Z, -X, -Y, -Z`. This is not
/// the [`Face::index`] order: positive faces come first.
pub const FACE_STORAGE_ORDER: [Face; 6] = [
    Face::PosX,
    Face::PosY,
    Face::PosZ,
    Face::NegX,
    Face::NegY,
    Face::NegZ,
];

/// Position of `face` in [`FACE_STORAGE_ORDER`]: `axis + 3 · negative`, in `0..6`.
#[inline]
#[must_use]
pub const fn storage_slot(face: Face) -> usize {
    face.axis() + 3 * face.is_negative() as usize
}

/// Most quads one regular face range of a chunk can hold: one per cell (a chunk of
/// alternating materials that greedy meshing cannot merge).
pub const MAX_REGULAR_RANGE_QUADS: u32 = CHUNK_VOLUME as u32;

/// Most quads one wall range can hold: one per cell of a chunk boundary.
pub const MAX_WALL_RANGE_QUADS: u32 = (CHUNK_SIZE * CHUNK_SIZE) as u32;

/// Most quads of one layer, and therefore of one draw: 202,752. The static index buffer holds
/// [`QUAD_INDEX_PATTERN`](cp_core::quad::QUAD_INDEX_PATTERN) repeated this many times.
pub const MAX_ITEM_QUADS: u32 = 6 * (MAX_REGULAR_RANGE_QUADS + MAX_WALL_RANGE_QUADS);

/// Number of LOD levels, `0..=MAX_LOD_LEVEL`.
pub const LOD_LEVELS: u32 = MAX_LOD_LEVEL as u32 + 1;

/// Pipelines that draw voxel quads, each with its own region of the indirect buffers: class 0
/// is the opaque pipeline (opaque and fluid layers), class 1 the cutout pipeline (the cutout
/// layer and every item that is fading, because both need `discard`).
pub const DRAW_CLASSES: u32 = 2;

/// Segments of the compacted draw lists: one per `(class, level)`, ordered by class, then by
/// level with level 0 first (near to far).
pub const SEGMENTS: u32 = DRAW_CLASSES * LOD_LEVELS;

/// Most draws one item can produce: maximal runs of visible ranges that no non-empty invisible
/// range interrupts, and twelve slots hold at most six such runs.
pub const MAX_ITEM_DRAWS: u32 = 6;

/// Threads per workgroup of the cull passes; also the number of items per block of the prefix
/// sum.
pub const CULL_WORKGROUP: u32 = 256;

/// LOD levels below this one sample the block textures; coarser levels draw the material's
/// average colour (which also covers cutout states meshed into the opaque layer from level 2
/// on, contract §5.2).
pub const TEXTURED_LEVELS: u8 = 2;

/// Relative margin of the frustum test. A box is culled only when it lies outside a plane by
/// more than this fraction of the magnitudes in the test, so that the GPU (which may fuse
/// multiply-adds) and the CPU cull only boxes that are outside in exact arithmetic.
pub const CULL_MARGIN: f32 = 1.0e-4;

/// How far the vertex shader moves every quad corner outward along both edges of its quad, in
/// pixels on screen (a recorded deviation from the exact lattice positions of contract §5.3).
///
/// Greedy meshes have T-junctions: corners of one quad on an edge of another. The rasteriser
/// snaps vertices to its subpixel grid (1/256 px on the baseline GPU), which leaves such a
/// corner up to about 0.006 px off the edge, and pixel centres in the sliver between them
/// would show the background. With every quad grown by this much, neighbours overlap instead;
/// the depth test hides the overlap and silhouettes grow by the same 1/20 px.
pub const EDGE_OVERLAP_PX: f32 = 0.05;

/// Most the edge overlap is lengthened along an edge direction: a corner moves at most this
/// many overlaps at its depth (`overlap · depth / focal length` metres, about 1.4 % of the
/// depth at 1080p and 60°) along each edge of its quad.
///
/// Surfaces seen at grazing angles need the stretch: all their directions project onto nearly
/// one screen line, so a corner has to move far along one edge to move the other edge sideways
/// by the overlap. With it the overlap closes cracks down to about 0.03° between view ray and
/// surface at the baseline's snapping; the corners of such slivers slide along their edges by
/// up to this many overlaps on screen, inside a wedge at most two overlaps wide. The bound
/// keeps the corners of edge-on quads from travelling far along their plane.
pub const OVERLAP_STRETCH_MAX: f32 = 256.0;

/// Most a corner may slide along an edge of its quad on screen, in pixels, whatever the
/// stretch allows.
///
/// Where the two edge directions of a quad project onto nearly one screen line but the quad
/// is not edge-on (the side of a one-metre terrace seen obliquely from above at a distance),
/// the stretch moved corners by many pixels along the edges, and the faces grew into visible
/// flags and spikes above the terrain (beauty review E2, round 1). Moving a corner further
/// than this along its edge never closes a crack that a pixel centre can fall into, so the
/// bound costs nothing where cracks occur.
pub const OVERLAP_SLIDE_MAX_PX: f32 = 0.75;

/// Words of the counter buffer: draws of class 0 and class 1 (the count buffer of path A),
/// quads in visible ranges, and one reserved word.
pub const COUNTER_WORDS: usize = 4;

/// Index of the visible-quads counter in the counter buffer.
pub const COUNTER_QUADS: usize = 2;

/// One of the mesh layers of a chunk (contract §5.2). The discriminant is the storage order.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(u8)]
pub enum MeshLayer {
    /// Opaque cube faces; from level 2 on also the faces of cutout states.
    Opaque = 0,
    /// Alpha-tested faces of cutout states (levels 0 and 1).
    Cutout = 1,
    /// Fluid surfaces; drawn opaque until a translucent pass exists.
    Fluid = 2,
}

impl MeshLayer {
    /// All layers in storage order.
    pub const ALL: [Self; MESH_LAYERS] = [Self::Opaque, Self::Cutout, Self::Fluid];

    /// Storage index `0..3`.
    #[inline]
    #[must_use]
    pub const fn index(self) -> usize {
        self as usize
    }

    /// The layer with storage index `index`.
    #[inline]
    #[must_use]
    pub const fn from_index(index: usize) -> Option<Self> {
        match index {
            0 => Some(Self::Opaque),
            1 => Some(Self::Cutout),
            2 => Some(Self::Fluid),
            _ => None,
        }
    }
}

/// The faces of a chunk whose wall ranges are drawn (contract §5.3): the faces towards a
/// neighbour of another LOD level, or between an edited and an unedited chunk of the same
/// level. Bit `k` is face `FACE_STORAGE_ORDER[k]`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct WallMask(u8);

impl WallMask {
    /// No wall range is drawn.
    pub const NONE: Self = Self(0);
    /// Every wall range is drawn.
    pub const ALL: Self = Self(0x3F);

    /// The mask with the low six `bits` (storage-slot order).
    #[inline]
    #[must_use]
    pub const fn from_bits(bits: u8) -> Self {
        Self(bits & 0x3F)
    }

    /// The mask of `faces`.
    #[must_use]
    pub fn from_faces(faces: &[Face]) -> Self {
        faces.iter().fold(Self::NONE, |mask, &face| mask.with(face))
    }

    /// The bits in storage-slot order.
    #[inline]
    #[must_use]
    pub const fn bits(self) -> u8 {
        self.0
    }

    /// The same mask with `face` added.
    #[inline]
    #[must_use]
    pub const fn with(self, face: Face) -> Self {
        Self(self.0 | 1 << storage_slot(face))
    }

    /// Whether the wall range of `face` is drawn.
    #[inline]
    #[must_use]
    pub const fn contains(self, face: Face) -> bool {
        self.0 & 1 << storage_slot(face) != 0
    }

    /// True when no wall range is drawn.
    #[inline]
    #[must_use]
    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }
}

/// The dithered cross-fade of an item (contract §6 `cp-stream`): a chunk that appears covers
/// `coverage` of its pixels with an ordered-dither pattern that rotates every frame; the chunk
/// it replaces draws the complement pattern with the remaining coverage, so the two together
/// cover every pixel exactly once.
///
/// Fading items are drawn by the cutout pipeline (draw class 1), which has the `discard` the
/// dither needs; fully covered items keep the early depth test of the opaque pipeline.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Fade {
    coverage: u8,
    complement: bool,
}

impl Default for Fade {
    fn default() -> Self {
        Self::FULL
    }
}

impl Fade {
    /// Every pixel is drawn (no fade).
    pub const FULL: Self = Self {
        coverage: 255,
        complement: false,
    };

    /// A fade that draws `coverage` (`0..=1`, quantised to 1/255; NaN counts as 0) of the
    /// pixels, with the complement pattern when `complement` is set.
    #[must_use]
    pub fn new(coverage: f32, complement: bool) -> Self {
        let scaled = if coverage.is_nan() {
            0.0
        } else {
            (coverage * 255.0).round()
        };
        let coverage = if scaled <= 0.0 {
            0
        } else if scaled >= 255.0 {
            255
        } else {
            // In 1..255 here.
            scaled as u8
        };
        Self {
            coverage,
            complement,
        }
    }

    /// Coverage in 1/255 units; 255 draws every pixel.
    #[inline]
    #[must_use]
    pub const fn coverage_bits(self) -> u8 {
        self.coverage
    }

    /// Whether the item draws the complement pattern (the chunk that is being replaced).
    #[inline]
    #[must_use]
    pub const fn complement(self) -> bool {
        self.complement
    }

    /// True when every pixel is drawn.
    #[inline]
    #[must_use]
    pub const fn is_full(self) -> bool {
        self.coverage == 255
    }

    /// The `fade` word of a [`GpuDrawItem`]: coverage in bits 0–7, complement in bit 8.
    #[inline]
    #[must_use]
    pub const fn pack(self) -> u32 {
        self.coverage as u32 | (self.complement as u32) << 8
    }
}

/// An axis-aligned box of chunk cells, both corners inclusive, every coordinate in `0..32`,
/// `min <= max` on every axis. The faces of its cells lie in the lattice box `[min, max + 1]`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct CellBounds {
    /// Smallest cell coordinate per axis.
    pub min: [u8; 3],
    /// Largest cell coordinate per axis.
    pub max: [u8; 3],
}

impl CellBounds {
    /// The whole chunk.
    pub const CHUNK: Self = Self {
        min: [0; 3],
        max: [31; 3],
    };

    /// The box of the single cell `cell`.
    #[inline]
    #[must_use]
    pub const fn cell(cell: [u8; 3]) -> Self {
        Self {
            min: cell,
            max: cell,
        }
    }

    /// The cells whose face `quad` covers. `quad` must be valid
    /// ([`PackedQuad::validate`]).
    #[must_use]
    pub fn of_quad(quad: PackedQuad) -> Self {
        let face = quad.face();
        let min = quad.cell();
        let mut max = min;
        max[face.u_axis()] += quad.width() - 1;
        max[face.v_axis()] += quad.height() - 1;
        Self { min, max }
    }

    /// The smallest box containing both.
    #[must_use]
    pub fn union(self, other: Self) -> Self {
        Self {
            min: [0, 1, 2].map(|axis| self.min[axis].min(other.min[axis])),
            max: [0, 1, 2].map(|axis| self.max[axis].max(other.max[axis])),
        }
    }

    /// The `cell_bounds` word of a [`GpuDrawItem`]: min x, y, z in bits 0–4, 5–9, 10–14, max
    /// in bits 15–19, 20–24, 25–29.
    #[inline]
    #[must_use]
    pub const fn pack(self) -> u32 {
        let [x0, y0, z0] = self.min;
        let [x1, y1, z1] = self.max;
        (x0 as u32 & 31)
            | (y0 as u32 & 31) << 5
            | (z0 as u32 & 31) << 10
            | (x1 as u32 & 31) << 15
            | (y1 as u32 & 31) << 20
            | (z1 as u32 & 31) << 25
    }

    /// The box of a packed `cell_bounds` word.
    #[inline]
    #[must_use]
    pub const fn unpack(bits: u32) -> Self {
        const fn field(bits: u32, shift: u32) -> u8 {
            ((bits >> shift) & 31) as u8
        }
        Self {
            min: [field(bits, 0), field(bits, 5), field(bits, 10)],
            max: [field(bits, 15), field(bits, 20), field(bits, 25)],
        }
    }
}

/// One drawable layer of one chunk: the 64-byte draw item of contract §5.3, WGSL `DrawItem`.
///
/// `origin` is the chunk's min corner in level-0 voxels of its grid. `lod_flags` holds the LOD
/// level in bits 0–4, the [`MeshLayer`] in bits 5–6 and the [`WallMask`] in bits 8–13.
/// `cell_bounds` is a packed [`CellBounds`] of every quad the item can draw. The item's quads
/// start at pool index `quad_base` and follow the twelve range slots without gaps; `counts[k]`
/// holds the regular count of storage slot `k` in bits 0–15 and its wall count in bits
/// 16–31. `object` indexes the object table, `fade` is a packed [`Fade`].
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Pod, Zeroable)]
pub struct GpuDrawItem {
    /// Min corner of the chunk, level-0 voxels, grid-local.
    pub origin: [i32; 3],
    /// Level, layer and wall mask.
    pub lod_flags: u32,
    /// Packed inclusive cell bounds.
    pub cell_bounds: u32,
    /// Pool index of the item's first quad.
    pub quad_base: u32,
    /// Regular and wall counts per storage slot.
    pub counts: [u32; 6],
    /// Object index.
    pub object: u32,
    /// Packed fade.
    pub fade: u32,
    /// Zero; pads the item to 64 bytes.
    pub reserved: [u32; 2],
}

const _: () = assert!(size_of::<GpuDrawItem>() == 64);

impl GpuDrawItem {
    /// The `lod_flags` word for `level`, `layer` and `walls`.
    #[inline]
    #[must_use]
    pub const fn pack_lod_flags(level: u8, layer: MeshLayer, walls: WallMask) -> u32 {
        (level as u32 & 31) | (layer as u32) << 5 | (walls.bits() as u32) << 8
    }

    /// LOD level, `0..=20`.
    #[inline]
    #[must_use]
    pub const fn level(&self) -> u8 {
        (self.lod_flags & 31) as u8
    }

    /// Mesh layer index (`0..3`).
    #[inline]
    #[must_use]
    pub const fn layer_index(&self) -> u32 {
        (self.lod_flags >> 5) & 3
    }

    /// The wall mask.
    #[inline]
    #[must_use]
    pub const fn walls(&self) -> WallMask {
        WallMask::from_bits(((self.lod_flags >> 8) & 0x3F) as u8)
    }

    /// Cell edge in metres: `2^level`.
    #[inline]
    #[must_use]
    pub const fn cell_size(&self) -> i32 {
        1 << self.level()
    }

    /// The draw class: 1 (cutout pipeline) for the cutout layer and for fading items, else 0.
    #[inline]
    #[must_use]
    pub const fn draw_class(&self) -> u32 {
        let faded = self.fade & 255 != 255;
        if self.layer_index() == MeshLayer::Cutout as u32 || faded {
            1
        } else {
            0
        }
    }

    /// The segment of the compacted draw lists: `class * LOD_LEVELS + level`.
    #[inline]
    #[must_use]
    pub const fn segment(&self) -> u32 {
        self.draw_class() * LOD_LEVELS + self.level() as u32
    }

    /// Quads in range slot `slot` (`0..6` regular, `6..12` walls, storage order).
    ///
    /// # Panics
    /// If `slot >= 12`.
    #[inline]
    #[must_use]
    pub const fn slot_count(&self, slot: usize) -> u32 {
        assert!(slot < RANGES_PER_LAYER, "range slot out of range");
        let word = self.counts[slot % 6];
        if slot < 6 { word & 0xFFFF } else { word >> 16 }
    }

    /// Quads of the item over all twelve slots.
    #[must_use]
    pub fn quad_count(&self) -> u32 {
        (0..RANGES_PER_LAYER)
            .map(|slot| self.slot_count(slot))
            .sum()
    }
}

/// One grid that is drawn (contract §5.3 "Objects"), WGSL `Object`, 224 bytes.
///
/// "Local" space is the grid's own axes with the origin at the integer `anchor` voxel: a
/// vertex's local position is `origin - anchor + lattice · cell_size`, computed in `i32` and
/// exact in `f32`. The matrices are composed in `f64` on the CPU and are column-major
/// (`[column][row]`): `local_to_clip` carries the fractional camera offset, `local_to_render`
/// maps to camera-relative render space, `prev_local_to_clip` is last frame's `local_to_clip`
/// for the same anchor (for velocity later). `camera_local` is the camera position in local
/// space, for the face test.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
pub struct GpuObject {
    /// Local → clip space.
    pub local_to_clip: [[f32; 4]; 4],
    /// Last frame's local → clip space, same anchor.
    pub prev_local_to_clip: [[f32; 4]; 4],
    /// Local → render space (camera-relative, axes of the camera's reference frame).
    pub local_to_render: [[f32; 4]; 4],
    /// Anchor voxel, grid-local.
    pub anchor: [i32; 3],
    /// Zero; reserved for per-object flags (moving objects that write velocity).
    pub flags: u32,
    /// Camera position in local space, metres.
    pub camera_local: [f32; 3],
    /// Zero.
    pub reserved: u32,
}

const _: () = assert!(size_of::<GpuObject>() == 224);

/// The per-frame uniform of every voxel pass, WGSL `Frame`.
///
/// `render_to_clip` is the projection times the view rotation (the camera sits at the origin
/// of render space), `clip_to_render` its inverse. Directions are unit vectors in render
/// space, colours linear Rec.709.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
pub struct GpuFrame {
    /// Render → clip space.
    pub render_to_clip: [[f32; 4]; 4],
    /// Clip → render space.
    pub clip_to_render: [[f32; 4]; 4],
    /// xyz: direction to the sun; w: cosine of the sun's angular radius.
    pub sun_direction: [f32; 4],
    /// rgb: sun illuminance colour; w: zero.
    pub sun_color: [f32; 4],
    /// rgb: sky colour at the zenith.
    pub sky_zenith: [f32; 4],
    /// rgb: sky colour at the horizon (also the fog colour).
    pub sky_horizon: [f32; 4],
    /// rgb: colour of the ground hemisphere for ambient light.
    pub ground: [f32; 4],
    /// xyz: the local up direction (away from gravity).
    pub up: [f32; 4],
    /// rgb per tint class: none, grass, foliage, water.
    pub tints: [[f32; 4]; 4],
    /// Width, height, 1 / width, 1 / height of the output in pixels.
    pub viewport: [f32; 4],
    /// Exposure (linear factor), ambient intensity, fog density (1/m), near plane (m).
    pub light: [f32; 4],
    /// Item count, block count, blocks per segment row, draw slots per class.
    pub cull: [u32; 4],
    /// Scratch offset of the block sums, scratch offset of the segment totals, frame index,
    /// zero.
    pub scratch: [u32; 4],
    /// Debug view, draw mode (0 A, 1 B, 2 direct), output is sRGB (1) or unorm (0), zero.
    pub mode: [u32; 4],
    /// Width and height of the block texture layers in texels; focal length in pixels (pixels
    /// per metre across the view ray at a depth of 1 m: half the output height times
    /// `1 / tan(fov_y / 2)`); the edge overlap of the quads in pixels ([`EDGE_OVERLAP_PX`]).
    pub raster: [f32; 4],
    /// rgb: sky colour at the horizon below the sun; w: zero.
    pub sky_sunset: [f32; 4],
    /// Fog falloff with height (1/m), camera altitude above the fog's reference level (m),
    /// start and end of the fade into the sky (m; equal values disable it).
    pub atmosphere: [f32; 4],
    /// xyz: camera position in its reference frame modulo the cloud anchor period (m); w: 1
    /// when the frame is accumulated by temporal anti-aliasing (dither patterns may then
    /// change from frame to frame), 0 otherwise.
    pub anchor: [f32; 4],
    /// The ground map ([`super::GroundMap`]): grid x and z (m) of its corner, `1 / extent`
    /// (1/m) and its side in texels.
    pub ground_map: [f32; 4],
    /// rgb: the light of the sky on a surface facing up, before the ambient intensity
    /// ([`super::Lighting::sky_light`]); w: the gain of the sky's own brightness
    /// ([`super::Lighting::sky_gain`]).
    pub sky_light: [f32; 4],
}

const _: () = assert!(size_of::<GpuFrame>() == 464);

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use cp_core::ids::MaterialId;
    use wgpu::naga;

    use super::*;
    use crate::shader::{ShaderFile, ShaderLibrary, ShaderTarget};

    /// The member offsets and the size of every struct named in `names`, from a program that
    /// binds them.
    fn wgsl_layouts(names: &[&str]) -> BTreeMap<String, (Vec<(String, u32)>, u32)> {
        let files: &'static [ShaderFile] = Vec::leak(
            ShaderLibrary::builtin()
                .files()
                .iter()
                .copied()
                .chain([ShaderFile::new(
                    "layout_probe.wgsl",
                    "#include \"voxel/common.wgsl\"\n\
                     @group(0) @binding(0) var<storage, read> items: array<DrawItem>;\n\
                     @group(0) @binding(1) var<storage, read> objects: array<Object>;\n\
                     @group(0) @binding(2) var<uniform> frame: Frame;\n",
                )])
                .collect(),
        );
        let validated = ShaderLibrary::new(files)
            .validate("layout_probe.wgsl", &ShaderTarget::baseline())
            .unwrap_or_else(|error| panic!("{error}"));
        let module = &validated.module;
        let mut layouts = BTreeMap::new();
        for (_, ty) in module.types.iter() {
            let Some(name) = ty.name.as_deref() else {
                continue;
            };
            if !names.contains(&name) {
                continue;
            }
            let naga::TypeInner::Struct { members, span } = &ty.inner else {
                panic!("{name} is not a struct");
            };
            let offsets = members
                .iter()
                .map(|member| (member.name.clone().unwrap(), member.offset))
                .collect();
            layouts.insert(name.to_owned(), (offsets, *span));
        }
        layouts
    }

    fn owned(fields: &[(&str, usize)]) -> Vec<(String, u32)> {
        fields
            .iter()
            .map(|&(name, offset)| (name.to_owned(), u32::try_from(offset).unwrap()))
            .collect()
    }

    #[test]
    fn the_wgsl_structs_have_the_rust_layouts() {
        use std::mem::offset_of;
        let layouts = wgsl_layouts(&["DrawItem", "Object", "Frame"]);
        let item = owned(&[
            ("origin", offset_of!(GpuDrawItem, origin)),
            ("lod_flags", offset_of!(GpuDrawItem, lod_flags)),
            ("cell_bounds", offset_of!(GpuDrawItem, cell_bounds)),
            ("quad_base", offset_of!(GpuDrawItem, quad_base)),
            ("counts", offset_of!(GpuDrawItem, counts)),
            ("object", offset_of!(GpuDrawItem, object)),
            ("fade", offset_of!(GpuDrawItem, fade)),
            ("reserved", offset_of!(GpuDrawItem, reserved)),
        ]);
        assert_eq!(layouts["DrawItem"], (item, 64));
        let object = owned(&[
            ("local_to_clip", offset_of!(GpuObject, local_to_clip)),
            (
                "prev_local_to_clip",
                offset_of!(GpuObject, prev_local_to_clip),
            ),
            ("local_to_render", offset_of!(GpuObject, local_to_render)),
            ("anchor", offset_of!(GpuObject, anchor)),
            ("flags", offset_of!(GpuObject, flags)),
            ("camera_local", offset_of!(GpuObject, camera_local)),
            ("reserved", offset_of!(GpuObject, reserved)),
        ]);
        assert_eq!(layouts["Object"], (object, 224));
        let frame = owned(&[
            ("render_to_clip", offset_of!(GpuFrame, render_to_clip)),
            ("clip_to_render", offset_of!(GpuFrame, clip_to_render)),
            ("sun_direction", offset_of!(GpuFrame, sun_direction)),
            ("sun_color", offset_of!(GpuFrame, sun_color)),
            ("sky_zenith", offset_of!(GpuFrame, sky_zenith)),
            ("sky_horizon", offset_of!(GpuFrame, sky_horizon)),
            ("ground", offset_of!(GpuFrame, ground)),
            ("up", offset_of!(GpuFrame, up)),
            ("tints", offset_of!(GpuFrame, tints)),
            ("viewport", offset_of!(GpuFrame, viewport)),
            ("light", offset_of!(GpuFrame, light)),
            ("cull", offset_of!(GpuFrame, cull)),
            ("scratch", offset_of!(GpuFrame, scratch)),
            ("mode", offset_of!(GpuFrame, mode)),
            ("raster", offset_of!(GpuFrame, raster)),
            ("sky_sunset", offset_of!(GpuFrame, sky_sunset)),
            ("atmosphere", offset_of!(GpuFrame, atmosphere)),
            ("anchor", offset_of!(GpuFrame, anchor)),
            ("ground_map", offset_of!(GpuFrame, ground_map)),
            ("sky_light", offset_of!(GpuFrame, sky_light)),
        ]);
        assert_eq!(layouts["Frame"], (frame, 464));
    }

    #[test]
    fn the_wgsl_constants_match_the_rust_ones() {
        let source = ShaderLibrary::builtin()
            .file("voxel/common.wgsl")
            .expect("the voxel include is in the library")
            .source;
        let expected = [
            format!("const LOD_LEVELS: u32 = {LOD_LEVELS}u;"),
            format!("const DRAW_CLASSES: u32 = {DRAW_CLASSES}u;"),
            format!("const SEGMENTS: u32 = {SEGMENTS}u;"),
            format!("const SLOTS: u32 = {RANGES_PER_LAYER}u;"),
            format!("const LAYER_CUTOUT: u32 = {}u;", MeshLayer::Cutout as u32),
            format!("const TEXTURED_LEVELS: u32 = {TEXTURED_LEVELS}u;"),
            format!("const CULL_GROUP: u32 = {CULL_WORKGROUP}u;"),
            format!("const COUNTER_QUADS: u32 = {COUNTER_QUADS}u;"),
            "const CULL_MARGIN: f32 = 1e-4;".to_owned(),
            "const OVERLAP_STRETCH_MAX: f32 = 256.0;".to_owned(),
            "const OVERLAP_SLIDE_MAX_PX: f32 = 0.75;".to_owned(),
        ];
        for line in expected {
            assert!(source.contains(&line), "voxel/common.wgsl lacks `{line}`");
        }
        assert_eq!(CULL_MARGIN, 1e-4);
        assert_eq!(OVERLAP_STRETCH_MAX, 256.0);
        assert_eq!(OVERLAP_SLIDE_MAX_PX, 0.75);
        assert_eq!(SEGMENTS, 42);
        assert_eq!(MAX_ITEM_QUADS, 202_752);
    }

    #[test]
    fn storage_slots_put_positive_faces_first() {
        for (slot, face) in FACE_STORAGE_ORDER.into_iter().enumerate() {
            assert_eq!(storage_slot(face), slot);
        }
        let faces: Vec<Face> = Face::ALL.to_vec();
        let mut by_slot = faces.clone();
        by_slot.sort_by_key(|&face| storage_slot(face));
        assert_eq!(by_slot, FACE_STORAGE_ORDER);
    }

    #[test]
    fn wall_masks_and_fades_pack_as_documented() {
        let mask = WallMask::from_faces(&[Face::PosY, Face::NegZ]);
        assert_eq!(mask.bits(), 0b10_0010);
        assert!(mask.contains(Face::PosY) && mask.contains(Face::NegZ));
        assert!(!mask.contains(Face::PosX));
        assert_eq!(WallMask::from_bits(0xFF), WallMask::ALL);
        assert!(WallMask::NONE.is_empty());

        assert_eq!(Fade::FULL.pack(), 255);
        assert!(Fade::FULL.is_full());
        assert_eq!(Fade::new(0.5, true).pack(), 128 | 256);
        assert_eq!(Fade::new(-1.0, false).coverage_bits(), 0);
        assert_eq!(Fade::new(f32::NAN, false).coverage_bits(), 0);
        assert_eq!(Fade::new(7.0, false), Fade::FULL);
        assert!(Fade::new(1.0, true).is_full());
        assert_eq!(Fade::default(), Fade::FULL);
    }

    #[test]
    fn items_pack_and_unpack_their_fields() {
        let bounds = CellBounds {
            min: [1, 2, 3],
            max: [31, 30, 29],
        };
        assert_eq!(CellBounds::unpack(bounds.pack()), bounds);
        assert_eq!(bounds.pack() >> 30, 0);
        assert_eq!(
            CellBounds::unpack(CellBounds::CHUNK.pack()),
            CellBounds::CHUNK
        );
        let mut item = GpuDrawItem {
            lod_flags: GpuDrawItem::pack_lod_flags(20, MeshLayer::Fluid, WallMask::ALL),
            fade: Fade::FULL.pack(),
            ..GpuDrawItem::default()
        };
        assert_eq!(item.level(), 20);
        assert_eq!(item.layer_index(), 2);
        assert_eq!(item.walls(), WallMask::ALL);
        assert_eq!(item.cell_size(), 1 << 20);
        assert_eq!(item.draw_class(), 0);
        assert_eq!(item.segment(), 20);
        item.lod_flags = GpuDrawItem::pack_lod_flags(3, MeshLayer::Cutout, WallMask::NONE);
        assert_eq!(item.draw_class(), 1);
        assert_eq!(item.segment(), LOD_LEVELS + 3);
        item.lod_flags = GpuDrawItem::pack_lod_flags(3, MeshLayer::Opaque, WallMask::NONE);
        item.fade = Fade::new(0.25, false).pack();
        assert_eq!(item.draw_class(), 1, "fading items use the cutout pipeline");
        item.counts = [5 | 7 << 16, 0, 1, 0, 0, 65535 | 1024 << 16];
        assert_eq!(item.slot_count(0), 5);
        assert_eq!(item.slot_count(6), 7);
        assert_eq!(item.slot_count(5), 65535);
        assert_eq!(item.slot_count(11), 1024);
        assert_eq!(item.quad_count(), 5 + 7 + 1 + 65535 + 1024);
    }

    #[test]
    fn quad_bounds_cover_the_quads_cells() {
        for face in Face::ALL {
            let quad = PackedQuad::new([2, 3, 4], 5, 6, face, MaterialId(1), 0, false).unwrap();
            let bounds = CellBounds::of_quad(quad);
            let mut min = [u8::MAX; 3];
            let mut max = [0_u8; 3];
            for cell in quad.covered_cells() {
                for axis in 0..3 {
                    min[axis] = min[axis].min(cell[axis]);
                    max[axis] = max[axis].max(cell[axis]);
                }
            }
            assert_eq!(bounds, CellBounds { min, max }, "{face:?}");
        }
        let a = CellBounds::cell([4, 9, 1]);
        let b = CellBounds::cell([2, 12, 5]);
        assert_eq!(
            a.union(b),
            CellBounds {
                min: [2, 9, 1],
                max: [4, 12, 5]
            }
        );
    }
}
