//! The material table: one storage buffer entry per material id (contract §5.3).
//!
//! The voxel shaders look up `materials[quad.material]` for the texture layer and for what far
//! LOD levels draw instead of a texture: the average colour, roughness, emission, metalness
//! and the tint class. cp-app fills the table from the content registry; cp-render knows no
//! content types (contract §3.1). The WGSL side is `struct Material` in
//! `common/material.wgsl`; a test holds both layouts against each other.

use bytemuck::{Pod, Zeroable};
use thiserror::Error;

use crate::error::RenderError;
use crate::gpu::GpuContext;
use crate::scope::capture;

/// Number of tint classes; `tint_class` is below it. The numbers are the stable discriminants
/// of `cp_content::TintClass`: 0 none, 1 grass, 2 foliage, 3 water.
pub const TINT_CLASS_COUNT: u32 = 4;

/// Most entries a table can have: one per [`cp_core::ids::MaterialId`] (16 bits).
pub const MAX_MATERIALS: usize = 1 << 16;

/// How the G-buffer pass may turn a material's texture from cell to cell, so that a surface of
/// many blocks does not show the same tile in the same orientation everywhere. Each 1 m cell
/// draws one of the allowed transforms by a hash of its position (stable from frame to frame
/// and on every LOD level that is textured).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum TextureTurn {
    /// Always as painted (the default; textures whose orientation carries meaning).
    #[default]
    Fixed = 0,
    /// Any of the eight rotations by quarter turns and mirror images (patterns without a
    /// direction: stone, dirt, gravel, grass seen from above, leaves).
    Free = 1,
    /// Only the transforms that keep the axes of the pattern: mirror images along either
    /// axis and the half turn (patterns with a grain, such as wind ripples or planks).
    Axial = 2,
    /// Only the mirror image left to right (patterns with an up side: the fringe of a grass
    /// side, bark, strata).
    Upright = 3,
}

/// Bits of [`GpuMaterial::variation`].
pub mod variation {
    /// Mask of the [`super::TextureTurn`] in bits 0–1.
    pub const TURN_MASK: u32 = 0b11;
    /// The surface is foliage: the lighting pass wraps the sun's light around it (light
    /// passes through leaves) and gives it no sky reflection.
    pub const FOLIAGE: u32 = 1 << 2;
    /// The material is ground that may be covered: [`COVER_LAYERS`] more texture layers follow
    /// its variants (flowers, blossoms, fallen leaves, straw, in this order), which the
    /// G-buffer pass lays on the tops of its cells by the ground map
    /// ([`crate::voxel::GroundMap`]).
    pub const GROUND_COVER: u32 = 1 << 3;
    /// Number of ground cover layers of a material with [`GROUND_COVER`].
    pub const COVER_LAYERS: u32 = 4;
    /// First bit of the number of extra texture layers (variants) in bits 8–11.
    pub const VARIANT_SHIFT: u32 = 8;
    /// Most extra texture layers a material may have.
    pub const MAX_VARIANTS: u32 = 15;
}

/// One material as the shaders see it: 48 bytes, the layout of WGSL `struct Material`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
pub struct GpuMaterial {
    /// Linear average albedo of the material's texture, `0..=1` per channel; far LOD levels
    /// draw this colour.
    pub average_color: [f32; 3],
    /// Perceptual roughness, `0..=1`.
    pub roughness: f32,
    /// Emission factor, finite and `>= 0` (the exposure model defines absolute brightness).
    pub emissive: f32,
    /// Metalness, `0..=1`.
    pub metallic: f32,
    /// Layer of the material's texture in the block texture arrays.
    pub texture_layer: u32,
    /// Tint class, below [`TINT_CLASS_COUNT`].
    pub tint_class: u32,
    /// Variation flags ([`variation`]): the [`TextureTurn`] in bits 0–1, [`variation::FOLIAGE`]
    /// in bit 2, [`variation::GROUND_COVER`] in bit 3, the number of extra texture layers in
    /// bits 8–11; every other bit is zero.
    pub variation: u32,
    /// First of the extra texture layers (variants of the texture painted with other seeds,
    /// one of which or the texture itself each cell shows, then the ground cover layers);
    /// ignored without variants and cover.
    pub variant_layer: u32,
    /// Zero.
    pub reserved: [u32; 2],
}

impl GpuMaterial {
    /// Size of one entry in bytes, also its array stride in a storage buffer.
    pub const SIZE: u64 = 48;

    /// The variation bits of a material: how its texture may turn, whether it is foliage, and
    /// how many extra texture layers it has (at most [`variation::MAX_VARIANTS`]; more are
    /// cut).
    #[must_use]
    pub const fn variation_bits(turn: TextureTurn, foliage: bool, variants: u32) -> u32 {
        let variants = if variants > variation::MAX_VARIANTS {
            variation::MAX_VARIANTS
        } else {
            variants
        };
        (turn as u32)
            | if foliage { variation::FOLIAGE } else { 0 }
            | (variants << variation::VARIANT_SHIFT)
    }

    /// Checks the value ranges the shaders rely on; `index` is the entry's place in its table
    /// and goes into the error.
    ///
    /// # Errors
    /// [`MaterialError::OutOfRange`] for the first field outside its range.
    pub fn validate(&self, index: usize) -> Result<(), MaterialError> {
        let unit = |value: f32| (0.0..=1.0).contains(&value);
        let out_of_range = |field: &'static str, value: String, allowed: &'static str| {
            Err(MaterialError::OutOfRange {
                index,
                field,
                value,
                allowed,
            })
        };
        if !self.average_color.iter().all(|&channel| unit(channel)) {
            return out_of_range(
                "average_color",
                format!("{:?}", self.average_color),
                "0..=1 per channel",
            );
        }
        if !unit(self.roughness) {
            return out_of_range("roughness", self.roughness.to_string(), "0..=1");
        }
        if !(self.emissive.is_finite() && self.emissive >= 0.0) {
            return out_of_range("emissive", self.emissive.to_string(), "finite and >= 0");
        }
        if !unit(self.metallic) {
            return out_of_range("metallic", self.metallic.to_string(), "0..=1");
        }
        if self.tint_class >= TINT_CLASS_COUNT {
            return out_of_range("tint_class", self.tint_class.to_string(), "0..=3");
        }
        let known = variation::TURN_MASK
            | variation::FOLIAGE
            | variation::GROUND_COVER
            | (variation::MAX_VARIANTS << variation::VARIANT_SHIFT);
        if self.variation & !known != 0 {
            return out_of_range(
                "variation",
                format!("{:#x}", self.variation),
                "bits 0-3 and 8-11",
            );
        }
        if self.reserved != [0, 0] {
            return out_of_range("reserved", format!("{:?}", self.reserved), "zero");
        }
        Ok(())
    }
}

const _: () = assert!(size_of::<GpuMaterial>() as u64 == GpuMaterial::SIZE);

/// Why a material table cannot hold the given entries.
#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub enum MaterialError {
    /// A field of an entry is outside the range the shaders rely on.
    #[error("material {index}: {field} = {value} is outside {allowed}")]
    OutOfRange {
        /// Index of the entry in its table.
        index: usize,
        /// Name of the field.
        field: &'static str,
        /// The value as text.
        value: String,
        /// The allowed range.
        allowed: &'static str,
    },
    /// A table must have at least one entry (a storage binding cannot be empty).
    #[error("a material table needs at least one entry")]
    Empty,
    /// More entries than material ids exist.
    #[error("{count} materials exceed the {MAX_MATERIALS} material ids")]
    TooMany {
        /// The number of entries given.
        count: usize,
    },
    /// An update reaches past the end of the table.
    #[error("updating {count} materials from {first} runs past the table of {len}")]
    UpdateOutOfBounds {
        /// First entry to update.
        first: usize,
        /// Number of entries given.
        count: usize,
        /// Entries in the table.
        len: usize,
    },
}

/// The material table on the GPU: a storage buffer of [`GpuMaterial`], bound read-only as
/// `array<Material>`. Its size is fixed at creation; parameters can be updated in place (hot
/// reload of material parameters, contract §4.4).
#[derive(Debug)]
pub struct MaterialTable {
    buffer: wgpu::Buffer,
    len: usize,
}

impl MaterialTable {
    /// Uploads `materials`; entry `i` is material id `i`.
    ///
    /// # Errors
    /// [`MaterialError`] for an empty or oversized table or an entry out of range,
    /// [`RenderError::Wgpu`] when the buffer cannot be created.
    pub fn new(gpu: &GpuContext, materials: &[GpuMaterial]) -> Result<Self, RenderError> {
        if materials.is_empty() {
            return Err(MaterialError::Empty.into());
        }
        if materials.len() > MAX_MATERIALS {
            return Err(MaterialError::TooMany {
                count: materials.len(),
            }
            .into());
        }
        check_all(materials, 0)?;
        let buffer = capture(gpu, "material table", |device| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("material table"),
                size: GpuMaterial::SIZE * materials.len() as u64,
                // COPY_SRC for read-backs in tests and debugging tools.
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            })
        })?;
        gpu.queue
            .write_buffer(&buffer, 0, bytemuck::cast_slice(materials));
        Ok(Self {
            buffer,
            len: materials.len(),
        })
    }

    /// Replaces the entries from `first` on with `materials`; visible to work submitted after
    /// this call.
    ///
    /// # Errors
    /// [`MaterialError::UpdateOutOfBounds`] and [`MaterialError::OutOfRange`].
    pub fn update(
        &self,
        gpu: &GpuContext,
        first: usize,
        materials: &[GpuMaterial],
    ) -> Result<(), RenderError> {
        if first
            .checked_add(materials.len())
            .is_none_or(|end| end > self.len)
        {
            return Err(MaterialError::UpdateOutOfBounds {
                first,
                count: materials.len(),
                len: self.len,
            }
            .into());
        }
        check_all(materials, first)?;
        gpu.queue.write_buffer(
            &self.buffer,
            GpuMaterial::SIZE * first as u64,
            bytemuck::cast_slice(materials),
        );
        Ok(())
    }

    /// The storage buffer.
    pub fn buffer(&self) -> &wgpu::Buffer {
        &self.buffer
    }

    /// Number of entries.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Always `false`: a table has at least one entry.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// The whole buffer as a bind group resource.
    pub fn binding(&self) -> wgpu::BindingResource<'_> {
        self.buffer.as_entire_binding()
    }
}

fn check_all(materials: &[GpuMaterial], first: usize) -> Result<(), MaterialError> {
    materials
        .iter()
        .enumerate()
        .try_for_each(|(offset, material)| material.validate(first + offset))
}

#[cfg(test)]
mod tests {
    use std::mem::offset_of;

    use wgpu::naga;

    use super::*;
    use crate::shader::{ShaderFile, ShaderLibrary, ShaderTarget};

    fn stone() -> GpuMaterial {
        GpuMaterial {
            average_color: [0.4, 0.41, 0.43],
            roughness: 0.9,
            emissive: 0.0,
            metallic: 0.0,
            texture_layer: 3,
            tint_class: 0,
            variation: GpuMaterial::variation_bits(TextureTurn::Free, false, 2),
            variant_layer: 40,
            reserved: [0, 0],
        }
    }

    #[test]
    fn the_wgsl_struct_has_the_rust_layout() {
        // The include alone, plus a binding so that the array type and its stride exist.
        let files: &'static [ShaderFile] = Vec::leak(
            ShaderLibrary::builtin()
                .files()
                .iter()
                .copied()
                .chain([ShaderFile::new(
                    "layout_probe.wgsl",
                    "#include \"common/material.wgsl\"\n\
                     @group(0) @binding(0) var<storage, read> materials: array<Material>;\n",
                )])
                .collect(),
        );
        let validated = ShaderLibrary::new(files)
            .validate("layout_probe.wgsl", &ShaderTarget::baseline())
            .unwrap();
        let module = &validated.module;
        let (_, material) = module
            .types
            .iter()
            .find(|(_, ty)| ty.name.as_deref() == Some("Material"))
            .expect("struct Material");
        let naga::TypeInner::Struct { members, span } = &material.inner else {
            panic!("Material is not a struct");
        };
        assert_eq!(u64::from(*span), GpuMaterial::SIZE);
        let offsets: Vec<(String, u32)> = members
            .iter()
            .map(|member| (member.name.clone().unwrap(), member.offset))
            .collect();
        let rust = [
            ("average_color", offset_of!(GpuMaterial, average_color)),
            ("roughness", offset_of!(GpuMaterial, roughness)),
            ("emissive", offset_of!(GpuMaterial, emissive)),
            ("metallic", offset_of!(GpuMaterial, metallic)),
            ("texture_layer", offset_of!(GpuMaterial, texture_layer)),
            ("tint_class", offset_of!(GpuMaterial, tint_class)),
            ("variation", offset_of!(GpuMaterial, variation)),
            ("variant_layer", offset_of!(GpuMaterial, variant_layer)),
            ("reserved", offset_of!(GpuMaterial, reserved)),
        ]
        .map(|(name, offset)| (name.to_owned(), u32::try_from(offset).unwrap()));
        assert_eq!(offsets, rust);

        let strides: Vec<u32> = module
            .types
            .iter()
            .filter_map(|(_, ty)| match ty.inner {
                naga::TypeInner::Array { stride, .. } => Some(stride),
                _ => None,
            })
            .collect();
        assert_eq!(strides, [48]);
        assert_eq!(size_of::<GpuMaterial>(), 48);
    }

    #[test]
    fn variation_bits_pack_turn_foliage_and_variants() {
        assert_eq!(GpuMaterial::variation_bits(TextureTurn::Fixed, false, 0), 0);
        assert_eq!(
            GpuMaterial::variation_bits(TextureTurn::Free, true, 2),
            1 | variation::FOLIAGE | (2 << variation::VARIANT_SHIFT)
        );
        assert_eq!(
            GpuMaterial::variation_bits(TextureTurn::Upright, false, 99) >> 8,
            variation::MAX_VARIANTS
        );
        assert_eq!(variation::GROUND_COVER, 8);
        assert_eq!(variation::COVER_LAYERS, 4);
        let source = ShaderLibrary::builtin()
            .file("common/material.wgsl")
            .unwrap()
            .source;
        for line in [
            "const TURN_FIXED: u32 = 0u;",
            "const TURN_FREE: u32 = 1u;",
            "const TURN_AXIAL: u32 = 2u;",
            "const TURN_UPRIGHT: u32 = 3u;",
            "const VARIATION_FOLIAGE: u32 = 4u;",
            "const VARIATION_GROUND_COVER: u32 = 8u;",
            "const VARIANT_SHIFT: u32 = 8u;",
            "const COVER_LAYERS: u32 = 4u;",
        ] {
            assert!(source.contains(line), "{line}");
        }
    }

    #[test]
    fn the_tint_constants_match_the_classes() {
        let source = ShaderLibrary::builtin()
            .file("common/material.wgsl")
            .unwrap()
            .source;
        for (name, number) in [
            ("TINT_NONE", 0),
            ("TINT_GRASS", 1),
            ("TINT_FOLIAGE", 2),
            ("TINT_WATER", 3),
        ] {
            let line = format!("const {name}: u32 = {number}u;");
            assert!(source.contains(&line), "{line}");
        }
        assert_eq!(TINT_CLASS_COUNT, 4);
    }

    /// Changes one field of a material to a value outside its range.
    type Spoil = fn(&mut GpuMaterial);

    #[test]
    fn entries_outside_their_ranges_are_rejected() {
        assert_eq!(stone().validate(0), Ok(()));
        let cases: [(&str, Spoil); 9] = [
            ("average_color", |m| m.average_color[1] = 1.5),
            ("average_color", |m| m.average_color[2] = f32::NAN),
            ("roughness", |m| m.roughness = -0.1),
            ("emissive", |m| m.emissive = f32::INFINITY),
            ("emissive", |m| m.emissive = -1.0),
            ("metallic", |m| m.metallic = 2.0),
            ("tint_class", |m| m.tint_class = 4),
            ("variation", |m| m.variation = 1 << 12),
            ("reserved", |m| m.reserved = [0, 7]),
        ];
        for (field, spoil) in cases {
            let mut material = stone();
            spoil(&mut material);
            match material.validate(7) {
                Err(MaterialError::OutOfRange {
                    index, field: got, ..
                }) => {
                    assert_eq!((index, got), (7, field));
                }
                other => panic!("{field}: expected a range error, got {other:?}"),
            }
        }
        // Edges of the ranges are allowed.
        let edge = GpuMaterial {
            average_color: [0.0, 1.0, 0.5],
            roughness: 1.0,
            emissive: 1000.0,
            metallic: 1.0,
            texture_layer: u32::MAX,
            tint_class: 3,
            variation: GpuMaterial::variation_bits(TextureTurn::Upright, true, 15),
            variant_layer: u32::MAX,
            reserved: [0, 0],
        };
        assert_eq!(edge.validate(0), Ok(()));
        let message = MaterialError::OutOfRange {
            index: 2,
            field: "roughness",
            value: "1.5".to_owned(),
            allowed: "0..=1",
        }
        .to_string();
        assert_eq!(message, "material 2: roughness = 1.5 is outside 0..=1");
    }

    #[test]
    fn the_byte_image_is_the_field_order() {
        let bytes = bytemuck::bytes_of(&stone()).to_vec();
        assert_eq!(bytes.len(), 48);
        assert_eq!(&bytes[24..28], &3_u32.to_le_bytes());
        assert_eq!(&bytes[12..16], &0.9_f32.to_le_bytes());
        assert_eq!(&bytes[36..40], &40_u32.to_le_bytes());
    }
}
