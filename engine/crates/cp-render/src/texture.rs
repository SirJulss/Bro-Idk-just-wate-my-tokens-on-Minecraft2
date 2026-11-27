//! 2D texture arrays for block textures (contract §5.3).
//!
//! Two arrays share their layer numbering: the albedo array (`Rgba8UnormSrgb`: sRGB colour,
//! linear alpha as cut-out mask) and the data array (`Rgba8Unorm`: normal, roughness, height
//! and similar, no colour conversion). Every layer carries its full mip chain, computed by the
//! caller (cp-app from cp-proctex), because a correct pixel-art minification needs more care
//! than a box filter in the wrong colour space. The pixel-art sampling of contract §5.3 samples
//! these arrays with explicit gradients.

use thiserror::Error;

use crate::error::RenderError;
use crate::gpu::GpuContext;
use crate::scope::capture;

/// Bytes per texel of both kinds of array.
const TEXEL_BYTES: u32 = 4;

/// What a texture array holds, and with that its format.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TextureArrayKind {
    /// Albedo: `Rgba8UnormSrgb`, colour channels sRGB-encoded, alpha linear.
    Albedo,
    /// Normal, roughness, height and other data: `Rgba8Unorm`, read as stored.
    Linear,
}

impl TextureArrayKind {
    /// The texture format of the kind.
    pub fn format(self) -> wgpu::TextureFormat {
        match self {
            Self::Albedo => wgpu::TextureFormat::Rgba8UnormSrgb,
            Self::Linear => wgpu::TextureFormat::Rgba8Unorm,
        }
    }
}

/// Why texture data does not fit its description or the device.
#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub enum TextureError {
    /// A side of the layers is zero.
    #[error("{label}: texture layers of {width}x{height} have no texels")]
    EmptyLayer {
        /// Label of the array.
        label: String,
        /// Requested width in texels.
        width: u32,
        /// Requested height in texels.
        height: u32,
    },
    /// A layer came with a different number of mip levels than the full chain.
    #[error("{label}: layer {layer} has {got} mip levels, the full chain has {expected}")]
    MipCount {
        /// Label of the array.
        label: String,
        /// Index the layer would have had.
        layer: u32,
        /// Mip levels given.
        got: usize,
        /// Mip levels of the full chain.
        expected: u32,
    },
    /// A mip level has the wrong number of bytes for its size.
    #[error(
        "{label}: mip level {level} of layer {layer} has {got} bytes, {width}x{height} RGBA8 \
         needs {expected}"
    )]
    MipSize {
        /// Label of the array.
        label: String,
        /// Index the layer would have had.
        layer: u32,
        /// The mip level.
        level: u32,
        /// Bytes given.
        got: usize,
        /// Bytes needed.
        expected: usize,
        /// Width of the level in texels.
        width: u32,
        /// Height of the level in texels.
        height: u32,
    },
    /// An array without layers cannot be created.
    #[error("{label}: a texture array needs at least one layer")]
    NoLayers {
        /// Label of the array.
        label: String,
    },
    /// More layers than the device allows.
    #[error("{label}: {layers} layers exceed the device limit of {limit}")]
    TooManyLayers {
        /// Label of the array.
        label: String,
        /// Layers pushed.
        layers: u32,
        /// `max_texture_array_layers` of the device.
        limit: u32,
    },
    /// Layers larger than the device allows.
    #[error("{label}: layers of {width}x{height} exceed the device limit of {limit} texels")]
    TooLarge {
        /// Label of the array.
        label: String,
        /// Width of the layers.
        width: u32,
        /// Height of the layers.
        height: u32,
        /// `max_texture_dimension_2d` of the device.
        limit: u32,
    },
}

/// Mip levels of the full chain of a `width` x `height` image: down to 1x1.
pub fn full_mip_level_count(width: u32, height: u32) -> u32 {
    let largest = width.max(height).max(1);
    u32::BITS - largest.leading_zeros()
}

/// Size of mip level `level` of a `width` x `height` image: each side halved per level,
/// rounded down, at least 1.
pub fn mip_size(width: u32, height: u32, level: u32) -> (u32, u32) {
    let shrink = |side: u32| side.checked_shr(level).unwrap_or(0).max(1);
    (shrink(width), shrink(height))
}

/// Collects the layers of a texture array on the CPU and uploads them in one go.
///
/// ```
/// use cp_render::texture::{TextureArrayBuilder, TextureArrayKind};
///
/// let mut albedo = TextureArrayBuilder::new("albedo", TextureArrayKind::Albedo, 16, 16)?;
/// assert_eq!(albedo.mip_level_count(), 5); // 16, 8, 4, 2, 1
/// let mips: Vec<Vec<u8>> = (0..5).map(|level| vec![255; (16 >> level) * (16 >> level) * 4]).collect();
/// let levels: Vec<&[u8]> = mips.iter().map(Vec::as_slice).collect();
/// assert_eq!(albedo.push_layer(&levels)?, 0);
/// # Ok::<(), cp_render::texture::TextureError>(())
/// ```
#[derive(Clone, Debug)]
pub struct TextureArrayBuilder {
    label: String,
    kind: TextureArrayKind,
    width: u32,
    height: u32,
    mip_level_count: u32,
    layers: u32,
    /// Per mip level, the texels of every layer pushed so far, layer after layer.
    levels: Vec<Vec<u8>>,
}

impl TextureArrayBuilder {
    /// An empty array of `kind` whose layers are `width` x `height` texels.
    ///
    /// # Errors
    /// [`TextureError::EmptyLayer`] when a side is zero.
    pub fn new(
        label: impl Into<String>,
        kind: TextureArrayKind,
        width: u32,
        height: u32,
    ) -> Result<Self, TextureError> {
        let label = label.into();
        if width == 0 || height == 0 {
            return Err(TextureError::EmptyLayer {
                label,
                width,
                height,
            });
        }
        let mip_level_count = full_mip_level_count(width, height);
        Ok(Self {
            label,
            kind,
            width,
            height,
            mip_level_count,
            layers: 0,
            levels: vec![Vec::new(); mip_level_count as usize],
        })
    }

    /// Mip levels every layer has: the full chain.
    pub fn mip_level_count(&self) -> u32 {
        self.mip_level_count
    }

    /// Layers pushed so far.
    pub fn layer_count(&self) -> u32 {
        self.layers
    }

    /// Adds a layer and returns its index. `mips[level]` holds the texels of that mip level,
    /// rows top to bottom, tightly packed RGBA8 (see [`mip_size`]); every level of the full
    /// chain must be given.
    ///
    /// # Errors
    /// [`TextureError::MipCount`] and [`TextureError::MipSize`]; the builder is unchanged.
    pub fn push_layer(&mut self, mips: &[&[u8]]) -> Result<u32, TextureError> {
        if mips.len() != self.mip_level_count as usize {
            return Err(TextureError::MipCount {
                label: self.label.clone(),
                layer: self.layers,
                got: mips.len(),
                expected: self.mip_level_count,
            });
        }
        for (level, data) in (0..).zip(mips) {
            let (width, height) = mip_size(self.width, self.height, level);
            let expected = width as usize * height as usize * TEXEL_BYTES as usize;
            if data.len() != expected {
                return Err(TextureError::MipSize {
                    label: self.label.clone(),
                    layer: self.layers,
                    level,
                    got: data.len(),
                    expected,
                    width,
                    height,
                });
            }
        }
        for (level, data) in self.levels.iter_mut().zip(mips) {
            level.extend_from_slice(data);
        }
        self.layers += 1;
        Ok(self.layers - 1)
    }

    /// Creates the texture array on the device and uploads every layer and mip level.
    ///
    /// # Errors
    /// [`TextureError::NoLayers`], [`TextureError::TooManyLayers`] and
    /// [`TextureError::TooLarge`] against the device limits; [`RenderError::Wgpu`] when the
    /// device rejects the texture.
    pub fn build(&self, gpu: &GpuContext) -> Result<TextureArray, RenderError> {
        let limits = &gpu.caps.enabled_limits;
        if self.layers == 0 {
            return Err(TextureError::NoLayers {
                label: self.label.clone(),
            }
            .into());
        }
        if self.layers > limits.max_texture_array_layers {
            return Err(TextureError::TooManyLayers {
                label: self.label.clone(),
                layers: self.layers,
                limit: limits.max_texture_array_layers,
            }
            .into());
        }
        if self.width.max(self.height) > limits.max_texture_dimension_2d {
            return Err(TextureError::TooLarge {
                label: self.label.clone(),
                width: self.width,
                height: self.height,
                limit: limits.max_texture_dimension_2d,
            }
            .into());
        }
        let format = self.kind.format();
        let texture = capture(gpu, &format!("texture array {}", self.label), |device| {
            let texture = device.create_texture(&wgpu::TextureDescriptor {
                label: Some(&self.label),
                size: wgpu::Extent3d {
                    width: self.width,
                    height: self.height,
                    depth_or_array_layers: self.layers,
                },
                mip_level_count: self.mip_level_count,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage: wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::COPY_DST
                    | wgpu::TextureUsages::COPY_SRC,
                view_formats: &[],
            });
            for (level, data) in (0..).zip(&self.levels) {
                let (width, height) = mip_size(self.width, self.height, level);
                gpu.queue.write_texture(
                    wgpu::TexelCopyTextureInfo {
                        texture: &texture,
                        mip_level: level,
                        origin: wgpu::Origin3d::ZERO,
                        aspect: wgpu::TextureAspect::All,
                    },
                    data,
                    wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(width * TEXEL_BYTES),
                        rows_per_image: Some(height),
                    },
                    wgpu::Extent3d {
                        width,
                        height,
                        depth_or_array_layers: self.layers,
                    },
                );
            }
            texture
        })?;
        // A view of one layer would otherwise default to the 2D dimension.
        let view = texture.create_view(&wgpu::TextureViewDescriptor {
            label: Some(&self.label),
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            ..Default::default()
        });
        Ok(TextureArray {
            texture,
            view,
            kind: self.kind,
            width: self.width,
            height: self.height,
            layers: self.layers,
            mip_level_count: self.mip_level_count,
        })
    }
}

/// A texture array on the device, with a `D2Array` view over every layer and mip level.
#[derive(Clone, Debug)]
pub struct TextureArray {
    texture: wgpu::Texture,
    view: wgpu::TextureView,
    kind: TextureArrayKind,
    width: u32,
    height: u32,
    layers: u32,
    mip_level_count: u32,
}

impl TextureArray {
    /// The texture (usages `TEXTURE_BINDING`, `COPY_DST`, `COPY_SRC`).
    pub fn texture(&self) -> &wgpu::Texture {
        &self.texture
    }

    /// The `texture_2d_array<f32>` view over everything.
    pub fn view(&self) -> &wgpu::TextureView {
        &self.view
    }

    /// What the array holds.
    pub fn kind(&self) -> TextureArrayKind {
        self.kind
    }

    /// Size of layer and mip level 0 in texels.
    pub fn size(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    /// Number of layers.
    pub fn layer_count(&self) -> u32 {
        self.layers
    }

    /// Mip levels per layer (the full chain).
    pub fn mip_level_count(&self) -> u32 {
        self.mip_level_count
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The full chain of a layer filled with `value`.
    fn chain(width: u32, height: u32, value: u8) -> Vec<Vec<u8>> {
        (0..full_mip_level_count(width, height))
            .map(|level| {
                let (w, h) = mip_size(width, height, level);
                vec![value; (w * h * TEXEL_BYTES) as usize]
            })
            .collect()
    }

    fn slices(levels: &[Vec<u8>]) -> Vec<&[u8]> {
        levels.iter().map(Vec::as_slice).collect()
    }

    #[test]
    fn the_full_chain_goes_down_to_one_texel() {
        assert_eq!(full_mip_level_count(1, 1), 1);
        assert_eq!(full_mip_level_count(16, 16), 5);
        assert_eq!(full_mip_level_count(16, 4), 5);
        assert_eq!(full_mip_level_count(17, 3), 5);
        assert_eq!(full_mip_level_count(4096, 1), 13);
        assert_eq!(mip_size(16, 4, 0), (16, 4));
        assert_eq!(mip_size(16, 4, 3), (2, 1));
        assert_eq!(mip_size(16, 4, 4), (1, 1));
        assert_eq!(mip_size(5, 3, 1), (2, 1));
        assert_eq!(mip_size(5, 3, 40), (1, 1));
        // The last level of the chain is 1x1, the one before it is not.
        for (width, height) in [(16, 16), (16, 4), (17, 3), (1, 9), (300, 200)] {
            let count = full_mip_level_count(width, height);
            assert_eq!(mip_size(width, height, count - 1), (1, 1));
            if count > 1 {
                assert_ne!(mip_size(width, height, count - 2), (1, 1));
            }
        }
    }

    #[test]
    fn layers_need_the_full_chain_with_exact_sizes() {
        let mut builder = TextureArrayBuilder::new("test", TextureArrayKind::Linear, 8, 4).unwrap();
        assert_eq!(builder.mip_level_count(), 4);
        let good = chain(8, 4, 7);
        assert_eq!(builder.push_layer(&slices(&good)).unwrap(), 0);
        assert_eq!(builder.push_layer(&slices(&good)).unwrap(), 1);

        let short = &slices(&good)[..3];
        assert_eq!(
            builder.push_layer(short),
            Err(TextureError::MipCount {
                label: "test".to_owned(),
                layer: 2,
                got: 3,
                expected: 4,
            })
        );
        let mut wrong = good.clone();
        wrong[2].pop();
        assert_eq!(
            builder.push_layer(&slices(&wrong)),
            Err(TextureError::MipSize {
                label: "test".to_owned(),
                layer: 2,
                level: 2,
                got: 7,
                expected: 8,
                width: 2,
                height: 1,
            })
        );
        // Failed pushes leave nothing behind.
        assert_eq!(builder.layer_count(), 2);
        let bytes: Vec<usize> = builder.levels.iter().map(Vec::len).collect();
        assert_eq!(bytes, [2 * 128, 2 * 32, 2 * 8, 2 * 4]);
    }

    #[test]
    fn empty_layers_are_rejected_and_formats_follow_the_kind() {
        assert!(matches!(
            TextureArrayBuilder::new("x", TextureArrayKind::Albedo, 0, 16),
            Err(TextureError::EmptyLayer { .. })
        ));
        assert_eq!(
            TextureArrayKind::Albedo.format(),
            wgpu::TextureFormat::Rgba8UnormSrgb
        );
        assert_eq!(
            TextureArrayKind::Linear.format(),
            wgpu::TextureFormat::Rgba8Unorm
        );
    }
}
