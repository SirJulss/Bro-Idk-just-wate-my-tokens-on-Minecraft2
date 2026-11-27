//! The ground map: what kind of land lies under the surfaces of the drawn grids, as a small
//! texture over their x-z plane that the G-buffer pass samples. It colours grass and leaves
//! by the land they grow on and lays ground cover (flowers, fallen leaves, straw) on the
//! tops of grass blocks, so that a heath, a forest floor and a lush meadow differ although
//! they are made of one grass block (contract §8: "generated far surfaces tinted from macro
//! maps"; until macro maps exist, the application builds the map from its generator).
//!
//! cp-render knows no generator: the map is plain data, four channels per texel.

use super::scene::VoxelError;

/// A ground map over a square of the grid plane.
///
/// Texel `(i, j)` (x fastest) covers the grid x from `origin[0] + i · texel_metres` and z from
/// `origin[1] + j · texel_metres`, one `texel_metres` square; the shader filters the texels
/// bilinearly between their centres and clamps at the border. The channels, `0..=255`:
///
/// | channel | meaning |
/// |---|---|
/// | r | dryness of the region: 0 lush lowland, 128 neither, 255 dry heath |
/// | g | wood: the share of the ground under trees (0 open land, 255 a forest's core) |
/// | b | how alpine the place is: 0 below, 255 near the snow line |
/// | a | 255 where the map holds data; 0 means "no map" and the shader keeps its defaults |
#[derive(Clone, Debug, PartialEq)]
pub struct GroundMap {
    /// Grid x and z (m) of the corner of texel `(0, 0)`.
    pub origin: [f64; 2],
    /// Edge of a texel (m), finite and > 0.
    pub texel_metres: f64,
    /// Texels per side, `1..=GroundMap::MAX_SIZE`.
    pub size: u32,
    /// `size * size` texels, x fastest.
    pub texels: Vec<[u8; 4]>,
}

impl GroundMap {
    /// Largest side of a map in texels.
    pub const MAX_SIZE: u32 = 4096;

    /// The texel of a place without data: neither dry nor lush, no wood, not alpine, and an
    /// alpha of 0, so the shader uses its defaults.
    pub const NO_DATA: [u8; 4] = [128, 0, 0, 0];

    /// A map that holds nothing (one texel of [`GroundMap::NO_DATA`]).
    #[must_use]
    pub fn empty() -> Self {
        Self {
            origin: [0.0, 0.0],
            texel_metres: 1.0,
            size: 1,
            texels: vec![Self::NO_DATA],
        }
    }

    /// Checks the documented ranges.
    ///
    /// # Errors
    /// [`VoxelError::InvalidGroundMap`] for a size out of range, a texel count that does not
    /// match it, or a texel size or origin that is not finite and positive.
    pub fn validate(&self) -> Result<(), VoxelError> {
        let invalid = |reason| Err(VoxelError::InvalidGroundMap { reason });
        if self.size == 0 || self.size > Self::MAX_SIZE {
            return invalid("the side must be 1 to 4096 texels");
        }
        if self.texels.len() != (self.size as usize) * (self.size as usize) {
            return invalid("the map must hold size * size texels");
        }
        if !(self.texel_metres.is_finite() && self.texel_metres > 0.0) {
            return invalid("the texel edge must be finite and positive");
        }
        if !self.origin.iter().all(|value| value.is_finite()) {
            return invalid("the origin must be finite");
        }
        Ok(())
    }

    /// The four values of the frame uniform: origin x and z (m) and `1 / (size · texel)`
    /// (the scale from metres to texture coordinates), and the side in texels.
    pub(crate) fn uniform(&self) -> [f32; 4] {
        let extent = f64::from(self.size) * self.texel_metres;
        [
            self.origin[0] as f32,
            self.origin[1] as f32,
            (1.0 / extent) as f32,
            self.size as f32,
        ]
    }
}

/// The ground map on the GPU: an `Rgba8Unorm` texture of one mip level and its view.
#[derive(Debug)]
pub(crate) struct GroundTexture {
    pub(crate) view: wgpu::TextureView,
    pub(crate) uniform: [f32; 4],
}

/// Format of the ground map texture.
pub(crate) const GROUND_MAP_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;

impl GroundTexture {
    /// Creates the texture of `map` (already validated) and uploads it.
    pub(crate) fn new(device: &wgpu::Device, queue: &wgpu::Queue, map: &GroundMap) -> Self {
        let size = wgpu::Extent3d {
            width: map.size,
            height: map.size,
            depth_or_array_layers: 1,
        };
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("voxel ground map"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: GROUND_MAP_FORMAT,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            map.texels.as_flattened(),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(4 * map.size),
                rows_per_image: Some(map.size),
            },
            size,
        );
        Self {
            view: texture.create_view(&wgpu::TextureViewDescriptor::default()),
            uniform: map.uniform(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn maps_are_checked_and_scaled() {
        let empty = GroundMap::empty();
        assert_eq!(empty.validate(), Ok(()));
        assert_eq!(empty.texels, [GroundMap::NO_DATA]);
        assert_eq!(GroundMap::NO_DATA[3], 0);
        let map = GroundMap {
            origin: [-1024.0, 512.0],
            texel_metres: 16.0,
            size: 128,
            texels: vec![[1, 2, 3, 255]; 128 * 128],
        };
        assert_eq!(map.validate(), Ok(()));
        assert_eq!(map.uniform(), [-1024.0, 512.0, 1.0 / 2048.0, 128.0]);
        let bad = [
            GroundMap {
                size: 0,
                texels: Vec::new(),
                ..map.clone()
            },
            GroundMap {
                texels: vec![[0; 4]; 3],
                ..map.clone()
            },
            GroundMap {
                texel_metres: 0.0,
                ..map.clone()
            },
            GroundMap {
                origin: [f64::NAN, 0.0],
                ..map.clone()
            },
        ];
        for map in bad {
            assert!(matches!(
                map.validate(),
                Err(VoxelError::InvalidGroundMap { .. })
            ));
        }
    }
}
