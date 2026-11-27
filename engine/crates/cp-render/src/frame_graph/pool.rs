//! The pool of transient textures: kept from frame to frame, re-created when a size changes.

use crate::error::RenderError;
use crate::gpu::GpuContext;
use crate::scope::capture;

/// Everything that decides whether a pooled texture can serve a transient.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct TextureKey {
    pub(crate) format: wgpu::TextureFormat,
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) mip_level_count: u32,
    pub(crate) usage: wgpu::TextureUsages,
}

impl TextureKey {
    /// Estimated bytes of the texture with its mip levels (no alignment or padding).
    fn bytes(&self) -> u64 {
        let texel = self
            .format
            .block_copy_size(None)
            .or_else(|| {
                self.format
                    .block_copy_size(Some(wgpu::TextureAspect::DepthOnly))
            })
            .map_or(4, u64::from);
        (0..self.mip_level_count)
            .map(|level| {
                let (width, height) = crate::texture::mip_size(self.width, self.height, level);
                u64::from(width) * u64::from(height) * texel
            })
            .sum()
    }
}

/// A texture and the default view over it (all mip levels).
#[derive(Clone, Debug)]
pub(crate) struct PooledTexture {
    pub(crate) texture: wgpu::Texture,
    pub(crate) view: wgpu::TextureView,
}

#[derive(Debug)]
struct Entry {
    key: TextureKey,
    texture: PooledTexture,
    /// Frame that last took the texture; a texture serves one transient per frame.
    last_used: u64,
}

/// What the pool holds, for reports and tests.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PoolStats {
    /// Textures alive in the pool.
    pub textures: usize,
    /// Textures created since the pool was made; constant from frame to frame while the
    /// output size stays the same.
    pub created_total: u64,
    /// Estimated bytes of the textures alive, mip levels included.
    pub bytes: u64,
}

/// Transient textures by [`TextureKey`].
#[derive(Debug, Default)]
pub(crate) struct TransientPool {
    entries: Vec<Entry>,
    created: u64,
}

impl TransientPool {
    /// A texture for `key` for `frame`: one from an earlier frame when the pool has it,
    /// otherwise a new one labelled `label`.
    pub(crate) fn acquire(
        &mut self,
        gpu: &GpuContext,
        key: &TextureKey,
        label: &str,
        frame: u64,
    ) -> Result<PooledTexture, RenderError> {
        if let Some(entry) = self
            .entries
            .iter_mut()
            .find(|entry| entry.key == *key && entry.last_used != frame)
        {
            entry.last_used = frame;
            return Ok(entry.texture.clone());
        }
        let texture = capture(gpu, &format!("transient texture {label}"), |device| {
            device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d {
                    width: key.width,
                    height: key.height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: key.mip_level_count,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: key.format,
                usage: key.usage,
                view_formats: &[],
            })
        })?;
        let view = texture.create_view(&wgpu::TextureViewDescriptor {
            label: Some(label),
            ..Default::default()
        });
        let pooled = PooledTexture { texture, view };
        self.created += 1;
        self.entries.push(Entry {
            key: *key,
            texture: pooled.clone(),
            last_used: frame,
        });
        Ok(pooled)
    }

    /// Drops every texture `frame` did not take: those of an output size or a declaration
    /// that is gone. wgpu keeps a texture alive while submitted work still uses it.
    pub(crate) fn evict_unused(&mut self, frame: u64) {
        self.entries.retain(|entry| entry.last_used == frame);
    }

    pub(crate) fn stats(&self) -> PoolStats {
        PoolStats {
            textures: self.entries.len(),
            created_total: self.created,
            bytes: self.entries.iter().map(|entry| entry.key.bytes()).sum(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn byte_estimates_count_every_mip_level() {
        let key = TextureKey {
            format: wgpu::TextureFormat::Rgba16Float,
            width: 4,
            height: 4,
            mip_level_count: 3,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        };
        assert_eq!(key.bytes(), (16 + 4 + 1) * 8);
        let depth = TextureKey {
            format: wgpu::TextureFormat::Depth32Float,
            mip_level_count: 1,
            ..key
        };
        assert_eq!(depth.bytes(), 16 * 4);
        let combined = TextureKey {
            format: wgpu::TextureFormat::Depth24PlusStencil8,
            ..depth
        };
        // No exact size is known; the estimate assumes 4 bytes.
        assert_eq!(combined.bytes(), 16 * 4);
    }
}
