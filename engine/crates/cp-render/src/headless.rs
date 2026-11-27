//! Rendering without a window: an offscreen colour target and its read-back (contract §5.3
//! "Infrastructure", §5.4 `cpe shot`).
//!
//! [`HeadlessTarget`] is a 4-byte RGBA8 or BGRA8 texture of a given size plus a read-back
//! buffer whose rows are padded to the 256 bytes buffer copies require. [`RgbaImage`] is what
//! comes back: tightly packed RGBA8 rows from top to bottom, ready for a PNG encoder (cp-app
//! owns the `png` dependency): colour type RGBA, 8 bits, and the sRGB chunk when
//! [`RgbaImage::srgb`] is set.

use std::sync::mpsc;

use thiserror::Error;

use crate::error::RenderError;
use crate::gpu::GpuContext;
use crate::scope::capture;

/// The formats a headless target can have.
pub const HEADLESS_FORMATS: [wgpu::TextureFormat; 4] = [
    wgpu::TextureFormat::Rgba8Unorm,
    wgpu::TextureFormat::Rgba8UnormSrgb,
    wgpu::TextureFormat::Bgra8Unorm,
    wgpu::TextureFormat::Bgra8UnormSrgb,
];

/// Why a headless target cannot be made as asked.
#[derive(Clone, Copy, Debug, Error, PartialEq, Eq)]
pub enum HeadlessError {
    /// Only the 4-byte formats of [`HEADLESS_FORMATS`] can be read back as RGBA8.
    #[error("a headless target must be RGBA8 or BGRA8 (unorm or sRGB), not {format:?}")]
    UnsupportedFormat {
        /// The format asked for.
        format: wgpu::TextureFormat,
    },
    /// A side is zero or larger than the device allows.
    #[error("a headless target of {width}x{height} is outside 1..={limit} texels per side")]
    InvalidSize {
        /// Width asked for.
        width: u32,
        /// Height asked for.
        height: u32,
        /// `max_texture_dimension_2d` of the device.
        limit: u32,
    },
}

/// An image read back from the GPU: `width * height` RGBA8 pixels, rows top to bottom,
/// without padding.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RgbaImage {
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
    /// `true` when the values are sRGB-encoded (the target had an sRGB format); `false` when
    /// they are what the shaders wrote, unconverted.
    pub srgb: bool,
    /// `4 * width * height` bytes: R, G, B, A per pixel.
    pub pixels: Vec<u8>,
}

impl RgbaImage {
    /// The pixel at column `x`, row `y` (row 0 is the top).
    ///
    /// # Panics
    /// When `x` or `y` is outside the image.
    pub fn pixel(&self, x: u32, y: u32) -> [u8; 4] {
        assert!(
            x < self.width && y < self.height,
            "pixel ({x}, {y}) outside the image"
        );
        let start = ((y as usize * self.width as usize) + x as usize) * 4;
        let mut pixel = [0; 4];
        pixel.copy_from_slice(&self.pixels[start..start + 4]);
        pixel
    }

    /// The bytes of row `y` (row 0 is the top).
    ///
    /// # Panics
    /// When `y` is outside the image.
    pub fn row(&self, y: u32) -> &[u8] {
        assert!(y < self.height, "row {y} outside the image");
        let stride = self.width as usize * 4;
        &self.pixels[y as usize * stride..(y as usize + 1) * stride]
    }
}

/// Bytes per row of a buffer copy of `width` 4-byte texels: padded to
/// `COPY_BYTES_PER_ROW_ALIGNMENT` (256). `width` is a texture width, far below the 2^30 at
/// which the product would overflow.
pub fn padded_bytes_per_row(width: u32) -> u32 {
    (width * 4).next_multiple_of(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT)
}

/// Removes the row padding of a read-back and swaps BGRA to RGBA when `bgra` is set.
fn unpad(padded: &[u8], width: u32, height: u32, padded_row: u32, bgra: bool) -> Vec<u8> {
    let row = width as usize * 4;
    let mut pixels = Vec::with_capacity(row * height as usize);
    for source in padded.chunks(padded_row as usize).take(height as usize) {
        pixels.extend_from_slice(&source[..row]);
    }
    if bgra {
        for pixel in pixels.as_chunks_mut::<4>().0 {
            pixel.swap(0, 2);
        }
    }
    pixels
}

/// An offscreen colour target with a read-back buffer.
///
/// The texture has the usages `RENDER_ATTACHMENT`, `TEXTURE_BINDING` and `COPY_SRC`.
#[derive(Debug)]
pub struct HeadlessTarget {
    texture: wgpu::Texture,
    view: wgpu::TextureView,
    format: wgpu::TextureFormat,
    width: u32,
    height: u32,
    readback: wgpu::Buffer,
    padded_row: u32,
}

impl HeadlessTarget {
    /// A `width` x `height` target of `format` (one of [`HEADLESS_FORMATS`]).
    ///
    /// # Errors
    /// [`HeadlessError`] for another format or a size outside the device limits,
    /// [`RenderError::Wgpu`] when the device cannot create the resources.
    pub fn new(
        gpu: &GpuContext,
        width: u32,
        height: u32,
        format: wgpu::TextureFormat,
    ) -> Result<Self, RenderError> {
        if !HEADLESS_FORMATS.contains(&format) {
            return Err(HeadlessError::UnsupportedFormat { format }.into());
        }
        let limit = gpu.caps.enabled_limits.max_texture_dimension_2d;
        if width == 0 || height == 0 || width > limit || height > limit {
            return Err(HeadlessError::InvalidSize {
                width,
                height,
                limit,
            }
            .into());
        }
        let padded_row = padded_bytes_per_row(width);
        let (texture, readback) = capture(gpu, "headless target", |device| {
            let texture = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("headless target"),
                size: wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::COPY_SRC,
                view_formats: &[],
            });
            let readback = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("headless read-back"),
                size: u64::from(padded_row) * u64::from(height),
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            (texture, readback)
        })?;
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        Ok(Self {
            texture,
            view,
            format,
            width,
            height,
            readback,
            padded_row,
        })
    }

    /// The colour texture.
    pub fn texture(&self) -> &wgpu::Texture {
        &self.texture
    }

    /// A view of the whole texture, for a colour attachment.
    pub fn view(&self) -> &wgpu::TextureView {
        &self.view
    }

    /// The texture format.
    pub fn format(&self) -> wgpu::TextureFormat {
        self.format
    }

    /// Width and height in pixels.
    pub fn size(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    /// Records the copy of the whole texture into the read-back buffer; submit it before
    /// [`Self::read_pixels`].
    pub fn encode_readback(&self, encoder: &mut wgpu::CommandEncoder) {
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &self.texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &self.readback,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(self.padded_row),
                    rows_per_image: Some(self.height),
                },
            },
            wgpu::Extent3d {
                width: self.width,
                height: self.height,
                depth_or_array_layers: 1,
            },
        );
    }

    /// Waits until the GPU has finished everything submitted, maps the read-back buffer and
    /// returns the image it holds. Blocks the calling thread.
    ///
    /// # Errors
    /// [`RenderError::Device`] when the device is lost or the wait fails,
    /// [`RenderError::Readback`] when the buffer cannot be mapped.
    pub fn read_pixels(&self, gpu: &GpuContext) -> Result<RgbaImage, RenderError> {
        let (sender, receiver) = mpsc::channel();
        self.readback
            .map_async(wgpu::MapMode::Read, .., move |result| {
                // The receiver outlives the wait below; a send can only fail after it gave up.
                let _ = sender.send(result);
            });
        gpu.poll(wgpu::PollType::wait_indefinitely())?;
        let failed = |reason: String| RenderError::Readback {
            what: "headless target".to_owned(),
            reason,
        };
        match receiver.try_recv() {
            Ok(Ok(())) => {}
            Ok(Err(error)) => return Err(failed(error.to_string())),
            Err(_) => return Err(failed("the mapping did not complete".to_owned())),
        }
        let pixels = {
            let mapped = self
                .readback
                .get_mapped_range(..)
                .map_err(|error| failed(error.to_string()))?;
            let bgra = matches!(
                self.format,
                wgpu::TextureFormat::Bgra8Unorm | wgpu::TextureFormat::Bgra8UnormSrgb
            );
            unpad(&mapped, self.width, self.height, self.padded_row, bgra)
        };
        self.readback.unmap();
        Ok(RgbaImage {
            width: self.width,
            height: self.height,
            srgb: self.format.is_srgb(),
            pixels,
        })
    }

    /// Copies the texture as it is after everything submitted so far, waits and returns it:
    /// [`Self::encode_readback`], a submission and [`Self::read_pixels`] in one call.
    ///
    /// # Errors
    /// The errors of [`Self::read_pixels`], and [`RenderError::Wgpu`] when the copy is
    /// rejected.
    pub fn capture_image(&self, gpu: &GpuContext) -> Result<RgbaImage, RenderError> {
        capture(gpu, "headless read-back copy", |device| {
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("headless read-back"),
            });
            self.encode_readback(&mut encoder);
            gpu.queue.submit([encoder.finish()]);
        })?;
        self.read_pixels(gpu)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rows_are_padded_to_256_bytes() {
        assert_eq!(padded_bytes_per_row(1), 256);
        assert_eq!(padded_bytes_per_row(64), 256);
        assert_eq!(padded_bytes_per_row(65), 512);
        assert_eq!(padded_bytes_per_row(1920), 7680);
        assert_eq!(padded_bytes_per_row(1921), 7936);
    }

    #[test]
    fn unpadding_keeps_rows_in_order_and_swizzles_bgra() {
        // 3x2 pixels, rows padded to 16 bytes; padding bytes are 0xEE.
        let width = 3;
        let padded_row = 16;
        let mut padded = Vec::new();
        for y in 0..2_u8 {
            for x in 0..3_u8 {
                padded.extend_from_slice(&[10 * y + x, 100, 200, 255]);
            }
            padded.extend_from_slice(&[0xEE; 4]);
        }
        let rgba = unpad(&padded, width, 2, padded_row, false);
        assert_eq!(rgba.len(), 24);
        assert_eq!(&rgba[..4], &[0, 100, 200, 255]);
        assert_eq!(&rgba[12..16], &[10, 100, 200, 255]);
        assert!(!rgba.contains(&0xEE));
        let swapped = unpad(&padded, width, 2, padded_row, true);
        assert_eq!(&swapped[..4], &[200, 100, 0, 255]);
        assert_eq!(&swapped[20..24], &[200, 100, 12, 255]);

        let image = RgbaImage {
            width: 3,
            height: 2,
            srgb: false,
            pixels: rgba,
        };
        assert_eq!(image.pixel(2, 1), [12, 100, 200, 255]);
        assert_eq!(image.row(1).len(), 12);
        assert_eq!(image.row(1)[0], 10);
    }

    #[test]
    #[should_panic(expected = "outside the image")]
    fn pixels_outside_the_image_panic() {
        let image = RgbaImage {
            width: 1,
            height: 1,
            srgb: true,
            pixels: vec![0; 4],
        };
        let _ = image.pixel(1, 0);
    }
}
