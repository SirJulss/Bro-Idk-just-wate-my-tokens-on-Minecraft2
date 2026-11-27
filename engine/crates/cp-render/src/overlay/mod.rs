//! Debug overlay: rectangles and bitmap text drawn over the frame (contract §5.3).
//!
//! An [`Overlay`] collects what to draw in target pixels (origin top left, y down): solid
//! rectangles and text in the engine's own [`font`]. An [`OverlayRenderer`] uploads it and
//! draws it in one instanced draw into a render pass. Glyphs are scaled by whole pixels and
//! never filtered, so text stays sharp and the output is exact to the pixel.
//!
//! ```no_run
//! use cp_render::gpu::{GpuContext, GpuOptions};
//! use cp_render::headless::HeadlessTarget;
//! use cp_render::overlay::{Overlay, OverlayColor, OverlayRenderer};
//! use cp_render::wgpu;
//!
//! let gpu = GpuContext::new_blocking(&GpuOptions::default(), None)?;
//! let target = HeadlessTarget::new(&gpu, 320, 180, wgpu::TextureFormat::Rgba8UnormSrgb)?;
//! let mut renderer = OverlayRenderer::new(&gpu, target.format())?;
//! let mut overlay = Overlay::new();
//! let lines = ["frame 1.92 ms", "gbuffer 0.81 ms"];
//! overlay.text_panel(8, 8, 2, &lines, OverlayColor::WHITE, OverlayColor::PANEL);
//! renderer.prepare(&gpu, &overlay, target.size())?;
//! let mut encoder =
//!     gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
//! {
//!     let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
//!         color_attachments: &[Some(wgpu::RenderPassColorAttachment {
//!             view: target.view(),
//!             depth_slice: None,
//!             resolve_target: None,
//!             ops: wgpu::Operations {
//!                 load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
//!                 store: wgpu::StoreOp::Store,
//!             },
//!         })],
//!         ..Default::default()
//!     });
//!     renderer.draw(&mut pass);
//! }
//! gpu.queue.submit([encoder.finish()]);
//! let image = target.capture_image(&gpu)?;
//! # let _ = image;
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

pub mod font;

use bytemuck::{Pod, Zeroable};

use crate::error::RenderError;
use crate::gpu::GpuContext;
use crate::scope::capture;
use crate::shader::ShaderLibrary;
use font::{GLYPH_ADVANCE, GLYPH_HEIGHT, GLYPH_WIDTH, LINE_HEIGHT};

/// An sRGB-encoded 8-bit colour with straight (not premultiplied) alpha: what an image editor
/// shows. The overlay converts it for sRGB targets so that the stored bytes equal these.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Pod, Zeroable)]
#[repr(C)]
pub struct OverlayColor(pub [u8; 4]);

impl OverlayColor {
    /// Opaque white.
    pub const WHITE: Self = Self([255, 255, 255, 255]);
    /// Opaque black.
    pub const BLACK: Self = Self([0, 0, 0, 255]);
    /// The yellow of warnings.
    pub const YELLOW: Self = Self([255, 214, 64, 255]);
    /// The red of errors.
    pub const RED: Self = Self([255, 80, 72, 255]);
    /// The translucent dark background of text panels.
    pub const PANEL: Self = Self([16, 18, 24, 192]);

    /// An opaque colour.
    pub const fn rgb(red: u8, green: u8, blue: u8) -> Self {
        Self([red, green, blue, 255])
    }
}

/// [`OverlayInstance::glyph`] of a solid rectangle.
pub const SOLID: u32 = u32::MAX;

/// One rectangle as the overlay shader reads it from vertex buffer 0 (32 bytes per instance).
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
#[repr(C)]
pub struct OverlayInstance {
    /// `[x0, y0, x1, y1]` in target pixels, whole numbers, `x0 <= x1`, `y0 <= y1`.
    pub rect: [f32; 4],
    /// Colour of the rectangle or of the glyph's ink.
    pub color: OverlayColor,
    /// Glyph index into the font atlas, or [`SOLID`].
    pub glyph: u32,
    /// Size of a glyph pixel in target pixels, at least 1; unused by solid rectangles.
    pub scale: u32,
    /// Keeps the stride at 32 bytes.
    pub padding: u32,
}

/// Width and height of a piece of text in target pixels.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TextExtent {
    /// Width of the longest line; the spacing after its last character is not included.
    pub width: u32,
    /// Height of all lines; the spacing after the last line is not included.
    pub height: u32,
}

/// Rectangles and text to draw over a frame, in the order they were added (later ones on top).
#[derive(Clone, Debug, Default)]
pub struct Overlay {
    instances: Vec<OverlayInstance>,
}

impl Overlay {
    /// An empty overlay.
    pub fn new() -> Self {
        Self::default()
    }

    /// Removes everything, keeping the allocation for the next frame.
    pub fn clear(&mut self) {
        self.instances.clear();
    }

    /// Number of instances (one per rectangle and per visible glyph).
    pub fn len(&self) -> usize {
        self.instances.len()
    }

    /// `true` when nothing is to be drawn.
    pub fn is_empty(&self) -> bool {
        self.instances.is_empty()
    }

    /// The instances as uploaded to the GPU.
    pub fn instances(&self) -> &[OverlayInstance] {
        &self.instances
    }

    /// A solid rectangle with its top left corner at `(x, y)`, in target pixels.
    pub fn rect(&mut self, x: i32, y: i32, width: u32, height: u32, color: OverlayColor) {
        if width == 0 || height == 0 {
            return;
        }
        self.instances.push(OverlayInstance {
            rect: rect(x, y, width, height),
            color,
            glyph: SOLID,
            scale: 1,
            padding: 0,
        });
    }

    /// Text with the top left corner of its first glyph box at `(x, y)`; every font pixel
    /// becomes `scale` x `scale` target pixels (0 counts as 1). `'\n'` starts a new line.
    /// Returns the extent of the text.
    pub fn text(
        &mut self,
        x: i32,
        y: i32,
        scale: u32,
        color: OverlayColor,
        text: &str,
    ) -> TextExtent {
        let scale = scale.max(1);
        for (row, line) in (0_i64..).zip(text.split('\n')) {
            let top = i64::from(y) + row * i64::from(LINE_HEIGHT * scale);
            for (column, character) in (0_i64..).zip(line.chars()) {
                if character == ' ' {
                    continue;
                }
                let left = i64::from(x) + column * i64::from(GLYPH_ADVANCE * scale);
                let (Ok(left), Ok(top)) = (i32::try_from(left), i32::try_from(top)) else {
                    continue;
                };
                self.instances.push(OverlayInstance {
                    rect: rect(left, top, GLYPH_WIDTH * scale, GLYPH_HEIGHT * scale),
                    color,
                    glyph: font::glyph_index(character),
                    scale,
                    padding: 0,
                });
            }
        }
        Self::measure(text, scale)
    }

    /// A panel for HUD lines: a `background` rectangle with a margin of two font pixels around
    /// `lines`, drawn in `color`. Returns the extent of the panel.
    pub fn text_panel(
        &mut self,
        x: i32,
        y: i32,
        scale: u32,
        lines: &[&str],
        color: OverlayColor,
        background: OverlayColor,
    ) -> TextExtent {
        let scale = scale.max(1);
        let margin = 2 * scale;
        let text = lines.join("\n");
        let inner = Self::measure(&text, scale);
        let extent = TextExtent {
            width: inner.width + 2 * margin,
            height: inner.height + 2 * margin,
        };
        self.rect(x, y, extent.width, extent.height, background);
        let inset = i32::try_from(margin).unwrap_or(i32::MAX);
        self.text(
            x.saturating_add(inset),
            y.saturating_add(inset),
            scale,
            color,
            &text,
        );
        extent
    }

    /// The extent [`Self::text`] gives `text` at `scale` (0 counts as 1).
    pub fn measure(text: &str, scale: u32) -> TextExtent {
        let scale = scale.max(1);
        let lines = text.split('\n').count() as u32;
        let longest = text
            .split('\n')
            .map(|line| line.chars().count() as u32)
            .max()
            .unwrap_or(0);
        TextExtent {
            width: if longest == 0 {
                0
            } else {
                ((longest - 1) * GLYPH_ADVANCE + GLYPH_WIDTH) * scale
            },
            height: ((lines - 1) * LINE_HEIGHT + GLYPH_HEIGHT) * scale,
        }
    }
}

/// The rect of an instance: whole target pixels as `f32` (exact below 2^24).
fn rect(x: i32, y: i32, width: u32, height: u32) -> [f32; 4] {
    let (x, y) = (f64::from(x), f64::from(y));
    [
        x as f32,
        y as f32,
        (x + f64::from(width)) as f32,
        (y + f64::from(height)) as f32,
    ]
}

/// Instances the first instance buffer has room for.
const INITIAL_CAPACITY: usize = 256;

/// The GPU side of the overlay for colour targets of one format.
///
/// [`Self::prepare`] uploads an [`Overlay`] and the target size, [`Self::draw`] records the
/// draw into a render pass whose only attachment is one colour target of
/// [`Self::target_format`], without depth. One draw per submission: the next `prepare`
/// overwrites the uploaded instances.
#[derive(Debug)]
pub struct OverlayRenderer {
    target_format: wgpu::TextureFormat,
    pipeline: wgpu::RenderPipeline,
    bind_group: wgpu::BindGroup,
    screen: wgpu::Buffer,
    instances: wgpu::Buffer,
    capacity: usize,
    count: u32,
}

impl OverlayRenderer {
    /// Creates the pipeline, the font atlas and the buffers for targets of `target_format`.
    ///
    /// # Errors
    /// [`RenderError::Shader`] or [`RenderError::Wgpu`] when the shader or the pipeline is
    /// rejected (for example a format that cannot be rendered to), [`RenderError::Device`] on
    /// a lost device.
    pub fn new(gpu: &GpuContext, target_format: wgpu::TextureFormat) -> Result<Self, RenderError> {
        let module = ShaderLibrary::builtin().create_module(gpu, "overlay.wgsl")?;
        let (pipeline, bind_group, screen) = capture(gpu, "overlay pipeline", |device| {
            let font = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("overlay font"),
                size: wgpu::Extent3d {
                    width: font::ATLAS_WIDTH,
                    height: font::ATLAS_HEIGHT,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::R8Unorm,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            });
            gpu.queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &font,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                &font::atlas_pixels(),
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(font::ATLAS_WIDTH),
                    rows_per_image: Some(font::ATLAS_HEIGHT),
                },
                font.size(),
            );
            let font_view = font.create_view(&wgpu::TextureViewDescriptor::default());
            let screen = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("overlay screen"),
                size: 16,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("overlay"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: wgpu::BufferSize::new(16),
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: false },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                ],
            });
            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("overlay"),
                layout: &layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: screen.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&font_view),
                    },
                ],
            });
            let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("overlay"),
                bind_group_layouts: &[Some(&layout)],
                immediate_size: 0,
            });
            let attributes = wgpu::vertex_attr_array![0 => Float32x4, 1 => Unorm8x4, 2 => Uint32x2];
            let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("overlay"),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &module,
                    entry_point: Some("vs_main"),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    buffers: &[Some(wgpu::VertexBufferLayout {
                        array_stride: size_of::<OverlayInstance>() as u64,
                        step_mode: wgpu::VertexStepMode::Instance,
                        attributes: &attributes,
                    })],
                },
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                fragment: Some(wgpu::FragmentState {
                    module: &module,
                    entry_point: Some("fs_main"),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: target_format,
                        blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                multiview_mask: None,
                cache: None,
            });
            (pipeline, bind_group, screen)
        })?;
        let instances = create_instance_buffer(gpu, INITIAL_CAPACITY)?;
        Ok(Self {
            target_format,
            pipeline,
            bind_group,
            screen,
            instances,
            capacity: INITIAL_CAPACITY,
            count: 0,
        })
    }

    /// The colour format the pipeline renders to.
    pub fn target_format(&self) -> wgpu::TextureFormat {
        self.target_format
    }

    /// Uploads `overlay` for a target of `target_size` pixels; the next [`Self::draw`] draws
    /// it. Grows the instance buffer when needed.
    ///
    /// # Errors
    /// [`RenderError::Wgpu`] when a larger instance buffer cannot be created.
    pub fn prepare(
        &mut self,
        gpu: &GpuContext,
        overlay: &Overlay,
        target_size: (u32, u32),
    ) -> Result<(), RenderError> {
        let (width, height) = target_size;
        let screen = [
            (width.max(1) as f32).to_bits(),
            (height.max(1) as f32).to_bits(),
            u32::from(self.target_format.is_srgb()),
            0,
        ];
        gpu.queue
            .write_buffer(&self.screen, 0, bytemuck::cast_slice(&screen));
        if overlay.len() > self.capacity {
            let capacity = overlay.len().next_power_of_two();
            self.instances = create_instance_buffer(gpu, capacity)?;
            self.capacity = capacity;
        }
        if !overlay.is_empty() {
            gpu.queue.write_buffer(
                &self.instances,
                0,
                bytemuck::cast_slice(overlay.instances()),
            );
        }
        self.count = u32::try_from(overlay.len()).unwrap_or(u32::MAX);
        Ok(())
    }

    /// Records the draw of the prepared overlay; nothing when it is empty.
    pub fn draw(&self, pass: &mut wgpu::RenderPass<'_>) {
        if self.count == 0 {
            return;
        }
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bind_group, &[]);
        pass.set_vertex_buffer(0, self.instances.slice(..));
        pass.draw(0..6, 0..self.count);
    }
}

fn create_instance_buffer(gpu: &GpuContext, capacity: usize) -> Result<wgpu::Buffer, RenderError> {
    capture(gpu, "overlay instances", |device| {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("overlay instances"),
            size: (capacity * size_of::<OverlayInstance>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn instances_are_32_bytes() {
        assert_eq!(size_of::<OverlayInstance>(), 32);
        assert_eq!(std::mem::offset_of!(OverlayInstance, color), 16);
        assert_eq!(std::mem::offset_of!(OverlayInstance, glyph), 20);
        assert_eq!(std::mem::offset_of!(OverlayInstance, scale), 24);
    }

    #[test]
    fn text_lays_out_glyphs_on_the_advance_and_line_grid() {
        let mut overlay = Overlay::new();
        let extent = overlay.text(10, 20, 2, OverlayColor::WHITE, "Ab c\nx");
        // The space draws nothing.
        let glyphs: Vec<(u32, [f32; 4])> = overlay
            .instances()
            .iter()
            .map(|instance| (instance.glyph, instance.rect))
            .collect();
        assert_eq!(
            glyphs,
            [
                (font::glyph_index('A'), [10.0, 20.0, 20.0, 36.0]),
                (font::glyph_index('b'), [22.0, 20.0, 32.0, 36.0]),
                (font::glyph_index('c'), [46.0, 20.0, 56.0, 36.0]),
                (font::glyph_index('x'), [10.0, 40.0, 20.0, 56.0]),
            ]
        );
        assert!(
            overlay
                .instances()
                .iter()
                .all(|instance| instance.scale == 2)
        );
        // Four characters of 6 px advance minus the last spacing, two lines of 10 px minus
        // the last spacing, all doubled.
        assert_eq!(
            extent,
            TextExtent {
                width: (3 * 6 + 5) * 2,
                height: (10 + 8) * 2
            }
        );
        assert_eq!(
            Overlay::measure("", 1),
            TextExtent {
                width: 0,
                height: 8
            }
        );
        assert_eq!(Overlay::measure("abc", 0), Overlay::measure("abc", 1));
    }

    #[test]
    fn panels_put_the_background_under_the_text() {
        let mut overlay = Overlay::new();
        let extent = overlay.text_panel(
            0,
            0,
            1,
            &["ab", "c"],
            OverlayColor::WHITE,
            OverlayColor::PANEL,
        );
        assert_eq!(
            extent,
            TextExtent {
                width: 11 + 4,
                height: 18 + 4
            }
        );
        let first = overlay.instances()[0];
        assert_eq!(first.glyph, SOLID);
        assert_eq!(first.rect, [0.0, 0.0, 15.0, 22.0]);
        assert_eq!(first.color, OverlayColor::PANEL);
        // The text starts inside the margin.
        assert_eq!(overlay.instances()[1].rect[..2], [2.0, 2.0]);
        assert_eq!(overlay.len(), 4);
        overlay.clear();
        assert!(overlay.is_empty());
        // Empty rectangles are dropped.
        overlay.rect(5, 5, 0, 9, OverlayColor::RED);
        assert!(overlay.is_empty());
    }

    #[test]
    fn unknown_characters_become_the_box_and_far_text_is_clipped_by_the_gpu() {
        let mut overlay = Overlay::new();
        overlay.text(-20, -30, 1, OverlayColor::YELLOW, "é");
        assert_eq!(overlay.instances()[0].glyph, font::UNKNOWN_GLYPH);
        assert_eq!(overlay.instances()[0].rect, [-20.0, -30.0, -15.0, -22.0]);
        // Positions beyond the i32 range are skipped instead of wrapping around.
        let mut overlay = Overlay::new();
        overlay.text(i32::MAX - 3, 0, 1, OverlayColor::WHITE, "ab");
        assert_eq!(overlay.len(), 1);
    }
}
