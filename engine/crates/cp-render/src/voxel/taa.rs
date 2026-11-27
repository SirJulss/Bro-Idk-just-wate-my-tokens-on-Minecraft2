//! Temporal anti-aliasing (contract §7 "Post", §11): the jitter sequence, the parameters of the
//! resolve pass and the two history textures it alternates between.
//!
//! Every frame offsets the projection by a sub-pixel amount from a Halton (2, 3) sequence of
//! [`JITTER_PHASES`] phases; the `taa` pass blends the frame into the history reprojected from
//! the last frame (static geometry: from depth and the two cameras) and clips the history to
//! the current neighbourhood. The history is `Rgba16Float` at the output size (contract §7
//! formats) and lives outside the frame graph's transient pool, because it must survive from
//! one frame to the next.

use bytemuck::{Pod, Zeroable};
use cp_core::math::{DMat4, DVec3};

use super::camera::FrameCamera;
use crate::error::RenderError;
use crate::gpu::GpuContext;
use crate::scope::capture;

/// Format of the history (contract §7: "HDR scene colour and TAA history `Rgba16Float`").
pub const HISTORY_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

/// Phases of the jitter sequence before it repeats.
pub const JITTER_PHASES: u64 = 16;

/// The least weight of the current frame in the blend: the history holds about ten frames.
pub const MIN_BLEND: f32 = 0.08;

/// The least weight of the current frame for a pixel that stands still (moves less than a
/// twentieth of a pixel per frame): the history then averages two full cycles of the
/// [`JITTER_PHASES`] phases, so that per-frame patterns (shadow and obscurance offsets, the
/// jitter itself) average out instead of leaving a fixed grain of the phases the shorter
/// history weights most.
pub const STILL_BLEND: f32 = 1.0 / (2 * JITTER_PHASES) as f32;

const _: () = assert!(STILL_BLEND < MIN_BLEND);

/// A camera that moves further than this between two frames (metres) cuts: the history is
/// dropped instead of smeared across the jump.
pub const CUT_DISTANCE: f64 = 50.0;

/// A camera that turns by more than this between two frames (radians) cuts as well.
pub const CUT_ANGLE: f64 = 0.5;

/// Element `index` (from 1) of the van der Corput sequence in `base`, in `0..1`.
fn radical_inverse(mut index: u64, base: u64) -> f64 {
    let mut result = 0.0;
    let mut fraction = 1.0 / base as f64;
    while index > 0 {
        result += (index % base) as f64 * fraction;
        index /= base;
        fraction /= base as f64;
    }
    result
}

/// The sub-pixel offset of frame `frame` in pixels, each axis in `-0.5..0.5`: Halton (2, 3),
/// [`JITTER_PHASES`] phases, centred on zero.
#[must_use]
pub fn jitter_pixels(frame: u64) -> [f32; 2] {
    let index = frame % JITTER_PHASES + 1;
    [
        (radical_inverse(index, 2) - 0.5) as f32,
        (radical_inverse(index, 3) - 0.5) as f32,
    ]
}

/// The offset of [`jitter_pixels`] in normalised device coordinates for an output of `size`
/// pixels (x right, y up; a pixel offset downwards is a negative NDC offset).
#[must_use]
pub fn jitter_ndc(frame: u64, size: (u32, u32)) -> [f32; 2] {
    let [x, y] = jitter_pixels(frame);
    [2.0 * x / size.0 as f32, -2.0 * y / size.1 as f32]
}

/// The parameters of the resolve pass, WGSL `Taa` in `voxel/taa.wgsl` (144 bytes).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
pub struct GpuTaa {
    /// Clip space without this frame's jitter → render space.
    pub clip_to_render: [[f32; 4]; 4],
    /// Render space of this frame → clip space of the last frame, without jitter.
    pub render_to_prev_clip: [[f32; 4]; 4],
    /// The least weight of the current frame, 1 when the history is valid, frames in the
    /// history, the least weight of a pixel that stands still ([`STILL_BLEND`]).
    pub params: [f32; 4],
}

const _: () = assert!(size_of::<GpuTaa>() == 144);

impl GpuTaa {
    /// The parameters for `now` after `prev`; `frames` is the number of frames in the valid
    /// history, `None` when there is none. Composed in `f64`: render space is
    /// camera-relative, so the last camera's render space is this one's shifted by the
    /// camera's move.
    pub(crate) fn new(now: &FrameCamera, prev: &FrameCamera, frames: Option<u32>) -> Self {
        let render_to_clip = now.camera.render_to_clip(now.aspect);
        let shift: DVec3 = now.camera.position - prev.camera.position;
        let render_to_prev_clip =
            prev.camera.render_to_clip(prev.aspect) * DMat4::from_translation(shift);
        Self {
            clip_to_render: render_to_clip.inverse().as_mat4().to_cols_array_2d(),
            render_to_prev_clip: render_to_prev_clip.as_mat4().to_cols_array_2d(),
            params: [
                MIN_BLEND,
                if frames.is_some() { 1.0 } else { 0.0 },
                frames.unwrap_or(0) as f32,
                STILL_BLEND,
            ],
        }
    }
}

/// Whether the camera jumped from `prev` to `now` (a cut): the history is dropped then.
pub(crate) fn is_cut(now: &FrameCamera, prev: &FrameCamera) -> bool {
    let moved = now.camera.position.distance(prev.camera.position);
    let turned = now
        .camera
        .orientation
        .angle_between(prev.camera.orientation);
    !(moved <= CUT_DISTANCE && turned <= CUT_ANGLE)
        || now.aspect != prev.aspect
        || now.camera.fov_y != prev.camera.fov_y
}

/// The two history textures of one output size and the views over them.
#[derive(Debug)]
pub(crate) struct History {
    pub(crate) size: (u32, u32),
    pub(crate) views: [wgpu::TextureView; 2],
    /// Which texture the next frame writes; it reads the other.
    pub(crate) next: usize,
    /// Frames accumulated in the texture the next frame reads; `None` when it holds nothing
    /// usable.
    pub(crate) frames: Option<u32>,
}

impl History {
    /// Two cleared history textures of `size` pixels.
    pub(crate) fn new(gpu: &GpuContext, size: (u32, u32)) -> Result<Self, RenderError> {
        let views = capture(gpu, "taa history", |device| {
            [0, 1].map(|index| {
                device
                    .create_texture(&wgpu::TextureDescriptor {
                        label: Some(if index == 0 {
                            "voxel.taa_history_a"
                        } else {
                            "voxel.taa_history_b"
                        }),
                        size: wgpu::Extent3d {
                            width: size.0,
                            height: size.1,
                            depth_or_array_layers: 1,
                        },
                        mip_level_count: 1,
                        sample_count: 1,
                        dimension: wgpu::TextureDimension::D2,
                        format: HISTORY_FORMAT,
                        usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                            | wgpu::TextureUsages::TEXTURE_BINDING,
                        view_formats: &[],
                    })
                    .create_view(&wgpu::TextureViewDescriptor::default())
            })
        })?;
        Ok(Self {
            size,
            views,
            next: 0,
            frames: None,
        })
    }

    /// Forgets what the history holds; the next frame starts it anew.
    pub(crate) fn reset(&mut self) {
        self.frames = None;
    }

    /// After a frame wrote texture `next`: it becomes the one the following frame reads.
    pub(crate) fn advance(&mut self) {
        self.frames = Some(self.frames.map_or(1, |frames| frames.saturating_add(1)));
        self.next = 1 - self.next;
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use cp_core::math::DVec4;
    use wgpu::naga;

    use super::*;
    use crate::shader::{ShaderFile, ShaderLibrary, ShaderTarget};
    use crate::voxel::Camera;

    #[test]
    fn the_jitter_covers_the_pixel_evenly_and_repeats() {
        let offsets: Vec<[f32; 2]> = (0..JITTER_PHASES).map(jitter_pixels).collect();
        // Distinct, inside the pixel, centred.
        let distinct: BTreeSet<(i32, i32)> = offsets
            .iter()
            .map(|o| ((o[0] * 1e4) as i32, (o[1] * 1e4) as i32))
            .collect();
        assert_eq!(distinct.len(), JITTER_PHASES as usize);
        let mean = offsets
            .iter()
            .fold([0.0_f32; 2], |sum, o| [sum[0] + o[0], sum[1] + o[1]]);
        for axis in 0..2 {
            assert!(offsets.iter().all(|o| o[axis].abs() < 0.5));
            assert!((mean[axis] / JITTER_PHASES as f32).abs() < 0.05, "{mean:?}");
        }
        // Every quarter of the pixel gets a quarter of the samples.
        for quadrant in 0..4 {
            let count = offsets
                .iter()
                .filter(|o| ((o[0] >= 0.0) as usize) + 2 * ((o[1] >= 0.0) as usize) == quadrant)
                .count();
            assert!((3..=5).contains(&count), "{quadrant}: {count}");
        }
        assert_eq!(jitter_pixels(3), jitter_pixels(3 + JITTER_PHASES));
        // In NDC: two units across the image, y up.
        let ndc = jitter_ndc(5, (200, 100));
        let pixels = jitter_pixels(5);
        assert!((ndc[0] - pixels[0] / 100.0).abs() < 1e-7);
        assert!((ndc[1] + pixels[1] / 50.0).abs() < 1e-7);
    }

    fn frame_camera(position: DVec3, target: DVec3) -> FrameCamera {
        FrameCamera {
            camera: Camera::look_at(position, target, DVec3::Y, 1.0),
            aspect: 1.5,
            jitter: [0.0, 0.0],
        }
    }

    #[test]
    fn reprojection_finds_where_a_point_was_in_the_last_frame() {
        let prev = frame_camera(DVec3::new(100.0, 20.0, 50.0), DVec3::new(0.0, 0.0, 0.0));
        let now = FrameCamera {
            jitter: [0.001, 0.002],
            ..frame_camera(DVec3::new(100.6, 20.1, 49.5), DVec3::new(0.0, 1.0, 0.0))
        };
        let taa = GpuTaa::new(&now, &prev, Some(3));
        assert_eq!(taa.params, [MIN_BLEND, 1.0, 3.0, STILL_BLEND]);
        // A still pixel averages two full cycles of the jitter (and moving ones less).
        assert_eq!(STILL_BLEND * (2 * JITTER_PHASES) as f32, 1.0);
        let clip_to_render = cp_core::math::Mat4::from_cols_array_2d(&taa.clip_to_render);
        let to_prev = cp_core::math::Mat4::from_cols_array_2d(&taa.render_to_prev_clip);
        for world in [
            DVec3::new(0.0, 0.0, 0.0),
            DVec3::new(10.0, 3.0, -4.0),
            DVec3::new(60.0, 5.0, 30.0),
        ] {
            // The point on this frame's screen, without the jitter.
            let render = world - now.camera.position;
            let clip = now.camera.render_to_clip(now.aspect) * render.extend(1.0);
            let ndc = (clip.truncate() / clip.w).as_vec3();
            let back = clip_to_render * ndc.extend(1.0);
            let prev_clip = to_prev * (back / back.w);
            let found = prev_clip.truncate() / prev_clip.w;
            let expected: DVec4 = prev.camera.frame_to_clip(prev.aspect) * world.extend(1.0);
            let expected = expected.truncate() / expected.w;
            assert!(
                (found.as_dvec3().truncate() - expected.truncate())
                    .abs()
                    .max_element()
                    < 1e-4,
                "{found} {expected}"
            );
        }
        let none = GpuTaa::new(&now, &prev, None);
        assert_eq!(none.params[1], 0.0);
    }

    #[test]
    fn cuts_are_jumps_turns_and_new_lenses() {
        let a = frame_camera(DVec3::new(0.0, 10.0, 0.0), DVec3::new(0.0, 10.0, -10.0));
        let near = frame_camera(DVec3::new(1.0, 10.0, 0.0), DVec3::new(1.0, 10.0, -10.0));
        let far = frame_camera(DVec3::new(80.0, 10.0, 0.0), DVec3::new(80.0, 10.0, -10.0));
        let turned = frame_camera(DVec3::new(0.0, 10.0, 0.0), DVec3::new(10.0, 10.0, 0.0));
        assert!(!is_cut(&near, &a));
        assert!(is_cut(&far, &a));
        assert!(is_cut(&turned, &a));
        let wider = FrameCamera {
            aspect: 2.0,
            ..near
        };
        assert!(is_cut(&wider, &a));
    }

    #[test]
    fn the_wgsl_struct_has_the_rust_layout() {
        use std::mem::offset_of;
        let files: &'static [ShaderFile] = Vec::leak(
            ShaderLibrary::builtin()
                .files()
                .iter()
                .copied()
                .chain([ShaderFile::new(
                    "taa_probe.wgsl",
                    "#include \"voxel/taa.wgsl\"\n",
                )])
                .collect(),
        );
        let validated = ShaderLibrary::new(files)
            .validate("taa_probe.wgsl", &ShaderTarget::baseline())
            .unwrap_or_else(|error| panic!("{error}"));
        let (_, ty) = validated
            .module
            .types
            .iter()
            .find(|(_, ty)| ty.name.as_deref() == Some("Taa"))
            .expect("struct Taa");
        let naga::TypeInner::Struct { members, span } = &ty.inner else {
            panic!("Taa is not a struct");
        };
        let offsets: Vec<(String, u32)> = members
            .iter()
            .map(|member| (member.name.clone().unwrap(), member.offset))
            .collect();
        let rust: Vec<(String, u32)> = [
            ("clip_to_render", offset_of!(GpuTaa, clip_to_render)),
            (
                "render_to_prev_clip",
                offset_of!(GpuTaa, render_to_prev_clip),
            ),
            ("params", offset_of!(GpuTaa, params)),
        ]
        .iter()
        .map(|&(name, offset)| (name.to_owned(), u32::try_from(offset).unwrap()))
        .collect();
        assert_eq!(offsets, rust);
        assert_eq!(*span, 144);
    }
}
