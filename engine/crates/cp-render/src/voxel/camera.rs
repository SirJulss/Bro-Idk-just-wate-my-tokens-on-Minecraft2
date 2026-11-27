//! Camera, grid poses, lighting parameters and the `f64` composition of the object matrices
//! (contract §2.4, §5.3 "Objects").
//!
//! Render space has its origin at the camera and the axes of the camera's reference frame
//! (contract §2.2). Absolute positions stay in `f64` until they are relative to an integer
//! anchor voxel next to the camera; only the small remainder reaches an `f32`.

use cp_core::math::{
    DMat3, DMat4, DQuat, DVec3, DVec4, IAabb, IVec3, Mat4, Vec3, Z_NEAR,
    perspective_infinite_reverse_rh,
};

use super::layout::GpuObject;

/// Margin around a grid's bounds inside which the anchor is the camera's own voxel: 1 km
/// (contract §5.3).
pub const ANCHOR_MARGIN_VOXELS: i32 = 1000;

/// A camera in the reference frame of render space.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Camera {
    /// Position in metres.
    pub position: DVec3,
    /// Rotation from view space (x right, y up, looking along -z) to the reference frame;
    /// unit length.
    pub orientation: DQuat,
    /// Full vertical field of view in radians, `0 < fov_y < π`.
    pub fov_y: f32,
}

impl Camera {
    /// A camera at `position` looking at `target`, with `up` (not parallel to the view
    /// direction) giving the screen's up side.
    #[must_use]
    pub fn look_at(position: DVec3, target: DVec3, up: DVec3, fov_y: f32) -> Self {
        let forward = (target - position).normalize();
        let right = forward.cross(up).normalize();
        let screen_up = right.cross(forward);
        let rotation = DMat3::from_cols(right, screen_up, -forward);
        Self {
            position,
            orientation: DQuat::from_mat3(&rotation).normalize(),
            fov_y,
        }
    }

    /// The projection of contract §2.4 for an output of `aspect` = width / height.
    #[must_use]
    pub fn projection(&self, aspect: f32) -> Mat4 {
        perspective_infinite_reverse_rh(self.fov_y, aspect, Z_NEAR)
    }

    /// Render space → clip space: the projection times the view rotation (the camera sits at
    /// the origin of render space).
    #[must_use]
    pub fn render_to_clip(&self, aspect: f32) -> DMat4 {
        let view = DMat4::from_quat(self.orientation.inverse());
        self.projection(aspect).as_dmat4() * view
    }

    /// Reference frame → clip space, for points given in `f64` frame coordinates (picking,
    /// tests). Precise only near the camera's position once narrowed to `f32`.
    #[must_use]
    pub fn frame_to_clip(&self, aspect: f32) -> DMat4 {
        self.render_to_clip(aspect) * DMat4::from_translation(-self.position)
    }
}

/// Where a voxel grid lies in the camera's reference frame: `p_frame = rotation · p_grid +
/// translation`, metres.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GridPose {
    /// Unit rotation from grid axes to frame axes.
    pub rotation: DQuat,
    /// Position of the grid origin in the frame.
    pub translation: DVec3,
}

impl Default for GridPose {
    fn default() -> Self {
        Self::IDENTITY
    }
}

impl GridPose {
    /// The grid's axes and origin are the frame's (a body seen from its own frame).
    pub const IDENTITY: Self = Self {
        rotation: DQuat::IDENTITY,
        translation: DVec3::ZERO,
    };

    /// A frame point in grid coordinates.
    #[must_use]
    pub fn frame_to_grid(&self, point: DVec3) -> DVec3 {
        self.rotation.inverse() * (point - self.translation)
    }
}

/// The light of an E2 frame: one sun with screen-space shadows, a hemispheric sky ambient with
/// screen-space ambient obscurance, height fog and a gradient sky with a layer of clouds
/// (the lighting pass derives the clouds from these colours). Directions are unit vectors in
/// render space; colours are linear Rec.709.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Lighting {
    /// The local up direction (away from gravity); orients the sky gradient and the ambient
    /// hemispheres.
    pub up: Vec3,
    /// Direction towards the sun.
    pub sun_direction: Vec3,
    /// Sun illuminance as a colour; a white Lambert surface facing the sun reflects
    /// `sun_color / π`.
    pub sun_color: Vec3,
    /// Angular radius of the sun disc in radians.
    pub sun_angular_radius: f32,
    /// Sky at the zenith.
    pub sky_zenith: Vec3,
    /// Sky at the horizon away from the sun; also the colour of the fog.
    pub sky_horizon: Vec3,
    /// Sky at the horizon below the sun. The horizon turns from [`Self::sky_horizon`] to this
    /// colour towards the sun's compass direction (a low sun makes it warm).
    pub sky_sunset: Vec3,
    /// Light from below the horizon: the bounce of the sun off the ground.
    pub ground: Vec3,
    /// The light of the sky on a surface facing up, before [`Self::ambient`]: mostly the sky's
    /// brightness, with as much of its colour as the light of the hour has (a blue fill at
    /// noon would turn every shadow cyan; at dusk the cool sky light against the warm sun is
    /// what separates shade from light). [`Lighting::sky_light_from`] derives it from the sky
    /// colours.
    pub sky_light: Vec3,
    /// Gain of the sky's own brightness (the sky behind the scene, its clouds, the fog colour
    /// and the sky mirrored by water) against the land's, which [`Self::exposure`] scales
    /// alike: 1 by day; below 1 at dusk, where the exposure rises for the dim land and the sky
    /// would otherwise glare above a dark landscape.
    pub sky_gain: f32,
    /// Scale of the hemispheric ambient light.
    pub ambient: f32,
    /// Fog density per metre at the fog's reference level (`fog = 1 - e^(-optical depth)`).
    pub fog_density: f32,
    /// The fog thins out with height above its reference level: density times
    /// `e^(-altitude · fog_falloff)`, per metre; 0 keeps it uniform.
    pub fog_falloff: f32,
    /// Height of the fog's reference level (the sea level of a world) along [`Self::up`] in
    /// the camera's reference frame, metres; the renderer measures the camera's altitude from
    /// it.
    pub fog_base: f32,
    /// Distances in metres (`start < end`) over which the scene fades completely into the
    /// fog colour, so that the edge of a finite world never shows; `[0, 0]` disables the
    /// fade.
    pub horizon_fade: [f32; 2],
    /// Linear exposure applied before the AgX tone map.
    pub exposure: f32,
    /// Albedo multipliers per tint class: none, grass, foliage, water. The G-buffer pass varies
    /// the grass and foliage tints around these across the land (red and blue by up to about
    /// 40 percent, green by less, smoothly over tens to hundreds of metres).
    pub tints: [Vec3; 4],
}

/// Period (metres) by which the camera position that anchors the cloud layer is reduced: a
/// multiple of the layer's pattern period of 2,400 m, small enough that the anchor stays
/// precise in `f32`. The pattern jumps once every period (154 km) of flight.
pub const CLOUD_ANCHOR_PERIOD: f64 = 2_400.0 * 64.0;

/// The cloud anchor of a camera at `position` (frame coordinates, metres): the position
/// reduced modulo [`CLOUD_ANCHOR_PERIOD`] in `f64`, then narrowed.
#[must_use]
pub fn cloud_anchor(position: DVec3) -> Vec3 {
    let reduce = |value: f64| value.rem_euclid(CLOUD_ANCHOR_PERIOD) as f32;
    Vec3::new(reduce(position.x), reduce(position.y), reduce(position.z))
}

impl Lighting {
    /// The light of a sky with the colours `horizon` and `zenith` on a surface facing up (see
    /// [`Self::sky_light`]): the mean of the sky dome, `zenith_weight` (0..1) of it the
    /// zenith's colour, desaturated towards its luminance so that it keeps `saturation` (0..1)
    /// of its colourfulness.
    #[must_use]
    pub fn sky_light_from(
        horizon: Vec3,
        zenith: Vec3,
        zenith_weight: f32,
        saturation: f32,
    ) -> Vec3 {
        let mean = horizon.lerp(zenith, zenith_weight);
        let luminance = mean.dot(Vec3::new(0.2126, 0.7152, 0.0722));
        Vec3::splat(luminance).lerp(mean, saturation)
    }
}

impl Default for Lighting {
    /// A clear late morning with +Y up.
    fn default() -> Self {
        let sky_zenith = Vec3::new(0.16, 0.34, 0.78);
        let sky_horizon = Vec3::new(0.62, 0.74, 0.92);
        Self {
            up: Vec3::Y,
            sun_direction: Vec3::new(0.35, 0.8, 0.45).normalize(),
            sun_color: Vec3::new(3.1, 2.95, 2.7),
            sun_angular_radius: 0.0047,
            sky_zenith,
            sky_horizon,
            sky_sunset: Vec3::new(0.66, 0.76, 0.92),
            ground: Vec3::new(0.32, 0.29, 0.24),
            sky_light: Self::sky_light_from(sky_horizon, sky_zenith, 0.35, 0.6),
            sky_gain: 1.0,
            ambient: 0.55,
            fog_density: 0.000_15,
            fog_falloff: 0.0,
            fog_base: 0.0,
            horizon_fade: [0.0, 0.0],
            exposure: 1.0,
            tints: [
                Vec3::ONE,
                Vec3::new(0.45, 0.72, 0.30),
                Vec3::new(0.38, 0.64, 0.26),
                Vec3::new(0.30, 0.50, 0.85),
            ],
        }
    }
}

/// The anchor of a grid with voxel `bounds` (half-open) for a camera at `camera_grid` (grid
/// coordinates, metres): the camera's voxel while it is inside the bounds expanded by
/// [`ANCHOR_MARGIN_VOXELS`], otherwise the nearest voxel of the expanded bounds (which keeps
/// the anchor continuous at the margin). `bounds` must not be empty.
#[must_use]
pub fn anchor_voxel(bounds: &IAabb, camera_grid: DVec3) -> IVec3 {
    let expanded = bounds.expand(ANCHOR_MARGIN_VOXELS);
    let floor = camera_grid.floor();
    let clamp = |value: f64, lo: i32, hi: i32| -> i32 {
        if value.is_nan() || value < f64::from(lo) {
            lo
        } else if value > f64::from(hi) {
            hi
        } else {
            // In range, integral after the floor.
            value as i32
        }
    };
    IVec3::new(
        clamp(floor.x, expanded.min.x, expanded.max.x - 1),
        clamp(floor.y, expanded.min.y, expanded.max.y - 1),
        clamp(floor.z, expanded.min.z, expanded.max.z - 1),
    )
}

/// The camera of one frame as the object composition needs it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct FrameCamera {
    pub(crate) camera: Camera,
    pub(crate) aspect: f32,
    /// Sub-pixel offset of the projection in normalised device coordinates (temporal
    /// anti-aliasing), zero without it.
    pub(crate) jitter: [f32; 2],
}

impl FrameCamera {
    /// Render → clip space with this frame's jitter.
    pub(crate) fn render_to_clip(&self) -> DMat4 {
        jitter_matrix(self.jitter) * self.camera.render_to_clip(self.aspect)
    }
}

/// The clip-space shift by `jitter` (NDC units): `J · (x, y, z, w) = (x + jx·w, y + jy·w, z,
/// w)`, so every point moves by the same fraction of a pixel on screen and depth is unchanged.
pub(crate) fn jitter_matrix(jitter: [f32; 2]) -> DMat4 {
    DMat4::from_cols(
        DVec4::X,
        DVec4::Y,
        DVec4::Z,
        DVec4::new(f64::from(jitter[0]), f64::from(jitter[1]), 0.0, 1.0),
    )
}

/// The GPU object of a grid with `bounds` at `pose` for `now`, with `prev_pose` and `prev`
/// for last frame's matrix (same anchor). Composed in `f64`; narrowed to `f32` at the end.
/// `local_to_clip` carries this frame's jitter; `prev_local_to_clip` is last frame's matrix
/// without jitter (what motion vectors need).
pub(crate) fn compose_object(
    bounds: &IAabb,
    pose: &GridPose,
    prev_pose: &GridPose,
    now: &FrameCamera,
    prev: &FrameCamera,
) -> GpuObject {
    let camera_grid = pose.frame_to_grid(now.camera.position);
    let anchor = anchor_voxel(bounds, camera_grid);
    let anchor_metres = anchor.as_dvec3();
    let local_to_render = local_to_render_matrix(pose, anchor_metres, &now.camera);
    let local_to_clip = now.render_to_clip() * local_to_render;
    let prev_local_to_clip = prev.camera.render_to_clip(prev.aspect)
        * local_to_render_matrix(prev_pose, anchor_metres, &prev.camera);
    GpuObject {
        local_to_clip: local_to_clip.as_mat4().to_cols_array_2d(),
        prev_local_to_clip: prev_local_to_clip.as_mat4().to_cols_array_2d(),
        local_to_render: local_to_render.as_mat4().to_cols_array_2d(),
        anchor: anchor.to_array(),
        flags: 0,
        camera_local: (camera_grid - anchor_metres).as_vec3().to_array(),
        reserved: 0,
    }
}

/// Local (grid axes, anchor-relative) → render space. The translation is the anchor's frame
/// position minus the camera position, both in `f64`, so it is small and exact to far below a
/// millimetre wherever the grid is.
fn local_to_render_matrix(pose: &GridPose, anchor: DVec3, camera: &Camera) -> DMat4 {
    let anchor_in_frame = pose.rotation * anchor + pose.translation;
    DMat4::from_rotation_translation(pose.rotation, anchor_in_frame - camera.position)
}

#[cfg(test)]
mod tests {
    use cp_core::det;

    use super::*;

    fn bounds() -> IAabb {
        IAabb::new(IVec3::new(-64, 0, -64), IVec3::new(64, 32, 64))
    }

    #[test]
    fn the_anchor_is_the_camera_voxel_inside_the_expanded_bounds() {
        let b = bounds();
        assert_eq!(
            anchor_voxel(&b, DVec3::new(3.7, 5.2, -0.5)),
            IVec3::new(3, 5, -1)
        );
        // 999 m above the top is still inside the margin.
        assert_eq!(
            anchor_voxel(&b, DVec3::new(0.5, 32.0 + 999.5, 0.5)),
            IVec3::new(0, 1031, 0)
        );
        // Beyond it: the nearest voxel of the expanded bounds, continuous at the margin.
        assert_eq!(
            anchor_voxel(&b, DVec3::new(0.5, 5000.0, 0.5)),
            IVec3::new(0, 1031, 0)
        );
        assert_eq!(
            anchor_voxel(&b, DVec3::new(-1.0e12, -1.0e12, 1.0e12)),
            IVec3::new(-1064, -1000, 1063)
        );
        assert_eq!(
            anchor_voxel(&b, DVec3::splat(f64::NAN)),
            IVec3::new(-1064, -1000, -1064)
        );
    }

    fn frame_camera(position: DVec3) -> FrameCamera {
        FrameCamera {
            camera: Camera::look_at(
                position,
                position + DVec3::new(1.0, -0.5, -2.0),
                DVec3::Y,
                1.0,
            ),
            aspect: 16.0 / 9.0,
            jitter: [0.0, 0.0],
        }
    }

    #[test]
    fn jitter_moves_every_point_by_the_same_ndc_offset() {
        let camera = frame_camera(DVec3::new(3.0, 4.0, 5.0));
        let jittered = FrameCamera {
            jitter: [0.002, -0.001],
            ..camera
        };
        for point in [
            DVec3::new(0.3, -0.2, -1.0),
            DVec3::new(10.0, 3.0, -40.0),
            DVec3::new(-500.0, 20.0, -2000.0),
        ] {
            let plain = camera.render_to_clip() * point.extend(1.0);
            let moved = jittered.render_to_clip() * point.extend(1.0);
            let (a, b) = (plain.truncate() / plain.w, moved.truncate() / moved.w);
            assert!((b.x - a.x - 0.002).abs() < 1e-9, "{a} {b}");
            assert!((b.y - a.y + 0.001).abs() < 1e-9, "{a} {b}");
            assert!((b.z - a.z).abs() < 1e-12, "depth is unchanged: {a} {b}");
        }
        // The object's matrix carries the jitter, its previous matrix does not.
        let bounds = IAabb::new(IVec3::ZERO, IVec3::splat(32));
        let pose = GridPose::IDENTITY;
        let object = compose_object(&bounds, &pose, &pose, &jittered, &jittered);
        let plain = compose_object(&bounds, &pose, &pose, &camera, &camera);
        assert_ne!(object.local_to_clip, plain.local_to_clip);
        assert_eq!(object.prev_local_to_clip, plain.prev_local_to_clip);
    }

    #[test]
    fn cloud_anchors_are_reduced_modulo_the_period() {
        let near = cloud_anchor(DVec3::new(10.5, -3.0, 2000.0));
        assert_eq!(
            near,
            Vec3::new(10.5, CLOUD_ANCHOR_PERIOD as f32 - 3.0, 2000.0)
        );
        let far = cloud_anchor(DVec3::new(10.5 + 7.0 * CLOUD_ANCHOR_PERIOD, 0.0, 0.0));
        assert!((far.x - 10.5).abs() < 1e-3, "{far}");
        assert!(CLOUD_ANCHOR_PERIOD.rem_euclid(2400.0) == 0.0);
    }

    #[test]
    fn objects_far_from_the_origin_get_the_matrices_of_near_ones() {
        // The same relative placement 3,000 km out: identical bits, because only the camera's
        // fractional offset from the anchor enters the matrices.
        let offset = DVec3::new(5.25, 3.5, 7.75);
        let near_bounds = IAabb::new(IVec3::ZERO, IVec3::splat(32));
        let far_origin = IVec3::new(3_000_000, 0, 0);
        let far_bounds = IAabb::new(far_origin, far_origin + IVec3::splat(32));
        let near_camera = frame_camera(offset);
        let far_camera = frame_camera(far_origin.as_dvec3() + offset);
        let pose = GridPose::IDENTITY;
        let near = compose_object(&near_bounds, &pose, &pose, &near_camera, &near_camera);
        let far = compose_object(&far_bounds, &pose, &pose, &far_camera, &far_camera);
        assert_eq!(near.anchor, [5, 3, 7]);
        assert_eq!(far.anchor, [3_000_005, 3, 7]);
        assert_eq!(near.local_to_clip, far.local_to_clip);
        assert_eq!(near.local_to_render, far.local_to_render);
        assert_eq!(near.camera_local, far.camera_local);
        assert_eq!(near.camera_local, [0.25, 0.5, 0.75]);
        assert_eq!(near.prev_local_to_clip, near.local_to_clip);
    }

    #[test]
    fn the_matrices_map_grid_points_where_the_camera_sees_them() {
        // A rotated, translated grid: local point → clip must equal the f64 reference.
        let pose = GridPose {
            rotation: det::dquat_from_axis_angle(DVec3::new(0.3, 1.0, 0.2).normalize(), 0.7),
            translation: DVec3::new(1.0e6 + 0.125, -2.0e5, 3.0e4),
        };
        let bounds = IAabb::new(IVec3::splat(-500), IVec3::splat(500));
        let grid_point = DVec3::new(12.0, 40.0, -7.0);
        let frame_point = pose.rotation * grid_point + pose.translation;
        let camera = frame_camera(frame_point + DVec3::new(-30.0, 20.0, 60.0));
        let object = compose_object(&bounds, &pose, &pose, &camera, &camera);
        let anchor = IVec3::from_array(object.anchor);
        let local = (grid_point.as_ivec3() - anchor).as_vec3();
        let clip = Mat4::from_cols_array_2d(&object.local_to_clip) * local.extend(1.0);
        let reference = camera.camera.frame_to_clip(camera.aspect) * frame_point.extend(1.0);
        let ndc = clip.truncate() / clip.w;
        let reference_ndc = reference.truncate() / reference.w;
        assert!(
            (ndc.as_dvec3() - reference_ndc).abs().max_element() < 1.0e-5,
            "{ndc:?} vs {reference_ndc:?}"
        );
        // The render-space position is camera-relative.
        let render = Mat4::from_cols_array_2d(&object.local_to_render) * local.extend(1.0);
        let expected = (frame_point - camera.camera.position).as_vec3();
        assert!((render.truncate() - expected).abs().max_element() < 1.0e-3);
        // The camera in local space is where the render origin maps back to.
        let back =
            Mat4::from_cols_array_2d(&object.local_to_render).inverse() * cp_core::math::Vec4::W;
        let camera_local = Vec3::from_array(object.camera_local);
        assert!((back.truncate() - camera_local).abs().max_element() < 1.0e-3);
    }

    #[test]
    fn look_at_points_the_view_axis_at_the_target() {
        let camera = Camera::look_at(
            DVec3::new(1.0, 2.0, 3.0),
            DVec3::new(1.0, 2.0, -7.0),
            DVec3::Y,
            1.0,
        );
        let forward = camera.orientation * DVec3::NEG_Z;
        assert!((forward - DVec3::NEG_Z).length() < 1e-12);
        let up = camera.orientation * DVec3::Y;
        assert!((up - DVec3::Y).length() < 1e-12);
        // The projection is the contract's: depth 1 at the near plane.
        let clip =
            camera.render_to_clip(1.0) * DVec3::new(0.0, 0.0, -f64::from(Z_NEAR)).extend(1.0);
        assert!((clip.z / clip.w - 1.0).abs() < 1e-6);
    }
}
