//! The scenes of `cpe run`, `cpe shot` and `cpe bench` (contract §5.4: "Scenes are
//! deterministic camera scripts").
//!
//! | scene | world | cameras |
//! |---|---|---|
//! | [`hills`] | cp-worldgen's rolling hills around each camera, level 0 near it and coarser rings out to the horizon | eight cameras found by a deterministic search, and a flight path |
//! | [`structures`] | cp-worldgen's structure field at levels 0 to 3 side by side | an overview and two close-ups |
//! | [`calib`] | cp-render's calibration scenes | one per sub-scene |
//!
//! Shared here: the scene names of the command line and the camera of a shot.

pub mod build;
pub mod calib;
pub mod hills;
pub mod lod;
pub mod sky;
pub mod structures;

use std::fmt;

use cp_core::math::DVec3;
use cp_render::voxel::Camera;

use sky::{Sun, view_direction};

/// The scenes of the command line.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SceneName {
    /// Rolling hills with lakes, ridges and forests ([`hills`]).
    Hills,
    /// The structure set at LOD levels 0 to 3 ([`structures`]).
    Structures,
    /// The renderer's calibration scenes ([`calib`]).
    Calib,
}

impl SceneName {
    /// Every scene, in the order of the usage text.
    pub const ALL: [Self; 3] = [Self::Hills, Self::Structures, Self::Calib];

    /// The name on the command line.
    pub const fn id(self) -> &'static str {
        match self {
            Self::Hills => "hills",
            Self::Structures => "structures",
            Self::Calib => "calib",
        }
    }

    /// The scene called `id`.
    pub fn from_id(id: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|scene| scene.id() == id)
    }
}

impl fmt::Display for SceneName {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.id())
    }
}

/// Steepest pitch of a scene camera: looking straight up or down has no defined yaw.
pub const MAX_PITCH: f64 = 1.55;

/// A camera of a scene script.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SceneCamera {
    /// Short name for logs and file names.
    pub name: &'static str,
    /// Eye position (metres, `+Y` up).
    pub position: DVec3,
    /// Compass direction of the view, radians clockwise from north (`-Z`) towards east (`+X`).
    pub yaw: f64,
    /// Angle of the view above the horizon, radians, within `±MAX_PITCH`.
    pub pitch: f64,
    /// Vertical field of view, radians.
    pub fov_y: f64,
    /// The sun of the shot unless the command line names one.
    pub sun: Sun,
}

impl SceneCamera {
    /// Unit vector of the view.
    pub fn forward(&self) -> DVec3 {
        view_direction(self.yaw, self.pitch.clamp(-MAX_PITCH, MAX_PITCH))
    }

    /// The renderer's camera.
    pub fn render_camera(&self) -> Camera {
        Camera::look_at(
            self.position,
            self.position + self.forward(),
            DVec3::Y,
            self.fov_y as f32,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scene_names_round_trip() {
        for scene in SceneName::ALL {
            assert_eq!(SceneName::from_id(scene.id()), Some(scene));
            assert_eq!(scene.to_string(), scene.id());
        }
        assert_eq!(SceneName::from_id("moon"), None);
    }

    #[test]
    fn scene_cameras_look_where_their_angles_say() {
        let camera = SceneCamera {
            name: "test",
            position: DVec3::new(1.0, 2.0, 3.0),
            yaw: std::f64::consts::FRAC_PI_2,
            pitch: 0.0,
            fov_y: 1.0,
            sun: Sun::default_from(0.0),
        };
        let forward = camera.render_camera().orientation * DVec3::NEG_Z;
        assert!((forward - DVec3::X).length() < 1e-9, "{forward:?}");
        let straight_down = SceneCamera {
            pitch: -10.0,
            ..camera
        };
        assert!(straight_down.forward().y > -1.0);
        assert!(straight_down.render_camera().orientation.is_finite());
    }
}
