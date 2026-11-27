//! The calibration scene: cp-render's calibration builder (contract §9 E2: G-buffer time at
//! 1.0 M visible triangles, cull time, nanoseconds per indirect draw).
//!
//! Every sub-scene is one [`CalibrationParams`]: resident chunks of which the camera sees a
//! known number, each visible chunk one draw. The first two hold 4,000 chunks with 4 M quads
//! resident and show 1.0 M and 2.0 M triangles; the next two draw 1.0 M triangles in 5,000 and
//! 20,000 indirect draws; the last three are one-quad draws for the cost per draw (5,000 and
//! 20,000 draws against a one-draw baseline).

use cp_render::voxel::CalibrationParams;

/// One sub-scene of the calibration scene.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CalibSubScene {
    /// Short name for logs and reports.
    pub name: &'static str,
    /// What it shows.
    pub description: &'static str,
    /// The calibration builder's parameters.
    pub params: CalibrationParams,
}

/// Material ids the calibration quads cycle through (the base pack has more).
const MATERIALS: u16 = 16;

/// The sub-scenes, in the order of `--camera`.
pub fn sub_scenes() -> Vec<CalibSubScene> {
    let scene = |name, description, resident, visible, quads| CalibSubScene {
        name,
        description,
        params: CalibrationParams {
            resident_chunks: resident,
            visible_chunks: visible,
            quads_per_chunk: quads,
            materials: MATERIALS,
            ..CalibrationParams::default()
        },
    };
    vec![
        scene(
            "triangles_1m",
            "4,000 chunks / 4 M quads resident, 1.0 M visible triangles in 500 draws",
            4_000,
            500,
            1_000,
        ),
        scene(
            "triangles_2m",
            "4,000 chunks / 4 M quads resident, 2.0 M visible triangles in 1,000 draws",
            4_000,
            1_000,
            1_000,
        ),
        scene(
            "draws_5k",
            "1.0 M visible triangles in 5,000 indirect draws",
            5_000,
            5_000,
            100,
        ),
        scene(
            "draws_20k",
            "1.0 M visible triangles in 20,000 indirect draws",
            20_000,
            20_000,
            25,
        ),
        scene(
            "baseline_1",
            "one one-quad draw: the fixed cost of the G-buffer pass",
            1,
            1,
            1,
        ),
        scene(
            "quads_5k",
            "5,000 one-quad draws: cost per draw",
            5_000,
            5_000,
            1,
        ),
        scene(
            "quads_20k",
            "20,000 one-quad draws: cost per draw",
            20_000,
            20_000,
            1,
        ),
    ]
}

/// Index of the one-draw baseline among [`sub_scenes`].
pub const BASELINE: usize = 4;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_sub_scenes_have_the_promised_loads() {
        let scenes = sub_scenes();
        let triangles = |s: &CalibSubScene| {
            2 * u64::from(s.params.visible_chunks) * u64::from(s.params.quads_per_chunk)
        };
        assert_eq!(triangles(&scenes[0]), 1_000_000);
        assert_eq!(triangles(&scenes[1]), 2_000_000);
        assert_eq!(
            u64::from(scenes[0].params.resident_chunks)
                * u64::from(scenes[0].params.quads_per_chunk),
            4_000_000
        );
        assert_eq!(scenes[2].params.visible_chunks, 5_000);
        assert_eq!(scenes[3].params.visible_chunks, 20_000);
        assert_eq!(triangles(&scenes[2]), 1_000_000);
        assert_eq!(triangles(&scenes[3]), 1_000_000);
        assert_eq!(scenes[BASELINE].params.visible_chunks, 1);
        for scene in &scenes {
            assert!(scene.params.visible_chunks <= scene.params.resident_chunks);
            assert!((1..=1024).contains(&scene.params.quads_per_chunk));
        }
    }
}
