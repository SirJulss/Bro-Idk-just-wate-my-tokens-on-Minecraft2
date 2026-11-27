//! Reference frames (contract §2.2): moving, rotating body frames and galaxy positions.
//!
//! All quantities are `f64`: metres, metres per second, radians per second. The transforms use
//! only exact IEEE operations (`+ - * /`, `sqrt`), so they are deterministic (contract §2.5).
//!
//! ```
//! use cp_core::det::dquat_from_axis_angle;
//! use cp_core::frames::BodyFrame;
//! use cp_core::math::DVec3;
//!
//! // A planet 1000 km along +X of its system, turned a quarter about +Y, spinning about +Y.
//! let planet = BodyFrame {
//!     position: DVec3::new(1.0e6, 0.0, 0.0),
//!     orientation: dquat_from_axis_angle(DVec3::Y, std::f64::consts::FRAC_PI_2),
//!     velocity: DVec3::new(0.0, 0.0, 30.0),
//!     angular_velocity: DVec3::new(0.0, 1.0e-3, 0.0),
//! };
//! // A point 10 km above the planet's +X face, in system coordinates and back.
//! let on_planet = DVec3::new(10_000.0, 0.0, 0.0);
//! let in_system = planet.point_to_parent(on_planet);
//! assert!((in_system - DVec3::new(1.0e6, 0.0, -10_000.0)).length() < 1e-6);
//! assert!((planet.point_to_body(in_system) - on_planet).length() < 1e-6);
//! // Standing still on the planet means moving with it in the system.
//! let velocity = planet.velocity_to_parent(on_planet, DVec3::ZERO);
//! assert!((velocity - DVec3::new(-10.0, 0.0, 30.0)).length() < 1e-9);
//! ```

use serde::{Deserialize, Serialize};

use crate::math::{DQuat, DVec3};

/// The state of a rigid frame (a body, a construct) relative to its parent frame.
///
/// It maps *body* coordinates (the frame's own, fixed to it) to *parent* coordinates
/// (for a planet: the inertial system frame).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct BodyFrame {
    /// Position of the body origin in parent coordinates, metres.
    pub position: DVec3,
    /// Rotation from body axes to parent axes. Must be a unit quaternion.
    pub orientation: DQuat,
    /// Velocity of the body origin in parent coordinates, metres per second.
    pub velocity: DVec3,
    /// Angular velocity of the body relative to the parent, expressed in **parent**
    /// coordinates, radians per second (right-hand rule about the vector).
    pub angular_velocity: DVec3,
}

impl Default for BodyFrame {
    fn default() -> Self {
        Self::IDENTITY
    }
}

impl BodyFrame {
    /// The frame that coincides with its parent and does not move.
    pub const IDENTITY: Self = Self {
        position: DVec3::ZERO,
        orientation: DQuat::IDENTITY,
        velocity: DVec3::ZERO,
        angular_velocity: DVec3::ZERO,
    };

    /// A frame at rest relative to its parent.
    #[inline]
    #[must_use]
    pub const fn at_rest(position: DVec3, orientation: DQuat) -> Self {
        Self {
            position,
            orientation,
            velocity: DVec3::ZERO,
            angular_velocity: DVec3::ZERO,
        }
    }

    /// Body point → parent point.
    #[inline]
    #[must_use]
    pub fn point_to_parent(&self, point: DVec3) -> DVec3 {
        self.position + self.orientation * point
    }

    /// Parent point → body point.
    #[inline]
    #[must_use]
    pub fn point_to_body(&self, point: DVec3) -> DVec3 {
        self.orientation.conjugate() * (point - self.position)
    }

    /// Body direction → parent direction (rotation only; lengths are preserved).
    #[inline]
    #[must_use]
    pub fn direction_to_parent(&self, direction: DVec3) -> DVec3 {
        self.orientation * direction
    }

    /// Parent direction → body direction.
    #[inline]
    #[must_use]
    pub fn direction_to_body(&self, direction: DVec3) -> DVec3 {
        self.orientation.conjugate() * direction
    }

    /// Velocity in the parent frame of something that is at the body point `point` and moves
    /// with `velocity` relative to the body (both in body coordinates). The result is in parent
    /// coordinates and includes the frame's own motion and rotation.
    #[inline]
    #[must_use]
    pub fn velocity_to_parent(&self, point: DVec3, velocity: DVec3) -> DVec3 {
        let arm = self.orientation * point;
        self.velocity + self.angular_velocity.cross(arm) + self.orientation * velocity
    }

    /// Inverse of [`BodyFrame::velocity_to_parent`]: `point` and `velocity` are in parent
    /// coordinates; the result is the velocity relative to the body, in body coordinates.
    /// Something that co-rotates with the body gets zero.
    #[inline]
    #[must_use]
    pub fn velocity_to_body(&self, point: DVec3, velocity: DVec3) -> DVec3 {
        let arm = point - self.position;
        self.orientation.conjugate() * (velocity - self.velocity - self.angular_velocity.cross(arm))
    }

    /// Chains two frames: `child` is given relative to `parent`, `parent` relative to some
    /// grandparent; the result is `child` relative to that grandparent.
    ///
    /// `compose(a, b).point_to_parent(p) == a.point_to_parent(b.point_to_parent(p))`, and
    /// likewise for directions and velocities.
    #[must_use]
    pub fn compose(parent: &Self, child: &Self) -> Self {
        Self {
            position: parent.point_to_parent(child.position),
            orientation: (parent.orientation * child.orientation).normalize(),
            velocity: parent.velocity_to_parent(child.position, child.velocity),
            angular_velocity: parent.angular_velocity + parent.orientation * child.angular_velocity,
        }
    }

    /// The rigid transform that converts coordinates in frame `from` to coordinates in frame
    /// `to`, at this instant. Both frames must be given relative to the same parent.
    ///
    /// `relative(from, to).transform_point(p) == to.point_to_body(from.point_to_parent(p))`.
    /// This is how the renderer places a body relative to the camera's reference frame.
    #[must_use]
    pub fn relative(from: &Self, to: &Self) -> RelativeTransform {
        let to_inverse = to.orientation.conjugate();
        RelativeTransform {
            rotation: (to_inverse * from.orientation).normalize(),
            translation: to_inverse * (from.position - to.position),
        }
    }
}

/// A rigid transform between two frames: `p ↦ rotation · p + translation`.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct RelativeTransform {
    /// Rotation from source axes to target axes (unit quaternion).
    pub rotation: DQuat,
    /// Position of the source origin in target coordinates, metres.
    pub translation: DVec3,
}

impl Default for RelativeTransform {
    fn default() -> Self {
        Self::IDENTITY
    }
}

impl RelativeTransform {
    /// The transform that changes nothing.
    pub const IDENTITY: Self = Self {
        rotation: DQuat::IDENTITY,
        translation: DVec3::ZERO,
    };

    /// Source point → target point.
    #[inline]
    #[must_use]
    pub fn transform_point(&self, point: DVec3) -> DVec3 {
        self.rotation * point + self.translation
    }

    /// Source direction → target direction (rotation only).
    #[inline]
    #[must_use]
    pub fn transform_direction(&self, direction: DVec3) -> DVec3 {
        self.rotation * direction
    }

    /// The transform in the opposite direction (target → source).
    #[inline]
    #[must_use]
    pub fn inverse(&self) -> Self {
        let rotation = self.rotation.conjugate();
        Self {
            rotation,
            translation: -(rotation * self.translation),
        }
    }
}

/// A position in the galaxy: an integer sector plus an offset inside it (contract §2.2).
///
/// Sectors are cubes of [`GalaxyPos::SECTOR_SIZE`] = 2^40 m (about 7.3 astronomical units).
/// Inside a sector neighbouring `f64` offsets are at most 2^-13 m (0.12 mm) apart, everywhere
/// in the galaxy. A position is *normalized* when `0 <= local < SECTOR_SIZE` on every axis;
/// the same place then has exactly one representation and `==` compares places.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct GalaxyPos {
    /// Sector index per axis.
    pub sector: [i64; 3],
    /// Offset from the sector's minimum corner, metres.
    pub local: DVec3,
}

impl Default for GalaxyPos {
    fn default() -> Self {
        Self::ORIGIN
    }
}

impl GalaxyPos {
    /// Edge length of a sector in metres: 2^40.
    pub const SECTOR_SIZE: f64 = 1_099_511_627_776.0;

    /// The minimum corner of sector `(0, 0, 0)`.
    pub const ORIGIN: Self = Self {
        sector: [0; 3],
        local: DVec3::ZERO,
    };

    /// A normalized position from any sector and offset.
    #[inline]
    #[must_use]
    pub fn new(sector: [i64; 3], local: DVec3) -> Self {
        Self { sector, local }.normalized()
    }

    /// The same place with `0 <= local < SECTOR_SIZE` on every axis, to within the sector
    /// resolution.
    ///
    /// Whole sectors contained in `local` move into `sector` (saturating at the `i64` limits).
    /// Per axis the place stays exactly the same when the offset is `>= 0` or a multiple of
    /// 2^-13 m (every offset `<= -2^39` m is such a multiple). Any other negative offset ends
    /// up in `[2^39, 2^40)` m of the sector below, where `f64` values are 2^-13 m (0.12 mm)
    /// apart, and is rounded to the nearest of them (ties to even): the place moves by at most
    /// 2^-14 m (0.06 mm). When the nearest value is the boundary itself, which happens for
    /// negative offsets of at most 2^-14 m, the result is `local = 0` in the same sector.
    /// Non-finite components are left unchanged.
    #[must_use]
    pub fn normalized(self) -> Self {
        let [sx, sy, sz] = self.sector;
        let (sx, x) = normalize_axis(sx, self.local.x);
        let (sy, y) = normalize_axis(sy, self.local.y);
        let (sz, z) = normalize_axis(sz, self.local.z);
        Self {
            sector: [sx, sy, sz],
            local: DVec3::new(x, y, z),
        }
    }

    /// The place `delta` metres away, normalized.
    #[inline]
    #[must_use]
    pub fn offset(self, delta: DVec3) -> Self {
        Self {
            sector: self.sector,
            local: self.local + delta,
        }
        .normalized()
    }

    /// The vector from `from` to `to` in metres (`to - from`).
    ///
    /// The sector difference is exact. For normalized positions the offset difference is below
    /// 2^40 m, so rounding it costs at most 2^-14 m (0.06 mm), and adding the sector term costs
    /// only the relative rounding of the `f64` result. Nearby places therefore keep the sector
    /// resolution however far they are from the galaxy origin.
    #[must_use]
    pub fn delta(from: &Self, to: &Self) -> DVec3 {
        let axis = |i: usize| {
            let sectors = i128::from(to.sector[i]) - i128::from(from.sector[i]);
            sectors as f64 * Self::SECTOR_SIZE + (to.local[i] - from.local[i])
        };
        DVec3::new(axis(0), axis(1), axis(2))
    }
}

/// Moves whole sectors out of `local`: returns the sector and the offset in
/// `[0, SECTOR_SIZE)` (non-finite offsets are returned unchanged).
///
/// The sector size is a power of two, so the floor of the quotient counts the whole sectors
/// exactly (apart from the underflow handled below) and `whole * SIZE` is exact. The true
/// remainder lies in `[0, 2^40)` and is a
/// multiple of the spacing of `local`, so `local - whole * SIZE` is exact whenever that spacing
/// is at least 2^-13 m (`|local| >= 2^39`), and trivially for `0 <= local < 2^39`, where
/// `whole = 0`. For `-2^39 < local < 0` the remainder `local + 2^40` lies in `(2^39, 2^40)`,
/// where `f64` values are 2^-13 m apart: the subtraction rounds it to the nearest of them, by
/// at most 2^-14 m (0.06 mm), and it is exact only when `local` is a multiple of 2^-13 m.
/// Rounding up to the full sector size means the boundary itself; the two corrections below
/// deal with that case and with a quotient that underflows.
fn normalize_axis(sector: i64, local: f64) -> (i64, f64) {
    const SIZE: f64 = GalaxyPos::SECTOR_SIZE;
    if !local.is_finite() {
        return (sector, local);
    }
    // Whole sectors below `local`: floor(local / SIZE), at most -1 for every negative offset.
    let mut whole = (local / SIZE).floor();
    // For negative offsets above about -2.7e-312 the quotient underflows to -0.0, whose floor
    // is not the -1 it stands for. Without this the offset would stay negative.
    if local < 0.0 && whole == 0.0 {
        whole = -1.0;
    }
    let mut rest = local - whole * SIZE;
    // A negative offset too small to resolve rounds `rest` up to the full sector size: that
    // is the minimum corner of the sector above.
    if rest >= SIZE {
        whole += 1.0;
        rest -= SIZE;
    }
    // The sector is adjusted once, with the final count, so that saturation at the `i64`
    // limits cannot be followed by a step back. The float-to-integer cast saturates too.
    (sector.saturating_add(whole as i64), rest)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::det::dquat_from_axis_angle;
    use crate::hash::Rng;

    fn random_vec(rng: &mut Rng, magnitude: f64) -> DVec3 {
        DVec3::new(
            rng.range_f64(-magnitude, magnitude),
            rng.range_f64(-magnitude, magnitude),
            rng.range_f64(-magnitude, magnitude),
        )
    }

    fn random_rotation(rng: &mut Rng) -> DQuat {
        loop {
            let axis = random_vec(rng, 1.0);
            if axis.length_squared() > 1e-3 && axis.length_squared() <= 1.0 {
                let angle = rng.range_f64(-core::f64::consts::PI, core::f64::consts::PI);
                return dquat_from_axis_angle(axis.normalize(), angle);
            }
        }
    }

    fn random_frame(rng: &mut Rng, distance: f64) -> BodyFrame {
        BodyFrame {
            position: random_vec(rng, distance),
            orientation: random_rotation(rng),
            velocity: random_vec(rng, 30_000.0),
            angular_velocity: random_vec(rng, 0.01),
        }
    }

    fn assert_close(a: DVec3, b: DVec3, tolerance: f64, what: &str) {
        assert!(
            (a - b).length() <= tolerance,
            "{what}: {a:?} vs {b:?} (tolerance {tolerance:e})"
        );
    }

    /// Spacing of `f64` values in the upper half of a sector, `[2^39, 2^40)` m: 2^-13 m.
    const SECTOR_STEP: f64 = 1.0 / 8192.0;

    fn random_sector(rng: &mut Rng) -> [i64; 3] {
        [
            i64::from(rng.next_u32() as i32),
            i64::from(rng.next_u32() as i32),
            i64::from(rng.next_u32() as i32),
        ]
    }

    /// A coordinate with 53 random significant bits, a random sign and a magnitude below 2^k m
    /// for a random `k` in `-30..=46`: fractions far finer than a sector resolves.
    fn fine_coordinate(rng: &mut Rng) -> f64 {
        let exponent = rng.range_u32(0, 77) as i32 - 30;
        let magnitude = rng.f64() * f64::from_bits(((1023 + exponent) as u64) << 52);
        if rng.next_u32().is_multiple_of(2) {
            magnitude
        } else {
            -magnitude
        }
    }

    /// Knuth's TwoSum: the rounded sum and its rounding error, `a + b == sum + error` exactly.
    fn two_sum(a: f64, b: f64) -> (f64, f64) {
        let sum = a + b;
        let b_part = sum - a;
        let a_part = sum - b_part;
        (sum, (a - a_part) + (b - b_part))
    }

    /// The rule documented on [`GalaxyPos::normalized`]: the offsets of one axis that keep
    /// exactly the same place.
    fn normalizes_exactly(local: f64) -> bool {
        local >= 0.0 || local % SECTOR_STEP == 0.0
    }

    /// Checks [`GalaxyPos::normalized`] against its documentation, axis by axis.
    fn assert_normalizes_as_documented(raw: GalaxyPos) {
        const S: f64 = GalaxyPos::SECTOR_SIZE;
        let normalized = raw.normalized();
        for axis in 0..3 {
            let (local, offset) = (raw.local[axis], normalized.local[axis]);
            assert!(
                (0.0..S).contains(&offset) && offset.is_sign_positive(),
                "{local:e} -> {normalized:?}"
            );
            // Sector and offset recombined, as an exact unevaluated sum.
            let moved = normalized.sector[axis] - raw.sector[axis];
            let (recombined, residue) = two_sum(moved as f64 * S, offset);
            if normalizes_exactly(local) {
                assert!(
                    recombined == local && residue == 0.0,
                    "{local:e} -> {normalized:?}"
                );
            } else {
                // A negative offset above -2^39 m with a finer fraction: the nearest `f64` of
                // the sector below, or the boundary itself. Both differences are exact here,
                // and the place does move.
                assert!(
                    moved == -1 || (moved == 0 && offset == 0.0),
                    "{local:e} -> {normalized:?}"
                );
                let error = (recombined - local) + residue;
                assert!(
                    error != 0.0 && error.abs() <= 0.5 * SECTOR_STEP,
                    "{local:e} moved by {error:e}"
                );
            }
        }
        assert_eq!(normalized.normalized(), normalized);
        // `delta` sees no difference where the place is kept, at most the rounding otherwise.
        let seen = GalaxyPos::delta(&raw, &normalized);
        if (0..3).all(|axis| normalizes_exactly(raw.local[axis])) {
            assert_eq!(seen, DVec3::ZERO, "{raw:?}");
        } else {
            assert!(
                seen.abs().max_element() <= 0.5 * SECTOR_STEP,
                "{raw:?}: {seen:?}"
            );
        }
    }

    #[test]
    fn identity_changes_nothing() {
        let frame = BodyFrame::IDENTITY;
        assert_eq!(BodyFrame::default(), frame);
        let p = DVec3::new(1.0, -2.0, 3.5);
        assert_eq!(frame.point_to_parent(p), p);
        assert_eq!(frame.point_to_body(p), p);
        assert_eq!(frame.direction_to_parent(p), p);
        assert_eq!(frame.velocity_to_parent(p, DVec3::X), DVec3::X);
        assert_eq!(frame.velocity_to_body(p, DVec3::X), DVec3::X);
        assert_eq!(RelativeTransform::default().transform_point(p), p);
        assert_eq!(
            BodyFrame::at_rest(p, DQuat::IDENTITY).point_to_parent(DVec3::ZERO),
            p
        );
    }

    #[test]
    fn known_configuration() {
        // A body 100 m along +X, turned a quarter about +Z (body X points along parent Y),
        // moving along +Y at 5 m/s and spinning about +Z at 2 rad/s.
        let frame = BodyFrame {
            position: DVec3::new(100.0, 0.0, 0.0),
            orientation: dquat_from_axis_angle(DVec3::Z, core::f64::consts::FRAC_PI_2),
            velocity: DVec3::new(0.0, 5.0, 0.0),
            angular_velocity: DVec3::new(0.0, 0.0, 2.0),
        };
        assert_close(
            frame.point_to_parent(DVec3::X),
            DVec3::new(100.0, 1.0, 0.0),
            1e-12,
            "point",
        );
        assert_close(
            frame.direction_to_parent(DVec3::Y),
            DVec3::NEG_X,
            1e-12,
            "direction",
        );
        assert_close(
            frame.point_to_body(DVec3::new(100.0, 1.0, 0.0)),
            DVec3::X,
            1e-12,
            "inverse",
        );
        // A point fixed 1 m along body X sits 1 m along parent Y from the centre; the spin
        // carries it towards -X at 2 m/s on top of the body's own 5 m/s.
        let fixed = frame.velocity_to_parent(DVec3::X, DVec3::ZERO);
        assert_close(
            fixed,
            DVec3::new(-2.0, 5.0, 0.0),
            1e-12,
            "velocity of a fixed point",
        );
        let relative = frame.velocity_to_body(DVec3::new(100.0, 1.0, 0.0), fixed);
        assert_close(
            relative,
            DVec3::ZERO,
            1e-12,
            "co-rotating point is at rest in the body",
        );
    }

    #[test]
    fn round_trips_to_1e9() {
        let mut rng = Rng::new(77, 1);
        for case in 0..5000 {
            // Body scale (tens of km): absolute tolerance 1e-9 m.
            let frame = random_frame(&mut rng, 50_000.0);
            let p = random_vec(&mut rng, 50_000.0);
            let v = random_vec(&mut rng, 500.0);
            assert_close(
                frame.point_to_body(frame.point_to_parent(p)),
                p,
                1e-9,
                "point",
            );
            assert_close(
                frame.point_to_parent(frame.point_to_body(p)),
                p,
                1e-9,
                "point back",
            );
            assert_close(
                frame.direction_to_body(frame.direction_to_parent(p)),
                p,
                1e-9,
                "direction",
            );
            assert_close(
                frame.direction_to_parent(frame.direction_to_body(p)),
                p,
                1e-9,
                "direction",
            );
            let lifted = frame.velocity_to_parent(p, v);
            let lowered = frame.velocity_to_body(frame.point_to_parent(p), lifted);
            assert_close(lowered, v, 1e-9, "velocity");
            // Rotations preserve lengths and angles.
            let q = random_vec(&mut rng, 10.0);
            let (pr, qr) = (frame.direction_to_parent(p), frame.direction_to_parent(q));
            assert!(
                (pr.dot(qr) - p.dot(q)).abs() <= 1e-9 * (1.0 + p.dot(q).abs()),
                "case {case}"
            );

            // System scale (tens of astronomical units): relative tolerance 1e-9.
            let far = random_frame(&mut rng, 1.0e13);
            let p = random_vec(&mut rng, 1.0e13);
            let scale = 1.0e13;
            assert_close(
                far.point_to_body(far.point_to_parent(p)),
                p,
                1e-9 * scale,
                "far point",
            );
            assert_close(
                far.point_to_parent(far.point_to_body(p)),
                p,
                1e-9 * scale,
                "far point",
            );
        }
    }

    #[test]
    fn velocity_transform_is_the_time_derivative() {
        // Move the frame and the point forward by a small time step and difference positions.
        let mut rng = Rng::new(78, 1);
        for _ in 0..300 {
            let frame = BodyFrame {
                position: random_vec(&mut rng, 1000.0),
                orientation: random_rotation(&mut rng),
                velocity: random_vec(&mut rng, 100.0),
                angular_velocity: random_vec(&mut rng, 1.0),
            };
            let p = random_vec(&mut rng, 100.0);
            let v = random_vec(&mut rng, 10.0);
            let dt = 1.0e-6;
            let angle = frame.angular_velocity.length() * dt;
            let spin = dquat_from_axis_angle(frame.angular_velocity.normalize(), angle);
            let later = BodyFrame {
                position: frame.position + frame.velocity * dt,
                orientation: spin * frame.orientation,
                ..frame
            };
            let numeric = (later.point_to_parent(p + v * dt) - frame.point_to_parent(p)) / dt;
            assert_close(
                frame.velocity_to_parent(p, v),
                numeric,
                2e-3,
                "finite difference",
            );
        }
    }

    #[test]
    fn compose_chains_transforms() {
        let mut rng = Rng::new(79, 1);
        for _ in 0..3000 {
            let parent = random_frame(&mut rng, 1.0e5);
            let child = random_frame(&mut rng, 1.0e4);
            let composed = BodyFrame::compose(&parent, &child);
            let p = random_vec(&mut rng, 1000.0);
            let v = random_vec(&mut rng, 100.0);

            let direct = parent.point_to_parent(child.point_to_parent(p));
            assert_close(composed.point_to_parent(p), direct, 1e-9, "point");
            assert_close(composed.point_to_body(direct), p, 1e-9, "point inverse");
            let direction = parent.direction_to_parent(child.direction_to_parent(p));
            assert_close(
                composed.direction_to_parent(p),
                direction,
                1e-9,
                "direction",
            );
            let velocity =
                parent.velocity_to_parent(child.point_to_parent(p), child.velocity_to_parent(p, v));
            assert_close(
                composed.velocity_to_parent(p, v),
                velocity,
                1e-9,
                "velocity",
            );
            assert!((composed.orientation.length() - 1.0).abs() < 1e-15);

            // Composition with the identity, and associativity.
            assert_close(
                BodyFrame::compose(&BodyFrame::IDENTITY, &child).point_to_parent(p),
                child.point_to_parent(p),
                1e-9,
                "left identity",
            );
            assert_close(
                BodyFrame::compose(&child, &BodyFrame::IDENTITY).point_to_parent(p),
                child.point_to_parent(p),
                1e-9,
                "right identity",
            );
            let grandchild = random_frame(&mut rng, 100.0);
            let left = BodyFrame::compose(&composed, &grandchild);
            let right = BodyFrame::compose(&parent, &BodyFrame::compose(&child, &grandchild));
            assert_close(
                left.point_to_parent(p),
                right.point_to_parent(p),
                1e-9,
                "associative",
            );
            assert_close(
                left.velocity_to_parent(p, v),
                right.velocity_to_parent(p, v),
                1e-9,
                "associative velocity",
            );
        }
    }

    #[test]
    fn relative_transform_links_sibling_frames() {
        let mut rng = Rng::new(80, 1);
        for _ in 0..3000 {
            let from = random_frame(&mut rng, 1.0e5);
            let to = random_frame(&mut rng, 1.0e5);
            let transform = BodyFrame::relative(&from, &to);
            let p = random_vec(&mut rng, 1000.0);

            let expected = to.point_to_body(from.point_to_parent(p));
            assert_close(transform.transform_point(p), expected, 1e-9, "point");
            let direction = to.direction_to_body(from.direction_to_parent(p));
            assert_close(
                transform.transform_direction(p),
                direction,
                1e-9,
                "direction",
            );

            // Inverse and the opposite relative transform agree and round-trip.
            let inverse = transform.inverse();
            assert_close(inverse.transform_point(expected), p, 1e-9, "inverse");
            let back = BodyFrame::relative(&to, &from);
            assert_close(
                back.transform_point(expected),
                p,
                1e-9,
                "opposite direction",
            );
            assert_close(
                back.translation,
                inverse.translation,
                1e-9,
                "inverse translation",
            );

            // A frame relative to itself is the identity; relative to the parent it is the frame.
            let own = BodyFrame::relative(&from, &from);
            assert_close(own.transform_point(p), p, 1e-9, "self");
            let to_parent = BodyFrame::relative(&from, &BodyFrame::IDENTITY);
            assert_close(
                to_parent.transform_point(p),
                from.point_to_parent(p),
                1e-9,
                "parent",
            );
        }
    }

    #[test]
    fn galaxy_positions_normalize_as_documented() {
        const S: f64 = GalaxyPos::SECTOR_SIZE;
        assert_eq!(S, (1_u64 << 40) as f64);
        assert_eq!(GalaxyPos::default(), GalaxyPos::ORIGIN);

        let p = GalaxyPos::new([0, 0, 0], DVec3::new(-1.0, S + 2.0, 3.5 * S));
        assert_eq!(p.sector, [-1, 1, 3]);
        assert_eq!(p.local, DVec3::new(S - 1.0, 2.0, 0.5 * S));
        assert_eq!(p.normalized(), p);

        // Exactly on a boundary belongs to the next sector.
        let boundary = GalaxyPos::new([5, 5, 5], DVec3::new(S, -S, 0.0));
        assert_eq!(boundary.sector, [6, 4, 5]);
        assert_eq!(boundary.local, DVec3::ZERO);

        // A negative offset too small to resolve inside the sector lands on the boundary.
        let tiny = GalaxyPos::new([0, 0, 0], DVec3::new(-1.0e-20, 0.0, 0.0));
        assert_eq!(tiny.sector, [0, 0, 0]);
        assert_eq!(tiny.local, DVec3::ZERO);

        // Limits and bad input do not panic.
        let edge = GalaxyPos::new(
            [i64::MAX, i64::MIN, 0],
            DVec3::new(2.0 * S, -2.0 * S, 1.0e300),
        );
        assert_eq!((edge.sector[0], edge.sector[1]), (i64::MAX, i64::MIN));
        let nan = GalaxyPos::new([1, 2, 3], DVec3::new(f64::NAN, f64::INFINITY, 1.0));
        assert_eq!(nan.sector, [1, 2, 3]);
        assert!(nan.local.x.is_nan() && nan.local.y.is_infinite() && nan.local.z == 1.0);

        // Uniform samples over ±40 sectors are multiples of 2^-9 m (80 sectors times the 2^-53
        // grid of the generator), so every one of them keeps exactly the same place.
        let mut rng = Rng::new(81, 1);
        for _ in 0..20_000 {
            let sector = random_sector(&mut rng);
            let local = random_vec(&mut rng, 40.0 * S);
            for axis in 0..3 {
                assert!(normalizes_exactly(local[axis]), "{:e}", local[axis]);
            }
            assert_normalizes_as_documented(GalaxyPos { sector, local });
        }
        // Fractions far below the sector resolution at every magnitude from 2^-30 m to 2^46 m,
        // so that negative offsets reach the rounding case about as often as the exact ones.
        let mut rng = Rng::new(81, 2);
        let mut rounded = 0;
        for _ in 0..20_000 {
            let sector = random_sector(&mut rng);
            let local = DVec3::new(
                fine_coordinate(&mut rng),
                fine_coordinate(&mut rng),
                fine_coordinate(&mut rng),
            );
            rounded += (0..3)
                .filter(|&axis| !normalizes_exactly(local[axis]))
                .count();
            assert_normalizes_as_documented(GalaxyPos { sector, local });
        }
        // Almost half of the 60 000 coordinates are negative and above -2^39 m.
        assert!(rounded > 20_000, "only {rounded} coordinates were rounded");
    }

    #[test]
    fn negative_offsets_with_fine_fractions_round_to_the_sector_below() {
        const S: f64 = GalaxyPos::SECTOR_SIZE;
        // Regression (v2 review): the documentation promised exactness here. -0.3 m becomes
        // 2^40 - 0.300048828125 m (2458 steps of 2^-13 m) in the sector below.
        let p = GalaxyPos::new([0; 3], DVec3::new(-0.3, 0.0, 0.0));
        assert_eq!(p.sector, [-1, 0, 0]);
        assert_eq!(p.local, DVec3::new(S - 0.300_048_828_125, 0.0, 0.0));
        assert_eq!(
            GalaxyPos::delta(&GalaxyPos::ORIGIN, &p).x,
            -0.300_048_828_125
        );

        // (offset, distance of the result below the boundary): the nearest `f64` of the
        // sector below, ties to even; multiples of a step stay exact.
        let cases = [
            (-0.3, 0.300_048_828_125),
            (-1.0e-4, SECTOR_STEP),
            (-12_345.678, 12_345.677_978_515_625),
            (-1.5 * SECTOR_STEP, 2.0 * SECTOR_STEP),
            (-2.5 * SECTOR_STEP, 2.0 * SECTOR_STEP),
            (-3.0 * SECTOR_STEP, 3.0 * SECTOR_STEP),
            (-0.5 * S + SECTOR_STEP, 0.5 * S - SECTOR_STEP),
        ];
        for sector in [[0; 3], [7, -3, 2], [i64::MIN + 1; 3], [i64::MAX; 3]] {
            for (offset, below_boundary) in cases {
                let p = GalaxyPos::new(sector, DVec3::splat(offset));
                assert_eq!(p.sector, sector.map(|s| s - 1), "{offset:e}");
                assert_eq!(p.local, DVec3::splat(S - below_boundary), "{offset:e}");
                assert_eq!(p.normalized(), p);
                assert_normalizes_as_documented(GalaxyPos {
                    sector,
                    local: DVec3::splat(offset),
                });
                // The positive counterpart is resolved inside the sector and stays as it is.
                let q = GalaxyPos::new(sector, DVec3::splat(-offset));
                assert_eq!((q.sector, q.local), (sector, DVec3::splat(-offset)));
            }
        }

        // `delta` across a sector boundary rounds the offset difference (almost 2^40 m) to its
        // spacing, so nearby places come out within 2^-14 m of their exact difference.
        let mut rng = Rng::new(83, 1);
        for _ in 0..10_000 {
            let steps = f64::from(rng.range_u32(1, 1 << 20));
            let below = GalaxyPos {
                sector: [3, -1, 0],
                local: DVec3::new(S - steps * SECTOR_STEP, 0.5, 0.5),
            };
            let fraction = rng.f64() * 64.0;
            let above = GalaxyPos {
                sector: [4, -1, 0],
                local: DVec3::new(fraction, 0.5, 0.5),
            };
            let measured = GalaxyPos::delta(&below, &above);
            let (exact, residue) = two_sum(steps * SECTOR_STEP, fraction);
            let error = (measured.x - exact) - residue;
            assert!(
                error.abs() <= 0.5 * SECTOR_STEP,
                "{steps} steps + {fraction:e}: off by {error:e}"
            );
            assert_eq!((measured.y, measured.z), (0.0, 0.0));
            assert_eq!(GalaxyPos::delta(&above, &below), -measured);
        }
    }

    #[test]
    fn offsets_below_the_resolution_normalize_onto_the_sector_corner() {
        const S: f64 = GalaxyPos::SECTOR_SIZE;
        // Spacing of `f64` values just below a sector boundary: 2^-13 m.
        const STEP: f64 = 1.0 / 8192.0;
        let corner = |sector| GalaxyPos {
            sector,
            local: DVec3::ZERO,
        };
        let bits = |v: DVec3| v.to_array().map(f64::to_bits);
        let sectors = [[0; 3], [7, -3, 2], [i64::MIN; 3], [i64::MAX; 3]];

        // Regression: for negative subnormal offsets the quotient `local / S` underflows to
        // -0.0. Such an offset used to keep its sector and stay negative, so `normalized()`
        // returned a position outside `0 <= local < S`. 2^-1035 (bit pattern 1 << 39) is the
        // largest magnitude whose quotient still rounds to zero.
        let tiny_offsets = [
            f64::from_bits(1), // 5e-324, the smallest subnormal
            1.0e-320,
            2.0e-312,
            f64::from_bits(1 << 39),
            f64::from_bits((1 << 39) + 1),
            f64::MIN_POSITIVE,
            1.0e-300,
            1.0e-20,
            0.5 * STEP, // the largest offset that still rounds to the corner
        ];
        for tiny in tiny_offsets {
            for sector in sectors {
                for raw in [
                    GalaxyPos::new(sector, DVec3::splat(-tiny)),
                    GalaxyPos {
                        sector,
                        local: DVec3::splat(-tiny),
                    }
                    .normalized(),
                    corner(sector).offset(DVec3::splat(-tiny)),
                ] {
                    // The same sector (also at the `i64` limits, where a borrow that is taken
                    // back again must not leave a trace) and exactly +0.0.
                    assert_eq!(raw.sector, sector, "-{tiny:e}");
                    assert_eq!(bits(raw.local), [0; 3], "-{tiny:e}: {:?}", raw.local);
                    assert_eq!(raw.normalized(), raw);
                }
                // The positive counterpart is representable and stays what it is.
                let positive = GalaxyPos::new(sector, DVec3::splat(tiny));
                assert_eq!(positive.sector, sector);
                assert_eq!(bits(positive.local), bits(DVec3::splat(tiny)));
            }
        }
        assert_eq!(
            GalaxyPos::new([0; 3], DVec3::new(-5.0e-324, 0.0, -1.0e-320)),
            GalaxyPos::ORIGIN
        );
        // Negative zero is zero.
        let negative_zero = GalaxyPos::new([3, 4, 5], DVec3::splat(-0.0));
        assert_eq!(negative_zero.sector, [3, 4, 5]);
        assert_eq!(bits(negative_zero.local), [0; 3]);

        // From just over half a step on, the offset is resolved in the sector below.
        for offset in [f64::from_bits((0.5 * STEP).to_bits() + 1), STEP] {
            let below = GalaxyPos::new([5, 5, 5], DVec3::splat(-offset));
            assert_eq!(below.sector, [4, 4, 4], "-{offset:e}");
            assert_eq!(below.local, DVec3::splat(S - STEP), "-{offset:e}");
        }

        // All magnitudes up to half a step, with every binary exponent (subnormals included)
        // equally likely; every other sample comes from the range around the underflow limit.
        let mut rng = Rng::new(82, 1);
        let half_step_bits = (0.5 * STEP).to_bits();
        for i in 0..200_000 {
            let range = if i % 2 == 0 { half_step_bits } else { 1 << 40 };
            let tiny = f64::from_bits(1 + rng.next_u64() % range);
            let sector = [
                i64::from(rng.next_u32() as i32),
                i64::from(rng.next_u32() as i32),
                i64::from(rng.next_u32() as i32),
            ];
            let negative = GalaxyPos::new(sector, DVec3::splat(-tiny));
            assert_eq!(negative.sector, sector, "-{tiny:e}");
            assert_eq!(bits(negative.local), [0; 3], "-{tiny:e}");
            let positive = GalaxyPos::new(sector, DVec3::splat(tiny));
            assert_eq!(positive.sector, sector, "{tiny:e}");
            assert_eq!(bits(positive.local), bits(DVec3::splat(tiny)), "{tiny:e}");
        }

        // The representable values next to sector boundaries, on both sides. Around the
        // boundary at zero these are the subnormals.
        for k in -3_i32..=3 {
            for steps in -8_i32..=8 {
                let mut local = f64::from(k) * S;
                for _ in 0..steps.abs() {
                    local = if steps > 0 {
                        local.next_up()
                    } else {
                        local.next_down()
                    };
                }
                let raw = GalaxyPos {
                    sector: [k.into(), 0, -1],
                    local: DVec3::splat(local),
                };
                let normalized = raw.normalized();
                for axis in 0..3 {
                    let offset = normalized.local[axis];
                    assert!(
                        (0.0..S).contains(&offset) && offset.is_sign_positive(),
                        "{local:e} -> {normalized:?}"
                    );
                }
                assert_eq!(normalized.normalized(), normalized);
                // The same place, to within what a sector can resolve.
                let moved = GalaxyPos::delta(&raw, &normalized).abs().max_element();
                assert!(moved <= 0.5 * STEP, "{local:e} moved by {moved:e}");
            }
        }
    }

    #[test]
    fn galaxy_offsets_and_deltas() {
        const S: f64 = GalaxyPos::SECTOR_SIZE;
        let a = GalaxyPos::new([10, -20, 30], DVec3::new(100.0, 200.0, 300.0));
        let step = DVec3::new(1.5, -250.0, S * 3.0 + 0.25);
        let b = a.offset(step);
        assert_eq!(b.sector, [10, -21, 33]);
        assert_eq!(b.local, DVec3::new(101.5, S - 50.0, 300.25));
        assert_eq!(GalaxyPos::delta(&a, &b), step);
        assert_eq!(GalaxyPos::delta(&b, &a), -step);
        assert_eq!(GalaxyPos::delta(&a, &a), DVec3::ZERO);
        assert_eq!(a.offset(DVec3::ZERO), a);

        // Millimetre steps stay millimetre steps a hundred thousand light years from the origin,
        // where a plain f64 coordinate (about 1e21 m) could not even resolve a kilometre.
        let far = GalaxyPos::new([860_000_000, -860_000_000, 12_345], DVec3::splat(0.5 * S));
        let near = far.offset(DVec3::new(0.001, -0.002, 0.003));
        let delta = GalaxyPos::delta(&far, &near);
        assert!(
            (delta - DVec3::new(0.001, -0.002, 0.003)).length() < 2.0e-4,
            "{delta:?}"
        );
        // Walking away in many small steps and measuring back is consistent.
        let mut walker = far;
        let stride = DVec3::new(123_456.789, -S / 7.0, 0.3 * S);
        for _ in 0..1000 {
            walker = walker.offset(stride);
        }
        let travelled = GalaxyPos::delta(&far, &walker);
        assert!(
            (travelled - stride * 1000.0).length() < 1.0,
            "{travelled:?}"
        );
        assert_eq!(walker.normalized(), walker);

        // Extreme sectors: the difference is formed without overflow.
        let low = GalaxyPos {
            sector: [i64::MIN; 3],
            local: DVec3::ZERO,
        };
        let high = GalaxyPos {
            sector: [i64::MAX; 3],
            local: DVec3::ZERO,
        };
        let span = GalaxyPos::delta(&low, &high);
        assert!(span.x > 2.0e31 && span.x.is_finite());
    }
}
