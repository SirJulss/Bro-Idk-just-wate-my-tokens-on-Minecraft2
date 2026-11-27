//! A star system: its bodies and their hierarchy, the analytic ephemeris and the spheres of
//! influence.
//!
//! # Frames
//! The *system frame* is inertial with its origin at the centre of the root star (see the crate
//! documentation for the axes). Every other body moves on a Kepler orbit around its parent's
//! centre and turns uniformly about its tilted spin axis:
//!
//! - [`SystemDef::body_frame`] gives a body's [`BodyFrame`] relative to the system frame:
//!   position, velocity, orientation (body → system axes) and angular velocity;
//! - [`SystemDef::frame_in_parent`] gives it relative to the parent's body-fixed, rotating
//!   frame ([`FrameId::Body`] of the parent; the system frame for the root), so that
//!   `body_frame(child) = BodyFrame::compose(&body_frame(parent), &frame_in_parent(child))`.
//!
//! Children do not pull their parents (the rails of a hierarchical Kepler model), so the root
//! star rests at the origin.

use core::f64::consts::{FRAC_PI_2, TAU};

use cp_core::det;
use cp_core::frames::BodyFrame;
use cp_core::ids::{BodyId, FrameId};
use cp_core::math::{DQuat, DVec3};

use crate::body::{BodyDef, BodyKind, BodyShape, MAX_ROOT_LEVEL, MIN_SURFACE_CLEARANCE, Rotation};
use crate::body::{best_grid_offset, root_extent};
use crate::error::DefinitionError;
use crate::gravity::GravityField;
use crate::orbit::{KeplerOrbit, max_equation_of_centre};

/// Uniform rotation of a body about its tilted spin axis, resolved from its [`Rotation`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Spin {
    /// Angular speed in radians per second, `> 0`.
    pub rate: f64,
    /// Rotation angle about the spin axis at the epoch, radians: the configured phase of a
    /// uniform rotation; `ω + M0` (argument of periapsis plus mean anomaly at the epoch) for a
    /// synchronous one.
    pub phase: f64,
    /// The tilt: rotation from the body axes at rotation angle 0 to the system axes, which
    /// carries body `+Y` to the spin axis. For a uniform rotation `R_Z(axial tilt)`, a rotation
    /// about system `+Z`. For a synchronous rotation `R_Y(Ω) · R_X(i - axial tilt)`: the frame
    /// of the orbit (body `+X` to the ascending node, `+Y` to the orbit normal) with the pole
    /// tilted about the line of nodes.
    pub tilt: DQuat,
}

impl Spin {
    /// The spin axis (body `+Y`, the north pole) in system axes.
    #[must_use]
    pub fn axis(&self) -> DVec3 {
        self.tilt * DVec3::Y
    }

    /// Sidereal rotation period in seconds.
    #[must_use]
    pub fn period(&self) -> f64 {
        TAU / self.rate
    }

    /// Rotation angle about the spin axis at time `t` (seconds since the epoch), in
    /// `[0, 2π)`.
    #[must_use]
    pub fn angle(&self, t: f64) -> f64 {
        (self.phase + self.rate * t).rem_euclid(TAU)
    }

    /// Orientation at time `t`: rotation from body axes to system axes,
    /// `tilt · R_Y(angle(t))`.
    #[must_use]
    pub fn orientation(&self, t: f64) -> DQuat {
        (self.tilt * det::dquat_from_axis_angle(DVec3::Y, self.angle(t))).normalize()
    }

    /// Angular velocity in system axes, radians per second.
    #[must_use]
    pub fn angular_velocity(&self) -> DVec3 {
        self.axis() * self.rate
    }
}

/// What the system derives from each body once.
#[derive(Clone, Debug, PartialEq)]
struct Derived {
    orbit: Option<KeplerOrbit>,
    spin: Spin,
    sphere_of_influence: f64,
    gravity: GravityField,
    children: Vec<BodyId>,
}

/// A validated star system (contract §6): a root star, planets around it and moons around
/// the planets, with their orbits and rotations.
///
/// Immutable after construction. [`BodyId`]s are indices into [`SystemDef::bodies`]; parents
/// come before their children.
#[derive(Clone, Debug, PartialEq)]
pub struct SystemDef {
    key: String,
    display_name: String,
    bodies: Vec<BodyDef>,
    derived: Vec<Derived>,
}

/// `namespace:name`, both parts made of `a-z`, `0-9` and `_` (contract §4.4).
pub(crate) fn is_valid_key(key: &str) -> bool {
    let ident = |part: &str| {
        !part.is_empty()
            && part
                .bytes()
                .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'_')
    };
    key.split_once(':')
        .is_some_and(|(namespace, name)| ident(namespace) && ident(name))
}

fn key_message(key: &str) -> String {
    format!(
        "`{key}` is not a valid key: keys have the form `namespace:name`, both parts made of \
         `a-z`, `0-9` and `_`"
    )
}

/// Laplace's sphere of influence `a (mu_body / mu_parent)^(2/5)` in metres: inside it the
/// body's own gravity dominates the parent's perturbation.
#[must_use]
pub fn sphere_of_influence(semi_major_axis: f64, mu_body: f64, mu_parent: f64) -> f64 {
    semi_major_axis * det::powf(mu_body / mu_parent, 0.4)
}

fn degrees(radians: f64) -> f64 {
    radians * (180.0 / core::f64::consts::PI)
}

impl SystemDef {
    /// Builds and validates a system from its bodies.
    ///
    /// `bodies[i]` gets the id `BodyId(i)`. Rules: keys follow the content key grammar and are
    /// unique; exactly one body, a star (a sphere), has no parent; planets (cubes) orbit the
    /// star, moons (cubes) orbit planets; parents come before their children; numbers are
    /// finite and in range; a tidally locked body keeps its `-X` face towards its parent (its
    /// axial tilt plus the largest equation of the centre of its orbit stays below 90°, see
    /// [`Rotation::Synchronous`]); every cube fits into its octree root set together with its grid
    /// offset and atmosphere, and its grid offset keeps the mean surface clear of chunk
    /// boundaries ([`MIN_SURFACE_CLEARANCE`]); orbits keep clear of their parent, every body
    /// fits into its own sphere of influence, a moon's sphere of influence stays inside its
    /// planet's, and the orbits of siblings, widened by their spheres of influence, do not
    /// overlap.
    ///
    /// # Errors
    /// The first broken rule, naming the body by its index and the field like its TOML key.
    pub fn new(
        key: impl Into<String>,
        display_name: impl Into<String>,
        bodies: Vec<BodyDef>,
    ) -> Result<Self, DefinitionError> {
        let key = key.into();
        let display_name = display_name.into();
        if !is_valid_key(&key) {
            return Err(DefinitionError::system("key", key_message(&key)));
        }
        if display_name.trim().is_empty() {
            return Err(DefinitionError::system("display_name", "must not be empty"));
        }
        if bodies.is_empty() {
            return Err(DefinitionError::system(
                "body",
                "a system needs at least one body, its star",
            ));
        }
        if u32::try_from(bodies.len()).is_err() {
            return Err(DefinitionError::system("body", "too many bodies"));
        }
        let mut derived: Vec<Derived> = Vec::with_capacity(bodies.len());
        for index in 0..bodies.len() {
            check_body(index, &bodies)?;
            derived.push(derive(index, &bodies)?);
        }
        for index in 0..bodies.len() {
            check_orbit_geometry(index, &bodies, &derived)?;
        }
        for (index, body) in bodies.iter().enumerate() {
            if let Some(parent) = body.parent {
                derived[parent.index()].children.push(body_id(index));
            }
        }
        Ok(Self {
            key,
            display_name,
            bodies,
            derived,
        })
    }

    /// Stable key of the system, `namespace:name`.
    #[must_use]
    pub fn key(&self) -> &str {
        &self.key
    }

    /// Name shown to players.
    #[must_use]
    pub fn display_name(&self) -> &str {
        &self.display_name
    }

    /// All bodies; `bodies()[i]` has the id `BodyId(i)`.
    #[must_use]
    pub fn bodies(&self) -> &[BodyDef] {
        &self.bodies
    }

    /// Number of bodies.
    #[must_use]
    pub fn body_count(&self) -> usize {
        self.bodies.len()
    }

    /// The ids of all bodies in order (parents before children).
    pub fn ids(&self) -> impl ExactSizeIterator<Item = BodyId> + '_ {
        (0..self.bodies.len()).map(body_id)
    }

    /// The body with this id, if it exists.
    #[must_use]
    pub fn get(&self, id: BodyId) -> Option<&BodyDef> {
        self.bodies.get(id.index())
    }

    /// The body with this id.
    ///
    /// # Panics
    /// If `id` is not an id of this system.
    #[must_use]
    pub fn body(&self, id: BodyId) -> &BodyDef {
        &self.bodies[id.index()]
    }

    /// The id of the body with this key.
    #[must_use]
    pub fn find(&self, key: &str) -> Option<BodyId> {
        self.bodies
            .iter()
            .position(|body| body.key == key)
            .map(body_id)
    }

    /// The root star.
    #[must_use]
    pub fn root(&self) -> BodyId {
        let index = self
            .bodies
            .iter()
            .position(|body| body.parent.is_none())
            .expect("a validated system has exactly one root");
        body_id(index)
    }

    /// The bodies that orbit `id`, in id order.
    ///
    /// # Panics
    /// If `id` is not an id of this system.
    #[must_use]
    pub fn children(&self, id: BodyId) -> &[BodyId] {
        &self.derived[id.index()].children
    }

    /// The reference frame a body's coordinates are given in: [`FrameId::Body`].
    #[must_use]
    pub fn frame_id(&self, id: BodyId) -> FrameId {
        FrameId::Body(id)
    }

    /// The orbit of `id` around its parent; `None` for the root.
    ///
    /// # Panics
    /// If `id` is not an id of this system.
    #[must_use]
    pub fn orbit(&self, id: BodyId) -> Option<&KeplerOrbit> {
        self.derived[id.index()].orbit.as_ref()
    }

    /// The resolved rotation of `id`.
    ///
    /// # Panics
    /// If `id` is not an id of this system.
    #[must_use]
    pub fn spin(&self, id: BodyId) -> &Spin {
        &self.derived[id.index()].spin
    }

    /// Radius of the sphere of influence of `id` in metres (Laplace, around the body's centre);
    /// infinite for the root.
    ///
    /// # Panics
    /// If `id` is not an id of this system.
    #[must_use]
    pub fn sphere_of_influence(&self, id: BodyId) -> f64 {
        self.derived[id.index()].sphere_of_influence
    }

    /// The gravity field of `id` in its body frame.
    ///
    /// # Panics
    /// If `id` is not an id of this system.
    #[must_use]
    pub fn gravity(&self, id: BodyId) -> &GravityField {
        &self.derived[id.index()].gravity
    }

    /// Mean interval between two noons on `id` in seconds: the synodic rotation period against
    /// the star, `1 / |1/P ∓ 1/Y|` with the sidereal period `P` and the orbital period `Y` of
    /// the planet the body belongs to (minus when the spin is prograde against that orbit).
    /// `None` for the root and for a body that keeps one face to the star.
    ///
    /// # Panics
    /// If `id` is not an id of this system.
    #[must_use]
    pub fn mean_solar_day(&self, id: BodyId) -> Option<f64> {
        let root = self.root();
        let mut planet = id;
        loop {
            match self.bodies[planet.index()].parent {
                None => return None,
                Some(parent) if parent == root => break,
                Some(parent) => planet = parent,
            }
        }
        let orbit = self.orbit(planet)?;
        let spin = self.spin(id);
        let sense = if spin.axis().dot(orbit.normal()) >= 0.0 {
            1.0
        } else {
            -1.0
        };
        let rate = spin.rate - sense * orbit.mean_motion();
        (rate != 0.0).then(|| TAU / rate.abs())
    }

    /// The frame of `id` relative to the system frame at time `t` (seconds since the epoch).
    ///
    /// # Panics
    /// If `id` is not an id of this system.
    #[must_use]
    pub fn body_frame(&self, id: BodyId, t: f64) -> BodyFrame {
        let spin = &self.derived[id.index()].spin;
        let own = BodyFrame {
            position: DVec3::ZERO,
            orientation: spin.orientation(t),
            velocity: DVec3::ZERO,
            angular_velocity: spin.angular_velocity(),
        };
        match (
            self.bodies[id.index()].parent,
            &self.derived[id.index()].orbit,
        ) {
            (Some(parent), Some(orbit)) => {
                let base = self.body_frame(parent, t);
                let state = orbit.state(t);
                BodyFrame {
                    position: base.position + state.position,
                    velocity: base.velocity + state.velocity,
                    ..own
                }
            }
            _ => own,
        }
    }

    /// The frames of all bodies relative to the system frame at time `t`, indexed by id; each
    /// equals [`SystemDef::body_frame`] bit for bit.
    #[must_use]
    pub fn frames(&self, t: f64) -> Vec<BodyFrame> {
        let mut frames: Vec<BodyFrame> = Vec::with_capacity(self.bodies.len());
        for index in 0..self.bodies.len() {
            let spin = &self.derived[index].spin;
            let mut frame = BodyFrame {
                position: DVec3::ZERO,
                orientation: spin.orientation(t),
                velocity: DVec3::ZERO,
                angular_velocity: spin.angular_velocity(),
            };
            if let (Some(parent), Some(orbit)) =
                (self.bodies[index].parent, &self.derived[index].orbit)
            {
                let base = frames[parent.index()];
                let state = orbit.state(t);
                frame.position = base.position + state.position;
                frame.velocity = base.velocity + state.velocity;
            }
            frames.push(frame);
        }
        frames
    }

    /// The frame of `id` relative to its parent's body-fixed, rotating frame at time `t`; for
    /// the root, relative to the system frame. Composing it with the parent's
    /// [`SystemDef::body_frame`] gives the body's.
    ///
    /// # Panics
    /// If `id` is not an id of this system.
    #[must_use]
    pub fn frame_in_parent(&self, id: BodyId, t: f64) -> BodyFrame {
        let (Some(parent), Some(orbit)) = (
            self.bodies[id.index()].parent,
            &self.derived[id.index()].orbit,
        ) else {
            return self.body_frame(id, t);
        };
        let base = self.body_frame(parent, t);
        let spin = &self.derived[id.index()].spin;
        let state = orbit.state(t);
        let inverse = base.orientation.conjugate();
        BodyFrame {
            position: inverse * state.position,
            orientation: (inverse * spin.orientation(t)).normalize(),
            velocity: inverse * (state.velocity - base.angular_velocity.cross(state.position)),
            angular_velocity: inverse * (spin.angular_velocity() - base.angular_velocity),
        }
    }

    /// The body whose gravity dominates at `position` (system frame, metres) at time `t`: the
    /// deepest body whose sphere of influence contains the point, descending from the root.
    /// Among siblings that both contain it (validated systems have none) the one with the
    /// smaller distance relative to its sphere wins, ties to the smaller id.
    #[must_use]
    pub fn dominant_body(&self, position: DVec3, t: f64) -> BodyId {
        self.dominant_body_in(position, &self.frames(t))
    }

    /// [`SystemDef::dominant_body`] with the frames of [`SystemDef::frames`] already at hand.
    ///
    /// # Panics
    /// If `frames` does not hold one frame per body.
    #[must_use]
    pub fn dominant_body_in(&self, position: DVec3, frames: &[BodyFrame]) -> BodyId {
        assert_eq!(frames.len(), self.bodies.len(), "one frame per body");
        let mut current = self.root();
        loop {
            let mut best: Option<(f64, BodyId)> = None;
            for &child in self.children(current) {
                let radius = self.sphere_of_influence(child);
                let distance = (position - frames[child.index()].position).length();
                if distance < radius {
                    let ratio = distance / radius;
                    if best.is_none_or(|(best_ratio, _)| ratio < best_ratio) {
                        best = Some((ratio, child));
                    }
                }
            }
            match best {
                Some((_, child)) => current = child,
                None => return current,
            }
        }
    }
}

fn body_id(index: usize) -> BodyId {
    BodyId(u32::try_from(index).expect("the body count was checked to fit into u32"))
}

/// The rules that concern one body on its own and its relation to its parent.
fn check_body(index: usize, bodies: &[BodyDef]) -> Result<(), DefinitionError> {
    let body = &bodies[index];
    let error = |field: &'static str, message: String| DefinitionError::body(index, field, message);
    if !is_valid_key(&body.key) {
        return Err(error("key", key_message(&body.key)));
    }
    if let Some(first) = bodies[..index]
        .iter()
        .position(|other| other.key == body.key)
    {
        return Err(error(
            "key",
            format!("`{}` is defined twice (first as body #{first})", body.key),
        ));
    }
    if body.display_name.trim().is_empty() {
        return Err(error("display_name", "must not be empty".to_owned()));
    }
    if !(body.surface_gravity.is_finite() && body.surface_gravity > 0.0) {
        return Err(error(
            "surface_gravity_m_s2",
            format!("must be positive, found {}", body.surface_gravity),
        ));
    }
    if !(0.0..=core::f64::consts::PI).contains(&body.axial_tilt) {
        return Err(error(
            "axial_tilt_deg",
            format!(
                "must be between 0 and 180 degrees, found {}",
                degrees(body.axial_tilt)
            ),
        ));
    }
    match body.rotation {
        Rotation::Uniform { period, phase } => {
            if !(period.is_finite() && period > 0.0) {
                return Err(error(
                    "rotation_period_s",
                    format!("must be positive, found {period}"),
                ));
            }
            if !phase.is_finite() {
                return Err(error(
                    "rotation_phase_deg",
                    format!("must be finite, found {phase}"),
                ));
            }
        }
        Rotation::Synchronous if body.parent.is_none() => {
            return Err(error(
                "tidally_locked",
                "only a body that orbits another one can be tidally locked".to_owned(),
            ));
        }
        Rotation::Synchronous => {}
    }
    check_shape(index, body)?;
    check_hierarchy(index, bodies)?;
    check_elements(index, body)?;
    check_lock(index, body)
}

/// A tidally locked body must keep its `-X` face towards its parent: the direction to the
/// parent stays within the axial tilt (libration in latitude) plus the largest equation of the
/// centre (libration in longitude) of the face normal, and that must stay below 90°.
fn check_lock(index: usize, body: &BodyDef) -> Result<(), DefinitionError> {
    let (Rotation::Synchronous, Some(elements)) = (body.rotation, &body.orbit) else {
        return Ok(());
    };
    let longitude = max_equation_of_centre(elements.eccentricity);
    if body.axial_tilt + longitude < FRAC_PI_2 {
        return Ok(());
    }
    Err(DefinitionError::body(
        index,
        "tidally_locked",
        format!(
            "a tidally locked body keeps its -X face towards its parent only while its axial \
             tilt (measured from the orbit normal, {:.1} degrees) plus the largest equation of \
             the centre of its orbit ({:.1} degrees at eccentricity {}) stays below 90 degrees",
            degrees(body.axial_tilt),
            degrees(longitude),
            elements.eccentricity
        ),
    ))
}

fn check_shape(index: usize, body: &BodyDef) -> Result<(), DefinitionError> {
    let error = |field: &'static str, message: String| DefinitionError::body(index, field, message);
    match (&body.shape, body.kind) {
        (BodyShape::Sphere { radius }, BodyKind::Star) => {
            if !(radius.is_finite() && *radius > 0.0) {
                return Err(error(
                    "radius_m",
                    format!("must be positive, found {radius}"),
                ));
            }
            Ok(())
        }
        (BodyShape::Cube(cube), BodyKind::Planet | BodyKind::Moon) => {
            if cube.edge < 2 || !cube.edge.is_multiple_of(2) {
                return Err(error(
                    "edge_m",
                    format!(
                        "must be an even number of metres, at least 2, found {}",
                        cube.edge
                    ),
                ));
            }
            let level = cube.root_level();
            if level > MAX_ROOT_LEVEL {
                return Err(error(
                    "edge_m",
                    format!(
                        "is too large for the octree: root level {level}, at most \
                         {MAX_ROOT_LEVEL}"
                    ),
                ));
            }
            let half_edge = cube.half_edge();
            if !(cube.sea_level.is_finite() && cube.sea_level.abs() < half_edge) {
                return Err(error(
                    "sea_level_m",
                    format!(
                        "must be finite and closer to the reference surface than the half edge \
                         ({half_edge} m), found {}",
                        cube.sea_level
                    ),
                ));
            }
            if !(cube.atmosphere_height.is_finite() && cube.atmosphere_height >= 0.0) {
                return Err(error(
                    "atmosphere_height_m",
                    format!("must be at least 0, found {}", cube.atmosphere_height),
                ));
            }
            if !is_valid_key(&cube.generator) {
                return Err(error("generator", key_message(&cube.generator)));
            }
            let extent = root_extent(level);
            let offset = f64::from(cube.grid_offset.unsigned_abs());
            if half_edge + offset + cube.atmosphere_height > extent {
                return Err(error(
                    "grid_offset",
                    format!(
                        "the cube (half edge {half_edge} m) shifted by the grid offset \
                         ({} voxels) and topped by its atmosphere ({} m) does not fit into its \
                         root chunks, which reach {extent} m from the lattice origin at root \
                         level {level}",
                        cube.grid_offset, cube.atmosphere_height
                    ),
                ));
            }
            let clearance = cube.surface_clearance();
            if clearance < MIN_SURFACE_CLEARANCE {
                let room = (extent - half_edge - cube.atmosphere_height).floor();
                let room = u32::try_from(room as i64).unwrap_or(0).min(8192);
                let best = best_grid_offset(cube.edge, cube.sea_level, room);
                let best_clearance =
                    crate::body::surface_clearance(cube.edge, cube.sea_level, best);
                return Err(error(
                    "grid_offset",
                    format!(
                        "{} keeps the mean surface only {:.1} % of a chunk from a chunk \
                         boundary on levels 4 to 8 (at least {:.1} % required); {best} keeps \
                         it {:.1} % away",
                        cube.grid_offset,
                        100.0 * clearance,
                        100.0 * MIN_SURFACE_CLEARANCE,
                        100.0 * best_clearance
                    ),
                ));
            }
            Ok(())
        }
        (_, kind) => Err(error(
            "kind",
            format!(
                "a {} is {}; stars are spheres (`radius_m`), planets and moons are cubes \
                 (`edge_m`)",
                kind.name(),
                match body.shape {
                    BodyShape::Sphere { .. } => "given a radius",
                    BodyShape::Cube(_) => "given an edge",
                }
            ),
        )),
    }
}

fn check_hierarchy(index: usize, bodies: &[BodyDef]) -> Result<(), DefinitionError> {
    let body = &bodies[index];
    let error = |field: &'static str, message: String| DefinitionError::body(index, field, message);
    match (body.kind, body.parent) {
        (BodyKind::Star, None) => {
            if let Some(root) = bodies[..index].iter().find(|other| other.parent.is_none()) {
                return Err(error(
                    "parent",
                    format!(
                        "only one body may orbit nothing; `{}` already is the root",
                        root.key
                    ),
                ));
            }
        }
        (BodyKind::Star, Some(_)) => {
            return Err(error(
                "parent",
                "a star orbits nothing (multiple stars are not supported)".to_owned(),
            ));
        }
        (_, None) => {
            return Err(error(
                "parent",
                "missing: planets orbit a star, moons orbit a planet".to_owned(),
            ));
        }
        (kind, Some(parent)) => {
            let parent_index = parent.index();
            if parent_index == index {
                return Err(error("parent", "a body cannot orbit itself".to_owned()));
            }
            let Some(parent_body) = bodies.get(parent_index) else {
                return Err(error("parent", format!("there is no body #{parent_index}")));
            };
            if parent_index > index {
                return Err(error(
                    "parent",
                    format!(
                        "`{}` must be defined before `{}`, which orbits it",
                        parent_body.key, body.key
                    ),
                ));
            }
            let expected = if kind == BodyKind::Planet {
                BodyKind::Star
            } else {
                BodyKind::Planet
            };
            if parent_body.kind != expected {
                return Err(error(
                    "parent",
                    format!(
                        "a {} orbits a {}, but `{}` is a {}",
                        kind.name(),
                        expected.name(),
                        parent_body.key,
                        parent_body.kind.name()
                    ),
                ));
            }
        }
    }
    Ok(())
}

fn check_elements(index: usize, body: &BodyDef) -> Result<(), DefinitionError> {
    let error = |field: &'static str, message: String| DefinitionError::body(index, field, message);
    let Some(elements) = &body.orbit else {
        return if body.parent.is_some() {
            Err(error(
                "orbit",
                "missing: every body but the star needs an orbit".to_owned(),
            ))
        } else {
            Ok(())
        };
    };
    if body.parent.is_none() {
        return Err(error(
            "orbit",
            "the star at the root orbits nothing; remove its orbit".to_owned(),
        ));
    }
    let a = elements.semi_major_axis;
    if !(a.is_finite() && a > 0.0) {
        return Err(error(
            "orbit.semi_major_axis_m",
            format!("must be positive, found {a}"),
        ));
    }
    let e = elements.eccentricity;
    if !(0.0..1.0).contains(&e) {
        return Err(error(
            "orbit.eccentricity",
            format!("must be at least 0 and below 1 (closed orbits only), found {e}"),
        ));
    }
    if !(0.0..=core::f64::consts::PI).contains(&elements.inclination) {
        return Err(error(
            "orbit.inclination_deg",
            format!(
                "must be between 0 and 180 degrees, found {}",
                degrees(elements.inclination)
            ),
        ));
    }
    for (field, value) in [
        ("orbit.ascending_node_deg", elements.ascending_node),
        ("orbit.periapsis_arg_deg", elements.periapsis_argument),
        ("orbit.mean_anomaly_deg", elements.mean_anomaly_at_epoch),
    ] {
        if !value.is_finite() {
            return Err(error(field, format!("must be finite, found {value}")));
        }
    }
    Ok(())
}

/// Orbit, spin, sphere of influence and gravity of body `index`, whose rules are checked.
fn derive(index: usize, bodies: &[BodyDef]) -> Result<Derived, DefinitionError> {
    let body = &bodies[index];
    let mu = body.mu();
    let (orbit, sphere_of_influence) = match (body.parent, &body.orbit) {
        (Some(parent), Some(elements)) => {
            let mu_parent = bodies[parent.index()].mu();
            let orbit = KeplerOrbit::new(*elements, mu_parent + mu);
            let radius = sphere_of_influence(elements.semi_major_axis, mu, mu_parent);
            (Some(orbit), radius)
        }
        _ => (None, f64::INFINITY),
    };
    let spin = match (body.rotation, &orbit) {
        (Rotation::Uniform { period, phase }, _) => Spin {
            rate: TAU / period,
            phase,
            tilt: det::dquat_from_axis_angle(DVec3::Z, body.axial_tilt),
        },
        (Rotation::Synchronous, Some(orbit)) => synchronous_spin(orbit, body.axial_tilt),
        (Rotation::Synchronous, None) => unreachable!("check_body rejects a locked root"),
    };
    let gravity = GravityField::for_body(body)
        .map_err(|error| DefinitionError::body(index, "surface_gravity_m_s2", error.to_string()))?;
    Ok(Derived {
        orbit,
        spin,
        sphere_of_influence,
        gravity,
        children: Vec::new(),
    })
}

/// The spin of a body tidally locked on `orbit` (see [`Rotation::Synchronous`]).
///
/// `R_Y(Ω) · R_X(i)` carries body `+X` to the ascending node and body `+Y` to the orbit
/// normal; turning it further by the argument of latitude `u` about `+Y` carries `+X` to the
/// direction of the point of the orbit at `u` (the orbit's own construction, see
/// [`KeplerOrbit::new`]). The body turns by `ω + M(t)` with the mean anomaly, so on a circular
/// orbit `-X` points exactly at the parent and the eccentricity only adds the equation of the
/// centre. The axial tilt turns the pole about the line of nodes as well, from the orbit
/// normal towards the orbit's northernmost point, so both rotations about `X` merge.
fn synchronous_spin(orbit: &KeplerOrbit, axial_tilt: f64) -> Spin {
    let elements = orbit.elements();
    let node = det::dquat_from_axis_angle(DVec3::Y, elements.ascending_node);
    let pole = det::dquat_from_axis_angle(DVec3::X, elements.inclination - axial_tilt);
    Spin {
        rate: orbit.mean_motion(),
        phase: (elements.periapsis_argument + elements.mean_anomaly_at_epoch).rem_euclid(TAU),
        tilt: (node * pole).normalize(),
    }
}

/// Rules that need the derived orbits and spheres of influence.
fn check_orbit_geometry(
    index: usize,
    bodies: &[BodyDef],
    derived: &[Derived],
) -> Result<(), DefinitionError> {
    let body = &bodies[index];
    let (Some(parent), Some(orbit)) = (body.parent, &derived[index].orbit) else {
        return Ok(());
    };
    let parent_body = &bodies[parent.index()];
    let soi = derived[index].sphere_of_influence;
    let reach = parent_body.bounding_radius() + parent_body.atmosphere_height();
    let own = body.bounding_radius() + body.atmosphere_height();
    let periapsis = orbit.periapsis_distance();
    if periapsis <= reach + own {
        return Err(DefinitionError::body(
            index,
            "orbit.semi_major_axis_m",
            format!(
                "the periapsis ({periapsis:.0} m from the centre of `{}`) is too close: `{}` \
                 reaches {reach:.0} m with its corners and atmosphere, this body {own:.0} m",
                parent_body.key, parent_body.key
            ),
        ));
    }
    if own >= soi {
        return Err(DefinitionError::body(
            index,
            "orbit.semi_major_axis_m",
            format!(
                "the body ({own:.0} m with its corners and atmosphere) does not fit into its own \
                 sphere of influence ({soi:.0} m); move it further out or make it heavier"
            ),
        ));
    }
    let parent_soi = derived[parent.index()].sphere_of_influence;
    if orbit.apoapsis_distance() + soi > parent_soi {
        return Err(DefinitionError::body(
            index,
            "orbit.semi_major_axis_m",
            format!(
                "the apoapsis ({:.0} m) plus the sphere of influence ({soi:.0} m) leaves the \
                 sphere of influence of `{}` ({parent_soi:.0} m)",
                orbit.apoapsis_distance(),
                parent_body.key
            ),
        ));
    }
    let range = |i: usize| -> (f64, f64) {
        let orbit = derived[i].orbit.as_ref().expect("siblings have orbits");
        let soi = derived[i].sphere_of_influence;
        (
            orbit.periapsis_distance() - soi,
            orbit.apoapsis_distance() + soi,
        )
    };
    let (low, high) = range(index);
    let earlier_siblings = bodies[..index]
        .iter()
        .enumerate()
        .filter(|(_, other)| other.parent == Some(parent));
    for (sibling, other) in earlier_siblings {
        let (other_low, other_high) = range(sibling);
        if low < other_high && other_low < high {
            return Err(DefinitionError::body(
                index,
                "orbit",
                format!(
                    "the orbit of `{}` widened by its sphere of influence ({:.0}–{:.0} km from \
                     `{}`) overlaps that of `{}` ({:.0}–{:.0} km)",
                    body.key,
                    low / 1000.0,
                    high / 1000.0,
                    parent_body.key,
                    other.key,
                    other_low / 1000.0,
                    other_high / 1000.0
                ),
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests;
