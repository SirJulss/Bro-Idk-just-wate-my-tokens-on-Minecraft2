//! `cp-universe`: the celestial model of the Cubeplanet engine.
//!
//! Responsibility: star systems and their bodies as data, loaded from
//! `content/<pack>/systems/*.toml` ([`SystemDef`], [`BodyDef`]); the analytic ephemeris (Kepler
//! orbits and uniform rotation, [`SystemDef::body_frame`]); the gravity potential of cube bodies
//! and stars ([`GravityField`]); spheres of influence ([`SystemDef::dominant_body`]); and the
//! procedural star catalogue of the galaxy ([`StarCatalogue`]). The crate follows the
//! determinism rules (contract §2.5): elementary functions only through `cp_core::det`, no
//! iteration order of hash maps in any result.
//!
//! Binding contract: `engine/docs/ARCHITEKTUR.md`, sections 2.1, 2.2, 2.5, 6 and 8.
//!
//! | module | contents |
//! |---|---|
//! | [`body`] | [`BodyDef`], [`CubeDef`], the voxel lattice of a body (root level, grid offset) |
//! | [`orbit`] | [`OrbitElements`], [`KeplerOrbit`], [`solve_kepler`] |
//! | [`system`] | [`SystemDef`]: validation, hierarchy, ephemeris, spheres of influence |
//! | [`schema`] | the TOML schema of system files, [`load_systems`] |
//! | [`gravity`] | [`GravityField`]: the blended cube potential, the uniform sphere |
//! | [`catalogue`] | [`StarCatalogue`], [`StarEntry`], [`blackbody_color`] |
//!
//! # Conventions
//! - Units: metres, seconds, kilograms, radians (degrees only in system files); `f64`
//!   throughout. Time `t` is seconds since the epoch, the simulation's tick 0
//!   (`cp_core::time::Tick::seconds`).
//! - **System frame:** inertial, origin at the centre of the root star (children do not pull
//!   their parents, so the star rests at the barycentre of the model), axes parallel to the
//!   galaxy's. The reference plane (ecliptic) is the XZ plane, north is `+Y`; prograde motion
//!   runs from `+X` towards `-Z`, counter-clockwise seen from `+Y`.
//! - **Orbits** are classical elements against the reference plane (see [`OrbitElements`]),
//!   relative to the parent's centre in system axes, valid at the epoch.
//! - **Body frame:** origin at the body (cube) centre; body `+Y` is the north pole and spin
//!   axis. The orientation (body → system axes) at time `t` is
//!   `R_Z(axial tilt) · R_Y(phase + 2π t / sidereal period)` for a uniform rotation and
//!   `R_Y(Ω) · R_X(i - axial tilt) · R_Y(ω + M(t))` for a tidally locked body, which turns in
//!   its orbit's frame and keeps its `-X` face towards its parent ([`Rotation::Synchronous`]).
//! - **Galaxy:** [`cp_core::frames::GalaxyPos`]; the galactic centre at the origin, the disc in
//!   the XZ plane.

pub mod body;
pub mod catalogue;
mod document;
mod error;
#[cfg(test)]
mod fixtures;
pub mod gravity;
pub mod orbit;
pub mod schema;
pub mod system;

pub use body::{
    BodyDef, BodyKind, BodyShape, CLEARANCE_LEVELS, CubeDef, MIN_SURFACE_CLEARANCE, Rotation,
    best_grid_offset, root_extent, root_level, surface_clearance,
};
pub use catalogue::{
    CATALOGUE_VERSION, CELL_SECTOR_BITS, CELL_SIZE, HOME_CELL, HOME_EXCLUSION_RADIUS,
    HOME_POSITION, HOME_SYSTEM_ID, HOME_SYSTEM_SEED, MAX_QUERY_RADIUS, MAX_STARS_PER_CELL, PARSEC,
    SOLAR_LUMINOSITY, SOLAR_TEMPERATURE, StarCatalogue, StarEntry, blackbody_color, cell_of,
    cell_origin, split_system_id, star_density, system_id,
};
pub use error::{DefinitionError, GravityError, UniverseError};
pub use gravity::{CubeGravity, CubeGravityShape, FieldSample, GravityField, SphereGravity};
pub use orbit::{KeplerOrbit, OrbitElements, OrbitState, solve_kepler};
pub use schema::load_systems;
pub use system::{Spin, SystemDef, sphere_of_influence};
