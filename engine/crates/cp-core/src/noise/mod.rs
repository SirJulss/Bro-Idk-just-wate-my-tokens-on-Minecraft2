//! Deterministic lattice noise (contract §4.1).
//!
//! All functions take a `u64` seed and `f64` coordinates in **lattice units**: features are
//! about one unit in size, so multiply world positions by a frequency before sampling
//! (`gradient3(seed, p * 0.01)` for 100 m features). The same seed gives independent fields in
//! the different noise families.
//!
//! | function | output | notes |
//! |---|---|---|
//! | [`value2`], [`value3`] | `[-1, 1]` | random values at lattice points, C2 interpolation |
//! | [`gradient2`], [`gradient3`] | `[-1, 1]` | Perlin-style gradient noise, quintic fade, C2 |
//! | [`worley2`], [`worley3`] | `(f1, f2)` distances | cellular noise, exact nearest two points |
//! | [`Fbm`] | `[-1, 1]` | fractal sums: plain, ridged, billow, domain warp |
//!
//! Every function except the domain warp has a `*_tiled(…, period)` variant that repeats
//! exactly every `period` lattice units on every axis, for seamless textures.
//!
//! For terrain and anything else that is thresholded, shaped or looked at closely, use
//! [`Fbm`] (also with a single octave) rather than the bare functions: see *Lattice
//! artefacts* below.
//!
//! # Guarantees
//! - **Determinism.** Only `+ - * /`, `sqrt` and integer operations are used; the gradient
//!   tables are computed at compile time. Results are bit-identical on every platform and are
//!   pinned by hashes of sample grids in the tests. They must never change: generated worlds
//!   depend on them.
//! - **Range.** The documented ranges hold for all finite inputs with `|coordinate| < 2^62`.
//!   The scale factors are the exact mathematical bounds, so the final clamp only removes
//!   rounding noise and never flattens a peak. Non-finite input gives an unspecified result
//!   (possibly NaN) and never panics.
//! - **Continuity.** Value, gradient and fractal noise are C2 (continuous second derivatives;
//!   ridged and billow have the creases their definitions ask for); Worley distances are
//!   continuous.
//! - **No axis bias.** Gradients come from 256 directions spread evenly over the circle or
//!   sphere, and [`Fbm`] rotates every octave by a golden-ratio angle, so features do not line
//!   up with the lattice axes.
//!
//! # Amplitude
//! The bare functions reach the bound of 1 only when all surrounding lattice points conspire;
//! their typical values are much smaller. Standard deviations: `value2` 0.45, `value3` 0.40,
//! `gradient2` 0.305, `gradient3` 0.22. [`Fbm`] stretches its octaves to use the range: one
//! octave has a standard deviation of 0.44 (2D) and 0.42 (3D).
//!
//! # Lattice artefacts
//! Plain gradient noise is exactly 0 at every lattice point (integer coordinates) and its
//! variance grows from there to the middle of each cell. Its zero set therefore passes through
//! a regular grid, which shows as soon as the noise is thresholded (coastlines) or folded
//! (ridges). Plain value noise has zero slope at every lattice point and looks blocky.
//! [`gradient2`]/[`gradient3`] and [`value2`]/[`value3`] are these classic functions,
//! unchanged, for uses where that does not matter. [`Fbm`] is built on *value-gradient noise*
//! instead, which gives every lattice point both a random value and a random gradient and has
//! neither artefact.
//!
//! ```
//! use cp_core::math::{DVec2, DVec3};
//! use cp_core::noise::{Fbm, worley3};
//!
//! let seed = 42;
//! // Terrain height in [-1, 1] with 400 m base features, five octaves.
//! let terrain = Fbm::new(5, 2.0, 0.5);
//! let position = DVec2::new(1250.0, -730.0);
//! let height = terrain.sample2(seed, position / 400.0);
//! assert!((-1.0..=1.0).contains(&height));
//! // The same call always returns the same bits.
//! assert_eq!(height.to_bits(), terrain.sample2(seed, position / 400.0).to_bits());
//!
//! // Mountain ridges, bent by a slow domain warp.
//! let warped = terrain.domain_warp2(seed + 1, position, 1.0 / 900.0, 150.0);
//! let ridges = terrain.ridged2(seed + 2, warped / 600.0);
//! assert!((-1.0..=1.0).contains(&ridges));
//!
//! // Cell borders of 3D cellular noise with 8 m cells.
//! let (f1, f2) = worley3(seed, DVec3::new(10.5, -3.25, 99.0) / 8.0);
//! assert!(0.0 <= f1 && f1 <= f2);
//! ```

mod basis;
mod fbm;
mod lattice;
mod worley;

#[cfg(test)]
mod tests;

pub use fbm::Fbm;
pub use worley::{worley2, worley2_tiled, worley3, worley3_tiled};

use crate::math::{DVec2, DVec3};
use basis::{gradient2_on, gradient3_on, value2_on, value3_on};
use lattice::{Tiled, Unbounded};

/// 2D value noise in `[-1, 1]`: uniformly random values at the lattice points, blended with
/// the quintic fade. Blocky; good for masks and cheap variation. Standard deviation 0.45.
#[inline]
#[must_use]
pub fn value2(seed: u64, p: DVec2) -> f64 {
    value2_on(seed, p, Unbounded)
}

/// [`value2`] that repeats every `period` lattice units on both axes (`period >= 1`, zero is
/// treated as 1).
#[inline]
#[must_use]
pub fn value2_tiled(seed: u64, p: DVec2, period: u32) -> f64 {
    value2_on(seed, p, Tiled::new(u64::from(period)))
}

/// 3D value noise in `[-1, 1]`, see [`value2`]. Standard deviation 0.40.
#[inline]
#[must_use]
pub fn value3(seed: u64, p: DVec3) -> f64 {
    value3_on(seed, p, Unbounded)
}

/// [`value3`] that repeats every `period` lattice units on all axes (`period >= 1`, zero is
/// treated as 1).
#[inline]
#[must_use]
pub fn value3_tiled(seed: u64, p: DVec3, period: u32) -> f64 {
    value3_on(seed, p, Tiled::new(u64::from(period)))
}

/// 2D gradient noise in `[-1, 1]` (Perlin-style, quintic fade, C2-continuous).
///
/// Zero at lattice points (see the [module documentation](self)); standard deviation 0.305.
#[inline]
#[must_use]
pub fn gradient2(seed: u64, p: DVec2) -> f64 {
    gradient2_on(seed, p, Unbounded)
}

/// [`gradient2`] that repeats every `period` lattice units on both axes (`period >= 1`, zero
/// is treated as 1).
#[inline]
#[must_use]
pub fn gradient2_tiled(seed: u64, p: DVec2, period: u32) -> f64 {
    gradient2_on(seed, p, Tiled::new(u64::from(period)))
}

/// 3D gradient noise in `[-1, 1]` (Perlin-style, quintic fade, C2-continuous).
///
/// Zero at lattice points (see the [module documentation](self)); standard deviation 0.22.
#[inline]
#[must_use]
pub fn gradient3(seed: u64, p: DVec3) -> f64 {
    gradient3_on(seed, p, Unbounded)
}

/// [`gradient3`] that repeats every `period` lattice units on all axes (`period >= 1`, zero
/// is treated as 1).
#[inline]
#[must_use]
pub fn gradient3_tiled(seed: u64, p: DVec3, period: u32) -> f64 {
    gradient3_on(seed, p, Tiled::new(u64::from(period)))
}
