//! `cp-core`: the foundation every other Cubeplanet engine crate builds on.
//!
//! Responsibility: maths, ids, coordinates, faces, time, hashing/RNG, noise, reference frames,
//! block/quad wire types and profiling. It depends on no other engine crate and follows the
//! determinism rules.
//!
//! Binding contract: `engine/docs/ARCHITEKTUR.md`, sections 2 and 4.1.
//!
//! | module | contents |
//! |---|---|
//! | [`math`] | `glam` re-export, floor division, boxes, frustum, projection, Morton codes |
//! | [`ids`] | typed ids of block states, materials, bodies, systems, entities, frames, grids; grid keys |
//! | [`coords`] | chunk constants, [`coords::LodKey`], chunk and cell keys, index conversions |
//! | [`face`] | [`face::Face`], tangent frames, the 24 [`face::Orientation`]s, texture rotation |
//! | [`block`] | per-state flag, render and LOD tables consumed by storage and meshing |
//! | [`quad`] | [`quad::PackedQuad`], the 8-byte mesh primitive shared with the shaders |
//! | [`time`] | fixed simulation ticks and the frame clock |
//! | [`hash`] | SplitMix/FNV hashing and the PCG32 random number generator |
//! | [`det`] | deterministic `sin`, `cos`, `exp`, `powi`, … backed by `libm`; `min`, `max`, `clamp` |
//! | [`noise`] | value, gradient and Worley noise, fractal sums, tileable variants |
//! | [`frames`] | moving body frames and galaxy positions |
//! | [`profile`] | CPU profiling scopes |
//!
//! Wire contracts shared with GPU shaders and saved data live in [`coords`], [`face`] and
//! [`quad`]; their layouts and tables are pinned by tests and must not change silently. The
//! texture rule on negative faces is the one contract v2 §2.3 adopted from this crate:
//! [`face::Face::uv_to_tex`] is normative, [`face::UV_TO_TEX`] is the same rule as a table.
//!
//! # Determinism of the maths
//! Deterministic crates use [`det`] for elementary functions, including `min`, `max` and
//! `clamp`, whose `std` versions leave the sign of a zero result open; the workspace
//! `clippy.toml` rejects the `std` methods (`disallowed-methods`). The `glam` types re-exported
//! by [`math`] are deterministic because `glam` is built with its `libm` feature; see [`math`].

pub mod block;
pub mod coords;
pub mod det;
pub mod face;
pub mod frames;
pub mod hash;
pub mod ids;
pub mod math;
pub mod noise;
pub mod profile;
pub mod quad;
pub mod time;
