//! `cp-worldgen`: the base-world generators of the Cubeplanet engine (contract §6).
//!
//! Every generator implements [`cp_voxel::BaseSource`]. E1 provides deterministic test worlds
//! for meshing, rendering and LOD tests; Terrain 2.0 (planet history, climate, erosion, biomes)
//! follows in E6.
//!
//! | source | world | identity key |
//! |---|---|---|
//! | [`FlatSource`] | layered flat ground | `cubeplanet:flat` |
//! | [`HillsSource`] | rolling hills, a few ridges, lakes, beaches and shores, rock, scree, snow, dry and lush regions, mixed woods, bushes and boulders | `cubeplanet:hills` |
//! | [`StructureField`] | the E1 structure set (pillar, one-thick wall, beam, hollow cube, tree) on flat ground | `cubeplanet:structure_field` |
//!
//! # Rules shared by all sources (contract §5.1)
//! - **Space.** The test worlds are planes with `+Y` up; voxel `(x, y, z)` spans `[x, x+1)` m.
//!   A cell of level `L` with coordinates `c` covers the voxels `[c * 2^L, (c + 1) * 2^L)` and
//!   has its centre at `(c + 0.5) * 2^L` m; `fill` regions are boxes of such cells, laid out
//!   x fastest, then z, then y ([`cp_voxel::base`]).
//! - **Point rule.** A cell is ground iff the generator's height field *at the cell centre*
//!   lies above the centre, on every level, without bias. Height fields are **band-limited per
//!   level**: level `L` keeps only the noise octaves whose wavelength spans at least
//!   [`BAND_LIMIT_CELLS`] of its cells, and evaluates the field once per column at the column's
//!   centre (column-first).
//! - **Layers.** Ground columns are layered from the top (grass over dirt over stone, …): a
//!   cell gets the layer its centre lies in, except that the top ground cell of a column always
//!   gets the surface layer, so a 1 m grass layer still colours a coarse cell.
//! - **Water** is an ordinary block state: cells the ground leaves open whose centre lies below
//!   the sea level.
//! - **Features** (trees, bushes, boulders) are placed independently of the level on a
//!   jittered grid and evaluated pull-based within a declared maximum radius
//!   ([`TREE_MAX_RADIUS`], [`TREE_MAX_HEIGHT`]); each feature class names the coarsest level
//!   at which it is rasterised (trees [`TREE_RASTER_MAX_LEVEL`], bushes and boulders
//!   [`SMALL_FEATURE_MAX_LEVEL`]) and its stand-in above it (trees: a canopy up to
//!   [`TREE_CANOPY_MAX_LEVEL`]; bushes and boulders: none). Output for a cell never depends
//!   on the region asked for.
//! - **States** come from the caller as a [`TerrainPalette`] of block state ids; which of them
//!   are `OPAQUE` comes from the flags table passed to each constructor. Parameters name
//!   [`PaletteRole`]s, never ids. This crate does not depend on `cp-content`.
//! - **Identity.** [`cp_voxel::GeneratorIdentity`] `{ key, version, seed, macro_hash }`: the test
//!   generators read no macro maps, so `macro_hash` carries the hash of every parameter that
//!   decides the cells (including which role each layer uses) and of the palette's state keys
//!   ([`TerrainPalette::key_hash`]); session-local ids never enter it. Every change of output
//!   bumps the `VERSION` constant of the source; golden hashes over fixed keys on every level
//!   in the tests enforce it.
//! - **Determinism** (contract §2.5): noise from `cp_core::noise`, randomness from
//!   `cp_core::hash`, `min`/`max`/`clamp` through `cp_core::det`, cell decisions in exact
//!   arithmetic, no iteration-order dependence.

mod config_hash;
mod error;
mod field;
mod flat;
mod hills;
mod palette;
mod region;
mod strata;
mod structures;
mod terrain;
mod trees;

pub use error::WorldgenError;
pub use field::BAND_LIMIT_CELLS;
pub use flat::{FlatConfig, FlatLayer, FlatSource};
pub use hills::HillsSource;
pub use palette::{PaletteKeys, PaletteRole, ROLE_COUNT, TerrainPalette};
pub use structures::{StructureField, StructureKind, StructureShape};
pub use terrain::HillsConfig;
pub use trees::{
    FeatureKind, Rock, SMALL_FEATURE_MAX_LEVEL, Species, TREE_CANOPY_MAX_LEVEL, TREE_GRID,
    TREE_MAX_HEIGHT, TREE_MAX_RADIUS, TREE_RASTER_MAX_LEVEL, TreeInfo,
};
