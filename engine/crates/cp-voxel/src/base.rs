//! The generator interface (contract §5.1): what a grid's procedurally generated base world
//! looks like, at any LOD level.
//!
//! `cp-worldgen` implements [`BaseSource`]; `cp-voxel` composes its output with player edits
//! ([`crate::EditStore`]) and caches it ([`crate::VoxelWorld`]).
//!
//! # Cell coordinates
//! A cell of level `L` with integer coordinates `c` covers the level-0 voxels
//! `[c * 2^L, (c + 1) * 2^L)` on every axis; level-0 cells are voxels. The chunk `key` covers
//! the level-`key.level` cells [`chunk_cells`]`(key)` = `[key.pos * 32, key.pos * 32 + 32)`.
//! Regions passed to [`BaseSource::fill`] are half-open boxes of such cells and are laid out
//! x fastest, then z, then y, like the cells of a chunk (contract §2.2).

use cp_core::coords::{CHUNK_SIZE, LodKey};
use cp_core::hash::{hash_bytes, hash_combine};
use cp_core::ids::BlockStateId;
use cp_core::math::{IAabb, IVec3};

/// Identifies the generator that produces a grid's base world (contract §5.1, §6).
///
/// Generator output for a given `(GeneratorIdentity, LodKey)` is identical on every machine
/// (contract §2.5 rule 6). A save pins the identity per body; every change of generator output
/// bumps `version`.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct GeneratorIdentity {
    /// Stable name of the generator, e.g. `"cubeplanet:flat"`.
    pub key: String,
    /// Output version of the generator; bumped on every change of its output.
    pub version: u32,
    /// World seed.
    pub seed: u64,
    /// Hash of the baked macro maps the generator reads (0 when it reads none).
    pub macro_hash: u64,
}

impl GeneratorIdentity {
    /// A deterministic 64-bit hash of all four fields, for cache keys (contract §6).
    ///
    /// Fixed recipe, identical on every machine: FNV-1a over the UTF-8 key, then
    /// [`hash_combine`] with the version, the seed and the macro hash in this order. Pinned by a
    /// test; it must never change.
    #[must_use]
    pub fn stable_hash(&self) -> u64 {
        let mut hash = hash_bytes(self.key.as_bytes());
        hash = hash_combine(hash, u64::from(self.version));
        hash = hash_combine(hash, self.seed);
        hash_combine(hash, self.macro_hash)
    }
}

/// Conservative classification of a chunk at its level (contract §5.1).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum NodeClass {
    /// Every cell of the chunk is air.
    Empty,
    /// Every cell of the chunk is `OPAQUE`.
    Opaque,
    /// Anything else; always an allowed answer.
    Mixed,
}

/// Error of [`BaseSource::fill`]: the caller's cancel callback returned `true` and the
/// generator stopped. The contents of the output buffer are unspecified.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, thiserror::Error)]
#[error("base generation was cancelled")]
pub struct Cancelled;

/// A deterministic base world: the generator interface implemented by `cp-worldgen`
/// (contract §5.1).
///
/// Rules for implementors: a cell at level `L` is solid iff the generator's band-limited field
/// at the cell centre says so (the same rule on every level, no bias); features are placed
/// independently of the level and evaluated pull-based within a declared maximum radius; every
/// feature class names the coarsest level at which it is still rasterised and its stand-in
/// above it. Output for a given `(identity, level, cell)` never depends on the region that
/// was asked for, on the thread or on the machine. Never returns
/// [`BlockStateId::UNCHANGED`].
pub trait BaseSource: Send + Sync {
    /// The identity that determines this source's output.
    fn identity(&self) -> &GeneratorIdentity;
    /// Fills `out` with the cells of `region` (cells of `level`, x fastest, then z, then y).
    ///
    /// `out.len()` equals the region's cell count ([`region_volume`]); `region` is not empty.
    /// Implementations poll `cancel` at a coarse granularity (per column or row) and return
    /// [`Cancelled`] as soon as it returns `true`.
    ///
    /// Cost: callers fill without asking [`BaseSource::classify`] first, so regions an
    /// implementation can decide without evaluating its field (for example all air above
    /// every terrain, feature and water level) should take a fast path here. Fills of a few
    /// cells (`cp-voxel` asks for at most 3³ around an edit) should cost about as much as
    /// those cells' columns.
    fn fill(
        &self,
        level: u8,
        region: IAabb,
        out: &mut [BlockStateId],
        cancel: &dyn Fn() -> bool,
    ) -> Result<(), Cancelled>;
    /// Exact-conservative for THIS chunk at its level: Empty ⇒ every cell is air,
    /// Opaque ⇒ every cell is OPAQUE. Mixed is always allowed.
    ///
    /// A pure function of the key, like `fill`: `cp-voxel` memoises the answer per edited
    /// chunk. Cost: implementations decide analytically where they can; for chunks they
    /// cannot (typically those that cross the surface) it may cost as much as evaluating the
    /// field over the chunk's columns, i.e. about as much as a `fill` of the chunk. Callers
    /// with a time budget must not assume it is cheap.
    fn classify(&self, key: LodKey) -> NodeClass;
    /// Conservative: true only if no cell of any descendant at any level is non-air.
    fn subtree_empty(&self, key: LodKey) -> bool;
}

/// The level-`key.level` cells of the chunk `key`: `[key.pos * 32, key.pos * 32 + 32)`.
#[inline]
#[must_use]
pub fn chunk_cells(key: LodKey) -> IAabb {
    let min = key.pos * CHUNK_SIZE;
    IAabb::new(min, min + IVec3::splat(CHUNK_SIZE))
}

/// The cells of the chunk `key` plus a one-cell halo on every side (34³ cells of
/// `key.level`), the region a [`crate::PaddedChunk`] is filled from.
#[inline]
#[must_use]
pub fn padded_cells(key: LodKey) -> IAabb {
    chunk_cells(key).expand(1)
}

/// Number of cells in `region`; 0 for an empty region.
#[inline]
#[must_use]
pub fn region_volume(region: IAabb) -> usize {
    if region.is_empty() {
        return 0;
    }
    let size = region.size();
    // Sizes are positive here; 64-bit targets only (contract §2.5), so the product fits.
    size.x as usize * size.y as usize * size.z as usize
}

/// Index of the cell `cell` inside the `fill` layout of `region` (x fastest, then z, then y).
///
/// `cell` must lie inside `region`.
#[inline]
#[must_use]
pub fn region_index(region: IAabb, cell: IVec3) -> usize {
    debug_assert!(region.contains(cell), "cell outside the region");
    let size = region.size();
    let local = cell - region.min;
    local.x as usize
        + local.z as usize * size.x as usize
        + local.y as usize * size.x as usize * size.z as usize
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn regions_of_a_chunk() {
        let key = LodKey::new(3, IVec3::new(-1, 0, 2));
        assert_eq!(
            chunk_cells(key),
            IAabb::new(IVec3::new(-32, 0, 64), IVec3::new(0, 32, 96))
        );
        assert_eq!(
            padded_cells(key),
            IAabb::new(IVec3::new(-33, -1, 63), IVec3::new(1, 33, 97))
        );
        assert_eq!(region_volume(chunk_cells(key)), 32 * 32 * 32);
        assert_eq!(region_volume(padded_cells(key)), 34 * 34 * 34);
        assert_eq!(region_volume(IAabb::EMPTY), 0);
        let region = padded_cells(key);
        assert_eq!(region_index(region, region.min), 0);
        assert_eq!(region_index(region, region.min + IVec3::X), 1);
        assert_eq!(region_index(region, region.min + IVec3::Z), 34);
        assert_eq!(region_index(region, region.min + IVec3::Y), 34 * 34);
        assert_eq!(
            region_index(region, region.max - IVec3::ONE),
            34 * 34 * 34 - 1
        );
    }

    #[test]
    fn identity_hash_is_pinned() {
        let identity = GeneratorIdentity {
            key: "cubeplanet:flat".to_owned(),
            version: 1,
            seed: 42,
            macro_hash: 0,
        };
        let hash = identity.stable_hash();
        // Recomputed from the documented recipe.
        let expected = hash_combine(
            hash_combine(hash_combine(hash_bytes(b"cubeplanet:flat"), 1), 42),
            0,
        );
        assert_eq!(hash, expected);
        let mut other = identity.clone();
        other.version = 2;
        assert_ne!(other.stable_hash(), hash);
        let mut other = identity.clone();
        other.seed = 43;
        assert_ne!(other.stable_hash(), hash);
        let mut other = identity;
        other.macro_hash = 1;
        assert_ne!(other.stable_hash(), hash);
    }

    #[test]
    fn cancelled_is_an_error() {
        let error: Box<dyn std::error::Error> = Box::new(Cancelled);
        assert_eq!(error.to_string(), "base generation was cancelled");
    }
}
