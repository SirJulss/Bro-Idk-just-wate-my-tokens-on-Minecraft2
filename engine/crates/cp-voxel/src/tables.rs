//! The per-state tables `cp-voxel` needs (contract §3.1, §4.4): plain slices, not the registry.

use std::sync::Arc;

use cp_core::block::{BlockFlags, LodInfo};
use cp_core::ids::BlockStateId;

/// Flag and LOD tables indexed by [`BlockStateId`], shared (`Arc`) and immutable.
///
/// Built from the content registry's `BlockTables` (`flags` and `lod`). Both tables have one
/// entry per state and must cover every state the base source and the edits use;
/// [`BlockStateId::UNCHANGED`] is never looked up.
#[derive(Clone)]
pub struct VoxelTables {
    /// Physical and visual flags per state.
    pub flags: Arc<[BlockFlags]>,
    /// LOD class and proxy per state.
    pub lod: Arc<[LodInfo]>,
}

impl core::fmt::Debug for VoxelTables {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("VoxelTables")
            .field("states", &self.flags.len())
            .finish_non_exhaustive()
    }
}

impl VoxelTables {
    /// Bundles the two tables.
    ///
    /// # Panics
    /// If the tables have different lengths (they come from one registry, so that is a
    /// programming error).
    #[must_use]
    pub fn new(flags: Arc<[BlockFlags]>, lod: Arc<[LodInfo]>) -> Self {
        assert_eq!(
            flags.len(),
            lod.len(),
            "VoxelTables::new: flag and LOD tables must have one entry per state"
        );
        Self { flags, lod }
    }

    /// Number of states covered.
    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        self.flags.len()
    }

    /// True when the tables cover no state.
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.flags.is_empty()
    }

    /// True when `state` has a table entry.
    #[inline]
    #[must_use]
    pub fn contains(&self, state: BlockStateId) -> bool {
        state.index() < self.flags.len() && state.index() < self.lod.len()
    }

    /// The flags of `state`.
    ///
    /// # Panics
    /// If `state` has no table entry.
    #[inline]
    #[must_use]
    pub fn flags_of(&self, state: BlockStateId) -> BlockFlags {
        self.flags[state.index()]
    }

    /// The LOD description of `state`.
    ///
    /// # Panics
    /// If `state` has no table entry.
    #[inline]
    #[must_use]
    pub fn lod_of(&self, state: BlockStateId) -> LodInfo {
        self.lod[state.index()]
    }
}
