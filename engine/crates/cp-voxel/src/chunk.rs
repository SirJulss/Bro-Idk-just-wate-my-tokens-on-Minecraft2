//! A chunk of one LOD level: its cells and which of them are player-authored (contract §5.1).

use cp_core::ids::BlockStateId;

use crate::chunk_data::ChunkData;
use crate::mask::{CellMask, MASK_WORDS};

/// The 32³ cells of one chunk at its level plus the cells that carry a player override.
///
/// `authored` marks cells that carry a player override at this level (at level 0 exactly the
/// cells an edit set; at coarser levels the cells with an authored child, contract §5.1).
/// `None` and an empty mask mean the same; methods of this crate keep `None` for "no authored
/// cell". Equality is semantic: cells compared cell by cell, `None` equal to an empty mask.
///
/// In an override layer ([`crate::EditStore`]) `blocks` holds
/// [`BlockStateId::UNCHANGED`] wherever the base world shows through; in a composite (what
/// [`crate::VoxelSnapshot::sample`] returns) it never does.
#[derive(Clone, Debug, Default)]
pub struct VoxelChunk {
    /// The cell states.
    pub blocks: ChunkData,
    /// The authored cells, if any.
    pub authored: Option<Box<CellMask>>,
}

impl VoxelChunk {
    /// A chunk with these cells and no authored cell.
    #[inline]
    #[must_use]
    pub const fn new(blocks: ChunkData) -> Self {
        Self {
            blocks,
            authored: None,
        }
    }

    /// A chunk whose cells all carry `state`, none authored.
    #[inline]
    #[must_use]
    pub const fn uniform(state: BlockStateId) -> Self {
        Self::new(ChunkData::Uniform(state))
    }

    /// The state of cell `index`.
    ///
    /// # Panics
    /// If `index >= CHUNK_VOLUME`.
    #[inline]
    #[must_use]
    pub fn get(&self, index: usize) -> BlockStateId {
        self.blocks.get(index)
    }

    /// True when cell `index` is authored.
    ///
    /// # Panics
    /// If `index >= CHUNK_VOLUME`.
    #[inline]
    #[must_use]
    pub fn is_authored(&self, index: usize) -> bool {
        self.authored.as_ref().is_some_and(|mask| mask.get(index))
    }

    /// Marks cell `index` as authored or not. Does not drop an emptied mask; call
    /// [`VoxelChunk::normalize_authored`] after a batch of changes.
    ///
    /// # Panics
    /// If `index >= CHUNK_VOLUME`.
    pub fn set_authored(&mut self, index: usize, authored: bool) {
        match (&mut self.authored, authored) {
            (Some(mask), _) => mask.set(index, authored),
            (None, true) => {
                let mut mask = Box::new(CellMask::new());
                mask.set(index, true);
                self.authored = Some(mask);
            }
            (None, false) => {}
        }
    }

    /// Replaces an empty authored mask by `None`.
    pub fn normalize_authored(&mut self) {
        if self.authored.as_ref().is_some_and(|mask| mask.is_empty()) {
            self.authored = None;
        }
    }

    /// True when at least one cell is authored.
    #[must_use]
    pub fn has_authored(&self) -> bool {
        self.authored.as_ref().is_some_and(|mask| !mask.is_empty())
    }

    /// The authored mask as plain words (`None` when no cell is authored), the form
    /// `cp-mesh` consumes.
    #[must_use]
    pub fn authored_words(&self) -> Option<&[u64; MASK_WORDS]> {
        self.authored
            .as_deref()
            .filter(|mask| !mask.is_empty())
            .map(CellMask::words)
    }

    /// Bytes this value occupies: its own size plus its heap allocations.
    #[must_use]
    pub fn memory_bytes(&self) -> usize {
        size_of::<Self>() - size_of::<ChunkData>()
            + self.blocks.memory_bytes()
            + self.authored.as_ref().map_or(0, |_| size_of::<CellMask>())
    }
}

impl PartialEq for VoxelChunk {
    fn eq(&self, other: &Self) -> bool {
        self.blocks == other.blocks && self.authored_words() == other.authored_words()
    }
}

impl Eq for VoxelChunk {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn authored_none_equals_empty_mask() {
        let mut a = VoxelChunk::uniform(BlockStateId(3));
        let b = VoxelChunk::uniform(BlockStateId(3));
        a.set_authored(10, true);
        assert!(a.is_authored(10) && a.has_authored());
        assert_ne!(a, b);
        a.set_authored(10, false);
        assert!(a.authored.is_some());
        assert_eq!(a, b, "an empty mask equals no mask");
        a.normalize_authored();
        assert!(a.authored.is_none());
        assert_eq!(a.authored_words(), None);
        a.set_authored(5, false);
        assert!(a.authored.is_none(), "clearing a bit never allocates");
    }

    #[test]
    fn memory_counts_the_mask() {
        let mut chunk = VoxelChunk::uniform(BlockStateId(1));
        let plain = chunk.memory_bytes();
        assert_eq!(plain, size_of::<VoxelChunk>());
        chunk.set_authored(0, true);
        assert_eq!(chunk.memory_bytes(), plain + 4096);
    }
}
