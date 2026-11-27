//! A chunk with a one-cell halo, the input of the mesher (contract §5.1, §5.2).

use cp_core::coords::{CHUNK_SIZE, LodKey};
use cp_core::ids::BlockStateId;
use cp_core::math::IVec3;

use crate::mask::{CellMask, MASK_WORDS};

/// Edge of a padded chunk in cells: 32 plus one halo cell on each side.
pub const PADDED_SIZE: usize = CHUNK_SIZE as usize + 2;

/// Number of cells of a padded chunk (`34³ = 39304`).
pub const PADDED_VOLUME: usize = PADDED_SIZE * PADDED_SIZE * PADDED_SIZE;

/// Padded index of the chunk cell `(x, y, z)`: `(x+1) + (z+1)*34 + (y+1)*1156` (contract §5.1).
///
/// Each coordinate lies in `-1..=32`; `-1` and `32` address the halo.
#[inline]
#[must_use]
pub const fn padded_index(x: i32, y: i32, z: i32) -> usize {
    debug_assert!(
        x >= -1 && x <= 32 && y >= -1 && y <= 32 && z >= -1 && z <= 32,
        "padded coordinate outside -1..=32"
    );
    let size = PADDED_SIZE as i32;
    ((x + 1) + (z + 1) * size + (y + 1) * size * size) as usize
}

/// The cells of one chunk at its level plus a one-cell halo: `34³` states, cell `(x, y, z)`
/// of the chunk at [`padded_index`]`(x, y, z)` (x fastest, then z, then y).
///
/// Face-halo cells (exactly one coordinate is `-1` or `32`) are exact copies of the
/// neighbouring chunks' cells at the same level; edge and corner halo cells are unspecified.
/// The chunk's own authored mask (32³, not padded) travels along.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PaddedChunk {
    key: LodKey,
    cells: Box<[BlockStateId]>,
    authored: Option<Box<CellMask>>,
}

impl PaddedChunk {
    /// A padded chunk from its cells in padded index order and the chunk's authored mask.
    ///
    /// # Panics
    /// If `cells.len() != PADDED_VOLUME`.
    #[must_use]
    pub fn new(key: LodKey, cells: Box<[BlockStateId]>, authored: Option<Box<CellMask>>) -> Self {
        assert_eq!(
            cells.len(),
            PADDED_VOLUME,
            "PaddedChunk::new: a padded chunk has exactly 34^3 cells"
        );
        let authored = authored.filter(|mask| !mask.is_empty());
        Self {
            key,
            cells,
            authored,
        }
    }

    /// A padded chunk whose cell at padded coordinates `p` (each component in `-1..=32`) is
    /// `f(p)`.
    #[must_use]
    pub fn from_fn(
        key: LodKey,
        authored: Option<Box<CellMask>>,
        mut f: impl FnMut(IVec3) -> BlockStateId,
    ) -> Self {
        let mut cells = Vec::with_capacity(PADDED_VOLUME);
        for y in -1..=32 {
            for z in -1..=32 {
                for x in -1..=32 {
                    cells.push(f(IVec3::new(x, y, z)));
                }
            }
        }
        Self::new(key, cells.into_boxed_slice(), authored)
    }

    /// The chunk this is the padded version of.
    #[inline]
    #[must_use]
    pub const fn key(&self) -> LodKey {
        self.key
    }

    /// Its LOD level (cells of `2^level` m).
    #[inline]
    #[must_use]
    pub const fn level(&self) -> u8 {
        self.key.level
    }

    /// All `34³` cells in padded index order.
    #[inline]
    #[must_use]
    pub fn cells(&self) -> &[BlockStateId] {
        &self.cells
    }

    /// All `34³` cells in padded index order, mutable.
    #[inline]
    #[must_use]
    pub fn cells_mut(&mut self) -> &mut [BlockStateId] {
        &mut self.cells
    }

    /// The chunk's authored mask as plain words (32³ bits, bit = chunk cell index), `None`
    /// when no cell is authored.
    #[inline]
    #[must_use]
    pub fn authored(&self) -> Option<&[u64; MASK_WORDS]> {
        self.authored.as_deref().map(CellMask::words)
    }

    /// The chunk's authored mask, `None` when no cell is authored.
    #[inline]
    #[must_use]
    pub fn authored_mask(&self) -> Option<&CellMask> {
        self.authored.as_deref()
    }

    /// The cell at padded coordinates `(x, y, z)`, each in `-1..=32`.
    ///
    /// # Panics
    /// If a coordinate is outside `-1..=32`.
    #[inline]
    #[must_use]
    pub fn get(&self, x: i32, y: i32, z: i32) -> BlockStateId {
        assert!(
            (-1..=32).contains(&x) && (-1..=32).contains(&y) && (-1..=32).contains(&z),
            "padded coordinate outside -1..=32"
        );
        self.cells[padded_index(x, y, z)]
    }

    /// Sets the cell at padded coordinates `(x, y, z)`, each in `-1..=32`.
    ///
    /// # Panics
    /// If a coordinate is outside `-1..=32`.
    #[inline]
    pub fn set(&mut self, x: i32, y: i32, z: i32, state: BlockStateId) {
        assert!(
            (-1..=32).contains(&x) && (-1..=32).contains(&y) && (-1..=32).contains(&z),
            "padded coordinate outside -1..=32"
        );
        self.cells[padded_index(x, y, z)] = state;
    }

    /// Key, cells and authored mask.
    #[must_use]
    pub fn into_parts(self) -> (LodKey, Box<[BlockStateId]>, Option<Box<CellMask>>) {
        (self.key, self.cells, self.authored)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn index_convention() {
        assert_eq!(PADDED_SIZE, 34);
        assert_eq!(PADDED_VOLUME, 39_304);
        assert_eq!(padded_index(-1, -1, -1), 0);
        assert_eq!(padded_index(0, 0, 0), 1 + 34 + 1156);
        assert_eq!(padded_index(32, 32, 32), PADDED_VOLUME - 1);
        assert_eq!(padded_index(1, 0, 0) - padded_index(0, 0, 0), 1);
        assert_eq!(padded_index(0, 0, 1) - padded_index(0, 0, 0), 34);
        assert_eq!(padded_index(0, 1, 0) - padded_index(0, 0, 0), 1156);
        let key = LodKey::new(2, IVec3::new(-1, 0, 3));
        let padded = PaddedChunk::from_fn(key, None, |p| {
            BlockStateId((p.x + 1 + 100 * (p.y + 1) + 10_000 * (p.z + 1)) as u32)
        });
        assert_eq!(padded.key(), key);
        assert_eq!(padded.level(), 2);
        for (x, y, z) in [(-1, -1, -1), (0, 5, 32), (32, -1, 7)] {
            assert_eq!(
                padded.get(x, y, z),
                BlockStateId((x + 1 + 100 * (y + 1) + 10_000 * (z + 1)) as u32)
            );
            assert_eq!(padded.cells()[padded_index(x, y, z)], padded.get(x, y, z));
        }
        // An empty mask is stored as none.
        let padded = PaddedChunk::new(
            key,
            vec![BlockStateId::AIR; PADDED_VOLUME].into_boxed_slice(),
            Some(Box::new(CellMask::new())),
        );
        assert!(padded.authored().is_none());
    }
}
