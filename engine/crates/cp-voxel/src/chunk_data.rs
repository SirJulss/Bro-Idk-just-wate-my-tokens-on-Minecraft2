//! Paletted storage of the 32³ cells of one chunk (contract §5.1).
//!
//! [`ChunkData`] is either one state for the whole chunk or a palette of distinct states plus
//! one palette index per cell, packed into `u64` words with 1, 2, 4, 8 or 16 bits per cell.
//! Cells are addressed by their canonical index `x + z * 32 + y * 1024`
//! ([`cp_core::coords::cell_index`]).
//!
//! The palette grows (wider indices) when a new state does not fit, a chunk whose cells all
//! carry one state collapses to [`ChunkData::Uniform`] at once, and [`ChunkData::compact`] /
//! [`ChunkData::canonicalize`] shrink the index width again. Equality is semantic (cell by
//! cell); [`ChunkData::write_canonical`] produces the canonical encoding of contract §2.5
//! rule 5, which hashes and saves use.

use core::fmt;

use cp_core::coords::CHUNK_VOLUME;
use cp_core::ids::BlockStateId;
use rustc_hash::FxHashMap;

/// `u64` words per bit of index width: a paletted chunk with `bits` bits per cell stores
/// `WORDS_PER_BIT * bits` words (`CHUNK_VOLUME / 64`).
pub const WORDS_PER_BIT: usize = CHUNK_VOLUME / 64;

/// The cells of one chunk (contract §5.1): one state for all cells, or a palette plus packed
/// per-cell palette indices.
///
/// Invariant: a [`ChunkData::Paletted`] chunk always uses at least two distinct states; a
/// chunk whose cells all carry one state is [`ChunkData::Uniform`] (every method collapses
/// it). `PartialEq` compares cell by cell, so two chunks with the same cells are equal
/// whatever their palette order or index width.
///
/// The type stores any [`BlockStateId`], including the [`BlockStateId::UNCHANGED`] sentinel of
/// edit override layers.
#[derive(Clone)]
pub enum ChunkData {
    /// Every cell carries this state.
    Uniform(BlockStateId),
    /// A palette of at least two states in use and one packed index per cell.
    Paletted(Paletted),
}

/// The paletted form of [`ChunkData`]: `palette`, `bits` per cell index and `data` words.
///
/// Invariants, kept by every method (the fields are private for that reason):
/// - `bits` is 1, 2, 4, 8 or 16 and the palette has at most `1 << bits` entries;
/// - [`Paletted::words`] holds `512 * bits` words; cell `i` keeps its palette index in word
///   `i / (64 / bits)` at bit `(i % (64 / bits)) * bits` (no index straddles two words);
/// - palette entries are pairwise distinct; an entry that no cell uses is a free slot, which
///   [`ChunkData::set`] reuses before the palette grows;
/// - at least two entries are in use.
#[derive(Clone)]
pub struct Paletted {
    palette: Vec<BlockStateId>,
    /// Cells per palette entry; the sum is `CHUNK_VOLUME`.
    counts: Vec<u16>,
    bits: u8,
    data: Box<[u64]>,
}

/// Smallest supported index width for `entries` palette entries (at least 1 bit).
#[inline]
const fn min_bits(entries: usize) -> u8 {
    match entries {
        0..=2 => 1,
        3..=4 => 2,
        5..=16 => 4,
        17..=256 => 8,
        _ => 16,
    }
}

#[inline]
const fn index_mask(bits: u8) -> u64 {
    (1_u64 << bits) - 1
}

/// Word and bit offset of cell `index` for `bits` bits per index.
#[inline]
const fn locate(bits: u8, index: usize) -> (usize, u32) {
    let log_bits = (bits as u32).trailing_zeros();
    let per_word_log = 6 - log_bits;
    let word = index >> per_word_log;
    let offset = ((index & ((1 << per_word_log) - 1)) as u32) << log_bits;
    (word, offset)
}

#[inline]
fn read_index(data: &[u64], bits: u8, index: usize) -> usize {
    let (word, offset) = locate(bits, index);
    ((data[word] >> offset) & index_mask(bits)) as usize
}

#[inline]
fn write_index(data: &mut [u64], bits: u8, index: usize, value: usize) {
    let (word, offset) = locate(bits, index);
    let mask = index_mask(bits) << offset;
    data[word] = (data[word] & !mask) | (((value as u64) << offset) & mask);
}

/// Calls `f(cell, entry)` for every cell in index order.
#[inline]
fn for_each_entry(data: &[u64], bits: u8, mut f: impl FnMut(usize, usize)) {
    let width = usize::from(bits);
    let per_word = 64 / width;
    let mask = index_mask(bits);
    for (word_index, &word) in data.iter().enumerate() {
        let mut word = word;
        let first = word_index * per_word;
        for cell in first..first + per_word {
            f(cell, (word & mask) as usize);
            word >>= width;
        }
    }
}

/// Packs one index per cell (`indices.len() == CHUNK_VOLUME`, each `< 1 << bits`).
fn pack(indices: &[u16], bits: u8) -> Box<[u64]> {
    debug_assert_eq!(indices.len(), CHUNK_VOLUME);
    let width = usize::from(bits);
    let per_word = 64 / width;
    indices
        .chunks_exact(per_word)
        .map(|group| {
            group.iter().enumerate().fold(0_u64, |word, (k, &value)| {
                word | (u64::from(value) << (k * width))
            })
        })
        .collect()
}

impl Paletted {
    /// The palette, indexed by the packed cell indices. It may contain free slots: entries
    /// that no cell uses ([`Paletted::count_of_entry`] is 0). Entries are pairwise distinct.
    #[inline]
    #[must_use]
    pub fn palette(&self) -> &[BlockStateId] {
        &self.palette
    }

    /// Bits per cell index: 1, 2, 4, 8 or 16.
    #[inline]
    #[must_use]
    pub fn bits(&self) -> u8 {
        self.bits
    }

    /// The packed index words, `512 * bits` of them (layout in the type documentation).
    #[inline]
    #[must_use]
    pub fn words(&self) -> &[u64] {
        &self.data
    }

    /// Number of cells whose index is the palette entry `entry`.
    ///
    /// # Panics
    /// If `entry` is not an index into [`Paletted::palette`].
    #[inline]
    #[must_use]
    pub fn count_of_entry(&self, entry: usize) -> usize {
        usize::from(self.counts[entry])
    }

    /// Palette index of the cell `index`.
    ///
    /// # Panics
    /// If `index >= CHUNK_VOLUME`.
    #[inline]
    #[must_use]
    pub fn entry_of(&self, index: usize) -> usize {
        read_index(&self.data, self.bits, index)
    }

    /// Number of palette entries in use (at least two).
    #[must_use]
    pub fn used_entries(&self) -> usize {
        self.counts.iter().filter(|&&count| count > 0).count()
    }

    #[inline]
    fn get(&self, index: usize) -> BlockStateId {
        self.palette[read_index(&self.data, self.bits, index)]
    }

    /// Sets one cell. Returns `Some(state)` when afterwards every cell carries `state`, i.e.
    /// the chunk must collapse to `Uniform(state)`.
    fn set(&mut self, index: usize, state: BlockStateId) -> Option<BlockStateId> {
        let old = read_index(&self.data, self.bits, index);
        if self.palette[old] == state {
            return None;
        }
        let new = match self.palette.iter().position(|&entry| entry == state) {
            Some(entry) => entry,
            None if self.counts[old] == 1 => {
                // This cell is the only user of its entry: rename the entry. `state` is new,
                // so no other cell carries it and the chunk cannot become uniform.
                self.palette[old] = state;
                return None;
            }
            None => self.allocate(state),
        };
        self.counts[old] -= 1;
        self.counts[new] += 1;
        write_index(&mut self.data, self.bits, index, new);
        (usize::from(self.counts[new]) == CHUNK_VOLUME).then_some(state)
    }

    /// A palette entry for the new state `state`: a free slot, a new entry, or a new entry
    /// after widening the indices. Its count is 0.
    fn allocate(&mut self, state: BlockStateId) -> usize {
        if let Some(free) = self.counts.iter().position(|&count| count == 0) {
            self.palette[free] = state;
            return free;
        }
        if self.palette.len() == 1 << self.bits {
            // Every entry is in use, so at most 32768 < 1 << 16 entries exist and widening
            // is always possible here.
            self.repack(self.bits * 2);
        }
        self.palette.push(state);
        self.counts.push(0);
        self.palette.len() - 1
    }

    /// Re-encodes the indices with `bits` bits each (every index must fit).
    fn repack(&mut self, bits: u8) {
        debug_assert!(matches!(bits, 1 | 2 | 4 | 8 | 16));
        let mut indices = vec![0_u16; CHUNK_VOLUME];
        for_each_entry(&self.data, self.bits, |cell, entry| {
            indices[cell] = entry as u16;
        });
        self.data = pack(&indices, bits);
        self.bits = bits;
    }

    /// Unpacks the palette index of every cell.
    fn indices(&self) -> Vec<u16> {
        let mut indices = vec![0_u16; CHUNK_VOLUME];
        for_each_entry(&self.data, self.bits, |cell, entry| {
            indices[cell] = entry as u16;
        });
        indices
    }
}

impl ChunkData {
    /// A chunk whose cells all carry `state`.
    #[inline]
    #[must_use]
    pub const fn uniform(state: BlockStateId) -> Self {
        Self::Uniform(state)
    }

    /// The chunk with the given cells in canonical index order.
    ///
    /// The result is compact and canonical: palette in order of first occurrence, minimal
    /// index width, `Uniform` when all cells are equal.
    ///
    /// # Panics
    /// If `cells.len() != CHUNK_VOLUME`.
    #[must_use]
    pub fn from_cells(cells: &[BlockStateId]) -> Self {
        assert_eq!(
            cells.len(),
            CHUNK_VOLUME,
            "ChunkData::from_cells: a chunk has exactly 32768 cells"
        );
        let first = cells[0];
        if cells.iter().all(|&cell| cell == first) {
            return Self::Uniform(first);
        }
        let mut palette: Vec<BlockStateId> = Vec::new();
        let mut counts: Vec<u16> = Vec::new();
        let mut lookup: FxHashMap<BlockStateId, u16> = FxHashMap::default();
        let mut indices = vec![0_u16; CHUNK_VOLUME];
        // Runs of equal states are common; only a change of state needs the hash map.
        let mut run_state = first;
        palette.push(first);
        counts.push(0);
        lookup.insert(first, 0);
        let mut run_entry: u16 = 0;
        for (index, &state) in cells.iter().enumerate() {
            if state != run_state {
                run_state = state;
                run_entry = *lookup.entry(state).or_insert_with(|| {
                    palette.push(state);
                    counts.push(0);
                    // At most CHUNK_VOLUME distinct states, so the entry fits in u16.
                    (palette.len() - 1) as u16
                });
            }
            indices[index] = run_entry;
            counts[usize::from(run_entry)] += 1;
        }
        let bits = min_bits(palette.len());
        Self::Paletted(Paletted {
            data: pack(&indices, bits),
            palette,
            counts,
            bits,
        })
    }

    /// The chunk whose cell `index` carries `f(index)`, for every index in canonical order
    /// (`index = x + z * 32 + y * 1024`; [`cp_core::coords::index_to_cell`] converts back).
    ///
    /// Compact and canonical like [`ChunkData::from_cells`].
    #[must_use]
    pub fn from_fn(mut f: impl FnMut(usize) -> BlockStateId) -> Self {
        let cells: Vec<BlockStateId> = (0..CHUNK_VOLUME).map(&mut f).collect();
        Self::from_cells(&cells)
    }

    /// The state of the cell `index`.
    ///
    /// # Panics
    /// If `index >= CHUNK_VOLUME`.
    #[inline]
    #[must_use]
    pub fn get(&self, index: usize) -> BlockStateId {
        match self {
            Self::Uniform(state) => {
                assert!(index < CHUNK_VOLUME, "cell index outside the chunk");
                *state
            }
            Self::Paletted(paletted) => paletted.get(index),
        }
    }

    /// Sets the cell `index` to `state`.
    ///
    /// Grows the palette (reusing free slots first, then widening the indices) when `state`
    /// is new, and collapses the chunk to `Uniform` when every cell ends up with one state.
    /// Amortised cost O(palette length).
    ///
    /// # Panics
    /// If `index >= CHUNK_VOLUME`.
    pub fn set(&mut self, index: usize, state: BlockStateId) {
        assert!(index < CHUNK_VOLUME, "cell index outside the chunk");
        match self {
            Self::Uniform(current) => {
                if *current == state {
                    return;
                }
                let mut data = vec![0_u64; WORDS_PER_BIT].into_boxed_slice();
                data[index >> 6] |= 1 << (index & 63);
                *self = Self::Paletted(Paletted {
                    palette: vec![*current, state],
                    counts: vec![(CHUNK_VOLUME - 1) as u16, 1],
                    bits: 1,
                    data,
                });
            }
            Self::Paletted(paletted) => {
                if let Some(state) = paletted.set(index, state) {
                    *self = Self::Uniform(state);
                }
            }
        }
    }

    /// Sets every cell to `state` (the chunk becomes `Uniform(state)`).
    #[inline]
    pub fn fill(&mut self, state: BlockStateId) {
        *self = Self::Uniform(state);
    }

    /// Writes all cells into `out` in canonical index order.
    ///
    /// # Panics
    /// If `out.len() != CHUNK_VOLUME`.
    pub fn decode_into(&self, out: &mut [BlockStateId]) {
        assert_eq!(
            out.len(),
            CHUNK_VOLUME,
            "ChunkData::decode_into: the output must hold 32768 cells"
        );
        match self {
            Self::Uniform(state) => out.fill(*state),
            Self::Paletted(paletted) => {
                for_each_entry(&paletted.data, paletted.bits, |cell, entry| {
                    out[cell] = paletted.palette[entry];
                });
            }
        }
    }

    /// All cells in canonical index order.
    #[must_use]
    pub fn to_cells(&self) -> Vec<BlockStateId> {
        let mut cells = vec![BlockStateId::AIR; CHUNK_VOLUME];
        self.decode_into(&mut cells);
        cells
    }

    /// Removes free palette slots and narrows the indices to the minimal width; keeps the
    /// order of the remaining palette entries. A no-op for compact chunks.
    pub fn compact(&mut self) {
        let Self::Paletted(paletted) = self else {
            return;
        };
        let used = paletted.used_entries();
        if used <= 1 {
            // Unreachable while the invariants hold (`set` collapses at once); kept so that
            // `compact` always yields a valid chunk.
            let state = paletted
                .palette
                .iter()
                .zip(&paletted.counts)
                .find(|(_, count)| **count > 0)
                .map_or(BlockStateId::AIR, |(&state, _)| state);
            *self = Self::Uniform(state);
            return;
        }
        let bits = min_bits(used);
        if used == paletted.palette.len() && bits == paletted.bits {
            return;
        }
        let mut remap = vec![u16::MAX; paletted.palette.len()];
        let mut palette = Vec::with_capacity(used);
        let mut counts = Vec::with_capacity(used);
        for (entry, (&state, &count)) in paletted.palette.iter().zip(&paletted.counts).enumerate() {
            if count > 0 {
                remap[entry] = palette.len() as u16;
                palette.push(state);
                counts.push(count);
            }
        }
        let indices: Vec<u16> = paletted
            .indices()
            .into_iter()
            .map(|entry| remap[usize::from(entry)])
            .collect();
        *paletted = Paletted {
            data: pack(&indices, bits),
            palette,
            counts,
            bits,
        };
    }

    /// Brings the chunk into its canonical in-memory form: compact, palette in order of first
    /// occurrence (cell index order), minimal index width. Two chunks with the same cells are
    /// structurally identical after this call.
    pub fn canonicalize(&mut self) {
        if let Self::Paletted(_) = self {
            *self = Self::from_cells(&self.to_cells());
        }
    }

    /// True when every cell carries one state.
    #[inline]
    #[must_use]
    pub fn is_uniform(&self) -> bool {
        matches!(self, Self::Uniform(_))
    }

    /// The state of every cell if the chunk is uniform.
    #[inline]
    #[must_use]
    pub fn uniform_state(&self) -> Option<BlockStateId> {
        match self {
            Self::Uniform(state) => Some(*state),
            Self::Paletted(_) => None,
        }
    }

    /// Bits per cell index: 0 for a uniform chunk, else 1, 2, 4, 8 or 16.
    #[inline]
    #[must_use]
    pub fn bits(&self) -> u8 {
        match self {
            Self::Uniform(_) => 0,
            Self::Paletted(paletted) => paletted.bits,
        }
    }

    /// The distinct states that at least one cell carries, in palette order.
    #[must_use]
    pub fn states(&self) -> States<'_> {
        States {
            inner: match self {
                Self::Uniform(state) => StatesInner::Uniform(Some(*state)),
                Self::Paletted(paletted) => {
                    StatesInner::Paletted(paletted.palette.iter().zip(paletted.counts.iter()))
                }
            },
        }
    }

    /// True when at least one cell carries `state`.
    #[must_use]
    pub fn contains(&self, state: BlockStateId) -> bool {
        self.count(state) > 0
    }

    /// Number of cells that carry `state`.
    #[must_use]
    pub fn count(&self, state: BlockStateId) -> usize {
        match self {
            Self::Uniform(current) => {
                if *current == state {
                    CHUNK_VOLUME
                } else {
                    0
                }
            }
            Self::Paletted(paletted) => paletted
                .palette
                .iter()
                .position(|&entry| entry == state)
                .map_or(0, |entry| usize::from(paletted.counts[entry])),
        }
    }

    /// Bytes this value occupies: its own size plus its heap allocations.
    #[must_use]
    pub fn memory_bytes(&self) -> usize {
        size_of::<Self>()
            + match self {
                Self::Uniform(_) => 0,
                Self::Paletted(paletted) => {
                    paletted.palette.capacity() * size_of::<BlockStateId>()
                        + paletted.counts.capacity() * size_of::<u16>()
                        + paletted.data.len() * size_of::<u64>()
                }
            }
    }

    /// The overlay `self.or(base)` (contract §5.1): every cell of `self` that is
    /// [`BlockStateId::UNCHANGED`] takes the cell of `base`, every other cell keeps its state.
    /// Used to compose an override layer with the base world.
    #[must_use]
    pub fn or(&self, base: &ChunkData) -> ChunkData {
        let unchanged = BlockStateId::UNCHANGED;
        match self {
            Self::Uniform(state) if state.is_unchanged() => base.clone(),
            Self::Uniform(state) => Self::Uniform(*state),
            Self::Paletted(paletted) => {
                let Some(hole) = paletted
                    .palette
                    .iter()
                    .zip(&paletted.counts)
                    .position(|(&state, &count)| state == unchanged && count > 0)
                else {
                    return self.clone();
                };
                if let Self::Uniform(fill) = base {
                    // Renaming the UNCHANGED entry keeps the palette distinct unless the base
                    // state is an entry already.
                    if !paletted.palette.contains(fill) {
                        let mut composite = paletted.clone();
                        composite.palette[hole] = *fill;
                        return Self::Paletted(composite);
                    }
                }
                let mut cells = base.to_cells();
                for_each_entry(&paletted.data, paletted.bits, |cell, entry| {
                    let state = paletted.palette[entry];
                    if state != unchanged {
                        cells[cell] = state;
                    }
                });
                Self::from_cells(&cells)
            }
        }
    }

    /// Appends the canonical encoding (contract §2.5 rule 5) to `out`.
    ///
    /// `key_of` maps a state to its persistent string key (for override layers it must also
    /// map [`BlockStateId::UNCHANGED`], e.g. to a reserved key). States that map to the same
    /// key count as one. The encoding depends only on the key of every cell, never on
    /// session-local ids, palette order or index width; all integers are little-endian:
    ///
    /// | field | encoding |
    /// |---|---|
    /// | `P` | `u32`: number of distinct keys, `>= 1` |
    /// | keys | `P` times: `u32` byte length, then the UTF-8 bytes; in order of first occurrence in cell index order |
    /// | `b` | `u8`: 0 when `P == 1` (uniform), else the smallest of 1, 2, 4, 8, 16 with `2^b >= P` |
    /// | cells | only if `b > 0`: `512 * b` words (`u64`); cell `i` stores its key index in word `i / (64 / b)` at bit `(i % (64 / b)) * b` |
    ///
    /// # Panics
    /// If a key is longer than `u32::MAX` bytes.
    pub fn write_canonical<'k>(
        &self,
        out: &mut Vec<u8>,
        mut key_of: impl FnMut(BlockStateId) -> &'k str,
    ) {
        let write_key = |out: &mut Vec<u8>, key: &str| {
            let length = u32::try_from(key.len()).expect("state key longer than u32::MAX bytes");
            out.extend_from_slice(&length.to_le_bytes());
            out.extend_from_slice(key.as_bytes());
        };
        let paletted = match self {
            Self::Uniform(state) => {
                out.extend_from_slice(&1_u32.to_le_bytes());
                write_key(out, key_of(*state));
                out.push(0);
                return;
            }
            Self::Paletted(paletted) => paletted,
        };
        // Canonical key index of every palette entry, assigned at the entry's first cell.
        let mut entry_key = vec![u32::MAX; paletted.palette.len()];
        let mut keys: Vec<&'k str> = Vec::new();
        let mut by_key: FxHashMap<&'k str, u32> = FxHashMap::default();
        let mut cells = vec![0_u16; CHUNK_VOLUME];
        for_each_entry(&paletted.data, paletted.bits, |cell, entry| {
            let mut key_index = entry_key[entry];
            if key_index == u32::MAX {
                let key = key_of(paletted.palette[entry]);
                key_index = *by_key.entry(key).or_insert_with(|| {
                    keys.push(key);
                    (keys.len() - 1) as u32
                });
                entry_key[entry] = key_index;
            }
            // At most CHUNK_VOLUME distinct keys, so the index fits in u16.
            cells[cell] = key_index as u16;
        });
        out.extend_from_slice(&(keys.len() as u32).to_le_bytes());
        for key in &keys {
            write_key(out, key);
        }
        if keys.len() == 1 {
            out.push(0);
            return;
        }
        let bits = min_bits(keys.len());
        out.push(bits);
        for word in pack(&cells, bits).iter() {
            out.extend_from_slice(&word.to_le_bytes());
        }
    }
}

impl Default for ChunkData {
    /// An all-air chunk.
    fn default() -> Self {
        Self::Uniform(BlockStateId::AIR)
    }
}

impl PartialEq for ChunkData {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Uniform(a), Self::Uniform(b)) => a == b,
            // A paletted chunk uses at least two states, so it never equals a uniform one.
            (Self::Uniform(_), Self::Paletted(_)) | (Self::Paletted(_), Self::Uniform(_)) => false,
            (Self::Paletted(a), Self::Paletted(b)) => {
                if a.bits == b.bits && a.palette == b.palette {
                    // Distinct palette entries: equal cells have equal indices.
                    return a.data == b.data;
                }
                (0..CHUNK_VOLUME).all(|index| a.get(index) == b.get(index))
            }
        }
    }
}

impl Eq for ChunkData {}

impl fmt::Debug for ChunkData {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Uniform(state) => f.debug_tuple("Uniform").field(state).finish(),
            Self::Paletted(paletted) => fmt::Debug::fmt(paletted, f),
        }
    }
}

impl fmt::Debug for Paletted {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let used: Vec<(BlockStateId, u16)> = self
            .palette
            .iter()
            .zip(&self.counts)
            .filter(|(_, count)| **count > 0)
            .map(|(&state, &count)| (state, count))
            .collect();
        f.debug_struct("Paletted")
            .field("bits", &self.bits)
            .field("palette_len", &self.palette.len())
            .field("used_states_and_counts", &used)
            .finish()
    }
}

/// Iterator over the distinct states of a chunk, see [`ChunkData::states`].
#[derive(Clone, Debug)]
pub struct States<'a> {
    inner: StatesInner<'a>,
}

#[derive(Clone, Debug)]
enum StatesInner<'a> {
    Uniform(Option<BlockStateId>),
    Paletted(core::iter::Zip<core::slice::Iter<'a, BlockStateId>, core::slice::Iter<'a, u16>>),
}

impl Iterator for States<'_> {
    type Item = BlockStateId;

    fn next(&mut self) -> Option<BlockStateId> {
        match &mut self.inner {
            StatesInner::Uniform(state) => state.take(),
            StatesInner::Paletted(entries) => entries
                .find(|(_, count)| **count > 0)
                .map(|(&state, _)| state),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn id(raw: u32) -> BlockStateId {
        BlockStateId(raw)
    }

    /// Checks the documented invariants of a paletted chunk.
    fn check(data: &ChunkData) {
        if let ChunkData::Paletted(p) = data {
            assert!(matches!(p.bits, 1 | 2 | 4 | 8 | 16));
            assert!(p.palette.len() <= 1 << p.bits);
            assert_eq!(p.data.len(), WORDS_PER_BIT * usize::from(p.bits));
            assert_eq!(p.palette.len(), p.counts.len());
            let mut counts = vec![0_usize; p.palette.len()];
            for index in 0..CHUNK_VOLUME {
                counts[p.entry_of(index)] += 1;
            }
            assert_eq!(
                counts,
                p.counts.iter().map(|&c| usize::from(c)).collect::<Vec<_>>()
            );
            for (i, a) in p.palette.iter().enumerate() {
                for b in &p.palette[i + 1..] {
                    assert_ne!(a, b, "palette entries must be distinct");
                }
            }
            assert!(p.used_entries() >= 2, "a paletted chunk uses two states");
        }
    }

    #[test]
    fn locate_matches_the_documented_layout() {
        for bits in [1_u8, 2, 4, 8, 16] {
            let per_word = 64 / usize::from(bits);
            for index in [0, 1, 63, 64, 100, 1000, CHUNK_VOLUME - 1] {
                let (word, offset) = locate(bits, index);
                assert_eq!(word, index / per_word);
                assert_eq!(offset as usize, (index % per_word) * usize::from(bits));
            }
        }
        assert_eq!(min_bits(2), 1);
        assert_eq!(min_bits(3), 2);
        assert_eq!(min_bits(16), 4);
        assert_eq!(min_bits(17), 8);
        assert_eq!(min_bits(256), 8);
        assert_eq!(min_bits(257), 16);
        assert_eq!(min_bits(65_536), 16);
    }

    #[test]
    fn set_grows_and_collapses() {
        let mut data = ChunkData::uniform(id(7));
        assert_eq!(data.bits(), 0);
        data.set(5, id(7));
        assert!(
            data.is_uniform(),
            "setting the uniform state changes nothing"
        );
        data.set(5, id(8));
        check(&data);
        assert_eq!(data.bits(), 1);
        assert_eq!(data.get(5), id(8));
        assert_eq!(data.get(4), id(7));
        assert_eq!(data.count(id(8)), 1);
        // Four states need two bits, five need four.
        data.set(6, id(9));
        check(&data);
        assert_eq!(data.bits(), 2);
        data.set(7, id(10));
        assert_eq!(data.bits(), 2);
        data.set(8, id(11));
        check(&data);
        assert_eq!(data.bits(), 4);
        // A free slot is reused before the palette grows.
        data.set(8, id(7));
        let ChunkData::Paletted(p) = &data else {
            panic!("still paletted")
        };
        assert_eq!(p.palette.len(), 5);
        data.set(9, id(12));
        let ChunkData::Paletted(p) = &data else {
            panic!("still paletted")
        };
        assert_eq!(p.palette.len(), 5, "slot of state 11 reused");
        check(&data);
        // Reverting every cell collapses to Uniform at once.
        for index in [5, 6, 7, 9] {
            data.set(index, id(7));
        }
        assert_eq!(data, ChunkData::uniform(id(7)));
        assert!(data.is_uniform());
    }

    #[test]
    fn sixteen_bit_palettes_and_compaction() {
        let mut data = ChunkData::from_fn(|index| id(index as u32 % 300));
        check(&data);
        assert_eq!(data.bits(), 16);
        for index in 0..CHUNK_VOLUME {
            assert_eq!(data.get(index), id(index as u32 % 300));
        }
        // Reduce to three states, then compact to two bits.
        for index in 0..CHUNK_VOLUME {
            data.set(index, id(index as u32 % 3));
        }
        check(&data);
        assert_eq!(data.bits(), 16, "set never narrows the indices");
        data.compact();
        check(&data);
        assert_eq!(data.bits(), 2);
        for index in 0..CHUNK_VOLUME {
            assert_eq!(data.get(index), id(index as u32 % 3));
        }
        // Cycling through more states than 16 bits can index works: free slots are reused.
        let mut data = ChunkData::uniform(id(0));
        for state in 1..70_000 {
            data.set(1, id(state));
            data.set(2, id(state + 1));
        }
        check(&data);
        assert_eq!(data.get(1), id(69_999));
        assert_eq!(data.get(2), id(70_000));
        assert!(data.bits() <= 2);
    }

    #[test]
    fn overlay_with_unchanged() {
        let unchanged = BlockStateId::UNCHANGED;
        let mut layer = ChunkData::uniform(unchanged);
        layer.set(3, id(5));
        let base = ChunkData::uniform(id(1));
        let composite = layer.or(&base);
        check(&composite);
        assert_eq!(composite.get(3), id(5));
        assert_eq!(composite.get(4), id(1));
        assert!(!composite.contains(unchanged));
        // Base state already in the layer's palette: general path.
        let mut layer = ChunkData::uniform(unchanged);
        layer.set(3, id(1));
        layer.set(4, id(2));
        layer.set(4, unchanged);
        let composite = layer.or(&base);
        assert_eq!(composite, ChunkData::uniform(id(1)));
        // Paletted base.
        let base = ChunkData::from_fn(|index| id(10 + (index % 4) as u32));
        let mut layer = ChunkData::uniform(unchanged);
        layer.set(0, id(3));
        let composite = layer.or(&base);
        for index in 0..CHUNK_VOLUME {
            let expected = if index == 0 {
                id(3)
            } else {
                id(10 + (index % 4) as u32)
            };
            assert_eq!(composite.get(index), expected);
        }
        assert_eq!(ChunkData::uniform(unchanged).or(&base), base);
        assert_eq!(
            ChunkData::uniform(id(9)).or(&base),
            ChunkData::uniform(id(9))
        );
    }

    #[test]
    fn states_lists_used_entries_only() {
        let mut data = ChunkData::uniform(id(1));
        assert_eq!(data.states().collect::<Vec<_>>(), [id(1)]);
        data.set(0, id(2));
        data.set(1, id(3));
        data.set(1, id(1));
        assert_eq!(data.states().collect::<Vec<_>>(), [id(1), id(2)]);
        assert!(data.contains(id(2)) && !data.contains(id(3)));
    }

    #[test]
    fn memory_accounting() {
        let uniform = ChunkData::uniform(id(1));
        assert_eq!(uniform.memory_bytes(), size_of::<ChunkData>());
        let two = ChunkData::from_fn(|index| id((index % 2) as u32));
        assert!(two.memory_bytes() >= size_of::<ChunkData>() + WORDS_PER_BIT * 8);
    }
}
