//! One bit per cell of a chunk (contract §5.1).

use core::fmt;

use cp_core::coords::CHUNK_VOLUME;

/// Number of `u64` words of a [`CellMask`].
pub const MASK_WORDS: usize = CHUNK_VOLUME / 64;

/// A set of cells of one chunk: 32768 bits in `[u64; 512]`, bit `i` = cell index `i`
/// (`index = x + z * 32 + y * 1024`), i.e. cell `i` is bit `i % 64` of word `i / 64`.
///
/// 4 KiB; chunks keep it boxed and only when at least one bit is set.
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct CellMask {
    words: [u64; MASK_WORDS],
}

impl CellMask {
    /// The empty set.
    #[inline]
    #[must_use]
    pub const fn new() -> Self {
        Self {
            words: [0; MASK_WORDS],
        }
    }

    /// The mask with the given words.
    #[inline]
    #[must_use]
    pub const fn from_words(words: [u64; MASK_WORDS]) -> Self {
        Self { words }
    }

    /// True when cell `index` is in the set.
    ///
    /// # Panics
    /// If `index >= CHUNK_VOLUME`.
    #[inline]
    #[must_use]
    pub fn get(&self, index: usize) -> bool {
        (self.words[index >> 6] >> (index & 63)) & 1 == 1
    }

    /// Adds (`true`) or removes (`false`) cell `index`.
    ///
    /// # Panics
    /// If `index >= CHUNK_VOLUME`.
    #[inline]
    pub fn set(&mut self, index: usize, value: bool) {
        let bit = 1_u64 << (index & 63);
        if value {
            self.words[index >> 6] |= bit;
        } else {
            self.words[index >> 6] &= !bit;
        }
    }

    /// True when no cell is in the set.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.words.iter().all(|&word| word == 0)
    }

    /// Number of cells in the set.
    #[must_use]
    pub fn count_ones(&self) -> usize {
        self.words
            .iter()
            .map(|word| word.count_ones() as usize)
            .sum()
    }

    /// The raw words.
    #[inline]
    #[must_use]
    pub const fn words(&self) -> &[u64; MASK_WORDS] {
        &self.words
    }

    /// The raw words, mutable.
    #[inline]
    #[must_use]
    pub fn words_mut(&mut self) -> &mut [u64; MASK_WORDS] {
        &mut self.words
    }

    /// The cell indices in the set, ascending.
    pub fn iter_ones(&self) -> impl Iterator<Item = usize> + '_ {
        self.words
            .iter()
            .enumerate()
            .flat_map(|(word_index, &word)| {
                let mut word = word;
                core::iter::from_fn(move || {
                    if word == 0 {
                        return None;
                    }
                    let bit = word.trailing_zeros() as usize;
                    word &= word - 1;
                    Some(word_index * 64 + bit)
                })
            })
    }
}

impl Default for CellMask {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Debug for CellMask {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CellMask")
            .field("cells", &self.count_ones())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bits_are_cell_indices() {
        let mut mask = CellMask::new();
        assert!(mask.is_empty());
        for index in [0, 1, 63, 64, 1000, CHUNK_VOLUME - 1] {
            mask.set(index, true);
            assert!(mask.get(index));
        }
        assert_eq!(mask.count_ones(), 6);
        assert_eq!(mask.words()[0], 1 | 2 | (1 << 63));
        assert_eq!(mask.words()[1], 1);
        assert_eq!(mask.words()[MASK_WORDS - 1], 1 << 63);
        assert_eq!(
            mask.iter_ones().collect::<Vec<_>>(),
            [0, 1, 63, 64, 1000, CHUNK_VOLUME - 1]
        );
        mask.set(63, false);
        assert!(!mask.get(63));
        assert_eq!(mask.count_ones(), 5);
        assert_eq!(CellMask::from_words(*mask.words()), mask);
        assert_eq!(size_of::<CellMask>(), 4096);
    }
}
