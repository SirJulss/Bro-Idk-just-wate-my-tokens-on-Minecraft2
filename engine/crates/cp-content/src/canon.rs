//! Canonical byte encoding behind [`Registry::content_hash`](crate::Registry::content_hash).
//!
//! The hash must be the same on every machine and in every run (contract §2.5). So the
//! registry is not hashed through `std::hash::Hash`, whose output is unspecified, but written
//! field by field into a byte string with fixed-width little-endian integers, and that string
//! is hashed with the pinned functions of `cp_core::hash`. Strings and lists carry their length
//! as a prefix, so no two different registries produce the same bytes.

use cp_core::hash::{hash_bytes, hash_combine};

/// Collects the canonical bytes of a value.
#[derive(Debug, Default)]
pub(crate) struct Canon {
    bytes: Vec<u8>,
}

impl Canon {
    pub(crate) fn u8(&mut self, value: u8) {
        self.bytes.push(value);
    }

    pub(crate) fn bool(&mut self, value: bool) {
        self.u8(u8::from(value));
    }

    pub(crate) fn u16(&mut self, value: u16) {
        self.bytes.extend_from_slice(&value.to_le_bytes());
    }

    pub(crate) fn u32(&mut self, value: u32) {
        self.bytes.extend_from_slice(&value.to_le_bytes());
    }

    pub(crate) fn u64(&mut self, value: u64) {
        self.bytes.extend_from_slice(&value.to_le_bytes());
    }

    /// The IEEE bit pattern. Callers store neither NaN nor `-0.0`, so equal values have equal
    /// bits.
    pub(crate) fn f32(&mut self, value: f32) {
        self.u32(value.to_bits());
    }

    /// A length or count, 64 bits wide whatever `usize` is.
    pub(crate) fn count(&mut self, count: usize) {
        self.u64(count as u64);
    }

    /// Length-prefixed UTF-8.
    pub(crate) fn str(&mut self, value: &str) {
        self.count(value.len());
        self.bytes.extend_from_slice(value.as_bytes());
    }

    /// The hash of everything written so far: FNV-1a over the bytes, then mixed with the
    /// length for avalanche.
    pub(crate) fn finish(&self) -> u64 {
        hash_combine(hash_bytes(&self.bytes), self.bytes.len() as u64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encoding_is_little_endian_and_length_prefixed() {
        let mut canon = Canon::default();
        canon.u8(0xAB);
        canon.bool(true);
        canon.bool(false);
        canon.u16(0x0102);
        canon.u32(0x0304_0506);
        canon.u64(0x0708_090A_0B0C_0D0E);
        canon.f32(1.0);
        canon.f32(f32::INFINITY);
        canon.count(3);
        canon.str("hé");
        let expected: Vec<u8> = [
            &[0xAB, 1, 0][..],
            &[0x02, 0x01],
            &[0x06, 0x05, 0x04, 0x03],
            &[0x0E, 0x0D, 0x0C, 0x0B, 0x0A, 0x09, 0x08, 0x07],
            &[0x00, 0x00, 0x80, 0x3F],
            &[0x00, 0x00, 0x80, 0x7F],
            &[3, 0, 0, 0, 0, 0, 0, 0],
            &[3, 0, 0, 0, 0, 0, 0, 0],
            "hé".as_bytes(),
        ]
        .concat();
        assert_eq!(canon.bytes, expected);
    }

    #[test]
    fn hash_is_pinned() {
        // Pins the recipe (FNV-1a, then hash_combine with the length) on every platform.
        assert_eq!(Canon::default().finish(), EMPTY);
        let mut canon = Canon::default();
        canon.str("cubeplanet:stone");
        canon.u32(7);
        assert_eq!(canon.finish(), STONE_7);
        assert_eq!(
            canon.finish(),
            hash_combine(hash_bytes(&canon.bytes), canon.bytes.len() as u64)
        );
    }

    // Computed with an independent implementation (Python) of FNV-1a 64 and cp-core's
    // `hash_combine`, not copied from this crate's output.
    const EMPTY: u64 = 0xc381_7c01_6ba4_ff30;
    const STONE_7: u64 = 0x4aa9_7e63_1a0f_38eb;

    #[test]
    fn boundaries_between_strings_matter() {
        let hash = |parts: &[&str]| {
            let mut canon = Canon::default();
            for part in parts {
                canon.str(part);
            }
            canon.finish()
        };
        assert_ne!(hash(&["ab", "c"]), hash(&["a", "bc"]));
        assert_ne!(hash(&["abc"]), hash(&["abc", ""]));
        assert_ne!(hash(&[]), hash(&[""]));
        assert_eq!(hash(&["ab", "c"]), hash(&["ab", "c"]));
    }
}
