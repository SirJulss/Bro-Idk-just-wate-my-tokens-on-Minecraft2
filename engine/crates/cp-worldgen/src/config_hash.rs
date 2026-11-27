//! The parameter hash that the test generators put into `GeneratorIdentity::macro_hash`.
//!
//! Values are appended as fixed-width little-endian bytes in a fixed order (contract §2.5 rule
//! 4) and hashed with FNV-1a, so the hash is identical on every machine. Session-local block
//! state ids never enter it; the palette contributes its key hash instead.

use cp_core::hash::hash_bytes;

/// Collects the parameters of one generator.
#[derive(Clone, Debug)]
pub(crate) struct ConfigHasher {
    bytes: Vec<u8>,
}

impl ConfigHasher {
    /// Starts with the generator's identity key, NUL-terminated, so that different generators
    /// with equal parameters never share a hash.
    pub(crate) fn new(key: &str) -> Self {
        let mut bytes = Vec::with_capacity(128);
        bytes.extend_from_slice(key.as_bytes());
        bytes.push(0);
        Self { bytes }
    }

    pub(crate) fn u32(&mut self, value: u32) -> &mut Self {
        self.bytes.extend_from_slice(&value.to_le_bytes());
        self
    }

    pub(crate) fn i32(&mut self, value: i32) -> &mut Self {
        self.bytes.extend_from_slice(&value.to_le_bytes());
        self
    }

    pub(crate) fn u64(&mut self, value: u64) -> &mut Self {
        self.bytes.extend_from_slice(&value.to_le_bytes());
        self
    }

    /// Hashes the bit pattern, so `0.0` and `-0.0` differ (they never occur as parameters).
    pub(crate) fn f64(&mut self, value: f64) -> &mut Self {
        self.bytes.extend_from_slice(&value.to_bits().to_le_bytes());
        self
    }

    /// Appends the byte length (`u64`) and then the UTF-8 bytes, so that consecutive strings
    /// never run into each other.
    pub(crate) fn str(&mut self, value: &str) -> &mut Self {
        // usize is 64 bits wide on every supported target (contract §2.5 rule 2).
        self.u64(value.len() as u64);
        self.bytes.extend_from_slice(value.as_bytes());
        self
    }

    pub(crate) fn finish(&self) -> u64 {
        hash_bytes(&self.bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hash_covers_key_values_and_order() {
        let base = ConfigHasher::new("a").u32(1).i32(-2).f64(0.5).finish();
        assert_eq!(
            base,
            ConfigHasher::new("a").u32(1).i32(-2).f64(0.5).finish()
        );
        assert_ne!(
            base,
            ConfigHasher::new("b").u32(1).i32(-2).f64(0.5).finish()
        );
        assert_ne!(
            base,
            ConfigHasher::new("a").u32(2).i32(-2).f64(0.5).finish()
        );
        assert_ne!(
            base,
            ConfigHasher::new("a").i32(-2).u32(1).f64(0.5).finish()
        );
        assert_ne!(
            base,
            ConfigHasher::new("a").u32(1).i32(-2).f64(0.25).finish()
        );
        // The NUL terminator keeps "ab" + 0x01.. apart from "a" + "b"...
        assert_ne!(
            ConfigHasher::new("ab").u32(1).finish(),
            ConfigHasher::new("a").u64(u64::from(b'b')).finish()
        );
        // Pinned: the recipe is FNV-1a over the documented bytes.
        let mut bytes = b"x\0".to_vec();
        bytes.extend_from_slice(&7_u64.to_le_bytes());
        assert_eq!(ConfigHasher::new("x").u64(7).finish(), hash_bytes(&bytes));
    }

    #[test]
    fn strings_are_length_prefixed() {
        let mut bytes = b"x\0".to_vec();
        bytes.extend_from_slice(&3_u64.to_le_bytes());
        bytes.extend_from_slice(b"dir");
        assert_eq!(
            ConfigHasher::new("x").str("dir").finish(),
            hash_bytes(&bytes)
        );
        // The split between two strings matters.
        assert_ne!(
            ConfigHasher::new("x").str("ab").str("c").finish(),
            ConfigHasher::new("x").str("a").str("bc").finish()
        );
        assert_ne!(
            ConfigHasher::new("x").str("").finish(),
            ConfigHasher::new("x").finish()
        );
    }
}
