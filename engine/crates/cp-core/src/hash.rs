//! Deterministic hashing and random numbers (contract §2.5, rule 3).
//!
//! Every function here is a fixed, platform-independent bit recipe. Generated worlds and saved
//! games depend on the exact outputs, so **none of these functions may ever change behaviour**;
//! the tests pin them with known vectors.
//!
//! These are not cryptographic hashes and offer no protection against crafted input.
//!
//! ```
//! use cp_core::hash::{Rng, hash_ivec3};
//! use cp_core::math::IVec3;
//!
//! // One generator per chunk, derived from the world seed and the chunk position ...
//! let chunk_seed = hash_ivec3(0xC0FFEE, IVec3::new(3, -1, 7));
//! let rng = Rng::new(chunk_seed, 0);
//! // ... and one independent stream per purpose, whatever the order of use.
//! let mut trees = rng.fork(1);
//! let mut ores = rng.fork(2);
//! let tree_count = trees.range_u32(0, 5);
//! let ore_depth = ores.range_f64(8.0, 40.0);
//! assert!(tree_count < 5 && (8.0..40.0).contains(&ore_depth));
//! ```

use glam::IVec3;
use serde::{Deserialize, Serialize};

/// 2^64 / φ, the usual odd "golden ratio" increment.
const GOLDEN: u64 = 0x9E37_79B9_7F4A_7C15;

/// Finaliser of SplitMix64 (Stafford's "Mix13"): a bijection on `u64` with full avalanche.
///
/// `mix64(0) == 0`; combine with a seed through [`hash_combine`] when that matters.
#[inline]
#[must_use]
pub const fn mix64(x: u64) -> u64 {
    let mut z = x;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Combines a running hash `seed` with one more 64-bit `value`.
///
/// The result depends on the order of the arguments and is a bijection in each argument for a
/// fixed other one, so chaining `hash_combine(hash_combine(seed, a), b)` never loses entropy.
/// `value` is scrambled before it meets `seed`; nearby seeds therefore give unrelated hashes.
#[inline]
#[must_use]
pub const fn hash_combine(seed: u64, value: u64) -> u64 {
    mix64(seed.wrapping_add(GOLDEN) ^ mix64(value))
}

/// Hash of an integer lattice point (voxel, chunk position, …) under `seed`.
///
/// All 2^96 coordinate triples contribute their full bits. Use it for per-cell decisions in
/// world generation; for smooth fields use [`crate::noise`].
#[inline]
#[must_use]
pub const fn hash_ivec3(seed: u64, p: IVec3) -> u64 {
    // Two's complement bits, zero-extended: negative coordinates hash as distinct values.
    let xy = (p.x as u32 as u64) | ((p.y as u32 as u64) << 32);
    let z = p.z as u32 as u64;
    hash_combine(hash_combine(seed, xy), z)
}

/// FNV-1a, 64 bit, over a byte string.
///
/// Stable across platforms because it consumes bytes; hash multi-byte integers through their
/// little-endian encoding (contract §2.5, rule 4).
#[inline]
#[must_use]
pub const fn hash_bytes(bytes: &[u8]) -> u64 {
    let mut hash: u64 = 0xCBF2_9CE4_8422_2325;
    let mut i = 0;
    while i < bytes.len() {
        hash ^= bytes[i] as u64;
        hash = hash.wrapping_mul(0x0000_0100_0000_01B3);
        i += 1;
    }
    hash
}

/// FNV-1a, 64 bit, over the UTF-8 bytes of `s`. Usable in constants for content keys.
#[inline]
#[must_use]
pub const fn hash_str(s: &str) -> u64 {
    hash_bytes(s.as_bytes())
}

/// Seeded pseudo random number generator: PCG-XSH-RR 64/32 (O'Neill, `pcg32`).
///
/// 64 bits of state, 32 bits of output per step, period 2^64 per stream and 2^63 independent
/// streams. The sequence for a given `(seed, stream)` is identical on every platform; this is the
/// only source of randomness allowed in deterministic crates.
///
/// Deliberately not `Copy`: copying a generator by accident repeats its numbers.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Rng {
    state: u64,
    /// Stream selector, always odd.
    inc: u64,
}

impl Rng {
    const MULTIPLIER: u64 = 6_364_136_223_846_793_005;

    /// Creates the generator for `seed` on stream `stream` (`pcg32_srandom_r`).
    ///
    /// Streams that differ only in the top bit coincide (the selector keeps 63 bits).
    #[must_use]
    pub const fn new(seed: u64, stream: u64) -> Self {
        let mut rng = Self {
            state: 0,
            inc: (stream << 1) | 1,
        };
        rng.step();
        rng.state = rng.state.wrapping_add(seed);
        rng.step();
        rng
    }

    #[inline]
    const fn step(&mut self) {
        self.state = self
            .state
            .wrapping_mul(Self::MULTIPLIER)
            .wrapping_add(self.inc);
    }

    /// Next 32 uniformly distributed bits.
    #[inline]
    pub const fn next_u32(&mut self) -> u32 {
        let old = self.state;
        self.step();
        // XSH-RR: xorshift the high bits down, then rotate by the top five bits.
        let xorshifted = (((old >> 18) ^ old) >> 27) as u32;
        xorshifted.rotate_right((old >> 59) as u32)
    }

    /// Next 64 uniformly distributed bits: two steps, the first one supplies the high half.
    #[inline]
    pub const fn next_u64(&mut self) -> u64 {
        let high = self.next_u32() as u64;
        let low = self.next_u32() as u64;
        (high << 32) | low
    }

    /// Uniform `f32` in `[0, 1)` with 24 random bits (every result is a multiple of 2^-24).
    #[inline]
    pub const fn f32(&mut self) -> f32 {
        (self.next_u32() >> 8) as f32 * (1.0 / 16_777_216.0)
    }

    /// Uniform `f64` in `[0, 1)` with 53 random bits (every result is a multiple of 2^-53).
    #[inline]
    pub const fn f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / 9_007_199_254_740_992.0)
    }

    /// Uniform integer in the half-open range `[lo, hi)`, without modulo bias.
    ///
    /// Returns `lo` when the range is empty (`hi <= lo`). Uses Lemire's multiply-and-reject
    /// method, so it consumes one step in almost all cases and occasionally more.
    #[inline]
    pub const fn range_u32(&mut self, lo: u32, hi: u32) -> u32 {
        if hi <= lo {
            return lo;
        }
        let span = hi - lo;
        let mut product = self.next_u32() as u64 * span as u64;
        if (product as u32) < span {
            // 2^32 mod span: draws whose low half falls below it would be over-represented.
            let threshold = span.wrapping_neg() % span;
            while (product as u32) < threshold {
                product = self.next_u32() as u64 * span as u64;
            }
        }
        lo + (product >> 32) as u32
    }

    /// Uniform `f64` in the half-open range `[lo, hi)`.
    ///
    /// Returns `lo` when the range is empty or inverted. Both bounds must be finite.
    #[inline]
    pub const fn range_f64(&mut self, lo: f64, hi: f64) -> f64 {
        let value = lo + (hi - lo) * self.f64();
        // Rounding can land exactly on `hi`; fold that case back to keep the range half-open.
        if value < hi { value } else { lo }
    }

    /// Derives an independent generator for the sub-stream `stream` without advancing `self`.
    ///
    /// The child depends only on the current state of `self` and on `stream`, so the order in
    /// which several children are forked does not matter:
    /// `rng.fork(1)` is the same generator whether or not `rng.fork(2)` was created before.
    /// Forking again after drawing numbers from `self` gives a different child.
    #[must_use]
    pub const fn fork(&self, stream: u64) -> Self {
        Self::new(
            hash_combine(self.state, stream),
            hash_combine(self.inc, stream),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pcg_reference_vector() {
        // Output of the reference implementation's demo (pcg32, seed 42, stream 54).
        let mut rng = Rng::new(42, 54);
        let expected: [u32; 6] = [
            0xa15c_02b7,
            0x7b47_f409,
            0xba1d_3330,
            0x83d2_f293,
            0xbfa4_784b,
            0xcbed_606e,
        ];
        for value in expected {
            assert_eq!(rng.next_u32(), value);
        }
    }

    #[test]
    fn pcg_matches_textbook_formulation() {
        // The C reference, transcribed literally, as an independent second implementation.
        struct Pcg32 {
            state: u64,
            inc: u64,
        }
        fn pcg32_random_r(rng: &mut Pcg32) -> u32 {
            let oldstate = rng.state;
            rng.state = oldstate
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(rng.inc);
            let xorshifted = (((oldstate >> 18) ^ oldstate) >> 27) as u32;
            let rot = (oldstate >> 59) as u32;
            (xorshifted >> rot) | (xorshifted << (rot.wrapping_neg() & 31))
        }
        fn pcg32_srandom_r(initstate: u64, initseq: u64) -> Pcg32 {
            let mut rng = Pcg32 {
                state: 0,
                inc: (initseq << 1) | 1,
            };
            pcg32_random_r(&mut rng);
            rng.state = rng.state.wrapping_add(initstate);
            pcg32_random_r(&mut rng);
            rng
        }
        for (seed, stream) in [
            (0, 0),
            (1, 2),
            (42, 54),
            (u64::MAX, u64::MAX),
            (0xDEAD_BEEF, 7),
        ] {
            let mut ours = Rng::new(seed, stream);
            let mut reference = pcg32_srandom_r(seed, stream);
            for _ in 0..1000 {
                assert_eq!(ours.next_u32(), pcg32_random_r(&mut reference));
            }
        }
    }

    #[test]
    fn next_u64_is_high_then_low() {
        let mut a = Rng::new(7, 3);
        let mut b = a.clone();
        let high = u64::from(b.next_u32());
        let low = u64::from(b.next_u32());
        assert_eq!(a.next_u64(), (high << 32) | low);
    }

    #[test]
    fn floats_stay_in_unit_interval() {
        let mut rng = Rng::new(1, 1);
        let (mut min32, mut max32, mut min64, mut max64) = (1.0_f32, 0.0_f32, 1.0_f64, 0.0_f64);
        let mut sum = 0.0;
        for _ in 0..200_000 {
            let a = rng.f32();
            let b = rng.f64();
            assert!((0.0..1.0).contains(&a));
            assert!((0.0..1.0).contains(&b));
            min32 = crate::det::min(min32, a);
            max32 = crate::det::max(max32, a);
            min64 = crate::det::min(min64, b);
            max64 = crate::det::max(max64, b);
            sum += b;
        }
        assert!(min32 < 1e-4 && max32 > 0.9999 && min64 < 1e-4 && max64 > 0.9999);
        assert!((sum / 200_000.0 - 0.5).abs() < 0.005);
        // The extreme raw outputs map to the ends of the half-open interval.
        assert_eq!(
            (u32::MAX >> 8) as f32 * (1.0 / 16_777_216.0),
            1.0 - f32::EPSILON / 2.0
        );
        assert_eq!(
            (u64::MAX >> 11) as f64 * (1.0 / 9_007_199_254_740_992.0),
            1.0 - f64::EPSILON / 2.0
        );
    }

    #[test]
    fn range_u32_is_in_range_and_unbiased() {
        let mut rng = Rng::new(99, 5);
        assert_eq!(rng.range_u32(5, 5), 5);
        assert_eq!(rng.range_u32(9, 3), 9);
        assert_eq!(rng.range_u32(17, 18), 17);
        // Small spans: every value appears with frequency 1/span.
        for span in [2_u32, 3, 5, 6, 7, 10] {
            let mut counts = vec![0_u32; span as usize];
            let draws = 60_000 * span;
            for _ in 0..draws {
                let v = rng.range_u32(100, 100 + span);
                assert!((100..100 + span).contains(&v));
                counts[(v - 100) as usize] += 1;
            }
            for (value, &count) in counts.iter().enumerate() {
                let deviation = (f64::from(count) - 60_000.0).abs() / 60_000.0;
                assert!(deviation < 0.02, "span {span}, value {value}: {count}");
            }
        }
        // A span just above 2^31 makes plain modulo reduction twice as likely for the low half
        // of the range; the unbiased method keeps both halves equal.
        let span = (1_u32 << 31) + (1 << 30);
        let mut low_half = 0_u32;
        for _ in 0..200_000 {
            if rng.range_u32(0, span) < span / 2 {
                low_half += 1;
            }
        }
        assert!(
            (f64::from(low_half) / 200_000.0 - 0.5).abs() < 0.01,
            "{low_half}"
        );
        // Full span.
        let mut seen_high = false;
        for _ in 0..64 {
            seen_high |= rng.range_u32(0, u32::MAX) > u32::MAX / 2;
        }
        assert!(seen_high);
    }

    #[test]
    fn range_f64_is_half_open() {
        let mut rng = Rng::new(3, 4);
        for _ in 0..100_000 {
            let v = rng.range_f64(-2.5, 7.25);
            assert!((-2.5..7.25).contains(&v));
        }
        assert_eq!(rng.range_f64(4.0, 4.0), 4.0);
        assert_eq!(rng.range_f64(4.0, 1.0), 4.0);
        // Adjacent floats: the only value in [lo, hi) is lo.
        let lo = 1.0_f64;
        let hi = f64::from_bits(lo.to_bits() + 1);
        for _ in 0..1000 {
            assert_eq!(rng.range_f64(lo, hi), lo);
        }
    }

    #[test]
    fn fork_is_order_independent_and_distinct() {
        let rng = Rng::new(123, 456);
        let mut a1 = rng.fork(1);
        let mut b = rng.fork(2);
        let mut a2 = rng.fork(1);
        assert_eq!(a1, a2);
        let xs: Vec<u32> = (0..16).map(|_| a1.next_u32()).collect();
        let ys: Vec<u32> = (0..16).map(|_| b.next_u32()).collect();
        let zs: Vec<u32> = (0..16).map(|_| a2.next_u32()).collect();
        assert_eq!(xs, zs);
        assert_ne!(xs, ys);
        // The parent is untouched, and a parent in another state forks differently.
        let mut parent = rng.clone();
        assert_eq!(parent, Rng::new(123, 456));
        parent.next_u32();
        assert_ne!(parent.fork(1), rng.fork(1));
    }

    #[test]
    fn fnv1a_reference_vectors() {
        assert_eq!(hash_bytes(b""), 0xcbf2_9ce4_8422_2325);
        assert_eq!(hash_bytes(b"a"), 0xaf63_dc4c_8601_ec8c);
        assert_eq!(hash_bytes(b"foobar"), 0x8594_4171_f739_67e8);
        assert_eq!(hash_str("foobar"), hash_bytes(b"foobar"));
        const KEY: u64 = hash_str("cubeplanet:stone");
        assert_eq!(KEY, hash_bytes(b"cubeplanet:stone"));
    }

    #[test]
    fn mix64_reference_vectors() {
        // SplitMix64 with seed 0 produces mix64(k * GOLDEN) for k = 1, 2, 3.
        assert_eq!(mix64(GOLDEN), 0xE220_A839_7B1D_CDAF);
        assert_eq!(mix64(GOLDEN.wrapping_mul(2)), 0x6E78_9E6A_A1B9_65F4);
        assert_eq!(mix64(GOLDEN.wrapping_mul(3)), 0x06C4_5D18_8009_454F);
        assert_eq!(mix64(0), 0);
    }

    #[test]
    fn mix64_is_a_bijection_with_avalanche() {
        // Inverse of each step; a bijection must round-trip.
        fn unmix64(mut z: u64) -> u64 {
            z ^= (z >> 31) ^ (z >> 62);
            z = z.wrapping_mul(0x3196_42B2_D24D_8EC3);
            z ^= (z >> 27) ^ (z >> 54);
            z = z.wrapping_mul(0x96DE_1B17_3F11_9089);
            z ^ (z >> 30) ^ (z >> 60)
        }
        let mut rng = Rng::new(5, 5);
        let mut flipped = 0_u64;
        for _ in 0..20_000 {
            let x = rng.next_u64();
            assert_eq!(unmix64(mix64(x)), x);
            let bit = rng.range_u32(0, 64);
            flipped += u64::from((mix64(x) ^ mix64(x ^ (1 << bit))).count_ones());
        }
        // Flipping one input bit flips half of the output bits on average.
        let average = flipped as f64 / 20_000.0;
        assert!((average - 32.0).abs() < 0.5, "{average}");
    }

    #[test]
    fn combined_hashes_are_pinned() {
        // hash_combine(0, 0) = mix64(GOLDEN), the first SplitMix64 output for seed 0.
        assert_eq!(hash_combine(0, 0), 0xE220_A839_7B1D_CDAF);
        let actual = [
            hash_combine(1, 2),
            hash_combine(u64::MAX, 0),
            hash_ivec3(0, IVec3::ZERO),
            hash_ivec3(42, IVec3::new(1, -2, 3)),
            hash_ivec3(42, IVec3::new(i32::MIN, i32::MAX, -1)),
        ];
        assert!(actual == PINNED_COMBINED, "hashes changed: {actual:#018x?}");
        // A forked generator is pinned through its first outputs.
        let mut child = Rng::new(42, 54).fork(7);
        let outputs = [child.next_u32(), child.next_u32(), child.next_u32()];
        assert!(outputs == PINNED_FORK, "fork changed: {outputs:#010x?}");
        // range_u32 consumes the stream in a fixed way.
        let mut rng = Rng::new(42, 54);
        let ranged = [
            rng.range_u32(0, 6),
            rng.range_u32(10, 20),
            rng.range_u32(0, 1_000_000),
        ];
        assert!(ranged == PINNED_RANGES, "range_u32 changed: {ranged:?}");
    }

    const PINNED_COMBINED: [u64; 5] = [
        0x83b9_7bf5_8b91_76c8,
        0xe4d9_7177_1b65_2c20,
        0xa706_dd2f_4d19_7e6f,
        0x8f34_6c92_f6b7_1ab1,
        0x2fe7_fa97_1b26_8458,
    ];
    const PINNED_FORK: [u32; 3] = [0xb80b_d308, 0x2296_eefc, 0x4e3a_5791];
    const PINNED_RANGES: [u32; 3] = [3, 14, 727_008];

    #[test]
    fn hash_combine_depends_on_order_and_both_arguments() {
        assert_ne!(hash_combine(1, 2), hash_combine(2, 1));
        assert_ne!(hash_combine(1, 2), hash_combine(1, 3));
        assert_ne!(hash_combine(1, 2), hash_combine(0, 2));
    }

    #[test]
    fn hash_ivec3_has_no_collisions_or_bias_on_a_grid() {
        use std::collections::HashSet;
        let mut seen = HashSet::new();
        let mut ones = [0_u32; 64];
        let mut count = 0_u32;
        for x in -20..20 {
            for y in -20..20 {
                for z in -20..20 {
                    let h = hash_ivec3(7, IVec3::new(x, y, z));
                    assert!(seen.insert(h), "collision at {x} {y} {z}");
                    for (bit, slot) in ones.iter_mut().enumerate() {
                        *slot += ((h >> bit) & 1) as u32;
                    }
                    count += 1;
                }
            }
        }
        for (bit, &n) in ones.iter().enumerate() {
            let p = f64::from(n) / f64::from(count);
            assert!((p - 0.5).abs() < 0.01, "bit {bit}: {p}");
        }
        // Axis order and sign matter, and so does the seed.
        let p = IVec3::new(1, 2, 3);
        assert_ne!(hash_ivec3(7, p), hash_ivec3(7, IVec3::new(3, 2, 1)));
        assert_ne!(hash_ivec3(7, p), hash_ivec3(7, -p));
        assert_ne!(hash_ivec3(7, p), hash_ivec3(8, p));
    }
}
