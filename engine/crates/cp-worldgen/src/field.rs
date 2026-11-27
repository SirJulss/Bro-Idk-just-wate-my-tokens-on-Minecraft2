//! Band-limited fractal noise (contract §5.1): each level sees only the octaves it can show.

use cp_core::coords::MAX_LOD_LEVEL;
use cp_core::math::DVec2;
use cp_core::noise::Fbm;

/// An octave is kept at level `L` only while its wavelength spans at least this many cells of
/// `L`; finer octaves are dropped there.
///
/// Four cells per period is twice the Nyquist limit: a level never samples detail it cannot
/// represent (no aliasing, no flicker between levels), and with octaves of halving wavelength
/// two neighbouring levels differ by at most one octave, whose amplitude is a small fraction of
/// the coarser cell.
pub const BAND_LIMIT_CELLS: f64 = 4.0;

/// Levels `0..=MAX_LOD_LEVEL`.
const LEVELS: usize = MAX_LOD_LEVEL as usize + 1;
/// Frequency ratio between octaves: 2, so that octave wavelengths halve exactly.
pub(crate) const LACUNARITY: f64 = 2.0;
/// Amplitude ratio between octaves.
pub(crate) const GAIN: f64 = 0.5;
/// `cp_core::noise::Fbm` ignores octaves beyond this.
pub(crate) const MAX_OCTAVES: u32 = 32;

/// Which fractal sum of `cp_core::noise::Fbm` a channel uses.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum NoiseShape {
    /// `Fbm::sample2`.
    Plain,
    /// `Fbm::ridged2`.
    Ridged,
    /// `Fbm::billow2`: rounded bumps between sharp creases (every octave `2|n| - 1`). Its
    /// octaves do not depend on each other, so the levels' partial sums are prefixes of the
    /// full sum as for the plain shape.
    Billow,
}

/// The octaves one level keeps.
#[derive(Clone, Copy, Debug, PartialEq)]
struct LevelNoise {
    /// The fractal sum of the kept octaves (a prefix of the full sum: octave `i` of an `Fbm`
    /// does not depend on how many octaves follow it).
    fbm: Fbm,
    /// Kept share of the full amplitude, `sum(gain^i, kept) / sum(gain^i, all)`, in `[0, 1]`.
    share: f64,
}

/// One 2D noise channel with per-level band limiting.
///
/// At full resolution it is `Fbm` with `octaves` octaves of the given first `wavelength`,
/// lacunarity 2 and gain 0.5. Level `L` keeps the octaves whose wavelength is at least
/// [`BAND_LIMIT_CELLS`]` * 2^L` m and returns their **partial sum** in units of the full sum's
/// amplitude: the value at level 0 lies in `[-1, 1]`, at level `L` in
/// `[-share(L), share(L)]`, and leaving out octaves never rescales the ones that remain.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct BandLimitedNoise {
    seed: u64,
    frequency: f64,
    shape: NoiseShape,
    levels: [LevelNoise; LEVELS],
}

impl BandLimitedNoise {
    /// `wavelength` in metres (> 0), `octaves` at most [`MAX_OCTAVES`] (more are ignored).
    pub(crate) fn new(seed: u64, wavelength: f64, octaves: u32, shape: NoiseShape) -> Self {
        let octaves = octaves.min(MAX_OCTAVES);
        let total = amplitude_sum(octaves);
        let levels = core::array::from_fn(|level| {
            // `level < LEVELS = 21`, so the conversion is lossless.
            let kept = kept_octaves(wavelength, octaves, level as u8);
            let share = if total > 0.0 {
                amplitude_sum(kept) / total
            } else {
                0.0
            };
            LevelNoise {
                fbm: Fbm::new(kept, LACUNARITY, GAIN),
                share,
            }
        });
        Self {
            seed,
            frequency: 1.0 / wavelength,
            shape,
            levels,
        }
    }

    /// The channel at `(x, z)` m as level `level` sees it, in `[-share(level), share(level)]`.
    ///
    /// # Panics
    /// If `level > MAX_LOD_LEVEL`.
    #[inline]
    pub(crate) fn sample(&self, level: u8, x: f64, z: f64) -> f64 {
        let noise = &self.levels[usize::from(level)];
        if noise.fbm.octaves == 0 {
            return 0.0;
        }
        let p = DVec2::new(x * self.frequency, z * self.frequency);
        let normalised = match self.shape {
            NoiseShape::Plain => noise.fbm.sample2(self.seed, p),
            NoiseShape::Ridged => noise.fbm.ridged2(self.seed, p),
            NoiseShape::Billow => noise.fbm.billow2(self.seed, p),
        };
        normalised * noise.share
    }

    /// Bound of `|sample(level, ..)|`: the kept share of the full amplitude, 1 at level 0
    /// (unless the channel has no octaves at all).
    pub(crate) fn share(&self, level: u8) -> f64 {
        self.levels[usize::from(level)].share
    }

    /// Number of octaves level `level` keeps.
    #[cfg(test)]
    pub(crate) fn kept(&self, level: u8) -> u32 {
        self.levels[usize::from(level)].fbm.octaves
    }
}

/// How many of the `octaves` octaves (first wavelength `wavelength` m, halving) level `level`
/// keeps: those with a wavelength of at least `BAND_LIMIT_CELLS * 2^level` m.
pub(crate) fn kept_octaves(wavelength: f64, octaves: u32, level: u8) -> u32 {
    let shortest = BAND_LIMIT_CELLS * f64::from(1_u32 << level);
    let mut kept = 0;
    let mut current = wavelength;
    while kept < octaves && current >= shortest {
        kept += 1;
        current *= 1.0 / LACUNARITY;
    }
    kept
}

/// `sum(gain^i, i < octaves)` for the gain of every channel (0.5), accumulated like `Fbm`
/// does.
pub(crate) fn amplitude_sum(octaves: u32) -> f64 {
    let mut sum = 0.0;
    let mut amplitude = 1.0;
    for _ in 0..octaves {
        sum += amplitude;
        amplitude *= GAIN;
    }
    sum
}

#[cfg(test)]
mod tests {
    use super::*;
    use cp_core::hash::Rng;

    #[test]
    fn octaves_are_dropped_at_four_cells_per_wavelength() {
        // 512 m, 7 octaves: wavelengths 512, 256, ..., 8 m.
        let kept: Vec<u32> = (0..=12)
            .map(|level| kept_octaves(512.0, 7, level))
            .collect();
        assert_eq!(kept, [7, 7, 6, 5, 4, 3, 2, 1, 0, 0, 0, 0, 0]);
        // A wavelength exactly at the limit is kept.
        assert_eq!(kept_octaves(16.0, 1, 2), 1);
        assert_eq!(kept_octaves(15.9, 1, 2), 0);
        // 1e9 m halves 27 times before it drops below 4 m.
        assert_eq!(kept_octaves(1.0e9, 40, 0), 28);
        assert_eq!(kept_octaves(1.0e18, 40, 0), 40);
        assert_eq!(kept_octaves(512.0, 0, 0), 0);
    }

    #[test]
    fn levels_are_partial_sums_of_the_full_field() {
        let noise = BandLimitedNoise::new(7, 512.0, 7, NoiseShape::Plain);
        let ridged = BandLimitedNoise::new(9, 300.0, 5, NoiseShape::Ridged);
        let billow = BandLimitedNoise::new(11, 40.0, 3, NoiseShape::Billow);
        assert_eq!(noise.share(0), 1.0);
        assert_eq!(noise.kept(8), 0);
        // 40 m, 20 m and 10 m: levels 0 and 1 keep all three, level 4 none.
        assert_eq!((billow.kept(1), billow.kept(2), billow.kept(4)), (3, 2, 0));
        let mut rng = Rng::new(5, 5);
        for _ in 0..2000 {
            let (x, z) = (rng.range_f64(-1.0e5, 1.0e5), rng.range_f64(-1.0e5, 1.0e5));
            let p = DVec2::new(x, z) / 512.0;
            // Level L carries exactly the first kept(L) octaves of the full sum.
            let full = noise.sample(0, x, z);
            assert!((full - Fbm::new(7, 2.0, 0.5).sample2(7, p)).abs() < 1.0e-15);
            for level in 0..=MAX_LOD_LEVEL {
                let value = noise.sample(level, x, z);
                assert!(value.abs() <= noise.share(level) * (1.0 + 1.0e-12));
                let kept = noise.kept(level);
                let partial: f64 = (0..kept)
                    .map(|i| {
                        // Octave i alone, as the difference of two prefix sums.
                        let upper = Fbm::new(i + 1, 2.0, 0.5).sample2(7, p) * amplitude_sum(i + 1);
                        let lower = Fbm::new(i, 2.0, 0.5).sample2(7, p) * amplitude_sum(i);
                        upper - lower
                    })
                    .sum::<f64>()
                    / amplitude_sum(7);
                assert!((value - partial).abs() < 1.0e-12, "{level}");
                let r = ridged.sample(level, x, z);
                assert!(r.abs() <= ridged.share(level) * (1.0 + 1.0e-12));
                // Billow octaves are independent, so its levels are prefixes as well.
                let b = billow.sample(level, x, z);
                assert!(b.abs() <= billow.share(level) * (1.0 + 1.0e-12));
                let kept = billow.kept(level);
                let q = DVec2::new(x * (1.0 / 40.0), z * (1.0 / 40.0));
                let prefix = Fbm::new(kept, 2.0, 0.5).billow2(11, q) * amplitude_sum(kept)
                    / amplitude_sum(3);
                assert!((b - prefix).abs() < 1.0e-12, "{level}");
            }
            assert_eq!(noise.sample(9, x, z), 0.0);
        }
    }

    #[test]
    fn the_dropped_octave_is_a_small_part_of_the_coarser_cell() {
        // Between two levels at most one octave goes; its amplitude bound relative to the
        // coarser cell is the same on every level for a self-similar sum.
        let noise = BandLimitedNoise::new(1, 512.0, 7, NoiseShape::Plain);
        for level in 0..8_u8 {
            let dropped = noise.share(level) - noise.share(level + 1);
            let cell = cell_size_m(level + 1);
            // 24 m of hill amplitude, as in the default hills.
            assert!(24.0 * dropped <= 0.1 * cell, "{level}: {dropped}");
        }
    }

    fn cell_size_m(level: u8) -> f64 {
        f64::from(1_u32 << level)
    }
}
