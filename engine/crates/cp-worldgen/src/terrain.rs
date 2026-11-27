//! The terrain of the hills test landscape: a band-limited height field, the surface kind of
//! each column and its layers.
//!
//! Height at `(x, z)` m as level `L` sees it:
//!
//! ```text
//! h_L = c_L + M * micro_mask_L * shore(c_L) * micro_L
//! c_L = base + A * (wc * hills_L(bend_L(x, z)) + wf * fine_L(bend_L(x, z)))
//!       + R * mask_L * ((ridges_L(warp(x, z)) + 1) / 2)^2
//! ```
//!
//! - `hills_L`: fractal Brownian motion (`Fbm::sample2`), the rolling hills, its octaves from
//!   the hill wavelength down to [`FINE_BELOW`] m; `fine_L` the octaves below that, with
//!   [`FINE_WEIGHT`] of their natural amplitude (`wc`, `wf`: the octaves' shares of the
//!   amplitude of the whole sum). Ripples of a few metres made the height hover around a cell
//!   boundary on flat ground, which broke it into single pits and humps of one metre. Both are
//!   sampled at coordinates bent by a gentle domain warp ([`HILL_WARP`], the same on every
//!   level), so that the contour lines around a hill are not concentric rings, and by a fine
//!   warp of 40 m ([`FINE_WARP`], band-limited: levels 0–3 only), so that the one-metre steps
//!   of a smooth slope meander and break instead of running on as ruled lines;
//! - `ridges_L`: a ridged multifractal (`Fbm::ridged2`), sharp crests near +1, sampled at
//!   coordinates bent by a domain warp of [`RIDGE_WARP`] (the same on every level), so peaks
//!   are lopsided and their crests bend instead of forming round cones;
//! - `mask_L = smoothstep(RIDGE_MASK_LOW, RIDGE_MASK_HIGH, regions_L)`: ridges only rise in
//!   a few regions, elsewhere the term is zero;
//! - `micro_L`: the **micro relief**, billow noise (`Fbm::billow2`) of [`MICRO_NOISE`]: rounded
//!   hummocks and hollows between sharp creases, [`MICRO_AMPLITUDE`] m at most, in full in the
//!   regions a [`MICRO_REGION_NOISE`] mask picks (hummocky moraine and pasture there, gently
//!   undulating meadows with half of it elsewhere), faded out at the water's edge and kept at
//!   half its height on the low ground of a shore, so that shores keep their beaches but their
//!   slopes do not rise in parallel terraces. Without it the field had almost no height below 32 m of wavelength,
//!   and gentle slopes broke into long, evenly spaced, concentric one-metre terraces that read
//!   as a contour map from afar; with it the steps break into short, irregular pieces between
//!   bumps and dips;
//!
//! each channel band-limited per level ([`crate::field`]). The height therefore lies in
//! `[base - A - M, base + A + R + M]` on every level.
//!
//! # Settled columns
//! The cells of a column follow its **settled** height ([`Terrain::settle`]): two passes of a
//! 3 × 3 median filter over the columns of the level. Where a smooth field hovers around a
//! cell boundary, quantising it to whole cells leaves single pits, humps, notches and tiny
//! plateaus of one or two cells, which from afar read as dark fins and dotted contour lines;
//! the median removes them and keeps slopes, terraces and cliffs. The settled height of a
//! column is one of the heights within two columns of it, so every bound of the field holds.
//!
//! # Surfaces
//! What covers a column depends on its height, its **slope** (the gradient of the level's
//! height field from the neighbouring columns of the same level) and a few band-limited noise
//! fields, so that the land is not banded by height alone ([`Terrain::surface`]):
//!
//! - lake floors (the columns under water) are sand, with gravel in the deep, on steep floors
//!   and on stony stretches of shore, and clay on muddy ones;
//! - beaches are a strip 3 to 20 m wide along the water (the width wanders along the shore
//!   every 90 m or so, and its inner edge frays on the scale of 16 m): low ground within that
//!   distance of the water and below the rise of the coarse field across the strip; about a
//!   quarter of the shores have no beach, and where the shore is so flat that such a strip
//!   would rise less than [`BEACH_MIN_RISE`], the meadow reaches the water behind a fringe of
//!   mud or gravel a few metres wide on muddy and stony stretches;
//! - snow lies above a snow line that wanders by ±25 m, climbs on steep slopes and on crests
//!   and reaches down the gullies in tongues ([`Terrain::convexity`]), and not at all on the
//!   steepest slopes and on the ribs of rock between the gullies;
//! - slopes steeper than about 45° (less on high ground, where soil is thin) are bare rock,
//!   stone or, by region, slate or limestone, with a fringe of scree (gravel) around them; a
//!   band of scree and rock patches lies below the snow line, its depth wandering from 4 to
//!   22 m, deeper down the gullies and shallower on the spurs;
//! - in the driest regions ([`Terrain::dryness`]) a few long dunes of sand with a rim of
//!   gravel lie in the heath, stretched along the wind.

use cp_core::coords::VOXEL_LIMIT;
use cp_core::det;
use cp_core::hash::hash_combine;
use cp_core::math::{DVec2, smoothstep};
use cp_core::noise::Fbm;

use crate::config_hash::ConfigHasher;
use crate::error::WorldgenError;
use crate::field::{BandLimitedNoise, GAIN, LACUNARITY, MAX_OCTAVES, NoiseShape, amplitude_sum};
use crate::palette::TerrainPalette;
use crate::region::{cell_size, centre, first_centre_above, last_centre_below};
use crate::strata::Column;

/// Parameters of [`crate::HillsSource`]. All lengths in metres, heights along `+Y`.
///
/// The defaults give gently rolling grassland around `y = 8` with lakes and coasts at the sea
/// level 0, beaches, a few ridge ranges up to about 80 m with snow caps and bare rock, and
/// mixed forests.
#[derive(Clone, Debug, PartialEq)]
pub struct HillsConfig {
    /// Mean ground height.
    pub base_height: f64,
    /// The rolling hills stay within `base_height ± hill_amplitude` (>= 0).
    pub hill_amplitude: f64,
    /// Wavelength of the largest hills (> 0).
    pub hill_wavelength: f64,
    /// Octaves of the hills at level 0, each of half the wavelength and amplitude of the one
    /// before (<= 32).
    pub hill_octaves: u32,
    /// Height the ridges add on top of the hills at their crests (>= 0; 0 disables ridges).
    pub ridge_height: f64,
    /// Wavelength of the largest ridge features (> 0).
    pub ridge_wavelength: f64,
    /// Octaves of the ridges at level 0 (<= 32).
    pub ridge_octaves: u32,
    /// Wavelength of the noise that decides which regions carry ridges (> 0).
    pub ridge_region_wavelength: f64,
    /// Sea level (voxel y): open cells whose centre lies below it hold water.
    pub sea_level: i32,
    /// How wide the beaches are: 2.5 gives strips that wander between 5 and 20 m along the
    /// shore; every metre more (or less) moves the wander a tenth of its range wider
    /// (narrower). (>= 0)
    pub beach_height: f64,
    /// Sea floors more than this below the sea level are gravel (>= 0).
    pub deep_water_depth: f64,
    /// Mean height of the snow line; it wanders by ±10 m and climbs on steep slopes.
    pub snow_line: f64,
    /// Share of the tree grid cells in the densest forests that hold a tree, in `[0, 1]`; 0
    /// disables trees, bushes and boulders.
    pub tree_density: f64,
}

impl Default for HillsConfig {
    fn default() -> Self {
        Self {
            base_height: 8.0,
            hill_amplitude: 24.0,
            hill_wavelength: 512.0,
            hill_octaves: 7,
            ridge_height: 56.0,
            ridge_wavelength: 1024.0,
            ridge_octaves: 6,
            ridge_region_wavelength: 2048.0,
            sea_level: 0,
            beach_height: 2.5,
            deep_water_depth: 8.0,
            snow_line: 52.0,
            tree_density: 0.9,
        }
    }
}

/// Heights beyond this magnitude (m) are rejected: they would leave the voxel limits long
/// before, and keeping sums small keeps every bound comfortably exact.
const MAX_HEIGHT_MAGNITUDE: f64 = 1.0e6;

impl HillsConfig {
    /// Checks every documented range.
    pub(crate) fn validate(&self) -> Result<(), WorldgenError> {
        let invalid = |message: String| Err(WorldgenError::InvalidConfig(message));
        let values = [
            ("base_height", self.base_height),
            ("hill_amplitude", self.hill_amplitude),
            ("hill_wavelength", self.hill_wavelength),
            ("ridge_height", self.ridge_height),
            ("ridge_wavelength", self.ridge_wavelength),
            ("ridge_region_wavelength", self.ridge_region_wavelength),
            ("beach_height", self.beach_height),
            ("deep_water_depth", self.deep_water_depth),
            ("snow_line", self.snow_line),
            ("tree_density", self.tree_density),
        ];
        for (name, value) in values {
            if !value.is_finite() {
                return invalid(format!("{name} must be finite, got {value}"));
            }
        }
        for (name, value) in [
            ("hill_amplitude", self.hill_amplitude),
            ("ridge_height", self.ridge_height),
            ("beach_height", self.beach_height),
            ("deep_water_depth", self.deep_water_depth),
        ] {
            if value < 0.0 {
                return invalid(format!("{name} must not be negative, got {value}"));
            }
        }
        for (name, value) in [
            ("hill_wavelength", self.hill_wavelength),
            ("ridge_wavelength", self.ridge_wavelength),
            ("ridge_region_wavelength", self.ridge_region_wavelength),
        ] {
            if value <= 0.0 {
                return invalid(format!("{name} must be positive, got {value}"));
            }
        }
        for (name, value) in [
            ("hill_octaves", self.hill_octaves),
            ("ridge_octaves", self.ridge_octaves),
        ] {
            if value > MAX_OCTAVES {
                return invalid(format!("{name} must be at most {MAX_OCTAVES}, got {value}"));
            }
        }
        if !(0.0..=1.0).contains(&self.tree_density) {
            return invalid(format!(
                "tree_density must lie in [0, 1], got {}",
                self.tree_density
            ));
        }
        let extent = self.base_height.abs() + self.hill_amplitude + self.ridge_height;
        if extent > MAX_HEIGHT_MAGNITUDE
            || self.snow_line.abs() > MAX_HEIGHT_MAGNITUDE
            || self.beach_height > MAX_HEIGHT_MAGNITUDE
            || self.deep_water_depth > MAX_HEIGHT_MAGNITUDE
        {
            return invalid(format!(
                "heights must stay within ±{MAX_HEIGHT_MAGNITUDE} m"
            ));
        }
        if self.sea_level.unsigned_abs() >= VOXEL_LIMIT.unsigned_abs() {
            return invalid(format!(
                "sea_level must lie inside the voxel limits, got {}",
                self.sea_level
            ));
        }
        Ok(())
    }

    /// Feeds every parameter, in declaration order, into an identity hash.
    pub(crate) fn hash_into(&self, hasher: &mut ConfigHasher) {
        hasher
            .f64(self.base_height)
            .f64(self.hill_amplitude)
            .f64(self.hill_wavelength)
            .u32(self.hill_octaves)
            .f64(self.ridge_height)
            .f64(self.ridge_wavelength)
            .u32(self.ridge_octaves)
            .f64(self.ridge_region_wavelength)
            .i32(self.sea_level)
            .f64(self.beach_height)
            .f64(self.deep_water_depth)
            .f64(self.snow_line)
            .f64(self.tree_density);
    }
}

/// Region noise below this carries no ridges at all.
const RIDGE_MASK_LOW: f64 = 0.05;
/// Region noise above this carries ridges at full height.
const RIDGE_MASK_HIGH: f64 = 0.35;
/// Octaves of the ridge region noise.
const RIDGE_REGION_OCTAVES: u32 = 2;

// Salts that give each noise channel its own seed.
const SALT_HILLS: u64 = 0x4849_4C4C_5300_0001;
const SALT_RIDGES: u64 = 0x5249_4447_4500_0002;
const SALT_REGIONS: u64 = 0x5245_4749_4F4E_0003;
const SALT_SNOW: u64 = 0x534E_4F57_0000_0006;
const SALT_BEACH: u64 = 0x4245_4143_4800_0007;
const SALT_SHORE: u64 = 0x5348_4F52_4500_0008;
const SALT_PATCHES: u64 = 0x5041_5443_4800_0009;
const SALT_ROCK: u64 = 0x524F_434B_0000_000A;
const SALT_DRY: u64 = 0x4452_5900_0000_000B;
const SALT_DRY_WARP_X: u64 = 0x4452_5958_0000_000C;
const SALT_DRY_WARP_Z: u64 = 0x4452_595A_0000_000D;
const SALT_FINE: u64 = 0x4649_4E45_0000_000E;
const SALT_RIDGE_WARP: u64 = 0x5257_4152_5000_000F;
const SALT_DUNES: u64 = 0x4455_4E45_0000_0010;
const SALT_MICRO: u64 = 0x4D49_4352_4F00_0011;
const SALT_MICRO_REGIONS: u64 = 0x4D52_4547_0000_0012;
const SALT_HILL_WARP: u64 = 0x4857_4152_5000_0013;
const SALT_FINE_WARP_X: u64 = 0x4657_4152_5800_0014;
const SALT_FINE_WARP_Z: u64 = 0x4657_4152_5A00_0015;
const SALT_ALPINE: u64 = 0x414C_5049_4E00_0016;
const SALT_FRAY: u64 = 0x4652_4159_0000_0017;

/// The micro relief: wavelength (m) and octaves of its billow noise (64 and 32 m, both of them
/// up to level 3, none from level 5 on; shorter octaves broke slopes into so many short steps
/// that their ends stood on the meadows like fins) ...
const MICRO_NOISE: (f64, u32) = (64.0, 2);
/// ... its amplitude (m) ...
pub(crate) const MICRO_AMPLITUDE: f64 = 2.2;
/// ... the noise of the regions that carry it in full, wavelength (m) and octaves, the range of
/// that noise (stretched to about `[-1, 1]`) over which it grows to its full height from the
/// share it keeps elsewhere: about half the land is hummocky, the rest gently undulating (even
/// there the one-metre steps of a slope must not run on as long, even rings) ...
const MICRO_REGION_NOISE: (f64, u32) = (700.0, 1);
const MICRO_REGION_RAMP: (f64, f64) = (-0.2, 0.4);
const MICRO_FLOOR: f64 = 0.55;
/// ... and how it fades in above the sea level: half of it over the first metres above the
/// water (the meadows of a shore keep hummocks, so that its slope does not rise in even,
/// parallel terraces), the other half over the heights (m) of the second ramp.
const MICRO_SHORE_RAMP: (f64, f64) = (0.4, 1.6);
const MICRO_LAND_RAMP: (f64, f64) = (1.6, 6.0);
/// Domain warp of the hills: wavelength (m), octaves and how far (m) it moves a sample.
const HILL_WARP: (f64, u32, f64) = (220.0, 1, 50.0);
/// The fine warp of the hills: wavelength (m) and how far (m) it moves a sample at most. It
/// bends the contour lines of a smooth slope on the scale of 10 to 30 m, so that the one-metre
/// steps of a hillside meander and break instead of running as ruled lines for a hundred
/// metres. Band-limited like an octave (levels 0–3 keep it, coarser levels do not see it), and
/// on ground of slope `s` it moves the height by at most `s` times its reach.
const FINE_WARP: (f64, f64) = (40.0, 9.0);

/// Octaves of the hills shorter than this wavelength (m) are the fine channel ...
pub(crate) const FINE_BELOW: f64 = 32.0;
/// ... which keeps this share of its natural amplitude (it adds a little roughness to slopes
/// without breaking flat ground into single cells).
pub(crate) const FINE_WEIGHT: f64 = 0.3;
/// Domain warp of the ridges: wavelength (m), octaves and how far (m) it moves a sample.
const RIDGE_WARP: (f64, u32, f64) = (900.0, 3, 380.0);
/// A beach is a strip this many metres wide along the water, at least and at most (the width
/// wanders along the shore with the beach noise): a narrow rim on most shores, a wide strand on
/// a few, so that the lakes are not all framed by the same band of sand.
const BEACH_WIDTH: (f64, f64) = (3.0, 14.0);
/// The top of a beach lies at most this high above the water (m).
const BEACH_MAX_RISE: f64 = 3.0;
/// Shores flatter than this many metres of rise across the beach width get no sand.
pub(crate) const BEACH_MIN_RISE: f64 = 0.4;
/// The water a beach lies along is looked for this many metres beyond the beach's width.
const NEAR_WATER_MARGIN: f64 = 5.0;
/// The flattest slope a beach is measured with (rise over run).
const BEACH_MIN_SLOPE: f64 = 0.015;
/// Dunes in the driest heath: wavelength (m) and octaves of their noise, the stretch along the
/// wind (the noise is sampled this much faster across it), and where they and their rim of
/// gravel begin (stretched noise), how dry the land must be and how flat.
const DUNE_NOISE: (f64, u32) = (80.0, 2);
const DUNE_STRETCH: f64 = 2.8;
const DUNE_FROM: f64 = 0.9;
const DUNE_RIM_FROM: f64 = 0.82;
const DUNE_DRYNESS: f64 = 0.6;
const DUNE_MAX_SLOPE: f64 = 0.15;

/// Wavelength (m) and octaves of the noise that moves the snow line.
const SNOW_NOISE: (f64, u32) = (220.0, 2);
/// The snow line wanders by this much (m) either way.
const SNOW_JITTER: f64 = 25.0;
/// Slope (rise over run) from which snow needs more height to stay, and how much more per unit
/// of slope (m).
const SNOW_SLOPE: (f64, f64) = (0.7, 12.0);
/// No snow stays on slopes steeper than this.
const SNOW_MAX_SLOPE: f64 = 0.8;
/// The relief of the mountains measures how convex a place is ([`Terrain::convexity`]): the
/// ridges of the level against the ridges of this level, which keeps only the features of 256 m
/// and more. Positive on crests and ribs, negative in gullies and hollows.
const CONVEX_LEVEL: u8 = 6;
/// The snow line climbs by this many metres per metre of convexity (and sinks in hollows), up
/// to [`CONVEX_MAX`] metres of convexity: wind strips the crests and the snow gathers in the
/// gullies, where it reaches down in tongues.
const CONVEX_GAIN: f64 = 1.4;
const CONVEX_MAX: f64 = 18.0;
/// Ribs of rock run through the snow where the relief is more convex than this (m): the crests
/// and spurs of the finer octaves of the ridges, between which the snow lies in gullies.
const RIB_FROM: f64 = 2.5;
/// The band of scree and rock patches below the snow line: wavelength (m) and octaves of the
/// noise that sets its depth, the depths (m) it wanders between, how much deeper it reaches
/// down the gullies and how much higher the grass climbs on the spurs (m, at full convexity).
const ALPINE_NOISE: (f64, u32) = (128.0, 2);
const ALPINE_DEPTH: (f64, f64) = (4.0, 22.0);
const SCREE_TONGUE: f64 = 13.0;
const GRASS_TONGUE: f64 = 8.0;
/// Wavelength (m) and octaves of the noise that moves the width of the beaches.
const BEACH_NOISE: (f64, u32) = (90.0, 2);
/// Below this (stretched) beach noise a shore has no beach at all (about a quarter of them).
const NO_BEACH: f64 = -0.45;
/// Wavelength (m) and octaves of the noise that frays the inner edge of a beach, and how far
/// (m) it moves that edge either way.
const FRAY_NOISE: (f64, u32) = (16.0, 2);
const FRAY_REACH: f64 = 3.0;
/// The fringe of a flat shore without a beach: the columns level with the water within this
/// many metres of it are mud or gravel, by the stretch of shore.
const FRINGE_REACH: f64 = 2.5;
/// Wavelength (m) and octaves of the noise that makes stretches of shore stony or muddy.
const SHORE_NOISE: (f64, u32) = (384.0, 2);
/// Wavelength (m) and octaves of the patches: ragged edges, sand on dry meadows, scree.
const PATCH_NOISE: (f64, u32) = (32.0, 2);
/// Wavelength (m) and octaves of the noise that picks the kind of rock.
const ROCK_NOISE: (f64, u32) = (700.0, 1);
/// Wavelength (m) of the regions (the dryness field), and the size (m) and wavelength of its
/// warp, which keeps the regions from looking like a blurred grid.
const DRY_NOISE: (f64, u32) = (2600.0, 2);
const DRY_WARP: (f64, f64) = (1000.0, 400.0);
/// Rock shows on slopes steeper than this at low ground ...
const ROCK_SLOPE_LOW: f64 = 1.05;
/// ... and on slopes steeper than this from 60 m up, where the soil is thin.
const ROCK_SLOPE_HIGH: f64 = 0.7;
/// Scree lies on slopes this much less steep than rock, above this height (m).
const SCREE_MARGIN: (f64, f64) = (0.3, 12.0);
/// Beaches steeper than this are gravel.
const STEEP_SHORE: f64 = 0.45;
/// Stretches of shore whose (stretched) shore noise lies above this are stony ...
const STONY_SHORE: f64 = 0.6;
/// ... and below minus this muddy.
const MUDDY_SHORE: f64 = 0.75;

// Cumulative layer depths below the surface (m).
const GRASS_DEPTH: f64 = 1.0;
const DIRT_DEPTH: f64 = 4.0;
const SAND_DEPTH: f64 = 4.0;
const GRAVEL_DEPTH: f64 = 2.0;
const SNOW_DEPTH: f64 = 1.0;
const CLAY_DEPTH: f64 = 3.0;
const ROCK_DEPTH: f64 = 4.0;
/// The deepest of the layer depths above: no layer of any surface kind reaches further below
/// the surface (checked by a test).
const MAX_LAYER_DEPTH: f64 = 4.0;

/// What covers a column, decided by its height, slope and the noise fields of
/// [`Terrain::surface`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Surface {
    /// 1 m grass over 3 m dirt over stone.
    Grass,
    /// 4 m sand over stone: beaches, shallow lake floors and sandy patches.
    Sand,
    /// 2 m gravel over stone: deep floors, stony shores, scree.
    Gravel,
    /// 1 m snow over stone.
    Snow,
    /// 3 m clay over stone: muddy shores and floors.
    Clay,
    /// Bare stone.
    Stone,
    /// 4 m slate over stone: bare rock of the slate regions.
    Slate,
    /// 4 m limestone over stone: bare rock of the limestone regions.
    Limestone,
    /// Water in the top cell over 4 m sand: ground that the point rule puts level with the
    /// water's surface although it lies below the sea level (the shallows along a shore, on
    /// coarse levels a band of the lake), so that it shows as water, as on finer levels.
    Shallow,
}

impl Surface {
    /// Every surface kind.
    #[cfg(test)]
    pub(crate) const ALL: [Self; 9] = [
        Self::Grass,
        Self::Sand,
        Self::Gravel,
        Self::Snow,
        Self::Clay,
        Self::Stone,
        Self::Slate,
        Self::Limestone,
        Self::Shallow,
    ];
}

/// The height field and the column rules of one hills landscape.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Terrain {
    base: f64,
    hill_amplitude: f64,
    ridge_height: f64,
    /// The octaves of the hills from the hill wavelength down to [`FINE_BELOW`] m, and their
    /// share of the whole sum's amplitude.
    hills: BandLimitedNoise,
    hills_weight: f64,
    /// The octaves below [`FINE_BELOW`] m, and their share times [`FINE_WEIGHT`].
    fine: BandLimitedNoise,
    fine_weight: f64,
    hill_warp_seed: u64,
    /// The two channels of the fine warp of the hills ([`FINE_WARP`]).
    fine_warp: [BandLimitedNoise; 2],
    /// The micro relief and the regions that carry it.
    micro: BandLimitedNoise,
    micro_regions: BandLimitedNoise,
    ridges: BandLimitedNoise,
    ridge_warp_seed: u64,
    regions: BandLimitedNoise,
    dunes: BandLimitedNoise,
    snow_noise: BandLimitedNoise,
    alpine_noise: BandLimitedNoise,
    beach_noise: BandLimitedNoise,
    fray_noise: BandLimitedNoise,
    shore_noise: BandLimitedNoise,
    patches: BandLimitedNoise,
    rock_noise: BandLimitedNoise,
    dry_noise: BandLimitedNoise,
    dry_warp: [BandLimitedNoise; 2],
    sea_level: f64,
    gravel_below: f64,
    /// Added to the beach noise: the configured beach height's offset from the default.
    beach_shift: f64,
    snow_from: f64,
    /// Added to the analytic height bounds to cover the rounding of the sum.
    bound_margin: f64,
}

impl Terrain {
    /// The terrain of `config` (already validated) for the world seed `seed`.
    pub(crate) fn new(seed: u64, config: &HillsConfig) -> Self {
        let sea_level = f64::from(config.sea_level);
        let channel = |salt: u64, (wavelength, octaves): (f64, u32)| {
            BandLimitedNoise::new(
                hash_combine(seed, salt),
                wavelength,
                octaves,
                NoiseShape::Plain,
            )
        };
        // The octaves of the hills down to FINE_BELOW m, and the finer ones.
        let mut coarse = 0;
        let mut wavelength = config.hill_wavelength;
        while coarse < config.hill_octaves && wavelength >= FINE_BELOW {
            coarse += 1;
            wavelength *= 1.0 / LACUNARITY;
        }
        let fine = config.hill_octaves - coarse;
        let total = amplitude_sum(config.hill_octaves);
        let shares = if total > 0.0 {
            let mut lead = 1.0;
            for _ in 0..coarse {
                lead *= GAIN;
            }
            (
                amplitude_sum(coarse) / total,
                FINE_WEIGHT * lead * amplitude_sum(fine) / total,
            )
        } else {
            (0.0, 0.0)
        };
        Self {
            base: config.base_height,
            hill_amplitude: config.hill_amplitude,
            ridge_height: config.ridge_height,
            hills: channel(SALT_HILLS, (config.hill_wavelength, coarse)),
            hills_weight: shares.0,
            fine: channel(SALT_FINE, (wavelength, fine)),
            fine_weight: shares.1,
            hill_warp_seed: hash_combine(seed, SALT_HILL_WARP),
            fine_warp: [
                channel(SALT_FINE_WARP_X, (FINE_WARP.0, 1)),
                channel(SALT_FINE_WARP_Z, (FINE_WARP.0, 1)),
            ],
            micro: BandLimitedNoise::new(
                hash_combine(seed, SALT_MICRO),
                MICRO_NOISE.0,
                MICRO_NOISE.1,
                NoiseShape::Billow,
            ),
            micro_regions: channel(SALT_MICRO_REGIONS, MICRO_REGION_NOISE),
            ridges: BandLimitedNoise::new(
                hash_combine(seed, SALT_RIDGES),
                config.ridge_wavelength,
                config.ridge_octaves,
                NoiseShape::Ridged,
            ),
            ridge_warp_seed: hash_combine(seed, SALT_RIDGE_WARP),
            regions: channel(
                SALT_REGIONS,
                (config.ridge_region_wavelength, RIDGE_REGION_OCTAVES),
            ),
            dunes: channel(SALT_DUNES, DUNE_NOISE),
            snow_noise: channel(SALT_SNOW, SNOW_NOISE),
            alpine_noise: channel(SALT_ALPINE, ALPINE_NOISE),
            beach_noise: channel(SALT_BEACH, BEACH_NOISE),
            fray_noise: channel(SALT_FRAY, FRAY_NOISE),
            shore_noise: channel(SALT_SHORE, SHORE_NOISE),
            patches: channel(SALT_PATCHES, PATCH_NOISE),
            rock_noise: channel(SALT_ROCK, ROCK_NOISE),
            dry_noise: channel(SALT_DRY, DRY_NOISE),
            dry_warp: [
                channel(SALT_DRY_WARP_X, (DRY_WARP.0, 1)),
                channel(SALT_DRY_WARP_Z, (DRY_WARP.0, 1)),
            ],
            sea_level,
            gravel_below: sea_level - config.deep_water_depth,
            beach_shift: 0.1 * (config.beach_height - HillsConfig::default().beach_height),
            snow_from: config.snow_line,
            bound_margin: 1.0e-9
                * (1.0
                    + config.base_height.abs()
                    + config.hill_amplitude
                    + config.ridge_height
                    + MICRO_AMPLITUDE),
        }
    }

    /// Surface height (m) at `(x, z)` m as level `level` sees it (band-limited), before the
    /// columns are settled ([`Terrain::settle`]).
    #[inline]
    pub(crate) fn height(&self, level: u8, x: f64, z: f64) -> f64 {
        let smooth = self.hill_height(level, x, z) + self.ridge_rise(level, x, z);
        smooth + self.micro_relief(level, x, z, smooth)
    }

    /// The point at which the hill channels are sampled for `(x, z)`: bent by the hill warp
    /// (a coordinate transform, the same on every level; it adds no detail of its own).
    #[inline]
    fn hill_point(&self, x: f64, z: f64) -> DVec2 {
        let (wavelength, octaves, amplitude) = HILL_WARP;
        Fbm::new(octaves, LACUNARITY, GAIN).domain_warp2(
            self.hill_warp_seed,
            DVec2::new(x, z),
            1.0 / wavelength,
            amplitude,
        )
    }

    /// [`Self::hill_point`] bent further by the fine warp as `level` sees it ([`FINE_WARP`]):
    /// band-limited like an octave of its wavelength, so the coarse levels, whose cells could
    /// not show its bends, do not see it.
    #[inline]
    fn fine_point(&self, level: u8, x: f64, z: f64) -> DVec2 {
        let p = self.hill_point(x, z);
        if self.fine_warp[0].share(level) <= 0.0 {
            return p;
        }
        p + DVec2::new(
            self.fine_warp[0].sample(level, x, z),
            self.fine_warp[1].sample(level, x, z),
        ) * FINE_WARP.1
    }

    /// The rolling hills alone (m): the base height and both hill channels.
    #[inline]
    fn hill_height(&self, level: u8, x: f64, z: f64) -> f64 {
        let p = self.fine_point(level, x, z);
        let hills = self.hills_weight * self.hills.sample(level, p.x, p.y)
            + self.fine_weight * self.fine.sample(level, p.x, p.y);
        self.base + self.hill_amplitude * hills
    }

    /// What the micro relief adds at `(x, z)` (m) on ground whose height without it is
    /// `smooth`: zero from level 4 on, outside its regions and at the water's edge, half of it
    /// on the low ground of a shore.
    #[inline]
    fn micro_relief(&self, level: u8, x: f64, z: f64, smooth: f64) -> f64 {
        let above = smooth - self.sea_level;
        let shore = 0.5 * smoothstep(MICRO_SHORE_RAMP.0, MICRO_SHORE_RAMP.1, above)
            + 0.5 * smoothstep(MICRO_LAND_RAMP.0, MICRO_LAND_RAMP.1, above);
        if shore <= 0.0 || self.micro.share(level) <= 0.0 {
            return 0.0;
        }
        let region = MICRO_FLOOR
            + (1.0 - MICRO_FLOOR)
                * smoothstep(
                    MICRO_REGION_RAMP.0,
                    MICRO_REGION_RAMP.1,
                    1.6 * self.micro_regions.sample(level, x, z),
                );
        MICRO_AMPLITUDE * region * shore * self.micro.sample(level, x, z)
    }

    /// What the ridges add at `(x, z)` (m); zero outside the ridge regions.
    #[inline]
    fn ridge_rise(&self, level: u8, x: f64, z: f64) -> f64 {
        self.ridge_frame(level, x, z)
            .map_or(0.0, |(mask, bent)| self.ridge_of(level, mask, bent))
    }

    /// The ridge region mask at `(x, z)` on `level` and the bent point at which the ridges are
    /// sampled; `None` outside the ridge regions, where the ridge noise is not even evaluated.
    #[inline]
    fn ridge_frame(&self, level: u8, x: f64, z: f64) -> Option<(f64, DVec2)> {
        if self.ridge_height <= 0.0 {
            return None;
        }
        let mask = smoothstep(
            RIDGE_MASK_LOW,
            RIDGE_MASK_HIGH,
            self.regions.sample(level, x, z),
        );
        if mask <= 0.0 {
            return None;
        }
        // The warp is a coordinate transform, the same on every level (it adds no detail of
        // its own: the ridges sampled at the bent coordinates keep their band limit).
        let (wavelength, octaves, amplitude) = RIDGE_WARP;
        let bent = Fbm::new(octaves, LACUNARITY, GAIN).domain_warp2(
            self.ridge_warp_seed,
            DVec2::new(x, z),
            1.0 / wavelength,
            amplitude,
        );
        Some((mask, bent))
    }

    /// The rise of the ridges on `level` with the region mask `mask` at the bent point `bent`.
    #[inline]
    fn ridge_of(&self, level: u8, mask: f64, bent: DVec2) -> f64 {
        let ridge = 0.5 * (self.ridges.sample(level, bent.x, bent.y) + 1.0);
        self.ridge_height * mask * (ridge * ridge)
    }

    /// How convex the mountains are at `(x, z)` as `level` sees them (m): the rise of the
    /// ridges on `level` less their rise on [`CONVEX_LEVEL`], which keeps only their features
    /// of 256 m and more. Positive on crests and spurs, negative in gullies and hollows, zero
    /// outside the ridge regions and from [`CONVEX_LEVEL`] on.
    pub(crate) fn convexity(&self, level: u8, x: f64, z: f64) -> f64 {
        if level >= CONVEX_LEVEL {
            return 0.0;
        }
        self.ridge_frame(level, x, z).map_or(0.0, |(mask, bent)| {
            self.ridge_of(level, mask, bent) - self.ridge_of(CONVEX_LEVEL, mask, bent)
        })
    }

    /// The bound of `|hills|` on `level`: the kept shares of both channels.
    fn hill_share(&self, level: u8) -> f64 {
        self.hills_weight * self.hills.share(level) + self.fine_weight * self.fine.share(level)
    }

    /// The bound of `|micro relief|` on `level` (m).
    fn micro_bound(&self, level: u8) -> f64 {
        MICRO_AMPLITUDE * self.micro.share(level)
    }

    /// A bound below every height of level `level` (m), settled or not.
    pub(crate) fn min_height(&self, level: u8) -> f64 {
        self.base
            - self.hill_amplitude * self.hill_share(level)
            - self.micro_bound(level)
            - self.bound_margin
    }

    /// A bound above every height of level `level` (m), settled or not.
    pub(crate) fn max_height(&self, level: u8) -> f64 {
        let ridge = 0.5 * (1.0 + self.ridges.share(level));
        self.base
            + self.hill_amplitude * self.hill_share(level)
            + self.ridge_height * ridge * ridge
            + self.micro_bound(level)
            + self.bound_margin
    }

    /// One pass of the settling filter on a column of `level`: `window` holds the heights of
    /// the 3 × 3 columns around it (x fastest, the column itself in the middle). The column
    /// takes the median of the nine heights when that ends in another top cell than its own
    /// (a median of heights is the height of the median top cell: the top cell grows with the
    /// height), and keeps its own height otherwise. A median of 3 × 3 keeps planes, slopes and
    /// straight steps of any height, and removes everything less than two cells across: single
    /// pits and humps, notches, and plateaus of two by two cells. The result is one of the
    /// window's heights, so it never leaves their range.
    pub(crate) fn settle_once(level: u8, window: [f64; 9]) -> f64 {
        let own = window[4];
        let mut sorted = window;
        sorted.sort_by(f64::total_cmp);
        let median = sorted[4];
        if last_centre_below(level, median) == last_centre_below(level, own) {
            own
        } else {
            median
        }
    }

    /// Two passes of [`Terrain::settle_once`] over a grid of `width` × `depth` column heights
    /// (x fastest): the settled heights of the inner `(width - 4)` × `(depth - 4)` columns,
    /// x fastest. Every column's result depends only on the heights within two columns of it
    /// (the 5 × 5 square around it) and lies within their range.
    ///
    /// # Panics
    /// If the grid is smaller than 5 × 5 or `heights` does not hold `width * depth` values.
    pub(crate) fn settle(level: u8, heights: &[f64], width: usize, depth: usize) -> Vec<f64> {
        assert!(width >= 5 && depth >= 5 && heights.len() == width * depth);
        let pass = |input: &[f64], w: usize, d: usize| -> Vec<f64> {
            let mut out = Vec::with_capacity((w - 2) * (d - 2));
            for j in 1..d - 1 {
                for i in 1..w - 1 {
                    let window =
                        core::array::from_fn(|k| input[(j + k / 3 - 1) * w + i + k % 3 - 1]);
                    out.push(Self::settle_once(level, window));
                }
            }
            out
        };
        let first = pass(heights, width, depth);
        pass(&first, width - 2, depth - 2)
    }

    /// The settled height (m) of the column `(cx, cz)` of `level` (cell coordinates): the
    /// evaluation of [`Terrain::settle`] for one column, from the 25 heights within two
    /// columns of it.
    ///
    /// Where the 3 × 3 columns around it share one top cell the column keeps its own height
    /// (a first pass leaves the middle column and the four beside it as they are, and changes
    /// at most the four corners, too few to move a median of nine), so the outer ring is only
    /// evaluated near a step.
    pub(crate) fn settled_height(&self, level: u8, cx: i64, cz: i64) -> f64 {
        let raw =
            |dx: i64, dz: i64| self.height(level, centre(level, cx + dx), centre(level, cz + dz));
        let inner: [f64; 9] = core::array::from_fn(|k| raw((k % 3) as i64 - 1, (k / 3) as i64 - 1));
        let top = last_centre_below(level, inner[4]);
        if inner
            .iter()
            .all(|&height| last_centre_below(level, height) == top)
        {
            return inner[4];
        }
        let grid: [f64; 25] = core::array::from_fn(|k| {
            let (dx, dz) = ((k % 5) as i64 - 2, (k / 5) as i64 - 2);
            if dx.abs() <= 1 && dz.abs() <= 1 {
                inner[((dz + 1) * 3 + dx + 1) as usize]
            } else {
                raw(dx, dz)
            }
        });
        let first: [f64; 9] = core::array::from_fn(|k| {
            let (i, j) = (k % 3 + 1, k / 3 + 1);
            Self::settle_once(
                level,
                core::array::from_fn(|w| grid[(j + w / 3 - 1) * 5 + i + w % 3 - 1]),
            )
        });
        Self::settle_once(level, first)
    }

    /// A bound above every height of every level (m).
    pub(crate) fn max_height_all_levels(&self) -> f64 {
        // Level 0 keeps every octave, and the bounds only shrink with fewer octaves.
        self.max_height(0)
    }

    /// The highest cell row of `level` that holds the base state (stone) in every column,
    /// whatever the column's height: the cell above it is ground as well, so it is not the top
    /// ground cell, and it lies at least [`MAX_LAYER_DEPTH`] below every possible surface, so no
    /// layer reaches it. Rows at or below it need no column evaluation.
    pub(crate) fn base_top(&self, level: u8) -> i64 {
        let low = self.min_height(level);
        // centre(c + 1) < low <= every height: the cell above c is ground in every column.
        let under_top_cell = last_centre_below(level, low) - 1;
        // centre(c) <= low - MAX_LAYER_DEPTH <= height - depth for every layer depth, with the
        // subtraction rounded as `Column::new` rounds it (rounding is monotonic).
        let under_layers = first_centre_above(level, low - MAX_LAYER_DEPTH) - 1;
        under_top_cell.min(under_layers)
    }

    /// Sea level (m).
    pub(crate) fn sea_level(&self) -> f64 {
        self.sea_level
    }

    /// Mean height of the snow line (m).
    pub(crate) fn snow_line(&self) -> f64 {
        self.snow_from
    }

    /// The dryness of the region around `(x, z)` m as `level` sees it, in `[-1, 1]`: positive
    /// in the dry heaths (sparse trees, mostly birch, sand in the meadows), negative in the
    /// lush lowlands (dense mixed woods, cherry by the water). Regions are about 1.3 km
    /// across, with bent borders.
    pub(crate) fn dryness(&self, level: u8, x: f64, z: f64) -> f64 {
        let amplitude = DRY_WARP.1;
        let wx = x + amplitude * self.dry_warp[0].sample(level, x, z);
        let wz = z + amplitude * self.dry_warp[1].sample(level, x, z);
        // Fbm rarely leaves ±0.6: stretch it to use the range.
        det::clamp(1.6 * self.dry_noise.sample(level, wx, wz), -1.0, 1.0)
    }

    /// The slope (rise over run) of the field of `level` at a column whose neighbours along x
    /// and z, `cell` metres away on each side, have the heights `[x-, x+, z-, z+]`.
    pub(crate) fn slope(neighbours: [f64; 4], cell: f64) -> f64 {
        let gx = (neighbours[1] - neighbours[0]) / (2.0 * cell);
        let gz = (neighbours[3] - neighbours[2]) / (2.0 * cell);
        (gx * gx + gz * gz).sqrt()
    }

    /// The slope of level `level` at the column centre `(x, z)` m, from the heights of the
    /// four neighbouring column centres of the level.
    pub(crate) fn slope_at(&self, level: u8, x: f64, z: f64) -> f64 {
        let cell = f64::from(1_u32 << level);
        Self::slope(
            [
                self.height(level, x - cell, z),
                self.height(level, x + cell, z),
                self.height(level, x, z - cell),
                self.height(level, x, z + cell),
            ],
            cell,
        )
    }

    /// The slope of the coarse field at `(x, z)` on `level`: the hills without their fine
    /// channel, and the ridges, from points `max(cell, 2 m)` to each side. Beaches are measured
    /// with it, so that their edge follows the shore instead of every ripple.
    fn shore_slope(&self, level: u8, x: f64, z: f64) -> f64 {
        let step = det::max(cell_size(level), 2.0);
        Self::slope(
            [
                self.coarse_height(level, x - step, z),
                self.coarse_height(level, x + step, z),
                self.coarse_height(level, x, z - step),
                self.coarse_height(level, x, z + step),
            ],
            step,
        )
    }

    /// The coarse field at `(x, z)` on `level` (m): the hills without their fine channel (and
    /// without the micro relief), and the ridges.
    fn coarse_height(&self, level: u8, x: f64, z: f64) -> f64 {
        let p = self.hill_point(x, z);
        self.base
            + self.hill_amplitude * self.hills_weight * self.hills.sample(level, p.x, p.y)
            + self.ridge_rise(level, x, z)
    }

    /// Whether the coarse field lies under the water somewhere within `reach` metres of
    /// `(x, z)` on `level`: at half and the full reach in eight directions. A beach is a
    /// strip along the water; low, flat land farther from it is meadow, however low it lies.
    fn near_water(&self, level: u8, x: f64, z: f64, reach: f64) -> bool {
        const DIRECTIONS: [(f64, f64); 8] = [
            (1.0, 0.0),
            (0.707, 0.707),
            (0.0, 1.0),
            (-0.707, 0.707),
            (-1.0, 0.0),
            (-0.707, -0.707),
            (0.0, -1.0),
            (0.707, -0.707),
        ];
        [0.5 * reach, reach].iter().any(|&distance| {
            DIRECTIONS.iter().any(|&(dx, dz)| {
                self.coarse_height(level, x + dx * distance, z + dz * distance) < self.sea_level
            })
        })
    }

    /// Whether the field of `level` itself (not the coarse one of [`Self::near_water`]) lies
    /// under the water within `reach` metres of `(x, z)`: at half and the full reach in eight
    /// directions. The fringe of a shore is a few metres wide, less than the coarse field's
    /// shoreline differs from the real one.
    fn water_within(&self, level: u8, x: f64, z: f64, reach: f64) -> bool {
        const DIRECTIONS: [(f64, f64); 8] = [
            (1.0, 0.0),
            (0.707, 0.707),
            (0.0, 1.0),
            (-0.707, 0.707),
            (-1.0, 0.0),
            (-0.707, -0.707),
            (0.0, -1.0),
            (0.707, -0.707),
        ];
        [0.5 * reach, reach].iter().any(|&distance| {
            DIRECTIONS.iter().any(|&(dx, dz)| {
                self.height(level, x + dx * distance, z + dz * distance) < self.sea_level
            })
        })
    }

    /// A bound above the height of every column whose surface is grass (m): grass only grows
    /// below the local snow line ([`Self::snow_line_at`]), and that line, even where it wanders
    /// and climbs on slopes and crests, stays below this (steeper ground is rock).
    pub(crate) fn grass_top(&self) -> f64 {
        det::min(
            self.snow_from
                + SNOW_JITTER
                + SNOW_SLOPE.1 * (ROCK_SLOPE_LOW - SNOW_SLOPE.0)
                + CONVEX_GAIN * CONVEX_MAX
                + 1.0,
            self.max_height(0),
        )
    }

    /// The snow line (m) at a column of `level` at `(x, z)` with the slope `slope` and the
    /// convexity `convex` ([`Self::convexity`]): the mean line, moved by the snow noise (±
    /// [`SNOW_JITTER`]), higher on steep slopes, higher on crests and lower in gullies.
    fn snow_line_at(&self, level: u8, x: f64, z: f64, slope: f64, convex: f64) -> f64 {
        // In [-1, 1], so the line wanders by at most SNOW_JITTER ([`Self::grass_top`]).
        let snow = det::clamp(1.6 * self.snow_noise.sample(level, x, z), -1.0, 1.0);
        self.snow_from - SNOW_JITTER * snow
            + SNOW_SLOPE.1 * det::max(slope - SNOW_SLOPE.0, 0.0)
            + CONVEX_GAIN * det::clamp(convex, -CONVEX_MAX, CONVEX_MAX)
    }

    /// The kind of bare rock at `(x, z)`: stone, or slate or limestone in their regions.
    pub(crate) fn rock_kind(&self, level: u8, x: f64, z: f64) -> Surface {
        self.rock(level, x, z)
    }

    /// The kind of bare rock at `(x, z)`: stone, or slate or limestone in their regions.
    fn rock(&self, level: u8, x: f64, z: f64) -> Surface {
        let kind = 1.6 * self.rock_noise.sample(level, x, z);
        if kind > 0.35 {
            Surface::Limestone
        } else if kind < -0.35 {
            Surface::Slate
        } else {
            Surface::Stone
        }
    }

    /// The surface kind of the column of `level` at `(x, z)` m whose surface lies at `height`
    /// m and whose slope is `slope` (see the module documentation).
    pub(crate) fn surface(&self, level: u8, x: f64, z: f64, height: f64, slope: f64) -> Surface {
        let sea = self.sea_level;
        // A lake floor lies under water: the cell above its top ground cell holds water. A
        // column whose top cell is level with the water although its surface lies below it
        // (by less than half a cell) is the shallows: its top cell holds water. Counted as
        // floor, such columns made sand flats of a hundred metres on flat shores and sand
        // rings around the lakes of the coarse levels.
        let submerged = centre(level, last_centre_below(level, height) + 1) < sea;
        if !submerged && height < sea {
            return Surface::Shallow;
        }
        if submerged {
            if height < self.gravel_below {
                return Surface::Gravel;
            }
            let shore = 1.6 * self.shore_noise.sample(level, x, z);
            return if shore < -MUDDY_SHORE && height > sea - 2.0 {
                Surface::Clay
            } else if shore > STONY_SHORE || slope > 0.8 {
                Surface::Gravel
            } else {
                Surface::Sand
            };
        }
        let patch = 1.6 * self.patches.sample(level, x, z);
        if height < sea + BEACH_MAX_RISE + 0.5 {
            if let Some(surface) = self.shore_surface(level, x, z, height, slope, patch) {
                return surface;
            }
        }
        // Only the mountains reach the snow and the band of scree below it: the convexity,
        // which samples the ridges twice, is needed there alone.
        let lowest_band =
            self.snow_from - SNOW_JITTER - CONVEX_GAIN * CONVEX_MAX - ALPINE_DEPTH.1 - SCREE_TONGUE;
        let convex = if height >= lowest_band {
            self.convexity(level, x, z)
        } else {
            0.0
        };
        let line = self.snow_line_at(level, x, z, slope, convex);
        if height >= line {
            // Above the snow line: snow, except on the steepest slopes and on the ribs and
            // crests of rock between the gullies, which the wind keeps bare.
            return if slope < SNOW_MAX_SLOPE && convex < RIB_FROM {
                Surface::Snow
            } else {
                self.rock(level, x, z)
            };
        }
        let rock_slope =
            ROCK_SLOPE_LOW - (ROCK_SLOPE_LOW - ROCK_SLOPE_HIGH) * smoothstep(20.0, 60.0, height);
        if slope > rock_slope {
            return self.rock(level, x, z);
        }
        if slope > rock_slope - SCREE_MARGIN.0 && height > SCREE_MARGIN.1 {
            return Surface::Gravel;
        }
        // The band of scree and rock below the snow line: its depth wanders, it reaches down
        // the gullies in tongues of scree, and the grass climbs higher on the spurs.
        let gully = smoothstep(2.0, 10.0, -convex);
        let spur = smoothstep(2.0, 10.0, convex);
        let depth = ALPINE_DEPTH.0
            + (ALPINE_DEPTH.1 - ALPINE_DEPTH.0)
                * det::clamp(0.5 + 0.8 * self.alpine_noise.sample(level, x, z), 0.0, 1.0)
            + SCREE_TONGUE * gully
            - GRASS_TONGUE * spur;
        if height > line - depth {
            if patch > 0.3 - 0.6 * gully {
                return Surface::Gravel;
            }
            if patch < -0.5 {
                return self.rock(level, x, z);
            }
        }
        if slope < DUNE_MAX_SLOPE {
            let dry = self.dryness(level, x, z);
            if dry > DUNE_DRYNESS {
                // Dunes stretched along the wind (the diagonal): the noise runs DUNE_STRETCH
                // times faster across it than along it.
                let along = 0.5 * (x + z);
                let across = 0.5 * (x - z) * DUNE_STRETCH;
                let dune = 1.6 * self.dunes.sample(level, along, across);
                if dune > DUNE_FROM {
                    return Surface::Sand;
                }
                if dune > DUNE_RIM_FROM {
                    return Surface::Gravel;
                }
            }
        }
        Surface::Grass
    }

    /// The surface of a column of `level` at `(x, z)` on the low ground of a shore (at most
    /// [`BEACH_MAX_RISE`] above the water), or `None` where the shore leaves it to the rest of
    /// the rules (the meadow):
    ///
    /// - a beach is a strip `width` m wide along the water: the height a column of it may
    ///   reach is the rise of the coarse ground across the strip, rounded to whole cells. The
    ///   configured beach height shifts the wandering width, and about a quarter of the shores
    ///   have no beach at all. A column is measured by the top face of its top cell, so that
    ///   all the columns of one step of the shore belong to the beach or none of them does (by
    ///   their exact heights the columns of a step alternated between sand and grass where the
    ///   field hovered around the beach's top), and the water is looked for a little beyond the
    ///   strip (NEAR_WATER_MARGIN m), for the same reason: a row of columns at the reach's
    ///   distance flickered in and out. The fray noise moves the inner edge of the strip and
    ///   the height of its top, so that it does not run parallel to the water;
    /// - a shore without a beach: the columns level with the water within a few metres of it
    ///   are a fringe of mud on muddy stretches and of gravel on stony ones, so that a flat
    ///   meadow does not run into the lake like a flooded lawn; the first step above the water
    ///   of a stony stretch is gravel.
    fn shore_surface(
        &self,
        level: u8,
        x: f64,
        z: f64,
        height: f64,
        slope: f64,
        patch: f64,
    ) -> Option<Surface> {
        let sea = self.sea_level;
        let cell = cell_size(level);
        let face = centre(level, last_centre_below(level, height)) + 0.5 * cell;
        let wander = 1.6 * self.beach_noise.sample(level, x, z) + self.beach_shift;
        let shore = 1.6 * self.shore_noise.sample(level, x, z);
        let fray = det::clamp(1.6 * self.fray_noise.sample(level, x, z), -1.0, 1.0);
        let width = BEACH_WIDTH.0 + (BEACH_WIDTH.1 - BEACH_WIDTH.0) * smoothstep(-0.2, 0.8, wander);
        let rise = width * det::max(self.shore_slope(level, x, z), BEACH_MIN_SLOPE);
        if wander > NO_BEACH && rise >= BEACH_MIN_RISE {
            let top = det::min(BEACH_MAX_RISE, rise) + 0.15 * patch + 0.35 * fray;
            let shelf = (top / cell + 0.5).floor() * cell;
            let reach = width + 2.0 * patch + FRAY_REACH * fray + NEAR_WATER_MARGIN;
            if face <= sea + shelf && self.near_water(level, x, z, reach) {
                // Mud only on the first step above the water.
                let muddy = shore < -MUDDY_SHORE && face <= sea + cell && slope < 0.3;
                return Some(if slope > STEEP_SHORE || shore > STONY_SHORE {
                    Surface::Gravel
                } else if muddy {
                    Surface::Clay
                } else {
                    Surface::Sand
                });
            }
        }
        let level_with_water = face <= sea + 0.25 * cell;
        let first_step = face <= sea + cell;
        if !first_step {
            return None;
        }
        let fringe = level_with_water
            && (shore < -0.25 || shore > 0.45)
            && self.water_within(level, x, z, det::max(FRINGE_REACH + 1.5 * fray, 1.0));
        if fringe {
            return Some(if shore < -0.25 {
                Surface::Clay
            } else {
                Surface::Gravel
            });
        }
        (shore > STONY_SHORE && !level_with_water).then_some(Surface::Gravel)
    }

    /// The column of `level` at `(x, z)` m with its surface at `height` m and the slope
    /// `slope`.
    pub(crate) fn column(
        &self,
        level: u8,
        (x, z): (f64, f64),
        height: f64,
        slope: f64,
        palette: &TerrainPalette,
    ) -> Column {
        let layers = self.layers(self.surface(level, x, z, height, slope), palette);
        Column::new(
            level,
            height,
            layers,
            palette.stone,
            (palette.water, self.sea_level),
        )
    }

    /// The two `(state, cumulative depth)` layers of a surface kind.
    pub(crate) fn layers(
        &self,
        surface: Surface,
        palette: &TerrainPalette,
    ) -> [(cp_core::ids::BlockStateId, f64); 2] {
        let single = |state, depth| [(state, depth), (state, depth)];
        match surface {
            Surface::Grass => [(palette.grass, GRASS_DEPTH), (palette.dirt, DIRT_DEPTH)],
            Surface::Sand => single(palette.sand, SAND_DEPTH),
            Surface::Gravel => single(palette.gravel, GRAVEL_DEPTH),
            Surface::Snow => [(palette.snow, SNOW_DEPTH), (palette.stone, SNOW_DEPTH)],
            Surface::Clay => single(palette.clay, CLAY_DEPTH),
            Surface::Stone => single(palette.stone, ROCK_DEPTH),
            Surface::Slate => single(palette.slate, ROCK_DEPTH),
            Surface::Limestone => single(palette.limestone, ROCK_DEPTH),
            // The top cell is always the first layer's; the water reaches no deeper.
            Surface::Shallow => [(palette.water, 0.0), (palette.sand, SAND_DEPTH)],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::palette::numbered_palette;
    use cp_core::coords::MAX_LOD_LEVEL;
    use cp_core::hash::Rng;

    #[test]
    fn default_config_is_valid_and_bad_values_are_rejected() {
        assert_eq!(HillsConfig::default().validate(), Ok(()));
        let cases: [fn(&mut HillsConfig); 9] = [
            |c| c.base_height = f64::NAN,
            |c| c.hill_amplitude = -1.0,
            |c| c.hill_wavelength = 0.0,
            |c| c.ridge_region_wavelength = f64::INFINITY,
            |c| c.ridge_octaves = 33,
            |c| c.tree_density = 1.5,
            |c| c.base_height = 2.0e6,
            |c| c.sea_level = i32::MIN,
            |c| c.deep_water_depth = -0.5,
        ];
        for (index, case) in cases.iter().enumerate() {
            let mut config = HillsConfig::default();
            case(&mut config);
            assert!(
                matches!(config.validate(), Err(WorldgenError::InvalidConfig(_))),
                "case {index}"
            );
        }
    }

    #[test]
    fn heights_stay_inside_the_bounds_on_every_level() {
        let config = HillsConfig::default();
        let terrain = Terrain::new(42, &config);
        let mut rng = Rng::new(42, 9);
        let mut lowest = f64::INFINITY;
        let mut highest = f64::NEG_INFINITY;
        for _ in 0..20_000 {
            let (x, z) = (rng.range_f64(-2.0e5, 2.0e5), rng.range_f64(-2.0e5, 2.0e5));
            for level in 0..=MAX_LOD_LEVEL {
                let h = terrain.height(level, x, z);
                assert!(h >= terrain.min_height(level) && h <= terrain.max_height(level));
                assert!(h <= terrain.max_height_all_levels());
                if level == 0 {
                    lowest = cp_core::det::min(lowest, h);
                    highest = cp_core::det::max(highest, h);
                }
            }
        }
        // The landscape uses its range: lakes below the sea, ridges well above the hills.
        assert!(lowest < 0.0, "{lowest}");
        assert!(
            highest > config.base_height + config.hill_amplitude,
            "{highest}"
        );
        // Fully band-limited levels are flat at the base height.
        assert_eq!(
            terrain.height(MAX_LOD_LEVEL, 1234.5, -77.0),
            config.base_height
        );
    }

    #[test]
    fn neighbouring_levels_differ_by_little() {
        // The point rule plus band limiting keep the surface of level L+1 within a fraction of
        // its cell of level L, column for column.
        let terrain = Terrain::new(7, &HillsConfig::default());
        let mut rng = Rng::new(7, 7);
        for _ in 0..5000 {
            let (x, z) = (rng.range_f64(-1.0e4, 1.0e4), rng.range_f64(-1.0e4, 1.0e4));
            for level in 0..10_u8 {
                let fine = terrain.height(level, x, z);
                let coarse = terrain.height(level + 1, x, z);
                let cell = f64::from(1_u32 << (level + 1));
                assert!(
                    (fine - coarse).abs() <= 0.25 * cell,
                    "{level}: {fine} {coarse}"
                );
            }
        }
    }

    #[test]
    fn no_layer_reaches_below_the_deepest_layer_depth() {
        let terrain = Terrain::new(1, &HillsConfig::default());
        let palette = numbered_palette();
        let depths: Vec<f64> = Surface::ALL
            .iter()
            .flat_map(|&surface| terrain.layers(surface, &palette).map(|layer| layer.1))
            .collect();
        assert!(depths.iter().all(|&depth| depth <= MAX_LAYER_DEPTH));
        // Tight: some layer reaches it.
        assert!(depths.contains(&MAX_LAYER_DEPTH));
    }

    #[test]
    fn rows_up_to_the_base_top_are_stone_in_every_column() {
        let config = HillsConfig::default();
        let palette = numbered_palette();
        let mut rng = Rng::new(5, 1);
        for seed in [1, 42] {
            let terrain = Terrain::new(seed, &config);
            for level in 0..=MAX_LOD_LEVEL {
                let top = terrain.base_top(level);
                let low = terrain.min_height(level);
                // The worst case is the lowest possible surface; then real heights of the field.
                let mut heights = vec![low, low + 1.0e-9, low + 0.5, terrain.max_height(level)];
                for _ in 0..300 {
                    let (x, z) = (rng.range_f64(-2.0e4, 2.0e4), rng.range_f64(-2.0e4, 2.0e4));
                    heights.push(terrain.height(level, x, z));
                }
                for &height in &heights {
                    // Every surface kind's layers, whatever kind the height would choose.
                    for surface in Surface::ALL {
                        let column = Column::new(
                            level,
                            height,
                            terrain.layers(surface, &palette),
                            palette.stone,
                            (palette.water, terrain.sea_level()),
                        );
                        for cy in top - 4..=top {
                            assert_eq!(
                                column.state(cy),
                                palette.stone,
                                "level {level} height {height} {surface:?} row {cy}"
                            );
                        }
                    }
                }
                // Not wastefully low: within two cells and the deepest layer of the bound.
                let cell = f64::from(1_u32 << level);
                assert!(
                    (top as f64 + 0.5) * cell > low - MAX_LAYER_DEPTH - 3.0 * cell,
                    "level {level}: {top} {low}"
                );
            }
        }
    }

    /// Counts the surface kinds of level 0 over a lattice of columns.
    fn census(terrain: &Terrain, half: f64, step: f64) -> Vec<(Surface, usize)> {
        let mut counts: Vec<(Surface, usize)> = Surface::ALL.iter().map(|&s| (s, 0)).collect();
        let mut x = -half;
        while x < half {
            let mut z = -half;
            while z < half {
                let height = terrain.height(0, x, z);
                let slope = terrain.slope_at(0, x, z);
                let surface = terrain.surface(0, x, z, height, slope);
                if let Some(entry) = counts.iter_mut().find(|(s, _)| *s == surface) {
                    entry.1 += 1;
                }
                z += step;
            }
            x += step;
        }
        counts
    }

    #[test]
    fn the_surfaces_follow_height_slope_and_place() {
        let config = HillsConfig {
            ridge_height: 96.0,
            snow_line: 64.0,
            hill_amplitude: 26.0,
            ..HillsConfig::default()
        };
        let terrain = Terrain::new(0x00C0_FFEE_2026, &config);
        // Fixed cases: deep water, flat meadow far from the water, a cliff.
        assert_eq!(terrain.surface(0, 10.5, 20.5, -9.0, 0.0), Surface::Gravel);
        assert_eq!(terrain.surface(0, 10.5, 20.5, 200.0, 0.0), Surface::Snow);
        let cliff = terrain.surface(0, 10.5, 20.5, 30.0, 3.0);
        assert!(
            matches!(cliff, Surface::Stone | Surface::Slate | Surface::Limestone),
            "{cliff:?}"
        );
        // Snow never lies on the steepest slopes, whatever the height.
        assert_ne!(terrain.surface(0, 10.5, 20.5, 200.0, 5.0), Surface::Snow);
        // Over a large area every kind occurs, and grass covers most of the land.
        let counts = census(&terrain, 3000.0, 6.0);
        let total: usize = counts.iter().map(|(_, n)| n).sum();
        for (surface, count) in &counts {
            assert!(*count > 0, "{surface:?} never occurs: {counts:?}");
        }
        let grass = counts[0].1;
        assert!(grass * 2 > total, "{counts:?}");
        // The snow line is not a contour: on the snowiest heights some columns are bare, and
        // some below the mean line are white.
        let mut white_below = 0;
        let mut bare_above = 0;
        let mut rng = Rng::new(3, 3);
        for _ in 0..40_000 {
            let (x, z) = (
                rng.range_f64(-6000.0, 6000.0),
                rng.range_f64(-6000.0, 6000.0),
            );
            let height = terrain.height(0, x, z);
            if (height - config.snow_line).abs() > 8.0 {
                continue;
            }
            let snow = terrain.surface(0, x, z, height, terrain.slope_at(0, x, z)) == Surface::Snow;
            if snow && height < config.snow_line {
                white_below += 1;
            }
            if !snow && height >= config.snow_line {
                bare_above += 1;
            }
        }
        assert!(
            white_below > 20 && bare_above > 20,
            "{white_below} {bare_above}"
        );
    }

    #[test]
    fn beaches_are_narrow_strips_that_vary_along_the_shore() {
        // Walk lines across the land and measure, wherever they come out of the water, how much
        // sand follows before the meadow (or anything else) begins.
        let terrain = Terrain::new(42, &HillsConfig::default());
        let sea = terrain.sea_level();
        let (mut widths, mut bare) = (Vec::new(), 0);
        for line in 0..160 {
            let z = f64::from(line) * 47.0 - 3760.5;
            let mut wet = true;
            let mut x = -4000.5;
            while x < 4000.0 {
                let height = terrain.height(0, x, z);
                let surface = terrain.surface(0, x, z, height, terrain.slope_at(0, x, z));
                let water = height < sea || surface == Surface::Shallow;
                if wet && !water {
                    let mut width = 0.0;
                    let mut ahead = x;
                    while width < 80.0 {
                        let h = terrain.height(0, ahead, z);
                        let kind = terrain.surface(0, ahead, z, h, terrain.slope_at(0, ahead, z));
                        if kind != Surface::Sand {
                            break;
                        }
                        width += 1.0;
                        ahead += 1.0;
                    }
                    if width == 0.0 {
                        bare += 1;
                    } else {
                        widths.push(width);
                    }
                }
                wet = water;
                x += 1.0;
            }
        }
        let shores = widths.len() + bare;
        assert!(shores > 100, "{shores} shores");
        // Many stretches of shore have no sand at all; where there is a beach it is a strip a
        // few to about twenty metres wide, not a plain of a hundred.
        assert!(bare * 5 > shores, "{bare} of {shores} without sand");
        widths.sort_by(f64::total_cmp);
        let median = widths[widths.len() / 2];
        let wide = widths[widths.len() * 9 / 10];
        assert!(
            (3.0..=20.0).contains(&median),
            "median {median}: {widths:?}"
        );
        assert!(wide <= 45.0, "90th percentile {wide}");
        // And the width varies along the shore.
        assert!(widths[widths.len() / 10] < 0.6 * wide, "{widths:?}");
    }

    #[test]
    fn settling_removes_small_blobs_and_keeps_slopes() {
        let level = 0;
        // A plain at 5.7 m (top cell 5) with a hump of one column and a plateau of two by two.
        let mut grid = vec![5.7_f64; 9 * 9];
        grid[2 * 9 + 2] = 6.7;
        for (i, j) in [(5, 5), (6, 5), (5, 6), (6, 6)] {
            grid[j * 9 + i] = 6.6;
        }
        let settled = Terrain::settle(level, &grid, 9, 9);
        assert!(
            settled.iter().all(|&h| (h - 5.7).abs() < 1e-12),
            "{settled:?}"
        );
        // A slope of one cell per column along x keeps every column.
        let slope: Vec<f64> = (0..81).map(|k| f64::from(k % 9) + 0.3).collect();
        let settled = Terrain::settle(level, &slope, 9, 9);
        for (k, &h) in settled.iter().enumerate() {
            assert_eq!(h, f64::from((k % 5) as u32 + 2) + 0.3);
        }
        // A wall three cells high stays a wall.
        let wall: Vec<f64> = (0..81).map(|k| if k % 9 < 4 { 2.5 } else { 5.5 }).collect();
        let settled = Terrain::settle(level, &wall, 9, 9);
        for (k, &h) in settled.iter().enumerate() {
            assert_eq!(h, if k % 5 + 2 < 4 { 2.5 } else { 5.5 });
        }
    }

    #[test]
    fn one_settled_column_equals_the_settled_grid() {
        // The single-column evaluation (with its shortcut on flat ground) and the batch over a
        // grid agree on every level, near steps and far from them.
        let terrain = Terrain::new(0x00C0_FFEE_2026, &HillsConfig::default());
        let mut rng = Rng::new(8, 1);
        let mut shortcuts = 0;
        for _ in 0..3000 {
            let level = rng.range_u32(0, 4) as u8;
            let cx = i64::from(rng.range_u32(0, 8000)) - 4000;
            let cz = i64::from(rng.range_u32(0, 8000)) - 4000;
            let grid: Vec<f64> = (0..25)
                .map(|k| {
                    let (dx, dz) = (k % 5 - 2, k / 5 - 2);
                    terrain.height(level, centre(level, cx + dx), centre(level, cz + dz))
                })
                .collect();
            let batch = Terrain::settle(level, &grid, 5, 5);
            assert_eq!(
                terrain.settled_height(level, cx, cz),
                batch[0],
                "{level} {cx} {cz}"
            );
            shortcuts += usize::from(batch[0] == grid[12]);
        }
        assert!(shortcuts > 1000, "{shortcuts}");
    }

    #[test]
    fn slopes_come_from_the_neighbours() {
        assert_eq!(Terrain::slope([1.0, 1.0, 1.0, 1.0], 1.0), 0.0);
        // Rising 1 m per metre along x and 0 along z: 45 degrees.
        assert!((Terrain::slope([0.0, 2.0, 5.0, 5.0], 1.0) - 1.0).abs() < 1e-12);
        // The same rise over twice the distance on level 1.
        assert!((Terrain::slope([0.0, 2.0, 5.0, 5.0], 2.0) - 0.5).abs() < 1e-12);
        let terrain = Terrain::new(9, &HillsConfig::default());
        let at = terrain.slope_at(0, 100.5, -40.5);
        let by_hand = Terrain::slope(
            [
                terrain.height(0, 99.5, -40.5),
                terrain.height(0, 101.5, -40.5),
                terrain.height(0, 100.5, -41.5),
                terrain.height(0, 100.5, -39.5),
            ],
            1.0,
        );
        assert_eq!(at, by_hand);
    }

    #[test]
    fn dryness_uses_its_range_in_regions() {
        let terrain = Terrain::new(42, &HillsConfig::default());
        let mut rng = Rng::new(1, 2);
        let (mut dry, mut wet) = (0, 0);
        for _ in 0..4000 {
            let (x, z) = (rng.range_f64(-2.0e4, 2.0e4), rng.range_f64(-2.0e4, 2.0e4));
            let d = terrain.dryness(0, x, z);
            assert!((-1.0..=1.0).contains(&d));
            dry += usize::from(d > 0.3);
            wet += usize::from(d < -0.3);
            // Smooth at the scale of a chunk.
            assert!((terrain.dryness(0, x + 8.0, z) - d).abs() < 0.05);
        }
        assert!(dry > 400 && wet > 400, "{dry} {wet}");
    }

    #[test]
    fn seeds_change_the_field() {
        let a = Terrain::new(1, &HillsConfig::default());
        let b = Terrain::new(2, &HillsConfig::default());
        let differs = (0..64).any(|i| {
            let x = f64::from(i) * 37.0;
            a.height(0, x, 11.0) != b.height(0, x, 11.0)
        });
        assert!(differs);
        assert_eq!(a.height(0, 12.5, 7.5), a.clone().height(0, 12.5, 7.5));
    }
}
