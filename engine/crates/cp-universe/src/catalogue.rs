//! The procedural star catalogue of the galaxy.
//!
//! The galaxy is a Milky-Way-like disc of about 6·10¹⁰ stars, centred on the galaxy origin
//! ([`GalaxyPos::ORIGIN`]) with its plane in the galaxy's XZ plane (north `+Y`). Space is cut
//! into *catalogue cells* of 2¹⁶ galaxy sectors per axis (2⁵⁶ m ≈ 2.34 pc), the unit of
//! generation: the stars of a cell are a pure function of the catalogue seed and the cell, so
//! any cell can be generated on its own, in any order, on any machine, with the same result
//! (contract §2.5). A galaxy sector (2⁴⁰ m) is far too small for that role: at the solar
//! neighbourhood's density of 0.1 stars per cubic parsec it holds 10⁻¹⁴ stars.
//!
//! Per cell: the expected count follows the galaxy's density model at the cell centre (an
//! exponential disc with two logarithmic spiral arms plus a bulge), the count is Poisson
//! distributed, positions are uniform in the cell, masses follow Kroupa's initial mass
//! function thinned by main-sequence lifetimes (dead stars are white dwarfs), and temperature
//! and luminosity follow main-sequence relations, with a share of red giants. About 0.8 % of
//! the stars are brighter than ten Suns, as in the solar neighbourhood. The home system sits at
//! the fixed [`HOME_POSITION`]; it is entry 0 of its cell, and no generated star comes closer
//! to it than [`HOME_EXCLUSION_RADIUS`].
//!
//! Every star is a visitable system: its [`StarEntry::id`] packs the cell and the star's index
//! in it, so [`StarCatalogue::star`] finds it again from the id alone, and its
//! [`StarEntry::system_seed`] seeds the generation of its planets.

use core::f64::consts::{PI, TAU};

use cp_core::det;
use cp_core::frames::GalaxyPos;
use cp_core::hash::{Rng, hash_combine, hash_str};
use cp_core::ids::SystemId;
use cp_core::math::DVec3;

/// Version of the generated catalogue; bumped whenever the stars of any cell change. Part of
/// the cell seed.
pub const CATALOGUE_VERSION: u32 = 1;

/// Bits of a catalogue cell in galaxy sectors per axis: a cell is 2¹⁶ sectors wide.
pub const CELL_SECTOR_BITS: u32 = 16;

/// Edge length of a catalogue cell in metres, 2⁵⁶ m (≈ 2.34 pc).
pub const CELL_SIZE: f64 = (1_u64 << (CELL_SECTOR_BITS + 40)) as f64;

/// One parsec in metres (`648000 / π` astronomical units).
pub const PARSEC: f64 = 149_597_870_700.0 * 648_000.0 / PI;

/// Nominal solar luminosity in watts (IAU 2015).
pub const SOLAR_LUMINOSITY: f64 = 3.828e26;

/// Nominal solar effective temperature in kelvin (IAU 2015).
pub const SOLAR_TEMPERATURE: f64 = 5772.0;

/// The fixed position of the home system's barycentre: 8.2 kpc from the galactic centre along
/// `+X`, 20 pc above the plane (like the Sun), in the middle of a galaxy sector.
pub const HOME_POSITION: GalaxyPos = GalaxyPos {
    sector: [230_000_000, 560_000, 32_768],
    local: DVec3::splat(549_755_813_888.0),
};

/// The catalogue cell of the home system.
pub const HOME_CELL: [i64; 3] = [
    HOME_POSITION.sector[0] >> CELL_SECTOR_BITS,
    HOME_POSITION.sector[1] >> CELL_SECTOR_BITS,
    HOME_POSITION.sector[2] >> CELL_SECTOR_BITS,
];

/// The id of the home system: entry 0 of [`HOME_CELL`].
pub const HOME_SYSTEM_ID: SystemId = pack_system_id(HOME_CELL, 0);

/// Seed of the home system (it is hand-made; the seed is for anything derived from it).
pub const HOME_SYSTEM_SEED: u64 = hash_str("cubeplanet:home");

/// No generated star lies closer to [`HOME_POSITION`] than this, in metres (one parsec; the
/// Sun's nearest neighbour is 1.3 pc away).
pub const HOME_EXCLUSION_RADIUS: f64 = PARSEC;

/// Largest radius [`StarCatalogue::stars_within`] accepts, in metres (50 pc; the cost grows
/// with the cube of the radius).
pub const MAX_QUERY_RADIUS: f64 = 50.0 * PARSEC;

/// Most stars one cell can hold (indices `1..=MAX_STARS_PER_CELL`; 0 is reserved for
/// hand-made systems).
pub const MAX_STARS_PER_CELL: u32 = u16::MAX as u32;

/// One star of the catalogue, the sun of a (possibly unvisited) system.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StarEntry {
    /// The system id: cell and index in the cell, see [`split_system_id`].
    pub id: SystemId,
    /// Position of the system's barycentre (the star).
    pub position: GalaxyPos,
    /// Effective temperature in kelvin.
    pub temperature: f64,
    /// Luminosity in watts.
    pub luminosity: f64,
    /// Seed of the system's procedural planets.
    pub system_seed: u64,
}

impl StarEntry {
    /// Colour of the star: its blackbody colour in linear sRGB, see [`blackbody_color`].
    #[must_use]
    pub fn color(&self) -> DVec3 {
        blackbody_color(self.temperature)
    }

    /// Luminosity in units of the Sun's.
    #[must_use]
    pub fn solar_luminosities(&self) -> f64 {
        self.luminosity / SOLAR_LUMINOSITY
    }
}

/// Packs a cell and an index into a [`SystemId`]: bits 48–63 the cell's x, 32–47 y, 16–31 z
/// (each as a 16-bit two's complement number), 0–15 the index.
const fn pack_system_id(cell: [i64; 3], index: u16) -> SystemId {
    let x = cell[0] as i16 as u16 as u64;
    let y = cell[1] as i16 as u16 as u64;
    let z = cell[2] as i16 as u16 as u64;
    SystemId((x << 48) | (y << 32) | (z << 16) | index as u64)
}

/// The id of entry `index` of `cell`; `None` for cells outside the catalogue's range of
/// `-32768..=32767` cells per axis (±76 kpc, beyond the galaxy).
#[must_use]
pub fn system_id(cell: [i64; 3], index: u16) -> Option<SystemId> {
    let fits = cell.iter().all(|&axis| i16::try_from(axis).is_ok());
    fits.then(|| pack_system_id(cell, index))
}

/// The cell and the index in the cell of a system id.
#[must_use]
pub fn split_system_id(id: SystemId) -> ([i64; 3], u16) {
    let axis = |shift: u32| i64::from((id.0 >> shift) as u16 as i16);
    ([axis(48), axis(32), axis(16)], (id.0 & 0xFFFF) as u16)
}

/// The catalogue cell that contains `position`.
#[must_use]
pub fn cell_of(position: &GalaxyPos) -> [i64; 3] {
    let normalized = position.normalized();
    normalized.sector.map(|sector| sector >> CELL_SECTOR_BITS)
}

/// The galaxy position of a cell's minimum corner.
#[must_use]
pub fn cell_origin(cell: [i64; 3]) -> GalaxyPos {
    GalaxyPos::new(cell.map(|axis| axis << CELL_SECTOR_BITS), DVec3::ZERO)
}

// ---------------------------------------------------------------------------------------------
// The galaxy model
// ---------------------------------------------------------------------------------------------

/// Stars per cubic parsec in the disc plane at the home system's distance from the centre.
const LOCAL_DENSITY: f64 = 0.1;
/// Galactocentric radius of the home system in parsecs.
const HOME_RADIUS: f64 = 8_200.0;
/// Radial scale length of the disc in parsecs.
const DISC_SCALE_LENGTH: f64 = 2_600.0;
/// Vertical scale height of the disc in parsecs.
const DISC_SCALE_HEIGHT: f64 = 300.0;
/// Contrast of the spiral arms: the disc density varies by this factor around its mean.
const ARM_CONTRAST: f64 = 0.5;
/// Pitch angle of the logarithmic spiral arms, radians (13°).
const ARM_PITCH: f64 = 13.0 * PI / 180.0;
/// Number of spiral arms.
const ARM_COUNT: f64 = 2.0;
/// Stars per cubic parsec at the centre of the bulge.
const BULGE_DENSITY: f64 = 40.0;
/// Scale radius of the exponential bulge in parsecs.
const BULGE_SCALE: f64 = 500.0;
/// No stars beyond this galactocentric radius in the plane, parsecs.
const GALAXY_RADIUS: f64 = 20_000.0;
/// No stars farther than this from the plane, parsecs.
const GALAXY_HALF_THICKNESS: f64 = 2_500.0;

/// Expected number of stars per cubic metre at `offset` metres from the galactic centre
/// (the galaxy model of the catalogue).
#[must_use]
pub fn star_density(offset: DVec3) -> f64 {
    let pc = offset / PARSEC;
    let planar = (pc.x * pc.x + pc.z * pc.z).sqrt();
    if planar > GALAXY_RADIUS || pc.y.abs() > GALAXY_HALF_THICKNESS {
        return 0.0;
    }
    let disc = LOCAL_DENSITY
        * det::exp((HOME_RADIUS - planar) / DISC_SCALE_LENGTH)
        * det::exp(-pc.y.abs() / DISC_SCALE_HEIGHT);
    // Two logarithmic spiral arms: the phase winds with ln(R) / tan(pitch).
    let angle = det::atan2(pc.z, pc.x);
    let winding = det::ln(det::max(planar, 1.0) / HOME_RADIUS) / det::tan(ARM_PITCH);
    let arms = 1.0 + ARM_CONTRAST * det::cos(ARM_COUNT * (angle - winding));
    let radius = pc.length();
    let bulge = BULGE_DENSITY * det::exp(-radius / BULGE_SCALE);
    (disc * arms + bulge) / (PARSEC * PARSEC * PARSEC)
}

// ---------------------------------------------------------------------------------------------
// Stellar population
// ---------------------------------------------------------------------------------------------

/// Masses in solar masses: Kroupa's initial mass function between these bounds (the upper
/// bound is where the simple relations below stop describing real stars).
const MASS_MIN: f64 = 0.08;
const MASS_BREAK: f64 = 0.5;
const MASS_MAX: f64 = 60.0;
/// Power-law slopes of `dN/dM ∝ M^-α` below and above the break.
const SLOPE_LOW: f64 = 1.3;
const SLOPE_HIGH: f64 = 2.3;
/// Share of living stars of at least [`GIANT_MIN_MASS`] that are red giants.
const GIANT_SHARE: f64 = 0.1;
const GIANT_MIN_MASS: f64 = 0.9;
/// Main-sequence lifetime `10 Gyr · M^-2.5` against a galaxy that formed stars evenly for
/// 10 Gyr: a star of more than one solar mass is still alive with probability `M^-2.5`,
/// otherwise it is a white dwarf. This turns the initial into the present-day mass function.
const LIFETIME_EXPONENT: f64 = -2.5;

/// Inverse CDF of a power law `M^-alpha` on `[low, high]`.
fn power_law(u: f64, low: f64, high: f64, alpha: f64) -> f64 {
    let exponent = 1.0 - alpha;
    let a = det::powf(low, exponent);
    let b = det::powf(high, exponent);
    det::powf(a + u * (b - a), 1.0 / exponent)
}

/// Integral of `M^-alpha` over `[low, high]`.
fn power_law_weight(low: f64, high: f64, alpha: f64) -> f64 {
    let exponent = 1.0 - alpha;
    (det::powf(high, exponent) - det::powf(low, exponent)) / exponent
}

fn sample_mass(rng: &mut Rng) -> f64 {
    let low = power_law_weight(MASS_MIN, MASS_BREAK, SLOPE_LOW);
    // Continuity at the break: the upper segment carries the factor MASS_BREAK^(α₂ - α₁).
    let high = det::powf(MASS_BREAK, SLOPE_HIGH - SLOPE_LOW)
        * power_law_weight(MASS_BREAK, MASS_MAX, SLOPE_HIGH);
    let segment = rng.f64();
    let u = rng.f64();
    if segment * (low + high) < low {
        power_law(u, MASS_MIN, MASS_BREAK, SLOPE_LOW)
    } else {
        power_law(u, MASS_BREAK, MASS_MAX, SLOPE_HIGH)
    }
}

/// Temperature (K) and luminosity (solar units) of a main-sequence star of `mass` solar
/// masses: the textbook piecewise mass–luminosity relation, the mass–radius fit of Demircan
/// and Kahraman (1991) and Stefan–Boltzmann. From 0.08 to 60 solar masses this spans late M
/// dwarfs near 3000 K to O stars near 60 000 K.
fn main_sequence(mass: f64) -> (f64, f64) {
    let luminosity = if mass < 0.43 {
        0.23 * det::powf(mass, 2.3)
    } else if mass < 2.0 {
        det::powi(mass, 4)
    } else if mass < 55.0 {
        1.4 * det::powf(mass, 3.5)
    } else {
        32_000.0 * mass
    };
    let radius = if mass < 1.66 {
        1.06 * det::powf(mass, 0.945)
    } else {
        1.33 * det::powf(mass, 0.555)
    };
    let temperature = SOLAR_TEMPERATURE * det::powf(luminosity / (radius * radius), 0.25);
    (temperature, luminosity)
}

/// Temperature (K) and luminosity (solar units) of one star of today's population: a mass from
/// the initial mass function, then a white dwarf if it has outlived its main-sequence lifetime,
/// a red giant with [`GIANT_SHARE`], or a main-sequence star.
fn sample_star(rng: &mut Rng) -> (f64, f64) {
    let mass = sample_mass(rng);
    if mass > 1.0 && rng.f64() >= det::powf(mass, LIFETIME_EXPONENT) {
        let temperature = rng.range_f64(5_000.0, 25_000.0);
        let luminosity = det::exp(rng.range_f64(det::ln(2e-4), det::ln(1e-2)));
        return (temperature, luminosity);
    }
    if mass >= GIANT_MIN_MASS && rng.f64() < GIANT_SHARE {
        let temperature = rng.range_f64(3_300.0, 5_200.0);
        let luminosity = det::exp(rng.range_f64(det::ln(30.0), det::ln(3_000.0)));
        return (temperature, luminosity);
    }
    main_sequence(mass)
}

/// A Poisson-distributed count with mean `mean`: by multiplication of uniforms for small means,
/// by the rounded normal approximation for large ones.
fn poisson(rng: &mut Rng, mean: f64) -> u32 {
    if mean <= 0.0 {
        return 0;
    }
    if mean < 30.0 {
        let limit = det::exp(-mean);
        let mut count = 0_u32;
        let mut product = rng.f64();
        while product > limit {
            count += 1;
            product *= rng.f64();
        }
        return count;
    }
    // Box–Muller; `1 - u` keeps the logarithm away from zero.
    let radius = (-2.0 * det::ln(1.0 - rng.f64())).sqrt();
    let normal = radius * det::cos(TAU * rng.f64());
    let count = (mean + mean.sqrt() * normal).round();
    if count <= 0.0 {
        0
    } else {
        // Saturating cast; the caller caps the count.
        count as u32
    }
}

/// The catalogue: the seeded procedural stars of the galaxy plus the fixed home system.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct StarCatalogue {
    seed: u64,
}

impl StarCatalogue {
    /// The catalogue of the galaxy with this seed.
    #[must_use]
    pub const fn new(seed: u64) -> Self {
        Self { seed }
    }

    /// The galaxy seed.
    #[must_use]
    pub const fn seed(&self) -> u64 {
        self.seed
    }

    /// The home system's entry: a Sun-like star at [`HOME_POSITION`].
    #[must_use]
    pub fn home(&self) -> StarEntry {
        StarEntry {
            id: HOME_SYSTEM_ID,
            position: HOME_POSITION,
            temperature: SOLAR_TEMPERATURE,
            luminosity: SOLAR_LUMINOSITY,
            system_seed: HOME_SYSTEM_SEED,
        }
    }

    /// Expected number of generated stars in `cell`: the model density at the cell centre
    /// times the cell volume.
    #[must_use]
    pub fn expected_stars(&self, cell: [i64; 3]) -> f64 {
        let centre = cell.map(|axis| (axis as f64 + 0.5) * CELL_SIZE);
        star_density(DVec3::from_array(centre)) * CELL_SIZE * CELL_SIZE * CELL_SIZE
    }

    fn cell_rng(&self, cell: [i64; 3]) -> Rng {
        let mut hash = hash_combine(self.seed, u64::from(CATALOGUE_VERSION));
        for axis in cell {
            hash = hash_combine(hash, axis as u64);
        }
        Rng::new(hash, 0x5354_4152)
    }

    /// All stars of `cell`, sorted by id (index in the cell). The home cell starts with the
    /// home system. Cells outside the catalogue's range are empty.
    #[must_use]
    pub fn cell_stars(&self, cell: [i64; 3]) -> Vec<StarEntry> {
        if system_id(cell, 0).is_none() {
            return Vec::new();
        }
        let mut stars = Vec::new();
        if cell == HOME_CELL {
            stars.push(self.home());
        }
        let rng = self.cell_rng(cell);
        let mut count_rng = rng.fork(0);
        let count = poisson(&mut count_rng, self.expected_stars(cell)).min(MAX_STARS_PER_CELL);
        let origin = cell.map(|axis| axis << CELL_SECTOR_BITS);
        for index in 1..=count {
            let index = u16::try_from(index).expect("the count is capped at u16::MAX");
            let mut star = rng.fork(u64::from(index));
            // 56 random bits per axis: 16 select the sector in the cell, 40 the metre in it.
            let offsets = [
                star.next_u64() >> 8,
                star.next_u64() >> 8,
                star.next_u64() >> 8,
            ];
            let position = GalaxyPos {
                sector: core::array::from_fn(|axis| origin[axis] + (offsets[axis] >> 40) as i64),
                local: DVec3::from_array(core::array::from_fn(|axis| {
                    (offsets[axis] & ((1 << 40) - 1)) as f64
                })),
            };
            if GalaxyPos::delta(&HOME_POSITION, &position).length() < HOME_EXCLUSION_RADIUS {
                continue;
            }
            let (temperature, luminosity) = sample_star(&mut star);
            let id = pack_system_id(cell, index);
            stars.push(StarEntry {
                id,
                position,
                temperature,
                luminosity: luminosity * SOLAR_LUMINOSITY,
                system_seed: hash_combine(hash_combine(self.seed, 0x5359_5354), id.0),
            });
        }
        stars
    }

    /// The star with this id, if the catalogue has it.
    #[must_use]
    pub fn star(&self, id: SystemId) -> Option<StarEntry> {
        let (cell, _) = split_system_id(id);
        self.cell_stars(cell)
            .into_iter()
            .find(|entry| entry.id == id)
    }

    /// Every star within `radius` metres of `center`, sorted by distance, ties by id.
    ///
    /// # Panics
    /// If `radius` is negative, not finite or above [`MAX_QUERY_RADIUS`].
    #[must_use]
    pub fn stars_within(&self, center: &GalaxyPos, radius: f64) -> Vec<StarEntry> {
        assert!(
            (0.0..=MAX_QUERY_RADIUS).contains(&radius),
            "stars_within: radius {radius} m outside 0..={MAX_QUERY_RADIUS} m"
        );
        let low = cell_of(&center.offset(DVec3::splat(-radius)));
        let high = cell_of(&center.offset(DVec3::splat(radius)));
        let mut found: Vec<(f64, StarEntry)> = Vec::new();
        for x in low[0]..=high[0] {
            for y in low[1]..=high[1] {
                for z in low[2]..=high[2] {
                    for star in self.cell_stars([x, y, z]) {
                        let distance = GalaxyPos::delta(center, &star.position).length();
                        if distance <= radius {
                            found.push((distance, star));
                        }
                    }
                }
            }
        }
        found.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.id.cmp(&b.1.id)));
        found.into_iter().map(|(_, star)| star).collect()
    }
}

// ---------------------------------------------------------------------------------------------
// Colour
// ---------------------------------------------------------------------------------------------

/// One lobe of the CIE 1931 colour matching function fit of Wyman, Sloan and Shirley (2013):
/// a Gaussian with different widths left and right of its centre, wavelengths in nm.
fn lobe(wavelength: f64, centre: f64, left: f64, right: f64) -> f64 {
    let width = if wavelength < centre { left } else { right };
    let t = (wavelength - centre) / width;
    det::exp(-0.5 * t * t)
}

/// The CIE 1931 2° observer (multi-lobe fit) at `wavelength` nm.
fn color_matching(wavelength: f64) -> DVec3 {
    let x = 1.056 * lobe(wavelength, 599.8, 37.9, 31.0)
        + 0.362 * lobe(wavelength, 442.0, 16.0, 26.7)
        - 0.065 * lobe(wavelength, 501.1, 20.4, 26.2);
    let y =
        0.821 * lobe(wavelength, 568.8, 46.9, 40.5) + 0.286 * lobe(wavelength, 530.9, 16.3, 31.1);
    let z =
        1.217 * lobe(wavelength, 437.0, 11.8, 36.0) + 0.681 * lobe(wavelength, 459.0, 26.0, 13.8);
    DVec3::new(x, y, z)
}

/// Second radiation constant `hc / k` in nm·K.
const RADIATION_C2: f64 = 1.438_777e7;

/// Colour of a blackbody of `temperature` kelvin as linear sRGB (D65), scaled so that the
/// largest component is 1 (negative components of colours outside the sRGB gamut are cut to
/// 0). The Sun (5772 K) is a faint yellowish white, 6500 K close to white, M dwarfs orange-red,
/// O and B stars blue-white. Brightness is not part of the colour; use the luminosity.
#[must_use]
pub fn blackbody_color(temperature: f64) -> DVec3 {
    let temperature = det::clamp(temperature, 500.0, 1.0e6);
    let mut xyz = DVec3::ZERO;
    let mut wavelength = 380.0;
    while wavelength <= 780.0 {
        // Planck's law up to a constant factor, wavelength in micrometres for the λ⁻⁵ term.
        let micrometres = wavelength / 1000.0;
        let radiance = 1.0
            / (micrometres * micrometres * micrometres * micrometres * micrometres)
            / det::exp_m1(RADIATION_C2 / (wavelength * temperature));
        xyz += color_matching(wavelength) * radiance;
        wavelength += 5.0;
    }
    let rgb = DVec3::new(
        3.2406 * xyz.x - 1.5372 * xyz.y - 0.4986 * xyz.z,
        -0.9689 * xyz.x + 1.8758 * xyz.y + 0.0415 * xyz.z,
        0.0557 * xyz.x - 0.2040 * xyz.y + 1.0570 * xyz.z,
    );
    let rgb = DVec3::new(
        det::max(rgb.x, 0.0),
        det::max(rgb.y, 0.0),
        det::max(rgb.z, 0.0),
    );
    let largest = det::max(rgb.x, det::max(rgb.y, rgb.z));
    if largest > 0.0 {
        rgb / largest
    } else {
        DVec3::ONE
    }
}

#[cfg(test)]
mod tests;
