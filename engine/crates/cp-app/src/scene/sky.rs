//! Sun, sky and fog of the E2 scenes: a small analytic daylight model that turns the sun's
//! elevation and azimuth into the [`Lighting`] of cp-render.
//!
//! Directions: the test worlds are planes with `+Y` up. Azimuths are measured clockwise from
//! north (`-Z`) towards east (`+X`), as on a map seen from above; elevations from the horizon.

use cp_core::det;
use cp_core::math::{DVec3, Vec3};
use cp_render::voxel::Lighting;

/// The sun of a frame.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Sun {
    /// Height above the horizon, degrees (`-90..=90`).
    pub elevation_deg: f64,
    /// Compass direction, degrees clockwise from north (`-Z`) towards east (`+X`).
    pub azimuth_deg: f64,
}

impl Sun {
    /// Elevation of the default sun of every shot (contract task: 45 degrees).
    pub const DEFAULT_ELEVATION_DEG: f64 = 45.0;

    /// The sun at the default elevation, from `azimuth_deg`.
    pub fn default_from(azimuth_deg: f64) -> Self {
        Self {
            elevation_deg: Self::DEFAULT_ELEVATION_DEG,
            azimuth_deg,
        }
    }

    /// Unit vector towards the sun.
    pub fn direction(self) -> DVec3 {
        let elevation = self.elevation_deg.to_radians();
        let azimuth = self.azimuth_deg.to_radians();
        let (sin_e, cos_e) = det::sin_cos(elevation);
        let (sin_a, cos_a) = det::sin_cos(azimuth);
        DVec3::new(sin_a * cos_e, sin_e, -cos_a * cos_e)
    }
}

/// Compass azimuth of a direction in degrees, `0..360` (north `-Z` = 0, east `+X` = 90).
pub fn azimuth_of(direction: DVec3) -> f64 {
    let degrees = det::atan2(direction.x, -direction.z).to_degrees();
    degrees.rem_euclid(360.0)
}

/// The unit vector of a view with compass `yaw` and `pitch` above the horizon (radians).
pub fn view_direction(yaw: f64, pitch: f64) -> DVec3 {
    let (sin_y, cos_y) = det::sin_cos(yaw);
    let (sin_p, cos_p) = det::sin_cos(pitch);
    DVec3::new(sin_y * cos_p, sin_p, -cos_y * cos_p)
}

/// Extinction of direct sunlight per unit air mass, per channel (linear Rec.709): Rayleigh and
/// a light aerosol load, enough to warm the sun towards the horizon and to keep a noon sun a
/// warm white.
const EXTINCTION: [f64; 3] = [0.075, 0.13, 0.30];
/// Sun illuminance above the atmosphere, in the units of `Lighting::sun_color`.
const SUN_TOP: f64 = 4.0;
/// Fog density at the sea level (1/m): 22 % of the light is lost to haze over 1 km, 63 % over
/// 4 km, so the far hills fade and the near ones keep their colour.
const FOG_DENSITY: f64 = 2.5e-4;
/// Scale height of the haze (m): it thins out above the sea level, so that peaks and the view
/// from above stand out of it.
const FOG_SCALE_HEIGHT: f64 = 300.0;
/// Albedo of the sunlit ground for the bounce light from below: grass, soil and sand seen
/// together, a warm grey (a yellow bounce would turn the shaded sides of green things neon).
const GROUND_ALBEDO: [f64; 3] = [0.34, 0.33, 0.25];
/// Share of the bounce that reaches a surface facing straight down (the ground is not an
/// infinite plane; neighbouring blocks and trees take part of it). Walls get half of it
/// (cp-render's lighting pass), which lifts the shaded risers of terraces out of black.
const BOUNCE: f64 = 0.7;
/// Ambient intensity at noon: the share of the sky's light that the lighting pass applies.
const AMBIENT: f64 = 0.6;
/// The sky at the zenith on a clear day (linear Rec.709): a deep, slightly violet blue.
const ZENITH_DAY: [f64; 3] = [0.06, 0.19, 0.66];
/// The biome colours of the hills scene (sRGB), until biomes provide their own: a meadow green
/// and a foliage green, both less saturated than cp-proctex's reference tints (which preview
/// textures on the contact sheet), so that sunlit grass reads as grass against a pale sky
/// instead of neon, and leaves in shade keep their colour.
pub const GRASS_TINT: [u8; 3] = [128, 163, 100];
/// See [`GRASS_TINT`].
pub const FOLIAGE_TINT: [u8; 3] = [112, 146, 98];

/// The relative air mass towards elevation `degrees` (Kasten and Young 1989); large but finite
/// at and below the horizon.
fn air_mass(degrees: f64) -> f64 {
    let clamped = det::max(degrees, -2.0);
    let sin = det::sin(clamped.to_radians());
    1.0 / det::max(
        sin + 0.505_72 * det::powf(clamped + 6.079_95, -1.636_4),
        0.02,
    )
}

/// How much the ambient light falls at dusk (the share it loses at the horizon): a little, so
/// that the sides a low sun lights glow against the others, not so much that the land turns
/// to mud under a bright sky.
const DUSK_AMBIENT_FALL: f64 = 0.15;
/// How much the exposure rises at dusk, as an eye adapts.
const DUSK_EXPOSURE_RISE: f64 = 1.4;
/// How much the sky's own brightness is held back at dusk against the land's
/// ([`Lighting::sky_gain`] = `1 / (1 + DUSK_SKY_HOLD · dusk)`): the exposure rises for the land,
/// and the sky, which keeps its brightness longer, would glare over a dark landscape (with
/// these values the sky ends up about as bright as with a tenth more exposure).
const DUSK_SKY_HOLD: f64 = 1.1;

/// The light of a clear day for `sun`, with `+Y` up and the haze based at the sea level
/// `y = 0`.
///
/// The sky is a deep blue at the zenith over a pale, bright band at the horizon; towards a low
/// sun the band turns warm (orange, then pink above the earth's shadow on the other side),
/// while the zenith darkens into a dusk blue. By day the ambient light is the sky's brightness
/// more than its colour, plus the warm bounce of the sun off the ground, so shadows stay
/// neutral instead of cyan; at dusk it keeps more of the sky's colour and of the zenith's blue,
/// so that shade turns cool against the warm light of the low sun. A low sun lowers the
/// ambient a little, so that the sides it lights glow against the ones it does not, and the
/// exposure rises, as an eye adapts at dusk, while the sky's own brightness is held back.
pub fn daylight(sun: Sun) -> Lighting {
    let elevation = sun.elevation_deg;
    let mass = air_mass(elevation);
    // Fades the sun out over the last degrees above the horizon.
    let above = smooth(-1.0, 3.0, elevation);
    let transmit = EXTINCTION.map(|k| det::exp(-k * mass));
    let sun_color = transmit.map(|t| SUN_TOP * t * above);
    // How much of the day the sky has: dim at dusk, full from about 20 degrees.
    let day = smooth(-6.0, 20.0, elevation);
    let low = 1.0 - smooth(2.0, 25.0, elevation);
    let mix = |a: [f64; 3], b: [f64; 3], t: f64| [0, 1, 2].map(|i| a[i] + (b[i] - a[i]) * t);
    let zenith_day = ZENITH_DAY;
    let zenith_dusk = [0.03, 0.06, 0.20];
    let horizon_day = [0.50, 0.66, 0.98];
    // Away from a low sun: the pink of the light scattered back over the earth's shadow.
    let horizon_dusk = [0.95, 0.62, 0.60];
    // Below a low sun: the warm glow of the long path through the haze, bright enough to
    // stay orange through the tone map.
    let sunset_low = [2.0, 0.95, 0.35];
    let sunset_day = [0.56, 0.70, 0.98];
    // The zenith darkens before the horizon does.
    let sky_zenith = mix(zenith_dusk, zenith_day, day * day);
    let dim = 0.25 + 0.75 * day;
    let sky_horizon = mix(horizon_day, horizon_dusk, low * 0.85).map(|c| c * dim);
    let sky_sunset = mix(sunset_day, sunset_low, low).map(|c| c * (0.5 + 0.5 * day));
    // Light coming up from the sunlit ground: its albedo times the sun on a level surface.
    let level = det::max(sun.direction().y, 0.0);
    let ground = [0, 1, 2].map(|i| GROUND_ALBEDO[i] * BOUNCE * sun_color[i] * level + 0.02 * dim);
    let vec = |c: [f64; 3]| Vec3::new(c[0] as f32, c[1] as f32, c[2] as f32);
    let srgb_tint = |rgb: [u8; 3]| vec(rgb.map(|c| srgb_to_linear(f64::from(c) / 255.0)));
    let direction = sun.direction();
    let dusk = 1.0 - smooth(0.0, 15.0, elevation);
    let sky_light = Lighting::sky_light_from(
        vec(sky_horizon),
        vec(sky_zenith),
        (0.35 + 0.15 * dusk) as f32,
        (0.6 + 0.25 * dusk) as f32,
    );
    Lighting {
        up: Vec3::Y,
        sun_direction: direction.as_vec3(),
        sun_color: vec(sun_color),
        sun_angular_radius: 0.004_65,
        sky_zenith: vec(sky_zenith),
        sky_horizon: vec(sky_horizon),
        sky_sunset: vec(sky_sunset),
        ground: vec(ground),
        sky_light,
        sky_gain: (1.0 / (1.0 + DUSK_SKY_HOLD * dusk)) as f32,
        ambient: (AMBIENT * (1.0 - DUSK_AMBIENT_FALL * dusk)) as f32,
        fog_density: FOG_DENSITY as f32,
        fog_falloff: (1.0 / FOG_SCALE_HEIGHT) as f32,
        fog_base: 0.0,
        // The scene sets the fade from the size of its world.
        horizon_fade: [0.0, 0.0],
        exposure: (1.0 + DUSK_EXPOSURE_RISE * dusk) as f32,
        // The scene's biome colours; water keeps cp-proctex's reference tint.
        tints: [
            Vec3::ONE,
            srgb_tint(GRASS_TINT),
            srgb_tint(FOLIAGE_TINT),
            srgb_tint(cp_proctex::Tint::Water.reference_color()),
        ],
    }
}

/// `smoothstep(edge0, edge1, x)` in `f64`.
fn smooth(edge0: f64, edge1: f64, x: f64) -> f64 {
    let t = det::clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// The sRGB transfer function, inverted: encoded `0..=1` → linear.
pub fn srgb_to_linear(encoded: f64) -> f64 {
    if encoded <= 0.040_45 {
        encoded / 12.92
    } else {
        det::powf((encoded + 0.055) / 1.055, 2.4)
    }
}

/// The sRGB transfer function: linear `0..=1` → encoded.
pub fn linear_to_srgb(linear: f64) -> f64 {
    let c = det::clamp(linear, 0.0, 1.0);
    if c <= 0.003_130_8 {
        12.92 * c
    } else {
        1.055 * det::powf(c, 1.0 / 2.4) - 0.055
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sun_directions_follow_the_compass() {
        let north = Sun {
            elevation_deg: 0.0,
            azimuth_deg: 0.0,
        }
        .direction();
        assert!((north - DVec3::NEG_Z).length() < 1e-12);
        let east = Sun {
            elevation_deg: 0.0,
            azimuth_deg: 90.0,
        }
        .direction();
        assert!((east - DVec3::X).length() < 1e-12);
        let zenith = Sun {
            elevation_deg: 90.0,
            azimuth_deg: 123.0,
        }
        .direction();
        assert!((zenith - DVec3::Y).length() < 1e-12);
        assert!((azimuth_of(DVec3::X) - 90.0).abs() < 1e-9);
        assert!((azimuth_of(DVec3::new(-1.0, 0.0, 0.0)) - 270.0).abs() < 1e-9);
        let view = view_direction(90.0_f64.to_radians(), 0.0);
        assert!((view - DVec3::X).length() < 1e-12);
    }

    #[test]
    fn dusk_glows_towards_the_sun_and_the_shadows_stay_neutral() {
        let noon = daylight(Sun::default_from(135.0));
        let dusk = daylight(Sun {
            elevation_deg: 8.0,
            azimuth_deg: 250.0,
        });
        let warmth = |c: Vec3| c.x / c.z;
        // Below a low sun the horizon is warm, away from it it is not; at noon both are blue.
        assert!(warmth(dusk.sky_sunset) > 2.0 * warmth(dusk.sky_horizon));
        assert!(warmth(noon.sky_sunset) < 1.0 && warmth(noon.sky_horizon) < 1.0);
        // The ambient falls at dusk while the exposure rises, so lit sides glow against the
        // others; the bounce light from the ground is warm.
        assert!(dusk.ambient < noon.ambient);
        assert!(dusk.exposure > noon.exposure);
        assert!(noon.ground.x >= noon.ground.z);
        // The sky's own brightness is held back at dusk, so that the raised exposure lifts the
        // land more than the sky; by day it is untouched.
        assert_eq!(noon.sky_gain, 1.0);
        assert!(dusk.sky_gain < 1.0);
        // The sky light keeps more of the sky's colour at dusk than by day: cool against the
        // warm sun, where the light of the day's mix would be a warm grey.
        let blue = |c: Vec3| c.z / c.y;
        let day_mix =
            |l: &Lighting| Lighting::sky_light_from(l.sky_horizon, l.sky_zenith, 0.35, 0.6);
        assert!(blue(dusk.sky_light) > 1.05 * blue(day_mix(&dusk)));
        assert!(blue(dusk.sky_light) > 3.0 * blue(dusk.sun_color));
        assert!((noon.sky_light - day_mix(&noon)).length() < 1e-6);
        // Height fog: thicker at the sea level than up high, no fade until the scene sets one.
        assert!(noon.fog_density > 0.0 && noon.fog_falloff > 0.0);
        assert_eq!(noon.horizon_fade, [0.0, 0.0]);
    }

    #[test]
    fn the_sun_warms_and_dims_towards_the_horizon() {
        let high = daylight(Sun::default_from(0.0));
        let low = daylight(Sun {
            elevation_deg: 4.0,
            azimuth_deg: 0.0,
        });
        let night = daylight(Sun {
            elevation_deg: -10.0,
            azimuth_deg: 0.0,
        });
        assert!(high.sun_color.x > low.sun_color.x);
        assert!(
            high.sun_color.x > high.sun_color.z,
            "a warm white sun at 45 degrees"
        );
        assert!(low.sun_color.x / low.sun_color.z > high.sun_color.x / high.sun_color.z);
        assert_eq!(night.sun_color, Vec3::ZERO);
        assert!(high.sky_zenith.z > high.sky_zenith.x);
        assert!((srgb_to_linear(linear_to_srgb(0.3)) - 0.3).abs() < 1e-9);
    }
}
