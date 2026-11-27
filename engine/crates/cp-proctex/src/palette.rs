//! Palettes: tonal ramps with the pack's hue shift, and the swatches recipes paint with.
//!
//! The art direction of the whole pack is encoded here once: every material colour becomes a
//! short ramp whose shadows lean towards blue-violet and whose highlights lean towards warm
//! yellow. Because every recipe (and every colour a caller passes in) goes through the same
//! builder, the textures share one light and sit together.

use cp_core::det;

use crate::color::{Oklab, Rgb8, oklab_to_rgb8, rgb8_to_linear, rgb8_to_oklab};
use crate::tint::takes_full_tint;

/// One palette entry: a colour and the surface properties of everything painted with it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Swatch {
    pub rgb: Rgb8,
    /// Perceptual roughness in `[0, 1]`.
    pub roughness: f64,
    /// Emitted light in `[0, 1]`, relative to the brightest texel of the pack.
    pub emissive: f64,
}

/// Swatches for `colors` with one roughness value each and no emission.
///
/// # Panics
/// If the slices differ in length (a mistake in a recipe).
pub(crate) fn swatches(colors: &[Rgb8], roughness: &[f64]) -> Vec<Swatch> {
    assert_eq!(colors.len(), roughness.len(), "one roughness per colour");
    colors
        .iter()
        .zip(roughness)
        .map(|(&rgb, &roughness)| Swatch {
            rgb,
            roughness,
            emissive: 0.0,
        })
        .collect()
}

/// Shape of a tonal ramp built around a base colour.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Shading {
    /// Number of tones, at least 2.
    pub tones: usize,
    /// OKLab lightness distance between the darkest and the lightest tone.
    pub spread: f64,
    /// Where the base colour sits in the ramp: 0 makes it the darkest tone, 1 the lightest.
    pub pivot: f64,
    /// Strength of the hue shift relative to the pack's standard (1.0).
    pub shift: f64,
    /// For the parts of a texture that take a biome tint: when the base colour is neutral
    /// enough to be tinted completely, every tone of the ramp is kept that neutral as well.
    pub tintable: bool,
}

impl Shading {
    /// A ramp of `tones` tones centred on the base colour with the standard hue shift.
    pub(crate) const fn new(tones: usize, spread: f64) -> Self {
        Self {
            tones,
            spread,
            pivot: 0.5,
            shift: 1.0,
            tintable: false,
        }
    }

    pub(crate) const fn pivot(mut self, pivot: f64) -> Self {
        self.pivot = pivot;
        self
    }

    pub(crate) const fn shift(mut self, shift: f64) -> Self {
        self.shift = shift;
        self
    }

    pub(crate) const fn tintable(mut self) -> Self {
        self.tintable = true;
        self
    }
}

/// OKLab chroma added per unit of lightness distance from the base tone.
const HUE_SHIFT: f64 = 0.15;
/// Direction of the shadow shift in the OKLab a/b plane: blue-violet (hue about 280°).
const COOL: (f64, f64) = (0.17, -0.985);
/// Direction of the highlight shift: warm yellow (hue about 84°).
const WARM: (f64, f64) = (0.10, 0.995);
/// How much chroma the lightest possible tone loses and the darkest gains, per unit of
/// lightness: highlights bleach, shadows stay rich.
const CHROMA_SLOPE: f64 = 0.9;

/// Lightness range ramps stay in: no tone is pure black or pure white, which would leave the
/// renderer's lighting nothing to work with. L 0.16 is a luminance of about 0.004 in grey.
const DARKEST: f64 = 0.16;
const LIGHTEST: f64 = 0.975;
/// Least luminance (linear, Rec. 709) of the darkest tone. OKLab lightness is not luminance:
/// a saturated blue at [`DARKEST`] reflects less light than a grey, so such a ramp slides up.
const MIN_LUMINANCE: f64 = 0.0045;

/// Builds a ramp from dark to light around `base`.
///
/// Tone `i` is `(i / (tones - 1) - pivot) * spread` lighter than the base. On top of the
/// lightness step, shadows are pushed towards blue-violet and highlights towards warm yellow,
/// proportionally to their distance from the base, so that even a neutral grey base yields a
/// ramp with a little colour temperature in it.
///
/// A ramp that would run past white or black slides back into range as a whole and keeps its
/// even steps (the base colour is then no longer one of its tones).
///
/// With [`Shading::tintable`] and a base colour that takes a biome tint completely, the hue
/// shift of each tone is held back as far as needed for the tone to be tinted completely too
/// (see [`crate::tint_weight`]): a near-neutral colour in gives a fully tintable ramp out.
pub(crate) fn ramp(base: Rgb8, shading: Shading) -> Vec<Rgb8> {
    let origin = rgb8_to_oklab(base);
    let tones = shading.tones.max(2);
    let last = (tones - 1) as f64;
    // The tone of lightness `l`, hue-shifted by its distance from the base.
    let shade = |l: f64| {
        let offset = l - origin.l;
        let chroma = 1.0 - CHROMA_SLOPE * offset;
        let push = offset.abs() * HUE_SHIFT * shading.shift;
        let (da, db) = if offset < 0.0 { COOL } else { WARM };
        Oklab {
            l,
            a: origin.a * chroma + push * da,
            b: origin.b * chroma + push * db,
        }
    };
    let mut darkest = det::max(
        det::min(
            origin.l - shading.pivot * shading.spread,
            LIGHTEST - shading.spread,
        ),
        DARKEST,
    );
    for _ in 0..64 {
        if luminance(oklab_to_rgb8(shade(darkest))) >= MIN_LUMINANCE {
            break;
        }
        darkest += 0.005;
    }
    let keep_neutral = shading.tintable && takes_full_tint(base);
    (0..tones)
        .map(|i| {
            let mut tone = shade(det::min(
                darkest + i as f64 / last * shading.spread,
                LIGHTEST,
            ));
            let mut rgb = oklab_to_rgb8(tone);
            // Fading the chroma ends in a grey, which is always tinted: the loop terminates.
            for _ in 0..96 {
                if !keep_neutral || takes_full_tint(rgb) {
                    break;
                }
                tone.a *= 0.92;
                tone.b *= 0.92;
                rgb = oklab_to_rgb8(tone);
            }
            rgb
        })
        .collect()
}

/// Relative luminance (Rec. 709) of an sRGB colour.
fn luminance(rgb: Rgb8) -> f64 {
    let [r, g, b] = rgb8_to_linear(rgb);
    0.2126 * r + 0.7152 * g + 0.0722 * b
}

/// Ramp of `tones` colours from `dark` to `light`, evenly spaced in OKLab. For materials whose
/// tones are not shades of one colour (lava runs from dark red to yellow).
pub(crate) fn blend(dark: Rgb8, light: Rgb8, tones: usize) -> Vec<Rgb8> {
    let (from, to) = (rgb8_to_oklab(dark), rgb8_to_oklab(light));
    let tones = tones.max(2);
    let last = (tones - 1) as f64;
    (0..tones)
        .map(|i| {
            let t = i as f64 / last;
            oklab_to_rgb8(Oklab {
                l: from.l + (to.l - from.l) * t,
                a: from.a + (to.a - from.a) * t,
                b: from.b + (to.b - from.b) * t,
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    const BASES: [Rgb8; 9] = [
        [139, 134, 130], // stone
        [134, 96, 66],   // dirt
        [226, 208, 158], // sand
        [236, 242, 248], // snow
        [74, 72, 82],    // basalt
        [108, 84, 56],   // bark
        [200, 204, 198], // neutral foliage
        [214, 124, 76],  // copper
        [128, 128, 128], // pure grey
    ];

    /// Warmth of a colour: positive when it leans to yellow/red, negative towards blue.
    fn warmth(rgb: Rgb8) -> f64 {
        rgb8_to_oklab(rgb).b
    }

    #[test]
    fn ramps_run_from_dark_to_light_in_even_steps() {
        // Includes white and black, whose ramps have to slide into range.
        for base in BASES.into_iter().chain([[255, 255, 255], [0, 0, 0]]) {
            for tones in 3..=6 {
                let colors = ramp(base, Shading::new(tones, 0.3));
                assert_eq!(colors.len(), tones);
                let lightness: Vec<f64> = colors.iter().map(|&c| rgb8_to_oklab(c).l).collect();
                for pair in lightness.windows(2) {
                    let step = pair[1] - pair[0];
                    assert!(
                        (step - 0.3 / (tones - 1) as f64).abs() < 0.012,
                        "{base:?}: {lightness:?}"
                    );
                }
                assert!(lightness[0] > 0.07 && lightness[tones - 1] < 0.985);
            }
        }
        // A spread wider than the range cannot be even; it still runs from dark to light.
        let wide = ramp([128, 128, 128], Shading::new(6, 1.5));
        let lightness: Vec<f64> = wide.iter().map(|&c| rgb8_to_oklab(c).l).collect();
        assert!(
            lightness.windows(2).all(|pair| pair[1] >= pair[0]),
            "{lightness:?}"
        );
    }

    #[test]
    fn shadows_are_cooler_and_highlights_warmer_than_plain_shades() {
        for base in BASES {
            let shading = Shading::new(5, 0.3);
            let colors = ramp(base, shading);
            let plain = ramp(base, shading.shift(0.0));
            // The darkest tone leans to blue, the lightest to yellow, compared to the same
            // tones without the hue shift ...
            let cooler = warmth(plain[0]) - warmth(colors[0]);
            let warmer = warmth(colors[4]) - warmth(plain[4]);
            assert!(cooler > 0.008, "{base:?}: {colors:?} vs {plain:?}");
            // (close to white there is no room for colour: such tones are exempt)
            let near_white = rgb8_to_oklab(colors[4]).l > 0.95;
            assert!(
                warmer > 0.008 || near_white,
                "{base:?}: {colors:?} vs {plain:?}"
            );
            // ... but only slightly: both ends stay recognisably the same material.
            assert!(cooler < 0.06 && warmer < 0.06, "{base:?}: {colors:?}");
            // A stronger shift moves them further.
            let strong = ramp(base, shading.shift(2.0));
            assert!(warmth(strong[0]) < warmth(colors[0]), "{base:?}");
        }
        // Without the shift the ramp of a grey is grey.
        let flat = ramp([128, 128, 128], Shading::new(5, 0.3).shift(0.0));
        assert!(
            flat.iter().all(|c| c[0] == c[1] && c[1] == c[2]),
            "{flat:?}"
        );
    }

    #[test]
    fn no_ramp_is_a_pure_grey_ramp() {
        // The art direction rules out neutral greys: even a grey base gets colour temperature.
        let colors = ramp([128, 128, 128], Shading::new(5, 0.3));
        let tinted = colors
            .iter()
            .filter(|c| c.iter().max() != c.iter().min())
            .count();
        assert!(tinted >= 4, "{colors:?}");
        assert!(colors[0][2] > colors[0][0], "shadow not cool: {colors:?}");
        assert!(
            colors[4][0] > colors[4][2],
            "highlight not warm: {colors:?}"
        );
    }

    #[test]
    fn tintable_ramps_of_neutral_colours_stay_neutral() {
        let neutrals: [Rgb8; 6] = [
            [204, 207, 200],
            [198, 203, 194],
            [164, 178, 182],
            [220, 223, 196],
            [128, 128, 128],
            [124, 112, 140],
        ];
        for base in neutrals {
            assert!(takes_full_tint(base), "{base:?}");
            for tones in 3..=6 {
                let shading = Shading::new(tones, 0.4).pivot(0.72);
                let plain = ramp(base, shading);
                let safe = ramp(base, shading.tintable());
                for (plain, safe) in plain.iter().zip(&safe) {
                    assert!(takes_full_tint(*safe), "{base:?}: {safe:?}");
                    // Only chroma is held back: the lightness steps stay as they were.
                    let (a, b) = (rgb8_to_oklab(*plain), rgb8_to_oklab(*safe));
                    assert!((a.l - b.l).abs() < 0.012, "{plain:?} {safe:?}");
                    // Tones that were fine already are not touched.
                    assert!(!takes_full_tint(*plain) || plain == safe);
                }
            }
        }
        // The flag is needed: the plain ramp of a cool neutral leaves the tintable range in
        // its shadows.
        let plain = ramp([164, 178, 182], Shading::new(5, 0.4).pivot(0.72));
        assert!(
            plain.iter().any(|&tone| !takes_full_tint(tone)),
            "{plain:?}"
        );
        // A coloured base is not meant to be tinted and is left alone.
        let green = [96, 152, 60];
        assert!(!takes_full_tint(green));
        assert_eq!(
            ramp(green, Shading::new(5, 0.36).tintable()),
            ramp(green, Shading::new(5, 0.36))
        );
    }

    #[test]
    fn the_pivot_moves_the_base_inside_the_ramp() {
        let base = [134, 96, 66];
        let centred = ramp(base, Shading::new(5, 0.3));
        assert_eq!(centred[2], base);
        let dark_based = ramp(base, Shading::new(5, 0.3).pivot(0.0));
        assert_eq!(dark_based[0], base);
        let light_based = ramp(base, Shading::new(5, 0.3).pivot(1.0));
        assert_eq!(light_based[4], base);
        // Fewer than two tones are bumped to two.
        assert_eq!(ramp(base, Shading::new(0, 0.3)).len(), 2);
    }

    #[test]
    fn blends_interpolate_between_their_ends() {
        let colors = blend([104, 38, 30], [255, 216, 112], 5);
        assert_eq!(colors.len(), 5);
        assert_eq!(colors[0], [104, 38, 30]);
        assert_eq!(colors[4], [255, 216, 112]);
        let lightness: Vec<f64> = colors.iter().map(|&c| rgb8_to_oklab(c).l).collect();
        assert!(lightness.windows(2).all(|pair| pair[1] > pair[0]));
        assert_eq!(blend([1, 2, 3], [4, 5, 6], 1).len(), 2);
    }

    #[test]
    fn swatches_pair_colours_with_roughness() {
        let list = swatches(&[[1, 2, 3], [4, 5, 6]], &[0.25, 0.75]);
        assert_eq!(list.len(), 2);
        assert_eq!(list[1].rgb, [4, 5, 6]);
        assert_eq!(list[1].roughness, 0.75);
        assert_eq!(list[0].emissive, 0.0);
    }
}
