//! Colour maths: the sRGB transfer function, linear RGB and the OKLab perceptual space.
//!
//! All transcendental functions go through `cp_core::det`, so palettes are bit-identical on
//! every machine.

use cp_core::det;

/// sRGB colour, 8 bits per channel.
pub(crate) type Rgb8 = [u8; 3];

/// Decodes one 8-bit sRGB channel to linear light in `[0, 1]`.
pub(crate) fn srgb_to_linear(channel: u8) -> f64 {
    let c = f64::from(channel) / 255.0;
    if c <= 0.040_45 {
        c / 12.92
    } else {
        det::powf((c + 0.055) / 1.055, 2.4)
    }
}

/// Encodes linear light as the nearest 8-bit sRGB channel value. Input outside `[0, 1]` is
/// clamped; NaN becomes 0.
pub(crate) fn linear_to_srgb(value: f64) -> u8 {
    let v = if value > 0.0 {
        det::min(value, 1.0)
    } else {
        0.0
    };
    let encoded = if v <= 0.003_130_8 {
        v * 12.92
    } else {
        1.055 * det::powf(v, 1.0 / 2.4) - 0.055
    };
    // Truncation after adding one half rounds to nearest; the value is within [0.5, 255.5].
    (encoded * 255.0 + 0.5) as u8
}

/// Linear RGB of an sRGB colour, each channel in `[0, 1]`.
pub(crate) fn rgb8_to_linear(rgb: Rgb8) -> [f64; 3] {
    rgb.map(srgb_to_linear)
}

/// Nearest sRGB colour of a linear RGB triple (channels clamped to `[0, 1]`).
pub(crate) fn linear_to_rgb8(linear: [f64; 3]) -> Rgb8 {
    linear.map(linear_to_srgb)
}

/// A colour in OKLab (Ottosson 2020): `l` is perceived lightness in `[0, 1]`, `a` runs from
/// green to red and `b` from blue to yellow. Equal distances look about equally large, which
/// makes it the right space for building tonal ramps.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Oklab {
    pub l: f64,
    pub a: f64,
    pub b: f64,
}

pub(crate) fn linear_to_oklab(rgb: [f64; 3]) -> Oklab {
    let [r, g, b] = rgb;
    let l = det::cbrt(0.412_221_470_8 * r + 0.536_332_536_3 * g + 0.051_445_992_9 * b);
    let m = det::cbrt(0.211_903_498_2 * r + 0.680_699_545_1 * g + 0.107_396_956_6 * b);
    let s = det::cbrt(0.088_302_461_9 * r + 0.281_718_837_6 * g + 0.629_978_700_5 * b);
    Oklab {
        l: 0.210_454_255_3 * l + 0.793_617_785 * m - 0.004_072_046_8 * s,
        a: 1.977_998_495_1 * l - 2.428_592_205 * m + 0.450_593_709_9 * s,
        b: 0.025_904_037_1 * l + 0.782_771_766_2 * m - 0.808_675_766 * s,
    }
}

/// Linear RGB of an OKLab colour. Channels leave `[0, 1]` when the colour is outside the sRGB
/// gamut.
pub(crate) fn oklab_to_linear(lab: Oklab) -> [f64; 3] {
    let l = lab.l + 0.396_337_777_4 * lab.a + 0.215_803_757_3 * lab.b;
    let m = lab.l - 0.105_561_345_8 * lab.a - 0.063_854_172_8 * lab.b;
    let s = lab.l - 0.089_484_177_5 * lab.a - 1.291_485_548 * lab.b;
    let (l, m, s) = (l * l * l, m * m * m, s * s * s);
    [
        4.076_741_662_1 * l - 3.307_711_591_3 * m + 0.230_969_929_2 * s,
        -1.268_438_004_6 * l + 2.609_757_401_1 * m - 0.341_319_396_5 * s,
        -0.004_196_086_3 * l - 0.703_418_614_7 * m + 1.707_614_701 * s,
    ]
}

pub(crate) fn rgb8_to_oklab(rgb: Rgb8) -> Oklab {
    linear_to_oklab(rgb8_to_linear(rgb))
}

/// Nearest displayable sRGB colour: when the colour is outside the gamut its chroma is reduced
/// step by step, which keeps hue and lightness instead of distorting them by clipping.
pub(crate) fn oklab_to_rgb8(lab: Oklab) -> Rgb8 {
    /// Channels this close to the gamut boundary round to the same 8-bit value anyway.
    const SLACK: f64 = 0.001;
    let l = det::clamp(lab.l, 0.0, 1.0);
    let mut chroma = 1.0;
    for _ in 0..48 {
        let linear = oklab_to_linear(Oklab {
            l,
            a: lab.a * chroma,
            b: lab.b * chroma,
        });
        if linear.iter().all(|c| (-SLACK..=1.0 + SLACK).contains(c)) {
            return linear_to_rgb8(linear);
        }
        chroma *= 0.9;
    }
    linear_to_rgb8(oklab_to_linear(Oklab { l, a: 0.0, b: 0.0 }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn srgb_transfer_round_trips_every_channel_value() {
        let mut previous = -1.0;
        for channel in 0..=255_u8 {
            let linear = srgb_to_linear(channel);
            assert!(linear > previous, "not increasing at {channel}");
            assert!((0.0..=1.0).contains(&linear));
            assert_eq!(linear_to_srgb(linear), channel);
            previous = linear;
        }
        assert_eq!(srgb_to_linear(0), 0.0);
        assert_eq!(srgb_to_linear(255), 1.0);
        // Middle grey of the display is about 21 % linear light.
        assert!((srgb_to_linear(128) - 0.2158).abs() < 1.0e-3);
    }

    #[test]
    fn encoding_clamps_bad_input() {
        assert_eq!(linear_to_srgb(-3.0), 0);
        assert_eq!(linear_to_srgb(7.0), 255);
        assert_eq!(linear_to_srgb(f64::NAN), 0);
        assert_eq!(linear_to_srgb(f64::INFINITY), 255);
    }

    #[test]
    fn oklab_matches_reference_values() {
        // White is L = 1 without chroma, black is the origin.
        let white = rgb8_to_oklab([255, 255, 255]);
        assert!((white.l - 1.0).abs() < 1.0e-6 && white.a.abs() < 1.0e-6 && white.b.abs() < 1.0e-6);
        let black = rgb8_to_oklab([0, 0, 0]);
        assert_eq!((black.l, black.a, black.b), (0.0, 0.0, 0.0));
        // Published values for the sRGB primaries (Ottosson's reference implementation).
        let red = rgb8_to_oklab([255, 0, 0]);
        assert!((red.l - 0.627_955).abs() < 1.0e-4, "{red:?}");
        assert!((red.a - 0.224_863).abs() < 1.0e-4 && (red.b - 0.125_846).abs() < 1.0e-4);
        let blue = rgb8_to_oklab([0, 0, 255]);
        assert!((blue.l - 0.452_014).abs() < 1.0e-4, "{blue:?}");
        assert!((blue.a + 0.032_457).abs() < 1.0e-4 && (blue.b + 0.311_528).abs() < 1.0e-4);
    }

    #[test]
    fn oklab_round_trips_the_colour_cube() {
        for r in (0..=255).step_by(15) {
            for g in (0..=255).step_by(15) {
                for b in (0..=255).step_by(15) {
                    let rgb = [r as u8, g as u8, b as u8];
                    assert_eq!(oklab_to_rgb8(rgb8_to_oklab(rgb)), rgb);
                }
            }
        }
    }

    #[test]
    fn out_of_gamut_colours_keep_their_lightness() {
        // A saturated yellow far outside the gamut comes back as a real colour of about the
        // same lightness instead of a clipped one.
        let wild = Oklab {
            l: 0.8,
            a: 0.1,
            b: 0.6,
        };
        let mapped = rgb8_to_oklab(oklab_to_rgb8(wild));
        assert!((mapped.l - 0.8).abs() < 0.02, "{mapped:?}");
        assert!(mapped.b > 0.1, "{mapped:?}");
        // Lightness itself is clamped.
        assert_eq!(
            oklab_to_rgb8(Oklab {
                l: 4.0,
                a: 0.0,
                b: 0.0
            }),
            [255, 255, 255]
        );
    }
}
