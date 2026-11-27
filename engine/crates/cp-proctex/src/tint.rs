//! The tint convention: which texels take a biome tint.

use cp_core::det;
use cp_core::math::smoothstep;

use crate::color::{Rgb8, linear_to_srgb, rgb8_to_linear, srgb_to_linear};

/// Texels whose linear saturation is at most this take the whole tint.
const FULL_TINT_BELOW: f32 = 0.45;
/// Texels whose linear saturation is at least this take none of it.
const NO_TINT_ABOVE: f32 = 0.6;

/// How much of a biome tint a texel takes, judged by its own colour: 1 for near-neutral
/// texels, 0 for coloured ones.
///
/// Textures of a tint class ([`crate::Tint`]) are painted in near-neutral tones and get their
/// colour by `albedo * mix(1, tint, weight)` in linear RGB. For most of them the weight is 1
/// everywhere. The exception is `grass_side`, whose soil must keep its own colour: its grass
/// fringe is near-neutral and its soil is clearly coloured, so this weight separates the two
/// without a mask channel.
///
/// The weight is `1 - smoothstep(0.45, 0.6, s)` of the saturation `s = (max - min) / max`
/// of the **linear** RGB colour (as a shader samples it from an sRGB texture); black counts
/// as neutral. Every texel of the tinted textures is fully on one side. The recipes see to
/// that for the neutral parts: whenever the colour parameter of a tinted part is itself
/// neutral enough for the whole tint (weight 1, as all defaults are), so is every tone
/// painted from it. The default soil is far on the other side. A caller who replaces the soil
/// of `grass_side` with a greyish colour should bake the grass colour in instead (parameter
/// `grass`) and not tint the texture.
///
/// ```
/// use cp_proctex::tint_weight;
///
/// assert_eq!(tint_weight([0.6, 0.62, 0.58]), 1.0); // pale grey: tinted
/// assert_eq!(tint_weight([0.24, 0.12, 0.05]), 0.0); // brown soil: keeps its colour
/// ```
#[must_use]
pub fn tint_weight(linear_rgb: [f32; 3]) -> f32 {
    let [r, g, b] = linear_rgb;
    let (max, min) = (det::max(det::max(r, g), b), det::min(det::min(r, g), b));
    if max <= 0.0 {
        return 1.0;
    }
    1.0 - smoothstep(FULL_TINT_BELOW, NO_TINT_ABOVE, (max - min) / max)
}

/// True when a colour is neutral enough to take a biome tint completely.
pub(crate) fn takes_full_tint(srgb: Rgb8) -> bool {
    tint_weight(rgb8_to_linear(srgb).map(|c| c as f32)) >= 1.0
}

/// The albedo texel as it looks once `tint` (sRGB) has been multiplied in.
pub(crate) fn apply_tint(albedo: [u8; 4], tint: [u8; 3]) -> [u8; 4] {
    let linear = [albedo[0], albedo[1], albedo[2]].map(srgb_to_linear);
    let weight = f64::from(tint_weight(linear.map(|c| c as f32)));
    let mut tinted = albedo;
    for channel in 0..3 {
        let factor = 1.0 + (srgb_to_linear(tint[channel]) - 1.0) * weight;
        tinted[channel] = linear_to_srgb(linear[channel] * factor);
    }
    tinted
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn neutral_colours_are_tinted_and_saturated_ones_are_not() {
        assert_eq!(tint_weight([0.0, 0.0, 0.0]), 1.0);
        assert_eq!(tint_weight([1.0, 1.0, 1.0]), 1.0);
        assert_eq!(tint_weight([0.5, 0.5, 0.5]), 1.0);
        assert_eq!(tint_weight([0.7, 0.72, 0.5]), 1.0);
        assert_eq!(tint_weight([1.0, 0.0, 0.0]), 0.0);
        assert_eq!(tint_weight([0.2, 0.1, 0.05]), 0.0);
        // In between the weight falls smoothly.
        let mut previous = 1.0;
        for step in 0..=100 {
            let low = 1.0 - step as f32 / 100.0;
            let weight = tint_weight([1.0, 1.0, low]);
            assert!((0.0..=1.0).contains(&weight));
            assert!(weight <= previous);
            previous = weight;
        }
        assert_eq!(previous, 0.0);
        let half = tint_weight([1.0, 1.0, 1.0 - 0.525]);
        assert!((half - 0.5).abs() < 1.0e-3, "{half}");
        assert!(takes_full_tint([200, 204, 198]) && !takes_full_tint([134, 96, 66]));
        // The weight depends on the hue-less saturation only, not on brightness.
        assert_eq!(tint_weight([0.5, 0.5, 0.26]), tint_weight([1.0, 1.0, 0.52]));
    }

    #[test]
    fn tinting_multiplies_in_linear_light() {
        // White takes the tint exactly, black stays black, alpha is untouched.
        assert_eq!(
            apply_tint([255, 255, 255, 200], [126, 188, 84]),
            [126, 188, 84, 200]
        );
        assert_eq!(apply_tint([0, 0, 0, 255], [126, 188, 84]), [0, 0, 0, 255]);
        // A white tint changes nothing.
        assert_eq!(
            apply_tint([90, 140, 200, 255], [255, 255, 255]),
            [90, 140, 200, 255]
        );
        // A coloured texel keeps its colour.
        assert_eq!(
            apply_tint([134, 96, 66, 255], [126, 188, 84]),
            [134, 96, 66, 255]
        );
        // Mid grey: the product of the two linear values.
        let tinted = apply_tint([128, 128, 128, 255], [255, 128, 0]);
        assert_eq!(tinted[0], 128);
        let expected = linear_to_srgb(srgb_to_linear(128) * srgb_to_linear(128));
        assert_eq!(tinted[1], expected);
        assert_eq!(tinted[2], 0);
    }
}
