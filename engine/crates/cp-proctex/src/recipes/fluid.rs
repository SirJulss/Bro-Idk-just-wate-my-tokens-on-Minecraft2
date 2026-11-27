//! Fluid recipes: water and lava.

use super::{color, heights, soften};
use crate::canvas::Canvas;
use crate::color::Rgb8;
use crate::grid::Grid;
use crate::paint::Painting;
use crate::palette::{Shading, Swatch, blend, ramp, swatches};
use crate::params::Params;
use crate::pattern::{DESIGN, fbm, ridged, scatter, value_fbm};

/// Near-neutral base of everything that takes the water tint.
const WATER: Rgb8 = [206, 212, 216];
/// The default lava ramp runs from cooled crust over the flowing melt to its hottest spots.
const LAVA_CRUST: Rgb8 = [88, 32, 30];
const LAVA_FLOW: Rgb8 = [232, 104, 34];
const LAVA_GLOW: Rgb8 = [255, 204, 96];

pub(super) fn water(canvas: &Canvas, params: &Params) -> Painting {
    let colors = ramp(
        color(params, "color", WATER),
        Shading::new(4, 0.2).pivot(0.6).tintable(),
    );
    let swell_seed = canvas.seed(1);
    // A slow swell in three calm tones: 0 trough, 1 surface, 2 crest.
    let swell = canvas.paint(|p| value_fbm(swell_seed, p, (3, 5), 2));
    let body = swell.quantize(&[0.2, 0.5, 0.3]).despeckled(|_| false);
    // Glints (3): short bright dashes, some with a shorter echo below. Dashes instead of
    // lines across the whole tile, which would repeat as stripes.
    let mut glints = Grid::filled(DESIGN, false);
    let mut rng = canvas.rng(2);
    for (x, y) in scatter(&mut rng, 6, 4.2) {
        let length = 3 + i64::from(rng.range_u32(0, 3));
        for i in 0..length {
            glints.set_wrapped(x + i, y, true);
        }
        if rng.range_u32(0, 2) == 0 {
            for i in 2..length {
                glints.set_wrapped(x + i + 1, y + 1, true);
            }
        }
    }
    let glints = canvas.stamp(&glints);
    let index = Grid::from_fn(canvas.size, |x, y| {
        if glints.get(x, y) { 3 } else { body.get(x, y) }
    });
    Painting {
        palette: swatches(&colors, &[0.1, 0.08, 0.07, 0.05]),
        height: soften(&swell.normalized()),
        index,
        holes: None,
        relief: 0.5,
    }
}

pub(super) fn lava(canvas: &Canvas, params: &Params) -> Painting {
    // The default runs from dark red crust to yellow melt; a caller's colour replaces it by
    // a ramp around that colour with a strong warm shift.
    let colors = match params.color("color") {
        Some(glow) => ramp(glow, Shading::new(5, 0.5).pivot(0.6).shift(2.0)),
        None => {
            // Two legs, so that the middle of the ramp is a saturated orange and not the
            // brown a straight line from dark red to yellow passes through.
            let mut colors = blend(LAVA_CRUST, LAVA_FLOW, 3);
            colors.extend(blend(LAVA_FLOW, LAVA_GLOW, 3).into_iter().skip(1));
            colors
        }
    };
    let (flow_seed, vein_seed) = (canvas.seed(1), canvas.seed(2));
    // Heat: broad flows with bright veins where the crust has torn.
    let heat = canvas.paint(|p| 0.55 * fbm(flow_seed, p, 4, 2) + 0.45 * ridged(vein_seed, p, 4, 2));
    let index = heat
        .quantize(&[0.15, 0.2, 0.35, 0.2, 0.1])
        .despeckled(|_| false);
    let glow = [0.08, 0.35, 0.62, 0.85, 1.0];
    let rough = [0.92, 0.8, 0.6, 0.45, 0.35];
    let palette = swatches(&colors, &rough)
        .into_iter()
        .zip(glow)
        .map(|(swatch, emissive)| Swatch { emissive, ..swatch })
        .collect();
    // The crust floats on the melt: dark is high, bright is low.
    let height = heights(&index, &[0.9, 0.7, 0.45, 0.3, 0.2]);
    Painting {
        palette,
        index,
        height,
        holes: None,
        relief: 1.2,
    }
}
