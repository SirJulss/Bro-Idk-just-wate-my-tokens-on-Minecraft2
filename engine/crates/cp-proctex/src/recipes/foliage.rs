//! Foliage: the cutout leaves recipe.

use super::color;
use crate::canvas::Canvas;
use crate::color::Rgb8;
use crate::grid::Grid;
use crate::paint::Painting;
use crate::palette::{Shading, ramp, swatches};
use crate::params::Params;
use crate::pattern::{DESIGN, fbm, scatter, value_fbm};

/// Near-neutral base of everything that takes the foliage tint.
const LEAF: Rgb8 = [198, 203, 194];

pub(super) fn leaves(canvas: &Canvas, params: &Params) -> Painting {
    let needle = params.variant("style") == Some("needle");
    let blossom = params.color("blossom");
    // With blossoms two palette entries go to the flowers and the leaves make do with four.
    let leaf_tones = if blossom.is_some() { 4 } else { 5 };
    let mut colors = ramp(
        color(params, "leaf", LEAF),
        Shading::new(leaf_tones, 0.40).pivot(0.72).tintable(),
    );
    if let Some(blossom) = blossom {
        colors.extend(ramp(blossom, Shading::new(2, 0.1).pivot(0.3)));
    }

    // Depth of the crown: clumps of leaves stand out, the space between them recedes. Round
    // clumps for broadleaf trees, narrow hanging ones for needles.
    let (depth_seed, light_seed) = (canvas.seed(1), canvas.seed(2));
    let depth = canvas
        .paint(|p| {
            if needle {
                value_fbm(depth_seed, p, (6, 3), 2)
            } else {
                fbm(depth_seed, p, 5, 2)
            }
        })
        .normalized();

    // The deepest places are open. Single stray texels are closed again (or opened), so the
    // outline stays in clumps.
    let open_share = if needle { 0.19 } else { 0.24 };
    let open_below = depth.threshold_above(1.0 - open_share);
    let holes = depth
        .map(|d| u8::from(d < open_below))
        .despeckled(|_| false)
        .map(|open| open == 1);

    // Shading: the clumps as relief under the pack's light, plus a little variation so that
    // equal slopes do not all get the same tone.
    let density = canvas.density();
    let variation = canvas.paint(|p| fbm(light_seed, p, 5, 1));
    let lit = Grid::from_fn(canvas.size, |x, y| {
        let here = depth.get(x, y);
        let toward_light = 0.5 * (depth.near(x, y, -1, 0) + depth.near(x, y, 0, -1));
        0.55 * here + 1.7 * (here - toward_light) * density + 0.1 * variation.get(x, y)
    });
    let shares: &[f64] = if leaf_tones == 5 {
        &[0.13, 0.22, 0.31, 0.23, 0.11]
    } else {
        &[0.15, 0.27, 0.35, 0.23]
    };
    let tones = lit.quantize(shares).despeckled(|_| false);
    let top = (leaf_tones - 1) as u8;
    // A gap is lit like a hollow: the leaves on its upper left are its shaded wall, the ones
    // on its lower right catch the light.
    let index = Grid::from_fn(canvas.size, |x, y| {
        let tone = tones.get(x, y);
        if holes.near(x, y, 1, 0) || holes.near(x, y, 0, 1) {
            tone.min(1)
        } else if holes.near(x, y, -1, 0) || holes.near(x, y, 0, -1) {
            tone.max(2).min(top)
        } else {
            tone
        }
    });
    let height = Grid::from_fn(canvas.size, |x, y| {
        if holes.get(x, y) {
            0.0
        } else {
            0.25 + 0.75 * depth.get(x, y)
        }
    });

    // Blossoms: pairs of light texels sitting on the leaves.
    let index = match blossom {
        None => index,
        Some(_) => {
            let mut layer = Grid::filled(DESIGN, 0_u8);
            let mut rng = canvas.rng(4);
            for (x, y) in scatter(&mut rng, 9, 3.4) {
                layer.set_wrapped(x, y, 2);
                let (dx, dy) = if rng.range_u32(0, 2) == 0 {
                    (1, 0)
                } else {
                    (0, 1)
                };
                layer.set_wrapped(x + dx, y + dy, 1);
            }
            let layer = canvas.stamp(&layer);
            Grid::from_fn(canvas.size, |x, y| match layer.get(x, y) {
                0 => index.get(x, y),
                petal => 3 + petal,
            })
        }
    };
    // A gap keeps the darkest leaf tone as its (invisible) colour: filtering across the
    // outline then darkens the rim a little instead of drawing a pale halo around it.
    let index = Grid::from_fn(canvas.size, |x, y| {
        if holes.get(x, y) { 0 } else { index.get(x, y) }
    });

    let roughness: &[f64] = if blossom.is_some() {
        &[0.8, 0.72, 0.66, 0.6, 0.7, 0.66]
    } else {
        &[0.8, 0.74, 0.68, 0.62, 0.58]
    };
    Painting {
        palette: swatches(&colors, roughness),
        index,
        height,
        holes: Some(holes),
        relief: 1.8,
    }
}
