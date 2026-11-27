//! The metal plate recipe.

use cp_core::det;
use cp_core::math::smoothstep;

use super::color;
use crate::canvas::Canvas;
use crate::color::Rgb8;
use crate::grid::Grid;
use crate::paint::Painting;
use crate::palette::{Shading, ramp, swatches};
use crate::params::Params;
use crate::pattern::{DESIGN, TILE, value_fbm, wrap};

const STEEL: Rgb8 = [164, 170, 180];

/// Centres of the rivet domes in design units: the corner shared by a rivet's lit head and
/// its shadow.
const RIVETS: [(f64, f64); 4] = [(3.0, 3.0), (14.0, 3.0), (3.0, 14.0), (14.0, 14.0)];
/// Squared radius of a rivet dome in design units. At 16 px the dome raises exactly the 2×2
/// texels of head, shadow and the two texels between them; the next texel centres lie on its
/// rim.
const RIVET_RADIUS_SQ: f64 = 2.5;

pub(super) fn metal(canvas: &Canvas, params: &Params) -> Painting {
    let colors = ramp(color(params, "color", STEEL), Shading::new(5, 0.30));
    let brush_seed = canvas.seed(1);
    // Brushed surface: faint horizontal streaks in the three middle tones.
    let brush = canvas.paint(|p| value_fbm(brush_seed, p, (1, 10), 2));
    let body = brush.quantize(&[0.2, 0.58, 0.22]).map(|tone| tone + 1);

    // The plate itself is drawn by hand. Layer values: 0 brushed face, then palette index + 1.
    let last = DESIGN - 1;
    let layer = Grid::from_fn(DESIGN, |x, y| {
        // Rivets sit two texels in from each corner: a lit head with a shadow below right.
        let head = (x == 2 || x == last - 2) && (y == 2 || y == last - 2);
        let shadow = (x == 3 || x == last - 1) && (y == 3 || y == last - 1);
        if (x == 0 && y == last) || (x == last && y == 0) {
            // The two corners where the lit and the shaded edge meet.
            3_u8
        } else if x == 0 || y == 0 {
            5
        } else if x == last || y == last {
            1
        } else if head {
            5
        } else if shadow {
            1
        } else if x == 1 || y == 1 {
            // Inner bevel: one step lighter than the face next to the lit edge.
            4
        } else if x == last - 1 || y == last - 1 {
            2
        } else {
            0
        }
    });
    let layer = canvas.stamp(&layer);
    let index = Grid::from_fn(canvas.size, |x, y| match layer.get(x, y) {
        0 => body.get(x, y),
        tone => tone - 1,
    });

    // The relief is that of a raised plate, not of the tones: the joint on the tile's border
    // is the low point and the bevel rises from it over two units, so the rims on the upper
    // left face the light and those on the lower right face away, as painted. A rivet is a
    // low dome centred between its lit head and its shadow. The streaks barely show.
    let height = canvas.paint(|p| {
        let (x, y) = (wrap(p.x), wrap(p.y));
        let edge = det::min(det::min(x, TILE - x), det::min(y, TILE - y));
        let bevel = smoothstep(0.0, 2.0, edge);
        let rivets: f64 = (RIVETS.iter())
            .map(|&(cx, cy)| {
                let (dx, dy) = (x - cx, y - cy);
                let inside = det::max(1.0 - (dx * dx + dy * dy) / RIVET_RADIUS_SQ, 0.0);
                inside * inside
            })
            .sum();
        0.1 + 0.66 * bevel + 0.2 * rivets
    });
    let height = Grid::from_fn(canvas.size, |x, y| {
        height.get(x, y) + 0.03 * brush.get(x, y)
    });
    Painting {
        palette: swatches(&colors, &[0.5, 0.42, 0.36, 0.3, 0.24]),
        height,
        index,
        holes: None,
        relief: 1.0,
    }
}
