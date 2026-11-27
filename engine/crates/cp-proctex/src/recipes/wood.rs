//! Wood recipes: bark, the cut end of a log and planks.

use cp_core::hash::hash_combine;

use super::{color, heights, lift};
use crate::canvas::Canvas;
use crate::color::Rgb8;
use crate::grid::Grid;
use crate::paint::Painting;
use crate::palette::{Shading, ramp, swatches};
use crate::params::Params;
use crate::pattern::{DESIGN, fbm, scatter, value, value_fbm, wrap};

const OAK_BARK: Rgb8 = [112, 88, 60];
const OAK_WOOD: Rgb8 = [186, 150, 98];
/// Default colour of the marks on pale (`birch`) bark.
const DARK_MARKS: Rgb8 = [66, 60, 58];

pub(super) fn log_side(canvas: &Canvas, params: &Params) -> Painting {
    let bark = color(params, "bark", OAK_BARK);
    match params.variant("style") {
        Some("birch") => birch_bark(canvas, bark, color(params, "marks", DARK_MARKS)),
        Some("scaly") => scaly_bark(canvas, bark),
        Some("smooth") => smooth_bark(canvas, bark, params.color("marks")),
        _ => furrowed_bark(canvas, bark),
    }
}

/// Ridges running up the trunk, lit from the left, with `periods` lattice cells per tile
/// across and along the trunk. Returns the tones (0 furrow, 1 shaded flank, 2 body, 3 lit
/// flank, 4 highlight, in the given shares) and the height of the ridges.
fn ridges(canvas: &Canvas, periods: (u32, u32), shares: &[f64; 5]) -> (Grid<u8>, Grid<f64>) {
    let ridge_seed = canvas.seed(1);
    let ridge = canvas
        .paint(|p| value_fbm(ridge_seed, p, periods, 2))
        .normalized();
    let density = canvas.density();
    let lit = Grid::from_fn(canvas.size, |x, y| {
        let here = ridge.get(x, y);
        0.7 * here + 1.1 * (here - ridge.near(x, y, -1, 0)) * density
    });
    (lit.quantize(shares), ridge)
}

/// Oak: long vertical ridges with dark furrows between them.
fn furrowed_bark(canvas: &Canvas, bark: Rgb8) -> Painting {
    let colors = ramp(bark, Shading::new(5, 0.32).pivot(0.55));
    let (index, height) = ridges(canvas, (6, 2), &[0.14, 0.26, 0.32, 0.20, 0.08]);
    Painting {
        palette: swatches(&colors, &[0.98, 0.95, 0.93, 0.9, 0.88]),
        index,
        height,
        holes: None,
        relief: 2.0,
    }
}

/// Birch: pale, slightly streaked bark with dark horizontal dashes (lenticels).
fn birch_bark(canvas: &Canvas, bark: Rgb8, marks: Rgb8) -> Painting {
    let pale = ramp(bark, Shading::new(3, 0.13).pivot(0.6));
    let dark = ramp(marks, Shading::new(2, 0.16).pivot(0.2));
    let streak_seed = canvas.seed(1);
    // Palette: 0 mark, 1 soft mark, 2..=4 the bark from shadow to light.
    let streak = canvas.paint(|p| value_fbm(streak_seed, p, (3, 2), 2));
    let body = streak
        .quantize(&[0.24, 0.5, 0.26])
        .map(|tone| tone + 2)
        .despeckled(|_| false);
    // Layer values: 1 mark, 2 soft end of a mark.
    let mut layer = Grid::filled(DESIGN, 0_u8);
    let mut rng = canvas.rng(2);
    for (x, y) in scatter(&mut rng, 9, 3.1) {
        let length = 2 + i64::from(rng.range_u32(0, 4));
        for i in 0..length {
            let soft = length > 2 && (i == 0 || i + 1 == length);
            layer.set_wrapped(x + i, y, if soft { 2 } else { 1 });
        }
        // Now and then a knot: the dash thickens downwards.
        if length >= 4 && rng.range_u32(0, 3) == 0 {
            layer.set_wrapped(x + 1, y + 1, 1);
            layer.set_wrapped(x + 2, y + 1, 2);
        }
    }
    let layer = canvas.stamp(&layer);
    let index = Grid::from_fn(canvas.size, |x, y| match layer.get(x, y) {
        1 => 0,
        2 => 1,
        _ => body.get(x, y),
    });
    let colors = [dark[0], dark[1], pale[0], pale[1], pale[2]];
    Painting {
        palette: swatches(&colors, &[0.95, 0.92, 0.82, 0.78, 0.74]),
        height: heights(&index, &[0.2, 0.35, 0.55, 0.65, 0.75]),
        index,
        holes: None,
        relief: 0.9,
    }
}

/// Spruce: short ridges broken into plates by cracks across the trunk.
fn scaly_bark(canvas: &Canvas, bark: Rgb8) -> Painting {
    let colors = ramp(bark, Shading::new(5, 0.32).pivot(0.5));
    let (ridged, ridge) = ridges(canvas, (5, 3), &[0.12, 0.28, 0.34, 0.20, 0.06]);
    // Cracks: short dark nicks (1), each with the lit upper edge of the plate below it (2).
    let mut layer = Grid::filled(DESIGN, 0_u8);
    let mut rng = canvas.rng(2);
    for (x, y) in scatter(&mut rng, 9, 3.2) {
        for i in 0..2 + i64::from(rng.range_u32(0, 2)) {
            layer.set_wrapped(x + i, y, 1);
            if layer.wrapped(x + i, y + 1) == 0 {
                layer.set_wrapped(x + i, y + 1, 2);
            }
        }
    }
    let layer = canvas.stamp(&layer);
    let index = Grid::from_fn(canvas.size, |x, y| match layer.get(x, y) {
        1 => 0,
        2 => ridged.get(x, y).max(3),
        _ => ridged.get(x, y),
    });
    let height = Grid::from_fn(canvas.size, |x, y| {
        if layer.get(x, y) == 1 {
            0.0
        } else {
            ridge.get(x, y)
        }
    });
    Painting {
        palette: swatches(&colors, &[0.98, 0.95, 0.93, 0.9, 0.88]),
        index,
        height,
        holes: None,
        relief: 2.0,
    }
}

/// Cherry: smooth bark in soft horizontal bands, crossed by short light dashes.
fn smooth_bark(canvas: &Canvas, bark: Rgb8, marks: Option<Rgb8>) -> Painting {
    let mut colors = ramp(bark, Shading::new(5, 0.30).pivot(0.4));
    if let Some(marks) = marks {
        // Through the ramp builder like every caller colour: a colour inside the pack's
        // range stays as it is, black or white are brought back into it.
        colors[4] = ramp(marks, Shading::new(2, 0.1).pivot(1.0))[1];
    }
    let band_seed = canvas.seed(1);
    // 1 shadow, 2 body, 3 light: calm bands that run around the trunk. 4 dashes, 0 the
    // shadow under a dash.
    let band = canvas.paint(|p| value_fbm(band_seed, p, (1, 5), 2));
    let body = band
        .quantize(&[0.22, 0.56, 0.22])
        .map(|tone| tone + 1)
        .despeckled(|_| false);
    let mut layer = Grid::filled(DESIGN, 0_u8);
    let mut rng = canvas.rng(2);
    for (x, y) in scatter(&mut rng, 8, 3.4) {
        let length = 3 + i64::from(rng.range_u32(0, 4));
        let shadowed = rng.range_u32(0, 3) > 0;
        for i in 0..length {
            layer.set_wrapped(x + i, y, 2);
            // The shadow is a little shorter than the dash, like a lip seen from above.
            if shadowed && i > 0 && layer.wrapped(x + i, y + 1) == 0 {
                layer.set_wrapped(x + i, y + 1, 1);
            }
        }
    }
    let layer = canvas.stamp(&layer);
    let index = Grid::from_fn(canvas.size, |x, y| match layer.get(x, y) {
        2 => 4,
        1 => 0,
        _ => body.get(x, y),
    });
    // A dash and its shadow are one lip.
    let height = lift(&heights(&body, &[0.45, 0.45, 0.52, 0.6]), |x, y| {
        layer.get(x, y) > 0
    });
    Painting {
        palette: swatches(&colors, &[0.9, 0.84, 0.8, 0.76, 0.8]),
        height,
        index,
        holes: None,
        relief: 0.9,
    }
}

pub(super) fn log_top(canvas: &Canvas, params: &Params) -> Painting {
    let bark = ramp(
        color(params, "bark", OAK_BARK),
        Shading::new(3, 0.16).pivot(0.8).shift(0.3),
    );
    let wood = ramp(
        color(params, "wood", OAK_WOOD),
        Shading::new(3, 0.2).pivot(0.6),
    );
    let (ring_seed, bark_seed) = (canvas.seed(1), canvas.seed(2));
    // Palette: 0 and 1 bark, 2 ring line, 3 wood, 4 pale wood.
    let index = canvas.paint(|p| {
        // Distance from the centre in the 4-norm: contours are squares with rounded corners,
        // the cross-section of a log that fills its block.
        let (dx, dy) = (wrap(p.x) - 8.0, wrap(p.y) - 8.0);
        let (dx2, dy2) = (dx * dx, dy * dy);
        let radius = (dx2 * dx2 + dy2 * dy2).sqrt().sqrt();
        if radius >= 7.05 {
            return u8::from(fbm(bark_seed, p, 8, 1) > -0.05);
        }
        let ring = radius + 0.45 * fbm(ring_seed, p, 3, 1);
        // Growth rings every two units; the pith in the middle is a ring as well.
        let phase = (ring / 2.0).rem_euclid(1.0);
        if radius < 0.8 || phase < 0.32 {
            2
        } else if radius >= 6.0 || phase > 0.7 {
            4
        } else {
            3
        }
    });
    let colors = [bark[0], bark[1], wood[0], wood[1], wood[2]];
    Painting {
        palette: swatches(&colors, &[0.96, 0.94, 0.82, 0.78, 0.74]),
        height: heights(&index, &[0.75, 0.9, 0.35, 0.5, 0.55]),
        index,
        holes: None,
        relief: 0.9,
    }
}

pub(super) fn planks(canvas: &Canvas, params: &Params) -> Painting {
    let colors = ramp(
        color(params, "wood", OAK_WOOD),
        Shading::new(5, 0.27).pivot(0.66),
    );
    // Four boards of four rows each. Every board has its own tone and one butt joint; the
    // joints of neighbouring boards are kept apart.
    let mut rng = canvas.rng(1);
    let mut boards = [(0_i64, 0_u8); 4];
    let mut previous = i64::from(rng.range_u32(0, DESIGN as u32));
    for board in &mut boards {
        let joint = (previous + 4 + i64::from(rng.range_u32(0, 9))) % DESIGN as i64;
        *board = (joint, 2 + rng.range_u32(0, 2) as u8);
        previous = joint;
    }
    let grain_seed = canvas.seed(2);
    // 0 joints, 1 dark grain, 2 and 3 the boards, 4 lit edges.
    let index = canvas.paint(|p| {
        let (x, y) = (wrap(p.x), wrap(p.y));
        let board = (y / 4.0) as usize % 4;
        let row = y - 4.0 * board as f64;
        let (joint, base) = boards[board];
        // Position along the board, measured from its butt joint.
        let along = (x - joint as f64).rem_euclid(DESIGN as f64);
        if row >= 3.0 || along < 1.0 {
            return 0;
        }
        // The grain runs along the board and does not continue into its neighbours.
        let grain = value(hash_combine(grain_seed, board as u64), p, (3, 8));
        let lit = row < 1.0 || (1.0..2.0).contains(&along);
        if lit && grain > -0.45 {
            (base + 1).min(4)
        } else if grain < -0.5 {
            1
        } else if grain > 0.55 {
            base + 1
        } else {
            base
        }
    });
    Painting {
        palette: swatches(&colors, &[0.92, 0.84, 0.8, 0.76, 0.72]),
        height: heights(&index, &[0.05, 0.6, 0.7, 0.76, 0.85]),
        index,
        holes: None,
        relief: 1.4,
    }
}
