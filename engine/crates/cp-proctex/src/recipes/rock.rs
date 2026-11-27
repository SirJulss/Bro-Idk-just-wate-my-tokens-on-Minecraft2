//! Rock and mineral recipes: stone, cobble, gravel, sandstone, clay, basalt, limestone, slate
//! and ore.

use cp_core::hash::hash_combine;
use cp_core::math::smoothstep;

use super::{bevel, chance, color, heights, lift};
use crate::canvas::Canvas;
use crate::color::Rgb8;
use crate::grid::Grid;
use crate::paint::Painting;
use crate::palette::{Shading, ramp, swatches};
use crate::params::Params;
use crate::pattern::{DESIGN, cells, fbm, honeycomb, scatter, unit, value, value_fbm, wrap};

const STONE: Rgb8 = [139, 134, 130];
const COBBLE: Rgb8 = [134, 129, 126];
const GRAVEL: Rgb8 = [140, 131, 124];
const SANDSTONE: Rgb8 = [214, 186, 134];
const CLAY: Rgb8 = [158, 164, 178];
const BASALT: Rgb8 = [78, 78, 90];
const LIMESTONE: Rgb8 = [206, 198, 176];
const SLATE: Rgb8 = [92, 102, 116];
const IRON_ORE: Rgb8 = [204, 186, 156];

/// Tones of the `stone` pattern: 0 deep shadow, 1 shadow, 2 body, 3 light, 4 highlight.
///
/// Light and dark patches come from two independent fields, both stretched sideways: flat,
/// overlapping blotches like the bedding of real rock.
fn stone_tones(canvas: &Canvas) -> Grid<u8> {
    let (light_seed, dark_seed) = (canvas.seed(1), canvas.seed(2));
    let light = canvas.paint(|p| value_fbm(light_seed, p, (4, 7), 2));
    let dark = canvas.paint(|p| value_fbm(dark_seed, p, (5, 8), 2));
    let (lit, bright) = (light.threshold_above(0.27), light.threshold_above(0.06));
    let shade = dark.threshold_above(0.24);
    let body = Grid::from_fn(canvas.size, |x, y| {
        let (l, d) = (light.get(x, y), dark.get(x, y));
        if l >= bright {
            4_u8
        } else if l >= lit {
            3
        } else if d >= shade {
            1
        } else {
            2
        }
    })
    .despeckled(|_| false);
    // Chips: short horizontal nicks, dark inside (1) with a lit lower lip (2).
    let mut layer = Grid::filled(DESIGN, 0_u8);
    let mut rng = canvas.rng(3);
    for (x, y) in scatter(&mut rng, 4, 5.0) {
        for i in 0..2 + i64::from(rng.range_u32(0, 2)) {
            layer.set_wrapped(x + i, y, 1);
            layer.set_wrapped(x + i, y + 1, 2);
        }
    }
    let layer = canvas.stamp(&layer);
    Grid::from_fn(canvas.size, |x, y| match layer.get(x, y) {
        1 => 0,
        2 => body.get(x, y).max(3),
        _ => body.get(x, y),
    })
}

pub(super) fn stone(canvas: &Canvas, params: &Params) -> Painting {
    let colors = ramp(color(params, "color", STONE), Shading::new(5, 0.30));
    let index = stone_tones(canvas);
    Painting {
        palette: swatches(&colors, &[0.94, 0.9, 0.86, 0.82, 0.78]),
        height: heights(&index, &[0.1, 0.38, 0.52, 0.66, 0.8]),
        index,
        holes: None,
        relief: 1.6,
    }
}

pub(super) fn cobble(canvas: &Canvas, params: &Params) -> Painting {
    let colors = ramp(
        color(params, "color", COBBLE),
        Shading::new(5, 0.38).pivot(0.6),
    );
    let (cell_seed, dent_seed) = (canvas.seed(1), canvas.seed(2));
    // Tones: 0 gap, 1 shaded rim, 2 and 3 the two stone tones, 4 lit rim.
    let (index, height) = canvas
        .paint(|p| {
            let cell = cells(cell_seed, p, (3, 3), 0.85);
            if cell.edge < 0.5 {
                return (0_u8, 0.0);
            }
            let base = if chance(cell.hash, 7, 0.55) { 2 } else { 3 };
            let dent = cell.edge > 1.6 && fbm(dent_seed, p, 6, 1) > 0.42;
            let tone = (base + bevel(&cell, 1.45) - i32::from(dent)).clamp(1, 4);
            let dome = smoothstep(0.5, 2.6, cell.edge);
            let tall = 0.75 + 0.25 * unit(hash_combine(cell.hash, 8));
            (tone as u8, 0.35 + 0.65 * dome * tall)
        })
        .unzip();
    Painting {
        palette: swatches(&colors, &[0.96, 0.9, 0.88, 0.85, 0.8]),
        index,
        height,
        holes: None,
        relief: 2.2,
    }
}

pub(super) fn gravel(canvas: &Canvas, params: &Params) -> Painting {
    let colors = ramp(
        color(params, "color", GRAVEL),
        Shading::new(5, 0.31).shift(1.2),
    );
    let cell_seed = canvas.seed(1);
    // Every pebble has its own tone; only the deepest gaps between them go dark.
    let (index, height) = canvas
        .paint(|p| {
            let cell = cells(cell_seed, p, (5, 5), 1.0);
            if cell.edge < 0.18 {
                return (0_u8, 0.0);
            }
            let base =
                [1, 2, 2, 2, 3, 3, 3, 4][(unit(hash_combine(cell.hash, 3)) * 8.0) as usize % 8];
            let tone = (base + bevel(&cell, 0.7)).clamp(1, 4);
            let dome = smoothstep(0.2, 1.4, cell.edge);
            (
                tone as u8,
                0.3 + dome * (0.4 + 0.3 * unit(hash_combine(cell.hash, 4))),
            )
        })
        .unzip();
    Painting {
        palette: swatches(&colors, &[0.96, 0.92, 0.9, 0.88, 0.84]),
        index,
        height,
        holes: None,
        relief: 1.6,
    }
}

pub(super) fn sandstone(canvas: &Canvas, params: &Params) -> Painting {
    let colors = ramp(
        color(params, "color", SANDSTONE),
        Shading::new(5, 0.26).pivot(0.6),
    );
    // Strata: bands two to four rows thick, each in one of three tones; about half of them
    // end in a dark bedding line. The stack starts at a random row so that no band boundary
    // is tied to the tile's edge.
    let mut rng = canvas.rng(1);
    let mut rows = [(2_u8, false); DESIGN];
    let start = rng.range_u32(0, DESIGN as u32) as usize;
    let (mut row, mut tone) = (0, 2_u8);
    while row < DESIGN {
        let thickness = rng.range_u32(2, 5) as usize;
        // One of the two other tones, so neighbouring bands always differ.
        tone = 1 + (tone + rng.range_u32(0, 2) as u8) % 3;
        let bedding = rng.range_u32(0, 2) == 0;
        for i in 0..thickness.min(DESIGN - row) {
            let last = i + 1 == thickness;
            rows[(start + row + i) % DESIGN] = (tone, bedding && last);
        }
        row += thickness;
    }

    let (wave_seed, ripple_seed, mottle_seed) = (canvas.seed(2), canvas.seed(3), canvas.seed(4));
    let index = canvas.paint(|p| {
        // The strata undulate by about a row.
        let wobble = 1.1 * value(wave_seed, p, (2, 1)) + 0.5 * value(ripple_seed, p, (4, 2));
        let (tone, bedding) = rows[wrap(wrap(p.y) + wobble) as usize % DESIGN];
        if bedding {
            0
        } else {
            tone + u8::from(fbm(mottle_seed, p, 6, 1) > 0.32)
        }
    });
    Painting {
        palette: swatches(&colors, &[0.9, 0.88, 0.86, 0.84, 0.82]),
        height: heights(&index, &[0.1, 0.45, 0.55, 0.65, 0.75]),
        index,
        holes: None,
        relief: 0.9,
    }
}

pub(super) fn clay(canvas: &Canvas, params: &Params) -> Painting {
    let colors = ramp(
        color(params, "color", CLAY),
        Shading::new(4, 0.2).pivot(0.6),
    );
    let smear_seed = canvas.seed(1);
    // Wide, flat smears: 0 pit, 1 shadow, 2 body, 3 light.
    let smear = canvas.paint(|p| value_fbm(smear_seed, p, (2, 5), 2));
    let body = smear
        .quantize(&[0.24, 0.48, 0.28])
        .map(|tone| tone + 1)
        .despeckled(|_| false);
    let mut pits = Grid::filled(DESIGN, false);
    for (x, y) in scatter(&mut canvas.rng(2), 4, 5.0) {
        pits.set_wrapped(x, y, true);
    }
    let pits = canvas.stamp(&pits);
    let index = Grid::from_fn(
        canvas.size,
        |x, y| {
            if pits.get(x, y) { 0 } else { body.get(x, y) }
        },
    );
    Painting {
        palette: swatches(&colors, &[0.75, 0.68, 0.62, 0.55]),
        height: heights(&index, &[0.15, 0.45, 0.55, 0.65]),
        index,
        holes: None,
        relief: 0.6,
    }
}

pub(super) fn basalt(canvas: &Canvas, params: &Params) -> Painting {
    let colors = ramp(
        color(params, "color", BASALT),
        Shading::new(5, 0.22).pivot(0.45).shift(0.6),
    );
    let cell_seed = canvas.seed(1);
    // Columns seen end-on, the mark of basalt: a honeycomb of flat faces, each in one tone,
    // with only the edge towards the light picked out. 0 joint, 1..=3 faces, 4 lit edge of
    // the lightest faces.
    let (faces, height) = canvas
        .paint(|p| {
            let cell = honeycomb(cell_seed, p, (3, 4), 0.3);
            if cell.edge < 0.5 {
                return (0_u8, 0.0);
            }
            let base = [1, 2, 2, 3][(unit(hash_combine(cell.hash, 5)) * 4.0) as usize % 4];
            let lit = i32::from(bevel(&cell, 1.15) > 0);
            (
                (base + lit) as u8,
                0.6 + 0.08 * f64::from(base) + 0.08 * f64::from(lit),
            )
        })
        .unzip();
    // Vesicles: a few gas bubbles, single dark texels.
    let mut pits = Grid::filled(DESIGN, false);
    for (x, y) in scatter(&mut canvas.rng(2), 5, 4.5) {
        pits.set_wrapped(x, y, true);
    }
    let pits = canvas.stamp(&pits);
    let index = Grid::from_fn(canvas.size, |x, y| {
        if pits.get(x, y) { 0 } else { faces.get(x, y) }
    });
    Painting {
        palette: swatches(&colors, &[0.92, 0.86, 0.82, 0.78, 0.7]),
        index,
        height,
        holes: None,
        relief: 1.1,
    }
}

pub(super) fn limestone(canvas: &Canvas, params: &Params) -> Painting {
    let colors = ramp(color(params, "color", LIMESTONE), Shading::new(5, 0.26));
    let (light_seed, dark_seed) = (canvas.seed(1), canvas.seed(2));
    // 0 pit, 1 shadow, 2 body, 3 light, 4 highlight.
    let light = canvas.paint(|p| fbm(light_seed, p, 3, 2));
    let dark = canvas.paint(|p| value_fbm(dark_seed, p, (3, 5), 2));
    let lit = light.threshold_above(0.26);
    let shade = dark.threshold_above(0.28);
    let body = Grid::from_fn(canvas.size, |x, y| {
        if light.get(x, y) >= lit {
            3_u8
        } else if dark.get(x, y) >= shade {
            1
        } else {
            2
        }
    })
    .despeckled(|_| false);
    // Pits: a dark hole whose far wall catches the light. The highlight tone is theirs alone.
    let mut layer = Grid::filled(DESIGN, 0_u8);
    for (x, y) in scatter(&mut canvas.rng(3), 7, 3.4) {
        layer.set_wrapped(x, y, 1);
        layer.set_wrapped(x + 1, y + 1, 2);
    }
    let layer = canvas.stamp(&layer);
    let index = Grid::from_fn(canvas.size, |x, y| match layer.get(x, y) {
        1 => 0,
        2 => 4,
        _ => body.get(x, y),
    });
    Painting {
        palette: swatches(&colors, &[0.92, 0.86, 0.82, 0.8, 0.76]),
        height: heights(&index, &[0.0, 0.42, 0.55, 0.66, 0.76]),
        index,
        holes: None,
        relief: 1.4,
    }
}

pub(super) fn slate(canvas: &Canvas, params: &Params) -> Painting {
    let colors = ramp(color(params, "color", SLATE), Shading::new(5, 0.32));
    let streak_seed = canvas.seed(1);
    // Long horizontal streaks: 1 shadow, 2 body, 3 light.
    let streak = canvas.paint(|p| value_fbm(streak_seed, p, (2, 8), 2));
    let body = streak.quantize(&[0.28, 0.44, 0.28]).map(|tone| tone + 1);
    // Ledges where a sheet has split off: a lit edge (4) over a line of shadow (0).
    let mut layer = Grid::filled(DESIGN, 0_u8);
    let mut rng = canvas.rng(2);
    for (x, y) in scatter(&mut rng, 5, 4.0) {
        let length = 4 + i64::from(rng.range_u32(0, 6));
        let step_at = 2 + i64::from(rng.range_u32(0, 4));
        for i in 0..length {
            // One step down along the way keeps the ledge from looking ruled.
            let row = y + i64::from(i >= step_at && length > 6);
            layer.set_wrapped(x + i, row, 2);
            layer.set_wrapped(x + i, row + 1, 1);
        }
    }
    let layer = canvas.stamp(&layer);
    let index = Grid::from_fn(canvas.size, |x, y| match layer.get(x, y) {
        2 => 4,
        1 => 0,
        _ => body.get(x, y),
    });
    // A ledge is the lip of the split sheet: its lit edge faces the light, its shadow faces
    // away.
    let height = lift(&heights(&body, &[0.4, 0.4, 0.5, 0.6]), |x, y| {
        layer.get(x, y) > 0
    });
    Painting {
        palette: swatches(&colors, &[0.8, 0.7, 0.64, 0.58, 0.5]),
        height,
        index,
        holes: None,
        relief: 1.2,
    }
}

/// Shapes of ore pieces; `#` is ore.
const NUGGETS: [&[&str]; 6] = [
    &[".##", "###"],
    &["##.", "###", ".#."],
    &[".#.", "###", "##."],
    &["###", ".##"],
    &["##", "##", ".#"],
    &[".##", "###", ".##"],
];
const LUMPS: [&[&str]; 4] = [
    &[".##.", "####", "####", ".##."],
    &["###.", "####", ".###"],
    &[".###", "####", "###."],
    &["##.", "###", "###", ".##"],
];
const FLECKS: [&[&str]; 4] = [&["#"], &["##"], &["#", "#"], &["#.", ".#"]];
/// Faceted crystals: a cross, two rhombs and a pyramid. Their rims catch the light on two
/// sides, so each one shows a lit and a shaded facet.
const CRYSTALS: [&[&str]; 4] = [
    &[".#.", "###", ".#."],
    &[".#", "##", "#."],
    &["#.", "##", ".#"],
    &[".#.", "###"],
];

/// Roughness of an accent: patina and rust are dull crusts.
const ACCENT_ROUGHNESS: f64 = 0.8;

pub(super) fn ore(canvas: &Canvas, params: &Params) -> Painting {
    let rock = ramp(color(params, "rock", STONE), Shading::new(5, 0.30));
    let nugget = ramp(
        color(params, "ore", IRON_ORE),
        Shading::new(3, 0.30).pivot(0.55),
    );
    // Every style draws its layout from a stream of its own: two ores of the same seed do not
    // put their pieces in the same places.
    let (shapes, count, spacing, roughness, stream): (&[&[&str]], usize, f64, [f64; 3], u64) =
        match params.variant("style") {
            Some("lumps") => (&LUMPS, 4, 5.6, [0.85, 0.78, 0.7], 2),
            Some("flecks") => (&FLECKS, 10, 3.2, [0.5, 0.42, 0.34], 3),
            Some("crystals") => (&CRYSTALS, 6, 4.4, [0.44, 0.34, 0.24], 4),
            _ => (&NUGGETS, 5, 4.8, [0.5, 0.42, 0.34], 1),
        };

    // Occupancy of the design grid by ore.
    let mut occupied = Grid::filled(DESIGN, false);
    let mut rng = canvas.rng(stream);
    for (x, y) in scatter(&mut rng, count, spacing) {
        let shape = shapes[rng.range_u32(0, shapes.len() as u32) as usize];
        for (dy, row) in shape.iter().enumerate() {
            for (dx, cell) in row.bytes().enumerate() {
                if cell == b'#' {
                    occupied.set_wrapped(x + dx as i64, y + dy as i64, true);
                }
            }
        }
    }
    // 0 rock, 1 rock in the shadow beside a piece, 5 rock right below a piece, 2 shaded ore,
    // 3 ore, 4 lit ore.
    let layer = Grid::from_fn(DESIGN, |x, y| {
        let has = |dx: i64, dy: i64| occupied.near(x, y, dx, dy);
        if !has(0, 0) {
            if has(0, -1) { 5 } else { u8::from(has(-1, 0)) }
        } else if !has(-1, 0) && !has(0, -1) && (has(1, 0) || has(0, 1)) {
            4
        } else if !has(1, 0) && !has(0, 1) && (has(-1, 0) || has(0, -1)) {
            2
        } else {
            3
        }
    });
    let layer = canvas.stamp(&layer);

    // The host rock is the stone texture of the same seed, reduced to three tones. Palette:
    // 0..=2 rock, 3 the side of the pieces facing away from the light, 4 ore, 5 lit ore.
    let host = stone_tones(&canvas.of_recipe("stone"));
    let shapes = Grid::from_fn(canvas.size, |x, y| match layer.get(x, y) {
        0 => [0, 0, 1, 2, 2][usize::from(host.get(x, y)).min(4)],
        1 | 5 => 0,
        piece => piece + 1,
    });
    // The relief follows the shapes, whatever colours they are painted in.
    let height = heights(&shapes, &[0.3, 0.45, 0.58, 0.7, 0.85, 1.0]);
    let [shaded, plain, lit] = roughness;
    // An accent takes over the shaded rims of the pieces and the rock right below them, where
    // weathering leaves its stain: green patina on copper, rust on iron.
    let accent = params.color("accent");
    let index = match accent {
        Some(_) => Grid::from_fn(canvas.size, |x, y| {
            if layer.get(x, y) == 5 {
                3
            } else {
                shapes.get(x, y)
            }
        }),
        None => shapes,
    };
    let (away, away_roughness) = match accent {
        Some(accent) => (ramp(accent, Shading::new(3, 0.30))[1], ACCENT_ROUGHNESS),
        None => (nugget[0], shaded),
    };
    let colors = [rock[1], rock[2], rock[3], away, nugget[1], nugget[2]];
    Painting {
        palette: swatches(&colors, &[0.9, 0.86, 0.82, away_roughness, plain, lit]),
        height,
        index,
        holes: None,
        relief: 1.3,
    }
}
