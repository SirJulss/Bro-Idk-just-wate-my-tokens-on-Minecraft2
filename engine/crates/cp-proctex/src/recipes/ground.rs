//! Ground cover recipes: dirt, grass, sand, snow and ice.

use core::f64::consts::TAU;

use cp_core::det;
use cp_core::math::DVec2;

use super::{color, heights, lift};
use crate::canvas::Canvas;
use crate::color::Rgb8;
use crate::grid::Grid;
use crate::paint::Painting;
use crate::palette::{Shading, ramp, swatches};
use crate::params::Params;
use crate::pattern::{DESIGN, TILE, fbm, scatter, value, value_fbm, wrap};

const DIRT: Rgb8 = [148, 108, 74];
/// Near-neutral base of everything that takes the grass tint.
const GRASS: Rgb8 = [204, 207, 200];
const SAND: Rgb8 = [226, 208, 158];
const SNOW: Rgb8 = [230, 241, 250];
const ICE: Rgb8 = [158, 204, 232];

fn dirt_colors(base: Rgb8) -> Vec<Rgb8> {
    ramp(base, Shading::new(5, 0.30))
}

/// The grass ramp. Its base is near-neutral and light, so that the biome tint multiplied into
/// it lands close to the tint itself; the cool-to-warm shift of the ramp survives the
/// multiplication as blue-green shadows and yellow-green highlights.
fn grass_colors(params: &Params) -> Vec<Rgb8> {
    ramp(
        color(params, "grass", GRASS),
        Shading::new(5, 0.36).pivot(0.7).tintable(),
    )
}

/// Tones of the `dirt` pattern (0 pit, 1 shadow, 2 body, 3 light, 4 pebble) and its relief.
fn dirt_tones(canvas: &Canvas) -> (Grid<u8>, Grid<f64>) {
    let (dark_seed, light_seed) = (canvas.seed(1), canvas.seed(2));
    let dark = canvas.paint(|p| fbm(dark_seed, p, 5, 2));
    let light = canvas.paint(|p| fbm(light_seed, p, 4, 2));
    let (shade, pit) = (dark.threshold_above(0.30), dark.threshold_above(0.07));
    let lit = light.threshold_above(0.26);
    let body = Grid::from_fn(canvas.size, |x, y| {
        let (d, l) = (dark.get(x, y), light.get(x, y));
        if d >= pit {
            0_u8
        } else if d >= shade {
            1
        } else if l >= lit {
            3
        } else {
            2
        }
    })
    .despeckled(|_| false);
    // Pebbles: single light texels, each with its shadow underneath.
    let mut layer = Grid::filled(DESIGN, 0_u8);
    for (x, y) in scatter(&mut canvas.rng(3), 5, 4.0) {
        layer.set_wrapped(x, y, 2);
        layer.set_wrapped(x, y + 1, 1);
    }
    let layer = canvas.stamp(&layer);
    let index = Grid::from_fn(canvas.size, |x, y| match layer.get(x, y) {
        2 => 4,
        1 => body.get(x, y).min(1),
        _ => body.get(x, y),
    });
    // A pebble and its shadow are one lip.
    let height = lift(&heights(&body, &[0.1, 0.35, 0.5, 0.62]), |x, y| {
        layer.get(x, y) > 0
    });
    (index, height)
}

pub(super) fn dirt(canvas: &Canvas, params: &Params) -> Painting {
    let colors = dirt_colors(color(params, "color", DIRT));
    let (index, height) = dirt_tones(canvas);
    Painting {
        palette: swatches(&colors, &[0.98, 0.96, 0.94, 0.92, 0.85]),
        height,
        index,
        holes: None,
        relief: 1.4,
    }
}

pub(super) fn grass_top(canvas: &Canvas, params: &Params) -> Painting {
    let colors = grass_colors(params);
    let mottle_seed = canvas.seed(1);
    // Ground between the blades: 1 shadow, 2 body.
    let mottle = canvas.paint(|p| fbm(mottle_seed, p, 4, 2));
    let shade = mottle.threshold_above(0.32);
    let body = mottle
        .map(|m| if m >= shade { 1_u8 } else { 2 })
        .despeckled(|_| false);

    // Blades: short upright strokes, some leaning, with a lit tip and a shadow at the foot.
    // Layer values: 1 shadow, 2 deep shadow, 3 blade, 4 lit tip.
    let mut layer = Grid::filled(DESIGN, 0_u8);
    let mut rng = canvas.rng(2);
    let mut shadows = Vec::new();
    for (x, y) in scatter(&mut rng, 26, 2.3) {
        let length = 2 + i64::from(rng.range_u32(0, 2));
        let lean = i64::from(rng.range_u32(0, 4) == 0);
        let lit_tip = rng.range_u32(0, 100) < 45;
        let deep_foot = rng.range_u32(0, 100) < 35;
        for i in 0..length {
            let tip = i + 1 == length;
            let tone = if tip && lit_tip { 4 } else { 3 };
            layer.set_wrapped(x + if tip { lean } else { 0 }, y - i, tone);
        }
        shadows.push((x + 1, y, 1_u8));
        if deep_foot {
            shadows.push((x, y + 1, 2));
        }
    }
    // Shadows never paint over a blade.
    for (x, y, tone) in shadows {
        if layer.wrapped(x, y) == 0 {
            layer.set_wrapped(x, y, tone);
        }
    }
    let layer = canvas.stamp(&layer);
    let index = Grid::from_fn(canvas.size, |x, y| match layer.get(x, y) {
        0 => body.get(x, y),
        1 => 1,
        2 => 0,
        tone => tone,
    });
    match params.variant("style") {
        Some("flowers") => with_flowers(canvas, &colors, &index, FLOWERS),
        Some("blossoms") => with_flowers(canvas, &colors, &index, BLOSSOMS),
        Some("litter") => with_litter(canvas, &colors, &index),
        Some("straw") => with_straw(canvas, &colors, &index),
        _ => Painting {
            palette: swatches(&colors, &[0.95, 0.92, 0.9, 0.86, 0.82]),
            height: heights(&index, &[0.1, 0.3, 0.45, 0.75, 0.95]),
            index,
            holes: None,
            relief: 1.2,
        },
    }
}

/// Flowers of a meadow: yellow and red ...
const FLOWERS: [Rgb8; 2] = [[226, 196, 92], [190, 84, 72]];
/// ... or violet and pink. Coloured enough to keep their colour under the grass tint.
const BLOSSOMS: [Rgb8; 2] = [[150, 112, 196], [232, 140, 180]];
/// Fallen leaves on a forest floor: dark, brown and tawny, with twigs of the dark tone. Muted
/// (beauty review E2, round 3: the earlier, brighter orange tones read as camouflage patches
/// on the grass instead of a mixed floor).
const LITTER: [Rgb8; 3] = [[71, 58, 44], [104, 83, 60], [134, 103, 69]];
/// Dry blades of a heath: shadowed, dull and light straw, close together in lightness, so that
/// a dry meadow reads as yellowed grass rather than bare earth (beauty review E2, round 3).
const STRAW: [Rgb8; 3] = [[160, 146, 98], [182, 166, 110], [200, 184, 124]];

/// Turf with flowers in it: the four lighter grass tones and two flower colours, each flower
/// one or two texels in clumps of a few.
fn with_flowers(canvas: &Canvas, grass: &[Rgb8], turf: &Grid<u8>, flowers: [Rgb8; 2]) -> Painting {
    let mut layer = Grid::filled(DESIGN, 0_u8);
    let mut rng = canvas.rng(5);
    for (i, (x, y)) in scatter(&mut rng, 9, 3.0).into_iter().enumerate() {
        let petal = 1 + u8::from(i % 3 == 2);
        layer.set_wrapped(x, y, petal);
        if rng.range_u32(0, 3) == 0 {
            layer.set_wrapped(x + 1, y, petal);
        }
    }
    let layer = canvas.stamp(&layer);
    let index = Grid::from_fn(canvas.size, |x, y| match layer.get(x, y) {
        0 => turf.get(x, y).max(1) - 1,
        petal => 3 + petal,
    });
    let colors = [
        grass[1], grass[2], grass[3], grass[4], flowers[0], flowers[1],
    ];
    Painting {
        palette: swatches(&colors, &[0.92, 0.9, 0.86, 0.82, 0.7, 0.7]),
        height: heights(&index, &[0.3, 0.45, 0.75, 0.95, 1.0, 1.0]),
        index,
        holes: None,
        relief: 1.2,
    }
}

/// A forest floor: three grass tones showing through patches of fallen leaves and a few
/// twigs, so that the ground under trees is not the lawn of a meadow.
fn with_litter(canvas: &Canvas, grass: &[Rgb8], turf: &Grid<u8>) -> Painting {
    let (cover_seed, tone_seed) = (canvas.seed(6), canvas.seed(7));
    let cover = canvas.paint(|p| fbm(cover_seed, p, 4, 2));
    let tone = canvas.paint(|p| fbm(tone_seed, p, 8, 1));
    let (litter_from, light_from, dark_below) = (
        cover.threshold_above(0.6),
        tone.threshold_above(0.3),
        tone.threshold_above(0.8),
    );
    // Twigs: short diagonal strokes of the darkest tone.
    let mut twigs = Grid::filled(DESIGN, false);
    let mut rng = canvas.rng(8);
    for (x, y) in scatter(&mut rng, 3, 6.0) {
        let length = 2 + i64::from(rng.range_u32(0, 2));
        for i in 0..length {
            twigs.set_wrapped(x + i, y + i / 2, true);
        }
    }
    let twigs = canvas.stamp(&twigs);
    let index = Grid::from_fn(canvas.size, |x, y| {
        if twigs.get(x, y) {
            3
        } else if cover.get(x, y) >= litter_from {
            let t = tone.get(x, y);
            if t >= light_from {
                5
            } else if t < dark_below {
                3
            } else {
                4
            }
        } else {
            [0, 0, 1, 2, 2][usize::from(turf.get(x, y)).min(4)]
        }
    })
    .despeckled(|_| false);
    let colors = [
        grass[1], grass[2], grass[3], LITTER[0], LITTER[1], LITTER[2],
    ];
    Painting {
        palette: swatches(&colors, &[0.92, 0.9, 0.86, 0.95, 0.93, 0.9]),
        height: heights(&index, &[0.2, 0.4, 0.6, 0.35, 0.55, 0.7]),
        index,
        holes: None,
        relief: 1.2,
    }
}

/// A dry heath: half the blades and much of the ground between them turn to straw.
fn with_straw(canvas: &Canvas, grass: &[Rgb8], turf: &Grid<u8>) -> Painting {
    let dry_seed = canvas.seed(9);
    let dry = canvas.paint(|p| fbm(dry_seed, p, 4, 2));
    let dry_from = dry.threshold_above(0.55);
    let index = Grid::from_fn(canvas.size, |x, y| {
        let tone = turf.get(x, y);
        if dry.get(x, y) >= dry_from {
            // Straw: the blade tones become light and dull straw, the ground its shadow.
            match tone {
                0 | 1 => 3,
                2 => 4,
                _ => 5,
            }
        } else {
            [0, 0, 1, 2, 2][usize::from(tone).min(4)]
        }
    });
    let colors = [grass[1], grass[2], grass[3], STRAW[0], STRAW[1], STRAW[2]];
    Painting {
        palette: swatches(&colors, &[0.92, 0.9, 0.86, 0.94, 0.9, 0.86]),
        height: heights(&index, &[0.2, 0.4, 0.75, 0.25, 0.5, 0.85]),
        index,
        holes: None,
        relief: 1.2,
    }
}

pub(super) fn grass_side(canvas: &Canvas, params: &Params) -> Painting {
    let grass = grass_colors(params);
    let soil = dirt_colors(color(params, "dirt", DIRT));
    // The soil is the `dirt` texture of the same seed, reduced to its three middle tones.
    let (dirt, dirt_height) = dirt_tones(&canvas.of_recipe("dirt"));

    // Depth of the fringe per column: a gentle wave of three to five rows, and here and there
    // a blade that hangs a row or two lower. A deep fringe keeps the one-metre steps of a
    // hillside green when they are seen from above, instead of drawing brown contour lines.
    let wave_seed = canvas.seed(1);
    let mut rng = canvas.rng(2);
    let mut depth = [0_i64; DESIGN];
    let mut blade = false;
    for (x, slot) in depth.iter_mut().enumerate() {
        let wave = value(wave_seed, DVec2::new(x as f64 + 0.5, 0.5), (5, 1));
        blade = !blade && rng.range_u32(0, 100) < 34;
        let hang = if blade {
            1 + i64::from(rng.range_u32(0, 2))
        } else {
            0
        };
        *slot = (4 + (wave * 1.4).round() as i64 + hang).clamp(3, 7);
    }
    // Layer values: 0 soil, 1 soil in the shadow of the fringe, 3 mid, 4 light grass. The
    // fringe has the middle and light tones of `grass_top` only, its tips and hanging blades
    // the middle one: dark tips drew dotted dark seams along every terrace of a hillside seen
    // from afar.
    let layer = Grid::from_fn(DESIGN, |x, y| {
        let column = |dx: usize| depth[(x + dx) % DESIGN];
        let (here, y) = (column(0), y as i64);
        if y > here {
            return 0_u8;
        } else if y == here {
            return 1;
        }
        let tip = y + 1 == here;
        let hanging = tip && here > column(DESIGN - 1) && here > column(1);
        let roll = rng.range_u32(0, 100);
        if y == 0 {
            if roll < 22 { 3 } else { 4 }
        } else if hanging || (tip && roll < 40) {
            3
        } else if y == 1 && roll < 45 {
            4
        } else {
            3
        }
    });
    let layer = canvas.stamp(&layer);
    let index = Grid::from_fn(canvas.size, |x, y| match layer.get(x, y) {
        0 => [0, 0, 1, 2, 2][usize::from(dirt.get(x, y)).min(4)],
        1 => 0,
        blade => blade,
    });
    let height = Grid::from_fn(canvas.size, |x, y| match layer.get(x, y) {
        0 => 0.55 * dirt_height.get(x, y),
        1 => 0.15,
        3 => 0.85,
        _ => 1.0,
    });
    let colors = [soil[1], soil[2], soil[3], grass[2], grass[3]];
    Painting {
        palette: swatches(&colors, &[0.96, 0.94, 0.92, 0.9, 0.84]),
        index,
        height,
        holes: None,
        relief: 1.4,
    }
}

pub(super) fn sand(canvas: &Canvas, params: &Params) -> Painting {
    let colors = ramp(
        color(params, "color", SAND),
        Shading::new(5, 0.22).pivot(0.55),
    );
    let warp_seed = canvas.seed(1);
    // Wind ripples: three crests per tile running slightly uphill, bent by a slow warp.
    // Tones: 1 trough, 2 body, 3 crest.
    let ripple = canvas.paint(|p| {
        let (u, v) = (wrap(p.x) / TILE, wrap(p.y) / TILE);
        det::cos(TAU * (u + 3.0 * v + 0.3 * fbm(warp_seed, p, 3, 1)))
    });
    let (crest, trough) = (ripple.threshold_above(0.26), ripple.threshold_above(0.80));
    let body = ripple
        .map(|r| {
            if r >= crest {
                3_u8
            } else if r < trough {
                1
            } else {
                2
            }
        })
        .despeckled(|_| false);
    // Single grains: dark ones (0) and bright ones that sparkle (4).
    let mut layer = Grid::filled(DESIGN, 0_u8);
    let mut rng = canvas.rng(2);
    for (i, (x, y)) in scatter(&mut rng, 18, 2.9).into_iter().enumerate() {
        layer.set_wrapped(x, y, if i % 2 == 0 { 1 } else { 2 });
    }
    let layer = canvas.stamp(&layer);
    let index = Grid::from_fn(canvas.size, |x, y| match layer.get(x, y) {
        1 => 0,
        2 => 4,
        _ => body.get(x, y),
    });
    Painting {
        palette: swatches(&colors, &[0.94, 0.92, 0.9, 0.88, 0.7]),
        height: heights(&index, &[0.3, 0.35, 0.5, 0.65, 0.75]),
        index,
        holes: None,
        relief: 0.6,
    }
}

pub(super) fn snow(canvas: &Canvas, params: &Params) -> Painting {
    let colors = ramp(
        color(params, "color", SNOW),
        Shading::new(4, 0.13).pivot(0.72).shift(1.1),
    );
    let drift_seed = canvas.seed(1);
    // Soft drifts, shaded as relief: what faces the light is bright, the lee side blue.
    let drift = canvas.paint(|p| fbm(drift_seed, p, 2, 2)).normalized();
    let density = canvas.density();
    let lit = Grid::from_fn(canvas.size, |x, y| {
        let here = drift.get(x, y);
        let toward_light = 0.5 * (drift.near(x, y, -1, 0) + drift.near(x, y, 0, -1));
        0.7 * here + 1.7 * (here - toward_light) * density
    });
    // Tones: 0 shadow, 1 body, 2 light; 3 is kept for sparkles.
    let body = lit
        .blurred()
        .quantize(&[0.18, 0.5, 0.32])
        .despeckled(|_| false);
    let mut sparkles = Grid::filled(DESIGN, false);
    for (x, y) in scatter(&mut canvas.rng(2), 7, 4.0) {
        sparkles.set_wrapped(x, y, true);
    }
    let sparkles = canvas.stamp(&sparkles);
    let index = Grid::from_fn(canvas.size, |x, y| {
        if sparkles.get(x, y) && body.get(x, y) > 0 {
            3
        } else {
            body.get(x, y)
        }
    });
    Painting {
        palette: swatches(&colors, &[0.82, 0.78, 0.72, 0.4]),
        height: drift,
        index,
        holes: None,
        relief: 1.6,
    }
}

pub(super) fn ice(canvas: &Canvas, params: &Params) -> Painting {
    let colors = ramp(
        color(params, "color", ICE),
        Shading::new(5, 0.30).pivot(0.45).shift(0.6),
    );
    let depth_seed = canvas.seed(1);
    // Depth of the ice in three calm tones: 1 deep, 2 body, 3 pale.
    let depth = canvas.paint(|p| value_fbm(depth_seed, p, (3, 3), 2));
    let body = depth
        .quantize(&[0.26, 0.48, 0.26])
        .map(|tone| tone + 1)
        .despeckled(|_| false);
    // Layer values: 1 crack (dark), 2 soft glint, 3 glint.
    let mut layer = Grid::filled(DESIGN, 0_u8);
    let mut rng = canvas.rng(2);
    // Glints: reflections running up to the right, each with a softer echo beside it.
    for (x, y) in scatter(&mut rng, 4, 5.0) {
        let length = 3 + i64::from(rng.range_u32(0, 4));
        for i in 0..length {
            layer.set_wrapped(x + i, y - i, 3);
            if i > 0 && i + 1 < length {
                layer.set_wrapped(x + i + 1, y - i, 2);
            }
        }
    }
    // Hairline cracks: long, running down to the right with the odd sidestep.
    for (x, y) in scatter(&mut rng, 2, 7.0) {
        let length = 6 + i64::from(rng.range_u32(0, 5));
        let (mut cx, mut cy) = (x, y);
        for _ in 0..length {
            if layer.wrapped(cx, cy) == 0 {
                layer.set_wrapped(cx, cy, 1);
            }
            cx += 1;
            cy += i64::from(rng.range_u32(0, 3) > 0);
        }
    }
    let layer = canvas.stamp(&layer);
    let index = Grid::from_fn(canvas.size, |x, y| match layer.get(x, y) {
        1 => 0,
        2 => body.get(x, y).max(3),
        3 => 4,
        _ => body.get(x, y),
    });
    Painting {
        palette: swatches(&colors, &[0.45, 0.2, 0.16, 0.12, 0.08]),
        height: heights(&index, &[0.2, 0.5, 0.55, 0.6, 0.65]),
        index,
        holes: None,
        relief: 0.5,
    }
}
