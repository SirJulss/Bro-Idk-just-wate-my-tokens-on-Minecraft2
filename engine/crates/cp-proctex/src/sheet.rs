//! The contact sheet: every recipe and preset on one image, for reviewing the pack as a whole.

use crate::params::Params;
use crate::presets::presets;
use crate::recipes::{Tint, recipe_info, recipes};
use crate::texture::{ALPHA_CUTOFF, PbrTexture};
use crate::tint::apply_tint;
use crate::{MAX_SIZE, MIN_SIZE, generate};

/// Seed of every texture on the sheet.
const SHEET_SEED: u64 = 1;
/// Largest edge length of the single magnified tile in screen pixels. It bounds the scale by
/// the texture size and with it the sheet: at most about 160 MB of RGBA, at 256 px textures
/// shown at scale 1.
const MAX_TILE: u32 = 256;

const BACKGROUND: [u8; 4] = [34, 36, 43, 255];
const LABEL: [u8; 4] = [226, 220, 204, 255];
/// The two tones of the checkerboard that shows through cutout textures.
const CHECKER: [[u8; 4]; 2] = [[58, 62, 74, 255], [46, 49, 60, 255]];

/// One picture inside a cell: a square RGBA image, magnified and optionally repeated.
pub(crate) struct View {
    /// Edge length of `pixels` in texels.
    pub size: u32,
    pub pixels: Vec<[u8; 4]>,
    /// Screen pixels per texel.
    pub scale: u32,
    /// How often the image is repeated on each axis.
    pub repeat: u32,
}

impl View {
    fn extent(&self) -> u32 {
        self.size * self.scale * self.repeat
    }
}

/// One labelled cell of a sheet.
pub(crate) struct Cell {
    pub label: String,
    /// Pictures side by side, left to right.
    pub views: Vec<View>,
    /// Colour swatches drawn under the first view.
    pub palette: Vec<[u8; 3]>,
}

/// An RGBA8 image under construction.
struct Image {
    width: u32,
    height: u32,
    data: Vec<u8>,
}

impl Image {
    fn new(width: u32, height: u32, fill: [u8; 4]) -> Self {
        let pixels = width as usize * height as usize;
        let mut data = Vec::with_capacity(pixels * 4);
        for _ in 0..pixels {
            data.extend_from_slice(&fill);
        }
        Self {
            width,
            height,
            data,
        }
    }

    fn fill_rect(&mut self, x: u32, y: u32, width: u32, height: u32, color: [u8; 4]) {
        for row in y..(y + height).min(self.height) {
            for column in x..(x + width).min(self.width) {
                let at = (row as usize * self.width as usize + column as usize) * 4;
                self.data[at..at + 4].copy_from_slice(&color);
            }
        }
    }

    /// Draws a view with its top left corner at `(x, y)`. Cut-out texels show a checkerboard.
    fn draw_view(&mut self, x: u32, y: u32, view: &View) {
        let tile = view.size * view.scale;
        // Checker squares of a quarter tile, at least one screen pixel.
        let square = (tile / 4).max(1);
        for ty in 0..view.size * view.repeat {
            for tx in 0..view.size * view.repeat {
                let texel = view.pixels[((ty % view.size) * view.size + tx % view.size) as usize];
                let (px, py) = (x + tx * view.scale, y + ty * view.scale);
                if texel[3] >= ALPHA_CUTOFF {
                    let [r, g, b, _] = texel;
                    self.fill_rect(px, py, view.scale, view.scale, [r, g, b, 255]);
                } else {
                    for sy in 0..view.scale {
                        for sx in 0..view.scale {
                            let parity = ((px - x + sx) / square + (py - y + sy) / square) % 2;
                            self.fill_rect(px + sx, py + sy, 1, 1, CHECKER[parity as usize]);
                        }
                    }
                }
            }
        }
    }

    /// Draws `text` in the built-in 3×5 font, magnified by `scale`.
    fn draw_text(&mut self, x: u32, y: u32, text: &str, scale: u32, color: [u8; 4]) {
        for (i, character) in text.chars().enumerate() {
            let rows = glyph(character);
            let left = x + i as u32 * (GLYPH_WIDTH + 1) * scale;
            for (row, bits) in rows.iter().enumerate() {
                for column in 0..GLYPH_WIDTH {
                    if bits & (1 << (GLYPH_WIDTH - 1 - column)) != 0 {
                        let (px, py) = (left + column * scale, y + row as u32 * scale);
                        self.fill_rect(px, py, scale, scale, color);
                    }
                }
            }
        }
    }
}

const GLYPH_WIDTH: u32 = 3;
const GLYPH_HEIGHT: u32 = 5;

/// Rows of a character in a 3×5 pixel font, top to bottom; bit 2 is the left column.
/// Lower case is drawn as upper case, unknown characters as a blank.
const fn glyph(character: char) -> [u8; 5] {
    match character.to_ascii_uppercase() {
        'A' => [0b010, 0b101, 0b111, 0b101, 0b101],
        'B' => [0b110, 0b101, 0b110, 0b101, 0b110],
        'C' => [0b011, 0b100, 0b100, 0b100, 0b011],
        'D' => [0b110, 0b101, 0b101, 0b101, 0b110],
        'E' => [0b111, 0b100, 0b110, 0b100, 0b111],
        'F' => [0b111, 0b100, 0b110, 0b100, 0b100],
        'G' => [0b011, 0b100, 0b101, 0b101, 0b011],
        'H' => [0b101, 0b101, 0b111, 0b101, 0b101],
        'I' => [0b111, 0b010, 0b010, 0b010, 0b111],
        'J' => [0b001, 0b001, 0b001, 0b101, 0b010],
        'K' => [0b101, 0b101, 0b110, 0b101, 0b101],
        'L' => [0b100, 0b100, 0b100, 0b100, 0b111],
        'M' => [0b101, 0b111, 0b111, 0b101, 0b101],
        'N' => [0b110, 0b101, 0b101, 0b101, 0b101],
        'O' => [0b111, 0b101, 0b101, 0b101, 0b111],
        'P' => [0b110, 0b101, 0b110, 0b100, 0b100],
        'Q' => [0b010, 0b101, 0b101, 0b110, 0b011],
        'R' => [0b110, 0b101, 0b110, 0b101, 0b101],
        'S' => [0b011, 0b100, 0b010, 0b001, 0b110],
        'T' => [0b111, 0b010, 0b010, 0b010, 0b010],
        'U' => [0b101, 0b101, 0b101, 0b101, 0b111],
        'V' => [0b101, 0b101, 0b101, 0b101, 0b010],
        'W' => [0b101, 0b101, 0b111, 0b111, 0b101],
        'X' => [0b101, 0b101, 0b010, 0b101, 0b101],
        'Y' => [0b101, 0b101, 0b010, 0b010, 0b010],
        'Z' => [0b111, 0b001, 0b010, 0b100, 0b111],
        '0' => [0b010, 0b101, 0b101, 0b101, 0b010],
        '1' => [0b010, 0b110, 0b010, 0b010, 0b111],
        '2' => [0b110, 0b001, 0b010, 0b100, 0b111],
        '3' => [0b110, 0b001, 0b010, 0b001, 0b110],
        '4' => [0b101, 0b101, 0b111, 0b001, 0b001],
        '5' => [0b111, 0b100, 0b110, 0b001, 0b110],
        '6' => [0b011, 0b100, 0b110, 0b101, 0b010],
        '7' => [0b111, 0b001, 0b010, 0b010, 0b010],
        '8' => [0b010, 0b101, 0b010, 0b101, 0b010],
        '9' => [0b010, 0b101, 0b011, 0b001, 0b110],
        '_' => [0b000, 0b000, 0b000, 0b000, 0b111],
        '-' => [0b000, 0b000, 0b111, 0b000, 0b000],
        '+' => [0b000, 0b010, 0b111, 0b010, 0b000],
        '*' => [0b000, 0b101, 0b010, 0b101, 0b000],
        '/' => [0b001, 0b001, 0b010, 0b100, 0b100],
        '.' => [0b000, 0b000, 0b000, 0b000, 0b010],
        ':' => [0b000, 0b010, 0b000, 0b010, 0b000],
        _ => [0; 5],
    }
}

/// Lays `cells` out in a grid of `columns` columns and renders them to RGBA8.
///
/// All cells are given the size of the largest one. `unit` is the basic spacing in screen
/// pixels: gaps are one unit, the margin around a cell two.
pub(crate) fn compose(cells: &[Cell], columns: u32, unit: u32) -> (u32, u32, Vec<u8>) {
    let unit = unit.max(1);
    let columns = columns.max(1);
    let font = (unit / 4).max(1);
    let label_height = GLYPH_HEIGHT * font + unit;
    let strip_height = 2 * unit;
    let inner_width = |cell: &Cell| {
        let views: u32 = cell.views.iter().map(View::extent).sum();
        let text = cell.label.chars().count() as u32 * (GLYPH_WIDTH + 1) * font;
        (views + unit * cell.views.len().saturating_sub(1) as u32).max(text)
    };
    let inner_height = |cell: &Cell| {
        let views = cell.views.iter().map(View::extent).max().unwrap_or(0);
        let first = cell.views.first().map_or(0, View::extent);
        // The swatches hang under the first view and only add height if nothing else is
        // taller.
        label_height + views.max(first + unit + strip_height)
    };
    let cell_width = cells.iter().map(inner_width).max().unwrap_or(0) + 2 * unit;
    let cell_height = cells.iter().map(inner_height).max().unwrap_or(0) + 2 * unit;
    let rows = (cells.len() as u32).div_ceil(columns);
    let mut image = Image::new(
        columns * cell_width + 2 * unit,
        rows * cell_height + 2 * unit,
        BACKGROUND,
    );
    for (i, cell) in cells.iter().enumerate() {
        let left = unit + (i as u32 % columns) * cell_width + unit;
        let top = unit + (i as u32 / columns) * cell_height + unit;
        image.draw_text(left, top, &cell.label, font, LABEL);
        let mut x = left;
        for view in &cell.views {
            image.draw_view(x, top + label_height, view);
            x += view.extent() + unit;
        }
        if let Some(first) = cell.views.first() {
            let count = cell.palette.len().max(1) as u32;
            let width = (first.extent() / count).min(4 * unit);
            let y = top + label_height + first.extent() + unit;
            for (j, &[r, g, b]) in cell.palette.iter().enumerate() {
                image.fill_rect(
                    left + j as u32 * width,
                    y,
                    width,
                    strip_height,
                    [r, g, b, 255],
                );
            }
        }
    }
    (image.width, image.height, image.data)
}

/// The distinct visible colours of a texture, dark to light.
pub(crate) fn palette_of(pixels: &[[u8; 4]]) -> Vec<[u8; 3]> {
    let mut colors: Vec<[u8; 3]> = pixels
        .iter()
        .filter(|texel| texel[3] >= ALPHA_CUTOFF)
        .map(|&[r, g, b, _]| [r, g, b])
        .collect();
    // Rec. 601 luma in integers is enough to order swatches.
    colors.sort_by_key(|&[r, g, b]| {
        (
            299 * u32::from(r) + 587 * u32::from(g) + 114 * u32::from(b),
            r,
            g,
            b,
        )
    });
    colors.dedup();
    colors
}

/// The albedo as the game shows it: with the reference tint of `tint` multiplied in.
pub(crate) fn shown_albedo(texture: &PbrTexture, tint: Tint) -> Vec<[u8; 4]> {
    match tint {
        Tint::None => texture.albedo.clone(),
        tint => (texture.albedo.iter())
            .map(|&texel| apply_tint(texel, tint.reference_color()))
            .collect(),
    }
}

/// A cell showing `pixels` once at `scale` and tiled 3×3 at half of it.
fn albedo_cell(label: String, size: u32, pixels: Vec<[u8; 4]>, scale: u32) -> Cell {
    Cell {
        label,
        palette: palette_of(&pixels),
        views: vec![
            View {
                size,
                pixels: pixels.clone(),
                scale,
                repeat: 1,
            },
            View {
                size,
                pixels,
                scale: (scale / 2).max(1),
                repeat: 3,
            },
        ],
    }
}

/// What the sheet shows: every recipe as authored, the tinted recipes again as they look in
/// the game, then every preset as it looks in the game. `(label, texture, tint to show)`.
pub(crate) fn sheet_entries(size: u32) -> Vec<(String, PbrTexture, Tint)> {
    let mut entries = Vec::new();
    let defaults = Params::new();
    for name in recipes() {
        if let Ok(texture) = generate(name, &defaults, SHEET_SEED, size) {
            entries.push(((*name).to_owned(), texture, Tint::None));
        }
    }
    for name in recipes() {
        let tint = recipe_info(name).map_or(Tint::None, |info| info.tint);
        if tint != Tint::None
            && let Ok(texture) = generate(name, &defaults, SHEET_SEED, size)
        {
            entries.push((format!("{name} + tint"), texture, tint));
        }
    }
    for preset in presets() {
        if let Ok(texture) = generate(preset.recipe, &preset.to_params(), SHEET_SEED, size) {
            entries.push((preset.name.to_owned(), texture, preset.tint));
        }
    }
    entries
}

/// The supported texture size for a requested one: clamped to the supported range and rounded
/// down to a power of two.
fn supported_size(size: u32) -> u32 {
    let clamped = size.clamp(MIN_SIZE, MAX_SIZE);
    1 << clamped.ilog2()
}

/// The scale for textures of `size` texels: at least 1, and no tile wider than [`MAX_TILE`].
fn sheet_scale(size: u32, scale: u32) -> u32 {
    scale.clamp(1, (MAX_TILE / size.max(1)).max(1))
}

/// An overview of the whole pack as an RGBA8 image: returns `(width, height, pixels)` with
/// `width * height * 4` bytes in row-major order, top row first.
///
/// Every cell shows one texture magnified `scale` times (nearest neighbour) next to a 3×3
/// tiling at half that scale, with its name above and its palette below. The sheet lists
/// every recipe with default parameters as authored, then the tinted recipes again with their
/// class's [`Tint::reference_color`] multiplied in, then every preset of [`crate::presets`]
/// as it appears in the game. Cut-out texels show a checkerboard.
///
/// `size` is the texture size; an unsupported value is clamped to [`MIN_SIZE`]..=[`MAX_SIZE`]
/// and rounded down to a power of two. `scale` is clamped to `1..=256 / size`, so that a
/// single tile is at most 256 screen pixels wide and the sheet stays below about 160 MB
/// whatever the arguments. The image is deterministic.
#[must_use]
pub fn contact_sheet(size: u32, scale: u32) -> (u32, u32, Vec<u8>) {
    let size = supported_size(size);
    let scale = sheet_scale(size, scale);
    let cells: Vec<Cell> = sheet_entries(size)
        .into_iter()
        .map(|(label, texture, tint)| albedo_cell(label, size, shown_albedo(&texture, tint), scale))
        .collect();
    // As many columns as make the sheet roughly square.
    let columns = (cells.len() as f64 * 0.62).sqrt().ceil() as u32;
    compose(&cells, columns, scale.max(2))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pixel(image: &(u32, u32, Vec<u8>), x: u32, y: u32) -> [u8; 4] {
        let at = (y as usize * image.0 as usize + x as usize) * 4;
        [
            image.2[at],
            image.2[at + 1],
            image.2[at + 2],
            image.2[at + 3],
        ]
    }

    #[test]
    fn the_sheet_lists_recipes_then_tinted_recipes_then_presets() {
        let entries = sheet_entries(16);
        let labels: Vec<&str> = entries.iter().map(|(label, _, _)| label.as_str()).collect();
        assert_eq!(labels.len(), recipes().len() + 4 + presets().len());
        assert_eq!(&labels[..recipes().len()], recipes());
        assert_eq!(
            labels[recipes().len()..recipes().len() + 4],
            [
                "grass_top + tint",
                "grass_side + tint",
                "leaves + tint",
                "water + tint"
            ]
        );
        assert_eq!(labels[recipes().len() + 4], presets()[0].name);
        for (label, texture, _) in &entries {
            assert_eq!(texture.size, 16, "{label}");
        }
    }

    #[test]
    fn the_sheet_is_a_deterministic_opaque_image() {
        let sheet = contact_sheet(16, 2);
        let (width, height, data) = &sheet;
        assert_eq!(data.len(), *width as usize * *height as usize * 4);
        let (pixels, rest) = data.as_chunks::<4>();
        assert!(rest.is_empty() && pixels.iter().all(|pixel| pixel[3] == 255));
        assert_eq!(sheet, contact_sheet(16, 2));
        // More than the background and the labels: the textures are on it.
        let mut colors = pixels.to_vec();
        colors.sort_unstable();
        colors.dedup();
        assert!(colors.len() > 100, "{}", colors.len());
        assert_eq!(pixel(&sheet, 0, 0), BACKGROUND);
        // A larger scale gives a larger sheet.
        let larger = contact_sheet(16, 4);
        assert!(larger.0 > *width && larger.1 > *height);
    }

    #[test]
    fn bad_arguments_are_replaced_not_rejected() {
        assert_eq!(supported_size(0), 16);
        assert_eq!(supported_size(16), 16);
        assert_eq!(supported_size(31), 16);
        assert_eq!(supported_size(100), 64);
        assert_eq!(supported_size(u32::MAX), 256);
        assert_eq!(contact_sheet(7, 0), contact_sheet(16, 1));
        assert_eq!(contact_sheet(17, 1), contact_sheet(16, 1));
        // The scale is bounded by the texture size: a tile is never wider than 256 pixels,
        // which bounds the memory of the sheet.
        assert_eq!(sheet_scale(16, 16), 16);
        assert_eq!(sheet_scale(16, 99), 16);
        assert_eq!(sheet_scale(16, 0), 1);
        assert_eq!(sheet_scale(64, 16), 4);
        assert_eq!(sheet_scale(256, 16), 1);
        assert_eq!(sheet_scale(256, 0), 1);
        for size in [16, 32, 64, 128, 256] {
            assert!(size * sheet_scale(size, u32::MAX) <= MAX_TILE);
        }
    }

    #[test]
    fn cells_show_one_tile_next_to_a_tiling() {
        // One red texel in the corner of an otherwise blue 2×2 image.
        let pixels = vec![
            [255, 0, 0, 255],
            [0, 0, 255, 255],
            [0, 0, 255, 255],
            [0, 0, 255, 0],
        ];
        let cell = albedo_cell("ab".to_owned(), 2, pixels, 8);
        assert_eq!(cell.palette, [[0, 0, 255], [255, 0, 0]]);
        let sheet = compose(&[cell], 1, 8);
        let (left, top) = (16, 16 + GLYPH_HEIGHT * 2 + 8);
        // The single view: 2 texels at scale 8.
        assert_eq!(pixel(&sheet, left, top), [255, 0, 0, 255]);
        assert_eq!(pixel(&sheet, left + 7, top + 7), [255, 0, 0, 255]);
        assert_eq!(pixel(&sheet, left + 8, top), [0, 0, 255, 255]);
        // The cut-out texel shows the checkerboard, not its hidden colour.
        assert!(CHECKER.contains(&pixel(&sheet, left + 12, top + 12)));
        // The tiling: 3×3 repeats at scale 4, one unit to the right.
        let tiled = left + 16 + 8;
        for repeat in 0..3 {
            assert_eq!(
                pixel(&sheet, tiled + repeat * 8, top + repeat * 8),
                [255, 0, 0, 255]
            );
            assert_eq!(pixel(&sheet, tiled + repeat * 8 + 4, top), [0, 0, 255, 255]);
        }
        assert_eq!(pixel(&sheet, tiled + 24, top), BACKGROUND);
        // The label is drawn above the views.
        let label_pixels = (0..GLYPH_HEIGHT * 2)
            .flat_map(|y| (0..16).map(move |x| (x, y)))
            .filter(|&(x, y)| pixel(&sheet, left + x, 16 + y) == LABEL)
            .count();
        assert!(label_pixels > 20, "{label_pixels}");
        // The swatches hang under the single view.
        let strip = top + 16 + 8;
        assert_eq!(pixel(&sheet, left, strip), [0, 0, 255, 255]);
        assert_eq!(pixel(&sheet, left + 8, strip), [255, 0, 0, 255]);
    }

    #[test]
    fn the_font_covers_recipe_and_preset_names() {
        for name in recipes()
            .iter()
            .copied()
            .chain(presets().iter().map(|p| p.name))
        {
            for character in name.chars() {
                assert_ne!(glyph(character), [0; 5], "no glyph for {character:?}");
            }
        }
        assert_eq!(glyph('a'), glyph('A'));
        assert_eq!(glyph(' '), [0; 5]);
        assert_eq!(glyph('~'), [0; 5]);
        // No glyph uses bits outside its three columns.
        for code in 0..128_u8 {
            assert!(glyph(char::from(code)).iter().all(|row| *row < 8));
        }
    }

    #[test]
    fn palettes_are_ordered_and_skip_hidden_colours() {
        let pixels = [
            [200, 200, 200, 255],
            [10, 10, 10, 255],
            [200, 200, 200, 255],
            [90, 0, 0, 0],
        ];
        assert_eq!(palette_of(&pixels), [[10, 10, 10], [200, 200, 200]]);
    }

    #[test]
    fn tinted_entries_are_shown_with_their_reference_tint() {
        let neutral = PbrTexture {
            size: 1,
            albedo: vec![[255, 255, 255, 255]],
            normal: vec![[128, 128]],
            roughness: vec![0],
            emissive: vec![0],
            height: vec![0],
        };
        assert_eq!(shown_albedo(&neutral, Tint::None), [[255, 255, 255, 255]]);
        let [r, g, b] = Tint::Grass.reference_color();
        assert_eq!(shown_albedo(&neutral, Tint::Grass), [[r, g, b, 255]]);
    }
}
