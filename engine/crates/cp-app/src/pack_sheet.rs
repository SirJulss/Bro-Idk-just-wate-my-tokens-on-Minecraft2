//! The texture sheet of content packs (`cpe proctex-sheet --pack <dir>`): every texture of the
//! packs made the way the game makes it (each texture definition's own recipe, parameters and
//! seed, or its image file), shown as it appears in the game.
//!
//! One cell per texture, in texture-layer order: the key and the source above; the texture
//! magnified, tinted like its material when the material takes a biome tint (with the tints
//! the scenes use, cp-proctex's reference tints); a 3 x 3 tiling at half the magnification to
//! check the seams; and a swatch of the colour that far LOD levels draw instead of the texture
//! (the linear average albedo, tinted the same way). Cut-out texels show a checkerboard.

use std::path::{Path, PathBuf};

use cp_content::{Registry, TextureSource, TintClass};
use cp_proctex::{ALPHA_CUTOFF, PbrTexture, Tint};
use cp_render::overlay::font::{
    GLYPH_ADVANCE, GLYPH_HEIGHT, GLYPH_WIDTH, LINE_HEIGHT, glyph_index, glyph_pixel,
};

use crate::Failure;
use crate::content_check::make_texture_sized;
use crate::scene::sky::{linear_to_srgb, srgb_to_linear};

const BACKGROUND: [u8; 4] = [30, 32, 38, 255];
const LABEL: [u8; 4] = [232, 226, 210, 255];
const DETAIL: [u8; 4] = [150, 156, 170, 255];
const CHECKER: [[u8; 4]; 2] = [[62, 66, 78, 255], [48, 51, 62, 255]];

/// One texture of the sheet.
struct Entry {
    key: String,
    source: String,
    tint: TintClass,
    texture: PbrTexture,
}

/// Renders the sheet of the textures of `packs` (load order) at `size` texels and magnification
/// `scale`; returns `(width, height, RGBA8 pixels)`.
///
/// # Errors
/// A [`Failure`] when the packs do not load or a texture cannot be made.
pub fn render(packs: &[PathBuf], size: u32, scale: u32) -> Result<(u32, u32, Vec<u8>), Failure> {
    let refs: Vec<&Path> = packs.iter().map(PathBuf::as_path).collect();
    let registry = Registry::load(&refs)
        .map_err(|error| Failure::new(format!("the content packs do not load: {error}")))?;
    let mut entries = Vec::new();
    let mut failures = Vec::new();
    for (layer, def) in registry.textures().iter().enumerate() {
        let texture = match make_texture_sized(def, size) {
            Ok(texture) => texture,
            Err(reason) => {
                failures.push(format!("texture `{}`: {reason}", def.key));
                continue;
            }
        };
        let source = match &def.source {
            TextureSource::Procedural {
                recipe,
                params,
                seed,
            } => {
                let params: Vec<String> = params
                    .iter()
                    .map(|(name, value)| match value {
                        cp_content::ParamValue::Color([r, g, b]) => {
                            format!("{name}=#{r:02x}{g:02x}{b:02x}")
                        }
                        cp_content::ParamValue::Variant(variant) => format!("{name}={variant}"),
                    })
                    .collect();
                format!("{recipe} seed {seed} {}", params.join(" "))
            }
            TextureSource::File { path, .. } => format!("file {path}"),
        };
        // The tint of the first material that uses the texture.
        let tint = registry
            .materials()
            .iter()
            .find(|material| material.texture.index() == layer)
            .map_or(TintClass::None, |material| material.tint);
        entries.push(Entry {
            key: def.key.clone(),
            source,
            tint,
            texture,
        });
    }
    if !failures.is_empty() {
        return Err(Failure::new(format!(
            "{} of {} textures cannot be made:\n  {}",
            failures.len(),
            registry.textures().len(),
            failures.join("\n  ")
        )));
    }
    Ok(compose(&entries, scale))
}

/// The sRGB tint of a tint class as the scenes use it (cp-proctex's reference tints).
fn tint_color(tint: TintClass) -> Option<[u8; 3]> {
    match tint {
        TintClass::None => None,
        TintClass::Grass => Some(Tint::Grass.reference_color()),
        TintClass::Foliage => Some(Tint::Foliage.reference_color()),
        TintClass::Water => Some(Tint::Water.reference_color()),
    }
}

/// A texel as the game shows it: `albedo * mix(1, tint, weight)` in linear light, with the
/// tint weight of cp-proctex.
fn shown(texel: [u8; 4], tint: Option<[u8; 3]>) -> [u8; 4] {
    let Some(tint) = tint else {
        return texel;
    };
    let linear = [0, 1, 2].map(|c| srgb_to_linear(f64::from(texel[c]) / 255.0));
    let weight = f64::from(cp_proctex::tint_weight(linear.map(|c| c as f32)));
    let mut out = texel;
    for c in 0..3 {
        let factor = 1.0 + (srgb_to_linear(f64::from(tint[c]) / 255.0) - 1.0) * weight;
        out[c] = (linear_to_srgb(linear[c] * factor) * 255.0).round() as u8;
    }
    out
}

/// An RGBA8 image under construction.
struct Canvas {
    width: u32,
    height: u32,
    pixels: Vec<u8>,
}

impl Canvas {
    fn new(width: u32, height: u32) -> Self {
        let mut pixels = Vec::with_capacity(width as usize * height as usize * 4);
        for _ in 0..width as usize * height as usize {
            pixels.extend_from_slice(&BACKGROUND);
        }
        Self {
            width,
            height,
            pixels,
        }
    }

    fn set(&mut self, x: u32, y: u32, color: [u8; 4]) {
        if x < self.width && y < self.height {
            let at = (y as usize * self.width as usize + x as usize) * 4;
            self.pixels[at..at + 4].copy_from_slice(&color);
        }
    }

    fn rect(&mut self, x: u32, y: u32, width: u32, height: u32, color: [u8; 4]) {
        for row in y..y + height {
            for column in x..x + width {
                self.set(column, row, color);
            }
        }
    }

    /// Text in the engine's 5 x 8 font, each font pixel `scale` x `scale`.
    fn text(&mut self, x: u32, y: u32, scale: u32, color: [u8; 4], text: &str) {
        for (index, character) in text.chars().enumerate() {
            let glyph = glyph_index(character);
            let left = x + index as u32 * GLYPH_ADVANCE * scale;
            for gy in 0..GLYPH_HEIGHT {
                for gx in 0..GLYPH_WIDTH {
                    if glyph_pixel(glyph, gx, gy) {
                        self.rect(left + gx * scale, y + gy * scale, scale, scale, color);
                    }
                }
            }
        }
    }

    /// A texture of `size` texels at `scale`, repeated `repeat` times per axis.
    fn texture(&mut self, x: u32, y: u32, size: u32, texels: &[[u8; 4]], scale: u32, repeat: u32) {
        let square = (size * scale / 4).max(1);
        for ty in 0..size * repeat {
            for tx in 0..size * repeat {
                let texel = texels[((ty % size) * size + tx % size) as usize];
                for sy in 0..scale {
                    for sx in 0..scale {
                        let (px, py) = (tx * scale + sx, ty * scale + sy);
                        let color = if texel[3] >= ALPHA_CUTOFF {
                            [texel[0], texel[1], texel[2], 255]
                        } else {
                            CHECKER[((px / square + py / square) % 2) as usize]
                        };
                        self.set(x + px, y + py, color);
                    }
                }
            }
        }
    }
}

/// Lays the entries out in a grid and draws them.
fn compose(entries: &[Entry], scale: u32) -> (u32, u32, Vec<u8>) {
    let scale = scale.max(1);
    let unit = (2 * scale).max(4);
    let font = (scale / 4).max(1);
    let largest = entries.iter().map(|e| e.texture.size).max().unwrap_or(16);
    let tile = largest * scale;
    let tiling = largest * 3 * (scale / 2).max(1);
    let swatch = tile / 2;
    let views = tile + unit + tiling + unit + swatch;
    let label_width = entries
        .iter()
        .map(|entry| entry.key.len().max(entry.source.len()) as u32 * GLYPH_ADVANCE * font)
        .max()
        .unwrap_or(0);
    let label_height = 3 * LINE_HEIGHT * font + unit;
    let cell_width = views.max(label_width) + 2 * unit;
    let cell_height = label_height + tile.max(tiling) + 2 * unit;
    let columns = ((entries.len() as f64 * 0.75).sqrt().ceil() as u32).max(1);
    let rows = (entries.len() as u32).div_ceil(columns).max(1);
    let mut canvas = Canvas::new(
        columns * cell_width + 2 * unit,
        rows * cell_height + 2 * unit,
    );
    for (index, entry) in entries.iter().enumerate() {
        let left = unit + (index as u32 % columns) * cell_width + unit;
        let top = unit + (index as u32 / columns) * cell_height + unit;
        let tint = tint_color(entry.tint);
        canvas.text(left, top, font, LABEL, &entry.key);
        canvas.text(left, top + LINE_HEIGHT * font, font, DETAIL, &entry.source);
        let note = match entry.tint {
            TintClass::None => "untinted".to_owned(),
            other => format!("tint {other:?} (shown tinted)").to_lowercase(),
        };
        canvas.text(left, top + 2 * LINE_HEIGHT * font, font, DETAIL, &note);
        let texels: Vec<[u8; 4]> = entry
            .texture
            .albedo
            .iter()
            .map(|&texel| shown(texel, tint))
            .collect();
        let size = entry.texture.size;
        let y = top + label_height;
        canvas.texture(left, y, size, &texels, tile / size, 1);
        let half = (tile / size / 2).max(1);
        canvas.texture(left + tile + unit, y, size, &texels, half, 3);
        // The far-LOD colour: the linear average albedo with the tint multiplied in.
        let average = cp_proctex::average_color(&entry.texture);
        let tint_linear = tint.map_or([1.0; 3], |t| {
            t.map(|c| srgb_to_linear(f64::from(c) / 255.0))
        });
        let far = [0, 1, 2].map(|c| {
            (linear_to_srgb(f64::from(average[c]) * tint_linear[c]) * 255.0).round() as u8
        });
        canvas.rect(
            left + tile + unit + tiling + unit,
            y,
            swatch,
            swatch,
            [far[0], far[1], far[2], 255],
        );
    }
    (canvas.width, canvas.height, canvas.pixels)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::content_check::default_pack_dir;

    #[test]
    fn the_base_pack_sheet_shows_every_texture() {
        let pack = default_pack_dir().expect("the workspace has its base pack");
        let (width, height, pixels) =
            render(std::slice::from_ref(&pack), 16, 4).expect("the sheet renders");
        assert_eq!(pixels.len(), width as usize * height as usize * 4);
        let registry = Registry::load(&[pack.as_path()]).expect("the base pack loads");
        // Every texture has a cell: the sheet is at least as large as the cells need.
        let cells = registry.textures().len() as u32;
        assert!(width * height >= cells * 64 * 64, "{width} x {height}");
        // The textures are on it, not just the background and the labels: four to six colours
        // per texture (the cp-proctex palettes) and the tinted and far-LOD variants.
        let mut colors: Vec<&[u8]> = pixels.chunks(4).collect();
        colors.sort_unstable();
        colors.dedup();
        assert!(
            colors.len() > 4 * registry.textures().len(),
            "{}",
            colors.len()
        );
    }

    #[test]
    fn tinting_follows_the_tint_weight() {
        // A neutral texel takes the whole tint; a saturated one keeps its colour.
        let grey = shown([200, 200, 200, 255], Some([126, 188, 84]));
        assert!(grey[1] > grey[0] && grey[1] > grey[2], "{grey:?}");
        let soil = [134, 96, 66, 255];
        assert_eq!(shown(soil, Some([126, 188, 84])), soil);
        assert_eq!(shown(soil, None), soil);
    }
}
