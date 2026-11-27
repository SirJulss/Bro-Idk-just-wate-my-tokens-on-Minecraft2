//! The review pages: the images and numbers the art was judged by, written to
//! `engine/target/proctex/review/` so that the next round of art direction starts from the
//! same view.

use std::fmt::Write as _;
use std::path::{Path, PathBuf};

use cp_core::det;

use super::channels::{RELIEF_SEEDS, height_explained_by_colour};
use super::tiling::edge_steps;
use super::{SEED, coverage, pack, paint, palette};
use crate::color::{linear_to_srgb, rgb8_to_oklab, srgb_to_linear};
use crate::sheet::{Cell, View, compose, palette_of, sheet_entries, shown_albedo};
use crate::{PbrTexture, decode_normal};

fn review_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/proctex/review")
}

fn write_png(path: &Path, image: &(u32, u32, Vec<u8>)) {
    let (width, height, data) = image;
    let file = std::fs::File::create(path).expect("create file");
    let mut encoder = png::Encoder::new(std::io::BufWriter::new(file), *width, *height);
    encoder.set_color(png::ColorType::Rgba);
    encoder.set_depth(png::BitDepth::Eight);
    let mut writer = encoder.write_header().expect("png header");
    writer.write_image_data(data).expect("png data");
    writer.finish().expect("png end");
}

fn grey(values: &[u8]) -> Vec<[u8; 4]> {
    values.iter().map(|&v| [v, v, v, 255]).collect()
}

/// The texture under a fixed light from the upper left: a rough preview of how the channels
/// work together in a renderer.
fn lit_preview(texture: &PbrTexture, albedo: &[[u8; 4]]) -> Vec<[u8; 4]> {
    // Towards the light: left, up, and out of the surface.
    let light = [-0.5_f64, 0.55, 0.67];
    let half = {
        let h = [light[0], light[1], light[2] + 1.0];
        let length = (h[0] * h[0] + h[1] * h[1] + h[2] * h[2]).sqrt();
        [h[0] / length, h[1] / length, h[2] / length]
    };
    (0..albedo.len())
        .map(|i| {
            let normal = decode_normal(texture.normal[i]).map(f64::from);
            let facing =
                |v: [f64; 3]| det::max(normal[0] * v[0] + normal[1] * v[1] + normal[2] * v[2], 0.0);
            let roughness = f64::from(texture.roughness[i]) / 255.0;
            let shininess = 2.0 + 120.0 * (1.0 - roughness) * (1.0 - roughness);
            let specular = 0.5 * (1.0 - roughness) * det::powf(facing(half), shininess);
            let emissive = f64::from(texture.emissive[i]) / 255.0;
            let diffuse = det::max(0.25 + facing(light), 1.6 * emissive);
            let [r, g, b, a] = albedo[i];
            let shade = |c: u8| linear_to_srgb(srgb_to_linear(c) * diffuse + specular);
            [shade(r), shade(g), shade(b), a]
        })
        .collect()
}

/// Every texture large and tiled, with its palette: nine to a page.
fn albedo_pages(dir: &Path) {
    let cells: Vec<Cell> = sheet_entries(16)
        .iter()
        .map(|(label, texture, tint)| {
            let pixels = shown_albedo(texture, *tint);
            Cell {
                label: label.clone(),
                palette: palette_of(&pixels),
                views: vec![
                    View {
                        size: 16,
                        pixels: pixels.clone(),
                        scale: 8,
                        repeat: 1,
                    },
                    View {
                        size: 16,
                        pixels,
                        scale: 4,
                        repeat: 3,
                    },
                ],
            }
        })
        .collect();
    for (page, cells) in cells.chunks(9).enumerate() {
        write_png(
            &dir.join(format!("albedo_{page}.png")),
            &compose(cells, 3, 8),
        );
    }
}

/// Albedo, normal map, height, roughness and a lit preview side by side.
fn channel_pages(dir: &Path) {
    let cells: Vec<Cell> = sheet_entries(16)
        .iter()
        .map(|(label, texture, tint)| {
            let albedo = shown_albedo(texture, *tint);
            let normal = (texture.normal.iter())
                .map(|&encoded| {
                    let z = decode_normal(encoded)[2];
                    [encoded[0], encoded[1], (z * 255.0) as u8, 255]
                })
                .collect();
            let view = |pixels: Vec<[u8; 4]>| View {
                size: 16,
                pixels,
                scale: 4,
                repeat: 1,
            };
            Cell {
                label: label.clone(),
                palette: Vec::new(),
                views: vec![
                    view(albedo.clone()),
                    view(normal),
                    view(grey(&texture.height)),
                    view(grey(&texture.roughness)),
                    View {
                        size: 16,
                        pixels: lit_preview(texture, &albedo),
                        scale: 2,
                        repeat: 2,
                    },
                ],
            }
        })
        .collect();
    for (page, cells) in cells.chunks(18).enumerate() {
        write_png(
            &dir.join(format!("channels_{page}.png")),
            &compose(cells, 3, 8),
        );
    }
}

/// Five other seeds of every texture: a recipe has to hold up for all of them, not only for
/// the one on the sheet.
fn seed_pages(dir: &Path) {
    let cells: Vec<Cell> = pack()
        .iter()
        .map(|entry| Cell {
            label: entry.label.clone(),
            palette: Vec::new(),
            views: (2..7)
                .map(|seed| View {
                    size: 16,
                    pixels: shown_albedo(&paint(entry, seed, 16), entry.tint),
                    scale: 4,
                    repeat: 1,
                })
                .collect(),
        })
        .collect();
    for (page, cells) in cells.chunks(18).enumerate() {
        write_png(
            &dir.join(format!("seeds_{page}.png")),
            &compose(cells, 3, 8),
        );
    }
}

/// The numbers behind the thresholds of the other tests, one line per texture.
fn metrics() -> String {
    let mut table = String::from(
        "texture | colours | coverage | height explained by colour | largest chroma | \
         lightness from..to | step across the edge / mean step inside (columns, rows)\n",
    );
    for entry in pack() {
        let texture = paint(&entry, SEED, 16);
        let colors = palette(&texture);
        let labs: Vec<_> = colors.iter().map(|&c| rgb8_to_oklab(c)).collect();
        let chroma = labs
            .iter()
            .map(|lab| (lab.a * lab.a + lab.b * lab.b).sqrt())
            .fold(0.0, det::max);
        let darkest = labs.iter().map(|lab| lab.l).fold(f64::INFINITY, det::min);
        let lightest = labs.iter().map(|lab| lab.l).fold(0.0, det::max);
        let explained: f64 = (0..RELIEF_SEEDS)
            .map(|seed| height_explained_by_colour(&paint(&entry, seed, 16)))
            .sum::<f64>()
            / RELIEF_SEEDS as f64;
        let seam = |columns: bool| {
            let (across, inside, _) = edge_steps(&entry, columns);
            across / inside
        };
        let _ = writeln!(
            table,
            "{} | {} | {:.3} | {explained:.2} | {chroma:.3} | {darkest:.2}..{lightest:.2} | \
             {:.2}, {:.2}",
            entry.label,
            colors.len(),
            coverage(&texture),
            seam(true),
            seam(false)
        );
    }
    table
}

/// Nothing is asserted about the art here: the test keeps the review reproducible.
#[test]
fn review_pages_are_written() {
    let dir = review_dir();
    std::fs::create_dir_all(&dir).expect("create the review directory");
    albedo_pages(&dir);
    channel_pages(&dir);
    seed_pages(&dir);
    std::fs::write(dir.join("metrics.txt"), metrics()).expect("write metrics");
    for name in [
        "albedo_0.png",
        "channels_0.png",
        "seeds_0.png",
        "metrics.txt",
    ] {
        let written = std::fs::metadata(dir.join(name)).expect("page exists");
        assert!(written.len() > 1000, "{name} is empty");
    }
}
