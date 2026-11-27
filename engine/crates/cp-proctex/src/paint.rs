//! A finished painting and its conversion into the texture channels.

use cp_core::det;

use crate::grid::Grid;
use crate::palette::Swatch;
use crate::pattern::DESIGN;
use crate::texture::{PbrTexture, normals_from_height};

/// What a recipe produces: palette indices plus a height field.
///
/// Colour, roughness and emission of a texel all come from its palette entry and the normals
/// are the slope of the height field, so the channels of the texture cannot disagree with
/// each other.
pub(crate) struct Painting {
    /// At most six entries, ordered from dark to light within each material.
    pub palette: Vec<Swatch>,
    /// Palette index of every texel.
    pub index: Grid<u8>,
    /// Height of every texel in `[0, 1]`. Steps may be sharp: above 16 px, [`Painting::finish`]
    /// softens them over about two design units, as sampling at 16 px does.
    pub height: Grid<f64>,
    /// Texels that are cut out (alpha 0), for cutout textures.
    pub holes: Option<Grid<bool>>,
    /// Height difference between 0 and 1 in design units (sixteenths of the tile); sets the
    /// strength of the normal map.
    pub relief: f64,
}

/// Converts a fraction in `[0, 1]` to a byte, rounding to nearest. NaN becomes 0.
fn to_byte(value: f64) -> u8 {
    (det::clamp(value, 0.0, 1.0) * 255.0 + 0.5) as u8
}

impl Painting {
    pub(crate) fn finish(&self) -> PbrTexture {
        let size = self.index.size();
        let last = self.palette.len().saturating_sub(1);
        let swatch = |index: u8| self.palette[usize::from(index).min(last)];
        let alpha = |texel: usize| match &self.holes {
            Some(holes) if holes.cells()[texel] => 0,
            _ => 255,
        };
        // Central differences span two texels: two design units at the native size, but only
        // 2 / n units at n texels per unit, where every step of the design would turn into a
        // cliff. A box of 2n - 1 texels (about two units) gives the relief back the slopes it
        // has at 16 px, so larger sizes show the same surface.
        let per_unit = (size / DESIGN).max(1);
        let height: Vec<u8> = (self.height.box_blurred(per_unit).cells().iter())
            .map(|&h| to_byte(h))
            .collect();
        PbrTexture {
            size: size as u32,
            albedo: (self.index.cells().iter().enumerate())
                .map(|(texel, &index)| {
                    let [r, g, b] = swatch(index).rgb;
                    [r, g, b, alpha(texel)]
                })
                .collect(),
            normal: normals_from_height(&height, size, self.relief),
            roughness: (self.index.cells().iter())
                .map(|&index| to_byte(swatch(index).roughness))
                .collect(),
            emissive: (self.index.cells().iter())
                .map(|&index| to_byte(swatch(index).emissive))
                .collect(),
            height,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn painting() -> Painting {
        Painting {
            palette: vec![
                Swatch {
                    rgb: [10, 20, 30],
                    roughness: 1.0,
                    emissive: 0.0,
                },
                Swatch {
                    rgb: [200, 180, 160],
                    roughness: 0.25,
                    emissive: 0.5,
                },
            ],
            index: Grid::from_fn(4, |x, _| u8::from(x >= 2)),
            height: Grid::from_fn(4, |x, _| if x >= 2 { 1.0 } else { 0.0 }),
            holes: None,
            relief: 1.0,
        }
    }

    #[test]
    fn channels_come_from_the_palette() {
        let texture = painting().finish();
        assert_eq!(texture.size, 4);
        for channel in [
            texture.albedo.len(),
            texture.normal.len(),
            texture.roughness.len(),
            texture.emissive.len(),
            texture.height.len(),
        ] {
            assert_eq!(channel, 16);
        }
        assert_eq!(texture.albedo[0], [10, 20, 30, 255]);
        assert_eq!(texture.albedo[3], [200, 180, 160, 255]);
        assert_eq!((texture.roughness[0], texture.roughness[3]), (255, 64));
        assert_eq!((texture.emissive[0], texture.emissive[3]), (0, 128));
        assert_eq!((texture.height[0], texture.height[3]), (0, 255));
        // The step between columns 1 and 2 tilts the normals of both towards the low side.
        assert!(texture.normal[1][0] < 128 && texture.normal[2][0] < 128);
        // ... and the step back down across the wrapped edge tilts them the other way.
        assert!(texture.normal[0][0] > 128 && texture.normal[3][0] > 128);
    }

    #[test]
    fn holes_become_transparent_texels_that_keep_their_colour() {
        let mut painting = painting();
        painting.holes = Some(Grid::from_fn(4, |x, y| x == 0 && y == 0));
        let texture = painting.finish();
        assert_eq!(texture.albedo[0], [10, 20, 30, 0]);
        assert_eq!(texture.albedo[1], [10, 20, 30, 255]);
        assert_eq!(texture.albedo.iter().filter(|t| t[3] == 0).count(), 1);
    }

    #[test]
    fn bad_values_are_clamped_instead_of_panicking() {
        let mut painting = painting();
        painting.index = Grid::filled(4, 9);
        painting.height = Grid::filled(4, f64::NAN);
        painting.palette[1].roughness = 7.0;
        let texture = painting.finish();
        assert!(texture.albedo.iter().all(|&t| t == [200, 180, 160, 255]));
        assert!(texture.roughness.iter().all(|&r| r == 255));
        assert!(texture.height.iter().all(|&h| h == 0));
        assert_eq!(to_byte(0.5), 128);
        assert_eq!(to_byte(-1.0), 0);
    }
}
