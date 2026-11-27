//! The texture type and its channel encodings.

use cp_core::det;

/// A square PBR texture that tiles seamlessly on both axes.
///
/// Every channel holds `size * size` texels in row-major order, row 0 at the **top** of the
/// image; texel `(x, y)` is at index `y * size + x`. Image right is `tex_u`, image up is
/// `tex_v` (contract §2.3).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PbrTexture {
    /// Edge length in texels.
    pub size: u32,
    /// Base colour as sRGB-encoded `[r, g, b]` plus alpha. Alpha is coverage for cutout
    /// textures (0 = hole, 255 = solid; see [`ALPHA_CUTOFF`]) and 255 everywhere else. The
    /// colour of a hole is the material's shadow tone, so filtering across the outline does
    /// not produce a halo.
    pub albedo: Vec<[u8; 4]>,
    /// Tangent-space normal, `x` and `y` only; decode with [`decode_normal`]. `x` points along
    /// image right, `y` along image up, `z` out of the surface. A flat texel is `[128, 128]`.
    pub normal: Vec<[u8; 2]>,
    /// Perceptual roughness, `value / 255`: 0 is a mirror, 255 fully diffuse.
    pub roughness: Vec<u8>,
    /// Emitted light as a fraction of the material's emissive strength, `value / 255`. The
    /// colour of the light is the albedo.
    pub emissive: Vec<u8>,
    /// Height above the lowest point of the surface, `value / 255`; the normals are the slope
    /// of this map.
    pub height: Vec<u8>,
}

/// Alpha values at or above this count as covered (about 0.5). The mip chain preserves the
/// share of covered texels with respect to this threshold; alpha tests should use the same.
pub const ALPHA_CUTOFF: u8 = 128;

/// Decodes a texel of [`PbrTexture::normal`] to a unit vector `[x, y, z]` with `z >= 0`.
///
/// Each stored byte `b` is `round(n * 127) + 128`, so `n = (b - 128) / 127` (the byte 0 does
/// not occur and decodes like 1). `z` is reconstructed as `sqrt(1 - x² - y²)`.
#[must_use]
pub fn decode_normal(encoded: [u8; 2]) -> [f32; 3] {
    let axis = |byte: u8| det::max((f32::from(byte) - 128.0) / 127.0, -1.0);
    let (x, y) = (axis(encoded[0]), axis(encoded[1]));
    let z = det::max(1.0 - x * x - y * y, 0.0).sqrt();
    [x, y, z]
}

/// Encodes the `x` and `y` components of a unit normal, the inverse of [`decode_normal`].
pub(crate) fn encode_normal(x: f64, y: f64) -> [u8; 2] {
    let axis = |n: f64| ((det::clamp(n, -1.0, 1.0) * 127.0).round() + 128.0) as u8;
    [axis(x), axis(y)]
}

/// Normal map of a height map on the torus.
///
/// `relief` is the height difference between the values 0 and 255 in design units (sixteenths
/// of the tile); slopes are central differences, so a step between two neighbouring texels
/// tilts both of them.
pub(crate) fn normals_from_height(height: &[u8], size: usize, relief: f64) -> Vec<[u8; 2]> {
    // Texels are 16 / size units apart and a central difference spans two of them.
    let gain = relief * size as f64 / (32.0 * 255.0);
    let at = |x: usize, y: usize| f64::from(height[(y % size) * size + x % size]);
    let mut normals = Vec::with_capacity(size * size);
    for y in 0..size {
        for x in 0..size {
            // Rise per unit towards image right and towards image up (row 0 is the top).
            let slope_u = (at(x + 1, y) - at(x + size - 1, y)) * gain;
            let slope_v = (at(x, y + size - 1) - at(x, y + 1)) * gain;
            let scale = 1.0 / (slope_u * slope_u + slope_v * slope_v + 1.0).sqrt();
            normals.push(encode_normal(-slope_u * scale, -slope_v * scale));
        }
    }
    normals
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normal_encoding_round_trips() {
        assert_eq!(encode_normal(0.0, 0.0), [128, 128]);
        assert_eq!(decode_normal([128, 128]), [0.0, 0.0, 1.0]);
        assert_eq!(encode_normal(1.0, -1.0), [255, 1]);
        assert_eq!(encode_normal(7.0, -7.0), [255, 1]);
        assert_eq!(decode_normal([255, 128]), [1.0, 0.0, 0.0]);
        assert_eq!(decode_normal([0, 128])[0], -1.0);
        for bx in 1..=255_u8 {
            for by in (1..=255_u8).step_by(7) {
                let [x, y, z] = decode_normal([bx, by]);
                assert!((-1.0..=1.0).contains(&x) && (-1.0..=1.0).contains(&y));
                assert!((0.0..=1.0).contains(&z));
                if x * x + y * y <= 1.0 {
                    assert!((x * x + y * y + z * z - 1.0).abs() < 1.0e-5);
                    assert_eq!(encode_normal(f64::from(x), f64::from(y)), [bx, by]);
                }
            }
        }
    }

    #[test]
    fn flat_height_gives_flat_normals() {
        for level in [0_u8, 97, 255] {
            let normals = normals_from_height(&[level; 64], 8, 2.0);
            assert!(normals.iter().all(|&n| n == [128, 128]));
        }
    }

    #[test]
    fn normals_lean_away_from_higher_ground() {
        // A ridge along the middle column: left of it the surface rises towards image right,
        // so the normal leans left (x < 0); right of it the other way round.
        let size = 8;
        let mut height = vec![0_u8; size * size];
        for y in 0..size {
            height[y * size + 4] = 255;
        }
        let normals = normals_from_height(&height, size, 1.0);
        for y in 0..size {
            assert!(normals[y * size + 3][0] < 128);
            assert!(normals[y * size + 5][0] > 128);
            assert_eq!(normals[y * size + 4], [128, 128]);
            assert_eq!(normals[y * size + 1], [128, 128]);
            assert!(normals.iter().all(|n| n[1] == 128));
        }
        // A ridge along a row: above it (smaller y) the surface rises towards image down, so
        // the normal leans up (y > 0).
        let mut height = vec![0_u8; size * size];
        for x in 0..size {
            height[4 * size + x] = 255;
        }
        let normals = normals_from_height(&height, size, 1.0);
        for x in 0..size {
            assert!(normals[3 * size + x][1] > 128);
            assert!(normals[5 * size + x][1] < 128);
        }
        // The slope wraps around the edges.
        let mut height = vec![0_u8; size * size];
        for y in 0..size {
            height[y * size] = 255;
        }
        let normals = normals_from_height(&height, size, 1.0);
        assert!(normals[size - 1][0] < 128 && normals[1][0] > 128);
    }

    #[test]
    fn relief_and_resolution_scale_the_slope() {
        // The same step is steeper with more relief, and a 45 degree slope decodes as such:
        // full height difference of 2 units across two texels of one unit each.
        let size = 16;
        let mut height = vec![0_u8; size * size];
        for y in 0..size {
            for x in 8..size {
                height[y * size + x] = 255;
            }
        }
        let gentle = normals_from_height(&height, size, 0.5);
        let steep = normals_from_height(&height, size, 2.0);
        assert!(steep[8][0] < gentle[8][0] && gentle[8][0] < 128);
        let [x, _, z] = decode_normal(steep[8]);
        assert!((x + z).abs() < 0.02, "{x} {z}");
    }
}
