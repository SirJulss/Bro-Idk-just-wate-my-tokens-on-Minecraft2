//! Mip chains and average colours.

use cp_core::det;

use crate::color::{linear_to_srgb, srgb_to_linear};
use crate::texture::{ALPHA_CUTOFF, PbrTexture, decode_normal, encode_normal};

/// True when every channel has `size * size` texels and the texture is not empty.
fn well_formed(tex: &PbrTexture) -> bool {
    let texels = (tex.size as usize).checked_mul(tex.size as usize);
    tex.size > 0
        && texels == Some(tex.albedo.len())
        && texels == Some(tex.normal.len())
        && texels == Some(tex.roughness.len())
        && texels == Some(tex.emissive.len())
        && texels == Some(tex.height.len())
}

/// What the alpha channel of a texture means, which decides how [`mip_chain`] filters it.
///
/// The textures of this crate are [`AlphaMode::Coverage`] when their recipe is
/// [`crate::RecipeInfo::cutout`] and [`AlphaMode::Opaque`] otherwise
/// ([`crate::RecipeInfo::alpha_mode`]); textures loaded from files say what they are.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum AlphaMode {
    /// Alpha means nothing: every texel counts fully in every average and every level,
    /// the first one included, is fully opaque (alpha 255).
    Opaque,
    /// Alpha is coverage for an alpha test against [`ALPHA_CUTOFF`] (cutout geometry): colour
    /// and the other channels are weighted by alpha, and every level keeps the share of
    /// covered texels of the base level.
    Coverage,
    /// Alpha is opacity for blending (translucent geometry): it is averaged plainly, so the
    /// mean opacity is the same at every level, and colour and the other channels are
    /// weighted by alpha.
    Blend,
}

/// The full mip chain of a texture: element `i` is mip level `i`, starting with a copy of
/// `tex` itself and halving the edge length (rounding down) until it reaches one texel.
///
/// Every level is filtered directly from the base level with a box filter:
/// - colour is averaged in linear light, weighted by alpha unless `alpha` is
///   [`AlphaMode::Opaque`], so the colour hidden in the holes of a cutout texture does not
///   leak into its rims;
/// - normals are averaged as vectors and renormalised;
/// - roughness, emission and height are averaged (weighted like the colour).
///
/// **Alpha** follows `alpha`. With [`AlphaMode::Coverage`] (cutout textures only) the share
/// of covered texels is preserved: plain averaging would let the share of texels that pass
/// the alpha test drift from level to level, and foliage would visibly thin out or fill in
/// with distance. Every level keeps the base level's share of texels with
/// `alpha >= ALPHA_CUTOFF`, exactly up to the resolution of the level: the difference is at
/// most half a texel, `0.5 / (size * size)`. The texels kept are those with the highest
/// averaged alpha (ties go to the denser neighbourhood), and alpha is rescaled so that the
/// cutoff falls between the kept and the dropped ones; fully opaque textures stay fully
/// opaque. With [`AlphaMode::Blend`] alpha is the plain average, and with
/// [`AlphaMode::Opaque`] it is 255 everywhere.
///
/// A malformed texture (a channel whose length is not `size * size`, or size 0) yields only
/// the first level.
#[must_use]
pub fn mip_chain(tex: &PbrTexture, alpha: AlphaMode) -> Vec<PbrTexture> {
    let mut first = tex.clone();
    if alpha == AlphaMode::Opaque {
        for texel in &mut first.albedo {
            texel[3] = u8::MAX;
        }
    }
    if !well_formed(tex) {
        return vec![first];
    }
    let mut chain = vec![first];
    let texels = tex.albedo.len();
    let covered = tex.albedo.iter().filter(|t| t[3] >= ALPHA_CUTOFF).count();
    let coverage = covered as f64 / texels as f64;
    let opaque = tex.albedo.iter().all(|t| t[3] == u8::MAX);
    let mut size = tex.size / 2;
    while size >= 1 {
        let mut level = downsample(tex, size, alpha != AlphaMode::Opaque);
        if alpha == AlphaMode::Coverage && !opaque {
            preserve_coverage(&mut level, coverage);
        }
        chain.push(level);
        size /= 2;
    }
    chain
}

/// Box-filters `tex` down to `size × size` texels (`1 <= size <= tex.size`). With `weighted`
/// the channels are weighted by alpha and alpha is averaged; without it every texel counts
/// fully and the result is opaque.
fn downsample(tex: &PbrTexture, size: u32, weighted: bool) -> PbrTexture {
    let (from, to) = (tex.size as usize, size as usize);
    let mut out = PbrTexture {
        size,
        albedo: Vec::with_capacity(to * to),
        normal: Vec::with_capacity(to * to),
        roughness: Vec::with_capacity(to * to),
        emissive: Vec::with_capacity(to * to),
        height: Vec::with_capacity(to * to),
    };
    for y in 0..to {
        for x in 0..to {
            // The source block of this texel; blocks partition the base level.
            let (x0, x1) = (x * from / to, (x + 1) * from / to);
            let (y0, y1) = (y * from / to, (y + 1) * from / to);
            let mut block = Block::default();
            for sy in y0..y1 {
                for sx in x0..x1 {
                    block.add(tex, sy * from + sx);
                }
            }
            block.write(&mut out, weighted);
        }
    }
    out
}

/// Running sums over the source texels of one mip texel. Each quantity is summed twice:
/// weighted by alpha, and plainly for blocks that are fully transparent.
#[derive(Default)]
struct Block {
    count: f64,
    alpha: f64,
    weighted: Sums,
    plain: Sums,
}

#[derive(Default)]
struct Sums {
    color: [f64; 3],
    normal: [f64; 3],
    roughness: f64,
    emissive: f64,
    height: f64,
}

impl Sums {
    fn add(&mut self, tex: &PbrTexture, texel: usize, weight: f64) {
        let [r, g, b, _] = tex.albedo[texel];
        let normal = decode_normal(tex.normal[texel]);
        for (channel, value) in [r, g, b].into_iter().enumerate() {
            self.color[channel] += weight * srgb_to_linear(value);
            self.normal[channel] += weight * f64::from(normal[channel]);
        }
        self.roughness += weight * f64::from(tex.roughness[texel]);
        self.emissive += weight * f64::from(tex.emissive[texel]);
        self.height += weight * f64::from(tex.height[texel]);
    }
}

impl Block {
    fn add(&mut self, tex: &PbrTexture, texel: usize) {
        let alpha = f64::from(tex.albedo[texel][3]) / 255.0;
        self.count += 1.0;
        self.alpha += alpha;
        self.weighted.add(tex, texel, alpha);
        self.plain.add(tex, texel, 1.0);
    }

    fn write(&self, out: &mut PbrTexture, weighted: bool) {
        let (sums, weight) = if weighted && self.alpha > 0.0 {
            (&self.weighted, self.alpha)
        } else {
            (&self.plain, self.count)
        };
        let byte = |sum: f64| det::clamp(sum / weight + 0.5, 0.0, 255.0) as u8;
        let [r, g, b] = sums.color.map(|sum| linear_to_srgb(sum / weight));
        let alpha = if weighted {
            (self.alpha / self.count * 255.0 + 0.5) as u8
        } else {
            u8::MAX
        };
        out.albedo.push([r, g, b, alpha]);
        let [nx, ny, nz] = sums.normal;
        let length = (nx * nx + ny * ny + nz * nz).sqrt();
        out.normal.push(if length > 0.0 {
            encode_normal(nx / length, ny / length)
        } else {
            encode_normal(0.0, 0.0)
        });
        out.roughness.push(byte(sums.roughness));
        out.emissive.push(byte(sums.emissive));
        out.height.push(byte(sums.height));
    }
}

/// Rewrites the alpha of `level` so that the share of texels at or above the cutoff equals
/// `coverage` as closely as the number of texels allows.
fn preserve_coverage(level: &mut PbrTexture, coverage: f64) {
    let size = level.size as usize;
    let texels = level.albedo.len();
    let keep = ((coverage * texels as f64).round() as usize).min(texels);
    let alpha = |texel: usize| f64::from(level.albedo[texel][3]);
    // Sum of alpha over the 3×3 neighbourhood on the torus: among texels of equal alpha, the
    // ones inside denser foliage stay.
    let density = |texel: usize| {
        let (x, y) = (texel % size, texel / size);
        let mut sum = 0.0;
        for dy in [size - 1, 0, 1] {
            for dx in [size - 1, 0, 1] {
                sum += alpha(((y + dy) % size) * size + (x + dx) % size);
            }
        }
        sum
    };
    let mut order: Vec<usize> = (0..texels).collect();
    order.sort_by(|&a, &b| {
        alpha(b)
            .total_cmp(&alpha(a))
            .then(density(b).total_cmp(&density(a)))
            .then(a.cmp(&b))
    });

    // Scale alpha so that the cutoff falls between the last kept and the first dropped texel.
    let cutoff = f64::from(ALPHA_CUTOFF) - 0.5;
    let scale = if keep == 0 || keep == texels {
        1.0
    } else {
        let between = 0.5 * (alpha(order[keep - 1]) + alpha(order[keep]));
        if between > 0.0 { cutoff / between } else { 1.0 }
    };
    let rescaled: Vec<u8> = (0..texels)
        .map(|texel| det::clamp(alpha(texel) * scale + 0.5, 0.0, 255.0) as u8)
        .collect();
    for (rank, &texel) in order.iter().enumerate() {
        // Texels of equal alpha can end up on both sides; the clamp puts each on its own.
        level.albedo[texel][3] = if rank < keep {
            rescaled[texel].max(ALPHA_CUTOFF)
        } else {
            rescaled[texel].min(ALPHA_CUTOFF - 1)
        };
    }
}

/// Average colour of the texture in **linear** RGB, each channel in `[0, 1]`, weighted by
/// alpha (the colour hidden in the holes of a cutout texture does not count). This is the
/// colour the surface has from far away, before any tint.
///
/// A texture without texels, or with a malformed albedo channel, is averaged over the texels
/// it has; no texels at all give black. A fully transparent texture gives the plain average.
#[must_use]
pub fn average_color(tex: &PbrTexture) -> [f32; 3] {
    let mut weighted = [0.0_f64; 3];
    let mut plain = [0.0_f64; 3];
    let mut weight = 0.0;
    for texel in &tex.albedo {
        let alpha = f64::from(texel[3]) / 255.0;
        weight += alpha;
        for channel in 0..3 {
            let linear = srgb_to_linear(texel[channel]);
            weighted[channel] += alpha * linear;
            plain[channel] += linear;
        }
    }
    let count = tex.albedo.len() as f64;
    if weight > 0.0 {
        weighted.map(|sum| (sum / weight) as f32)
    } else if count > 0.0 {
        plain.map(|sum| (sum / count) as f32)
    } else {
        [0.0; 3]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cp_core::hash::Rng;

    fn solid(size: u32, albedo: [u8; 4]) -> PbrTexture {
        let texels = (size * size) as usize;
        PbrTexture {
            size,
            albedo: vec![albedo; texels],
            normal: vec![[128, 128]; texels],
            roughness: vec![200; texels],
            emissive: vec![10; texels],
            height: vec![99; texels],
        }
    }

    fn coverage(tex: &PbrTexture) -> f64 {
        let covered = tex.albedo.iter().filter(|t| t[3] >= ALPHA_CUTOFF).count();
        covered as f64 / tex.albedo.len() as f64
    }

    /// A cutout texture with blobby holes and binary alpha.
    fn cutout(size: u32, seed: u64, share: u32) -> PbrTexture {
        let mut tex = solid(size, [60, 140, 50, 255]);
        let mut rng = Rng::new(seed, 9);
        for y in 0..size {
            for x in 0..size {
                // 2×2 blocks of equal fate make the holes clump.
                let mut block = Rng::new(seed, u64::from(x / 2 + (y / 2) * size));
                if block.range_u32(0, 100) >= share || rng.range_u32(0, 100) < 6 {
                    tex.albedo[(y * size + x) as usize] = [20, 40, 20, 0];
                }
            }
        }
        tex
    }

    #[test]
    fn the_chain_halves_down_to_one_texel() {
        let chain = mip_chain(&solid(16, [10, 20, 30, 255]), AlphaMode::Coverage);
        let sizes: Vec<u32> = chain.iter().map(|level| level.size).collect();
        assert_eq!(sizes, [16, 8, 4, 2, 1]);
        for level in &chain {
            assert!(well_formed(level));
            // A uniform texture stays what it is at every level.
            assert!(level.albedo.iter().all(|&t| t == [10, 20, 30, 255]));
            assert!(level.normal.iter().all(|&n| n == [128, 128]));
            assert!(level.roughness.iter().all(|&r| r == 200));
            assert!(level.emissive.iter().all(|&e| e == 10));
            assert!(level.height.iter().all(|&h| h == 99));
        }
        assert_eq!(chain[0], solid(16, [10, 20, 30, 255]));
        // Odd sizes round down; a single texel is its own chain.
        let sizes: Vec<u32> = mip_chain(&solid(12, [1, 2, 3, 255]), AlphaMode::Coverage)
            .iter()
            .map(|level| level.size)
            .collect();
        assert_eq!(sizes, [12, 6, 3, 1]);
        assert_eq!(
            mip_chain(&solid(1, [1, 2, 3, 255]), AlphaMode::Coverage).len(),
            1
        );
    }

    #[test]
    fn malformed_textures_yield_only_themselves() {
        let mut broken = solid(8, [1, 2, 3, 255]);
        broken.height.pop();
        assert_eq!(
            mip_chain(&broken, AlphaMode::Coverage),
            vec![broken.clone()]
        );
        let mut empty = solid(4, [1, 2, 3, 255]);
        empty.size = 0;
        assert_eq!(mip_chain(&empty, AlphaMode::Coverage).len(), 1);
        let mut huge = solid(2, [1, 2, 3, 255]);
        huge.size = u32::MAX;
        assert_eq!(mip_chain(&huge, AlphaMode::Coverage).len(), 1);
    }

    #[test]
    fn colour_is_averaged_in_linear_light() {
        // A black and white checker averages to the sRGB value of 50 % light, not to 128.
        let mut tex = solid(4, [0, 0, 0, 255]);
        for (i, texel) in tex.albedo.iter_mut().enumerate() {
            if (i % 4 + i / 4) % 2 == 0 {
                *texel = [255, 255, 255, 255];
            }
        }
        let chain = mip_chain(&tex, AlphaMode::Coverage);
        let expected = linear_to_srgb(0.5);
        assert_eq!(expected, 188);
        assert!(chain[1].albedo.iter().all(|&t| t == [188, 188, 188, 255]));
        assert_eq!(chain[2].albedo, [[188, 188, 188, 255]]);
    }

    #[test]
    fn hidden_colour_does_not_leak_into_the_rims() {
        // Half of every block is a hole with a wildly different hidden colour.
        let mut tex = solid(4, [200, 100, 50, 255]);
        for (i, texel) in tex.albedo.iter_mut().enumerate() {
            if i % 2 == 0 {
                *texel = [0, 255, 255, 0];
            }
        }
        let chain = mip_chain(&tex, AlphaMode::Coverage);
        for level in &chain[1..] {
            for texel in &level.albedo {
                assert_eq!(texel[..3], [200, 100, 50]);
            }
        }
        // A block that is entirely hole keeps its hidden colour.
        let chain = mip_chain(&solid(4, [9, 99, 199, 0]), AlphaMode::Coverage);
        assert!(chain[1].albedo.iter().all(|&t| t == [9, 99, 199, 0]));
    }

    #[test]
    fn normals_are_averaged_as_vectors() {
        let mut tex = solid(2, [128, 128, 128, 255]);
        tex.normal = vec![[255, 128], [1, 128], [128, 255], [128, 1]];
        // Opposite tilts cancel: the average points straight up.
        assert_eq!(mip_chain(&tex, AlphaMode::Coverage)[1].normal, [[128, 128]]);
        tex.normal = vec![[200, 128]; 4];
        assert_eq!(mip_chain(&tex, AlphaMode::Coverage)[1].normal, [[200, 128]]);
    }

    #[test]
    fn coverage_is_preserved_to_half_a_texel() {
        for seed in 0..40 {
            for share in [35, 55, 72, 90] {
                let base = cutout(16, seed, share);
                let wanted = coverage(&base);
                let chain = mip_chain(&base, AlphaMode::Coverage);
                assert_eq!(chain.len(), 5);
                for level in &chain[1..] {
                    let texels = level.albedo.len() as f64;
                    let error = (coverage(level) - wanted).abs();
                    assert!(
                        error <= 0.5 / texels + 1.0e-12,
                        "seed {seed}, share {share}, size {}: {error}",
                        level.size
                    );
                }
            }
        }
    }

    #[test]
    fn plain_averaging_would_not_preserve_coverage() {
        // The reason the rescaling exists: without it the share of covered texels drifts.
        let mut worst: f64 = 0.0;
        for seed in 0..40 {
            let base = cutout(16, seed, 72);
            let plain = downsample(&base, 4, true);
            worst = det::max(worst, (coverage(&plain) - coverage(&base)).abs());
        }
        assert!(worst > 0.1, "{worst}");
    }

    #[test]
    fn rescaled_alpha_keeps_the_order_of_the_averages() {
        let base = cutout(16, 3, 72);
        let plain = downsample(&base, 8, true);
        let kept = &mip_chain(&base, AlphaMode::Coverage)[1];
        for (a, (pa, ka)) in plain.albedo.iter().zip(&kept.albedo).enumerate() {
            for (pb, kb) in plain.albedo.iter().zip(&kept.albedo).skip(a + 1) {
                // More averaged alpha never ends up with less final alpha.
                assert!(pa[3] <= pb[3] || ka[3] >= kb[3]);
                assert!(pa[3] >= pb[3] || ka[3] <= kb[3]);
            }
        }
        // Colour is untouched by the alpha rescaling.
        for (p, k) in plain.albedo.iter().zip(&kept.albedo) {
            assert_eq!(p[..3], k[..3]);
        }
    }

    #[test]
    fn opaque_and_empty_textures_keep_their_alpha() {
        for level in mip_chain(&solid(8, [5, 5, 5, 255]), AlphaMode::Coverage) {
            assert!(level.albedo.iter().all(|t| t[3] == 255));
        }
        for level in mip_chain(&solid(8, [5, 5, 5, 0]), AlphaMode::Coverage) {
            assert!(level.albedo.iter().all(|t| t[3] == 0));
        }
        // Translucent but everywhere above the cutoff: stays fully covered.
        for level in mip_chain(&solid(8, [5, 5, 5, 180]), AlphaMode::Coverage) {
            assert!(level.albedo.iter().all(|t| t[3] >= ALPHA_CUTOFF));
        }
    }

    #[test]
    fn average_colour_is_linear_and_alpha_weighted() {
        assert_eq!(
            average_color(&solid(4, [255, 255, 255, 255])),
            [1.0, 1.0, 1.0]
        );
        assert_eq!(average_color(&solid(4, [0, 0, 0, 255])), [0.0, 0.0, 0.0]);
        let mut tex = solid(2, [255, 0, 0, 255]);
        tex.albedo[0] = [0, 0, 255, 255];
        tex.albedo[1] = [0, 0, 255, 255];
        assert_eq!(average_color(&tex), [0.5, 0.0, 0.5]);
        // Holes do not count ...
        tex.albedo[0][3] = 0;
        tex.albedo[1][3] = 0;
        assert_eq!(average_color(&tex), [1.0, 0.0, 0.0]);
        // ... unless there is nothing else.
        assert_eq!(
            average_color(&solid(2, [255, 255, 255, 0])),
            [1.0, 1.0, 1.0]
        );
        let mut none = solid(1, [1, 1, 1, 255]);
        none.albedo.clear();
        assert_eq!(average_color(&none), [0.0, 0.0, 0.0]);
        // sRGB 188 is half of full light.
        let half = average_color(&solid(2, [188, 188, 188, 255]));
        assert!((half[0] - 0.5).abs() < 0.005);
    }

    fn mean_alpha(tex: &PbrTexture) -> f64 {
        tex.albedo.iter().map(|t| f64::from(t[3])).sum::<f64>() / tex.albedo.len() as f64
    }

    #[test]
    fn translucent_alpha_is_averaged_not_cut() {
        // A pane of alpha 90 in an opaque frame along two edges (mean alpha about 110).
        let mut pane = solid(16, [90, 140, 200, 90]);
        for i in 0..16 {
            pane.albedo[i] = [40, 40, 40, 255];
            pane.albedo[i * 16] = [40, 40, 40, 255];
        }
        let wanted = mean_alpha(&pane);
        for level in mip_chain(&pane, AlphaMode::Blend) {
            assert!((mean_alpha(&level) - wanted).abs() < 1.0, "{}", level.size);
        }
        // A checker of two opacities stays at their mean instead of 127 and 128.
        let mut checker = solid(16, [200, 200, 200, 100]);
        for (i, texel) in checker.albedo.iter_mut().enumerate() {
            if (i % 16 + i / 16) % 2 == 0 {
                texel[3] = 200;
            }
        }
        let chain = mip_chain(&checker, AlphaMode::Blend);
        for level in &chain[1..] {
            assert!(level.albedo.iter().all(|t| t[3] == 150), "{}", level.size);
        }
        // Colour stays weighted by alpha: the hidden colour of a hole does not leak.
        let mut holes = solid(4, [200, 100, 50, 255]);
        for texel in holes.albedo.iter_mut().step_by(2) {
            *texel = [0, 255, 255, 0];
        }
        let chain = mip_chain(&holes, AlphaMode::Blend);
        assert!(chain[1].albedo.iter().all(|t| *t == [200, 100, 50, 128]));
        // The coverage mode is the one that binarises.
        let cut = mip_chain(&checker, AlphaMode::Coverage);
        assert!(cut[1].albedo.iter().all(|t| t[3] != 150));
    }

    #[test]
    fn opaque_mode_ignores_alpha() {
        let mut tex = solid(4, [255, 255, 255, 255]);
        for texel in tex.albedo.iter_mut().step_by(2) {
            *texel = [0, 0, 0, 0];
        }
        let chain = mip_chain(&tex, AlphaMode::Opaque);
        assert_eq!(chain.len(), 3);
        for level in &chain {
            assert!(level.albedo.iter().all(|t| t[3] == 255));
        }
        // Every texel counts: black and white average to half of the light.
        assert!(chain[1].albedo.iter().all(|t| t[..3] == [188, 188, 188]));
        assert_eq!(chain[0].albedo[1], [255, 255, 255, 255]);
        // For an opaque texture the three modes agree.
        let opaque = solid(8, [10, 90, 30, 255]);
        let reference = mip_chain(&opaque, AlphaMode::Opaque);
        assert_eq!(mip_chain(&opaque, AlphaMode::Coverage), reference);
        assert_eq!(mip_chain(&opaque, AlphaMode::Blend), reference);
    }

    #[test]
    fn the_last_level_holds_the_average_colour() {
        let base = cutout(16, 5, 80);
        let chain = mip_chain(&base, AlphaMode::Coverage);
        let average = average_color(&base);
        let last = chain[4].albedo[0];
        for channel in 0..3 {
            let linear = srgb_to_linear(last[channel]) as f32;
            assert!(
                (linear - average[channel]).abs() < 0.01,
                "{last:?} {average:?}"
            );
        }
    }
}
