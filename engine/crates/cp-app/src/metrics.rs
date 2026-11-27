//! Measured criteria of a rendered image, printed by `cpe shot` for the beauty review
//! (contract §9): numbers that back up what the judges see. They do not replace a judge; they
//! make a washed-out, single-hued, flat, featureless or clipped frame visible at a glance and
//! comparable between versions and cameras. Since round 3 of the review of E2 they also carry
//! the judges' own measures: the share of muddy, over-saturated darks and of near-black pixels
//! (a dusk frame must keep its shadows readable) and the hue of the meadows in the lower part
//! of the frame (the medians of a set of views must spread, or every place has one green).

use std::fmt;

/// Hue bins of [`ImageMetrics::hue_variety`] (20 degrees each).
const HUE_BINS: usize = 18;

/// Measures of an sRGB-encoded RGBA8 image.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ImageMetrics {
    /// Colourfulness after Hasler and Süsstrunk (2003) on the encoded channels: about 15 is
    /// slightly, 33 moderately, 45 quite, 59 highly colourful.
    pub colourfulness: f64,
    /// RMS contrast: the standard deviation of the luma (0 to 1), at most 0.5.
    pub contrast: f64,
    /// Variety of colour: the entropy in bits of the hue histogram (18 bins of 20 degrees) of
    /// the pixels that have a colour (saturation and value at least 0.15); 0 for a single
    /// hue, at most log2(18) = 4.17 bits.
    pub hue_variety: f64,
    /// Share of the pixels that have a colour in that sense.
    pub coloured: f64,
    /// Fine detail: the mean absolute luma difference between neighbouring pixels, 0 to 1.
    pub detail: f64,
    /// Mean luma, 0 to 1.
    pub mean_luma: f64,
    /// Share of the pixels with luma at most 2 / 255 (crushed shadows).
    pub black: f64,
    /// Share of the pixels with luma at least 253 / 255 (blown highlights).
    pub white: f64,
    /// Share of the pixels that are dark and over-saturated (HSV saturation above 0.75 and
    /// value below 0.45): mud olive and ink green rather than shade.
    pub dark_saturated: f64,
    /// Share of the pixels with an HSV value below 0.2: near black, unreadable.
    pub very_dark: f64,
    /// Median hue (degrees) of the green pixels (hue 40 to 170 degrees, saturation and value at
    /// least 0.15) in the lower 60 percent of the image, where the land is; `None` without
    /// any.
    pub green_hue: Option<f64>,
    /// The spread of that hue: from its 10th to its 90th percentile, degrees.
    pub green_spread: f64,
}

impl ImageMetrics {
    /// Measures `rgba`, `width` x `height` pixels of four bytes, rows top to bottom. An image
    /// without pixels, or with fewer bytes than its size needs, measures all zero.
    pub fn measure(width: u32, height: u32, rgba: &[u8]) -> Self {
        let (width, height) = (width as usize, height as usize);
        let count = width * height;
        if count == 0 || rgba.len() < count * 4 {
            return Self::default();
        }
        let mut luma = Vec::with_capacity(count);
        let (mut rg_sum, mut rg_squares, mut yb_sum, mut yb_squares) = (0.0, 0.0, 0.0, 0.0);
        let (mut black, mut white, mut coloured) = (0_usize, 0_usize, 0_usize);
        let (mut dark_saturated, mut very_dark) = (0_usize, 0_usize);
        let mut greens = Vec::new();
        let land_from = height * 2 / 5;
        let mut hues = [0_usize; HUE_BINS];
        let (pixels, _) = rgba.as_chunks::<4>();
        for (index, pixel) in pixels.iter().take(count).enumerate() {
            let [r, g, b, _] = pixel.map(f64::from);
            let rg = r - g;
            let yb = 0.5 * (r + g) - b;
            rg_sum += rg;
            rg_squares += rg * rg;
            yb_sum += yb;
            yb_squares += yb * yb;
            let y = (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255.0;
            luma.push(y);
            black += usize::from(y <= 2.0 / 255.0);
            white += usize::from(y >= 253.0 / 255.0);
            if let Some(bin) = hue_bin(r, g, b) {
                hues[bin] += 1;
                coloured += 1;
            }
            let (hue, saturation, value) = hsv(r, g, b);
            dark_saturated += usize::from(saturation > 0.75 && value < 0.45);
            very_dark += usize::from(value < 0.2);
            if index / width >= land_from
                && (40.0..170.0).contains(&hue)
                && saturation >= 0.15
                && value >= 0.15
            {
                greens.push(hue);
            }
        }
        greens.sort_by(f64::total_cmp);
        let percentile = |p: f64| greens[((greens.len() - 1) as f64 * p).round() as usize];
        let (green_hue, green_spread) = if greens.is_empty() {
            (None, 0.0)
        } else {
            (Some(percentile(0.5)), percentile(0.9) - percentile(0.1))
        };
        let n = count as f64;
        let deviation = |sum: f64, squares: f64| (squares / n - (sum / n) * (sum / n)).max(0.0);
        let (rg_mean, yb_mean) = (rg_sum / n, yb_sum / n);
        let colourfulness = (deviation(rg_sum, rg_squares) + deviation(yb_sum, yb_squares)).sqrt()
            + 0.3 * (rg_mean * rg_mean + yb_mean * yb_mean).sqrt();
        let mean_luma = luma.iter().sum::<f64>() / n;
        let contrast = (luma.iter().map(|y| (y - mean_luma).powi(2)).sum::<f64>() / n).sqrt();
        let hue_variety = if coloured == 0 {
            0.0
        } else {
            hues.iter()
                .filter(|&&bin| bin > 0)
                .map(|&bin| {
                    let p = bin as f64 / coloured as f64;
                    -p * p.log2()
                })
                .sum()
        };
        let (mut differences, mut pairs) = (0.0, 0_usize);
        for y in 0..height {
            for x in 0..width {
                let here = luma[y * width + x];
                if x + 1 < width {
                    differences += (luma[y * width + x + 1] - here).abs();
                    pairs += 1;
                }
                if y + 1 < height {
                    differences += (luma[(y + 1) * width + x] - here).abs();
                    pairs += 1;
                }
            }
        }
        Self {
            colourfulness,
            contrast,
            hue_variety,
            coloured: coloured as f64 / n,
            detail: if pairs == 0 {
                0.0
            } else {
                differences / pairs as f64
            },
            mean_luma,
            black: black as f64 / n,
            white: white as f64 / n,
            dark_saturated: dark_saturated as f64 / n,
            very_dark: very_dark as f64 / n,
            green_hue,
            green_spread,
        }
    }
}

/// Hue (degrees, `0..360`), saturation and value (`0..=1`) of an encoded colour (channels
/// `0..=255`); a grey has hue 0.
fn hsv(r: f64, g: f64, b: f64) -> (f64, f64, f64) {
    let high = r.max(g).max(b);
    let low = r.min(g).min(b);
    let span = high - low;
    let saturation = if high > 0.0 { span / high } else { 0.0 };
    let hue = if span <= 0.0 {
        0.0
    } else if high == r {
        60.0 * ((g - b) / span).rem_euclid(6.0)
    } else if high == g {
        60.0 * ((b - r) / span + 2.0)
    } else {
        60.0 * ((r - g) / span + 4.0)
    };
    (hue, saturation, high / 255.0)
}

/// The 20-degree hue bin of an encoded colour, or `None` when its saturation or value is
/// below 0.15.
fn hue_bin(r: f64, g: f64, b: f64) -> Option<usize> {
    let high = r.max(g).max(b);
    let low = r.min(g).min(b);
    if high < 0.15 * 255.0 || high - low < 0.15 * high {
        return None;
    }
    let span = high - low;
    let sector = if high == r {
        ((g - b) / span).rem_euclid(6.0)
    } else if high == g {
        (b - r) / span + 2.0
    } else {
        (r - g) / span + 4.0
    };
    // Degrees / 20 = sector * 60 / 20.
    Some(((sector * 3.0) as usize).min(HUE_BINS - 1))
}

impl fmt::Display for ImageMetrics {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "colourfulness {:.1}, contrast {:.3}, hue variety {:.2} bits over {:.0} % coloured \
             pixels, detail {:.4}, mean luma {:.3}, black {:.2} %, white {:.2} %, dark \
             saturated {:.1} %, very dark {:.1} %",
            self.colourfulness,
            self.contrast,
            self.hue_variety,
            100.0 * self.coloured,
            self.detail,
            self.mean_luma,
            100.0 * self.black,
            100.0 * self.white,
            100.0 * self.dark_saturated,
            100.0 * self.very_dark
        )?;
        match self.green_hue {
            Some(hue) => write!(
                f,
                ", meadow hue {hue:.0} deg (spread {:.0} deg)",
                self.green_spread
            ),
            None => write!(f, ", no meadow"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn image(width: u32, height: u32, pixel: impl Fn(u32, u32) -> [u8; 3]) -> Vec<u8> {
        (0..height)
            .flat_map(|y| (0..width).map(move |x| (x, y)))
            .flat_map(|(x, y)| {
                let [r, g, b] = pixel(x, y);
                [r, g, b, 255]
            })
            .collect()
    }

    #[test]
    fn a_grey_image_has_no_colour_contrast_or_detail() {
        let metrics = ImageMetrics::measure(8, 4, &image(8, 4, |_, _| [128, 128, 128]));
        assert_eq!(metrics.colourfulness, 0.0);
        assert_eq!(metrics.contrast, 0.0);
        assert_eq!(metrics.hue_variety, 0.0);
        assert_eq!(metrics.coloured, 0.0);
        assert_eq!(metrics.detail, 0.0);
        assert!((metrics.mean_luma - 128.0 / 255.0).abs() < 1e-9);
        assert_eq!((metrics.black, metrics.white), (0.0, 0.0));
    }

    #[test]
    fn two_hues_carry_one_bit_and_strong_colour() {
        let pixels = image(
            10,
            10,
            |x, _| if x < 5 { [220, 30, 30] } else { [30, 40, 220] },
        );
        let metrics = ImageMetrics::measure(10, 10, &pixels);
        assert!((metrics.hue_variety - 1.0).abs() < 1e-9, "{metrics}");
        assert_eq!(metrics.coloured, 1.0);
        assert!(metrics.colourfulness > 59.0, "{metrics}");
        // One column boundary: 10 of the 180 neighbour pairs differ.
        assert!(metrics.detail > 0.0 && metrics.detail < 0.05, "{metrics}");
    }

    #[test]
    fn a_checkerboard_is_all_contrast_detail_and_clipping() {
        let pixels = image(6, 6, |x, y| {
            if (x + y) % 2 == 0 {
                [255, 255, 255]
            } else {
                [0, 0, 0]
            }
        });
        let metrics = ImageMetrics::measure(6, 6, &pixels);
        assert!((metrics.contrast - 0.5).abs() < 1e-9);
        assert!((metrics.detail - 1.0).abs() < 1e-9);
        assert_eq!((metrics.black, metrics.white), (0.5, 0.5));
        assert_eq!(metrics.hue_variety, 0.0);
    }

    #[test]
    fn hue_bins_follow_the_colour_wheel() {
        assert_eq!(hue_bin(255.0, 0.0, 0.0), Some(0));
        assert_eq!(hue_bin(255.0, 255.0, 0.0), Some(3));
        assert_eq!(hue_bin(0.0, 255.0, 0.0), Some(6));
        assert_eq!(hue_bin(0.0, 255.0, 255.0), Some(9));
        assert_eq!(hue_bin(0.0, 0.0, 255.0), Some(12));
        assert_eq!(hue_bin(255.0, 0.0, 255.0), Some(15));
        assert_eq!(hue_bin(255.0, 0.0, 10.0), Some(17));
        // Too grey or too dark to count.
        assert_eq!(hue_bin(200.0, 190.0, 185.0), None);
        assert_eq!(hue_bin(30.0, 0.0, 0.0), None);
    }

    #[test]
    fn dark_saturated_very_dark_and_meadow_hues_are_counted() {
        // A row of black, four rows of mud olive (dark and saturated, 61 degrees), five rows of
        // a meadow green of 90 degrees.
        let pixels = image(10, 10, |_, y| match y {
            0 => [0, 0, 0],
            1..=4 => [83, 84, 12],
            _ => [118, 150, 86],
        });
        let metrics = ImageMetrics::measure(10, 10, &pixels);
        assert!((metrics.dark_saturated - 0.4).abs() < 1e-9, "{metrics}");
        assert!((metrics.very_dark - 0.1).abs() < 1e-9, "{metrics}");
        // The lower 60 percent are rows 4 to 9: one olive row, five meadow rows.
        let hue = metrics.green_hue.expect("greens");
        assert!((hue - 90.0).abs() < 0.5, "{metrics}");
        assert!(metrics.green_spread > 25.0, "{metrics}");
        assert!(format!("{metrics}").contains("meadow hue 90 deg"));
        let grey = ImageMetrics::measure(4, 4, &image(4, 4, |_, _| [128, 128, 128]));
        assert_eq!(grey.green_hue, None);
        assert_eq!((grey.dark_saturated, grey.very_dark), (0.0, 0.0));
        assert_eq!(hsv(255.0, 0.0, 0.0), (0.0, 1.0, 1.0));
        assert_eq!(hsv(0.0, 255.0, 0.0).0, 120.0);
        assert_eq!(hsv(0.0, 0.0, 255.0).0, 240.0);
    }

    #[test]
    fn short_or_empty_images_measure_zero() {
        assert_eq!(ImageMetrics::measure(0, 0, &[]), ImageMetrics::default());
        assert_eq!(
            ImageMetrics::measure(4, 4, &[0; 12]),
            ImageMetrics::default()
        );
    }
}
