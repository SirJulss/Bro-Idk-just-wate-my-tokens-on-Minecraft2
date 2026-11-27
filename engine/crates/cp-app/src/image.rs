//! PNG files: writing RGBA8 images and reading any 8- or 16-bit PNG as RGBA8.

use std::io::Cursor;

/// Encodes `pixels` (`width * height` RGBA8 texels, top row first) as a PNG file.
pub(crate) fn encode_rgba(width: u32, height: u32, pixels: &[u8]) -> Result<Vec<u8>, String> {
    encode_rgba_with(width, height, pixels, false)
}

/// [`encode_rgba`] with the PNG `sRGB` chunk (perceptual intent) when `srgb` is set: the
/// pixels are sRGB-encoded, as rendered images read back from an sRGB target are.
pub(crate) fn encode_rgba_with(
    width: u32,
    height: u32,
    pixels: &[u8],
    srgb: bool,
) -> Result<Vec<u8>, String> {
    let expected = u64::from(width) * u64::from(height) * 4;
    if pixels.len() as u64 != expected {
        return Err(format!(
            "{} bytes for a {width} x {height} RGBA image, expected {expected}",
            pixels.len()
        ));
    }
    let mut bytes = Vec::new();
    let mut encoder = png::Encoder::new(&mut bytes, width, height);
    encoder.set_color(png::ColorType::Rgba);
    encoder.set_depth(png::BitDepth::Eight);
    if srgb {
        encoder.set_source_srgb(png::SrgbRenderingIntent::Perceptual);
    }
    let mut writer = encoder.write_header().map_err(|error| error.to_string())?;
    writer
        .write_image_data(pixels)
        .map_err(|error| error.to_string())?;
    writer.finish().map_err(|error| error.to_string())?;
    Ok(bytes)
}

/// Decodes a PNG file to `(width, height, texels)`, texels as sRGB `[r, g, b, a]`, top row
/// first. Palettes, grey and 16-bit channels are converted; missing alpha becomes 255.
/// Only the first frame of an animated PNG is read.
pub(crate) fn decode_rgba(bytes: &[u8]) -> Result<(u32, u32, Vec<[u8; 4]>), String> {
    let mut decoder = png::Decoder::new(Cursor::new(bytes));
    decoder.set_transformations(png::Transformations::EXPAND | png::Transformations::STRIP_16);
    let mut reader = decoder.read_info().map_err(|error| error.to_string())?;
    let length = reader
        .output_buffer_size()
        .ok_or_else(|| "the image is too large".to_owned())?;
    let mut buffer = vec![0; length];
    let info = reader
        .next_frame(&mut buffer)
        .map_err(|error| error.to_string())?;
    if info.bit_depth != png::BitDepth::Eight {
        return Err(format!("unexpected bit depth {:?}", info.bit_depth));
    }
    let data = &buffer[..info.buffer_size()];
    let texels: Vec<[u8; 4]> = match info.color_type {
        png::ColorType::Rgba => data.as_chunks::<4>().0.to_vec(),
        png::ColorType::Rgb => data
            .as_chunks::<3>()
            .0
            .iter()
            .map(|&[r, g, b]| [r, g, b, 255])
            .collect(),
        png::ColorType::GrayscaleAlpha => data
            .as_chunks::<2>()
            .0
            .iter()
            .map(|&[grey, alpha]| [grey, grey, grey, alpha])
            .collect(),
        png::ColorType::Grayscale => data.iter().map(|&grey| [grey, grey, grey, 255]).collect(),
        // `EXPAND` turns palette images into RGB or RGBA.
        png::ColorType::Indexed => return Err("palette image was not expanded".to_owned()),
    };
    let expected = u64::from(info.width) * u64::from(info.height);
    if texels.len() as u64 != expected {
        return Err(format!(
            "decoded {} texels for {} x {} px",
            texels.len(),
            info.width,
            info.height
        ));
    }
    Ok((info.width, info.height, texels))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rgba_images_survive_a_round_trip() {
        let pixels: Vec<u8> = (0..3 * 2 * 4).map(|value| (value * 11) as u8).collect();
        let png = encode_rgba(3, 2, &pixels).expect("encodes");
        let (width, height, texels) = decode_rgba(&png).expect("decodes");
        assert_eq!((width, height), (3, 2));
        assert_eq!(texels.concat(), pixels);
    }

    #[test]
    fn grey_and_rgb_images_become_rgba() {
        let encode = |color: png::ColorType, data: &[u8]| {
            let mut bytes = Vec::new();
            let mut encoder = png::Encoder::new(&mut bytes, 2, 1);
            encoder.set_color(color);
            encoder.set_depth(png::BitDepth::Eight);
            let mut writer = encoder.write_header().expect("header");
            writer.write_image_data(data).expect("pixels");
            writer.finish().expect("finish");
            bytes
        };
        let rgb = encode(png::ColorType::Rgb, &[1, 2, 3, 4, 5, 6]);
        assert_eq!(
            decode_rgba(&rgb).expect("decodes").2,
            [[1, 2, 3, 255], [4, 5, 6, 255]]
        );
        let grey = encode(png::ColorType::GrayscaleAlpha, &[7, 8, 9, 10]);
        assert_eq!(
            decode_rgba(&grey).expect("decodes").2,
            [[7, 7, 7, 8], [9, 9, 9, 10]]
        );
    }

    #[test]
    fn srgb_images_carry_the_srgb_chunk() {
        let pixels = [10_u8, 20, 30, 255, 40, 50, 60, 255];
        let tagged = encode_rgba_with(2, 1, &pixels, true).expect("encodes");
        let plain = encode_rgba_with(2, 1, &pixels, false).expect("encodes");
        let has_chunk = |bytes: &[u8]| bytes.windows(4).any(|window| window == b"sRGB");
        assert!(has_chunk(&tagged));
        assert!(!has_chunk(&plain));
        let reader = png::Decoder::new(Cursor::new(tagged))
            .read_info()
            .expect("a PNG header");
        assert_eq!(
            reader.info().srgb,
            Some(png::SrgbRenderingIntent::Perceptual)
        );
        assert_eq!(decode_rgba(&plain).expect("decodes").2.concat(), pixels);
    }

    #[test]
    fn broken_input_is_an_error() {
        assert!(decode_rgba(b"not a png").is_err());
        assert!(encode_rgba(2, 2, &[0; 15]).is_err());
    }
}
