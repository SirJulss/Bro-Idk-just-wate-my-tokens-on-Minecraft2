//! A minimal PNG writer for the images of the renderer tests: RGBA8, deflate "stored" blocks
//! (no compression), CRC-32 and Adler-32 computed here, so cp-render needs no image crate
//! (cp-app owns the `png` dependency).

/// The PNG file of `width` x `height` RGBA8 pixels (rows top to bottom), with the sRGB chunk
/// when `srgb` is set.
pub fn encode_rgba(width: u32, height: u32, pixels: &[u8], srgb: bool) -> Vec<u8> {
    assert_eq!(pixels.len(), width as usize * height as usize * 4);
    let row = width as usize * 4;
    let mut filtered = Vec::with_capacity((row + 1) * height as usize);
    for line in pixels.chunks(row) {
        // Filter type 0 (none) per row.
        filtered.push(0);
        filtered.extend_from_slice(line);
    }
    let mut png = b"\x89PNG\r\n\x1a\n".to_vec();
    let mut header = Vec::new();
    header.extend_from_slice(&width.to_be_bytes());
    header.extend_from_slice(&height.to_be_bytes());
    // 8 bits, colour type 6 (RGBA), deflate, adaptive filtering, no interlace.
    header.extend_from_slice(&[8, 6, 0, 0, 0]);
    chunk(&mut png, b"IHDR", &header);
    if srgb {
        // Rendering intent 0 (perceptual).
        chunk(&mut png, b"sRGB", &[0]);
    }
    chunk(&mut png, b"IDAT", &zlib_stored(&filtered));
    chunk(&mut png, b"IEND", &[]);
    png
}

fn chunk(out: &mut Vec<u8>, kind: &[u8; 4], data: &[u8]) {
    out.extend_from_slice(
        &u32::try_from(data.len())
            .expect("a PNG chunk fits u32")
            .to_be_bytes(),
    );
    let start = out.len();
    out.extend_from_slice(kind);
    out.extend_from_slice(data);
    let crc = crc32(&out[start..]);
    out.extend_from_slice(&crc.to_be_bytes());
}

/// A zlib stream of stored (uncompressed) deflate blocks.
fn zlib_stored(data: &[u8]) -> Vec<u8> {
    // CMF: deflate with a 32 KiB window; FLG: no dictionary, check bits make it a multiple of 31.
    let mut out = vec![0x78, 0x01];
    let blocks: Vec<&[u8]> = if data.is_empty() {
        vec![&[]]
    } else {
        data.chunks(65_535).collect()
    };
    let last = blocks.len() - 1;
    for (index, block) in blocks.into_iter().enumerate() {
        out.push(u8::from(index == last));
        let len = u16::try_from(block.len()).expect("stored blocks hold at most 65535 bytes");
        out.extend_from_slice(&len.to_le_bytes());
        out.extend_from_slice(&(!len).to_le_bytes());
        out.extend_from_slice(block);
    }
    out.extend_from_slice(&adler32(data).to_be_bytes());
    out
}

fn crc32(bytes: &[u8]) -> u32 {
    let mut crc = 0xFFFF_FFFF_u32;
    for &byte in bytes {
        crc ^= u32::from(byte);
        for _ in 0..8 {
            crc = if crc & 1 == 1 {
                (crc >> 1) ^ 0xEDB8_8320
            } else {
                crc >> 1
            };
        }
    }
    !crc
}

fn adler32(bytes: &[u8]) -> u32 {
    let (mut a, mut b) = (1_u32, 0_u32);
    for &byte in bytes {
        a = (a + u32::from(byte)) % 65_521;
        b = (b + a) % 65_521;
    }
    b << 16 | a
}

#[test]
fn checksums_match_known_values() {
    // Known values: CRC-32 of "123456789" and Adler-32 of "Wikipedia".
    assert_eq!(crc32(b"123456789"), 0xCBF4_3926);
    assert_eq!(adler32(b"Wikipedia"), 0x11E6_0398);
    let png = encode_rgba(2, 1, &[255, 0, 0, 255, 0, 255, 0, 255], true);
    assert_eq!(&png[..8], b"\x89PNG\r\n\x1a\n");
    assert_eq!(&png[12..16], b"IHDR");
}
