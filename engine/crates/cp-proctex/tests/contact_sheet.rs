//! The public API as the application uses it: paints the whole pack and writes its contact
//! sheet to `engine/target/proctex/contact_sheet.png`.

use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::path::{Path, PathBuf};

use cp_core::det;
use cp_proctex::{
    ALPHA_CUTOFF, ParamKind, Params, Tint, average_color, contact_sheet, decode_normal, generate,
    mip_chain, presets, recipe_info, recipes, tint_weight,
};

/// `engine/target/proctex`, next to the build output of the workspace.
fn output_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/proctex")
}

#[test]
fn the_contact_sheet_is_written_as_a_png() {
    let (width, height, pixels) = contact_sheet(16, 8);
    assert_eq!(pixels.len(), width as usize * height as usize * 4);
    // Every texture magnified eight times next to its tiling makes for a large sheet.
    assert!(width > 1000 && height > 1000, "{width} x {height}");

    let path = output_dir().join("contact_sheet.png");
    std::fs::create_dir_all(output_dir()).expect("create the output directory");
    let file = File::create(&path).expect("create the sheet");
    let mut encoder = png::Encoder::new(BufWriter::new(file), width, height);
    encoder.set_color(png::ColorType::Rgba);
    encoder.set_depth(png::BitDepth::Eight);
    let mut writer = encoder.write_header().expect("write the header");
    writer.write_image_data(&pixels).expect("write the pixels");
    writer.finish().expect("finish the file");

    // The file on disk is the sheet, pixel for pixel.
    let file = File::open(&path).expect("open the sheet");
    let mut reader = png::Decoder::new(BufReader::new(file))
        .read_info()
        .expect("read the header");
    let mut decoded = vec![0; reader.output_buffer_size().expect("buffer size")];
    let info = reader.next_frame(&mut decoded).expect("read the pixels");
    assert_eq!((info.width, info.height), (width, height));
    assert_eq!(info.color_type, png::ColorType::Rgba);
    assert_eq!(info.bit_depth, png::BitDepth::Eight);
    assert!(decoded[..info.buffer_size()] == pixels[..], "pixels differ");
}

#[test]
fn a_content_pack_can_drive_everything_through_text() {
    // What the glue between content and renderer does for one texture definition:
    // recipe + parameters (as text) + seed in, texture and mips out.
    let mut params = Params::new();
    for (key, text) in [
        ("bark", "#e2decc"),
        ("marks", "#423c3a"),
        ("style", "birch"),
    ] {
        params.set_text(key, text).expect("well-formed value");
    }
    let texture = generate("log_side", &params, 42, 16).expect("valid request");
    let alpha = recipe_info("log_side").expect("a recipe").alpha_mode();
    let chain = mip_chain(&texture, alpha);
    assert_eq!(chain.len(), 5);
    assert_eq!(chain[0], texture);
    let [r, g, b] = average_color(&texture);
    assert!(r > g && g > b, "birch bark is a warm white: {r} {g} {b}");

    // Mistakes in a pack are reported with the recipe and the key.
    let mut typo = Params::new();
    typo.set_text("barc", "#e2decc").expect("well-formed value");
    let error = generate("log_side", &typo, 42, 16).expect_err("unknown key");
    assert_eq!(
        error.to_string(),
        "recipe `log_side` has no parameter `barc`"
    );
    assert!(Params::new().set_text("bark", "#e2de").is_err());
    assert!(generate("log_sides", &Params::new(), 42, 16).is_err());
    assert!(generate("log_side", &Params::new(), 42, 20).is_err());
}

#[test]
fn every_recipe_and_preset_is_documented_and_paints() {
    assert!(recipes().len() >= 22);
    for name in recipes() {
        let info = recipe_info(name).expect("listed recipes are documented");
        let texture = generate(name, &Params::new(), 0, 16).expect("defaults are valid");
        // The description matches what is painted.
        let holes = texture.albedo.iter().any(|texel| texel[3] < ALPHA_CUTOFF);
        assert_eq!(holes, info.cutout, "{name}");
        // Every documented variant is accepted.
        for param in info.params {
            if let ParamKind::Variant(names) = param.kind {
                for variant in names {
                    let params = Params::new().with_variant(param.key, variant);
                    assert!(generate(name, &params, 0, 16).is_ok(), "{name}.{variant}");
                }
            }
        }
        // Normals decode to unit vectors pointing out of the surface.
        for &encoded in &texture.normal {
            let [x, y, z] = decode_normal(encoded);
            assert!(
                (x * x + y * y + z * z - 1.0).abs() < 1.0e-4 && z > 0.0,
                "{name}"
            );
        }
    }
    for preset in presets() {
        let texture = generate(preset.recipe, &preset.to_params(), 0, 16)
            .unwrap_or_else(|error| panic!("{}: {error}", preset.name));
        assert_eq!(texture.size, 16);
    }
}

#[test]
fn tinted_textures_come_out_in_the_colour_of_their_tint() {
    // The renderer's side of the tint convention, in linear RGB.
    let tinted_average = |name: &str, tint: Tint| {
        let texture = generate(name, &Params::new(), 3, 16).expect("valid request");
        // Roughly the display's transfer function; the test only compares channels.
        let to_linear = |c: u8| det::powf(f32::from(c) / 255.0, 2.2);
        let tint = tint.reference_color().map(to_linear);
        let mut sum = [0.0_f32; 3];
        for texel in &texture.albedo {
            let linear = [texel[0], texel[1], texel[2]].map(to_linear);
            let weight = tint_weight(linear);
            for channel in 0..3 {
                sum[channel] += linear[channel] * (1.0 + (tint[channel] - 1.0) * weight);
            }
        }
        sum.map(|channel| channel / texture.albedo.len() as f32)
    };
    let [r, g, b] = tinted_average("grass_top", Tint::Grass);
    assert!(g > r && r > b, "grass is green: {r} {g} {b}");
    let [r, g, b] = tinted_average("water", Tint::Water);
    assert!(b > g && g > r, "water is blue: {r} {g} {b}");
    // The side of a grass block stays mostly soil-coloured.
    let [r, g, b] = tinted_average("grass_side", Tint::Grass);
    assert!(r > b && g > b, "soil with a green fringe: {r} {g} {b}");
    // Untinted, the same textures are close to neutral.
    let grass = generate("grass_top", &Params::new(), 3, 16).expect("valid request");
    let [r, g, b] = average_color(&grass);
    assert!((r - g).abs() < 0.05 && (g - b).abs() < 0.08, "{r} {g} {b}");
    assert_eq!(
        recipe_info("grass_top").map(|info| info.tint),
        Some(Tint::Grass)
    );
}
