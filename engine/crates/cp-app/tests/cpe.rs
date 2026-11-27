//! The binary `cpe` run as a user runs it: arguments in, exit code, stdout, stderr and files
//! out.

// cp-app is the crate that prints (contract 2.6); a skipped test says so on stderr.
#![allow(clippy::print_stderr)]

use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use cp_content::Registry;

/// Runs `cpe` with `args` and without the environment variables that change its behaviour.
fn cpe(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_cpe"))
        .args(args)
        .env_remove("CP_LOG")
        .env_remove("CP_BACKEND")
        .env_remove("CP_DXC")
        .output()
        .expect("cpe starts")
}

fn stdout(output: &Output) -> String {
    String::from_utf8_lossy(&output.stdout).into_owned()
}

fn stderr(output: &Output) -> String {
    String::from_utf8_lossy(&output.stderr).into_owned()
}

/// A fresh directory below cargo's scratch directory for this test.
fn scratch(test: &str) -> PathBuf {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR"))
        .join(format!("cp-app-{test}-{}", std::process::id()));
    if dir.exists() {
        fs::remove_dir_all(&dir).expect("a stale scratch directory can be removed");
    }
    fs::create_dir_all(&dir).expect("the scratch directory can be created");
    dir
}

/// `engine/content/base`; CARGO_MANIFEST_DIR is `engine/crates/cp-app`.
fn base_pack() -> PathBuf {
    let engine = Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .expect("the crate lies two levels below the workspace");
    engine.join("content").join("base")
}

#[test]
fn without_arguments_and_with_help_the_usage_is_printed() {
    let plain = cpe(&[]);
    assert_eq!(plain.status.code(), Some(0));
    let text = stdout(&plain);
    for synopsis in [
        "cpe caps [--out <file.md>] [--json <file.json>]",
        "cpe proctex-sheet --out <file.png>",
        "cpe content-check [--pack <dir>]...",
        "cpe run [--scene <name>]",
        "cpe shot --scene <name> --out <file.png>",
        "cpe bench --scene <name> --out <file.json>",
    ] {
        assert!(text.contains(synopsis), "{synopsis}\n{text}");
    }
    for spelling in [&["--help"][..], &["-h"], &["help"]] {
        let output = cpe(spelling);
        assert_eq!(output.status.code(), Some(0), "{spelling:?}");
        assert_eq!(stdout(&output), text, "{spelling:?}");
    }
    let one = cpe(&["proctex-sheet", "--help"]);
    assert_eq!(one.status.code(), Some(0));
    assert!(stdout(&one).contains("--scale <n>"));
}

#[test]
fn bad_command_lines_fail_with_exit_code_2_and_a_message() {
    for args in [
        &["frobnicate"][..],
        &["--verbose"],
        &["caps", "--md", "x.md"],
        &["caps", "--out"],
        &["proctex-sheet"],
        &["proctex-sheet", "--out", "x.png", "--size", "20"],
        &["proctex-sheet", "--out", "x.png", "--scale", "many"],
        &["content-check", "extra"],
        &["help", "frobnicate"],
        &["run", "--scene", "moon"],
        &["shot", "--scene", "hills"],
        &[
            "shot", "--scene", "hills", "--out", "x.png", "--view", "pretty",
        ],
        &["bench", "--out", "x.json"],
    ] {
        let output = cpe(args);
        assert_eq!(output.status.code(), Some(2), "{args:?}");
        assert!(stdout(&output).is_empty(), "{args:?}");
        let message = stderr(&output);
        assert!(message.starts_with("error: "), "{args:?}: {message}");
        assert!(!message.contains("panicked"), "{args:?}: {message}");
    }
}

#[test]
fn proctex_sheet_writes_the_contact_sheet() {
    let dir = scratch("sheet");
    let path = dir.join("nested").join("sheet.png");
    let path_text = path.to_str().expect("a Unicode path");
    let output = cpe(&["proctex-sheet", "--out", path_text, "--scale", "1"]);
    assert_eq!(output.status.code(), Some(0), "{}", stderr(&output));
    assert!(stdout(&output).contains("wrote"));

    let (width, height, pixels) = cp_proctex::contact_sheet(16, 1);
    let file = fs::File::open(&path).expect("the sheet exists");
    let mut reader = png::Decoder::new(std::io::BufReader::new(file))
        .read_info()
        .expect("a PNG header");
    let mut decoded = vec![0; reader.output_buffer_size().expect("a buffer size")];
    let info = reader.next_frame(&mut decoded).expect("PNG pixels");
    assert_eq!((info.width, info.height), (width, height));
    assert_eq!(info.color_type, png::ColorType::Rgba);
    assert!(decoded[..info.buffer_size()] == pixels[..], "pixels differ");
    fs::remove_dir_all(&dir).expect("the scratch directory can be removed");
}

#[test]
fn content_check_passes_on_the_base_pack() {
    let output = cpe(&["content-check"]);
    assert_eq!(output.status.code(), Some(0), "{}", stderr(&output));
    let text = stdout(&output);
    let registry = Registry::load(&[base_pack().as_path()]).expect("the base pack loads");
    assert!(
        text.contains(&format!("content hash: {:#018x}", registry.content_hash())),
        "{text}"
    );
    assert!(
        text.contains(&format!("{:<14}{}\n", "states:", registry.state_count())),
        "{text}"
    );
    for material in registry.materials() {
        assert!(text.contains(&material.key), "{}", material.key);
    }
    // Naming the base pack explicitly gives the same report.
    let pack = base_pack();
    let explicit = cpe(&["content-check", "--pack", pack.to_str().expect("Unicode")]);
    assert_eq!(explicit.status.code(), Some(0), "{}", stderr(&explicit));
    let hash_line = |text: &str| {
        text.lines()
            .find(|line| line.starts_with("content hash:"))
            .map(str::to_owned)
    };
    assert_eq!(hash_line(&stdout(&explicit)), hash_line(&text));
}

#[test]
fn content_check_fails_when_a_texture_cannot_be_made() {
    let dir = scratch("broken-pack");
    let pack = dir.join("broken");
    fs::create_dir_all(pack.join("textures")).expect("a pack directory");
    fs::write(
        pack.join("pack.toml"),
        "format = 1\nname = \"broken\"\nversion = \"1.0\"\n",
    )
    .expect("pack.toml");
    // Overrides two textures of the base pack: one with a recipe that does not exist, one
    // with a variant its recipe does not know.
    fs::write(
        pack.join("textures").join("broken.toml"),
        "[[texture]]\nkey = \"cubeplanet:stone\"\nprovenance = \"own\"\n\
         procedural = { recipe = \"no_such_recipe\", seed = 1 }\n\n\
         [[texture]]\nkey = \"cubeplanet:ore_coal\"\nprovenance = \"own\"\n\
         procedural = { recipe = \"ore\", seed = 1, params = { style = \"bubbles\" } }\n",
    )
    .expect("a texture file");
    let base = base_pack();
    let output = cpe(&[
        "content-check",
        "--pack",
        base.to_str().expect("Unicode"),
        "--pack",
        pack.to_str().expect("Unicode"),
    ]);
    assert_eq!(output.status.code(), Some(1), "{}", stdout(&output));
    let message = stderr(&output);
    assert!(message.contains("2 of"), "{message}");
    assert!(message.contains("`cubeplanet:stone`"), "{message}");
    assert!(message.contains("no_such_recipe"), "{message}");
    assert!(message.contains("`cubeplanet:ore_coal`"), "{message}");
    assert!(message.contains("bubbles"), "{message}");
    assert!(stdout(&output).is_empty());

    // A pack that does not load at all fails as well.
    let missing = dir.join("missing");
    let output = cpe(&[
        "content-check",
        "--pack",
        missing.to_str().expect("Unicode"),
    ]);
    assert_eq!(output.status.code(), Some(1));
    assert!(
        stderr(&output).contains("do not load"),
        "{}",
        stderr(&output)
    );
    fs::remove_dir_all(&dir).expect("the scratch directory can be removed");
}

#[test]
fn proctex_sheet_with_a_pack_shows_the_pack_textures() {
    let dir = scratch("pack-sheet");
    let path = dir.join("pack.png");
    let pack = base_pack();
    let output = cpe(&[
        "proctex-sheet",
        "--pack",
        pack.to_str().expect("Unicode"),
        "--scale",
        "4",
        "--out",
        path.to_str().expect("Unicode"),
    ]);
    assert_eq!(output.status.code(), Some(0), "{}", stderr(&output));
    let (width, height, pixels) = decode(&path);
    assert_eq!(pixels.len(), width as usize * height as usize * 4);
    // Not the contact sheet of the recipes: a different image.
    let (sheet_width, sheet_height, _) = cp_proctex::contact_sheet(16, 4);
    assert_ne!((width, height), (sheet_width, sheet_height));
    // A pack that does not load fails.
    let missing = dir.join("missing");
    let output = cpe(&[
        "proctex-sheet",
        "--pack",
        missing.to_str().expect("Unicode"),
        "--out",
        path.to_str().expect("Unicode"),
    ]);
    assert_eq!(output.status.code(), Some(1));
    assert!(
        stderr(&output).contains("do not load"),
        "{}",
        stderr(&output)
    );
    fs::remove_dir_all(&dir).expect("the scratch directory can be removed");
}

/// Width, height and pixels of a PNG file.
fn decode(path: &Path) -> (u32, u32, Vec<u8>) {
    let file = fs::File::open(path).expect("the image exists");
    let mut reader = png::Decoder::new(std::io::BufReader::new(file))
        .read_info()
        .expect("a PNG header");
    let mut pixels = vec![0; reader.output_buffer_size().expect("a buffer size")];
    let info = reader.next_frame(&mut pixels).expect("PNG pixels");
    pixels.truncate(info.buffer_size());
    (info.width, info.height, pixels)
}

/// True when the run failed for want of a GPU; the test then skips (contract 2.7).
fn skipped_without_gpu(output: &Output, test: &str) -> bool {
    if output.status.code() == Some(0) {
        return false;
    }
    let message = stderr(output);
    if message.contains("no GPU adapter found") {
        eprintln!("{test}: skipped: no GPU adapter ({message})");
        return true;
    }
    false
}

/// Needs a GPU adapter. Small shots of every scene: a PNG of the asked size, sRGB-tagged, with
/// content.
#[test]
fn shots_of_every_scene_are_written_as_srgb_png() {
    let dir = scratch("shots");
    for (scene, camera) in [("structures", "1"), ("calib", "4"), ("hills", "2")] {
        let path = dir.join(format!("{scene}.png"));
        let output = cpe(&[
            "shot",
            "--scene",
            scene,
            "--camera",
            camera,
            "--width",
            "320",
            "--height",
            "180",
            "--out",
            path.to_str().expect("Unicode"),
        ]);
        if skipped_without_gpu(&output, "shots_of_every_scene_are_written_as_srgb_png") {
            return;
        }
        assert_eq!(
            output.status.code(),
            Some(0),
            "{scene}: {}",
            stderr(&output)
        );
        assert!(stdout(&output).contains("wrote"), "{}", stdout(&output));
        // The measured criteria of the image follow on a line of their own.
        assert!(
            stdout(&output).contains("\nimage: colourfulness "),
            "{}",
            stdout(&output)
        );
        let bytes = fs::read(&path).expect("the shot exists");
        assert!(bytes.windows(4).any(|window| window == b"sRGB"), "{scene}");
        let (width, height, pixels) = decode(&path);
        assert_eq!((width, height), (320, 180), "{scene}");
        let mut colours: Vec<&[u8]> = pixels.chunks(4).collect();
        colours.sort_unstable();
        colours.dedup();
        assert!(colours.len() > 20, "{scene}: {} colours", colours.len());
    }
    // An unknown camera is a failure with a message, not a panic.
    let output = cpe(&[
        "shot",
        "--scene",
        "structures",
        "--camera",
        "9",
        "--out",
        dir.join("none.png").to_str().expect("Unicode"),
    ]);
    assert_eq!(output.status.code(), Some(1));
    assert!(
        stderr(&output).contains("has no camera 9"),
        "{}",
        stderr(&output)
    );
    fs::remove_dir_all(&dir).expect("the scratch directory can be removed");
}

/// Needs a GPU adapter. The two evening shots of the beauty review (the valley and the shore
/// cameras of the hills under a sun 8 degrees high in the west-south-west) keep their shadows
/// readable: less than 5 percent of the pixels are dark and over-saturated (mud olive; 37
/// percent of the valley shot were in round 3 of the review of E2) and less than 5 percent
/// near black (15 percent of the shore shot were).
#[test]
fn evening_shots_keep_their_shadows_readable() {
    let dir = scratch("evening");
    for camera in ["0", "2"] {
        let path = dir.join(format!("evening-{camera}.png"));
        let output = cpe(&[
            "shot",
            "--scene",
            "hills",
            "--camera",
            camera,
            "--sun",
            "8,250",
            "--width",
            "320",
            "--height",
            "180",
            "--out",
            path.to_str().expect("Unicode"),
        ]);
        if skipped_without_gpu(&output, "evening_shots_keep_their_shadows_readable") {
            return;
        }
        assert_eq!(
            output.status.code(),
            Some(0),
            "camera {camera}: {}",
            stderr(&output)
        );
        let (width, height, pixels) = decode(&path);
        let metrics = cp_app::metrics::ImageMetrics::measure(width, height, &pixels);
        assert!(
            metrics.dark_saturated < 0.05 && metrics.very_dark < 0.05,
            "camera {camera}: {metrics}"
        );
    }
    fs::remove_dir_all(&dir).expect("the scratch directory can be removed");
}

/// Benchmarks refuse to measure a debug build; in a release test run this test has nothing to
/// check.
#[test]
fn bench_refuses_debug_builds() {
    if !cfg!(debug_assertions) {
        return;
    }
    let dir = scratch("bench");
    let path = dir.join("bench.json");
    let output = cpe(&[
        "bench",
        "--scene",
        "calib",
        "--out",
        path.to_str().expect("Unicode"),
    ]);
    assert_eq!(output.status.code(), Some(1));
    assert!(
        stderr(&output).contains("release builds only"),
        "{}",
        stderr(&output)
    );
    assert!(!path.exists());
    fs::remove_dir_all(&dir).expect("the scratch directory can be removed");
}

/// Needs a GPU adapter; without one the test only checks that the failure is reported and
/// returns (contract 2.7).
#[test]
fn caps_prints_and_saves_the_report() {
    let dir = scratch("caps");
    let markdown = dir.join("caps.md");
    let json = dir.join("caps.json");
    let output = cpe(&[
        "caps",
        "--out",
        markdown.to_str().expect("Unicode"),
        "--json",
        json.to_str().expect("Unicode"),
    ]);
    if output.status.code() != Some(0) {
        let message = stderr(&output);
        assert_eq!(output.status.code(), Some(1), "{message}");
        assert!(message.contains("no GPU adapter found"), "{message}");
        eprintln!("skipped: no GPU adapter ({message})");
        return;
    }
    let printed = stdout(&output);
    assert!(printed.starts_with("# GPU capability report"));
    assert_eq!(fs::read_to_string(&markdown).expect("caps.md"), printed);
    let json_text = fs::read_to_string(&json).expect("caps.json");
    assert!(json_text.contains("\"schema\": 2"), "{json_text}");
    assert!(json_text.contains("\"draw_path\""));
    fs::remove_dir_all(&dir).expect("the scratch directory can be removed");
}
