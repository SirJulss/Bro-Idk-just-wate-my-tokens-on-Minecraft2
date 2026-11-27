//! Packs written to disk and loaded through `Registry::load`: overriding, error reports with
//! real file paths, and the behaviour of the content hash.

use std::fs;
use std::path::{Path, PathBuf};

use cp_content::{ContentError, DefKind, Orient, Registry, TextureSource};
use cp_core::block::BlockFlags;
use cp_core::ids::BlockStateId;

/// A directory below cargo's scratch directory for integration tests, removed again on drop.
struct TempDir {
    path: PathBuf,
}

impl TempDir {
    /// `test` must be unique among the tests of this file; the process id keeps concurrent
    /// runs of the test binary apart.
    fn new(test: &str) -> Self {
        let path = Path::new(env!("CARGO_TARGET_TMPDIR"))
            .join(format!("cp-content-{test}-{}", std::process::id()));
        if path.exists() {
            fs::remove_dir_all(&path).expect("a stale test directory can be removed");
        }
        fs::create_dir_all(&path).expect("the test directory can be created");
        Self { path }
    }

    fn path(&self) -> &Path {
        &self.path
    }

    /// Writes `text` to the file `relative` (with `/` separators), creating its directories.
    fn write(&self, relative: &str, text: &str) -> PathBuf {
        self.write_bytes(relative, text.as_bytes())
    }

    fn write_bytes(&self, relative: &str, bytes: &[u8]) -> PathBuf {
        let path = self.path.join(relative);
        let parent = path.parent().expect("files lie inside the test directory");
        fs::create_dir_all(parent).expect("directories can be created");
        fs::write(&path, bytes).expect("the file can be written");
        path
    }

    /// A pack directory `name` with the standard test content.
    fn standard_pack(&self, name: &str) -> PathBuf {
        self.pack(name, TEXTURES, MATERIALS, BLOCKS)
    }

    /// A pack directory `name` with one file per kind.
    fn pack(&self, name: &str, textures: &str, materials: &str, blocks: &str) -> PathBuf {
        self.write(&format!("{name}/pack.toml"), &manifest(name));
        self.write(&format!("{name}/textures/all.toml"), textures);
        self.write(&format!("{name}/materials/all.toml"), materials);
        self.write(&format!("{name}/blocks/all.toml"), blocks);
        self.path.join(name)
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        // Best effort: a leftover directory only wastes space below `target/tmp`.
        let _ = fs::remove_dir_all(&self.path);
    }
}

fn manifest(name: &str) -> String {
    format!("format = 1\nname = \"{name}\"\nversion = \"1.0\"\n")
}

const TEXTURES: &str = r##"
[[texture]]
key = "test:rock"
provenance = "own"
procedural = { recipe = "stone", seed = 1 }

[[texture]]
key = "test:moss"
provenance = "own"
procedural = { recipe = "leaves", seed = 2, params = { leaf = "#336633", style = "broad" } }
"##;

const MATERIALS: &str = r#"
[[material]]
key = "test:rock"
texture = "test:rock"
roughness = 0.9

[[material]]
key = "test:moss"
texture = "test:moss"
roughness = 0.8
tint = "foliage"
"#;

const BLOCKS: &str = r#"
[[block]]
key = "test:rock"
faces = { all = "test:rock" }
hardness = 5.0
density = 2500.0
tags = ["rock"]

[[block]]
key = "test:mossy_rock"
orient = "up6"
faces = { top = "test:moss", side = "test:rock", bottom = "test:rock" }
hardness = 4.0
density = 2400.0
"#;

fn load(dirs: &[&Path]) -> Registry {
    Registry::load(dirs).unwrap_or_else(|error| panic!("the packs must load: {error}"))
}

fn load_error(dirs: &[&Path]) -> ContentError {
    match Registry::load(dirs) {
        Ok(_) => panic!("loading must fail"),
        Err(error) => error,
    }
}

/// The error text with `/` as the only path separator.
fn message(error: &ContentError) -> String {
    error.to_string().replace('\\', "/")
}

/// Kind and key of the definition an error is about.
fn subject(error: &ContentError) -> Option<(DefKind, &str)> {
    match error {
        ContentError::DuplicateKey { kind, key, .. }
        | ContentError::UnknownReference { kind, key, .. }
        | ContentError::Invalid { kind, key, .. } => Some((*kind, key.as_str())),
        _ => None,
    }
}

#[test]
fn a_pack_on_disk_loads() {
    let temp = TempDir::new("loads");
    let dir = temp.standard_pack("rocks");
    let registry = load(&[&dir]);

    assert_eq!(registry.packs().len(), 1);
    assert_eq!(registry.packs()[0].name, "rocks");
    assert_eq!(registry.packs()[0].version, "1.0");
    // Air is there although the pack does not define it.
    let keys: Vec<&str> = (0..registry.state_count())
        .map(|index| registry.state_key(BlockStateId(index as u32)))
        .collect();
    assert_eq!(
        keys,
        [
            "cubeplanet:air",
            "test:mossy_rock",
            "test:mossy_rock[up=+x]",
            "test:mossy_rock[up=+z]",
            "test:mossy_rock[up=-x]",
            "test:mossy_rock[up=-y]",
            "test:mossy_rock[up=-z]",
            "test:rock",
        ]
    );
    for (index, key) in keys.iter().enumerate() {
        assert_eq!(registry.state(key), Some(BlockStateId(index as u32)));
    }
    assert_eq!(registry.tables().flags[0], BlockFlags::empty());
    assert_eq!(
        registry.block("test:mossy_rock").unwrap().orient,
        Orient::Up6
    );
    assert_eq!(registry.materials().len(), 2);
    assert_eq!(registry.textures().len(), 2);
}

#[test]
fn a_second_pack_overrides_by_key() {
    let temp = TempDir::new("override");
    let base = temp.standard_pack("base");
    // The overlay changes the rock block, repaints the moss material and adds a block of its
    // own that uses a material of the first pack.
    let overlay = temp.pack(
        "overlay",
        r#"
[[texture]]
key = "over:lichen"
provenance = "own"
procedural = { recipe = "leaves", seed = 77 }
"#,
        r#"
[[material]]
key = "test:moss"
texture = "over:lichen"
roughness = 0.3
"#,
        r#"
[[block]]
key = "test:rock"
faces = { all = "test:rock" }
hardness = 50.0
density = 2500.0

[[block]]
key = "over:pillar"
orient = "axis3"
faces = { top = "test:moss", side = "test:rock", bottom = "test:moss" }
hardness = 5.0
density = 2500.0
"#,
    );

    let alone = load(&[&base]);
    let both = load(&[&base, &overlay]);
    let names: Vec<&str> = both.packs().iter().map(|pack| pack.name.as_str()).collect();
    assert_eq!(names, ["base", "overlay"]);

    // The block of the first pack is replaced as a whole.
    assert_eq!(alone.block("test:rock").unwrap().hardness, 5.0);
    assert!(alone.block("test:rock").unwrap().has_tag("rock"));
    let rock = both.block("test:rock").unwrap();
    assert_eq!(rock.hardness, 50.0);
    assert!(
        rock.tags.is_empty(),
        "fields are not merged: {:?}",
        rock.tags
    );

    // The repainted material is seen by the untouched block of the first pack.
    let moss = &both.materials()[both.material("test:moss").unwrap().index()];
    assert_eq!(moss.roughness, 0.3);
    assert_eq!(both.textures()[moss.texture.index()].key, "over:lichen");
    let mossy = both.block("test:mossy_rock").unwrap();
    assert_eq!(mossy.faces.unwrap()[2], both.material("test:moss").unwrap());
    // The texture of the first pack is still there, just no longer used by that material.
    assert!(both.texture("test:moss").is_some());

    // New definitions are added; ids are reassigned over the union.
    assert!(alone.block("over:pillar").is_none());
    assert_eq!(both.block("over:pillar").unwrap().orient, Orient::Axis3);
    assert_eq!(both.state_count(), alone.state_count() + 3);
    assert_eq!(both.state_key(BlockStateId(1)), "over:pillar");
    assert_eq!(alone.state_key(BlockStateId(1)), "test:mossy_rock");
    assert_ne!(alone.content_hash(), both.content_hash());

    // The order of the packs decides who wins.
    let reversed = load(&[&overlay, &base]);
    assert_eq!(reversed.block("test:rock").unwrap().hardness, 5.0);
    let moss = &reversed.materials()[reversed.material("test:moss").unwrap().index()];
    assert_eq!(moss.roughness, 0.8);
    assert!(reversed.block("over:pillar").is_some());
    assert_ne!(reversed.content_hash(), both.content_hash());
}

#[test]
fn duplicate_keys_are_rejected() {
    let temp = TempDir::new("duplicate");
    let dir = temp.standard_pack("dup");
    // A second block file in the same pack defines `test:rock` again.
    temp.write(
        "dup/blocks/more.toml",
        "# more rocks\n\n[[block]]\nkey = \"test:rock\"\nfaces = { all = \"test:moss\" }\n\
         hardness = 1.0\ndensity = 1.0\n",
    );
    let error = load_error(&[&dir]);
    match &error {
        ContentError::DuplicateKey {
            kind,
            key,
            file,
            line,
            first_file,
            first_line,
        } => {
            assert_eq!((*kind, key.as_str()), (DefKind::Block, "test:rock"));
            // Files are read in the order of their names: `all.toml` before `more.toml`.
            assert_eq!(file, &dir.join("blocks").join("more.toml"));
            assert_eq!(first_file, &dir.join("blocks").join("all.toml"));
            assert_eq!((*line, *first_line), (3, 2));
        }
        other => panic!("expected a duplicate key, got {other:?}"),
    }
    let text = message(&error);
    assert!(
        text.contains("dup/blocks/more.toml:3: duplicate block `test:rock`"),
        "{text}"
    );
    assert!(text.contains("dup/blocks/all.toml:2"), "{text}");
    assert_eq!(error.key(), Some("test:rock"));

    // The same holds inside one file, and for the other kinds.
    let again = "\n[[material]]\nkey = \"test:moss\"\ntexture = \"test:rock\"\nroughness = 1\n";
    let twice = temp.pack("twice", TEXTURES, &format!("{MATERIALS}{again}"), BLOCKS);
    let error = load_error(&[&twice]);
    let text = message(&error);
    assert!(
        text.contains("twice/materials/all.toml:13: duplicate material `test:moss`"),
        "{text}"
    );
    assert!(text.contains("materials/all.toml:7)"), "{text}");

    let textures = temp.pack(
        "textures",
        &format!("{TEXTURES}{TEXTURES}"),
        MATERIALS,
        BLOCKS,
    );
    let error = load_error(&[&textures]);
    assert!(
        matches!(error, ContentError::DuplicateKey { .. }),
        "{error:?}"
    );
    assert_eq!(subject(&error), Some((DefKind::Texture, "test:rock")));
}

#[test]
fn dangling_references_are_rejected() {
    let temp = TempDir::new("dangling");

    // A block face names a material that does not exist.
    let blocks = BLOCKS.replace("top = \"test:moss\"", "top = \"test:lichen\"");
    let dir = temp.pack("material", TEXTURES, MATERIALS, &blocks);
    let error = load_error(&[&dir]);
    match &error {
        ContentError::UnknownReference {
            kind,
            key,
            file,
            line,
            target_kind,
            target,
        } => {
            assert_eq!((*kind, key.as_str()), (DefKind::Block, "test:mossy_rock"));
            assert_eq!(
                (*target_kind, target.as_str()),
                (DefKind::Material, "test:lichen")
            );
            assert_eq!(file, &dir.join("blocks").join("all.toml"));
            assert_eq!(*line, 9);
        }
        other => panic!("expected an unknown reference, got {other:?}"),
    }
    let text = message(&error);
    assert!(
        text.ends_with(
            "material/blocks/all.toml:9: block `test:mossy_rock` refers to material \
             `test:lichen`, which no loaded pack defines"
        ),
        "{text}"
    );
    assert_eq!(error.key(), Some("test:mossy_rock"));
    assert_eq!(
        error.path(),
        Some(dir.join("blocks").join("all.toml").as_path())
    );

    // A material names a texture that does not exist.
    let materials = MATERIALS.replace("texture = \"test:moss\"", "texture = \"test:fern\"");
    let dir = temp.pack("texture", TEXTURES, &materials, BLOCKS);
    let error = load_error(&[&dir]);
    match &error {
        ContentError::UnknownReference {
            kind,
            key,
            target_kind,
            target,
            line,
            ..
        } => {
            assert_eq!((*kind, key.as_str()), (DefKind::Material, "test:moss"));
            assert_eq!(
                (*target_kind, target.as_str()),
                (DefKind::Texture, "test:fern")
            );
            assert_eq!(*line, 7);
        }
        other => panic!("expected an unknown reference, got {other:?}"),
    }
    let text = message(&error);
    assert!(
        text.ends_with(
            "texture/materials/all.toml:7: material `test:moss` refers to texture \
             `test:fern`, which no loaded pack defines"
        ),
        "{text}"
    );

    // A block key is not a material key: the key spaces are separate.
    let blocks = BLOCKS.replace("all = \"test:rock\"", "all = \"test:mossy_rock\"");
    let dir = temp.pack("kinds", TEXTURES, MATERIALS, &blocks);
    match load_error(&[&dir]) {
        ContentError::UnknownReference {
            target_kind,
            target,
            ..
        } => assert_eq!(
            (target_kind, target.as_str()),
            (DefKind::Material, "test:mossy_rock")
        ),
        other => panic!("expected an unknown reference, got {other:?}"),
    }
}

#[test]
fn unknown_fields_are_rejected() {
    let temp = TempDir::new("unknown-fields");

    let cases: [(&str, DefKind, String, &str, u32, &str); 3] = [
        (
            "block",
            DefKind::Block,
            BLOCKS.replace("density = 2400.0", "density = 2400.0\nhardnes = 3.0"),
            "test:mossy_rock",
            15,
            "hardnes",
        ),
        (
            "material",
            DefKind::Material,
            MATERIALS.replace("tint = \"foliage\"", "colour = \"green\""),
            "test:moss",
            11,
            "colour",
        ),
        (
            "texture",
            DefKind::Texture,
            TEXTURES.replace("seed = 1 }", "seed = 1 }\nauthor = \"me\""),
            "test:rock",
            6,
            "author",
        ),
    ];
    for (name, kind, text, key, line, field) in cases {
        let (textures, materials, blocks) = match kind {
            DefKind::Texture => (text.as_str(), MATERIALS, BLOCKS),
            DefKind::Material => (TEXTURES, text.as_str(), BLOCKS),
            DefKind::Block => (TEXTURES, MATERIALS, text.as_str()),
        };
        let dir = temp.pack(name, textures, materials, blocks);
        let error = load_error(&[&dir]);
        match &error {
            ContentError::Parse {
                file,
                line: reported_line,
                column,
                key: reported_key,
                message,
            } => {
                assert_eq!(file, &dir.join(kind.directory()).join("all.toml"));
                assert_eq!((*reported_line, *column), (line, 1), "{name}");
                assert_eq!(reported_key.as_deref(), Some(key));
                assert!(
                    message.starts_with(&format!("unknown field `{field}`, expected one of")),
                    "{message}"
                );
            }
            other => panic!("expected a parse error, got {other:?}"),
        }
        let text = message(&error);
        let location = format!(
            "{name}/{}/all.toml:{line}:1: `{key}`: unknown field",
            kind.directory()
        );
        assert!(text.contains(&location), "{text}");
        assert_eq!(error.key(), Some(key));
    }

    // Nested tables are strict as well.
    let textures = TEXTURES.replace("seed = 1 }", "seed = 1, size = 32 }");
    let dir = temp.pack("nested", &textures, MATERIALS, BLOCKS);
    let text = message(&load_error(&[&dir]));
    assert!(text.contains("`test:rock`: unknown field `size`"), "{text}");
    let blocks = BLOCKS.replace(
        "{ all = \"test:rock\" }",
        "{ all = \"test:rock\", front = \"test:rock\" }",
    );
    let dir = temp.pack("faces", TEXTURES, MATERIALS, &blocks);
    let text = message(&load_error(&[&dir]));
    assert!(
        text.contains("`test:rock`: unknown field `front`"),
        "{text}"
    );

    // So is pack.toml.
    let dir = temp.standard_pack("manifest");
    temp.write(
        "manifest/pack.toml",
        &format!("{}author = \"me\"\n", manifest("manifest")),
    );
    let error = load_error(&[&dir]);
    let text = message(&error);
    assert!(
        text.contains("manifest/pack.toml:4:1: unknown field `author`"),
        "{text}"
    );
    assert_eq!(error.key(), None);

    // A definition in the wrong directory is an unknown top-level table.
    let dir = temp.standard_pack("misplaced");
    temp.write("misplaced/blocks/stray.toml", MATERIALS);
    let text = message(&load_error(&[&dir]));
    assert!(
        text.contains("misplaced/blocks/stray.toml:2:3: unknown top-level key `material`"),
        "{text}"
    );
}

#[test]
fn invalid_values_name_file_and_key() {
    let temp = TempDir::new("invalid");

    let dir = temp.pack(
        "range",
        TEXTURES,
        &MATERIALS.replace("roughness = 0.8", "roughness = 8"),
        BLOCKS,
    );
    let error = load_error(&[&dir]);
    assert!(
        matches!(error, ContentError::Invalid { line: 7, .. }),
        "{error:?}"
    );
    assert_eq!(subject(&error), Some((DefKind::Material, "test:moss")));
    let text = message(&error);
    assert!(
        text.ends_with(
            "range/materials/all.toml:7: material `test:moss`: `roughness` must be between 0 \
             and 1, got 8.0"
        ),
        "{text}"
    );

    let dir = temp.pack(
        "key",
        TEXTURES,
        MATERIALS,
        &BLOCKS.replace("key = \"test:rock\"", "key = \"Rock\""),
    );
    let text = message(&load_error(&[&dir]));
    assert!(
        text.contains("key/blocks/all.toml:2: block `Rock`:"),
        "{text}"
    );
    assert!(text.contains("namespace:name"), "{text}");

    let dir = temp.pack(
        "type",
        TEXTURES,
        MATERIALS,
        &BLOCKS.replace("hardness = 5.0", "hardness = \"very\""),
    );
    let text = message(&load_error(&[&dir]));
    assert!(
        text.contains("type/blocks/all.toml:5:12: `test:rock`: invalid type"),
        "{text}"
    );

    let dir = temp.pack(
        "syntax",
        TEXTURES,
        MATERIALS,
        &BLOCKS.replace("[[block]]", "[[block]"),
    );
    let error = load_error(&[&dir]);
    assert!(
        matches!(error, ContentError::Parse { line: 2, .. }),
        "{error:?}"
    );
    assert!(message(&error).contains("syntax/blocks/all.toml:2:"));
}

#[test]
fn broken_pack_directories_are_reported() {
    let temp = TempDir::new("broken");

    let missing = temp.path().join("nowhere");
    let error = load_error(&[&missing]);
    assert!(
        matches!(&error, ContentError::Pack { path, .. } if path == &missing),
        "{error:?}"
    );
    assert!(message(&error).ends_with("nowhere: the content pack directory does not exist"));

    // A file is not a pack, not even a pack.toml.
    let file = temp.write("lonely/pack.toml", &manifest("lonely"));
    let error = load_error(&[&file]);
    assert!(
        matches!(&error, ContentError::Pack { path, .. } if path == &file),
        "{error:?}"
    );
    assert!(message(&error).ends_with("pack.toml: a content pack is a directory, this is not one"));
    assert!(Registry::load(&[&temp.path().join("lonely")]).is_ok());

    // A directory without pack.toml is not a pack, whatever else it contains.
    temp.write("bare/blocks/all.toml", BLOCKS);
    let bare = temp.path().join("bare");
    let error = load_error(&[&bare]);
    assert!(
        matches!(&error, ContentError::Pack { path, .. } if path == &bare),
        "{error:?}"
    );
    assert!(message(&error).ends_with("bare: not a content pack: `pack.toml` is missing"));

    // An unsupported format version.
    let dir = temp.standard_pack("future");
    temp.write(
        "future/pack.toml",
        "format = 2\nname = \"future\"\nversion = \"1\"\n",
    );
    let error = load_error(&[&dir]);
    assert!(
        matches!(&error, ContentError::Pack { path, .. } if path == &dir.join("pack.toml")),
        "{error:?}"
    );
    assert!(message(&error).contains("pack format 2 is not supported"));

    // A definition file that is not UTF-8.
    let dir = temp.standard_pack("binary");
    let file = temp.write_bytes("binary/blocks/bad.toml", &[0x5B, 0xFF, 0xFE, 0x5D]);
    let error = load_error(&[&dir]);
    assert!(
        matches!(&error, ContentError::Io { path, .. } if path == &file),
        "{error:?}"
    );

    // One broken pack fails the whole load, wherever it stands in the list.
    let good = temp.standard_pack("good");
    assert!(Registry::load(&[&good, &bare]).is_err());
    assert!(Registry::load(&[&bare, &good]).is_err());
    assert!(Registry::load(&[&good]).is_ok());
}

#[test]
fn definition_files_are_found_and_other_files_left_alone() {
    let temp = TempDir::new("ignored");
    let dir = temp.standard_pack("pack");
    let plain = load(&[&dir]);

    temp.write("pack/README.md", "not content");
    temp.write("pack/textures/notes.txt", "[[texture]] this is not parsed");
    temp.write_bytes("pack/textures/img/rock.png", &[0x89, b'P', b'N', b'G']);
    // Other crates own other directories of a pack.
    temp.write("pack/systems/home.toml", "[[body]]\nname = \"home\"\n");
    // A definition directory may be organised in sub-directories, and the extension counts
    // in any case.
    let block = |name: &str| {
        format!(
            "[[block]]\nkey = \"test:{name}\"\nfaces = {{ all = \"test:rock\" }}\n\
             hardness = 1\ndensity = 1\n"
        )
    };
    temp.write("pack/blocks/EXTRA.TOML", &block("extra"));
    temp.write("pack/blocks/deep/down/nested.toml", &block("nested"));

    let registry = load(&[&dir]);
    assert!(registry.block("test:extra").is_some());
    assert!(registry.block("test:nested").is_some());
    assert_eq!(registry.blocks().len(), plain.blocks().len() + 2);

    // A nested file is a definition file like any other: its problems are reported, with
    // its path, and its keys must be new.
    let nested = temp.write("pack/blocks/deep/more.toml", &block("nested"));
    let error = load_error(&[&dir]);
    match &error {
        ContentError::DuplicateKey {
            file, first_file, ..
        } => {
            // Files are read in the order of their paths: `deep/down/…` before `deep/more…`.
            assert_eq!(file, &nested);
            assert!(first_file.ends_with("deep/down/nested.toml"), "{error}");
        }
        other => panic!("expected a duplicate key, got {other:?}"),
    }
    assert_eq!(subject(&error), Some((DefKind::Block, "test:nested")));
    temp.write(
        "pack/blocks/deep/more.toml",
        "[[block]]\nkey = \"test:broken\"\n",
    );
    let text = message(&load_error(&[&dir]));
    assert!(
        text.contains("pack/blocks/deep/more.toml:1:1: `test:broken`: missing field"),
        "{text}"
    );

    // No directory is required: a pack may consist of pack.toml alone.
    temp.write("empty/pack.toml", &manifest("empty"));
    let empty = load(&[&temp.path().join("empty")]);
    assert_eq!(empty.state_count(), 1);
    assert_eq!(empty.packs()[0].name, "empty");
}

#[test]
fn file_textures_are_resolved_inside_their_pack() {
    let temp = TempDir::new("file-textures");
    let textures = r#"
[[texture]]
key = "test:rock"
provenance = "placeholder"
licence = "CC0, example.org"
file = "textures/img/rock.png"

[[texture]]
key = "test:moss"
provenance = "own"
procedural = { recipe = "leaves", seed = 2 }
"#;
    let dir = temp.pack("pack", textures, MATERIALS, BLOCKS);

    // The file has to exist.
    let error = load_error(&[&dir]);
    assert!(
        matches!(error, ContentError::Invalid { line: 2, .. }),
        "{error:?}"
    );
    assert_eq!(subject(&error), Some((DefKind::Texture, "test:rock")));
    let text = message(&error);
    assert!(
        text.contains("texture file `textures/img/rock.png` does not exist"),
        "{text}"
    );
    assert!(
        text.contains("pack/textures/all.toml:2: texture `test:rock`"),
        "{text}"
    );

    let image = temp.write_bytes("pack/textures/img/rock.png", &[0x89, b'P', b'N', b'G']);
    let registry = load(&[&dir]);
    let texture = &registry.textures()[registry.texture("test:rock").unwrap().index()];
    match &texture.source {
        TextureSource::File { path, resolved } => {
            assert_eq!(path, "textures/img/rock.png");
            assert_eq!(resolved, &dir.join("textures/img/rock.png"));
            assert!(resolved.is_file());
            assert_eq!(fs::read(resolved).unwrap(), fs::read(&image).unwrap());
        }
        other => panic!("expected a file texture, got {other:?}"),
    }
    assert_eq!(texture.licence.as_deref(), Some("CC0, example.org"));

    // The same pack somewhere else: other resolved paths, same content hash.
    let copy = temp.pack("copy", textures, MATERIALS, BLOCKS);
    temp.write_bytes("copy/textures/img/rock.png", &[1, 2, 3]);
    let copied = load(&[&copy]);
    assert_eq!(copied.content_hash(), registry.content_hash());
    let moved = &copied.textures()[copied.texture("test:rock").unwrap().index()];
    assert_ne!(moved.source, texture.source);

    // Paths must stay inside the pack.
    for bad in [
        "../pack/textures/img/rock.png",
        "/textures/img/rock.png",
        "textures\\\\img\\\\rock.png",
    ] {
        let escaping = textures.replace("textures/img/rock.png", bad);
        let dir = temp.pack("escape", &escaping, MATERIALS, BLOCKS);
        let text = message(&load_error(&[&dir]));
        assert!(
            text.contains("must be relative to the pack directory"),
            "{bad}: {text}"
        );
    }

    // A placeholder without a licence is refused.
    let unlicensed = textures.replace("licence = \"CC0, example.org\"\n", "");
    let dir = temp.pack("unlicensed", &unlicensed, MATERIALS, BLOCKS);
    temp.write_bytes("unlicensed/textures/img/rock.png", &[0]);
    let text = message(&load_error(&[&dir]));
    assert!(
        text.contains("a placeholder texture must state its `licence`"),
        "{text}"
    );
}

/// On Windows and macOS a file is found under any spelling of its case, on Linux only under
/// its own. A pack must load everywhere or nowhere.
#[test]
fn file_texture_names_must_match_exactly() {
    let temp = TempDir::new("file-spelling");
    let pack = |name: &str, path: &str| {
        let textures = format!(
            "[[texture]]\nkey = \"test:rock\"\nprovenance = \"own\"\nfile = \"{path}\"\n\n\
             [[texture]]\nkey = \"test:moss\"\nprovenance = \"own\"\n\
             procedural = {{ recipe = \"leaves\", seed = 2 }}\n"
        );
        let dir = temp.pack(name, &textures, MATERIALS, BLOCKS);
        temp.write_bytes(&format!("{name}/textures/img/grate.png"), &[0x89, b'P']);
        dir
    };
    let path_of = |registry: &Registry| match &registry.textures()
        [registry.texture("test:rock").unwrap().index()]
    .source
    {
        TextureSource::File { path, .. } => path.clone(),
        other => panic!("expected a file texture, got {other:?}"),
    };
    let exact = load(&[&pack("exact", "textures/img/grate.png")]);
    assert_eq!(path_of(&exact), "textures/img/grate.png");

    // Another case anywhere in the path fails on every system and names the real spelling.
    for (name, path, written, real) in [
        (
            "file_case",
            "textures/img/Grate.png",
            "Grate.png",
            "grate.png",
        ),
        (
            "ext_case",
            "textures/img/grate.PNG",
            "grate.PNG",
            "grate.png",
        ),
        ("dir_case", "textures/IMG/grate.png", "IMG", "img"),
        ("top_case", "Textures/img/grate.png", "Textures", "textures"),
    ] {
        let error = load_error(&[&pack(name, path)]);
        assert!(
            matches!(error, ContentError::Invalid { line: 1, .. }),
            "{error:?}"
        );
        assert_eq!(subject(&error), Some((DefKind::Texture, "test:rock")));
        let text = message(&error);
        assert!(
            text.contains(&format!(
                "texture file `{path}` does not exist in its pack: `{written}` is spelled \
                 `{real}` on disk"
            )),
            "{text}"
        );
    }

    // Names that Windows would read as another file are refused before the disk is asked.
    for (name, path, problem) in [
        ("dot", "textures/img/grate.png.", "ends in a dot or a space"),
        (
            "space",
            "textures/img/grate.png ",
            "ends in a dot or a space",
        ),
        (
            "device",
            "textures/img/nul.png",
            "names a device on Windows",
        ),
        ("wildcard", "textures/img/grate?.png", "contains '?'"),
    ] {
        let text = message(&load_error(&[&pack(name, path)]));
        assert!(text.contains(problem), "{path}: {text}");
    }

    // The path must end at a file and pass through directories only.
    let text = message(&load_error(&[&pack("dir", "textures/img")]));
    assert!(
        text.contains("texture file `textures/img` is not a file"),
        "{text}"
    );
    let text = message(&load_error(&[&pack(
        "through",
        "textures/img/grate.png/x.png",
    )]));
    assert!(text.contains("is not a directory"), "{text}");

    // Two names that differ only in case cannot both exist on Windows or macOS. Where the
    // file system holds both (Linux), a pack that relies on one of them is refused.
    let dir = pack("twins", "textures/img/grate.png");
    temp.write_bytes("twins/textures/img/Grate.png", &[1]);
    let names = fs::read_dir(dir.join("textures/img")).unwrap().count();
    if names == 2 {
        let text = message(&load_error(&[&dir]));
        assert!(
            text.contains("`grate.png` and `Grate.png` lie in")
                && text.contains("differ only in case"),
            "{text}"
        );
    } else {
        assert_eq!(names, 1, "a case-insensitive file system keeps one file");
        assert_eq!(path_of(&load(&[&dir])), "textures/img/grate.png");
    }
}

/// Git for Windows checks text files out with CRLF line endings by default. A pack must mean
/// the same either way, down to the bytes of its multi-line strings and its content hash.
#[test]
fn line_endings_change_no_string_and_no_hash() {
    let temp = TempDir::new("line-endings");
    let textures = r#"
[[texture]]
key = "test:rock"
provenance = "placeholder"
licence = """
CC-BY 4.0
by someone
"""

[texture.procedural]
recipe = "stone"
seed = 1

[texture.procedural.params]
style = '''
rough
'''

[[texture]]
key = "test:moss"
provenance = "own"
procedural = { recipe = "leaves", seed = 2 }
"#;
    let blocks = r#"
[[block]]
key = "test:rock"
faces = { all = "test:rock" }
hardness = 5.0
density = 2500.0

[block.ext.lore]
text = '''
Old rock.
Very old.
'''
lines = ["""
one
""", "two"]
escaped = "kept\r\nas written"

[block.ext.lore.nested]
note = """
deep\
  er
"""

[[block]]
key = "test:mossy_rock"
faces = { all = "test:moss" }
hardness = 4.0
density = 2400.0
"#;
    let crlf = |text: &str| text.replace('\n', "\r\n");
    let lf = load(&[&temp.pack("lf", textures, MATERIALS, blocks)]);
    temp.write("crlf/pack.toml", &crlf(&manifest("crlf")));
    temp.write("crlf/textures/all.toml", &crlf(textures));
    temp.write("crlf/materials/all.toml", &crlf(MATERIALS));
    temp.write("crlf/blocks/all.toml", &crlf(blocks));
    let windows = load(&[&temp.path().join("crlf")]);

    let rock = |registry: &Registry| {
        registry.textures()[registry.texture("test:rock").unwrap().index()].clone()
    };
    let texture = rock(&lf);
    assert_eq!(texture.licence.as_deref(), Some("CC-BY 4.0\nby someone\n"));
    match &texture.source {
        TextureSource::Procedural { params, .. } => {
            assert_eq!(params.variant("style"), Some("rough\n"));
        }
        other => panic!("expected a procedural texture, got {other:?}"),
    }
    let lore = lf.block("test:rock").unwrap().ext("lore").unwrap().clone();
    assert_eq!(lore["text"].as_str(), Some("Old rock.\nVery old.\n"));
    assert_eq!(lore["lines"][0].as_str(), Some("one\n"));
    assert_eq!(lore["nested"]["note"].as_str(), Some("deeper\n"));
    // A line break spelled with escapes is the author's data and stays as written.
    assert_eq!(lore["escaped"].as_str(), Some("kept\r\nas written"));

    assert_eq!(rock(&windows), texture);
    assert_eq!(windows.block("test:rock").unwrap().ext("lore"), Some(&lore));
    assert_eq!(windows.content_hash(), lf.content_hash());
}

#[test]
fn hash_is_stable_and_changes_with_every_value() {
    let temp = TempDir::new("hash");
    let dir = temp.standard_pack("pack");
    let original = load(&[&dir]).content_hash();
    assert_eq!(load(&[&dir]).content_hash(), original);

    // One changed value at a time, in every kind of definition: `(kind, from, to)`.
    use DefKind::{Block, Material, Texture};
    let changes = [
        (Block, "hardness = 5.0", "hardness = 5.5"),
        (Block, "density = 2400.0", "density = 2401.0"),
        (Block, r#"orient = "up6""#, r#"orient = "axis3""#),
        (Block, r#"tags = ["rock"]"#, r#"tags = ["stone"]"#),
        (Block, r#"tags = ["rock"]"#, r#"tags = ["rock", "hard"]"#),
        (Block, r#"bottom = "test:rock""#, r#"bottom = "test:moss""#),
        (Block, "density = 2500.0", "density = 2500.0\nlight = 1"),
        (
            Block,
            "density = 2500.0",
            "density = 2500.0\nopaque = false",
        ),
        (Block, "density = 2500.0", "density = 2500.0\nsolid = false"),
        (
            Block,
            "density = 2500.0",
            "density = 2500.0\nstructural = true",
        ),
        (Block, r#"orient = "up6""#, r#"orient = "rot24""#),
        (
            Block,
            "density = 2500.0",
            "density = 2500.0\nopaque = false\ncull_group = \"rocks\"",
        ),
        (
            Block,
            r#"tags = ["rock"]"#,
            "tags = [\"rock\"]\n[block.ext.mine]\nore = true",
        ),
        (Block, r#"key = "test:rock""#, r#"key = "test:rocks""#),
        (Material, "roughness = 0.9", "roughness = 0.91"),
        (
            Material,
            "roughness = 0.9",
            "roughness = 0.9\nmetallic = 0.5",
        ),
        (
            Material,
            "roughness = 0.9",
            "roughness = 0.9\nemissive = 0.5",
        ),
        (Material, r#"tint = "foliage""#, r#"tint = "grass""#),
        (
            Material,
            r#"texture = "test:moss""#,
            r#"texture = "test:rock""#,
        ),
        (Texture, "seed = 1 }", "seed = 3 }"),
        (Texture, "#336633", "#336634"),
        (Texture, r#"style = "broad""#, r#"style = "needle""#),
        (Texture, r#"style = "broad""#, r##"style = "#b0a""##),
        (Texture, r#", style = "broad""#, ""),
        (Texture, r#"recipe = "stone""#, r#"recipe = "cobble""#),
        (
            Texture,
            r#"provenance = "own""#,
            "provenance = \"own\"\nlicence = \"mine\"",
        ),
    ];
    let mut hashes = vec![original];
    for (index, (kind, from, to)) in changes.into_iter().enumerate() {
        let source = match kind {
            DefKind::Texture => TEXTURES,
            DefKind::Material => MATERIALS,
            DefKind::Block => BLOCKS,
        };
        assert!(source.contains(from), "`{from}` is not in the test pack");
        let changed = source.replacen(from, to, 1);
        let (textures, materials, blocks) = match kind {
            DefKind::Texture => (changed.as_str(), MATERIALS, BLOCKS),
            DefKind::Material => (TEXTURES, changed.as_str(), BLOCKS),
            DefKind::Block => (TEXTURES, MATERIALS, changed.as_str()),
        };
        let dir = temp.pack(&format!("change{index}"), textures, materials, blocks);
        let hash = load(&[&dir]).content_hash();
        assert!(
            !hashes.contains(&hash),
            "`{from}` -> `{to}` did not change the hash"
        );
        hashes.push(hash);
    }

    // Material flags count as well. A cut-out material needs a `cutout` block and a
    // translucent one a block that is not opaque, so the block is changed first and the flag
    // on top of that.
    for (flag, block_change, refusal) in [
        (
            "cutout",
            "orient = \"up6\"\nrender = \"cutout\"",
            "block `test:mossy_rock`: a `cube` block cannot use the cut-out material `test:moss`",
        ),
        (
            "translucent",
            "orient = \"up6\"\nopaque = false",
            "block `test:mossy_rock`: an `opaque` block cannot use the translucent material \
             `test:moss`",
        ),
    ] {
        let blocks = BLOCKS.replace("orient = \"up6\"", block_change);
        let materials = MATERIALS.replace(
            "roughness = 0.8",
            &format!("roughness = 0.8\nflags = [\"{flag}\"]"),
        );
        let plain = temp.pack(&format!("{flag}_off"), TEXTURES, MATERIALS, &blocks);
        let flagged = temp.pack(&format!("{flag}_on"), TEXTURES, &materials, &blocks);
        let (plain, flagged) = (load(&[&plain]), load(&[&flagged]));
        assert_ne!(plain.content_hash(), flagged.content_hash(), "{flag}");
        assert_ne!(plain.content_hash(), original, "{flag}");

        // The flag alone would be refused: the mossy rock is an opaque cube.
        let dir = temp.pack(&format!("{flag}_refused"), TEXTURES, &materials, BLOCKS);
        let text = message(&load_error(&[&dir]));
        assert!(text.contains(refusal), "{text}");
    }
}

#[test]
fn hash_ignores_everything_that_is_not_content() {
    let temp = TempDir::new("hash-layout");
    let dir = temp.standard_pack("pack");
    let original = load(&[&dir]).content_hash();

    // Another directory, another pack name, version and description.
    temp.write(
        "elsewhere/deep/down/pack.toml",
        "format = 1\nname = \"renamed\"\nversion = \"9.9\"\ndescription = \"moved\"\n",
    );
    temp.write("elsewhere/deep/down/textures/t.toml", TEXTURES);
    temp.write("elsewhere/deep/down/materials/m.toml", MATERIALS);
    temp.write("elsewhere/deep/down/blocks/b.toml", BLOCKS);
    let moved = load(&[&temp.path().join("elsewhere/deep/down")]);
    assert_eq!(moved.content_hash(), original);
    assert_eq!(moved.packs()[0].name, "renamed");

    // The same definitions in another order, split over several files, with other spacing,
    // comments, Windows line endings and numbers written differently.
    let (rock, mossy) = BLOCKS
        .split_once("\n[[block]]\nkey = \"test:mossy_rock\"")
        .expect("the test pack has two blocks");
    let mossy = format!("[[block]]\nkey = \"test:mossy_rock\"{mossy}");
    temp.write("split/pack.toml", &manifest("split"));
    temp.write("split/textures/t.toml", &TEXTURES.replace('\n', "\r\n"));
    temp.write(
        "split/materials/m.toml",
        &format!("# materials\n\n\n{MATERIALS}"),
    );
    temp.write(
        "split/blocks/a_mossy.toml",
        &mossy.replace("4.0", "4").replace(" = ", "="),
    );
    temp.write("split/blocks/b_rock.toml", &rock.replace("2500.0", "2.5e3"));
    let split = load(&[&temp.path().join("split")]);
    assert_eq!(split.content_hash(), original);

    // Two packs that together hold the same definitions.
    temp.write("art/pack.toml", &manifest("art"));
    temp.write("art/textures/t.toml", TEXTURES);
    temp.write("art/materials/m.toml", MATERIALS);
    temp.write("world/pack.toml", &manifest("world"));
    temp.write("world/blocks/b.toml", BLOCKS);
    let art = temp.path().join("art");
    let world = temp.path().join("world");
    assert_eq!(load(&[&art, &world]).content_hash(), original);
    assert_eq!(load(&[&world, &art]).content_hash(), original);

    // Describing air explicitly, exactly as the engine provides it, changes nothing either.
    let air = r#"
[[block]]
key = "cubeplanet:air"
render = "none"
hardness = 0
density = 0
"#;
    let dir = temp.pack("air", TEXTURES, MATERIALS, &format!("{BLOCKS}{air}"));
    assert_eq!(load(&[&dir]).content_hash(), original);
}

#[test]
fn the_base_pack_hashes_the_same_wherever_it_lies() {
    let source = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../content/base");
    let temp = TempDir::new("base-copy");
    temp.write(
        "copy/pack.toml",
        &fs::read_to_string(source.join("pack.toml")).unwrap(),
    );
    let mut files = 0;
    for kind in DefKind::ALL {
        let directory = kind.directory();
        for entry in fs::read_dir(source.join(directory)).unwrap() {
            let path = entry.unwrap().path();
            let name = path.file_name().unwrap().to_str().unwrap().to_owned();
            temp.write(
                &format!("copy/{directory}/{name}"),
                &fs::read_to_string(&path).unwrap(),
            );
            files += 1;
        }
    }
    assert!(
        files >= 3,
        "the base pack has definition files of every kind"
    );

    let original = Registry::load(&[&source]).unwrap();
    let copy = Registry::load(&[&temp.path().join("copy")]).unwrap();
    assert_eq!(copy.content_hash(), original.content_hash());
    assert_eq!(copy.state_count(), original.state_count());
    for index in 0..original.state_count() {
        let id = BlockStateId(index as u32);
        assert_eq!(copy.state_key(id), original.state_key(id));
    }
    assert_eq!(copy.tables(), original.tables());
    assert_eq!(copy.cull_groups(), original.cull_groups());
    assert_eq!(copy.blocks(), original.blocks());
    assert_eq!(copy.materials(), original.materials());
    assert_eq!(copy.textures(), original.textures());
}
