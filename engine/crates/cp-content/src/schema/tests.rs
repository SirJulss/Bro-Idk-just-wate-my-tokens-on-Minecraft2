//! Tests of the TOML schema and of the positions and keys that parse errors carry.

use super::*;

fn blocks(text: &str) -> Result<Vec<Entry<RawBlock>>, ContentError> {
    parse_entries(Path::new("pack/blocks/test.toml"), text, "block")
}

/// `(line, column, key, message)` of the parse error that `text` must cause.
fn parse_failure(text: &str) -> (u32, u32, Option<String>, String) {
    match blocks(text) {
        Err(ContentError::Parse {
            file,
            line,
            column,
            key,
            message,
        }) => {
            assert_eq!(file, Path::new("pack/blocks/test.toml"));
            (line, column, key, message)
        }
        other => panic!("expected a parse error, got {other:?}"),
    }
}

/// The obviously correct way: walk the text character by character.
fn slow_position(text: &str, offset: usize) -> (u32, u32) {
    let (mut line, mut column) = (1, 1);
    for (index, character) in text.char_indices() {
        if index >= offset {
            break;
        }
        if character == '\n' {
            line += 1;
            column = 1;
        } else {
            column += 1;
        }
    }
    (line, column)
}

#[test]
fn lines_and_columns() {
    let text = "ab\ncd\r\n\näöü=1\n";
    let lines = Lines::new(text);
    assert_eq!(lines.starts, [0, 3, 7, 8, 17]);
    assert_eq!(lines.position(0), (1, 1));
    assert_eq!(lines.position(1), (1, 2));
    // A newline still belongs to the line it ends.
    assert_eq!(lines.position(2), (1, 3));
    assert_eq!(lines.position(3), (2, 1));
    assert_eq!(lines.position(5), (2, 3));
    assert_eq!(lines.position(7), (3, 1));
    assert_eq!(lines.position(8), (4, 1));
    // "äöü" are two bytes each: the `=` is the fourth character of line 4.
    let equals = text.find('=').unwrap();
    assert_eq!(equals, 14);
    assert_eq!(lines.position(equals), (4, 4));
    // The end of the text, and offsets past it, are the start of the line after the last
    // newline.
    assert_eq!(lines.position(text.len()), (5, 1));
    assert_eq!(lines.position(10_000), (5, 1));

    for sample in [
        text,
        "",
        "x",
        "\n",
        "\n\n",
        "no newline at the end",
        "a\n\nb\nçé\n",
    ] {
        let lines = Lines::new(sample);
        for (offset, _) in sample.char_indices().chain([(sample.len(), ' ')]) {
            assert_eq!(
                lines.position(offset),
                slow_position(sample, offset),
                "{sample:?} at {offset}"
            );
        }
    }
}

#[test]
fn entries_keep_their_order_and_lines() {
    let entries = blocks(
        r#"# comment

[[block]]
key = "t:b"
hardness = 1
density = 2.5

[[block]]
key = "t:a"
render = "none"
hardness = 0.0
density = 0
"#,
    )
    .unwrap();
    let summary: Vec<_> = entries
        .iter()
        .map(|entry| (entry.raw.key(), entry.line))
        .collect();
    assert_eq!(summary, [("t:b", 3), ("t:a", 8)]);
    // Integers are accepted where numbers are expected; omitted fields take their defaults.
    let first = &entries[0].raw;
    assert_eq!((first.hardness, first.density), (1.0, 2.5));
    assert_eq!(
        (first.render, first.orient),
        (RawRender::Cube, Orient::None)
    );
    assert_eq!((first.solid, first.opaque, first.light), (None, None, 0));
    assert!(first.tags.is_empty() && first.faces.is_none());
    assert!(!first.structural && first.ext.is_empty());
    assert_eq!((&first.cull_group, &first.lod_proxy), (&None, &None));
    assert_eq!(entries[1].raw.render, RawRender::None);

    // The inline spelling of an array of tables is the same thing.
    let inline = blocks(
        r#"block = [
    { key = "t:b", hardness = 1, density = 2.5 },
    { key = "t:a", hardness = 0, density = 0, render = "none" },
]
"#,
    )
    .unwrap();
    let summary: Vec<_> = inline
        .iter()
        .map(|entry| (entry.raw.key(), entry.line))
        .collect();
    assert_eq!(summary, [("t:b", 2), ("t:a", 3)]);
}

#[test]
fn empty_files_are_fine() {
    assert!(blocks("").unwrap().is_empty());
    assert!(blocks("# nothing here yet\n").unwrap().is_empty());
}

#[test]
fn unknown_fields_name_the_entry_and_the_line() {
    let (line, column, key, message) = parse_failure(
        r#"[[block]]
key = "t:ok"
hardness = 1
density = 1

[[block]]
key = "t:bad"
hardness = 1
density = 1
colour = "red"
"#,
    );
    assert_eq!((line, column), (10, 1));
    assert_eq!(key.as_deref(), Some("t:bad"));
    assert_eq!(
        message,
        "unknown field `colour`, expected one of `key`, `orient`, `render`, `solid`, \
         `opaque`, `structural`, `faces`, `hardness`, `density`, `light`, `tags`, \
         `cull_group`, `lod_proxy`, `ext`"
    );
}

#[test]
fn unknown_fields_in_nested_tables_are_rejected() {
    let (line, column, key, message) = parse_failure(
        r#"[[block]]
key = "t:b"
hardness = 1
density = 1
faces = { all = "t:m", front = "t:m" }
"#,
    );
    assert_eq!((line, column), (5, 24));
    assert_eq!(key.as_deref(), Some("t:b"));
    assert!(message.starts_with("unknown field `front`"), "{message}");
    assert!(
        message.contains("`+x`"),
        "lists the known fields: {message}"
    );
}

#[test]
fn missing_fields_and_wrong_types() {
    let (line, _, key, message) = parse_failure("\n[[block]]\nkey = \"t:b\"\ndensity = 1\n");
    assert_eq!((line, key.as_deref()), (2, Some("t:b")));
    assert_eq!(message, "missing field `hardness`");

    let block = |line: &str| {
        parse_failure(&format!(
            "[[block]]\nkey = \"t:b\"\nhardness = 1\ndensity = 1\n{line}\n"
        ))
    };
    let (line, column, key, message) = block("solid = \"yes\"");
    assert_eq!((line, column, key.as_deref()), (5, 9, Some("t:b")));
    assert!(message.starts_with("invalid type: string"), "{message}");

    let (line, _, key, message) = block("light = 300");
    assert_eq!((line, key.as_deref()), (5, Some("t:b")));
    assert!(message.contains("expected u8"), "{message}");

    let (_, _, key, message) = block("light = 1.5");
    assert_eq!(key.as_deref(), Some("t:b"));
    assert!(
        message.starts_with("invalid type: floating point"),
        "{message}"
    );

    let (_, _, key, message) = block("orient = \"up4\"");
    assert_eq!(key.as_deref(), Some("t:b"));
    assert_eq!(
        message,
        "unknown variant `up4`, expected one of `none`, `up6`, `axis3`, `rot24`"
    );

    let (_, _, _, message) = block("render = \"glass\"");
    assert_eq!(
        message,
        "unknown variant `glass`, expected one of `none`, `cube`, `cutout`, `fluid`, `model`"
    );

    let (_, _, key, message) = block("structural = 1");
    assert_eq!(key.as_deref(), Some("t:b"));
    assert!(message.starts_with("invalid type: integer"), "{message}");
    let (_, _, _, message) = block("cull_group = [\"leaves\"]");
    assert!(message.starts_with("invalid type: sequence"), "{message}");
    let (_, _, _, message) = block("ext = { forestry = [1] }");
    assert!(message.starts_with("invalid type: sequence"), "{message}");

    // Without a readable key the error still has a position.
    let (line, _, key, message) = parse_failure("[[block]]\nhardness = 1\ndensity = 1\n");
    assert_eq!((line, key), (1, None));
    assert_eq!(message, "missing field `key`");
    let (line, _, key, _) = parse_failure("[[block]]\nkey = 5\nhardness = 1\ndensity = 1\n");
    assert_eq!((line, key), (2, None));
}

#[test]
fn syntax_errors_have_a_position() {
    // An unterminated string.
    let (line, column, key, message) = parse_failure("[[block]]\nkey = \"t:b\nhardness = 1\n");
    assert_eq!((line, key), (2, None));
    assert!(column > 1, "{column}");
    assert!(!message.is_empty());

    // The same field twice is a TOML error.
    let (line, column, _, message) =
        parse_failure("[[block]]\nkey = \"t:b\"\nhardness = 1\nhardness = 2\ndensity = 1\n");
    assert_eq!((line, column), (4, 1));
    assert!(message.contains("duplicate key"), "{message}");
}

#[test]
fn only_the_expected_table_is_allowed() {
    let (line, column, key, message) = parse_failure(
        r#"[[block]]
key = "t:b"
hardness = 1
density = 1

[[material]]
"#,
    );
    assert_eq!((line, column, key), (6, 3, None));
    assert_eq!(
        message,
        "unknown top-level key `material`: this file may only contain `[[block]]` entries"
    );

    let (line, _, _, message) = parse_failure("title = \"x\"\n");
    assert_eq!(line, 1);
    assert!(
        message.starts_with("unknown top-level key `title`"),
        "{message}"
    );

    let expected =
        "`block` must be an array of tables: start each entry with `[[block]]`".to_owned();
    let (line, _, _, message) = parse_failure("\nblock = 5\n");
    assert_eq!((line, message), (2, expected.clone()));
    let (_, _, _, message) = parse_failure("[block]\nkey = \"t:b\"\n");
    assert_eq!(message, expected);

    // An array whose elements are not tables fails on its first element.
    let (line, _, key, message) = parse_failure("block = [\n  5,\n]\n");
    assert_eq!((line, key), (2, None));
    assert!(
        message.starts_with("invalid type: integer `5`"),
        "{message}"
    );
}

#[test]
fn faces_forms() {
    fn faces(text: &str) -> Result<Vec<String>, String> {
        let raw: RawFaces = toml::from_str(text).unwrap();
        raw.local_faces()
            .map(|faces| faces.iter().map(|face| (*face).to_owned()).collect())
    }
    assert_eq!(faces(r#"all = "a""#).unwrap(), ["a"; 6]);
    assert_eq!(
        faces("top = \"t\"\nside = \"s\"\nbottom = \"b\"").unwrap(),
        ["s", "s", "t", "b", "s", "s"]
    );
    let six = r#"
"+x" = "1"
"-x" = "2"
"+y" = "3"
"-y" = "4"
"+z" = "5"
"-z" = "6"
"#;
    assert_eq!(faces(six).unwrap(), ["1", "2", "3", "4", "5", "6"]);

    let problem = |text: &str| faces(text).unwrap_err();
    assert!(problem("").starts_with("`faces` is empty: give `all`, or"));
    let five = six.replace("\"-z\" = \"6\"", "");
    for incomplete in [
        "top = \"t\"\nside = \"s\"",
        "bottom = \"b\"",
        "\"+x\" = \"1\"\n\"-y\" = \"4\"",
        five.as_str(),
    ] {
        let message = problem(incomplete);
        assert!(message.starts_with("`faces` is incomplete"), "{message}");
    }
    let all_and_six = format!("all = \"a\"\n{six}");
    for mixed in [
        "all = \"a\"\ntop = \"t\"",
        "all = \"a\"\n\"+x\" = \"1\"",
        "top = \"t\"\nside = \"s\"\nbottom = \"b\"\n\"+x\" = \"1\"",
        all_and_six.as_str(),
    ] {
        let message = problem(mixed);
        assert!(
            message.starts_with("`faces` mixes several forms"),
            "{message}"
        );
    }
}

#[test]
fn textures_and_materials_parse() {
    let textures: Vec<Entry<RawTexture>> = parse_entries(
        Path::new("t.toml"),
        r##"[[texture]]
key = "t:a"
provenance = "own"
procedural = { recipe = "ore", seed = 7, params = { ore = "#102030" } }

[[texture]]
key = "t:b"
provenance = "placeholder"
licence = "CC0"
file = "textures/b.png"

[[texture]]
key = "t:c"
provenance = "own"
[texture.procedural]
recipe = "stone"
seed = 9
"##,
        "texture",
    )
    .unwrap();
    let procedural = textures[0].raw.procedural.as_ref().unwrap();
    assert_eq!((procedural.recipe.as_str(), procedural.seed), ("ore", 7));
    assert_eq!(procedural.params.color("ore"), Some([0x10, 0x20, 0x30]));
    assert!(textures[0].raw.file.is_none() && textures[0].raw.licence.is_none());
    assert_eq!(textures[1].raw.file.as_deref(), Some("textures/b.png"));
    assert_eq!(textures[1].raw.provenance, Provenance::Placeholder);
    assert_eq!(textures[1].raw.licence.as_deref(), Some("CC0"));
    assert!(textures[1].raw.procedural.is_none());
    // The table spelling of `procedural`, without parameters.
    let procedural = textures[2].raw.procedural.as_ref().unwrap();
    assert_eq!((procedural.recipe.as_str(), procedural.seed), ("stone", 9));
    assert!(procedural.params.is_empty());
    assert_eq!(textures[2].line, 12);

    let materials: Vec<Entry<RawMaterial>> = parse_entries(
        Path::new("m.toml"),
        r#"[[material]]
key = "t:a"
texture = "t:a"
roughness = 0.5
tint = "foliage"
flags = ["cutout", "translucent"]

[[material]]
key = "t:b"
texture = "t:a"
"#,
        "material",
    )
    .unwrap();
    let material = &materials[0].raw;
    assert_eq!(
        (material.roughness, material.metallic, material.emissive),
        (0.5, 0.0, 0.0)
    );
    assert_eq!(material.tint, TintClass::Foliage);
    assert_eq!(
        material.flags,
        [RawMaterialFlag::Cutout, RawMaterialFlag::Translucent]
    );
    // A material may be no more than a name for a texture: the texture as it is painted.
    let plain = &materials[1].raw;
    assert_eq!(
        (plain.roughness, plain.metallic, plain.emissive),
        (1.0, 0.0, 0.0)
    );
    assert_eq!(plain.tint, TintClass::None);
    assert!(plain.flags.is_empty());
}

#[test]
fn errors_inside_textures_and_materials_name_the_key() {
    let texture = |procedural: &str| {
        let text = format!(
            "[[texture]]\nkey = \"t:a\"\nprovenance = \"own\"\nprocedural = {procedural}\n"
        );
        parse_entries::<RawTexture>(Path::new("t.toml"), &text, "texture").unwrap_err()
    };
    for (procedural, problem) in [
        (r#"{ recipe = "ore", seed = -1 }"#, "expected u64"),
        (r#"{ recipe = "ore" }"#, "missing field `seed`"),
        (r#"{ seed = 1 }"#, "missing field `recipe`"),
        (
            r#"{ recipe = "ore", seed = 1, size = 16 }"#,
            "unknown field `size`",
        ),
        (
            r##"{ recipe = "ore", seed = 1, params = { ore = "#12" } }"##,
            "`#12` is not a colour",
        ),
        (
            r#"{ recipe = "ore", seed = 1, params = { ore = [1, 2, 3] } }"#,
            "invalid type: sequence",
        ),
        (r#""ore""#, "invalid type: string"),
    ] {
        let error = texture(procedural);
        assert_eq!(error.key(), Some("t:a"), "{procedural}");
        assert!(error.to_string().contains(problem), "{procedural}: {error}");
        assert!(error.to_string().starts_with("t.toml:4:"), "{error}");
    }

    let material = |line: &str| {
        let text =
            format!("[[material]]\nkey = \"t:a\"\ntexture = \"t:a\"\nroughness = 0.5\n{line}\n");
        parse_entries::<RawMaterial>(Path::new("m.toml"), &text, "material").unwrap_err()
    };
    for (line, problem) in [
        (
            r#"flags = ["shiny"]"#,
            "unknown variant `shiny`, expected `cutout` or `translucent`",
        ),
        (r#"flags = "cutout""#, "invalid type: string"),
        (
            r#"tint = "lava""#,
            "unknown variant `lava`, expected one of `none`, `grass`, `foliage`, `water`",
        ),
        ("metallic = true", "invalid type: boolean"),
    ] {
        let error = material(line);
        assert_eq!(error.key(), Some("t:a"), "{line}");
        assert!(error.to_string().contains(problem), "{line}: {error}");
        assert!(error.to_string().starts_with("m.toml:5:"), "{error}");
    }
}

#[test]
fn manifest() {
    let manifest = parse_manifest(
        Path::new("pack.toml"),
        "format = 1\nname = \"base\"\nversion = \"0.1.0\"\n",
    )
    .unwrap();
    assert_eq!(
        (
            manifest.format,
            manifest.name.as_str(),
            manifest.version.as_str()
        ),
        (1, "base", "0.1.0")
    );
    assert_eq!(manifest.description, "");

    let error = parse_manifest(
        Path::new("pack.toml"),
        "format = 1\nname = \"base\"\nversion = \"0.1.0\"\nauthor = \"me\"\n",
    )
    .unwrap_err();
    assert_eq!(
        error.to_string(),
        "pack.toml:4:1: unknown field `author`, expected one of `format`, `name`, \
         `version`, `description`"
    );
    assert_eq!(error.key(), None);
}
