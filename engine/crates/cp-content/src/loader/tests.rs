//! Tests of the loader, on packs that exist only in memory.

use cp_core::block::BlockFlags;
use cp_core::ids::BlockStateId;

use super::*;
use crate::block::Orient;
use crate::material::TintClass;

const MANIFEST: &str = r#"
format = 1
name = "test"
version = "1"
"#;

const TEXTURES: &str = r##"
[[texture]]
key = "t:a"
provenance = "own"
procedural = { recipe = "stone", seed = 1 }

[[texture]]
key = "t:b"
provenance = "own"
procedural = { recipe = "dirt", seed = 2, params = { tone = "#102030", style = "coarse" } }
"##;

const MATERIALS: &str = r#"
[[material]]
key = "t:a"
texture = "t:a"
roughness = 0.9

[[material]]
key = "t:b"
texture = "t:b"
roughness = 0.5
"#;

/// A pack with the standard textures and materials plus the given block file.
fn pack_with_blocks(blocks: &str) -> PackSource {
    PackSource::embedded(
        "<test>",
        MANIFEST,
        &[
            ("t.toml", DefKind::Texture, TEXTURES),
            ("m.toml", DefKind::Material, MATERIALS),
            ("b.toml", DefKind::Block, blocks),
        ],
    )
}

fn load_blocks(blocks: &str) -> Result<Registry, ContentError> {
    assemble(&[pack_with_blocks(blocks)])
}

/// The standard pack with one block `t:x` that has a hardness, a density and `lines`.
fn block_with(lines: &str) -> Result<Registry, ContentError> {
    load_blocks(&format!(
        "[[block]]\nkey = \"t:x\"\nhardness = 1\ndensity = 1\n{lines}\n"
    ))
}

/// A pack that consists of one definition file.
fn pack_of(kind: DefKind, text: &str) -> PackSource {
    PackSource::embedded("<test>", MANIFEST, &[("x.toml", kind, text)])
}

/// `kind key: message` of the expected [`ContentError::Invalid`].
fn invalid_message(result: Result<Registry, ContentError>) -> String {
    match result {
        Err(ContentError::Invalid {
            kind, key, message, ..
        }) => format!("{kind} {key}: {message}"),
        other => panic!("expected an invalid definition, got {other:?}"),
    }
}

#[test]
fn no_packs_still_have_air() {
    let registry = assemble(&[]).unwrap();
    assert_eq!(registry.state_count(), 1);
    assert_eq!(registry.state_key(BlockStateId::AIR), AIR_KEY);
    assert_eq!(*registry.tables().flags, [BlockFlags::empty()]);
    assert!(registry.materials().is_empty() && registry.textures().is_empty());
    assert!(registry.packs().is_empty());
    let air = &registry.blocks()[0];
    assert_eq!((air.key.as_str(), air.render), (AIR_KEY, RenderClass::None));
    assert_eq!((air.hardness, air.density, air.light), (0.0, 0.0, 0));
}

#[test]
fn render_class_sets_the_defaults() {
    let registry = load_blocks(
        r#"
[[block]]
key = "t:cube"
faces = { all = "t:a" }
hardness = 1
density = 1

[[block]]
key = "t:cutout"
render = "cutout"
faces = { all = "t:a" }
hardness = 1
density = 1

[[block]]
key = "t:fluid"
render = "fluid"
faces = { all = "t:a" }
hardness = 0
density = 1

[[block]]
key = "t:none"
render = "none"
hardness = 0
density = 1

[[block]]
key = "t:glass"
opaque = false
faces = { all = "t:a" }
hardness = 1
density = 1

[[block]]
key = "t:barrier"
render = "none"
solid = true
hardness = inf
density = 1
"#,
    )
    .unwrap();
    let summary = |key: &str| {
        let block = registry.block(key).unwrap();
        (block.render, block.solid, block.opaque)
    };
    assert_eq!(summary("t:cube"), (RenderClass::Cube, true, true));
    assert_eq!(summary("t:cutout"), (RenderClass::Cutout, true, false));
    assert_eq!(summary("t:fluid"), (RenderClass::Fluid, false, false));
    assert_eq!(summary("t:none"), (RenderClass::None, false, false));
    assert_eq!(summary("t:glass"), (RenderClass::Cube, true, false));
    assert_eq!(summary("t:barrier"), (RenderClass::None, true, false));
    assert!(registry.block("t:barrier").unwrap().is_unbreakable());
    assert_eq!(registry.block("t:none").unwrap().faces, None);
}

#[test]
fn opaque_must_cover_the_cell() {
    let cases = [
        (
            "render = \"none\"\nopaque = true",
            "a block with `render = \"none\"`",
        ),
        (
            "render = \"cutout\"\nopaque = true\nfaces = { all = \"t:a\" }",
            "a `cutout` block",
        ),
        (
            "render = \"fluid\"\nopaque = true\nfaces = { all = \"t:a\" }",
            "a `fluid` block",
        ),
    ];
    for (lines, what) in cases {
        let message = invalid_message(block_with(lines));
        let expected = format!("block t:x: {what} cannot be `opaque`");
        assert!(message.starts_with(&expected), "{message}");
    }
    let message = invalid_message(block_with(
        "render = \"fluid\"\nsolid = true\nfaces = { all = \"t:a\" }",
    ));
    assert_eq!(message, "block t:x: a `fluid` block cannot be `solid`");
}

#[test]
fn opaque_blocks_need_opaque_materials() {
    let materials = format!(
        "{MATERIALS}\n\
         [[material]]\nkey = \"t:holes\"\ntexture = \"t:a\"\nflags = [\"cutout\"]\n\n\
         [[material]]\nkey = \"t:glass\"\ntexture = \"t:a\"\nflags = [\"translucent\"]\n"
    );
    let block = |lines: &str| {
        let blocks = format!("[[block]]\nkey = \"t:x\"\nhardness = 1\ndensity = 1\n{lines}\n");
        assemble(&[PackSource::embedded(
            "<test>",
            MANIFEST,
            &[
                ("t.toml", DefKind::Texture, TEXTURES),
                ("m.toml", DefKind::Material, &materials),
                ("b.toml", DefKind::Block, &blocks),
            ],
        )])
    };
    // One see-through face is enough to leave a hole in what the block hides.
    for faces in [
        r#"{ all = "t:glass" }"#,
        r#"{ top = "t:a", side = "t:a", bottom = "t:glass" }"#,
    ] {
        let message = invalid_message(block(&format!("faces = {faces}")));
        assert_eq!(
            message,
            "block t:x: an `opaque` block cannot use the translucent material `t:glass`: it \
             would hide what shows through it; set `opaque = false`"
        );
    }
    // Only the cutout layer cuts holes: cut-out materials need `render = "cutout"`, also on
    // cubes that hide nothing, where the holes would show as solid colour.
    for lines in [
        "faces = { all = \"t:holes\" }",
        "opaque = false\nfaces = { all = \"t:holes\" }",
        "opaque = false\nfaces = { top = \"t:a\", side = \"t:glass\", bottom = \"t:holes\" }",
    ] {
        assert_eq!(
            invalid_message(block(lines)),
            "block t:x: a `cube` block cannot use the cut-out material `t:holes`: it is drawn \
             without cutting holes; set `render = \"cutout\"`",
            "{lines}"
        );
    }
    assert_eq!(
        invalid_message(block("render = \"fluid\"\nfaces = { all = \"t:holes\" }")),
        "block t:x: a `fluid` block cannot use the cut-out material `t:holes`: fluids are \
         drawn without cutting holes; use a material without the `cutout` flag"
    );
    // The advice of each message leads to a block that loads and is drawn as intended.
    let advised = block("render = \"cutout\"\nfaces = { all = \"t:holes\" }").unwrap();
    let state = advised.state("t:x").unwrap();
    assert_eq!(
        advised.tables().flags[state.index()],
        BlockFlags::SOLID | BlockFlags::CUTOUT
    );
    assert_eq!(
        advised.tables().render[state.index()].class,
        RenderClass::Cutout
    );
    let advised = block("opaque = false\nfaces = { all = \"t:glass\" }").unwrap();
    let state = advised.state("t:x").unwrap();
    assert_eq!(advised.tables().flags[state.index()], BlockFlags::SOLID);
    // See-through faces are fine on blocks that hide nothing and draw them in their layer.
    for lines in [
        "render = \"cutout\"\nfaces = { top = \"t:a\", side = \"t:holes\", bottom = \"t:holes\" }",
        "render = \"fluid\"\nfaces = { all = \"t:glass\" }",
        "opaque = false\nfaces = { top = \"t:a\", side = \"t:glass\", bottom = \"t:glass\" }",
    ] {
        let registry = block(lines).unwrap();
        assert!(!registry.block("t:x").unwrap().opaque, "{lines}");
    }
}

#[test]
fn faces_must_match_the_render_class() {
    let message = invalid_message(block_with(""));
    assert!(message.contains("`faces` is missing"), "{message}");
    let message = invalid_message(block_with("render = \"none\"\nfaces = { all = \"t:a\" }"));
    assert!(message.contains("cannot have `faces`"), "{message}");
    for (faces, problem) in [
        ("{}", "`faces` is empty"),
        (r#"{ top = "t:a" }"#, "`faces` is incomplete"),
        (
            r#"{ all = "t:a", side = "t:b" }"#,
            "`faces` mixes several forms",
        ),
    ] {
        let message = invalid_message(block_with(&format!("faces = {faces}")));
        assert!(message.contains(problem), "{message}");
    }
}

#[test]
fn face_forms_resolve_to_local_faces() {
    let registry = load_blocks(
        r#"
[[block]]
key = "t:all"
faces = { all = "t:b" }
hardness = 1
density = 1

[[block]]
key = "t:tsb"
faces = { top = "t:b", side = "t:a", bottom = "t:b" }
hardness = 1
density = 1

[[block]]
key = "t:six"
hardness = 1
density = 1
[block.faces]
"+x" = "t:a"
"-x" = "t:b"
"+y" = "t:b"
"-y" = "t:a"
"+z" = "t:b"
"-z" = "t:b"
"#,
    )
    .unwrap();
    let (a, b) = (
        registry.material("t:a").unwrap(),
        registry.material("t:b").unwrap(),
    );
    let faces = |key: &str| registry.block(key).unwrap().faces.unwrap();
    assert_eq!(faces("t:all"), [b; 6]);
    assert_eq!(faces("t:tsb"), [a, a, b, b, a, a]);
    assert_eq!(faces("t:six"), [a, b, b, a, b, b]);
}

#[test]
fn block_values_are_range_checked() {
    let faces = r#"faces = { all = "t:a" }"#;
    for (line, problem) in [
        ("light = 16", "`light` must be between 0 and 15, got 16"),
        (r#"tags = ["Rock"]"#, "tag `Rock` must be made of"),
        (r#"tags = ["a", "b", "a"]"#, "tag `a` is listed twice"),
    ] {
        let message = invalid_message(block_with(&format!("{faces}\n{line}")));
        assert!(message.contains(problem), "{message}");
    }

    let finite = "must be a finite number >= 0";
    for (field, value, problem) in [
        ("hardness", "-1", format!("`hardness` {finite}, got -1.0")),
        ("hardness", "nan", format!("`hardness` {finite}, got NaN")),
        (
            "hardness",
            "-inf",
            "(`inf` makes a block unbreakable)".to_owned(),
        ),
        ("density", "-0.5", format!("`density` {finite}, got -0.5")),
        ("density", "inf", format!("`density` {finite}, got inf")),
        ("density", "1e300", format!("`density` {finite}, got 1e300")),
    ] {
        let other = if field == "hardness" {
            "density"
        } else {
            "hardness"
        };
        let message = invalid_message(load_blocks(&format!(
            "[[block]]\nkey = \"t:x\"\n{field} = {value}\n{other} = 1\n{faces}\n"
        )));
        assert!(message.contains(&problem), "{message}");
    }

    let registry = block_with(&format!("{faces}\nlight = 15\ntags = [\"wood\", \"log\"]")).unwrap();
    let block = registry.block("t:x").unwrap();
    assert_eq!(block.light, 15);
    assert_eq!(block.tags, ["log", "wood"], "tags are sorted");

    // An unbreakable block, and a minus zero that must not reach the hash as such.
    let registry = load_blocks(&format!(
        "[[block]]\nkey = \"t:x\"\nhardness = inf\ndensity = -0.0\n{faces}\n"
    ))
    .unwrap();
    let block = registry.block("t:x").unwrap();
    assert_eq!(block.hardness, f32::INFINITY);
    assert_eq!(block.density.to_bits(), 0.0_f32.to_bits());
    // Values are rounded to the nearest `f32`.
    let registry = load_blocks(&format!(
        "[[block]]\nkey = \"t:x\"\nhardness = 0.1\ndensity = 16777217\n{faces}\n"
    ))
    .unwrap();
    let block = registry.block("t:x").unwrap();
    assert_eq!(block.hardness, 0.1_f32);
    assert_eq!(block.density, 16_777_216.0);
}

#[test]
fn material_rules() {
    let material = |lines: &str| {
        let text = format!("[[material]]\nkey = \"t:m\"\ntexture = \"t:a\"\n{lines}\n");
        assemble(&[PackSource::embedded(
            "<test>",
            MANIFEST,
            &[
                ("t.toml", DefKind::Texture, TEXTURES),
                ("m.toml", DefKind::Material, &text),
            ],
        )])
    };
    for (lines, problem) in [
        (
            "roughness = 1.5",
            "`roughness` must be between 0 and 1, got 1.5",
        ),
        (
            "roughness = -0.1",
            "`roughness` must be between 0 and 1, got -0.1",
        ),
        (
            "roughness = nan",
            "`roughness` must be between 0 and 1, got NaN",
        ),
        (
            "roughness = 1\nmetallic = 2",
            "`metallic` must be between 0 and 1, got 2.0",
        ),
        (
            "roughness = 1\nemissive = -1",
            "`emissive` must be a finite number >= 0, got -1.0",
        ),
        (
            "roughness = 1\nemissive = inf",
            "`emissive` must be a finite number >= 0, got inf",
        ),
        (
            "roughness = 1\nflags = [\"cutout\", \"cutout\"]",
            "flag `cutout` is listed twice",
        ),
    ] {
        let message = invalid_message(material(lines));
        assert_eq!(message, format!("material t:m: {problem}"));
    }

    let registry = material(
        r#"
roughness = 0.25
metallic = 1
emissive = 3.5
tint = "water"
flags = ["translucent", "cutout"]
"#,
    )
    .unwrap();
    let def = &registry.materials()[registry.material("t:m").unwrap().index()];
    assert_eq!(
        (def.roughness, def.metallic, def.emissive),
        (0.25, 1.0, 3.5)
    );
    assert_eq!(def.tint, TintClass::Water);
    assert_eq!(
        def.flags,
        MaterialFlags::CUTOUT | MaterialFlags::TRANSLUCENT
    );
    assert_eq!(registry.textures()[def.texture.index()].key, "t:a");

    // Everything but the texture has a default: the texture as it is painted, no glow.
    let registry = material("").unwrap();
    let def = &registry.materials()[0];
    assert_eq!((def.roughness, def.metallic, def.emissive), (1.0, 0.0, 0.0));
    assert_eq!(
        (def.tint, def.flags),
        (TintClass::None, MaterialFlags::empty())
    );
}

#[test]
fn texture_rules() {
    let texture = |lines: &str| {
        let text = format!("[[texture]]\nkey = \"t:t\"\n{lines}\n");
        assemble(&[pack_of(DefKind::Texture, &text)])
    };
    // A texture of our own with the given recipe and further fields of `procedural`.
    let own = |recipe: &str, more: &str| {
        format!("provenance = \"own\"\nprocedural = {{ recipe = \"{recipe}\", seed = 1{more} }}")
    };
    let stone = "procedural = { recipe = \"stone\", seed = 1 }";
    let cases = [
        ("provenance = \"own\"".to_owned(), "no source"),
        (
            format!("provenance = \"own\"\nfile = \"a.png\"\n{stone}"),
            "give either `procedural` or `file`, not both",
        ),
        (own("Stone", ""), "recipe name `Stone` must be made of"),
        (own("", ""), "recipe name `` must be made of"),
        (
            own("stone", ", params = { \"Leaf Colour\" = \"#fff\" }"),
            "parameter name `Leaf Colour` must be made of",
        ),
        (
            format!("provenance = \"placeholder\"\n{stone}"),
            "a placeholder texture must state its `licence`",
        ),
        (
            format!("provenance = \"own\"\nlicence = \"  \"\n{stone}"),
            "`licence` must not be empty",
        ),
        (
            "provenance = \"own\"\nfile = \"textures/a.png\"".to_owned(),
            "file textures need a pack directory",
        ),
    ];
    for (lines, problem) in cases {
        let message = invalid_message(texture(&lines));
        assert!(message.starts_with("texture t:t: "), "{message}");
        assert!(message.contains(problem), "{message}");
    }

    let registry = texture(
        r##"
provenance = "placeholder"
licence = "CC-BY 4.0, A. Artist"
[texture.procedural]
recipe = "ore"
seed = 18446744073709551615
params = { ore = "#e0843c" }
"##,
    )
    .unwrap();
    let def = &registry.textures()[0];
    assert_eq!(def.provenance, Provenance::Placeholder);
    assert_eq!(def.licence.as_deref(), Some("CC-BY 4.0, A. Artist"));
    let TextureSource::Procedural {
        recipe,
        params,
        seed,
    } = &def.source
    else {
        panic!("expected a procedural texture, got {:?}", def.source);
    };
    assert_eq!((recipe.as_str(), *seed), ("ore", u64::MAX));
    assert_eq!(params.color("ore"), Some([0xe0, 0x84, 0x3c]));

    // Own art needs no licence, but may state one.
    let registry = texture(&format!("provenance = \"own\"\n{stone}")).unwrap();
    assert_eq!(registry.textures()[0].licence, None);
    let licensed = format!("provenance = \"own\"\nlicence = \"house style\"\n{stone}");
    let registry = texture(&licensed).unwrap();
    assert_eq!(
        registry.textures()[0].licence.as_deref(),
        Some("house style")
    );
}

#[test]
fn pack_paths_are_portable() {
    for good in [
        "a.png",
        "textures/a.png",
        "textures/img/a b.png",
        "a/b/c.d.png",
        "..a/b..png",
    ] {
        assert_eq!(file_path::check(good), Ok(()), "{good}");
    }
    for bad in [
        "",
        "/abs.png",
        "C:/abs.png",
        "c:rel.png",
        "textures\\a.png",
        "../other/a.png",
        "textures/../../a.png",
        "textures/..",
        "./a.png",
        "textures//a.png",
        "textures/",
    ] {
        let message = file_path::check(bad).expect_err(bad);
        assert!(
            message.contains("relative to the pack directory"),
            "{message}"
        );
    }
}

#[test]
fn keys_are_checked_for_every_kind() {
    let cases = [
        (
            DefKind::Texture,
            r#"[[texture]]
key = "Stone"
provenance = "own"
procedural = { recipe = "stone", seed = 1 }
"#,
        ),
        (
            DefKind::Material,
            r#"[[material]]
key = "t:Stone"
texture = "t:a"
roughness = 1
"#,
        ),
        (
            DefKind::Block,
            r#"[[block]]
key = "t:stone[up=+x]"
hardness = 1
density = 1
"#,
        ),
    ];
    for (kind, text) in cases {
        match assemble(&[pack_of(kind, text)]) {
            Err(ContentError::Invalid {
                kind: reported,
                message,
                line,
                ..
            }) => {
                assert_eq!((reported, line), (kind, 1));
                assert!(message.contains("namespace:name"), "{message}");
            }
            other => panic!("expected an invalid key, got {other:?}"),
        }
    }
}

#[test]
fn air_stays_empty() {
    // A pack may describe air, as long as it stays the empty block.
    let registry = load_blocks(
        r#"
[[block]]
key = "cubeplanet:air"
render = "none"
hardness = 0
density = 1.2
tags = ["gas"]
"#,
    )
    .unwrap();
    let air = registry.block(AIR_KEY).unwrap();
    assert_eq!(air.default_state(), BlockStateId::AIR);
    assert_eq!(air.density, 1.2);
    assert!(air.has_tag("gas"));
    assert_eq!(registry.tables().flags[0], BlockFlags::empty());

    for lines in [
        "faces = { all = \"t:a\" }",
        // Drawn, although it would carry no flag.
        "solid = false\nopaque = false\nfaces = { all = \"t:a\" }",
        "render = \"none\"\nsolid = true",
        "render = \"none\"\nlight = 3",
        "render = \"none\"\norient = \"up6\"",
        "render = \"none\"\norient = \"axis3\"",
        "render = \"none\"\norient = \"rot24\"",
        "render = \"none\"\nsolid = true\nstructural = true",
        "render = \"fluid\"\nfaces = { all = \"t:a\" }",
    ] {
        let message = invalid_message(load_blocks(&format!(
            "[[block]]\nkey = \"cubeplanet:air\"\nhardness = 0\ndensity = 0\n{lines}\n"
        )));
        assert!(
            message.starts_with("block cubeplanet:air: `cubeplanet:air` is the empty block"),
            "{lines}: {message}"
        );
    }
}

#[test]
fn ids_follow_the_state_keys_even_across_blocks() {
    // `t:a2` sorts between the default state of `t:a` and its other states, and
    // `aaa:first` sorts before `cubeplanet:air`, which keeps state 0 all the same.
    let registry = load_blocks(
        r#"
[[block]]
key = "t:a"
orient = "up6"
hardness = 1
density = 1
faces = { top = "t:a", side = "t:b", bottom = "t:b" }

[[block]]
key = "t:a2"
hardness = 1
density = 1
faces = { all = "t:a" }

[[block]]
key = "aaa:first"
orient = "axis3"
hardness = 1
density = 1
faces = { all = "t:b" }
"#,
    )
    .unwrap();
    let keys: Vec<&str> = (0..registry.state_count())
        .map(|index| registry.state_key(BlockStateId(index as u32)))
        .collect();
    assert_eq!(
        keys,
        [
            "cubeplanet:air",
            "aaa:first",
            "aaa:first[axis=x]",
            "aaa:first[axis=z]",
            "t:a",
            "t:a2",
            "t:a[up=+x]",
            "t:a[up=+z]",
            "t:a[up=-x]",
            "t:a[up=-y]",
            "t:a[up=-z]",
        ]
    );
    // The states of `t:a` are not contiguous, yet each of them knows its block, its
    // orientation and its row of the tables.
    let block = registry.block("t:a").unwrap();
    let ids: Vec<u32> = block.states().iter().map(|state| state.0).collect();
    assert_eq!(ids, [6, 8, 4, 9, 7, 10]);
    for face in cp_core::face::Face::ALL {
        let state = block.state_for_up(face);
        assert_eq!(registry.block_of(state).key, "t:a");
        assert_eq!(registry.orientation(state).up(), face);
        let render = registry.tables().render[state.index()];
        assert_eq!(
            render.face(face).material,
            registry.material("t:a").unwrap()
        );
        assert_eq!(
            render.face(face.opposite()).material,
            registry.material("t:b").unwrap()
        );
    }
    assert_eq!(registry.block_of(BlockStateId(5)).key, "t:a2");
    assert_eq!(
        registry.block("aaa:first").unwrap().states(),
        [BlockStateId(2), BlockStateId(1), BlockStateId(3)]
    );
}

#[test]
fn duplicates_inside_a_pack_are_errors() {
    let result = load_blocks(
        r#"[[block]]
key = "t:x"
hardness = 1
density = 1
faces = { all = "t:a" }

[[block]]
key = "t:x"
hardness = 2
density = 1
faces = { all = "t:a" }
"#,
    );
    match result {
        Err(ContentError::DuplicateKey {
            kind,
            key,
            line,
            first_line,
            file,
            first_file,
        }) => {
            assert_eq!((kind, key.as_str()), (DefKind::Block, "t:x"));
            assert_eq!((line, first_line), (7, 1));
            assert_eq!(file, first_file);
            assert!(file.ends_with("b.toml"));
        }
        other => panic!("expected a duplicate key, got {other:?}"),
    }
}

#[test]
fn the_same_key_may_name_one_definition_of_each_kind() {
    // `t:a` is a texture, a material and a block here.
    let registry = load_blocks(
        r#"
[[block]]
key = "t:a"
hardness = 1
density = 1
faces = { all = "t:a" }
"#,
    )
    .unwrap();
    assert!(registry.block("t:a").is_some());
    assert!(registry.material("t:a").is_some());
    assert!(registry.texture("t:a").is_some());
    assert_eq!(registry.state("t:a"), Some(BlockStateId(1)));
}

#[test]
fn later_packs_override_whole_definitions() {
    let base = pack_with_blocks(
        r#"
[[block]]
key = "t:x"
orient = "up6"
hardness = 1
density = 1
faces = { all = "t:a" }
tags = ["old"]
"#,
    );
    let overlay = PackSource::embedded(
        "<overlay>",
        "format = 1\nname = \"overlay\"\nversion = \"2\"\n",
        &[
            (
                "m.toml",
                DefKind::Material,
                r#"
[[material]]
key = "t:a"
texture = "t:b"
roughness = 0.1
"#,
            ),
            (
                "b.toml",
                DefKind::Block,
                r#"
[[block]]
key = "t:x"
hardness = 9
density = 1
faces = { all = "t:b" }

[[block]]
key = "o:new"
hardness = 1
density = 1
faces = { all = "t:a" }
"#,
            ),
        ],
    );

    let alone = assemble(std::slice::from_ref(&base)).unwrap();
    assert_eq!(alone.block("t:x").unwrap().orient, Orient::Up6);
    assert_eq!(alone.state_count(), 1 + 6);

    let both = assemble(&[base, overlay]).unwrap();
    let names: Vec<_> = both.packs().iter().map(|pack| pack.name.as_str()).collect();
    assert_eq!(names, ["test", "overlay"]);
    // The block is replaced as a whole: fields the overlay leaves out fall back to their
    // defaults, not to the values of the overridden definition.
    let block = both.block("t:x").unwrap();
    assert_eq!(block.orient, Orient::None);
    assert_eq!(block.hardness, 9.0);
    assert!(block.tags.is_empty());
    assert_eq!(block.faces, Some([both.material("t:b").unwrap(); 6]));
    assert_eq!(both.state_count(), 1 + 1 + 1);
    assert!(both.state("t:x[up=+x]").is_none());
    // The overridden material is seen by a block of either pack.
    let material = &both.materials()[both.material("t:a").unwrap().index()];
    assert_eq!(material.roughness, 0.1);
    assert_eq!(both.textures()[material.texture.index()].key, "t:b");
    assert!(both.block("o:new").is_some());
    assert_ne!(alone.content_hash(), both.content_hash());
}

#[test]
fn references_are_resolved_after_merging() {
    let block = "[[block]]\nkey = \"t:x\"\nhardness = 1\ndensity = 1\nfaces = { all = \"t:a\" }\n";
    let blocks =
        || PackSource::embedded("<blocks>", MANIFEST, &[("b.toml", DefKind::Block, block)]);
    let art = || {
        PackSource::embedded(
            "<art>",
            "format = 1\nname = \"art\"\nversion = \"1\"\n",
            &[
                ("t.toml", DefKind::Texture, TEXTURES),
                ("m.toml", DefKind::Material, MATERIALS),
            ],
        )
    };

    // The block pack alone is broken: its block names a material that it does not have.
    match assemble(&[blocks()]) {
        Err(ContentError::UnknownReference {
            kind,
            key,
            target_kind,
            target,
            line,
            ..
        }) => {
            assert_eq!((kind, key.as_str(), line), (DefKind::Block, "t:x", 1));
            assert_eq!((target_kind, target.as_str()), (DefKind::Material, "t:a"));
        }
        other => panic!("expected an unknown reference, got {other:?}"),
    }
    // A pack that brings the material and its texture completes it, whether it is loaded
    // before or after.
    let forward = assemble(&[blocks(), art()]).unwrap();
    let backward = assemble(&[art(), blocks()]).unwrap();
    assert!(forward.block("t:x").is_some());
    assert_eq!(forward.content_hash(), backward.content_hash());
}

#[test]
fn dangling_texture_references_are_errors() {
    let result = assemble(&[pack_of(
        DefKind::Material,
        "\n\n[[material]]\nkey = \"t:m\"\ntexture = \"t:nowhere\"\nroughness = 1\n",
    )]);
    match result {
        Err(ContentError::UnknownReference {
            kind,
            key,
            target_kind,
            target,
            line,
            ..
        }) => {
            assert_eq!((kind, key.as_str(), line), (DefKind::Material, "t:m", 3));
            assert_eq!(
                (target_kind, target.as_str()),
                (DefKind::Texture, "t:nowhere")
            );
        }
        other => panic!("expected an unknown reference, got {other:?}"),
    }
}

#[test]
fn errors_are_reported_in_key_order() {
    // Two broken blocks: whichever comes first in the file, the one with the smaller key is
    // reported, and textures are checked before materials before blocks.
    let faces = "faces = { all = \"t:a\" }";
    let broken =
        |key: &str| format!("[[block]]\nkey = \"{key}\"\nhardness = -1\ndensity = 1\n{faces}\n");
    for blocks in [
        format!("{}{}", broken("t:b"), broken("t:a")),
        format!("{}{}", broken("t:a"), broken("t:b")),
    ] {
        let message = invalid_message(load_blocks(&blocks));
        assert!(message.starts_with("block t:a: "), "{message}");
    }

    let pack = PackSource::embedded(
        "<test>",
        MANIFEST,
        &[
            ("t.toml", DefKind::Texture, TEXTURES),
            (
                "m.toml",
                DefKind::Material,
                "[[material]]\nkey = \"t:m\"\ntexture = \"t:a\"\nroughness = 7\n",
            ),
            ("b.toml", DefKind::Block, &broken("t:a")),
        ],
    );
    let message = invalid_message(assemble(&[pack]));
    assert!(message.starts_with("material t:m: "), "{message}");
}

#[test]
fn id_limits_are_errors_not_panics() {
    assert!(check_limit("things", 0, 3).is_ok());
    assert!(check_limit("things", 3, 3).is_ok());
    assert!(matches!(
        check_limit("things", 4, 3),
        Err(ContentError::Limit {
            what: "things",
            count: 4,
            limit: 3
        })
    ));

    let materials = |count: usize| {
        let mut text = String::new();
        for index in 0..count {
            text.push_str(&format!(
                "[[material]]\nkey = \"t:m{index:05}\"\ntexture = \"t:a\"\nroughness = 1\n"
            ));
        }
        assemble(&[PackSource::embedded(
            "<test>",
            MANIFEST,
            &[
                ("t.toml", DefKind::Texture, TEXTURES),
                ("m.toml", DefKind::Material, &text),
            ],
        )])
    };

    // Exactly as many materials as a `MaterialId` can address: the last one gets the last id.
    let full = materials(MAX_MATERIALS).unwrap();
    assert_eq!(full.materials().len(), 65_536);
    assert_eq!(full.material("t:m65535"), Some(MaterialId(u16::MAX)));
    assert_eq!(full.material("t:m00000"), Some(MaterialId(0)));

    // One more does not fit.
    match materials(MAX_MATERIALS + 1) {
        Err(ContentError::Limit { what, count, limit }) => {
            assert_eq!((what, count, limit), ("materials", 65_537, 65_536));
        }
        other => panic!("expected a limit error, got {:?}", other.map(|_| ())),
    }
}

/// Six materials `t:px` … `t:nz`, one per face of a block, on top of the standard ones.
fn six_materials() -> String {
    let mut text = MATERIALS.to_owned();
    for name in ["px", "nx", "py", "ny", "pz", "nz"] {
        text.push_str(&format!(
            "\n[[material]]\nkey = \"t:{name}\"\ntexture = \"t:a\"\n"
        ));
    }
    text
}

/// The standard textures, [`six_materials`] and the given blocks.
fn load_with_six(blocks: &str) -> Result<Registry, ContentError> {
    let materials = six_materials();
    assemble(&[PackSource::embedded(
        "<test>",
        MANIFEST,
        &[
            ("t.toml", DefKind::Texture, TEXTURES),
            ("m.toml", DefKind::Material, &materials),
            ("b.toml", DefKind::Block, blocks),
        ],
    )])
}

const MACHINE: &str = r#"
[[block]]
key = "t:machine"
orient = "rot24"
hardness = 1
density = 1
[block.faces]
"+x" = "t:px"
"-x" = "t:nx"
"+y" = "t:py"
"-y" = "t:ny"
"+z" = "t:pz"
"-z" = "t:nz"
"#;

#[test]
fn rot24_states_carry_their_front_to_the_world_face_of_the_orientation() {
    use cp_core::face::{Face, Orientation};

    let registry = load_with_six(MACHINE).unwrap();
    let block = registry.block("t:machine").unwrap();
    assert_eq!(block.orient, Orient::Rot24);
    assert_eq!(block.states().len(), 24);
    assert_eq!(registry.state_count(), 1 + 24);
    let local =
        ["t:px", "t:nx", "t:py", "t:ny", "t:pz", "t:nz"].map(|key| registry.material(key).unwrap());
    let front = registry.material("t:pz").unwrap();

    for orientation in Orientation::ALL {
        let state = block.state_for(orientation);
        assert_eq!(block.states()[orientation.index()], state);
        assert_eq!(registry.orientation(state), orientation);
        // The key writes what differs from `turn=0`, `up=+y`, sorted by name.
        let key = registry.state_key(state);
        let mut expected = Vec::new();
        if orientation.turn() != 0 {
            expected.push(format!("turn={}", orientation.turn()));
        }
        if orientation.up() != Face::PosY {
            let sign = if orientation.up().is_negative() {
                '-'
            } else {
                '+'
            };
            let axis = ["x", "y", "z"][orientation.up().axis()];
            expected.push(format!("up={sign}{axis}"));
        }
        let expected = if expected.is_empty() {
            "t:machine".to_owned()
        } else {
            format!("t:machine[{}]", expected.join(","))
        };
        assert_eq!(key, expected);
        assert_eq!(registry.state(key), Some(state));

        let render = registry.tables().render[state.index()];
        assert_eq!(render.class, RenderClass::Cube);
        // The front material shows on the world face the orientation's front points to.
        assert_eq!(render.face(orientation.front()).material, front, "{key}");
        for face in Face::ALL {
            let world = orientation.world_face(face);
            assert_eq!(render.face(world).material, local[face.index()], "{key}");
        }
        for world in Face::ALL {
            assert_eq!(
                render.face(world).uv_rot,
                orientation.uv_rot(world),
                "{key}"
            );
        }
    }
    assert_eq!(block.default_state(), registry.state("t:machine").unwrap());
    assert_eq!(
        registry.orientation(block.default_state()),
        Orientation::IDENTITY
    );
}

#[test]
fn rot24_rotation_composes() {
    use cp_core::face::Orientation;

    let registry = load_with_six(MACHINE).unwrap();
    let block = registry.block("t:machine").unwrap();
    for &state in block.states() {
        for a in Orientation::ALL {
            let once = registry.rotated(state, a);
            assert_eq!(
                registry.orientation(once),
                a.compose(registry.orientation(state))
            );
            for b in Orientation::ALL {
                assert_eq!(
                    registry.rotated(once, b),
                    registry.rotated(state, b.compose(a))
                );
            }
            // Turning back restores the state.
            assert_eq!(registry.rotated(once, a.inverse()), state);
        }
    }
}

#[test]
fn model_blocks_and_their_lod_proxies() {
    use cp_core::block::{LodClass, LodInfo};
    use cp_core::face::Orientation;

    let registry = load_blocks(
        r#"
[[block]]
key = "t:stone"
faces = { all = "t:a" }
hardness = 1
density = 1

[[block]]
key = "t:log"
orient = "axis3"
structural = true
faces = { all = "t:b" }
hardness = 1
density = 1

[[block]]
key = "t:water"
render = "fluid"
faces = { all = "t:b" }
hardness = 0
density = 1

[[block]]
key = "t:machine"
orient = "rot24"
render = "model"
lod_proxy = "t:stone"
hardness = 1
density = 1

[[block]]
key = "t:pile"
render = "model"
structural = true
lod_proxy = "t:log[axis=z]"
hardness = 1
density = 1

[[block]]
key = "t:pump"
render = "model"
solid = false
lod_proxy = "t:water"
hardness = 1
density = 1

[[block]]
key = "t:lever"
orient = "up6"
render = "model"
solid = false
hardness = 1
density = 1

[[block]]
key = "t:glass"
opaque = false
faces = { all = "t:a" }
hardness = 1
density = 1

[[block]]
key = "t:ghost"
solid = false
opaque = false
faces = { all = "t:a" }
hardness = 1
density = 1

[[block]]
key = "t:barrier"
render = "none"
solid = true
hardness = 1
density = 1
"#,
    )
    .unwrap();
    let id = |key: &str| registry.state(key).unwrap();
    let tables = registry.tables();
    let lod = |key: &str| tables.lod[id(key).index()];

    let machine = registry.block("t:machine").unwrap();
    assert_eq!(machine.render, RenderClass::Model);
    assert!(machine.solid && !machine.opaque && machine.faces.is_none());
    assert_eq!(machine.lod_proxy.as_deref(), Some("t:stone"));
    for &state in machine.states() {
        assert_eq!(tables.flags[state.index()], BlockFlags::SOLID);
        assert_eq!(tables.render[state.index()].class, RenderClass::Model);
        assert_eq!(
            tables.lod[state.index()],
            LodInfo {
                class: LodClass::Solid,
                proxy: id("t:stone")
            }
        );
        assert_eq!(
            tables.material_color[state.index()],
            crate::NO_MATERIAL_COLOR
        );
        assert_eq!(tables.cull_group[state.index()], 0);
    }
    assert_eq!(machine.states().len(), 24);
    assert_eq!(
        registry.rotated(machine.default_state(), Orientation::ALL[9]),
        machine.states()[9]
    );

    assert_eq!(
        tables.flags[id("t:pile").index()],
        BlockFlags::SOLID | BlockFlags::STRUCTURAL
    );
    assert_eq!(
        lod("t:pile"),
        LodInfo {
            class: LodClass::Solid,
            proxy: id("t:log[axis=z]")
        }
    );
    assert_eq!(
        lod("t:pump"),
        LodInfo {
            class: LodClass::Fluid,
            proxy: id("t:water")
        }
    );
    assert_eq!(tables.flags[id("t:pump").index()], BlockFlags::empty());
    for key in ["t:lever", "t:lever[up=-y]"] {
        assert_eq!(lod(key), LodInfo::EMPTY, "{key}");
    }
    // Cube-shaped states stand in for themselves, whatever their class.
    for (key, class) in [
        ("t:stone", LodClass::Solid),
        ("t:log[axis=x]", LodClass::Solid),
        ("t:glass", LodClass::Solid),
        ("t:water", LodClass::Fluid),
        ("t:ghost", LodClass::Empty),
        ("t:barrier", LodClass::Empty),
    ] {
        assert_eq!(
            lod(key),
            LodInfo {
                class,
                proxy: id(key)
            },
            "{key}"
        );
    }
}

#[test]
fn rotated_models_turn_their_lod_proxy() {
    use cp_core::block::{LodClass, LodInfo};
    use cp_core::face::{Face, Orientation};

    let registry = load_blocks(
        r#"
[[block]]
key = "t:log"
orient = "axis3"
structural = true
faces = { all = "t:b" }
hardness = 1
density = 1

[[block]]
key = "t:furnace"
orient = "rot24"
faces = { all = "t:a" }
hardness = 1
density = 1

[[block]]
key = "t:stone"
faces = { all = "t:a" }
hardness = 1
density = 1

[[block]]
key = "t:shaft"
orient = "rot24"
render = "model"
lod_proxy = "t:log"
hardness = 1
density = 1

[[block]]
key = "t:crusher"
orient = "rot24"
render = "model"
lod_proxy = "t:furnace[turn=1]"
hardness = 1
density = 1

[[block]]
key = "t:vent"
orient = "up6"
render = "model"
lod_proxy = "t:furnace"
hardness = 1
density = 1

[[block]]
key = "t:gear"
orient = "rot24"
render = "model"
lod_proxy = "t:stone"
hardness = 1
density = 1
"#,
    )
    .unwrap();
    let id = |key: &str| registry.state(key).unwrap();
    let lod = |key: &str| registry.tables().lod[id(key).index()];

    // A shaft along +y stands in as a log along y; laid along x or z, as a log along x or z,
    // whatever its turn about its own axis.
    for (shaft, log) in [
        ("t:shaft", "t:log"),
        ("t:shaft[turn=1]", "t:log"),
        ("t:shaft[up=-y]", "t:log"),
        ("t:shaft[up=+x]", "t:log[axis=x]"),
        ("t:shaft[turn=3,up=-x]", "t:log[axis=x]"),
        ("t:shaft[up=+z]", "t:log[axis=z]"),
        ("t:shaft[turn=2,up=-z]", "t:log[axis=z]"),
    ] {
        assert_eq!(
            lod(shaft),
            LodInfo {
                class: LodClass::Solid,
                proxy: id(log)
            },
            "{shaft}"
        );
    }
    // A vent on the -Y face of a planet shows its stand-in's top towards -Y as well.
    assert_eq!(lod("t:vent[up=-y]").proxy, id("t:furnace[up=-y]"));
    assert_eq!(lod("t:vent[up=+z]").proxy, id("t:furnace[up=+z]"));

    // In general every state of a model gets the written proxy turned by the state's
    // orientation: the default state keeps it as written, a proxy that cannot be rotated
    // stays the same, and the class is always the proxy block's.
    for model in ["t:shaft", "t:crusher", "t:vent", "t:gear"] {
        let block = registry.block(model).unwrap();
        let written = id(block.lod_proxy.as_deref().unwrap());
        let written_block = registry.block_of(written);
        assert_eq!(
            registry.tables().lod[block.default_state().index()].proxy,
            written,
            "{model}"
        );
        for &state in block.states() {
            let info = registry.tables().lod[state.index()];
            assert_eq!(
                info.proxy,
                registry.rotated(written, registry.orientation(state)),
                "{}",
                registry.state_key(state)
            );
            assert_eq!(info.class, LodClass::Solid);
            assert_eq!(registry.block_of(info.proxy).key, written_block.key);
        }
    }
    for &state in registry.block("t:gear").unwrap().states() {
        assert_eq!(registry.tables().lod[state.index()].proxy, id("t:stone"));
    }

    // Turning a placed model turns its stand-in the same way.
    let crusher = id("t:crusher[turn=2,up=+x]");
    for o in Orientation::ALL {
        let turned = registry.rotated(crusher, o);
        assert_eq!(
            registry.tables().lod[turned.index()].proxy,
            registry.rotated(registry.tables().lod[crusher.index()].proxy, o),
            "{o:?}"
        );
    }
    let quarter = Orientation::from_up(Face::PosX);
    assert_eq!(
        registry.tables().lod[registry.rotated(id("t:crusher"), quarter).index()].proxy,
        registry.rotated(id("t:furnace[turn=1]"), quarter)
    );
}

#[test]
fn lod_proxies_must_name_cube_shaped_states() {
    let blocks = |extra: &str| {
        format!(
            r#"
[[block]]
key = "t:stone"
faces = {{ all = "t:a" }}
hardness = 1
density = 1

[[block]]
key = "t:log"
orient = "axis3"
faces = {{ all = "t:b" }}
hardness = 1
density = 1

[[block]]
key = "t:barrier"
render = "none"
hardness = 1
density = 1

[[block]]
key = "t:gear"
render = "model"
hardness = 1
density = 1
{extra}
"#
        )
    };
    let model = |proxy: &str| {
        blocks(&format!(
            "\n[[block]]\nkey = \"t:x\"\nrender = \"model\"\nlod_proxy = \"{proxy}\"\n\
             hardness = 1\ndensity = 1\n"
        ))
    };
    for (proxy, problem) in [
        (
            "t:Stone",
            "`lod_proxy` is not a state key: `t:Stone` is not a valid key",
        ),
        ("t:log[]", "`lod_proxy` is not a state key"),
        ("t:log[up=+x,axis=z]", "`lod_proxy` is not a state key"),
        (
            "t:stone[up=+x]",
            "`lod_proxy` `t:stone[up=+x]` is not a state of `t:stone`",
        ),
        (
            "t:log[axis=y]",
            "`lod_proxy` `t:log[axis=y]` is not a state of `t:log`",
        ),
        ("t:log[axis=w]", "is not a state of `t:log`"),
        (
            "t:gear",
            "must be a state of a `cube`, `cutout` or `fluid` block",
        ),
        (
            "t:barrier",
            "must be a state of a `cube`, `cutout` or `fluid` block",
        ),
        (
            "t:x",
            "must be a state of a `cube`, `cutout` or `fluid` block",
        ),
    ] {
        let message = invalid_message(load_blocks(&model(proxy)));
        assert!(message.starts_with("block t:x: "), "{proxy}: {message}");
        assert!(message.contains(problem), "{proxy}: {message}");
    }
    match load_blocks(&model("t:nothing[axis=x]")) {
        Err(ContentError::UnknownReference {
            key,
            target_kind,
            target,
            ..
        }) => {
            assert_eq!(key, "t:x");
            assert_eq!(
                (target_kind, target.as_str()),
                (DefKind::Block, "t:nothing")
            );
        }
        other => panic!("expected an unknown reference, got {other:?}"),
    }
    // A proxy may name any state of a cube-shaped block.
    assert!(load_blocks(&model("t:log[axis=x]")).is_ok());

    // Only model blocks have a proxy, and they have no faces and cover nothing.
    let message = invalid_message(block_with("faces = { all = \"t:a\" }\nlod_proxy = \"t:x\""));
    assert!(
        message.contains("only a `model` block has a `lod_proxy`"),
        "{message}"
    );
    let message = invalid_message(block_with("render = \"model\"\nfaces = { all = \"t:a\" }"));
    assert!(message.contains("cannot have `faces`"), "{message}");
    let message = invalid_message(block_with("render = \"model\"\nopaque = true"));
    assert!(
        message.contains("a `model` block cannot be `opaque`"),
        "{message}"
    );
}

#[test]
fn cull_groups_belong_to_see_through_meshed_blocks() {
    for (lines, problem) in [
        (
            "render = \"cutout\"\nfaces = { all = \"t:a\" }\ncull_group = \"Leaves\"",
            "cull group `Leaves` must be made of",
        ),
        (
            "render = \"cutout\"\nfaces = { all = \"t:a\" }\ncull_group = \"\"",
            "cull group `` must be made of",
        ),
        (
            "faces = { all = \"t:a\" }\ncull_group = \"stone\"",
            "an `opaque` block cannot have a `cull_group`",
        ),
        (
            "render = \"none\"\ncull_group = \"air\"",
            "can have a `cull_group`",
        ),
        (
            "render = \"model\"\ncull_group = \"gears\"",
            "can have a `cull_group`",
        ),
    ] {
        let message = invalid_message(block_with(lines));
        assert!(message.contains(problem), "{lines}: {message}");
    }
    for lines in [
        "render = \"cutout\"\nfaces = { all = \"t:a\" }\ncull_group = \"leaves\"",
        "render = \"fluid\"\nfaces = { all = \"t:a\" }\ncull_group = \"water\"",
        "opaque = false\nfaces = { all = \"t:a\" }\ncull_group = \"glass_2\"",
    ] {
        let registry = block_with(lines).unwrap();
        let state = registry.state("t:x").unwrap();
        assert_eq!(registry.tables().cull_group[state.index()], 1, "{lines}");
        assert_eq!(registry.cull_groups().len(), 1);
    }
}

#[test]
fn cull_groups_are_numbered_by_name_whatever_the_pack_order() {
    let pack = |name: &str, blocks: &str| {
        PackSource::embedded(
            "<test>",
            &format!("format = 1\nname = \"{name}\"\nversion = \"1\"\n"),
            &[
                ("t.toml", DefKind::Texture, TEXTURES),
                ("m.toml", DefKind::Material, MATERIALS),
                ("b.toml", DefKind::Block, blocks),
            ],
        )
    };
    let see_through = |key: &str, group: &str| {
        format!(
            "[[block]]\nkey = \"{key}\"\nrender = \"cutout\"\nfaces = {{ all = \"t:a\" }}\n\
             cull_group = \"{group}\"\nhardness = 1\ndensity = 1\n\n"
        )
    };
    let first = || {
        pack(
            "first",
            &format!(
                "{}{}[[block]]\nkey = \"a:stone\"\nfaces = {{ all = \"t:a\" }}\nhardness = 1\n\
                 density = 1\n",
                see_through("a:water", "water"),
                see_through("a:leaves", "leaves")
            ),
        )
    };
    let second = || {
        pack(
            "second",
            &format!(
                "{}{}",
                see_through("b:glass", "glass"),
                see_through("b:leaves", "leaves")
            ),
        )
    };
    let forward = assemble(&[first(), second()]).unwrap();
    let backward = assemble(&[second(), first()]).unwrap();
    for registry in [&forward, &backward] {
        assert_eq!(registry.cull_groups(), ["glass", "leaves", "water"]);
        let group = |key: &str| registry.tables().cull_group[registry.state(key).unwrap().index()];
        assert_eq!(group("b:glass"), 1);
        assert_eq!(group("a:leaves"), 2);
        assert_eq!(group("b:leaves"), 2);
        assert_eq!(group("a:water"), 3);
        assert_eq!(group("a:stone"), 0);
        assert_eq!(group(AIR_KEY), 0);
    }
    assert_eq!(forward.tables(), backward.tables());
    assert_eq!(forward.content_hash(), backward.content_hash());
    // A pack alone numbers only the groups it knows.
    let alone = assemble(&[second()]).unwrap();
    assert_eq!(alone.cull_groups(), ["glass", "leaves"]);
}

#[test]
fn structural_blocks_are_solid() {
    for lines in [
        "render = \"fluid\"\nfaces = { all = \"t:a\" }\nstructural = true",
        "solid = false\nopaque = false\nfaces = { all = \"t:a\" }\nstructural = true",
        "render = \"none\"\nstructural = true",
    ] {
        let message = invalid_message(block_with(lines));
        assert!(
            message.contains("only a `solid` block can be `structural`"),
            "{lines}: {message}"
        );
    }
    let registry = block_with("faces = { all = \"t:a\" }\nstructural = true").unwrap();
    let state = registry.state("t:x").unwrap();
    assert_eq!(
        registry.tables().flags[state.index()],
        BlockFlags::SOLID | BlockFlags::OPAQUE | BlockFlags::STRUCTURAL
    );
    assert!(registry.block("t:x").unwrap().structural);
}

#[test]
fn ext_tables_are_stored_as_written() {
    let table_form = r#"
[[block]]
key = "t:x"
hardness = 1
density = 1
faces = { all = "t:a" }
[block.ext.forestry]
burn_time = 300
fuel = true
grades = [1.5, -2, "high"]
[block.ext.forestry.drops]
sapling = 0.05
[block.ext.zz_other]
when = 1979-05-27T07:32:00Z
"#;
    let registry = load_blocks(table_form).unwrap();
    let block = registry.block("t:x").unwrap();
    let namespaces: Vec<&str> = block.ext.keys().map(String::as_str).collect();
    assert_eq!(namespaces, ["forestry", "zz_other"]);
    let forestry = block.ext("forestry").unwrap();
    assert_eq!(forestry["burn_time"].as_integer(), Some(300));
    assert_eq!(forestry["fuel"].as_bool(), Some(true));
    assert_eq!(forestry["grades"].as_array().map(Vec::len), Some(3));
    assert_eq!(forestry["drops"]["sapling"].as_float(), Some(0.05));
    assert!(block.ext("zz_other").unwrap()["when"].is_datetime());
    assert!(block.ext("nothing").is_none());
    // The engine reads nothing from it: the tables are those of a plain block.
    let plain = block_with("faces = { all = \"t:a\" }").unwrap();
    assert_eq!(registry.tables(), plain.tables());

    // The same data written inline and in another order hashes the same.
    let inline_form = r#"
[[block]]
key = "t:x"
ext = { zz_other = { when = 1979-05-27T07:32:00Z }, forestry = { drops = { sapling = 0.05 }, grades = [1.5, -2, "high"], fuel = true, burn_time = 300 } }
hardness = 1
density = 1
faces = { all = "t:a" }
"#;
    let inline = load_blocks(inline_form).unwrap();
    assert_eq!(inline.block("t:x").unwrap().ext, block.ext);
    assert_eq!(inline.content_hash(), registry.content_hash());
    assert_ne!(plain.content_hash(), registry.content_hash());

    // Every value counts.
    let mut hashes = vec![registry.content_hash(), plain.content_hash()];
    for (from, to) in [
        ("burn_time = 300", "burn_time = 301"),
        ("fuel = true", "fuel = false"),
        ("[1.5, -2, \"high\"]", "[1.5, -2, \"low\"]"),
        ("[1.5, -2, \"high\"]", "[1.5, \"high\", -2]"),
        ("sapling = 0.05", "sapling = 0.06"),
        ("sapling = 0.05", "sapling = \"0.05\""),
        ("07:32:00Z", "07:32:01Z"),
        ("[block.ext.zz_other]", "[block.ext.zz_another]"),
        ("burn_time = 300", "burn_time = 300\nnew = {}"),
    ] {
        assert!(table_form.contains(from), "{from}");
        let changed = load_blocks(&table_form.replacen(from, to, 1)).unwrap();
        let hash = changed.content_hash();
        assert!(
            !hashes.contains(&hash),
            "`{from}` -> `{to}` did not change the hash"
        );
        hashes.push(hash);
    }

    // Namespaces follow the key alphabet; each one is a table.
    let message = invalid_message(block_with(
        "faces = { all = \"t:a\" }\n[block.ext.\"My-Mod\"]\nx = 1",
    ));
    assert!(
        message.contains("`ext` namespace `My-Mod` must be made of"),
        "{message}"
    );
    match block_with("faces = { all = \"t:a\" }\next = { forestry = 5 }") {
        Err(ContentError::Parse { key, message, .. }) => {
            assert_eq!(key.as_deref(), Some("t:x"));
            assert!(message.contains("invalid type: integer `5`"), "{message}");
        }
        other => panic!("expected a parse error, got {other:?}"),
    }
}

#[test]
fn hash_follows_every_table_relevant_value() {
    let base = r#"
[[block]]
key = "t:x"
orient = "up6"
render = "cutout"
faces = { all = "t:a" }
hardness = 1
density = 1

[[block]]
key = "t:m"
render = "model"
lod_proxy = "t:x"
hardness = 1
density = 1
"#;
    let original = load_blocks(base).unwrap().content_hash();
    assert_eq!(load_blocks(base).unwrap().content_hash(), original);
    let mut hashes = vec![original];
    for (from, to) in [
        ("orient = \"up6\"", "orient = \"rot24\""),
        ("orient = \"up6\"", "orient = \"axis3\""),
        ("orient = \"up6\"", "orient = \"none\""),
        (
            "render = \"cutout\"",
            "render = \"cutout\"\ncull_group = \"leaves\"",
        ),
        (
            "render = \"cutout\"",
            "render = \"cutout\"\nstructural = true",
        ),
        ("render = \"cutout\"", "render = \"cutout\"\nsolid = false"),
        ("render = \"cutout\"", "render = \"cube\"\nopaque = false"),
        ("lod_proxy = \"t:x\"", "lod_proxy = \"t:x[up=-y]\""),
        ("lod_proxy = \"t:x\"\n", ""),
        ("render = \"model\"", "render = \"model\"\nsolid = false"),
        (
            "render = \"model\"",
            "render = \"model\"\norient = \"rot24\"",
        ),
        (
            "render = \"model\"",
            "render = \"model\"\nstructural = true",
        ),
    ] {
        assert!(base.contains(from), "{from}");
        let hash = load_blocks(&base.replacen(from, to, 1))
            .unwrap()
            .content_hash();
        assert!(
            !hashes.contains(&hash),
            "`{from}` -> `{to}` did not change the hash"
        );
        hashes.push(hash);
    }
    // Renaming a cull group changes the hash, although its number stays 1.
    let named = |group: &str| {
        load_blocks(&base.replacen(
            "render = \"cutout\"",
            &format!("render = \"cutout\"\ncull_group = \"{group}\""),
            1,
        ))
        .unwrap()
    };
    let (leaves, foliage) = (named("leaves"), named("foliage"));
    assert_eq!(leaves.tables(), foliage.tables());
    assert_ne!(leaves.content_hash(), foliage.content_hash());
}
