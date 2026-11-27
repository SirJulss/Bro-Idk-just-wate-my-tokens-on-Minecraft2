//! How a registry assigns its ids, checked through `Registry::load` on packs written to disk:
//! state 0 is `cubeplanet:air` whatever sorts before it, and the order in which a pack lists
//! its definitions changes no id, no table and no hash.

use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};

use cp_content::{NO_MATERIAL_COLOR, Registry};
use cp_core::block::{BlockFlags, BlockRender, LodClass, LodInfo, RenderClass};
use cp_core::face::{Face, Orientation};
use cp_core::hash::Rng;
use cp_core::ids::{BlockStateId, MaterialId};

/// A directory below cargo's scratch directory for integration tests, removed again on drop.
struct TempDir {
    path: PathBuf,
}

impl TempDir {
    /// `test` must be unique among the tests of this file; the process id keeps concurrent
    /// runs of the test binary apart.
    fn new(test: &str) -> Self {
        let path = Path::new(env!("CARGO_TARGET_TMPDIR"))
            .join(format!("cp-content-ids-{test}-{}", std::process::id()));
        if path.exists() {
            fs::remove_dir_all(&path).expect("a stale test directory can be removed");
        }
        fs::create_dir_all(&path).expect("the test directory can be created");
        Self { path }
    }

    /// Writes a pack directory `name`: its `pack.toml` and `files`, given as `(path inside the
    /// pack with '/' separators, text)`.
    fn pack(&self, name: &str, files: &[(String, String)]) -> PathBuf {
        let dir = self.path.join(name);
        let manifest = format!("format = 1\nname = \"{name}\"\nversion = \"1\"\n");
        write(&dir.join("pack.toml"), &manifest);
        for (relative, text) in files {
            write(&dir.join(relative), text);
        }
        dir
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        // Best effort: a leftover directory only wastes space below `target/tmp`.
        let _ = fs::remove_dir_all(&self.path);
    }
}

fn write(path: &Path, text: &str) {
    let parent = path.parent().expect("files lie inside the test directory");
    fs::create_dir_all(parent).expect("directories can be created");
    fs::write(path, text).expect("the file can be written");
}

fn load(dir: &Path) -> Registry {
    Registry::load(&[dir]).unwrap_or_else(|error| panic!("the pack must load: {error}"))
}

/// Every state key of `registry`, by id.
fn state_keys(registry: &Registry) -> Vec<&str> {
    (0..registry.state_count())
        .map(|index| registry.state_key(BlockStateId(index as u32)))
        .collect()
}

/// A cube block `key` with `orient`, drawn with the one material of the test packs.
fn early_block(key: &str, orient: &str) -> String {
    format!(
        "[[block]]\nkey = \"{key}\"\norient = \"{orient}\"\nfaces = {{ all = \"early:rock\" }}\n\
         hardness = 1.0\ndensity = 1.0\n\n"
    )
}

const EARLY_TEXTURE: &str = r#"
[[texture]]
key = "early:rock"
provenance = "own"
procedural = { recipe = "stone", seed = 3 }
"#;

const EARLY_MATERIAL: &str = r#"
[[material]]
key = "early:rock"
texture = "early:rock"
"#;

/// Air as the engine provides it.
const AIR: &str = r#"
[[block]]
key = "cubeplanet:air"
render = "none"
hardness = 0
density = 0
"#;

#[test]
fn air_is_state_zero_whatever_sorts_before_it() {
    // Byte order puts all of these before `cubeplanet:air`: other namespaces ('0' < '_' < 'a' <
    // 'c'), names in its own namespace that sort before `air`, and `cubeplanet:ai`, a prefix
    // of it. Two keys sort after air, one of them with air's key as its prefix.
    let blocks: String = [
        ("zzz:last", "none"),
        ("cubeplanet:air_vent", "none"),
        ("cubeplanet:ai", "none"),
        ("cubeplanet:aaa", "none"),
        ("cubeplanet:a", "axis3"),
        ("aaa:stone", "up6"),
        ("_:under", "none"),
        ("0:first", "none"),
    ]
    .iter()
    .map(|(key, orient)| early_block(key, orient))
    .collect();
    let expected = [
        "cubeplanet:air",
        "0:first",
        "_:under",
        "aaa:stone",
        "aaa:stone[up=+x]",
        "aaa:stone[up=+z]",
        "aaa:stone[up=-x]",
        "aaa:stone[up=-y]",
        "aaa:stone[up=-z]",
        "cubeplanet:a",
        "cubeplanet:a[axis=x]",
        "cubeplanet:a[axis=z]",
        "cubeplanet:aaa",
        "cubeplanet:ai",
        "cubeplanet:air_vent",
        "zzz:last",
    ];
    // Every key but air's sorts in byte order, and thirteen of them sort before air's.
    assert!(expected[1..].is_sorted());
    assert_eq!(
        expected[1..]
            .iter()
            .filter(|key| **key < "cubeplanet:air")
            .count(),
        13
    );

    let temp = TempDir::new("air-first");
    let pack = |name: &str, block_files: &[(&str, &str)]| {
        let mut files = vec![
            ("textures/t.toml".to_owned(), EARLY_TEXTURE.to_owned()),
            ("materials/m.toml".to_owned(), EARLY_MATERIAL.to_owned()),
        ];
        files.extend(
            block_files
                .iter()
                .map(|(path, text)| ((*path).to_owned(), (*text).to_owned())),
        );
        load(&temp.pack(name, &files))
    };
    // Air provided by the engine, or written out: in a file read before the others, or at the
    // end of the block file.
    let with_air_last = format!("{blocks}{AIR}");
    let registries = [
        pack("implicit", &[("blocks/b.toml", blocks.as_str())]),
        pack(
            "air_first",
            &[
                ("blocks/0_air.toml", AIR),
                ("blocks/b.toml", blocks.as_str()),
            ],
        ),
        pack("air_last", &[("blocks/b.toml", with_air_last.as_str())]),
    ];

    for registry in &registries {
        assert_eq!(state_keys(registry), expected);
        for (index, key) in expected.iter().enumerate() {
            assert_eq!(
                registry.state(key),
                Some(BlockStateId(index as u32)),
                "{key}"
            );
        }
        assert_eq!(registry.state("cubeplanet:air"), Some(BlockStateId::AIR));
        assert_eq!(registry.block_of(BlockStateId::AIR).key, "cubeplanet:air");
        assert_eq!(
            registry.blocks()[0].key,
            "0:first",
            "blocks are sorted by key"
        );
        let tables = registry.tables();
        assert_eq!(tables.flags[0], BlockFlags::empty());
        assert_eq!(tables.render[0], BlockRender::NONE);
        assert_eq!(tables.cull_group[0], 0);
        assert_eq!(tables.lod[0], LodInfo::EMPTY);
        assert_eq!(tables.material_color[0], NO_MATERIAL_COLOR);
        // Every other state is one of the solid cubes, before air in key order or not.
        for index in 1..registry.state_count() {
            let id = BlockStateId(index as u32);
            assert_eq!(
                tables.flags[index],
                BlockFlags::SOLID | BlockFlags::OPAQUE,
                "{}",
                registry.state_key(id)
            );
            assert_eq!(tables.render[index].class, RenderClass::Cube);
            assert_eq!(
                tables.lod[index],
                LodInfo {
                    class: LodClass::Solid,
                    proxy: id
                }
            );
        }
        assert_eq!(registry.content_hash(), registries[0].content_hash());
    }
}

const TEXTURES: [&str; 5] = [
    r#"
[[texture]]
key = "ord:granite"
provenance = "own"
procedural = { recipe = "stone", seed = 11 }
"#,
    r##"
[[texture]]
key = "ord:moss"
provenance = "own"
procedural = { recipe = "leaves", seed = 12, params = { leaf = "#3a6b2f", style = "broad" } }
"##,
    r##"
[[texture]]
key = "ord:bark"
provenance = "own"
procedural = { recipe = "log_side", seed = 13, params = { bark = "#5a3d24", style = "furrowed" } }
"##,
    r#"
[[texture]]
key = "ord:rings"
provenance = "own"
procedural = { recipe = "log_top", seed = 13 }
"#,
    r#"
[[texture]]
key = "ord:water"
provenance = "own"
procedural = { recipe = "water", seed = 15 }
"#,
];

/// Material keys chosen so that their sorted order differs from the order written here.
const MATERIALS: [&str; 6] = [
    r#"
[[material]]
key = "ord:z_granite"
texture = "ord:granite"
roughness = 0.9
"#,
    r#"
[[material]]
key = "ord:a_moss"
texture = "ord:moss"
tint = "foliage"
flags = ["cutout"]
"#,
    r#"
[[material]]
key = "ord:m_bark"
texture = "ord:bark"
"#,
    r#"
[[material]]
key = "ord:m_rings"
texture = "ord:rings"
metallic = 0.25
"#,
    r#"
[[material]]
key = "ord:b_water"
texture = "ord:water"
tint = "water"
flags = ["translucent"]
"#,
    r#"
[[material]]
key = "ord:c_glow"
texture = "ord:granite"
emissive = 2.0
"#,
];

/// Every orientation kind, cull groups whose names sort against the order of the blocks, a
/// model whose stand-in is a turned state of another block, an `[ext]` table, a key that sorts
/// before air, and air itself.
const BLOCKS: [&str; 9] = [
    r#"
[[block]]
key = "cubeplanet:air"
render = "none"
hardness = 0
density = 0
"#,
    r#"
[[block]]
key = "ord:rock"
faces = { all = "ord:z_granite" }
hardness = 5.0
density = 2600.0
tags = ["rock", "hard"]
"#,
    r#"
[[block]]
key = "ord:log"
orient = "axis3"
structural = true
hardness = 2.0
density = 700.0
[block.faces]
top = "ord:m_rings"
side = "ord:m_bark"
bottom = "ord:m_rings"
"#,
    r#"
[[block]]
key = "ord:leaves"
render = "cutout"
faces = { all = "ord:a_moss" }
cull_group = "zz_leaves"
hardness = 0.2
density = 100.0
"#,
    r#"
[[block]]
key = "ord:water"
render = "fluid"
faces = { all = "ord:b_water" }
cull_group = "aa_water"
hardness = 0.0
density = 1000.0
"#,
    r#"
[[block]]
key = "ord:glass"
opaque = false
faces = { all = "ord:z_granite" }
cull_group = "mm_glass"
hardness = 0.5
density = 2500.0
"#,
    r#"
[[block]]
key = "ord:machine"
orient = "rot24"
hardness = 3.0
density = 4000.0
light = 7
[block.faces]
"+x" = "ord:m_bark"
"-x" = "ord:m_rings"
"+y" = "ord:z_granite"
"-y" = "ord:c_glow"
"+z" = "ord:c_glow"
"-z" = "ord:m_bark"
"#,
    r#"
[[block]]
key = "ord:shaft"
render = "model"
orient = "up6"
lod_proxy = "ord:log[axis=x]"
hardness = 1.0
density = 900.0
"#,
    r#"
[[block]]
key = "aaa:first"
orient = "up6"
faces = { top = "ord:c_glow", side = "ord:z_granite", bottom = "ord:m_bark" }
hardness = 1.0
density = 1.0
[block.ext.lore]
note = "sorts before air"
weights = [3, 1, 2]
"#,
];

/// `entries` in a random order (Fisher-Yates).
fn shuffled<'a>(entries: &[&'a str], rng: &mut Rng) -> Vec<&'a str> {
    let mut order = entries.to_vec();
    for last in (1..order.len()).rev() {
        let bound = u32::try_from(last + 1).expect("few entries");
        order.swap(last, rng.range_u32(0, bound) as usize);
    }
    order
}

/// The definition files of `directory` that hold `entries` dealt out over `files` files, the
/// last of them in a sub-directory; as `(path inside the pack, text)`.
///
/// Files are read in the order of their paths, so the entries are inserted in yet another
/// order than the one given.
fn deal(directory: &str, entries: &[&str], files: usize) -> Vec<(String, String)> {
    let mut texts = vec![String::new(); files];
    for (index, entry) in entries.iter().enumerate() {
        texts[index % files].push_str(entry);
    }
    texts
        .into_iter()
        .enumerate()
        .map(|(file, text)| {
            let path = if file > 0 && file + 1 == files {
                format!("{directory}/nested/part{file}.toml")
            } else {
                format!("{directory}/part{file}.toml")
            };
            (path, text)
        })
        .collect()
}

/// The key of every `[[block]]`, `[[material]]` or `[[texture]]` entry, in the order written.
fn entry_keys<'a>(entries: &[&'a str]) -> Vec<&'a str> {
    entries
        .iter()
        .map(|entry| {
            let (_, rest) = entry.split_once("key = \"").expect("every entry has a key");
            rest.split_once('"').expect("the key is quoted").0
        })
        .collect()
}

/// Asserts that `other` has the same ids, tables and definitions as `reference` and the same
/// content hash.
fn assert_same_registry(reference: &Registry, other: &Registry, what: &str) {
    assert_eq!(
        other.content_hash(),
        reference.content_hash(),
        "{what}: content hash"
    );
    assert_eq!(
        state_keys(other),
        state_keys(reference),
        "{what}: state ids"
    );
    for index in 0..reference.state_count() {
        let id = BlockStateId(index as u32);
        let key = reference.state_key(id);
        assert_eq!(other.state(key), Some(id), "{what}: id of `{key}`");
        assert_eq!(
            other.orientation(id),
            reference.orientation(id),
            "{what}: `{key}`"
        );
        assert_eq!(
            other.block_of(id).key,
            reference.block_of(id).key,
            "{what}: `{key}`"
        );
        for turn in Orientation::ALL {
            assert_eq!(
                other.rotated(id, turn),
                reference.rotated(id, turn),
                "{what}: `{key}` turned by {turn:?}"
            );
        }
    }
    for material in reference.materials() {
        assert_eq!(
            other.material(&material.key),
            reference.material(&material.key),
            "{what}: id of material `{}`",
            material.key
        );
    }
    for texture in reference.textures() {
        assert_eq!(
            other.texture(&texture.key),
            reference.texture(&texture.key),
            "{what}: layer of texture `{}`",
            texture.key
        );
    }
    assert_eq!(other.tables(), reference.tables(), "{what}: tables");
    assert_eq!(other.cull_groups(), reference.cull_groups(), "{what}");
    assert_eq!(other.blocks(), reference.blocks(), "{what}: blocks");
    assert_eq!(
        other.materials(),
        reference.materials(),
        "{what}: materials"
    );
    assert_eq!(other.textures(), reference.textures(), "{what}: textures");
}

#[test]
fn insertion_order_inside_a_pack_changes_no_id_table_or_hash() {
    const SHUFFLES: usize = 24;
    let temp = TempDir::new("insertion-order");
    let write_pack = |name: &str, textures: &[&str], materials: &[&str], blocks: &[&str], files| {
        let mut all = deal("textures", textures, files);
        all.extend(deal("materials", materials, files));
        all.extend(deal("blocks", blocks, files));
        temp.pack(name, &all)
    };

    // The definitions in the order written above, one file per kind.
    let reference = load(&write_pack("written", &TEXTURES, &MATERIALS, &BLOCKS, 1));

    // The reference itself follows the rules, so that agreeing with it means something.
    let keys = state_keys(&reference);
    // Air, aaa:first (up6), glass, leaves, log (axis3), machine (rot24), rock, shaft (up6), water.
    assert_eq!(keys.len(), 1 + 6 + 1 + 1 + 3 + 24 + 1 + 6 + 1);
    assert_eq!(
        keys[..3],
        ["cubeplanet:air", "aaa:first", "aaa:first[up=+x]"]
    );
    assert!(keys[1..].is_sorted());
    let material = |key: &str| reference.material(key).expect("the material exists");
    let material_ids: Vec<u16> = [
        "ord:a_moss",
        "ord:b_water",
        "ord:c_glow",
        "ord:m_bark",
        "ord:m_rings",
        "ord:z_granite",
    ]
    .map(|key| material(key).0)
    .to_vec();
    assert_eq!(material_ids, [0, 1, 2, 3, 4, 5]);
    assert_eq!(
        reference.cull_groups(),
        ["aa_water", "mm_glass", "zz_leaves"]
    );
    let state = |key: &str| reference.state(key).expect("the state exists");
    let tables = reference.tables();
    assert_eq!(tables.cull_group[state("ord:water").index()], 1);
    assert_eq!(tables.cull_group[state("ord:glass").index()], 2);
    assert_eq!(tables.cull_group[state("ord:leaves").index()], 3);
    // The stand-in turns with the model: the unturned shaft stands in as the log along x that
    // the pack names, the shaft turned to +x as a log along y.
    assert_eq!(
        tables.lod[state("ord:shaft").index()].proxy,
        state("ord:log[axis=x]")
    );
    assert_eq!(
        tables.lod[state("ord:shaft[up=+x]").index()].proxy,
        state("ord:log")
    );
    assert_eq!(
        tables.render[state("ord:rock").index()]
            .face(Face::PosY)
            .material,
        MaterialId(5)
    );

    // Every definition in another order, dealt out over one to three files.
    let mut rng = Rng::new(0x0D_E5_07, 1);
    let mut orders = BTreeSet::new();
    let mut candidates = vec![(
        "reversed".to_owned(),
        TEXTURES.iter().rev().copied().collect::<Vec<_>>(),
        MATERIALS.iter().rev().copied().collect::<Vec<_>>(),
        BLOCKS.iter().rev().copied().collect::<Vec<_>>(),
        3,
    )];
    for round in 0..SHUFFLES {
        let files = 1 + rng.range_u32(0, 3) as usize;
        candidates.push((
            format!("shuffled{round}"),
            shuffled(&TEXTURES, &mut rng),
            shuffled(&MATERIALS, &mut rng),
            shuffled(&BLOCKS, &mut rng),
            files,
        ));
    }
    for (name, textures, materials, blocks, files) in candidates {
        orders.insert((entry_keys(&materials), entry_keys(&blocks)));
        let registry = load(&write_pack(&name, &textures, &materials, &blocks, files));
        assert_same_registry(&reference, &registry, &name);
    }
    // The shuffles did try many different orders.
    assert!(orders.len() > SHUFFLES / 2, "{} orders", orders.len());
}
