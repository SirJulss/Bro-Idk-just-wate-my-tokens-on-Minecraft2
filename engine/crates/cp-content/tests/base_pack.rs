//! The real base pack, `engine/content/base`, loaded through the public API.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use cp_content::{
    BlockDef, MaterialDef, MaterialFlags, Orient, ParamValue, Params, Provenance, Registry,
    StateKey, TextureDef, TextureSource, TintClass,
};
use cp_core::block::{BlockFlags, BlockRender, LodClass, LodInfo, RenderClass};
use cp_core::det;
use cp_core::face::{Face, Orientation};
use cp_core::ids::{BlockStateId, MaterialId};

const SPECIES: [&str; 4] = ["oak", "birch", "spruce", "cherry"];
const ORES: [&str; 4] = ["copper", "tin", "iron", "coal"];

/// What a recipe parameter takes.
#[derive(Clone, Copy, Debug)]
enum Kind {
    /// A colour.
    Color,
    /// One of these variant names.
    Variant(&'static [&'static str]),
}

const COLOR: &[(&str, Kind)] = &[("color", Kind::Color)];

/// The recipes that contract §4.3 promises, each with the parameters `cp-proctex` documents
/// for it (`cp_proctex::recipe_info`, transcribed on 2026-11-26).
///
/// `cp-content` must not depend on `cp-proctex`, so this table is a copy. It catches a misspelt
/// recipe, parameter or variant in the base pack here, long before a texture is generated;
/// the real check against the generator is `cpe content-check`.
const RECIPES: [(&str, &[(&str, Kind)]); 22] = [
    ("stone", COLOR),
    ("cobble", COLOR),
    ("gravel", COLOR),
    ("dirt", COLOR),
    ("grass_top", &[("grass", Kind::Color)]),
    (
        "grass_side",
        &[("grass", Kind::Color), ("dirt", Kind::Color)],
    ),
    ("sand", COLOR),
    ("sandstone", COLOR),
    ("snow", COLOR),
    ("ice", COLOR),
    ("clay", COLOR),
    ("basalt", COLOR),
    ("limestone", COLOR),
    ("slate", COLOR),
    (
        "log_side",
        &[
            ("bark", Kind::Color),
            ("marks", Kind::Color),
            (
                "style",
                Kind::Variant(&["furrowed", "birch", "scaly", "smooth"]),
            ),
        ],
    ),
    ("log_top", &[("bark", Kind::Color), ("wood", Kind::Color)]),
    ("planks", &[("wood", Kind::Color)]),
    (
        "leaves",
        &[
            ("leaf", Kind::Color),
            ("blossom", Kind::Color),
            ("style", Kind::Variant(&["broad", "needle"])),
        ],
    ),
    ("water", COLOR),
    ("lava", COLOR),
    (
        "ore",
        &[
            ("ore", Kind::Color),
            ("accent", Kind::Color),
            ("rock", Kind::Color),
            (
                "style",
                Kind::Variant(&["nuggets", "lumps", "flecks", "crystals"]),
            ),
        ],
    ),
    ("metal", COLOR),
];

/// The foliage green that `cp-proctex` previews tinted textures with and suggests as a stand-in
/// until biomes provide their own (`Tint::Foliage.reference_color()`).
const REFERENCE_FOLIAGE: [u8; 3] = [92, 160, 66];

fn base_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../content/base")
}

fn base() -> Registry {
    Registry::load(&[&base_dir()])
        .unwrap_or_else(|error| panic!("the base pack must load: {error}"))
}

fn block<'a>(registry: &'a Registry, name: &str) -> &'a BlockDef {
    let key = format!("cubeplanet:{name}");
    registry
        .block(&key)
        .unwrap_or_else(|| panic!("block `{key}` is missing"))
}

fn material_id(registry: &Registry, name: &str) -> MaterialId {
    let key = format!("cubeplanet:{name}");
    registry
        .material(&key)
        .unwrap_or_else(|| panic!("material `{key}` is missing"))
}

fn material<'a>(registry: &'a Registry, name: &str) -> &'a MaterialDef {
    &registry.materials()[material_id(registry, name).index()]
}

fn texture<'a>(registry: &'a Registry, name: &str) -> &'a TextureDef {
    let key = format!("cubeplanet:{name}");
    let layer = registry
        .texture(&key)
        .unwrap_or_else(|| panic!("texture `{key}` is missing"));
    &registry.textures()[layer.index()]
}

/// Recipe, parameters and seed of the procedural texture `name`.
fn procedural<'a>(registry: &'a Registry, name: &str) -> (&'a str, &'a Params, u64) {
    match &texture(registry, name).source {
        TextureSource::Procedural {
            recipe,
            params,
            seed,
        } => (recipe, params, *seed),
        other => panic!("texture `{name}` is not procedural: {other:?}"),
    }
}

fn recipe<'a>(registry: &'a Registry, name: &str) -> &'a str {
    procedural(registry, name).0
}

fn color(registry: &Registry, name: &str, param: &str) -> [u8; 3] {
    procedural(registry, name)
        .1
        .color(param)
        .unwrap_or_else(|| panic!("texture `{name}` has no colour parameter `{param}`"))
}

fn variant<'a>(registry: &'a Registry, name: &str, param: &str) -> &'a str {
    procedural(registry, name)
        .1
        .variant(param)
        .unwrap_or_else(|| panic!("texture `{name}` has no variant parameter `{param}`"))
}

fn state(registry: &Registry, key: &str) -> BlockStateId {
    registry
        .state(key)
        .unwrap_or_else(|| panic!("state `{key}` is missing"))
}

fn render(registry: &Registry, key: &str) -> BlockRender {
    registry.tables().render[state(registry, key).index()]
}

fn flags(registry: &Registry, key: &str) -> BlockFlags {
    registry.tables().flags[state(registry, key).index()]
}

fn cull_group(registry: &Registry, key: &str) -> u16 {
    registry.tables().cull_group[state(registry, key).index()]
}

fn lod(registry: &Registry, key: &str) -> LodInfo {
    registry.tables().lod[state(registry, key).index()]
}

#[test]
fn loads_with_the_block_list_of_the_contract() {
    let registry = base();
    let mut expected: Vec<String> = [
        "air",
        "stone",
        "dirt",
        "grass",
        "sand",
        "sandstone",
        "gravel",
        "snow",
        "ice",
        "clay",
        "basalt",
        "limestone",
        "slate",
        "core",
        "water",
        "lava",
    ]
    .iter()
    .map(|name| format!("cubeplanet:{name}"))
    .collect();
    for species in SPECIES {
        for part in ["log", "leaves", "planks"] {
            expected.push(format!("cubeplanet:{part}_{species}"));
        }
    }
    for ore in ORES {
        expected.push(format!("cubeplanet:ore_{ore}"));
    }
    expected.sort();
    let actual: Vec<&str> = registry.blocks().iter().map(|b| b.key.as_str()).collect();
    assert_eq!(actual, expected);
    assert_eq!(registry.blocks().len(), 32);

    // 32 blocks; grass has six states and each of the four logs three.
    assert_eq!(registry.state_count(), 32 + 5 + 4 * 2);
    let tables = registry.tables();
    assert_eq!(tables.flags.len(), registry.state_count());
    assert_eq!(tables.render.len(), registry.state_count());
    assert_eq!(tables.cull_group.len(), registry.state_count());
    assert_eq!(tables.lod.len(), registry.state_count());
    assert_eq!(tables.material_color.len(), registry.state_count());

    assert_eq!(registry.packs().len(), 1);
    assert_eq!(registry.packs()[0].name, "base");
    assert!(!registry.packs()[0].version.is_empty());
}

#[test]
fn air_is_state_zero() {
    let registry = base();
    assert_eq!(registry.state("cubeplanet:air"), Some(BlockStateId::AIR));
    assert_eq!(registry.state_key(BlockStateId::AIR), "cubeplanet:air");
    let tables = registry.tables();
    assert_eq!(tables.flags[0], BlockFlags::empty());
    assert_eq!(tables.render[0], BlockRender::NONE);
    assert_eq!(tables.cull_group[0], 0);
    assert_eq!(tables.lod[0], LodInfo::EMPTY);
    let air = registry.block_of(BlockStateId::AIR);
    assert_eq!(air.render, RenderClass::None);
    assert!(!air.solid && !air.opaque && air.faces.is_none());
}

#[test]
fn every_state_key_round_trips() {
    let registry = base();
    let mut seen = BTreeSet::new();
    let mut previous: Option<&str> = None;
    for index in 0..registry.state_count() {
        let id = BlockStateId(index as u32);
        let key = registry.state_key(id);
        assert_eq!(registry.state(key), Some(id), "{key}");
        assert!(seen.insert(key), "state key `{key}` occurs twice");

        // Ids follow the byte order of the keys, except that air comes first.
        if index >= 2 {
            assert!(previous < Some(key), "{previous:?} must sort before {key}");
        }
        previous = Some(key);

        // The key is the block key plus, sorted by name, the properties of the state's
        // orientation whose values are not the defaults (contract §4.4).
        let block = registry.block_of(id);
        let orientation = registry.orientation(id);
        let up = orientation.up();
        let axis = ["x", "y", "z"][up.axis()];
        let sign = if up.is_negative() { '-' } else { '+' };
        let mut properties = Vec::new();
        match block.orient {
            Orient::None => {}
            Orient::Up6 => {
                if up != Face::PosY {
                    properties.push(format!("up={sign}{axis}"));
                }
            }
            Orient::Axis3 => {
                if up.axis() != 1 {
                    properties.push(format!("axis={axis}"));
                }
            }
            Orient::Rot24 => {
                if orientation.turn() != 0 {
                    properties.push(format!("turn={}", orientation.turn()));
                }
                if up != Face::PosY {
                    properties.push(format!("up={sign}{axis}"));
                }
            }
        }
        let expected = if properties.is_empty() {
            block.key.clone()
        } else {
            format!("{}[{}]", block.key, properties.join(","))
        };
        assert_eq!(key, expected);
        assert!(block.states().contains(&id));
        assert_eq!(block.state_for(orientation), id);

        // The grammar reads the key back as it is.
        let parsed = StateKey::parse(key).unwrap_or_else(|error| panic!("{error}"));
        assert_eq!(parsed.to_string(), key);
        assert_eq!(parsed.block(), block.key);
        assert_eq!(parsed.properties().len(), properties.len());
    }
    assert_eq!(registry.state_key(BlockStateId(1)), "cubeplanet:basalt");

    // Every block is reachable by its bare key, and that is its default state.
    for block in registry.blocks() {
        assert_eq!(registry.state(&block.key), Some(block.default_state()));
        assert_eq!(
            registry.orientation(block.default_state()),
            Orientation::IDENTITY
        );
        assert_eq!(block.states().len(), block.orient.state_count());
    }
    assert_eq!(registry.state("cubeplanet:unobtainium"), None);
    assert_eq!(registry.state("cubeplanet:stone[up=+x]"), None);
}

#[test]
fn grass_turns_with_its_up_face() {
    let registry = base();
    let grass = block(&registry, "grass");
    assert_eq!(grass.orient, Orient::Up6);
    let top = material_id(&registry, "grass_top");
    let side = material_id(&registry, "grass_side");
    let bottom = material_id(&registry, "dirt");
    assert_eq!(grass.faces, Some([side, side, top, bottom, side, side]));

    let keys = [
        (Face::PosX, "cubeplanet:grass[up=+x]"),
        (Face::NegX, "cubeplanet:grass[up=-x]"),
        (Face::PosY, "cubeplanet:grass"),
        (Face::NegY, "cubeplanet:grass[up=-y]"),
        (Face::PosZ, "cubeplanet:grass[up=+z]"),
        (Face::NegZ, "cubeplanet:grass[up=-z]"),
    ];
    for (up, key) in keys {
        let id = state(&registry, key);
        assert_eq!(grass.state_for_up(up), id);
        assert_eq!(registry.orientation(id), Orientation::from_up(up));
        let render = registry.tables().render[id.index()];
        assert_eq!(render.class, RenderClass::Cube);
        for world in Face::ALL {
            let appearance = render.face(world);
            let expected = if world == up {
                top
            } else if world == up.opposite() {
                bottom
            } else {
                side
            };
            assert_eq!(appearance.material, expected, "{key}, face {world:?}");
            assert_eq!(appearance.uv_rot, Orientation::from_up(up).uv_rot(world));
            // On the four sides the image's up direction is the block's up direction, so the
            // fringe of grass hangs down from the top edge whichever way the block lies.
            if world.axis() != up.axis() {
                let (_, image_up) = world.quad_tex_axes(appearance.uv_rot);
                assert_eq!(image_up, up.normal(), "{key}, face {world:?}");
            }
        }
        assert_eq!(
            registry.tables().flags[id.index()],
            BlockFlags::SOLID | BlockFlags::OPAQUE
        );
        // Grass placed on any face of a planet is the default state turned onto that face.
        let placed = registry.rotated(grass.default_state(), Orientation::from_up(up));
        assert_eq!(placed, id);
    }
    // The default value is never written: the bare key is the only key of the default state.
    assert_eq!(registry.state("cubeplanet:grass[up=+y]"), None);
}

#[test]
fn logs_show_their_end_grain_on_both_axis_faces() {
    let registry = base();
    for species in SPECIES {
        let log = block(&registry, &format!("log_{species}"));
        assert_eq!(log.orient, Orient::Axis3);
        let end_grain = material_id(&registry, &format!("log_{species}_top"));
        let bark = material_id(&registry, &format!("log_{species}_side"));
        assert_ne!(end_grain, bark);

        let variants = [
            (0, format!("cubeplanet:log_{species}[axis=x]")),
            (1, format!("cubeplanet:log_{species}")),
            (2, format!("cubeplanet:log_{species}[axis=z]")),
        ];
        for (axis, key) in variants {
            let id = state(&registry, &key);
            assert_eq!(log.states()[axis], id);
            let orientation = registry.orientation(id);
            assert_eq!(orientation.up(), Face::from_axis(axis, false));
            let render = registry.tables().render[id.index()];
            for world in Face::ALL {
                let appearance = render.face(world);
                if world.axis() == axis {
                    assert_eq!(appearance.material, end_grain, "{key}, face {world:?}");
                } else {
                    assert_eq!(appearance.material, bark, "{key}, face {world:?}");
                    // The bark's grain runs along the log.
                    let (_, image_up) = world.quad_tex_axes(appearance.uv_rot);
                    assert_eq!(image_up, Face::from_axis(axis, false).normal());
                }
                assert_eq!(appearance.uv_rot, orientation.uv_rot(world));
            }
            // Both ends of the axis give the same state.
            assert_eq!(log.state_for_up(Face::from_axis(axis, false)), id);
            assert_eq!(log.state_for_up(Face::from_axis(axis, true)), id);
            // Logs are structural: a trunk survives the derivation of coarser levels.
            assert_eq!(
                registry.tables().flags[id.index()],
                BlockFlags::SOLID | BlockFlags::OPAQUE | BlockFlags::STRUCTURAL
            );
            // Turning a log keeps it on the axis its up face is turned to.
            for o in Orientation::ALL {
                let turned = registry.rotated(id, o);
                let along = o.world_face(Face::from_axis(axis, false)).axis();
                assert_eq!(turned, log.states()[along]);
            }
        }
        assert_eq!(
            registry.state(&format!("cubeplanet:log_{species}[axis=y]")),
            None
        );
    }
}

#[test]
fn unrotated_blocks_use_the_identity_orientation() {
    let registry = base();
    let stone = render(&registry, "cubeplanet:stone");
    let material = material_id(&registry, "stone");
    for face in Face::ALL {
        assert_eq!(stone.face(face).material, material);
        assert_eq!(stone.face(face).uv_rot, Orientation::IDENTITY.uv_rot(face));
    }
}

#[test]
fn flags_follow_the_kind_of_block() {
    let registry = base();
    let solid_cube = BlockFlags::SOLID | BlockFlags::OPAQUE;
    for name in [
        "stone",
        "dirt",
        "grass",
        "sand",
        "sandstone",
        "gravel",
        "snow",
        "ice",
        "clay",
        "basalt",
        "limestone",
        "slate",
        "core",
    ] {
        let key = format!("cubeplanet:{name}");
        assert_eq!(flags(&registry, &key), solid_cube, "{key}");
        assert_eq!(render(&registry, &key).class, RenderClass::Cube, "{key}");
    }
    for species in SPECIES {
        let key = format!("cubeplanet:planks_{species}");
        assert_eq!(flags(&registry, &key), solid_cube, "{key}");
        assert_eq!(render(&registry, &key).class, RenderClass::Cube, "{key}");
        let key = format!("cubeplanet:log_{species}");
        let structural = solid_cube | BlockFlags::STRUCTURAL;
        assert_eq!(flags(&registry, &key), structural, "{key}");
        assert_eq!(render(&registry, &key).class, RenderClass::Cube, "{key}");
        // Leaves are cut out: solid to walk on, but they hide nothing behind them.
        let key = format!("cubeplanet:leaves_{species}");
        assert_eq!(
            flags(&registry, &key),
            BlockFlags::SOLID | BlockFlags::CUTOUT,
            "{key}"
        );
        assert_eq!(render(&registry, &key).class, RenderClass::Cutout, "{key}");
        let leaves = block(&registry, &format!("leaves_{species}"));
        assert!(leaves.solid && !leaves.opaque);
        assert!(
            material(&registry, &format!("leaves_{species}"))
                .flags
                .contains(MaterialFlags::CUTOUT)
        );
    }
    for ore in ORES {
        let key = format!("cubeplanet:ore_{ore}");
        assert_eq!(flags(&registry, &key), solid_cube, "{key}");
        assert!(block(&registry, &format!("ore_{ore}")).has_tag("ore"));
    }

    // Fluids: neither solid nor opaque.
    let fluid = BlockFlags::FLUID;
    assert_eq!(flags(&registry, "cubeplanet:water"), fluid);
    assert_eq!(
        flags(&registry, "cubeplanet:lava"),
        fluid | BlockFlags::EMISSIVE
    );
    assert_eq!(
        render(&registry, "cubeplanet:water").class,
        RenderClass::Fluid
    );
    assert_eq!(
        render(&registry, "cubeplanet:lava").class,
        RenderClass::Fluid
    );
    assert_eq!(block(&registry, "lava").light, 15);
    assert!(material(&registry, "lava").emissive > 0.0);
    assert!(
        material(&registry, "water")
            .flags
            .contains(MaterialFlags::TRANSLUCENT)
    );
    assert!(material(&registry, "lava").flags.is_empty());

    // Only lava glows.
    for block in registry.blocks() {
        assert_eq!(
            block.light > 0,
            block.key == "cubeplanet:lava",
            "{}",
            block.key
        );
    }
    // Only the core cannot be mined.
    for block in registry.blocks() {
        assert_eq!(
            block.is_unbreakable(),
            block.key == "cubeplanet:core",
            "{}",
            block.key
        );
    }
}

#[test]
fn physical_values_are_plausible() {
    let registry = base();
    for block in registry.blocks() {
        assert!(block.hardness >= 0.0, "{}", block.key);
        assert!(
            block.density.is_finite() && block.density >= 0.0,
            "{}",
            block.key
        );
        assert!(block.tags.is_sorted(), "{}", block.key);
        if block.solid {
            assert!(block.density > 0.0, "{} is solid but weightless", block.key);
        }
    }
    let density = |name: &str| block(&registry, name).density;
    assert_eq!(density("water"), 1000.0);
    assert!(density("ice") < density("water"), "ice floats");
    for species in SPECIES {
        assert!(
            density(&format!("log_{species}")) < density("water"),
            "wood floats"
        );
        assert_eq!(
            density(&format!("log_{species}")),
            density(&format!("planks_{species}"))
        );
    }
    assert!(density("lava") > density("stone"));
    assert!(density("core") > density("basalt"));
    let hardness = |name: &str| block(&registry, name).hardness;
    assert!(hardness("dirt") < hardness("stone") && hardness("stone") < hardness("basalt"));
    assert!(hardness("leaves_oak") < hardness("planks_oak"));
}

#[test]
fn all_art_is_own_and_procedural() {
    let registry = base();
    assert_eq!(registry.textures().len(), 36);
    for texture in registry.textures() {
        assert_eq!(texture.provenance, Provenance::Own, "{}", texture.key);
        assert!(
            matches!(texture.source, TextureSource::Procedural { .. }),
            "`{}` is a file texture",
            texture.key
        );
    }
}

#[test]
fn textures_use_documented_recipes_and_parameters() {
    let registry = base();
    for texture in registry.textures() {
        let TextureSource::Procedural { recipe, params, .. } = &texture.source else {
            panic!("`{}` is a file texture", texture.key);
        };
        let Some((_, known)) = RECIPES.iter().find(|(name, _)| name == recipe) else {
            panic!(
                "texture `{}` uses the recipe `{recipe}`, which the contract does not list",
                texture.key
            );
        };
        for (name, value) in params.iter() {
            let context = format!("texture `{}`, recipe `{recipe}`", texture.key);
            let Some((_, kind)) = known.iter().find(|(key, _)| *key == name) else {
                panic!("{context}: the recipe has no parameter `{name}`");
            };
            match (kind, value) {
                (Kind::Color, ParamValue::Color(_)) => {}
                (Kind::Variant(names), ParamValue::Variant(variant)) => assert!(
                    names.contains(&variant.as_str()),
                    "{context}: `{name}` has no variant `{variant}`, only {names:?}"
                ),
                _ => panic!("{context}: `{name}` = {value:?} is not a {kind:?}"),
            }
        }
    }
}

#[test]
fn every_block_has_the_texture_of_its_kind() {
    let registry = base();
    for (name, expected) in [
        ("stone", "stone"),
        ("dirt", "dirt"),
        ("grass_top", "grass_top"),
        ("grass_side", "grass_side"),
        ("sand", "sand"),
        ("sandstone", "sandstone"),
        ("gravel", "gravel"),
        ("snow", "snow"),
        ("ice", "ice"),
        ("clay", "clay"),
        ("basalt", "basalt"),
        ("limestone", "limestone"),
        ("slate", "slate"),
        ("core", "basalt"),
        ("water", "water"),
        ("lava", "lava"),
    ] {
        assert_eq!(recipe(&registry, name), expected, "{name}");
        // A block, a material and a texture of the same name belong together.
        let block = block(
            &registry,
            if name.starts_with("grass") {
                "grass"
            } else {
                name
            },
        );
        let material = material_id(&registry, name);
        assert!(block.faces.is_some_and(|faces| faces.contains(&material)));
        let layer = registry.materials()[material.index()].texture;
        assert_eq!(
            registry.textures()[layer.index()].key,
            format!("cubeplanet:{name}")
        );
    }
    for species in SPECIES {
        assert_eq!(
            recipe(&registry, &format!("log_{species}_side")),
            "log_side"
        );
        assert_eq!(recipe(&registry, &format!("log_{species}_top")), "log_top");
        assert_eq!(recipe(&registry, &format!("planks_{species}")), "planks");
        assert_eq!(recipe(&registry, &format!("leaves_{species}")), "leaves");
    }
    for ore in ORES {
        assert_eq!(recipe(&registry, &format!("ore_{ore}")), "ore");
    }

    // The core is not plain basalt: it has a colour of its own.
    assert!(procedural(&registry, "basalt").1.is_empty());
    let core = color(&registry, "core", "color");
    // cp-proctex paints basalt in the blue-grey [78, 78, 90] unless told otherwise.
    assert!(difference(core, [78, 78, 90]) >= CLEARLY_DISTINCT);
    assert!(block(&registry, "core").is_unbreakable());
}

#[test]
fn textures_that_continue_each_other_share_a_seed() {
    let registry = base();
    let seed = |name: &str| procedural(&registry, name).2;
    // The side of a grass block shows the dirt of the block below it.
    assert_eq!(seed("grass_side"), seed("dirt"));
    // Ore is painted into the stone of its own seed.
    for ore in ORES {
        assert_eq!(seed(&format!("ore_{ore}")), seed("stone"), "{ore}");
    }
}

/// An sRGB channel as linear light, `0.0..=1.0`.
fn to_linear(channel: u8) -> f64 {
    let value = f64::from(channel) / 255.0;
    if value <= 0.04045 {
        value / 12.92
    } else {
        det::powf((value + 0.055) / 1.055, 2.4)
    }
}

/// Linear light as an sRGB channel, rounded.
fn to_srgb(linear: f64) -> u8 {
    let value = if linear <= 0.003_130_8 {
        12.92 * linear
    } else {
        1.055 * det::powf(linear, 1.0 / 2.4) - 0.055
    };
    (det::clamp(value, 0.0, 1.0) * 255.0).round() as u8
}

/// What a texture colour becomes under a tint: the product of the two in linear light.
fn tinted(color: [u8; 3], tint: [u8; 3]) -> [u8; 3] {
    [0, 1, 2].map(|channel| to_srgb(to_linear(color[channel]) * to_linear(tint[channel])))
}

/// The saturation `(max - min) / max` of a colour in linear light. `cp-proctex` lets a texel
/// take the whole tint when this is at most 0.45, and none of it from 0.6 on.
fn linear_saturation(color: [u8; 3]) -> f64 {
    let [red, green, blue] = color.map(to_linear);
    let max = det::max(det::max(red, green), blue);
    let min = det::min(det::min(red, green), blue);
    if max > 0.0 { (max - min) / max } else { 0.0 }
}

/// An sRGB colour in CIELAB (D65), the space in which equal distances look about equally
/// different: `[L*, a*, b*]`.
fn lab(color: [u8; 3]) -> [f64; 3] {
    let [red, green, blue] = color.map(to_linear);
    let pivot = |t: f64| {
        if t > 0.008_856 {
            det::cbrt(t)
        } else {
            7.787 * t + 16.0 / 116.0
        }
    };
    let x = pivot((0.4124 * red + 0.3576 * green + 0.1805 * blue) / 0.950_47);
    let y = pivot(0.2126 * red + 0.7152 * green + 0.0722 * blue);
    let z = pivot((0.0193 * red + 0.1192 * green + 0.9505 * blue) / 1.088_83);
    [116.0 * y - 16.0, 500.0 * (x - y), 200.0 * (y - z)]
}

/// The CIE76 colour difference ΔE of two sRGB colours. About 2.3 is just noticeable; black and
/// white are 100 apart.
fn difference(a: [u8; 3], b: [u8; 3]) -> f64 {
    let (a, b) = (lab(a), lab(b));
    (0..3)
        .map(|i| (a[i] - b[i]) * (a[i] - b[i]))
        .sum::<f64>()
        .sqrt()
}

/// Colours that players must tell apart at a glance differ by at least this ΔE.
const CLEARLY_DISTINCT: f64 = 25.0;

/// Asserts that all colours differ clearly from each other.
fn assert_distinct(what: &str, colors: &[(&str, [u8; 3])]) {
    for (index, (first_name, first)) in colors.iter().enumerate() {
        for (second_name, second) in &colors[index + 1..] {
            let apart = difference(*first, *second);
            assert!(
                apart >= CLEARLY_DISTINCT,
                "{what}: {first_name} {first:?} and {second_name} {second:?} are only \
                 {apart:.1} apart"
            );
        }
    }
}

#[test]
fn colour_difference_matches_known_values() {
    assert_eq!(difference([18, 52, 86], [18, 52, 86]), 0.0);
    let black_white = difference([0, 0, 0], [255, 255, 255]);
    assert!((black_white - 100.0).abs() < 0.01, "{black_white}");
    // Pure red is L* 53.2, a* 80.1, b* 67.2.
    let red = lab([255, 0, 0]);
    for (channel, expected) in red.into_iter().zip([53.24, 80.09, 67.20]) {
        assert!((channel - expected).abs() < 0.05, "{red:?}");
    }
    // Neighbouring greys are far below the threshold, complementary colours far above.
    assert!(difference([128, 128, 128], [133, 133, 133]) < 3.0);
    assert!(difference([255, 0, 0], [0, 255, 0]) > 150.0);

    // Tinting multiplies in linear light: white takes the tint, a white tint changes nothing.
    assert_eq!(tinted([255, 255, 255], [92, 160, 66]), [92, 160, 66]);
    assert_eq!(tinted([90, 140, 200], [255, 255, 255]), [90, 140, 200]);
    assert_eq!(tinted([0, 0, 0], [92, 160, 66]), [0, 0, 0]);
    for channel in [0, 1, 12, 127, 128, 200, 255] {
        assert_eq!(to_srgb(to_linear(channel)), channel);
    }
    assert_eq!(linear_saturation([128, 128, 128]), 0.0);
    assert_eq!(linear_saturation([255, 0, 0]), 1.0);
}

/// The colour of a species' leaves as a player sees it: the colour of the texture, under the
/// reference foliage green if the material is tinted.
fn leaf_colour(registry: &Registry, species: &str) -> [u8; 3] {
    let name = format!("leaves_{species}");
    let painted = color(registry, &name, "leaf");
    match material(registry, &name).tint {
        TintClass::None => painted,
        TintClass::Foliage => {
            assert!(
                linear_saturation(painted) <= 0.45,
                "`{name}` is tinted, so its texture must be near-neutral, not {painted:?}"
            );
            tinted(painted, REFERENCE_FOLIAGE)
        }
        other => panic!("leaves are not tinted as {other:?}"),
    }
}

#[test]
fn tree_species_have_distinct_colours() {
    let registry = base();
    let collect = |texture: &str, param: &str| -> Vec<(&str, [u8; 3])> {
        SPECIES
            .iter()
            .map(|species| {
                let name = texture.replace("{}", species);
                (*species, color(&registry, &name, param))
            })
            .collect()
    };
    assert_distinct("bark", &collect("log_{}_side", "bark"));
    assert_distinct("planks", &collect("planks_{}", "wood"));
    let leaves: Vec<(&str, [u8; 3])> = SPECIES
        .iter()
        .map(|species| (*species, leaf_colour(&registry, species)))
        .collect();
    assert_distinct("leaves", &leaves);

    // Oak leaves follow the foliage colour of their surroundings: under the reference green
    // they are the fresh green the pack documents. The other species keep their own colour.
    assert_eq!(material(&registry, "leaves_oak").tint, TintClass::Foliage);
    assert!(difference(leaf_colour(&registry, "oak"), [0x44, 0x80, 0x2a]) < 2.0);
    for species in ["birch", "spruce", "cherry"] {
        let name = format!("leaves_{species}");
        assert_eq!(material(&registry, &name).tint, TintClass::None, "{name}");
    }

    // Every species has its own bark pattern and leaf shape.
    let styles: Vec<&str> = SPECIES
        .iter()
        .map(|species| variant(&registry, &format!("log_{species}_side"), "style"))
        .collect();
    assert_eq!(styles, ["furrowed", "birch", "scaly", "smooth"]);
    let shapes: Vec<&str> = SPECIES
        .iter()
        .map(|species| variant(&registry, &format!("leaves_{species}"), "style"))
        .collect();
    assert_eq!(shapes, ["broad", "broad", "needle", "broad"]);
    // The dashes on birch and cherry bark stand out from the bark.
    for species in ["birch", "cherry"] {
        let name = format!("log_{species}_side");
        let (bark, marks) = (
            color(&registry, &name, "bark"),
            color(&registry, &name, "marks"),
        );
        assert!(difference(bark, marks) >= CLEARLY_DISTINCT, "{species}");
    }

    for species in SPECIES {
        // The end grain shows the same wood as the planks, inside the same bark as the side.
        assert_eq!(
            color(&registry, &format!("log_{species}_top"), "wood"),
            color(&registry, &format!("planks_{species}"), "wood")
        );
        assert_eq!(
            color(&registry, &format!("log_{species}_top"), "bark"),
            color(&registry, &format!("log_{species}_side"), "bark")
        );
        // Bark and cut wood of one species differ too, if less: they also differ in pattern.
        let bark = color(&registry, &format!("log_{species}_side"), "bark");
        let wood = color(&registry, &format!("planks_{species}"), "wood");
        assert!(difference(bark, wood) >= 10.0, "{species}");
    }
}

#[test]
fn ores_have_distinct_colours() {
    let registry = base();
    let colors: Vec<(&str, [u8; 3])> = ORES
        .iter()
        .map(|ore| (*ore, color(&registry, &format!("ore_{ore}"), "ore")))
        .collect();
    assert_distinct("ores", &colors);

    // Ore must also stand out from the rock it sits in; cp-proctex paints stone in the warm
    // grey [139, 134, 130] unless told otherwise.
    for (ore, colour) in &colors {
        assert!(
            difference(*colour, [139, 134, 130]) >= CLEARLY_DISTINCT,
            "{ore}"
        );
    }
    // Every ore has pieces of a shape of its own as well (the cp-proctex presets).
    let style = |ore: &str| variant(&registry, &format!("ore_{ore}"), "style");
    assert_eq!(ORES.map(style), ["nuggets", "crystals", "flecks", "lumps"]);
}

#[test]
fn tints_mark_what_follows_the_surroundings() {
    let registry = base();
    let tinted_materials: Vec<(&str, TintClass)> = registry
        .materials()
        .iter()
        .filter(|material| material.tint != TintClass::None)
        .map(|material| (material.key.as_str(), material.tint))
        .collect();
    assert_eq!(
        tinted_materials,
        [
            ("cubeplanet:grass_side", TintClass::Grass),
            ("cubeplanet:grass_top", TintClass::Grass),
            ("cubeplanet:leaves_oak", TintClass::Foliage),
            ("cubeplanet:water", TintClass::Water),
        ]
    );
    // Grass and water leave their colour to the tint: the recipes paint them near-neutral
    // unless a colour is given.
    for name in ["grass_top", "grass_side", "water"] {
        assert!(procedural(&registry, name).1.is_empty(), "{name}");
    }
    for material in registry.materials() {
        assert!(
            (0.0..=1.0).contains(&material.roughness),
            "{}",
            material.key
        );
        assert!((0.0..=1.0).contains(&material.metallic), "{}", material.key);
        assert!(material.emissive >= 0.0, "{}", material.key);
    }
}

#[test]
fn nothing_is_defined_in_vain() {
    let registry = base();
    assert_eq!(registry.materials().len(), 36);
    let mut used_materials = vec![false; registry.materials().len()];
    for block in registry.blocks() {
        for material in block.faces.iter().flatten() {
            used_materials[material.index()] = true;
        }
    }
    for (material, used) in registry.materials().iter().zip(&used_materials) {
        assert!(used, "no block uses the material `{}`", material.key);
    }
    let mut used_textures = vec![false; registry.textures().len()];
    for material in registry.materials() {
        used_textures[material.texture.index()] = true;
    }
    for (texture, used) in registry.textures().iter().zip(&used_textures) {
        assert!(used, "no material uses the texture `{}`", texture.key);
    }
}

#[test]
fn hash_is_the_same_for_every_load() {
    let first = base();
    let second = base();
    assert_eq!(first.content_hash(), second.content_hash());
    assert_ne!(first.content_hash(), 0);
    assert_ne!(
        first.content_hash(),
        Registry::builtin_minimal().content_hash()
    );
    // Loading the same pack twice over itself overrides everything with itself.
    let dir = base_dir();
    let twice = Registry::load(&[&dir, &dir]).unwrap();
    assert_eq!(twice.content_hash(), first.content_hash());
    assert_eq!(twice.packs().len(), 2);
}

/// A definition with its references spelled out as keys, so that definitions of two registries
/// can be compared although their ids differ.
fn portable_block(registry: &Registry, block: &BlockDef) -> String {
    let faces = block.faces.map(|faces| {
        faces.map(|material| {
            let material = &registry.materials()[material.index()];
            let texture = &registry.textures()[material.texture.index()];
            format!(
                "{} {} {} {} {:?} {:?} / {texture:?}",
                material.key,
                material.roughness,
                material.metallic,
                material.emissive,
                material.tint,
                material.flags
            )
        })
    });
    format!(
        "{} {:?} {} {} {} {:?} {} {} {} {:?} {:?} {:?} {:?} {faces:?}",
        block.key,
        block.orient,
        block.solid,
        block.opaque,
        block.structural,
        block.render,
        block.hardness,
        block.density,
        block.light,
        block.tags,
        block.cull_group,
        block.lod_proxy,
        block.ext
    )
}

#[test]
fn builtin_minimal_is_a_part_of_the_base_pack() {
    let registry = base();
    let minimal = Registry::builtin_minimal();
    for small in minimal.blocks() {
        let big = registry
            .block(&small.key)
            .unwrap_or_else(|| panic!("`{}` is built in but not in the base pack", small.key));
        assert_eq!(
            portable_block(&minimal, small),
            portable_block(&registry, big)
        );
    }
    let (big, small) = (registry.tables(), minimal.tables());
    for index in 0..minimal.state_count() {
        let key = minimal.state_key(BlockStateId(index as u32));
        let id = state(&registry, key);
        assert_eq!(big.flags[id.index()], small.flags[index], "{key}");
        assert_eq!(
            big.render[id.index()].class,
            small.render[index].class,
            "{key}"
        );
        assert_eq!(big.lod[id.index()].class, small.lod[index].class, "{key}");
        // Cull groups are numbered within each registry; their names agree.
        let name = |registry: &Registry, group: u16| {
            (group > 0).then(|| registry.cull_groups()[usize::from(group) - 1].clone())
        };
        assert_eq!(
            name(&registry, big.cull_group[id.index()]),
            name(&minimal, small.cull_group[index]),
            "{key}"
        );
    }
}

#[test]
fn see_through_blocks_have_the_cull_groups_of_the_contract() {
    let registry = base();
    assert_eq!(registry.cull_groups(), ["lava", "leaves", "water"]);
    // Every species of leaves shares one group: a canopy has no inner faces.
    for species in SPECIES {
        assert_eq!(
            cull_group(&registry, &format!("cubeplanet:leaves_{species}")),
            2
        );
    }
    // Water and lava have their own groups, so where they meet both surfaces are drawn.
    assert_eq!(cull_group(&registry, "cubeplanet:lava"), 1);
    assert_eq!(cull_group(&registry, "cubeplanet:water"), 3);
    // Opaque blocks hide their neighbours anyway and have none.
    for block in registry.blocks() {
        if block.opaque || block.render == RenderClass::None {
            assert_eq!(block.cull_group, None, "{}", block.key);
        }
    }
    // Ice is opaque until the renderer has a translucent layer.
    let ice = block(&registry, "ice");
    assert!(ice.opaque && ice.cull_group.is_none());
    assert_eq!(
        flags(&registry, "cubeplanet:ice"),
        BlockFlags::SOLID | BlockFlags::OPAQUE
    );
}

#[test]
fn every_state_stands_in_for_itself_at_a_distance() {
    let registry = base();
    for index in 0..registry.state_count() {
        let id = BlockStateId(index as u32);
        let block = registry.block_of(id);
        let info = registry.tables().lod[index];
        // The base pack has no model blocks: every state is its own proxy.
        assert_ne!(block.render, RenderClass::Model);
        assert_eq!(info.proxy, id, "{}", block.key);
        let expected = match block.render {
            RenderClass::Fluid => LodClass::Fluid,
            RenderClass::None => LodClass::Empty,
            _ => LodClass::Solid,
        };
        assert_eq!(info.class, expected, "{}", block.key);
    }
    assert_eq!(lod(&registry, "cubeplanet:air"), LodInfo::EMPTY);
    assert_eq!(
        lod(&registry, "cubeplanet:leaves_cherry").class,
        LodClass::Solid
    );
}
