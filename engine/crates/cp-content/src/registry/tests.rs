//! Tests of the registry and its tables, on the built-in minimal content.

use std::iter;
use std::sync::Arc;

use cp_core::block::{BlockFlags, BlockRender};

use super::*;
use crate::block::Orient;
use crate::error::DefKind;
use crate::key::StateKey;
use crate::material::{MaterialFlags, TintClass};

#[test]
fn builtin_air_is_state_zero_and_empty() {
    let registry = Registry::builtin_minimal();
    assert_eq!(registry.state(AIR_KEY), Some(BlockStateId::AIR));
    assert_eq!(registry.state_key(BlockStateId::AIR), AIR_KEY);
    let tables = registry.tables();
    assert_eq!(tables.flags[0], BlockFlags::empty());
    assert_eq!(tables.render[0], BlockRender::NONE);
    assert_eq!(tables.cull_group[0], 0);
    assert_eq!(tables.lod[0], LodInfo::EMPTY);
    assert_eq!(tables.material_color[0], NO_MATERIAL_COLOR);
    assert_eq!(
        registry.orientation(BlockStateId::AIR),
        Orientation::IDENTITY
    );
    let air = registry.block_of(BlockStateId::AIR);
    assert_eq!(air.key, AIR_KEY);
    assert_eq!(air.states(), [BlockStateId::AIR]);
    assert_eq!(
        (air.solid, air.opaque, air.structural, air.render),
        (false, false, false, RenderClass::None)
    );
    assert_eq!(air.faces, None);
    assert_eq!(
        registry.rotated(BlockStateId::AIR, Orientation::ALL[17]),
        BlockStateId::AIR
    );
}

#[test]
fn builtin_states_are_sorted_and_complete() {
    let registry = Registry::builtin_minimal();
    let keys: Vec<&str> = (0..registry.state_count())
        .map(|index| registry.state_key(BlockStateId(index as u32)))
        .collect();
    assert_eq!(
        keys,
        [
            "cubeplanet:air",
            "cubeplanet:dirt",
            "cubeplanet:grass",
            "cubeplanet:grass[up=+x]",
            "cubeplanet:grass[up=+z]",
            "cubeplanet:grass[up=-x]",
            "cubeplanet:grass[up=-y]",
            "cubeplanet:grass[up=-z]",
            "cubeplanet:lava",
            "cubeplanet:leaves_oak",
            "cubeplanet:log_oak",
            "cubeplanet:log_oak[axis=x]",
            "cubeplanet:log_oak[axis=z]",
            "cubeplanet:sand",
            "cubeplanet:stone",
            "cubeplanet:water",
        ]
    );
    // Everything after air is in plain byte order.
    assert!(keys[1..].is_sorted());
    let tables = registry.tables();
    assert_eq!(tables.flags.len(), keys.len());
    assert_eq!(tables.render.len(), keys.len());
    assert_eq!(tables.cull_group.len(), keys.len());
    assert_eq!(tables.lod.len(), keys.len());
    assert_eq!(tables.material_color.len(), keys.len());

    let block_keys: Vec<&str> = registry.blocks().iter().map(|b| b.key.as_str()).collect();
    assert_eq!(
        block_keys,
        [
            "cubeplanet:air",
            "cubeplanet:dirt",
            "cubeplanet:grass",
            "cubeplanet:lava",
            "cubeplanet:leaves_oak",
            "cubeplanet:log_oak",
            "cubeplanet:sand",
            "cubeplanet:stone",
            "cubeplanet:water",
        ]
    );
    assert_eq!(registry.packs().len(), 1);
    assert_eq!(registry.packs()[0].name, "minimal");
}

#[test]
fn builtin_tables_are_consistent() {
    let registry = Registry::builtin_minimal();
    assert!(registry.blocks().is_sorted_by(|a, b| a.key < b.key));
    assert!(registry.materials().is_sorted_by(|a, b| a.key < b.key));
    assert!(registry.textures().is_sorted_by(|a, b| a.key < b.key));

    for (index, material) in registry.materials().iter().enumerate() {
        assert_eq!(
            registry.material(&material.key),
            Some(MaterialId(index as u16))
        );
        assert!(material.texture.index() < registry.textures().len());
    }
    for (index, texture) in registry.textures().iter().enumerate() {
        assert_eq!(
            registry.texture(&texture.key),
            Some(TextureLayer(index as u16))
        );
        assert_eq!(texture.provenance, Provenance::Own);
        assert!(matches!(texture.source, TextureSource::Procedural { .. }));
    }
    assert_eq!(registry.material("cubeplanet:nope"), None);
    assert_eq!(registry.texture("cubeplanet:nope"), None);
    assert!(registry.block("cubeplanet:nope").is_none());

    let tables = registry.tables();
    let mut seen = vec![false; registry.state_count()];
    for block in registry.blocks() {
        assert_eq!(block.states().len(), block.orient.state_count());
        assert_eq!(registry.state(&block.key), Some(block.default_state()));
        assert_eq!(registry.block(&block.key), Some(block));
        assert_eq!(block.faces.is_some(), block.render != RenderClass::None);
        for &state in block.states() {
            assert!(!seen[state.index()], "state {state} belongs to two blocks");
            seen[state.index()] = true;
            assert_eq!(registry.block_of(state), block);
            assert_eq!(tables.flags[state.index()], block.flags());
            let orientation = registry.orientation(state);
            assert_eq!(block.state_for(orientation), state);
            let render = tables.render[state.index()];
            assert_eq!(render, block.render_for(orientation));
            assert_eq!(render.class, block.render);
            for face in Face::ALL {
                assert!(render.face(face).material.index() < registry.materials().len());
                assert!(render.face(face).uv_rot < 4);
            }
            // Cube-shaped states stand in for themselves.
            assert_eq!(tables.lod[state.index()].proxy, state);
        }
    }
    assert!(seen.iter().all(|&covered| covered));
}

#[test]
fn builtin_flags() {
    let registry = Registry::builtin_minimal();
    let flags = |key: &str| registry.tables().flags[registry.state(key).unwrap().index()];
    let solid_cube = BlockFlags::SOLID | BlockFlags::OPAQUE;
    assert_eq!(flags("cubeplanet:stone"), solid_cube);
    assert_eq!(flags("cubeplanet:dirt"), solid_cube);
    assert_eq!(flags("cubeplanet:sand"), solid_cube);
    assert_eq!(flags("cubeplanet:grass[up=-z]"), solid_cube);
    for log in [
        "cubeplanet:log_oak",
        "cubeplanet:log_oak[axis=x]",
        "cubeplanet:log_oak[axis=z]",
    ] {
        assert_eq!(flags(log), solid_cube | BlockFlags::STRUCTURAL, "{log}");
    }
    assert_eq!(
        flags("cubeplanet:leaves_oak"),
        BlockFlags::SOLID | BlockFlags::CUTOUT
    );
    assert_eq!(flags("cubeplanet:water"), BlockFlags::FLUID);
    assert_eq!(
        flags("cubeplanet:lava"),
        BlockFlags::FLUID | BlockFlags::EMISSIVE
    );
    let class = |key: &str| registry.tables().render[registry.state(key).unwrap().index()].class;
    assert_eq!(class("cubeplanet:stone"), RenderClass::Cube);
    assert_eq!(class("cubeplanet:leaves_oak"), RenderClass::Cutout);
    assert_eq!(class("cubeplanet:water"), RenderClass::Fluid);
    assert_eq!(class("cubeplanet:air"), RenderClass::None);
}

#[test]
fn builtin_cull_groups_and_lod() {
    let registry = Registry::builtin_minimal();
    assert_eq!(registry.cull_groups(), ["lava", "leaves", "water"]);
    let tables = registry.tables();
    let id = |key: &str| registry.state(key).unwrap();
    let group = |key: &str| tables.cull_group[id(key).index()];
    assert_eq!(group("cubeplanet:lava"), 1);
    assert_eq!(group("cubeplanet:leaves_oak"), 2);
    assert_eq!(group("cubeplanet:water"), 3);
    for key in [
        "cubeplanet:stone",
        "cubeplanet:grass[up=+x]",
        "cubeplanet:log_oak",
    ] {
        assert_eq!(group(key), 0, "{key}");
    }

    let lod = |key: &str| tables.lod[id(key).index()];
    for (key, class) in [
        ("cubeplanet:stone", LodClass::Solid),
        ("cubeplanet:grass[up=-x]", LodClass::Solid),
        ("cubeplanet:log_oak[axis=z]", LodClass::Solid),
        ("cubeplanet:leaves_oak", LodClass::Solid),
        ("cubeplanet:water", LodClass::Fluid),
        ("cubeplanet:lava", LodClass::Fluid),
        ("cubeplanet:air", LodClass::Empty),
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
fn builtin_materials() {
    let registry = Registry::builtin_minimal();
    let material = |key: &str| &registry.materials()[registry.material(key).unwrap().index()];
    assert_eq!(material("cubeplanet:grass_top").tint, TintClass::Grass);
    assert_eq!(material("cubeplanet:leaves_oak").tint, TintClass::Foliage);
    assert_eq!(
        material("cubeplanet:leaves_oak").flags,
        MaterialFlags::CUTOUT
    );
    assert_eq!(material("cubeplanet:water").tint, TintClass::Water);
    assert_eq!(
        material("cubeplanet:water").flags,
        MaterialFlags::TRANSLUCENT
    );
    assert!(material("cubeplanet:lava").emissive > 0.0);
    assert_eq!(material("cubeplanet:stone").flags, MaterialFlags::empty());
    assert_eq!(material("cubeplanet:stone").emissive, 0.0);
    // A material's texture is found through its layer.
    let stone = material("cubeplanet:stone");
    assert_eq!(
        registry.textures()[stone.texture.index()].key,
        "cubeplanet:stone"
    );
}

#[test]
fn builtin_blocks() {
    let registry = Registry::builtin_minimal();
    let block = |key: &str| registry.block(key).unwrap();
    assert_eq!(block("cubeplanet:grass").orient, Orient::Up6);
    assert_eq!(block("cubeplanet:log_oak").orient, Orient::Axis3);
    assert!(block("cubeplanet:log_oak").structural);
    assert_eq!(block("cubeplanet:stone").orient, Orient::None);
    assert_eq!(block("cubeplanet:lava").light, 15);
    assert!(block("cubeplanet:log_oak").has_tag("log"));
    assert_eq!(
        block("cubeplanet:leaves_oak").cull_group.as_deref(),
        Some("leaves")
    );
    for block in registry.blocks() {
        assert!(
            block.hardness >= 0.0 && block.density >= 0.0,
            "{}",
            block.key
        );
        assert!(block.tags.is_sorted(), "{}", block.key);
        assert!(
            block.ext.is_empty() && block.lod_proxy.is_none(),
            "{}",
            block.key
        );
    }
}

#[test]
fn state_keys_are_canonical() {
    let registry = Registry::builtin_minimal();
    for index in 0..registry.state_count() {
        let id = BlockStateId(index as u32);
        let key = registry.state_key(id);
        let parsed = StateKey::parse(key).unwrap_or_else(|error| panic!("{key}: {error}"));
        assert_eq!(parsed.to_string(), key);
        let block = registry.block_of(id);
        assert_eq!(parsed.block(), block.key);
        let variant = block.orient.resolve(parsed.properties());
        assert_eq!(
            variant.map(|index| block.states()[index]),
            Some(id),
            "{key}"
        );
    }
    // Spelled-out defaults and every other spelling are not keys (contract §4.4: only
    // non-default values are written).
    for key in [
        "cubeplanet:grass[up=+y]",
        "cubeplanet:log_oak[axis=y]",
        "cubeplanet:stone[up=+y]",
        "cubeplanet:stone[]",
        "cubeplanet:grass[up=+Y]",
        "cubeplanet:grass[up=y]",
        "cubeplanet:grass[axis=y]",
        "cubeplanet:grass [up=+x]",
        "cubeplanet:grass[up=+x,up=+x]",
        "cubeplanet:log_oak[axis=+x]",
        "cubeplanet:log_oak[up=+x]",
        "cubeplanet:grass[turn=1]",
        "grass",
        "",
    ] {
        assert_eq!(registry.state(key), None, "{key}");
    }
}

#[test]
fn rotation_composes_and_projects() {
    let registry = Registry::builtin_minimal();
    let grass = registry.block("cubeplanet:grass").unwrap();
    let log = registry.block("cubeplanet:log_oak").unwrap();
    let stone = registry.state("cubeplanet:stone").unwrap();
    for o in Orientation::ALL {
        // `none` never turns.
        assert_eq!(registry.rotated(stone, o), stone);
        for &state in grass.states() {
            let rotated = registry.rotated(state, o);
            let up = o.world_face(registry.orientation(state).up());
            assert_eq!(registry.orientation(rotated), Orientation::from_up(up));
            assert_eq!(registry.block_of(rotated).key, grass.key);
        }
        for &state in log.states() {
            let rotated = registry.rotated(state, o);
            let axis = o.world_face(registry.orientation(state).up()).axis();
            assert_eq!(
                registry.orientation(rotated),
                Orientation::from_up(Face::from_axis(axis, false))
            );
        }
    }
    // Turning by the identity changes nothing.
    for index in 0..registry.state_count() {
        let id = BlockStateId(index as u32);
        assert_eq!(registry.rotated(id, Orientation::IDENTITY), id);
    }
    // Lying grass is turned back up by the inverse rotation.
    let lying = registry.state("cubeplanet:grass[up=-z]").unwrap();
    let back = registry.orientation(lying).inverse();
    assert_eq!(registry.rotated(lying, back), grass.default_state());
}

#[test]
fn material_colours_default_to_grey_and_average_the_faces() {
    let mut registry = Registry::builtin_minimal();
    let hash = registry.content_hash();
    let before = registry.tables().clone();
    for (index, colour) in before.material_color.iter().enumerate() {
        let drawn = registry
            .block_of(BlockStateId(index as u32))
            .faces
            .is_some();
        let expected = if drawn {
            DEFAULT_MATERIAL_COLOR
        } else {
            NO_MATERIAL_COLOR
        };
        assert_eq!(*colour, expected, "state {index}");
    }

    // Material `m` gets the colour (m, 2m, 0.5).
    let colors: Vec<[f32; 3]> = (0..registry.materials().len())
        .map(|index| [index as f32, 2.0 * index as f32, 0.5])
        .collect();
    registry.set_material_colors(&colors);
    let tables = registry.tables();
    let color = |key: &str| tables.material_color[registry.state(key).unwrap().index()];
    let id = |key: &str| f32::from(registry.material(key).unwrap().0);
    let stone = id("cubeplanet:stone");
    assert_eq!(color("cubeplanet:stone"), [stone, 2.0 * stone, 0.5]);
    // Grass: one top, four sides, one bottom, whichever way it lies.
    let mean =
        (id("cubeplanet:grass_top") + 4.0 * id("cubeplanet:grass_side") + id("cubeplanet:dirt"))
            / 6.0;
    for key in ["cubeplanet:grass", "cubeplanet:grass[up=-x]"] {
        let [red, green, blue] = color(key);
        assert!(
            (red - mean).abs() < 1e-5 && (green - 2.0 * mean).abs() < 1e-5,
            "{key}"
        );
        assert_eq!(blue, 0.5);
    }
    assert_eq!(color("cubeplanet:air"), NO_MATERIAL_COLOR);

    // Only the colours changed; earlier clones keep theirs; the hash is not affected.
    assert!(Arc::ptr_eq(&tables.flags, &before.flags));
    assert!(Arc::ptr_eq(&tables.render, &before.render));
    assert!(Arc::ptr_eq(&tables.lod, &before.lod));
    assert!(!Arc::ptr_eq(&tables.material_color, &before.material_color));
    assert_eq!(before.material_color[1], DEFAULT_MATERIAL_COLOR);
    assert_eq!(registry.content_hash(), hash);
}

#[test]
#[should_panic(expected = "one colour per material")]
fn material_colours_need_one_entry_per_material() {
    let mut registry = Registry::builtin_minimal();
    registry.set_material_colors(&[[0.5; 3]]);
}

#[test]
#[should_panic(expected = "finite and >= 0")]
fn material_colours_must_be_finite() {
    let mut registry = Registry::builtin_minimal();
    let mut colors = vec![[0.5; 3]; registry.materials().len()];
    colors[0][1] = f32::NAN;
    registry.set_material_colors(&colors);
}

#[test]
fn tables_are_shared_not_copied() {
    let registry = Registry::builtin_minimal();
    let copy = registry.tables().clone();
    assert!(Arc::ptr_eq(&copy.flags, &registry.tables().flags));
    assert!(Arc::ptr_eq(&copy.cull_group, &registry.tables().cull_group));
    assert_eq!(&copy, registry.tables());
}

#[test]
fn builtin_hash_is_pinned() {
    // The hash must not depend on the machine, the run or the build profile. If this value
    // changes although the built-in content did not, the encoding changed by accident; if
    // the content changed on purpose, recompute it with `tests/reference_hash.py`, an
    // independent Python port of the hash that reads the same TOML files, and update it.
    let registry = Registry::builtin_minimal();
    assert_eq!(
        registry.content_hash(),
        BUILTIN_HASH,
        "actual hash: {:#018x}",
        registry.content_hash()
    );
    assert_eq!(Registry::builtin_minimal().content_hash(), BUILTIN_HASH);
    assert_eq!(registry.clone().content_hash(), BUILTIN_HASH);
}

/// Re-pinned for hash format 2 (contract v2: `u16` flags, structural, cull groups, LOD,
/// `[ext]` tables, material keys instead of ids) and the v2 built-in content (structural log,
/// cull groups of leaves, water and lava). Computed by `tests/reference_hash.py`.
const BUILTIN_HASH: u64 = 0x0267_e89c_5f77_97a7;

#[test]
fn registry_refuses_to_grow_to_the_unchanged_sentinel() {
    // The limit is the number of ids below the sentinel: the highest id a full registry hands
    // out is the one right before `UNCHANGED`.
    assert_eq!(MAX_STATES, u32::MAX as usize);
    let highest = BlockStateId(u32::try_from(MAX_STATES - 1).expect("every state id fits 32 bits"));
    assert!(!highest.is_unchanged());
    assert_eq!(highest.0 + 1, BlockStateId::UNCHANGED.0);

    // Synthetic content right at the limit, counted the way `from_defs` counts states before it
    // enumerates them, but without a single block in memory: as many `rot24` blocks as fit,
    // topped up with two `up6` blocks and one `axis3` block.
    let rot24 = Orient::Rot24.state_count();
    let top_up = [Orient::Up6, Orient::Up6, Orient::Axis3].map(Orient::state_count);
    assert_eq!(MAX_STATES % rot24, top_up.iter().sum::<usize>());
    let at_limit = || iter::repeat_n(rot24, MAX_STATES / rot24).chain(top_up);
    assert_eq!(checked_state_count(at_limit()).ok(), Some(MAX_STATES));

    let refusal = |result: Result<usize, ContentError>| match result {
        Err(ContentError::Limit { what, count, limit }) => (what, count, limit),
        other => panic!("expected a limit error, got {other:?}"),
    };
    // One `none` block more, and the last id would be the sentinel.
    let one_more = at_limit().chain([Orient::None.state_count()]);
    assert_eq!(
        refusal(checked_state_count(one_more)),
        ("block states", MAX_STATES + 1, MAX_STATES)
    );
    // Totals beyond `usize` saturate instead of wrapping around to a small count.
    assert_eq!(
        refusal(checked_state_count([usize::MAX, 2])),
        ("block states", usize::MAX, MAX_STATES)
    );

    // For real content the count is the number of states `from_defs` builds.
    let registry = Registry::builtin_minimal();
    let counted = checked_state_count(
        registry
            .blocks()
            .iter()
            .map(|block| block.orient.state_count()),
    );
    assert_eq!(counted.ok(), Some(registry.state_count()));
    assert!(registry.state_count() < MAX_STATES);
}

#[test]
fn builtin_sources_are_the_documented_kinds() {
    let source = builtin::source();
    let kinds: Vec<DefKind> = source.files.iter().map(|file| file.kind).collect();
    assert_eq!(kinds, DefKind::ALL);
    assert_eq!(source.dir, None);
}

#[test]
fn registries_and_errors_can_cross_threads() {
    fn assert_send_sync<T: Send + Sync + 'static>() {}
    // Workers share one registry; a load error may be handed to another thread.
    assert_send_sync::<Registry>();
    assert_send_sync::<BlockTables>();
    assert_send_sync::<ContentError>();

    let registry = Registry::builtin_minimal();
    let stone = registry.state("cubeplanet:stone");
    let seen: Vec<_> = std::thread::scope(|scope| {
        let workers: Vec<_> = (0..4)
            .map(|_| scope.spawn(|| registry.state("cubeplanet:stone")))
            .collect();
        workers
            .into_iter()
            .map(|worker| worker.join().unwrap())
            .collect()
    });
    assert_eq!(seen, [stone; 4]);
}
