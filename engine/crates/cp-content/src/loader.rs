//! Turns the text of content packs into a [`Registry`].
//!
//! 1. **Merge.** Every pack is parsed on its own; a key defined twice inside one pack is an
//!    error. Packs are then laid over each other in load order: a later pack replaces a whole
//!    definition of an earlier one that has the same kind and key.
//! 2. **Validate and resolve.** Only the merged result is checked, in key order: textures,
//!    then materials (which name textures), then blocks (which name materials). A reference is
//!    therefore free to point into another pack, and an override is seen by everything that
//!    refers to the overridden key.
//! 3. **Assemble.** [`Registry::from_defs`] enumerates the block states and builds the tables.
//!
//! All intermediate maps are `BTreeMap`s, so neither the result nor the error that is reported
//! first depends on hashing or on the order of files on disk.

use std::collections::BTreeMap;
use std::collections::btree_map::Entry as Slot;
use std::path::Path;
use std::rc::Rc;

use cp_core::block::RenderClass;
use cp_core::ids::{MaterialId, TextureLayer};

use crate::block::BlockDef;
use crate::error::{ContentError, DefKind};
use crate::file_path::{self, Listings};
use crate::key::{AIR_KEY, StateKey, check_key, is_ident};
use crate::material::{MaterialDef, MaterialFlags};
use crate::pack::{PackInfo, PackSource};
use crate::registry::Registry;
use crate::schema::{
    Entry, Keyed, RawBlock, RawMaterial, RawMaterialFlag, RawRender, RawTexture, parse_entries,
};
use crate::texture::{Provenance, TextureDef, TextureSource};

/// `MaterialId` is 16 bits wide (and so is the material field of a packed quad).
const MAX_MATERIALS: usize = 1 << 16;
/// `TextureLayer` is 16 bits wide.
const MAX_TEXTURES: usize = 1 << 16;
/// Highest light emission level of a block.
const MAX_LIGHT: u8 = 15;

/// Builds the registry of `packs`; later packs override earlier ones.
pub(crate) fn assemble(packs: &[PackSource]) -> Result<Registry, ContentError> {
    let mut merged = Merged::default();
    let mut infos = Vec::with_capacity(packs.len());
    for pack in packs {
        infos.push(merged.add_pack(pack)?);
    }
    merged.provide_air();

    let textures = resolve_textures(&merged.textures)?;
    let materials = resolve_materials(&merged.materials, &textures)?;
    let blocks = resolve_blocks(&merged.blocks, &materials)?;
    check_lod_proxies(&merged.blocks, &blocks)?;
    Registry::from_defs(infos, textures, materials, blocks)
}

/// Where a definition was written.
#[derive(Clone, Debug)]
struct Origin {
    file: Rc<Path>,
    /// Line of the definition's header; 0 for definitions the engine provides.
    line: u32,
    /// Directory of the defining pack, `None` for embedded packs.
    pack_dir: Option<Rc<Path>>,
}

impl Origin {
    fn invalid(&self, kind: DefKind, key: &str, message: impl Into<String>) -> ContentError {
        ContentError::Invalid {
            kind,
            key: key.to_owned(),
            file: self.file.to_path_buf(),
            line: self.line,
            message: message.into(),
        }
    }

    fn unknown_reference(
        &self,
        kind: DefKind,
        key: &str,
        target_kind: DefKind,
        target: &str,
    ) -> ContentError {
        ContentError::UnknownReference {
            kind,
            key: key.to_owned(),
            file: self.file.to_path_buf(),
            line: self.line,
            target_kind,
            target: target.to_owned(),
        }
    }
}

/// A raw definition and where it came from.
#[derive(Debug)]
struct Def<T> {
    raw: T,
    origin: Origin,
}

type Defs<T> = BTreeMap<String, Def<T>>;

/// The definitions of one pack, or of all packs laid over each other, by key.
#[derive(Debug, Default)]
struct Merged {
    textures: Defs<RawTexture>,
    materials: Defs<RawMaterial>,
    blocks: Defs<RawBlock>,
}

impl Merged {
    /// Parses `pack` and lays it over what is already there.
    fn add_pack(&mut self, pack: &PackSource) -> Result<PackInfo, ContentError> {
        let info = pack.info()?;
        let pack_dir: Option<Rc<Path>> = pack.dir.as_deref().map(Rc::from);
        let mut own = Merged::default();
        for file in &pack.files {
            let path: Rc<Path> = Rc::from(file.path.as_path());
            let origin = |line| Origin {
                file: Rc::clone(&path),
                line,
                pack_dir: pack_dir.clone(),
            };
            let table = file.kind.name();
            match file.kind {
                DefKind::Texture => {
                    let entries = parse_entries(&file.path, &file.text, table)?;
                    insert_all(file.kind, &mut own.textures, entries, origin)?;
                }
                DefKind::Material => {
                    let entries = parse_entries(&file.path, &file.text, table)?;
                    insert_all(file.kind, &mut own.materials, entries, origin)?;
                }
                DefKind::Block => {
                    let entries = parse_entries(&file.path, &file.text, table)?;
                    insert_all(file.kind, &mut own.blocks, entries, origin)?;
                }
            }
        }
        lay_over(DefKind::Texture, &info, &mut self.textures, own.textures);
        lay_over(DefKind::Material, &info, &mut self.materials, own.materials);
        lay_over(DefKind::Block, &info, &mut self.blocks, own.blocks);
        Ok(info)
    }

    /// `cubeplanet:air` exists in every registry, whether or not a pack defines it.
    fn provide_air(&mut self) {
        self.blocks
            .entry(AIR_KEY.to_owned())
            .or_insert_with(|| Def {
                raw: RawBlock::air(AIR_KEY),
                origin: Origin {
                    file: Rc::from(Path::new("<builtin>")),
                    line: 0,
                    pack_dir: None,
                },
            });
    }
}

/// Adds the entries of one file to the definitions of its pack. A key may occur once per pack.
fn insert_all<T: Keyed>(
    kind: DefKind,
    defs: &mut Defs<T>,
    entries: Vec<Entry<T>>,
    origin: impl Fn(u32) -> Origin,
) -> Result<(), ContentError> {
    for entry in entries {
        let origin = origin(entry.line);
        let key = entry.raw.key().to_owned();
        check_key(&key).map_err(|message| origin.invalid(kind, &key, message))?;
        match defs.entry(key) {
            Slot::Vacant(slot) => {
                slot.insert(Def {
                    raw: entry.raw,
                    origin,
                });
            }
            Slot::Occupied(slot) => {
                let first = &slot.get().origin;
                return Err(ContentError::DuplicateKey {
                    kind,
                    key: slot.key().clone(),
                    file: origin.file.to_path_buf(),
                    line: origin.line,
                    first_file: first.file.to_path_buf(),
                    first_line: first.line,
                });
            }
        }
    }
    Ok(())
}

/// Lays the definitions of a pack over those of the packs loaded before it.
fn lay_over<T>(kind: DefKind, pack: &PackInfo, merged: &mut Defs<T>, own: Defs<T>) {
    for (key, def) in own {
        if let Some(previous) = merged.get(&key) {
            log::debug!(
                "content: pack `{}` overrides {kind} `{key}` (was {}:{})",
                pack.name,
                previous.origin.file.display(),
                previous.origin.line
            );
        }
        merged.insert(key, def);
    }
}

fn check_limit(what: &'static str, count: usize, limit: usize) -> Result<(), ContentError> {
    if count > limit {
        Err(ContentError::Limit { what, count, limit })
    } else {
        Ok(())
    }
}

/// Index of the definition named `key` in a list sorted by key.
fn find<T>(sorted: &[T], key_of: impl Fn(&T) -> &str, key: &str) -> Option<usize> {
    sorted.binary_search_by(|def| key_of(def).cmp(key)).ok()
}

fn resolve_textures(defs: &Defs<RawTexture>) -> Result<Vec<TextureDef>, ContentError> {
    check_limit("textures", defs.len(), MAX_TEXTURES)?;
    let mut listings = Listings::default();
    defs.iter()
        .map(|(key, def)| texture_def(key, def, &mut listings))
        .collect()
}

fn texture_def(
    key: &str,
    def: &Def<RawTexture>,
    listings: &mut Listings,
) -> Result<TextureDef, ContentError> {
    let raw = &def.raw;
    let invalid = |message: String| def.origin.invalid(DefKind::Texture, key, message);

    let source = match (&raw.procedural, &raw.file) {
        (Some(procedural), None) => {
            if !is_ident(&procedural.recipe) {
                return Err(invalid(format!(
                    "recipe name `{}` must be made of `a-z`, `0-9` and `_`",
                    procedural.recipe
                )));
            }
            let mut names = procedural.params.iter().map(|(name, _)| name);
            if let Some(name) = names.find(|name| !is_ident(name)) {
                return Err(invalid(format!(
                    "parameter name `{name}` must be made of `a-z`, `0-9` and `_`"
                )));
            }
            TextureSource::Procedural {
                recipe: procedural.recipe.clone(),
                params: procedural.params.clone(),
                seed: procedural.seed,
            }
        }
        (None, Some(path)) => {
            file_path::check(path).map_err(invalid)?;
            let Some(pack_dir) = &def.origin.pack_dir else {
                return Err(invalid(
                    "file textures need a pack directory; built-in content has none".to_owned(),
                ));
            };
            TextureSource::File {
                path: path.clone(),
                resolved: listings.find(pack_dir, path, invalid)?,
            }
        }
        (Some(_), Some(_)) => {
            return Err(invalid(
                "give either `procedural` or `file`, not both".to_owned(),
            ));
        }
        (None, None) => {
            return Err(invalid(
                "no source: give `procedural = { recipe = \"…\", seed = … }` or `file = \"…\"`"
                    .to_owned(),
            ));
        }
    };

    let licence = raw.licence.clone();
    if licence.as_ref().is_some_and(|text| text.trim().is_empty()) {
        return Err(invalid("`licence` must not be empty".to_owned()));
    }
    if raw.provenance == Provenance::Placeholder && licence.is_none() {
        return Err(invalid(
            "a placeholder texture must state its `licence`".to_owned(),
        ));
    }
    Ok(TextureDef {
        key: key.to_owned(),
        source,
        provenance: raw.provenance,
        licence,
    })
}

fn resolve_materials(
    defs: &Defs<RawMaterial>,
    textures: &[TextureDef],
) -> Result<Vec<MaterialDef>, ContentError> {
    check_limit("materials", defs.len(), MAX_MATERIALS)?;
    defs.iter()
        .map(|(key, def)| material_def(key, def, textures))
        .collect()
}

fn material_def(
    key: &str,
    def: &Def<RawMaterial>,
    textures: &[TextureDef],
) -> Result<MaterialDef, ContentError> {
    let raw = &def.raw;
    let kind = DefKind::Material;
    let invalid = |message: String| def.origin.invalid(kind, key, message);

    let texture = find(textures, |texture| &texture.key, &raw.texture).ok_or_else(|| {
        def.origin
            .unknown_reference(kind, key, DefKind::Texture, &raw.texture)
    })?;

    let mut flags = MaterialFlags::empty();
    for raw_flag in &raw.flags {
        let (name, flag) = match raw_flag {
            RawMaterialFlag::Cutout => ("cutout", MaterialFlags::CUTOUT),
            RawMaterialFlag::Translucent => ("translucent", MaterialFlags::TRANSLUCENT),
        };
        if flags.contains(flag) {
            return Err(invalid(format!("flag `{name}` is listed twice")));
        }
        flags |= flag;
    }

    Ok(MaterialDef {
        key: key.to_owned(),
        // The limit check in `resolve_textures` keeps every index within 16 bits.
        texture: TextureLayer(texture as u16),
        roughness: unit_interval("roughness", raw.roughness).map_err(invalid)?,
        metallic: unit_interval("metallic", raw.metallic).map_err(invalid)?,
        emissive: non_negative("emissive", raw.emissive).map_err(invalid)?,
        tint: raw.tint,
        flags,
    })
}

/// A number in `0.0..=1.0`.
fn unit_interval(name: &str, value: f64) -> Result<f32, String> {
    if (0.0..=1.0).contains(&value) {
        // Adding zero turns -0.0 into +0.0, so equal values hash equally.
        Ok(value as f32 + 0.0)
    } else {
        Err(format!("`{name}` must be between 0 and 1, got {value:?}"))
    }
}

/// A finite number `>= 0` that fits an `f32`.
fn non_negative(name: &str, value: f64) -> Result<f32, String> {
    if (0.0..=f64::from(f32::MAX)).contains(&value) {
        Ok(value as f32 + 0.0)
    } else {
        Err(format!(
            "`{name}` must be a finite number >= 0, got {value:?}"
        ))
    }
}

/// A number `>= 0`; infinity is allowed and means "cannot be mined".
fn hardness(value: f64) -> Result<f32, String> {
    if value == f64::INFINITY {
        Ok(f32::INFINITY)
    } else {
        non_negative("hardness", value)
            .map_err(|message| format!("{message} (`inf` makes a block unbreakable)"))
    }
}

fn resolve_blocks(
    defs: &Defs<RawBlock>,
    materials: &[MaterialDef],
) -> Result<Vec<BlockDef>, ContentError> {
    defs.iter()
        .map(|(key, def)| block_def(key, def, materials))
        .collect()
}

fn block_def(
    key: &str,
    def: &Def<RawBlock>,
    materials: &[MaterialDef],
) -> Result<BlockDef, ContentError> {
    let raw = &def.raw;
    let kind = DefKind::Block;
    let invalid = |message: String| def.origin.invalid(kind, key, message);

    let render = match raw.render {
        RawRender::None => RenderClass::None,
        RawRender::Cube => RenderClass::Cube,
        RawRender::Cutout => RenderClass::Cutout,
        RawRender::Fluid => RenderClass::Fluid,
        RawRender::Model => RenderClass::Model,
    };
    let solid = raw.solid.unwrap_or(matches!(
        render,
        RenderClass::Cube | RenderClass::Cutout | RenderClass::Model
    ));
    let opaque = raw.opaque.unwrap_or(render == RenderClass::Cube);

    // An opaque block hides the faces of its neighbours, so it must itself cover its whole
    // cell with opaque pixels; anything else would leave holes in the world.
    let not_opaque = |what: &str| -> Result<(), ContentError> {
        if opaque {
            Err(invalid(format!(
                "{what} cannot be `opaque`: it would hide its neighbours' faces without \
                 covering them"
            )))
        } else {
            Ok(())
        }
    };
    match render {
        RenderClass::None => not_opaque("a block with `render = \"none\"`")?,
        RenderClass::Cutout => not_opaque("a `cutout` block")?,
        RenderClass::Fluid => {
            not_opaque("a `fluid` block")?;
            if solid {
                return Err(invalid("a `fluid` block cannot be `solid`".to_owned()));
            }
        }
        RenderClass::Model => not_opaque("a `model` block")?,
        RenderClass::Cube => {}
    }
    // The flag keeps solid cells in coarser levels; on anything else it would mean nothing.
    if raw.structural && !solid {
        return Err(invalid(
            "only a `solid` block can be `structural`: the flag keeps solid cells when coarser \
             levels are derived"
                .to_owned(),
        ));
    }

    let faces = match (&raw.faces, render) {
        (None, RenderClass::None | RenderClass::Model) => None,
        (Some(_), RenderClass::None) => {
            return Err(invalid(
                "a block with `render = \"none\"` is not drawn and cannot have `faces`".to_owned(),
            ));
        }
        (Some(_), RenderClass::Model) => {
            return Err(invalid(
                "a `model` block is drawn by the model path and cannot have `faces`".to_owned(),
            ));
        }
        (None, _) => {
            return Err(invalid(
                "`faces` is missing: a drawn block needs a material for every face".to_owned(),
            ));
        }
        (Some(raw_faces), _) => {
            let names = raw_faces.local_faces().map_err(invalid)?;
            let mut faces = [MaterialId(0); 6];
            for (face, name) in faces.iter_mut().zip(names) {
                let index = find(materials, |material| &material.key, name).ok_or_else(|| {
                    def.origin
                        .unknown_reference(kind, key, DefKind::Material, name)
                })?;
                let material = &materials[index];
                // Only the cutout layer cuts holes (contract §5.2): `cube` blocks are meshed
                // into the opaque layer and fluids into the fluid layer, which draw every
                // texel, so the holes would show as solid colour. This also covers the
                // opaque blocks, which are all cubes.
                if material.flags.contains(MaterialFlags::CUTOUT) {
                    match render {
                        RenderClass::Cube => {
                            return Err(invalid(format!(
                                "a `cube` block cannot use the cut-out material `{}`: it is \
                                 drawn without cutting holes; set `render = \"cutout\"`",
                                material.key
                            )));
                        }
                        RenderClass::Fluid => {
                            return Err(invalid(format!(
                                "a `fluid` block cannot use the cut-out material `{}`: fluids \
                                 are drawn without cutting holes; use a material without the \
                                 `cutout` flag",
                                material.key
                            )));
                        }
                        _ => {}
                    }
                }
                // The same reasoning as above, one level down: a see-through face does not
                // cover what an opaque block hides.
                if opaque && material.flags.contains(MaterialFlags::TRANSLUCENT) {
                    return Err(invalid(format!(
                        "an `opaque` block cannot use the translucent material `{}`: it would \
                         hide what shows through it; set `opaque = false`",
                        material.key
                    )));
                }
                // The limit check in `resolve_materials` keeps every index within 16 bits.
                *face = MaterialId(index as u16);
            }
            Some(faces)
        }
    };

    if raw.light > MAX_LIGHT {
        return Err(invalid(format!(
            "`light` must be between 0 and {MAX_LIGHT}, got {}",
            raw.light
        )));
    }

    let mut tags = raw.tags.clone();
    if let Some(tag) = tags.iter().find(|tag| !is_ident(tag)) {
        return Err(invalid(format!(
            "tag `{tag}` must be made of `a-z`, `0-9` and `_`"
        )));
    }
    tags.sort();
    if let Some(pair) = tags.windows(2).find(|pair| pair[0] == pair[1]) {
        return Err(invalid(format!("tag `{}` is listed twice", pair[0])));
    }

    if let Some(group) = &raw.cull_group {
        if !is_ident(group) {
            return Err(invalid(format!(
                "cull group `{group}` must be made of `a-z`, `0-9` and `_`"
            )));
        }
        // A face is hidden by a neighbour of its own group. An opaque block hides its
        // neighbours anyway, and the faces it would lose behind see-through members of its
        // group would leave holes; blocks that are not meshed would hide faces they do not
        // replace.
        if opaque {
            return Err(invalid(
                "an `opaque` block cannot have a `cull_group`: it hides its neighbours' faces \
                 anyway, and its own faces behind see-through members of the group would leave \
                 holes"
                    .to_owned(),
            ));
        }
        if matches!(render, RenderClass::None | RenderClass::Model) {
            return Err(invalid(
                "only `cube`, `cutout` and `fluid` blocks are meshed and can have a `cull_group`"
                    .to_owned(),
            ));
        }
    }

    // The proxy itself is resolved in `check_lod_proxies`, once all blocks are known.
    if raw.lod_proxy.is_some() && render != RenderClass::Model {
        return Err(invalid(
            "only a `model` block has a `lod_proxy`: cube-shaped states stand in for themselves \
             in coarser levels"
                .to_owned(),
        ));
    }

    if let Some(namespace) = raw.ext.keys().find(|namespace| !is_ident(namespace)) {
        return Err(invalid(format!(
            "`ext` namespace `{namespace}` must be made of `a-z`, `0-9` and `_`"
        )));
    }

    let block = BlockDef {
        key: key.to_owned(),
        orient: raw.orient,
        solid,
        opaque,
        structural: raw.structural,
        render,
        faces,
        hardness: hardness(raw.hardness).map_err(invalid)?,
        density: non_negative("density", raw.density).map_err(invalid)?,
        light: raw.light,
        tags,
        cull_group: raw.cull_group.clone(),
        lod_proxy: raw.lod_proxy.clone(),
        ext: raw.ext.clone(),
        states: Vec::new(),
    };

    // State 0 is the empty cell everywhere in the engine: storage, meshing and collision rely
    // on air being undrawn, having no flags, no cull group and exactly one state.
    let empty = block.render == RenderClass::None
        && block.flags().is_empty()
        && block.orient.state_count() == 1;
    if key == AIR_KEY && !empty {
        return Err(invalid(format!(
            "`{AIR_KEY}` is the empty block: it must keep `orient = \"none\"` and \
             `render = \"none\"`, and be neither `solid` nor emit `light`"
        )));
    }
    Ok(block)
}

/// Checks that the `lod_proxy` of every `model` block names a state of a cube-shaped block.
///
/// `defs` and `blocks` hold the same blocks in the same (key) order.
fn check_lod_proxies(defs: &Defs<RawBlock>, blocks: &[BlockDef]) -> Result<(), ContentError> {
    let kind = DefKind::Block;
    for ((key, def), block) in defs.iter().zip(blocks) {
        let Some(proxy) = &block.lod_proxy else {
            continue;
        };
        let invalid = |message: String| def.origin.invalid(kind, key, message);
        let parsed = StateKey::parse(proxy)
            .map_err(|error| invalid(format!("`lod_proxy` is not a state key: {error}")))?;
        let target = find(blocks, |block| &block.key, parsed.block()).ok_or_else(|| {
            def.origin
                .unknown_reference(kind, key, DefKind::Block, parsed.block())
        })?;
        let target = &blocks[target];
        if target.orient.resolve(parsed.properties()).is_none() {
            return Err(invalid(format!(
                "`lod_proxy` `{proxy}` is not a state of `{}`: write the bare key for its \
                 default state and only properties of its `orient`, with values other than \
                 their defaults",
                target.key
            )));
        }
        if !matches!(
            target.render,
            RenderClass::Cube | RenderClass::Cutout | RenderClass::Fluid
        ) {
            return Err(invalid(format!(
                "`lod_proxy` `{proxy}` must be a state of a `cube`, `cutout` or `fluid` block"
            )));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests;
