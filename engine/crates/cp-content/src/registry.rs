//! The registry: every definition of the loaded packs plus the id-indexed tables derived from
//! them.

use std::collections::BTreeSet;
use std::path::Path;

use cp_core::block::{LodClass, LodInfo, RenderClass};
use cp_core::face::{Face, Orientation};
use cp_core::ids::{BlockStateId, MaterialId, TextureLayer};
use rustc_hash::FxHashMap;

use crate::block::BlockDef;
use crate::builtin;
use crate::canon::Canon;
use crate::error::ContentError;
use crate::key::{AIR_KEY, format_state_key};
use crate::loader;
use crate::material::MaterialDef;
use crate::pack::{PackInfo, PackSource};
use crate::tables::{BlockTables, DEFAULT_MATERIAL_COLOR, NO_MATERIAL_COLOR};
use crate::texture::{ParamValue, Provenance, TextureDef, TextureSource};

/// Version of the byte encoding behind [`Registry::content_hash`]. Bump it whenever
/// `hash_content` changes what it writes, so that old and new hashes can never be confused.
const HASH_FORMAT: u32 = 2;

/// Highest number of cull groups: their numbers `1..` must fit the `u16` of
/// [`BlockTables::cull_group`], where 0 means none.
const MAX_CULL_GROUPS: usize = u16::MAX as usize;

/// Highest number of block states: their ids `0..MAX_STATES` all lie below
/// [`BlockStateId::UNCHANGED`], the sentinel of edit layers that no registry hands out.
pub(crate) const MAX_STATES: usize = BlockStateId::UNCHANGED.index();

/// The number of block states of blocks that have `per_block` states each.
///
/// # Errors
/// [`ContentError::Limit`] when the total is above [`MAX_STATES`]; a total beyond `usize` is
/// reported as `usize::MAX`. The states are counted before any of them is enumerated, so that
/// content with too many states is refused before memory is spent on it.
fn checked_state_count(per_block: impl IntoIterator<Item = usize>) -> Result<usize, ContentError> {
    let count = per_block.into_iter().fold(0, usize::saturating_add);
    if count > MAX_STATES {
        return Err(ContentError::Limit {
            what: "block states",
            count,
            limit: MAX_STATES,
        });
    }
    Ok(count)
}

/// All blocks, block states, materials and textures of a set of content packs.
///
/// A registry is immutable apart from [`Registry::set_material_colors`]. Its ids
/// ([`BlockStateId`], [`MaterialId`], [`TextureLayer`]) are positions in its tables: they are
/// assigned deterministically, by sorting the keys, but they change whenever a pack adds or
/// removes a definition. They are therefore valid for one session only; anything persistent
/// stores keys ([`Registry::state_key`]) and translates them back with [`Registry::state`].
///
/// Invariants:
/// - state 0 is `cubeplanet:air`: no flags, not drawn, cull group 0, [`LodInfo::EMPTY`];
/// - every other state id follows the byte order of the state keys, so comparing two ids
///   other than air compares their keys;
/// - every table of [`Registry::tables`] has [`Registry::state_count`] entries;
/// - [`Registry::blocks`], [`Registry::materials`] and [`Registry::textures`] are sorted by
///   key, and every id stored in a definition is a valid index.
#[derive(Clone, Debug)]
pub struct Registry {
    packs: Vec<PackInfo>,
    blocks: Vec<BlockDef>,
    materials: Vec<MaterialDef>,
    textures: Vec<TextureDef>,
    /// Canonical key of every state, by state id.
    state_keys: Vec<String>,
    /// Index into `blocks` of every state's block, by state id.
    state_blocks: Vec<u32>,
    state_orientations: Vec<Orientation>,
    tables: BlockTables,
    /// Names of the cull groups, sorted; group `n` is `cull_groups[n - 1]`.
    cull_groups: Vec<String>,
    /// The canonical key of every state. Only ever queried, never iterated (contract §2.5).
    state_lookup: FxHashMap<String, BlockStateId>,
    content_hash: u64,
}

impl Registry {
    /// Loads the content packs in the directories `pack_dirs`.
    ///
    /// The packs are laid over each other in the given order: a definition of a later pack
    /// replaces, as a whole, a definition of an earlier pack with the same kind and key.
    /// References are resolved after that, so they may cross packs. Without any pack the
    /// registry holds `cubeplanet:air` only.
    ///
    /// # Errors
    /// The first problem found, with the file and, where it applies, the key it concerns:
    /// unreadable files, TOML and schema errors (unknown fields included), a key defined twice
    /// in one pack, references to keys that no pack defines, and values that break a rule.
    /// See [`ContentError`]. Which problem is found first does not depend on the machine.
    pub fn load(pack_dirs: &[&Path]) -> Result<Self, ContentError> {
        let mut sources = Vec::with_capacity(pack_dirs.len());
        for dir in pack_dirs {
            sources.push(PackSource::read(dir)?);
        }
        let registry = loader::assemble(&sources)?;
        log::info!(
            "content: {} pack(s), {} blocks, {} states, {} materials, {} textures, hash {:016x}",
            registry.packs.len(),
            registry.blocks.len(),
            registry.state_keys.len(),
            registry.materials.len(),
            registry.textures.len(),
            registry.content_hash
        );
        for texture in &registry.textures {
            if texture.provenance == Provenance::Placeholder {
                log::warn!(
                    "content: texture `{}` is placeholder art ({}) and must not ship",
                    texture.key,
                    texture.licence.as_deref().unwrap_or("no licence given")
                );
            }
        }
        Ok(registry)
    }

    /// A small registry that needs no files, for tests of this and other crates.
    ///
    /// It holds `cubeplanet:` `air`, `stone`, `dirt`, `sand`, `grass` (`up6`), `log_oak`
    /// (`axis3`, structural), `leaves_oak` (cut-out, cull group `leaves`), `water` and `lava`
    /// (fluids with cull groups of their own; lava emits light), with one material per
    /// distinct face and procedural textures. The same definitions exist in the base pack.
    #[must_use]
    pub fn builtin_minimal() -> Self {
        loader::assemble(&[builtin::source()])
            .expect("the built-in minimal content is valid; a unit test loads it")
    }

    /// The id of the state named `key`, e.g. `cubeplanet:stone` or `cubeplanet:log_oak[axis=z]`.
    ///
    /// Only the one canonical key of a state is accepted (contract §4.4): the bare block key
    /// for the default state, and no spelled-out default values (`cubeplanet:grass[up=+y]` is
    /// not a key). `None` for every other text.
    #[must_use]
    pub fn state(&self, key: &str) -> Option<BlockStateId> {
        self.state_lookup.get(key).copied()
    }

    /// The canonical key of the state `id`; [`Registry::state`] maps it back to `id`.
    ///
    /// # Panics
    /// If `id` is not a state of this registry (`id.index() >= state_count()`).
    #[must_use]
    pub fn state_key(&self, id: BlockStateId) -> &str {
        &self.state_keys[id.index()]
    }

    /// Number of block states; valid ids are `0..state_count()`. Never zero, and always below
    /// [`BlockStateId::UNCHANGED`], which no registry hands out.
    #[must_use]
    pub fn state_count(&self) -> usize {
        self.state_keys.len()
    }

    /// The per-state tables for storage, meshing and rendering, indexed by [`BlockStateId`].
    #[must_use]
    pub fn tables(&self) -> &BlockTables {
        &self.tables
    }

    /// The state of the block of `id` after rotating it by `o`: the state whose orientation is
    /// `o ∘ orientation(id)` ([`Orientation::compose`]) projected onto the block's orientation
    /// kind ([`Orient::project`](crate::Orient::project)): `up6` drops the turn, `axis3` keeps
    /// the axis, `none` stays as it is.
    ///
    /// # Panics
    /// If `id` is not a state of this registry.
    #[must_use]
    pub fn rotated(&self, id: BlockStateId, o: Orientation) -> BlockStateId {
        self.block_of(id).state_for(o.compose(self.orientation(id)))
    }

    /// All blocks, sorted by key.
    #[must_use]
    pub fn blocks(&self) -> &[BlockDef] {
        &self.blocks
    }

    /// The block named `key`.
    #[must_use]
    pub fn block(&self, key: &str) -> Option<&BlockDef> {
        let index = self
            .blocks
            .binary_search_by(|block| block.key.as_str().cmp(key))
            .ok()?;
        Some(&self.blocks[index])
    }

    /// The block that the state `id` is a state of.
    ///
    /// # Panics
    /// If `id` is not a state of this registry.
    #[must_use]
    pub fn block_of(&self, id: BlockStateId) -> &BlockDef {
        &self.blocks[self.state_blocks[id.index()] as usize]
    }

    /// How the block is rotated in the state `id`, one of the block's
    /// [`Orient::orientations`](crate::Orient::orientations); [`Orientation::IDENTITY`] for
    /// blocks that cannot be rotated.
    ///
    /// # Panics
    /// If `id` is not a state of this registry.
    #[must_use]
    pub fn orientation(&self, id: BlockStateId) -> Orientation {
        self.state_orientations[id.index()]
    }

    /// The names of all cull groups, sorted. Group number `n` of
    /// [`BlockTables::cull_group`] is `cull_groups()[n - 1]`; 0 is no group. The numbering
    /// depends only on the set of names, not on the order of packs or files.
    #[must_use]
    pub fn cull_groups(&self) -> &[String] {
        &self.cull_groups
    }

    /// All materials, sorted by key; the index of a material is its [`MaterialId`].
    #[must_use]
    pub fn materials(&self) -> &[MaterialDef] {
        &self.materials
    }

    /// The id of the material named `key`.
    #[must_use]
    pub fn material(&self, key: &str) -> Option<MaterialId> {
        let index = self
            .materials
            .binary_search_by(|material| material.key.as_str().cmp(key))
            .ok()?;
        // The loader rejects registries with more materials than a `u16` can index.
        Some(MaterialId(index as u16))
    }

    /// All textures, sorted by key; the index of a texture is its [`TextureLayer`].
    #[must_use]
    pub fn textures(&self) -> &[TextureDef] {
        &self.textures
    }

    /// The layer of the texture named `key`.
    #[must_use]
    pub fn texture(&self, key: &str) -> Option<TextureLayer> {
        let index = self
            .textures
            .binary_search_by(|texture| texture.key.as_str().cmp(key))
            .ok()?;
        // The loader rejects registries with more textures than a `u16` can index.
        Some(TextureLayer(index as u16))
    }

    /// The loaded packs in load order.
    #[must_use]
    pub fn packs(&self) -> &[PackInfo] {
        &self.packs
    }

    /// Fills [`BlockTables::material_color`] from the colours of the materials.
    ///
    /// `colors[m]` is the linear RGB average albedo of material `m` ([`MaterialId`] order),
    /// computed by the application from the material's texture. Every state with faces gets
    /// the average of its six face materials; states without faces keep
    /// [`NO_MATERIAL_COLOR`]. The other tables are untouched; bundles cloned from
    /// [`Registry::tables`] before the call keep the old colours.
    ///
    /// The colours are not content: [`Registry::content_hash`] does not change.
    ///
    /// # Panics
    /// If `colors` does not have one entry per material, or a channel is negative or not
    /// finite. Both are mistakes of the caller, not of a content pack.
    pub fn set_material_colors(&mut self, colors: &[[f32; 3]]) {
        assert_eq!(
            colors.len(),
            self.materials.len(),
            "set_material_colors needs one colour per material"
        );
        assert!(
            colors
                .iter()
                .flatten()
                .all(|channel| channel.is_finite() && *channel >= 0.0),
            "material colours must be finite and >= 0"
        );
        let table: Vec<[f32; 3]> = self
            .state_blocks
            .iter()
            .map(|&block| match self.blocks[block as usize].faces {
                Some(faces) => average_color(faces.map(|material| colors[material.index()])),
                None => NO_MATERIAL_COLOR,
            })
            .collect();
        self.tables.material_color = table.into();
    }

    /// A hash of everything the registry defines: all textures, materials and blocks with all
    /// their values (`[ext]` tables included), and the state tables derived from them, except
    /// the material colours that the application fills in later.
    ///
    /// Two registries with equal content have equal hashes on every machine and in every run,
    /// wherever the packs lie on disk and however their definitions are split into files; a
    /// change of any value changes the hash (up to 64-bit collisions). Pack names, versions and
    /// descriptions are not content and do not count, and neither do the pixels of image files:
    /// a file texture is hashed by its path inside the pack. Session-local ids never enter the
    /// hash; references are written as keys.
    #[must_use]
    pub fn content_hash(&self) -> u64 {
        self.content_hash
    }

    /// Enumerates the states of `blocks`, assigns the ids and builds the tables.
    ///
    /// `textures`, `materials` and `blocks` must be sorted by key with all references
    /// resolved (every `lod_proxy` names a state of a cube-shaped block), and `blocks` must
    /// contain `cubeplanet:air`; the loader guarantees all of it.
    ///
    /// The LOD stand-in of a rotated `model` state is its block's `lod_proxy` turned by the
    /// state's orientation (see [`BlockTables::lod`]).
    pub(crate) fn from_defs(
        packs: Vec<PackInfo>,
        textures: Vec<TextureDef>,
        materials: Vec<MaterialDef>,
        mut blocks: Vec<BlockDef>,
    ) -> Result<Self, ContentError> {
        // Ids stay below `BlockStateId::UNCHANGED`, the edit sentinel that is never a state.
        let count = checked_state_count(blocks.iter().map(|block| block.orient.state_count()))?;
        // (state key, block index, variant index)
        let mut pending: Vec<(String, usize, usize)> = Vec::with_capacity(count);
        for (block_index, block) in blocks.iter().enumerate() {
            for (variant, &orientation) in block.orient.orientations().iter().enumerate() {
                let key = format_state_key(&block.key, &block.orient.properties(orientation));
                pending.push((key, block_index, variant));
            }
        }
        debug_assert_eq!(pending.len(), count, "every state is counted once");
        // Air first, everything else in the byte order of the state keys.
        pending.sort_by(|a, b| {
            (a.0 != AIR_KEY)
                .cmp(&(b.0 != AIR_KEY))
                .then_with(|| a.0.cmp(&b.0))
        });
        assert_eq!(
            pending.first().map(|state| state.0.as_str()),
            Some(AIR_KEY),
            "the loader always provides the air block"
        );

        let cull_groups: Vec<String> = blocks
            .iter()
            .filter_map(|block| block.cull_group.clone())
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect();
        if cull_groups.len() > MAX_CULL_GROUPS {
            return Err(ContentError::Limit {
                what: "cull groups",
                count: cull_groups.len(),
                limit: MAX_CULL_GROUPS,
            });
        }
        let group_number = |name: &str| -> u16 {
            let index = cull_groups
                .binary_search_by(|group| group.as_str().cmp(name))
                .expect("every cull group name is collected above");
            // Checked above: at most `u16::MAX` groups, numbered from 1.
            (index + 1) as u16
        };

        for block in &mut blocks {
            block.states = vec![BlockStateId::AIR; block.orient.state_count()];
        }
        let mut state_keys = Vec::with_capacity(count);
        let mut state_blocks = Vec::with_capacity(count);
        let mut state_orientations = Vec::with_capacity(count);
        let mut flags = Vec::with_capacity(count);
        let mut render = Vec::with_capacity(count);
        let mut cull_group = Vec::with_capacity(count);
        let mut material_color = Vec::with_capacity(count);
        let mut state_lookup = FxHashMap::default();
        state_lookup.reserve(count);
        for (index, (key, block_index, variant)) in pending.into_iter().enumerate() {
            // Checked above: the number of states fits a `u32`.
            let id = BlockStateId(index as u32);
            let block = &mut blocks[block_index];
            let orientation = block.orient.orientations()[variant];
            block.states[variant] = id;
            flags.push(block.flags());
            render.push(block.render_for(orientation));
            cull_group.push(block.cull_group.as_deref().map_or(0, group_number));
            material_color.push(if block.faces.is_some() {
                DEFAULT_MATERIAL_COLOR
            } else {
                NO_MATERIAL_COLOR
            });
            state_blocks.push(block_index as u32);
            state_orientations.push(orientation);
            state_lookup.insert(key.clone(), id);
            state_keys.push(key);
        }

        // A model block's proxy may be any state, so the LOD table needs all ids first.
        let lod: Vec<LodInfo> = state_blocks
            .iter()
            .enumerate()
            .map(|(index, &block_index)| {
                let block = &blocks[block_index as usize];
                if block.render != RenderClass::Model {
                    return LodInfo {
                        class: lod_class(block),
                        proxy: BlockStateId(index as u32),
                    };
                }
                let Some(key) = &block.lod_proxy else {
                    return LodInfo::EMPTY;
                };
                let written = *state_lookup
                    .get(key)
                    .expect("the loader resolves every lod_proxy to a state");
                // The key names the stand-in of the unrotated model. A rotated state of the
                // model gets that stand-in turned the same way, as `Registry::rotated` turns a
                // state, so that its orientation survives into coarser levels; a stand-in that
                // cannot be rotated stays the written state.
                let proxy_block = &blocks[state_blocks[written.index()] as usize];
                let turned = state_orientations[index].compose(state_orientations[written.index()]);
                LodInfo {
                    class: lod_class(proxy_block),
                    proxy: proxy_block.state_for(turned),
                }
            })
            .collect();

        let tables = BlockTables {
            flags: flags.into(),
            render: render.into(),
            cull_group: cull_group.into(),
            lod: lod.into(),
            material_color: material_color.into(),
        };
        let content_hash = hash_content(
            &textures,
            &materials,
            &blocks,
            &cull_groups,
            &state_keys,
            &state_blocks,
            &tables,
        );
        Ok(Self {
            packs,
            blocks,
            materials,
            textures,
            state_keys,
            state_blocks,
            state_orientations,
            tables,
            cull_groups,
            state_lookup,
            content_hash,
        })
    }
}

/// The LOD class of the states of a cube-shaped block (contract §4.4): solid cubes and cutouts
/// count as solid, fluids as fluid, everything else (air, undrawn blocks, cubes that are not
/// solid) vanishes at a distance.
fn lod_class(block: &BlockDef) -> LodClass {
    match block.render {
        RenderClass::Cube | RenderClass::Cutout if block.solid => LodClass::Solid,
        RenderClass::Fluid => LodClass::Fluid,
        _ => LodClass::Empty,
    }
}

/// The plain average of six colours, summed in face order so the bits are the same everywhere.
///
/// The sum is taken in `f64`: there the sum of six equal `f32` values is exact (it needs at
/// most 27 significant bits), so a block with one material on all faces gets exactly that
/// material's colour, which an `f32` sum misses by a rounding step.
fn average_color(colors: [[f32; 3]; 6]) -> [f32; 3] {
    let mut sum = [0.0_f64; 3];
    for color in colors {
        for (total, channel) in sum.iter_mut().zip(color) {
            *total += f64::from(channel);
        }
    }
    sum.map(|total| (total / 6.0) as f32)
}

/// The bits every NaN float of an `[ext]` table is hashed as: TOML's `nan`, `+nan` and `-nan`
/// are one value to the hash.
const CANONICAL_NAN: u64 = 0x7FF8_0000_0000_0000;

/// Writes a TOML value of an `[ext]` table: a tag byte, then the payload. Tables are written in
/// key order, whatever order the TOML library keeps.
fn hash_toml_value(canon: &mut Canon, value: &toml::Value) {
    match value {
        toml::Value::String(text) => {
            canon.u8(0);
            canon.str(text);
        }
        toml::Value::Integer(number) => {
            canon.u8(1);
            // Two's complement.
            canon.u64(*number as u64);
        }
        toml::Value::Float(number) => {
            canon.u8(2);
            canon.u64(if number.is_nan() {
                CANONICAL_NAN
            } else {
                number.to_bits()
            });
        }
        toml::Value::Boolean(flag) => {
            canon.u8(3);
            canon.bool(*flag);
        }
        toml::Value::Datetime(datetime) => {
            canon.u8(4);
            canon.str(&datetime.to_string());
        }
        toml::Value::Array(items) => {
            canon.u8(5);
            canon.count(items.len());
            for item in items {
                hash_toml_value(canon, item);
            }
        }
        toml::Value::Table(table) => {
            canon.u8(6);
            hash_toml_table(canon, table);
        }
    }
}

fn hash_toml_table(canon: &mut Canon, table: &toml::Table) {
    let mut entries: Vec<(&String, &toml::Value)> = table.iter().collect();
    entries.sort_by(|a, b| a.0.cmp(b.0));
    canon.count(entries.len());
    for (key, value) in entries {
        canon.str(key);
        hash_toml_value(canon, value);
    }
}

/// Writes the whole content in a fixed order and hashes it. References are written as keys,
/// so the bytes describe the content itself rather than this session's ids; the derived state
/// tables follow at the end because meshes and saves depend on exactly those.
///
/// The byte layout is spelled out in `tests/reference_hash.py`, an independent port that must
/// produce the same hashes. Change both together, and bump [`HASH_FORMAT`].
fn hash_content(
    textures: &[TextureDef],
    materials: &[MaterialDef],
    blocks: &[BlockDef],
    cull_groups: &[String],
    state_keys: &[String],
    state_blocks: &[u32],
    tables: &BlockTables,
) -> u64 {
    let mut canon = Canon::default();
    canon.str("cp-content");
    canon.u32(HASH_FORMAT);

    canon.count(textures.len());
    for texture in textures {
        canon.str(&texture.key);
        match &texture.source {
            TextureSource::Procedural {
                recipe,
                params,
                seed,
            } => {
                canon.u8(0);
                canon.str(recipe);
                canon.u64(*seed);
                canon.count(params.len());
                for (name, value) in params.iter() {
                    canon.str(name);
                    match value {
                        ParamValue::Color([red, green, blue]) => {
                            canon.u8(0);
                            canon.u8(*red);
                            canon.u8(*green);
                            canon.u8(*blue);
                        }
                        ParamValue::Variant(variant) => {
                            canon.u8(1);
                            canon.str(variant);
                        }
                    }
                }
            }
            // Only the path inside the pack: `resolved` differs from machine to machine.
            TextureSource::File { path, resolved: _ } => {
                canon.u8(1);
                canon.str(path);
            }
        }
        canon.u8(match texture.provenance {
            Provenance::Own => 0,
            Provenance::Placeholder => 1,
        });
        canon.bool(texture.licence.is_some());
        canon.str(texture.licence.as_deref().unwrap_or(""));
    }

    canon.count(materials.len());
    for material in materials {
        canon.str(&material.key);
        canon.str(&textures[material.texture.index()].key);
        canon.f32(material.roughness);
        canon.f32(material.metallic);
        canon.f32(material.emissive);
        canon.u8(material.tint as u8);
        canon.u8(material.flags.bits());
    }

    canon.count(blocks.len());
    for block in blocks {
        canon.str(&block.key);
        canon.u8(block.orient as u8);
        canon.bool(block.solid);
        canon.bool(block.opaque);
        canon.bool(block.structural);
        canon.u8(block.render as u8);
        canon.bool(block.faces.is_some());
        for material in block.faces.iter().flatten() {
            canon.str(&materials[material.index()].key);
        }
        canon.f32(block.hardness);
        canon.f32(block.density);
        canon.u8(block.light);
        canon.count(block.tags.len());
        for tag in &block.tags {
            canon.str(tag);
        }
        canon.bool(block.cull_group.is_some());
        canon.str(block.cull_group.as_deref().unwrap_or(""));
        canon.bool(block.lod_proxy.is_some());
        canon.str(block.lod_proxy.as_deref().unwrap_or(""));
        // `BTreeMap`: namespaces in key order.
        canon.count(block.ext.len());
        for (namespace, table) in &block.ext {
            canon.str(namespace);
            hash_toml_table(&mut canon, table);
        }
    }

    canon.count(cull_groups.len());
    for group in cull_groups {
        canon.str(group);
    }

    canon.count(state_keys.len());
    for (index, key) in state_keys.iter().enumerate() {
        let block = &blocks[state_blocks[index] as usize];
        canon.str(key);
        canon.u16(tables.flags[index].bits());
        let render = &tables.render[index];
        canon.u8(render.class as u8);
        canon.bool(block.faces.is_some());
        if block.faces.is_some() {
            for face in Face::ALL {
                let appearance = render.face(face);
                canon.str(&materials[appearance.material.index()].key);
                canon.u8(appearance.uv_rot);
            }
        }
        canon.u16(tables.cull_group[index]);
        let lod = tables.lod[index];
        canon.u8(lod.class as u8);
        canon.str(&state_keys[lod.proxy.index()]);
    }
    canon.finish()
}

#[cfg(test)]
mod tests;
