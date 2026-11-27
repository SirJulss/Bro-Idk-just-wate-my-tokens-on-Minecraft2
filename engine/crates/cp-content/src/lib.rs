//! `cp-content`: the data registry of the Cubeplanet engine.
//!
//! Responsibility: blocks, block states, materials and textures as data, loaded from content
//! packs. It follows the determinism rules.
//!
//! Binding contract: `engine/docs/ARCHITEKTUR.md`, section 4.4.
//!
//! # What the registry provides
//!
//! A [`Registry`] is loaded once from a list of pack directories ([`Registry::load`]) and is
//! immutable afterwards, apart from the material colours the application fills in
//! ([`Registry::set_material_colors`]). It hands the rest of the engine dense tables indexed by
//! ids:
//!
//! - [`Registry::tables`], indexed by [`BlockStateId`]: flags, per-face render data, cull
//!   groups, LOD classes and proxies, material colours ([`BlockTables`]); what `cp-voxel`,
//!   `cp-mesh` and the renderer consume instead of the registry itself;
//! - [`Registry::materials`], indexed by [`MaterialId`]: the material table of the renderer;
//! - [`Registry::textures`], indexed by [`TextureLayer`]: the layers of the block texture
//!   array, which the application generates with `cp-proctex` or loads from files;
//! - [`Registry::blocks`], sorted by key: what gameplay needs (hardness, density, light,
//!   tags, `[ext]` tables), reached from a state through [`Registry::block_of`].
//!
//! Ids are assigned by sorting keys, with `cubeplanet:air` as state 0. They are the same for
//! the same set of definitions on every machine, but they shift when a pack adds a definition:
//! saved data stores keys ([`Registry::state_key`]) and maps them back with
//! [`Registry::state`]. [`Registry::content_hash`] tells whether two registries have the same
//! content.
//!
//! ```
//! use cp_content::Registry;
//! use cp_core::block::BlockFlags;
//! use cp_core::face::{Face, Orientation};
//! use cp_core::ids::BlockStateId;
//!
//! let registry = Registry::builtin_minimal();
//! assert_eq!(registry.state("cubeplanet:air"), Some(BlockStateId::AIR));
//!
//! // A grass block lying on the +X face of a cube planet: its top points to +X.
//! let grass = registry.state("cubeplanet:grass[up=+x]").expect("grass exists");
//! assert_eq!(registry.state_key(grass), "cubeplanet:grass[up=+x]");
//! let tables = registry.tables();
//! assert_eq!(tables.flags[grass.index()], BlockFlags::SOLID | BlockFlags::OPAQUE);
//! let top = tables.render[grass.index()].face(Face::PosX).material;
//! assert_eq!(registry.material("cubeplanet:grass_top"), Some(top));
//!
//! // The same state, found through the block, and by turning the default state.
//! let block = registry.block("cubeplanet:grass").expect("grass exists");
//! assert_eq!(block.state_for_up(Face::PosX), grass);
//! assert_eq!(registry.block_of(grass).key, "cubeplanet:grass");
//! let quarter = Orientation::from_up(Face::PosX);
//! assert_eq!(registry.rotated(block.default_state(), quarter), grass);
//! ```
//!
//! # Content packs
//!
//! A pack is a directory:
//!
//! ```text
//! <pack>/
//!   pack.toml             required
//!   textures/**/*.toml    [[texture]] entries
//!   materials/**/*.toml   [[material]] entries
//!   blocks/**/*.toml      [[block]] entries
//! ```
//!
//! Every `.toml` file below one of the three directories is a definition file of that kind;
//! how the definitions are split into files and sub-directories is up to the pack. Anything
//! else in a pack is left alone.
//!
//! Every definition has a `key` of the form `namespace:name` (`a-z`, `0-9`, `_`). Textures,
//! materials and blocks have separate key spaces. Blocks name materials and materials name
//! textures by key; a reference may point into another loaded pack.
//!
//! The schema is strict: an unknown field, a missing required field, a wrong type, a malformed
//! key, a value out of range, a key defined twice in one pack and a reference to nothing are
//! all errors, each reported as a [`ContentError`] with the file, the line and the key.
//!
//! **Overriding.** Packs are laid over each other in load order. A definition in a later pack
//! replaces the definition of the same kind and key from an earlier pack as a whole; fields
//! are not merged.
//!
//! ## `pack.toml`
//!
//! ```toml
//! format = 1                # pack format version; this engine reads 1
//! name = "base"             # a-z, 0-9, _
//! version = "0.1.0"
//! description = "…"         # optional
//! ```
//!
//! ## `[[texture]]`
//!
//! ```toml
//! [[texture]]
//! key = "cubeplanet:ore_copper"
//! provenance = "own"        # "own" | "placeholder"
//! procedural = { recipe = "ore", seed = 101, params = { ore = "#d27a4c", style = "nuggets" } }
//! # or, instead of `procedural`:  file = "textures/ore_copper.png"
//! # licence = "…"           # required for placeholders, optional for own art
//! ```
//!
//! `procedural` names a `cp-proctex` recipe. `params` is optional. A parameter is a string:
//! one that starts with `#` is a colour (`"#rrggbb"` or `"#rgb"`), any other names a variant
//! of the recipe ([`ParamValue`]). This crate does not depend on `cp-proctex`: whether a
//! recipe exists and knows its parameters is checked where the textures are generated.
//! `file` is a path relative to the pack directory, with `/` as separator; the file must
//! exist, spelled exactly as on disk, upper and lower case included. So that a pack loads on
//! every system, a name may not hold control characters or any of `<>:"\|?*`, end in a dot
//! or a space, or be a Windows device name (`con`, `nul`, `com1`, … with any extension).
//!
//! Pack files may use LF or CRLF line endings; strings are read with LF either way.
//!
//! ## `[[material]]`
//!
//! ```toml
//! [[material]]
//! key = "cubeplanet:leaves_oak"
//! texture = "cubeplanet:leaves_oak"
//! roughness = 1.0           # 0..1, default 1: factor on the texture's roughness
//! metallic = 0.0            # 0..1, default 0
//! emissive = 0.0            # >= 0, default 0: factor on the texture's emission
//! tint = "foliage"          # "none" (default) | "grass" | "foliage" | "water"
//! flags = ["cutout"]        # any of "cutout", "translucent"; default none
//! ```
//!
//! Textures carry roughness and emission per texel, so a material only scales them; most
//! materials are no more than a name for a texture. A tinted material takes on the grass,
//! foliage or water colour of its surroundings and expects a texture painted in near-neutral
//! tones.
//!
//! ## `[[block]]`
//!
//! ```toml
//! [[block]]
//! key = "cubeplanet:log_oak"
//! orient = "axis3"          # "none" (default) | "up6" | "axis3" | "rot24"
//! render = "cube"           # "cube" (default) | "cutout" | "fluid" | "model" | "none"
//! solid = true              # default: true for cube, cutout and model
//! opaque = true             # default: true for cube
//! structural = true         # default false; only for solid blocks
//! hardness = 3.0            # >= 0; inf = cannot be mined
//! density = 750.0           # kg/m³, >= 0
//! light = 0                 # 0..15, default 0
//! tags = ["log", "wood"]    # default none
//! # cull_group = "leaves"   # default none; a-z, 0-9, _
//! # lod_proxy = "ns:b[…]"   # model blocks only
//! [block.faces]             # or inline: faces = { all = "cubeplanet:dirt" }
//! top = "cubeplanet:log_oak_top"
//! side = "cubeplanet:log_oak_side"
//! bottom = "cubeplanet:log_oak_top"
//! [block.ext.forestry]      # any number of [block.ext.<namespace>] tables
//! burn_time = 300
//! ```
//!
//! `faces` gives one material per face of the *unrotated* block, in one of three forms:
//! `all`; or `top`, `side` and `bottom`; or all six of `"+x"`, `"-x"`, `"+y"`, `"-y"`, `"+z"`,
//! `"-z"` (`+y` is the top, `+z` the front). Blocks with `render = "none"` or `"model"` have
//! no faces: the first are not drawn, the second will be drawn by the instanced model path,
//! which does not exist yet. Only a `cube` block may be `opaque`, and then none of its
//! materials may be `translucent`: an opaque block hides its neighbours' faces, so it has to
//! cover them. `cutout` materials belong to `cutout` blocks: only their layer cuts holes, so a
//! `cube` or `fluid` block cannot use one. A `fluid` block is never `solid`.
//!
//! The flags of a block's states follow from these fields ([`BlockDef::flags`]): `SOLID`,
//! `OPAQUE` and `STRUCTURAL` from the fields of the same name, `EMISSIVE` from `light > 0`,
//! `CUTOUT` and `FLUID` from `render`. `ENTITY` is reserved.
//!
//! **Cull groups.** A face between two cells of the same cull group is not drawn (leaves next
//! to leaves, water next to water). Groups are numbered by sorted name from 1, 0 is no group
//! ([`BlockTables::cull_group`], [`Registry::cull_groups`]), so the numbers do not depend on
//! the order of packs. Only see-through blocks that are meshed (`cutout`, `fluid`, `cube` that
//! is not `opaque`) can have one.
//!
//! **LOD.** In coarser levels every state counts as solid (solid `cube` and `cutout` blocks),
//! fluid (`fluid` blocks) or empty (everything else), and is drawn as itself. A `model` block
//! is not cube-shaped: it names the cube-shaped state that stands in for its default state
//! (`lod_proxy`, a state key of a `cube`, `cutout` or `fluid` block) and then counts as that
//! state's class, or vanishes without one ([`BlockTables::lod`]). A rotated state of the model
//! gets the stand-in turned the same way ([`Registry::rotated`]): a shaft along `+y` with
//! `lod_proxy = "ns:log"` is a log along `x` when it is turned to lie along `x`.
//!
//! **Extensions.** `[block.ext.<namespace>]` tables hold data for code outside the engine.
//! They are stored as written ([`BlockDef::ext`]) and only their namespaces are checked
//! (`a-z`, `0-9`, `_`); they count for the content hash.
//!
//! # Block states
//!
//! `orient` decides how many states a block has ([`Orient`]). A state key is the block key
//! followed by the state's properties, sorted by name, with only the values written that
//! differ from their defaults ([`StateKey`]); the bare key is the default state, the unrotated
//! block:
//!
//! | `orient` | states of a block `ns:b` |
//! |---|---|
//! | `none` | `ns:b` |
//! | `up6` | `ns:b` (`up=+y`), `ns:b[up=+x]`, `[up=-x]`, `[up=-y]`, `[up=+z]`, `[up=-z]` |
//! | `axis3` | `ns:b` (`axis=y`), `ns:b[axis=x]`, `ns:b[axis=z]` |
//! | `rot24` | `ns:b` (`turn=0`, `up=+y`), `ns:b[turn=1]`, …, `ns:b[up=-z]`, `ns:b[turn=3,up=-z]` |
//!
//! `up` is the world face the block's top points to and `turn` the quarter turns about it
//! ([`Orientation`]); `axis=x` and `axis=z` are the orientations with up `+X` and `+Z`. The
//! render table holds, for every state, the materials per **world** face and the `uv_rot` of
//! [`Orientation::uv_rot`], so the mesher needs to know nothing about orientation.
//! [`Registry::rotated`] turns a placed state.
//!
//! `cubeplanet:air` is the empty block. It exists in every registry, is always state 0 and can
//! only be defined as empty.
//!
//! [`BlockStateId`]: cp_core::ids::BlockStateId
//! [`MaterialId`]: cp_core::ids::MaterialId
//! [`TextureLayer`]: cp_core::ids::TextureLayer
//! [`Orientation`]: cp_core::face::Orientation
//! [`Orientation::uv_rot`]: cp_core::face::Orientation::uv_rot

mod block;
mod builtin;
mod canon;
mod error;
mod file_path;
mod key;
mod loader;
mod material;
mod pack;
mod registry;
mod schema;
mod tables;
mod texture;

pub use block::{BlockDef, Orient};
pub use error::{ContentError, DefKind};
pub use key::{KeyError, StateKey};
pub use material::{MaterialDef, MaterialFlags, TintClass};
pub use pack::PackInfo;
pub use registry::Registry;
pub use tables::{BlockTables, DEFAULT_MATERIAL_COLOR, NO_MATERIAL_COLOR};
pub use texture::{ParamValue, Params, Provenance, TextureDef, TextureSource};
