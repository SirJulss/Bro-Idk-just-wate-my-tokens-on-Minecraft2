//! The per-state tables that storage, meshing and rendering consume instead of the registry
//! (contract §3.1, §4.4).

use std::sync::Arc;

use cp_core::block::{BlockFlags, BlockRender, LodInfo};

/// Linear RGB albedo of every drawn state until
/// [`Registry::set_material_colors`](crate::Registry::set_material_colors) provides the real
/// colours: the photographic middle grey, 18 % reflectance in every channel, about the albedo
/// of bare rock and soil, so an unpainted world still has a plausible brightness.
pub const DEFAULT_MATERIAL_COLOR: [f32; 3] = [0.18, 0.18, 0.18];

/// Linear RGB albedo of states that have no faces (`render = "none"` or `"model"`): black.
/// They are never meshed, so the value is never drawn.
pub const NO_MATERIAL_COLOR: [f32; 3] = [0.0, 0.0, 0.0];

/// One bundle of dense tables indexed by [`BlockStateId`](cp_core::ids::BlockStateId).
///
/// Every table has [`Registry::state_count`](crate::Registry::state_count) entries, and entry
/// 0 describes air: no flags, [`BlockRender::NONE`], cull group 0,
/// [`LodInfo::EMPTY`], [`NO_MATERIAL_COLOR`]. The tables are immutable; cloning the bundle
/// clones five `Arc`s, so workers can each hold one.
#[derive(Clone, Debug, PartialEq)]
pub struct BlockTables {
    /// Flags of each state ([`BlockDef::flags`](crate::BlockDef::flags)).
    pub flags: Arc<[BlockFlags]>,
    /// Materials and `uv_rot` per **world** face of each state, the state's orientation
    /// applied ([`BlockDef::render_for`](crate::BlockDef::render_for)).
    pub render: Arc<[BlockRender]>,
    /// Cull group of each state: 0 for none, otherwise `n` for the `n`-th name of
    /// [`Registry::cull_groups`](crate::Registry::cull_groups) (1-based, names sorted). A face
    /// between two cells of the same group is hidden.
    pub cull_group: Arc<[u16]>,
    /// Class and stand-in of each state in coarser LOD levels. Cube-shaped states stand in for
    /// themselves; `model` states for their block's `lod_proxy`, turned by the state's
    /// orientation like [`Registry::rotated`](crate::Registry::rotated) turns a state (the
    /// default state gets the proxy as written), or vanish ([`LodInfo`] with class `Empty`
    /// and proxy air) without one. The class is the proxy block's.
    pub lod: Arc<[LodInfo]>,
    /// Linear RGB albedo of each state: the average of its six face materials once
    /// [`Registry::set_material_colors`](crate::Registry::set_material_colors) has run,
    /// [`DEFAULT_MATERIAL_COLOR`] before; [`NO_MATERIAL_COLOR`] for states without faces.
    pub material_color: Arc<[[f32; 3]]>,
}
