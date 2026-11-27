//! Block definitions and the states they expand to.
//!
//! A block is a kind of thing (`cubeplanet:grass`); a block *state* is a block together with
//! the values of its state properties (`cubeplanet:grass[up=+x]`). Chunks store states, never
//! blocks. The only state properties so far are those of the orientation, chosen by [`Orient`].

use std::collections::BTreeMap;

use cp_core::block::{BlockFlags, BlockRender, RenderClass};
use cp_core::face::{Face, Orientation};
use cp_core::ids::{BlockStateId, MaterialId};
use serde::Deserialize;

/// How a block may be rotated when it is placed (contract §4.4). Every possible rotation is one
/// block state.
///
/// | TOML | states | properties of a state key (default in bold) |
/// |---|---|---|
/// | `"none"` | 1 | – |
/// | `"up6"` | 6 | `up` ∈ `+x -x` **`+y`** `-y +z -z` |
/// | `"axis3"` | 3 | `axis` ∈ `x` **`y`** `z` |
/// | `"rot24"` | 24 | `turn` ∈ **`0`** `1 2 3`, `up` ∈ `+x -x` **`+y`** `-y +z -z` |
///
/// Only values that differ from the default are written, sorted by property name:
/// `ns:b`, `ns:b[up=+x]`, `ns:b[axis=z]`, `ns:b[turn=2]`, `ns:b[turn=1,up=-z]`. The bare key
/// always names the default state, the unrotated block ([`Orientation::IDENTITY`]).
///
/// `up` is the world face the block's top (local `+Y`) points to and `turn` the quarter turns
/// about it ([`Orientation`]); a block's front is its local `+Z` face. `axis=a` is the
/// orientation `up = +a`, `turn = 0`.
///
/// The discriminants are part of [`Registry::content_hash`](crate::Registry::content_hash)
/// and must not be renumbered.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default, Deserialize)]
#[serde(rename_all = "snake_case")]
#[repr(u8)]
pub enum Orient {
    /// One state; the block is never rotated.
    #[default]
    None = 0,
    /// Six states, property `up`: the block's top points to any of the six faces, never
    /// turned about it. For blocks with a distinct top and bottom, such as grass on a cube
    /// planet.
    Up6 = 1,
    /// Three states, property `axis`: the block's local Y axis lies along the X, Y or Z axis.
    /// For blocks whose two ends look alike, such as logs.
    Axis3 = 2,
    /// All 24 rotations, properties `turn` and `up`. For blocks with a front, such as machines
    /// that must face any direction on any face of a planet.
    Rot24 = 3,
}

/// Property values of `up`, by [`Face::index`].
const UP_VALUES: [&str; 6] = ["+x", "-x", "+y", "-y", "+z", "-z"];
/// Property values of `axis`, by axis.
const AXIS_VALUES: [&str; 3] = ["x", "y", "z"];
/// Property values of `turn`.
const TURN_VALUES: [&str; 4] = ["0", "1", "2", "3"];

const NONE: [Orientation; 1] = [Orientation::IDENTITY];
/// By [`Face::index`] of the up face: `Orientation::ALL[..6]`.
const UP6: [Orientation; 6] = [
    Orientation::from_up(Face::PosX),
    Orientation::from_up(Face::NegX),
    Orientation::from_up(Face::PosY),
    Orientation::from_up(Face::NegY),
    Orientation::from_up(Face::PosZ),
    Orientation::from_up(Face::NegZ),
];
/// By axis.
const AXIS3: [Orientation; 3] = [
    Orientation::from_up(Face::PosX),
    Orientation::from_up(Face::PosY),
    Orientation::from_up(Face::PosZ),
];

impl Orient {
    /// Every kind, in the order of the discriminants.
    pub const ALL: [Self; 4] = [Self::None, Self::Up6, Self::Axis3, Self::Rot24];

    /// Number of states of a block of this kind: 1, 6, 3 or 24.
    #[must_use]
    pub const fn state_count(self) -> usize {
        self.orientations().len()
    }

    /// The orientation of every state of a block of this kind, in the order of
    /// [`BlockDef::states`]: the identity; by up face ([`Face::index`]); by axis; by
    /// [`Orientation::index`].
    #[must_use]
    pub const fn orientations(self) -> &'static [Orientation] {
        match self {
            Self::None => &NONE,
            Self::Up6 => &UP6,
            Self::Axis3 => &AXIS3,
            Self::Rot24 => &Orientation::ALL,
        }
    }

    /// Position in [`Orient::orientations`] of the state that `orientation` is projected onto:
    /// `none` has only its one state, `up6` keeps the up face and drops the turn, `axis3` keeps
    /// the axis of the up face, `rot24` keeps everything.
    #[must_use]
    pub const fn variant_index(self, orientation: Orientation) -> usize {
        match self {
            Self::None => 0,
            Self::Up6 => orientation.up().index(),
            Self::Axis3 => orientation.up().axis(),
            Self::Rot24 => orientation.index(),
        }
    }

    /// `orientation` projected onto the orientations this kind can express (see
    /// [`Orient::variant_index`]).
    #[must_use]
    pub const fn project(self, orientation: Orientation) -> Orientation {
        self.orientations()[self.variant_index(orientation)]
    }

    /// The properties written in the key of the state with `orientation`, one of
    /// [`Orient::orientations`]: sorted by name, only those that differ from their default.
    pub(crate) fn properties(self, orientation: Orientation) -> Vec<(&'static str, &'static str)> {
        let up = orientation.up();
        let mut properties = Vec::new();
        match self {
            Self::None => {}
            Self::Up6 => {
                if up != Face::PosY {
                    properties.push(("up", UP_VALUES[up.index()]));
                }
            }
            Self::Axis3 => {
                if up.axis() != 1 {
                    properties.push(("axis", AXIS_VALUES[up.axis()]));
                }
            }
            Self::Rot24 => {
                if orientation.turn() != 0 {
                    properties.push(("turn", TURN_VALUES[usize::from(orientation.turn())]));
                }
                if up != Face::PosY {
                    properties.push(("up", UP_VALUES[up.index()]));
                }
            }
        }
        properties
    }

    /// The position in [`Orient::orientations`] of the state whose key writes exactly
    /// `properties` (as parsed by [`StateKey`](crate::StateKey): sorted, each named once).
    /// `None` when a block of this kind has no such state: an unknown property, an unknown
    /// value, or a default value written out.
    pub(crate) fn resolve(self, properties: &[(&str, &str)]) -> Option<usize> {
        let mut up = Face::PosY;
        let mut turn = 0;
        let mut axis = 1;
        for &(name, value) in properties {
            match (self, name) {
                (Self::Up6 | Self::Rot24, "up") => {
                    let index = UP_VALUES.iter().position(|&known| known == value)?;
                    up = Face::ALL[index];
                    if up == Face::PosY {
                        return None;
                    }
                }
                (Self::Rot24, "turn") => {
                    turn = TURN_VALUES.iter().position(|&known| known == value)?;
                    if turn == 0 {
                        return None;
                    }
                }
                (Self::Axis3, "axis") => {
                    axis = AXIS_VALUES.iter().position(|&known| known == value)?;
                    if axis == 1 {
                        return None;
                    }
                }
                _ => return None,
            }
        }
        let orientation = match self {
            Self::None => Orientation::IDENTITY,
            Self::Up6 => Orientation::from_up(up),
            Self::Axis3 => Orientation::from_up(Face::from_axis(axis, false)),
            // `turn` is one of the four values of `TURN_VALUES`.
            Self::Rot24 => Orientation::new(up, turn as u8),
        };
        Some(self.variant_index(orientation))
    }
}

/// A block as defined by a content pack, with its references resolved.
///
/// Obtained from [`Registry::blocks`](crate::Registry::blocks), where the blocks are sorted by
/// key. Ids are valid for the registry that handed them out and for one session only.
#[derive(Clone, Debug, PartialEq)]
pub struct BlockDef {
    /// The block's key, `namespace:name`. Also the key of its default state.
    pub key: String,
    /// How the block may be rotated; determines its states.
    pub orient: Orient,
    /// Occupies its cell for collision and counts as solid when coarser LODs are derived.
    pub solid: bool,
    /// A full opaque cube: hides every neighbouring face that touches it.
    pub opaque: bool,
    /// Kept conservatively when coarser LODs are derived, so that thin structures (logs) do
    /// not vanish at a distance. Only solid blocks are structural.
    pub structural: bool,
    /// How the block is drawn, if at all.
    pub render: RenderClass,
    /// Materials of the six faces of the *unrotated* block, indexed by [`Face::index`]
    /// (`+X -X +Y -Y +Z -Z`; `+Y` is the top, `+Z` the front). `None` exactly when `render`
    /// is [`RenderClass::None`] or [`RenderClass::Model`].
    pub faces: Option<[MaterialId; 6]>,
    /// Resistance to mining, dimensionless and `>= 0`: mining time is proportional to it.
    /// `f32::INFINITY` means the block cannot be mined. Blocks that are never mined (air,
    /// fluids) carry 0.
    pub hardness: f32,
    /// Mass per cell volume in kg/m³, finite and `>= 0`.
    pub density: f32,
    /// Light emission level, `0..=15`; 0 emits nothing.
    pub light: u8,
    /// Free-form labels for gameplay and world generation (`"rock"`, `"log"`, …), sorted and
    /// without duplicates.
    pub tags: Vec<String>,
    /// Name of the block's cull group: faces between two cells of the same group are not
    /// drawn (leaves next to leaves, water next to water). Numbered in
    /// [`BlockTables::cull_group`](crate::BlockTables::cull_group).
    pub cull_group: Option<String>,
    /// For blocks that are not cube-shaped (`render = "model"`): the key of the cube-shaped
    /// state that stands in for the block's default state in coarser LOD levels. A rotated
    /// state of the block gets this state turned the same way
    /// ([`BlockTables::lod`](crate::BlockTables::lod)).
    pub lod_proxy: Option<String>,
    /// `[ext.<namespace>]` tables, by namespace: data for code outside the engine, stored as
    /// written and never interpreted here.
    pub ext: BTreeMap<String, toml::Table>,
    /// State ids in the order of [`Orient::orientations`]. Filled in by the registry.
    pub(crate) states: Vec<BlockStateId>,
}

impl BlockDef {
    /// The ids of all states of this block, in the order of [`Orient::orientations`]: one for
    /// [`Orient::None`], six by the [`Face::index`] of the up face for [`Orient::Up6`], three by
    /// axis for [`Orient::Axis3`], 24 by [`Orientation::index`] for [`Orient::Rot24`].
    #[must_use]
    pub fn states(&self) -> &[BlockStateId] {
        &self.states
    }

    /// The unrotated state, named by the bare block key.
    #[must_use]
    pub fn default_state(&self) -> BlockStateId {
        self.state_for(Orientation::IDENTITY)
    }

    /// The state to place for a block rotated by `orientation`, as far as the block can be
    /// rotated ([`Orient::project`]).
    #[must_use]
    pub fn state_for(&self, orientation: Orientation) -> BlockStateId {
        self.states[self.orient.variant_index(orientation)]
    }

    /// The state to place so that the block's top points to `up`, unturned: exactly that state
    /// for [`Orient::Up6`] and [`Orient::Rot24`], the state of `up`'s axis for
    /// [`Orient::Axis3`], the only state for [`Orient::None`].
    ///
    /// On a cube planet, `up` is the face of the planet the block is placed on.
    #[must_use]
    pub fn state_for_up(&self, up: Face) -> BlockStateId {
        self.state_for(Orientation::from_up(up))
    }

    /// The flags shared by all states of this block (contract §4.4): `SOLID`, `OPAQUE` and
    /// `STRUCTURAL` from the fields of the same name, `EMISSIVE` from `light > 0`, `CUTOUT` and
    /// `FLUID` from the render class. `ENTITY` is reserved and never set.
    #[must_use]
    pub fn flags(&self) -> BlockFlags {
        let mut flags = BlockFlags::empty();
        flags.set(BlockFlags::SOLID, self.solid);
        flags.set(BlockFlags::OPAQUE, self.opaque);
        flags.set(BlockFlags::STRUCTURAL, self.structural);
        flags.set(BlockFlags::EMISSIVE, self.light > 0);
        match self.render {
            RenderClass::None | RenderClass::Cube | RenderClass::Model => {}
            RenderClass::Cutout => flags |= BlockFlags::CUTOUT,
            RenderClass::Fluid => flags |= BlockFlags::FLUID,
        }
        flags
    }

    /// The render description of this block rotated by `orientation`: per **world** face the
    /// material of the local face that ends up there and the `uv_rot` that keeps its texture
    /// attached to the block. Blocks without faces keep the materials of
    /// [`BlockRender::NONE`].
    #[must_use]
    pub fn render_for(&self, orientation: Orientation) -> BlockRender {
        match self.faces {
            Some(local) => BlockRender::oriented(self.render, local, orientation),
            None => BlockRender {
                class: self.render,
                ..BlockRender::NONE
            },
        }
    }

    /// True when the block carries the tag `tag`.
    #[must_use]
    pub fn has_tag(&self, tag: &str) -> bool {
        self.tags
            .binary_search_by(|own| own.as_str().cmp(tag))
            .is_ok()
    }

    /// True when the block cannot be mined (`hardness` is infinite).
    #[must_use]
    pub fn is_unbreakable(&self) -> bool {
        self.hardness.is_infinite()
    }

    /// The `[ext.<namespace>]` table of the block, if it has one.
    #[must_use]
    pub fn ext(&self, namespace: &str) -> Option<&toml::Table> {
        self.ext.get(namespace)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::key::format_state_key;

    fn keys(orient: Orient) -> Vec<String> {
        orient
            .orientations()
            .iter()
            .map(|&orientation| format_state_key("ns:b", &orient.properties(orientation)))
            .collect()
    }

    #[test]
    fn state_keys_per_kind() {
        assert_eq!(keys(Orient::None), ["ns:b"]);
        assert_eq!(
            keys(Orient::Up6),
            [
                "ns:b[up=+x]",
                "ns:b[up=-x]",
                "ns:b",
                "ns:b[up=-y]",
                "ns:b[up=+z]",
                "ns:b[up=-z]"
            ]
        );
        assert_eq!(
            keys(Orient::Axis3),
            ["ns:b[axis=x]", "ns:b", "ns:b[axis=z]"]
        );
        let rot24 = keys(Orient::Rot24);
        assert_eq!(
            rot24[..12],
            [
                "ns:b[up=+x]",
                "ns:b[up=-x]",
                "ns:b",
                "ns:b[up=-y]",
                "ns:b[up=+z]",
                "ns:b[up=-z]",
                "ns:b[turn=1,up=+x]",
                "ns:b[turn=1,up=-x]",
                "ns:b[turn=1]",
                "ns:b[turn=1,up=-y]",
                "ns:b[turn=1,up=+z]",
                "ns:b[turn=1,up=-z]",
            ]
        );
        assert_eq!(rot24[20], "ns:b[turn=3]");
        assert_eq!(rot24[23], "ns:b[turn=3,up=-z]");

        for orient in Orient::ALL {
            let keys = keys(orient);
            assert_eq!(keys.len(), orient.state_count());
            let mut unique = keys.clone();
            unique.sort();
            unique.dedup();
            assert_eq!(unique.len(), keys.len(), "{orient:?}: keys are distinct");
            // Exactly one state is the default one, and it is the unrotated block.
            let defaults: Vec<_> = orient
                .orientations()
                .iter()
                .filter(|&&orientation| orient.properties(orientation).is_empty())
                .collect();
            assert_eq!(defaults, [&Orientation::IDENTITY], "{orient:?}");
        }
    }

    #[test]
    fn orientations_are_indexed_as_documented() {
        assert_eq!(UP6, Orientation::ALL[..6]);
        assert_eq!(Orient::Rot24.orientations(), Orientation::ALL);
        for face in Face::ALL {
            assert_eq!(UP6[face.index()], Orientation::from_up(face));
            assert_eq!(AXIS3[face.axis()].up().axis(), face.axis());
            assert!(!AXIS3[face.axis()].up().is_negative());
        }
        for orient in Orient::ALL {
            for (index, &orientation) in orient.orientations().iter().enumerate() {
                assert_eq!(orient.variant_index(orientation), index, "{orient:?}");
                assert_eq!(orient.project(orientation), orientation, "{orient:?}");
            }
        }
    }

    #[test]
    fn projection_drops_what_a_kind_cannot_express() {
        for orientation in Orientation::ALL {
            assert_eq!(Orient::None.project(orientation), Orientation::IDENTITY);
            let up6 = Orient::Up6.project(orientation);
            assert_eq!((up6.up(), up6.turn()), (orientation.up(), 0));
            let axis3 = Orient::Axis3.project(orientation);
            assert_eq!(axis3.turn(), 0);
            assert_eq!(axis3.up(), Face::from_axis(orientation.up().axis(), false));
            assert_eq!(Orient::Rot24.project(orientation), orientation);
        }
    }

    #[test]
    fn properties_resolve_to_their_state() {
        for orient in Orient::ALL {
            for (index, &orientation) in orient.orientations().iter().enumerate() {
                let properties = orient.properties(orientation);
                assert_eq!(orient.resolve(&properties), Some(index), "{properties:?}");
            }
        }
        let resolve = |orient: Orient, properties: &[(&str, &str)]| orient.resolve(properties);
        // Default values are never written.
        assert_eq!(resolve(Orient::Up6, &[("up", "+y")]), None);
        assert_eq!(resolve(Orient::Axis3, &[("axis", "y")]), None);
        assert_eq!(resolve(Orient::Rot24, &[("turn", "0")]), None);
        assert_eq!(resolve(Orient::Rot24, &[("turn", "0"), ("up", "+x")]), None);
        assert_eq!(resolve(Orient::Rot24, &[("turn", "1"), ("up", "+y")]), None);
        // Properties of another kind, unknown values.
        assert_eq!(resolve(Orient::None, &[("up", "+x")]), None);
        assert_eq!(resolve(Orient::Up6, &[("axis", "x")]), None);
        assert_eq!(resolve(Orient::Up6, &[("turn", "1")]), None);
        assert_eq!(resolve(Orient::Axis3, &[("up", "+x")]), None);
        assert_eq!(resolve(Orient::Axis3, &[("axis", "+x")]), None);
        assert_eq!(resolve(Orient::Up6, &[("up", "x")]), None);
        assert_eq!(resolve(Orient::Rot24, &[("turn", "4")]), None);
        assert_eq!(resolve(Orient::Rot24, &[("turn", "01")]), None);
        assert_eq!(resolve(Orient::Rot24, &[("lit", "1")]), None);
        // The bare key is the default state of every kind.
        for orient in Orient::ALL {
            let default = orient.variant_index(Orientation::IDENTITY);
            assert_eq!(resolve(orient, &[]), Some(default), "{orient:?}");
        }
    }

    #[test]
    fn orient_parses_from_its_toml_names() {
        #[derive(Deserialize)]
        struct Holder {
            orient: Orient,
        }
        for (text, expected) in [
            ("none", Orient::None),
            ("up6", Orient::Up6),
            ("axis3", Orient::Axis3),
            ("rot24", Orient::Rot24),
        ] {
            let holder: Holder = toml::from_str(&format!("orient = \"{text}\"")).unwrap();
            assert_eq!(holder.orient, expected);
        }
        assert!(toml::from_str::<Holder>("orient = \"up4\"").is_err());
        assert_eq!(Orient::default(), Orient::None);
        let numbers: Vec<u8> = Orient::ALL.iter().map(|&orient| orient as u8).collect();
        assert_eq!(numbers, [0, 1, 2, 3]);
    }

    fn block(orient: Orient, render: RenderClass) -> BlockDef {
        let faces = matches!(
            render,
            RenderClass::Cube | RenderClass::Cutout | RenderClass::Fluid
        )
        .then(|| [10, 11, 20, 30, 12, 13].map(MaterialId));
        let states = (0..orient.state_count() as u32)
            .map(|index| BlockStateId(100 + index))
            .collect();
        BlockDef {
            key: "ns:b".to_owned(),
            orient,
            solid: true,
            opaque: render == RenderClass::Cube,
            structural: false,
            render,
            faces,
            hardness: 1.0,
            density: 1000.0,
            light: 0,
            tags: vec!["log".to_owned(), "wood".to_owned()],
            cull_group: None,
            lod_proxy: None,
            ext: BTreeMap::new(),
            states,
        }
    }

    #[test]
    fn states_are_found_by_orientation() {
        let fixed = block(Orient::None, RenderClass::Cube);
        let up6 = block(Orient::Up6, RenderClass::Cube);
        let axis3 = block(Orient::Axis3, RenderClass::Cube);
        let rot24 = block(Orient::Rot24, RenderClass::Cube);
        assert_eq!(fixed.default_state(), BlockStateId(100));
        assert_eq!(up6.default_state(), BlockStateId(100 + 2));
        assert_eq!(axis3.default_state(), BlockStateId(100 + 1));
        assert_eq!(rot24.default_state(), BlockStateId(100 + 2));
        for face in Face::ALL {
            assert_eq!(fixed.state_for_up(face), BlockStateId(100));
            let index = face.index() as u32;
            assert_eq!(up6.state_for_up(face), BlockStateId(100 + index));
            assert_eq!(rot24.state_for_up(face), BlockStateId(100 + index));
            let axis = face.axis() as u32;
            assert_eq!(axis3.state_for_up(face), BlockStateId(100 + axis));
        }
        for orientation in Orientation::ALL {
            let index = orientation.index() as u32;
            assert_eq!(rot24.state_for(orientation), BlockStateId(100 + index));
            let up = orientation.up().index() as u32;
            assert_eq!(up6.state_for(orientation), BlockStateId(100 + up));
        }
        assert_eq!(up6.states().len(), 6);
        assert_eq!(rot24.states().len(), 24);
    }

    #[test]
    fn flags_follow_the_definition() {
        let stone = block(Orient::None, RenderClass::Cube);
        assert_eq!(stone.flags(), BlockFlags::SOLID | BlockFlags::OPAQUE);

        let mut log = block(Orient::Axis3, RenderClass::Cube);
        log.structural = true;
        assert_eq!(
            log.flags(),
            BlockFlags::SOLID | BlockFlags::OPAQUE | BlockFlags::STRUCTURAL
        );

        let leaves = block(Orient::None, RenderClass::Cutout);
        assert_eq!(leaves.flags(), BlockFlags::SOLID | BlockFlags::CUTOUT);

        let mut lava = block(Orient::None, RenderClass::Fluid);
        lava.solid = false;
        lava.light = 15;
        assert_eq!(lava.flags(), BlockFlags::FLUID | BlockFlags::EMISSIVE);

        let machine = block(Orient::Rot24, RenderClass::Model);
        assert_eq!(machine.flags(), BlockFlags::SOLID);
        assert_eq!(
            machine.render_for(Orientation::ALL[7]),
            BlockRender {
                class: RenderClass::Model,
                ..BlockRender::NONE
            }
        );

        let mut air = block(Orient::None, RenderClass::None);
        air.solid = false;
        assert_eq!(air.flags(), BlockFlags::empty());
        assert_eq!(air.render_for(Orientation::IDENTITY), BlockRender::NONE);

        // `ENTITY` is reserved: no definition sets it.
        for render in [
            RenderClass::None,
            RenderClass::Cube,
            RenderClass::Cutout,
            RenderClass::Fluid,
            RenderClass::Model,
        ] {
            let mut any = block(Orient::None, render);
            any.structural = true;
            any.light = 1;
            assert!(!any.flags().contains(BlockFlags::ENTITY), "{render:?}");
        }
    }

    #[test]
    fn render_follows_the_orientation() {
        let machine = block(Orient::Rot24, RenderClass::Cube);
        let local = machine.faces.unwrap();
        for orientation in Orientation::ALL {
            let render = machine.render_for(orientation);
            assert_eq!(render.class, RenderClass::Cube);
            for face in Face::ALL {
                let world = orientation.world_face(face);
                assert_eq!(render.face(world).material, local[face.index()]);
            }
            assert_eq!(
                render.face(orientation.front()).material,
                local[Face::PosZ.index()]
            );
            for world in Face::ALL {
                assert_eq!(render.face(world).uv_rot, orientation.uv_rot(world));
            }
        }
    }

    #[test]
    fn tags_hardness_and_ext_helpers() {
        let mut log = block(Orient::Axis3, RenderClass::Cube);
        assert!(log.has_tag("log") && log.has_tag("wood"));
        assert!(!log.has_tag("lo") && !log.has_tag("rock") && !log.has_tag(""));
        assert!(!log.is_unbreakable());
        log.hardness = f32::INFINITY;
        assert!(log.is_unbreakable());

        assert!(log.ext("forestry").is_none());
        let table: toml::Table = toml::from_str("burn_time = 300").unwrap();
        log.ext.insert("forestry".to_owned(), table.clone());
        assert_eq!(log.ext("forestry"), Some(&table));
    }
}
