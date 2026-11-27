//! Per-block-state tables consumed by storage, meshing and rendering (contract §4.1, §4.4).
//!
//! `cp-voxel` and `cp-mesh` never see the content registry; they get plain slices indexed by
//! [`BlockStateId`]: `&[BlockFlags]`, `&[BlockRender]`, `&[LodInfo]` (and the `u16` cull groups
//! of the registry; contract §3.1). This module defines the element types.

use bitflags::bitflags;
use serde::{Deserialize, Serialize};

use crate::face::{Face, Orientation};
use crate::ids::{BlockStateId, MaterialId};

bitflags! {
    /// Physical and visual properties of a block state, two bytes per state.
    ///
    /// Air has no flags. `CUTOUT` and `FLUID` mirror the state's [`RenderClass`]; the content
    /// registry derives them from it. Bits 7–15 are unused.
    #[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
    #[repr(transparent)]
    pub struct BlockFlags: u16 {
        /// Occupies its cell for collision and counts as solid when coarser LODs are derived.
        const SOLID = 1 << 0;
        /// Hides every face of a neighbouring cell that touches it (a full, opaque cube).
        const OPAQUE = 1 << 1;
        /// Drawn alpha-tested (leaves, grates): neighbours stay visible through it.
        const CUTOUT = 1 << 2;
        /// A fluid (water, lava): not solid, drawn in the fluid layer.
        const FLUID = 1 << 3;
        /// Emits light.
        const EMISSIVE = 1 << 4;
        /// Reserved: the cell is backed by an entity with its own state (machines, containers).
        const ENTITY = 1 << 5;
        /// Kept conservatively when coarser LODs are derived, so thin structures (logs,
        /// walls) do not vanish at a distance (downsample rule, contract §5.1).
        const STRUCTURAL = 1 << 6;
    }
}

/// How the mesher treats a block state.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[repr(u8)]
pub enum RenderClass {
    /// Not drawn (air).
    #[default]
    None = 0,
    /// Opaque cube faces.
    Cube = 1,
    /// Alpha-tested cube faces.
    Cutout = 2,
    /// Fluid surface faces.
    Fluid = 3,
    /// Not a full cube or animated (machines, stairs, levers): drawn by the instanced model path
    /// (contract §1). Reserved: the mesher emits nothing for it yet.
    Model = 4,
}

/// What one face of a block state looks like: the two values the mesher copies into every quad
/// of that face. Quads merge only when both are equal.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub struct FaceAppearance {
    /// Surface material of the face.
    pub material: MaterialId,
    /// Texture rotation in quarter turns, `0..4` (see [`Face::uv_to_tex`]).
    pub uv_rot: u8,
}

/// Render description of a block state: its class and the appearance of its six faces.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub struct BlockRender {
    /// Which mesh layer the state goes to, if any.
    pub class: RenderClass,
    /// Appearance per **world** face, indexed by [`Face::index`]. The block's orientation is
    /// already applied.
    pub faces: [FaceAppearance; 6],
}

impl BlockRender {
    /// The description of a state that is not drawn (air).
    pub const NONE: Self = Self {
        class: RenderClass::None,
        faces: [FaceAppearance {
            material: MaterialId(0),
            uv_rot: 0,
        }; 6],
    };

    /// Builds the world-face table of a block placed with `orientation` (any of the 24).
    ///
    /// `local_materials` is indexed by the face of the *unrotated* block
    /// ([`Face::index`]: `+X -X +Y -Y +Z -Z`, where `+Y` is the block's top and `+Z` its front).
    /// Each world face gets the material of the local face that ends up there and the `uv_rot`
    /// that keeps that face's texture attached to the block ([`Orientation::uv_rot`]).
    #[must_use]
    pub fn oriented(
        class: RenderClass,
        local_materials: [MaterialId; 6],
        orientation: Orientation,
    ) -> Self {
        let faces = Face::ALL.map(|world| FaceAppearance {
            material: local_materials[orientation.local_face(world).index()],
            uv_rot: orientation.uv_rot(world),
        });
        Self { class, faces }
    }

    /// Appearance of the world face `face`.
    #[inline]
    #[must_use]
    pub const fn face(&self, face: Face) -> FaceAppearance {
        self.faces[face.index()]
    }
}

/// What a block state counts as when coarser LOD levels are derived (downsample rule,
/// contract §5.1).
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug, Default)]
#[repr(u8)]
pub enum LodClass {
    /// Air and everything that vanishes at a distance.
    #[default]
    Empty = 0,
    /// Water, lava: counts towards a fluid parent cell.
    Fluid = 1,
    /// Opaque and cutout solids: counts towards a solid parent cell.
    Solid = 2,
}

/// The LOD description of a block state: its class and the state that stands in for it in
/// coarser levels (`proxy`: the state itself for cube-shaped states, a cube-shaped stand-in
/// for others).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub struct LodInfo {
    /// Class counted by the downsample rule.
    pub class: LodClass,
    /// State written into coarser levels in place of this one.
    pub proxy: BlockStateId,
}

impl LodInfo {
    /// The description of air: [`LodClass::Empty`], proxy [`BlockStateId::AIR`].
    pub const EMPTY: Self = Self {
        class: LodClass::Empty,
        proxy: BlockStateId::AIR,
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flags_are_distinct_bits_of_two_bytes() {
        assert_eq!(size_of::<BlockFlags>(), 2);
        let all = [
            BlockFlags::SOLID,
            BlockFlags::OPAQUE,
            BlockFlags::CUTOUT,
            BlockFlags::FLUID,
            BlockFlags::EMISSIVE,
            BlockFlags::ENTITY,
            BlockFlags::STRUCTURAL,
        ];
        let mut union = BlockFlags::empty();
        for (bit, flag) in all.into_iter().enumerate() {
            assert_eq!(flag.bits(), 1 << bit);
            assert!(!union.intersects(flag));
            union |= flag;
        }
        assert_eq!(union, BlockFlags::all());
        assert_eq!(union.bits(), 0b111_1111);
        assert_eq!(BlockFlags::default(), BlockFlags::empty());
        let stone = BlockFlags::SOLID | BlockFlags::OPAQUE;
        assert!(stone.contains(BlockFlags::OPAQUE) && !stone.contains(BlockFlags::FLUID));
        assert_eq!(BlockFlags::from_bits(1 << 7), None);
        assert_eq!(BlockFlags::from_bits(1 << 15), None);
        assert_eq!(
            BlockFlags::from_bits_truncate(0xFFFF),
            BlockFlags::all(),
            "bits 7-15 are not flags"
        );
    }

    #[test]
    fn defaults_describe_air() {
        assert_eq!(RenderClass::default(), RenderClass::None);
        assert_eq!(BlockRender::default(), BlockRender::NONE);
        assert_eq!(
            BlockRender::NONE.face(Face::NegZ),
            FaceAppearance::default()
        );
        assert_eq!(size_of::<RenderClass>(), 1);
        assert_eq!(
            [
                RenderClass::None,
                RenderClass::Cube,
                RenderClass::Cutout,
                RenderClass::Fluid,
                RenderClass::Model
            ]
            .map(|class| class as u8),
            [0, 1, 2, 3, 4]
        );
        assert_eq!(LodInfo::default(), LodInfo::EMPTY);
        assert_eq!(LodClass::default(), LodClass::Empty);
        assert_eq!(LodInfo::EMPTY.proxy, BlockStateId::AIR);
        assert_eq!(
            [LodClass::Empty, LodClass::Fluid, LodClass::Solid].map(|class| class as u8),
            [0, 1, 2]
        );
        assert_eq!(size_of::<LodClass>(), 1);
    }

    #[test]
    fn oriented_blocks_carry_their_faces_along() {
        // A machine-like block: distinct top, bottom, front, back and two distinct sides.
        let local = [10, 11, 20, 30, 12, 13].map(MaterialId);
        let unrotated = BlockRender::oriented(RenderClass::Cube, local, Orientation::IDENTITY);
        assert_eq!(unrotated.class, RenderClass::Cube);
        for face in Face::ALL {
            assert_eq!(unrotated.face(face).material, local[face.index()]);
            assert_eq!(
                unrotated.face(face).uv_rot,
                Orientation::IDENTITY.uv_rot(face)
            );
        }
        for orientation in Orientation::ALL {
            let render = BlockRender::oriented(RenderClass::Cutout, local, orientation);
            let up = orientation.up();
            // The top material shows on the up face, the bottom material opposite to it, the
            // front material on the front face.
            assert_eq!(render.face(up).material, MaterialId(20));
            assert_eq!(render.face(up.opposite()).material, MaterialId(30));
            assert_eq!(render.face(orientation.front()).material, MaterialId(12));
            for local_face in Face::ALL {
                let world = orientation.world_face(local_face);
                assert_eq!(
                    render.faces[world.index()].material,
                    local[local_face.index()]
                );
                assert_eq!(
                    render.faces[world.index()].uv_rot,
                    orientation.uv_rot(world)
                );
            }
            // All six materials are still present exactly once.
            let mut materials: Vec<u16> = render.faces.iter().map(|face| face.material.0).collect();
            materials.sort_unstable();
            assert_eq!(materials, [10, 11, 12, 13, 20, 30]);
        }
    }
}
