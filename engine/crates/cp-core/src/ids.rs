//! Typed identifiers.
//!
//! Thin newtypes over integers so that a block state cannot be passed where a material is
//! expected. All of them are `Copy + Eq + Ord + Hash + Debug` and serialisable. The integer
//! newtypes serialise as their bare number; they and [`EntityId`] are plain-old-data for GPU
//! buffers and chunk storage. [`FrameId`] and [`GridId`] are enums over them. The keys of a
//! grid's chunks and voxels, [`ChunkKey`] and [`CellAddress`], are defined in
//! [`coords`](crate::coords) next to [`LodKey`](crate::coords::LodKey) and re-exported here,
//! where contract §4.1 lists them; both paths name the same types.
//!
//! Ids from the content registry ([`BlockStateId`], [`MaterialId`], [`TextureLayer`]) are valid
//! for one session only; persistent data stores string keys (contract §4.4).

use core::fmt;

use bytemuck::{Pod, Zeroable};
use serde::{Deserialize, Serialize};

/// The chunk and voxel keys of a grid (contract §2.2), defined in [`crate::coords`].
pub use crate::coords::{CellAddress, ChunkKey};

macro_rules! id_newtype {
    ($(#[$doc:meta])* $name:ident($inner:ty)) => {
        $(#[$doc])*
        #[derive(
            Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug, Default,
            Serialize, Deserialize, Pod, Zeroable,
        )]
        #[repr(transparent)]
        #[serde(transparent)]
        pub struct $name(pub $inner);

        impl From<$inner> for $name {
            #[inline]
            fn from(raw: $inner) -> Self {
                Self(raw)
            }
        }

        impl From<$name> for $inner {
            #[inline]
            fn from(id: $name) -> Self {
                id.0
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                fmt::Display::fmt(&self.0, f)
            }
        }
    };
}

/// Adds `index()` for ids that address dense tables.
macro_rules! id_index {
    ($name:ident) => {
        impl $name {
            /// The id as a table index.
            #[inline]
            #[must_use]
            pub const fn index(self) -> usize {
                self.0 as usize
            }
        }
    };
}

id_newtype!(
    /// A block state: a block plus the values of its state properties (orientation, axis, …).
    ///
    /// Index into the registry's per-state tables (`cp_content::Registry::tables()`, a
    /// `BlockTables` bundle).
    BlockStateId(u32)
);
id_index!(BlockStateId);

impl BlockStateId {
    /// Air, the empty state. Always id 0 (contract §4.4).
    pub const AIR: Self = Self(0);

    /// "No override": the sentinel of sparse edit layers, whose cells are either `UNCHANGED`
    /// (the base world shows through) or an explicit state (contract §5.1). Reserved: never
    /// the id of a registry state, so it must never reach the per-state tables.
    pub const UNCHANGED: Self = Self(u32::MAX);

    /// True for [`BlockStateId::AIR`].
    #[inline]
    #[must_use]
    pub const fn is_air(self) -> bool {
        self.0 == 0
    }

    /// True for the [`BlockStateId::UNCHANGED`] sentinel.
    #[inline]
    #[must_use]
    pub const fn is_unchanged(self) -> bool {
        self.0 == u32::MAX
    }
}

id_newtype!(
    /// A surface material; packed into 16 bits of every quad (contract §4.1 `quad`).
    MaterialId(u16)
);
id_index!(MaterialId);

id_newtype!(
    /// A layer of the block texture array.
    TextureLayer(u16)
);
id_index!(TextureLayer);

id_newtype!(
    /// A celestial body (star, planet, moon). Moving constructs are entities ([`EntityId`]).
    BodyId(u32)
);
id_index!(BodyId);

id_newtype!(
    /// A star system. 64 bits so procedurally generated systems can derive it from their
    /// galaxy position.
    SystemId(u64)
);

/// A generational entity handle: `index` addresses a slot, `generation` tells reuses of that
/// slot apart, so a stale handle never aliases a newer entity.
///
/// Ordered by `index`, then `generation`.
#[derive(
    Clone,
    Copy,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    Hash,
    Debug,
    Default,
    Serialize,
    Deserialize,
    Pod,
    Zeroable,
)]
#[repr(C)]
pub struct EntityId {
    /// Slot index.
    pub index: u32,
    /// Incremented every time the slot is reused.
    pub generation: u32,
}

impl EntityId {
    /// Handle from slot index and generation.
    #[inline]
    #[must_use]
    pub const fn new(index: u32, generation: u32) -> Self {
        Self { index, generation }
    }
}

impl fmt::Display for EntityId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}v{}", self.index, self.generation)
    }
}

/// Names a reference frame (contract §2.2): the inertial frame of a system, the rotating,
/// body-fixed frame of a body, or the local frame of a moving construct.
///
/// Ordered `System < Body < Construct`, then by the inner id.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug, Serialize, Deserialize)]
pub enum FrameId {
    /// Inertial frame with its origin at the system barycentre.
    System(SystemId),
    /// Body-fixed rotating frame with its origin at the body (cube) centre.
    Body(BodyId),
    /// Local frame of a moving construct; its pose is given relative to its parent frame.
    Construct(EntityId),
}

/// Names a voxel grid (contract §2.2): the body-fixed grid of a body or the local grid of a
/// moving construct. Terrain, buildings and constructs share storage, meshes and draw path.
///
/// Ordered `Body < Construct`, then by the inner id.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug, Serialize, Deserialize)]
pub enum GridId {
    /// The voxel grid of a body.
    Body(BodyId),
    /// The voxel grid of a construct entity.
    Construct(EntityId),
}

impl GridId {
    /// The reference frame the grid's coordinates are given in.
    #[inline]
    #[must_use]
    pub const fn frame(self) -> FrameId {
        match self {
            Self::Body(body) => FrameId::Body(body),
            Self::Construct(entity) => FrameId::Construct(entity),
        }
    }
}

impl From<BodyId> for GridId {
    #[inline]
    fn from(body: BodyId) -> Self {
        Self::Body(body)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::{BTreeSet, HashSet};

    /// The trait set the contract promises for every id type.
    fn assert_id_traits<T>()
    where
        T: Copy + Eq + Ord + core::hash::Hash + fmt::Debug + Serialize + for<'de> Deserialize<'de>,
    {
    }

    #[test]
    fn every_id_has_the_contract_traits() {
        assert_id_traits::<BlockStateId>();
        assert_id_traits::<MaterialId>();
        assert_id_traits::<TextureLayer>();
        assert_id_traits::<BodyId>();
        assert_id_traits::<SystemId>();
        assert_id_traits::<EntityId>();
        assert_id_traits::<FrameId>();
        assert_id_traits::<GridId>();
        // The grid keys re-exported from `coords` have the same traits.
        assert_id_traits::<ChunkKey>();
        assert_id_traits::<CellAddress>();
    }

    #[test]
    fn air_is_zero_and_default() {
        assert_eq!(BlockStateId::AIR, BlockStateId(0));
        assert_eq!(BlockStateId::default(), BlockStateId::AIR);
        assert!(BlockStateId::AIR.is_air());
        assert!(!BlockStateId(1).is_air());
        // The edit sentinel is the largest id, distinct from air and from every real state.
        assert_eq!(BlockStateId::UNCHANGED, BlockStateId(u32::MAX));
        assert!(BlockStateId::UNCHANGED.is_unchanged());
        assert!(!BlockStateId::UNCHANGED.is_air());
        assert!(!BlockStateId::AIR.is_unchanged());
        assert!(!BlockStateId(u32::MAX - 1).is_unchanged());
        assert!(BlockStateId(u32::MAX - 1) < BlockStateId::UNCHANGED);
        assert_eq!(BlockStateId(77).index(), 77);
        assert_eq!(MaterialId(u16::MAX).index(), 65_535);
        assert_eq!(TextureLayer(3).index(), 3);
        assert_eq!(BodyId(u32::MAX).index(), u32::MAX as usize);
    }

    #[test]
    fn layout_is_transparent() {
        assert_eq!(size_of::<BlockStateId>(), 4);
        assert_eq!(size_of::<MaterialId>(), 2);
        assert_eq!(size_of::<TextureLayer>(), 2);
        assert_eq!(size_of::<BodyId>(), 4);
        assert_eq!(size_of::<SystemId>(), 8);
        assert_eq!(size_of::<EntityId>(), 8);
        assert_eq!(align_of::<EntityId>(), 4);
        // Pod: slices of ids are slices of their integers and the other way round.
        let states = [BlockStateId(1), BlockStateId(0x0102_0304)];
        assert_eq!(bytemuck::cast_slice::<_, u32>(&states), &[1, 0x0102_0304]);
        let entity = EntityId::new(7, 9);
        assert_eq!(bytemuck::cast::<_, [u32; 2]>(entity), [7, 9]);
    }

    #[test]
    fn ordering_hashing_and_conversions() {
        let mut sorted = BTreeSet::new();
        sorted.extend([BlockStateId(5), BlockStateId(1), BlockStateId(3)]);
        assert_eq!(
            sorted.into_iter().collect::<Vec<_>>(),
            [BlockStateId(1), BlockStateId(3), BlockStateId(5)]
        );
        assert!(EntityId::new(1, 9) < EntityId::new(2, 0));
        assert!(EntityId::new(2, 0) < EntityId::new(2, 1));
        assert!(FrameId::System(SystemId(9)) < FrameId::Body(BodyId(0)));
        assert!(FrameId::Body(BodyId(1)) < FrameId::Body(BodyId(2)));
        assert!(FrameId::Body(BodyId(u32::MAX)) < FrameId::Construct(EntityId::new(0, 0)));
        assert!(GridId::Body(BodyId(u32::MAX)) < GridId::Construct(EntityId::new(0, 0)));
        assert!(GridId::Construct(EntityId::new(1, 5)) < GridId::Construct(EntityId::new(2, 0)));

        // Every grid lives in the frame of its owner.
        assert_eq!(GridId::Body(BodyId(4)).frame(), FrameId::Body(BodyId(4)));
        let ship = EntityId::new(7, 1);
        assert_eq!(GridId::Construct(ship).frame(), FrameId::Construct(ship));
        assert_eq!(GridId::from(BodyId(3)), GridId::Body(BodyId(3)));

        let mut set = HashSet::new();
        assert!(set.insert(FrameId::Body(BodyId(3))));
        assert!(!set.insert(FrameId::Body(BodyId(3))));
        assert!(set.insert(FrameId::System(SystemId(3))));

        assert_eq!(MaterialId::from(4_u16), MaterialId(4));
        assert_eq!(u64::from(SystemId(1 << 40)), 1 << 40);
        assert_eq!(format!("{} {:?}", BodyId(12), BodyId(12)), "12 BodyId(12)");
        assert_eq!(EntityId::new(4, 2).to_string(), "4v2");
    }
}
