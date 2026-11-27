//! `cp-voxel`: voxel storage for the Cubeplanet engine.
//!
//! Responsibility: chunk storage, edit overrides, the LOD pyramid of edits, the generator
//! interface [`BaseSource`], snapshots and level-0 queries. It takes plain flag tables
//! ([`VoxelTables`]), not the content registry, and follows the determinism rules of contract
//! §2.5 (results never depend on hash-map order, threads or time).
//!
//! Binding contract: `engine/docs/ARCHITEKTUR.md`, section 5.1.
//!
//! | module | contents |
//! |---|---|
//! | [`base`] | [`BaseSource`], [`GeneratorIdentity`], [`NodeClass`], [`Cancelled`], cell regions |
//! | [`chunk_data`] | [`ChunkData`]: uniform or paletted cells, canonical encoding |
//! | [`mask`] | [`CellMask`]: one bit per cell |
//! | [`chunk`] | [`VoxelChunk`]: cells plus authored mask |
//! | [`padded`] | [`PaddedChunk`]: a chunk with a one-cell halo, the mesher's input |
//! | [`tables`] | [`VoxelTables`]: flag and LOD tables |
//! | [`downsample`] | the normative downsample rule |
//! | [`edit`] | [`EditStore`], [`CellEdit`], [`EditReceipt`]: sparse overrides and pyramid |
//! | [`world`] | [`VoxelWorld`]: base + edits + base-chunk cache of one grid |
//! | [`snapshot`] | [`VoxelSnapshot`]: O(1) frozen views; classify, sample, sample_padded |
//! | [`query`] | raycast and box overlap at level 0 |
//! | [`stamp`] | [`VoxelSnapshot::apron_stamp`]: persistent stamp of a chunk's override data |
//!
//! ```
//! use std::sync::Arc;
//! use cp_core::block::{BlockFlags, LodClass, LodInfo};
//! use cp_core::coords::LodKey;
//! use cp_core::ids::BlockStateId;
//! use cp_core::math::{DVec3, IAabb, IVec3};
//! use cp_voxel::{
//!     BaseSource, Cancelled, CellEdit, GeneratorIdentity, NodeClass, VoxelTables, VoxelWorld,
//!     WorldConfig, region_volume,
//! };
//!
//! /// Stone below y = 0, air above, at every level.
//! struct Flat(GeneratorIdentity);
//! impl BaseSource for Flat {
//!     fn identity(&self) -> &GeneratorIdentity {
//!         &self.0
//!     }
//!     fn fill(
//!         &self,
//!         level: u8,
//!         region: IAabb,
//!         out: &mut [BlockStateId],
//!         _cancel: &dyn Fn() -> bool,
//!     ) -> Result<(), Cancelled> {
//!         assert_eq!(out.len(), region_volume(region));
//!         let mut cells = out.iter_mut();
//!         for y in region.min.y..region.max.y {
//!             // Cell y of `level` covers voxels [y << level, (y + 1) << level).
//!             let state = if y < 0 { BlockStateId(1) } else { BlockStateId::AIR };
//!             for _ in 0..(region.size().x * region.size().z) {
//!                 *cells.next().unwrap() = state;
//!             }
//!         }
//!         let _ = level;
//!         Ok(())
//!     }
//!     fn classify(&self, key: LodKey) -> NodeClass {
//!         match key.pos.y {
//!             y if y < 0 => NodeClass::Opaque,
//!             _ => NodeClass::Empty,
//!         }
//!     }
//!     fn subtree_empty(&self, key: LodKey) -> bool {
//!         key.pos.y >= 0
//!     }
//! }
//!
//! let stone = LodInfo { class: LodClass::Solid, proxy: BlockStateId(1) };
//! let tables = VoxelTables::new(
//!     Arc::from([BlockFlags::empty(), BlockFlags::SOLID | BlockFlags::OPAQUE]),
//!     Arc::from([LodInfo::EMPTY, stone]),
//! );
//! let identity = GeneratorIdentity {
//!     key: "doc:flat".into(), version: 1, seed: 0, macro_hash: 0,
//! };
//! let config = WorldConfig { top_level: 4, ..WorldConfig::default() };
//! let mut world = VoxelWorld::new(Arc::new(Flat(identity)), tables, config);
//!
//! let before = world.snapshot();
//! // Dig the voxel below the origin.
//! let receipt = world.apply(&[CellEdit { cell: IVec3::new(0, -1, 0), state: BlockStateId::AIR }]);
//! assert!(receipt.dirty.contains(&LodKey::lod0(IVec3::new(0, -1, 0))));
//! let after = world.snapshot();
//!
//! let down = DVec3::new(0.5, 3.0, 0.5);
//! let hit_before = before.raycast(down, DVec3::NEG_Y, 10.0).unwrap();
//! let hit_after = after.raycast(down, DVec3::NEG_Y, 10.0).unwrap();
//! assert_eq!(hit_before.cell, IVec3::new(0, -1, 0));
//! assert_eq!(hit_after.cell, IVec3::new(0, -2, 0));
//! assert_eq!(hit_after.distance, 4.0);
//! ```

pub mod base;
pub mod chunk;
pub mod chunk_data;
pub mod downsample;
pub mod edit;
pub mod mask;
pub mod padded;
pub mod query;
pub mod snapshot;
pub mod stamp;
pub mod tables;
pub mod world;

mod cache;
mod class;
mod layers;

pub use base::{
    BaseSource, Cancelled, GeneratorIdentity, NodeClass, chunk_cells, padded_cells, region_index,
    region_volume,
};
pub use cache::CacheUsage;
pub use chunk::VoxelChunk;
pub use chunk_data::{ChunkData, Paletted, States, WORDS_PER_BIT};
pub use downsample::{Downsampled, downsample_cell, downsample_chunk};
pub use edit::{CellEdit, ClassChange, EditReceipt, EditStore};
pub use mask::{CellMask, MASK_WORDS};
pub use padded::{PADDED_SIZE, PADDED_VOLUME, PaddedChunk, padded_index};
pub use query::RayHit;
pub use snapshot::VoxelSnapshot;
pub use stamp::{APRON_STAMP_VERSION, UNCHANGED_KEY};
pub use tables::VoxelTables;
pub use world::{VoxelWorld, WorldConfig};
