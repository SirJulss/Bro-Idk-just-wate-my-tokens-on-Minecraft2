//! One voxel grid: base world, edits and the base-chunk cache (contract §5.1, "Snapshots").

use core::fmt;
use std::sync::Arc;

use cp_core::coords::MAX_LOD_LEVEL;

use crate::base::BaseSource;
use crate::cache::{BaseCache, CacheUsage};
use crate::edit::{CellEdit, EditReceipt, EditStore};
use crate::snapshot::VoxelSnapshot;
use crate::tables::VoxelTables;

/// Settings of a [`VoxelWorld`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WorldConfig {
    /// Coarsest level of the edit pyramid: the body's root level (contract §2.2), at most
    /// `MAX_LOD_LEVEL`.
    pub top_level: u8,
    /// Byte budget of the cache of generated base chunks.
    pub cache_bytes: usize,
}

impl Default for WorldConfig {
    /// Pyramid up to `MAX_LOD_LEVEL`, 128 MiB of cached base chunks.
    fn default() -> Self {
        Self {
            top_level: MAX_LOD_LEVEL,
            cache_bytes: 128 << 20,
        }
    }
}

/// The voxel data of one grid (body or construct): its base world, its [`EditStore`] and a
/// bounded single-flight cache of generated base chunks (contract §5.1).
///
/// Mutated only by its owner (the simulation thread) at tick boundaries through
/// [`VoxelWorld::apply`]; everyone else reads [`VoxelSnapshot`]s.
pub struct VoxelWorld {
    base: Arc<dyn BaseSource>,
    edits: EditStore,
    cache: Arc<BaseCache>,
}

impl VoxelWorld {
    /// A world without edits.
    ///
    /// # Panics
    /// If `config.top_level > MAX_LOD_LEVEL`.
    #[must_use]
    pub fn new(base: Arc<dyn BaseSource>, tables: VoxelTables, config: WorldConfig) -> Self {
        Self {
            base,
            edits: EditStore::new(tables, config.top_level),
            cache: Arc::new(BaseCache::new(config.cache_bytes)),
        }
    }

    /// Applies cell edits and updates the pyramid synchronously ([`EditStore::apply`]).
    /// Snapshots taken earlier keep seeing the old state.
    pub fn apply(&mut self, edits: &[CellEdit]) -> EditReceipt {
        self.edits.apply(edits, &*self.base)
    }

    /// An O(1) snapshot of the current state.
    #[must_use]
    pub fn snapshot(&self) -> VoxelSnapshot {
        VoxelSnapshot {
            base: Arc::clone(&self.base),
            tables: self.edits.tables().clone(),
            layers: self.edits.layer_map().clone(),
            cache: Arc::clone(&self.cache),
            sequence: self.edits.sequence(),
            top_level: self.edits.top_level(),
        }
    }

    /// The edits.
    #[must_use]
    pub fn edits(&self) -> &EditStore {
        &self.edits
    }

    /// The base world.
    #[must_use]
    pub fn base(&self) -> &Arc<dyn BaseSource> {
        &self.base
    }

    /// The per-state tables.
    #[must_use]
    pub fn tables(&self) -> &VoxelTables {
        self.edits.tables()
    }

    /// Size of the base-chunk cache.
    #[must_use]
    pub fn cache_usage(&self) -> CacheUsage {
        self.cache.usage()
    }
}

impl fmt::Debug for VoxelWorld {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("VoxelWorld")
            .field("generator", self.base.identity())
            .field("edits", &self.edits)
            .field("cache", &self.cache.usage())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::base::{Cancelled, GeneratorIdentity, NodeClass};
    use crate::edit::CellEdit;
    use crate::layers::{LayerMap, RegionKey};
    use cp_core::block::{BlockFlags, LodClass, LodInfo};
    use cp_core::coords::LodKey;
    use cp_core::ids::BlockStateId;
    use cp_core::math::{IAabb, IVec3};
    use std::collections::BTreeMap;

    struct Air(GeneratorIdentity);

    impl BaseSource for Air {
        fn identity(&self) -> &GeneratorIdentity {
            &self.0
        }
        fn fill(
            &self,
            _level: u8,
            _region: IAabb,
            out: &mut [BlockStateId],
            _cancel: &dyn Fn() -> bool,
        ) -> Result<(), Cancelled> {
            out.fill(BlockStateId::AIR);
            Ok(())
        }
        fn classify(&self, _key: LodKey) -> NodeClass {
            NodeClass::Empty
        }
        fn subtree_empty(&self, _key: LodKey) -> bool {
            true
        }
    }

    fn world() -> VoxelWorld {
        let stone = LodInfo {
            class: LodClass::Solid,
            proxy: BlockStateId(1),
        };
        let tables = VoxelTables::new(
            Arc::from([BlockFlags::empty(), BlockFlags::SOLID | BlockFlags::OPAQUE]),
            Arc::from([LodInfo::EMPTY, stone]),
        );
        let identity = GeneratorIdentity {
            key: "test:air".into(),
            version: 1,
            seed: 0,
            macro_hash: 0,
        };
        let config = WorldConfig {
            top_level: 4,
            ..WorldConfig::default()
        };
        VoxelWorld::new(Arc::new(Air(identity)), tables, config)
    }

    /// Region index pointers of a layer map.
    fn regions(map: &LayerMap) -> BTreeMap<RegionKey, *const ()> {
        map.regions
            .iter()
            .map(|(key, region)| (*key, Arc::as_ptr(region).cast::<()>()))
            .collect()
    }

    #[test]
    fn first_write_after_a_snapshot_copies_one_region_index_and_chunk_per_level() {
        let mut world = world();
        let stone = BlockStateId(1);
        // Layers in several regions on every level, and several chunks per region.
        let mut edits = Vec::new();
        for (i, base) in [
            IVec3::new(100, 100, 100),
            IVec3::new(-3000, 40, 900),
            IVec3::new(9000, -700, -5000),
        ]
        .into_iter()
        .enumerate()
        {
            for chunk in 0..4 {
                edits.push(CellEdit {
                    cell: base + IVec3::new(32 * chunk, i as i32, 0),
                    state: stone,
                });
            }
        }
        world.apply(&edits);
        let frozen = world.snapshot();
        // A cell in the chunk of the first edit whose ancestors on levels 1..=4 carry nothing
        // yet (so every level changes), interior on every level (local x 24, 28, 30, 15, 7;
        // y and z 5, 18, 25, 12, 6), so no halo neighbour is touched.
        let cell = IVec3::new(120, 101, 101);
        let receipt = world.apply(&[CellEdit { cell, state: stone }]);
        let written: Vec<LodKey> = (0..=4)
            .map(|level| LodKey::containing(level, cell))
            .collect();
        assert_eq!(receipt.dirty, written, "one chunk per level, no neighbour");

        let old = regions(&frozen.layers);
        let new = regions(world.edits.layer_map());
        assert_eq!(old.len(), new.len());
        let copied: Vec<RegionKey> = old
            .iter()
            .filter(|(key, ptr)| new[*key] != **ptr)
            .map(|(key, _)| *key)
            .collect();
        let expected: Vec<RegionKey> = written.iter().map(|&key| LayerMap::locate(key).0).collect();
        assert_eq!(
            copied, expected,
            "exactly one region index per written chunk"
        );
        // Inside a copied region only the written chunk is new; the others are shared.
        for &key in &written {
            let (region, _) = LayerMap::locate(key);
            let old_region = &frozen.layers.regions[&region];
            let new_region = &world.edits.layer_map().regions[&region];
            for (local, slot) in &new_region.slots {
                let chunk = LayerMap::key_of(region, *local);
                let (Some(new_layer), Some(old_layer)) = (
                    slot.layer.as_ref(),
                    old_region.slots.get(local).and_then(|s| s.layer.as_ref()),
                ) else {
                    continue;
                };
                assert_eq!(
                    !Arc::ptr_eq(new_layer, old_layer),
                    chunk == key,
                    "{chunk:?}"
                );
            }
        }

        // A second write to the same chunk before the next snapshot copies nothing.
        let region0 = LayerMap::locate(written[0]).0;
        let region_before = Arc::as_ptr(&world.edits.layer_map().regions[&region0]);
        let chunk_before = Arc::as_ptr(world.edits.layer_map().layer(written[0]).expect("layer"));
        world.apply(&[CellEdit {
            cell: cell + IVec3::X,
            state: stone,
        }]);
        assert_eq!(
            Arc::as_ptr(&world.edits.layer_map().regions[&region0]),
            region_before
        );
        assert_eq!(
            Arc::as_ptr(world.edits.layer_map().layer(written[0]).expect("layer")),
            chunk_before
        );
        // The frozen snapshot still has the old cell.
        assert_eq!(frozen.cell_state(cell), BlockStateId::AIR);
        assert_eq!(world.snapshot().cell_state(cell), stone);
    }
}
