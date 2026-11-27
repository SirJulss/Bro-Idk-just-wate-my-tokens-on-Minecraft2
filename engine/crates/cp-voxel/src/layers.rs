//! Structurally shared storage of override layers and chunk revisions (contract §5.1,
//! "Snapshots").
//!
//! Two levels of `Arc`: the map of regions (16³ chunks of one level) and, per region, an index
//! of its chunk slots, each slot holding an `Arc` of its override layer. Cloning a
//! [`LayerMap`] (what a snapshot does) is one reference-count increment. A write after a clone
//! copies the region map (one pointer per region), the one region index it writes to, and the
//! chunk it changes; every other region index and every other chunk stays shared.

use std::collections::BTreeMap;
use std::sync::Arc;

use cp_core::coords::LodKey;
use cp_core::math::IVec3;

use crate::chunk::VoxelChunk;
use crate::class::BaseClasses;

/// log2 of the region edge in chunks.
pub(crate) const REGION_BITS: u32 = 4;
/// Region edge in chunks.
pub(crate) const REGION_SIZE: i32 = 1 << REGION_BITS;

/// A region: 16³ chunks of one level, `pos = chunk pos >> 4`. Ordered by level, then x, y, z.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub(crate) struct RegionKey {
    pub(crate) level: u8,
    pub(crate) x: i32,
    pub(crate) y: i32,
    pub(crate) z: i32,
}

/// What is stored for one chunk: its override layer (if any), its revision and what is known
/// about its base class.
#[derive(Clone, Debug, Default)]
pub(crate) struct Slot {
    /// Override layer: `UNCHANGED` where the base shows through, plus authored bits. `None`
    /// when the chunk carries no override and no authored cell.
    pub(crate) layer: Option<Arc<VoxelChunk>>,
    /// The edit sequence number at which the chunk or its face halo last changed; 0 = never.
    pub(crate) revision: u64,
    /// Memo of `BaseSource::classify` for this chunk (the base never changes); kept when the
    /// layer goes away.
    pub(crate) base_class: BaseClasses,
}

/// The slots of one region, keyed by the local chunk index `x + 16 * z + 256 * y`.
#[derive(Clone, Debug, Default)]
pub(crate) struct Region {
    pub(crate) slots: BTreeMap<u16, Slot>,
}

/// The persistent map region → region index → chunk slot.
#[derive(Clone, Debug, Default)]
pub(crate) struct LayerMap {
    pub(crate) regions: Arc<BTreeMap<RegionKey, Arc<Region>>>,
}

impl LayerMap {
    /// Region and local slot index of a chunk.
    #[inline]
    pub(crate) fn locate(key: LodKey) -> (RegionKey, u16) {
        let region = key.pos >> REGION_BITS;
        let local = key.pos & (REGION_SIZE - 1);
        (
            RegionKey {
                level: key.level,
                x: region.x,
                y: region.y,
                z: region.z,
            },
            (local.x | (local.z << REGION_BITS) | (local.y << (2 * REGION_BITS))) as u16,
        )
    }

    /// Inverse of [`LayerMap::locate`].
    #[inline]
    pub(crate) fn key_of(region: RegionKey, local: u16) -> LodKey {
        let local = i32::from(local);
        let mask = REGION_SIZE - 1;
        let offset = IVec3::new(
            local & mask,
            (local >> (2 * REGION_BITS)) & mask,
            (local >> REGION_BITS) & mask,
        );
        LodKey::new(
            region.level,
            IVec3::new(region.x, region.y, region.z) * REGION_SIZE + offset,
        )
    }

    /// The slot of a chunk, if one exists.
    #[inline]
    pub(crate) fn slot(&self, key: LodKey) -> Option<&Slot> {
        let (region, local) = Self::locate(key);
        self.regions.get(&region)?.slots.get(&local)
    }

    /// The override layer of a chunk.
    #[inline]
    pub(crate) fn layer(&self, key: LodKey) -> Option<&Arc<VoxelChunk>> {
        self.slot(key)?.layer.as_ref()
    }

    /// The revision of a chunk (0 if it never changed).
    #[inline]
    pub(crate) fn revision(&self, key: LodKey) -> u64 {
        self.slot(key).map_or(0, |slot| slot.revision)
    }

    /// The slot of a chunk for writing, created if absent. Copies the region map and the
    /// region index first when they are shared with a snapshot (copy on write); the layer
    /// inside stays shared until the caller writes to it through `Arc::make_mut`.
    pub(crate) fn slot_mut(&mut self, key: LodKey) -> &mut Slot {
        let (region, local) = Self::locate(key);
        let regions = Arc::make_mut(&mut self.regions);
        let region = regions.entry(region).or_default();
        Arc::make_mut(region).slots.entry(local).or_default()
    }

    /// Every override layer with its chunk, sorted by key.
    pub(crate) fn layers(&self) -> Vec<(LodKey, &Arc<VoxelChunk>)> {
        let mut layers: Vec<(LodKey, &Arc<VoxelChunk>)> = self
            .regions
            .iter()
            .flat_map(|(&region, index)| {
                index.slots.iter().filter_map(move |(&local, slot)| {
                    slot.layer
                        .as_ref()
                        .map(|layer| (Self::key_of(region, local), layer))
                })
            })
            .collect();
        layers.sort_unstable_by_key(|(key, _)| *key);
        layers
    }

    /// True when a chunk of `level` (`<= key.level`) inside `key` has an override layer.
    pub(crate) fn has_layer_within(&self, key: LodKey, level: u8) -> bool {
        debug_assert!(level <= key.level);
        let shift = u32::from(key.level - level);
        // The chunks of `level` inside `key`, and the regions that hold them.
        let first = key.pos << shift;
        let last = ((key.pos + 1) << shift) - 1;
        let (first_region, last_region) = (first >> REGION_BITS, last >> REGION_BITS);
        let from = RegionKey {
            level,
            x: first_region.x,
            y: i32::MIN,
            z: i32::MIN,
        };
        let to = RegionKey {
            level,
            x: last_region.x,
            y: i32::MAX,
            z: i32::MAX,
        };
        self.regions.range(from..=to).any(|(&region, index)| {
            let inside = (first_region.y..=last_region.y).contains(&region.y)
                && (first_region.z..=last_region.z).contains(&region.z);
            inside
                && index.slots.iter().any(|(&local, slot)| {
                    let chunk = Self::key_of(region, local).pos;
                    slot.layer.is_some() && chunk.cmpge(first).all() && chunk.cmple(last).all()
                })
        })
    }

    /// Number of override layers.
    pub(crate) fn layer_count(&self) -> usize {
        self.regions
            .values()
            .map(|index| {
                index
                    .slots
                    .values()
                    .filter(|slot| slot.layer.is_some())
                    .count()
            })
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn locate_round_trips() {
        for level in [0_u8, 3, 20] {
            for pos in [
                IVec3::ZERO,
                IVec3::new(-1, -1, -1),
                IVec3::new(15, 16, -17),
                IVec3::new(-1000, 77, 4096),
            ] {
                let key = LodKey::new(level, pos);
                let (region, local) = LayerMap::locate(key);
                assert!(local < 4096);
                assert_eq!(region.level, level);
                assert_eq!(IVec3::new(region.x, region.y, region.z), pos >> 4);
                assert_eq!(LayerMap::key_of(region, local), key);
            }
        }
    }

    #[test]
    fn copy_on_write_shares_untouched_parts() {
        let mut map = LayerMap::default();
        let a = LodKey::lod0(IVec3::new(1, 2, 3));
        let b = LodKey::lod0(IVec3::new(2, 2, 3));
        let far = LodKey::lod0(IVec3::new(100, 0, 0));
        for key in [a, b, far] {
            map.slot_mut(key).layer = Some(Arc::new(VoxelChunk::default()));
        }
        let frozen = map.clone();
        assert!(
            Arc::ptr_eq(&frozen.regions, &map.regions),
            "a clone is O(1)"
        );
        let slot = map.slot_mut(a);
        let layer = slot.layer.as_mut().expect("layer exists");
        Arc::make_mut(layer)
            .blocks
            .set(0, cp_core::ids::BlockStateId(5));
        // The written chunk and its region index were copied, nothing else.
        assert!(!Arc::ptr_eq(&frozen.regions, &map.regions));
        let (region_a, _) = LayerMap::locate(a);
        let (region_far, _) = LayerMap::locate(far);
        assert!(!Arc::ptr_eq(
            &frozen.regions[&region_a],
            &map.regions[&region_a]
        ));
        assert!(Arc::ptr_eq(
            &frozen.regions[&region_far],
            &map.regions[&region_far]
        ));
        assert!(!Arc::ptr_eq(
            frozen.layer(a).expect("old layer"),
            map.layer(a).expect("new layer")
        ));
        assert!(Arc::ptr_eq(
            frozen.layer(b).expect("old layer"),
            map.layer(b).expect("new layer")
        ));
        assert_eq!(
            frozen.layer(a).expect("old").get(0),
            cp_core::ids::BlockStateId(0)
        );
        assert_eq!(
            map.layer(a).expect("new").get(0),
            cp_core::ids::BlockStateId(5)
        );
        assert_eq!(map.layer_count(), 3);
        assert_eq!(
            map.layers().iter().map(|(key, _)| *key).collect::<Vec<_>>(),
            [a, b, far]
        );
    }

    #[test]
    fn layers_within_a_coarser_chunk() {
        let mut map = LayerMap::default();
        let inner = LodKey::new(2, IVec3::new(-17, 3, 40));
        map.slot_mut(inner).layer = Some(Arc::new(VoxelChunk::default()));
        // A slot with a revision only does not count.
        map.slot_mut(LodKey::new(2, IVec3::new(-18, 3, 40)))
            .revision = 4;
        for level in 2..=8 {
            let ancestor = inner.ancestor_at(level).expect("valid level");
            assert!(map.has_layer_within(ancestor, 2), "level {level}");
            for offset in [IVec3::X, IVec3::NEG_Y, IVec3::Z] {
                let neighbour = LodKey::new(level, ancestor.pos + offset);
                assert!(
                    !map.has_layer_within(neighbour, 2),
                    "level {level} {offset}"
                );
            }
        }
        assert!(map.has_layer_within(inner, 2));
        assert!(!map.has_layer_within(LodKey::new(3, IVec3::ZERO), 3));
        assert!(!map.has_layer_within(LodKey::new(3, IVec3::new(-9, 1, 20)), 3));
    }
}
