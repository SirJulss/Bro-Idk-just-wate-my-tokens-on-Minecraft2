//! Player edits as sparse override layers plus the derived LOD pyramid (contract §5.1).
//!
//! Level 0 stores, per touched chunk, a [`VoxelChunk`] whose cells are
//! [`BlockStateId::UNCHANGED`] (the base shows through) or an explicit state (`AIR` = dug);
//! the authored mask marks the explicit cells. `composite(0) = override.or(base)`.
//!
//! Level `L+1` overrides are a pure function of the eight level-`L` composites and the base: a
//! parent cell carries an override only if at least one of its children carries one and the
//! downsampled result ([`crate::downsample_cell`]) differs from `base(L+1)`; it is authored
//! iff a child is. `composite(L) = O_L.or(base(L))`. [`EditStore::apply`] keeps this pyramid
//! up to date synchronously, touching only the parent cells of changed cells, level by level.
//! The simulation never reads levels above 0.
//!
//! The class changes of a receipt are kept cheap: a write changes a chunk's class only if it
//! changes the distinct states of the chunk's layer (an authored bit never does), and the base
//! class, which [`BaseSource::classify`] may compute from the whole base chunk, is asked at
//! most once per chunk and remembered; most surface chunks are decided by a few base cells
//! around the edit instead (see `crate::class`).

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use cp_core::block::BlockFlags;
use cp_core::coords::{
    CHUNK_SIZE, LodKey, MAX_LOD_LEVEL, VOXEL_LIMIT, cell_index, index_to_cell, voxel_to_chunk,
};
use cp_core::face::Face;
use cp_core::ids::BlockStateId;
use cp_core::math::{IAabb, IVec3};

use crate::base::{BaseSource, NodeClass, chunk_cells, region_index, region_volume};
use crate::chunk::VoxelChunk;
use crate::class::{BaseClasses, CellEvidence, LayerClass, composite_class};
use crate::downsample::{child_offset, downsample_cell};
use crate::layers::LayerMap;
use crate::tables::VoxelTables;

/// One edit of a level-0 cell.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct CellEdit {
    /// The voxel (grid-local level-0 coordinates); must lie in `[-VOXEL_LIMIT, VOXEL_LIMIT)`.
    pub cell: IVec3,
    /// The new override: a state (`AIR` = dug), or [`BlockStateId::UNCHANGED`] to revert the
    /// cell to the base world.
    pub state: BlockStateId,
}

/// A change of [`crate::VoxelSnapshot::classify`] caused by an edit.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ClassChange {
    /// The chunk.
    pub key: LodKey,
    /// Its class before the edit.
    pub before: NodeClass,
    /// Its class after the edit.
    pub after: NodeClass,
}

/// What one [`EditStore::apply`] changed.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct EditReceipt {
    /// The edit sequence number of this apply; every dirty key's revision is now this value.
    pub revision: u64,
    /// Every chunk whose cells, authored mask or face halo changed, sorted: the level-0 chunks
    /// with changed cells, the face neighbours whose halo contains a cell whose state changed,
    /// and the same for every affected pyramid level. A chunk whose authored mask alone changed
    /// is dirty, its neighbours are not (a halo carries states only, so their padded chunks
    /// and meshes are unchanged); a consumer that keeps flags depending on whether a neighbour
    /// is edited (such as wall masks, contract §5.3) derives them from the dirty chunk.
    ///
    /// Exact on pyramid levels. On level 0 a cell counts as changed when its override changed,
    /// so setting or removing an override equal to the base state lists the chunk's face
    /// neighbours although their halo did not change (the base cell is not generated here).
    pub dirty: Vec<LodKey>,
    /// Chunks whose [`NodeClass`] changed, sorted by key.
    pub class_changes: Vec<ClassChange>,
    /// Level-0 cells whose override changed (no-op edits are not counted).
    pub changed_cells: usize,
    /// Edits ignored because their cell lies outside `[-VOXEL_LIMIT, VOXEL_LIMIT)` or their
    /// state has no table entry.
    pub rejected: usize,
}

/// One cell of an override layer to (re)write.
#[derive(Clone, Copy, Debug)]
struct CellChange {
    index: u16,
    state: BlockStateId,
    authored: bool,
}

/// Bookkeeping of one apply.
#[derive(Default)]
struct Work {
    dirty: BTreeSet<LodKey>,
    class_changes: Vec<ClassChange>,
}

/// The edits of one grid: sparse override layers at level 0 and the derived pyramid up to
/// `top_level` (contract §5.1). Never stores whole copies of base chunks.
///
/// Mutated only by its owner; [`crate::VoxelWorld::snapshot`] shares its layers with
/// snapshots, so the first write after a snapshot copies only what it touches.
#[derive(Clone)]
pub struct EditStore {
    tables: VoxelTables,
    top_level: u8,
    layers: LayerMap,
    sequence: u64,
}

impl core::fmt::Debug for EditStore {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("EditStore")
            .field("top_level", &self.top_level)
            .field("sequence", &self.sequence)
            .field("layers", &self.layers.layer_count())
            .field("states", &self.tables.len())
            .finish()
    }
}

fn never() -> bool {
    false
}

/// `base.fill` of a region; generation inside `apply` is never cancelled.
fn fill_region(base: &dyn BaseSource, level: u8, region: IAabb) -> Vec<BlockStateId> {
    let mut cells = vec![BlockStateId::AIR; region_volume(region)];
    base.fill(level, region, &mut cells, &never)
        .expect("BaseSource::fill returned Cancelled although cancel never fired");
    debug_assert!(
        !cells.contains(&BlockStateId::UNCHANGED),
        "a BaseSource produced the UNCHANGED sentinel"
    );
    cells
}

/// Tile (8³ parent cells) of a parent-local cell; base cells are fetched per tile so that
/// edits far apart inside one chunk do not make it fetch the whole chunk.
#[inline]
fn tile_of(local: IVec3) -> usize {
    ((local.x >> 3) | ((local.z >> 3) << 2) | ((local.y >> 3) << 4)) as usize
}

/// Marks the face neighbours of `key` whose halo contains the cell `index` (a cell on the
/// chunk's border).
fn mark_halo_neighbours(dirty: &mut BTreeSet<LodKey>, key: LodKey, index: u16) {
    let local = index_to_cell(usize::from(index));
    for (axis, &coordinate) in local.iter().enumerate() {
        let side = match coordinate {
            0 => Face::from_axis(axis, true),
            31 => Face::from_axis(axis, false),
            _ => continue,
        };
        let neighbour = LodKey::new(key.level, key.pos + side.normal());
        if neighbour.is_valid() {
            dirty.insert(neighbour);
        }
    }
}

/// Evidence about the base class of `key` from the base cells of the 3³ box around the cell
/// `index`, clipped to the chunk: one small fill instead of a `classify`, which may evaluate
/// the field over the whole chunk. Edits happen where solid meets open space, so the box
/// usually holds both and proves the chunk `Mixed`.
fn probe(base: &dyn BaseSource, key: LodKey, index: u16, flags: &[BlockFlags]) -> CellEvidence {
    let [x, y, z] = index_to_cell(usize::from(index));
    let cell = key.pos * CHUNK_SIZE + IVec3::new(i32::from(x), i32::from(y), i32::from(z));
    let region = IAabb::new(cell - 1, cell + 2)
        .intersection(&chunk_cells(key))
        .expect("a cell of the chunk lies inside the chunk");
    CellEvidence::of(&fill_region(base, key.level, region), flags)
}

impl EditStore {
    /// An empty store whose pyramid reaches up to `top_level` (the body's root level; levels
    /// above it are never derived).
    ///
    /// # Panics
    /// If `top_level > MAX_LOD_LEVEL`.
    #[must_use]
    pub fn new(tables: VoxelTables, top_level: u8) -> Self {
        assert!(
            top_level <= MAX_LOD_LEVEL,
            "EditStore::new: top_level above MAX_LOD_LEVEL"
        );
        Self {
            tables,
            top_level,
            layers: LayerMap::default(),
            sequence: 0,
        }
    }

    /// The coarsest pyramid level kept up to date.
    #[inline]
    #[must_use]
    pub fn top_level(&self) -> u8 {
        self.top_level
    }

    /// The tables the downsample rule reads.
    #[inline]
    #[must_use]
    pub fn tables(&self) -> &VoxelTables {
        &self.tables
    }

    /// Number of applies so far; the revision the next apply will stamp is one more.
    #[inline]
    #[must_use]
    pub fn sequence(&self) -> u64 {
        self.sequence
    }

    /// The override layer of a chunk at any level (`UNCHANGED` where the base shows through),
    /// `None` when the chunk has no override and no authored cell.
    #[must_use]
    pub fn layer(&self, key: LodKey) -> Option<&VoxelChunk> {
        self.layers.layer(key).map(Arc::as_ref)
    }

    /// Every override layer, sorted by key (all levels).
    #[must_use]
    pub fn layers(&self) -> Vec<(LodKey, &VoxelChunk)> {
        self.layers
            .layers()
            .into_iter()
            .map(|(key, layer)| (key, layer.as_ref()))
            .collect()
    }

    /// Number of override layers (all levels).
    #[must_use]
    pub fn layer_count(&self) -> usize {
        self.layers.layer_count()
    }

    /// True when no chunk carries an override or an authored cell.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.layer_count() == 0
    }

    /// The level-0 override of a voxel: its explicit state or [`BlockStateId::UNCHANGED`].
    #[must_use]
    pub fn override_at(&self, cell: IVec3) -> BlockStateId {
        let (chunk, [x, y, z]) = voxel_to_chunk(cell);
        self.layer(LodKey::lod0(chunk))
            .map_or(BlockStateId::UNCHANGED, |layer| {
                layer.get(cell_index(x, y, z))
            })
    }

    /// The revision of a chunk: the sequence number of the last apply that listed the chunk
    /// as dirty ([`EditReceipt::dirty`]), 0 if none ever did.
    ///
    /// Session-local: the sequence starts at 0 with every store, so a revision must never
    /// enter a persistent key or hash (contract §2.5 rule 5); use
    /// [`crate::VoxelSnapshot::apron_stamp`] for those.
    #[must_use]
    pub fn revision(&self, key: LodKey) -> u64 {
        self.layers.revision(key)
    }

    /// The class of the composite chunk `key` (base ⊕ overrides), exact-conservative like
    /// [`BaseSource::classify`].
    ///
    /// Asks `base` only when the chunk's override layer does not decide the class alone and
    /// no apply has memoised the base's class of the chunk; that call may evaluate the base
    /// field over the chunk (see [`BaseSource::classify`]).
    #[must_use]
    pub fn classify(&self, key: LodKey, base: &dyn BaseSource) -> NodeClass {
        composite_class(&self.layers, key, &self.tables.flags, base)
    }

    /// The layers, for snapshots.
    #[inline]
    pub(crate) fn layer_map(&self) -> &LayerMap {
        &self.layers
    }

    /// Applies `edits` in order (a later edit of the same cell wins) and updates the pyramid
    /// synchronously. `base` must be the grid's base world.
    ///
    /// Edits whose cell lies outside `[-VOXEL_LIMIT, VOXEL_LIMIT)` or whose state has no
    /// table entry are ignored and counted in [`EditReceipt::rejected`].
    ///
    /// Cost: proportional to the changed cells and their ancestors, plus base generation of
    /// the few cells the downsample rule needs. For the class changes, a chunk whose layer
    /// gains or loses a distinct state may need its base class: a fill of at most 3³ base
    /// cells around the edit decides most chunks, and only the rest cost one
    /// [`BaseSource::classify`], once per chunk for the life of the store (the answer is kept
    /// with the chunk). Authored-only changes on pyramid levels never ask the base.
    pub fn apply(&mut self, edits: &[CellEdit], base: &dyn BaseSource) -> EditReceipt {
        cp_core::profile_scope!("cp_voxel::EditStore::apply");
        self.sequence += 1;
        let mut receipt = EditReceipt {
            revision: self.sequence,
            ..EditReceipt::default()
        };
        // Final requested override per level-0 cell, grouped by chunk.
        let mut requested: BTreeMap<LodKey, BTreeMap<u16, BlockStateId>> = BTreeMap::new();
        for edit in edits {
            if !self.accepts(edit) {
                receipt.rejected += 1;
                continue;
            }
            let (chunk, [x, y, z]) = voxel_to_chunk(edit.cell);
            requested
                .entry(LodKey::lod0(chunk))
                .or_default()
                .insert(cell_index(x, y, z) as u16, edit.state);
        }

        let mut work = Work::default();
        let mut changed: BTreeMap<LodKey, Vec<u16>> = BTreeMap::new();
        for (key, cells) in requested {
            let layer = self.layer(key);
            let changes: Vec<CellChange> = cells
                .into_iter()
                .filter(|&(index, state)| {
                    layer.map_or(BlockStateId::UNCHANGED, |layer| {
                        layer.get(usize::from(index))
                    }) != state
                })
                .map(|(index, state)| CellChange {
                    index,
                    state,
                    authored: !state.is_unchanged(),
                })
                .collect();
            if changes.is_empty() {
                continue;
            }
            receipt.changed_cells += changes.len();
            self.write_changes(key, &changes, CellEvidence::default(), base, &mut work);
            changed.insert(key, changes.iter().map(|change| change.index).collect());
        }

        for _ in 0..self.top_level {
            if changed.is_empty() {
                break;
            }
            let mut parents: BTreeMap<LodKey, BTreeSet<u16>> = BTreeMap::new();
            for (key, cells) in &changed {
                let parent = key
                    .parent()
                    .expect("levels below top_level <= MAX_LOD_LEVEL have a parent");
                // The child chunk covers one octant (16³ cells) of its parent.
                let octant = (key.pos & 1) * (CHUNK_SIZE / 2);
                let parent_cells = parents.entry(parent).or_default();
                for &index in cells {
                    let [x, y, z] = index_to_cell(usize::from(index));
                    let local = octant
                        + IVec3::new(i32::from(x >> 1), i32::from(y >> 1), i32::from(z >> 1));
                    parent_cells
                        .insert(cell_index(local.x as u8, local.y as u8, local.z as u8) as u16);
                }
            }
            let mut next = BTreeMap::new();
            for (parent, cells) in parents {
                let (changes, evidence) = self.recompute_parents(parent, &cells, base);
                if changes.is_empty() {
                    continue;
                }
                self.write_changes(parent, &changes, evidence, base, &mut work);
                next.insert(parent, changes.iter().map(|change| change.index).collect());
            }
            changed = next;
        }

        for &key in &work.dirty {
            self.layers.slot_mut(key).revision = receipt.revision;
        }
        receipt.dirty = work.dirty.into_iter().collect();
        receipt.class_changes = work.class_changes;
        receipt
    }

    fn accepts(&self, edit: &CellEdit) -> bool {
        let inside = edit.cell.cmpge(IVec3::splat(-VOXEL_LIMIT)).all()
            && edit.cell.cmplt(IVec3::splat(VOXEL_LIMIT)).all();
        inside && (edit.state.is_unchanged() || self.tables.contains(edit.state))
    }

    /// Writes changed cells into the layer of `key` (creating or removing it), records the
    /// dirty keys and the class change. `evidence` comes from base cells of `key` that the
    /// caller has generated anyway.
    fn write_changes(
        &mut self,
        key: LodKey,
        changes: &[CellChange],
        evidence: CellEvidence,
        base: &dyn BaseSource,
        work: &mut Work,
    ) {
        // Disjoint fields: the tables are read while one slot of the layers is written.
        let flags: &[BlockFlags] = &self.tables.flags;
        let slot = self.layers.slot_mut(key);
        let before = LayerClass::of(slot.layer.as_deref(), flags);
        let layer = slot
            .layer
            .get_or_insert_with(|| Arc::new(VoxelChunk::uniform(BlockStateId::UNCHANGED)));
        let chunk = Arc::make_mut(layer);
        work.dirty.insert(key);
        let mut first_state_change = None;
        for change in changes {
            let index = usize::from(change.index);
            if chunk.get(index) != change.state {
                chunk.blocks.set(index, change.state);
                // A face halo carries states only: an authored bit alone never reaches it.
                mark_halo_neighbours(&mut work.dirty, key, change.index);
                if first_state_change.is_none() {
                    first_state_change = Some(change.index);
                }
            }
            chunk.set_authored(index, change.authored);
        }
        chunk.normalize_authored();
        if chunk.blocks.uniform_state() == Some(BlockStateId::UNCHANGED) && chunk.authored.is_none()
        {
            slot.layer = None;
        }
        // Evidence that came for free is kept, whether or not this write needs the class.
        slot.base_class = slot.base_class.narrow(evidence);
        let after = LayerClass::of(slot.layer.as_deref(), flags);
        if before == after {
            // The class is a function of the layer class and the (fixed) base class.
            return;
        }
        let classes =
            |base_class: NodeClass| (before.with_base(base_class), after.with_base(base_class));
        let mut known = slot.base_class;
        let decided = match known.decide(classes) {
            Some(decided) => decided,
            None => {
                // Only states enter a layer class, so some state changed.
                let index = first_state_change.expect("the layer class changed with no state");
                known = known.narrow(probe(base, key, index, flags));
                match known.decide(classes) {
                    Some(decided) => decided,
                    None => {
                        let base_class = base.classify(key);
                        known = BaseClasses::exactly(base_class);
                        classes(base_class)
                    }
                }
            }
        };
        slot.base_class = known;
        let (before, after) = decided;
        if before != after {
            work.class_changes.push(ClassChange { key, before, after });
        }
    }

    /// New override and authored bit of the given parent-local cells of `parent` (level
    /// `>= 1`), from the layers one level finer (already up to date) and the base. Returns
    /// only the cells that differ from the current layer of `parent`, and the evidence about
    /// the base class of `parent` from the base cells generated on the way.
    fn recompute_parents(
        &self,
        parent: LodKey,
        cells: &BTreeSet<u16>,
        base: &dyn BaseSource,
    ) -> (Vec<CellChange>, CellEvidence) {
        struct Gathered {
            index: u16,
            local: IVec3,
            overrides: [BlockStateId; 8],
            authored: u8,
        }
        struct TileBase {
            parent_region: IAabb,
            parent_cells: Vec<BlockStateId>,
            child_region: IAabb,
            child_cells: Vec<BlockStateId>,
        }

        let child_layers = parent.children().map(|key| self.layer(key));
        let parent_layer = self.layer(parent);
        // First parent cell of the chunk, in parent-level cells.
        let origin = parent.pos * CHUNK_SIZE;

        let mut gathered = Vec::with_capacity(cells.len());
        let mut tiles = [IAabb::EMPTY; 64];
        for &index in cells {
            let [px, py, pz] = index_to_cell(usize::from(index));
            let octant =
                usize::from(px >> 4) | (usize::from(pz >> 4) << 1) | (usize::from(py >> 4) << 2);
            let [bx, by, bz] = [px & 15, py & 15, pz & 15].map(|coordinate| coordinate * 2);
            let mut overrides = [BlockStateId::UNCHANGED; 8];
            let mut authored = 0_u8;
            if let Some(layer) = child_layers[octant] {
                for (child, slot) in overrides.iter_mut().enumerate() {
                    let [dx, dy, dz] = child_offset(child);
                    let child_index = cell_index(bx + dx, by + dy, bz + dz);
                    *slot = layer.get(child_index);
                    if layer.is_authored(child_index) {
                        authored |= 1 << child;
                    }
                }
            }
            let local = IVec3::new(i32::from(px), i32::from(py), i32::from(pz));
            if overrides.iter().any(|state| !state.is_unchanged()) {
                let tile = &mut tiles[tile_of(local)];
                *tile = tile.union(&IAabb::from_min_size(local, IVec3::ONE));
            }
            gathered.push(Gathered {
                index,
                local,
                overrides,
                authored,
            });
        }

        let tile_bases: Vec<Option<TileBase>> = tiles
            .iter()
            .map(|tile| {
                (!tile.is_empty()).then(|| {
                    let parent_region = IAabb::new(origin + tile.min, origin + tile.max);
                    let child_region = IAabb::new(parent_region.min * 2, parent_region.max * 2);
                    TileBase {
                        parent_cells: fill_region(base, parent.level, parent_region),
                        child_cells: fill_region(base, parent.level - 1, child_region),
                        parent_region,
                        child_region,
                    }
                })
            })
            .collect();
        let mut evidence = CellEvidence::default();
        for tile in tile_bases.iter().flatten() {
            if evidence.is_conclusive() {
                break;
            }
            evidence = evidence.merge(CellEvidence::of(&tile.parent_cells, &self.tables.flags));
        }

        let mut changes = Vec::new();
        for cell in gathered {
            let has_override = cell.overrides.iter().any(|state| !state.is_unchanged());
            let state = match &tile_bases[tile_of(cell.local)] {
                Some(tile) if has_override => {
                    let parent_cell = origin + cell.local;
                    let mut composite = [BlockStateId::AIR; 8];
                    for (child, slot) in composite.iter_mut().enumerate() {
                        let own = cell.overrides[child];
                        *slot = if own.is_unchanged() {
                            let [dx, dy, dz] = child_offset(child);
                            let child_cell = parent_cell * 2
                                + IVec3::new(i32::from(dx), i32::from(dy), i32::from(dz));
                            tile.child_cells[region_index(tile.child_region, child_cell)]
                        } else {
                            own
                        };
                    }
                    let down = downsample_cell(
                        &composite,
                        cell.authored,
                        parent.level,
                        &self.tables.flags,
                        &self.tables.lod,
                    );
                    let base_state =
                        tile.parent_cells[region_index(tile.parent_region, parent_cell)];
                    if down.state == base_state {
                        BlockStateId::UNCHANGED
                    } else {
                        down.state
                    }
                }
                _ => BlockStateId::UNCHANGED,
            };
            let authored = cell.authored != 0;
            let index = usize::from(cell.index);
            let current = parent_layer.map_or((BlockStateId::UNCHANGED, false), |layer| {
                (layer.get(index), layer.is_authored(index))
            });
            if current != (state, authored) {
                changes.push(CellChange {
                    index: cell.index,
                    state,
                    authored,
                });
            }
        }
        (changes, evidence)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dirty_neighbours_follow_the_halo() {
        let key = LodKey::lod0(IVec3::new(2, -1, 0));
        let mut dirty = BTreeSet::new();
        mark_halo_neighbours(&mut dirty, key, cell_index(5, 6, 7) as u16);
        assert!(dirty.is_empty(), "an interior cell is in no halo");
        mark_halo_neighbours(&mut dirty, key, cell_index(0, 31, 7) as u16);
        let expected: BTreeSet<LodKey> = [
            LodKey::lod0(IVec3::new(1, -1, 0)),
            LodKey::lod0(IVec3::new(2, 0, 0)),
        ]
        .into_iter()
        .collect();
        assert_eq!(dirty, expected);
        assert_eq!(tile_of(IVec3::new(31, 31, 31)), 63);
        assert_eq!(tile_of(IVec3::new(8, 0, 0)), 1);
        assert_eq!(tile_of(IVec3::new(0, 0, 8)), 4);
        assert_eq!(tile_of(IVec3::new(0, 8, 0)), 16);
    }
}
