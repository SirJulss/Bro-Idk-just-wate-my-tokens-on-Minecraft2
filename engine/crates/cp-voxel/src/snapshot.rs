//! Immutable, cheaply cloned views of a voxel world (contract §5.1, "Snapshots").

use core::fmt;
use std::sync::Arc;

use cp_core::coords::{CHUNK_VOLUME, LodKey, cell_index};
use cp_core::face::Face;
use cp_core::ids::BlockStateId;

use crate::base::{BaseSource, Cancelled, NodeClass, chunk_cells, padded_cells};
use crate::cache::{BaseCache, CacheUsage};
use crate::chunk::VoxelChunk;
use crate::chunk_data::ChunkData;
use crate::class::composite_class;
use crate::layers::LayerMap;
use crate::padded::{PADDED_VOLUME, PaddedChunk, padded_index};
use crate::tables::VoxelTables;

fn never() -> bool {
    false
}

/// A frozen view of one grid: the base world, the override layers as they were when the
/// snapshot was taken, and the shared base-chunk cache (contract §5.1).
///
/// Taking a snapshot is O(1) and it stays valid however the world changes afterwards.
/// `Clone`, `Send` and `Sync`: meshing and streaming jobs read it on worker threads. Reads
/// generate missing base chunks (single-flight through the shared cache) and block until
/// they are ready; the `try_` variants take a cancel callback.
#[derive(Clone)]
pub struct VoxelSnapshot {
    pub(crate) base: Arc<dyn BaseSource>,
    pub(crate) tables: VoxelTables,
    pub(crate) layers: LayerMap,
    pub(crate) cache: Arc<BaseCache>,
    pub(crate) sequence: u64,
    /// Coarsest pyramid level of the edits.
    pub(crate) top_level: u8,
}

/// Generates one base chunk with a single `fill`.
///
/// `classify` is not asked first: for the chunks it cannot decide analytically it may cost
/// about as much as the fill itself, and for the ones it can, the fill is expected to take a
/// fast path of its own ([`BaseSource::fill`]); a uniform result collapses to `Uniform`.
fn generate_base(
    base: &dyn BaseSource,
    key: LodKey,
    cancel: &dyn Fn() -> bool,
) -> Result<VoxelChunk, Cancelled> {
    cp_core::profile_scope!("cp_voxel::generate_base");
    let mut cells = vec![BlockStateId::AIR; CHUNK_VOLUME];
    base.fill(key.level, chunk_cells(key), &mut cells, cancel)?;
    debug_assert!(
        !cells.contains(&BlockStateId::UNCHANGED),
        "a BaseSource produced the UNCHANGED sentinel"
    );
    Ok(VoxelChunk::new(ChunkData::from_cells(&cells)))
}

impl VoxelSnapshot {
    /// The class of the composite chunk `key` (base ⊕ overrides): exact-conservative like
    /// [`BaseSource::classify`]. Never generates cells.
    ///
    /// Not necessarily cheap: unless the chunk's override layer decides the class alone or an
    /// apply has memoised the base's class of the chunk, this calls [`BaseSource::classify`],
    /// which may evaluate the base field over the whole chunk (about as costly as a fill for
    /// chunks that cross the surface). Callers with a frame budget should keep the answer or
    /// ask from a job; it changes only with [`crate::EditReceipt::class_changes`].
    #[must_use]
    pub fn classify(&self, key: LodKey) -> NodeClass {
        composite_class(&self.layers, key, &self.tables.flags, &*self.base)
    }

    /// Conservative for the composite world: true only if no cell of `key` or of any
    /// descendant, at any level, is non-air. Use this rather than
    /// [`BaseSource::subtree_empty`], which knows nothing of player builds.
    ///
    /// Every level-0 override leaves an authored layer on each pyramid level up to the top
    /// level, so one layer lookup (or one scan of the top level's layers for keys above it)
    /// tells whether edits reach into the subtree; if they do, the answer is `false`.
    #[must_use]
    pub fn subtree_empty(&self, key: LodKey) -> bool {
        self.base.subtree_empty(key)
            && !self
                .layers
                .has_layer_within(key, key.level.min(self.top_level))
    }

    /// The composite chunk `key` at its level: base cells with the overrides of that level
    /// applied, and the chunk's authored mask. Unedited chunks are shared with the cache.
    #[must_use]
    pub fn sample(&self, key: LodKey) -> Arc<VoxelChunk> {
        self.try_sample(key, &never)
            .expect("sampling without a cancel request cannot be cancelled")
    }

    /// [`VoxelSnapshot::sample`] that gives up with [`Cancelled`] when `cancel` returns
    /// `true` (polled by the generator and while waiting for another thread's generation).
    pub fn try_sample(
        &self,
        key: LodKey,
        cancel: &dyn Fn() -> bool,
    ) -> Result<Arc<VoxelChunk>, Cancelled> {
        let Some(layer) = self.layers.layer(key) else {
            return self.base_chunk(key, cancel);
        };
        let blocks = if layer.blocks.contains(BlockStateId::UNCHANGED) {
            layer.blocks.or(&self.base_chunk(key, cancel)?.blocks)
        } else {
            layer.blocks.clone()
        };
        Ok(Arc::new(VoxelChunk {
            blocks,
            authored: layer.authored.clone(),
        }))
    }

    /// The composite chunk `key` with a one-cell halo of its face neighbours at the same
    /// level (contract §5.1), the mesher's input.
    ///
    /// One `base.fill` over the 34³ region, then the overrides of the chunk and of its six
    /// face neighbours on top. Edge and corner halo cells are base cells (unspecified by the
    /// contract).
    #[must_use]
    pub fn sample_padded(&self, key: LodKey) -> PaddedChunk {
        self.try_sample_padded(key, &never)
            .expect("sampling without a cancel request cannot be cancelled")
    }

    /// [`VoxelSnapshot::sample_padded`] with a cancel callback.
    pub fn try_sample_padded(
        &self,
        key: LodKey,
        cancel: &dyn Fn() -> bool,
    ) -> Result<PaddedChunk, Cancelled> {
        cp_core::profile_scope!("cp_voxel::sample_padded");
        let mut cells = vec![BlockStateId::AIR; PADDED_VOLUME].into_boxed_slice();
        self.base
            .fill(key.level, padded_cells(key), &mut cells, cancel)?;
        let own = self.layers.layer(key);
        if let Some(layer) = own {
            overlay_interior(&mut cells, layer);
        }
        for face in Face::ALL {
            let neighbour = LodKey::new(key.level, key.pos + face.normal());
            if let Some(layer) = self.layers.layer(neighbour) {
                overlay_face(&mut cells, layer, face);
            }
        }
        let authored = own.and_then(|layer| layer.authored.clone());
        Ok(PaddedChunk::new(key, cells, authored))
    }

    /// The revision of chunk `key`: the sequence number of the last apply that listed it as
    /// dirty ([`crate::EditReceipt::dirty`]), 0 for a chunk that no edit ever reached. It
    /// changes whenever the chunk's cells, its authored mask or its face halo change, and
    /// otherwise only in the rare level-0 case documented there.
    ///
    /// Session-local: valid within one world, restarting at 0 with every new one. It must
    /// never enter a persistent key or hash (contract §2.5 rule 5); the on-disk mesh cache
    /// (contract §6) uses [`VoxelSnapshot::apron_stamp`] instead.
    #[must_use]
    pub fn revision(&self, key: LodKey) -> u64 {
        self.layers.revision(key)
    }

    /// The world's apply count when the snapshot was taken.
    #[must_use]
    pub fn sequence(&self) -> u64 {
        self.sequence
    }

    /// The override layer of `key` (`UNCHANGED` where the base shows through), if any.
    #[must_use]
    pub fn override_layer(&self, key: LodKey) -> Option<&Arc<VoxelChunk>> {
        self.layers.layer(key)
    }

    /// Every override layer of every level, sorted by key: what a save writes (level 0) and
    /// may cache (the pyramid).
    #[must_use]
    pub fn overrides(&self) -> Vec<(LodKey, &Arc<VoxelChunk>)> {
        self.layers.layers()
    }

    /// The base world.
    #[must_use]
    pub fn base(&self) -> &Arc<dyn BaseSource> {
        &self.base
    }

    /// The per-state tables.
    #[must_use]
    pub fn tables(&self) -> &VoxelTables {
        &self.tables
    }

    /// Size of the shared base-chunk cache.
    #[must_use]
    pub fn cache_usage(&self) -> CacheUsage {
        self.cache.usage()
    }

    /// The base chunk `key`, cached.
    pub(crate) fn base_chunk(
        &self,
        key: LodKey,
        cancel: &dyn Fn() -> bool,
    ) -> Result<Arc<VoxelChunk>, Cancelled> {
        self.cache
            .get_or_generate(key, cancel, || generate_base(&*self.base, key, cancel))
    }
}

/// Writes the explicit cells of `layer` into the interior of a padded chunk.
fn overlay_interior(cells: &mut [BlockStateId], layer: &VoxelChunk) {
    if layer.blocks.uniform_state() == Some(BlockStateId::UNCHANGED) {
        return;
    }
    let decoded = layer.blocks.to_cells();
    let mut index = 0;
    for y in 0..32 {
        for z in 0..32 {
            let row = padded_index(0, y, z);
            for (x, &state) in decoded[index..index + 32].iter().enumerate() {
                if !state.is_unchanged() {
                    cells[row + x] = state;
                }
            }
            index += 32;
        }
    }
}

/// Writes the explicit cells of the face neighbour on side `face` that border the chunk into
/// the face halo of a padded chunk.
fn overlay_face(cells: &mut [BlockStateId], layer: &VoxelChunk, face: Face) {
    if layer.blocks.uniform_state() == Some(BlockStateId::UNCHANGED) {
        return;
    }
    let axis = face.axis();
    // The neighbour's slice that touches the chunk, and where it goes in the halo.
    let (theirs, ours) = if face.is_negative() {
        (31, -1)
    } else {
        (0, 32)
    };
    for a in 0..32_u8 {
        for b in 0..32_u8 {
            let mut local = [0_u8; 3];
            local[axis] = theirs;
            local[(axis + 1) % 3] = a;
            local[(axis + 2) % 3] = b;
            let state = layer.get(cell_index(local[0], local[1], local[2]));
            if state.is_unchanged() {
                continue;
            }
            let mut padded = [0_i32; 3];
            padded[axis] = ours;
            padded[(axis + 1) % 3] = i32::from(a);
            padded[(axis + 2) % 3] = i32::from(b);
            cells[padded_index(padded[0], padded[1], padded[2])] = state;
        }
    }
}

impl fmt::Debug for VoxelSnapshot {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("VoxelSnapshot")
            .field("generator", self.base.identity())
            .field("sequence", &self.sequence)
            .field("override_layers", &self.layers.layer_count())
            .finish_non_exhaustive()
    }
}
