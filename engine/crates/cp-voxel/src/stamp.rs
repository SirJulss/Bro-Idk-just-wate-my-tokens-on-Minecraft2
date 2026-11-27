//! Persistent stamps of the override data a chunk's mesh depends on (contract §6, mesh cache;
//! §2.5 rules 4 and 5).
//!
//! A padded chunk ([`VoxelSnapshot::sample_padded`]) and its authored mask are a function of
//! the base world, the state tables and the override data around the chunk: the chunk's own
//! override layer and, of each face neighbour, the slice of its layer that forms the face
//! halo. The generator identity pins the base and the content hash pins the tables, so a mesh
//! cache that outlives the session keys on those plus [`VoxelSnapshot::apron_stamp`].
//! [`VoxelSnapshot::revision`] cannot serve there: it counts the applies of one session.

use cp_core::coords::{CHUNK_SIZE, LodKey, cell_index};
use cp_core::face::Face;
use cp_core::hash::hash_bytes;
use cp_core::ids::BlockStateId;
use rustc_hash::FxHashMap;

use crate::chunk::VoxelChunk;
use crate::snapshot::VoxelSnapshot;

/// Version of the stamp recipe of [`VoxelSnapshot::apron_stamp`]. It is part of the hashed
/// bytes and is bumped whenever the recipe changes, so stamps of different recipes differ.
pub const APRON_STAMP_VERSION: u32 = 1;

/// The key that stands for [`BlockStateId::UNCHANGED`] in a stamp. Never a registry key: the
/// key grammar (contract §4.4) has no `#`.
pub const UNCHANGED_KEY: &str = "#unchanged";

/// First bytes of every stamp's input.
const STAMP_TAG: &[u8] = b"cp-voxel:apron";

/// Cells of one face halo slice.
const SLICE_CELLS: usize = (CHUNK_SIZE * CHUNK_SIZE) as usize;

impl VoxelSnapshot {
    /// A persistent stamp of the override data the padded chunk `key` and its authored mask
    /// depend on: the override layer of `key` (states and authored mask) and, for each face
    /// neighbour, the slice of the neighbour's override layer that forms the face halo of
    /// `key`. The key of the mesh cache (contract §6) combines it with the generator identity,
    /// the grid, `key`, the content hash and the mesher versions.
    ///
    /// Persistent where [`VoxelSnapshot::revision`] is not: a pure function of that data and
    /// of the persistent state keys, identical in every session and on every machine for the
    /// same override data whatever edits led to it, and independent of session-local state ids
    /// (contract §2.5 rules 4 and 5). Equal stamps mean equal override data, up to collisions
    /// of a 64-bit hash; a neighbour's cells off the halo slice never enter. An override equal
    /// to the base cell changes the stamp although not the padded chunk: a cache miss, never a
    /// stale hit.
    ///
    /// `key_of` maps a state to its persistent key (the registry's state key); it is never
    /// called with [`BlockStateId::UNCHANGED`], which is written as [`UNCHANGED_KEY`]. States
    /// with the same key count as one.
    ///
    /// Recipe ([`APRON_STAMP_VERSION`] 1), FNV-1a ([`hash_bytes`]) over the bytes
    /// - `"cp-voxel:apron"`, then the version as `u32` (little-endian like every integer);
    /// - the own layer: `0u8` when the chunk has no override data (no layer, or only
    ///   `UNCHANGED` cells and no authored cell), else `1u8`, the canonical encoding of the
    ///   layer's states ([`crate::ChunkData::write_canonical`]), then `0u8` without an authored
    ///   cell or `1u8` and the 512 words of the authored mask as `u64`;
    /// - for each face in [`Face::ALL`] order (`+X, -X, +Y, -Y, +Z, -Z`), the halo slice: the
    ///   32 × 32 cells of the neighbour across that face that touch the chunk, cell `(a, b)` at
    ///   neighbour-local `n = 0` (positive faces) or `n = 31` (negative faces), `u = a`,
    ///   `v = b`, where `n` is the face's axis and `u`, `v` its tangent axes (contract §2.3),
    ///   `a` outer and `b` inner; `0u8` when all are `UNCHANGED` (or the neighbour has no
    ///   layer), else `1u8`, the number `P` of distinct keys as `u32`, the keys in order of
    ///   first occurrence (each a `u32` byte length and its UTF-8 bytes) and one `u16` key
    ///   index per cell.
    ///
    /// Cost: seven layer lookups; for each layer found, one pass over its cells (own layer)
    /// or over one slice of 1024 cells (neighbours).
    ///
    /// # Panics
    /// If a key is longer than `u32::MAX` bytes.
    #[must_use]
    pub fn apron_stamp<'k>(
        &self,
        key: LodKey,
        mut key_of: impl FnMut(BlockStateId) -> &'k str,
    ) -> u64 {
        let mut key_of = |state: BlockStateId| {
            if state.is_unchanged() {
                UNCHANGED_KEY
            } else {
                key_of(state)
            }
        };
        let mut bytes = Vec::with_capacity(64);
        bytes.extend_from_slice(STAMP_TAG);
        bytes.extend_from_slice(&APRON_STAMP_VERSION.to_le_bytes());
        match self
            .layers
            .layer(key)
            .filter(|layer| has_override_data(layer))
        {
            None => bytes.push(0),
            Some(layer) => {
                bytes.push(1);
                layer.blocks.write_canonical(&mut bytes, &mut key_of);
                match layer.authored_words() {
                    None => bytes.push(0),
                    Some(words) => {
                        bytes.push(1);
                        for word in words {
                            bytes.extend_from_slice(&word.to_le_bytes());
                        }
                    }
                }
            }
        }
        for face in Face::ALL {
            let neighbour = LodKey::new(key.level, key.pos + face.normal());
            match self
                .layers
                .layer(neighbour)
                .and_then(|layer| halo_slice(layer, face))
            {
                None => bytes.push(0),
                Some(states) => {
                    bytes.push(1);
                    write_states(&mut bytes, &states, &mut key_of);
                }
            }
        }
        hash_bytes(&bytes)
    }
}

/// True unless the layer holds only `UNCHANGED` cells and no authored cell.
fn has_override_data(layer: &VoxelChunk) -> bool {
    layer.blocks.uniform_state() != Some(BlockStateId::UNCHANGED) || layer.has_authored()
}

/// The cells of `layer` (the neighbour across `face`) that form the face halo of the chunk,
/// in stamp order; `None` when all of them are `UNCHANGED`.
fn halo_slice(layer: &VoxelChunk, face: Face) -> Option<Vec<BlockStateId>> {
    if layer.blocks.uniform_state() == Some(BlockStateId::UNCHANGED) {
        return None;
    }
    let normal_axis = face.axis();
    let border = if face.is_negative() { 31 } else { 0 };
    let mut states = Vec::with_capacity(SLICE_CELLS);
    for a in 0..32_u8 {
        for b in 0..32_u8 {
            let mut local = [0_u8; 3];
            local[normal_axis] = border;
            local[face.u_axis()] = a;
            local[face.v_axis()] = b;
            states.push(layer.get(cell_index(local[0], local[1], local[2])));
        }
    }
    states
        .iter()
        .any(|state| !state.is_unchanged())
        .then_some(states)
}

/// Appends `P`, the distinct keys of `states` in order of first occurrence and one `u16` key
/// index per state (at most 65536 distinct keys).
fn write_states<'k>(
    out: &mut Vec<u8>,
    states: &[BlockStateId],
    mut key_of: impl FnMut(BlockStateId) -> &'k str,
) {
    let mut keys: Vec<&'k str> = Vec::new();
    let mut index_of: FxHashMap<&'k str, u16> = FxHashMap::default();
    let mut indices = Vec::with_capacity(states.len());
    // Runs of equal states are common; only a change of state needs a key lookup.
    let mut run: Option<(BlockStateId, u16)> = None;
    for &state in states {
        let index = match run {
            Some((run_state, index)) if run_state == state => index,
            _ => {
                let key = key_of(state);
                let index = *index_of.entry(key).or_insert_with(|| {
                    keys.push(key);
                    u16::try_from(keys.len() - 1).expect("at most 65536 distinct keys")
                });
                run = Some((state, index));
                index
            }
        };
        indices.push(index);
    }
    out.extend_from_slice(&(keys.len() as u32).to_le_bytes());
    for key in &keys {
        let length = u32::try_from(key.len()).expect("state key longer than u32::MAX bytes");
        out.extend_from_slice(&length.to_le_bytes());
        out.extend_from_slice(key.as_bytes());
    }
    for index in indices {
        out.extend_from_slice(&index.to_le_bytes());
    }
}
