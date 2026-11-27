//! Palette growth and collapse, semantic equality and the canonical encoding.

use cp_core::coords::{CHUNK_VOLUME, cell_index};
use cp_core::hash::hash_bytes;
use cp_core::ids::BlockStateId;
use cp_voxel::{ChunkData, WORDS_PER_BIT};

use crate::common::{
    AIR, DIRT, GRASS, KEYS, LOG, STATE_COUNT, STONE, WATER, key_of, range_i32, rng,
};

fn canonical(data: &ChunkData) -> Vec<u8> {
    let mut out = Vec::new();
    data.write_canonical(&mut out, key_of);
    out
}

/// A layered terrain column pattern with a few odd cells.
fn terrain(index: usize) -> BlockStateId {
    let [x, y, z] = cp_core::coords::index_to_cell(index).map(u32::from);
    match y {
        0..=11 => STONE,
        12..=14 => DIRT,
        15 => GRASS,
        _ if (x * 7 + z * 3) % 29 == 0 && y < 20 => LOG,
        _ if y < 18 && x > 25 => WATER,
        _ => AIR,
    }
}

#[test]
fn construction_order_does_not_change_the_canonical_form() {
    let reference = ChunkData::from_fn(terrain);
    // Same cells built cell by cell in a scrambled order, starting from another state and
    // passing through extra states that end up unused (free slots, wider indices).
    let mut scrambled = ChunkData::uniform(BlockStateId(9));
    let mut rng = rng(1);
    let mut order: Vec<usize> = (0..CHUNK_VOLUME).collect();
    for i in (1..order.len()).rev() {
        let j = rng.range_u32(0, i as u32 + 1) as usize;
        order.swap(i, j);
    }
    for (step, &index) in order.iter().enumerate() {
        // Detours through other states first.
        scrambled.set(index, BlockStateId((step % STATE_COUNT) as u32));
        scrambled.set(index, BlockStateId(((step * 7) % STATE_COUNT) as u32));
        scrambled.set(index, terrain(index));
    }
    assert_eq!(scrambled, reference, "semantic equality");
    let bytes = canonical(&reference);
    assert_eq!(canonical(&scrambled), bytes);
    // After canonicalisation the two are identical down to palette order and width.
    let mut compacted = scrambled.clone();
    compacted.compact();
    assert_eq!(compacted, reference);
    assert_eq!(canonical(&compacted), bytes);
    scrambled.canonicalize();
    match (&scrambled, &reference) {
        (ChunkData::Paletted(a), ChunkData::Paletted(b)) => {
            assert_eq!(a.palette(), b.palette());
            assert_eq!(a.bits(), b.bits());
            assert_eq!(a.words(), b.words());
        }
        _ => panic!("terrain is paletted"),
    }
    // Ids do not leak into the encoding: a permuted id space with the same keys encodes the
    // same.
    let permuted = ChunkData::from_fn(|index| BlockStateId(100 + terrain(index).0));
    let mut out = Vec::new();
    permuted.write_canonical(&mut out, |state| KEYS[(state.0 - 100) as usize]);
    assert_eq!(out, bytes);
}

#[test]
fn canonical_encoding_layout() {
    let uniform = ChunkData::uniform(STONE);
    let mut expected = Vec::new();
    expected.extend_from_slice(&1_u32.to_le_bytes());
    expected.extend_from_slice(&(KEYS[1].len() as u32).to_le_bytes());
    expected.extend_from_slice(KEYS[1].as_bytes());
    expected.push(0);
    assert_eq!(canonical(&uniform), expected);

    // Air first in cell order, then stone: key indices 0 and 1, one bit per cell.
    let mut two = ChunkData::uniform(STONE);
    two.set(0, AIR);
    two.set(cell_index(1, 31, 31), DIRT);
    let bytes = canonical(&two);
    let mut header = Vec::new();
    header.extend_from_slice(&3_u32.to_le_bytes());
    for key in [KEYS[0], KEYS[1], KEYS[2]] {
        header.extend_from_slice(&(key.len() as u32).to_le_bytes());
        header.extend_from_slice(key.as_bytes());
    }
    header.push(2);
    assert_eq!(&bytes[..header.len()], &header[..]);
    let words = &bytes[header.len()..];
    assert_eq!(words.len(), 2 * WORDS_PER_BIT * 8);
    let word = |i: usize| u64::from_le_bytes(words[i * 8..i * 8 + 8].try_into().unwrap());
    // Cell 0 = key 0 (air), cells 1.. = key 1 (stone): 2 bits per cell, 32 cells per word.
    assert_eq!(word(0), 0x5555_5555_5555_5554);
    let dirt_cell = cell_index(1, 31, 31);
    let (w, bit) = (dirt_cell / 32, (dirt_cell % 32) * 2);
    assert_eq!((word(w) >> bit) & 3, 2);
    // A uniform chunk never depends on how it was reached.
    let mut back = two.clone();
    back.set(0, STONE);
    back.set(cell_index(1, 31, 31), STONE);
    assert!(back.is_uniform());
    assert_eq!(canonical(&back), canonical(&uniform));
}

#[test]
fn equal_keys_merge_in_the_encoding() {
    // Two session ids that name the same persistent key count as one key.
    let data = ChunkData::from_fn(|index| if index % 2 == 0 { STONE } else { DIRT });
    let mut out = Vec::new();
    data.write_canonical(&mut out, |_| "test:same");
    let mut uniform = Vec::new();
    ChunkData::uniform(LOG).write_canonical(&mut uniform, |_| "test:same");
    assert_eq!(out, uniform);
    assert_ne!(hash_bytes(&canonical(&data)), hash_bytes(&uniform));
}

/// Random operations against a plain array model.
#[test]
fn random_operations_match_a_reference_model() {
    let mut rng = rng(2);
    for round in 0..40 {
        let mut model = vec![AIR; CHUNK_VOLUME];
        let mut data = ChunkData::uniform(AIR);
        let palette_size = [2, 3, 5, 17, 300, 5000][round % 6];
        for _ in 0..3000 {
            match rng.range_u32(0, 100) {
                0 => {
                    let state = BlockStateId(rng.range_u32(0, palette_size));
                    data.fill(state);
                    model.fill(state);
                }
                1 => data.compact(),
                2 => data.canonicalize(),
                3..=10 => {
                    // A run of cells, the common case for terrain and builds.
                    let start = rng.range_u32(0, CHUNK_VOLUME as u32) as usize;
                    let length = range_i32(&mut rng, 1, 2000) as usize;
                    let state = BlockStateId(rng.range_u32(0, palette_size));
                    let end = (start + length).min(CHUNK_VOLUME);
                    for (index, cell) in model.iter_mut().enumerate().take(end).skip(start) {
                        data.set(index, state);
                        *cell = state;
                    }
                }
                _ => {
                    let index = rng.range_u32(0, CHUNK_VOLUME as u32) as usize;
                    let state = BlockStateId(rng.range_u32(0, palette_size));
                    data.set(index, state);
                    model[index] = state;
                }
            }
        }
        assert_eq!(data.to_cells(), model, "round {round}");
        let rebuilt = ChunkData::from_cells(&model);
        assert_eq!(data, rebuilt);
        assert_eq!(canonical_ids(&data), canonical_ids(&rebuilt));
        // Uniform iff all cells are equal.
        assert_eq!(data.is_uniform(), model.iter().all(|&s| s == model[0]));
        for state in data.states() {
            assert_eq!(
                data.count(state),
                model.iter().filter(|&&s| s == state).count()
            );
        }
        let mut compacted = data.clone();
        compacted.compact();
        assert!(compacted.memory_bytes() <= data.memory_bytes());
        assert_eq!(compacted, data);
    }
}

/// Canonical bytes with numeric keys, for palettes beyond the test table.
fn canonical_ids(data: &ChunkData) -> Vec<u8> {
    let keys: Vec<String> = (0..6000).map(|i| format!("k{i}")).collect();
    let mut out = Vec::new();
    data.write_canonical(&mut out, |state| keys[state.index()].as_str());
    out
}

#[test]
fn semantic_equality_ignores_representation() {
    let a = ChunkData::from_fn(terrain);
    let mut b = ChunkData::from_fn(|index| BlockStateId(5000 + (index % 7) as u32));
    for index in 0..CHUNK_VOLUME {
        b.set(index, terrain(index));
    }
    assert!(b.bits() >= a.bits());
    assert_eq!(a, b);
    let mut c = b.clone();
    c.set(77, BlockStateId(3));
    assert_ne!(a, c);
    assert_ne!(ChunkData::uniform(STONE), ChunkData::uniform(DIRT));
    assert_ne!(ChunkData::uniform(STONE), a);
}
