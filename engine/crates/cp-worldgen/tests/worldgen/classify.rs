//! `classify` and `subtree_empty` never lie (contract §5.1): every `Empty` chunk is checked to
//! hold only air and every `Opaque` chunk only `OPAQUE` cells, cell by cell; every key that
//! `subtree_empty` calls empty has sampled descendants on every level checked the same way.

use cp_core::coords::LodKey;
use cp_core::hash::{Rng, hash_combine};
use cp_core::math::IVec3;
use cp_voxel::{BaseSource, NodeClass};

use crate::common::{chunk, flat, hills, hills_without_trees, is_opaque, structures};

#[derive(Default, Debug)]
struct Counts {
    empty: usize,
    opaque: usize,
    mixed: usize,
}

/// Checks the classification of every key against its filled cells.
fn check_classify(source: &dyn BaseSource, keys: &[LodKey]) -> Counts {
    let mut counts = Counts::default();
    for &key in keys {
        match source.classify(key) {
            NodeClass::Empty => {
                counts.empty += 1;
                let cells = chunk(source, key);
                assert!(cells.iter().all(|s| s.is_air()), "{key:?} is not empty");
            }
            NodeClass::Opaque => {
                counts.opaque += 1;
                let cells = chunk(source, key);
                assert!(cells.iter().all(|&s| is_opaque(s)), "{key:?} is not opaque");
            }
            NodeClass::Mixed => counts.mixed += 1,
        }
    }
    counts
}

/// Keys of levels 0–3 on a block around the origin, from below the ground to the sky.
fn block_keys() -> Vec<LodKey> {
    let mut keys = Vec::new();
    for level in 0..=3 {
        for y in -2..=3 {
            for z in -3..=3 {
                for x in -3..=3 {
                    keys.push(LodKey::new(level, IVec3::new(x, y, z)));
                }
            }
        }
    }
    keys
}

/// Random keys of levels 0–6 across a wide area and height range.
fn random_keys(seed: u64, count: usize) -> Vec<LodKey> {
    let mut rng = Rng::new(seed, 1);
    (0..count)
        .map(|_| {
            let level = rng.range_u32(0, 7) as u8;
            let reach = (8192 >> level) / 32;
            let mut coordinate = |span: i32| rng.range_u32(0, 2 * span as u32) as i32 - span;
            let pos = IVec3::new(coordinate(reach), 0, coordinate(reach));
            let y = coordinate(4.max(160 >> (5 + level)));
            LodKey::new(level, pos.with_y(y))
        })
        .collect()
}

#[test]
fn hills_classify_never_lies() {
    let source = hills(42);
    let counts = check_classify(&source, &block_keys());
    assert!(
        counts.empty > 150 && counts.opaque > 150 && counts.mixed > 150,
        "{counts:?}"
    );
    let counts = check_classify(&source, &random_keys(7, 600));
    assert!(counts.empty > 50 && counts.opaque > 50, "{counts:?}");
    // Without trees the chunks right above the ground can be empty as well.
    let counts = check_classify(&hills_without_trees(42), &block_keys());
    assert!(counts.empty > 150, "{counts:?}");
}

#[test]
fn flat_and_structure_classify_never_lie() {
    for source in [&flat() as &dyn BaseSource, &structures()] {
        let counts = check_classify(source, &block_keys());
        assert!(counts.empty > 100 && counts.opaque > 100, "{counts:?}");
    }
}

/// For every key that `subtree_empty` calls empty: the key itself and a few descendant chains
/// down to level 0 are all air.
fn check_subtree(source: &dyn BaseSource, keys: &[LodKey]) -> usize {
    let mut empty = 0;
    for &key in keys {
        if !source.subtree_empty(key) {
            continue;
        }
        empty += 1;
        assert_eq!(source.classify(key), NodeClass::Empty, "{key:?}");
        assert!(chunk(source, key).iter().all(|s| s.is_air()), "{key:?}");
        // Three chains of descendants; one of them always takes the lowest child (`0`, nearest
        // the ground), the others are chosen by a hash.
        for chain in 0..3_u64 {
            let mut node = key;
            while node.level > 0 {
                let choice = hash_combine(chain, u64::from(node.level)) % 8;
                let octant = if chain == 0 { 0 } else { choice as usize };
                node = node.children()[octant];
                let cells = chunk(source, node);
                assert!(cells.iter().all(|s| s.is_air()), "{key:?} -> {node:?}");
            }
        }
    }
    empty
}

/// Keys of levels 1–8 in a column over the origin and its neighbours, from the ground up.
fn column_keys() -> Vec<LodKey> {
    let mut keys = Vec::new();
    for level in 0..=8_u8 {
        let extent = 32 << level;
        for y in 0..(1024 / extent).max(2) + 1 {
            for (x, z) in [(0, 0), (-1, 0), (0, 1), (2, -3)] {
                keys.push(LodKey::new(level, IVec3::new(x, y, z)));
            }
        }
    }
    keys
}

#[test]
fn subtree_empty_never_lies() {
    let source = hills(42);
    let empty = check_subtree(&source, &column_keys());
    assert!(empty > 20, "{empty}");
    // The answer is exact on level 0 and switches to true right above the analytic bound.
    for key in column_keys().into_iter().filter(|key| key.level == 0) {
        assert_eq!(
            source.subtree_empty(key),
            source.classify(key) == NodeClass::Empty,
            "{key:?}"
        );
    }
    // Keys that hold ground are never subtree-empty.
    assert!(!source.subtree_empty(LodKey::new(5, IVec3::new(0, 0, 0))));
    assert!(!source.subtree_empty(LodKey::new(0, IVec3::new(0, 0, 0))));
    for source in [
        &flat() as &dyn BaseSource,
        &structures(),
        &hills_without_trees(3),
    ] {
        let empty = check_subtree(source, &column_keys());
        assert!(empty > 20, "{empty}");
    }
}
