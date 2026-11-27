//! Shared helpers: a small test registry (ids, keys, flags, LOD info), the three sources,
//! chunk fills, the canonical hash and the normative downsample of a parent from its children.

use cp_core::block::{BlockFlags, LodClass, LodInfo};
use cp_core::coords::{CHUNK_VOLUME, LodKey};
use cp_core::hash::hash_bytes;
use cp_core::ids::BlockStateId;
use cp_voxel::{BaseSource, ChunkData, VoxelChunk, chunk_cells, downsample_chunk};
use cp_worldgen::{
    FlatSource, HillsConfig, HillsSource, PaletteKeys, StructureField, TerrainPalette,
};

/// Persistent keys of the test states. Like the content registry, the test registry numbers
/// its states by sorted key (air first), so ties of the downsample rule ("smallest id") fall
/// as they do with the base pack: clay < dirt < grass < gravel < leaves < … < log < … < sand <
/// slate < snow < stone < water. (The relative order of the states the generators had before
/// the birch, spruce, cherry and rock roles came is unchanged.)
pub const KEYS: [&str; 19] = [
    "test:air",
    "test:clay",
    "test:dirt",
    "test:grass",
    "test:gravel",
    "test:leaves",
    "test:leaves_birch",
    "test:leaves_cherry",
    "test:leaves_spruce",
    "test:limestone",
    "test:log",
    "test:log_birch",
    "test:log_cherry",
    "test:log_spruce",
    "test:sand",
    "test:slate",
    "test:snow",
    "test:stone",
    "test:water",
];

pub const AIR: BlockStateId = BlockStateId(0);
pub const CLAY: BlockStateId = BlockStateId(1);
pub const DIRT: BlockStateId = BlockStateId(2);
pub const GRASS: BlockStateId = BlockStateId(3);
pub const GRAVEL: BlockStateId = BlockStateId(4);
pub const LEAVES: BlockStateId = BlockStateId(5);
pub const LEAVES_BIRCH: BlockStateId = BlockStateId(6);
pub const LEAVES_CHERRY: BlockStateId = BlockStateId(7);
pub const LEAVES_SPRUCE: BlockStateId = BlockStateId(8);
pub const LIMESTONE: BlockStateId = BlockStateId(9);
pub const LOG: BlockStateId = BlockStateId(10);
pub const LOG_BIRCH: BlockStateId = BlockStateId(11);
pub const LOG_CHERRY: BlockStateId = BlockStateId(12);
pub const LOG_SPRUCE: BlockStateId = BlockStateId(13);
pub const SAND: BlockStateId = BlockStateId(14);
pub const SLATE: BlockStateId = BlockStateId(15);
pub const SNOW: BlockStateId = BlockStateId(16);
pub const STONE: BlockStateId = BlockStateId(17);
pub const WATER: BlockStateId = BlockStateId(18);

/// The palette keys of the test registry.
pub const PALETTE_KEYS: PaletteKeys<'static> = PaletteKeys {
    keys: [
        "test:grass",
        "test:dirt",
        "test:stone",
        "test:sand",
        "test:gravel",
        "test:snow",
        "test:water",
        "test:log",
        "test:leaves",
        "test:log_birch",
        "test:leaves_birch",
        "test:log_spruce",
        "test:leaves_spruce",
        "test:log_cherry",
        "test:leaves_cherry",
        "test:clay",
        "test:slate",
        "test:limestone",
    ],
};

pub fn palette() -> TerrainPalette {
    TerrainPalette::resolve(&PALETTE_KEYS, |key| {
        KEYS.iter()
            .position(|&k| k == key)
            .map(|index| BlockStateId(index as u32))
    })
    .expect("the test registry has every palette key")
}

/// The leaves of the four species.
pub const ALL_LEAVES: [BlockStateId; 4] = [LEAVES, LEAVES_BIRCH, LEAVES_SPRUCE, LEAVES_CHERRY];
/// The logs of the four species.
pub const ALL_LOGS: [BlockStateId; 4] = [LOG, LOG_BIRCH, LOG_SPRUCE, LOG_CHERRY];

/// Flags as the base pack derives them: cube states solid and opaque, logs structural, leaves
/// cutout, water fluid.
pub fn flags() -> Vec<BlockFlags> {
    let cube = BlockFlags::SOLID | BlockFlags::OPAQUE;
    (0..KEYS.len() as u32)
        .map(|id| match BlockStateId(id) {
            AIR => BlockFlags::empty(),
            WATER => BlockFlags::FLUID,
            state if ALL_LEAVES.contains(&state) => BlockFlags::SOLID | BlockFlags::CUTOUT,
            state if ALL_LOGS.contains(&state) => cube | BlockFlags::STRUCTURAL,
            _ => cube,
        })
        .collect()
}

/// LOD info as the base pack derives it: every state is its own proxy.
pub fn lod() -> Vec<LodInfo> {
    (0..KEYS.len() as u32)
        .map(|id| {
            let state = BlockStateId(id);
            LodInfo {
                class: class(state),
                proxy: state,
            }
        })
        .collect()
}

pub fn class(state: BlockStateId) -> LodClass {
    match state {
        AIR => LodClass::Empty,
        WATER => LodClass::Fluid,
        _ => LodClass::Solid,
    }
}

pub fn is_opaque(state: BlockStateId) -> bool {
    flags()[state.index()].contains(BlockFlags::OPAQUE)
}

/// The ground states (everything a column of terrain is made of).
pub fn is_ground(state: BlockStateId) -> bool {
    [
        STONE, DIRT, GRASS, SAND, GRAVEL, SNOW, CLAY, SLATE, LIMESTONE,
    ]
    .contains(&state)
}

pub fn hills(seed: u64) -> HillsSource {
    HillsSource::new(seed, HillsConfig::default(), palette(), &flags())
        .expect("the default hills are valid")
}

pub fn hills_without_trees(seed: u64) -> HillsSource {
    let config = HillsConfig {
        tree_density: 0.0,
        ..HillsConfig::default()
    };
    HillsSource::new(seed, config, palette(), &flags()).expect("valid hills without trees")
}

pub fn flat() -> FlatSource {
    FlatSource::grassland(0, &palette(), 6, &flags()).expect("valid grassland")
}

pub fn structures() -> StructureField {
    StructureField::new(0, palette(), &flags()).expect("valid structure field")
}

/// The 32³ cells of `key`.
pub fn chunk(source: &dyn BaseSource, key: LodKey) -> Vec<BlockStateId> {
    let mut cells = vec![BlockStateId::UNCHANGED; CHUNK_VOLUME];
    source
        .fill(key.level, chunk_cells(key), &mut cells, &|| false)
        .expect("never cancelled");
    assert!(cells.iter().all(|state| !state.is_unchanged()));
    cells
}

/// FNV-1a of the canonical encoding (contract §2.5 rule 5, `ChunkData::write_canonical`):
/// independent of session-local ids.
pub fn canonical_hash(cells: &[BlockStateId]) -> u64 {
    let mut bytes = Vec::new();
    ChunkData::from_cells(cells).write_canonical(&mut bytes, |state| KEYS[state.index()]);
    hash_bytes(&bytes)
}

/// The parent `key` downsampled from its eight generated children by the normative rule.
pub fn downsampled(source: &dyn BaseSource, key: LodKey) -> Vec<BlockStateId> {
    let children: Vec<VoxelChunk> = key
        .children()
        .iter()
        .map(|&child| VoxelChunk::new(ChunkData::from_cells(&chunk(source, child))))
        .collect();
    let children: [&VoxelChunk; 8] = core::array::from_fn(|i| &children[i]);
    downsample_chunk(children, key.level, &flags(), &lod())
        .blocks
        .to_cells()
}
