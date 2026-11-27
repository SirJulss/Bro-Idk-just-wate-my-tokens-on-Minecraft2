//! Fixtures shared by the integration tests: a small state table and test base sources.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use cp_core::block::{BlockFlags, LodClass, LodInfo};
use cp_core::coords::{CHUNK_VOLUME, LodKey, index_to_cell};
use cp_core::det;
use cp_core::hash::hash_ivec3;
use cp_core::ids::BlockStateId;
use cp_core::math::{IAabb, IVec3};
use cp_voxel::{
    BaseSource, Cancelled, ChunkData, GeneratorIdentity, NodeClass, VoxelChunk, VoxelTables,
    chunk_cells, region_volume,
};

pub const AIR: BlockStateId = BlockStateId(0);
pub const STONE: BlockStateId = BlockStateId(1);
pub const DIRT: BlockStateId = BlockStateId(2);
pub const GRASS: BlockStateId = BlockStateId(3);
pub const WATER: BlockStateId = BlockStateId(4);
pub const LOG: BlockStateId = BlockStateId(5);
pub const LEAVES: BlockStateId = BlockStateId(6);
pub const FLOWER: BlockStateId = BlockStateId(7);
pub const PLANK: BlockStateId = BlockStateId(8);
pub const GLASS: BlockStateId = BlockStateId(9);
pub const LAVA: BlockStateId = BlockStateId(10);
pub const STATE_COUNT: usize = 11;

/// Persistent keys of the test states (sorted like their ids, as the registry does).
pub const KEYS: [&str; STATE_COUNT] = [
    "cubeplanet:air",
    "test:a_stone",
    "test:b_dirt",
    "test:c_grass",
    "test:d_water",
    "test:e_log",
    "test:f_leaves",
    "test:g_flower",
    "test:h_plank",
    "test:i_glass",
    "test:j_lava",
];

/// Key lookup for canonical encodings; the edit sentinel gets a reserved key.
pub fn key_of(state: BlockStateId) -> &'static str {
    if state.is_unchanged() {
        "#unchanged"
    } else {
        KEYS[state.index()]
    }
}

pub fn tables() -> VoxelTables {
    let opaque = BlockFlags::SOLID | BlockFlags::OPAQUE;
    let flags = [
        BlockFlags::empty(),
        opaque,
        opaque,
        opaque,
        BlockFlags::FLUID,
        opaque | BlockFlags::STRUCTURAL,
        BlockFlags::SOLID | BlockFlags::CUTOUT,
        BlockFlags::CUTOUT,
        opaque,
        BlockFlags::SOLID | BlockFlags::CUTOUT,
        BlockFlags::FLUID | BlockFlags::EMISSIVE,
    ];
    let solid = |proxy| LodInfo {
        class: LodClass::Solid,
        proxy,
    };
    let fluid = |proxy| LodInfo {
        class: LodClass::Fluid,
        proxy,
    };
    let lod = [
        LodInfo::EMPTY,
        solid(STONE),
        solid(DIRT),
        solid(GRASS),
        fluid(WATER),
        solid(LOG),
        solid(LEAVES),
        // Flowers vanish at a distance.
        LodInfo::EMPTY,
        solid(PLANK),
        solid(GLASS),
        fluid(LAVA),
    ];
    VoxelTables::new(Arc::from(flags), Arc::from(lod))
}

pub fn identity(key: &str) -> GeneratorIdentity {
    GeneratorIdentity {
        key: key.to_owned(),
        version: 1,
        seed: 7,
        macro_hash: 0,
    }
}

/// Fills `out` row by row from `cell(level, c)`, polling `cancel` per row.
pub fn fill_with(
    level: u8,
    region: IAabb,
    out: &mut [BlockStateId],
    cancel: &dyn Fn() -> bool,
    cell: impl Fn(u8, IVec3) -> BlockStateId,
) -> Result<(), Cancelled> {
    assert_eq!(out.len(), region_volume(region), "fill buffer size");
    let mut index = 0;
    for y in region.min.y..region.max.y {
        for z in region.min.z..region.max.z {
            if cancel() {
                return Err(Cancelled);
            }
            for x in region.min.x..region.max.x {
                out[index] = cell(level, IVec3::new(x, y, z));
                index += 1;
            }
        }
    }
    Ok(())
}

/// A base defined cell by cell by a closure `(level, cell) -> state`; always `Mixed`.
pub struct FnBase<F> {
    identity: GeneratorIdentity,
    cell: F,
}

impl<F: Fn(u8, IVec3) -> BlockStateId + Send + Sync> FnBase<F> {
    pub fn new(name: &str, cell: F) -> Self {
        Self {
            identity: identity(name),
            cell,
        }
    }
}

impl<F: Fn(u8, IVec3) -> BlockStateId + Send + Sync> BaseSource for FnBase<F> {
    fn identity(&self) -> &GeneratorIdentity {
        &self.identity
    }

    fn fill(
        &self,
        level: u8,
        region: IAabb,
        out: &mut [BlockStateId],
        cancel: &dyn Fn() -> bool,
    ) -> Result<(), Cancelled> {
        fill_with(level, region, out, cancel, &self.cell)
    }

    fn classify(&self, _key: LodKey) -> NodeClass {
        NodeClass::Mixed
    }

    fn subtree_empty(&self, _key: LodKey) -> bool {
        false
    }
}

/// Sea level of [`HillsBase`] in metres.
pub const SEA_LEVEL: f64 = 4.0;
/// Highest terrain of [`HillsBase`] (8 + 6 + 5 m) plus room for trunks.
const TOP: f64 = 8.0 + 6.0 + 5.0 + 6.0;
/// Lowest terrain of [`HillsBase`] minus the soil depth.
const BOTTOM: f64 = 8.0 - 6.0 - 5.0 - 4.0;

/// Rolling hills with soil, a sea and trunks, consistent at every level: a cell is decided by
/// the field at its centre (contract §5.1).
pub struct HillsBase {
    identity: GeneratorIdentity,
    /// Classify by the chunk's own column heights (exact-conservative and, like cp-worldgen's
    /// hills generator, one height evaluation per column) instead of the global bounds only.
    column_classify: bool,
}

impl HillsBase {
    pub fn new() -> Self {
        Self {
            identity: identity("test:hills"),
            column_classify: false,
        }
    }

    /// The same landscape whose `classify` evaluates every column of the chunk.
    pub fn column_classified() -> Self {
        Self {
            column_classify: true,
            ..Self::new()
        }
    }

    pub fn height(x: f64, z: f64) -> f64 {
        8.0 + 6.0 * det::sin(x / 23.0) + 5.0 * det::cos(z / 17.0)
    }

    /// Terrain height at the centre of the column `(x, z)` of `level` cells.
    fn column_height(level: u8, x: i32, z: i32) -> f64 {
        let size = f64::from(1_i32 << level);
        Self::height((f64::from(x) + 0.5) * size, (f64::from(z) + 0.5) * size)
    }

    /// The highest ground voxel (level 0) of the column `(x, z)`.
    pub fn ground_top(x: i32, z: i32) -> i32 {
        let height = Self::column_height(0, x, z);
        let mut y = height.floor() as i32;
        while f64::from(y) + 0.5 >= height {
            y -= 1;
        }
        while f64::from(y) + 1.5 < height {
            y += 1;
        }
        y
    }

    /// Exact-conservative class from the chunk's own columns: one height per column.
    fn classify_by_columns(key: LodKey) -> NodeClass {
        let level = key.level;
        let size = f64::from(1_i32 << level);
        let cells = chunk_cells(key);
        let mut low = f64::INFINITY;
        let mut high = f64::NEG_INFINITY;
        for z in cells.min.z..cells.max.z {
            for x in cells.min.x..cells.max.x {
                let height = Self::column_height(level, x, z);
                low = det::min(low, height);
                high = det::max(high, height);
            }
        }
        let lowest = (f64::from(cells.min.y) + 0.5) * size;
        let highest = (f64::from(cells.max.y) - 0.5) * size;
        // Ground (grass, dirt, stone) is opaque; water and trunks only occur above it.
        if highest < low {
            return NodeClass::Opaque;
        }
        // Trunks exist on level 0 only and end 5 m above the ground.
        let trunk_top = if level == 0 { high + 5.0 } else { high };
        if lowest >= high && lowest >= SEA_LEVEL && lowest >= trunk_top {
            NodeClass::Empty
        } else {
            NodeClass::Mixed
        }
    }

    /// The state of cell `c` of `level`.
    pub fn cell(level: u8, c: IVec3) -> BlockStateId {
        Self::cell_in_column(level, c, Self::column_height(level, c.x, c.z))
    }

    fn cell_in_column(level: u8, c: IVec3, height: f64) -> BlockStateId {
        let size = f64::from(1_i32 << level);
        let centre = (c.as_dvec3() + 0.5) * size;
        if centre.y < height {
            let depth = height - centre.y;
            return if depth < 1.0 {
                GRASS
            } else if depth < 4.0 {
                DIRT
            } else {
                STONE
            };
        }
        // Trunks: level 0 only, on hashed columns above dry land.
        if level == 0 && height > SEA_LEVEL && centre.y < height + 5.0 {
            let column = IVec3::new(c.x, 0, c.z);
            if hash_ivec3(11, column).is_multiple_of(61) {
                return LOG;
            }
        }
        if centre.y < SEA_LEVEL { WATER } else { AIR }
    }
}

impl BaseSource for HillsBase {
    fn identity(&self) -> &GeneratorIdentity {
        &self.identity
    }

    fn fill(
        &self,
        level: u8,
        region: IAabb,
        out: &mut [BlockStateId],
        cancel: &dyn Fn() -> bool,
    ) -> Result<(), Cancelled> {
        assert_eq!(out.len(), region_volume(region), "fill buffer size");
        let size = f64::from(1_i32 << level);
        if (f64::from(region.min.y) + 0.5) * size >= TOP {
            // Above every ground, water and trunk cell: all air, decided without the field (the
            // fast path `BaseSource::fill` asks implementations for).
            out.fill(AIR);
            return Ok(());
        }
        // Column-first: one height per (x, z) of the footprint.
        let mut heights = Vec::new();
        for z in region.min.z..region.max.z {
            if cancel() {
                return Err(Cancelled);
            }
            for x in region.min.x..region.max.x {
                heights.push(Self::column_height(level, x, z));
            }
        }
        let mut index = 0;
        for y in region.min.y..region.max.y {
            let mut column = 0;
            for z in region.min.z..region.max.z {
                for x in region.min.x..region.max.x {
                    out[index] = Self::cell_in_column(level, IVec3::new(x, y, z), heights[column]);
                    index += 1;
                    column += 1;
                }
            }
        }
        Ok(())
    }

    fn classify(&self, key: LodKey) -> NodeClass {
        let aabb = key.aabb_voxels();
        if f64::from(aabb.min.y) >= TOP {
            NodeClass::Empty
        } else if f64::from(aabb.max.y) <= BOTTOM {
            NodeClass::Opaque
        } else if self.column_classify {
            Self::classify_by_columns(key)
        } else {
            NodeClass::Mixed
        }
    }

    fn subtree_empty(&self, key: LodKey) -> bool {
        f64::from(key.aabb_voxels().min.y) >= TOP
    }
}

/// Wraps a base, counts `fill` and `classify` calls and optionally sleeps in each fill.
pub struct CountingBase<B> {
    pub inner: B,
    pub fills: AtomicUsize,
    pub classifies: AtomicUsize,
    pub delay: Duration,
}

impl<B: BaseSource> CountingBase<B> {
    pub fn new(inner: B, delay: Duration) -> Self {
        Self {
            inner,
            fills: AtomicUsize::new(0),
            classifies: AtomicUsize::new(0),
            delay,
        }
    }

    pub fn fills(&self) -> usize {
        self.fills.load(Ordering::SeqCst)
    }

    pub fn classifies(&self) -> usize {
        self.classifies.load(Ordering::SeqCst)
    }
}

impl<B: BaseSource> BaseSource for CountingBase<B> {
    fn identity(&self) -> &GeneratorIdentity {
        self.inner.identity()
    }

    fn fill(
        &self,
        level: u8,
        region: IAabb,
        out: &mut [BlockStateId],
        cancel: &dyn Fn() -> bool,
    ) -> Result<(), Cancelled> {
        self.fills.fetch_add(1, Ordering::SeqCst);
        if !self.delay.is_zero() {
            std::thread::sleep(self.delay);
        }
        self.inner.fill(level, region, out, cancel)
    }

    fn classify(&self, key: LodKey) -> NodeClass {
        self.classifies.fetch_add(1, Ordering::SeqCst);
        self.inner.classify(key)
    }

    fn subtree_empty(&self, key: LodKey) -> bool {
        self.inner.subtree_empty(key)
    }
}

/// Wraps a base and answers `classify` exactly, by filling the chunk and looking at every
/// cell: the most precise answer the contract allows (Empty and Opaque wherever true).
pub struct ExactClassify<B>(pub B);

impl<B: BaseSource> BaseSource for ExactClassify<B> {
    fn identity(&self) -> &GeneratorIdentity {
        self.0.identity()
    }

    fn fill(
        &self,
        level: u8,
        region: IAabb,
        out: &mut [BlockStateId],
        cancel: &dyn Fn() -> bool,
    ) -> Result<(), Cancelled> {
        self.0.fill(level, region, out, cancel)
    }

    fn classify(&self, key: LodKey) -> NodeClass {
        let flags = tables().flags;
        let cells = base_chunk(&self.0, key).blocks.to_cells();
        if cells.iter().all(|state| state.is_air()) {
            NodeClass::Empty
        } else if cells
            .iter()
            .all(|state| flags[state.index()].contains(BlockFlags::OPAQUE))
        {
            NodeClass::Opaque
        } else {
            NodeClass::Mixed
        }
    }

    fn subtree_empty(&self, key: LodKey) -> bool {
        self.0.subtree_empty(key)
    }
}

/// One base cell, by a one-cell fill.
pub fn base_cell(base: &dyn BaseSource, level: u8, cell: IVec3) -> BlockStateId {
    let mut out = [AIR];
    base.fill(
        level,
        IAabb::from_min_size(cell, IVec3::ONE),
        &mut out,
        &|| false,
    )
    .expect("never cancelled");
    out[0]
}

/// The base chunk `key`, generated directly.
pub fn base_chunk(base: &dyn BaseSource, key: LodKey) -> VoxelChunk {
    let mut cells = vec![AIR; CHUNK_VOLUME];
    base.fill(key.level, chunk_cells(key), &mut cells, &|| false)
        .expect("never cancelled");
    VoxelChunk::new(ChunkData::from_cells(&cells))
}

/// Global cell coordinates (cells of the key's level) of the chunk cell `index`.
pub fn global_cell(key: LodKey, index: usize) -> IVec3 {
    let [x, y, z] = index_to_cell(index);
    key.pos * 32 + IVec3::new(i32::from(x), i32::from(y), i32::from(z))
}

/// Deterministic test RNG helpers.
pub fn rng(seed: u64) -> cp_core::hash::Rng {
    cp_core::hash::Rng::new(seed, 99)
}

pub fn range_i32(rng: &mut cp_core::hash::Rng, lo: i32, hi: i32) -> i32 {
    lo + rng.range_u32(0, (hi - lo) as u32) as i32
}
