//! The hills test landscape as a [`BaseSource`].

use cp_core::block::BlockFlags;
use cp_core::coords::LodKey;
use cp_core::det;
use cp_core::ids::BlockStateId;
use cp_core::math::IAabb;
use cp_voxel::{BaseSource, Cancelled, GeneratorIdentity, NodeClass, chunk_cells};

use crate::config_hash::ConfigHasher;
use crate::error::WorldgenError;
use crate::palette::{TerrainPalette, is_opaque};
use crate::region::{Region, cell_size, centre, checked_request, last_centre_below};
use crate::strata::Column;
use crate::terrain::{HillsConfig, Terrain};
use crate::trees::{
    Canvas, LANDSCAPE_TREES, TREE_CANOPY_MAX_LEVEL, TREE_MAX_HEIGHT, TreeInfo, TreePlacer,
};

/// Columns of the ring around a filled region whose heights settle the region's columns
/// ([`Terrain::settle`] reads two columns to each side).
const SETTLE_RING: usize = 2;

/// A deterministic test landscape on a plane (`+Y` up): rolling hills, a few higher ridge
/// ranges, water up to a sea level, beaches, stony and muddy shores, bare rock and scree on
/// steep slopes, snow caps, dry and lush regions, and mixed woods of oak, birch, spruce and
/// cherry with bushes and boulders ([`HillsConfig`]).
///
/// # Levels (contract §5.1)
/// Column-first: one evaluation of the band-limited height field per column of the requested
/// region (and of the ring of columns around it, for the slopes), at the column centre; a cell
/// is ground iff its centre lies below that height, water iff it is not ground and its centre
/// lies below the sea level. Ground columns are layered by their surface kind, which follows
/// the column's height, its slope (from the neighbouring columns of the same level) and
/// band-limited noise (see `terrain.rs`): grass over dirt over stone, sand on beaches and
/// lake floors, gravel on deep floors, stony shores and scree, clay on muddy shores, snow on
/// high ground, bare stone, slate or limestone on steep slopes. Features (trees, bushes,
/// boulders) have a declared radius (see [`crate::TREE_MAX_RADIUS`]); trees are rasterised up
/// to level 1 with a canopy stand-in up to level 3, bushes and boulders up to level 1.
///
/// # Cost
/// `fill` evaluates the field once per column of the region and its two-column ring, except
/// for rows that are stone in every column whatever its height (below every layer of the
/// lowest surface the level can have, [`HillsSource::height_bounds`]) and regions above
/// everything the landscape holds: those are filled without evaluating anything.
///
/// # Classification
/// [`BaseSource::classify`] is exact-conservative: a chunk is `Opaque` when all its cell
/// centres lie below the lowest of its columns (and the ground states are opaque), `Empty` when
/// they lie above the highest column and the sea level and no tree can reach in, `Mixed`
/// otherwise. It decides from the analytic height bounds of the level where it can and
/// otherwise evaluates the chunk's own columns, stopping at the first column that rules out
/// both `Empty` and `Opaque`; a chunk that the surface crosses therefore costs a few columns.
/// [`BaseSource::subtree_empty`] decides with the analytic bound of the field over all levels,
/// and exactly for level-0 keys.
#[derive(Clone, Debug)]
pub struct HillsSource {
    identity: GeneratorIdentity,
    config: HillsConfig,
    palette: TerrainPalette,
    terrain: Terrain,
    trees: Option<TreePlacer>,
    /// All ground states are `OPAQUE`.
    ground_opaque: bool,
    /// Every feature cell's centre lies below this height (m).
    tree_top: f64,
    /// Every non-air cell's centre lies below this height (m), on every level.
    top_bound: f64,
}

impl HillsSource {
    /// Identity key of this generator.
    pub const KEY: &'static str = "cubeplanet:hills";
    /// Output version; bumped on every change of the generated cells. 2: surfaces by slope and
    /// noise, regions, four tree species, bushes and boulders (beauty review E2, round 1).
    /// 3: settled columns, damped fine octaves, warped ridges, beaches by width, dunes, spruce
    /// cones, copses, boulder fields, bushes of several mounds and crowns drawn for the distance
    /// (beauty review E2, round 2). 4: micro relief in hummocky regions, warped hills, beaches
    /// decided by whole cells and narrower on most shores, clumped forests, trees anywhere in
    /// their grid cell, sizes that vary more, young trees and veterans (beauty review E2,
    /// round 3).
    pub const VERSION: u32 = 4;

    /// The landscape of `config` for the world seed `seed`. `flags` is the per-state flag table
    /// of the registry the palette's ids come from; it decides which states count as `OPAQUE`
    /// for [`BaseSource::classify`].
    ///
    /// # Errors
    /// [`WorldgenError::InvalidConfig`] when a parameter is out of its documented range or the
    /// palette contains [`BlockStateId::UNCHANGED`].
    pub fn new(
        seed: u64,
        config: HillsConfig,
        palette: TerrainPalette,
        flags: &[BlockFlags],
    ) -> Result<Self, WorldgenError> {
        config.validate()?;
        palette.validate()?;
        let mut hasher = ConfigHasher::new(Self::KEY);
        config.hash_into(&mut hasher);
        hasher.u64(palette.key_hash);
        let identity = GeneratorIdentity {
            key: Self::KEY.to_owned(),
            version: Self::VERSION,
            seed,
            macro_hash: hasher.finish(),
        };
        let terrain = Terrain::new(seed, &config);
        let trees = (config.tree_density > 0.0).then(|| TreePlacer::new(seed, config.tree_density));
        let ground_opaque = [
            palette.grass,
            palette.dirt,
            palette.stone,
            palette.sand,
            palette.gravel,
            palette.snow,
            palette.clay,
            palette.slate,
            palette.limestone,
        ]
        .iter()
        .all(|&state| is_opaque(flags, state));
        // Features stand on grass, which ends below a bound near the snow line, and below the
        // highest ground. A feature cell's centre lies less than TREE_MAX_HEIGHT above the
        // base, which lies less than one metre above the ground height; 8 m covers the cells
        // of the coarsest canopy level.
        let tree_top = det::min(terrain.grass_top(), terrain.max_height(0))
            + 1.0
            + f64::from(TREE_MAX_HEIGHT)
            + f64::from(1_u32 << TREE_CANOPY_MAX_LEVEL);
        let mut top_bound = det::max(terrain.max_height_all_levels(), terrain.sea_level());
        if trees.is_some() {
            top_bound = det::max(top_bound, tree_top);
        }
        log::debug!("hills source {identity:?}");
        Ok(Self {
            identity,
            config,
            palette,
            terrain,
            trees,
            ground_opaque,
            tree_top,
            top_bound,
        })
    }

    /// The parameters.
    #[must_use]
    pub fn config(&self) -> &HillsConfig {
        &self.config
    }

    /// The block states.
    #[must_use]
    pub fn palette(&self) -> &TerrainPalette {
        &self.palette
    }

    /// Surface height (m) of the band-limited field of `level` at `(x, z)` m: the cells of
    /// `level` whose centre lies below it are ground, when `(x, z)` is their column centre.
    ///
    /// # Panics
    /// If `level > MAX_LOD_LEVEL`.
    #[must_use]
    pub fn height(&self, level: u8, x: f64, z: f64) -> f64 {
        self.terrain.height(level, x, z)
    }

    /// Bounds `(low, high)` in m of [`HillsSource::height`] on `level` over the whole plane.
    ///
    /// # Panics
    /// If `level > MAX_LOD_LEVEL`.
    #[must_use]
    pub fn height_bounds(&self, level: u8) -> (f64, f64) {
        (
            self.terrain.min_height(level),
            self.terrain.max_height(level),
        )
    }

    /// The highest ground voxel of the column `(x, z)` at level 0 (trees and water excluded).
    #[must_use]
    pub fn ground_top(&self, x: i32, z: i32) -> i32 {
        let height = self.terrain.settled_height(0, i64::from(x), i64::from(z));
        // The validated configuration keeps heights within ±1e6 m.
        last_centre_below(0, height) as i32
    }

    /// The wood density at `(x, z)` m in `[0, 1]`: near 1 in the cores of the forests, the
    /// share of tree cells elsewhere (copses, lone trees). 0 without trees.
    #[must_use]
    pub fn wood(&self, x: f64, z: f64) -> f64 {
        self.trees
            .as_ref()
            .map_or(0.0, |placer| placer.wood(x, z, &self.terrain))
    }

    /// The mean height of the snow line (m).
    #[must_use]
    pub fn snow_line(&self) -> f64 {
        self.terrain.snow_line()
    }

    /// The dryness of the region around `(x, z)` m at level 0, in `[-1, 1]`: positive in the
    /// dry heaths (sparse trees, mostly birch, sand in the meadows), negative in the lush
    /// lowlands. Regions are about 1.3 km across.
    #[must_use]
    pub fn dryness(&self, x: f64, z: f64) -> f64 {
        self.terrain.dryness(0, x, z)
    }

    /// The slope (rise over run) of the level-0 ground at the column centre `(x, z)` m.
    #[must_use]
    pub fn slope(&self, x: f64, z: f64) -> f64 {
        self.terrain.slope_at(0, x, z)
    }

    /// The state of the highest ground voxel of the column `(x, z)` at level 0 (grass, sand,
    /// snow, …; trees and water excluded).
    #[must_use]
    pub fn ground_state(&self, x: i32, z: i32) -> BlockStateId {
        let (cx, cz) = (f64::from(x) + 0.5, f64::from(z) + 0.5);
        let height = self.terrain.settled_height(0, i64::from(x), i64::from(z));
        let slope = self.terrain.slope_at(0, cx, cz);
        let surface = self.terrain.surface(0, cx, cz, height, slope);
        self.terrain.layers(surface, &self.palette)[0].0
    }

    /// The features (trees, bushes, boulders) whose trunk column lies inside the x/z extent of
    /// `voxels` (its y extent is ignored), ordered by grid cell (z, then x).
    #[must_use]
    pub fn trees_in(&self, voxels: IAabb) -> Vec<TreeInfo> {
        let Some(placer) = &self.trees else {
            return Vec::new();
        };
        placer
            .trees_in(
                [voxels.min.x.into(), voxels.max.x.into()],
                [voxels.min.z.into(), voxels.max.z.into()],
                &self.terrain,
            )
            .iter()
            .map(|tree| tree.info())
            .collect()
    }

    fn opaque_ground(&self) -> NodeClass {
        if self.ground_opaque {
            NodeClass::Opaque
        } else {
            NodeClass::Mixed
        }
    }

    /// Scans the columns of `cells` (one height each) for what the terrain makes of them:
    /// returns `(may_be_empty, may_be_opaque)`, where `may_be_empty` starts as `empty_possible`
    /// and turns false at the first column with ground above `lowest` (the region's lowest cell
    /// centre), and `may_be_opaque` turns false at the first column that ends at or below
    /// `highest` (its highest cell centre). Stops as soon as both are false: the region is then
    /// mixed, whatever the remaining columns hold. A settled column takes one of the heights
    /// within [`SETTLE_RING`] columns of it, so the scan covers that ring around the region.
    fn scan_columns(
        &self,
        level: u8,
        cells: &Region,
        [lowest, highest]: [f64; 2],
        empty_possible: bool,
    ) -> (bool, bool) {
        let ring = SETTLE_RING as i64;
        let mut may_be_empty = empty_possible;
        let mut may_be_opaque = true;
        for z in cells.min[2] - ring..cells.max[2] + ring {
            for x in cells.min[0] - ring..cells.max[0] + ring {
                let height = self
                    .terrain
                    .height(level, centre(level, x), centre(level, z));
                // The point rule: the column's lowest cell is ground iff lowest < height, its
                // highest cell iff highest < height.
                may_be_empty &= height <= lowest;
                may_be_opaque &= highest < height;
                if !may_be_empty && !may_be_opaque {
                    return (false, false);
                }
            }
        }
        (may_be_empty, may_be_opaque)
    }

    /// True when some tree may draw a cell into `cells` (conservative: its cell box meets
    /// them). `lowest` is the lowest cell centre of the region.
    fn trees_may_reach(&self, level: u8, cells: &Region, lowest: f64) -> bool {
        let Some(placer) = &self.trees else {
            return false;
        };
        level <= TREE_CANOPY_MAX_LEVEL
            && lowest < self.tree_top
            && !placer
                .trees_for(level, cells, LANDSCAPE_TREES, &self.terrain)
                .is_empty()
    }

    /// Draws the trees into a filled region whose terrain columns are `columns`.
    fn grow_trees(
        &self,
        canvas: &mut Canvas<'_>,
        columns: &[Column],
        cancel: &dyn Fn() -> bool,
    ) -> Result<(), Cancelled> {
        let Some(placer) = &self.trees else {
            return Ok(());
        };
        let level = canvas.level;
        let region = canvas.region;
        let lowest_open = columns
            .iter()
            .map(Column::open_from)
            .min()
            .unwrap_or(i64::MAX);
        if level > TREE_CANOPY_MAX_LEVEL
            || centre(level, region.min[1]) >= self.tree_top
            || region.max[1] <= lowest_open
        {
            // No tree cell can lie here, or every cell is ground or water already.
            return Ok(());
        }
        let trees = placer.trees_for(level, &region, LANDSCAPE_TREES, &self.terrain);
        if cancel() {
            return Err(Cancelled);
        }
        // Region extents fit in i64; column index of (x, z) in the column list.
        let nx = region.max[0] - region.min[0];
        let writable = |x: i64, y: i64, z: i64| {
            let index = (z - region.min[2]) * nx + (x - region.min[0]);
            y >= columns[index as usize].open_from()
        };
        // On level 0 crowns (and boulders) first, then trunks: logs win over leaves whatever
        // the order of the features. From level 1 on the other way round: leaves win, so that
        // a coarse trunk does not stick out of its crown. Among crowns the later feature in
        // grid order wins.
        for part in LANDSCAPE_TREES.drawing_order(level) {
            for tree in &trees {
                tree.draw(canvas, LANDSCAPE_TREES, part, &self.palette, writable);
            }
        }
        Ok(())
    }
}

impl BaseSource for HillsSource {
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
        let region = checked_request(level, region, out);
        if region.is_empty() {
            return Ok(());
        }
        if centre(level, region.min[1]) >= self.top_bound {
            // Above every ground, water and tree cell of every level: all air, exactly.
            out.fill(BlockStateId::AIR);
            return Ok(());
        }
        // Rows up to here are stone in every column: below every layer of the lowest surface
        // the level can have. Trees and water only take cells the ground leaves open.
        let base_top = self.terrain.base_top(level);
        if region.max[1] - 1 <= base_top {
            out.fill(self.palette.stone);
            return Ok(());
        }
        // Column-first: one height per column and per column of the two rings around the
        // region (the neighbours that settle the columns and give the slopes), then the
        // settled columns, then the cells in memory order.
        let (nx, nz) = (region.extent(0), region.extent(2));
        let ring = SETTLE_RING;
        let (width, depth) = (nx + 2 * ring, nz + 2 * ring);
        let reach = ring as i64;
        let mut heights = Vec::with_capacity(width * depth);
        for z in region.min[2] - reach..region.max[2] + reach {
            for x in region.min[0] - reach..region.max[0] + reach {
                if cancel() {
                    return Err(Cancelled);
                }
                heights.push(
                    self.terrain
                        .height(level, centre(level, x), centre(level, z)),
                );
            }
        }
        let settled = Terrain::settle(level, &heights, width, depth);
        let cell = cell_size(level);
        let mut columns = Vec::with_capacity(nx * nz);
        for (j, z) in (region.min[2]..region.max[2]).enumerate() {
            for (i, x) in (region.min[0]..region.max[0]).enumerate() {
                // The slope of the unsettled field, from the neighbouring columns.
                let at = |di: usize, dj: usize| heights[(j + dj) * width + i + di];
                let slope = Terrain::slope(
                    [
                        at(ring - 1, ring),
                        at(ring + 1, ring),
                        at(ring, ring - 1),
                        at(ring, ring + 1),
                    ],
                    cell,
                );
                columns.push(self.terrain.column(
                    level,
                    (centre(level, x), centre(level, z)),
                    settled[j * nx + i],
                    slope,
                    &self.palette,
                ));
            }
        }
        for (row, y) in out
            .chunks_exact_mut(columns.len())
            .zip(region.min[1]..region.max[1])
        {
            if y <= base_top {
                row.fill(self.palette.stone);
                continue;
            }
            for (cell, column) in row.iter_mut().zip(&columns) {
                *cell = column.state(y);
            }
        }
        let mut canvas = Canvas { level, region, out };
        self.grow_trees(&mut canvas, &columns, cancel)
    }

    fn classify(&self, key: LodKey) -> NodeClass {
        if !key.is_valid() {
            return NodeClass::Mixed;
        }
        let level = key.level;
        let cells = Region::from_aabb(chunk_cells(key));
        let lowest = centre(level, cells.min[1]);
        let highest = centre(level, cells.max[1] - 1);
        // The top cell of a column that lies level with the water holds water (the shallows):
        // a chunk that may contain such a cell is never opaque.
        let shallows = highest >= self.terrain.sea_level() - cell_size(level);
        // Decided by the analytic bounds, without evaluating the field.
        if highest < self.terrain.min_height(level) && !shallows {
            return self.opaque_ground();
        }
        if lowest >= self.top_bound {
            return NodeClass::Empty;
        }
        // Open cells below the sea level hold water: such a chunk is never empty.
        let above_sea = lowest >= self.terrain.sea_level();
        // No column of this level reaches the lowest centre: no ground and no water.
        let above_ground = above_sea && lowest >= self.terrain.max_height(level);
        let (may_be_empty, may_be_opaque) = if above_ground {
            (true, false)
        } else {
            // Exact: this chunk's own columns, until the answer is known to be Mixed.
            self.scan_columns(level, &cells, [lowest, highest], above_sea)
        };
        if may_be_opaque && !shallows {
            self.opaque_ground()
        } else if !may_be_empty || self.trees_may_reach(level, &cells, lowest) {
            NodeClass::Mixed
        } else {
            NodeClass::Empty
        }
    }

    fn subtree_empty(&self, key: LodKey) -> bool {
        if !key.is_valid() {
            return false;
        }
        // Every cell of the key and of its descendants has its centre at least half a metre
        // above the key's floor.
        let lowest = f64::from(key.origin_voxel().y) + 0.5;
        if lowest >= self.top_bound {
            return true;
        }
        // Finer levels see more octaves than any bound evaluated here, so below the global
        // bound only a level-0 key (no descendants) can be decided, exactly.
        key.level == 0 && self.classify(key) == NodeClass::Empty
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::palette::{ROLE_COUNT, numbered_palette};
    use crate::strata::layered_state;
    use crate::trees::{Tree, TreePart, TreeStyle};
    use cp_core::coords::MAX_LOD_LEVEL;
    use cp_core::math::IVec3;

    const AIR: BlockStateId = BlockStateId::AIR;

    /// Role `i` is state `i + 1`: grass 1, dirt 2, stone 3, sand 4, gravel 5, snow 6, water 7,
    /// oak log 8, oak leaves 9, birch 10 and 11, spruce 12 and 13, cherry 14 and 15, clay 16,
    /// slate 17, limestone 18.
    fn palette() -> TerrainPalette {
        numbered_palette()
    }

    fn flags() -> Vec<BlockFlags> {
        let solid = BlockFlags::SOLID | BlockFlags::OPAQUE;
        let palette = palette();
        let mut flags = vec![solid; ROLE_COUNT + 1];
        flags[0] = BlockFlags::empty();
        flags[palette.water.index()] = BlockFlags::FLUID;
        for log in [
            palette.log,
            palette.log_birch,
            palette.log_spruce,
            palette.log_cherry,
        ] {
            flags[log.index()] = solid | BlockFlags::STRUCTURAL;
        }
        for leaves in [
            palette.leaves,
            palette.leaves_birch,
            palette.leaves_spruce,
            palette.leaves_cherry,
        ] {
            flags[leaves.index()] = BlockFlags::SOLID | BlockFlags::CUTOUT;
        }
        flags
    }

    /// The states that features are made of.
    fn is_feature(palette: &TerrainPalette, state: BlockStateId) -> bool {
        [
            palette.log,
            palette.leaves,
            palette.log_birch,
            palette.leaves_birch,
            palette.log_spruce,
            palette.leaves_spruce,
            palette.log_cherry,
            palette.leaves_cherry,
        ]
        .contains(&state)
    }

    fn source(seed: u64) -> HillsSource {
        HillsSource::new(seed, HillsConfig::default(), palette(), &flags()).unwrap()
    }

    /// The last feature of `trees` (grid order) that draws `part` into the cell, with `style`.
    fn feature_state(
        trees: &[Tree],
        level: u8,
        style: TreeStyle,
        part: TreePart,
        cell: [i64; 3],
        palette: &TerrainPalette,
    ) -> Option<BlockStateId> {
        trees
            .iter()
            .rev()
            .find(|t| t.has_cell(level, style, part, cell))
            .map(|t| t.state(part, palette))
    }

    /// The landscape cell by cell, the slow and obvious way: the point rule per cell with the
    /// slope from the four neighbouring columns, every feature of a wide window tested against
    /// every cell (trunks win over crowns; among crowns the last feature in grid order).
    fn reference(source: &HillsSource, level: u8, region: &Region) -> Vec<BlockStateId> {
        let voxels = region.voxels(level);
        let margin = 64;
        let trees: Vec<Tree> = source.trees.as_ref().map_or_else(Vec::new, |placer| {
            placer.trees_in_level(
                level,
                [voxels.min[0] - margin, voxels.max[0] + margin],
                [voxels.min[2] - margin, voxels.max[2] + margin],
                &source.terrain,
            )
        });
        let palette = source.palette;
        let terrain = &source.terrain;
        let mut out = Vec::with_capacity(region.volume());
        for y in region.min[1]..region.max[1] {
            for z in region.min[2]..region.max[2] {
                for x in region.min[0]..region.max[0] {
                    let (cx, cz) = (centre(level, x), centre(level, z));
                    let height = terrain.settled_height(level, x, z);
                    let slope = terrain.slope_at(level, cx, cz);
                    let layers =
                        terrain.layers(terrain.surface(level, cx, cz, height, slope), &palette);
                    let cell = [x, y, z];
                    // The part drawn last wins: logs on level 0, leaves from level 1 on.
                    let [first, last] = LANDSCAPE_TREES.drawing_order(level);
                    let feature =
                        |part| feature_state(&trees, level, LANDSCAPE_TREES, part, cell, &palette);
                    let state = if let Some(ground) =
                        layered_state(level, y, height, &layers, palette.stone)
                    {
                        ground
                    } else if centre(level, y) < terrain.sea_level() {
                        palette.water
                    } else if let Some(state) = feature(last) {
                        state
                    } else if let Some(state) = feature(first) {
                        state
                    } else {
                        AIR
                    };
                    out.push(state);
                }
            }
        }
        out
    }

    fn fill(source: &HillsSource, level: u8, region: &Region) -> Vec<BlockStateId> {
        let mut out = vec![BlockStateId::UNCHANGED; region.volume()];
        let aabb = IAabb::new(
            IVec3::new(
                region.min[0] as i32,
                region.min[1] as i32,
                region.min[2] as i32,
            ),
            IVec3::new(
                region.max[0] as i32,
                region.max[1] as i32,
                region.max[2] as i32,
            ),
        );
        source.fill(level, aabb, &mut out, &|| false).unwrap();
        out
    }

    /// A region of `size` cells around the surface at the voxel column `(x, z)`.
    fn surface_region(source: &HillsSource, level: u8, x: i64, z: i64, size: [i64; 3]) -> Region {
        let height = source.height(level, x as f64, z as f64);
        let cy = last_centre_below(level, height);
        let c = [x >> level, cy, z >> level];
        Region::new(
            [c[0] - size[0] / 2, c[1] - size[1] / 2, c[2] - size[2] / 2],
            [
                c[0] + (size[0] + 1) / 2,
                c[1] + (size[1] + 1) / 2,
                c[2] + (size[2] + 1) / 2,
            ],
        )
    }

    #[test]
    fn fill_equals_the_reference_on_every_tree_level() {
        let source = source(42);
        // Find a forest, a coast and a ridge.
        let trees = source.trees_in(IAabb::new(IVec3::splat(-400), IVec3::splat(400)));
        assert!(!trees.is_empty());
        let forest = trees[trees.len() / 2].base;
        let mut places = vec![(i64::from(forest.x), i64::from(forest.z))];
        let mut coast = None;
        let mut ridge = None;
        for i in 0..4000_i64 {
            let (x, z) = ((i % 63) * 97 - 3000, (i / 63) * 89 - 3000);
            let height = source.height(0, x as f64, z as f64);
            if coast.is_none() && (-1.0..1.0).contains(&height) {
                coast = Some((x, z));
            }
            if ridge.is_none() && height > 45.0 {
                ridge = Some((x, z));
            }
        }
        places.extend(coast);
        places.extend(ridge);
        assert_eq!(places.len(), 3, "landscape lacks a coast or a ridge");
        let mut cells_with_trees = 0;
        for &(x, z) in &places {
            for level in 0..=4_u8 {
                for size in [[34, 34, 34], [7, 41, 13]] {
                    let region = surface_region(&source, level, x, z, size);
                    let filled = fill(&source, level, &region);
                    assert_eq!(
                        filled,
                        reference(&source, level, &region),
                        "{level} {x} {z}"
                    );
                    cells_with_trees += filled
                        .iter()
                        .filter(|&&s| is_feature(&source.palette, s))
                        .count();
                }
            }
        }
        assert!(cells_with_trees > 100, "{cells_with_trees}");
    }

    #[test]
    fn regions_at_and_above_the_content_bound() {
        let source = source(42);
        // The highest place of a scan, where the ridges come closest to the bound.
        let (x, z) = (0..400_i64)
            .map(|i| ((i % 20) * 211 - 2000, (i / 20) * 199 - 2000))
            .max_by(|a, b| {
                let height = |p: &(i64, i64)| source.height(0, p.0 as f64, p.1 as f64);
                height(a).total_cmp(&height(b))
            })
            .unwrap();
        for level in 0..=3_u8 {
            let first_free = last_centre_below(level, source.top_bound) + 1;
            let (cx, cz) = (x >> level, z >> level);
            // Entirely above the bound (filled without evaluating anything) and straddling it.
            for (low, high) in [
                (first_free, first_free + 6),
                (first_free - 4, first_free + 2),
            ] {
                let region = Region::new([cx - 3, low, cz - 3], [cx + 3, high, cz + 3]);
                let filled = fill(&source, level, &region);
                assert_eq!(filled, reference(&source, level, &region), "{level} {low}");
                if low == first_free {
                    assert!(filled.iter().all(|s| s.is_air()));
                }
            }
        }
    }

    #[test]
    fn regions_at_and_below_the_stone_bound() {
        let source = source(42);
        for level in 0..=6_u8 {
            let top = source.terrain.base_top(level);
            for (x, z) in [(0_i64, 0_i64), (-1234, 777), (5000, -40)] {
                let (cx, cz) = (x >> level, z >> level);
                // Entirely at or below the bound (filled without evaluating anything) and
                // straddling it.
                for (low, high) in [(top - 6, top + 1), (top - 2, top + 5)] {
                    let region = Region::new([cx - 3, low, cz - 2], [cx + 4, high, cz + 3]);
                    let filled = fill(&source, level, &region);
                    assert_eq!(filled, reference(&source, level, &region), "{level} {low}");
                    if high == top + 1 {
                        assert!(filled.iter().all(|&s| s == source.palette.stone));
                    }
                }
            }
            // From just below the bound up through the ground surface and the trees.
            let surface = surface_region(&source, level, 100, -200, [5, 7, 5]);
            let region = Region::new(
                [surface.min[0], top - 1, surface.min[2]],
                [surface.max[0], surface.max[1], surface.max[2]],
            );
            assert_eq!(
                fill(&source, level, &region),
                reference(&source, level, &region)
            );
        }
    }

    /// The classification the slow way: the exact height bounds of all the chunk's columns,
    /// then the same three tests, with no shortcut and no early exit.
    /// The classification the slow way: the analytic bounds of the level, then the extreme
    /// heights of all the columns that settle the chunk's columns, with no early exit.
    fn reference_classify(source: &HillsSource, key: LodKey) -> NodeClass {
        let level = key.level;
        let terrain = &source.terrain;
        let cells = Region::from_aabb(chunk_cells(key));
        let lowest = centre(level, cells.min[1]);
        let highest = centre(level, cells.max[1] - 1);
        let shallows = highest >= terrain.sea_level() - cell_size(level);
        if highest < terrain.min_height(level) && !shallows {
            return source.opaque_ground();
        }
        if lowest >= source.top_bound {
            return NodeClass::Empty;
        }
        // Every height that may settle a column of the chunk: the chunk's columns and the
        // ring of SETTLE_RING columns around them.
        let ring = SETTLE_RING as i64;
        let (mut low, mut high) = (f64::INFINITY, f64::NEG_INFINITY);
        for z in cells.min[2] - ring..cells.max[2] + ring {
            for x in cells.min[0] - ring..cells.max[0] + ring {
                let height = terrain.height(level, centre(level, x), centre(level, z));
                low = det::min(low, height);
                high = det::max(high, height);
            }
        }
        let above_sea = lowest >= terrain.sea_level();
        let above_ground = above_sea && lowest >= terrain.max_height(level);
        let opaque = !above_ground && highest < low && !shallows;
        let empty = above_sea && (above_ground || high <= lowest);
        if opaque {
            source.opaque_ground()
        } else if !empty || source.trees_may_reach(level, &cells, lowest) {
            NodeClass::Mixed
        } else {
            NodeClass::Empty
        }
    }

    #[test]
    fn classify_equals_the_full_column_scan() {
        let normal_flags = flags();
        let mut dirt_not_opaque = flags();
        dirt_not_opaque[2] = BlockFlags::SOLID;
        let without_trees = HillsConfig {
            tree_density: 0.0,
            ..HillsConfig::default()
        };
        let sources = [
            source(42),
            HillsSource::new(42, without_trees, palette(), &flags()).unwrap(),
            HillsSource::new(7, HillsConfig::default(), palette(), &dirt_not_opaque).unwrap(),
        ];
        let mut rng = cp_core::hash::Rng::new(9, 9);
        let (mut empty, mut opaque, mut mixed) = (0, 0, 0);
        for source in &sources {
            for level in 0..=8_u8 {
                for _ in 0..12 {
                    let x = rng.range_u32(0, 12_000) as i32 - 6000;
                    let z = rng.range_u32(0, 12_000) as i32 - 6000;
                    let ground = IVec3::new(x, source.ground_top(x, z), z);
                    let surface = LodKey::containing(level, ground);
                    for dy in -2..=3 {
                        let key = LodKey::new(level, surface.pos + IVec3::Y * dy);
                        let class = source.classify(key);
                        assert_eq!(class, reference_classify(source, key), "{key:?}");
                        // And it is true of the cells (contract §5.1): Empty is all air,
                        // Opaque all opaque.
                        if level <= 4 && class != NodeClass::Mixed {
                            let cells = chunk_cells(key);
                            let mut out = vec![AIR; 32 * 32 * 32];
                            source.fill(level, cells, &mut out, &|| false).unwrap();
                            let table = if std::ptr::eq(source, &sources[2]) {
                                &dirt_not_opaque
                            } else {
                                &normal_flags
                            };
                            let ok = |state: &BlockStateId| match class {
                                NodeClass::Empty => state.is_air(),
                                _ => is_opaque(table, *state),
                            };
                            assert!(out.iter().all(ok), "{key:?} {class:?}");
                        }
                        match class {
                            NodeClass::Empty => empty += 1,
                            NodeClass::Opaque => opaque += 1,
                            NodeClass::Mixed => mixed += 1,
                        }
                    }
                }
            }
        }
        assert!(
            empty > 100 && opaque > 100 && mixed > 100,
            "{empty} {opaque} {mixed}"
        );
    }

    #[test]
    fn padded_halos_equal_the_neighbour_cells() {
        let source = source(5);
        let trees = source.trees_in(IAabb::new(IVec3::splat(-200), IVec3::splat(200)));
        let tree = trees[0].base;
        for level in 0..=3_u8 {
            let key = LodKey::containing(level, tree);
            let padded = cp_voxel::padded_cells(key);
            let mut halo = vec![AIR; cp_voxel::region_volume(padded)];
            source.fill(level, padded, &mut halo, &|| false).unwrap();
            // Every cell of the padded fill equals the same cell of a fill of its own chunk.
            for neighbour in (0..27).map(|i| IVec3::new(i % 3 - 1, i / 9 - 1, (i / 3) % 3 - 1)) {
                let other = LodKey::new(level, key.pos + neighbour);
                let cells = chunk_cells(other);
                let mut chunk = vec![AIR; 32 * 32 * 32];
                source.fill(level, cells, &mut chunk, &|| false).unwrap();
                for y in cells.min.y..cells.max.y {
                    for z in cells.min.z..cells.max.z {
                        for x in cells.min.x..cells.max.x {
                            let cell = IVec3::new(x, y, z);
                            if padded.contains(cell) {
                                assert_eq!(
                                    halo[cp_voxel::region_index(padded, cell)],
                                    chunk[cp_voxel::region_index(cells, cell)]
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn cancel_stops_the_fill() {
        let source = source(1);
        let region = cp_voxel::padded_cells(LodKey::lod0(IVec3::ZERO));
        let mut out = vec![AIR; cp_voxel::region_volume(region)];
        assert_eq!(source.fill(0, region, &mut out, &|| true), Err(Cancelled));
        let calls = core::cell::Cell::new(0);
        let late = || {
            calls.set(calls.get() + 1);
            calls.get() > 500
        };
        assert_eq!(source.fill(0, region, &mut out, &late), Err(Cancelled));
        assert_eq!(calls.get(), 501);
    }

    #[test]
    #[should_panic(expected = "fill: the output holds")]
    fn a_wrong_buffer_length_is_a_bug() {
        let source = source(1);
        let mut out = vec![AIR; 10];
        let _ = source.fill(0, chunk_cells(LodKey::lod0(IVec3::ZERO)), &mut out, &|| {
            false
        });
    }

    #[test]
    fn identity_covers_seed_config_and_palette() {
        let a = source(1);
        assert_eq!(a.identity().key, "cubeplanet:hills");
        assert_eq!(a.identity().version, HillsSource::VERSION);
        assert_eq!(a.identity().seed, 1);
        assert_eq!(a.identity(), source(1).identity());
        assert_ne!(a.identity(), source(2).identity());
        let config = HillsConfig {
            sea_level: 3,
            ..HillsConfig::default()
        };
        let b = HillsSource::new(1, config, palette(), &flags()).unwrap();
        assert_ne!(a.identity().macro_hash, b.identity().macro_hash);
        let other_palette = TerrainPalette {
            key_hash: 77,
            ..palette()
        };
        let c = HillsSource::new(1, HillsConfig::default(), other_palette, &flags()).unwrap();
        assert_ne!(a.identity().macro_hash, c.identity().macro_hash);
        let bad = TerrainPalette {
            water: BlockStateId::UNCHANGED,
            ..palette()
        };
        assert!(HillsSource::new(1, HillsConfig::default(), bad, &flags()).is_err());
    }

    #[test]
    fn invalid_keys_get_safe_answers() {
        let source = source(1);
        let key = LodKey::new(MAX_LOD_LEVEL + 1, IVec3::ZERO);
        assert_eq!(source.classify(key), NodeClass::Mixed);
        assert!(!source.subtree_empty(key));
    }

    #[test]
    fn without_opaque_ground_nothing_is_opaque() {
        let mut flags = flags();
        flags[2] = BlockFlags::SOLID; // dirt not opaque
        let source = HillsSource::new(1, HillsConfig::default(), palette(), &flags).unwrap();
        let deep = LodKey::new(2, IVec3::new(0, -3, 0));
        assert_eq!(source.classify(deep), NodeClass::Mixed);
        assert_eq!(self::source(1).classify(deep), NodeClass::Opaque);
    }

    #[test]
    fn helpers_agree_with_the_fill() {
        let source = source(3);
        // The ground states of the test palette are the ids 1 to 6 and 16 to 18.
        let is_ground =
            |state: BlockStateId| (1..=6).contains(&state.0) || (16..=18).contains(&state.0);
        for (x, z) in [(0, 0), (-77, 140), (1000, -3)] {
            let top = source.ground_top(x, z);
            let column = IAabb::new(IVec3::new(x, top - 2, z), IVec3::new(x + 1, top + 2, z + 1));
            let mut out = vec![AIR; 4];
            source.fill(0, column, &mut out, &|| false).unwrap();
            assert!(out[..3].iter().all(|&s| is_ground(s)), "{out:?}");
            assert!(!is_ground(out[3]), "{out:?}");
        }
        let (low, high) = source.height_bounds(0);
        assert!(low < high);
        assert!(
            source
                .trees_in(IAabb::new(IVec3::ZERO, IVec3::ZERO))
                .is_empty()
        );
    }
}
