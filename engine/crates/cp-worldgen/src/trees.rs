//! Features: the trees, bushes and boulders of the hills landscape, and the tree of the
//! structure field (contract §5.1).
//!
//! # Placement
//! One candidate per cell of a jittered grid of [`TREE_GRID`] m, independent of the level. A
//! hash of the world seed and the grid cell decides where its trunk stands (anywhere in the
//! grid cell, so that neighbours stand from one to fifteen metres apart), its size, age,
//! species and shape; a few low-frequency noise fields decide whether it exists and what it
//! is:
//!
//! - the **wood** density: forests (256 m) with dense cores, soft edges, groves (48 m) and
//!   clearings, thinned out in the dry regions, and inside them clumps (40 m noise): groups
//!   of a few trees with gaps of 10 to 30 m between them instead of the even spacing of a
//!   plantation; in the open land copses of a few trees (70 m noise) and only rarely a lone
//!   tree, so that meadows do not look like a park;
//! - **bushes** along forest edges and around copses, rarely in the open; **boulders** in
//!   boulder fields (24 m noise) on slopes and at forest edges, of stone, slate or limestone
//!   as the rock of the region;
//! - the **species**: spruce takes over from 22 m up and stands in dark groves lower down,
//!   birch grows in groves (350 m) and in the dry regions, cherry stands alone in meadows and
//!   by the water, oak elsewhere;
//! - the **size** varies from tree to tree more than with the wood density (a little taller in
//!   the cores, smaller at the edges, where a mantle of young trees and bushes leads into the
//!   wood), and the **age** adds young trees (one in eight, a third of the size) and veterans
//!   (one in sixteen, wider crowns and a little taller), so that a stand's crowns do not form
//!   one flat roof;
//! - above the tree line a few trees grow as bushes (krummholz); nothing grows on anything
//!   but grass, and trees keep 3.5 m above the water.
//!
//! Its base is the first voxel above the level-0 ground. Generation is pull-based: a region
//! asks for the grid cells within the declared maximum radius and evaluates only those, so the
//! result never depends on which chunk is generated first.
//!
//! # Shape
//! - oak: a trunk of 3–10 logs and a broad crown of up to three overlapping ellipsoids;
//! - birch: a slender trunk of 5–13 logs and a narrow, tall crown;
//! - spruce: 7–17 m, a cone of tiers from 1 m above the ground to the tip, its trunk ending
//!   4 m below the tip inside a core of needles, so no bare trunk shows above the cone;
//! - cherry: a short trunk of 3–8 logs under a wide, flat crown;
//! - bush: an irregular mound of two or three overlapping ellipsoids of leaves, 2–4 m across;
//! - boulder: an ellipsoid of rock half sunk into the ground.
//!
//! A cell belongs to the crown when its centre lies inside one of the shapes (exact integer
//! tests in half metres) or when it contains the crown's anchor point. On level 0 logs win
//! over leaves; features only grow into cells the terrain leaves as air.
//!
//! # Levels
//! - Up to [`TREE_RASTER_MAX_LEVEL`] trees are rasterised: the trunk conservatively (every
//!   cell that overlaps a trunk voxel, as the downsample rule keeps structural cells) and the
//!   crown by the point rule plus the cell that contains the anchor.
//! - Above it, up to [`TREE_CANOPY_MAX_LEVEL`], the stand-in is a **canopy** of leaf cells
//!   around the crown: the crown alone, without a trunk. The anchor cell keeps every tree
//!   visible as at least one leaf cell.
//! - Landscape trees are drawn for the distance from level 1 on: the crown's shapes grow by
//!   half a cell, leaves win over logs, and the trunk ends at the crown base, so that a
//!   2 m-wide trunk does not stick out of a crown that the point rule has thinned; on the
//!   canopy levels the slender crowns of birch and spruce are replaced by one compact
//!   ellipsoid at most 1.3 times as tall as wide (a column of single cells reads as a post).
//!   The tree of the structure field keeps the plain rule on every level.
//! - Bushes and boulders are drawn up to [`SMALL_FEATURE_MAX_LEVEL`] and have no stand-in
//!   above it.
//! - Above [`TREE_CANOPY_MAX_LEVEL`] a tree is smaller than a cell and has no stand-in; the
//!   grass below carries the colour.

use cp_core::coords::MAX_LOD_LEVEL;
use cp_core::det;
use cp_core::hash::{Rng, hash_combine};
use cp_core::ids::BlockStateId;
use cp_core::math::{DVec2, IVec3, smoothstep};
use cp_core::noise::Fbm;

use crate::palette::TerrainPalette;
use crate::region::{Region, ceil_div, floor_div, last_centre_below};
use crate::terrain::{Surface, Terrain};

/// Edge of the feature placement grid (m): at most one feature per grid cell.
pub const TREE_GRID: i32 = 8;
/// Declared maximum feature radius (m): no feature voxel lies further than this from the trunk
/// column along x or z (contract §5.1). Coarser levels add at most one cell to it.
pub const TREE_MAX_RADIUS: i32 = 6;
/// Every feature voxel lies below `base + TREE_MAX_HEIGHT` (voxels above the base voxel).
pub const TREE_MAX_HEIGHT: i32 = 18;
/// Coarsest level at which trees are rasterised with trunk and crown.
pub const TREE_RASTER_MAX_LEVEL: u8 = 1;
/// Coarsest level of the canopy stand-in; above it trees are not drawn.
pub const TREE_CANOPY_MAX_LEVEL: u8 = 3;
/// Coarsest level at which bushes and boulders are drawn; they have no stand-in.
pub const SMALL_FEATURE_MAX_LEVEL: u8 = 1;

/// Trunks keep this distance (m) from the border of their grid cell: none, so that two trees
/// of neighbouring cells may stand side by side.
const GRID_MARGIN: i64 = 0;

/// Forest mask: two octaves of 256 m.
const FOREST: Fbm = Fbm::new(2, 2.0, 0.5);
const FOREST_FREQUENCY: f64 = 1.0 / 256.0;
/// The forest mask from this value on starts to carry trees; dense from 0.35 above it.
const FOREST_THRESHOLD: f64 = -0.1;
const FOREST_RAMP: f64 = 0.35;
/// Groves and clearings inside the forests: one octave of 48 m.
const GROVE: Fbm = Fbm::new(1, 2.0, 0.5);
const GROVE_FREQUENCY: f64 = 1.0 / 48.0;
/// Clumps of trees inside the forests: one octave of 40 m, and the range of the noise (mapped
/// to about `[0, 1]`) over which a clump fades in. Between clumps a quarter of the trees stand.
const CLUMP: Fbm = Fbm::new(1, 2.0, 0.5);
const CLUMP_FREQUENCY: f64 = 1.0 / 40.0;
const CLUMP_RAMP: (f64, f64) = (0.25, 0.75);
const CLUMP_FLOOR: f64 = 0.25;
/// The age of a tree: the share of young trees and their size factor, and the share of
/// veterans.
const YOUNG: (f64, f64) = (0.12, 0.35);
const VETERANS: f64 = 0.06;
/// At a forest edge (an edge measure `4 w (1 - w)` above this, for the wood density `w`) trees
/// keep this share of their size: the mantle of a wood.
const EDGE_FROM: f64 = 0.5;
const EDGE_SIZE: f64 = 0.7;
/// Birch groves: one octave of 350 m.
const BIRCH_GROVE: Fbm = Fbm::new(1, 2.0, 0.5);
const BIRCH_FREQUENCY: f64 = 1.0 / 350.0;
/// The wood density in the open: a rare lone tree in the meadows.
const OPEN_WOOD: f64 = 0.01;
/// Copses in the open land: one octave of 70 m; where it is high, groups of a few trees.
const COPSE: Fbm = Fbm::new(1, 2.0, 0.5);
const COPSE_FREQUENCY: f64 = 1.0 / 70.0;
/// The copse noise (stretched to about `[-1, 1]`) from which copses start, and where they are
/// densest; and the wood density at their core.
const COPSE_RAMP: (f64, f64) = (0.6, 0.85);
const COPSE_WOOD: f64 = 0.35;
/// Boulder fields: one octave of 24 m; boulders lie only where it is above this.
const BOULDER_FIELD: Fbm = Fbm::new(1, 2.0, 0.5);
const BOULDER_FREQUENCY: f64 = 1.0 / 24.0;
const BOULDER_FIELD_FROM: f64 = 0.5;
/// How much the dry regions thin the forests out (they keep dense birch groves).
const DRY_THINNING: f64 = 0.6;
/// Spruce takes over between these heights (m).
const SPRUCE_HEIGHTS: (f64, f64) = (22.0, 44.0);
/// Stands of spruce in the low groves grow from this height (m) up.
const SPRUCE_STAND_HEIGHT: f64 = 10.0;
/// Trees stand at least this high above the sea level (m).
const SHORE_CLEARANCE: f64 = 3.5;
/// The tree line: trees thin out from this far below the snow line to none at that far (m).
const TREE_LINE: (f64, f64) = (30.0, 14.0);
/// The share of the trees above the tree line that grow as bushes instead.
const KRUMMHOLZ: f64 = 0.3;

const SALT_TREES: u64 = 0x5452_4545_5300_0004;
const SALT_FOREST: u64 = 0x464F_5245_5354_0005;
const SALT_GROVE: u64 = 0x4752_4F56_4500_0006;
const SALT_BIRCH: u64 = 0x4249_5243_4800_0007;
const SALT_COPSE: u64 = 0x434F_5053_4500_0008;
const SALT_BOULDERS: u64 = 0x424F_554C_4400_0009;
const SALT_CLUMP: u64 = 0x434C_554D_5000_000A;

/// A tree species: its logs and leaves in the palette and its shape.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Species {
    /// Broad crown of several lobes; the leaves take the foliage tint.
    Oak,
    /// White bark, a narrow, tall crown of light leaves.
    Birch,
    /// A dark cone of tiers.
    Spruce,
    /// A wide, flat crown of pink blossoms.
    Cherry,
}

/// The rock of a boulder.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Rock {
    /// [`TerrainPalette::stone`].
    Stone,
    /// [`TerrainPalette::slate`].
    Slate,
    /// [`TerrainPalette::limestone`].
    Limestone,
}

/// What a feature is.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum FeatureKind {
    /// A tree with trunk and crown.
    Tree(Species),
    /// A mound of leaves of a species, without a trunk.
    Bush(Species),
    /// An ellipsoid of rock half sunk into the ground.
    Boulder(Rock),
}

/// One feature as the generators place it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TreeInfo {
    /// The lowest trunk voxel (the voxel above the ground for bushes and boulders); it stands
    /// on the ground voxel below it.
    pub base: IVec3,
    /// Number of trunk voxels, 0 for bushes and boulders.
    pub trunk_height: i32,
    /// Horizontal radius of the main part of the crown (or of the boulder) in half metres.
    pub crown_radius_halves: i32,
    /// What it is.
    pub kind: FeatureKind,
}

/// An axis-aligned ellipsoid in half metres: the cells whose centre lies inside belong to it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Blob {
    pub(crate) centre: [i64; 3],
    /// Radii along x, y, z in half metres (> 0).
    pub(crate) radii: [i64; 3],
}

impl Blob {
    /// True when the point `p` (half metres) lies inside or on the ellipsoid: exact in
    /// integers, `Σ d_i² Π_{j≠i} r_j² <= Π r_i²`.
    fn contains(&self, p: [i64; 3]) -> bool {
        let [rx, ry, rz] = self.radii.map(|r| r * r);
        let d = [0, 1, 2].map(|i| {
            let d = p[i] - self.centre[i];
            d * d
        });
        d[0] * ry * rz + d[1] * rx * rz + d[2] * rx * ry <= rx * ry * rz
    }

    /// The same ellipsoid with every radius grown by `grow` half metres (>= 0).
    fn grown(&self, grow: i64) -> Self {
        Self {
            centre: self.centre,
            radii: self.radii.map(|r| r + grow),
        }
    }

    /// The inclusive bounds (half metres) of the points inside.
    fn bounds(&self) -> ([i64; 3], [i64; 3]) {
        (
            [0, 1, 2].map(|i| self.centre[i] - self.radii[i]),
            [0, 1, 2].map(|i| self.centre[i] + self.radii[i]),
        )
    }
}

/// A cone of tiers around a vertical axis, in half metres: from `bottom` to `top`, the radius
/// falling from `radius` at the bottom to zero at the top along a slightly convex profile,
/// each tier of [`TIER`] half metres wide at its foot and narrower at its head, around a core
/// of needles next to the axis (see [`Cone::contains`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Cone {
    pub(crate) axis: [i64; 2],
    pub(crate) bottom: i64,
    pub(crate) top: i64,
    pub(crate) radius: i64,
}

/// Height of a tier of a cone in half metres (3 m).
const TIER: i64 = 6;
/// The core of a cone: points within `sqrt(CORE_RADIUS_SQ4 / 4)` = 2.5 half metres of the axis
/// belong to it down from `CORE_BELOW_TIP` half metres (1.5 m) under the tip, so the four
/// columns next to a trunk are needles all the way up and no trunk shows through the cone.
const CORE_RADIUS_SQ4: i64 = 25;
const CORE_BELOW_TIP: i64 = 3;

impl Cone {
    /// True when `p` (half metres) lies inside the cone grown by `grow` half metres. With
    /// `h = top - bottom`, `l = top - y` (clamped to `0..=h`) and `u` the position in the tier,
    /// the radius is `r = R · l (5h - l) / (4h²) · (10T - 2u) / (10T)`: `R` at the foot, a
    /// little fuller than straight lines above it (`(1 - t)^0.8` closely), and each tier
    /// narrowing to 80 percent at its head. Exact in integers:
    /// `d² · D² <= (N + grow · D)²` with `r = N / D`.
    fn contains(&self, p: [i64; 3], grow: i64) -> bool {
        if p[1] < self.bottom - grow || p[1] > self.top + grow {
            return false;
        }
        let (dx, dz) = (p[0] - self.axis[0], p[2] - self.axis[1]);
        let d2 = dx * dx + dz * dz;
        let height = self.top - self.bottom;
        let left = (self.top - p[1]).clamp(0, height);
        if left >= CORE_BELOW_TIP && 4 * d2 <= CORE_RADIUS_SQ4 {
            return true;
        }
        let u = (p[1] - self.bottom).clamp(0, height).rem_euclid(TIER);
        let full = 10 * TIER;
        let numerator = self.radius * left * (5 * height - left) * (full - 2 * u);
        let denominator = 4 * height * height * full;
        let reach = numerator + grow * denominator;
        d2 * denominator * denominator <= reach * reach
    }

    /// The inclusive bounds (half metres) of the points inside the cone grown by `grow`.
    fn bounds(&self, grow: i64) -> ([i64; 3], [i64; 3]) {
        let radius = self.radius + grow;
        (
            [
                self.axis[0] - radius,
                self.bottom - grow,
                self.axis[1] - radius,
            ],
            [
                self.axis[0] + radius,
                self.top + grow,
                self.axis[1] + radius,
            ],
        )
    }
}

/// The shapes a crown is drawn with on one level: ellipsoids and an optional cone, all grown
/// by `grow` half metres.
#[derive(Clone, Copy, Debug)]
struct CrownShapes {
    blobs: [Blob; 3],
    count: usize,
    cone: Option<Cone>,
    grow: i64,
}

impl CrownShapes {
    fn blobs(&self) -> &[Blob] {
        &self.blobs[..self.count]
    }

    fn contains(&self, p: [i64; 3]) -> bool {
        self.blobs().iter().any(|blob| blob.contains(p))
            || self.cone.is_some_and(|cone| cone.contains(p, self.grow))
    }
}

/// A feature in internal coordinates: trunk column `(x, z)`, trunk voxels `base..base +
/// height`, the crown as up to three ellipsoids and a cone, and the anchor point whose cell is
/// always part of the crown.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Tree {
    pub(crate) x: i64,
    pub(crate) z: i64,
    pub(crate) base: i64,
    /// Trunk voxels; 0 for bushes and boulders.
    pub(crate) height: i64,
    pub(crate) kind: FeatureKind,
    blobs: [Blob; 3],
    blob_count: usize,
    cone: Option<Cone>,
    /// Half metres.
    anchor: [i64; 3],
}

/// Up to which level each part of a feature is drawn, and whether it is drawn for the
/// distance on the coarse levels.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct TreeStyle {
    pub(crate) trunk_max_level: u8,
    pub(crate) crown_max_level: u8,
    /// From level 1 on: crowns grown by half a cell and drawn over the trunk, the trunk ended
    /// at the crown base, compact stand-ins for slender crowns on the canopy levels (see the
    /// module documentation).
    pub(crate) distant_crowns: bool,
}

impl TreeStyle {
    /// Whether `level` draws this style's crowns for the distance.
    fn distant(&self, level: u8) -> bool {
        self.distant_crowns && level >= 1
    }

    /// The order in which the parts are drawn on `level`: the later part wins where both
    /// want a cell. Logs win on level 0 and wherever the crowns are not drawn for the
    /// distance; leaves win from level 1 on otherwise.
    pub(crate) fn drawing_order(&self, level: u8) -> [TreePart; 2] {
        if self.distant(level) {
            [TreePart::Trunk, TreePart::Crown]
        } else {
            [TreePart::Crown, TreePart::Trunk]
        }
    }
}

/// Features of the landscape: trees rasterised up to level 1 with a canopy stand-in up to
/// level 3, bushes and boulders up to level 1, crowns drawn for the distance.
pub(crate) const LANDSCAPE_TREES: TreeStyle = TreeStyle {
    trunk_max_level: TREE_RASTER_MAX_LEVEL,
    crown_max_level: TREE_CANOPY_MAX_LEVEL,
    distant_crowns: true,
};

/// The tree of the structure set: a structure, kept (conservatively) on every level.
pub(crate) const STRUCTURE_TREES: TreeStyle = TreeStyle {
    trunk_max_level: MAX_LOD_LEVEL,
    crown_max_level: MAX_LOD_LEVEL,
    distant_crowns: false,
};

/// Stand-ins of the canopy levels are at most this many times as tall as wide (in tenths).
const STAND_IN_ASPECT_TENTHS: i64 = 13;

/// Which part of the features a drawing pass writes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum TreePart {
    /// Leaves (or the rock of a boulder).
    Crown,
    /// Logs.
    Trunk,
}

/// A region being filled: its level, its cells and the output buffer in `fill` layout.
pub(crate) struct Canvas<'a> {
    pub(crate) level: u8,
    pub(crate) region: Region,
    pub(crate) out: &'a mut [BlockStateId],
}

/// The cells of `level` whose centres lie in the inclusive half-metre interval `[lo, hi]`, as
/// a half-open index range.
fn cells_with_centres_in(level: u8, lo: i64, hi: i64) -> (i64, i64) {
    let size = 1_i64 << level;
    // Cells whose centre (2c + 1) * size lies within [lo, hi].
    (
        ceil_div(lo - size, 2 * size),
        floor_div(hi - size, 2 * size) + 1,
    )
}

impl Tree {
    /// A trunk of `height` logs with one spherical crown of `radius_halves` half metres around
    /// the centre of the top trunk voxel: the tree of the structure field.
    pub(crate) fn sphere(x: i64, z: i64, base: i64, height: i64, radius_halves: i64) -> Self {
        let top = base + height - 1;
        let centre = [2 * x + 1, 2 * top + 1, 2 * z + 1];
        let blob = Blob {
            centre,
            radii: [radius_halves; 3],
        };
        Self {
            x,
            z,
            base,
            height,
            kind: FeatureKind::Tree(Species::Oak),
            blobs: [blob; 3],
            blob_count: 1,
            cone: None,
            anchor: centre,
        }
    }

    /// The top trunk voxel.
    fn top(&self) -> i64 {
        self.base + self.height - 1
    }

    /// The style `style` limited to what this kind of feature has: bushes and boulders end at
    /// [`SMALL_FEATURE_MAX_LEVEL`].
    fn limited(&self, style: TreeStyle) -> TreeStyle {
        match self.kind {
            FeatureKind::Tree(_) => style,
            FeatureKind::Bush(_) | FeatureKind::Boulder(_) => TreeStyle {
                trunk_max_level: style.trunk_max_level.min(SMALL_FEATURE_MAX_LEVEL),
                crown_max_level: style.crown_max_level.min(SMALL_FEATURE_MAX_LEVEL),
                ..style
            },
        }
    }

    /// The shapes of the crown on `level` with `style`: as made on level 0 and wherever the
    /// style does not draw for the distance; from level 1 on grown by half a cell
    /// (`2^level` half metres); on the canopy levels (above the rasterised ones) a slender
    /// crown, birch or spruce, becomes one ellipsoid at most 1.3 times as tall as wide: for
    /// birch the hull of its two ellipsoids, for spruce the lower 60 percent of the cone.
    fn crown_shapes(&self, level: u8, style: TreeStyle) -> CrownShapes {
        let mut shapes = CrownShapes {
            blobs: self.blobs,
            count: self.blob_count,
            cone: self.cone,
            grow: 0,
        };
        if !style.distant(level) {
            return shapes;
        }
        let grow = 1_i64 << level;
        let canopy = level > style.trunk_max_level;
        let stand_in = match self.kind {
            FeatureKind::Tree(Species::Spruce) if canopy => self.cone.map(|cone| {
                let height = cone.top - cone.bottom;
                let ry = (3 * height) / 10;
                Blob {
                    centre: [cone.axis[0], cone.bottom + ry, cone.axis[1]],
                    radii: [cone.radius, ry, cone.radius],
                }
            }),
            FeatureKind::Tree(Species::Birch) if canopy && self.blob_count >= 1 => {
                let (lo, hi) = self.blobs().iter().map(Blob::bounds).fold(
                    self.blobs[0].bounds(),
                    |(lo, hi), (a, b)| {
                        (
                            [0, 1, 2].map(|i| lo[i].min(a[i])),
                            [0, 1, 2].map(|i| hi[i].max(b[i])),
                        )
                    },
                );
                let centre = [0, 1, 2].map(|i| floor_div(lo[i] + hi[i], 2));
                Some(Blob {
                    centre,
                    radii: [0, 1, 2].map(|i| (hi[i] - lo[i] + 1) / 2),
                })
            }
            _ => None,
        };
        if let Some(blob) = stand_in {
            let mut blob = blob.grown(grow);
            let width = blob.radii[0].min(blob.radii[2]);
            blob.radii[1] = blob.radii[1].min((STAND_IN_ASPECT_TENTHS * width) / 10);
            // A stand-in whose radius is under one cell (`2 * grow` half metres) is one or two
            // cells wide, wherever it lies on the lattice: it may be no taller than wide, or
            // the cell centres it holds stack into a post.
            if width < 2 * grow {
                blob.radii[1] = blob.radii[1].min(width);
            }
            shapes.blobs[0] = blob;
            shapes.count = 1;
            shapes.cone = None;
        } else {
            for blob in &mut shapes.blobs[..shapes.count] {
                *blob = blob.grown(grow);
            }
            shapes.grow = grow;
        }
        shapes
    }

    /// The trunk voxel at which the trunk ends on `level` with `style`: its top voxel, or,
    /// where the crown is drawn for the distance, the voxel below the crown's lowest point.
    fn trunk_end(&self, level: u8, style: TreeStyle) -> i64 {
        if !style.distant(level) {
            return self.top();
        }
        let (lo, _) = self.crown_bounds(0, style);
        self.top().min(floor_div(lo[1], 2) - 1)
    }

    /// Whether `part` of this feature is drawn on `level` with `style`.
    pub(crate) fn draws(&self, level: u8, style: TreeStyle, part: TreePart) -> bool {
        let style = self.limited(style);
        match part {
            TreePart::Crown => level <= style.crown_max_level,
            TreePart::Trunk => self.height > 0 && level <= style.trunk_max_level,
        }
    }

    /// The state of `part` from `palette`.
    pub(crate) fn state(&self, part: TreePart, palette: &TerrainPalette) -> BlockStateId {
        let species = |species: Species| match (species, part) {
            (Species::Oak, TreePart::Trunk) => palette.log,
            (Species::Oak, TreePart::Crown) => palette.leaves,
            (Species::Birch, TreePart::Trunk) => palette.log_birch,
            (Species::Birch, TreePart::Crown) => palette.leaves_birch,
            (Species::Spruce, TreePart::Trunk) => palette.log_spruce,
            (Species::Spruce, TreePart::Crown) => palette.leaves_spruce,
            (Species::Cherry, TreePart::Trunk) => palette.log_cherry,
            (Species::Cherry, TreePart::Crown) => palette.leaves_cherry,
        };
        match self.kind {
            FeatureKind::Tree(s) | FeatureKind::Bush(s) => species(s),
            FeatureKind::Boulder(Rock::Stone) => palette.stone,
            FeatureKind::Boulder(Rock::Slate) => palette.slate,
            FeatureKind::Boulder(Rock::Limestone) => palette.limestone,
        }
    }

    /// The cell of `level` that contains the anchor point.
    fn anchor_cell(&self, level: u8) -> [i64; 3] {
        self.anchor.map(|a| floor_div(a, 2) >> level)
    }

    fn blobs(&self) -> &[Blob] {
        &self.blobs[..self.blob_count]
    }

    /// True when `cell` of `level` belongs to the crown `shapes` (the crown of `level`, see
    /// [`Tree::crown_shapes`]): its centre lies inside a shape, or it contains the anchor.
    fn in_crown(&self, shapes: &CrownShapes, level: u8, cell: [i64; 3]) -> bool {
        let size = 1_i64 << level;
        let centre = cell.map(|c| (2 * c + 1) * size);
        shapes.contains(centre) || cell == self.anchor_cell(level)
    }

    /// True when `cell` of `level` overlaps a trunk voxel at or below `end`.
    fn in_trunk(&self, level: u8, end: i64, cell: [i64; 3]) -> bool {
        self.height > 0
            && end >= self.base
            && cell[0] == self.x >> level
            && cell[2] == self.z >> level
            && cell[1] >= self.base >> level
            && cell[1] <= end >> level
    }

    /// True when `style` draws `part` into `cell` of `level` (the plain rule that `draw`
    /// evaluates over the cell box; tests check against it).
    #[cfg(test)]
    pub(crate) fn has_cell(
        &self,
        level: u8,
        style: TreeStyle,
        part: TreePart,
        cell: [i64; 3],
    ) -> bool {
        self.draws(level, style, part)
            && match part {
                TreePart::Crown => self.in_crown(&self.crown_shapes(level, style), level, cell),
                TreePart::Trunk => self.in_trunk(level, self.trunk_end(level, style), cell),
            }
    }

    /// The inclusive half-metre bounds of the crown's shapes on `level` with `style`.
    fn crown_bounds(&self, level: u8, style: TreeStyle) -> ([i64; 3], [i64; 3]) {
        let shapes = self.crown_shapes(level, style);
        let mut lo = self.anchor;
        let mut hi = self.anchor;
        let bounds = shapes
            .blobs()
            .iter()
            .map(Blob::bounds)
            .chain(shapes.cone.as_ref().map(|cone| cone.bounds(shapes.grow)));
        for (a, b) in bounds {
            for axis in 0..3 {
                lo[axis] = lo[axis].min(a[axis]);
                hi[axis] = hi[axis].max(b[axis]);
            }
        }
        (lo, hi)
    }

    /// A box of `level` cells that contains every cell `style` draws for this feature, or
    /// `None` when it draws nothing on that level.
    pub(crate) fn cell_box(&self, level: u8, style: TreeStyle) -> Option<Region> {
        let mut bounds: Option<Region> = None;
        if self.draws(level, style, TreePart::Crown) {
            let (lo, hi) = self.crown_bounds(level, style);
            let anchor = self.anchor_cell(level);
            let mut min = [0; 3];
            let mut max = [0; 3];
            for axis in 0..3 {
                let (first, end) = cells_with_centres_in(level, lo[axis], hi[axis]);
                min[axis] = first.min(anchor[axis]);
                max[axis] = end.max(anchor[axis] + 1);
            }
            bounds = Some(Region::new(min, max));
        }
        let end = self.trunk_end(level, style);
        if self.draws(level, style, TreePart::Trunk) && end >= self.base {
            let trunk = Region::new(
                [self.x >> level, self.base >> level, self.z >> level],
                [
                    (self.x >> level) + 1,
                    (end >> level) + 1,
                    (self.z >> level) + 1,
                ],
            );
            bounds = Some(bounds.map_or(trunk, |b| b.union(&trunk)));
        }
        bounds
    }

    /// The level-0 voxels of the feature: trunk and the bounding box of the crown's shapes.
    pub(crate) fn voxel_box(&self) -> Region {
        let (lo, hi) = self.crown_bounds(0, STRUCTURE_TREES);
        let crown = Region::new(lo.map(|c| floor_div(c, 2)), hi.map(|c| ceil_div(c, 2)));
        if self.height == 0 {
            return crown;
        }
        let trunk = Region::new(
            [self.x, self.base, self.z],
            [self.x + 1, self.top() + 1, self.z + 1],
        );
        crown.union(&trunk)
    }

    /// Writes the state of `part` from `palette` into the cells of that part that lie in the
    /// canvas and that `writable` allows (cells the terrain left as air).
    pub(crate) fn draw(
        &self,
        canvas: &mut Canvas<'_>,
        style: TreeStyle,
        part: TreePart,
        palette: &TerrainPalette,
        writable: impl Fn(i64, i64, i64) -> bool,
    ) {
        let level = canvas.level;
        if !self.draws(level, style, part) {
            return;
        }
        let Some(cells) = self
            .cell_box(level, style)
            .and_then(|bounds| bounds.intersection(&canvas.region))
        else {
            return;
        };
        let state = self.state(part, palette);
        let shapes = self.crown_shapes(level, style);
        let end = self.trunk_end(level, style);
        for y in cells.min[1]..cells.max[1] {
            for z in cells.min[2]..cells.max[2] {
                for x in cells.min[0]..cells.max[0] {
                    let hit = match part {
                        TreePart::Crown => self.in_crown(&shapes, level, [x, y, z]),
                        TreePart::Trunk => self.in_trunk(level, end, [x, y, z]),
                    };
                    if hit && writable(x, y, z) {
                        let index = canvas.region.index(x, y, z);
                        canvas.out[index] = state;
                    }
                }
            }
        }
    }

    pub(crate) fn info(&self) -> TreeInfo {
        // Features only exist inside the voxel limits, so the coordinates fit in i32.
        let radius = self
            .blobs()
            .first()
            .map(|blob| blob.radii[0])
            .or(self.cone.map(|cone| cone.radius))
            .unwrap_or(0);
        TreeInfo {
            base: IVec3::new(self.x as i32, self.base as i32, self.z as i32),
            trunk_height: self.height as i32,
            crown_radius_halves: radius as i32,
            kind: self.kind,
        }
    }
}

/// The random draws of one candidate, each in `0..2^16`, made in a fixed order whatever the
/// outcome of the tests.
#[derive(Clone, Copy, Debug)]
struct Draws {
    exists: u32,
    size: u32,
    species: u32,
    shape: [u32; 4],
    age: u32,
}

/// A draw in `0..2^16` as a fraction in `[0, 1)`.
fn unit(draw: u32) -> f64 {
    f64::from(draw) / 65_536.0
}

/// `value` rounded to the nearest integer, for the sizes of the shapes (values are small and
/// positive).
fn round(value: f64) -> i64 {
    (value + 0.5).floor() as i64
}

/// Horizontal directions in half metres, eight around the circle.
const LOBE_DIRECTIONS: [[i64; 2]; 8] = [
    [3, 0],
    [2, 2],
    [0, 3],
    [-2, 2],
    [-3, 0],
    [-2, -2],
    [0, -3],
    [2, -2],
];

/// Adds an ellipsoid to the crown of `tree` (radii at least one metre).
fn add(tree: &mut Tree, centre: [i64; 3], radii: [i64; 3]) {
    tree.blobs[tree.blob_count] = Blob {
        centre,
        radii: radii.map(|r| r.max(2)),
    };
    tree.blob_count += 1;
}

/// Adds `count` lobes of radius `radius` (half metres; vertical radius `flat / 4` of it)
/// around the crown centre `centre`, in directions taken from `draws`.
fn lobes(tree: &mut Tree, centre: [i64; 3], radius: i64, flat: i64, count: u32, draws: [u32; 4]) {
    for k in 0..count {
        let [ox, oz] = LOBE_DIRECTIONS[((draws[1] + 3 * k) % 8) as usize];
        let dy = i64::from((draws[2] >> (2 * k)) % 3) - 1;
        add(
            tree,
            [centre[0] + ox, centre[1] + dy, centre[2] + oz],
            [radius, (radius * flat) / 4, radius],
        );
    }
}

/// The shape of a feature of `kind` with its base voxel at `(x, base, z)`, of size `size` in
/// `[0, 1]`, with the shape draws `draws`; a `veteran` tree has a wider crown and (except a
/// spruce, which already reaches [`TREE_MAX_HEIGHT`]) a taller trunk. Every shape keeps inside
/// [`TREE_MAX_RADIUS`] and [`TREE_MAX_HEIGHT`], and its bare trunk below the crown short
/// enough for the distant crowns of the coarse levels (tests check both).
fn shape(
    kind: FeatureKind,
    [x, base, z]: [i64; 3],
    size: f64,
    draws: [u32; 4],
    veteran: bool,
) -> Tree {
    let old = i64::from(veteran);
    let mut tree = Tree {
        x,
        z,
        base,
        height: 0,
        kind,
        blobs: [Blob {
            centre: [0; 3],
            radii: [1; 3],
        }; 3],
        blob_count: 0,
        cone: None,
        anchor: [2 * x + 1, 2 * base + 1, 2 * z + 1],
    };
    match kind {
        FeatureKind::Tree(Species::Oak) => {
            tree.height = 3 + round(6.0 * size) + old;
            let top = tree.top();
            let r = 3 + round(6.0 * size) + 2 * old;
            let centre = [2 * x + 1, 2 * top + 1, 2 * z + 1];
            add(
                &mut tree,
                [centre[0], centre[1] - 1, centre[2]],
                [r, (3 * r) / 4, r],
            );
            lobes(&mut tree, centre, (7 * r) / 10, 3, 1 + draws[0] % 2, draws);
            tree.anchor = centre;
        }
        FeatureKind::Tree(Species::Birch) => {
            tree.height = 5 + round(6.0 * size) + 2 * old;
            let top = tree.top();
            let r = 3 + round(2.0 * size) + 2 * old;
            let ry = r + (7 * r) / 10;
            let centre = [2 * x + 1, 2 * top + 1 - ry / 3, 2 * z + 1];
            add(&mut tree, centre, [r, ry, r]);
            let small = (6 * r) / 10;
            add(
                &mut tree,
                [centre[0], centre[1] + (2 * ry) / 3, centre[2]],
                [small, small, small],
            );
            tree.anchor = [2 * x + 1, 2 * top + 1, 2 * z + 1];
        }
        FeatureKind::Tree(Species::Spruce) => {
            let total = 7 + round(10.0 * size);
            // The trunk ends 5 m below the tip, inside the core of the cone.
            tree.height = total - 5;
            let radius = 5 + round(3.0 * size) + 2 * old;
            tree.cone = Some(Cone {
                axis: [2 * x + 1, 2 * z + 1],
                bottom: 2 * (base + 1),
                top: 2 * (base + total) + 1,
                radius,
            });
            tree.anchor = [2 * x + 1, 2 * (base + total / 2) + 1, 2 * z + 1];
        }
        FeatureKind::Tree(Species::Cherry) => {
            tree.height = 3 + round(3.0 * size) + 2 * old;
            let top = tree.top();
            let r = 5 + round(3.0 * size) + 2 * old;
            let centre = [2 * x + 1, 2 * top + 1, 2 * z + 1];
            add(&mut tree, centre, [r, (6 * r) / 10, r]);
            lobes(&mut tree, centre, (7 * r) / 10, 2, 2, draws);
            tree.anchor = centre;
        }
        FeatureKind::Bush(_) => {
            // Two or three overlapping mounds, 2 to 3 half metres apart: an irregular hump
            // of 2 to 4 m instead of a cube.
            let centre = [2 * x + 1, 2 * base + 1, 2 * z + 1];
            let r = 2 + i64::from(draws[0] % 2) + round(size);
            let ry = 2 + i64::from(draws[1] % 2);
            add(&mut tree, centre, [r, ry, r]);
            for k in 0..1 + draws[3] % 2 {
                let [ox, oz] = LOBE_DIRECTIONS[((draws[2] + 3 * k) % 8) as usize];
                let shift = 2 + i64::from((draws[3] >> (k + 1)) % 2);
                let small = (r - 1).max(2);
                add(
                    &mut tree,
                    [
                        centre[0] + (ox * shift) / 3,
                        centre[1] - i64::from(k),
                        centre[2] + (oz * shift) / 3,
                    ],
                    [small, ry, small],
                );
            }
        }
        FeatureKind::Boulder(_) => {
            let r = [2, 3, 3, 4][(draws[0] % 4) as usize];
            let ry = 2 + i64::from(draws[1] % 2);
            // A little longer one way than the other.
            let (rx, rz) = if draws[2].is_multiple_of(2) {
                (r, (r * 3) / 4)
            } else {
                ((r * 3) / 4, r)
            };
            add(&mut tree, [2 * x + 1, 2 * base, 2 * z + 1], [rx, ry, rz]);
        }
    }
    tree
}

/// Places the features of one landscape.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct TreePlacer {
    seed: u64,
    forest_seed: u64,
    grove_seed: u64,
    birch_seed: u64,
    copse_seed: u64,
    boulder_seed: u64,
    clump_seed: u64,
    /// The share of grid cells that hold a tree in the densest forest, in `[0, 1]`.
    density: f64,
}

impl TreePlacer {
    /// Features for the world seed `seed`; `density` in `[0, 1]` is the share of grid cells in
    /// the densest forests that hold a tree.
    pub(crate) fn new(seed: u64, density: f64) -> Self {
        Self {
            seed: hash_combine(seed, SALT_TREES),
            forest_seed: hash_combine(seed, SALT_FOREST),
            grove_seed: hash_combine(seed, SALT_GROVE),
            birch_seed: hash_combine(seed, SALT_BIRCH),
            copse_seed: hash_combine(seed, SALT_COPSE),
            boulder_seed: hash_combine(seed, SALT_BOULDERS),
            clump_seed: hash_combine(seed, SALT_CLUMP),
            density,
        }
    }

    /// The clumping of the forest at `(x, z)` m in `[CLUMP_FLOOR, 1]`: 1 in the core of a
    /// group of trees, [`CLUMP_FLOOR`] in the gaps between groups.
    fn clumping(&self, x: f64, z: f64) -> f64 {
        let noise = 0.5 + 0.8 * CLUMP.sample2(self.clump_seed, DVec2::new(x, z) * CLUMP_FREQUENCY);
        CLUMP_FLOOR + (1.0 - CLUMP_FLOOR) * smoothstep(CLUMP_RAMP.0, CLUMP_RAMP.1, noise)
    }

    /// How much of a copse `(x, z)` m lies in, in `[0, 1]`: groups of a few trees in the open
    /// land, about 70 m apart where they occur.
    fn copse(&self, x: f64, z: f64) -> f64 {
        let noise = 1.6 * COPSE.sample2(self.copse_seed, DVec2::new(x, z) * COPSE_FREQUENCY);
        smoothstep(COPSE_RAMP.0, COPSE_RAMP.1, noise)
    }

    /// The wood density at `(x, z)` m in `[0, 1]`: forest cores near 1, soft edges, groves
    /// and clearings, clumps of trees with gaps between them, thinned out in dry regions; in
    /// the open [`OPEN_WOOD`] plus the copses. The density of the trees, and of the shade under
    /// them (the forest floor of a ground map, for example).
    pub(crate) fn wood(&self, x: f64, z: f64, terrain: &Terrain) -> f64 {
        let p = DVec2::new(x, z);
        let open = OPEN_WOOD + COPSE_WOOD * self.copse(x, z);
        let forest = FOREST.sample2(self.forest_seed, p * FOREST_FREQUENCY);
        let core = smoothstep(FOREST_THRESHOLD, FOREST_THRESHOLD + FOREST_RAMP, forest);
        if core <= 0.0 {
            return open;
        }
        let grove = GROVE.sample2(self.grove_seed, p * GROVE_FREQUENCY);
        let groves = 0.55 + 0.45 * det::clamp(0.5 + 0.8 * grove, 0.0, 1.0);
        let dry = terrain.dryness(0, x, z);
        let thinned = 1.0 - DRY_THINNING * smoothstep(0.2, 0.7, dry);
        open + (1.0 - open) * core * groves * thinned * self.clumping(x, z)
    }

    /// The species of a tree with the species draw `draw` at `(x, z)` m on ground at `height`
    /// m, in wood of density `wood`.
    fn species(
        &self,
        draw: u32,
        (x, z): (f64, f64),
        height: f64,
        wood: f64,
        terrain: &Terrain,
    ) -> Species {
        // One field of groves: birch where it is high, dark stands of spruce where it is low.
        let grove = 1.6 * BIRCH_GROVE.sample2(self.birch_seed, DVec2::new(x, z) * BIRCH_FREQUENCY);
        let stand = if grove < -0.45 && height > SPRUCE_STAND_HEIGHT {
            0.55
        } else {
            0.0
        };
        let spruce = det::max(
            smoothstep(SPRUCE_HEIGHTS.0, SPRUCE_HEIGHTS.1, height),
            stand,
        );
        let u = unit(draw);
        if u < spruce {
            return Species::Spruce;
        }
        let u = (u - spruce) / (1.0 - spruce);
        let dry = terrain.dryness(0, x, z);
        let birch = det::max(
            if grove > 0.3 { 0.65 } else { 0.12 },
            0.6 * smoothstep(0.15, 0.6, dry),
        );
        let lowland = 1.0 - smoothstep(12.0, 25.0, height);
        let cherry = 0.03 + (0.08 * (1.0 - wood) + 0.06 * smoothstep(0.0, -0.6, dry)) * lowland;
        if u < cherry {
            Species::Cherry
        } else if u < cherry + birch * (1.0 - cherry) {
            Species::Birch
        } else {
            Species::Oak
        }
    }

    /// The feature of grid cell `(gx, gz)` if its trunk column lies in `[x0, x1) × [z0, z1)`,
    /// standing on the settled level-0 ground (`settled`) or, for the stand-ins of the canopy
    /// levels, on the unsettled field (at most a few metres apart, inside one coarse cell;
    /// whether and what a feature is never depends on it).
    fn tree_at(
        &self,
        cell: [i64; 2],
        x: [i64; 2],
        z: [i64; 2],
        terrain: &Terrain,
        settled: bool,
    ) -> Option<Tree> {
        // The grid cell coordinates enter the hash as their two's complement bits.
        let hash = hash_combine(hash_combine(self.seed, cell[0] as u64), cell[1] as u64);
        let mut rng = Rng::new(hash, 0);
        // Every draw happens in a fixed order, whatever the outcome of the tests below.
        let mut draw = || rng.range_u32(0, 1 << 16);
        let exists = draw();
        let span = (i64::from(TREE_GRID) - 2 * GRID_MARGIN) as u32;
        let tx = cell[0] * i64::from(TREE_GRID) + GRID_MARGIN + i64::from(draw() % span);
        let tz = cell[1] * i64::from(TREE_GRID) + GRID_MARGIN + i64::from(draw() % span);
        let draws = Draws {
            exists,
            size: draw(),
            species: draw(),
            shape: [draw(), draw(), draw(), draw()],
            age: draw(),
        };
        if self.density <= 0.0 || tx < x[0] || tx >= x[1] || tz < z[0] || tz >= z[1] {
            return None;
        }
        let (cx, cz) = (tx as f64 + 0.5, tz as f64 + 0.5);
        let wood = self.wood(cx, cz, terrain);
        let edge = 4.0 * wood * (1.0 - wood);
        let copse = self.copse(cx, cz);
        let p_tree = self.density * wood;
        // Bushes along forest edges and around copses, rarely alone in a meadow.
        let p_bush = (0.008 + 0.3 * edge + 0.15 * copse) * self.density;
        // Boulders in boulder fields, more of them on slopes and at forest edges.
        let field =
            BOULDER_FIELD.sample2(self.boulder_seed, DVec2::new(cx, cz) * BOULDER_FREQUENCY);
        let p_boulder = if 1.6 * field > BOULDER_FIELD_FROM {
            let slope = terrain.slope_at(0, cx, cz);
            (0.004 + 0.1 * smoothstep(0.3, 0.8, slope) + 0.05 * edge) * self.density
        } else {
            0.0
        };
        let u = unit(draws.exists);
        let open = 1.0 - p_tree;
        enum Class {
            Tree,
            Bush,
            Boulder,
        }
        let mut class = if u < p_tree {
            Class::Tree
        } else if u < p_tree + open * p_bush {
            Class::Bush
        } else if u < p_tree + open * (p_bush + p_boulder) {
            Class::Boulder
        } else {
            return None;
        };
        // Whether and what the feature is depends on the full-resolution field at its trunk,
        // on every level alike.
        let ground = terrain.height(0, cx, cz);
        if ground < terrain.sea_level() + SHORE_CLEARANCE {
            return None;
        }
        let slope = terrain.slope_at(0, cx, cz);
        if terrain.surface(0, cx, cz, ground, slope) != Surface::Grass {
            return None;
        }
        // Above the tree line a few trees turn into bushes (krummholz), the others are gone.
        let line = terrain.snow_line();
        let below_line = smoothstep(line - TREE_LINE.1, line - TREE_LINE.0, ground);
        if matches!(class, Class::Tree) && unit(draws.shape[3]) >= below_line {
            if unit(draws.shape[2]) >= KRUMMHOLZ {
                return None;
            }
            class = Class::Bush;
        }
        // The size varies from tree to tree more than with the density of the wood; the
        // edge of a wood is a mantle of smaller trees; young trees are a third of the size,
        // veterans wider and taller (see `shape`).
        let mut size = det::clamp(
            0.2 + 0.45 * wood + 0.8 * (unit(draws.size) - 0.5),
            0.05,
            1.0,
        );
        if edge > EDGE_FROM {
            size *= EDGE_SIZE;
        }
        let age = unit(draws.age);
        if age < YOUNG.0 {
            size *= YOUNG.1;
        }
        let veteran = age >= 1.0 - VETERANS;
        let kind = match class {
            Class::Tree => {
                FeatureKind::Tree(self.species(draws.species, (cx, cz), ground, wood, terrain))
            }
            // Bushes are oak (its leaves take the foliage tint of the place) or, among
            // conifers, spruce; never birch, whose light leaves made yellow cubes.
            Class::Bush => FeatureKind::Bush(
                match self.species(draws.species, (cx, cz), ground, wood, terrain) {
                    Species::Spruce => Species::Spruce,
                    Species::Birch | Species::Oak | Species::Cherry => Species::Oak,
                },
            ),
            Class::Boulder => FeatureKind::Boulder(match terrain.rock_kind(0, cx, cz) {
                Surface::Slate => Rock::Slate,
                Surface::Limestone => Rock::Limestone,
                _ => Rock::Stone,
            }),
        };
        // It stands on the settled ground (levels 0 and 1, and every query for the trees of a
        // place), where its trunk meets the cells of the terrain.
        let floor = if settled {
            terrain.settled_height(0, tx, tz)
        } else {
            ground
        };
        let base = last_centre_below(0, floor) + 1;
        Some(shape(kind, [tx, base, tz], size, draws.shape, veteran))
    }

    /// Every feature whose trunk column lies in `[x0, x1) × [z0, z1)` (voxels), in grid order
    /// (z, then x), standing on the settled ground.
    pub(crate) fn trees_in(&self, x: [i64; 2], z: [i64; 2], terrain: &Terrain) -> Vec<Tree> {
        self.trees_on(x, z, terrain, true)
    }

    /// [`TreePlacer::trees_in`] for `level`: on the canopy levels the stand-ins stand on the
    /// unsettled field ([`TreePlacer::tree_at`]).
    pub(crate) fn trees_in_level(
        &self,
        level: u8,
        x: [i64; 2],
        z: [i64; 2],
        terrain: &Terrain,
    ) -> Vec<Tree> {
        self.trees_on(x, z, terrain, level <= TREE_RASTER_MAX_LEVEL)
    }

    fn trees_on(&self, x: [i64; 2], z: [i64; 2], terrain: &Terrain, settled: bool) -> Vec<Tree> {
        let grid = i64::from(TREE_GRID);
        // Grid cells whose trunk range [g * grid + margin, (g + 1) * grid - margin) meets the
        // interval.
        let first = |lo: i64| ceil_div(lo - grid + GRID_MARGIN + 1, grid);
        let last = |hi: i64| ceil_div(hi - GRID_MARGIN, grid) - 1;
        let mut trees = Vec::new();
        for gz in first(z[0])..=last(z[1]) {
            for gx in first(x[0])..=last(x[1]) {
                if let Some(tree) = self.tree_at([gx, gz], x, z, terrain, settled) {
                    trees.push(tree);
                }
            }
        }
        trees
    }

    /// Every feature that `style` draws into `region` (cells of `level`), in grid order.
    pub(crate) fn trees_for(
        &self,
        level: u8,
        region: &Region,
        style: TreeStyle,
        terrain: &Terrain,
    ) -> Vec<Tree> {
        if level > style.crown_max_level && level > style.trunk_max_level {
            return Vec::new();
        }
        // On level L a feature's cells reach at most TREE_MAX_RADIUS + 2^L m from its trunk.
        let reach = i64::from(TREE_MAX_RADIUS) + (1_i64 << level) + 1;
        let voxels = region.voxels(level);
        let mut trees = self.trees_in_level(
            level,
            [voxels.min[0] - reach, voxels.max[0] + reach],
            [voxels.min[2] - reach, voxels.max[2] + reach],
            terrain,
        );
        trees.retain(|tree| {
            tree.cell_box(level, style)
                .is_some_and(|bounds| bounds.intersects(region))
        });
        trees
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::palette::numbered_palette;
    use crate::terrain::HillsConfig;

    /// Every cell of `level` that the feature draws, by brute force over a generous box; the
    /// part the style draws last wins (the trunk on level 0, the crown from level 1 on for
    /// the landscape).
    fn cells(tree: &Tree, level: u8, style: TreeStyle) -> Vec<([i64; 3], TreePart)> {
        let [first, last] = style.drawing_order(level);
        let mut found = Vec::new();
        let span = 24;
        let around = [tree.x >> level, tree.base >> level, tree.z >> level];
        for y in around[1] - span..around[1] + span {
            for z in around[2] - span..around[2] + span {
                for x in around[0] - span..around[0] + span {
                    let cell = [x, y, z];
                    if tree.has_cell(level, style, last, cell) {
                        found.push((cell, last));
                    } else if tree.has_cell(level, style, first, cell) {
                        found.push((cell, first));
                    }
                }
            }
        }
        found
    }

    /// The level-0 voxel box of a feature grown by what its crown grows on `level` with
    /// `style` (half a cell from level 1 on for the landscape), rounded out to whole metres.
    fn grown_voxel_box(tree: &Tree, level: u8, style: TreeStyle) -> Region {
        let voxels = tree.voxel_box();
        if !style.distant(level) {
            return voxels;
        }
        let grow = ((1_i64 << level) + 1) / 2;
        Region::new(voxels.min.map(|c| c - grow), voxels.max.map(|c| c + grow))
    }

    /// Features of every kind, size and age at random places.
    fn samples(count: usize) -> Vec<Tree> {
        let mut rng = Rng::new(3, 3);
        let kinds = [
            FeatureKind::Tree(Species::Oak),
            FeatureKind::Tree(Species::Birch),
            FeatureKind::Tree(Species::Spruce),
            FeatureKind::Tree(Species::Cherry),
            FeatureKind::Bush(Species::Oak),
            FeatureKind::Bush(Species::Spruce),
            FeatureKind::Boulder(Rock::Stone),
            FeatureKind::Boulder(Rock::Limestone),
        ];
        (0..count)
            .map(|i| {
                let place = [
                    i64::from(rng.range_u32(0, 200)) - 100,
                    i64::from(rng.range_u32(0, 80)) - 40,
                    i64::from(rng.range_u32(0, 200)) - 100,
                ];
                let size = unit(rng.range_u32(0, 1 << 16));
                let draws = [0; 4].map(|_| rng.range_u32(0, 1 << 16));
                // Every third feature of a kind a veteran, and the largest sizes often.
                let size = if i % 7 == 0 { 1.0 } else { size };
                shape(kinds[i % kinds.len()], place, size, draws, i % 3 == 0)
            })
            .collect()
    }

    #[test]
    fn the_structure_tree_is_a_sphere_on_a_trunk() {
        let t = Tree::sphere(5, -3, 10, 5, 6);
        let found = cells(&t, 0, LANDSCAPE_TREES);
        let trunk: Vec<_> = found.iter().filter(|c| c.1 == TreePart::Trunk).collect();
        assert_eq!(trunk.len(), 5);
        assert!(trunk.iter().all(|c| c.0[0] == 5 && c.0[2] == -3));
        // A 3 m crown around voxel (5, 14, -3): every voxel within 3 m by its centre.
        let crown = found.iter().filter(|c| c.1 == TreePart::Crown).count();
        let expected = (-3_i64..=3)
            .flat_map(|dx| {
                (-3_i64..=3).flat_map(move |dy| (-3_i64..=3).map(move |dz| (dx, dy, dz)))
            })
            .filter(|&(dx, dy, dz)| dx * dx + dy * dy + dz * dz <= 9)
            .filter(|&(dx, dy, dz)| !(dx == 0 && dz == 0 && (-4..=0).contains(&dy)))
            .count();
        assert_eq!(crown, expected);
    }

    #[test]
    fn every_feature_stays_inside_the_declared_radius_and_height() {
        for t in samples(400) {
            let found = cells(&t, 0, LANDSCAPE_TREES);
            assert!(!found.is_empty(), "{t:?}");
            for (cell, _) in &found {
                assert!((cell[0] - t.x).abs() < i64::from(TREE_MAX_RADIUS), "{t:?}");
                assert!((cell[2] - t.z).abs() < i64::from(TREE_MAX_RADIUS), "{t:?}");
                // Boulders sink up to 1.5 m into the ground (where the terrain keeps its own
                // cells).
                assert!(cell[1] >= t.base - 2, "{t:?}");
                assert!(cell[1] < t.base + i64::from(TREE_MAX_HEIGHT), "{t:?}");
            }
        }
    }

    #[test]
    fn shapes_differ_by_species() {
        let base = [0, 0, 0];
        let draws = [1, 2, 3, 4];
        let leaves = |kind| {
            let t = shape(kind, base, 0.8, draws, false);
            let found = cells(&t, 0, LANDSCAPE_TREES);
            let crown: Vec<[i64; 3]> = found
                .iter()
                .filter(|c| c.1 == TreePart::Crown)
                .map(|c| c.0)
                .collect();
            let height = crown.iter().map(|c| c[1]).max().unwrap()
                - crown.iter().map(|c| c[1]).min().unwrap()
                + 1;
            let width = crown.iter().map(|c| c[0]).max().unwrap()
                - crown.iter().map(|c| c[0]).min().unwrap()
                + 1;
            (t.height, width, height, crown.len())
        };
        let oak = leaves(FeatureKind::Tree(Species::Oak));
        let birch = leaves(FeatureKind::Tree(Species::Birch));
        let spruce = leaves(FeatureKind::Tree(Species::Spruce));
        let cherry = leaves(FeatureKind::Tree(Species::Cherry));
        // Birch: taller trunk and a crown taller than wide; oak and cherry broader than high;
        // spruce the tallest.
        assert!(birch.0 > oak.0, "{birch:?} {oak:?}");
        assert!(birch.2 > birch.1, "{birch:?}");
        assert!(oak.1 >= oak.2, "{oak:?}");
        assert!(cherry.1 > cherry.2, "{cherry:?}");
        assert!(spruce.0 + spruce.2 > oak.0 + oak.2, "{spruce:?} {oak:?}");
        // Bushes and boulders sit on the ground without a trunk.
        for kind in [
            FeatureKind::Bush(Species::Oak),
            FeatureKind::Boulder(Rock::Slate),
        ] {
            let t = shape(kind, base, 0.5, draws, false);
            assert_eq!(t.height, 0);
            let found = cells(&t, 0, LANDSCAPE_TREES);
            assert!(found.iter().all(|c| c.1 == TreePart::Crown));
            assert!(found.iter().any(|c| c.0[1] == 0), "{kind:?}");
            assert!(found.len() <= 40, "{kind:?}: {}", found.len());
        }
        let palette = numbered_palette();
        let boulder = shape(FeatureKind::Boulder(Rock::Slate), base, 0.5, draws, false);
        assert_eq!(boulder.state(TreePart::Crown, &palette), palette.slate);
        let spruce = shape(FeatureKind::Tree(Species::Spruce), base, 0.5, draws, false);
        assert_eq!(spruce.state(TreePart::Trunk, &palette), palette.log_spruce);
        assert_eq!(
            spruce.state(TreePart::Crown, &palette),
            palette.leaves_spruce
        );
    }

    #[test]
    fn boxes_contain_every_drawn_cell_on_every_level() {
        for t in samples(240) {
            for style in [LANDSCAPE_TREES, STRUCTURE_TREES] {
                for level in 0..=5_u8 {
                    let found = cells(&t, level, style);
                    let bounds = t.cell_box(level, style);
                    let voxels = grown_voxel_box(&t, level, style);
                    assert_eq!(found.is_empty(), bounds.is_none(), "{level} {t:?}");
                    for (cell, _) in &found {
                        assert!(bounds.is_some_and(|b| b.contains(*cell)), "{level}");
                        // Every drawn cell meets the voxel box (grown by the crown's growth on
                        // the distant levels), so the box bounds the subtree.
                        let size = 1_i64 << level;
                        let cell_voxels =
                            Region::new(cell.map(|c| c * size), cell.map(|c| (c + 1) * size));
                        assert!(cell_voxels.intersects(&voxels), "{level} {t:?}");
                        // And it lies within the reach used for pulling features.
                        let reach = i64::from(TREE_MAX_RADIUS) + size + 1;
                        assert!(
                            cell_voxels.min[0] >= t.x - reach
                                && cell_voxels.max[0] <= t.x + reach + 1
                        );
                        assert!(
                            cell_voxels.min[2] >= t.z - reach
                                && cell_voxels.max[2] <= t.z + reach + 1
                        );
                    }
                }
            }
            // Landscape trees are a canopy without trunk on levels 2 and 3, and gone above;
            // bushes and boulders end at level 1.
            let small = !matches!(t.kind, FeatureKind::Tree(_));
            assert!(
                cells(&t, 2, LANDSCAPE_TREES)
                    .iter()
                    .all(|c| c.1 == TreePart::Crown)
            );
            assert_eq!(cells(&t, 2, LANDSCAPE_TREES).is_empty(), small, "{t:?}");
            assert_eq!(!cells(&t, 3, LANDSCAPE_TREES).is_empty(), !small);
            assert!(cells(&t, 4, LANDSCAPE_TREES).is_empty());
            assert!(!cells(&t, 1, LANDSCAPE_TREES).is_empty());
        }
    }

    #[test]
    fn spruce_trunks_stay_inside_their_needles() {
        // Every size and shape draw of a spruce: above the lowest needles every log has
        // needles on at least three of its four sides, and the top 3 m are needles only.
        let mut rng = Rng::new(11, 4);
        for _ in 0..200 {
            let size = unit(rng.range_u32(0, 1 << 16));
            let draws = [0; 4].map(|_| rng.range_u32(0, 1 << 16));
            let veteran = rng.range_u32(0, 2) == 1;
            let t = shape(
                FeatureKind::Tree(Species::Spruce),
                [0, 0, 0],
                size,
                draws,
                veteran,
            );
            let found = cells(&t, 0, LANDSCAPE_TREES);
            let crown: Vec<[i64; 3]> = found
                .iter()
                .filter(|c| c.1 == TreePart::Crown)
                .map(|c| c.0)
                .collect();
            let lowest = crown.iter().map(|c| c[1]).min().expect("needles");
            let tip = crown.iter().map(|c| c[1]).max().expect("needles");
            for (cell, part) in &found {
                if *part != TreePart::Trunk {
                    continue;
                }
                assert!(
                    cell[1] <= tip - 3,
                    "a log in the top 3 m: {cell:?}, tip {tip}"
                );
                if cell[1] >= lowest {
                    let around = [[1, 0], [-1, 0], [0, 1], [0, -1]]
                        .iter()
                        .filter(|[dx, dz]| crown.contains(&[cell[0] + dx, cell[1], cell[2] + dz]))
                        .count();
                    assert!(
                        around >= 3,
                        "bare log at {cell:?}: {around} sides of needles"
                    );
                }
            }
        }
    }

    #[test]
    fn distant_crowns_hide_their_trunks_and_stay_compact() {
        for t in samples(240) {
            if !matches!(t.kind, FeatureKind::Tree(_)) {
                continue;
            }
            for level in 1..=3_u8 {
                let found = cells(&t, level, LANDSCAPE_TREES);
                let crown: Vec<[i64; 3]> = found
                    .iter()
                    .filter(|c| c.1 == TreePart::Crown)
                    .map(|c| c.0)
                    .collect();
                assert!(!crown.is_empty());
                // The trunk ends under the crown: at most three cells of it (6 m, the bare
                // trunk of a tall birch), all below the leaves of its own column, so no
                // coarse trunk shows through or above a crown.
                let trunk: Vec<[i64; 3]> = found
                    .iter()
                    .filter(|c| c.1 == TreePart::Trunk)
                    .map(|c| c.0)
                    .collect();
                assert!(trunk.len() <= 3, "level {level}: {trunk:?} of {t:?}");
                for cell in &trunk {
                    let mut column = crown.iter().filter(|c| c[0] == cell[0] && c[2] == cell[2]);
                    assert!(
                        column.all(|c| c[1] > cell[1]),
                        "level {level}: trunk cell {cell:?} among the leaves of {t:?}"
                    );
                }
                // On the canopy levels no stand-in is a post: at most 1.3 times as tall as
                // wide, give or take the cell that holds the anchor.
                if level >= 2 {
                    let extent = |axis: usize| {
                        crown.iter().map(|c| c[axis]).max().unwrap()
                            - crown.iter().map(|c| c[axis]).min().unwrap()
                            + 1
                    };
                    let width = extent(0).max(extent(2));
                    assert!(
                        10 * extent(1) <= 13 * width + 10,
                        "level {level}: {} tall, {width} wide: {t:?}",
                        extent(1)
                    );
                }
            }
        }
    }

    #[test]
    fn structure_trees_stay_connected_on_every_level() {
        let t = Tree::sphere(125, 8, 0, 6, 6);
        for level in 0..=6_u8 {
            let found: Vec<[i64; 3]> = cells(&t, level, STRUCTURE_TREES)
                .into_iter()
                .map(|c| c.0)
                .collect();
            // Flood fill over face neighbours from the first cell reaches all of them.
            let mut reached = vec![found[0]];
            let mut frontier = vec![found[0]];
            while let Some(cell) = frontier.pop() {
                for other in &found {
                    let distance: i64 = (0..3).map(|a| (other[a] - cell[a]).abs()).sum();
                    if distance == 1 && !reached.contains(other) {
                        reached.push(*other);
                        frontier.push(*other);
                    }
                }
            }
            assert_eq!(reached.len(), found.len(), "level {level}");
        }
    }

    #[test]
    fn placement_is_pull_based_and_level_independent() {
        let config = HillsConfig::default();
        let terrain = Terrain::new(99, &config);
        let placer = TreePlacer::new(99, config.tree_density);
        let all = placer.trees_in([-300, 300], [-300, 300], &terrain);
        assert!(all.len() > 150, "{}", all.len());
        // Any sub-rectangle yields exactly the features of the big query that stand in it.
        for (x, z) in [
            ([-300, 0], [0, 300]),
            ([-17, 23], [-5, 41]),
            ([7, 8], [-300, 300]),
        ] {
            let part = placer.trees_in(x, z, &terrain);
            let expected: Vec<Tree> = all
                .iter()
                .filter(|t| t.x >= x[0] && t.x < x[1] && t.z >= z[0] && t.z < z[1])
                .copied()
                .collect();
            assert_eq!(part, expected);
        }
        // At most one feature per grid cell, inside its margin; bases sit on grass.
        for t in &all {
            let grid = i64::from(TREE_GRID);
            let local = [t.x.rem_euclid(grid), t.z.rem_euclid(grid)];
            assert!(
                local
                    .iter()
                    .all(|&l| (GRID_MARGIN..grid - GRID_MARGIN).contains(&l)),
                "{t:?}"
            );
            let (cx, cz) = (t.x as f64 + 0.5, t.z as f64 + 0.5);
            let ground = terrain.height(0, cx, cz);
            let slope = terrain.slope_at(0, cx, cz);
            assert_eq!(terrain.surface(0, cx, cz, ground, slope), Surface::Grass);
            let settled = terrain.settled_height(0, t.x, t.z);
            assert_eq!(t.base, last_centre_below(0, settled) + 1);
        }
        // No density, no features.
        assert!(
            TreePlacer::new(99, 0.0)
                .trees_in([-300, 300], [-300, 300], &terrain)
                .is_empty()
        );
    }

    #[test]
    fn the_landscape_mixes_species_bushes_boulders_and_clearings() {
        let config = HillsConfig {
            ridge_height: 96.0,
            snow_line: 64.0,
            hill_amplitude: 26.0,
            ..HillsConfig::default()
        };
        let terrain = Terrain::new(0x00C0_FFEE_2026, &config);
        let placer = TreePlacer::new(0x00C0_FFEE_2026, config.tree_density);
        let all = placer.trees_in([-1500, 1500], [-1500, 1500], &terrain);
        let count =
            |pick: &dyn Fn(FeatureKind) -> bool| all.iter().filter(|t| pick(t.kind)).count();
        let oak = count(&|k| k == FeatureKind::Tree(Species::Oak));
        let birch = count(&|k| k == FeatureKind::Tree(Species::Birch));
        let spruce = count(&|k| k == FeatureKind::Tree(Species::Spruce));
        let cherry = count(&|k| k == FeatureKind::Tree(Species::Cherry));
        let bushes = count(&|k| matches!(k, FeatureKind::Bush(_)));
        let boulders = count(&|k| matches!(k, FeatureKind::Boulder(_)));
        let trees = oak + birch + spruce + cherry;
        assert!(trees > 5000, "{trees}");
        for (name, n, share) in [
            ("oak", oak, 0.2),
            ("birch", birch, 0.1),
            ("spruce", spruce, 0.03),
            ("cherry", cherry, 0.01),
        ] {
            assert!(n as f64 > share * trees as f64, "{name}: {n} of {trees}");
        }
        assert!(bushes as f64 > 0.05 * trees as f64, "{bushes}");
        assert!(boulders as f64 > 0.01 * trees as f64, "{boulders}");
        // Clearings: many 64 m squares without a single tree among forested ones, and sizes
        // that vary.
        let mut squares = std::collections::BTreeMap::new();
        for t in &all {
            if matches!(t.kind, FeatureKind::Tree(_)) {
                *squares
                    .entry((t.x.div_euclid(64), t.z.div_euclid(64)))
                    .or_insert(0) += 1;
            }
        }
        let dense = squares.values().filter(|&&n| n >= 30).count();
        let sparse = squares.values().filter(|&&n| n <= 3).count();
        assert!(dense > 50 && sparse > 50, "{dense} {sparse}");
        let heights: std::collections::BTreeSet<i64> = all.iter().map(|t| t.height).collect();
        assert!(heights.len() >= 8, "{heights:?}");
    }
}
