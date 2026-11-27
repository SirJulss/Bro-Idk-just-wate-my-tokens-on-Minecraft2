//! The E1 structure set on flat ground as a [`BaseSource`] (contract §9, E1 acceptance).

use cp_core::block::BlockFlags;
use cp_core::coords::LodKey;
use cp_core::ids::BlockStateId;
use cp_core::math::{IAabb, IVec3};
use cp_voxel::{BaseSource, Cancelled, GeneratorIdentity, NodeClass, chunk_cells};

use crate::config_hash::ConfigHasher;
use crate::error::WorldgenError;
use crate::palette::{TerrainPalette, is_opaque};
use crate::region::{Region, cells_overlapping, centre, checked_request};
use crate::strata::FlatGround;
use crate::trees::{Canvas, STRUCTURE_TREES, Tree, TreePart};

/// The shapes of the structure set.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum StructureKind {
    /// A free-standing 1 × 12 × 1 column of logs.
    Pillar,
    /// A wall one voxel thick, 16 m long and 8 m high; the first copy crosses a level-0
    /// chunk border.
    Wall,
    /// A 16 m horizontal beam, 1 × 1, resting on two 7 m posts.
    Beam,
    /// A 9 m cube with a one-voxel shell around an empty interior.
    HollowCube,
    /// A tree: a 6 m log trunk with a 3 m crown of leaves.
    Tree,
}

/// One shape of a [`StructureField`] and the level-0 voxels it occupies (half-open box).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct StructureShape {
    /// What it is.
    pub kind: StructureKind,
    /// Its voxels lie inside this box; it touches all six sides.
    pub bounds: IAabb,
}

/// The structure set of the E1 acceptance (pillar, one-thick wall, beam, hollow cube, tree)
/// on flat grassland, for LOD tests and golden images.
///
/// The ground is grass (1 m) over dirt (3 m) over stone below [`StructureField::GROUND_Y`].
/// The layout stands twice: at the origin and shifted by [`StructureField::SECOND_COPY`] (an
/// offset that is not a multiple of any coarse cell, so the copies meet the cell grid of every
/// level differently). See [`StructureField::shapes`].
///
/// # Levels
/// The shapes are structures, kept **conservatively** on every level so that each one stays a
/// single connected component: a cell is a log iff it overlaps a log voxel of a shape (the
/// downsample rule keeps structural cells the same way), and the tree's crown is drawn by the
/// point rule plus the cell that contains the crown centre, which is always a trunk cell. A
/// one-thick wall therefore becomes a wall one coarse cell thick, the hollow cube fills up once
/// its interior is smaller than a cell.
#[derive(Clone, Debug)]
pub struct StructureField {
    identity: GeneratorIdentity,
    palette: TerrainPalette,
    ground: FlatGround,
    shapes: Vec<StructureShape>,
    /// Log voxel boxes (half-open) of all shapes.
    boxes: Vec<Region>,
    trees: Vec<Tree>,
    ground_opaque: bool,
}

/// A box of voxels relative to a layout copy's origin: `(min, max)`, half-open.
type LayoutBox = ([i64; 3], [i64; 3]);

/// One copy of the layout: each shape with its log boxes. The tree is added separately.
const LAYOUT: [(StructureKind, &[LayoutBox]); 4] = [
    (StructureKind::Pillar, &[([8, 0, 8], [9, 12, 9])]),
    (StructureKind::Wall, &[([27, 0, 8], [43, 8, 9])]),
    (
        StructureKind::Beam,
        &[
            ([61, 0, 8], [62, 7, 9]),
            ([76, 0, 8], [77, 7, 9]),
            ([61, 7, 8], [77, 8, 9]),
        ],
    ),
    (
        StructureKind::HollowCube,
        &[
            ([95, 0, 4], [96, 9, 13]),
            ([103, 0, 4], [104, 9, 13]),
            ([96, 0, 4], [103, 1, 13]),
            ([96, 8, 4], [103, 9, 13]),
            ([96, 1, 4], [103, 8, 5]),
            ([96, 1, 12], [103, 8, 13]),
        ],
    ),
];
/// The tree's trunk column relative to the copy's origin, its trunk height and crown radius in
/// half metres.
const LAYOUT_TREE: ([i64; 2], i64, i64) = ([125, 8], 6, 6);

impl StructureField {
    /// Identity key of this generator.
    pub const KEY: &'static str = "cubeplanet:structure_field";
    /// Output version; bumped on every change of the layout or of the generated cells.
    pub const VERSION: u32 = 1;
    /// Height of the ground surface (voxel y); the shapes stand on it.
    pub const GROUND_Y: i32 = 0;
    /// Offset `(x, z)` of the layout's second copy.
    pub const SECOND_COPY: [i32; 2] = [5, 43];

    /// The structure field with the block states of `palette`; shapes are made of
    /// `palette.log`, the tree's crown of `palette.leaves`. `seed` is recorded in the identity
    /// (the layout does not use it); `flags` decides which ground states are `OPAQUE`.
    ///
    /// # Errors
    /// [`WorldgenError::InvalidConfig`] if the palette contains [`BlockStateId::UNCHANGED`].
    pub fn new(
        seed: u64,
        palette: TerrainPalette,
        flags: &[BlockFlags],
    ) -> Result<Self, WorldgenError> {
        palette.validate()?;
        let identity = GeneratorIdentity {
            key: Self::KEY.to_owned(),
            version: Self::VERSION,
            seed,
            macro_hash: ConfigHasher::new(Self::KEY)
                .i32(Self::GROUND_Y)
                .u64(palette.key_hash)
                .finish(),
        };
        let mut shapes = Vec::new();
        let mut boxes = Vec::new();
        let mut trees = Vec::new();
        let ground_y = i64::from(Self::GROUND_Y);
        for origin in [[0, 0], Self::SECOND_COPY.map(i64::from)] {
            let shift = |p: [i64; 3]| [p[0] + origin[0], p[1] + ground_y, p[2] + origin[1]];
            for (kind, parts) in LAYOUT {
                let parts: Vec<Region> = parts
                    .iter()
                    .map(|&(min, max)| Region::new(shift(min), shift(max)))
                    .collect();
                let bounds = parts
                    .iter()
                    .skip(1)
                    .fold(parts[0], |bounds, part| bounds.union(part));
                shapes.push(StructureShape {
                    kind,
                    bounds: to_aabb(&bounds),
                });
                boxes.extend(parts);
            }
            let ([x, z], height, radius_halves) = LAYOUT_TREE;
            let tree = Tree::sphere(
                x + origin[0],
                z + origin[1],
                ground_y,
                height,
                radius_halves,
            );
            shapes.push(StructureShape {
                kind: StructureKind::Tree,
                bounds: to_aabb(&tree.voxel_box()),
            });
            trees.push(tree);
        }
        let ground = FlatGround::new(
            f64::from(Self::GROUND_Y),
            vec![(palette.grass, 1.0), (palette.dirt, 4.0)],
            palette.stone,
        );
        let ground_opaque = ground.states().all(|state| is_opaque(flags, state));
        log::debug!("structure field {identity:?}");
        Ok(Self {
            identity,
            palette,
            ground,
            shapes,
            boxes,
            trees,
            ground_opaque,
        })
    }

    /// The shapes of both copies, in layout order (pillar, wall, beam, hollow cube, tree; then
    /// the same for the second copy).
    #[must_use]
    pub fn shapes(&self) -> &[StructureShape] {
        &self.shapes
    }

    /// True when some shape draws a cell of `level` into `cells` (conservative: the shape's
    /// cell box meets them).
    fn shapes_reach(&self, level: u8, cells: &Region) -> bool {
        self.boxes
            .iter()
            .any(|part| overlapping_cells(level, part).intersects(cells))
            || self.trees.iter().any(|tree| {
                tree.cell_box(level, STRUCTURE_TREES)
                    .is_some_and(|bounds| bounds.intersects(cells))
            })
    }
}

/// The cells of `level` that overlap any voxel of `voxels`.
fn overlapping_cells(level: u8, voxels: &Region) -> Region {
    let x = cells_overlapping(level, voxels.min[0], voxels.max[0]);
    let y = cells_overlapping(level, voxels.min[1], voxels.max[1]);
    let z = cells_overlapping(level, voxels.min[2], voxels.max[2]);
    Region::new([x.0, y.0, z.0], [x.1, y.1, z.1])
}

fn to_aabb(region: &Region) -> IAabb {
    // The layout lies within a few hundred metres of the origin.
    let point = |p: [i64; 3]| IVec3::new(p[0] as i32, p[1] as i32, p[2] as i32);
    IAabb::new(point(region.min), point(region.max))
}

impl BaseSource for StructureField {
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
        let slab = region.extent(0) * region.extent(2);
        for (rows, y) in out.chunks_exact_mut(slab).zip(region.min[1]..region.max[1]) {
            if cancel() {
                return Err(Cancelled);
            }
            rows.fill(self.ground.row_state(level, y));
        }
        let open_from = self.ground.open_from(level);
        if region.max[1] <= open_from {
            return Ok(());
        }
        let writable = |_x: i64, y: i64, _z: i64| y >= open_from;
        let mut canvas = Canvas { level, region, out };
        // Crowns first: logs win over leaves. The trees are oaks: oak leaves and logs.
        for part in [TreePart::Crown, TreePart::Trunk] {
            for tree in &self.trees {
                tree.draw(&mut canvas, STRUCTURE_TREES, part, &self.palette, writable);
            }
        }
        for part in &self.boxes {
            let Some(cells) = overlapping_cells(level, part).intersection(&region) else {
                continue;
            };
            for y in cells.min[1].max(open_from)..cells.max[1] {
                for z in cells.min[2]..cells.max[2] {
                    for x in cells.min[0]..cells.max[0] {
                        canvas.out[region.index(x, y, z)] = self.palette.log;
                    }
                }
            }
        }
        Ok(())
    }

    fn classify(&self, key: LodKey) -> NodeClass {
        if !key.is_valid() {
            return NodeClass::Mixed;
        }
        let level = key.level;
        let cells = Region::from_aabb(chunk_cells(key));
        let ground = f64::from(Self::GROUND_Y);
        if centre(level, cells.max[1] - 1) < ground {
            // Every cell is ground.
            return if self.ground_opaque {
                NodeClass::Opaque
            } else {
                NodeClass::Mixed
            };
        }
        if centre(level, cells.min[1]) >= ground && !self.shapes_reach(level, &cells) {
            return NodeClass::Empty;
        }
        NodeClass::Mixed
    }

    fn subtree_empty(&self, key: LodKey) -> bool {
        if !key.is_valid() || key.origin_voxel().y < Self::GROUND_Y {
            return false;
        }
        // A shape's cell on any level overlaps its voxels or (crown) has its centre in its
        // bounds; one voxel of margin keeps the test safe at the box borders.
        let voxels = Region::from_aabb(key.aabb_voxels());
        !self
            .shapes
            .iter()
            .any(|shape| Region::from_aabb(shape.bounds.expand(1)).intersects(&voxels))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cp_core::coords::MAX_LOD_LEVEL;

    /// Grass 1, dirt 2, stone 3, oak log 8, oak leaves 9 (the roles the field uses).
    fn palette() -> TerrainPalette {
        crate::palette::numbered_palette()
    }

    fn field() -> StructureField {
        let solid = BlockFlags::SOLID | BlockFlags::OPAQUE;
        let mut flags = vec![solid; 10];
        flags[0] = BlockFlags::empty();
        StructureField::new(0, palette(), &flags).unwrap()
    }

    /// The cells of `level` inside `bounds` (voxels, grown by a margin) that hold logs or
    /// leaves.
    fn shape_cells(field: &StructureField, level: u8, bounds: IAabb) -> Vec<[i64; 3]> {
        let voxels = Region::from_aabb(bounds.expand(2));
        let cells = overlapping_cells(level, &voxels);
        let mut out = vec![BlockStateId::AIR; cells.volume()];
        field
            .fill(level, to_aabb(&cells), &mut out, &|| false)
            .unwrap();
        let mut found = Vec::new();
        for y in cells.min[1]..cells.max[1] {
            for z in cells.min[2]..cells.max[2] {
                for x in cells.min[0]..cells.max[0] {
                    let state = out[cells.index(x, y, z)];
                    if state == field.palette.log || state == field.palette.leaves {
                        found.push([x, y, z]);
                    }
                }
            }
        }
        found
    }

    /// Number of face-connected components of a set of cells.
    fn components(cells: &[[i64; 3]]) -> usize {
        let mut seen = vec![false; cells.len()];
        let mut count = 0;
        while let Some(start) = seen.iter().position(|&s| !s) {
            count += 1;
            seen[start] = true;
            let mut stack = vec![start];
            while let Some(i) = stack.pop() {
                for (j, other) in cells.iter().enumerate() {
                    let distance: i64 = (0..3).map(|a| (other[a] - cells[i][a]).abs()).sum();
                    if !seen[j] && distance == 1 {
                        seen[j] = true;
                        stack.push(j);
                    }
                }
            }
        }
        count
    }

    #[test]
    fn layout_has_the_structure_set_twice() {
        let field = field();
        let kinds: Vec<StructureKind> = field.shapes().iter().map(|s| s.kind).collect();
        let set = [
            StructureKind::Pillar,
            StructureKind::Wall,
            StructureKind::Beam,
            StructureKind::HollowCube,
            StructureKind::Tree,
        ];
        assert_eq!(kinds, [set, set].concat());
        // The first wall crosses the border between the level-0 chunks 0 and 1.
        let wall = field.shapes()[1].bounds;
        assert!(wall.min.x < 32 && wall.max.x > 32);
        // Shapes stay apart on level 3 (cells of 8 m), so each is its own component there.
        for (i, a) in field.shapes().iter().enumerate() {
            for b in &field.shapes()[i + 1..] {
                let grown = Region::from_aabb(a.bounds.expand(16));
                assert!(
                    !grown.intersects(&Region::from_aabb(b.bounds)),
                    "{a:?} {b:?}"
                );
            }
        }
    }

    #[test]
    fn shapes_are_exact_on_level_zero() {
        let field = field();
        let pillar = shape_cells(&field, 0, field.shapes()[0].bounds);
        assert_eq!(pillar.len(), 12);
        let wall = shape_cells(&field, 0, field.shapes()[1].bounds);
        assert_eq!(wall.len(), 16 * 8);
        let beam = shape_cells(&field, 0, field.shapes()[2].bounds);
        assert_eq!(beam.len(), 7 + 7 + 16);
        let cube = shape_cells(&field, 0, field.shapes()[3].bounds);
        assert_eq!(cube.len(), 9 * 9 * 9 - 7 * 7 * 7);
        let tree = shape_cells(&field, 0, field.shapes()[4].bounds);
        assert!(tree.len() > 6 + 100, "{}", tree.len());
        for shape in field.shapes() {
            // The bounds are tight on level 0.
            let cells = shape_cells(&field, 0, shape.bounds);
            for axis in 0..3 {
                let min = cells.iter().map(|c| c[axis]).min().unwrap();
                let max = cells.iter().map(|c| c[axis]).max().unwrap() + 1;
                assert_eq!(min, i64::from(shape.bounds.min[axis]), "{shape:?}");
                assert_eq!(max, i64::from(shape.bounds.max[axis]), "{shape:?}");
            }
        }
    }

    #[test]
    fn every_shape_stays_one_connected_component_on_levels_one_to_three() {
        let field = field();
        for shape in field.shapes() {
            for level in 0..=3_u8 {
                let cells = shape_cells(&field, level, shape.bounds);
                assert!(!cells.is_empty(), "{shape:?} level {level}");
                assert_eq!(components(&cells), 1, "{shape:?} level {level}");
            }
        }
        // The hollow cube is still hollow on level 1.
        let cube = shape_cells(&field, 1, field.shapes()[3].bounds);
        let interior = overlapping_cells(1, &Region::new([96, 1, 5], [103, 8, 12]));
        let inside = cube
            .iter()
            .filter(|c| (0..3).all(|a| c[a] > interior.min[a] && c[a] < interior.max[a] - 1))
            .count();
        assert_eq!(inside, 0);
    }

    #[test]
    fn classify_and_subtree_never_lie() {
        let field = field();
        let (mut empty, mut opaque) = (0, 0);
        for level in 0..=MAX_LOD_LEVEL.min(8) {
            for y in -2..=2 {
                for z in -2..=2 {
                    for x in -2..=5 {
                        let key = LodKey::new(level, IVec3::new(x, y, z));
                        let class = field.classify(key);
                        if class == NodeClass::Mixed && !field.subtree_empty(key) {
                            continue;
                        }
                        let mut out = vec![BlockStateId::UNCHANGED; 32 * 32 * 32];
                        field
                            .fill(level, chunk_cells(key), &mut out, &|| false)
                            .unwrap();
                        match class {
                            NodeClass::Empty => {
                                empty += 1;
                                assert!(out.iter().all(|s| s.is_air()), "{key:?}");
                            }
                            NodeClass::Opaque => {
                                opaque += 1;
                                assert!(out.iter().all(|s| (1..=3).contains(&s.0)), "{key:?}");
                            }
                            NodeClass::Mixed => {}
                        }
                        if field.subtree_empty(key) {
                            assert_eq!(class, NodeClass::Empty, "{key:?}");
                            // Sampled descendants are empty as well.
                            let mut child = key;
                            while child.level > 0 {
                                child = child.children()[(child.pos.x & 7) as usize];
                                let mut cells = vec![BlockStateId::UNCHANGED; 32 * 32 * 32];
                                field
                                    .fill(child.level, chunk_cells(child), &mut cells, &|| false)
                                    .unwrap();
                                assert!(cells.iter().all(|s| s.is_air()), "{child:?}");
                            }
                        }
                    }
                }
            }
        }
        assert!(empty > 100 && opaque > 100, "{empty} {opaque}");
    }
}
