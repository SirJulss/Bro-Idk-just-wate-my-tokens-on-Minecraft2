//! A static LOD selection for the E2 scenes: which chunks of which level to generate and mesh
//! around one camera, and which of their walls to draw.
//!
//! This is not cp-stream (E3): it runs once, at start-up or for a shot, and has no hysteresis,
//! cross-fade or budget governor. It selects columns of chunks in the x/z plane with a quadtree:
//! a column of level `L` is split into its four children while it is coarser than the level
//! its distance asks for. Level 0 covers everything within `level0_radius` of the camera, and
//! beyond it level `L` reaches out to `level0_radius * growth^L`, so a cell of level `L`
//! appears about as large on screen at the outer edge of its ring as a level-0 cell at the edge
//! of the level-0 disc. The plan ends at `max_radius`.
//!
//! Where two levels meet, the surfaces of the coarse and the fine level do not match exactly
//! (contract §5.1 allows a coarse cell of offset). Both sides therefore draw their walls on the
//! faces towards the other level (contract §5.3: wall ranges at LOD borders), which closes every
//! gap whichever surface lies higher.

use std::collections::BTreeSet;

use cp_core::coords::{CHUNK_SIZE, LodKey, MAX_LOD_LEVEL};
use cp_core::det;
use cp_core::face::Face;
use cp_core::math::{DVec3, IVec3};
use cp_render::voxel::WallMask;

/// A column of chunks of one level: the chunks `(level, (x, *, z))`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ColumnKey {
    /// LOD level, `0..=MAX_LOD_LEVEL`.
    pub level: u8,
    /// Chunk position along x, in chunks of `level`.
    pub x: i32,
    /// Chunk position along z, in chunks of `level`.
    pub z: i32,
}

impl ColumnKey {
    /// Edge of the column in metres: `32 << level`.
    pub fn extent(self) -> f64 {
        f64::from(CHUNK_SIZE << self.level)
    }

    /// Smallest x and z of the column, metres.
    pub fn min_xz(self) -> (f64, f64) {
        let extent = self.extent();
        (f64::from(self.x) * extent, f64::from(self.z) * extent)
    }

    /// The four columns one level finer that make up this one.
    ///
    /// # Panics
    /// If `level == 0`.
    pub fn children(self) -> [Self; 4] {
        assert!(self.level > 0, "a level-0 column has no children");
        let level = self.level - 1;
        let (x, z) = (self.x * 2, self.z * 2);
        [(0, 0), (1, 0), (0, 1), (1, 1)].map(|(dx, dz)| Self {
            level,
            x: x + dx,
            z: z + dz,
        })
    }

    /// Horizontal distance from `(x, z)` to the column, 0 inside it (metres).
    pub fn horizontal_distance(self, x: f64, z: f64) -> f64 {
        let (min_x, min_z) = self.min_xz();
        let extent = self.extent();
        let dx = det::max(det::max(min_x - x, x - (min_x + extent)), 0.0);
        let dz = det::max(det::max(min_z - z, z - (min_z + extent)), 0.0);
        (dx * dx + dz * dz).sqrt()
    }
}

/// What a plan covers.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LodParams {
    /// The point the rings are centred on (metres).
    pub center: DVec3,
    /// Level 0 everywhere within this distance (metres, > 0).
    pub level0_radius: f64,
    /// Ring growth per level, > 1: level `L` reaches `level0_radius * growth^L`.
    pub growth: f64,
    /// Nothing beyond this horizontal distance (metres).
    pub max_radius: f64,
    /// Coarsest level used.
    pub top_level: u8,
    /// Whether the outer edge of the plan draws walls too (a skirt down to the bottom of the
    /// outermost chunks) or stays open.
    pub edge_walls: bool,
}

/// One selected column and the walls of its chunks.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PlannedColumn {
    /// The column.
    pub key: ColumnKey,
    /// Walls towards neighbours of another level (only ±X and ±Z faces).
    pub walls: WallMask,
    /// Distance from the plan's centre to the column (metres), for priorities.
    pub distance: f64,
}

/// One chunk to generate and mesh.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PlannedChunk {
    /// The chunk.
    pub key: LodKey,
    /// Its walls.
    pub walls: WallMask,
    /// Distance from the plan's centre (metres).
    pub distance: f64,
}

impl LodParams {
    /// The level a column at `distance` metres (closest point) asks for.
    pub fn level_for(&self, distance: f64) -> u8 {
        if distance <= self.level0_radius {
            return 0;
        }
        // Ring L covers (r0 * g^(L-1), r0 * g^L].
        let rings = det::ln(distance / self.level0_radius) / det::ln(self.growth);
        let level = det::max(rings.ceil(), 1.0);
        if level >= f64::from(self.top_level) {
            self.top_level
        } else {
            // In 1..top_level.
            level as u8
        }
    }
}

/// Selects the columns of a plan. `vertical(level)` gives the height range `(low, high)` in
/// metres that the content of `level` can occupy; it enters the distance to a column.
pub fn plan_columns(params: &LodParams, vertical: impl Fn(u8) -> (f64, f64)) -> Vec<PlannedColumn> {
    let top = params.top_level.min(MAX_LOD_LEVEL);
    let center = params.center;
    let root_extent = f64::from(CHUNK_SIZE << top);
    let reach = params.max_radius;
    let first = |value: f64| ((value - reach) / root_extent).floor() as i32;
    let last = |value: f64| ((value + reach) / root_extent).floor() as i32;
    let mut stack = Vec::new();
    for x in first(center.x)..=last(center.x) {
        for z in first(center.z)..=last(center.z) {
            stack.push(ColumnKey { level: top, x, z });
        }
    }
    let mut leaves = Vec::new();
    while let Some(column) = stack.pop() {
        let horizontal = column.horizontal_distance(center.x, center.z);
        if horizontal > reach {
            continue;
        }
        let (low, high) = vertical(column.level);
        let dy = det::max(det::max(low - center.y, center.y - high), 0.0);
        let distance = (horizontal * horizontal + dy * dy).sqrt();
        if column.level > params.level_for(distance) {
            stack.extend(column.children());
        } else {
            leaves.push((column, distance));
        }
    }
    let set: BTreeSet<ColumnKey> = leaves.iter().map(|(column, _)| *column).collect();
    let mut planned: Vec<PlannedColumn> = leaves
        .into_iter()
        .map(|(key, distance)| {
            let mut walls = WallMask::NONE;
            for (face, dx, dz) in [
                (Face::PosX, 1, 0),
                (Face::NegX, -1, 0),
                (Face::PosZ, 0, 1),
                (Face::NegZ, 0, -1),
            ] {
                let neighbour = ColumnKey {
                    level: key.level,
                    x: key.x + dx,
                    z: key.z + dz,
                };
                if set.contains(&neighbour) {
                    continue;
                }
                let inside = neighbour.horizontal_distance(center.x, center.z) <= reach;
                if inside || params.edge_walls {
                    walls = walls.with(face);
                }
            }
            PlannedColumn {
                key,
                walls,
                distance,
            }
        })
        .collect();
    planned.sort_by(|a, b| {
        a.distance
            .total_cmp(&b.distance)
            .then_with(|| a.key.cmp(&b.key))
    });
    planned
}

/// The chunks of the planned columns between the heights `vertical(level)` (metres), nearest
/// first.
pub fn plan_chunks(
    columns: &[PlannedColumn],
    vertical: impl Fn(u8) -> (f64, f64),
) -> Vec<PlannedChunk> {
    let mut chunks = Vec::new();
    for column in columns {
        let (low, high) = vertical(column.key.level);
        let extent = column.key.extent();
        let first = (low / extent).floor() as i32;
        let last = (high / extent).floor() as i32;
        for y in first..=last {
            chunks.push(PlannedChunk {
                key: LodKey::new(column.key.level, IVec3::new(column.key.x, y, column.key.z)),
                walls: column.walls,
                distance: column.distance,
            });
        }
    }
    chunks
}

#[cfg(test)]
mod tests {
    use super::*;

    fn params() -> LodParams {
        LodParams {
            center: DVec3::new(10.0, 20.0, -30.0),
            level0_radius: 200.0,
            growth: 2.0,
            max_radius: 3000.0,
            top_level: 6,
            edge_walls: false,
        }
    }

    #[test]
    fn levels_grow_with_the_rings() {
        let p = params();
        assert_eq!(p.level_for(0.0), 0);
        assert_eq!(p.level_for(200.0), 0);
        assert_eq!(p.level_for(201.0), 1);
        assert_eq!(p.level_for(400.0), 1);
        assert_eq!(p.level_for(401.0), 2);
        assert_eq!(p.level_for(1.0e9), 6);
    }

    #[test]
    fn the_plan_tiles_the_disc_without_overlap() {
        let p = params();
        let columns = plan_columns(&p, |_| (0.0, 40.0));
        assert!(!columns.is_empty());
        // No two columns overlap, and the camera's column is level 0.
        let boxes: Vec<(f64, f64, f64)> = columns
            .iter()
            .map(|column| {
                let (x, z) = column.key.min_xz();
                (x, z, column.key.extent())
            })
            .collect();
        for (i, a) in boxes.iter().enumerate() {
            for b in &boxes[i + 1..] {
                let apart =
                    a.0 + a.2 <= b.0 || b.0 + b.2 <= a.0 || a.1 + a.2 <= b.1 || b.1 + b.2 <= a.1;
                assert!(apart, "{a:?} {b:?}");
            }
        }
        let own = columns
            .iter()
            .find(|column| column.key.horizontal_distance(10.0, -30.0) == 0.0)
            .expect("the camera's column is planned");
        assert_eq!(own.key.level, 0);
        // Every point of the disc of the max radius minus one root column is covered.
        for (x, z) in [
            (10.0, -30.0),
            (500.0, 500.0),
            (-1500.0, 200.0),
            (2000.0, -1800.0),
        ] {
            let covered = columns
                .iter()
                .any(|column| column.key.horizontal_distance(x, z) == 0.0);
            assert!(covered, "({x}, {z})");
        }
        // Sorted by distance.
        assert!(columns.windows(2).all(|w| w[0].distance <= w[1].distance));
    }

    #[test]
    fn walls_face_other_levels_only() {
        let p = params();
        let columns = plan_columns(&p, |_| (0.0, 40.0));
        let set: BTreeSet<ColumnKey> = columns.iter().map(|column| column.key).collect();
        let mut with_walls = 0;
        for column in &columns {
            for (face, dx, dz) in [
                (Face::PosX, 1, 0),
                (Face::NegX, -1, 0),
                (Face::PosZ, 0, 1),
                (Face::NegZ, 0, -1),
            ] {
                let neighbour = ColumnKey {
                    level: column.key.level,
                    x: column.key.x + dx,
                    z: column.key.z + dz,
                };
                if set.contains(&neighbour) {
                    assert!(!column.walls.contains(face), "{column:?} {face:?}");
                }
            }
            assert!(!column.walls.contains(Face::PosY) && !column.walls.contains(Face::NegY));
            with_walls += usize::from(!column.walls.is_empty());
        }
        assert!(with_walls > 0);
        // The chunks of a column span the vertical range.
        let chunks = plan_chunks(&columns[..1], |_| (-5.0, 40.0));
        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[0].key.pos.y, -1);
        assert_eq!(chunks[2].key.pos.y, 1);
    }
}
