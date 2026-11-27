//! Voxel and chunk coordinates (contract §2.2).
//!
//! - A **voxel** `(x, y, z)` (`IVec3`, body space) spans `[x, x+1)` m on every axis.
//! - A **chunk** is a cube of 32³ **cells**. At LOD `level` a cell has an edge of `2^level` m,
//!   so the chunk spans `32 << level` voxels; its minimum corner sits at
//!   `pos * (32 << level)` voxels. Level 0 cells are voxels.
//! - Chunk and local coordinates use floor semantics: `chunk = voxel >> 5`, `local = voxel & 31`.
//! - Cells inside a chunk are stored in the canonical order `index = x + z * 32 + y * 1024`.
//! - Every voxel grid (a body or a moving construct, [`GridId`]) has its own octree:
//!   [`ChunkKey`] `{ grid, lod }` names a chunk of it, [`CellAddress`] `{ grid, cell }` a voxel.
//!
//! These rules are shared with the GPU shaders and with saved data; they must not change.
//!
//! ```
//! use cp_core::coords::{LodKey, cell_index, voxel_to_chunk};
//! use cp_core::math::IVec3;
//!
//! // Voxel (-1, 40, 5) lives in chunk (-1, 1, 0) at local cell (31, 8, 5).
//! let (chunk, [x, y, z]) = voxel_to_chunk(IVec3::new(-1, 40, 5));
//! assert_eq!((chunk, [x, y, z]), (IVec3::new(-1, 1, 0), [31, 8, 5]));
//! assert_eq!(cell_index(x, y, z), 31 + 5 * 32 + 8 * 1024);
//!
//! // The same place two LOD levels up: cells of 4 m, chunks of 128 m.
//! let key = LodKey::lod0(chunk).ancestor_at(2).unwrap();
//! assert_eq!(key, LodKey::new(2, IVec3::new(-1, 0, 0)));
//! assert_eq!(key.origin_voxel(), IVec3::new(-128, 0, 0));
//! assert!(key.contains(&LodKey::lod0(chunk)));
//! ```

use core::cmp::Ordering;

use serde::{Deserialize, Serialize};

use crate::ids::GridId;
use crate::math::{IAabb, IVec3};

/// log2 of the chunk edge in cells.
pub const CHUNK_BITS: u32 = 5;
/// Chunk edge in cells.
pub const CHUNK_SIZE: i32 = 1 << CHUNK_BITS;
/// Mask extracting the local cell coordinate from a voxel coordinate (`CHUNK_SIZE - 1`).
pub const CHUNK_MASK: i32 = CHUNK_SIZE - 1;
/// Number of cells in a chunk.
pub const CHUNK_VOLUME: usize = 1 << (3 * CHUNK_BITS);
/// Coarsest LOD level: cells of 2^20 m, chunks of 2^25 m.
pub const MAX_LOD_LEVEL: u8 = 20;
/// Voxel coordinates of [valid](LodKey::is_valid) chunks lie in
/// `[-VOXEL_LIMIT, VOXEL_LIMIT)` on every axis (2^30, half the `i32` range, so that sums and
/// differences of valid coordinates cannot overflow).
pub const VOXEL_LIMIT: i32 = 1 << 30;

/// Identifies one chunk of a body's LOD octree: `pos` counts chunks of level `level`.
///
/// The chunk covers the voxels `[pos * (32 << level), (pos + 1) * (32 << level))`. A key is
/// [valid](LodKey::is_valid) when `level <= MAX_LOD_LEVEL` and that range lies inside
/// `[-VOXEL_LIMIT, VOXEL_LIMIT)`; the methods assume valid keys and return unspecified (but
/// never undefined) results otherwise. Parents, children and ancestors of valid keys are valid.
///
/// Ordered by `level`, then `pos.x`, `pos.y`, `pos.z` — the order to use wherever results must
/// not depend on hash-map iteration (contract §2.5).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Serialize, Deserialize)]
pub struct LodKey {
    /// LOD level, `0..=MAX_LOD_LEVEL`. Cell size is `2^level` m.
    pub level: u8,
    /// Chunk position in units of this level's chunk extent.
    pub pos: IVec3,
}

impl LodKey {
    /// Key from level and chunk position.
    #[inline]
    #[must_use]
    pub const fn new(level: u8, pos: IVec3) -> Self {
        Self { level, pos }
    }

    /// The full-resolution (level 0) chunk at chunk position `pos`.
    #[inline]
    #[must_use]
    pub const fn lod0(pos: IVec3) -> Self {
        Self { level: 0, pos }
    }

    /// The chunk of `level` that contains the level-0 voxel `voxel`.
    #[inline]
    #[must_use]
    pub fn containing(level: u8, voxel: IVec3) -> Self {
        // The clamp only matters for invalid levels, where it avoids an overflowing shift.
        Self {
            level,
            pos: voxel >> (CHUNK_BITS + u32::from(level)).min(31),
        }
    }

    /// True when the level is in range and the chunk lies inside the voxel limits.
    #[inline]
    #[must_use]
    pub fn is_valid(&self) -> bool {
        if self.level > MAX_LOD_LEVEL {
            return false;
        }
        // Chunks per axis on each side of the origin: VOXEL_LIMIT / extent.
        let limit = VOXEL_LIMIT >> (CHUNK_BITS + u32::from(self.level));
        self.pos.cmpge(IVec3::splat(-limit)).all() && self.pos.cmplt(IVec3::splat(limit)).all()
    }

    /// Edge length of one cell in voxels (= metres): `2^level`.
    #[inline]
    #[must_use]
    pub const fn cell_size(&self) -> i32 {
        1_i32.wrapping_shl(self.level as u32)
    }

    /// Edge length of the chunk in voxels (= metres): `32 << level`.
    #[inline]
    #[must_use]
    pub const fn chunk_extent(&self) -> i32 {
        CHUNK_SIZE.wrapping_shl(self.level as u32)
    }

    /// Minimum corner of the chunk in level-0 voxels: `pos * chunk_extent()`.
    #[inline]
    #[must_use]
    pub const fn origin_voxel(&self) -> IVec3 {
        let extent = self.chunk_extent();
        IVec3::new(
            self.pos.x.wrapping_mul(extent),
            self.pos.y.wrapping_mul(extent),
            self.pos.z.wrapping_mul(extent),
        )
    }

    /// The level-0 voxels covered by the chunk (half-open box).
    #[inline]
    #[must_use]
    pub const fn aabb_voxels(&self) -> IAabb {
        let min = self.origin_voxel();
        let extent = self.chunk_extent();
        IAabb::new(
            min,
            IVec3::new(
                min.x.wrapping_add(extent),
                min.y.wrapping_add(extent),
                min.z.wrapping_add(extent),
            ),
        )
    }

    /// The chunk one level coarser that contains this one, or `None` at [`MAX_LOD_LEVEL`].
    #[inline]
    #[must_use]
    pub fn parent(&self) -> Option<Self> {
        self.ancestor_at(self.level.checked_add(1)?)
    }

    /// The ancestor at `level`, which may be this key itself. `None` when `level` is finer
    /// than this key's level or above [`MAX_LOD_LEVEL`].
    #[inline]
    #[must_use]
    pub fn ancestor_at(&self, level: u8) -> Option<Self> {
        if level < self.level || level > MAX_LOD_LEVEL {
            return None;
        }
        // Arithmetic shift is floor division, which keeps negative positions correct.
        Some(Self {
            level,
            pos: self.pos >> u32::from(level - self.level),
        })
    }

    /// The eight chunks one level finer that make up this chunk.
    ///
    /// Child `i` sits at offset `(i & 1, (i >> 2) & 1, (i >> 1) & 1)` in `(x, y, z)`, i.e.
    /// `i = x + z * 2 + y * 4`: x fastest, y slowest, like the canonical cell order.
    ///
    /// # Panics
    /// If `level == 0`; level-0 chunks have no children.
    #[inline]
    #[must_use]
    pub fn children(&self) -> [Self; 8] {
        assert!(
            self.level > 0,
            "LodKey::children: a level-0 chunk has no children"
        );
        core::array::from_fn(|octant| Self {
            level: self.level - 1,
            pos: self
                .pos
                .wrapping_mul(IVec3::splat(2))
                .wrapping_add(Self::octant_offset(octant)),
        })
    }

    /// Index of this chunk within [`LodKey::children`] of its parent (`0..8`).
    #[inline]
    #[must_use]
    pub const fn octant(&self) -> usize {
        ((self.pos.x & 1) | ((self.pos.z & 1) << 1) | ((self.pos.y & 1) << 2)) as usize
    }

    /// Offset of child `octant` (`0..8`) in child chunks: each component is 0 or 1.
    #[inline]
    #[must_use]
    pub const fn octant_offset(octant: usize) -> IVec3 {
        IVec3::new(
            (octant & 1) as i32,
            ((octant >> 2) & 1) as i32,
            ((octant >> 1) & 1) as i32,
        )
    }

    /// True when `other` lies inside this chunk: same chunk or a descendant of it.
    #[inline]
    #[must_use]
    pub fn contains(&self, other: &Self) -> bool {
        other.ancestor_at(self.level) == Some(*self)
    }
}

impl Ord for LodKey {
    fn cmp(&self, other: &Self) -> Ordering {
        (self.level, self.pos.x, self.pos.y, self.pos.z).cmp(&(
            other.level,
            other.pos.x,
            other.pos.y,
            other.pos.z,
        ))
    }
}

impl PartialOrd for LodKey {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// A chunk of a specific voxel grid (contract §2.2). Ordered by `grid`, then `lod`.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug, Serialize, Deserialize)]
pub struct ChunkKey {
    /// The grid (body or construct) the chunk belongs to.
    pub grid: GridId,
    /// The chunk within that grid's octree.
    pub lod: LodKey,
}

impl ChunkKey {
    /// Key from grid and chunk.
    #[inline]
    #[must_use]
    pub const fn new(grid: GridId, lod: LodKey) -> Self {
        Self { grid, lod }
    }
}

/// One voxel (level-0 cell) of a specific grid (contract §2.2): the address of edits,
/// raycast hits and block entities.
///
/// Ordered by `grid`, then `cell.x`, `cell.y`, `cell.z`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Serialize, Deserialize)]
pub struct CellAddress {
    /// The grid the voxel belongs to.
    pub grid: GridId,
    /// Voxel coordinates in the grid; voxel `(x, y, z)` spans `[x, x+1)` m.
    pub cell: IVec3,
}

impl CellAddress {
    /// Address from grid and voxel.
    #[inline]
    #[must_use]
    pub const fn new(grid: GridId, cell: IVec3) -> Self {
        Self { grid, cell }
    }

    /// The level-0 chunk that contains the voxel and the local cell `[x, y, z]` inside it
    /// ([`voxel_to_chunk`] within the same grid).
    #[inline]
    #[must_use]
    pub const fn to_chunk(self) -> (ChunkKey, [u8; 3]) {
        let (chunk, local) = voxel_to_chunk(self.cell);
        (ChunkKey::new(self.grid, LodKey::lod0(chunk)), local)
    }
}

impl Ord for CellAddress {
    fn cmp(&self, other: &Self) -> Ordering {
        (self.grid, self.cell.x, self.cell.y, self.cell.z).cmp(&(
            other.grid,
            other.cell.x,
            other.cell.y,
            other.cell.z,
        ))
    }
}

impl PartialOrd for CellAddress {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Splits a voxel coordinate into its level-0 chunk position and the local cell `[x, y, z]`
/// inside that chunk: `chunk = voxel >> 5`, `local = voxel & 31`.
#[inline]
#[must_use]
pub const fn voxel_to_chunk(voxel: IVec3) -> (IVec3, [u8; 3]) {
    (
        IVec3::new(
            voxel.x >> CHUNK_BITS,
            voxel.y >> CHUNK_BITS,
            voxel.z >> CHUNK_BITS,
        ),
        [
            (voxel.x & CHUNK_MASK) as u8,
            (voxel.y & CHUNK_MASK) as u8,
            (voxel.z & CHUNK_MASK) as u8,
        ],
    )
}

/// Inverse of [`voxel_to_chunk`]: `chunk * 32 + local`.
#[inline]
#[must_use]
pub const fn chunk_to_voxel(chunk: IVec3, local: [u8; 3]) -> IVec3 {
    IVec3::new(
        chunk
            .x
            .wrapping_mul(CHUNK_SIZE)
            .wrapping_add(local[0] as i32),
        chunk
            .y
            .wrapping_mul(CHUNK_SIZE)
            .wrapping_add(local[1] as i32),
        chunk
            .z
            .wrapping_mul(CHUNK_SIZE)
            .wrapping_add(local[2] as i32),
    )
}

/// Canonical index of the cell `(x, y, z)` inside a chunk: `x + z * 32 + y * 1024`
/// (x fastest, y slowest). Each coordinate must be `< 32`.
#[inline]
#[must_use]
pub const fn cell_index(x: u8, y: u8, z: u8) -> usize {
    debug_assert!(
        x < 32 && y < 32 && z < 32,
        "cell coordinate outside the chunk"
    );
    (x as usize) | ((z as usize) << CHUNK_BITS) | ((y as usize) << (2 * CHUNK_BITS))
}

/// Inverse of [`cell_index`]: the cell `[x, y, z]` of a canonical index `< CHUNK_VOLUME`.
#[inline]
#[must_use]
pub const fn index_to_cell(index: usize) -> [u8; 3] {
    debug_assert!(index < CHUNK_VOLUME, "cell index outside the chunk");
    let mask = CHUNK_MASK as usize;
    [
        (index & mask) as u8,
        ((index >> (2 * CHUNK_BITS)) & mask) as u8,
        ((index >> CHUNK_BITS) & mask) as u8,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hash::Rng;
    use crate::ids::{BodyId, EntityId};
    use crate::math::{div_floor_ivec3, mod_floor_ivec3};

    #[test]
    fn constants_match_the_contract() {
        assert_eq!(CHUNK_BITS, 5);
        assert_eq!(CHUNK_SIZE, 32);
        assert_eq!(CHUNK_MASK, 31);
        assert_eq!(CHUNK_VOLUME, 32_768);
        assert_eq!(MAX_LOD_LEVEL, 20);
    }

    #[test]
    fn voxel_to_chunk_floors_negative_coordinates() {
        assert_eq!(
            voxel_to_chunk(IVec3::new(0, 31, 32)),
            (IVec3::new(0, 0, 1), [0, 31, 0])
        );
        assert_eq!(
            voxel_to_chunk(IVec3::new(-1, -32, -33)),
            (IVec3::new(-1, -1, -2), [31, 0, 31])
        );
        assert_eq!(
            voxel_to_chunk(IVec3::new(i32::MIN, i32::MAX, 0)),
            (IVec3::new(-(1 << 26), (1 << 26) - 1, 0), [0, 31, 0])
        );
        for x in -200..200 {
            let voxel = IVec3::new(x, -x, x * 7 - 3);
            let (chunk, local) = voxel_to_chunk(voxel);
            // Floor semantics of the contract, against the generic floor maths.
            assert_eq!(chunk, div_floor_ivec3(voxel, CHUNK_SIZE));
            let local_vec = IVec3::new(local[0].into(), local[1].into(), local[2].into());
            assert_eq!(local_vec, mod_floor_ivec3(voxel, CHUNK_SIZE));
            assert_eq!(chunk_to_voxel(chunk, local), voxel);
            assert_eq!(LodKey::containing(0, voxel), LodKey::lod0(chunk));
            assert!(LodKey::lod0(chunk).aabb_voxels().contains(voxel));
        }
    }

    #[test]
    fn cell_index_is_x_fastest_y_slowest() {
        assert_eq!(cell_index(0, 0, 0), 0);
        assert_eq!(cell_index(1, 0, 0), 1);
        assert_eq!(cell_index(0, 0, 1), 32);
        assert_eq!(cell_index(0, 1, 0), 1024);
        assert_eq!(cell_index(31, 31, 31), CHUNK_VOLUME - 1);
        let mut expected = 0;
        for y in 0..32_u8 {
            for z in 0..32_u8 {
                for x in 0..32_u8 {
                    let index = cell_index(x, y, z);
                    assert_eq!(index, expected);
                    assert_eq!(
                        index,
                        usize::from(x) + usize::from(z) * 32 + usize::from(y) * 1024
                    );
                    assert_eq!(index_to_cell(index), [x, y, z]);
                    expected += 1;
                }
            }
        }
        assert_eq!(expected, CHUNK_VOLUME);
    }

    #[test]
    fn lod_key_geometry() {
        let key = LodKey::lod0(IVec3::new(2, -1, 0));
        assert_eq!(key, LodKey::new(0, IVec3::new(2, -1, 0)));
        assert_eq!(key.cell_size(), 1);
        assert_eq!(key.chunk_extent(), 32);
        assert_eq!(key.origin_voxel(), IVec3::new(64, -32, 0));
        assert_eq!(
            key.aabb_voxels(),
            IAabb::new(IVec3::new(64, -32, 0), IVec3::new(96, 0, 32))
        );

        let key = LodKey::new(3, IVec3::new(-2, 1, 5));
        assert_eq!(key.cell_size(), 8);
        assert_eq!(key.chunk_extent(), 256);
        assert_eq!(key.origin_voxel(), IVec3::new(-512, 256, 1280));
        assert_eq!(key.aabb_voxels().size(), IVec3::splat(256));

        let coarsest = LodKey::new(MAX_LOD_LEVEL, IVec3::new(-32, 31, 0));
        assert_eq!(coarsest.cell_size(), 1 << 20);
        assert_eq!(coarsest.chunk_extent(), 1 << 25);
        assert_eq!(
            coarsest.origin_voxel(),
            IVec3::new(-VOXEL_LIMIT, VOXEL_LIMIT - (1 << 25), 0)
        );
        assert_eq!(
            coarsest.aabb_voxels().max,
            IVec3::new(-VOXEL_LIMIT + (1 << 25), VOXEL_LIMIT, 1 << 25)
        );
    }

    #[test]
    fn validity_limits() {
        assert!(LodKey::lod0(IVec3::ZERO).is_valid());
        let edge = VOXEL_LIMIT / CHUNK_SIZE;
        assert!(LodKey::lod0(IVec3::splat(edge - 1)).is_valid());
        assert!(LodKey::lod0(IVec3::splat(-edge)).is_valid());
        assert!(!LodKey::lod0(IVec3::new(edge, 0, 0)).is_valid());
        assert!(!LodKey::lod0(IVec3::new(0, -edge - 1, 0)).is_valid());
        assert!(LodKey::new(MAX_LOD_LEVEL, IVec3::new(-32, 31, 0)).is_valid());
        assert!(!LodKey::new(MAX_LOD_LEVEL, IVec3::new(0, 0, 32)).is_valid());
        assert!(!LodKey::new(MAX_LOD_LEVEL, IVec3::new(0, 0, -33)).is_valid());
        assert!(!LodKey::new(MAX_LOD_LEVEL + 1, IVec3::ZERO).is_valid());
        assert!(!LodKey::new(u8::MAX, IVec3::ZERO).is_valid());
        // Invalid keys never panic, whatever they return.
        let wild = LodKey::new(u8::MAX, IVec3::new(i32::MIN, i32::MAX, -1));
        let _ = (
            wild.cell_size(),
            wild.chunk_extent(),
            wild.origin_voxel(),
            wild.aabb_voxels(),
        );
        assert_eq!(wild.parent(), None);
        assert_eq!(wild.ancestor_at(3), None);
        // Every valid key's box stays inside the limits, on every level.
        for level in 0..=MAX_LOD_LEVEL {
            let limit = VOXEL_LIMIT >> (CHUNK_BITS + u32::from(level));
            for pos in [
                IVec3::splat(-limit),
                IVec3::splat(limit - 1),
                IVec3::new(-limit, 0, limit - 1),
            ] {
                let key = LodKey::new(level, pos);
                assert!(key.is_valid());
                let aabb = key.aabb_voxels();
                assert!(aabb.min.cmpge(IVec3::splat(-VOXEL_LIMIT)).all());
                assert!(aabb.max.cmple(IVec3::splat(VOXEL_LIMIT)).all());
                assert_eq!(aabb.size(), IVec3::splat(key.chunk_extent()));
            }
        }
    }

    #[test]
    fn parent_children_round_trip() {
        let mut rng = Rng::new(31, 3);
        let mut keys = vec![
            LodKey::new(1, IVec3::ZERO),
            LodKey::new(1, IVec3::NEG_ONE),
            LodKey::new(5, IVec3::new(-3, 2, -1)),
            LodKey::new(MAX_LOD_LEVEL, IVec3::new(-32, 31, -1)),
        ];
        for _ in 0..2000 {
            let level = rng.range_u32(1, u32::from(MAX_LOD_LEVEL) + 1) as u8;
            let limit = VOXEL_LIMIT >> (CHUNK_BITS + u32::from(level));
            let mut coordinate = || {
                // Mix positions near the origin, where signs change, with far ones.
                let span = if rng.range_u32(0, 2) == 0 {
                    limit.min(4)
                } else {
                    limit
                };
                rng.range_u32(0, 2 * span as u32) as i32 - span
            };
            keys.push(LodKey::new(
                level,
                IVec3::new(coordinate(), coordinate(), coordinate()),
            ));
        }
        for key in keys {
            assert!(key.is_valid());
            let children = key.children();
            let parent_box = key.aabb_voxels();
            let mut covered = 0_i128;
            for (octant, child) in children.iter().enumerate() {
                assert!(child.is_valid());
                assert_eq!(child.level, key.level - 1);
                assert_eq!(child.parent(), Some(key));
                assert_eq!(child.octant(), octant);
                assert_eq!(child.pos, key.pos * 2 + LodKey::octant_offset(octant));
                assert_eq!(child.ancestor_at(key.level), Some(key));
                assert!(key.contains(child) && !child.contains(&key));
                // The child's box is the matching octant of the parent's box.
                let child_box = child.aabb_voxels();
                let expected_min =
                    parent_box.min + LodKey::octant_offset(octant) * child.chunk_extent();
                assert_eq!(child_box.min, expected_min);
                assert_eq!(parent_box.intersection(&child_box), Some(child_box));
                let size = child_box.size();
                covered += i128::from(size.x) * i128::from(size.y) * i128::from(size.z);
                for other in &children[..octant] {
                    assert!(!other.aabb_voxels().intersects(&child_box));
                }
            }
            // Eight disjoint boxes inside the parent with the parent's volume tile it exactly.
            assert_eq!(covered, i128::from(key.chunk_extent()).pow(3));
            // The octant order is x fastest, y slowest.
            assert_eq!(children[1].pos - children[0].pos, IVec3::X);
            assert_eq!(children[2].pos - children[0].pos, IVec3::Z);
            assert_eq!(children[4].pos - children[0].pos, IVec3::Y);
        }
    }

    #[test]
    fn ancestors_and_containment() {
        let key = LodKey::lod0(IVec3::new(-1, 37, -70));
        assert_eq!(key.ancestor_at(0), Some(key));
        assert_eq!(key.parent(), Some(LodKey::new(1, IVec3::new(-1, 18, -35))));
        assert_eq!(
            key.ancestor_at(3),
            Some(LodKey::new(3, IVec3::new(-1, 4, -9)))
        );
        assert_eq!(
            key.ancestor_at(MAX_LOD_LEVEL),
            Some(LodKey::new(MAX_LOD_LEVEL, IVec3::NEG_ONE.with_y(0)))
        );
        assert_eq!(key.ancestor_at(MAX_LOD_LEVEL + 1), None);
        assert_eq!(LodKey::new(4, IVec3::ZERO).ancestor_at(3), None);
        assert_eq!(LodKey::new(MAX_LOD_LEVEL, IVec3::ZERO).parent(), None);
        assert!(key.contains(&key));

        let mut rng = Rng::new(32, 3);
        for _ in 0..3000 {
            let level = rng.range_u32(0, u32::from(MAX_LOD_LEVEL)) as u8;
            let limit = VOXEL_LIMIT >> (CHUNK_BITS + u32::from(level));
            let span = limit.min(1000);
            let mut coordinate = || rng.range_u32(0, 2 * span as u32) as i32 - span;
            let key = LodKey::new(level, IVec3::new(coordinate(), coordinate(), coordinate()));
            let mut ancestor = key;
            // Walking up with parent() agrees with ancestor_at() and with the voxel boxes.
            while let Some(parent) = ancestor.parent() {
                assert!(parent.is_valid());
                assert_eq!(key.ancestor_at(parent.level), Some(parent));
                assert!(parent.contains(&key));
                assert!(!key.contains(&parent));
                let outer = parent.aabb_voxels();
                assert_eq!(
                    outer.intersection(&key.aabb_voxels()),
                    Some(key.aabb_voxels())
                );
                assert_eq!(LodKey::containing(parent.level, key.origin_voxel()), parent);
                ancestor = parent;
            }
            assert_eq!(ancestor.level, MAX_LOD_LEVEL);
            // A neighbour on the same level is never contained.
            let neighbour = LodKey::new(level, key.pos + IVec3::X);
            assert!(!key.contains(&neighbour) && !neighbour.contains(&key));
        }
    }

    #[test]
    fn ordering_is_level_then_x_y_z() {
        let mut keys = vec![
            LodKey::new(1, IVec3::new(0, 0, 0)),
            LodKey::new(0, IVec3::new(1, 0, 0)),
            LodKey::new(0, IVec3::new(0, 1, 0)),
            LodKey::new(0, IVec3::new(0, 0, 1)),
            LodKey::new(0, IVec3::new(-1, 5, 5)),
            LodKey::new(0, IVec3::new(0, 0, 0)),
        ];
        keys.sort();
        assert_eq!(
            keys,
            [
                LodKey::new(0, IVec3::new(-1, 5, 5)),
                LodKey::new(0, IVec3::new(0, 0, 0)),
                LodKey::new(0, IVec3::new(0, 0, 1)),
                LodKey::new(0, IVec3::new(0, 1, 0)),
                LodKey::new(0, IVec3::new(1, 0, 0)),
                LodKey::new(1, IVec3::new(0, 0, 0)),
            ]
        );
        let body = |id| GridId::Body(BodyId(id));
        let a = ChunkKey::new(body(1), LodKey::new(9, IVec3::ONE));
        let b = ChunkKey::new(body(2), LodKey::lod0(IVec3::ZERO));
        assert!(a < b);
        assert!(ChunkKey::new(body(2), LodKey::lod0(IVec3::NEG_ONE)) < b);
        // Body grids come before construct grids.
        let construct = GridId::Construct(EntityId::new(0, 0));
        assert!(b < ChunkKey::new(construct, LodKey::lod0(IVec3::ZERO)));
        assert_eq!(b.grid, body(2));
        assert_eq!(b.lod, LodKey::lod0(IVec3::ZERO));

        // Cell addresses: grid, then x, y, z.
        let mut cells = vec![
            CellAddress::new(construct, IVec3::ZERO),
            CellAddress::new(body(1), IVec3::new(0, 0, 1)),
            CellAddress::new(body(1), IVec3::new(0, 1, 0)),
            CellAddress::new(body(1), IVec3::new(-1, 9, 9)),
            CellAddress::new(body(0), IVec3::splat(100)),
        ];
        cells.sort();
        assert_eq!(
            cells,
            [
                CellAddress::new(body(0), IVec3::splat(100)),
                CellAddress::new(body(1), IVec3::new(-1, 9, 9)),
                CellAddress::new(body(1), IVec3::new(0, 0, 1)),
                CellAddress::new(body(1), IVec3::new(0, 1, 0)),
                CellAddress::new(construct, IVec3::ZERO),
            ]
        );
    }

    #[test]
    fn cell_addresses_split_like_voxels() {
        let grid = GridId::Construct(EntityId::new(3, 1));
        for voxel in [
            IVec3::new(-1, 40, 5),
            IVec3::ZERO,
            IVec3::new(31, 32, -33),
            IVec3::new(i32::MIN, i32::MAX, -VOXEL_LIMIT),
        ] {
            let (chunk, local) = CellAddress::new(grid, voxel).to_chunk();
            let (expected_chunk, expected_local) = voxel_to_chunk(voxel);
            assert_eq!(chunk, ChunkKey::new(grid, LodKey::lod0(expected_chunk)));
            assert_eq!(local, expected_local);
            assert_eq!(chunk_to_voxel(chunk.lod.pos, local), voxel);
        }
    }
}
