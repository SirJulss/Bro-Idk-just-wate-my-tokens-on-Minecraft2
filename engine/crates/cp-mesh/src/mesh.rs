//! The mesh of one chunk: packed quads in layer and face ranges (contract §5.2).

use core::ops::Range;

use cp_core::block::RenderClass;
use cp_core::face::Face;
use cp_core::quad::PackedQuad;

/// Highest LOD level that has a cutout layer. On coarser levels cutout states are meshed into
/// the opaque layer (contract §5.2).
pub const CUTOUT_MAX_LEVEL: u8 = 1;

/// The faces in the storage order of a layer's ranges: `+X, +Y, +Z, -X, -Y, -Z`.
///
/// This is not the [`Face::index`] order: positive faces come first.
pub const FACE_STORAGE_ORDER: [Face; 6] = [
    Face::PosX,
    Face::PosY,
    Face::PosZ,
    Face::NegX,
    Face::NegY,
    Face::NegZ,
];

/// Ranges per layer: the six regular face ranges, then the six wall ranges, each group in
/// [`FACE_STORAGE_ORDER`].
pub const RANGES_PER_LAYER: usize = 12;

/// Range slot of the first wall range; wall slot = `WALL_RANGE_OFFSET + storage_slot(face)`.
pub const WALL_RANGE_OFFSET: usize = 6;

/// Position of `face` in [`FACE_STORAGE_ORDER`]: `axis + 3 · negative`, in `0..6`.
#[inline]
#[must_use]
pub const fn storage_slot(face: Face) -> usize {
    face.axis() + 3 * face.is_negative() as usize
}

/// A mesh layer: the pass that draws a quad. Discriminants are the storage order of the layers
/// in [`ChunkMesh::quads`] and the indices of [`ChunkMesh::layers`].
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
#[repr(u8)]
pub enum Layer {
    /// Opaque cube faces; from level 2 on also the faces of cutout states.
    Opaque = 0,
    /// Alpha-tested faces of cutout states, on levels 0 and 1 only.
    Cutout = 1,
    /// Fluid surfaces.
    Fluid = 2,
}

impl Layer {
    /// Number of layers.
    pub const COUNT: usize = 3;

    /// All layers in storage order.
    pub const ALL: [Self; Self::COUNT] = [Self::Opaque, Self::Cutout, Self::Fluid];

    /// Storage index `0..3`.
    #[inline]
    #[must_use]
    pub const fn index(self) -> usize {
        self as usize
    }

    /// The layer that the faces of a state of render class `class` go to in a chunk of LOD
    /// `level`; `None` for the classes that emit no faces (`None`, `Model`).
    #[inline]
    #[must_use]
    pub const fn of_class(class: RenderClass, level: u8) -> Option<Self> {
        match class {
            RenderClass::Cube => Some(Self::Opaque),
            RenderClass::Cutout if level <= CUTOUT_MAX_LEVEL => Some(Self::Cutout),
            RenderClass::Cutout => Some(Self::Opaque),
            RenderClass::Fluid => Some(Self::Fluid),
            RenderClass::None | RenderClass::Model => None,
        }
    }
}

/// An axis-aligned box of chunk cells `[x, y, z]`, both corners inclusive, every coordinate in
/// `0..32`, `min <= max` on every axis.
///
/// The faces of the cells in the box lie inside the lattice box `[min, max + 1]` (in cells of
/// the chunk; multiply by the cell size for metres).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct CellBounds {
    /// Smallest cell coordinate on each axis.
    pub min: [u8; 3],
    /// Largest cell coordinate on each axis.
    pub max: [u8; 3],
}

impl CellBounds {
    /// The box of the single cell `cell`.
    #[inline]
    #[must_use]
    pub const fn cell(cell: [u8; 3]) -> Self {
        Self {
            min: cell,
            max: cell,
        }
    }

    /// The smallest box containing `self` and `other`.
    #[inline]
    #[must_use]
    pub fn union(self, other: Self) -> Self {
        Self {
            min: [0, 1, 2].map(|axis| self.min[axis].min(other.min[axis])),
            max: [0, 1, 2].map(|axis| self.max[axis].max(other.max[axis])),
        }
    }

    /// The smallest box containing both boxes; `None` stands for "no cells".
    #[inline]
    #[must_use]
    pub(crate) fn union_of(a: Option<Self>, b: Option<Self>) -> Option<Self> {
        match (a, b) {
            (Some(a), Some(b)) => Some(a.union(b)),
            (a, b) => a.or(b),
        }
    }

    /// True if `cell` lies inside the box.
    #[inline]
    #[must_use]
    pub fn contains(self, cell: [u8; 3]) -> bool {
        (0..3).all(|axis| self.min[axis] <= cell[axis] && cell[axis] <= self.max[axis])
    }
}

/// Where one layer's quads are in [`ChunkMesh::quads`], and their bounds.
///
/// The layer occupies `first .. first + len()`; inside it the twelve ranges follow each other
/// without gaps in slot order: slot `storage_slot(face)` holds the regular quads of `face`,
/// slot `WALL_RANGE_OFFSET + storage_slot(face)` its wall quads.
///
/// The bounds are tight boxes of the cells whose faces the quads cover. [`bounds`] is the
/// layer's box of contract §5.2 and covers everything a draw of the layer can show, walls
/// included; a draw whose wall mask is empty (contract §5.3) may cull with the smaller
/// [`regular_bounds`] instead. Invariant: `bounds` is the union of the other two.
///
/// [`bounds`]: LayerRanges::bounds
/// [`regular_bounds`]: LayerRanges::regular_bounds
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub struct LayerRanges {
    /// Index of the layer's first quad in [`ChunkMesh::quads`].
    pub first: u32,
    /// Number of quads in each of the twelve ranges, by slot.
    pub counts: [u32; RANGES_PER_LAYER],
    /// Tight bounds of all quads of the layer, regular and wall; `None` if the layer is
    /// empty. A chunk whose faces are all culled by its halo has only walls, and this box
    /// covers them.
    pub bounds: Option<CellBounds>,
    /// Tight bounds of the six regular ranges; `None` if they are empty.
    pub regular_bounds: Option<CellBounds>,
    /// Tight bounds of the six wall ranges; `None` if they are empty.
    pub wall_bounds: Option<CellBounds>,
}

impl LayerRanges {
    /// Number of quads in the layer, regular and wall.
    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        self.counts.iter().map(|&count| count as usize).sum()
    }

    /// True if the layer has no quads.
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.counts.iter().all(|&count| count == 0)
    }

    /// Number of regular quads (slots `0..6`).
    #[inline]
    #[must_use]
    pub fn regular_len(&self) -> usize {
        self.counts[..WALL_RANGE_OFFSET]
            .iter()
            .map(|&count| count as usize)
            .sum()
    }

    /// Number of wall quads (slots `6..12`).
    #[inline]
    #[must_use]
    pub fn wall_len(&self) -> usize {
        self.counts[WALL_RANGE_OFFSET..]
            .iter()
            .map(|&count| count as usize)
            .sum()
    }

    /// The whole layer as indices into [`ChunkMesh::quads`].
    #[inline]
    #[must_use]
    pub fn range(&self) -> Range<usize> {
        let first = self.first as usize;
        first..first + self.len()
    }

    /// Range slot `slot` (`0..12`) as indices into [`ChunkMesh::quads`].
    ///
    /// # Panics
    /// If `slot >= RANGES_PER_LAYER`.
    #[inline]
    #[must_use]
    pub fn slot(&self, slot: usize) -> Range<usize> {
        let start = self.first as usize
            + self.counts[..slot]
                .iter()
                .map(|&count| count as usize)
                .sum::<usize>();
        start..start + self.counts[slot] as usize
    }

    /// The regular quads of `face` as indices into [`ChunkMesh::quads`].
    #[inline]
    #[must_use]
    pub fn regular(&self, face: Face) -> Range<usize> {
        self.slot(storage_slot(face))
    }

    /// The wall quads of `face` as indices into [`ChunkMesh::quads`].
    #[inline]
    #[must_use]
    pub fn walls(&self, face: Face) -> Range<usize> {
        self.slot(WALL_RANGE_OFFSET + storage_slot(face))
    }
}

/// The mesh of one chunk (contract §5.2), as made by [`Mesher`](crate::Mesher).
///
/// Invariants of a mesh made by the mesher: the three layers lie in [`Layer`] order without
/// gaps and together cover `quads` exactly; every quad is valid with payload 0 and lies on the
/// face of its range; wall quads lie on the chunk boundary their face points to; inside a range
/// the quads are sorted by `(plane on the normal axis, v, u)` of their min cell, ascending; no
/// two quads cover the same face of the same cell.
#[derive(Clone, PartialEq, Eq, Hash, Debug, Default)]
pub struct ChunkMesh {
    /// All quads, layer after layer, range after range.
    pub quads: Vec<PackedQuad>,
    /// Ranges and bounds per layer, indexed by [`Layer::index`].
    pub layers: [LayerRanges; Layer::COUNT],
}

impl ChunkMesh {
    /// An empty mesh.
    #[inline]
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Removes every quad and range, keeping the allocation.
    pub fn clear(&mut self) {
        self.quads.clear();
        self.layers = [LayerRanges::default(); Layer::COUNT];
    }

    /// True if the mesh has no quads.
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.quads.is_empty()
    }

    /// Ranges and bounds of `layer`.
    #[inline]
    #[must_use]
    pub fn layer(&self, layer: Layer) -> &LayerRanges {
        &self.layers[layer.index()]
    }

    /// All quads of `layer`, regular and wall.
    #[inline]
    #[must_use]
    pub fn layer_quads(&self, layer: Layer) -> &[PackedQuad] {
        &self.quads[self.layer(layer).range()]
    }

    /// The regular quads of `layer` on `face`.
    #[inline]
    #[must_use]
    pub fn regular_quads(&self, layer: Layer, face: Face) -> &[PackedQuad] {
        &self.quads[self.layer(layer).regular(face)]
    }

    /// The wall quads of `layer` on `face`.
    #[inline]
    #[must_use]
    pub fn wall_quads(&self, layer: Layer, face: Face) -> &[PackedQuad] {
        &self.quads[self.layer(layer).walls(face)]
    }

    /// Number of regular quads over all layers.
    #[inline]
    #[must_use]
    pub fn regular_len(&self) -> usize {
        self.layers.iter().map(LayerRanges::regular_len).sum()
    }

    /// Number of wall quads over all layers.
    #[inline]
    #[must_use]
    pub fn wall_len(&self) -> usize {
        self.layers.iter().map(LayerRanges::wall_len).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn storage_order_puts_positive_faces_first() {
        for (slot, face) in FACE_STORAGE_ORDER.into_iter().enumerate() {
            assert_eq!(storage_slot(face), slot);
        }
        assert_eq!(storage_slot(Face::PosX), 0);
        assert_eq!(storage_slot(Face::NegX), 3);
        assert_eq!(storage_slot(Face::NegZ), 5);
        assert_eq!(WALL_RANGE_OFFSET + 6, RANGES_PER_LAYER);
    }

    #[test]
    fn layers_follow_class_and_level() {
        for level in 0..=20 {
            assert_eq!(
                Layer::of_class(RenderClass::Cube, level),
                Some(Layer::Opaque)
            );
            assert_eq!(
                Layer::of_class(RenderClass::Fluid, level),
                Some(Layer::Fluid)
            );
            assert_eq!(Layer::of_class(RenderClass::None, level), None);
            assert_eq!(Layer::of_class(RenderClass::Model, level), None);
            let cutout = if level <= 1 {
                Layer::Cutout
            } else {
                Layer::Opaque
            };
            assert_eq!(Layer::of_class(RenderClass::Cutout, level), Some(cutout));
        }
        assert_eq!(Layer::ALL.map(Layer::index), [0, 1, 2]);
    }

    #[test]
    fn ranges_are_contiguous_slots() {
        let mut counts = [0; RANGES_PER_LAYER];
        for (slot, count) in counts.iter_mut().enumerate() {
            *count = slot as u32 + 1;
        }
        let ranges = LayerRanges {
            first: 100,
            counts,
            bounds: None,
            regular_bounds: None,
            wall_bounds: None,
        };
        assert_eq!(ranges.len(), 78);
        assert_eq!(ranges.regular_len(), 21);
        assert_eq!(ranges.wall_len(), 57);
        assert_eq!(ranges.range(), 100..178);
        let mut next = 100;
        for slot in 0..RANGES_PER_LAYER {
            let range = ranges.slot(slot);
            assert_eq!(range.start, next);
            assert_eq!(range.len(), slot + 1);
            next = range.end;
        }
        assert_eq!(next, 178);
        assert_eq!(ranges.regular(Face::PosY), ranges.slot(1));
        assert_eq!(ranges.walls(Face::NegX), ranges.slot(9));
        assert!(LayerRanges::default().is_empty());
        assert!(!ranges.is_empty());
    }

    #[test]
    fn bounds_union_and_containment() {
        let a = CellBounds::cell([1, 2, 3]);
        let b = CellBounds {
            min: [0, 5, 3],
            max: [4, 6, 31],
        };
        let both = a.union(b);
        assert_eq!(
            both,
            CellBounds {
                min: [0, 2, 3],
                max: [4, 6, 31]
            }
        );
        assert!(both.contains([1, 2, 3]) && both.contains([4, 6, 31]));
        assert!(!both.contains([5, 2, 3]) && !a.contains([1, 2, 4]));
        assert_eq!(CellBounds::union_of(Some(a), Some(b)), Some(both));
        assert_eq!(CellBounds::union_of(Some(a), None), Some(a));
        assert_eq!(CellBounds::union_of(None, Some(b)), Some(b));
        assert_eq!(CellBounds::union_of(None, None), None);
    }
}
