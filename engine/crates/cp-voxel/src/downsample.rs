//! The normative downsample rule (contract §5.1): one coarse cell from its eight children.
//!
//! Input: the eight child cells of a parent cell, child index `dx + 2*dz + 4*dy` (the order of
//! [`cp_core::coords::LodKey::children`]), the children's authored bits and the per-state
//! tables. Per parent cell the classes of the children are counted (`lod[state].class`);
//! `structural` is the number of Solid children that are `STRUCTURAL` or authored. The parent
//! is **Solid** if `solid >= 4`, or `structural >= 1` when the parent level is at most 2, or
//! `structural >= 2` above; else **Fluid** if `solid + fluid >= 4`; else **Empty** (air).
//! Material: the most frequent `proxy` among the Solid children that share a face with a
//! non-Solid child inside the 2×2×2 group (all Solid children if none does); a Fluid parent
//! takes the most frequent fluid state; ties go to the smallest id. A parent cell is authored
//! iff any child is.
//!
//! "Touching" is read as sharing a face: inside a 2×2×2 group every two cells share at least a
//! corner, so a corner-based reading would make the "if none touches" case unreachable.

use cp_core::block::{BlockFlags, LodClass, LodInfo};
use cp_core::coords::{CHUNK_VOLUME, cell_index};
use cp_core::ids::BlockStateId;

use crate::chunk::VoxelChunk;
use crate::chunk_data::ChunkData;
use crate::mask::CellMask;

/// A downsampled parent cell.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Downsampled {
    /// The parent's state.
    pub state: BlockStateId,
    /// True when at least one child is authored.
    pub authored: bool,
}

/// Child `i` of a 2×2×2 group sits at `(i & 1, (i >> 2) & 1, (i >> 1) & 1)` in `(x, y, z)`.
#[inline]
pub(crate) const fn child_offset(child: usize) -> [u8; 3] {
    [
        (child & 1) as u8,
        ((child >> 2) & 1) as u8,
        ((child >> 1) & 1) as u8,
    ]
}

/// Bit `i` of the result is bit `i ^ axis_bit` of `mask`: the face neighbour of every child
/// along the axis with bit `axis_bit` (1 = x, 2 = z, 4 = y).
#[inline]
const fn neighbours(mask: u8, axis_bit: u8) -> u8 {
    match axis_bit {
        1 => ((mask & 0x55) << 1) | ((mask & 0xAA) >> 1),
        2 => ((mask & 0x33) << 2) | ((mask & 0xCC) >> 2),
        _ => ((mask & 0x0F) << 4) | ((mask & 0xF0) >> 4),
    }
}

/// The most frequent value among the children in `mask`; ties go to the smallest value.
#[inline]
fn most_frequent(mask: u8, value: impl Fn(usize) -> BlockStateId) -> BlockStateId {
    let mut best = BlockStateId::AIR;
    let mut best_count = 0;
    for child in (0..8).filter(|child| mask & (1 << child) != 0) {
        let candidate = value(child);
        let count = (0..8)
            .filter(|&other| mask & (1 << other) != 0 && value(other) == candidate)
            .count();
        if count > best_count || (count == best_count && candidate < best) {
            best = candidate;
            best_count = count;
        }
    }
    best
}

/// Downsamples one parent cell of level `parent_level` (`>= 1`) from its eight children
/// (contract §5.1, rule in the [module documentation](self)).
///
/// `children` are composite states (never [`BlockStateId::UNCHANGED`]) in child order
/// `dx + 2*dz + 4*dy`; bit `i` of `authored` is set when child `i` is authored. `flags` and
/// `lod` are indexed by state.
///
/// # Panics
/// If a child state has no entry in `flags` or `lod`.
#[must_use]
pub fn downsample_cell(
    children: &[BlockStateId; 8],
    authored: u8,
    parent_level: u8,
    flags: &[BlockFlags],
    lod: &[LodInfo],
) -> Downsampled {
    let mut solid_mask = 0_u8;
    let mut fluid_mask = 0_u8;
    let mut structural_mask = 0_u8;
    for (child, &state) in children.iter().enumerate() {
        match lod[state.index()].class {
            LodClass::Solid => {
                solid_mask |= 1 << child;
                if flags[state.index()].contains(BlockFlags::STRUCTURAL) {
                    structural_mask |= 1 << child;
                }
            }
            LodClass::Fluid => fluid_mask |= 1 << child,
            LodClass::Empty => {}
        }
    }
    let solid = solid_mask.count_ones();
    let fluid = fluid_mask.count_ones();
    let structural = (solid_mask & (structural_mask | authored)).count_ones();
    let structural_needed = if parent_level <= 2 { 1 } else { 2 };
    let state = if solid >= 4 || structural >= structural_needed {
        let open = !solid_mask;
        let touching =
            solid_mask & (neighbours(open, 1) | neighbours(open, 2) | neighbours(open, 4));
        let candidates = if touching != 0 { touching } else { solid_mask };
        most_frequent(candidates, |child| lod[children[child].index()].proxy)
    } else if solid + fluid >= 4 {
        most_frequent(fluid_mask, |child| children[child])
    } else {
        BlockStateId::AIR
    };
    Downsampled {
        state,
        authored: authored != 0,
    }
}

/// Downsamples a whole chunk: the parent chunk of level `parent_level` (`>= 1`) from its
/// eight child chunks, given as composites (no [`BlockStateId::UNCHANGED`]) in the order of
/// [`cp_core::coords::LodKey::children`]. Applies [`downsample_cell`] to every parent cell;
/// the result's authored mask is the union rule of contract §5.1.
///
/// This is how a coarse level of player edits is derived; `cp-worldgen` uses it to measure
/// how well its own coarse levels agree with downsampled fine ones.
///
/// # Panics
/// If a child state has no entry in `flags` or `lod`.
#[must_use]
pub fn downsample_chunk(
    children: [&VoxelChunk; 8],
    parent_level: u8,
    flags: &[BlockFlags],
    lod: &[LodInfo],
) -> VoxelChunk {
    // One octant of the parent (16³ cells) comes from one child chunk.
    let mut cells = vec![BlockStateId::AIR; CHUNK_VOLUME];
    let mut authored_out = CellMask::new();
    for (octant, child) in children.iter().enumerate() {
        let [ox, oy, oz] = child_offset(octant).map(|offset| offset * 16);
        if let (Some(state), false) = (child.blocks.uniform_state(), child.has_authored()) {
            // A uniform child gives a uniform octant.
            let value = downsample_cell(&[state; 8], 0, parent_level, flags, lod).state;
            for y in oy..oy + 16 {
                for z in oz..oz + 16 {
                    for x in ox..ox + 16 {
                        cells[cell_index(x, y, z)] = value;
                    }
                }
            }
            continue;
        }
        let decoded = child.blocks.to_cells();
        for y in 0..16_u8 {
            for z in 0..16_u8 {
                for x in 0..16_u8 {
                    let mut group = [BlockStateId::AIR; 8];
                    let mut authored = 0_u8;
                    for (i, slot) in group.iter_mut().enumerate() {
                        let [dx, dy, dz] = child_offset(i);
                        let index = cell_index(2 * x + dx, 2 * y + dy, 2 * z + dz);
                        *slot = decoded[index];
                        if child.is_authored(index) {
                            authored |= 1 << i;
                        }
                    }
                    let result = downsample_cell(&group, authored, parent_level, flags, lod);
                    let parent = cell_index(ox + x, oy + y, oz + z);
                    cells[parent] = result.state;
                    if result.authored {
                        authored_out.set(parent, true);
                    }
                }
            }
        }
    }
    VoxelChunk {
        blocks: ChunkData::from_cells(&cells),
        authored: (!authored_out.is_empty()).then(|| Box::new(authored_out)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const AIR: BlockStateId = BlockStateId(0);
    const STONE: BlockStateId = BlockStateId(1);
    const DIRT: BlockStateId = BlockStateId(2);
    const WATER: BlockStateId = BlockStateId(3);
    const LOG: BlockStateId = BlockStateId(4);
    const LAVA: BlockStateId = BlockStateId(5);
    const SLAB: BlockStateId = BlockStateId(6);

    fn tables() -> (Vec<BlockFlags>, Vec<LodInfo>) {
        let solid = BlockFlags::SOLID | BlockFlags::OPAQUE;
        let flags = vec![
            BlockFlags::empty(),
            solid,
            solid,
            BlockFlags::FLUID,
            solid | BlockFlags::STRUCTURAL,
            BlockFlags::FLUID,
            BlockFlags::SOLID,
        ];
        let lod = |class, proxy| LodInfo { class, proxy };
        let lod = vec![
            LodInfo::EMPTY,
            lod(LodClass::Solid, STONE),
            lod(LodClass::Solid, DIRT),
            lod(LodClass::Fluid, WATER),
            lod(LodClass::Solid, LOG),
            lod(LodClass::Fluid, LAVA),
            // A non-cube state whose stand-in is stone.
            lod(LodClass::Solid, STONE),
        ];
        (flags, lod)
    }

    fn run(children: [BlockStateId; 8], authored: u8, level: u8) -> Downsampled {
        let (flags, lod) = tables();
        downsample_cell(&children, authored, level, &flags, &lod)
    }

    #[test]
    fn neighbour_masks_flip_one_axis() {
        for axis_bit in [1_u8, 2, 4] {
            for child in 0..8 {
                assert_eq!(
                    neighbours(1 << child, axis_bit),
                    1 << (child ^ usize::from(axis_bit))
                );
            }
        }
        assert_eq!(child_offset(1), [1, 0, 0]);
        assert_eq!(child_offset(2), [0, 0, 1]);
        assert_eq!(child_offset(4), [0, 1, 0]);
    }

    #[test]
    fn counting_rules() {
        // Natural terrain: four solids make a solid parent, three do not.
        let four = [STONE, STONE, STONE, STONE, AIR, AIR, AIR, AIR];
        assert_eq!(run(four, 0, 5).state, STONE);
        let three = [STONE, STONE, STONE, AIR, AIR, AIR, AIR, AIR];
        assert_eq!(run(three, 0, 1).state, AIR);
        // Structural: one log keeps the parent solid up to level 2, two are needed above.
        let one_log = [LOG, AIR, AIR, AIR, AIR, AIR, AIR, AIR];
        assert_eq!(run(one_log, 0, 1).state, LOG);
        assert_eq!(run(one_log, 0, 2).state, LOG);
        assert_eq!(run(one_log, 0, 3).state, AIR);
        let two_logs = [LOG, AIR, AIR, AIR, LOG, AIR, AIR, AIR];
        assert_eq!(run(two_logs, 0, 3).state, LOG);
        // Authored solids count as structural; authored air does not.
        let one_stone = [AIR, STONE, AIR, AIR, AIR, AIR, AIR, AIR];
        assert_eq!(run(one_stone, 0b10, 2).state, STONE);
        assert_eq!(run(one_stone, 0b01, 2).state, AIR);
        assert!(run(one_stone, 0b01, 2).authored);
        assert!(!run(one_stone, 0, 2).authored);
        // Fluids: solid + fluid >= 4 makes a fluid parent with the most frequent fluid.
        let wet = [STONE, WATER, WATER, LAVA, AIR, AIR, AIR, AIR];
        assert_eq!(run(wet, 0, 4).state, WATER);
        let tie = [STONE, STONE, WATER, LAVA, AIR, AIR, AIR, AIR];
        assert_eq!(run(tie, 0, 4).state, WATER, "tie goes to the smaller id");
        let dry = [STONE, WATER, AIR, AIR, AIR, AIR, AIR, AIR];
        assert_eq!(run(dry, 0, 4).state, AIR);
    }

    #[test]
    fn material_prefers_the_exposed_surface() {
        // Bottom layer (dy = 0) dirt under stone? Top layer air; the bottom children touch
        // air above them, so all four are candidates: three dirt beat one stone.
        let surface = [DIRT, DIRT, DIRT, STONE, AIR, AIR, AIR, AIR];
        assert_eq!(run(surface, 0, 3).state, DIRT);
        // Seven stone and one dirt: only the three stones next to the air child are exposed
        // and they win; a buried majority does not matter.
        let buried = [AIR, STONE, STONE, DIRT, STONE, DIRT, DIRT, DIRT];
        // Children touching child 0 (air) are 1, 2 and 4: all stone.
        assert_eq!(run(buried, 0, 3).state, STONE);
        // Fully solid group: every child is a candidate, majority wins, ties smallest id.
        let full = [DIRT, DIRT, DIRT, DIRT, STONE, STONE, STONE, STONE];
        assert_eq!(run(full, 0, 3).state, STONE);
        // Proxies are counted, not states: two slabs (proxy stone) beat one dirt.
        let proxies = [SLAB, SLAB, DIRT, AIR, AIR, AIR, AIR, STONE];
        assert_eq!(run(proxies, 0, 1).state, STONE);
    }

    #[test]
    fn uniform_children_give_uniform_octants() {
        let (flags, lod) = tables();
        let stone = VoxelChunk::uniform(STONE);
        let air = VoxelChunk::uniform(AIR);
        let mut mixed =
            VoxelChunk::new(ChunkData::from_fn(
                |index| {
                    if index % 3 == 0 { STONE } else { AIR }
                },
            ));
        mixed.set_authored(0, true);
        let children = [&stone, &air, &stone, &air, &mixed, &air, &air, &stone];
        let parent = downsample_chunk(children, 1, &flags, &lod);
        for (octant, child) in children.iter().enumerate() {
            let [ox, oy, oz] = child_offset(octant).map(|offset| offset * 16);
            for (x, y, z) in [(0, 0, 0), (15, 15, 15), (3, 9, 12)] {
                let mut group = [AIR; 8];
                let mut authored = 0;
                for (i, slot) in group.iter_mut().enumerate() {
                    let [dx, dy, dz] = child_offset(i);
                    let index = cell_index(2 * x + dx, 2 * y + dy, 2 * z + dz);
                    *slot = child.get(index);
                    if child.is_authored(index) {
                        authored |= 1 << i;
                    }
                }
                let expected = downsample_cell(&group, authored, 1, &flags, &lod);
                let index = cell_index(ox + x, oy + y, oz + z);
                assert_eq!(parent.get(index), expected.state);
                assert_eq!(parent.is_authored(index), expected.authored);
            }
        }
        assert!(parent.is_authored(cell_index(0, 16, 0)));
        assert_eq!(
            parent.authored.as_ref().map(|mask| mask.count_ones()),
            Some(1)
        );
    }
}
