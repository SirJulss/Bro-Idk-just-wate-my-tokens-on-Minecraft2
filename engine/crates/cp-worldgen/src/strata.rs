//! Layered ground: which state each cell of a column gets (contract §5.1).
//!
//! A cell of level `L` is ground iff its centre lies below the column's surface height `h`, the
//! same point rule on every level. Its state follows the layers from the top: a layer
//! `(state, depth)` holds the cells whose centre lies less than `depth` metres below the
//! surface (depths cumulative and non-decreasing), the base state everything below the last
//! layer. One exception keeps the surface material visible at every level: the **top ground
//! cell** of a column always gets the first layer's state, even when that layer is thinner than
//! the cell. A 1 m grass layer therefore still colours the top 8 m cell of level 3, which is also
//! what the downsample rule produces from finer levels (the surface children touch air).

use cp_core::ids::BlockStateId;

use crate::region::{centre, first_centre_above, last_centre_below};

/// The state of cell `cy` (level `level`) of a column whose ground reaches up to `surface` m,
/// or `None` when the cell is not ground. This is the rule in its plain form; [`Column`] and the
/// flat ground evaluate it faster and are checked against it.
pub(crate) fn layered_state(
    level: u8,
    cy: i64,
    surface: f64,
    layers: &[(BlockStateId, f64)],
    base: BlockStateId,
) -> Option<BlockStateId> {
    let c = centre(level, cy);
    if c >= surface {
        return None;
    }
    if centre(level, cy + 1) >= surface {
        return Some(layers.first().map_or(base, |layer| layer.0));
    }
    for &(state, depth) in layers {
        if c > surface - depth {
            return Some(state);
        }
    }
    Some(base)
}

/// One column of cells of one level, resolved into index thresholds so that filling it costs
/// a few integer comparisons per cell. Cell for cell it equals [`layered_state`] with two
/// layers, plus water in the open cells whose centre lies below the water level.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Column {
    /// Highest ground cell.
    ground_top: i64,
    /// Highest cell whose centre lies below the water level.
    water_top: i64,
    /// Lowest cell of the first and of the second layer (the second may be empty).
    starts: [i64; 2],
    /// First layer, second layer, base.
    states: [BlockStateId; 3],
    water: BlockStateId,
}

impl Column {
    /// A column of `level` with its surface at `surface` m, two `(state, depth)` layers over
    /// `base`, and `water.0` filling the open cells whose centre lies below `water.1` m.
    pub(crate) fn new(
        level: u8,
        surface: f64,
        layers: [(BlockStateId, f64); 2],
        base: BlockStateId,
        water: (BlockStateId, f64),
    ) -> Self {
        let ground_top = last_centre_below(level, surface);
        // Layer j starts at the first cell whose centre lies above `surface - depth_j`; the top
        // cell always belongs to the first layer, and a later layer never starts above an
        // earlier one. `surface - depth` is rounded exactly as in `layered_state`.
        let start0 = first_centre_above(level, surface - layers[0].1).min(ground_top);
        let start1 = first_centre_above(level, surface - layers[1].1).min(start0);
        Self {
            ground_top,
            water_top: last_centre_below(level, water.1),
            starts: [start0, start1],
            states: [layers[0].0, layers[1].0, base],
            water: water.0,
        }
    }

    /// The state of cell `cy`.
    #[inline]
    pub(crate) fn state(&self, cy: i64) -> BlockStateId {
        if cy <= self.ground_top {
            if cy >= self.starts[0] {
                self.states[0]
            } else if cy >= self.starts[1] {
                self.states[1]
            } else {
                self.states[2]
            }
        } else if cy <= self.water_top {
            self.water
        } else {
            BlockStateId::AIR
        }
    }

    /// Highest ground cell.
    #[cfg(test)]
    pub(crate) fn ground_top(&self) -> i64 {
        self.ground_top
    }

    /// Lowest cell that the terrain leaves as air; features grow from here up.
    #[inline]
    pub(crate) fn open_from(&self) -> i64 {
        self.ground_top.max(self.water_top) + 1
    }
}

/// Flat layered ground: every column has the same surface height, so a cell's state depends
/// only on its row.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct FlatGround {
    surface: f64,
    layers: Vec<(BlockStateId, f64)>,
    base: BlockStateId,
}

impl FlatGround {
    /// Ground below `surface` m: `layers` from the top as `(state, cumulative depth)`, `base`
    /// below them.
    pub(crate) fn new(surface: f64, layers: Vec<(BlockStateId, f64)>, base: BlockStateId) -> Self {
        Self {
            surface,
            layers,
            base,
        }
    }

    /// The state of every cell in row `cy` of `level` (air above the ground).
    pub(crate) fn row_state(&self, level: u8, cy: i64) -> BlockStateId {
        layered_state(level, cy, self.surface, &self.layers, self.base).unwrap_or(BlockStateId::AIR)
    }

    /// Lowest row of `level` above the ground.
    pub(crate) fn open_from(&self, level: u8) -> i64 {
        last_centre_below(level, self.surface) + 1
    }

    /// Every state the ground can contain.
    pub(crate) fn states(&self) -> impl Iterator<Item = BlockStateId> + '_ {
        self.layers
            .iter()
            .map(|layer| layer.0)
            .chain(core::iter::once(self.base))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cp_core::hash::Rng;

    const GRASS: BlockStateId = BlockStateId(3);
    const DIRT: BlockStateId = BlockStateId(2);
    const STONE: BlockStateId = BlockStateId(1);
    const WATER: BlockStateId = BlockStateId(7);

    #[test]
    fn the_rule_on_level_zero_is_plain_layering() {
        // Surface at 10.3 m: voxels 0..=9 are ground (centres up to 9.5), the top one is grass,
        // voxels whose centre lies less than 4 m deep are dirt.
        let layers = [(GRASS, 1.0), (DIRT, 4.0)];
        let state = |y| layered_state(0, y, 10.3, &layers, STONE);
        assert_eq!(state(10), None);
        assert_eq!(state(9), Some(GRASS));
        assert_eq!(state(8), Some(DIRT));
        assert_eq!(state(7), Some(DIRT));
        assert_eq!(state(6), Some(DIRT)); // centre 6.5, 3.8 m deep
        assert_eq!(state(5), Some(STONE));
        assert_eq!(state(-100), Some(STONE));
        // A surface exactly on a cell centre leaves that cell open.
        assert_eq!(layered_state(0, 3, 3.5, &layers, STONE), None);
        assert_eq!(layered_state(0, 2, 3.5, &layers, STONE), Some(GRASS));
    }

    #[test]
    fn the_top_cell_keeps_the_surface_layer_on_coarse_levels() {
        let layers = [(GRASS, 1.0), (DIRT, 4.0)];
        // Level 3, surface at 37 m: the cell [32, 40) has its centre at 36 < 37 and is the top
        // ground cell; 1 m deep by its centre, so plain layering would make it dirt.
        assert_eq!(layered_state(3, 4, 37.0, &layers, STONE), Some(GRASS));
        assert_eq!(layered_state(3, 3, 37.0, &layers, STONE), Some(STONE));
        assert_eq!(layered_state(3, 5, 37.0, &layers, STONE), None);
    }

    #[test]
    fn columns_equal_the_reference_rule() {
        let mut rng = Rng::new(11, 2);
        for _ in 0..3000 {
            let level = rng.range_u32(0, 9) as u8;
            let surface = rng.range_f64(-300.0, 300.0);
            let first = rng.range_f64(0.0, 6.0);
            let second = first + rng.range_f64(0.0, 9.0);
            let water_level = rng.range_f64(-300.0, 300.0);
            let layers = [(GRASS, first), (DIRT, second)];
            let column = Column::new(level, surface, layers, STONE, (WATER, water_level));
            let size = 1_i64 << level;
            let around = (surface / size as f64) as i64;
            for cy in around - 40..around + 40 {
                let expected = layered_state(level, cy, surface, &layers, STONE).unwrap_or(
                    if centre(level, cy) < water_level {
                        WATER
                    } else {
                        BlockStateId::AIR
                    },
                );
                assert_eq!(column.state(cy), expected, "{level} {surface} {cy}");
            }
            let ground = layered_state(level, column.ground_top(), surface, &layers, STONE);
            assert!(ground.is_some());
            assert!(
                layered_state(level, column.ground_top() + 1, surface, &layers, STONE).is_none()
            );
            for cy in column.open_from()..column.open_from() + 5 {
                assert_eq!(column.state(cy), BlockStateId::AIR);
            }
            assert_ne!(column.state(column.open_from() - 1), BlockStateId::AIR);
        }
    }

    #[test]
    fn flat_ground_rows() {
        let ground = FlatGround::new(0.0, vec![(GRASS, 1.0), (DIRT, 4.0)], STONE);
        assert_eq!(ground.row_state(0, 0), BlockStateId::AIR);
        assert_eq!(ground.row_state(0, -1), GRASS);
        assert_eq!(ground.row_state(0, -4), DIRT);
        assert_eq!(ground.row_state(0, -5), STONE);
        assert_eq!(ground.row_state(4, -1), GRASS);
        assert_eq!(ground.row_state(4, -2), STONE);
        assert_eq!(ground.open_from(0), 0);
        assert_eq!(ground.open_from(5), 0);
        let states: Vec<_> = ground.states().collect();
        assert_eq!(states, [GRASS, DIRT, STONE]);
        let empty = FlatGround::new(2.5, Vec::new(), STONE);
        assert_eq!(empty.row_state(0, 1), STONE);
        assert_eq!(empty.row_state(0, 2), BlockStateId::AIR);
        assert_eq!(empty.open_from(0), 2);
    }
}
