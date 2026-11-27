//! The downsample rule on the structure set (E1 acceptance) and on natural terrain.

use std::collections::{BTreeSet, VecDeque};
use std::sync::Arc;

use cp_core::block::LodClass;
use cp_core::coords::{CHUNK_VOLUME, LodKey, cell_index, index_to_cell};
use cp_core::ids::BlockStateId;
use cp_core::math::IVec3;
use cp_voxel::{
    CellEdit, ChunkData, VoxelChunk, VoxelTables, VoxelWorld, WorldConfig, downsample_cell,
    downsample_chunk,
};

use crate::common::{
    AIR, DIRT, FLOWER, FnBase, GRASS, HillsBase, LAVA, LEAVES, LOG, STONE, WATER, base_chunk, rng,
    tables,
};

/// A structure inside level-0 chunk (0, 0, 0): its cells and whether it is player-built.
struct Structure {
    name: &'static str,
    cells: Vec<(IVec3, BlockStateId)>,
    authored: bool,
}

fn structures() -> Vec<Structure> {
    let mut list = Vec::new();
    // A 1×1×32 pillar of a STRUCTURAL block, generated (not authored).
    list.push(Structure {
        name: "pillar",
        cells: (0..32).map(|y| (IVec3::new(5, y, 9), LOG)).collect(),
        authored: false,
    });
    // A one-thick wall of plain stone, player-built, starting at odd coordinates.
    let mut wall = Vec::new();
    for y in 3..27 {
        for z in 1..30 {
            wall.push((IVec3::new(7, y, z), STONE));
        }
    }
    list.push(Structure {
        name: "wall",
        cells: wall,
        authored: true,
    });
    // A horizontal beam of plain stone, player-built.
    list.push(Structure {
        name: "beam",
        cells: (3..29).map(|z| (IVec3::new(11, 13, z), STONE)).collect(),
        authored: true,
    });
    // A hollow cube (one-thick shell, 12 m edge), player-built.
    let mut cube = Vec::new();
    for y in 5..17 {
        for z in 9..21 {
            for x in 9..21 {
                let shell = x == 9 || x == 20 || y == 5 || y == 16 || z == 9 || z == 20;
                if shell {
                    cube.push((IVec3::new(x, y, z), STONE));
                }
            }
        }
    }
    list.push(Structure {
        name: "hollow cube",
        cells: cube,
        authored: true,
    });
    // A tree: STRUCTURAL trunk and a canopy of (non-structural) leaves, generated.
    let mut tree: Vec<(IVec3, BlockStateId)> =
        (0..10).map(|y| (IVec3::new(16, y, 16), LOG)).collect();
    for y in 7..13_i32 {
        for z in 13..20_i32 {
            for x in 13..20_i32 {
                let (dx, dz) = (x - 16, z - 16);
                let corner = dx.abs() == 3 && dz.abs() == 3;
                let trunk = dx == 0 && dz == 0 && y < 10;
                if !corner && !trunk && dx * dx + dz * dz + (y - 10) * (y - 10) <= 13 {
                    tree.push((IVec3::new(x, y, z), LEAVES));
                }
            }
        }
    }
    list.push(Structure {
        name: "tree",
        cells: tree,
        authored: false,
    });
    list
}

/// The structure as a level-0 chunk (0, 0, 0) in air.
fn level0_chunk(structure: &Structure) -> VoxelChunk {
    let mut cells = vec![AIR; CHUNK_VOLUME];
    let mut chunk = VoxelChunk::default();
    for &(cell, state) in &structure.cells {
        let index = cell_index(cell.x as u8, cell.y as u8, cell.z as u8);
        cells[index] = state;
        if structure.authored {
            chunk.set_authored(index, true);
        }
    }
    chunk.blocks = ChunkData::from_cells(&cells);
    chunk
}

/// Composite of `key` derived from level 0 with the pure chunk downsample: the structure in
/// chunk (0, 0, 0), air everywhere else.
fn derived(structure_chunk: &VoxelChunk, key: LodKey, tables: &VoxelTables) -> VoxelChunk {
    if key.level == 0 {
        return if key.pos == IVec3::ZERO {
            structure_chunk.clone()
        } else {
            VoxelChunk::uniform(AIR)
        };
    }
    let children: Vec<VoxelChunk> = key
        .children()
        .iter()
        .map(|&child| derived(structure_chunk, child, tables))
        .collect();
    let refs: [&VoxelChunk; 8] = std::array::from_fn(|i| &children[i]);
    downsample_chunk(refs, key.level, &tables.flags, &tables.lod)
}

/// Solid cells of a chunk, and whether they form one face-connected component.
fn solid_component_count(chunk: &VoxelChunk, tables: &VoxelTables) -> (usize, usize) {
    let solid: BTreeSet<usize> = (0..CHUNK_VOLUME)
        .filter(|&index| tables.lod_of(chunk.get(index)).class == LodClass::Solid)
        .collect();
    let mut seen = BTreeSet::new();
    let mut components = 0;
    for &start in &solid {
        if !seen.insert(start) {
            continue;
        }
        components += 1;
        let mut queue = VecDeque::from([start]);
        while let Some(index) = queue.pop_front() {
            let [x, y, z] = index_to_cell(index).map(i32::from);
            for (dx, dy, dz) in [
                (1, 0, 0),
                (-1, 0, 0),
                (0, 1, 0),
                (0, -1, 0),
                (0, 0, 1),
                (0, 0, -1),
            ] {
                let (nx, ny, nz) = (x + dx, y + dy, z + dz);
                if !(0..32).contains(&nx) || !(0..32).contains(&ny) || !(0..32).contains(&nz) {
                    continue;
                }
                let neighbour = cell_index(nx as u8, ny as u8, nz as u8);
                if solid.contains(&neighbour) && seen.insert(neighbour) {
                    queue.push_back(neighbour);
                }
            }
        }
    }
    (solid.len(), components)
}

#[test]
fn structure_set_stays_connected_at_levels_1_to_3() {
    let tables = tables();
    for structure in structures() {
        let chunk = level0_chunk(&structure);
        let (cells, components) = solid_component_count(&chunk, &tables);
        assert!(
            cells > 0 && components == 1,
            "{} at level 0",
            structure.name
        );
        for level in 1..=3 {
            let key = LodKey::new(level, IVec3::ZERO);
            let coarse = derived(&chunk, key, &tables);
            let (cells, components) = solid_component_count(&coarse, &tables);
            assert!(cells > 0, "{} vanished at level {level}", structure.name);
            assert_eq!(
                components, 1,
                "{} falls apart at level {level} ({cells} solid cells)",
                structure.name
            );
            // Authored structures stay authored; generated ones never become authored.
            assert_eq!(
                coarse.has_authored(),
                structure.authored,
                "{}",
                structure.name
            );
        }
    }
}

#[test]
fn authored_structures_survive_through_the_edit_pyramid() {
    // The same structures built by a player in an empty world: the pyramid of the
    // EditStore must give the same coarse composites as the pure downsample.
    let tables = tables();
    for structure in structures().into_iter().filter(|s| s.authored) {
        let base = Arc::new(FnBase::new("test:air", |_, _| AIR));
        let config = WorldConfig {
            top_level: 3,
            ..WorldConfig::default()
        };
        let mut world = VoxelWorld::new(base, tables.clone(), config);
        let edits: Vec<CellEdit> = structure
            .cells
            .iter()
            .map(|&(cell, state)| CellEdit { cell, state })
            .collect();
        world.apply(&edits);
        let snapshot = world.snapshot();
        let chunk = level0_chunk(&structure);
        for level in 0..=3 {
            let key = LodKey::new(level, IVec3::ZERO);
            let sampled = snapshot.sample(key);
            assert_eq!(
                *sampled,
                derived(&chunk, key, &tables),
                "{} level {level}",
                structure.name
            );
            let (cells, components) = solid_component_count(&sampled, &tables);
            assert!(
                cells > 0 && components == 1,
                "{} level {level}",
                structure.name
            );
        }
    }
}

#[test]
fn natural_terrain_uses_the_four_of_eight_rule() {
    let tables = tables();
    // Every combination of natural (non-structural, non-authored) children on several levels.
    let natural = [AIR, STONE, DIRT, GRASS, WATER, LAVA, FLOWER];
    let mut rng = rng(5);
    for _ in 0..20_000 {
        let children: [BlockStateId; 8] =
            std::array::from_fn(|_| natural[rng.range_u32(0, natural.len() as u32) as usize]);
        let level = rng.range_u32(1, 8) as u8;
        let result = downsample_cell(&children, 0, level, &tables.flags, &tables.lod);
        let solid = children
            .iter()
            .filter(|&&s| tables.lod_of(s).class == LodClass::Solid)
            .count();
        let fluid = children
            .iter()
            .filter(|&&s| tables.lod_of(s).class == LodClass::Fluid)
            .count();
        let class = tables.lod_of(result.state).class;
        if solid >= 4 {
            assert_eq!(class, LodClass::Solid, "{children:?}");
        } else if solid + fluid >= 4 {
            assert_eq!(class, LodClass::Fluid, "{children:?}");
        } else {
            assert_eq!(result.state, AIR, "{children:?}");
        }
        assert!(!result.authored);
    }
    // A generated terrain chunk: each parent cell is solid iff at least four children are.
    let hills = HillsBase::new();
    for pos in [
        IVec3::new(0, 0, 0),
        IVec3::new(-1, 0, 2),
        IVec3::new(3, -1, -2),
    ] {
        let parent = LodKey::new(1, pos);
        let children = parent.children().map(|key| base_chunk(&hills, key));
        let refs: [&VoxelChunk; 8] = std::array::from_fn(|i| &children[i]);
        let down = downsample_chunk(refs, 1, &tables.flags, &tables.lod);
        for index in 0..CHUNK_VOLUME {
            let [x, y, z] = index_to_cell(index).map(i32::from);
            let octant = ((x >> 4) | ((z >> 4) << 1) | ((y >> 4) << 2)) as usize;
            let child = &children[octant];
            let mut solid = 0;
            let mut structural = 0;
            for dy in 0..2 {
                for dz in 0..2 {
                    for dx in 0..2 {
                        let cx = ((x & 15) * 2 + dx) as u8;
                        let cy = ((y & 15) * 2 + dy) as u8;
                        let cz = ((z & 15) * 2 + dz) as u8;
                        let state = child.get(cell_index(cx, cy, cz));
                        if tables.lod_of(state).class == LodClass::Solid {
                            solid += 1;
                            if state == LOG {
                                structural += 1;
                            }
                        }
                    }
                }
            }
            let parent_solid = tables.lod_of(down.get(index)).class == LodClass::Solid;
            assert_eq!(parent_solid, solid >= 4 || structural >= 1, "cell {index}");
        }
    }
}
