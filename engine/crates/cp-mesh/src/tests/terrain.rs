//! Terrain-like test chunks: a heightfield with grass, dirt, sand and stone layers, a sea and
//! trees with cutout leaves; with the base pack's states also caves, ores, gravel, mountains
//! on coarse levels and dense forest canopies.

use cp_core::face::Face;
use cp_core::hash::{hash_combine, hash_ivec3, mix64};
use cp_core::ids::BlockStateId;
use cp_core::math::{DVec2, DVec3, IVec3};
use cp_core::noise::Fbm;

use super::fixtures::{Blocks, PackStates, padded_from_fn};
use super::reference::{
    assert_same_faces, assert_same_mesh, check_invariants, mesh_faces, reference_faces,
    reference_mesh,
};
use crate::{Layer, MeshInput, Mesher, PADDED_SIZE, padded_index};

/// Water fills every air cell at or below this height.
pub(crate) const SEA_LEVEL: i32 = 2;

/// Leaves reach this far from the trunk (columns), so trees of neighbouring columns matter.
const CROWN_RADIUS: i32 = 2;

/// What [`Terrain::pack_chunk`] generates besides the layered ground.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Style {
    /// Ridged mountains (slopes of several cells per column on coarse levels) instead of
    /// rolling hills.
    pub rough: bool,
    /// Worm caves, ore clusters and gravel pockets below the surface.
    pub caves: bool,
}

pub(crate) struct Terrain {
    seed: u64,
    hills: Fbm,
    /// Three octaves for the 3D fields (caves, gravel), which are sampled per cell.
    volume: Fbm,
}

impl Terrain {
    pub(crate) fn new(seed: u64) -> Self {
        Self {
            seed,
            hills: Fbm::new(4, 2.0, 0.5),
            volume: Fbm::new(3, 2.0, 0.5),
        }
    }

    /// Surface height of `style` at the level-0 column `(x, z)`.
    fn surface(&self, style: Style, x: i32, z: i32) -> i32 {
        if !style.rough {
            return self.height(x, z);
        }
        let p = DVec2::new(f64::from(x), f64::from(z));
        let ridges = self.hills.ridged2(self.seed ^ 0x0011_D9E5, p / 700.0);
        let detail = self.hills.sample2(self.seed ^ 0x0D37_A11E, p / 90.0);
        (ridges * 240.0 + detail * 30.0).floor() as i32 + 40
    }

    /// The padded chunk at chunk position `chunk` of LOD `level` with the base pack's states.
    ///
    /// Every cell is decided at its centre, by the same rule on every level (contract §5.1):
    /// solid below the surface, grass on top, three metres of dirt (sand on beaches), stone
    /// below, water up to the sea level; with `style.caves` also worm caves, ore clusters and
    /// gravel pockets in the rock; trees on level 0. Edge and corner halo cells are
    /// `UNCHANGED`, as the mesher never reads them.
    pub(crate) fn pack_chunk(
        &self,
        states: &PackStates,
        style: Style,
        level: u8,
        chunk: [i32; 3],
    ) -> Vec<BlockStateId> {
        let size = 1_i32 << level;
        // Level-0 voxel at the centre of padded cell `c` (in -1..=32) along one axis.
        let centre = |axis: usize, c: i32| ((chunk[axis] * 32 + c) << level) + (size >> 1);
        let heights: Vec<i32> = (0..PADDED_SIZE * PADDED_SIZE)
            .map(|i| {
                let (x, z) = ((i % PADDED_SIZE) as i32 - 1, (i / PADDED_SIZE) as i32 - 1);
                self.surface(style, centre(0, x), centre(2, z))
            })
            .collect();
        let height = |x: i32, z: i32| heights[(z + 1) as usize * PADDED_SIZE + (x + 1) as usize];
        let mut volume = padded_from_fn(|x, y, z| {
            let h = height(x, z);
            let p = IVec3::new(centre(0, x), centre(1, y), centre(2, z));
            // Depth of the cell centre below the top of the surface cell's column.
            let depth = h - p.y;
            let beach = h <= SEA_LEVEL + 1;
            if depth < 0 {
                if p.y <= SEA_LEVEL {
                    states.water
                } else {
                    states.air
                }
            } else if depth < size {
                if beach { states.sand } else { states.grass }
            } else if depth < size + 3 {
                if beach { states.sand } else { states.dirt }
            } else if style.caves && depth >= size + 4 {
                self.rock(states, p)
            } else {
                states.stone
            }
        });
        if level == 0 {
            self.plant_trees(states, style, chunk, &mut volume);
        }
        volume
    }

    /// The rock at the level-0 voxel `p`: a cave (air), an ore cluster, a gravel pocket or
    /// stone.
    fn rock(&self, states: &PackStates, p: IVec3) -> BlockStateId {
        let at = DVec3::new(f64::from(p.x), f64::from(p.y), f64::from(p.z));
        // Tubes where two independent fields are both near zero.
        let a = self.volume.sample3(self.seed ^ 0xCA7E_0001, at / 40.0);
        let b = self.volume.sample3(self.seed ^ 0xCA7E_0002, at / 40.0);
        if a * a + b * b < 0.012 {
            return states.air;
        }
        // Clusters on a 4 m lattice: 8 % of the lattice cells hold one kind of ore, in about
        // half of their voxels.
        let lattice = IVec3::new(p.x >> 2, p.y >> 2, p.z >> 2);
        let cluster = hash_ivec3(self.seed ^ 0x0BE5, lattice);
        if cluster % 100 < 8 && hash_ivec3(self.seed ^ 0x0BE6, p) % 100 < 55 {
            return states.ores[(cluster >> 32) as usize % 4];
        }
        if self.volume.sample3(self.seed ^ 0x6EA7, at / 18.0) > 0.42 {
            return states.gravel;
        }
        states.stone
    }

    /// Oak and birch trees on the level-0 columns of `chunk` and its crown margin, written into
    /// the readable cells of the padded `volume`.
    fn plant_trees(
        &self,
        states: &PackStates,
        style: Style,
        chunk: [i32; 3],
        volume: &mut [BlockStateId],
    ) {
        let origin = chunk.map(|c| c * 32);
        let margin = 1 + CROWN_RADIUS;
        for tz in -margin..32 + margin {
            for tx in -margin..32 + margin {
                let (wx, wz) = (origin[0] + tx, origin[2] + tz);
                let Some(trunk) = self.tree(wx, wz) else {
                    continue;
                };
                let h = self.surface(style, wx, wz);
                if h <= SEA_LEVEL + 1 {
                    continue;
                }
                let leaves = states.leaves[(trunk % 2) as usize];
                let top = h + trunk;
                for dy in -2..=1_i32 {
                    for dz in -CROWN_RADIUS..=CROWN_RADIUS {
                        for dx in -CROWN_RADIUS..=CROWN_RADIUS {
                            let reach = dx.abs() + dz.abs() + (dy + 1).abs();
                            let (x, y, z) = (tx + dx, top + dy - origin[1], tz + dz);
                            if reach <= 3 && Self::readable(x, y, z) {
                                let cell = &mut volume[padded_index(x, y, z)];
                                if *cell == states.air {
                                    *cell = leaves;
                                }
                            }
                        }
                    }
                }
                for wy in h + 1..=top {
                    let y = wy - origin[1];
                    if Self::readable(tx, y, tz) {
                        volume[padded_index(tx, y, tz)] = states.log;
                    }
                }
            }
        }
    }

    /// A level-0 forest chunk: ground up to `y = 3`, trunks on about every 23rd column and a
    /// 16-cell canopy (`y = 12..28`) of oak and birch leaves with a quarter of its cells open
    /// at random. A stress case: the crowns of cp-worldgen's trees are solid spheres, so real
    /// forests show far fewer faces.
    pub(crate) fn canopy_chunk(&self, states: &PackStates, chunk: [i32; 2]) -> Vec<BlockStateId> {
        let origin = chunk.map(|c| c * 32);
        padded_from_fn(|x, y, z| {
            let (wx, wz) = (origin[0] + x, origin[1] + z);
            let column = hash_ivec3(self.seed ^ 0x7121, IVec3::new(wx, 0, wz));
            let cell = hash_ivec3(self.seed ^ 0x1EAF, IVec3::new(wx, y, wz));
            match y {
                ..=1 => states.stone,
                2 => states.dirt,
                3 => states.grass,
                _ if column.is_multiple_of(23) && y < 26 => states.log,
                12..28 if !cell.is_multiple_of(4) => states.leaves[(cell >> 40) as usize % 2],
                _ => states.air,
            }
        })
    }

    /// True for padded cells that the mesher reads.
    fn readable(x: i32, y: i32, z: i32) -> bool {
        Self::inside(x, y, z) && super::fixtures::is_readable(x, y, z)
    }

    /// Height of the top solid cell of column `(x, z)` (level-0 voxels).
    fn height(&self, x: i32, z: i32) -> i32 {
        let p = DVec2::new(f64::from(x), f64::from(z)) / 140.0;
        (self.hills.sample2(self.seed, p) * 34.0).floor() as i32 + 8
    }

    /// Trunk length of the tree on column `(x, z)`, if one grows there.
    fn tree(&self, x: i32, z: i32) -> Option<i32> {
        let column = (u64::from(x as u32) << 32) | u64::from(z as u32);
        let hash = mix64(hash_combine(self.seed ^ 0x7EE5_7EE5, column));
        hash.is_multiple_of(47)
            .then_some(4 + ((hash >> 20) % 3) as i32)
    }

    /// The padded level-0 chunk at chunk position `chunk` (edge and corner halo cells are
    /// `UNCHANGED`, as the mesher never reads them).
    pub(crate) fn padded_chunk(&self, blocks: &Blocks, chunk: [i32; 3]) -> Vec<BlockStateId> {
        let origin = chunk.map(|c| c * 32);
        let grass = blocks.grass[Face::PosY.index()];
        let log = blocks.logs[1];
        let margin = 1 + CROWN_RADIUS;
        let span = PADDED_SIZE as i32 + 2 * CROWN_RADIUS;
        // Heights of the padded columns and the crown margin around them.
        let heights: Vec<i32> = (0..span * span)
            .map(|i| {
                let (dx, dz) = (i % span, i / span);
                self.height(origin[0] + dx - margin, origin[2] + dz - margin)
            })
            .collect();
        let height = |x: i32, z: i32| heights[((z + margin) * span + x + margin) as usize];
        let mut volume = vec![blocks.air; PADDED_SIZE * PADDED_SIZE * PADDED_SIZE];
        for y in -1..=32 {
            for z in -1..=32 {
                for x in -1..=32 {
                    let (wy, h) = (origin[1] + y, height(x, z));
                    let beach = h <= SEA_LEVEL + 1;
                    volume[padded_index(x, y, z)] = if wy < h - 3 {
                        blocks.stone
                    } else if wy < h {
                        if beach { blocks.sand } else { blocks.dirt }
                    } else if wy == h {
                        if beach { blocks.sand } else { grass }
                    } else if wy <= SEA_LEVEL {
                        blocks.water
                    } else {
                        blocks.air
                    };
                }
            }
        }
        // Trees whose trunk stands in the padded area or close enough for its crown.
        for tz in -margin..32 + margin {
            for tx in -margin..32 + margin {
                let h = height(tx, tz);
                let Some(trunk) = self.tree(origin[0] + tx, origin[2] + tz) else {
                    continue;
                };
                if h <= SEA_LEVEL + 1 {
                    continue;
                }
                let top = h + trunk;
                for dy in -2..=1_i32 {
                    for dz in -CROWN_RADIUS..=CROWN_RADIUS {
                        for dx in -CROWN_RADIUS..=CROWN_RADIUS {
                            let reach = dx.abs() + dz.abs() + (dy + 1).abs();
                            let (x, y, z) = (tx + dx, top + dy - origin[1], tz + dz);
                            if reach <= 3 && Self::inside(x, y, z) {
                                let cell = &mut volume[padded_index(x, y, z)];
                                if *cell == blocks.air {
                                    *cell = blocks.leaves;
                                }
                            }
                        }
                    }
                }
                for wy in h + 1..=top {
                    let y = wy - origin[1];
                    if Self::inside(tx, y, tz) {
                        volume[padded_index(tx, y, tz)] = log;
                    }
                }
            }
        }
        padded_from_fn(|x, y, z| volume[padded_index(x, y, z)])
    }

    fn inside(x: i32, y: i32, z: i32) -> bool {
        [x, y, z].iter().all(|c| (-1..=32).contains(c))
    }
}

/// The chunk positions of the benchmark: 16 × 16 columns of three chunks around the surface.
pub(crate) fn bench_chunks() -> Vec<[i32; 3]> {
    let mut chunks = Vec::new();
    for cy in -1..=1 {
        for cz in 0..16 {
            for cx in 0..16 {
                chunks.push([cx, cy, cz]);
            }
        }
    }
    chunks
}

#[test]
fn terrain_chunks_match_the_reference() {
    let (palette, blocks) = super::fixtures::palette();
    let terrain = Terrain::new(5);
    let mut mesher = Mesher::new();
    let mut layers_seen = [false; Layer::COUNT];
    for chunk in bench_chunks().into_iter().step_by(23) {
        let cells = terrain.padded_chunk(&blocks, chunk);
        let input = MeshInput {
            cells: &cells,
            authored: None,
            level: 0,
            tables: palette.tables(),
        };
        let mesh = mesher.mesh(&input).expect("valid terrain");
        let context = format!("terrain chunk {chunk:?}");
        check_invariants(&mesh, &context);
        assert_same_faces(&mesh_faces(&mesh), &reference_faces(&input), &context);
        assert_same_mesh(&mesh, &reference_mesh(&input), &context);
        for layer in Layer::ALL {
            layers_seen[layer.index()] |= !mesh.layer(layer).is_empty();
        }
    }
    assert_eq!(
        layers_seen, [true; 3],
        "terrain has opaque ground, leaves and sea"
    );
}

#[test]
fn base_pack_scenes_match_the_reference() {
    // The benchmark's scenes with the base pack's ids and with ids spread 64 apart (every
    // state in one slot of any small id-keyed table): caves with ores and gravel, a mountain
    // surface on level 3, a forest canopy.
    let (pack, states) = super::fixtures::base_pack();
    let terrain = Terrain::new(5);
    let caves = Style {
        rough: false,
        caves: true,
    };
    let mountains = Style {
        rough: true,
        caves: true,
    };
    let chunks = [
        (
            "caves",
            terrain.pack_chunk(&states, caves, 0, [1, -2, 3]),
            0,
        ),
        ("caves", terrain.pack_chunk(&states, caves, 0, [6, 0, 2]), 0),
        (
            "forest",
            terrain.pack_chunk(&states, caves, 0, [2, 0, 5]),
            0,
        ),
        (
            "mountains",
            terrain.pack_chunk(&states, mountains, 3, [2, 0, 4]),
            3,
        ),
        (
            "mountains",
            terrain.pack_chunk(&states, mountains, 1, [9, 0, 1]),
            1,
        ),
        ("canopy", terrain.canopy_chunk(&states, [3, 1]), 0),
    ];
    let spread = super::fixtures::spread(&pack, 64);
    let ore_materials = states
        .ores
        .map(|ore| pack.render[ore.index()].face(Face::PosX).material);
    let mut mesher = Mesher::new();
    let mut ores_shown = false;
    for (name, cells, level) in &chunks {
        let spread_cells: Vec<BlockStateId> = cells
            .iter()
            .map(|&state| super::fixtures::spread_id(state, 64))
            .collect();
        for (ids, palette, cells) in [("pack", &pack, cells), ("spread", &spread, &spread_cells)] {
            let input = MeshInput {
                cells,
                authored: None,
                level: *level,
                tables: palette.tables(),
            };
            let mesh = mesher.mesh(&input).expect("valid scene");
            let context = format!("{name} level {level}, {ids} ids");
            check_invariants(&mesh, &context);
            assert_same_faces(&mesh_faces(&mesh), &reference_faces(&input), &context);
            assert_same_mesh(&mesh, &reference_mesh(&input), &context);
            assert!(mesh.regular_len() > 0, "{context}: a scene shows faces");
            ores_shown |= mesh
                .quads
                .iter()
                .any(|quad| ore_materials.contains(&quad.material()));
        }
    }
    assert!(ores_shown, "cave walls show ores");
}
