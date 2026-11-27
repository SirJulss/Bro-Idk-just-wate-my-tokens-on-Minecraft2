//! Test fixtures: a block palette with every render class, cull-group case and orientation,
//! padded-chunk builders and random chunks.

use cp_core::block::{BlockFlags, BlockRender, RenderClass};
use cp_core::face::{Face, Orientation};
use cp_core::hash::Rng;
use cp_core::ids::{BlockStateId, MaterialId};

use crate::{MeshInput, MeshTables, PADDED_VOLUME, padded_index};

/// Cull group numbers of the palette. `ODD` differs from `GLASS` only in bit 15, so equal low
/// bits must not be mistaken for an equal group.
pub(crate) const GLASS: u16 = 1;
pub(crate) const LEAVES: u16 = 2;
pub(crate) const WATER: u16 = 3;
pub(crate) const LAVA: u16 = 4;
pub(crate) const ODD: u16 = 0x8001;

/// Per-state tables of a test palette.
#[derive(Clone, Default)]
pub(crate) struct Palette {
    pub flags: Vec<BlockFlags>,
    pub render: Vec<BlockRender>,
    pub cull_group: Vec<u16>,
}

impl Palette {
    pub(crate) fn tables(&self) -> MeshTables<'_> {
        MeshTables {
            flags: &self.flags,
            render: &self.render,
            cull_group: &self.cull_group,
        }
    }

    /// Adds a state; `local` are the materials of the unrotated block's faces in
    /// [`Face::index`] order.
    fn add(
        &mut self,
        flags: BlockFlags,
        class: RenderClass,
        local: [u16; 6],
        orientation: Orientation,
        group: u16,
    ) -> BlockStateId {
        let id = BlockStateId(self.flags.len() as u32);
        self.flags.push(flags);
        self.render.push(match class {
            RenderClass::None | RenderClass::Model => BlockRender {
                class,
                ..BlockRender::NONE
            },
            _ => BlockRender::oriented(class, local.map(MaterialId), orientation),
        });
        self.cull_group.push(group);
        id
    }
}

/// The states of [`palette`].
pub(crate) struct Blocks {
    pub air: BlockStateId,
    pub stone: BlockStateId,
    pub dirt: BlockStateId,
    pub sand: BlockStateId,
    /// Grass with its top towards each world face, by [`Face::index`]; `grass[2]` is upright.
    pub grass: [BlockStateId; 6],
    /// Logs along x, y and z.
    pub logs: [BlockStateId; 3],
    /// A block with six different faces in all 24 orientations, by [`Orientation::index`].
    pub furnace: [BlockStateId; 24],
    /// Cutout, cull group `LEAVES`.
    pub leaves: BlockStateId,
    /// Cutout without a cull group.
    pub grate: BlockStateId,
    /// Non-opaque cube, cull group `GLASS`.
    pub glass: BlockStateId,
    /// Cutout, cull group `GLASS` (another class in the same group).
    pub pane: BlockStateId,
    /// Non-opaque cube without a cull group.
    pub clear: BlockStateId,
    /// Non-opaque cube, cull group `ODD`.
    pub odd: BlockStateId,
    /// Fluid, cull group `WATER`.
    pub water: BlockStateId,
    /// Fluid, cull group `LAVA`.
    pub lava: BlockStateId,
    /// Opaque model: emits nothing, hides its neighbours.
    pub machine: BlockStateId,
    /// Non-opaque model: emits nothing, hides nothing.
    pub lever: BlockStateId,
    /// Render class `None` but opaque: emits nothing, hides its neighbours.
    pub barrier: BlockStateId,
}

impl Blocks {
    /// Opaque cubes of many materials and orientations.
    pub(crate) fn opaque_mix(&self) -> Vec<BlockStateId> {
        let mut mix = vec![self.stone, self.dirt, self.sand];
        mix.extend(self.grass);
        mix.extend(self.logs);
        mix.extend(self.furnace.iter().step_by(5));
        mix
    }

    pub(crate) fn cutout_mix(&self) -> Vec<BlockStateId> {
        vec![
            self.leaves,
            self.grate,
            self.pane,
            self.stone,
            self.logs[1],
            self.glass,
        ]
    }

    pub(crate) fn fluid_mix(&self) -> Vec<BlockStateId> {
        vec![
            self.water,
            self.lava,
            self.sand,
            self.stone,
            self.leaves,
            self.water,
        ]
    }

    pub(crate) fn glass_mix(&self) -> Vec<BlockStateId> {
        vec![self.glass, self.pane, self.clear, self.odd, self.grate]
    }

    pub(crate) fn model_mix(&self) -> Vec<BlockStateId> {
        vec![
            self.machine,
            self.lever,
            self.barrier,
            self.stone,
            self.leaves,
            self.water,
        ]
    }

    pub(crate) fn everything(&self) -> Vec<BlockStateId> {
        let mut mix = self.opaque_mix();
        mix.extend(self.cutout_mix());
        mix.extend(self.fluid_mix());
        mix.extend(self.glass_mix());
        mix.extend(self.model_mix());
        mix.extend(self.furnace);
        mix
    }
}

/// The test palette: every render class, cull groups across classes and sharing low bits,
/// opaque non-rendered states and blocks in many orientations.
pub(crate) fn palette() -> (Palette, Blocks) {
    let solid = BlockFlags::SOLID;
    let opaque = BlockFlags::SOLID | BlockFlags::OPAQUE;
    let identity = Orientation::IDENTITY;
    let mut p = Palette::default();
    let air = p.add(BlockFlags::empty(), RenderClass::None, [0; 6], identity, 0);
    let stone = p.add(opaque, RenderClass::Cube, [1; 6], identity, 0);
    let dirt = p.add(opaque, RenderClass::Cube, [2; 6], identity, 0);
    let sand = p.add(opaque, RenderClass::Cube, [5; 6], identity, 0);
    let grass = Face::ALL.map(|up| {
        p.add(
            opaque,
            RenderClass::Cube,
            [4, 4, 3, 2, 4, 4],
            Orientation::from_up(up),
            0,
        )
    });
    let logs = [Face::PosX, Face::PosY, Face::PosZ].map(|up| {
        p.add(
            opaque | BlockFlags::STRUCTURAL,
            RenderClass::Cube,
            [7, 7, 6, 6, 7, 7],
            Orientation::from_up(up),
            0,
        )
    });
    let furnace =
        Orientation::ALL.map(|o| p.add(opaque, RenderClass::Cube, [20, 21, 22, 23, 24, 25], o, 0));
    let cutout = solid | BlockFlags::CUTOUT;
    let leaves = p.add(cutout, RenderClass::Cutout, [8; 6], identity, LEAVES);
    let grate = p.add(cutout, RenderClass::Cutout, [9; 6], identity, 0);
    let glass = p.add(solid, RenderClass::Cube, [10; 6], identity, GLASS);
    let pane = p.add(cutout, RenderClass::Cutout, [12; 6], identity, GLASS);
    let clear = p.add(solid, RenderClass::Cube, [11; 6], identity, 0);
    let odd = p.add(solid, RenderClass::Cube, [13; 6], identity, ODD);
    let water = p.add(
        BlockFlags::FLUID,
        RenderClass::Fluid,
        [14; 6],
        identity,
        WATER,
    );
    let lava = p.add(
        BlockFlags::FLUID | BlockFlags::EMISSIVE,
        RenderClass::Fluid,
        [15, 15, 16, 16, 15, 15],
        Orientation::from_up(Face::NegZ),
        LAVA,
    );
    let machine = p.add(opaque, RenderClass::Model, [0; 6], identity, 0);
    let lever = p.add(BlockFlags::empty(), RenderClass::Model, [0; 6], identity, 0);
    let barrier = p.add(opaque, RenderClass::None, [0; 6], identity, 0);
    let blocks = Blocks {
        air,
        stone,
        dirt,
        sand,
        grass,
        logs,
        furnace,
        leaves,
        grate,
        glass,
        pane,
        clear,
        odd,
        water,
        lava,
        machine,
        lever,
        barrier,
    };
    (p, blocks)
}

/// How a state of the base pack is built (`engine/content/base/blocks/*.toml`).
#[derive(Clone, Copy)]
enum PackBlock {
    Air,
    /// An opaque cube with one material on every face.
    Cube(&'static str),
    /// `up6` grass with its top towards the face.
    Grass(Face),
    /// `axis3` log with the materials (end grain, bark) along the axis of the face.
    Log((&'static str, &'static str), Face),
    /// Cutout leaves in the cull group `leaves`.
    Leaves(&'static str),
    /// A fluid: material, extra flags, cull group.
    Fluid(&'static str, BlockFlags, u16),
}

/// Every state of the base pack in sorted key order, which is the id order (contract §4.4).
/// Cull groups are numbered by sorted name: `lava` 1, `leaves` 2, `water` 3.
const BASE_PACK: [(&str, PackBlock); 45] = {
    use PackBlock::{Air, Cube, Fluid, Grass, Leaves, Log};
    const BIRCH: (&str, &str) = ("log_birch_top", "log_birch_side");
    const CHERRY: (&str, &str) = ("log_cherry_top", "log_cherry_side");
    const OAK: (&str, &str) = ("log_oak_top", "log_oak_side");
    const SPRUCE: (&str, &str) = ("log_spruce_top", "log_spruce_side");
    [
        ("cubeplanet:air", Air),
        ("cubeplanet:basalt", Cube("basalt")),
        ("cubeplanet:clay", Cube("clay")),
        ("cubeplanet:core", Cube("core")),
        ("cubeplanet:dirt", Cube("dirt")),
        ("cubeplanet:grass", Grass(Face::PosY)),
        ("cubeplanet:grass[up=+x]", Grass(Face::PosX)),
        ("cubeplanet:grass[up=+z]", Grass(Face::PosZ)),
        ("cubeplanet:grass[up=-x]", Grass(Face::NegX)),
        ("cubeplanet:grass[up=-y]", Grass(Face::NegY)),
        ("cubeplanet:grass[up=-z]", Grass(Face::NegZ)),
        ("cubeplanet:gravel", Cube("gravel")),
        ("cubeplanet:ice", Cube("ice")),
        ("cubeplanet:lava", Fluid("lava", BlockFlags::EMISSIVE, 1)),
        ("cubeplanet:leaves_birch", Leaves("leaves_birch")),
        ("cubeplanet:leaves_cherry", Leaves("leaves_cherry")),
        ("cubeplanet:leaves_oak", Leaves("leaves_oak")),
        ("cubeplanet:leaves_spruce", Leaves("leaves_spruce")),
        ("cubeplanet:limestone", Cube("limestone")),
        ("cubeplanet:log_birch", Log(BIRCH, Face::PosY)),
        ("cubeplanet:log_birch[axis=x]", Log(BIRCH, Face::PosX)),
        ("cubeplanet:log_birch[axis=z]", Log(BIRCH, Face::PosZ)),
        ("cubeplanet:log_cherry", Log(CHERRY, Face::PosY)),
        ("cubeplanet:log_cherry[axis=x]", Log(CHERRY, Face::PosX)),
        ("cubeplanet:log_cherry[axis=z]", Log(CHERRY, Face::PosZ)),
        ("cubeplanet:log_oak", Log(OAK, Face::PosY)),
        ("cubeplanet:log_oak[axis=x]", Log(OAK, Face::PosX)),
        ("cubeplanet:log_oak[axis=z]", Log(OAK, Face::PosZ)),
        ("cubeplanet:log_spruce", Log(SPRUCE, Face::PosY)),
        ("cubeplanet:log_spruce[axis=x]", Log(SPRUCE, Face::PosX)),
        ("cubeplanet:log_spruce[axis=z]", Log(SPRUCE, Face::PosZ)),
        ("cubeplanet:ore_coal", Cube("ore_coal")),
        ("cubeplanet:ore_copper", Cube("ore_copper")),
        ("cubeplanet:ore_iron", Cube("ore_iron")),
        ("cubeplanet:ore_tin", Cube("ore_tin")),
        ("cubeplanet:planks_birch", Cube("planks_birch")),
        ("cubeplanet:planks_cherry", Cube("planks_cherry")),
        ("cubeplanet:planks_oak", Cube("planks_oak")),
        ("cubeplanet:planks_spruce", Cube("planks_spruce")),
        ("cubeplanet:sand", Cube("sand")),
        ("cubeplanet:sandstone", Cube("sandstone")),
        ("cubeplanet:slate", Cube("slate")),
        ("cubeplanet:snow", Cube("snow")),
        ("cubeplanet:stone", Cube("stone")),
        ("cubeplanet:water", Fluid("water", BlockFlags::empty(), 3)),
    ]
};

/// The states of [`base_pack`] that the terrain generators place.
#[derive(Clone, Copy, Debug)]
pub(crate) struct PackStates {
    pub air: BlockStateId,
    pub stone: BlockStateId,
    pub dirt: BlockStateId,
    /// Grass with its top up (+Y).
    pub grass: BlockStateId,
    pub sand: BlockStateId,
    pub gravel: BlockStateId,
    /// Copper, tin, iron and coal ore.
    pub ores: [BlockStateId; 4],
    /// An upright oak log.
    pub log: BlockStateId,
    /// Oak and birch leaves (one cull group).
    pub leaves: [BlockStateId; 2],
    pub water: BlockStateId,
}

/// The 45 states of the base content pack (`engine/content/base/blocks`) with the ids, flags,
/// render classes and cull groups cp-content gives them: ids by sorted state key, air first,
/// cull groups numbered by sorted name (`lava` 1, `leaves` 2, `water` 3; contract §4.4).
///
/// Benchmarks use it because the ids of real content collide in small id-keyed tables in a
/// way a hand-made palette does not: stone (43) and gravel (11), air (0), oak leaves (16) and
/// copper ore (32) are equal modulo 16.
pub(crate) fn base_pack() -> (Palette, PackStates) {
    // The id rule of cp-content: ids in sorted key order, air first.
    assert!(BASE_PACK.windows(2).all(|pair| pair[0].0 < pair[1].0));
    assert_eq!(BASE_PACK[0].0, "cubeplanet:air");
    // Materials are numbered in order of first use; only their equality matters here.
    let mut materials: Vec<&str> = Vec::new();
    let mut material = |name: &'static str| {
        let index = materials.iter().position(|&known| known == name);
        let index = index.unwrap_or_else(|| {
            materials.push(name);
            materials.len() - 1
        });
        index as u16
    };
    let opaque = BlockFlags::SOLID | BlockFlags::OPAQUE;
    let mut palette = Palette::default();
    for (_, block) in BASE_PACK {
        match block {
            PackBlock::Air => {
                palette.add(
                    BlockFlags::empty(),
                    RenderClass::None,
                    [0; 6],
                    Orientation::IDENTITY,
                    0,
                );
            }
            PackBlock::Cube(name) => {
                let local = [material(name); 6];
                palette.add(opaque, RenderClass::Cube, local, Orientation::IDENTITY, 0);
            }
            PackBlock::Grass(up) => {
                let (side, top, bottom) = (
                    material("grass_side"),
                    material("grass_top"),
                    material("dirt"),
                );
                let local = [side, side, top, bottom, side, side];
                let orientation = Orientation::from_up(up);
                palette.add(opaque, RenderClass::Cube, local, orientation, 0);
            }
            PackBlock::Log((end, bark), axis) => {
                let (end, bark) = (material(end), material(bark));
                let local = [bark, bark, end, end, bark, bark];
                let flags = opaque | BlockFlags::STRUCTURAL;
                palette.add(
                    flags,
                    RenderClass::Cube,
                    local,
                    Orientation::from_up(axis),
                    0,
                );
            }
            PackBlock::Leaves(name) => {
                let flags = BlockFlags::SOLID | BlockFlags::CUTOUT;
                let local = [material(name); 6];
                palette.add(flags, RenderClass::Cutout, local, Orientation::IDENTITY, 2);
            }
            PackBlock::Fluid(name, extra, group) => {
                let flags = BlockFlags::FLUID | extra;
                let local = [material(name); 6];
                palette.add(
                    flags,
                    RenderClass::Fluid,
                    local,
                    Orientation::IDENTITY,
                    group,
                );
            }
        }
    }
    let id = |key: &str| {
        let index = BASE_PACK.iter().position(|&(known, _)| known == key);
        BlockStateId(index.expect("a base pack key") as u32)
    };
    let states = PackStates {
        air: id("cubeplanet:air"),
        stone: id("cubeplanet:stone"),
        dirt: id("cubeplanet:dirt"),
        grass: id("cubeplanet:grass"),
        sand: id("cubeplanet:sand"),
        gravel: id("cubeplanet:gravel"),
        ores: [
            id("cubeplanet:ore_copper"),
            id("cubeplanet:ore_tin"),
            id("cubeplanet:ore_iron"),
            id("cubeplanet:ore_coal"),
        ],
        log: id("cubeplanet:log_oak"),
        leaves: [id("cubeplanet:leaves_oak"), id("cubeplanet:leaves_birch")],
        water: id("cubeplanet:water"),
    };
    // The ids whose collisions modulo 16 the cache review measured.
    let pinned = [
        states.stone,
        states.gravel,
        states.leaves[0],
        states.ores[0],
    ];
    assert_eq!(pinned.map(|state| state.0), [43, 11, 16, 32]);
    assert_eq!(states.water.0, 44);
    (palette, states)
}

/// `palette` with state `i` moved to id `i · stride`; the ids in between are unused states
/// that look like air. With `stride` a power of two, every state falls into the same slot of
/// any id-keyed direct-mapped table with at most `stride` entries.
pub(crate) fn spread(palette: &Palette, stride: u32) -> Palette {
    let len = (palette.flags.len() - 1) * stride as usize + 1;
    let mut spread = Palette {
        flags: vec![BlockFlags::empty(); len],
        render: vec![BlockRender::NONE; len],
        cull_group: vec![0; len],
    };
    for state in 0..palette.flags.len() {
        let id = spread_id(BlockStateId(state as u32), stride).index();
        spread.flags[id] = palette.flags[state];
        spread.render[id] = palette.render[state];
        spread.cull_group[id] = palette.cull_group[state];
    }
    spread
}

/// The id of `state` in [`spread`]`(_, stride)`; [`BlockStateId::UNCHANGED`] stays itself.
pub(crate) fn spread_id(state: BlockStateId, stride: u32) -> BlockStateId {
    if state.is_unchanged() {
        state
    } else {
        BlockStateId(state.0 * stride)
    }
}

/// True for the padded cells the mesher may read: chunk cells and face-halo cells (at most one
/// coordinate outside `0..32`).
pub(crate) fn is_readable(x: i32, y: i32, z: i32) -> bool {
    [x, y, z]
        .into_iter()
        .filter(|coordinate| !(0..32).contains(coordinate))
        .count()
        <= 1
}

/// A padded chunk with `f(x, y, z)` in every readable cell and [`BlockStateId::UNCHANGED`],
/// which no table describes, in the edge and corner halo cells: reading those would fail.
pub(crate) fn padded_from_fn(
    mut f: impl FnMut(i32, i32, i32) -> BlockStateId,
) -> Vec<BlockStateId> {
    let mut cells = vec![BlockStateId::UNCHANGED; PADDED_VOLUME];
    for y in -1..=32 {
        for z in -1..=32 {
            for x in -1..=32 {
                if is_readable(x, y, z) {
                    cells[padded_index(x, y, z)] = f(x, y, z);
                }
            }
        }
    }
    cells
}

/// One cell per call, uniformly from `mix` with probability `density`, else air.
pub(crate) fn random_cells(rng: &mut Rng, mix: &[BlockStateId], density: f64) -> Vec<BlockStateId> {
    padded_from_fn(|_, _, _| {
        if rng.f64() < density {
            mix[rng.range_u32(0, mix.len() as u32) as usize]
        } else {
            BlockStateId::AIR
        }
    })
}

/// Overlapping random boxes of random states (air included): large faces of several keys next
/// to each other, which is what the greedy merge has to get right.
pub(crate) fn random_boxes(rng: &mut Rng, mix: &[BlockStateId], boxes: u32) -> Vec<BlockStateId> {
    let mut volume = vec![BlockStateId::AIR; PADDED_VOLUME];
    for _ in 0..boxes {
        let state = if rng.range_u32(0, 5) == 0 {
            BlockStateId::AIR
        } else {
            mix[rng.range_u32(0, mix.len() as u32) as usize]
        };
        // Half-open ranges of padded coordinates per axis, inside -1..=32.
        let [rx, ry, rz] = [0, 1, 2].map(|_| {
            let low = rng.range_u32(0, 34) as i32 - 1;
            let size = rng.range_u32(1, 20) as i32;
            low..(low + size).min(33)
        });
        for y in ry {
            for z in rz.clone() {
                for x in rx.clone() {
                    volume[padded_index(x, y, z)] = state;
                }
            }
        }
    }
    padded_from_fn(|x, y, z| volume[padded_index(x, y, z)])
}

/// A random authored mask: `None`, sparse, half, full or one box, by `kind % 5`.
pub(crate) fn random_authored(rng: &mut Rng, kind: u32) -> Option<Box<[u64; 512]>> {
    let mut mask = Box::new([0_u64; 512]);
    match kind % 5 {
        0 => return None,
        1 | 2 => {
            let density = if kind % 5 == 1 { 0.05 } else { 0.5 };
            for index in 0..32_768 {
                if rng.f64() < density {
                    mask[index / 64] |= 1 << (index % 64);
                }
            }
        }
        3 => mask.fill(u64::MAX),
        _ => {
            let low = [0, 1, 2].map(|_| rng.range_u32(0, 32) as usize);
            let high = low.map(|low| (low + rng.range_u32(1, 24) as usize).min(32));
            for y in low[1]..high[1] {
                for z in low[2]..high[2] {
                    for x in low[0]..high[0] {
                        let index = x + 32 * z + 1024 * y;
                        mask[index / 64] |= 1 << (index % 64);
                    }
                }
            }
        }
    }
    Some(mask)
}

/// A mesh input over `cells`.
pub(crate) fn input<'a>(
    palette: &'a Palette,
    cells: &'a [BlockStateId],
    authored: Option<&'a [u64; 512]>,
    level: u8,
) -> MeshInput<'a> {
    MeshInput {
        cells,
        authored,
        level,
        tables: palette.tables(),
    }
}
