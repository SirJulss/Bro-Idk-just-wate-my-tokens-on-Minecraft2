//! Layered flat ground as a [`BaseSource`].

use cp_core::block::BlockFlags;
use cp_core::coords::{LodKey, VOXEL_LIMIT};
use cp_core::ids::BlockStateId;
use cp_core::math::IAabb;
use cp_voxel::{BaseSource, Cancelled, GeneratorIdentity, NodeClass, chunk_cells};

use crate::config_hash::ConfigHasher;
use crate::error::WorldgenError;
use crate::palette::{PaletteRole, TerrainPalette, is_opaque};
use crate::region::{Region, checked_request};
use crate::strata::FlatGround;

/// One layer of [`FlatConfig`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct FlatLayer {
    /// The palette role whose state fills the layer.
    pub role: PaletteRole,
    /// Thickness in metres (>= 1).
    pub thickness: u32,
}

/// Parameters of a [`FlatSource`].
///
/// Session-independent: the layers name [`PaletteRole`]s, never block state ids, so the
/// source's identity can hash exactly what decides its cells (together with the palette's
/// [`TerrainPalette::key_hash`]).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct FlatConfig {
    /// Height of the ground surface (voxel y): cells whose centre lies below it are ground.
    pub surface_y: i32,
    /// The layers below the surface, top first.
    pub layers: Vec<FlatLayer>,
    /// The role of everything below the last layer.
    pub bottom: PaletteRole,
}

impl FlatConfig {
    /// Grass (1 m) over dirt (3 m) over stone, surface at `surface_y`.
    #[must_use]
    pub fn grassland(surface_y: i32) -> Self {
        Self {
            surface_y,
            layers: vec![
                FlatLayer {
                    role: PaletteRole::Grass,
                    thickness: 1,
                },
                FlatLayer {
                    role: PaletteRole::Dirt,
                    thickness: 3,
                },
            ],
            bottom: PaletteRole::Stone,
        }
    }
}

/// Flat layered ground on a plane: the simplest base world, for meshing and rendering tests.
///
/// The cells of a row share one state, so `classify` is exact: `Empty` iff every row of the
/// chunk is air, `Opaque` iff every row's state is `OPAQUE`. Layers follow the common layer
/// rule (crate documentation): on coarse levels the top ground cell keeps the first layer's
/// state, thinner layers below are point-sampled at the cell centres.
///
/// # Identity
/// `macro_hash` covers the surface height, the palette's key hash and, in order, the role and
/// thickness of every layer and the bottom role. Two flat sources therefore share an identity
/// only if their cells carry the same state keys, whatever ids the session gave those states
/// (for palettes whose key hash is known, see [`TerrainPalette::key_hash`]).
#[derive(Clone, Debug)]
pub struct FlatSource {
    identity: GeneratorIdentity,
    surface_y: i32,
    ground: FlatGround,
    /// The ground states that are `OPAQUE`, sorted.
    opaque: Vec<BlockStateId>,
}

impl FlatSource {
    /// Identity key of this generator.
    pub const KEY: &'static str = "cubeplanet:flat";
    /// Output version; bumped on every change of the generated cells.
    pub const VERSION: u32 = 1;

    /// Flat ground of `config` with the states of `palette`, for the world seed `seed`
    /// (recorded in the identity; flat ground does not use it). `flags` is the per-state flag
    /// table of the registry the palette's ids come from; it decides which states are
    /// `OPAQUE`.
    ///
    /// # Errors
    /// [`WorldgenError::InvalidConfig`] for a layer thinner than 1 m, a stack that leaves the
    /// voxel limits, or a palette that contains [`BlockStateId::UNCHANGED`].
    pub fn new(
        seed: u64,
        config: &FlatConfig,
        palette: &TerrainPalette,
        flags: &[BlockFlags],
    ) -> Result<Self, WorldgenError> {
        let invalid = |message: &str| Err(WorldgenError::InvalidConfig(message.to_owned()));
        if config.layers.iter().any(|layer| layer.thickness == 0) {
            return invalid("flat layers must be at least 1 m thick");
        }
        palette.validate()?;
        let depth: u64 = config
            .layers
            .iter()
            .map(|layer| u64::from(layer.thickness))
            .sum();
        if config.surface_y.unsigned_abs() >= VOXEL_LIMIT.unsigned_abs()
            || depth >= u64::from(VOXEL_LIMIT.unsigned_abs())
        {
            return invalid("the flat layers must lie inside the voxel limits");
        }
        let identity = GeneratorIdentity {
            key: Self::KEY.to_owned(),
            version: Self::VERSION,
            seed,
            macro_hash: Self::parameter_hash(config, palette),
        };
        let mut bottom_depth = 0_u64;
        let layers = config
            .layers
            .iter()
            .map(|layer| {
                bottom_depth += u64::from(layer.thickness);
                // Exact: the depth is below 2^30.
                (palette.state(layer.role), bottom_depth as f64)
            })
            .collect();
        let ground = FlatGround::new(
            f64::from(config.surface_y),
            layers,
            palette.state(config.bottom),
        );
        let mut opaque: Vec<BlockStateId> = ground
            .states()
            .filter(|&state| is_opaque(flags, state))
            .collect();
        opaque.sort_unstable();
        opaque.dedup();
        log::debug!("flat source {identity:?}");
        Ok(Self {
            identity,
            surface_y: config.surface_y,
            ground,
            opaque,
        })
    }

    /// Grass over dirt over stone with its surface at `surface_y` ([`FlatConfig::grassland`]).
    ///
    /// # Errors
    /// As [`FlatSource::new`].
    pub fn grassland(
        seed: u64,
        palette: &TerrainPalette,
        surface_y: i32,
        flags: &[BlockFlags],
    ) -> Result<Self, WorldgenError> {
        Self::new(seed, &FlatConfig::grassland(surface_y), palette, flags)
    }

    /// The state of every cell in row `y` (cells of `level`).
    ///
    /// # Panics
    /// If `level > MAX_LOD_LEVEL`.
    #[must_use]
    pub fn row_state(&self, level: u8, y: i32) -> BlockStateId {
        assert!(
            level <= cp_core::coords::MAX_LOD_LEVEL,
            "level {level} out of range"
        );
        self.ground.row_state(level, y.into())
    }

    /// `GeneratorIdentity::macro_hash` of a validated `config` with `palette`: everything that
    /// decides the cells, in a fixed order, and nothing session-local.
    fn parameter_hash(config: &FlatConfig, palette: &TerrainPalette) -> u64 {
        let mut hasher = ConfigHasher::new(Self::KEY);
        // Every layer is at least 1 m thick and the stack less than 2^30 m deep, so the count
        // fits in u32.
        hasher
            .i32(config.surface_y)
            .u64(palette.key_hash)
            .u32(config.layers.len() as u32);
        for layer in &config.layers {
            hasher.str(layer.role.name()).u32(layer.thickness);
        }
        hasher.str(config.bottom.name()).finish()
    }

    fn is_opaque(&self, state: BlockStateId) -> bool {
        self.opaque.binary_search(&state).is_ok()
    }
}

impl BaseSource for FlatSource {
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
        Ok(())
    }

    fn classify(&self, key: LodKey) -> NodeClass {
        if !key.is_valid() {
            return NodeClass::Mixed;
        }
        let cells = Region::from_aabb(chunk_cells(key));
        let mut all_air = true;
        let mut all_opaque = true;
        for y in cells.min[1]..cells.max[1] {
            let state = self.ground.row_state(key.level, y);
            all_air &= state.is_air();
            all_opaque &= self.is_opaque(state);
        }
        if all_air {
            NodeClass::Empty
        } else if all_opaque {
            NodeClass::Opaque
        } else {
            NodeClass::Mixed
        }
    }

    fn subtree_empty(&self, key: LodKey) -> bool {
        // Every cell of the key and its descendants has its centre above the key's floor,
        // which is at or above the surface.
        key.is_valid() && key.origin_voxel().y >= self.surface_y
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::palette::ROLE_COUNT;
    use cp_core::math::IVec3;

    const STONE: BlockStateId = BlockStateId(1);
    const DIRT: BlockStateId = BlockStateId(2);
    const GRASS: BlockStateId = BlockStateId(3);

    /// States 1–9 in the order stone, dirt, grass, sand, gravel, snow, water, log, leaves,
    /// the other roles 10–18 in their order.
    fn palette() -> TerrainPalette {
        let mut states = [BlockStateId::AIR; ROLE_COUNT];
        let first = [
            (PaletteRole::Stone, STONE),
            (PaletteRole::Dirt, DIRT),
            (PaletteRole::Grass, GRASS),
            (PaletteRole::Sand, BlockStateId(4)),
            (PaletteRole::Gravel, BlockStateId(5)),
            (PaletteRole::Snow, BlockStateId(6)),
            (PaletteRole::Water, BlockStateId(7)),
            (PaletteRole::Log, BlockStateId(8)),
            (PaletteRole::Leaves, BlockStateId(9)),
        ];
        for (role, state) in first {
            states[role.index()] = state;
        }
        for (index, role) in PaletteRole::ALL[9..].iter().enumerate() {
            states[role.index()] = BlockStateId(10 + index as u32);
        }
        TerrainPalette::from_states(states, 5)
    }

    fn flags() -> Vec<BlockFlags> {
        let solid = BlockFlags::SOLID | BlockFlags::OPAQUE;
        let mut flags = vec![solid; ROLE_COUNT + 1];
        flags[0] = BlockFlags::empty();
        flags[7] = BlockFlags::FLUID;
        flags
    }

    fn layer(role: PaletteRole, thickness: u32) -> FlatLayer {
        FlatLayer { role, thickness }
    }

    fn grassland(surface_y: i32) -> FlatSource {
        FlatSource::grassland(9, &palette(), surface_y, &flags()).unwrap()
    }

    fn chunk(source: &FlatSource, key: LodKey) -> Vec<BlockStateId> {
        let mut out = vec![BlockStateId::UNCHANGED; 32 * 32 * 32];
        source
            .fill(key.level, chunk_cells(key), &mut out, &|| false)
            .unwrap();
        out
    }

    #[test]
    fn rows_follow_the_layers_on_every_level() {
        let flat = grassland(64);
        assert_eq!(flat.row_state(0, 64), BlockStateId::AIR);
        assert_eq!(flat.row_state(0, 63), GRASS);
        assert_eq!(flat.row_state(0, 62), DIRT);
        assert_eq!(flat.row_state(0, 60), DIRT);
        assert_eq!(flat.row_state(0, 59), STONE);
        // Level 3: rows of 8 m; [56, 64) is the top ground row and stays grass.
        assert_eq!(flat.row_state(3, 8), BlockStateId::AIR);
        assert_eq!(flat.row_state(3, 7), GRASS);
        assert_eq!(flat.row_state(3, 6), STONE);
        // Level 1: [62, 64) grass, [60, 62) centre 61 lies 3 m deep: dirt.
        assert_eq!(flat.row_state(1, 31), GRASS);
        assert_eq!(flat.row_state(1, 30), DIRT);
        assert_eq!(flat.row_state(1, 29), STONE);
    }

    #[test]
    fn layers_take_the_states_of_their_roles() {
        let config = FlatConfig {
            surface_y: 10,
            layers: vec![layer(PaletteRole::Snow, 2), layer(PaletteRole::Sand, 3)],
            bottom: PaletteRole::Gravel,
        };
        let palette = palette();
        let flat = FlatSource::new(0, &config, &palette, &flags()).unwrap();
        assert_eq!(flat.row_state(0, 10), BlockStateId::AIR);
        assert_eq!(flat.row_state(0, 9), palette.snow);
        assert_eq!(flat.row_state(0, 8), palette.snow);
        assert_eq!(flat.row_state(0, 7), palette.sand);
        assert_eq!(flat.row_state(0, 5), palette.sand);
        assert_eq!(flat.row_state(0, 4), palette.gravel);
        assert_eq!(flat.row_state(0, -1000), palette.gravel);
    }

    #[test]
    fn fill_classify_and_subtree_agree() {
        let flat = grassland(5);
        for level in 0..=6_u8 {
            for y in -3..=2 {
                let key = LodKey::new(level, IVec3::new(1, y, -2));
                let cells = chunk_cells(key);
                let out = chunk(&flat, key);
                let all_air = out.iter().all(|s| s.is_air());
                let all_opaque = out.iter().all(|&s| s == GRASS || s == DIRT || s == STONE);
                let expected = if all_air {
                    NodeClass::Empty
                } else if all_opaque {
                    NodeClass::Opaque
                } else {
                    NodeClass::Mixed
                };
                // Exact, not just conservative.
                assert_eq!(flat.classify(key), expected, "{key:?}");
                if flat.subtree_empty(key) {
                    assert!(all_air);
                }
                for (i, &state) in out.iter().enumerate() {
                    let row = cells.min.y + (i / 1024) as i32;
                    assert_eq!(state, flat.row_state(level, row));
                }
            }
        }
        assert!(flat.subtree_empty(LodKey::new(4, IVec3::new(0, 1, 0))));
        assert!(!flat.subtree_empty(LodKey::new(4, IVec3::new(0, 0, 0))));
    }

    #[test]
    fn water_layers_are_mixed() {
        let config = FlatConfig {
            surface_y: 0,
            layers: vec![layer(PaletteRole::Water, 64)],
            bottom: PaletteRole::Stone,
        };
        let flat = FlatSource::new(0, &config, &palette(), &flags()).unwrap();
        assert_eq!(
            flat.classify(LodKey::lod0(IVec3::new(0, -1, 0))),
            NodeClass::Mixed
        );
        assert_eq!(
            flat.classify(LodKey::lod0(IVec3::new(0, -3, 0))),
            NodeClass::Opaque
        );
        assert_eq!(
            flat.classify(LodKey::lod0(IVec3::new(0, 0, 0))),
            NodeClass::Empty
        );
    }

    #[test]
    fn bad_configs_are_rejected() {
        let mut config = FlatConfig {
            surface_y: 0,
            layers: vec![layer(PaletteRole::Grass, 0)],
            bottom: PaletteRole::Stone,
        };
        assert!(FlatSource::new(0, &config, &palette(), &flags()).is_err());
        config.layers[0].thickness = 1;
        assert!(FlatSource::new(0, &config, &palette(), &flags()).is_ok());
        // The edit sentinel is rejected in every role, used by the layers or not.
        for role in PaletteRole::ALL {
            let mut ids = palette().states();
            ids[role.index()] = BlockStateId::UNCHANGED;
            let states = TerrainPalette::from_states(ids, 5);
            assert!(
                FlatSource::new(0, &config, &states, &flags()).is_err(),
                "{role:?}"
            );
        }
        config.surface_y = i32::MAX;
        assert!(FlatSource::new(0, &config, &palette(), &flags()).is_err());
        config.surface_y = 0;
        config.layers.push(layer(PaletteRole::Dirt, u32::MAX));
        assert!(FlatSource::new(0, &config, &palette(), &flags()).is_err());
    }

    /// Regression: the identity used to hash the layer thicknesses and a palette-wide key hash
    /// only, so flat worlds with other layer states shared one identity.
    #[test]
    fn identity_covers_every_layer_role_and_the_bottom() {
        use PaletteRole::{Dirt, Grass, Gravel, Sand, Snow, Stone, Water};
        let palette = palette();
        let source = |config: &FlatConfig| FlatSource::new(0, config, &palette, &flags()).unwrap();
        // Same palette, same thicknesses, other states: different identities and cells.
        let grass = FlatConfig::grassland(6);
        let beach = FlatConfig {
            surface_y: 6,
            layers: vec![layer(Sand, 1), layer(Gravel, 3)],
            bottom: Snow,
        };
        let (a, b) = (source(&grass), source(&beach));
        assert_ne!(a.identity(), b.identity());
        assert_ne!(a.identity().stable_hash(), b.identity().stable_hash());
        let key = LodKey::lod0(IVec3::ZERO);
        assert_ne!(chunk(&a, key), chunk(&b, key));
        // Configs that differ in one thing each: one layer's role, the bottom, a thickness, the
        // surface, the order of the layers, an extra layer.
        let one = |layers: Vec<FlatLayer>, bottom| FlatConfig {
            surface_y: 6,
            layers,
            bottom,
        };
        let configs = [
            grass.clone(),
            one(vec![layer(Grass, 1), layer(Sand, 3)], Stone),
            one(vec![layer(Snow, 1), layer(Dirt, 3)], Stone),
            one(vec![layer(Grass, 1), layer(Dirt, 3)], Gravel),
            one(vec![layer(Grass, 1), layer(Dirt, 3)], Water),
            one(vec![layer(Grass, 2), layer(Dirt, 3)], Stone),
            one(vec![layer(Grass, 1), layer(Dirt, 2)], Stone),
            one(vec![layer(Dirt, 3), layer(Grass, 1)], Stone),
            one(
                vec![layer(Grass, 1), layer(Dirt, 3), layer(Stone, 1)],
                Stone,
            ),
            one(vec![layer(Grass, 4)], Stone),
            one(Vec::new(), Grass),
            FlatConfig::grassland(7),
            beach,
        ];
        let sources: Vec<FlatSource> = configs.iter().map(source).collect();
        for (i, x) in sources.iter().enumerate() {
            for (j, y) in sources.iter().enumerate() {
                let same = x.identity() == y.identity();
                assert_eq!(same, i == j, "{:?} {:?}", configs[i], configs[j]);
            }
        }
        // The palette's key hash enters as well; the seed only through its own field.
        let other_keys = TerrainPalette {
            key_hash: 6,
            ..palette
        };
        let c = FlatSource::new(0, &grass, &other_keys, &flags()).unwrap();
        assert_ne!(c.identity().macro_hash, a.identity().macro_hash);
        let seeded = FlatSource::new(3, &grass, &palette, &flags()).unwrap();
        assert_eq!(seeded.identity().macro_hash, a.identity().macro_hash);
        assert_ne!(seeded.identity(), a.identity());
    }

    /// Two sessions that numbered the same states differently: equal identities, and cells
    /// that are equal key by key.
    #[test]
    fn identity_does_not_depend_on_session_ids() {
        let first = palette();
        // The second session numbers the same states in reverse: stone is 18, the last 1.
        let count = ROLE_COUNT as u32 + 1;
        let reversed = |id: BlockStateId| BlockStateId(count - id.0);
        let second = TerrainPalette::from_states(first.states().map(reversed), first.key_hash);
        let key_of = |palette: &TerrainPalette, state: BlockStateId| {
            PaletteRole::ALL
                .iter()
                .find(|&&role| palette.state(role) == state)
                .map_or("air", |role| role.name())
        };
        let mut second_flags = flags();
        second_flags.reverse();
        second_flags.rotate_right(1);
        // Sand in the top rows and gravel below on every tested level (water only on level 0).
        let config = FlatConfig {
            surface_y: 0,
            layers: vec![layer(PaletteRole::Sand, 2), layer(PaletteRole::Water, 1)],
            bottom: PaletteRole::Gravel,
        };
        let a = FlatSource::new(4, &config, &first, &flags()).unwrap();
        let b = FlatSource::new(4, &config, &second, &second_flags).unwrap();
        assert_eq!(a.identity(), b.identity());
        for level in [0_u8, 1, 3] {
            let key = LodKey::new(level, IVec3::new(0, -1, 0));
            let keys_a: Vec<&str> = chunk(&a, key).iter().map(|&s| key_of(&first, s)).collect();
            let keys_b: Vec<&str> = chunk(&b, key).iter().map(|&s| key_of(&second, s)).collect();
            assert_eq!(keys_a, keys_b, "level {level}");
            assert!(keys_a.contains(&"sand") && keys_a.contains(&"gravel"));
        }
    }
}
