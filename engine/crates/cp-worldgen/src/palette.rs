//! The block states the test generators write.

use cp_core::block::BlockFlags;
use cp_core::hash::hash_bytes;
use cp_core::ids::BlockStateId;

use crate::error::WorldgenError;

/// Number of roles of a [`TerrainPalette`].
pub const ROLE_COUNT: usize = 18;

/// The roles of a [`TerrainPalette`] (and of [`PaletteKeys`]), in their fixed order: the ground
/// (grass, dirt, stone, sand, gravel, snow), water, oak log and leaves, then the logs and leaves
/// of birch, spruce and cherry and the rocks and soils of variety (clay, slate, limestone).
///
/// Generator parameters name roles, never block state ids: a role is session-independent, so
/// it may enter an identity hash, and together with [`TerrainPalette::key_hash`] it determines
/// the state key a cell gets.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum PaletteRole {
    /// [`TerrainPalette::grass`].
    Grass,
    /// [`TerrainPalette::dirt`].
    Dirt,
    /// [`TerrainPalette::stone`].
    Stone,
    /// [`TerrainPalette::sand`].
    Sand,
    /// [`TerrainPalette::gravel`].
    Gravel,
    /// [`TerrainPalette::snow`].
    Snow,
    /// [`TerrainPalette::water`].
    Water,
    /// [`TerrainPalette::log`] (oak).
    Log,
    /// [`TerrainPalette::leaves`] (oak).
    Leaves,
    /// [`TerrainPalette::log_birch`].
    LogBirch,
    /// [`TerrainPalette::leaves_birch`].
    LeavesBirch,
    /// [`TerrainPalette::log_spruce`].
    LogSpruce,
    /// [`TerrainPalette::leaves_spruce`].
    LeavesSpruce,
    /// [`TerrainPalette::log_cherry`].
    LogCherry,
    /// [`TerrainPalette::leaves_cherry`].
    LeavesCherry,
    /// [`TerrainPalette::clay`].
    Clay,
    /// [`TerrainPalette::slate`].
    Slate,
    /// [`TerrainPalette::limestone`].
    Limestone,
}

impl PaletteRole {
    /// Every role, in the fixed order.
    pub const ALL: [Self; ROLE_COUNT] = [
        Self::Grass,
        Self::Dirt,
        Self::Stone,
        Self::Sand,
        Self::Gravel,
        Self::Snow,
        Self::Water,
        Self::Log,
        Self::Leaves,
        Self::LogBirch,
        Self::LeavesBirch,
        Self::LogSpruce,
        Self::LeavesSpruce,
        Self::LogCherry,
        Self::LeavesCherry,
        Self::Clay,
        Self::Slate,
        Self::Limestone,
    ];

    /// The role's name: `"grass"`, `"dirt"`, …, the field name in [`TerrainPalette`]. Fixed
    /// forever, because [`PaletteKeys::hash`] and generator identities hash it.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Grass => "grass",
            Self::Dirt => "dirt",
            Self::Stone => "stone",
            Self::Sand => "sand",
            Self::Gravel => "gravel",
            Self::Snow => "snow",
            Self::Water => "water",
            Self::Log => "log",
            Self::Leaves => "leaves",
            Self::LogBirch => "log_birch",
            Self::LeavesBirch => "leaves_birch",
            Self::LogSpruce => "log_spruce",
            Self::LeavesSpruce => "leaves_spruce",
            Self::LogCherry => "log_cherry",
            Self::LeavesCherry => "leaves_cherry",
            Self::Clay => "clay",
            Self::Slate => "slate",
            Self::Limestone => "limestone",
        }
    }

    /// Position of the role in [`Self::ALL`].
    #[must_use]
    pub const fn index(self) -> usize {
        self as usize
    }
}

/// The block states of the test worlds, one per role, chosen by the caller (normally resolved
/// from the content registry with [`TerrainPalette::resolve`]).
///
/// The generators only copy these ids into the cells they fill; which of them are `OPAQUE`
/// they learn from the flags table passed to their constructors. Ids are session-local
/// (contract §4.4), so they never enter a hash: [`TerrainPalette::key_hash`] carries the
/// session-independent identity of the choice into the generators' identities.
///
/// Roles should name distinct states. [`BlockStateId::UNCHANGED`] is rejected by every
/// generator constructor (a base source never returns it); air is allowed and simply leaves
/// the cells of that role empty.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TerrainPalette {
    /// Top layer of land below the snow line (1 m).
    pub grass: BlockStateId,
    /// Below grass (3 m).
    pub dirt: BlockStateId,
    /// Everything below the surface layers; bare rock on steep slopes.
    pub stone: BlockStateId,
    /// Beaches and shallow sea floors (4 m).
    pub sand: BlockStateId,
    /// Deep sea floors, stony shores and scree (2 m).
    pub gravel: BlockStateId,
    /// Top layer on high ground (1 m).
    pub snow: BlockStateId,
    /// The cells below the sea level that the ground leaves open.
    pub water: BlockStateId,
    /// Oak trunks and the structure test shapes. Should be `STRUCTURAL`, so that the LOD
    /// pyramid keeps thin structures (contract §5.1); so should the other logs.
    pub log: BlockStateId,
    /// Oak crowns, bushes and the structure test tree.
    pub leaves: BlockStateId,
    /// Birch trunks.
    pub log_birch: BlockStateId,
    /// Birch crowns.
    pub leaves_birch: BlockStateId,
    /// Spruce trunks.
    pub log_spruce: BlockStateId,
    /// Spruce crowns.
    pub leaves_spruce: BlockStateId,
    /// Cherry trunks.
    pub log_cherry: BlockStateId,
    /// Cherry crowns.
    pub leaves_cherry: BlockStateId,
    /// Muddy shores and lake floors (2 m).
    pub clay: BlockStateId,
    /// Dark rock of the slate regions.
    pub slate: BlockStateId,
    /// Pale rock of the limestone regions.
    pub limestone: BlockStateId,
    /// Session-independent hash of the state keys the ids stand for ([`PaletteKeys::hash`]),
    /// or 0 when the caller did not say. It enters the generators' identities, so worlds built
    /// from palettes with different keys never share an identity, provided the hash is known:
    /// palettes with the hash 0 cannot be told apart, so 0 is only for worlds whose identity
    /// is never persisted or compared (unit tests). [`TerrainPalette::resolve`] always fills
    /// it in.
    pub key_hash: u64,
}

/// The state keys of a [`TerrainPalette`], one per role.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PaletteKeys<'a> {
    /// Key per role, in [`PaletteRole::ALL`] order.
    pub keys: [&'a str; ROLE_COUNT],
}

impl PaletteKeys<'static> {
    /// The terrain states of the base content pack.
    pub const BASE: Self = Self {
        keys: [
            "cubeplanet:grass",
            "cubeplanet:dirt",
            "cubeplanet:stone",
            "cubeplanet:sand",
            "cubeplanet:gravel",
            "cubeplanet:snow",
            "cubeplanet:water",
            "cubeplanet:log_oak",
            "cubeplanet:leaves_oak",
            "cubeplanet:log_birch",
            "cubeplanet:leaves_birch",
            "cubeplanet:log_spruce",
            "cubeplanet:leaves_spruce",
            "cubeplanet:log_cherry",
            "cubeplanet:leaves_cherry",
            "cubeplanet:clay",
            "cubeplanet:slate",
            "cubeplanet:limestone",
        ],
    };
}

impl<'a> PaletteKeys<'a> {
    /// The key of `role`.
    #[must_use]
    pub const fn key(&self, role: PaletteRole) -> &'a str {
        self.keys[role.index()]
    }

    /// The same keys with `role` mapped to `key`.
    #[must_use]
    pub const fn with(mut self, role: PaletteRole, key: &'a str) -> Self {
        self.keys[role.index()] = key;
        self
    }

    /// `(role name, key)` pairs in the fixed role order ([`PaletteRole::ALL`]).
    #[must_use]
    pub fn roles(&self) -> [(&'static str, &'a str); ROLE_COUNT] {
        PaletteRole::ALL.map(|role| (role.name(), self.key(role)))
    }

    /// FNV-1a over `role=key` lines (each ended by `\n`) in [`PaletteKeys::roles`] order. A
    /// fixed recipe: the same keys give the same hash on every machine and in every session.
    #[must_use]
    pub fn hash(&self) -> u64 {
        let mut bytes = Vec::with_capacity(512);
        for (role, key) in self.roles() {
            bytes.extend_from_slice(role.as_bytes());
            bytes.push(b'=');
            bytes.extend_from_slice(key.as_bytes());
            bytes.push(b'\n');
        }
        hash_bytes(&bytes)
    }
}

impl TerrainPalette {
    /// A palette from one state per role in [`PaletteRole::ALL`] order and the key hash.
    #[must_use]
    pub const fn from_states(states: [BlockStateId; ROLE_COUNT], key_hash: u64) -> Self {
        let [
            grass,
            dirt,
            stone,
            sand,
            gravel,
            snow,
            water,
            log,
            leaves,
            log_birch,
            leaves_birch,
            log_spruce,
            leaves_spruce,
            log_cherry,
            leaves_cherry,
            clay,
            slate,
            limestone,
        ] = states;
        Self {
            grass,
            dirt,
            stone,
            sand,
            gravel,
            snow,
            water,
            log,
            leaves,
            log_birch,
            leaves_birch,
            log_spruce,
            leaves_spruce,
            log_cherry,
            leaves_cherry,
            clay,
            slate,
            limestone,
            key_hash,
        }
    }

    /// Looks every role's key up (normally `|key| registry.state(key)`) and records
    /// [`PaletteKeys::hash`] as the palette's key hash.
    ///
    /// # Errors
    /// [`WorldgenError::UnknownState`] for the first role whose key the lookup does not know,
    /// [`WorldgenError::InvalidConfig`] if the lookup returns [`BlockStateId::UNCHANGED`].
    pub fn resolve(
        keys: &PaletteKeys<'_>,
        mut lookup: impl FnMut(&str) -> Option<BlockStateId>,
    ) -> Result<Self, WorldgenError> {
        let mut states = [BlockStateId::AIR; ROLE_COUNT];
        for (slot, (role, key)) in states.iter_mut().zip(keys.roles()) {
            *slot = lookup(key).ok_or_else(|| WorldgenError::UnknownState {
                role,
                key: key.to_owned(),
            })?;
        }
        let palette = Self::from_states(states, keys.hash());
        palette.validate()?;
        Ok(palette)
    }

    /// The state of `role`.
    #[must_use]
    pub const fn state(&self, role: PaletteRole) -> BlockStateId {
        match role {
            PaletteRole::Grass => self.grass,
            PaletteRole::Dirt => self.dirt,
            PaletteRole::Stone => self.stone,
            PaletteRole::Sand => self.sand,
            PaletteRole::Gravel => self.gravel,
            PaletteRole::Snow => self.snow,
            PaletteRole::Water => self.water,
            PaletteRole::Log => self.log,
            PaletteRole::Leaves => self.leaves,
            PaletteRole::LogBirch => self.log_birch,
            PaletteRole::LeavesBirch => self.leaves_birch,
            PaletteRole::LogSpruce => self.log_spruce,
            PaletteRole::LeavesSpruce => self.leaves_spruce,
            PaletteRole::LogCherry => self.log_cherry,
            PaletteRole::LeavesCherry => self.leaves_cherry,
            PaletteRole::Clay => self.clay,
            PaletteRole::Slate => self.slate,
            PaletteRole::Limestone => self.limestone,
        }
    }

    /// The states in [`PaletteRole::ALL`] order.
    #[must_use]
    pub fn states(&self) -> [BlockStateId; ROLE_COUNT] {
        PaletteRole::ALL.map(|role| self.state(role))
    }

    /// Rejects the edit sentinel, which a base source must never return.
    pub(crate) fn validate(&self) -> Result<(), WorldgenError> {
        if self.states().iter().any(|state| state.is_unchanged()) {
            return Err(WorldgenError::InvalidConfig(
                "a palette state is BlockStateId::UNCHANGED".to_owned(),
            ));
        }
        Ok(())
    }
}

/// True when `flags` marks `state` as `OPAQUE`. States outside the table count as not opaque,
/// which keeps `classify` conservative.
pub(crate) fn is_opaque(flags: &[BlockFlags], state: BlockStateId) -> bool {
    flags
        .get(state.index())
        .is_some_and(|flags| flags.contains(BlockFlags::OPAQUE))
}

/// A palette whose role `i` is state `i + 1` (air stays free), with key hash 0: unit tests.
#[cfg(test)]
pub(crate) fn numbered_palette() -> TerrainPalette {
    TerrainPalette::from_states(core::array::from_fn(|i| BlockStateId(i as u32 + 1)), 0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolve_looks_up_every_role_in_order() {
        let mut asked = Vec::new();
        let palette = TerrainPalette::resolve(&PaletteKeys::BASE, |key| {
            asked.push(key.to_owned());
            Some(BlockStateId(asked.len() as u32))
        })
        .unwrap();
        assert_eq!(
            asked,
            [
                "cubeplanet:grass",
                "cubeplanet:dirt",
                "cubeplanet:stone",
                "cubeplanet:sand",
                "cubeplanet:gravel",
                "cubeplanet:snow",
                "cubeplanet:water",
                "cubeplanet:log_oak",
                "cubeplanet:leaves_oak",
                "cubeplanet:log_birch",
                "cubeplanet:leaves_birch",
                "cubeplanet:log_spruce",
                "cubeplanet:leaves_spruce",
                "cubeplanet:log_cherry",
                "cubeplanet:leaves_cherry",
                "cubeplanet:clay",
                "cubeplanet:slate",
                "cubeplanet:limestone",
            ]
        );
        assert_eq!(palette.states(), numbered_palette().states());
        assert_eq!(palette.key_hash, PaletteKeys::BASE.hash());
        assert_ne!(palette.key_hash, 0);
    }

    #[test]
    fn resolve_reports_the_missing_role() {
        let error = TerrainPalette::resolve(&PaletteKeys::BASE, |key| {
            (key != "cubeplanet:snow").then_some(BlockStateId(3))
        })
        .unwrap_err();
        assert_eq!(
            error,
            WorldgenError::UnknownState {
                role: "snow",
                key: "cubeplanet:snow".to_owned()
            }
        );
        let error = TerrainPalette::resolve(&PaletteKeys::BASE, |_| Some(BlockStateId::UNCHANGED))
            .unwrap_err();
        assert!(matches!(error, WorldgenError::InvalidConfig(_)));
    }

    #[test]
    fn key_hash_is_pinned_and_depends_on_every_key() {
        let mut bytes = Vec::new();
        for (role, key) in PaletteKeys::BASE.roles() {
            bytes.extend_from_slice(format!("{role}={key}\n").as_bytes());
        }
        assert_eq!(PaletteKeys::BASE.hash(), hash_bytes(&bytes));
        for role in PaletteRole::ALL {
            let other = PaletteKeys::BASE.with(role, "cubeplanet:core");
            assert_ne!(other.hash(), PaletteKeys::BASE.hash(), "{role:?}");
        }
        // Swapping two roles changes the hash although the set of keys is the same.
        let swapped = PaletteKeys::BASE
            .with(PaletteRole::Grass, PaletteKeys::BASE.key(PaletteRole::Dirt))
            .with(PaletteRole::Dirt, PaletteKeys::BASE.key(PaletteRole::Grass));
        assert_ne!(swapped.hash(), PaletteKeys::BASE.hash());
    }

    #[test]
    fn roles_name_the_fields_in_the_fixed_order() {
        assert_eq!(
            PaletteRole::ALL.map(PaletteRole::name),
            [
                "grass",
                "dirt",
                "stone",
                "sand",
                "gravel",
                "snow",
                "water",
                "log",
                "leaves",
                "log_birch",
                "leaves_birch",
                "log_spruce",
                "leaves_spruce",
                "log_cherry",
                "leaves_cherry",
                "clay",
                "slate",
                "limestone",
            ]
        );
        // The derived order is the fixed order, and the index is the position.
        assert!(PaletteRole::ALL.windows(2).all(|pair| pair[0] < pair[1]));
        for (index, role) in PaletteRole::ALL.into_iter().enumerate() {
            assert_eq!(role.index(), index);
        }
        let palette =
            TerrainPalette::from_states(core::array::from_fn(|i| BlockStateId(i as u32 + 11)), 0);
        assert_eq!(
            PaletteRole::ALL.map(|role| palette.state(role).0),
            core::array::from_fn(|i| i as u32 + 11)
        );
        assert_eq!(palette.states(), PaletteRole::ALL.map(|r| palette.state(r)));
        assert_eq!(palette.log_spruce, BlockStateId(22));
        assert_eq!(palette.limestone, BlockStateId(28));
        let keys = PaletteKeys::BASE;
        assert_eq!(keys.key(PaletteRole::Grass), "cubeplanet:grass");
        assert_eq!(keys.key(PaletteRole::Gravel), "cubeplanet:gravel");
        assert_eq!(keys.key(PaletteRole::Log), "cubeplanet:log_oak");
        assert_eq!(keys.key(PaletteRole::Leaves), "cubeplanet:leaves_oak");
        assert_eq!(
            keys.key(PaletteRole::LeavesCherry),
            "cubeplanet:leaves_cherry"
        );
        for (index, role) in PaletteRole::ALL.into_iter().enumerate() {
            assert_eq!(keys.roles()[index], (role.name(), keys.key(role)));
        }
    }

    #[test]
    fn opacity_comes_from_the_table() {
        let flags = [
            BlockFlags::empty(),
            BlockFlags::SOLID | BlockFlags::OPAQUE,
            BlockFlags::SOLID | BlockFlags::CUTOUT,
        ];
        assert!(!is_opaque(&flags, BlockStateId(0)));
        assert!(is_opaque(&flags, BlockStateId(1)));
        assert!(!is_opaque(&flags, BlockStateId(2)));
        assert!(!is_opaque(&flags, BlockStateId(3)));
        assert!(!is_opaque(&flags, BlockStateId::UNCHANGED));
    }
}
