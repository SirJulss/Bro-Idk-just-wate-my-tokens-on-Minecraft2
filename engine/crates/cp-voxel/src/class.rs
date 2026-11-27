//! Class bookkeeping of composite chunks (contract §5.1): the [`NodeClass`] of a chunk from
//! its override layer and what is known about its base chunk.
//!
//! The class of a composite chunk depends on two things only: the [`LayerClass`] of its
//! override layer (does some cell show the base, is every explicit state air, is every one
//! `OPAQUE`) and the class [`BaseSource::classify`] gives the base chunk. A write therefore
//! changes the class only if it changes the layer class, and the base class is needed only
//! where the result depends on it.
//!
//! `classify` may evaluate the base field over the whole chunk, so its answer is memoised per
//! chunk as a [`BaseClasses`] set (the base never changes, and `classify` is a pure function
//! of the key). Base cells seen in a fill narrow the set without asking `classify` at all:
//! by its exact-conservative contract a chunk with a non-air cell is not `Empty` and one with
//! a non-`OPAQUE` cell is not `Opaque`, so both together mean `Mixed`.

use cp_core::block::BlockFlags;
use cp_core::coords::LodKey;
use cp_core::ids::BlockStateId;

use crate::base::{BaseSource, NodeClass};
use crate::chunk::VoxelChunk;
use crate::layers::LayerMap;

/// What an override layer alone says about the class of its composite chunk.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct LayerClass {
    /// Some cell is `UNCHANGED`: the base shows through there.
    shows_base: bool,
    /// Every explicit (not `UNCHANGED`) state is air; true when there is none.
    all_air: bool,
    /// Every explicit state is `OPAQUE`; true when there is none.
    all_opaque: bool,
}

impl LayerClass {
    /// No layer, or one of `UNCHANGED` cells only: the composite is the base.
    pub(crate) const BASE: Self = Self {
        shows_base: true,
        all_air: true,
        all_opaque: true,
    };

    /// The layer class of `layer` (`None`: no override layer). Looks at the distinct states
    /// of the layer only, not at its cells.
    pub(crate) fn of(layer: Option<&VoxelChunk>, flags: &[BlockFlags]) -> Self {
        let Some(layer) = layer else {
            return Self::BASE;
        };
        let mut class = Self {
            shows_base: false,
            all_air: true,
            all_opaque: true,
        };
        for state in layer.blocks.states() {
            if state.is_unchanged() {
                class.shows_base = true;
            } else {
                class.all_air &= state.is_air();
                class.all_opaque &= flags[state.index()].contains(BlockFlags::OPAQUE);
            }
        }
        class
    }

    /// The class of the composite chunk when the base chunk has class `base`.
    ///
    /// Exact-conservative like [`BaseSource::classify`]: `Empty` only if every cell is air,
    /// `Opaque` only if every cell is `OPAQUE`.
    pub(crate) fn with_base(self, base: NodeClass) -> NodeClass {
        let base_allows = |class: NodeClass| !self.shows_base || base == class;
        if self.all_air && base_allows(NodeClass::Empty) {
            NodeClass::Empty
        } else if self.all_opaque && base_allows(NodeClass::Opaque) {
            NodeClass::Opaque
        } else {
            NodeClass::Mixed
        }
    }

    /// The class of the composite chunk. Calls `base_class` only when neither this layer
    /// class nor `known` decides it.
    pub(crate) fn resolve(
        self,
        known: BaseClasses,
        base_class: impl FnOnce() -> NodeClass,
    ) -> NodeClass {
        known
            .decide(|base| self.with_base(base))
            .unwrap_or_else(|| self.with_base(base_class()))
    }
}

/// What base cells seen in a fill say about the base class of their chunk.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct CellEvidence {
    /// A non-air base cell was seen: the base class is not `Empty`.
    non_air: bool,
    /// A non-`OPAQUE` base cell was seen: the base class is not `Opaque`.
    non_opaque: bool,
}

impl CellEvidence {
    /// The evidence of `cells`, base cells of one chunk at the chunk's level.
    pub(crate) fn of(cells: &[BlockStateId], flags: &[BlockFlags]) -> Self {
        let mut evidence = Self::default();
        for &state in cells {
            evidence.non_air |= !state.is_air();
            evidence.non_opaque |= !flags[state.index()].contains(BlockFlags::OPAQUE);
            if evidence.is_conclusive() {
                break;
            }
        }
        evidence
    }

    /// Both kinds of evidence together.
    #[must_use]
    pub(crate) fn merge(self, other: Self) -> Self {
        Self {
            non_air: self.non_air | other.non_air,
            non_opaque: self.non_opaque | other.non_opaque,
        }
    }

    /// True when the evidence alone proves the base class `Mixed`.
    pub(crate) fn is_conclusive(self) -> bool {
        self.non_air && self.non_opaque
    }
}

/// The classes [`BaseSource::classify`] may still return for one chunk, as far as known: a
/// set of [`NodeClass`]es that always contains the base's actual answer. It starts with all
/// three; facts only remove members.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct BaseClasses(u8);

const fn bit(class: NodeClass) -> u8 {
    match class {
        NodeClass::Empty => 1,
        NodeClass::Opaque => 2,
        NodeClass::Mixed => 4,
    }
}

impl BaseClasses {
    /// Nothing known yet.
    pub(crate) const UNKNOWN: Self = Self(0b111);

    /// The base's answer `class` itself.
    pub(crate) const fn exactly(class: NodeClass) -> Self {
        Self(bit(class))
    }

    const fn contains(self, class: NodeClass) -> bool {
        self.0 & bit(class) != 0
    }

    /// Removes the classes `evidence` rules out. Never empties the set: `Mixed` is never ruled
    /// out, and an exact answer of the base is kept even if a cell contradicts it (a base
    /// that breaks its contract makes classes wrong, not the bookkeeping inconsistent).
    #[must_use]
    pub(crate) fn narrow(self, evidence: CellEvidence) -> Self {
        let mut ruled_out = 0;
        if evidence.non_air {
            ruled_out |= bit(NodeClass::Empty);
        }
        if evidence.non_opaque {
            ruled_out |= bit(NodeClass::Opaque);
        }
        match self.0 & !ruled_out {
            0 => self,
            narrowed => Self(narrowed),
        }
    }

    /// `f(class)` when it is the same for every class still possible, else `None`.
    pub(crate) fn decide<T: PartialEq>(self, f: impl Fn(NodeClass) -> T) -> Option<T> {
        let mut decided: Option<T> = None;
        for class in [NodeClass::Empty, NodeClass::Opaque, NodeClass::Mixed] {
            if !self.contains(class) {
                continue;
            }
            let value = f(class);
            match &decided {
                None => decided = Some(value),
                Some(previous) if *previous == value => {}
                Some(_) => return None,
            }
        }
        decided
    }
}

impl Default for BaseClasses {
    /// [`BaseClasses::UNKNOWN`].
    fn default() -> Self {
        Self::UNKNOWN
    }
}

/// The class of the composite chunk `key` of `layers` (base ⊕ overrides), exact-conservative.
/// Asks `base` only when neither the chunk's layer nor its memoised base class decides it.
pub(crate) fn composite_class(
    layers: &LayerMap,
    key: LodKey,
    flags: &[BlockFlags],
    base: &dyn BaseSource,
) -> NodeClass {
    let slot = layers.slot(key);
    let layer = slot.and_then(|slot| slot.layer.as_deref());
    let known = slot.map_or(BaseClasses::UNKNOWN, |slot| slot.base_class);
    LayerClass::of(layer, flags).resolve(known, || base.classify(key))
}

#[cfg(test)]
mod tests {
    use super::*;

    const AIR: BlockStateId = BlockStateId::AIR;
    const STONE: BlockStateId = BlockStateId(1);
    const GLASS: BlockStateId = BlockStateId(2);
    const ALL: [NodeClass; 3] = [NodeClass::Empty, NodeClass::Opaque, NodeClass::Mixed];

    fn flags() -> [BlockFlags; 3] {
        [
            BlockFlags::empty(),
            BlockFlags::SOLID | BlockFlags::OPAQUE,
            BlockFlags::SOLID | BlockFlags::CUTOUT,
        ]
    }

    /// The composite class straight from the definition, cell by cell: every cell of the
    /// layer that is UNCHANGED stands for "some cell of a base chunk of class `base`".
    fn reference(layer: &VoxelChunk, base: NodeClass, flags: &[BlockFlags]) -> NodeClass {
        let states: Vec<BlockStateId> = layer.blocks.states().collect();
        let explicit = states.iter().filter(|state| !state.is_unchanged());
        let shows_base = states.iter().any(|state| state.is_unchanged());
        let all_air = explicit.clone().all(|state| state.is_air());
        let all_opaque = explicit
            .clone()
            .all(|state| flags[state.index()].contains(BlockFlags::OPAQUE));
        let empty = all_air && (!shows_base || base == NodeClass::Empty);
        let opaque = all_opaque && (!shows_base || base == NodeClass::Opaque);
        if empty {
            NodeClass::Empty
        } else if opaque {
            NodeClass::Opaque
        } else {
            NodeClass::Mixed
        }
    }

    #[test]
    fn layer_class_matches_the_definition() {
        let flags = flags();
        let mut layers = vec![VoxelChunk::uniform(BlockStateId::UNCHANGED)];
        for states in [
            &[AIR][..],
            &[STONE],
            &[GLASS],
            &[AIR, STONE],
            &[STONE, GLASS],
        ] {
            for full in [false, true] {
                let mut layer = if full {
                    VoxelChunk::uniform(states[0])
                } else {
                    VoxelChunk::uniform(BlockStateId::UNCHANGED)
                };
                for (index, &state) in states.iter().enumerate() {
                    layer.blocks.set(index, state);
                }
                layers.push(layer);
            }
        }
        for layer in &layers {
            let class = LayerClass::of(Some(layer), &flags);
            for base in ALL {
                assert_eq!(
                    class.with_base(base),
                    reference(layer, base, &flags),
                    "{layer:?} over {base:?}"
                );
            }
        }
        for base in ALL {
            assert_eq!(LayerClass::of(None, &flags).with_base(base), base);
            assert_eq!(LayerClass::BASE.with_base(base), base);
        }
        // A layer without UNCHANGED decides alone; the base is not asked.
        let full = LayerClass::of(Some(&VoxelChunk::uniform(STONE)), &flags);
        let never = || panic!("base asked");
        assert_eq!(full.resolve(BaseClasses::UNKNOWN, never), NodeClass::Opaque);
        // A layer with air and stone among UNCHANGED is Mixed whatever the base.
        let mut both = VoxelChunk::uniform(BlockStateId::UNCHANGED);
        both.blocks.set(0, AIR);
        both.blocks.set(1, STONE);
        let class = LayerClass::of(Some(&both), &flags);
        assert_eq!(class.resolve(BaseClasses::UNKNOWN, never), NodeClass::Mixed);
    }

    #[test]
    fn evidence_narrows_and_decides() {
        let flags = flags();
        let air = CellEvidence::of(&[AIR, AIR], &flags);
        let stone = CellEvidence::of(&[STONE], &flags);
        let glass = CellEvidence::of(&[GLASS], &flags);
        assert!(!air.is_conclusive() && !stone.is_conclusive());
        assert!(air.merge(stone).is_conclusive());
        assert!(
            glass.is_conclusive(),
            "one non-air, non-opaque cell is enough"
        );
        let identity = |class: NodeClass| class;
        assert_eq!(BaseClasses::UNKNOWN.decide(identity), None);
        assert_eq!(
            BaseClasses::UNKNOWN
                .narrow(air.merge(stone))
                .decide(identity),
            Some(NodeClass::Mixed)
        );
        let not_opaque = BaseClasses::UNKNOWN.narrow(air);
        assert_eq!(not_opaque.decide(identity), None);
        assert_eq!(
            not_opaque.decide(|class| class == NodeClass::Opaque),
            Some(false)
        );
        for class in ALL {
            assert_eq!(BaseClasses::exactly(class).decide(identity), Some(class));
            // Evidence never empties a set.
            for evidence in [air, stone, glass] {
                let narrowed = BaseClasses::exactly(class).narrow(evidence);
                assert!(
                    narrowed.decide(identity).is_some(),
                    "{class:?} {evidence:?}"
                );
            }
        }
        assert_eq!(BaseClasses::default(), BaseClasses::UNKNOWN);
    }
}
