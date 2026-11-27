//! Parameter sets for the species and ores of the base game.

use crate::params::{ParamValue, Params};
use crate::recipes::Tint;

/// A named set of parameters for a recipe: the colours and variants of one tree species or
/// one ore, chosen to sit well with the rest of the pack.
///
/// Presets are plain data in the text syntax of content packs, so a pack can copy them into
/// its texture definitions verbatim. They are also what the contact sheet shows after the
/// bare recipes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Preset {
    /// Unique name, `<species or metal>_<recipe>`.
    pub name: &'static str,
    /// The recipe the parameters are for.
    pub recipe: &'static str,
    /// `(key, value)` pairs; values in the syntax of [`ParamValue::parse`].
    pub params: &'static [(&'static str, &'static str)],
    /// Tint class the texture is authored for with these parameters. Differs from the
    /// recipe's own class when the preset bakes a colour in (cherry leaves are pink whatever
    /// the biome).
    pub tint: Tint,
}

impl Preset {
    /// The preset's parameters, ready for [`crate::generate`].
    #[must_use]
    pub fn to_params(&self) -> Params {
        let mut params = Params::new();
        for (key, text) in self.params {
            // The table below is checked by the tests, so the fallback is never taken; it only
            // keeps this function free of panics.
            let value = ParamValue::parse(text).unwrap_or(ParamValue::Variant((*text).to_owned()));
            params.set(key, value);
        }
        params
    }
}

const fn preset(
    name: &'static str,
    recipe: &'static str,
    params: &'static [(&'static str, &'static str)],
) -> Preset {
    Preset {
        name,
        recipe,
        params,
        tint: Tint::None,
    }
}

const fn foliage(mut preset: Preset) -> Preset {
    preset.tint = Tint::Foliage;
    preset
}

const fn grass(mut preset: Preset) -> Preset {
    preset.tint = Tint::Grass;
    preset
}

const OAK_BARK: &str = "#70583c";
const OAK_WOOD: &str = "#ba9662";
const BIRCH_BARK: &str = "#e2decc";
const BIRCH_WOOD: &str = "#dcc896";
const SPRUCE_BARK: &str = "#5c4234";
const SPRUCE_WOOD: &str = "#8e6c48";
const CHERRY_BARK: &str = "#6e4842";
const CHERRY_WOOD: &str = "#d89a86";

static PRESETS: [Preset; 24] = [
    preset(
        "oak_log_side",
        "log_side",
        &[("bark", OAK_BARK), ("style", "furrowed")],
    ),
    preset(
        "oak_log_top",
        "log_top",
        &[("bark", OAK_BARK), ("wood", OAK_WOOD)],
    ),
    preset("oak_planks", "planks", &[("wood", OAK_WOOD)]),
    foliage(preset("oak_leaves", "leaves", &[("style", "broad")])),
    preset(
        "birch_log_side",
        "log_side",
        &[
            ("bark", BIRCH_BARK),
            ("marks", "#423c3a"),
            ("style", "birch"),
        ],
    ),
    preset(
        "birch_log_top",
        "log_top",
        &[("bark", BIRCH_BARK), ("wood", BIRCH_WOOD)],
    ),
    preset("birch_planks", "planks", &[("wood", BIRCH_WOOD)]),
    // Lighter and a touch yellow before the tint: birches stand out pale among other trees.
    foliage(preset(
        "birch_leaves",
        "leaves",
        &[("leaf", "#dcdfc4"), ("style", "broad")],
    )),
    preset(
        "spruce_log_side",
        "log_side",
        &[("bark", SPRUCE_BARK), ("style", "scaly")],
    ),
    preset(
        "spruce_log_top",
        "log_top",
        &[("bark", SPRUCE_BARK), ("wood", SPRUCE_WOOD)],
    ),
    preset("spruce_planks", "planks", &[("wood", SPRUCE_WOOD)]),
    // Darker and cooler before the tint: needles end up a deep blue-green.
    foliage(preset(
        "spruce_leaves",
        "leaves",
        &[("leaf", "#a4b2b6"), ("style", "needle")],
    )),
    preset(
        "cherry_log_side",
        "log_side",
        &[
            ("bark", CHERRY_BARK),
            ("marks", "#b88c7c"),
            ("style", "smooth"),
        ],
    ),
    preset(
        "cherry_log_top",
        "log_top",
        &[("bark", CHERRY_BARK), ("wood", CHERRY_WOOD)],
    ),
    preset("cherry_planks", "planks", &[("wood", CHERRY_WOOD)]),
    // Baked pink, not tinted.
    preset(
        "cherry_leaves",
        "leaves",
        &[
            ("leaf", "#e6a0b8"),
            ("blossom", "#fbe0e8"),
            ("style", "broad"),
        ],
    ),
    // The four ores differ in colour and in the shape of their pieces, so that each one can
    // be told from the others at a glance: orange nuggets with green patina, pale beige
    // specks with rust, pale silver-blue crystals, black lumps.
    preset(
        "copper_ore",
        "ore",
        &[
            ("ore", "#d27a4c"),
            ("accent", "#4a8c84"),
            ("style", "nuggets"),
        ],
    ),
    preset(
        "tin_ore",
        "ore",
        &[("ore", "#b0c4da"), ("style", "crystals")],
    ),
    preset(
        "iron_ore",
        "ore",
        &[
            ("ore", "#ccba9c"),
            ("accent", "#965234"),
            ("style", "flecks"),
        ],
    ),
    preset("coal_ore", "ore", &[("ore", "#38363e"), ("style", "lumps")]),
    // Ground cover: turf with flowers, a forest floor and a dry heath. Their grass takes the
    // tint of the place; flowers, fallen leaves and straw keep their own colour.
    grass(preset(
        "flowers_grass_top",
        "grass_top",
        &[("style", "flowers")],
    )),
    grass(preset(
        "blossoms_grass_top",
        "grass_top",
        &[("style", "blossoms")],
    )),
    grass(preset(
        "litter_grass_top",
        "grass_top",
        &[("style", "litter")],
    )),
    grass(preset(
        "straw_grass_top",
        "grass_top",
        &[("style", "straw")],
    )),
];

/// Parameter sets for the base game's four tree species (oak, birch, spruce, cherry: bark,
/// log end, planks, leaves), four ores (copper, tin, iron, coal) and four kinds of ground
/// cover (`grass_top` with flowers, blossoms, fallen leaves or straw).
///
/// ```
/// use cp_proctex::{generate, presets};
///
/// let birch = presets()
///     .iter()
///     .find(|preset| preset.name == "birch_log_side")
///     .expect("a base game preset");
/// let texture = generate(birch.recipe, &birch.to_params(), 1, 16)?;
/// assert_eq!(texture.albedo.len(), 256);
/// # Ok::<(), cp_proctex::ProcTexError>(())
/// ```
#[must_use]
pub fn presets() -> &'static [Preset] {
    &PRESETS
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::recipes::{find, recipe_info};

    #[test]
    fn every_preset_is_valid_for_its_recipe() {
        for (i, preset) in presets().iter().enumerate() {
            let recipe = find(preset.recipe).unwrap_or_else(|| panic!("{}", preset.name));
            assert_eq!(recipe.check(&preset.to_params()), Ok(()), "{}", preset.name);
            assert_eq!(
                preset.to_params().len(),
                preset.params.len(),
                "{}",
                preset.name
            );
            for (_, text) in preset.params {
                assert!(ParamValue::parse(text).is_ok(), "{}: {text}", preset.name);
            }
            assert!(
                presets()[..i].iter().all(|other| other.name != preset.name),
                "duplicate {}",
                preset.name
            );
            assert!(preset.name.ends_with(preset.recipe), "{}", preset.name);
            // A preset is tinted only if its recipe is, and then with the recipe's class.
            let class = recipe_info(preset.recipe).map(|info| info.tint);
            assert!(preset.tint == Tint::None || Some(preset.tint) == class);
        }
    }

    #[test]
    fn the_base_game_species_and_ores_are_covered() {
        let names: Vec<&str> = presets().iter().map(|preset| preset.name).collect();
        for species in ["oak", "birch", "spruce", "cherry"] {
            for part in ["log_side", "log_top", "planks", "leaves"] {
                assert!(names.contains(&format!("{species}_{part}").as_str()));
            }
        }
        for metal in ["copper", "tin", "iron", "coal"] {
            assert!(names.contains(&format!("{metal}_ore").as_str()));
        }
        // Cherry leaves are baked, the others take the foliage tint.
        let tint = |name: &str| presets().iter().find(|p| p.name == name).map(|p| p.tint);
        assert_eq!(tint("cherry_leaves"), Some(Tint::None));
        assert_eq!(tint("oak_leaves"), Some(Tint::Foliage));
        assert_eq!(tint("spruce_leaves"), Some(Tint::Foliage));
    }

    #[test]
    fn presets_convert_to_typed_parameters() {
        let birch = presets()
            .iter()
            .find(|preset| preset.name == "birch_log_side")
            .expect("preset exists");
        let params = birch.to_params();
        assert_eq!(params.color("bark"), Some([0xe2, 0xde, 0xcc]));
        assert_eq!(params.variant("style"), Some("birch"));
    }
}
