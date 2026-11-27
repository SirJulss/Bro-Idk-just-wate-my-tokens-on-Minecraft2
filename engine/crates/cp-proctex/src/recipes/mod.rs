//! The recipes: one painter per material, and the table that names and documents them.
//!
//! # Shared rules of the pack
//! - Light comes from the upper left. Rims that face it are a tone lighter, rims that face
//!   away a tone darker, and the darkest tone is kept for gaps and joints.
//! - A recipe composes its tones from a few independent layers (light patches, dark patches,
//!   placed details). Thresholding a single noise field into all tones would nest them like
//!   contour lines, which is the look of generated textures.
//! - Shares of tones are fixed by rank, so every seed has the same balance of light and dark.
//! - Small details are placed by hand on the 16×16 design grid, spaced apart, never sprinkled.

mod fluid;
mod foliage;
mod ground;
mod metal;
mod rock;
mod wood;

use core::f64::consts::FRAC_1_SQRT_2;

use cp_core::hash::hash_combine;
use cp_core::math::DVec2;

use crate::canvas::Canvas;
use crate::color::Rgb8;
use crate::error::ProcTexError;
use crate::grid::Grid;
use crate::mip::AlphaMode;
use crate::paint::Painting;
use crate::params::{ParamValue, Params};
use crate::pattern::{Cell, DESIGN, unit};

/// The tint class a texture is authored for (mirrors the material tint classes of the content
/// registry).
///
/// Textures of a class other than [`Tint::None`] are painted in near-neutral, light tones and
/// only become grass, foliage or water colour when the renderer multiplies the biome's tint
/// into them; see [`crate::tint_weight`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Tint {
    /// Final colours; never tinted.
    None,
    /// Multiplied by the biome's grass colour.
    Grass,
    /// Multiplied by the biome's foliage colour.
    Foliage,
    /// Multiplied by the water colour.
    Water,
}

impl Tint {
    /// A representative sRGB tint for the class: what the contact sheet previews with, and a
    /// sensible stand-in until biomes provide their own. White for [`Tint::None`].
    #[must_use]
    pub const fn reference_color(self) -> [u8; 3] {
        match self {
            Self::None => [255, 255, 255],
            Self::Grass => [126, 188, 84],
            Self::Foliage => [92, 160, 66],
            Self::Water => [72, 138, 214],
        }
    }
}

/// What kind of value a parameter takes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ParamKind {
    /// An sRGB colour ([`ParamValue::Color`]).
    Color,
    /// One of the listed variant names ([`ParamValue::Variant`]); the first is the default.
    Variant(&'static [&'static str]),
}

/// Description of one parameter of a recipe.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ParamInfo {
    /// Key to use in [`Params`].
    pub key: &'static str,
    /// Kind of value the key takes.
    pub kind: ParamKind,
    /// What the parameter changes, and its default.
    pub summary: &'static str,
}

/// Description of one recipe.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RecipeInfo {
    /// Name to pass to [`crate::generate`].
    pub name: &'static str,
    /// One line on what the texture shows.
    pub summary: &'static str,
    /// Tint class the default colours are authored for. With [`Tint::None`] the colours are
    /// final; otherwise the tinted parts are near-neutral until a caller overrides the colour.
    pub tint: Tint,
    /// True when the texture has holes (alpha 0) and belongs on cutout geometry.
    pub cutout: bool,
    /// The parameters the recipe accepts.
    pub params: &'static [ParamInfo],
}

impl RecipeInfo {
    /// What the alpha of the recipe's textures means: [`AlphaMode::Coverage`] for cutout
    /// recipes, [`AlphaMode::Opaque`] for all others. The mode to build their
    /// [`crate::mip_chain`] with.
    #[must_use]
    pub const fn alpha_mode(&self) -> AlphaMode {
        if self.cutout {
            AlphaMode::Coverage
        } else {
            AlphaMode::Opaque
        }
    }
}

/// A recipe: its description and its painter.
pub(crate) struct Recipe {
    pub info: RecipeInfo,
    pub paint: fn(&Canvas, &Params) -> Painting,
}

impl Recipe {
    /// Rejects parameters the recipe does not know or cannot read.
    pub(crate) fn check(&self, params: &Params) -> Result<(), ProcTexError> {
        for (key, value) in params.iter() {
            let recipe = || self.info.name.to_owned();
            let Some(info) = self.info.params.iter().find(|info| info.key == key) else {
                return Err(ProcTexError::UnknownParam {
                    recipe: recipe(),
                    key: key.to_owned(),
                });
            };
            match (info.kind, value) {
                (ParamKind::Color, ParamValue::Color(_)) => {}
                (ParamKind::Variant(names), ParamValue::Variant(name)) => {
                    if !names.contains(&name.as_str()) {
                        return Err(ProcTexError::UnknownVariant {
                            recipe: recipe(),
                            key: key.to_owned(),
                            value: name.clone(),
                        });
                    }
                }
                (kind, _) => {
                    return Err(ProcTexError::WrongParamKind {
                        recipe: recipe(),
                        key: key.to_owned(),
                        expected: match kind {
                            ParamKind::Color => "colour",
                            ParamKind::Variant(_) => "variant name",
                        },
                    });
                }
            }
        }
        Ok(())
    }
}

const fn color_param(key: &'static str, summary: &'static str) -> ParamInfo {
    ParamInfo {
        key,
        kind: ParamKind::Color,
        summary,
    }
}

const fn recipe(
    name: &'static str,
    summary: &'static str,
    params: &'static [ParamInfo],
    paint: fn(&Canvas, &Params) -> Painting,
) -> Recipe {
    Recipe {
        info: RecipeInfo {
            name,
            summary,
            tint: Tint::None,
            cutout: false,
            params,
        },
        paint,
    }
}

const fn tinted(mut recipe: Recipe, tint: Tint) -> Recipe {
    recipe.info.tint = tint;
    recipe
}

const fn cutout(mut recipe: Recipe) -> Recipe {
    recipe.info.cutout = true;
    recipe
}

/// Names of all recipes, in the order of [`RECIPES`].
static NAMES: [&str; RECIPE_COUNT] = {
    let mut names = [""; RECIPE_COUNT];
    let mut i = 0;
    while i < RECIPE_COUNT {
        names[i] = RECIPES[i].info.name;
        i += 1;
    }
    names
};

const RECIPE_COUNT: usize = 22;

static RECIPES: [Recipe; RECIPE_COUNT] = [
    recipe(
        "stone",
        "Bedrock in flat, layered blotches",
        &[color_param("color", "rock colour; default warm grey")],
        rock::stone,
    ),
    recipe(
        "cobble",
        "Rounded stones set in dark gaps",
        &[color_param("color", "stone colour; default warm grey")],
        rock::cobble,
    ),
    recipe(
        "gravel",
        "Loose pebbles of mixed tones",
        &[color_param("color", "pebble colour; default grey-brown")],
        rock::gravel,
    ),
    recipe(
        "dirt",
        "Crumbly soil with a few pebbles",
        &[color_param("color", "soil colour; default warm brown")],
        ground::dirt,
    ),
    tinted(
        recipe(
            "grass_top",
            "Turf seen from above, near-neutral for the grass tint",
            &[
                color_param(
                    "grass",
                    "blade colour; default near-neutral (set it to bake a colour in)",
                ),
                ParamInfo {
                    key: "style",
                    kind: ParamKind::Variant(&["turf", "flowers", "blossoms", "litter", "straw"]),
                    summary: "turf: grass only (default); flowers: with yellow and red flowers;                               blossoms: with violet and pink flowers; litter: a forest floor of                               fallen leaves and twigs; straw: a dry heath of straw-coloured                               blades. Flowers, leaves and straw keep their colour under the tint",
                },
            ],
            ground::grass_top,
        ),
        Tint::Grass,
    ),
    tinted(
        recipe(
            "grass_side",
            "Soil of the same seed's `dirt` under a ragged, near-neutral grass fringe",
            &[
                color_param(
                    "grass",
                    "fringe colour; default near-neutral (set it to bake a colour in)",
                ),
                color_param("dirt", "soil colour; default as `dirt`"),
            ],
            ground::grass_side,
        ),
        Tint::Grass,
    ),
    recipe(
        "sand",
        "Pale sand with wind ripples and single grains",
        &[color_param("color", "sand colour; default pale yellow")],
        ground::sand,
    ),
    recipe(
        "sandstone",
        "Compacted sand in wavy strata",
        &[color_param("color", "stone colour; default ochre")],
        rock::sandstone,
    ),
    recipe(
        "snow",
        "Soft drifts with blue shadows and a few sparkles",
        &[color_param("color", "snow colour; default cool white")],
        ground::snow,
    ),
    recipe(
        "ice",
        "Glossy ice with streaks of reflected light and hairline cracks",
        &[color_param("color", "ice colour; default light blue")],
        ground::ice,
    ),
    recipe(
        "clay",
        "Smooth, damp clay in wide smears",
        &[color_param("color", "clay colour; default blue-grey")],
        rock::clay,
    ),
    recipe(
        "basalt",
        "Dark volcanic rock broken into polygonal columns",
        &[color_param("color", "rock colour; default charcoal")],
        rock::basalt,
    ),
    recipe(
        "limestone",
        "Pale, pitted sedimentary rock",
        &[color_param("color", "rock colour; default cream")],
        rock::limestone,
    ),
    recipe(
        "slate",
        "Dark rock splitting into thin horizontal sheets",
        &[color_param("color", "rock colour; default blue-grey")],
        rock::slate,
    ),
    recipe(
        "log_side",
        "Bark of a log, in one of four styles",
        &[
            color_param("bark", "bark colour; default oak brown"),
            color_param(
                "marks",
                "colour of the marks of the `birch` and `smooth` styles",
            ),
            ParamInfo {
                key: "style",
                kind: ParamKind::Variant(&["furrowed", "birch", "scaly", "smooth"]),
                summary: "furrowed: vertical ridges (oak); birch: pale with dark dashes; \
                          scaly: plates (spruce); smooth: banded with light dashes (cherry)",
            },
        ],
        wood::log_side,
    ),
    recipe(
        "log_top",
        "Cut end of a log: growth rings inside a ring of bark",
        &[
            color_param("bark", "bark colour; default oak brown"),
            color_param("wood", "wood colour; default oak"),
        ],
        wood::log_top,
    ),
    recipe(
        "planks",
        "Four boards with staggered butt joints",
        &[color_param("wood", "wood colour; default oak")],
        wood::planks,
    ),
    cutout(tinted(
        recipe(
            "leaves",
            "Clumps of foliage with gaps, near-neutral for the foliage tint",
            &[
                color_param(
                    "leaf",
                    "leaf colour; default near-neutral (set it to bake a colour in)",
                ),
                color_param("blossom", "adds blossoms of this colour; default none"),
                ParamInfo {
                    key: "style",
                    kind: ParamKind::Variant(&["broad", "needle"]),
                    summary: "broad: round clumps; needle: dense, narrow hanging tufts",
                },
            ],
            foliage::leaves,
        ),
        Tint::Foliage,
    )),
    tinted(
        recipe(
            "water",
            "Calm water with thin glints, near-neutral for the water tint",
            &[color_param(
                "color",
                "water colour; default near-neutral (set it to bake a colour in)",
            )],
            fluid::water,
        ),
        Tint::Water,
    ),
    recipe(
        "lava",
        "Molten rock: glowing flows between patches of dark crust (emissive)",
        &[color_param("color", "colour of the glow; default orange")],
        fluid::lava,
    ),
    recipe(
        "ore",
        "Pieces of ore embedded in the same seed's `stone`",
        &[
            color_param("ore", "colour of the pieces; default pale iron"),
            color_param(
                "accent",
                "patina or rust on the shaded rims of the pieces and on the rock right below \
                 them; default none (a shade of `ore` and the rock's own shadow)",
            ),
            color_param("rock", "host rock colour; default as `stone`"),
            ParamInfo {
                key: "style",
                kind: ParamKind::Variant(&["nuggets", "lumps", "flecks", "crystals"]),
                summary: "nuggets: a few shiny pieces (copper); lumps: large dull chunks \
                          (coal); flecks: many small specks (iron); crystals: small glinting \
                          facets (tin)",
            },
        ],
        rock::ore,
    ),
    recipe(
        "metal",
        "Riveted plate with a bevelled edge",
        &[color_param("color", "metal colour; default steel")],
        metal::metal,
    ),
];

/// Names of all recipes.
///
/// ```
/// assert!(cp_proctex::recipes().contains(&"grass_top"));
/// ```
#[must_use]
pub fn recipes() -> &'static [&'static str] {
    &NAMES
}

/// Description of the recipe `name`: what it shows, whether it is tinted or cut out, and the
/// parameters it accepts. `None` for unknown names.
///
/// ```
/// use cp_proctex::{Tint, recipe_info};
///
/// let leaves = recipe_info("leaves").expect("a contract recipe");
/// assert!(leaves.cutout);
/// assert_eq!(leaves.tint, Tint::Foliage);
/// assert!(leaves.params.iter().any(|param| param.key == "leaf"));
/// assert!(recipe_info("granite").is_none());
/// ```
#[must_use]
pub fn recipe_info(name: &str) -> Option<&'static RecipeInfo> {
    find(name).map(|recipe| &recipe.info)
}

pub(crate) fn find(name: &str) -> Option<&'static Recipe> {
    RECIPES.iter().find(|recipe| recipe.info.name == name)
}

// ---------------------------------------------------------------------------------------------
// Helpers shared by the painters
// ---------------------------------------------------------------------------------------------

/// Direction towards the light in image space (x right, y down): the upper left.
const LIGHT: DVec2 = DVec2::new(-FRAC_1_SQRT_2, -FRAC_1_SQRT_2);

/// The colour parameter `key`, or the recipe's default.
fn color(params: &Params, key: &str, default: Rgb8) -> Rgb8 {
    params.color(key).unwrap_or(default)
}

/// Tone step for the rim of a cell: +1 where the rim faces the light, -1 where it faces away,
/// 0 inside the cell (further than `rim` from the border) and on the flanks.
fn bevel(cell: &Cell, rim: f64) -> i32 {
    if cell.edge >= rim {
        return 0;
    }
    let facing = cell.outward.dot(LIGHT);
    if facing > 0.3 {
        1
    } else if facing < -0.3 {
        -1
    } else {
        0
    }
}

/// A yes/no decision that is fixed for a cell: true with the given probability.
fn chance(hash: u64, salt: u64, probability: f64) -> bool {
    unit(hash_combine(hash, salt)) < probability
}

/// How far a lip stands out of the surface around it, as a share of the height range.
///
/// A light detail painted over its own shadow (the ledges of `slate`, the dashes of smooth
/// bark, the pebbles of `dirt`) is a lip whose crest lies between the two: raising both
/// texels tilts the light one towards the light and the shadow away from it. Heights taken
/// from the tones would make the shadow a pit and tilt the highlight away from the light.
const LIP: f64 = 0.35;

/// `height` with the texels for which `raised(x, y)` holds lifted by [`LIP`].
fn lift(height: &Grid<f64>, raised: impl Fn(usize, usize) -> bool) -> Grid<f64> {
    Grid::from_fn(height.size(), |x, y| {
        height.get(x, y) + if raised(x, y) { LIP } else { 0.0 }
    })
}

/// Height field that follows the tones: every tone has a level, softened by [`soften`] so
/// that the normal map gets slopes instead of cliffs.
fn heights(index: &Grid<u8>, levels: &[f64]) -> Grid<f64> {
    let last = levels.len().saturating_sub(1);
    soften(&index.map(|tone| levels[usize::from(tone).min(last)]))
}

/// Blurs a height field painted on the canvas with a tent that reaches two design units to
/// each side. At the native 16 px that is exactly [`Grid::blurred`]; at larger sizes it is the
/// same kernel measured in design units, so a softened step has the same slope at every size.
fn soften(height: &Grid<f64>) -> Grid<f64> {
    match (height.size() / DESIGN).max(1) {
        1 => height.blurred(),
        per_unit => height.tent_blurred(2 * per_unit),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_table_holds_the_contract_recipes() {
        let contract = [
            "stone",
            "cobble",
            "gravel",
            "dirt",
            "grass_top",
            "grass_side",
            "sand",
            "sandstone",
            "snow",
            "ice",
            "clay",
            "basalt",
            "limestone",
            "slate",
            "log_side",
            "log_top",
            "planks",
            "leaves",
            "water",
            "lava",
            "ore",
            "metal",
        ];
        assert_eq!(recipes(), contract);
        for name in contract {
            let info = recipe_info(name).expect("listed recipe");
            assert_eq!(info.name, name);
            assert!(!info.summary.is_empty());
            assert!(!info.params.is_empty(), "{name} takes no colour");
            for (i, param) in info.params.iter().enumerate() {
                assert!(!param.summary.is_empty());
                assert!(
                    info.params[..i].iter().all(|other| other.key != param.key),
                    "{name}: duplicate key {}",
                    param.key
                );
                if let ParamKind::Variant(names) = param.kind {
                    assert!(names.len() >= 2, "{name}.{}", param.key);
                }
            }
        }
        assert!(recipe_info("Stone").is_none());
        assert!(recipe_info("").is_none());
    }

    #[test]
    fn tint_classes_and_cutout_are_declared() {
        let tinted: Vec<(&str, Tint)> = recipes()
            .iter()
            .filter_map(|name| {
                let info = recipe_info(name)?;
                (info.tint != Tint::None).then_some((*name, info.tint))
            })
            .collect();
        assert_eq!(
            tinted,
            [
                ("grass_top", Tint::Grass),
                ("grass_side", Tint::Grass),
                ("leaves", Tint::Foliage),
                ("water", Tint::Water),
            ]
        );
        let cutouts: Vec<&str> = recipes()
            .iter()
            .copied()
            .filter(|name| recipe_info(name).is_some_and(|info| info.cutout))
            .collect();
        assert_eq!(cutouts, ["leaves"]);
        for name in recipes() {
            let info = recipe_info(name).expect("listed recipe");
            let expected = if info.cutout {
                AlphaMode::Coverage
            } else {
                AlphaMode::Opaque
            };
            assert_eq!(info.alpha_mode(), expected, "{name}");
        }
        assert_eq!(Tint::None.reference_color(), [255, 255, 255]);
        for tint in [Tint::Grass, Tint::Foliage, Tint::Water] {
            let [r, g, b] = tint.reference_color();
            assert!(r != g || g != b);
        }
    }

    #[test]
    fn parameters_are_validated_against_the_table() {
        let ore = find("ore").expect("ore exists");
        assert_eq!(ore.check(&Params::new()), Ok(()));
        let good = Params::new()
            .with_color("ore", [1, 2, 3])
            .with_color("rock", [4, 5, 6])
            .with_variant("style", "lumps");
        assert_eq!(ore.check(&good), Ok(()));
        assert_eq!(
            ore.check(&Params::new().with_color("colour", [1, 2, 3])),
            Err(ProcTexError::UnknownParam {
                recipe: "ore".into(),
                key: "colour".into()
            })
        );
        assert_eq!(
            ore.check(&Params::new().with_variant("style", "blobs")),
            Err(ProcTexError::UnknownVariant {
                recipe: "ore".into(),
                key: "style".into(),
                value: "blobs".into()
            })
        );
        assert_eq!(
            ore.check(&Params::new().with_color("style", [1, 2, 3])),
            Err(ProcTexError::WrongParamKind {
                recipe: "ore".into(),
                key: "style".into(),
                expected: "variant name"
            })
        );
        assert_eq!(
            ore.check(&Params::new().with_variant("ore", "red")),
            Err(ProcTexError::WrongParamKind {
                recipe: "ore".into(),
                key: "ore".into(),
                expected: "colour"
            })
        );
    }

    #[test]
    fn bevels_face_the_light() {
        let cell = |outward: DVec2, edge: f64| Cell {
            hash: 0,
            edge,
            outward,
        };
        // Border towards the upper left, close by: lit rim.
        assert_eq!(bevel(&cell(DVec2::new(-1.0, 0.0), 0.5), 1.0), 1);
        assert_eq!(bevel(&cell(DVec2::new(0.0, -1.0), 0.5), 1.0), 1);
        // Border towards the lower right: shaded rim.
        assert_eq!(bevel(&cell(DVec2::new(1.0, 0.0), 0.5), 1.0), -1);
        assert_eq!(bevel(&cell(DVec2::new(0.0, 1.0), 0.5), 1.0), -1);
        // Flanks and the interior stay as they are.
        assert_eq!(
            bevel(&cell(DVec2::new(FRAC_1_SQRT_2, -FRAC_1_SQRT_2), 0.5), 1.0),
            0
        );
        assert_eq!(bevel(&cell(DVec2::new(-1.0, 0.0), 1.5), 1.0), 0);
    }

    #[test]
    fn softening_is_measured_in_design_units() {
        let step = |size: usize| Grid::from_fn(size, |x, _| if x * 2 >= size { 1.0 } else { 0.0 });
        assert_eq!(soften(&step(16)), step(16).blurred());
        for size in [32, 64, 256] {
            let soft = soften(&step(size));
            let per_unit = (size / DESIGN) as f64;
            // The step in the middle becomes a ramp of three to four units at every size ...
            let ramp = (size / 4..3 * size / 4)
                .filter(|&x| (1.0e-9..1.0 - 1.0e-9).contains(&soft.get(x, 0)))
                .count();
            assert!(
                (3.0..=4.0).contains(&(ramp as f64 / per_unit)),
                "{size}: {ramp}"
            );
            // ... that never rises by more than half the step per unit.
            for x in 1..size - 1 {
                let slope = (soft.get(x + 1, 0) - soft.get(x - 1, 0)) / 2.0 * per_unit;
                assert!(slope.abs() <= 0.5 + 1.0e-9, "{size}: {slope}");
            }
        }
    }

    #[test]
    fn heights_follow_tones() {
        let index = Grid::from_fn(8, |x, _| u8::from(x >= 4));
        let height = heights(&index, &[0.2, 0.8]);
        assert!(height.get(1, 0) < height.get(6, 0));
        assert!((height.get(1, 3) - 0.2).abs() < 1.0e-12);
        assert!((height.get(6, 3) - 0.8).abs() < 1.0e-12);
        // Tones beyond the table use its last level.
        let flat = heights(&Grid::filled(4, 9), &[0.1, 0.4]);
        assert!(flat.cells().iter().all(|&h| (h - 0.4).abs() < 1.0e-12));
        assert!(chance(1, 2, 1.0) && !chance(1, 2, 0.0));
    }
}
