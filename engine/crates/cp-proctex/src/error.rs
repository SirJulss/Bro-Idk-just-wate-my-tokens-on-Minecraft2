//! The error type of the crate.

use core::fmt;

/// Why a texture request was rejected. Requests come from content packs, so every variant
/// names what to fix.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ProcTexError {
    /// No recipe has this name; [`crate::recipes`] lists the valid ones.
    UnknownRecipe {
        /// The name that was asked for.
        recipe: String,
    },
    /// The edge length is not a power of two between [`crate::MIN_SIZE`] and
    /// [`crate::MAX_SIZE`].
    UnsupportedSize {
        /// The edge length that was asked for, in texels.
        size: u32,
    },
    /// The recipe has no parameter with this key; [`crate::recipe_info`] lists the valid ones.
    UnknownParam {
        /// Recipe the parameters were meant for.
        recipe: String,
        /// The key the recipe does not know.
        key: String,
    },
    /// The parameter exists, but takes the other kind of value (colour or variant).
    WrongParamKind {
        /// Recipe the parameters were meant for.
        recipe: String,
        /// Key of the offending parameter.
        key: String,
        /// The kind of value the recipe expects for this key.
        expected: &'static str,
    },
    /// The variant parameter exists, but has no variant of this name.
    UnknownVariant {
        /// Recipe the parameters were meant for.
        recipe: String,
        /// Key of the offending parameter.
        key: String,
        /// The variant name the recipe does not know.
        value: String,
    },
    /// A colour was not written as `#rrggbb` or `#rgb`.
    InvalidColor {
        /// The text that could not be read.
        text: String,
    },
}

impl fmt::Display for ProcTexError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnknownRecipe { recipe } => write!(f, "unknown texture recipe `{recipe}`"),
            Self::UnsupportedSize { size } => write!(
                f,
                "unsupported texture size {size}: must be a power of two from {} to {}",
                crate::MIN_SIZE,
                crate::MAX_SIZE
            ),
            Self::UnknownParam { recipe, key } => {
                write!(f, "recipe `{recipe}` has no parameter `{key}`")
            }
            Self::WrongParamKind {
                recipe,
                key,
                expected,
            } => write!(
                f,
                "parameter `{key}` of recipe `{recipe}` must be a {expected}"
            ),
            Self::UnknownVariant { recipe, key, value } => write!(
                f,
                "parameter `{key}` of recipe `{recipe}` has no variant `{value}`"
            ),
            Self::InvalidColor { text } => {
                write!(f, "`{text}` is not a colour (expected #rrggbb or #rgb)")
            }
        }
    }
}

impl core::error::Error for ProcTexError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn messages_name_what_to_fix() {
        let cases = [
            (
                ProcTexError::UnknownRecipe {
                    recipe: "granite".into(),
                },
                "unknown texture recipe `granite`",
            ),
            (
                ProcTexError::UnsupportedSize { size: 12 },
                "unsupported texture size 12: must be a power of two from 16 to 256",
            ),
            (
                ProcTexError::UnknownParam {
                    recipe: "ore".into(),
                    key: "colour".into(),
                },
                "recipe `ore` has no parameter `colour`",
            ),
            (
                ProcTexError::WrongParamKind {
                    recipe: "ore".into(),
                    key: "style".into(),
                    expected: "variant name",
                },
                "parameter `style` of recipe `ore` must be a variant name",
            ),
            (
                ProcTexError::UnknownVariant {
                    recipe: "ore".into(),
                    key: "style".into(),
                    value: "blobs".into(),
                },
                "parameter `style` of recipe `ore` has no variant `blobs`",
            ),
            (
                ProcTexError::InvalidColor { text: "#12".into() },
                "`#12` is not a colour (expected #rrggbb or #rgb)",
            ),
        ];
        for (error, message) in cases {
            assert_eq!(error.to_string(), message);
            let as_dyn: &dyn core::error::Error = &error;
            assert!(as_dyn.source().is_none());
        }
    }
}
