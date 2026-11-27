//! Caller-chosen colours and variants of a recipe.

use std::collections::BTreeMap;

use crate::error::ProcTexError;

/// One parameter value.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ParamValue {
    /// An sRGB colour, 8 bits per channel.
    Color([u8; 3]),
    /// The name of a structural variant, for example a bark style.
    Variant(String),
}

impl ParamValue {
    /// Reads a value from text, as content packs write it: `#rrggbb` or `#rgb` (hexadecimal,
    /// either case) is a colour, any text that does not start with `#` names a variant.
    ///
    /// # Errors
    /// [`ProcTexError::InvalidColor`] for text that starts with `#` but is not a colour.
    pub fn parse(text: &str) -> Result<Self, ProcTexError> {
        let Some(digits) = text.strip_prefix('#') else {
            return Ok(Self::Variant(text.to_owned()));
        };
        let invalid = || ProcTexError::InvalidColor {
            text: text.to_owned(),
        };
        let nibbles = digits
            .chars()
            .map(|c| c.to_digit(16).map(|d| d as u8))
            .collect::<Option<Vec<u8>>>()
            .ok_or_else(invalid)?;
        match nibbles[..] {
            [r, g, b] => Ok(Self::Color([r * 17, g * 17, b * 17])),
            [r1, r0, g1, g0, b1, b0] => Ok(Self::Color([
                (r1 << 4) | r0,
                (g1 << 4) | g0,
                (b1 << 4) | b0,
            ])),
            _ => Err(invalid()),
        }
    }
}

/// The colours and variants a caller picks for a recipe: bark and leaf colours per tree
/// species, the colour of an ore, the style of a bark, and so on.
///
/// Parameters are addressed by key; every recipe documents its keys through
/// [`crate::recipe_info`], and a parameter that is not set keeps the recipe's default.
/// [`crate::generate`] rejects keys the recipe does not know, values of the wrong kind and
/// unknown variant names, so a typo in a content pack never goes unnoticed.
///
/// ```
/// use cp_proctex::{ParamValue, Params};
///
/// // Typed, for code ...
/// let birch = Params::new()
///     .with_color("bark", [226, 222, 208])
///     .with_variant("style", "birch");
/// // ... or from text, for content packs.
/// let mut from_text = Params::new();
/// from_text.set_text("bark", "#e2ded0")?;
/// from_text.set_text("style", "birch")?;
/// assert_eq!(birch, from_text);
/// assert_eq!(birch.color("bark"), Some([226, 222, 208]));
/// assert_eq!(birch.get("style"), Some(&ParamValue::Variant("birch".into())));
/// # Ok::<(), cp_proctex::ProcTexError>(())
/// ```
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Params {
    // Ordered, so that validation reports the same first error on every run.
    values: BTreeMap<String, ParamValue>,
}

impl Params {
    /// No parameters: the recipe's defaults.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds or replaces the colour `key` (sRGB).
    #[must_use]
    pub fn with_color(mut self, key: &str, srgb: [u8; 3]) -> Self {
        self.set(key, ParamValue::Color(srgb));
        self
    }

    /// Adds or replaces the variant `key`.
    #[must_use]
    pub fn with_variant(mut self, key: &str, name: &str) -> Self {
        self.set(key, ParamValue::Variant(name.to_owned()));
        self
    }

    /// Sets `key` and returns the value it had before, if any.
    pub fn set(&mut self, key: &str, value: ParamValue) -> Option<ParamValue> {
        self.values.insert(key.to_owned(), value)
    }

    /// Sets `key` from text; see [`ParamValue::parse`] for the syntax.
    ///
    /// # Errors
    /// [`ProcTexError::InvalidColor`] if the text starts with `#` but is not a colour.
    pub fn set_text(&mut self, key: &str, text: &str) -> Result<(), ProcTexError> {
        self.set(key, ParamValue::parse(text)?);
        Ok(())
    }

    /// The value of `key`, if set.
    #[must_use]
    pub fn get(&self, key: &str) -> Option<&ParamValue> {
        self.values.get(key)
    }

    /// The colour `key`, if it is set to a colour.
    #[must_use]
    pub fn color(&self, key: &str) -> Option<[u8; 3]> {
        match self.values.get(key) {
            Some(&ParamValue::Color(srgb)) => Some(srgb),
            _ => None,
        }
    }

    /// The variant `key`, if it is set to a variant name.
    #[must_use]
    pub fn variant(&self, key: &str) -> Option<&str> {
        match self.values.get(key) {
            Some(ParamValue::Variant(name)) => Some(name),
            _ => None,
        }
    }

    /// All parameters in key order.
    pub fn iter(&self) -> impl Iterator<Item = (&str, &ParamValue)> {
        self.values.iter().map(|(key, value)| (key.as_str(), value))
    }

    /// Number of parameters that are set.
    #[must_use]
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// True when no parameter is set.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn colours_parse_in_both_notations() {
        assert_eq!(
            ParamValue::parse("#8a5a2b"),
            Ok(ParamValue::Color([0x8a, 0x5a, 0x2b]))
        );
        assert_eq!(
            ParamValue::parse("#FFffFF"),
            Ok(ParamValue::Color([255, 255, 255]))
        );
        assert_eq!(
            ParamValue::parse("#000000"),
            Ok(ParamValue::Color([0, 0, 0]))
        );
        assert_eq!(
            ParamValue::parse("#f80"),
            Ok(ParamValue::Color([255, 136, 0]))
        );
        assert_eq!(
            ParamValue::parse("birch"),
            Ok(ParamValue::Variant("birch".into()))
        );
        assert_eq!(
            ParamValue::parse(""),
            Ok(ParamValue::Variant(String::new()))
        );
    }

    #[test]
    fn malformed_colours_are_errors() {
        for text in [
            "#", "#12", "#1234", "#12345", "#1234567", "#gggggg", "#12 456", "#ééé",
        ] {
            assert_eq!(
                ParamValue::parse(text),
                Err(ProcTexError::InvalidColor { text: text.into() }),
                "{text}"
            );
        }
        let mut params = Params::new();
        assert!(params.set_text("bark", "#nothex").is_err());
        assert!(params.is_empty());
    }

    #[test]
    fn params_store_typed_values_by_key() {
        let mut params = Params::new()
            .with_color("ore", [1, 2, 3])
            .with_variant("style", "lumps");
        assert_eq!(params.len(), 2);
        assert!(!params.is_empty());
        assert_eq!(params.color("ore"), Some([1, 2, 3]));
        assert_eq!(params.variant("style"), Some("lumps"));
        // The typed getters do not coerce.
        assert_eq!(params.color("style"), None);
        assert_eq!(params.variant("ore"), None);
        assert_eq!(params.get("missing"), None);
        // Setting again replaces and reports the old value.
        let old = params.set("ore", ParamValue::Color([9, 9, 9]));
        assert_eq!(old, Some(ParamValue::Color([1, 2, 3])));
        assert_eq!(params.color("ore"), Some([9, 9, 9]));
        // Iteration is in key order, whatever the insertion order.
        let other = Params::new()
            .with_variant("style", "lumps")
            .with_color("ore", [9, 9, 9]);
        assert_eq!(params, other);
        let keys: Vec<&str> = params.iter().map(|(key, _)| key).collect();
        assert_eq!(keys, ["ore", "style"]);
        assert_eq!(Params::new(), Params::default());
    }
}
