//! Key grammar (contract §4.4).
//!
//! Every definition is named by a key `namespace:name`. A block *state* is named by its block's
//! key followed by the state's properties: `namespace:name[prop=value,…]`. The grammar is fixed
//! forever once saves exist:
//!
//! - namespace and name are non-empty and made of `a-z`, `0-9` and `_`;
//! - property names use the same alphabet, values `a-z`, `0-9`, `_`, `+` and `-`;
//! - properties are sorted by name and each is named once;
//! - only values that differ from the property's default are written, so the bare block key is
//!   the default state and an empty list `[]` never occurs.
//!
//! The narrow alphabet keeps keys unambiguous in save palettes and file names, and makes their
//! byte order the same as their alphabetical order. Every state has exactly one key.
//!
//! [`StateKey::parse`] checks the syntax. Which properties and values a block accepts depends on
//! its orientation kind ([`Orient`](crate::Orient)), so whether a well-formed key names a state
//! is decided by [`Registry::state`](crate::Registry::state).

use core::fmt;

use thiserror::Error;

/// The key of the empty block. Its default state is always
/// [`BlockStateId::AIR`](cp_core::ids::BlockStateId::AIR).
pub(crate) const AIR_KEY: &str = "cubeplanet:air";

/// True for a non-empty string of `a-z`, `0-9` and `_`: a namespace, a name or a property name.
pub(crate) fn is_ident(text: &str) -> bool {
    !text.is_empty()
        && text
            .bytes()
            .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'_')
}

/// True for a non-empty string of `a-z`, `0-9`, `_`, `+` and `-`: a property value.
fn is_value(text: &str) -> bool {
    !text.is_empty()
        && text.bytes().all(|byte| {
            byte.is_ascii_lowercase() || byte.is_ascii_digit() || matches!(byte, b'_' | b'+' | b'-')
        })
}

/// True for `namespace:name`.
fn is_block_key(key: &str) -> bool {
    key.split_once(':')
        .is_some_and(|(namespace, name)| is_ident(namespace) && is_ident(name))
}

/// Checks that `key` has the form `namespace:name` (a definition key, without properties); the
/// error is a message for the pack author.
pub(crate) fn check_key(key: &str) -> Result<(), String> {
    if is_block_key(key) {
        Ok(())
    } else {
        Err(KeyError::Block {
            key: key.to_owned(),
        }
        .to_string())
    }
}

/// Writes `block[name=value,…]`, or the bare `block` when there are no properties.
fn write_key(out: &mut impl fmt::Write, block: &str, properties: &[(&str, &str)]) -> fmt::Result {
    out.write_str(block)?;
    for (index, (name, value)) in properties.iter().enumerate() {
        out.write_str(if index == 0 { "[" } else { "," })?;
        out.write_str(name)?;
        out.write_char('=')?;
        out.write_str(value)?;
    }
    if !properties.is_empty() {
        out.write_char(']')?;
    }
    Ok(())
}

/// Formats the key of a state. `properties` must be sorted by name, each named once.
pub(crate) fn format_state_key(block: &str, properties: &[(&str, &str)]) -> String {
    debug_assert!(properties.is_sorted_by(|a, b| a.0 < b.0));
    let mut key = String::with_capacity(block.len() + 16 * properties.len());
    // Writing into a `String` cannot fail.
    let _ = write_key(&mut key, block, properties);
    key
}

/// Why a text is not a well-formed key.
#[derive(Clone, Debug, PartialEq, Eq, Error)]
#[non_exhaustive]
pub enum KeyError {
    /// The part before the properties is not `namespace:name`.
    #[error(
        "`{key}` is not a valid key: keys have the form `namespace:name`, both parts made of \
         `a-z`, `0-9` and `_`"
    )]
    Block {
        /// The whole text.
        key: String,
    },
    /// A `[` whose list does not end the key with exactly one `]`.
    #[error("`{key}`: the properties of a state are written as `[name=value,…]` at the end")]
    Brackets {
        /// The whole text.
        key: String,
    },
    /// `[]`: the default state is written as the bare block key.
    #[error("`{key}`: an empty property list is not written; the bare key is the default state")]
    EmptyProperties {
        /// The whole text.
        key: String,
    },
    /// An item of the property list is not `name=value`.
    #[error(
        "`{key}`: `{property}` is not a property: properties have the form `name=value`, the \
         name made of `a-z`, `0-9` and `_`, the value of `a-z`, `0-9`, `_`, `+` and `-`"
    )]
    Property {
        /// The whole text.
        key: String,
        /// The offending item.
        property: String,
    },
    /// Properties out of order, or one named twice.
    #[error("`{key}`: properties are sorted by name and named once each; `{name}` is out of place")]
    Order {
        /// The whole text.
        key: String,
        /// The first property name that breaks the order.
        name: String,
    },
}

/// A well-formed block state key, split into the block key and the written properties.
///
/// Parsing checks the grammar only; [`Registry::state`](crate::Registry::state) tells whether
/// the key names a state of a loaded block. Formatting (`Display`) gives back exactly the parsed
/// text.
///
/// ```
/// use cp_content::StateKey;
///
/// let key = StateKey::parse("cubeplanet:furnace[turn=1,up=-x]").unwrap();
/// assert_eq!(key.block(), "cubeplanet:furnace");
/// assert_eq!((key.namespace(), key.name()), ("cubeplanet", "furnace"));
/// assert_eq!(key.properties(), [("turn", "1"), ("up", "-x")]);
/// assert_eq!(key.property("up"), Some("-x"));
/// assert_eq!(key.to_string(), "cubeplanet:furnace[turn=1,up=-x]");
///
/// assert!(StateKey::parse("cubeplanet:furnace[up=-x,turn=1]").is_err()); // not sorted
/// assert!(StateKey::parse("cubeplanet:stone[]").is_err());
/// ```
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct StateKey<'a> {
    block: &'a str,
    properties: Vec<(&'a str, &'a str)>,
}

impl<'a> StateKey<'a> {
    /// Parses `key`.
    ///
    /// # Errors
    /// [`KeyError`] when `key` breaks the grammar of the module documentation.
    pub fn parse(key: &'a str) -> Result<Self, KeyError> {
        let (block, list) = match key.split_once('[') {
            Some((block, rest)) => (block, Some(rest)),
            None => (key, None),
        };
        if !is_block_key(block) {
            return Err(KeyError::Block {
                key: key.to_owned(),
            });
        }
        let mut properties: Vec<(&'a str, &'a str)> = Vec::new();
        if let Some(rest) = list {
            let brackets = || KeyError::Brackets {
                key: key.to_owned(),
            };
            let list = rest.strip_suffix(']').ok_or_else(brackets)?;
            if list.contains(['[', ']']) {
                return Err(brackets());
            }
            if list.is_empty() {
                return Err(KeyError::EmptyProperties {
                    key: key.to_owned(),
                });
            }
            for property in list.split(',') {
                let Some((name, value)) = property
                    .split_once('=')
                    .filter(|&(name, value)| is_ident(name) && is_value(value))
                else {
                    return Err(KeyError::Property {
                        key: key.to_owned(),
                        property: property.to_owned(),
                    });
                };
                if properties
                    .last()
                    .is_some_and(|&(previous, _)| previous >= name)
                {
                    return Err(KeyError::Order {
                        key: key.to_owned(),
                        name: name.to_owned(),
                    });
                }
                properties.push((name, value));
            }
        }
        Ok(Self { block, properties })
    }

    /// The block key, `namespace:name`.
    #[must_use]
    pub fn block(&self) -> &'a str {
        self.block
    }

    /// The part of the block key before the `:`.
    #[must_use]
    pub fn namespace(&self) -> &'a str {
        self.block
            .split_once(':')
            .map_or("", |(namespace, _)| namespace)
    }

    /// The part of the block key after the `:`.
    #[must_use]
    pub fn name(&self) -> &'a str {
        self.block.split_once(':').map_or("", |(_, name)| name)
    }

    /// The written properties as `(name, value)`, sorted by name. Empty for a bare key.
    #[must_use]
    pub fn properties(&self) -> &[(&'a str, &'a str)] {
        &self.properties
    }

    /// The written value of the property `name`; `None` when the key does not write it (the
    /// property then has its default value).
    #[must_use]
    pub fn property(&self, name: &str) -> Option<&'a str> {
        self.properties
            .iter()
            .find(|(own, _)| *own == name)
            .map(|&(_, value)| value)
    }
}

impl fmt::Display for StateKey<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write_key(f, self.block, &self.properties)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identifiers_and_values() {
        for good in ["a", "stone", "log_oak", "2x4", "_", "a1_b2"] {
            assert!(is_ident(good), "{good}");
            assert!(is_value(good), "{good}");
        }
        for bad in ["", "Stone", "log-oak", "a b", "a:b", "ä", "a[", "a=", "+x"] {
            assert!(!is_ident(bad), "{bad}");
        }
        for good in ["+x", "-z", "0", "a+b-c_1"] {
            assert!(is_value(good), "{good}");
        }
        for bad in ["", "+X", "a b", "a=b", "a,b", "a]", "ü", "x."] {
            assert!(!is_value(bad), "{bad}");
        }
    }

    #[test]
    fn definition_keys() {
        for good in [
            AIR_KEY,
            "cubeplanet:stone",
            "cubeplanet:log_oak",
            "mod_2:block_17",
            "a:1",
            // The grammar allows any identifier as namespace (contract §4.4).
            "1mod:stone",
            "_mod:stone",
        ] {
            assert_eq!(check_key(good), Ok(()), "{good}");
        }
        for bad in [
            "",
            "stone",
            ":stone",
            "cubeplanet:",
            "cubeplanet:Stone",
            "Cubeplanet:stone",
            "a:b:c",
            "cubeplanet:grass[up=+x]",
            "cubeplanet: stone",
            "cubeplanet:log oak",
            "cubeplanet:log-oak",
        ] {
            let message = check_key(bad).expect_err(bad);
            assert!(message.contains("namespace:name"), "{message}");
            assert!(message.contains(&format!("`{bad}`")), "{message}");
        }
    }

    #[test]
    fn well_formed_state_keys_round_trip() {
        for (text, block, properties) in [
            ("cubeplanet:stone", "cubeplanet:stone", &[][..]),
            ("ns:b[up=+x]", "ns:b", &[("up", "+x")][..]),
            ("ns:b[axis=z]", "ns:b", &[("axis", "z")][..]),
            (
                "ns:b[turn=3,up=-z]",
                "ns:b",
                &[("turn", "3"), ("up", "-z")][..],
            ),
            (
                "a:b[a=1,b=2,c=3]",
                "a:b",
                &[("a", "1"), ("b", "2"), ("c", "3")][..],
            ),
            ("a:b[lit_2=on-off+]", "a:b", &[("lit_2", "on-off+")][..]),
        ] {
            let key = StateKey::parse(text).unwrap_or_else(|error| panic!("{text}: {error}"));
            assert_eq!(key.block(), block);
            assert_eq!(key.properties(), properties);
            assert_eq!(key.to_string(), text);
            let pairs: Vec<(&str, &str)> = properties.to_vec();
            assert_eq!(format_state_key(block, &pairs), text);
            for (name, value) in properties {
                assert_eq!(key.property(name), Some(*value));
            }
            assert_eq!(key.property("missing"), None);
        }
        let key = StateKey::parse("mod_2:block_17[up=+y]").unwrap();
        assert_eq!((key.namespace(), key.name()), ("mod_2", "block_17"));
    }

    #[test]
    fn malformed_state_keys_are_rejected() {
        let block = |key: &str| KeyError::Block {
            key: key.to_owned(),
        };
        let brackets = |key: &str| KeyError::Brackets {
            key: key.to_owned(),
        };
        let property = |key: &str, item: &str| KeyError::Property {
            key: key.to_owned(),
            property: item.to_owned(),
        };
        let order = |key: &str, name: &str| KeyError::Order {
            key: key.to_owned(),
            name: name.to_owned(),
        };
        let cases = [
            ("", block("")),
            ("stone", block("stone")),
            ("Ns:b", block("Ns:b")),
            ("ns:b c", block("ns:b c")),
            ("ns:[up=+x]", block("ns:[up=+x]")),
            ("ns:b]", block("ns:b]")),
            ("[up=+x]", block("[up=+x]")),
            ("ns:b[up=+x", brackets("ns:b[up=+x")),
            ("ns:b[up=+x]x", brackets("ns:b[up=+x]x")),
            ("ns:b[up=+x]]", brackets("ns:b[up=+x]]")),
            ("ns:b[up=[+x]]", brackets("ns:b[up=[+x]]")),
            ("ns:b[up=+x][a=1]", brackets("ns:b[up=+x][a=1]")),
            (
                "ns:b[]",
                KeyError::EmptyProperties {
                    key: "ns:b[]".to_owned(),
                },
            ),
            ("ns:b[up]", property("ns:b[up]", "up")),
            ("ns:b[up=]", property("ns:b[up=]", "up=")),
            ("ns:b[=+x]", property("ns:b[=+x]", "=+x")),
            ("ns:b[up=+X]", property("ns:b[up=+X]", "up=+X")),
            ("ns:b[Up=+x]", property("ns:b[Up=+x]", "Up=+x")),
            ("ns:b[up=+x=y]", property("ns:b[up=+x=y]", "up=+x=y")),
            ("ns:b[up = +x]", property("ns:b[up = +x]", "up = +x")),
            ("ns:b[a=1,]", property("ns:b[a=1,]", "")),
            ("ns:b[,a=1]", property("ns:b[,a=1]", "")),
            ("ns:b[up=+x,turn=1]", order("ns:b[up=+x,turn=1]", "turn")),
            ("ns:b[up=+x,up=-x]", order("ns:b[up=+x,up=-x]", "up")),
            ("ns:b[a=1,c=2,b=3]", order("ns:b[a=1,c=2,b=3]", "b")),
        ];
        for (text, expected) in cases {
            assert_eq!(StateKey::parse(text), Err(expected), "{text}");
        }
        // Every message names the offending text.
        for text in ["ns:b[up=+x,turn=1]", "ns:b[]", "Ns:b", "ns:b[up]", "ns:b[a"] {
            let message = StateKey::parse(text).unwrap_err().to_string();
            assert!(message.contains(&format!("`{text}`")), "{message}");
        }
    }
}
