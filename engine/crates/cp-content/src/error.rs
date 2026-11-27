//! Errors of content loading.
//!
//! Content packs are external input, so every problem in them is reported as a
//! [`ContentError`] and never as a panic (contract §2.6). Each error names the file it is about
//! and, when the problem lies inside one definition, the key of that definition.

use std::fmt;
use std::io;
use std::path::{Path, PathBuf};

use thiserror::Error;

/// The kinds of definitions a content pack holds.
///
/// Each kind has its own key space: a block, a material and a texture may all be called
/// `cubeplanet:stone`.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub enum DefKind {
    /// A texture, defined in `textures/*.toml` as `[[texture]]`.
    Texture,
    /// A material, defined in `materials/*.toml` as `[[material]]`.
    Material,
    /// A block, defined in `blocks/*.toml` as `[[block]]`.
    Block,
}

impl DefKind {
    /// All kinds, in the order in which a pack is read and validated (references point from
    /// blocks to materials to textures).
    pub const ALL: [Self; 3] = [Self::Texture, Self::Material, Self::Block];

    /// The lower-case name used in messages; also the name of the TOML array of tables:
    /// `texture`, `material`, `block`.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Texture => "texture",
            Self::Material => "material",
            Self::Block => "block",
        }
    }

    /// The sub-directory of a pack that holds the files of this kind: `textures`, `materials`,
    /// `blocks`.
    #[must_use]
    pub const fn directory(self) -> &'static str {
        match self {
            Self::Texture => "textures",
            Self::Material => "materials",
            Self::Block => "blocks",
        }
    }
}

impl fmt::Display for DefKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// Why a set of content packs could not be loaded.
///
/// Lines and columns are 1-based. A definition's `line` is the line of its `[[block]]`,
/// `[[material]]` or `[[texture]]` header.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum ContentError {
    /// A pack directory or one of its files could not be read.
    #[error("{}: {source}", .path.display())]
    Io {
        /// The directory or file that could not be read.
        path: PathBuf,
        /// The operating system's reason.
        #[source]
        source: io::Error,
    },

    /// A directory is not a usable content pack: it does not exist, has no `pack.toml`, or its
    /// `pack.toml` declares something this engine cannot read.
    #[error("{}: {message}", .path.display())]
    Pack {
        /// The pack directory or its `pack.toml`.
        path: PathBuf,
        /// What is wrong.
        message: String,
    },

    /// A file is not valid TOML or does not match the schema: a syntax error, an unknown field,
    /// a missing field or a value of the wrong type.
    #[error("{}:{line}:{column}: {}{message}", .file.display(), key_prefix(.key))]
    Parse {
        /// The offending file.
        file: PathBuf,
        /// Line of the problem.
        line: u32,
        /// Column of the problem, counted in characters.
        column: u32,
        /// Key of the definition the problem is in, when the definition has a readable `key`.
        key: Option<String>,
        /// The parser's description, e.g. ``unknown field `colour`, expected one of …``.
        message: String,
    },

    /// A key is defined twice inside one pack. (Defining a key again in a *later* pack is an
    /// override, not an error.)
    #[error(
        "{}:{line}: duplicate {kind} `{key}` (first defined at {}:{first_line})",
        .file.display(),
        .first_file.display()
    )]
    DuplicateKey {
        /// Kind of the definition.
        kind: DefKind,
        /// The key that occurs twice.
        key: String,
        /// File of the second definition.
        file: PathBuf,
        /// Line of the second definition.
        line: u32,
        /// File of the first definition.
        first_file: PathBuf,
        /// Line of the first definition.
        first_line: u32,
    },

    /// A definition refers to a key that no loaded pack defines: a block face names a missing
    /// material, or a material names a missing texture.
    #[error(
        "{}:{line}: {kind} `{key}` refers to {target_kind} `{target}`, which no loaded pack \
         defines",
        .file.display()
    )]
    UnknownReference {
        /// Kind of the referring definition.
        kind: DefKind,
        /// Key of the referring definition.
        key: String,
        /// File of the referring definition.
        file: PathBuf,
        /// Line of the referring definition.
        line: u32,
        /// Kind of the missing definition.
        target_kind: DefKind,
        /// The key that could not be found.
        target: String,
    },

    /// A definition matches the schema but breaks a rule: a malformed key, a value out of
    /// range, a combination of fields that makes no sense.
    #[error("{}:{line}: {kind} `{key}`: {message}", .file.display())]
    Invalid {
        /// Kind of the definition.
        kind: DefKind,
        /// Key of the definition.
        key: String,
        /// File of the definition.
        file: PathBuf,
        /// Line of the definition.
        line: u32,
        /// The rule that is broken.
        message: String,
    },

    /// The loaded packs define more entries than an id type can address.
    #[error("too many {what}: {count}, the limit is {limit}")]
    Limit {
        /// What there is too much of: `"materials"`, `"textures"` or `"block states"`.
        what: &'static str,
        /// How many the packs define.
        count: usize,
        /// How many the engine can address.
        limit: usize,
    },
}

impl ContentError {
    /// The file or directory the error is about; `None` only for [`ContentError::Limit`].
    #[must_use]
    pub fn path(&self) -> Option<&Path> {
        match self {
            Self::Io { path, .. } | Self::Pack { path, .. } => Some(path),
            Self::Parse { file, .. }
            | Self::DuplicateKey { file, .. }
            | Self::UnknownReference { file, .. }
            | Self::Invalid { file, .. } => Some(file),
            Self::Limit { .. } => None,
        }
    }

    /// The key of the definition the error is about, when it is about one.
    #[must_use]
    pub fn key(&self) -> Option<&str> {
        match self {
            Self::Parse { key, .. } => key.as_deref(),
            Self::DuplicateKey { key, .. }
            | Self::UnknownReference { key, .. }
            | Self::Invalid { key, .. } => Some(key),
            Self::Io { .. } | Self::Pack { .. } | Self::Limit { .. } => None,
        }
    }
}

/// `` `key`: `` when the key of the offending definition is known, nothing otherwise.
fn key_prefix(key: &Option<String>) -> String {
    key.as_ref()
        .map_or_else(String::new, |key| format!("`{key}`: "))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kinds_name_their_tables_and_directories() {
        let names: Vec<_> = DefKind::ALL
            .iter()
            .map(|kind| (kind.name(), kind.directory(), kind.to_string()))
            .collect();
        assert_eq!(
            names,
            [
                ("texture", "textures", "texture".to_owned()),
                ("material", "materials", "material".to_owned()),
                ("block", "blocks", "block".to_owned()),
            ]
        );
    }

    #[test]
    fn messages_name_file_line_and_key() {
        let parse = ContentError::Parse {
            file: PathBuf::from("pack/blocks/a.toml"),
            line: 7,
            column: 3,
            key: Some("test:stone".to_owned()),
            message: "unknown field `colour`".to_owned(),
        };
        assert_eq!(
            parse.to_string().replace('\\', "/"),
            "pack/blocks/a.toml:7:3: `test:stone`: unknown field `colour`"
        );
        assert_eq!(parse.key(), Some("test:stone"));
        assert_eq!(parse.path(), Some(Path::new("pack/blocks/a.toml")));

        let syntax = ContentError::Parse {
            file: PathBuf::from("pack/pack.toml"),
            line: 1,
            column: 9,
            key: None,
            message: "invalid string".to_owned(),
        };
        assert_eq!(
            syntax.to_string().replace('\\', "/"),
            "pack/pack.toml:1:9: invalid string"
        );
        assert_eq!(syntax.key(), None);

        let duplicate = ContentError::DuplicateKey {
            kind: DefKind::Material,
            key: "test:stone".to_owned(),
            file: PathBuf::from("b.toml"),
            line: 9,
            first_file: PathBuf::from("a.toml"),
            first_line: 2,
        };
        assert_eq!(
            duplicate.to_string(),
            "b.toml:9: duplicate material `test:stone` (first defined at a.toml:2)"
        );

        let dangling = ContentError::UnknownReference {
            kind: DefKind::Block,
            key: "test:stone".to_owned(),
            file: PathBuf::from("a.toml"),
            line: 4,
            target_kind: DefKind::Material,
            target: "test:missing".to_owned(),
        };
        assert_eq!(
            dangling.to_string(),
            "a.toml:4: block `test:stone` refers to material `test:missing`, \
             which no loaded pack defines"
        );
        assert_eq!(dangling.key(), Some("test:stone"));

        let invalid = ContentError::Invalid {
            kind: DefKind::Texture,
            key: "test:t".to_owned(),
            file: PathBuf::from("t.toml"),
            line: 1,
            message: "give either `procedural` or `file`, not both".to_owned(),
        };
        assert_eq!(
            invalid.to_string(),
            "t.toml:1: texture `test:t`: give either `procedural` or `file`, not both"
        );

        let limit = ContentError::Limit {
            what: "materials",
            count: 70_000,
            limit: 65_536,
        };
        assert_eq!(
            limit.to_string(),
            "too many materials: 70000, the limit is 65536"
        );
        assert_eq!((limit.path(), limit.key()), (None, None));
    }

    #[test]
    fn io_errors_keep_their_source() {
        use std::error::Error as _;
        let error = ContentError::Io {
            path: PathBuf::from("missing/pack.toml"),
            source: io::Error::new(io::ErrorKind::NotFound, "gone"),
        };
        assert!(error.to_string().ends_with(": gone"));
        assert!(error.source().is_some());
        assert_eq!(error.path(), Some(Path::new("missing/pack.toml")));
    }
}
