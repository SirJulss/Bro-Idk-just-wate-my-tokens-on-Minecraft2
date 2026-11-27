//! Errors of loading and building star systems.
//!
//! System files are external input, so every problem in them is a [`UniverseError`], never a
//! panic (contract §2.6). Each error names the file and, when the problem lies inside one
//! definition, the definition (`system`, ``body `cubeplanet:heimat` ``) and the TOML key.
//! Systems built in code report the same rules as a [`DefinitionError`], which names the body by
//! its index instead of a file position.

use std::fmt;
use std::io;
use std::path::PathBuf;

use thiserror::Error;

/// Why a system file or a directory of system files could not be loaded.
///
/// Lines and columns are 1-based; columns count characters.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum UniverseError {
    /// A file or directory could not be read.
    #[error("{}: {source}", .path.display())]
    Io {
        /// The file or directory.
        path: PathBuf,
        /// The operating system's reason.
        #[source]
        source: io::Error,
    },

    /// The file is not TOML, or uses a TOML construct the system reader does not support
    /// (dotted keys in key/value pairs, multi-line strings, dates and times).
    #[error("{}:{line}:{column}: {message}", .file.display())]
    Syntax {
        /// The offending file.
        file: PathBuf,
        /// Line of the problem.
        line: u32,
        /// Column of the problem.
        column: u32,
        /// What is wrong.
        message: String,
    },

    /// The file is valid TOML but breaks the schema (unknown, missing or mistyped key) or a rule
    /// of the system model (a value out of range, inconsistent bodies).
    #[error("{}:{line}: {}{message}", .file.display(), location(.item, .key))]
    Definition {
        /// The offending file.
        file: PathBuf,
        /// Line of the key, or of the table header when the key is missing.
        line: u32,
        /// The definition: `system`, ``body `cubeplanet:heimat` `` or `body #3` when the body's
        /// key is unreadable.
        item: String,
        /// The TOML key the problem is about, e.g. `orbit.eccentricity`.
        key: Option<String>,
        /// What is wrong.
        message: String,
    },

    /// Two files of one directory define a system with the same key.
    #[error(
        "{}: duplicate system `{key}` (first defined in {})",
        .file.display(),
        .first_file.display()
    )]
    DuplicateSystem {
        /// The key that occurs twice.
        key: String,
        /// The file of the second definition.
        file: PathBuf,
        /// The file of the first definition.
        first_file: PathBuf,
    },
}

/// `item: `key`: ` for messages.
fn location(item: &str, key: &Option<String>) -> String {
    match key {
        Some(key) => format!("{item}: `{key}`: "),
        None => format!("{item}: "),
    }
}

/// A rule of the system model that a definition breaks, for systems built in code
/// ([`crate::SystemDef::new`]).
///
/// The loader turns it into a [`UniverseError::Definition`] with file and line.
#[derive(Clone, Debug, PartialEq, Eq, Error)]
pub struct DefinitionError {
    /// Index of the offending body in the list given to [`crate::SystemDef::new`]; `None` when
    /// the problem is with the system itself.
    pub body: Option<usize>,
    /// The field the problem is about, named like its TOML key (`edge_m`,
    /// `orbit.eccentricity`); `None` when it concerns the whole definition.
    pub field: Option<&'static str>,
    /// What is wrong.
    pub message: String,
}

impl DefinitionError {
    /// An error about field `field` of body `body`.
    pub(crate) fn body(body: usize, field: &'static str, message: impl Into<String>) -> Self {
        Self {
            body: Some(body),
            field: Some(field),
            message: message.into(),
        }
    }

    /// An error about field `field` of the system itself.
    pub(crate) fn system(field: &'static str, message: impl Into<String>) -> Self {
        Self {
            body: None,
            field: Some(field),
            message: message.into(),
        }
    }
}

impl fmt::Display for DefinitionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.body {
            Some(index) => write!(f, "body #{index}: ")?,
            None => f.write_str("system: ")?,
        }
        if let Some(field) = self.field {
            write!(f, "`{field}`: ")?;
        }
        f.write_str(&self.message)
    }
}

/// Why custom gravity shape parameters were rejected ([`crate::CubeGravity::new`]).
#[derive(Clone, Debug, PartialEq, Eq, Error)]
#[error("invalid cube gravity parameters: {0}")]
pub struct GravityError(pub(crate) String);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn messages_name_file_line_item_and_key() {
        let error = UniverseError::Definition {
            file: PathBuf::from("systems/home.toml"),
            line: 42,
            item: "body `cubeplanet:kiesel`".to_owned(),
            key: Some("orbit.eccentricity".to_owned()),
            message: "must be at least 0 and below 1, found 1.5".to_owned(),
        };
        assert_eq!(
            error.to_string().replace('\\', "/"),
            "systems/home.toml:42: body `cubeplanet:kiesel`: `orbit.eccentricity`: must be at \
             least 0 and below 1, found 1.5"
        );
        let whole = UniverseError::Definition {
            file: PathBuf::from("a.toml"),
            line: 1,
            item: "system".to_owned(),
            key: None,
            message: "no bodies".to_owned(),
        };
        assert_eq!(whole.to_string(), "a.toml:1: system: no bodies");
        let syntax = UniverseError::Syntax {
            file: PathBuf::from("a.toml"),
            line: 3,
            column: 7,
            message: "unterminated string".to_owned(),
        };
        assert_eq!(syntax.to_string(), "a.toml:3:7: unterminated string");
        let duplicate = UniverseError::DuplicateSystem {
            key: "cubeplanet:home".to_owned(),
            file: PathBuf::from("b.toml"),
            first_file: PathBuf::from("a.toml"),
        };
        assert_eq!(
            duplicate.to_string(),
            "b.toml: duplicate system `cubeplanet:home` (first defined in a.toml)"
        );
    }

    #[test]
    fn definition_errors_name_body_and_field() {
        let error = DefinitionError::body(3, "edge_m", "must be even");
        assert_eq!(error.to_string(), "body #3: `edge_m`: must be even");
        let error = DefinitionError::system("key", "bad key");
        assert_eq!(error.to_string(), "system: `key`: bad key");
        let error = DefinitionError {
            body: None,
            field: None,
            message: "no bodies".to_owned(),
        };
        assert_eq!(error.to_string(), "system: no bodies");
    }

    #[test]
    fn io_errors_keep_their_source() {
        use std::error::Error as _;
        let error = UniverseError::Io {
            path: PathBuf::from("missing"),
            source: io::Error::new(io::ErrorKind::NotFound, "gone"),
        };
        assert!(error.to_string().ends_with(": gone"));
        assert!(error.source().is_some());
    }
}
