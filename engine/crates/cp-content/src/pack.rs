//! Content packs as they lie on disk.
//!
//! ```text
//! <pack>/
//!   pack.toml             format, name, version, description
//!   textures/**/*.toml    [[texture]] entries
//!   materials/**/*.toml   [[material]] entries
//!   blocks/**/*.toml      [[block]] entries
//! ```
//!
//! Only `pack.toml` is required. Every `.toml` file below the three directories is a
//! definition file, in sub-directories too; they are read in the order of their paths. Other
//! files and directories (image files, `systems/` of `cp-universe`, …) are left alone. This
//! module only reads text; `schema` parses it and `loader` gives it meaning.
//!
//! Every text is read with LF line endings ([`unix_newlines`]), so a pack means the same
//! whether its files were saved with LF or CRLF.

use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use crate::error::{ContentError, DefKind};
use crate::key::is_ident;
use crate::schema::parse_manifest;

/// The pack format this engine reads (`format` in `pack.toml`).
pub(crate) const PACK_FORMAT: u32 = 1;

/// What a loaded content pack says about itself in `pack.toml`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PackInfo {
    /// Short name of the pack: `a-z`, `0-9` and `_`.
    pub name: String,
    /// Version of the pack's content, free-form.
    pub version: String,
    /// One-line description; may be empty.
    pub description: String,
}

/// One definition file of a pack.
#[derive(Debug)]
pub(crate) struct SourceFile {
    /// Path used in error messages.
    pub path: PathBuf,
    /// What the file defines, given by the directory it lies in.
    pub kind: DefKind,
    /// The file's content, with LF line endings.
    pub text: String,
}

/// The text of one content pack, read from a directory or embedded in the binary.
#[derive(Debug)]
pub(crate) struct PackSource {
    /// The pack directory, against which texture file paths are resolved. `None` for embedded
    /// packs, which cannot contain file textures.
    pub dir: Option<PathBuf>,
    /// Path of `pack.toml`, for error messages.
    pub manifest_path: PathBuf,
    /// Content of `pack.toml`, with LF line endings.
    pub manifest: String,
    /// The definition files: textures, then materials, then blocks, each sorted by path.
    pub files: Vec<SourceFile>,
}

impl PackSource {
    /// Reads the pack in the directory `dir`.
    pub(crate) fn read(dir: &Path) -> Result<Self, ContentError> {
        if !dir.is_dir() {
            let message = if dir.exists() {
                "a content pack is a directory, this is not one"
            } else {
                "the content pack directory does not exist"
            };
            return Err(ContentError::Pack {
                path: dir.to_path_buf(),
                message: message.to_owned(),
            });
        }
        let manifest_path = dir.join("pack.toml");
        let manifest = match fs::read_to_string(&manifest_path) {
            Ok(text) => unix_newlines(text),
            Err(error) if error.kind() == io::ErrorKind::NotFound => {
                return Err(ContentError::Pack {
                    path: dir.to_path_buf(),
                    message: "not a content pack: `pack.toml` is missing".to_owned(),
                });
            }
            Err(source) => {
                return Err(ContentError::Io {
                    path: manifest_path,
                    source,
                });
            }
        };

        let mut files = Vec::new();
        for kind in DefKind::ALL {
            for path in toml_files(&dir.join(kind.directory()))? {
                let text = fs::read_to_string(&path).map_err(|source| ContentError::Io {
                    path: path.clone(),
                    source,
                })?;
                files.push(SourceFile {
                    path,
                    kind,
                    text: unix_newlines(text),
                });
            }
        }
        Ok(Self {
            dir: Some(dir.to_path_buf()),
            manifest_path,
            manifest,
            files,
        })
    }

    /// A pack whose text is compiled into the binary. `label` stands in for the directory in
    /// error messages; `files` are `(name, kind, text)`.
    pub(crate) fn embedded(label: &str, manifest: &str, files: &[(&str, DefKind, &str)]) -> Self {
        let root = Path::new(label);
        Self {
            dir: None,
            manifest_path: root.join("pack.toml"),
            manifest: unix_newlines(manifest.to_owned()),
            files: files
                .iter()
                .map(|&(name, kind, text)| SourceFile {
                    path: root.join(kind.directory()).join(name),
                    kind,
                    text: unix_newlines(text.to_owned()),
                })
                .collect(),
        }
    }

    /// Parses and checks `pack.toml`.
    pub(crate) fn info(&self) -> Result<PackInfo, ContentError> {
        let raw = parse_manifest(&self.manifest_path, &self.manifest)?;
        let invalid = |message: String| ContentError::Pack {
            path: self.manifest_path.clone(),
            message,
        };
        if raw.format != PACK_FORMAT {
            return Err(invalid(format!(
                "pack format {} is not supported; this engine reads format {PACK_FORMAT}",
                raw.format
            )));
        }
        if !is_ident(&raw.name) {
            return Err(invalid(format!(
                "pack name `{}` must be made of `a-z`, `0-9` and `_`",
                raw.name
            )));
        }
        if raw.version.trim().is_empty() {
            return Err(invalid("`version` must not be empty".to_owned()));
        }
        Ok(PackInfo {
            name: raw.name,
            version: raw.version,
            description: raw.description,
        })
    }
}

/// `text` with every CRLF line ending turned into LF.
///
/// TOML reads both as a newline, but the parser keeps a CRLF inside a multi-line string as
/// written. Git for Windows checks text files out with CRLF by default, so without this a
/// multi-line `licence` or `[ext]` string, and with it the content hash, would depend on how a
/// pack was checked out. The TOML specification allows the conversion. A CR on its own is not
/// a TOML newline and is kept, and so is a line break written as the escapes `\r\n`: that is
/// data the author spelled out.
fn unix_newlines(text: String) -> String {
    if text.contains("\r\n") {
        text.replace("\r\n", "\n")
    } else {
        text
    }
}

/// The `*.toml` files in `dir` and its sub-directories, sorted by path. A missing directory
/// has none.
fn toml_files(dir: &Path) -> Result<Vec<PathBuf>, ContentError> {
    let mut paths = Vec::new();
    match collect_toml_files(dir, &mut paths) {
        Err(ContentError::Io { path, source })
            if path == dir && source.kind() == io::ErrorKind::NotFound => {}
        other => other?,
    }
    // `read_dir` yields entries in an order that depends on the file system.
    paths.sort();
    Ok(paths)
}

fn collect_toml_files(dir: &Path, paths: &mut Vec<PathBuf>) -> Result<(), ContentError> {
    let io_error = |source| ContentError::Io {
        path: dir.to_path_buf(),
        source,
    };
    for entry in fs::read_dir(dir).map_err(io_error)? {
        let entry = entry.map_err(io_error)?;
        let path = entry.path();
        // `file_type` does not follow symbolic links: a link to a directory is not entered, so
        // a cycle of links cannot trap the walk.
        if entry.file_type().map_err(io_error)?.is_dir() {
            collect_toml_files(&path, paths)?;
        } else if path.is_file()
            && path
                .extension()
                .is_some_and(|extension| extension.eq_ignore_ascii_case("toml"))
        {
            paths.push(path);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pack(manifest: &str) -> PackSource {
        PackSource::embedded("<test>", manifest, &[])
    }

    #[test]
    fn embedded_packs_place_files_by_kind() {
        let source = PackSource::embedded(
            "<builtin>",
            "format = 1\nname = \"minimal\"\nversion = \"1\"\ndescription = \"d\"\n",
            &[
                ("a.toml", DefKind::Texture, "# t"),
                ("b.toml", DefKind::Block, "# b"),
            ],
        );
        assert_eq!(source.dir, None);
        assert_eq!(
            source.manifest_path,
            Path::new("<builtin>").join("pack.toml")
        );
        let files: Vec<_> = source
            .files
            .iter()
            .map(|file| (file.path.clone(), file.kind, file.text.as_str()))
            .collect();
        assert_eq!(
            files,
            [
                (
                    Path::new("<builtin>").join("textures").join("a.toml"),
                    DefKind::Texture,
                    "# t"
                ),
                (
                    Path::new("<builtin>").join("blocks").join("b.toml"),
                    DefKind::Block,
                    "# b"
                ),
            ]
        );
        assert_eq!(
            source.info().unwrap(),
            PackInfo {
                name: "minimal".to_owned(),
                version: "1".to_owned(),
                description: "d".to_owned(),
            }
        );
    }

    #[test]
    fn texts_are_read_with_lf_line_endings() {
        assert_eq!(unix_newlines("a\r\nb\r\n".to_owned()), "a\nb\n");
        assert_eq!(unix_newlines("a\nb\n".to_owned()), "a\nb\n");
        // A lone CR is not a line ending: in CR CR LF only the CRLF is one (as in Git and in
        // Python's `tomllib`, which `tests/reference_hash.py` uses).
        assert_eq!(unix_newlines("a\rb\r\r\n".to_owned()), "a\rb\r\n");
        // The escapes `\r\n` are text, not a line ending.
        assert_eq!(unix_newlines(r"a\r\nb".to_owned()), r"a\r\nb");

        let source = PackSource::embedded(
            "<test>",
            "format = 1\r\nname = \"p\"\r\nversion = \"1\"\r\ndescription = \"\"\"\r\nd\r\n\"\"\"\r\n",
            &[("a.toml", DefKind::Texture, "# t\r\n")],
        );
        assert_eq!(source.files[0].text, "# t\n");
        assert_eq!(source.info().unwrap().description, "d\n");
    }

    #[test]
    fn manifest_rules() {
        let message = |manifest: &str| match pack(manifest).info() {
            Err(ContentError::Pack { path, message }) => {
                assert_eq!(path, Path::new("<test>").join("pack.toml"));
                message
            }
            other => panic!("expected a pack error, got {other:?}"),
        };
        assert!(
            message("format = 2\nname = \"p\"\nversion = \"1\"\n")
                .contains("format 2 is not supported")
        );
        assert!(message("format = 1\nname = \"My Pack\"\nversion = \"1\"\n").contains("`My Pack`"));
        assert!(message("format = 1\nname = \"\"\nversion = \"1\"\n").contains("pack name"));
        assert!(message("format = 1\nname = \"p\"\nversion = \" \"\n").contains("`version`"));

        // Schema problems are parse errors with a position.
        let error = pack("format = 1\nversion = \"1\"\n").info().unwrap_err();
        assert!(matches!(error, ContentError::Parse { .. }), "{error:?}");
        assert!(
            error.to_string().contains("missing field `name`"),
            "{error}"
        );
        let error = pack("format = \"one\"\nname = \"p\"\nversion = \"1\"\n")
            .info()
            .unwrap_err();
        assert!(error.to_string().contains("pack.toml:1:10"), "{error}");
    }
}
