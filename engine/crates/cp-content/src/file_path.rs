//! Paths of file textures: what a pack may write, and finding the file a path names.
//!
//! A pack is written on one machine and loaded on others, so a path has to name the same file
//! on every file system the engine runs on. Otherwise a pack loads on its author's machine and
//! fails elsewhere:
//!
//! - Windows forbids some characters in file names, drops dots and spaces at the end of a name
//!   and reserves device names (`con`, `nul`, `com1`, …), whatever extension follows them;
//! - Windows and macOS compare names without regard to case by default, Linux does not.
//!
//! [`check`] covers the first point on the text alone. [`Listings::find`] covers the second:
//! it looks the path up one component at a time and accepts only names spelled exactly as on
//! disk, so a path that loads anywhere loads everywhere, and one file has one spelling (the
//! content hash takes the path as written).

use std::collections::BTreeMap;
use std::collections::btree_map::Entry;
use std::ffi::OsString;
use std::fs;
use std::path::{Path, PathBuf};

use crate::error::ContentError;

/// Characters that Windows does not allow in a file name, besides the control characters and
/// the separators `/`, `\` and `:`, which [`check`] reports on their own.
const RESERVED_CHARACTERS: [char; 6] = ['<', '>', '"', '|', '?', '*'];

/// Checks the text of a file path: relative to the pack directory, `/`-separated, without
/// `.` or `..` components, and made of names that every file system stores as written.
pub(crate) fn check(path: &str) -> Result<(), String> {
    let structured = !path.is_empty()
        && !path.contains(['\\', ':'])
        && path
            .split('/')
            .all(|component| !matches!(component, "" | "." | ".."));
    if !structured {
        return Err(format!(
            "file path `{path}` must be relative to the pack directory, use `/` as separator \
             and contain no `.` or `..` components"
        ));
    }
    for component in path.split('/') {
        let reserved = |c: char| c.is_control() || RESERVED_CHARACTERS.contains(&c);
        if let Some(character) = component.chars().find(|&c| reserved(c)) {
            return Err(format!(
                "file path `{}` contains {character:?}, which Windows does not allow in file \
                 names",
                path.escape_debug()
            ));
        }
        if component.ends_with(['.', ' ']) {
            return Err(format!(
                "file path `{path}`: the name `{component}` ends in a dot or a space, which \
                 Windows drops, so it would open another file there"
            ));
        }
        if is_device_name(component) {
            return Err(format!(
                "file path `{path}`: `{component}` names a device on Windows, not a file"
            ));
        }
    }
    Ok(())
}

/// True for the names Windows reserves for devices: `con`, `prn`, `aux`, `nul`, `conin$`,
/// `conout$`, and `com`/`lpt` followed by a digit, in any case. Windows ignores what follows
/// the first dot and spaces before it, so `nul.png` and `nul .png` are devices too.
fn is_device_name(component: &str) -> bool {
    let stem = component
        .split_once('.')
        .map_or(component, |(stem, _)| stem)
        .trim_end_matches(' ')
        .to_ascii_uppercase();
    if matches!(
        stem.as_str(),
        "CON" | "PRN" | "AUX" | "NUL" | "CONIN$" | "CONOUT$"
    ) {
        return true;
    }
    let (Some(prefix), Some(number)) = (stem.get(..3), stem.get(3..)) else {
        return false;
    };
    let mut digits = number.chars();
    matches!(prefix, "COM" | "LPT")
        && matches!(
            (digits.next(), digits.next()),
            (Some('0'..='9' | '¹' | '²' | '³'), None)
        )
}

/// The names in the directories that file texture paths pass through, each directory read
/// once while the textures of a registry are resolved.
#[derive(Debug, Default)]
pub(crate) struct Listings {
    /// The entries of each directory read so far, sorted by name.
    names: BTreeMap<PathBuf, Vec<Name>>,
}

/// One entry of a directory listing.
#[derive(Debug)]
struct Name {
    /// The name as the file system stores it.
    exact: OsString,
    /// The name in lower case, for names that are UTF-8.
    folded: Option<String>,
}

impl Listings {
    /// The file that `path` names below `pack_dir`, with every component spelled exactly as
    /// the directory lists it. `path` must have passed [`check`].
    ///
    /// # Errors
    /// [`ContentError::Io`] when a directory on the way cannot be listed; otherwise the
    /// problem with the path, turned into an error by `invalid`.
    pub(crate) fn find(
        &mut self,
        pack_dir: &Path,
        path: &str,
        invalid: impl Fn(String) -> ContentError,
    ) -> Result<PathBuf, ContentError> {
        let mut current = pack_dir.to_path_buf();
        let mut components = path.split('/').peekable();
        while let Some(component) = components.next() {
            let names = self.names(&current)?;
            let exact = names.iter().any(|name| name.exact == component);
            if let Some(other) = differs_in_case(names, component) {
                return Err(invalid(if exact {
                    format!(
                        "texture file `{path}`: `{component}` and `{other}` lie in {} and \
                         differ only in case; Windows and macOS cannot tell them apart",
                        current.display()
                    )
                } else {
                    format!(
                        "texture file `{path}` does not exist in its pack: `{component}` is \
                         spelled `{other}` on disk (in {}), and names must match exactly to \
                         load on every file system",
                        current.display()
                    )
                }));
            }
            if !exact {
                return Err(invalid(format!(
                    "texture file `{path}` does not exist in its pack (looked for {})",
                    pack_dir.join(path).display()
                )));
            }
            current.push(component);
            if components.peek().is_some() && !current.is_dir() {
                return Err(invalid(format!(
                    "texture file `{path}` does not exist in its pack: {} is not a directory",
                    current.display()
                )));
            }
        }
        if !current.is_file() {
            return Err(invalid(format!(
                "texture file `{path}` is not a file (looked at {})",
                current.display()
            )));
        }
        Ok(pack_dir.join(path))
    }

    /// The entries of `dir`, read on first use.
    fn names(&mut self, dir: &Path) -> Result<&[Name], ContentError> {
        match self.names.entry(dir.to_path_buf()) {
            Entry::Occupied(slot) => Ok(slot.into_mut().as_slice()),
            Entry::Vacant(slot) => {
                let io_error = |source| ContentError::Io {
                    path: dir.to_path_buf(),
                    source,
                };
                let mut names = Vec::new();
                for entry in fs::read_dir(dir).map_err(io_error)? {
                    let exact = entry.map_err(io_error)?.file_name();
                    let folded = exact.to_str().map(str::to_lowercase);
                    names.push(Name { exact, folded });
                }
                // `read_dir` yields an order that depends on the file system; sorted, the
                // name an error reports does not.
                names.sort_by(|a, b| a.exact.cmp(&b.exact));
                Ok(slot.insert(names).as_slice())
            }
        }
    }
}

/// The first of `names` that equals `component` except for case, if any.
fn differs_in_case<'a>(names: &'a [Name], component: &str) -> Option<&'a str> {
    let folded = component.to_lowercase();
    names.iter().find_map(|name| {
        let text = name.exact.to_str()?;
        (text != component && name.folded.as_deref() == Some(folded.as_str())).then_some(text)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn portable_paths_pass() {
        for path in [
            "rock.png",
            "textures/img/rock.png",
            "textures/Rock Face.PNG",
            "a/.hidden/b.png",
            "con_rock.png",
            "console.png",
            "com10.png",
            "lpt.png",
            "nul_/x.png",
            "ünïcode/石.png",
        ] {
            assert_eq!(check(path), Ok(()), "{path}");
        }
    }

    #[test]
    fn paths_outside_the_pack_fail() {
        for path in [
            "",
            "/rock.png",
            "../rock.png",
            "a/../rock.png",
            "./rock.png",
            "a//rock.png",
            "a/",
            "a\\rock.png",
            "c:rock.png",
        ] {
            let message = check(path).unwrap_err();
            assert!(
                message.contains("must be relative to the pack directory"),
                "{path}: {message}"
            );
        }
    }

    #[test]
    fn names_windows_cannot_store_fail() {
        for (path, problem) in [
            ("a<b.png", "contains '<'"),
            ("a>b.png", "contains '>'"),
            ("a\"b.png", "contains '\"'"),
            ("a|b.png", "contains '|'"),
            ("a?.png", "contains '?'"),
            ("img/*.png", "contains '*'"),
            ("a\tb.png", "contains '\\t'"),
            ("a\r\nb.png", "contains '\\r'"),
            ("a\u{7f}.png", "contains '\\u{7f}'"),
            ("rock.png.", "ends in a dot or a space"),
            ("rock.png ", "ends in a dot or a space"),
            ("img./rock.png", "ends in a dot or a space"),
            ("img /rock.png", "ends in a dot or a space"),
            ("nul", "names a device"),
            ("NUL.png", "names a device"),
            ("img/con.tar.gz", "names a device"),
            ("Aux/rock.png", "names a device"),
            ("prn .png", "names a device"),
            ("com1.png", "names a device"),
            ("LPT9", "names a device"),
            ("com0.png", "names a device"),
            ("lpt¹.png", "names a device"),
            ("conin$.png", "names a device"),
            ("CONOUT$", "names a device"),
        ] {
            let message = check(path).unwrap_err();
            assert!(message.contains(problem), "{path:?}: {message}");
        }
        // Control characters are shown escaped, so the message stays on one line.
        assert!(!check("a\nb.png").unwrap_err().contains('\n'));
    }
}
