//! The `#include` preprocessor of the shader library.
//!
//! A line whose first non-blank character is `#` is a directive; WGSL itself never uses `#`.
//! The only directive is
//!
//! ```text
//! #include "common/color.wgsl"
//! ```
//!
//! which is replaced by the preprocessed text of that file. Paths are relative to the root of
//! the library, use `/` and end in `.wgsl`. Every file is included at most once per program
//! (an automatic include guard): a second `#include` of a file that is already in the output
//! adds nothing. An `#include` of a file that is still being expanded is a cycle and an error.
//! Only a comment may follow the closing quote.

use std::fmt;

use super::ShaderError;

/// A line of a library file: its path and its 1-based line number.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct SourceLine {
    /// Path of the file relative to the library root, e.g. `common/color.wgsl`.
    pub file: String,
    /// 1-based line number in that file.
    pub line: u32,
}

impl fmt::Display for SourceLine {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}", self.file, self.line)
    }
}

/// A program after preprocessing: one WGSL text and, for every line of it, the file and line
/// it came from.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PreprocessedSource {
    program: String,
    text: String,
    /// Every file that contributed, in the order of first inclusion; the program first.
    files: Vec<String>,
    /// Per output line: index into `files` and 1-based line in that file.
    origins: Vec<(usize, u32)>,
}

impl PreprocessedSource {
    /// Path of the program the text was made from.
    pub fn program(&self) -> &str {
        &self.program
    }

    /// The WGSL text with every include expanded; each line ends with `\n`.
    pub fn text(&self) -> &str {
        &self.text
    }

    /// Every file that contributed, in the order of first inclusion; the program comes first.
    pub fn files(&self) -> &[String] {
        &self.files
    }

    /// Number of lines of [`Self::text`].
    pub fn line_count(&self) -> usize {
        self.origins.len()
    }

    /// Where line `line_number` (1-based, as naga reports it) of [`Self::text`] came from.
    /// `None` outside the text.
    pub fn origin(&self, line_number: u32) -> Option<SourceLine> {
        let index = usize::try_from(line_number.checked_sub(1)?).ok()?;
        let &(file, line) = self.origins.get(index)?;
        Some(SourceLine {
            file: self.files[file].clone(),
            line,
        })
    }
}

/// Expands every `#include` of the file `program`; `lookup` maps a path to the text of a file.
///
/// # Errors
/// [`ShaderError::UnknownFile`] for a path `lookup` does not know (with the line of the
/// `#include`, unless it is `program` itself), [`ShaderError::IncludeCycle`],
/// [`ShaderError::BadDirective`] and [`ShaderError::BadIncludePath`].
pub fn preprocess<'s>(
    program: &str,
    lookup: impl Fn(&str) -> Option<&'s str>,
) -> Result<PreprocessedSource, ShaderError> {
    let mut expander = Expander {
        lookup: &lookup,
        text: String::new(),
        files: Vec::new(),
        origins: Vec::new(),
        open: Vec::new(),
        done: Vec::new(),
    };
    expander.expand(program, None)?;
    Ok(PreprocessedSource {
        program: program.to_owned(),
        text: expander.text,
        files: expander.files,
        origins: expander.origins,
    })
}

/// State of one preprocessing run.
struct Expander<'l, F> {
    lookup: &'l F,
    text: String,
    files: Vec<String>,
    origins: Vec<(usize, u32)>,
    /// Files whose expansion has begun and not ended: the current include chain.
    open: Vec<String>,
    /// Files whose expansion has ended; including them again adds nothing.
    done: Vec<String>,
}

impl<'s, F: Fn(&str) -> Option<&'s str>> Expander<'_, F> {
    fn expand(&mut self, path: &str, site: Option<SourceLine>) -> Result<(), ShaderError> {
        if let Some(start) = self.open.iter().position(|open| open == path) {
            let mut chain = self.open[start..].to_vec();
            chain.push(path.to_owned());
            return Err(ShaderError::IncludeCycle { chain });
        }
        if self.done.iter().any(|done| done == path) {
            return Ok(());
        }
        let Some(source) = (self.lookup)(path) else {
            return Err(ShaderError::UnknownFile {
                path: path.to_owned(),
                included_from: site,
            });
        };
        let file = self.files.len();
        self.files.push(path.to_owned());
        self.open.push(path.to_owned());
        for (index, line) in source.lines().enumerate() {
            let number = u32::try_from(index + 1).unwrap_or(u32::MAX);
            let at = || SourceLine {
                file: path.to_owned(),
                line: number,
            };
            match classify(line) {
                Line::Code => {
                    self.text.push_str(line);
                    self.text.push('\n');
                    self.origins.push((file, number));
                }
                Line::Include(target) => {
                    if !is_library_path(target) {
                        return Err(ShaderError::BadIncludePath {
                            at: at(),
                            path: target.to_owned(),
                        });
                    }
                    self.expand(target, Some(at()))?;
                }
                Line::Malformed => {
                    return Err(ShaderError::BadDirective {
                        at: at(),
                        text: line.trim().to_owned(),
                    });
                }
            }
        }
        self.open.pop();
        self.done.push(path.to_owned());
        Ok(())
    }
}

/// What a line of a library file is.
#[derive(Debug, PartialEq, Eq)]
enum Line<'a> {
    /// WGSL, copied as it is.
    Code,
    /// `#include "path"`.
    Include(&'a str),
    /// Starts with `#` but is no well-formed include.
    Malformed,
}

fn classify(line: &str) -> Line<'_> {
    let Some(directive) = line.trim_start().strip_prefix('#') else {
        return Line::Code;
    };
    let Some(arguments) = directive.strip_prefix("include") else {
        return Line::Malformed;
    };
    let quoted = arguments.trim_start();
    // `#includefoo` is not an include.
    if quoted.len() == arguments.len() {
        return Line::Malformed;
    }
    let Some(quoted) = quoted.strip_prefix('"') else {
        return Line::Malformed;
    };
    let Some((path, rest)) = quoted.split_once('"') else {
        return Line::Malformed;
    };
    let rest = rest.trim();
    if rest.is_empty() || rest.starts_with("//") {
        Line::Include(path)
    } else {
        Line::Malformed
    }
}

/// A path relative to the library root: `/`-separated names of `[A-Za-z0-9_.-]`, none of them
/// empty, `.` or `..`, ending in `.wgsl`.
fn is_library_path(path: &str) -> bool {
    let valid_name = |name: &str| {
        !name.is_empty()
            && name != "."
            && name != ".."
            && name
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || b"_.-".contains(&byte))
    };
    path.ends_with(".wgsl") && path.split('/').all(valid_name)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A lookup over a fixed set of files.
    fn files<'s>(set: &'s [(&'s str, &'s str)]) -> impl Fn(&str) -> Option<&'s str> {
        move |path| {
            set.iter()
                .find(|(name, _)| *name == path)
                .map(|(_, text)| *text)
        }
    }

    #[test]
    fn includes_are_expanded_in_place_and_recursively() {
        let set = [
            (
                "main.wgsl",
                "// main\n#include \"a.wgsl\"\nfn main_body() {}\n",
            ),
            ("a.wgsl", "#include \"sub/b.wgsl\"\nfn a() {}"),
            ("sub/b.wgsl", "fn b() {}\n"),
        ];
        let source = preprocess("main.wgsl", files(&set)).unwrap();
        assert_eq!(
            source.text(),
            "// main\nfn b() {}\nfn a() {}\nfn main_body() {}\n"
        );
        assert_eq!(source.program(), "main.wgsl");
        assert_eq!(source.files(), ["main.wgsl", "a.wgsl", "sub/b.wgsl"]);
        assert_eq!(source.line_count(), 4);
        let origin = |line| source.origin(line).map(|at| at.to_string());
        assert_eq!(origin(1).as_deref(), Some("main.wgsl:1"));
        assert_eq!(origin(2).as_deref(), Some("sub/b.wgsl:1"));
        assert_eq!(origin(3).as_deref(), Some("a.wgsl:2"));
        assert_eq!(origin(4).as_deref(), Some("main.wgsl:3"));
        assert_eq!(source.origin(0), None);
        assert_eq!(source.origin(5), None);
    }

    #[test]
    fn a_file_is_included_once_per_program() {
        // A diamond: both branches include the same leaf, which appears once, where it is
        // first included.
        let set = [
            (
                "main.wgsl",
                "#include \"left.wgsl\"\n#include \"right.wgsl\"\n#include \"leaf.wgsl\"\nmain\n",
            ),
            ("left.wgsl", "#include \"leaf.wgsl\"\nleft\n"),
            ("right.wgsl", "#include \"leaf.wgsl\"\nright\n"),
            ("leaf.wgsl", "leaf\n"),
        ];
        let source = preprocess("main.wgsl", files(&set)).unwrap();
        assert_eq!(source.text(), "leaf\nleft\nright\nmain\n");
        assert_eq!(
            source.files(),
            ["main.wgsl", "left.wgsl", "leaf.wgsl", "right.wgsl"]
        );
    }

    #[test]
    fn cycles_are_rejected_with_their_chain() {
        let set = [("self.wgsl", "#include \"self.wgsl\"\n")];
        assert_eq!(
            preprocess("self.wgsl", files(&set)),
            Err(ShaderError::IncludeCycle {
                chain: vec!["self.wgsl".to_owned(), "self.wgsl".to_owned()]
            })
        );

        let set = [
            ("main.wgsl", "#include \"a.wgsl\"\n"),
            ("a.wgsl", "#include \"b.wgsl\"\n"),
            ("b.wgsl", "fn b() {}\n#include \"a.wgsl\"\n"),
        ];
        let error = preprocess("main.wgsl", files(&set)).unwrap_err();
        assert_eq!(
            error,
            ShaderError::IncludeCycle {
                chain: ["a.wgsl", "b.wgsl", "a.wgsl"].map(str::to_owned).to_vec()
            }
        );
        assert_eq!(
            error.to_string(),
            "include cycle: a.wgsl -> b.wgsl -> a.wgsl"
        );
    }

    #[test]
    fn unknown_files_name_the_include_site() {
        let set = [("main.wgsl", "\n#include \"missing.wgsl\"\n")];
        let error = preprocess("main.wgsl", files(&set)).unwrap_err();
        assert_eq!(
            error,
            ShaderError::UnknownFile {
                path: "missing.wgsl".to_owned(),
                included_from: Some(SourceLine {
                    file: "main.wgsl".to_owned(),
                    line: 2
                }),
            }
        );
        assert!(error.to_string().contains("main.wgsl:2"), "{error}");

        let error = preprocess("nowhere.wgsl", files(&set)).unwrap_err();
        assert_eq!(
            error,
            ShaderError::UnknownFile {
                path: "nowhere.wgsl".to_owned(),
                included_from: None,
            }
        );
    }

    #[test]
    fn directives_other_than_include_are_errors() {
        for line in [
            "#define X 1",
            "#include",
            "#include common.wgsl",
            "#include \"unterminated.wgsl",
            "#includeX \"a.wgsl\"",
            "#include \"a.wgsl\" trailing",
            "  # include \"a.wgsl\"",
        ] {
            let text = format!("fn f() {{}}\n{line}\n");
            let set = [("main.wgsl", text.as_str()), ("a.wgsl", "")];
            match preprocess("main.wgsl", files(&set)) {
                Err(ShaderError::BadDirective { at, text }) => {
                    assert_eq!(at.to_string(), "main.wgsl:2", "{line}");
                    assert_eq!(text, line.trim(), "{line}");
                }
                other => panic!("{line:?} must be rejected, got {other:?}"),
            }
        }
        // Blanks before the directive and a trailing comment are fine.
        let set = [
            (
                "main.wgsl",
                "   #include   \"a.wgsl\"   // colour helpers\n",
            ),
            ("a.wgsl", "fn a() {}\n"),
        ];
        assert_eq!(
            preprocess("main.wgsl", files(&set)).unwrap().text(),
            "fn a() {}\n"
        );
    }

    #[test]
    fn include_paths_stay_inside_the_library() {
        for path in [
            "../escape.wgsl",
            "/absolute.wgsl",
            "common\\color.wgsl",
            "common//color.wgsl",
            "./color.wgsl",
            "color.txt",
            "",
            "with space.wgsl",
        ] {
            let text = format!("#include \"{path}\"\n");
            let set = [("main.wgsl", text.as_str())];
            match preprocess("main.wgsl", files(&set)) {
                Err(ShaderError::BadIncludePath { at, path: reported }) => {
                    assert_eq!(at.line, 1);
                    assert_eq!(reported, path);
                }
                other => panic!("{path:?} must be rejected, got {other:?}"),
            }
        }
        for path in ["a.wgsl", "common/color.wgsl", "x/y-z_1.v2.wgsl"] {
            assert!(is_library_path(path), "{path}");
        }
    }

    #[test]
    fn windows_line_endings_are_read_as_lines() {
        let set = [
            ("main.wgsl", "#include \"a.wgsl\"\r\nfn main() {}\r\n"),
            ("a.wgsl", "fn a() {}\r\n"),
        ];
        let source = preprocess("main.wgsl", files(&set)).unwrap();
        assert_eq!(source.text(), "fn a() {}\nfn main() {}\n");
    }
}
