//! Determinism rule 1 of contract §2.5 is enforced by clippy: `engine/clippy.toml` must list
//! every method the rule forbids, for `f64` and for `f32`, in `disallowed-methods`.
//!
//! The file is parsed, not searched, so that a commented-out entry or one under another key
//! does not count. `cp-core` has no TOML dependency; the small parser below reads the part of
//! TOML that `clippy.toml` uses and refuses everything else, so that the test fails loudly
//! instead of overlooking an entry it cannot read.

use std::collections::BTreeSet;
use std::fs;
use std::path::Path;

/// Contract §2.5 rule 1 (v2): the transcendental functions that the deterministic crates must
/// not call on `f64` and on `f32` (`f64::{…}` "and the `f32` equivalents").
const TRANSCENDENTAL: [&str; 20] = [
    "sin", "cos", "tan", "asin", "acos", "atan", "atan2", "sinh", "cosh", "tanh", "exp", "exp2",
    "ln", "log2", "log10", "powf", "powi", "cbrt", "hypot", "mul_add",
];

/// Contract §2.5 rule 1 (v2): the comparisons that the deterministic crates make through
/// `det::{min, max, clamp}` only.
const COMPARISONS: [&str; 3] = ["min", "max", "clamp"];

/// The float types of the rule.
const FLOATS: [&str; 2] = ["f64", "f32"];

/// A value of the TOML subset that `clippy.toml` uses.
#[derive(Clone, Debug, PartialEq)]
enum Value {
    Boolean(bool),
    Integer(i64),
    String(String),
    Array(Vec<Value>),
    /// An inline table, keys in the order written.
    Table(Vec<(String, Value)>),
}

/// Reads top-level `key = value` lines whose values are booleans, integers, single-line
/// strings, arrays and inline tables, with comments and blank lines in between. Table
/// headers, dotted keys, floats, dates and multi-line strings are refused.
struct Parser<'a> {
    text: &'a str,
    /// Byte offset of the next character.
    position: usize,
}

impl<'a> Parser<'a> {
    /// The top-level entries of `text` in the order written.
    fn document(text: &'a str) -> Result<Vec<(String, Value)>, String> {
        let mut parser = Parser { text, position: 0 };
        let mut entries: Vec<(String, Value)> = Vec::new();
        loop {
            parser.skip_blank(true);
            match parser.peek() {
                None => return Ok(entries),
                Some('[') => return Err(parser.error("table headers are not supported")),
                Some(_) => {}
            }
            let key = parser.key()?;
            if entries.iter().any(|(existing, _)| *existing == key) {
                return Err(parser.error(&format!("`{key}` is set twice")));
            }
            parser.skip_blank(false);
            parser.expect('=')?;
            parser.skip_blank(false);
            let value = parser.value()?;
            parser.skip_blank(false);
            match parser.peek() {
                None | Some('\n') => {}
                Some('\r') if parser.rest().starts_with("\r\n") => {}
                Some(other) => {
                    return Err(parser.error(&format!("`{other}` after a value")));
                }
            }
            entries.push((key, value));
        }
    }

    fn rest(&self) -> &'a str {
        &self.text[self.position..]
    }

    fn peek(&self) -> Option<char> {
        self.rest().chars().next()
    }

    fn bump(&mut self) -> Option<char> {
        let next = self.peek()?;
        self.position += next.len_utf8();
        Some(next)
    }

    /// `message` with the line and column of the next character.
    fn error(&self, message: &str) -> String {
        let before = &self.text[..self.position];
        let line = before.matches('\n').count() + 1;
        let column = before
            .rsplit('\n')
            .next()
            .map_or(0, |part| part.chars().count())
            + 1;
        format!("{line}:{column}: {message}")
    }

    fn expect(&mut self, wanted: char) -> Result<(), String> {
        match self.bump() {
            Some(found) if found == wanted => Ok(()),
            found => Err(self.error(&format!("expected `{wanted}`, found {found:?}"))),
        }
    }

    /// Skips spaces, tabs and comments, and line breaks too if `newlines`.
    fn skip_blank(&mut self, newlines: bool) {
        while let Some(next) = self.peek() {
            match next {
                ' ' | '\t' => {}
                '\n' | '\r' if newlines => {}
                '#' => {
                    let line_end = self.rest().find('\n').unwrap_or(self.rest().len());
                    self.position += line_end;
                    continue;
                }
                _ => return,
            }
            self.bump();
        }
    }

    fn key(&mut self) -> Result<String, String> {
        if self.peek() == Some('"') {
            return self.basic_string();
        }
        let length = self
            .rest()
            .find(|c: char| !(c.is_ascii_alphanumeric() || c == '_' || c == '-'))
            .unwrap_or(self.rest().len());
        if length == 0 {
            return Err(self.error("expected a key"));
        }
        let key = self.rest()[..length].to_owned();
        self.position += length;
        if self.peek() == Some('.') {
            return Err(self.error("dotted keys are not supported"));
        }
        Ok(key)
    }

    fn value(&mut self) -> Result<Value, String> {
        match self.peek() {
            Some('"') if self.rest().starts_with("\"\"\"") => {
                Err(self.error("multi-line strings are not supported"))
            }
            Some('"') => self.basic_string().map(Value::String),
            Some('\'') if self.rest().starts_with("'''") => {
                Err(self.error("multi-line strings are not supported"))
            }
            Some('\'') => self.literal_string().map(Value::String),
            Some('[') => self.array(),
            Some('{') => self.inline_table(),
            Some('t' | 'f') => self.boolean(),
            Some('+' | '-' | '0'..='9') => self.integer(),
            other => Err(self.error(&format!("expected a value, found {other:?}"))),
        }
    }

    fn basic_string(&mut self) -> Result<String, String> {
        self.expect('"')?;
        let mut text = String::new();
        loop {
            match self.bump() {
                None | Some('\n') => return Err(self.error("unterminated string")),
                Some('"') => return Ok(text),
                Some('\\') => {
                    let escaped = match self.bump() {
                        Some('"') => '"',
                        Some('\\') => '\\',
                        Some('n') => '\n',
                        Some('t') => '\t',
                        Some('r') => '\r',
                        Some('b') => '\u{8}',
                        Some('f') => '\u{c}',
                        Some(unicode @ ('u' | 'U')) => self.unicode_escape(unicode)?,
                        other => return Err(self.error(&format!("unknown escape {other:?}"))),
                    };
                    text.push(escaped);
                }
                Some(other) => text.push(other),
            }
        }
    }

    /// The character of a `\uXXXX` or `\UXXXXXXXX` escape whose letter has been read.
    fn unicode_escape(&mut self, letter: char) -> Result<char, String> {
        let digits = if letter == 'u' { 4 } else { 8 };
        let hex = self.rest().get(..digits).unwrap_or_default();
        let code = u32::from_str_radix(hex, 16)
            .ok()
            .filter(|_| hex.len() == digits)
            .and_then(char::from_u32);
        match code {
            Some(character) => {
                self.position += digits;
                Ok(character)
            }
            None => Err(self.error("invalid unicode escape")),
        }
    }

    fn literal_string(&mut self) -> Result<String, String> {
        self.expect('\'')?;
        let length = self.rest().find(['\'', '\n']).unwrap_or(self.rest().len());
        if !self.rest()[length..].starts_with('\'') {
            return Err(self.error("unterminated string"));
        }
        let text = self.rest()[..length].to_owned();
        self.position += length + 1;
        Ok(text)
    }

    fn array(&mut self) -> Result<Value, String> {
        self.expect('[')?;
        let mut items = Vec::new();
        loop {
            self.skip_blank(true);
            if self.peek() == Some(']') {
                self.bump();
                return Ok(Value::Array(items));
            }
            items.push(self.value()?);
            self.skip_blank(true);
            match self.bump() {
                Some(',') => {}
                Some(']') => return Ok(Value::Array(items)),
                other => return Err(self.error(&format!("expected `,` or `]`, found {other:?}"))),
            }
        }
    }

    fn inline_table(&mut self) -> Result<Value, String> {
        self.expect('{')?;
        let mut fields: Vec<(String, Value)> = Vec::new();
        self.skip_blank(false);
        if self.peek() == Some('}') {
            self.bump();
            return Ok(Value::Table(fields));
        }
        loop {
            self.skip_blank(false);
            let key = self.key()?;
            if fields.iter().any(|(existing, _)| *existing == key) {
                return Err(self.error(&format!("`{key}` is set twice")));
            }
            self.skip_blank(false);
            self.expect('=')?;
            self.skip_blank(false);
            fields.push((key, self.value()?));
            self.skip_blank(false);
            match self.bump() {
                Some(',') => {}
                Some('}') => return Ok(Value::Table(fields)),
                other => return Err(self.error(&format!("expected `,` or `}}`, found {other:?}"))),
            }
        }
    }

    fn boolean(&mut self) -> Result<Value, String> {
        for (word, value) in [("true", true), ("false", false)] {
            if self.rest().starts_with(word) {
                self.position += word.len();
                return Ok(Value::Boolean(value));
            }
        }
        Err(self.error("expected `true` or `false`"))
    }

    fn integer(&mut self) -> Result<Value, String> {
        let length = self
            .rest()
            .char_indices()
            .find(|&(index, c)| {
                !(c.is_ascii_digit() || c == '_' || (index == 0 && "+-".contains(c)))
            })
            .map_or(self.rest().len(), |(index, _)| index);
        let written = &self.rest()[..length];
        if self.rest()[length..].starts_with(['.', 'e', 'E', ':', 'x', 'o', 'b']) {
            return Err(self.error("only decimal integers are supported"));
        }
        let number = written
            .replace('_', "")
            .parse::<i64>()
            .map_err(|error| self.error(&format!("`{written}`: {error}")))?;
        self.position += length;
        Ok(Value::Integer(number))
    }
}

/// `engine/clippy.toml`, parsed.
fn clippy_toml() -> Vec<(String, Value)> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../clippy.toml");
    let text = fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("{} cannot be read: {error}", path.display()));
    Parser::document(&text).unwrap_or_else(|error| panic!("clippy.toml:{error}"))
}

/// The paths of the `disallowed-methods` of a parsed `clippy.toml`, in the order written. An
/// entry is a path string or a table with a `path` string.
fn disallowed_methods(document: &[(String, Value)]) -> Vec<String> {
    let (_, list) = document
        .iter()
        .find(|(key, _)| key == "disallowed-methods")
        .expect("clippy.toml sets `disallowed-methods`");
    let Value::Array(entries) = list else {
        panic!("`disallowed-methods` is not an array: {list:?}");
    };
    entries
        .iter()
        .map(|entry| match entry {
            Value::String(path) => path.clone(),
            Value::Table(fields) => match fields.iter().find(|(key, _)| key == "path") {
                Some((_, Value::String(path))) => path.clone(),
                _ => panic!("an entry of `disallowed-methods` has no `path` string: {entry:?}"),
            },
            other => panic!("unexpected entry in `disallowed-methods`: {other:?}"),
        })
        .collect()
}

/// Every `f64::…` and `f32::…` path that rule 1 forbids.
fn rule_1_paths() -> Vec<String> {
    FLOATS
        .iter()
        .flat_map(|float| {
            TRANSCENDENTAL
                .iter()
                .chain(&COMPARISONS)
                .map(move |method| format!("{float}::{method}"))
        })
        .collect()
}

/// The paths of [`rule_1_paths`] that `forbidden` lacks, in that order.
fn missing_from(forbidden: &BTreeSet<String>) -> Vec<String> {
    rule_1_paths()
        .into_iter()
        .filter(|path| !forbidden.contains(path))
        .collect()
}

#[test]
fn clippy_forbids_every_method_of_rule_1_on_f64_and_f32() {
    assert_eq!(rule_1_paths().len(), 2 * (20 + 3));
    let forbidden: BTreeSet<String> = disallowed_methods(&clippy_toml()).into_iter().collect();
    let missing = missing_from(&forbidden);
    assert!(
        missing.is_empty(),
        "clippy.toml lets the deterministic crates call {missing:?} (contract §2.5 rule 1)"
    );
}

#[test]
fn a_method_missing_from_the_list_is_reported() {
    let mut forbidden: BTreeSet<String> = rule_1_paths().into_iter().collect();
    assert!(missing_from(&forbidden).is_empty());
    for path in ["f32::hypot", "f64::clamp"] {
        assert!(forbidden.remove(path));
    }
    assert_eq!(missing_from(&forbidden), ["f64::clamp", "f32::hypot"]);
}

#[test]
fn clippy_forbids_the_same_float_methods_on_f64_and_f32() {
    let methods = disallowed_methods(&clippy_toml());
    let unique: BTreeSet<&String> = methods.iter().collect();
    assert_eq!(unique.len(), methods.len(), "an entry is listed twice");
    let names = |float: &str| -> BTreeSet<String> {
        let prefix = format!("{float}::");
        methods
            .iter()
            .filter_map(|path| path.strip_prefix(&prefix))
            .map(str::to_owned)
            .collect()
    };
    assert_eq!(names("f64"), names("f32"));
    assert!(names("f64").len() >= TRANSCENDENTAL.len() + COMPARISONS.len());
}

#[test]
fn transcription_of_rule_1_matches_the_contract() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../docs/ARCHITEKTUR.md");
    let contract = fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("{} cannot be read: {error}", path.display()));
    let section_start = contract
        .find("### 2.5")
        .expect("the contract has a section 2.5");
    let section = &contract[section_start..];
    let section = &section[..section[1..]
        .find("\n#")
        .map_or(section.len(), |end| end + 1)];
    let rule_start = section.find("\n1. ").expect("section 2.5 has a rule 1");
    let rule = &section[rule_start..];
    let rule = &rule[..rule[1..].find("\n2. ").map_or(rule.len(), |end| end + 1)];
    // One line, so that lists broken over several lines read as written.
    let rule = rule.split_whitespace().collect::<Vec<_>>().join(" ");

    let listed = |opening: &str| -> Vec<String> {
        let start = rule.find(opening).unwrap_or_else(|| {
            panic!("rule 1 of contract §2.5 no longer lists `{opening}…}}`; update this test")
        }) + opening.len();
        let end = start + rule[start..].find("}`").expect("the list is closed");
        rule[start..end]
            .split(',')
            .map(|name| name.trim().to_owned())
            .collect()
    };
    assert_eq!(
        listed("`f64::{"),
        TRANSCENDENTAL,
        "the transcendental functions"
    );
    assert!(
        rule.contains("`f32` equivalents"),
        "rule 1 no longer names the f32 equivalents: {rule}"
    );
    assert_eq!(listed("`det::{"), COMPARISONS, "the comparisons");
}

#[test]
fn parser_reads_the_subset_that_clippy_toml_uses() {
    let text = "\
# A comment with \"quotes\", ] and = in it.
allow-unwrap-in-tests = true   # trailing comment
threshold = -1_000
\"quoted key\" = 'C:\\literal'
list = [
    \"a # not a comment\",
    { path = \"f64::sin\", reason = \"say \\\"why\\\" \\u00e4\" },
    # { path = \"f64::cos\" },
    [1, 2,],
    {},
]
";
    let document = Parser::document(text).expect("the sample is valid");
    assert_eq!(
        document,
        [
            ("allow-unwrap-in-tests".to_owned(), Value::Boolean(true)),
            ("threshold".to_owned(), Value::Integer(-1000)),
            (
                "quoted key".to_owned(),
                Value::String("C:\\literal".to_owned())
            ),
            (
                "list".to_owned(),
                Value::Array(vec![
                    Value::String("a # not a comment".to_owned()),
                    Value::Table(vec![
                        ("path".to_owned(), Value::String("f64::sin".to_owned())),
                        (
                            "reason".to_owned(),
                            Value::String("say \"why\" \u{e4}".to_owned())
                        ),
                    ]),
                    Value::Array(vec![Value::Integer(1), Value::Integer(2)]),
                    Value::Table(Vec::new()),
                ]),
            ),
        ]
    );
    // Windows line endings read the same.
    assert_eq!(
        Parser::document(&text.replace('\n', "\r\n")).expect("the sample is valid"),
        document
    );

    // Entries that are commented out, or listed under another key, do not count.
    let methods = Parser::document(
        "\
disallowed-types = [\"f64::exp\"]
disallowed-methods = [
    \"f32::tan\",
    # \"f64::cos\",
    { path = \"f64::sin\", reason = \"#1\" }, # { path = \"f64::ln\" },
]
",
    )
    .expect("the sample is valid");
    assert_eq!(disallowed_methods(&methods), ["f32::tan", "f64::sin"]);
}

#[test]
fn parser_refuses_what_it_does_not_understand() {
    for text in [
        "[table]\nkey = 1\n",
        "a.b = 1\n",
        "key = 1.5\n",
        "key = 1e3\n",
        "key = 0x10\n",
        "key = 1979-05-27\n",
        "key = \"\"\"multi\nline\"\"\"\n",
        "key = '''multi'''\n",
        "key = \"unterminated\n",
        "key = \"bad \\q escape\"\n",
        "key = [1, 2\n",
        "key = { a = 1 b = 2 }\n",
        "key = { a = 1, a = 2 }\n",
        "key = 1\nkey = 2\n",
        "key = 1 2\n",
        "key = yes\n",
        "= 1\n",
    ] {
        assert!(Parser::document(text).is_err(), "accepted {text:?}");
    }
}
