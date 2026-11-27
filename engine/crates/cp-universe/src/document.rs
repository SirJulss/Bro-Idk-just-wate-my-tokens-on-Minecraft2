//! A strict reader for the subset of TOML that system files use.
//!
//! Contract §3.2 lists no TOML library for `cp-universe`, so the crate reads TOML itself.
//! Supported: comments; `[table]` and `[[array.of.tables]]` headers with bare or quoted keys,
//! including sub-tables of the last element of an array of tables (`[body.orbit]`); `key =
//! value` pairs with bare or quoted keys; single-line basic and literal strings; integers
//! (decimal, `0x`, `0o`, `0b`, underscores between digits); floats (fraction and/or exponent,
//! `inf`, `nan`); booleans; arrays (may span lines, trailing comma allowed); single-line inline
//! tables. The other TOML constructs are rejected as unsupported instead of being misread:
//! dotted keys in key/value pairs, multi-line strings, dates and times.
//!
//! Every key and value remembers its line, so the schema layer can name it in errors.

use core::fmt;

/// A TOML value.
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum Value {
    String(String),
    Integer(i64),
    Float(f64),
    Boolean(bool),
    Array(Vec<Item>),
    Table(Table),
    /// An array of tables, built by `[[key]]` headers; never empty.
    TableArray(Vec<Table>),
}

impl Value {
    /// "a string", "an integer", … for messages.
    pub(crate) fn type_name(&self) -> &'static str {
        match self {
            Self::String(_) => "a string",
            Self::Integer(_) => "an integer",
            Self::Float(_) => "a float",
            Self::Boolean(_) => "a boolean",
            Self::Array(_) => "an array",
            Self::Table(_) => "a table",
            Self::TableArray(_) => "an array of tables",
        }
    }
}

/// An element of an array, with the line it starts on.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Item {
    pub line: u32,
    pub value: Value,
}

/// A key of a table with its value and the line of the key.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Entry {
    pub key: String,
    pub line: u32,
    pub value: Value,
}

/// How a table came into existence; decides what may still be added to it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Origin {
    /// The document itself.
    Root,
    /// Defined by a `[header]`.
    Header,
    /// Created as the parent of a header (`[a.b]` creates `a`); may still get its own header.
    Implicit,
    /// An inline table `{ … }`; closed.
    Inline,
    /// An element of an array of tables.
    Element,
}

/// A TOML table: its entries in the order they were written.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Table {
    /// Line of the header that defined the table (1 for the document).
    pub line: u32,
    origin: Origin,
    pub entries: Vec<Entry>,
}

impl Table {
    fn new(line: u32, origin: Origin) -> Self {
        Self {
            line,
            origin,
            entries: Vec::new(),
        }
    }

    /// The entry with this key.
    pub(crate) fn get(&self, key: &str) -> Option<&Entry> {
        self.entries.iter().find(|entry| entry.key == key)
    }

    fn position(&self, key: &str) -> Option<usize> {
        self.entries.iter().position(|entry| entry.key == key)
    }
}

/// A syntax error or an unsupported construct, at a 1-based line and character column.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct SyntaxError {
    pub line: u32,
    pub column: u32,
    pub message: String,
}

impl fmt::Display for SyntaxError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}: {}", self.line, self.column, self.message)
    }
}

/// Parses a whole document into its root table.
pub(crate) fn parse(text: &str) -> Result<Table, SyntaxError> {
    let text = text.strip_prefix('\u{feff}').unwrap_or(text);
    Parser {
        chars: text.chars().collect(),
        pos: 0,
        line: 1,
        line_start: 0,
    }
    .document()
}

struct Parser {
    chars: Vec<char>,
    pos: usize,
    line: u32,
    line_start: usize,
}

type Parsed<T> = Result<T, SyntaxError>;

fn is_bare_key_char(c: char) -> bool {
    c.is_ascii_alphanumeric() || c == '_' || c == '-'
}

/// Characters that may appear in a number, a boolean or a date token.
fn is_token_char(c: char) -> bool {
    c.is_ascii_alphanumeric() || matches!(c, '_' | '+' | '-' | '.' | ':')
}

impl Parser {
    fn peek(&self) -> Option<char> {
        self.chars.get(self.pos).copied()
    }

    fn peek_at(&self, offset: usize) -> Option<char> {
        self.chars.get(self.pos + offset).copied()
    }

    fn starts_with(&self, text: &str) -> bool {
        text.chars()
            .enumerate()
            .all(|(offset, c)| self.peek_at(offset) == Some(c))
    }

    fn bump(&mut self) -> Option<char> {
        let c = self.peek()?;
        self.pos += 1;
        if c == '\n' {
            self.line += 1;
            self.line_start = self.pos;
        }
        Some(c)
    }

    fn column(&self) -> u32 {
        u32::try_from(self.pos - self.line_start + 1).unwrap_or(u32::MAX)
    }

    fn error_here(&self, message: impl Into<String>) -> SyntaxError {
        SyntaxError {
            line: self.line,
            column: self.column(),
            message: message.into(),
        }
    }

    fn skip_whitespace(&mut self) {
        while matches!(self.peek(), Some(' ' | '\t')) {
            self.bump();
        }
    }

    fn skip_comment(&mut self) {
        if self.peek() == Some('#') {
            while !matches!(self.peek(), None | Some('\n' | '\r')) {
                self.bump();
            }
        }
    }

    /// Consumes a line break (`\n` or `\r\n`); `false` when there is none here.
    fn line_break(&mut self) -> Parsed<bool> {
        match self.peek() {
            Some('\n') => {
                self.bump();
                Ok(true)
            }
            Some('\r') if self.peek_at(1) == Some('\n') => {
                self.bump();
                self.bump();
                Ok(true)
            }
            Some('\r') => Err(self.error_here("a carriage return must be followed by a line feed")),
            _ => Ok(false),
        }
    }

    /// After a header or a key/value pair: only whitespace and a comment may follow on the line.
    fn end_of_line(&mut self) -> Parsed<()> {
        self.skip_whitespace();
        self.skip_comment();
        if self.peek().is_none() || self.line_break()? {
            Ok(())
        } else {
            Err(self.error_here(format!(
                "expected the end of the line, found `{}`",
                self.peek().unwrap_or(' ')
            )))
        }
    }

    /// Skips whitespace, comments and line breaks (between array elements).
    fn skip_blank(&mut self) -> Parsed<()> {
        loop {
            self.skip_whitespace();
            self.skip_comment();
            if !self.line_break()? {
                return Ok(());
            }
        }
    }

    fn expect(&mut self, c: char, what: &str) -> Parsed<()> {
        if self.peek() == Some(c) {
            self.bump();
            Ok(())
        } else {
            Err(self.error_here(format!("expected {what}")))
        }
    }

    fn document(mut self) -> Parsed<Table> {
        let mut root = Table::new(1, Origin::Root);
        let mut current: Vec<String> = Vec::new();
        loop {
            self.skip_whitespace();
            match self.peek() {
                None => return Ok(root),
                Some('#') => {
                    self.skip_comment();
                    self.end_of_line()?;
                }
                Some('\n' | '\r') => {
                    self.line_break()?;
                }
                Some('[') => {
                    let (line, column) = (self.line, self.column());
                    let array = self.peek_at(1) == Some('[');
                    self.bump();
                    if array {
                        self.bump();
                    }
                    self.skip_whitespace();
                    let path = self.key_path()?;
                    self.skip_whitespace();
                    if array {
                        if !self.starts_with("]]") {
                            return Err(self.error_here("expected `]]` to close the header"));
                        }
                        self.bump();
                        self.bump();
                    } else {
                        self.expect(']', "`]` to close the header")?;
                    }
                    self.end_of_line()?;
                    let result = if array {
                        append_table(&mut root, &path, line)
                    } else {
                        define_table(&mut root, &path, line)
                    };
                    result.map_err(|message| SyntaxError {
                        line,
                        column,
                        message,
                    })?;
                    current = path;
                }
                Some(_) => {
                    let (line, column) = (self.line, self.column());
                    let (key, value) = self.key_value()?;
                    self.end_of_line()?;
                    insert(navigate(&mut root, &current), key, line, value).map_err(|message| {
                        SyntaxError {
                            line,
                            column,
                            message,
                        }
                    })?;
                }
            }
        }
    }

    /// `key = value`, positioned at the key.
    fn key_value(&mut self) -> Parsed<(String, Value)> {
        let key = self.key()?;
        self.skip_whitespace();
        if self.peek() == Some('.') {
            return Err(self.error_here(
                "dotted keys are not supported in key/value pairs; use a [table] header",
            ));
        }
        self.expect('=', "`=` after the key")?;
        self.skip_whitespace();
        let value = self.value()?;
        Ok((key, value))
    }

    fn key(&mut self) -> Parsed<String> {
        match self.peek() {
            Some('"') => self.basic_string(),
            Some('\'') => self.literal_string(),
            Some(c) if is_bare_key_char(c) => {
                let mut key = String::new();
                while let Some(c) = self.peek().filter(|&c| is_bare_key_char(c)) {
                    key.push(c);
                    self.bump();
                }
                Ok(key)
            }
            Some(c) => Err(self.error_here(format!("expected a key, found `{c}`"))),
            None => Err(self.error_here("expected a key, found the end of the file")),
        }
    }

    /// `key(.key)*` inside a header.
    fn key_path(&mut self) -> Parsed<Vec<String>> {
        let mut path = vec![self.key()?];
        loop {
            self.skip_whitespace();
            if self.peek() != Some('.') {
                return Ok(path);
            }
            self.bump();
            self.skip_whitespace();
            path.push(self.key()?);
        }
    }

    fn value(&mut self) -> Parsed<Value> {
        match self.peek() {
            Some('"') if self.starts_with("\"\"\"") => {
                Err(self.error_here("multi-line strings are not supported"))
            }
            Some('\'') if self.starts_with("'''") => {
                Err(self.error_here("multi-line strings are not supported"))
            }
            Some('"') => Ok(Value::String(self.basic_string()?)),
            Some('\'') => Ok(Value::String(self.literal_string()?)),
            Some('[') => self.array(),
            Some('{') => self.inline_table(),
            Some(c) if is_token_char(c) => self.scalar(),
            Some('\n' | '\r') => {
                Err(self.error_here("expected a value, found the end of the line"))
            }
            Some(c) => Err(self.error_here(format!("expected a value, found `{c}`"))),
            None => Err(self.error_here("expected a value, found the end of the file")),
        }
    }

    /// A basic string `"…"` with escapes, positioned at the opening quote.
    fn basic_string(&mut self) -> Parsed<String> {
        if self.starts_with("\"\"\"") {
            return Err(self.error_here("multi-line strings are not supported"));
        }
        self.bump();
        let mut text = String::new();
        loop {
            match self.peek() {
                None | Some('\n' | '\r') => return Err(self.error_here("unterminated string")),
                Some('"') => {
                    self.bump();
                    return Ok(text);
                }
                Some('\\') => {
                    self.bump();
                    text.push(self.escape()?);
                }
                Some(c) if is_forbidden_control(c) => {
                    return Err(self.error_here(format!(
                        "control character U+{:04X} in a string",
                        u32::from(c)
                    )));
                }
                Some(c) => {
                    text.push(c);
                    self.bump();
                }
            }
        }
    }

    /// The character after a backslash.
    fn escape(&mut self) -> Parsed<char> {
        let c = self
            .peek()
            .ok_or_else(|| self.error_here("unterminated string"))?;
        let simple = match c {
            'b' => Some('\u{8}'),
            't' => Some('\t'),
            'n' => Some('\n'),
            'f' => Some('\u{c}'),
            'r' => Some('\r'),
            '"' => Some('"'),
            '\\' => Some('\\'),
            _ => None,
        };
        if let Some(simple) = simple {
            self.bump();
            return Ok(simple);
        }
        let digits = match c {
            'u' => 4,
            'U' => 8,
            _ => return Err(self.error_here(format!("unknown escape sequence `\\{c}`"))),
        };
        self.bump();
        let mut code = 0_u32;
        for _ in 0..digits {
            let digit = self
                .peek()
                .and_then(|c| c.to_digit(16))
                .ok_or_else(|| self.error_here("expected a hexadecimal digit in a \\u escape"))?;
            code = code * 16 + digit;
            self.bump();
        }
        char::from_u32(code)
            .ok_or_else(|| self.error_here(format!("U+{code:X} is not a Unicode scalar value")))
    }

    /// A literal string `'…'`, positioned at the opening quote.
    fn literal_string(&mut self) -> Parsed<String> {
        if self.starts_with("'''") {
            return Err(self.error_here("multi-line strings are not supported"));
        }
        self.bump();
        let mut text = String::new();
        loop {
            match self.peek() {
                None | Some('\n' | '\r') => return Err(self.error_here("unterminated string")),
                Some('\'') => {
                    self.bump();
                    return Ok(text);
                }
                Some(c) if is_forbidden_control(c) => {
                    return Err(self.error_here(format!(
                        "control character U+{:04X} in a string",
                        u32::from(c)
                    )));
                }
                Some(c) => {
                    text.push(c);
                    self.bump();
                }
            }
        }
    }

    fn array(&mut self) -> Parsed<Value> {
        self.bump();
        let mut items = Vec::new();
        loop {
            self.skip_blank()?;
            if self.peek() == Some(']') {
                self.bump();
                return Ok(Value::Array(items));
            }
            let line = self.line;
            let value = self.value()?;
            items.push(Item { line, value });
            self.skip_blank()?;
            match self.peek() {
                Some(',') => {
                    self.bump();
                }
                Some(']') => {
                    self.bump();
                    return Ok(Value::Array(items));
                }
                _ => return Err(self.error_here("expected `,` or `]` in an array")),
            }
        }
    }

    fn inline_table(&mut self) -> Parsed<Value> {
        let line = self.line;
        self.bump();
        let mut table = Table::new(line, Origin::Inline);
        self.skip_whitespace();
        if self.peek() == Some('}') {
            self.bump();
            return Ok(Value::Table(table));
        }
        loop {
            self.skip_whitespace();
            let (key_line, key_column) = (self.line, self.column());
            let (key, value) = self.key_value()?;
            insert(&mut table, key, key_line, value).map_err(|message| SyntaxError {
                line: key_line,
                column: key_column,
                message,
            })?;
            self.skip_whitespace();
            match self.peek() {
                Some(',') => {
                    self.bump();
                    self.skip_whitespace();
                    if self.peek() == Some('}') {
                        return Err(self.error_here("trailing comma in an inline table"));
                    }
                }
                Some('}') => {
                    self.bump();
                    return Ok(Value::Table(table));
                }
                Some('\n' | '\r') | None => {
                    return Err(self.error_here("an inline table must end on the line it starts"));
                }
                Some(c) => {
                    return Err(self.error_here(format!(
                        "expected `,` or `}}` in an inline table, found `{c}`"
                    )));
                }
            }
        }
    }

    /// A number or a boolean.
    fn scalar(&mut self) -> Parsed<Value> {
        let (line, column) = (self.line, self.column());
        let mut token = String::new();
        while let Some(c) = self.peek().filter(|&c| is_token_char(c)) {
            token.push(c);
            self.bump();
        }
        scalar_value(&token).map_err(|message| SyntaxError {
            line,
            column,
            message,
        })
    }
}

/// Control characters other than tab may not appear in strings.
fn is_forbidden_control(c: char) -> bool {
    (c.is_control() && c != '\t') || c == '\u{7f}'
}

/// Interprets the token of a bare value.
fn scalar_value(token: &str) -> Result<Value, String> {
    match token {
        "true" => return Ok(Value::Boolean(true)),
        "false" => return Ok(Value::Boolean(false)),
        "inf" | "+inf" => return Ok(Value::Float(f64::INFINITY)),
        "-inf" => return Ok(Value::Float(f64::NEG_INFINITY)),
        "nan" | "+nan" | "-nan" => return Ok(Value::Float(f64::NAN)),
        _ => {}
    }
    if token.contains(':') || looks_like_date(token) {
        return Err(format!("`{token}`: dates and times are not supported"));
    }
    for (prefix, radix) in [("0x", 16), ("0o", 8), ("0b", 2)] {
        if let Some(digits) = token.strip_prefix(prefix) {
            if !grouped_digits(digits, radix) {
                return Err(format!("`{token}` is not a valid integer"));
            }
            return i64::from_str_radix(&digits.replace('_', ""), radix)
                .map(Value::Integer)
                .map_err(|_| format!("`{token}` does not fit into a 64-bit integer"));
        }
    }
    let unsigned = token.strip_prefix(['+', '-']).unwrap_or(token);
    if unsigned.contains(['.', 'e', 'E']) {
        if !is_toml_float(unsigned) {
            return Err(format!("`{token}` is not a valid number"));
        }
        // Rust's float parser rounds correctly on every platform.
        return token
            .replace('_', "")
            .parse::<f64>()
            .map(Value::Float)
            .map_err(|_| format!("`{token}` is not a valid number"));
    }
    if !is_decimal_integer(unsigned) {
        return Err(format!("`{token}` is not a valid value"));
    }
    token
        .replace('_', "")
        .parse::<i64>()
        .map(Value::Integer)
        .map_err(|_| format!("`{token}` does not fit into a 64-bit integer"))
}

/// `1979-05-27…`: four digits and a dash.
fn looks_like_date(token: &str) -> bool {
    let bytes = token.as_bytes();
    bytes.len() >= 5 && bytes[..4].iter().all(u8::is_ascii_digit) && bytes[4] == b'-'
}

/// Digits of `radix`, with single underscores only between digits.
fn grouped_digits(text: &str, radix: u32) -> bool {
    !text.is_empty()
        && !text.starts_with('_')
        && !text.ends_with('_')
        && !text.contains("__")
        && text.chars().all(|c| c == '_' || c.is_digit(radix))
}

/// A decimal integer without sign and without leading zeros.
fn is_decimal_integer(text: &str) -> bool {
    grouped_digits(text, 10) && (text == "0" || !text.starts_with('0'))
}

/// The unsigned part of a TOML float: an integer part, then a fraction, an exponent or both.
fn is_toml_float(text: &str) -> bool {
    let (mantissa, exponent) = match text.find(['e', 'E']) {
        Some(index) => (&text[..index], Some(&text[index + 1..])),
        None => (text, None),
    };
    let (integer, fraction) = match mantissa.split_once('.') {
        Some((integer, fraction)) => (integer, Some(fraction)),
        None => (mantissa, None),
    };
    if !is_decimal_integer(integer) || (fraction.is_none() && exponent.is_none()) {
        return false;
    }
    if fraction.is_some_and(|fraction| !grouped_digits(fraction, 10)) {
        return false;
    }
    exponent.is_none_or(|exponent| {
        grouped_digits(exponent.strip_prefix(['+', '-']).unwrap_or(exponent), 10)
    })
}

/// The table that key/value pairs after the header `path` go into.
fn navigate<'t>(root: &'t mut Table, path: &[String]) -> &'t mut Table {
    let mut table = root;
    for key in path {
        let index = table
            .position(key)
            .expect("the header that set the current path created it");
        table = match &mut table.entries[index].value {
            Value::Table(child) => child,
            Value::TableArray(list) => list
                .last_mut()
                .expect("arrays of tables are created with one element"),
            _ => unreachable!("the header that set the current path checked it"),
        };
    }
    table
}

/// The table `key` inside `table`, created implicitly when missing; for an array of tables its
/// last element. `name` is the dotted path so far, for messages.
fn descend<'t>(
    table: &'t mut Table,
    key: &str,
    line: u32,
    name: &str,
) -> Result<&'t mut Table, String> {
    let index = match table.position(key) {
        Some(index) => index,
        None => {
            table.entries.push(Entry {
                key: key.to_owned(),
                line,
                value: Value::Table(Table::new(line, Origin::Implicit)),
            });
            table.entries.len() - 1
        }
    };
    let entry_line = table.entries[index].line;
    match &mut table.entries[index].value {
        Value::Table(child) if child.origin == Origin::Inline => Err(format!(
            "`{name}` is an inline table (line {entry_line}) and cannot be extended"
        )),
        Value::Table(child) => Ok(child),
        Value::TableArray(list) => Ok(list
            .last_mut()
            .expect("arrays of tables are created with one element")),
        other => Err(format!(
            "`{name}` is {} (line {entry_line}), not a table",
            other.type_name()
        )),
    }
}

/// Walks all but the last key of a header path, creating implicit tables.
fn header_parent<'t>(
    root: &'t mut Table,
    path: &[String],
    line: u32,
) -> Result<(&'t mut Table, String), String> {
    let (last, prefix) = path.split_last().expect("a header has at least one key");
    let mut table = root;
    for (depth, key) in prefix.iter().enumerate() {
        table = descend(table, key, line, &path[..=depth].join("."))?;
    }
    Ok((table, last.clone()))
}

/// `[path]`.
fn define_table(root: &mut Table, path: &[String], line: u32) -> Result<(), String> {
    let name = path.join(".");
    let (table, last) = header_parent(root, path, line)?;
    let Some(index) = table.position(&last) else {
        table.entries.push(Entry {
            key: last,
            line,
            value: Value::Table(Table::new(line, Origin::Header)),
        });
        return Ok(());
    };
    let entry_line = table.entries[index].line;
    match &mut table.entries[index].value {
        Value::Table(child) if child.origin == Origin::Implicit => {
            child.origin = Origin::Header;
            child.line = line;
            Ok(())
        }
        Value::Table(child) if child.origin == Origin::Inline => Err(format!(
            "`{name}` is an inline table (line {entry_line}) and cannot be extended"
        )),
        Value::Table(_) => Err(format!(
            "table `{name}` is defined twice (first on line {entry_line})"
        )),
        Value::TableArray(_) => Err(format!(
            "`{name}` is an array of tables (line {entry_line}); write `[[{name}]]`"
        )),
        other => Err(format!(
            "`{name}` already has a value, {} (line {entry_line})",
            other.type_name()
        )),
    }
}

/// `[[path]]`.
fn append_table(root: &mut Table, path: &[String], line: u32) -> Result<(), String> {
    let name = path.join(".");
    let (table, last) = header_parent(root, path, line)?;
    let Some(index) = table.position(&last) else {
        table.entries.push(Entry {
            key: last,
            line,
            value: Value::TableArray(vec![Table::new(line, Origin::Element)]),
        });
        return Ok(());
    };
    let entry_line = table.entries[index].line;
    match &mut table.entries[index].value {
        Value::TableArray(list) => {
            list.push(Table::new(line, Origin::Element));
            Ok(())
        }
        other => Err(format!(
            "`{name}` is {} (line {entry_line}), not an array of tables",
            other.type_name()
        )),
    }
}

/// Adds `key = value` to `table`.
fn insert(table: &mut Table, key: String, line: u32, value: Value) -> Result<(), String> {
    if let Some(existing) = table.get(&key) {
        return Err(format!(
            "duplicate key `{key}` (first defined on line {})",
            existing.line
        ));
    }
    table.entries.push(Entry { key, line, value });
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn value<'t>(table: &'t Table, key: &str) -> &'t Value {
        &table
            .get(key)
            .unwrap_or_else(|| panic!("no key {key}"))
            .value
    }

    fn table<'t>(table: &'t Table, key: &str) -> &'t Table {
        match value(table, key) {
            Value::Table(table) => table,
            other => panic!("{key} is {other:?}"),
        }
    }

    fn error(text: &str) -> SyntaxError {
        parse(text).expect_err("the document must be rejected")
    }

    #[test]
    fn scalars() {
        let doc = parse(
            "a = \"x\\ty\\u00e9\\U0001F600\\\"\\\\\"\nb = 'C:\\path'\nc = 1_000\nd = -17\n\
             e = 0xff_ff\nf = 0o17\ng = 0b101\nh = 1.5e-3\ni = -2E+2\nj = 6.0_25\nk = inf\n\
             l = -inf\nm = nan\nn = true\no = false\np = +0\nq = 0.0\nr = 1e5\n",
        )
        .unwrap();
        assert_eq!(value(&doc, "a"), &Value::String("x\tyé😀\"\\".to_owned()));
        assert_eq!(value(&doc, "b"), &Value::String("C:\\path".to_owned()));
        assert_eq!(value(&doc, "c"), &Value::Integer(1000));
        assert_eq!(value(&doc, "d"), &Value::Integer(-17));
        assert_eq!(value(&doc, "e"), &Value::Integer(0xffff));
        assert_eq!(value(&doc, "f"), &Value::Integer(0o17));
        assert_eq!(value(&doc, "g"), &Value::Integer(5));
        assert_eq!(value(&doc, "h"), &Value::Float(1.5e-3));
        assert_eq!(value(&doc, "i"), &Value::Float(-200.0));
        assert_eq!(value(&doc, "j"), &Value::Float(6.025));
        assert_eq!(value(&doc, "k"), &Value::Float(f64::INFINITY));
        assert_eq!(value(&doc, "l"), &Value::Float(f64::NEG_INFINITY));
        assert!(matches!(value(&doc, "m"), Value::Float(x) if x.is_nan()));
        assert_eq!(value(&doc, "n"), &Value::Boolean(true));
        assert_eq!(value(&doc, "o"), &Value::Boolean(false));
        assert_eq!(value(&doc, "p"), &Value::Integer(0));
        assert_eq!(value(&doc, "q"), &Value::Float(0.0));
        assert_eq!(value(&doc, "r"), &Value::Float(1.0e5));
        // Lines are remembered per key.
        assert_eq!(doc.get("a").unwrap().line, 1);
        assert_eq!(doc.get("r").unwrap().line, 18);
    }

    #[test]
    fn tables_arrays_and_comments() {
        let doc = parse(
            "# A comment\n\
             title = \"x\" # trailing\n\
             \n\
             [system]\n\
             key = \"a:b\"\n\
             [[body]]\n\
             key = \"one\"\n\
             [body.orbit]\n\
             a = 1.0\n\
             [[body]]\n\
             key = \"two\"\n\
             list = [ 1, 2.5, # inside\n\
             \"three\", [4], ]\n\
             inline = { x = 1, 'y z' = \"w\" }\n\
             empty = []\n\
             [\"quoted key\" . sub]\n\
             q = 1\n",
        )
        .unwrap();
        assert_eq!(value(&doc, "title"), &Value::String("x".to_owned()));
        let system = table(&doc, "system");
        assert_eq!(system.line, 4);
        assert_eq!(value(system, "key"), &Value::String("a:b".to_owned()));
        let Value::TableArray(bodies) = value(&doc, "body") else {
            panic!("body is not an array of tables");
        };
        assert_eq!(bodies.len(), 2);
        assert_eq!((bodies[0].line, bodies[1].line), (6, 10));
        let orbit = table(&bodies[0], "orbit");
        assert_eq!(value(orbit, "a"), &Value::Float(1.0));
        assert!(bodies[1].get("orbit").is_none());
        let Value::Array(list) = value(&bodies[1], "list") else {
            panic!("list is not an array");
        };
        assert_eq!(list.len(), 4);
        assert_eq!(list[2].value, Value::String("three".to_owned()));
        assert_eq!(list[2].line, 13);
        assert_eq!(
            list[3].value,
            Value::Array(vec![Item {
                line: 13,
                value: Value::Integer(4)
            }])
        );
        let inline = table(&bodies[1], "inline");
        assert_eq!(value(inline, "y z"), &Value::String("w".to_owned()));
        assert_eq!(value(&bodies[1], "empty"), &Value::Array(Vec::new()));
        let quoted = table(table(&doc, "quoted key"), "sub");
        assert_eq!(value(quoted, "q"), &Value::Integer(1));
    }

    #[test]
    fn implicit_tables_may_be_defined_once() {
        let doc = parse("[a.b]\nx = 1\n[a]\ny = 2\n").unwrap();
        let a = table(&doc, "a");
        assert_eq!(a.line, 3);
        assert_eq!(value(table(a, "b"), "x"), &Value::Integer(1));
        assert_eq!(value(a, "y"), &Value::Integer(2));
        let twice = error("[a.b]\n[a]\n[a]\n");
        assert_eq!(twice.line, 3);
        assert!(twice.message.contains("defined twice"), "{twice:?}");
    }

    #[test]
    fn crlf_and_byte_order_mark_are_accepted() {
        let doc = parse("\u{feff}a = 1\r\n[t]\r\nb = 'x'\r\n").unwrap();
        assert_eq!(value(&doc, "a"), &Value::Integer(1));
        assert_eq!(value(table(&doc, "t"), "b"), &Value::String("x".to_owned()));
        assert_eq!(table(&doc, "t").line, 2);
    }

    #[test]
    fn errors_have_positions() {
        let cases: &[(&str, u32, u32, &str)] = &[
            (
                "a = 1\na = 2\n",
                2,
                1,
                "duplicate key `a` (first defined on line 1)",
            ),
            ("[t]\n[t]\n", 2, 1, "defined twice"),
            ("a = \"open\n", 1, 10, "unterminated string"),
            ("a = \"\\q\"\n", 1, 7, "unknown escape"),
            ("a = \"\"\"x\"\"\"\n", 1, 5, "multi-line strings"),
            ("a = '''x'''\n", 1, 5, "multi-line strings"),
            ("a.b = 1\n", 1, 2, "dotted keys"),
            ("d = 1979-05-27\n", 1, 5, "dates and times"),
            ("t = 07:32:00\n", 1, 5, "dates and times"),
            ("n = 012\n", 1, 5, "not a valid value"),
            ("n = 1__0\n", 1, 5, "not a valid value"),
            ("n = 1.\n", 1, 5, "not a valid number"),
            ("n = .5\n", 1, 5, "not a valid number"),
            ("n = 1e\n", 1, 5, "not a valid number"),
            ("n = 99999999999999999999\n", 1, 5, "64-bit"),
            ("a = 1 b = 2\n", 1, 7, "expected the end of the line"),
            ("a = { x = 1, }\n", 1, 14, "trailing comma"),
            ("a = { x = 1\n", 1, 12, "must end on the line"),
            ("a = [1 2]\n", 1, 8, "expected `,` or `]`"),
            ("a = { x = 1 }\n[a]\n", 2, 1, "inline table"),
            ("a = { x = 1 }\n[a.b]\n", 2, 1, "inline table"),
            ("a = [ { x = 1 } ]\n[[a]]\n", 2, 1, "not an array of tables"),
            ("a = 1\n[a.b]\n", 2, 1, "not a table"),
            ("[[a]]\n[a]\n", 2, 1, "array of tables"),
            ("[a\n", 1, 3, "`]`"),
            ("[[a]\n", 1, 4, "`]]`"),
            ("= 1\n", 1, 1, "expected a key"),
            ("a 1\n", 1, 3, "`=`"),
            ("a =\n", 1, 4, "expected a value"),
            ("a = \"x\ty\u{1}\"\n", 1, 9, "control character"),
            ("a = 1\r", 1, 6, "carriage return"),
        ];
        for &(text, line, column, message) in cases {
            let error = error(text);
            assert!(
                error.message.contains(message),
                "{text:?}: expected {message:?}, got {error:?}"
            );
            assert_eq!(
                (error.line, error.column),
                (line, column),
                "{text:?}: {error}"
            );
        }
    }

    #[test]
    fn sub_tables_follow_the_last_array_element() {
        let doc = parse("[[a]]\n[a.b]\nx = 1\n[[a]]\n[a.b]\nx = 2\n").unwrap();
        let Value::TableArray(list) = value(&doc, "a") else {
            panic!("a is not an array of tables");
        };
        assert_eq!(value(table(&list[0], "b"), "x"), &Value::Integer(1));
        assert_eq!(value(table(&list[1], "b"), "x"), &Value::Integer(2));
        // The same sub-table twice inside one element is a duplicate.
        let twice = error("[[a]]\n[a.b]\n[a.b]\n");
        assert_eq!(twice.line, 3);
    }
}
