//! A minimal JSON document model and writer for the capability report.
//!
//! The report has to be machine-readable and the crate has no JSON dependency (contract §3.2),
//! so this module provides the little that is needed: a value tree and a pretty-printing writer
//! with string escaping after RFC 8259. Numbers are unsigned integers because the report
//! contains no others.

/// A JSON value. Objects keep their insertion order, so the output is deterministic.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum Json {
    Null,
    Bool(bool),
    Number(u64),
    String(String),
    Array(Vec<Json>),
    /// Keys must be unique; the writer does not check it, the report tests do.
    Object(Vec<(String, Json)>),
}

impl Json {
    /// An object from `(key, value)` pairs, in the given order.
    pub(crate) fn object<K: Into<String>>(entries: impl IntoIterator<Item = (K, Json)>) -> Self {
        Self::Object(
            entries
                .into_iter()
                .map(|(key, value)| (key.into(), value))
                .collect(),
        )
    }

    /// An array of strings.
    pub(crate) fn strings<S: Into<String>>(items: impl IntoIterator<Item = S>) -> Self {
        Self::Array(
            items
                .into_iter()
                .map(|item| Self::String(item.into()))
                .collect(),
        )
    }

    /// The document as indented text (two spaces per level, `\n` line ends, final newline).
    pub(crate) fn to_pretty_string(&self) -> String {
        let mut out = String::new();
        self.write(&mut out, 0);
        out.push('\n');
        out
    }

    fn write(&self, out: &mut String, depth: usize) {
        match self {
            Self::Null => out.push_str("null"),
            Self::Bool(true) => out.push_str("true"),
            Self::Bool(false) => out.push_str("false"),
            Self::Number(number) => out.push_str(&number.to_string()),
            Self::String(text) => write_string(out, text),
            Self::Array(items) => {
                if items.is_empty() {
                    out.push_str("[]");
                    return;
                }
                out.push('[');
                for (index, item) in items.iter().enumerate() {
                    out.push_str(if index == 0 { "\n" } else { ",\n" });
                    indent(out, depth + 1);
                    item.write(out, depth + 1);
                }
                out.push('\n');
                indent(out, depth);
                out.push(']');
            }
            Self::Object(entries) => {
                if entries.is_empty() {
                    out.push_str("{}");
                    return;
                }
                out.push('{');
                for (index, (key, value)) in entries.iter().enumerate() {
                    out.push_str(if index == 0 { "\n" } else { ",\n" });
                    indent(out, depth + 1);
                    write_string(out, key);
                    out.push_str(": ");
                    value.write(out, depth + 1);
                }
                out.push('\n');
                indent(out, depth);
                out.push('}');
            }
        }
    }
}

impl From<bool> for Json {
    fn from(value: bool) -> Self {
        Self::Bool(value)
    }
}

impl From<u32> for Json {
    fn from(value: u32) -> Self {
        Self::Number(u64::from(value))
    }
}

impl From<u64> for Json {
    fn from(value: u64) -> Self {
        Self::Number(value)
    }
}

impl From<&str> for Json {
    fn from(value: &str) -> Self {
        Self::String(value.to_owned())
    }
}

impl From<String> for Json {
    fn from(value: String) -> Self {
        Self::String(value)
    }
}

fn indent(out: &mut String, depth: usize) {
    for _ in 0..depth {
        out.push_str("  ");
    }
}

/// Writes `text` as a JSON string literal.
///
/// Quotation mark, backslash and all control characters below U+0020 are escaped, as RFC 8259
/// requires; everything else, including non-ASCII text, is written as UTF-8.
fn write_string(out: &mut String, text: &str) {
    out.push('"');
    for character in text.chars() {
        match character {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{08}' => out.push_str("\\b"),
            '\u{0C}' => out.push_str("\\f"),
            control if control < '\u{20}' => {
                out.push_str(&format!("\\u{:04x}", u32::from(control)));
            }
            other => out.push(other),
        }
    }
    out.push('"');
}

/// A strict JSON reader, used by the tests to prove that the writer's output is valid JSON and
/// to inspect reports structurally.
#[cfg(test)]
pub(crate) mod read {
    use super::Json;

    /// Nesting depth after which the reader gives up; the report nests four levels deep.
    const MAX_DEPTH: usize = 32;

    /// Parses a complete JSON text after RFC 8259.
    ///
    /// Stricter than the RFC in two points that would be bugs in a report: duplicate keys in an
    /// object are rejected, and so are numbers other than unsigned integers that fit a `u64`.
    pub(crate) fn parse(text: &str) -> Result<Json, String> {
        let mut reader = Reader {
            bytes: text.as_bytes(),
            position: 0,
        };
        reader.skip_whitespace();
        let value = reader.value(0)?;
        reader.skip_whitespace();
        if reader.position != reader.bytes.len() {
            return Err(reader.error("unexpected characters after the document"));
        }
        Ok(value)
    }

    struct Reader<'a> {
        bytes: &'a [u8],
        position: usize,
    }

    impl Reader<'_> {
        fn error(&self, message: &str) -> String {
            format!("{message} at byte {}", self.position)
        }

        fn peek(&self) -> Option<u8> {
            self.bytes.get(self.position).copied()
        }

        fn skip_whitespace(&mut self) {
            while matches!(self.peek(), Some(b' ' | b'\t' | b'\n' | b'\r')) {
                self.position += 1;
            }
        }

        fn expect(&mut self, byte: u8) -> Result<(), String> {
            if self.peek() == Some(byte) {
                self.position += 1;
                Ok(())
            } else {
                Err(self.error(&format!("expected '{}'", char::from(byte))))
            }
        }

        fn literal(&mut self, word: &str, value: Json) -> Result<Json, String> {
            if self.bytes[self.position..].starts_with(word.as_bytes()) {
                self.position += word.len();
                Ok(value)
            } else {
                Err(self.error("invalid literal"))
            }
        }

        fn value(&mut self, depth: usize) -> Result<Json, String> {
            if depth > MAX_DEPTH {
                return Err(self.error("nesting too deep"));
            }
            match self.peek() {
                Some(b'{') => self.object(depth),
                Some(b'[') => self.array(depth),
                Some(b'"') => self.string().map(Json::String),
                Some(b't') => self.literal("true", Json::Bool(true)),
                Some(b'f') => self.literal("false", Json::Bool(false)),
                Some(b'n') => self.literal("null", Json::Null),
                Some(b'-' | b'0'..=b'9') => self.number(),
                Some(_) => Err(self.error("unexpected character")),
                None => Err(self.error("unexpected end of input")),
            }
        }

        fn object(&mut self, depth: usize) -> Result<Json, String> {
            self.expect(b'{')?;
            let mut entries: Vec<(String, Json)> = Vec::new();
            self.skip_whitespace();
            if self.peek() == Some(b'}') {
                self.position += 1;
                return Ok(Json::Object(entries));
            }
            loop {
                self.skip_whitespace();
                let key = self.string()?;
                if entries.iter().any(|(existing, _)| *existing == key) {
                    return Err(self.error(&format!("duplicate key {key:?}")));
                }
                self.skip_whitespace();
                self.expect(b':')?;
                self.skip_whitespace();
                let value = self.value(depth + 1)?;
                entries.push((key, value));
                self.skip_whitespace();
                match self.peek() {
                    Some(b',') => self.position += 1,
                    Some(b'}') => {
                        self.position += 1;
                        return Ok(Json::Object(entries));
                    }
                    _ => return Err(self.error("expected ',' or '}'")),
                }
            }
        }

        fn array(&mut self, depth: usize) -> Result<Json, String> {
            self.expect(b'[')?;
            let mut items = Vec::new();
            self.skip_whitespace();
            if self.peek() == Some(b']') {
                self.position += 1;
                return Ok(Json::Array(items));
            }
            loop {
                self.skip_whitespace();
                items.push(self.value(depth + 1)?);
                self.skip_whitespace();
                match self.peek() {
                    Some(b',') => self.position += 1,
                    Some(b']') => {
                        self.position += 1;
                        return Ok(Json::Array(items));
                    }
                    _ => return Err(self.error("expected ',' or ']'")),
                }
            }
        }

        fn string(&mut self) -> Result<String, String> {
            self.expect(b'"')?;
            let mut text: Vec<u8> = Vec::new();
            loop {
                let Some(byte) = self.peek() else {
                    return Err(self.error("unterminated string"));
                };
                self.position += 1;
                match byte {
                    b'"' => break,
                    b'\\' => {
                        let character = self.escape()?;
                        text.extend_from_slice(character.encode_utf8(&mut [0; 4]).as_bytes());
                    }
                    0x00..=0x1F => {
                        self.position -= 1;
                        return Err(self.error("unescaped control character in string"));
                    }
                    // The input is a `&str`, so copying bytes keeps the text valid UTF-8.
                    other => text.push(other),
                }
            }
            String::from_utf8(text).map_err(|_| self.error("string is not valid UTF-8"))
        }

        fn escape(&mut self) -> Result<char, String> {
            let Some(byte) = self.peek() else {
                return Err(self.error("unterminated escape"));
            };
            self.position += 1;
            Ok(match byte {
                b'"' => '"',
                b'\\' => '\\',
                b'/' => '/',
                b'b' => '\u{08}',
                b'f' => '\u{0C}',
                b'n' => '\n',
                b'r' => '\r',
                b't' => '\t',
                b'u' => {
                    let unit = self.hex4()?;
                    let code_point = match unit {
                        0xD800..=0xDBFF => {
                            if !self.bytes[self.position..].starts_with(b"\\u") {
                                return Err(self.error("high surrogate without a low surrogate"));
                            }
                            self.position += 2;
                            let low = self.hex4()?;
                            if !(0xDC00..=0xDFFF).contains(&low) {
                                return Err(self.error("invalid low surrogate"));
                            }
                            0x10000 + ((unit - 0xD800) << 10) + (low - 0xDC00)
                        }
                        0xDC00..=0xDFFF => return Err(self.error("unpaired low surrogate")),
                        other => other,
                    };
                    char::from_u32(code_point).ok_or_else(|| self.error("invalid code point"))?
                }
                _ => {
                    self.position -= 1;
                    return Err(self.error("invalid escape"));
                }
            })
        }

        fn hex4(&mut self) -> Result<u32, String> {
            let digits = self
                .bytes
                .get(self.position..self.position + 4)
                .ok_or_else(|| self.error("truncated \\u escape"))?;
            let mut value = 0;
            for &digit in digits {
                let nibble = char::from(digit)
                    .to_digit(16)
                    .ok_or_else(|| self.error("invalid hex digit in \\u escape"))?;
                value = value * 16 + nibble;
            }
            self.position += 4;
            Ok(value)
        }

        fn number(&mut self) -> Result<Json, String> {
            let start = self.position;
            let negative = self.peek() == Some(b'-');
            if negative {
                self.position += 1;
            }
            let integer_start = self.position;
            while matches!(self.peek(), Some(b'0'..=b'9')) {
                self.position += 1;
            }
            let integer = &self.bytes[integer_start..self.position];
            if integer.is_empty() {
                return Err(self.error("number without digits"));
            }
            if integer.len() > 1 && integer[0] == b'0' {
                self.position = integer_start;
                return Err(self.error("number with a leading zero"));
            }
            if negative || matches!(self.peek(), Some(b'.' | b'e' | b'E')) {
                self.position = start;
                return Err(self.error("only unsigned integers are supported"));
            }
            let digits = std::str::from_utf8(integer).map_err(|_| self.error("invalid number"))?;
            digits
                .parse::<u64>()
                .map(Json::Number)
                .map_err(|_| self.error("integer does not fit into 64 bits"))
        }
    }

    impl Json {
        /// The value of `key` when `self` is an object that has it.
        pub(crate) fn get(&self, key: &str) -> Option<&Json> {
            match self {
                Json::Object(entries) => entries
                    .iter()
                    .find(|(name, _)| name == key)
                    .map(|(_, value)| value),
                _ => None,
            }
        }

        /// Follows a path of object keys; panics with the missing key, for test brevity.
        pub(crate) fn at(&self, path: &[&str]) -> &Json {
            path.iter().fold(self, |value, key| {
                value
                    .get(key)
                    .unwrap_or_else(|| panic!("missing key {key:?} in path {path:?}"))
            })
        }

        pub(crate) fn as_str(&self) -> Option<&str> {
            match self {
                Json::String(text) => Some(text),
                _ => None,
            }
        }

        pub(crate) fn as_bool(&self) -> Option<bool> {
            match self {
                Json::Bool(value) => Some(*value),
                _ => None,
            }
        }

        pub(crate) fn as_u64(&self) -> Option<u64> {
            match self {
                Json::Number(value) => Some(*value),
                _ => None,
            }
        }

        pub(crate) fn as_array(&self) -> Option<&[Json]> {
            match self {
                Json::Array(items) => Some(items),
                _ => None,
            }
        }

        pub(crate) fn as_object(&self) -> Option<&[(String, Json)]> {
            match self {
                Json::Object(entries) => Some(entries),
                _ => None,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::read::parse;
    use super::*;

    fn sample() -> Json {
        Json::object([
            ("null", Json::Null),
            ("yes", Json::from(true)),
            ("no", Json::from(false)),
            ("zero", Json::from(0u32)),
            ("big", Json::from(u64::MAX)),
            ("text", Json::from("plain")),
            ("empty_array", Json::Array(Vec::new())),
            ("empty_object", Json::Object(Vec::new())),
            ("names", Json::strings(["a", "b"])),
            (
                "nested",
                Json::object([(
                    "list",
                    Json::Array(vec![Json::from(1u32), Json::object([("deep", Json::Null)])]),
                )]),
            ),
        ])
    }

    #[test]
    fn pretty_output_has_the_expected_shape() {
        let document = Json::object([
            ("name", Json::from("GTX")),
            ("layers", Json::from(2048u32)),
            ("flags", Json::strings(["A", "B"])),
            ("none", Json::Array(Vec::new())),
        ]);
        let expected = [
            "{",
            "  \"name\": \"GTX\",",
            "  \"layers\": 2048,",
            "  \"flags\": [",
            "    \"A\",",
            "    \"B\"",
            "  ],",
            "  \"none\": []",
            "}",
            "",
        ]
        .join("\n");
        assert_eq!(document.to_pretty_string(), expected);
    }

    #[test]
    fn written_documents_parse_back_to_the_same_value() {
        let document = sample();
        let text = document.to_pretty_string();
        assert_eq!(parse(&text).unwrap(), document);
        // Scalars are documents too.
        for scalar in [
            Json::Null,
            Json::from(true),
            Json::from(7u32),
            Json::from("x"),
        ] {
            assert_eq!(parse(&scalar.to_pretty_string()).unwrap(), scalar);
        }
    }

    #[test]
    fn strings_are_escaped_after_rfc_8259() {
        let mut out = String::new();
        write_string(
            &mut out,
            "a\"b\\c/d\n\r\t\u{08}\u{0C}\u{00}\u{1F}\u{7F}é漢😀",
        );
        assert_eq!(
            out,
            "\"a\\\"b\\\\c/d\\n\\r\\t\\b\\f\\u0000\\u001f\u{7F}é漢😀\""
        );
    }

    #[test]
    fn hostile_strings_survive_a_round_trip() {
        let every_control: String = (0u8..0x20).map(char::from).collect();
        let hostile = [
            "",
            "\"",
            "\\",
            "\\\"",
            "\\u0041",
            "line\nbreak",
            "tab\tand\rreturn",
            "NVIDIA \"GeForce\" | GTX\\1080",
            "ünïcödé 漢字 😀 \u{2028}\u{2029}",
            "}{][,:",
            every_control.as_str(),
        ];
        for text in hostile {
            let document = Json::object([(text, Json::from(text))]);
            let written = document.to_pretty_string();
            assert!(
                !written.bytes().any(|byte| byte < 0x20 && byte != b'\n'),
                "raw control character written for {text:?}"
            );
            assert_eq!(parse(&written).unwrap(), document, "{text:?}");
        }
    }

    #[test]
    fn the_reader_understands_every_escape() {
        let text = r#""\" \\ \/ \b \f \n \r \t \u0041 \u00e9 \uD83D\uDE00""#;
        assert_eq!(
            parse(text).unwrap(),
            Json::from("\" \\ / \u{08} \u{0C} \n \r \t A é 😀")
        );
    }

    #[test]
    fn the_reader_accepts_compact_and_spaced_documents() {
        let compact = r#"{"a":[1,2,{"b":null}],"c":true,"d":"e"}"#;
        let spaced =
            " {\n\t\"a\" : [ 1 , 2 , { \"b\" : null } ] ,\r\n \"c\" : true , \"d\" : \"e\" } ";
        assert_eq!(parse(compact).unwrap(), parse(spaced).unwrap());
        assert_eq!(
            parse("18446744073709551615").unwrap(),
            Json::Number(u64::MAX)
        );
    }

    #[test]
    fn the_reader_rejects_invalid_json() {
        let invalid = [
            "",
            " ",
            "{",
            "}",
            "[",
            "[1,]",
            "[1 2]",
            "{\"a\":1,}",
            "{\"a\" 1}",
            "{a:1}",
            "{\"a\":1,\"a\":2}",
            "{\"a\":1}{}",
            "{\"a\":1} x",
            "'a'",
            "\"unterminated",
            "\"raw\nnewline\"",
            "\"raw\ttab\"",
            "\"bad \\x escape\"",
            "\"\\u12\"",
            "\"\\u12G4\"",
            "\"\\uD83D\"",
            "\"\\uD83Dx\"",
            "\"\\uD83D\\u0041\"",
            "\"\\uDE00\"",
            "tru",
            "nul",
            "True",
            "01",
            "-",
            "-1",
            "1.5",
            "1e3",
            "18446744073709551616",
            "NaN",
        ];
        for text in invalid {
            assert!(parse(text).is_err(), "{text:?} must be rejected");
        }
        let deep = "[".repeat(100) + &"]".repeat(100);
        assert!(parse(&deep).is_err());
    }

    #[test]
    fn accessors_navigate_a_document() {
        let document = sample();
        assert_eq!(document.at(&["text"]).as_str(), Some("plain"));
        assert_eq!(document.at(&["yes"]).as_bool(), Some(true));
        assert_eq!(document.at(&["big"]).as_u64(), Some(u64::MAX));
        assert_eq!(
            document.at(&["names"]).as_array().map(<[Json]>::len),
            Some(2)
        );
        assert_eq!(
            document
                .at(&["nested"])
                .as_object()
                .map(<[(String, Json)]>::len),
            Some(1)
        );
        assert_eq!(document.get("missing"), None);
        assert_eq!(document.at(&["text"]).as_u64(), None);
    }
}
