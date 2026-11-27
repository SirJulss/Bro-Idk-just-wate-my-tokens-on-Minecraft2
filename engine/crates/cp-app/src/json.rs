//! A small JSON writer for the benchmark reports (the workspace has no JSON crate, contract
//! §3.2). Objects keep their keys in insertion order, so a report reads in the order it was
//! built.

use std::fmt::Write as _;

/// A JSON value.
#[derive(Clone, Debug, PartialEq)]
pub enum Json {
    /// `null`.
    Null,
    /// `true` or `false`.
    Bool(bool),
    /// A number; non-finite numbers are written as `null`.
    Number(f64),
    /// An integer, written without a fraction.
    Integer(i64),
    /// A string.
    String(String),
    /// An array.
    Array(Vec<Json>),
    /// An object; keys in insertion order.
    Object(Vec<(String, Json)>),
}

impl Json {
    /// An empty object.
    pub fn object() -> Self {
        Self::Object(Vec::new())
    }

    /// Adds `key: value` to an object (a no-op on other values) and returns it.
    #[must_use]
    pub fn with(mut self, key: &str, value: impl Into<Json>) -> Self {
        if let Self::Object(entries) = &mut self {
            entries.push((key.to_owned(), value.into()));
        }
        self
    }

    /// The value as indented JSON text with a final newline.
    pub fn render(&self) -> String {
        let mut out = String::new();
        self.write(&mut out, 0);
        out.push('\n');
        out
    }

    fn write(&self, out: &mut String, indent: usize) {
        match self {
            Self::Null => out.push_str("null"),
            Self::Bool(value) => out.push_str(if *value { "true" } else { "false" }),
            Self::Number(value) => {
                if *value == 0.0 {
                    // Both zeros.
                    out.push('0');
                } else if value.is_finite() {
                    // Nine decimals, without trailing zeros.
                    let text = format!("{value:.9}");
                    out.push_str(text.trim_end_matches('0').trim_end_matches('.'));
                } else {
                    out.push_str("null");
                }
            }
            Self::Integer(value) => {
                let _ = write!(out, "{value}");
            }
            Self::String(text) => write_string(out, text),
            Self::Array(items) => {
                if items.is_empty() {
                    out.push_str("[]");
                    return;
                }
                out.push('[');
                for (index, item) in items.iter().enumerate() {
                    out.push_str(if index == 0 { "\n" } else { ",\n" });
                    push_indent(out, indent + 1);
                    item.write(out, indent + 1);
                }
                out.push('\n');
                push_indent(out, indent);
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
                    push_indent(out, indent + 1);
                    write_string(out, key);
                    out.push_str(": ");
                    value.write(out, indent + 1);
                }
                out.push('\n');
                push_indent(out, indent);
                out.push('}');
            }
        }
    }
}

fn push_indent(out: &mut String, indent: usize) {
    for _ in 0..indent {
        out.push_str("  ");
    }
}

fn write_string(out: &mut String, text: &str) {
    out.push('"');
    for character in text.chars() {
        match character {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if u32::from(c) < 0x20 => {
                let _ = write!(out, "\\u{:04x}", u32::from(c));
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

impl From<bool> for Json {
    fn from(value: bool) -> Self {
        Self::Bool(value)
    }
}

impl From<f64> for Json {
    fn from(value: f64) -> Self {
        Self::Number(value)
    }
}

impl From<u32> for Json {
    fn from(value: u32) -> Self {
        Self::Integer(i64::from(value))
    }
}

impl From<u64> for Json {
    fn from(value: u64) -> Self {
        Self::Integer(i64::try_from(value).unwrap_or(i64::MAX))
    }
}

impl From<usize> for Json {
    fn from(value: usize) -> Self {
        Self::Integer(i64::try_from(value).unwrap_or(i64::MAX))
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

impl<T: Into<Json>> From<Option<T>> for Json {
    fn from(value: Option<T>) -> Self {
        value.map_or(Self::Null, Into::into)
    }
}

impl From<Vec<Json>> for Json {
    fn from(value: Vec<Json>) -> Self {
        Self::Array(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn values_render_as_json() {
        let value = Json::object()
            .with("name", "a \"b\"\n")
            .with("count", 3_u32)
            .with("ms", 0.125)
            .with("whole", 2.0)
            .with("none", Option::<u32>::None)
            .with("nan", f64::NAN)
            .with("list", vec![Json::Bool(true), Json::Number(-0.5)])
            .with("empty", Json::Array(Vec::new()));
        assert_eq!(
            value.render(),
            "{\n  \"name\": \"a \\\"b\\\"\\n\",\n  \"count\": 3,\n  \"ms\": 0.125,\n  \"whole\": 2,\n  \
             \"none\": null,\n  \"nan\": null,\n  \"list\": [\n    true,\n    -0.5\n  ],\n  \
             \"empty\": []\n}\n"
        );
        assert_eq!(Json::Number(0.0).render(), "0\n");
        assert_eq!(Json::Number(-0.0).render(), "0\n");
    }
}
