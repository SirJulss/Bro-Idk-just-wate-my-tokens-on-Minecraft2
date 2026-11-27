//! The TOML schema of content pack files.
//!
//! Every struct here mirrors one TOML table and rejects unknown fields, so a misspelt field is
//! an error instead of a silently ignored line. The structs hold values exactly as written;
//! `loader` checks their meaning and resolves references.
//!
//! A definition file is a list of `[[texture]]`, `[[material]]` or `[[block]]` tables. It is
//! parsed into a spanned document first and each entry is then deserialized on its own, so that
//! an error can name the entry's `key` and the line it happened on.

use std::collections::BTreeMap;
use std::path::Path;

use serde::Deserialize;
use serde::de::DeserializeOwned;
use toml::de::{DeTable, DeValue, ValueDeserializer};

use crate::block::Orient;
use crate::error::ContentError;
use crate::material::TintClass;
use crate::texture::{Params, Provenance};

/// A raw definition that carries its own key.
pub(crate) trait Keyed {
    /// The `key` field as written.
    fn key(&self) -> &str;
}

/// `pack.toml`.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct RawManifest {
    /// Version of the pack format the pack is written for.
    pub format: u32,
    /// Short name of the pack, used in log messages.
    pub name: String,
    /// Version of the pack's content, free-form.
    pub version: String,
    /// One-line description.
    #[serde(default)]
    pub description: String,
}

/// One `[[texture]]` entry.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct RawTexture {
    pub key: String,
    pub procedural: Option<RawProcedural>,
    pub file: Option<String>,
    pub provenance: Provenance,
    pub licence: Option<String>,
}

/// The `procedural` table of a texture.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct RawProcedural {
    pub recipe: String,
    pub seed: u64,
    #[serde(default)]
    pub params: Params,
}

/// One `[[material]]` entry.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct RawMaterial {
    pub key: String,
    pub texture: String,
    /// A factor on the texture's own roughness; 1 leaves it as painted.
    #[serde(default = "one")]
    pub roughness: f64,
    #[serde(default)]
    pub metallic: f64,
    #[serde(default)]
    pub emissive: f64,
    #[serde(default)]
    pub tint: TintClass,
    #[serde(default)]
    pub flags: Vec<RawMaterialFlag>,
}

fn one() -> f64 {
    1.0
}

/// A name in a material's `flags` list.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum RawMaterialFlag {
    Cutout,
    Translucent,
}

/// One `[[block]]` entry.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct RawBlock {
    pub key: String,
    #[serde(default)]
    pub orient: Orient,
    #[serde(default)]
    pub render: RawRender,
    /// Defaults to true for `cube`, `cutout` and `model` blocks.
    pub solid: Option<bool>,
    /// Defaults to true for `cube` blocks.
    pub opaque: Option<bool>,
    #[serde(default)]
    pub structural: bool,
    pub faces: Option<RawFaces>,
    pub hardness: f64,
    pub density: f64,
    #[serde(default)]
    pub light: u8,
    #[serde(default)]
    pub tags: Vec<String>,
    /// Name of a cull group.
    pub cull_group: Option<String>,
    /// State key of the stand-in of a `model` block in coarser LOD levels.
    pub lod_proxy: Option<String>,
    /// `[ext.<namespace>]` tables, kept as written.
    #[serde(default)]
    pub ext: BTreeMap<String, toml::Table>,
}

impl RawBlock {
    /// The empty block the engine provides when no pack defines `cubeplanet:air`.
    pub(crate) fn air(key: &str) -> Self {
        Self {
            key: key.to_owned(),
            orient: Orient::None,
            render: RawRender::None,
            solid: None,
            opaque: None,
            structural: false,
            faces: None,
            hardness: 0.0,
            density: 0.0,
            light: 0,
            tags: Vec::new(),
            cull_group: None,
            lod_proxy: None,
            ext: BTreeMap::new(),
        }
    }
}

/// The `render` field of a block.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum RawRender {
    None,
    #[default]
    Cube,
    Cutout,
    Fluid,
    Model,
}

/// The `faces` table of a block: material keys in one of three forms.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct RawFaces {
    pub all: Option<String>,
    pub top: Option<String>,
    pub side: Option<String>,
    pub bottom: Option<String>,
    #[serde(rename = "+x")]
    pub pos_x: Option<String>,
    #[serde(rename = "-x")]
    pub neg_x: Option<String>,
    #[serde(rename = "+y")]
    pub pos_y: Option<String>,
    #[serde(rename = "-y")]
    pub neg_y: Option<String>,
    #[serde(rename = "+z")]
    pub pos_z: Option<String>,
    #[serde(rename = "-z")]
    pub neg_z: Option<String>,
}

impl RawFaces {
    /// The material keys of the six block-local faces in face index order
    /// (`+X -X +Y -Y +Z -Z`). Exactly one of the three forms must be given, completely.
    pub(crate) fn local_faces(&self) -> Result<[&str; 6], String> {
        const FORMS: &str = "give `all`, or `top`, `side` and `bottom`, or all six of `+x`, \
                             `-x`, `+y`, `-y`, `+z`, `-z`";
        let six = [
            &self.pos_x,
            &self.neg_x,
            &self.pos_y,
            &self.neg_y,
            &self.pos_z,
            &self.neg_z,
        ];
        let any_six = six.iter().any(|face| face.is_some());
        let any_tsb = self.top.is_some() || self.side.is_some() || self.bottom.is_some();
        match (&self.all, any_tsb, any_six) {
            (Some(all), false, false) => Ok([all.as_str(); 6]),
            (None, true, false) => match (&self.top, &self.side, &self.bottom) {
                (Some(top), Some(side), Some(bottom)) => Ok([side, side, top, bottom, side, side]),
                _ => Err(format!("`faces` is incomplete: {FORMS}")),
            },
            (None, false, true) => {
                let mut faces = [""; 6];
                for (slot, face) in faces.iter_mut().zip(six) {
                    *slot = face
                        .as_deref()
                        .ok_or_else(|| format!("`faces` is incomplete: {FORMS}"))?;
                }
                Ok(faces)
            }
            (None, false, false) => Err(format!("`faces` is empty: {FORMS}")),
            _ => Err(format!("`faces` mixes several forms: {FORMS}")),
        }
    }
}

impl Keyed for RawTexture {
    fn key(&self) -> &str {
        &self.key
    }
}

impl Keyed for RawMaterial {
    fn key(&self) -> &str {
        &self.key
    }
}

impl Keyed for RawBlock {
    fn key(&self) -> &str {
        &self.key
    }
}

/// One entry of a definition file with the line of its header.
#[derive(Debug)]
pub(crate) struct Entry<T> {
    pub raw: T,
    /// 1-based line of the entry's `[[…]]` header.
    pub line: u32,
}

/// Parses a definition file: any number of `[[table]]` entries and nothing else.
pub(crate) fn parse_entries<T: DeserializeOwned>(
    file: &Path,
    text: &str,
    table: &str,
) -> Result<Vec<Entry<T>>, ContentError> {
    let lines = Lines::new(text);
    let root = DeTable::parse(text).map_err(|error| parse_error(file, &lines, &error, None))?;
    let mut entries = Vec::new();
    for (name, value) in root.into_inner() {
        let name_text: &str = name.get_ref();
        if name_text != table {
            return Err(error_at(
                file,
                &lines,
                name.span().start,
                format!(
                    "unknown top-level key `{name_text}`: this file may only contain \
                     `[[{table}]]` entries"
                ),
            ));
        }
        let start = value.span().start;
        let DeValue::Array(items) = value.into_inner() else {
            return Err(error_at(
                file,
                &lines,
                start,
                format!(
                    "`{table}` must be an array of tables: start each entry with `[[{table}]]`"
                ),
            ));
        };
        for item in items {
            let (line, _) = lines.position(item.span().start);
            let key = item
                .get_ref()
                .get("key")
                .and_then(|key| key.get_ref().as_str())
                .map(str::to_owned);
            let raw = T::deserialize(ValueDeserializer::from(item))
                .map_err(|error| parse_error(file, &lines, &error, key))?;
            entries.push(Entry { raw, line });
        }
    }
    Ok(entries)
}

/// Parses `pack.toml`.
pub(crate) fn parse_manifest(file: &Path, text: &str) -> Result<RawManifest, ContentError> {
    toml::from_str(text).map_err(|error| parse_error(file, &Lines::new(text), &error, None))
}

/// Converts a TOML error into a [`ContentError::Parse`] with line and column.
fn parse_error(
    file: &Path,
    lines: &Lines<'_>,
    error: &toml::de::Error,
    key: Option<String>,
) -> ContentError {
    let (line, column) = error
        .span()
        .map_or((1, 1), |span| lines.position(span.start));
    ContentError::Parse {
        file: file.to_path_buf(),
        line,
        column,
        key,
        message: error.message().to_owned(),
    }
}

fn error_at(file: &Path, lines: &Lines<'_>, offset: usize, message: String) -> ContentError {
    let (line, column) = lines.position(offset);
    ContentError::Parse {
        file: file.to_path_buf(),
        line,
        column,
        key: None,
        message,
    }
}

/// Maps byte offsets of a text to lines and columns.
///
/// Built once per file: a definition file may hold thousands of entries, and each of them needs
/// its line, so rescanning the text for every entry would cost quadratic time.
struct Lines<'a> {
    text: &'a str,
    /// Byte offset at which each line starts, ascending; the first is 0.
    starts: Vec<usize>,
}

impl<'a> Lines<'a> {
    fn new(text: &'a str) -> Self {
        let after_newlines = text
            .bytes()
            .enumerate()
            .filter(|&(_, byte)| byte == b'\n')
            .map(|(index, _)| index + 1);
        Self {
            text,
            starts: core::iter::once(0).chain(after_newlines).collect(),
        }
    }

    /// 1-based line and column (in characters) of the byte `offset`. Offsets past the end count
    /// as the end of the text.
    fn position(&self, offset: usize) -> (u32, u32) {
        let offset = offset.min(self.text.len());
        // The number of lines that start at or before `offset`: at least one, as line 1 starts
        // at offset 0.
        let line = self.starts.partition_point(|&start| start <= offset);
        let line_start = self.starts[line - 1];
        // Count characters, not bytes: every byte that is not a UTF-8 continuation byte starts
        // one.
        let column = self.text.as_bytes()[line_start..offset]
            .iter()
            .filter(|&&byte| byte & 0xC0 != 0x80)
            .count()
            + 1;
        (
            u32::try_from(line).unwrap_or(u32::MAX),
            u32::try_from(column).unwrap_or(u32::MAX),
        )
    }
}

#[cfg(test)]
mod tests;
