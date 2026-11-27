//! The TOML schema of system files (`content/<pack>/systems/*.toml`) and the loader.
//!
//! One file defines one system:
//!
//! ```toml
//! [system]
//! key = "cubeplanet:home"            # namespace:name, unique among the loaded systems
//! display_name = "Heimatsystem"
//!
//! [[body]]                           # parents before their children; the first id is 0
//! key = "cubeplanet:sonne"
//! display_name = "Sonne"
//! kind = "star"                      # star | planet | moon
//! radius_m = 70000.0                 # stars only
//! surface_gravity_m_s2 = 147.5       # mu = g * R² (R = radius, or half edge for cubes)
//! rotation_period_s = 43200.0        # sidereal; or `tidally_locked = true` for children
//! rotation_phase_deg = 0.0           # optional, default 0; not with tidally_locked
//! axial_tilt_deg = 0.0               # spin axis against the system's +Y, leaning to -X;
//!                                    # tidally locked: against the orbit normal, leaning
//!                                    # to the system's +Y (crate::Rotation::Synchronous)
//!
//! [[body]]
//! key = "cubeplanet:heimat"
//! display_name = "Heimat"
//! kind = "planet"
//! parent = "cubeplanet:sonne"        # required for planets and moons
//! edge_m = 40960                     # planets and moons: even integer
//! surface_gravity_m_s2 = 9.81
//! sea_level_m = 0.0                  # above the reference surface (the cube's faces)
//! grid_offset = 1365                 # voxel lattice shift, integer
//! atmosphere_height_m = 8000.0       # 0 = airless
//! generator = "cubeplanet:hills"     # BaseSource key
//! generator_seed = 14159             # optional, default: FNV-1a hash of the body key
//! rotation_period_s = 6912.0
//! axial_tilt_deg = 23.5
//!
//! [body.orbit]                       # every body but the star; all six elements required
//! semi_major_axis_m = 8192000.0
//! eccentricity = 0.0
//! inclination_deg = 0.0
//! ascending_node_deg = 0.0
//! periapsis_arg_deg = 0.0
//! mean_anomaly_deg = 270.0           # at the epoch, t = 0
//! ```
//!
//! Every key not listed here is an error, as is a key that does not belong to the body's kind.
//! Numbers may be written as integers or floats except where an integer is required. Angles
//! are degrees in the file and radians in [`crate::BodyDef`].

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use cp_core::hash::hash_str;
use cp_core::ids::BodyId;

use crate::body::{BodyDef, BodyKind, BodyShape, CubeDef, Rotation, radians};
use crate::document::{self, Entry, Table, Value};
use crate::error::{DefinitionError, UniverseError};
use crate::orbit::OrbitElements;
use crate::system::SystemDef;

const TOP_KEYS: &[&str] = &["system", "body"];
const SYSTEM_KEYS: &[&str] = &["key", "display_name"];
const BODY_KEYS: &[&str] = &[
    "key",
    "display_name",
    "kind",
    "parent",
    "radius_m",
    "edge_m",
    "surface_gravity_m_s2",
    "sea_level_m",
    "grid_offset",
    "atmosphere_height_m",
    "generator",
    "generator_seed",
    "rotation_period_s",
    "rotation_phase_deg",
    "tidally_locked",
    "axial_tilt_deg",
    "orbit",
];
const ORBIT_KEYS: &[&str] = &[
    "semi_major_axis_m",
    "eccentricity",
    "inclination_deg",
    "ascending_node_deg",
    "periapsis_arg_deg",
    "mean_anomaly_deg",
];
/// Keys of cube bodies that a star must not have.
const CUBE_ONLY: &[&str] = &[
    "edge_m",
    "sea_level_m",
    "grid_offset",
    "atmosphere_height_m",
    "generator",
    "generator_seed",
];

/// Largest integer magnitude that converts to `f64` exactly.
const EXACT_INTEGER: u64 = 1 << 53;

/// Lines of the keys a definition was read from, by field name (`edge_m`,
/// `orbit.eccentricity`), and the line of its table header.
#[derive(Default)]
struct Lines {
    header: u32,
    fields: BTreeMap<&'static str, u32>,
}

impl Lines {
    fn of(&self, field: Option<&str>) -> u32 {
        field
            .and_then(|field| self.fields.get(field).copied())
            .unwrap_or(self.header)
    }
}

/// Reads the keys of one table, records their lines and rejects the ones nobody asked for.
struct Reader<'a> {
    file: &'a Path,
    item: String,
    table: &'a Table,
    /// Prefix of the field names of this table, e.g. `orbit.`.
    prefix: &'static str,
    known: &'static [&'static str],
    used: Vec<bool>,
    lines: Lines,
}

type Read<T> = Result<T, UniverseError>;

impl<'a> Reader<'a> {
    fn new(
        file: &'a Path,
        item: String,
        table: &'a Table,
        prefix: &'static str,
        known: &'static [&'static str],
    ) -> Self {
        Self {
            file,
            item,
            table,
            prefix,
            known,
            used: vec![false; table.entries.len()],
            lines: Lines {
                header: table.line,
                fields: BTreeMap::new(),
            },
        }
    }

    fn error(&self, line: u32, field: &str, message: impl Into<String>) -> UniverseError {
        UniverseError::Definition {
            file: self.file.to_path_buf(),
            line,
            item: self.item.clone(),
            key: Some(field.to_owned()),
            message: message.into(),
        }
    }

    fn local<'k>(&self, field: &'k str) -> &'k str {
        field
            .strip_prefix(self.prefix)
            .expect("field names carry the prefix of their table")
    }

    /// The entry of `field`, marked as read.
    fn entry(&mut self, field: &'static str) -> Option<&'a Entry> {
        let key = self.local(field);
        let index = self
            .table
            .entries
            .iter()
            .position(|entry| entry.key == key)?;
        self.used[index] = true;
        let entry = &self.table.entries[index];
        self.lines.fields.insert(field, entry.line);
        Some(entry)
    }

    fn required(&mut self, field: &'static str) -> Read<&'a Entry> {
        self.entry(field)
            .ok_or_else(|| self.error(self.table.line, field, "missing"))
    }

    fn wrong_type(&self, entry: &Entry, field: &str, expected: &str) -> UniverseError {
        self.error(
            entry.line,
            field,
            format!("must be {expected}, found {}", entry.value.type_name()),
        )
    }

    fn string_of(&self, entry: &Entry, field: &str) -> Read<String> {
        match &entry.value {
            Value::String(text) => Ok(text.clone()),
            _ => Err(self.wrong_type(entry, field, "a string")),
        }
    }

    fn string(&mut self, field: &'static str) -> Read<String> {
        let entry = self.required(field)?;
        self.string_of(entry, field)
    }

    fn number_of(&self, entry: &Entry, field: &str) -> Read<f64> {
        match entry.value {
            Value::Float(value) => Ok(value),
            Value::Integer(value) if value.unsigned_abs() <= EXACT_INTEGER => Ok(value as f64),
            Value::Integer(value) => Err(self.error(
                entry.line,
                field,
                format!("{value} is too large to be represented exactly; write it as a float"),
            )),
            _ => Err(self.wrong_type(entry, field, "a number")),
        }
    }

    fn number(&mut self, field: &'static str) -> Read<f64> {
        let entry = self.required(field)?;
        self.number_of(entry, field)
    }

    fn optional_number(&mut self, field: &'static str) -> Read<Option<f64>> {
        self.entry(field)
            .map(|entry| self.number_of(entry, field))
            .transpose()
    }

    fn integer_of(&self, entry: &Entry, field: &str) -> Read<i64> {
        match entry.value {
            Value::Integer(value) => Ok(value),
            _ => Err(self.wrong_type(entry, field, "an integer")),
        }
    }

    fn integer<T: TryFrom<i64>>(&mut self, field: &'static str, range: &str) -> Read<T> {
        let entry = self.required(field)?;
        let value = self.integer_of(entry, field)?;
        T::try_from(value)
            .map_err(|_| self.error(entry.line, field, format!("must be {range}, found {value}")))
    }

    fn optional_integer<T: TryFrom<i64>>(
        &mut self,
        field: &'static str,
        range: &str,
    ) -> Read<Option<T>> {
        let Some(entry) = self.entry(field) else {
            return Ok(None);
        };
        let value = self.integer_of(entry, field)?;
        T::try_from(value)
            .map(Some)
            .map_err(|_| self.error(entry.line, field, format!("must be {range}, found {value}")))
    }

    fn optional_boolean(&mut self, field: &'static str) -> Read<Option<bool>> {
        let Some(entry) = self.entry(field) else {
            return Ok(None);
        };
        match entry.value {
            Value::Boolean(value) => Ok(Some(value)),
            _ => Err(self.wrong_type(entry, field, "a boolean")),
        }
    }

    /// A sub-table written as `[parent.field]` or as an inline table.
    fn optional_table(&mut self, field: &'static str) -> Read<Option<&'a Table>> {
        let Some(entry) = self.entry(field) else {
            return Ok(None);
        };
        match &entry.value {
            Value::Table(table) => Ok(Some(table)),
            _ => Err(self.wrong_type(
                entry,
                field,
                "a table (a `[...]` header or an inline table)",
            )),
        }
    }

    /// An array of tables written as `[[field]]`.
    fn table_array(&mut self, field: &'static str) -> Read<&'a [Table]> {
        let entry = self.required(field)?;
        match &entry.value {
            Value::TableArray(tables) => Ok(tables),
            _ => Err(self.wrong_type(
                entry,
                field,
                &format!("an array of tables, written `[[{field}]]`"),
            )),
        }
    }

    /// Rejects `field` with `reason` if it is present.
    fn forbid(&mut self, field: &'static str, reason: &str) -> Read<()> {
        match self.entry(field) {
            Some(entry) => Err(self.error(entry.line, field, reason)),
            None => Ok(()),
        }
    }

    /// Fails on the first key that was not read; returns the recorded lines.
    fn finish(self) -> Read<Lines> {
        if let Some(index) = self.used.iter().position(|used| !used) {
            let entry = &self.table.entries[index];
            let expected = self
                .known
                .iter()
                .map(|key| format!("`{key}`"))
                .collect::<Vec<_>>()
                .join(", ");
            let full = format!("{}{}", self.prefix, entry.key);
            return Err(self.error(
                entry.line,
                &full,
                format!("unknown key; expected one of {expected}"),
            ));
        }
        Ok(self.lines)
    }
}

/// Parses one system file. `file` is used in error messages only.
pub(crate) fn read_system(text: &str, file: &Path) -> Read<SystemDef> {
    let root = document::parse(text).map_err(|error| UniverseError::Syntax {
        file: file.to_path_buf(),
        line: error.line,
        column: error.column,
        message: error.message,
    })?;
    let mut top = Reader::new(file, "top level".to_owned(), &root, "", TOP_KEYS);
    let system_table = top
        .optional_table("system")?
        .ok_or_else(|| top.error(1, "system", "missing; a system file starts with [system]"))?;
    let body_tables = top.table_array("body")?;
    top.finish()?;

    let mut system = Reader::new(file, "system".to_owned(), system_table, "", SYSTEM_KEYS);
    let key = system.string("key")?;
    let display_name = system.string("display_name")?;
    let system_lines = system.finish()?;

    // Parents are named by key; resolve them against every body of the file, so that a parent
    // written after its child gets the precise "must be defined before" message.
    let mut index_of: BTreeMap<&str, usize> = BTreeMap::new();
    for (index, table) in body_tables.iter().enumerate() {
        if let Some(Entry {
            value: Value::String(key),
            ..
        }) = table.get("key")
        {
            index_of.entry(key.as_str()).or_insert(index);
        }
    }
    let mut bodies = Vec::with_capacity(body_tables.len());
    let mut lines = Vec::with_capacity(body_tables.len());
    let mut items = Vec::with_capacity(body_tables.len());
    for (index, table) in body_tables.iter().enumerate() {
        let item = match table.get("key") {
            Some(Entry {
                value: Value::String(key),
                ..
            }) => format!("body `{key}`"),
            _ => format!("body #{index}"),
        };
        let (body, body_lines) = read_body(file, item.clone(), table, &index_of)?;
        bodies.push(body);
        lines.push(body_lines);
        items.push(item);
    }
    SystemDef::new(key, display_name, bodies)
        .map_err(|error| locate(file, error, &system_lines, &lines, &items))
}

/// Turns a rule violation into an error with the file line of the offending key.
fn locate(
    file: &Path,
    error: DefinitionError,
    system_lines: &Lines,
    lines: &[Lines],
    items: &[String],
) -> UniverseError {
    let (line, item) = match error.body {
        Some(index) => (lines[index].of(error.field), items[index].clone()),
        None => (system_lines.of(error.field), "system".to_owned()),
    };
    UniverseError::Definition {
        file: file.to_path_buf(),
        line,
        item,
        key: error.field.map(str::to_owned),
        message: error.message,
    }
}

fn read_body(
    file: &Path,
    item: String,
    table: &Table,
    index_of: &BTreeMap<&str, usize>,
) -> Read<(BodyDef, Lines)> {
    let mut reader = Reader::new(file, item.clone(), table, "", BODY_KEYS);
    let key = reader.string("key")?;
    let display_name = reader.string("display_name")?;
    let kind_entry = reader.required("kind")?;
    let kind_name = reader.string_of(kind_entry, "kind")?;
    let kind = BodyKind::from_name(&kind_name).ok_or_else(|| {
        reader.error(
            kind_entry.line,
            "kind",
            format!("`{kind_name}` is not a body kind; expected `star`, `planet` or `moon`"),
        )
    })?;
    let parent = match reader.entry("parent") {
        None => None,
        Some(entry) => {
            let name = reader.string_of(entry, "parent")?;
            let Some(&index) = index_of.get(name.as_str()) else {
                return Err(reader.error(
                    entry.line,
                    "parent",
                    format!("no body of this file has the key `{name}`"),
                ));
            };
            Some(BodyId(u32::try_from(index).unwrap_or(u32::MAX)))
        }
    };
    let surface_gravity = reader.number("surface_gravity_m_s2")?;
    let axial_tilt = radians(reader.number("axial_tilt_deg")?);
    let rotation = read_rotation(&mut reader)?;
    let shape = if kind == BodyKind::Star {
        for &field in CUBE_ONLY {
            reader.forbid(
                field,
                "only planets and moons (cube bodies) have this key; a star has `radius_m`",
            )?;
        }
        BodyShape::Sphere {
            radius: reader.number("radius_m")?,
        }
    } else {
        reader.forbid(
            "radius_m",
            "only stars have a radius; planets and moons are cubes with `edge_m`",
        )?;
        let edge = reader.integer::<u32>("edge_m", "a positive number of metres")?;
        let sea_level = reader.number("sea_level_m")?;
        let grid_offset = reader.integer::<i32>("grid_offset", "a 32-bit integer")?;
        let atmosphere_height = reader.number("atmosphere_height_m")?;
        let generator = reader.string("generator")?;
        let generator_seed = reader
            .optional_integer::<u64>("generator_seed", "at least 0")?
            .unwrap_or_else(|| hash_str(&key));
        BodyShape::Cube(CubeDef {
            edge,
            sea_level,
            grid_offset,
            atmosphere_height,
            generator,
            generator_seed,
        })
    };
    let orbit = match reader.optional_table("orbit")? {
        None => None,
        Some(orbit_table) => {
            let mut orbit = Reader::new(file, item, orbit_table, "orbit.", ORBIT_KEYS);
            let elements = OrbitElements {
                semi_major_axis: orbit.number("orbit.semi_major_axis_m")?,
                eccentricity: orbit.number("orbit.eccentricity")?,
                inclination: radians(orbit.number("orbit.inclination_deg")?),
                ascending_node: radians(orbit.number("orbit.ascending_node_deg")?),
                periapsis_argument: radians(orbit.number("orbit.periapsis_arg_deg")?),
                mean_anomaly_at_epoch: radians(orbit.number("orbit.mean_anomaly_deg")?),
            };
            let orbit_lines = orbit.finish()?;
            reader.lines.fields.extend(orbit_lines.fields);
            Some(elements)
        }
    };
    let lines = reader.finish()?;
    let body = BodyDef {
        key,
        display_name,
        kind,
        parent,
        shape,
        surface_gravity,
        rotation,
        axial_tilt,
        orbit,
    };
    Ok((body, lines))
}

fn read_rotation(reader: &mut Reader<'_>) -> Read<Rotation> {
    let period = reader.optional_number("rotation_period_s")?;
    let locked = reader.optional_boolean("tidally_locked")?.unwrap_or(false);
    let phase = reader.optional_number("rotation_phase_deg")?;
    match (period, locked) {
        (Some(_), true) => Err(reader.error(
            reader.lines.of(Some("tidally_locked")),
            "tidally_locked",
            "a tidally locked body takes its rotation period from its orbit; remove \
             `rotation_period_s`",
        )),
        (Some(period), false) => Ok(Rotation::Uniform {
            period,
            phase: radians(phase.unwrap_or(0.0)),
        }),
        (None, true) if phase.is_some() => Err(reader.error(
            reader.lines.of(Some("rotation_phase_deg")),
            "rotation_phase_deg",
            "a tidally locked body turns its -X face to its parent; it has no free phase",
        )),
        (None, true) => Ok(Rotation::Synchronous),
        (None, false) => Err(reader.error(
            reader.table.line,
            "rotation_period_s",
            "missing (or write `tidally_locked = true` for a body that keeps one face to its \
             parent)",
        )),
    }
}

impl SystemDef {
    /// Parses a system from the text of a system file; `file` names it in errors.
    ///
    /// # Errors
    /// [`UniverseError::Syntax`] for text that is not TOML or uses an unsupported construct,
    /// [`UniverseError::Definition`] for a schema violation or a broken rule of
    /// [`SystemDef::new`], with the line of the offending key.
    pub fn from_toml(text: &str, file: &Path) -> Result<Self, UniverseError> {
        read_system(text, file)
    }

    /// Reads and parses one system file.
    ///
    /// # Errors
    /// [`UniverseError::Io`] when the file cannot be read (or is not UTF-8), otherwise as
    /// [`SystemDef::from_toml`].
    pub fn load(path: &Path) -> Result<Self, UniverseError> {
        let text = std::fs::read_to_string(path).map_err(|source| UniverseError::Io {
            path: path.to_path_buf(),
            source,
        })?;
        read_system(&text, path)
    }
}

/// Loads every `*.toml` file in `dir` (not recursive) as one system, in the order of their file
/// names; other files are ignored.
///
/// # Errors
/// [`UniverseError::Io`] when the directory or a file cannot be read,
/// [`UniverseError::DuplicateSystem`] when two files define the same system key, and every
/// error of [`SystemDef::load`].
pub fn load_systems(dir: &Path) -> Result<Vec<SystemDef>, UniverseError> {
    let io = |source| UniverseError::Io {
        path: dir.to_path_buf(),
        source,
    };
    let mut files: Vec<PathBuf> = Vec::new();
    for entry in std::fs::read_dir(dir).map_err(io)? {
        let entry = entry.map_err(io)?;
        let path = entry.path();
        let is_file = entry.file_type().map_err(io)?.is_file();
        if is_file
            && path
                .extension()
                .is_some_and(|extension| extension == "toml")
        {
            files.push(path);
        }
    }
    files.sort();
    let mut systems: Vec<SystemDef> = Vec::with_capacity(files.len());
    let mut first_file: BTreeMap<String, PathBuf> = BTreeMap::new();
    for file in files {
        let system = SystemDef::load(&file)?;
        if let Some(first) = first_file.get(system.key()) {
            return Err(UniverseError::DuplicateSystem {
                key: system.key().to_owned(),
                file,
                first_file: first.clone(),
            });
        }
        first_file.insert(system.key().to_owned(), file);
        systems.push(system);
    }
    Ok(systems)
}

#[cfg(test)]
mod tests;
