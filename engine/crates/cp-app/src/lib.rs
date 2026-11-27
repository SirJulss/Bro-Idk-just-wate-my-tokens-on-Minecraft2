//! `cp-app`: the application layer of the Cubeplanet engine, built as the binary `cpe`.
//!
//! Responsibility: window, input, debug views, headless shots, benchmarks and tools. It glues
//! content, textures and the renderer together and is the only crate that prints.
//!
//! Binding contract: `engine/docs/ARCHITEKTUR.md`, sections 4.5 and 5.4.
//!
//! | command | module | what it does |
//! |---|---|---|
//! | `cpe caps` | [`caps`] | opens the GPU headless and prints or saves the capability report |
//! | `cpe proctex-sheet` | [`sheet`], [`pack_sheet`] | writes the contact sheet of the procedural textures, or of a pack's textures, as PNG |
//! | `cpe content-check` | [`content_check`] | loads content packs, generates their textures, fills the material colours |
//! | `cpe run` | [`run`](mod@run) | a window over a scene with a free-fly camera, HUD, debug views and draw paths |
//! | `cpe shot` | [`shot`] | one headless frame of a scene camera as PNG |
//! | `cpe bench` | [`bench`] | renderer measurements per sub-scene as JSON |
//!
//! The glue between the crates: [`assets`] turns content packs into the renderer's material
//! table and texture array, cp-worldgen's palette and the tables of cp-voxel and cp-mesh;
//! [`scene`] holds the scenes, the static LOD selection around a camera and the generation and
//! meshing of chunks on the job system; [`stage`] loads a scene into a renderer.
//!
//! [`cli::parse`] turns the command line into a [`cli::Command`], [`run()`] executes it and
//! writes what it has to say to the given output. Log records of all crates reach stderr
//! through [`logger`].

// The application is not under the determinism rules (contract 2.5): it may use the maths of
// the standard library, whose results can differ in the last bit between machines. The scene
// scripts use `cp_core::det` all the same, so cameras land on the same bits everywhere.
#![allow(clippy::disallowed_methods)]

pub mod assets;
pub mod bench;
pub mod caps;
pub mod cli;
pub mod content_check;
mod image;
pub mod json;
pub mod logger;
pub mod metrics;
pub mod pack_sheet;
pub mod run;
pub mod scene;
pub mod sheet;
pub mod shot;
pub mod stage;
pub mod stats;

use std::fmt;
use std::io::Write;
use std::path::Path;

use cli::Command;

/// Exit code of a command that failed while running (a GPU, file or content problem).
pub const EXIT_FAILURE: u8 = 1;
/// Exit code of a command line that cannot be run (unknown command, bad option or value).
pub const EXIT_USAGE: u8 = 2;

/// Why a command failed, as a message for the user: what went wrong and what it concerns
/// (a file, a texture, the GPU).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Failure {
    message: String,
}

impl Failure {
    /// A failure with this message.
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }

    /// The message, without a trailing newline.
    #[must_use]
    pub fn message(&self) -> &str {
        &self.message
    }
}

impl fmt::Display for Failure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for Failure {}

/// Runs `command` and writes its result (usage text or report) to `out`.
///
/// Files the command was asked to write are written before this returns; progress and
/// diagnostics go to the log.
///
/// # Errors
/// A [`Failure`] when the command cannot do its job, and when `out` cannot be written.
pub fn run(command: &Command, out: &mut dyn Write) -> Result<(), Failure> {
    match command {
        Command::Help(topic) => write_out(out, &cli::usage(*topic)),
        Command::Caps(args) => {
            let report = caps::run(args)?;
            write_out(out, &report)
        }
        Command::ProctexSheet(args) => {
            let sheet = sheet::run(args)?;
            write_out(out, &format!("{sheet}\n"))
        }
        Command::ContentCheck(args) => {
            let report = content_check::run(args)?;
            write_out(out, &report.to_string())
        }
        Command::Run(args) => {
            let summary = run::run(args)?;
            write_out(out, &format!("{summary}\n"))
        }
        Command::Shot(args) => {
            let info = shot::run(args)?;
            write_out(out, &format!("{info}\n"))
        }
        Command::Bench(args) => {
            let info = bench::run(args)?;
            write_out(out, &format!("{info}\n"))
        }
    }
}

/// Writes `text` to the command's output.
fn write_out(out: &mut dyn Write, text: &str) -> Result<(), Failure> {
    out.write_all(text.as_bytes())
        .and_then(|()| out.flush())
        .map_err(|error| Failure::new(format!("cannot write the output: {error}")))
}

/// Writes `bytes` to the file `path`, creating its parent directories first.
fn write_file(path: &Path, bytes: &[u8]) -> Result<(), Failure> {
    if let Some(parent) = path.parent().filter(|dir| !dir.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent).map_err(|error| {
            Failure::new(format!(
                "cannot create the directory {}: {error}",
                parent.display()
            ))
        })?;
    }
    std::fs::write(path, bytes)
        .map_err(|error| Failure::new(format!("cannot write {}: {error}", path.display())))?;
    log::info!("wrote {} ({} bytes)", path.display(), bytes.len());
    Ok(())
}
