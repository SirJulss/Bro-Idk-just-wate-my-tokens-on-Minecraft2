//! The content of [`Registry::builtin_minimal`](crate::Registry::builtin_minimal).
//!
//! It is written in the pack format and goes through the same loader as a pack on disk, so
//! tests that use it exercise the real code path. The definitions are copies of the ones in
//! `content/base`; an integration test keeps the two from drifting apart.

use crate::error::DefKind;
use crate::pack::PackSource;

const MANIFEST: &str = "\
format = 1
name = \"minimal\"
version = \"1\"
description = \"Built-in minimal content for tests.\"
";

/// The built-in pack. Its files are named `<builtin>/…` in error messages.
pub(crate) fn source() -> PackSource {
    PackSource::embedded(
        "<builtin>",
        MANIFEST,
        &[
            (
                "minimal.toml",
                DefKind::Texture,
                include_str!("builtin/textures.toml"),
            ),
            (
                "minimal.toml",
                DefKind::Material,
                include_str!("builtin/materials.toml"),
            ),
            (
                "minimal.toml",
                DefKind::Block,
                include_str!("builtin/blocks.toml"),
            ),
        ],
    )
}
