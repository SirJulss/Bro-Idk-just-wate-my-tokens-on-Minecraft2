//! WGSL shaders: the embedded library, the `#include` preprocessor, validation with
//! `wgpu::naga` and module creation that reports a bad shader as an error (contract §5.3,
//! "Infrastructure"; §4.5 error handling).
//!
//! # The library
//! Every WGSL file of the crate lives under `src/shader/wgsl/` and is embedded with
//! `include_str!` in [`BUILTIN_FILES`]. A file is named by its path relative to that directory,
//! with `/` separators: `common/color.wgsl`. Every file is a program of its own (include files
//! simply have no entry points), and the tests of this module check that
//!
//! - every `.wgsl` file of the directory is listed in [`BUILTIN_FILES`] and vice versa,
//! - every listed file preprocesses, parses and validates for [`ShaderTarget::baseline`] plus
//!   its [`ShaderFile::required_features`], and for a desktop device with every optional
//!   feature of the engine.
//!
//! A new shader is one file in the directory and one `wgsl_file!("path.wgsl")` line in the
//! table.
//!
//! # Includes
//! See [`preprocess`]: `#include "common/color.wgsl"` on a line of its own, every file at most
//! once per program, cycles are errors. Errors of naga are mapped back to the file and line
//! of the library ([`CompileFailure::at`]).
//!
//! # Creation
//! [`ShaderLibrary::create_module`] preprocesses, validates with the capabilities of the
//! device (so a bad shader fails with a readable location before wgpu sees it) and creates the
//! module inside an error scope ([`create_shader_module`]); a failure is an `Err`, never a
//! panic.

mod preprocess;
mod validate;

use thiserror::Error;

pub use preprocess::{PreprocessedSource, SourceLine, preprocess};
pub use validate::{CompileFailure, ShaderTarget, ValidatedShader, validate};

use crate::error::RenderError;
use crate::gpu::GpuContext;
use crate::scope::capture;

/// Why a program could not be preprocessed or validated.
#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub enum ShaderError {
    /// A file is not in the library.
    #[error("{}", unknown_file_message(.path, .included_from.as_ref()))]
    UnknownFile {
        /// The path that was asked for.
        path: String,
        /// The `#include` that asked for it; `None` for the program itself.
        included_from: Option<SourceLine>,
    },
    /// A file includes itself, directly or through others.
    #[error("include cycle: {}", .chain.join(" -> "))]
    IncludeCycle {
        /// The files of the cycle, the repeated file at both ends.
        chain: Vec<String>,
    },
    /// A line starts with `#` but is no `#include "path"`.
    #[error("{at}: malformed directive `{text}`; the only directive is #include \"path.wgsl\"")]
    BadDirective {
        /// Where the line is.
        at: SourceLine,
        /// The line without surrounding blanks.
        text: String,
    },
    /// An include path leaves the library or does not name a `.wgsl` file.
    #[error(
        "{at}: include path {path:?} must be relative to the library root, use `/` and end \
         in .wgsl"
    )]
    BadIncludePath {
        /// Where the `#include` is.
        at: SourceLine,
        /// The path as written.
        path: String,
    },
    /// The program, or a file it includes, needs device features the target does not have.
    #[error("{program} needs device features the device does not have: {missing:?}")]
    MissingFeatures {
        /// The program.
        program: String,
        /// The required features the target lacks.
        missing: wgpu::Features,
    },
    /// A resource uses a bind group index the device does not have.
    #[error("{program}: `{variable}` is in bind group {group}, the device has {limit}")]
    BindGroupOutOfRange {
        /// The program.
        program: String,
        /// Name of the global variable.
        variable: String,
        /// Its `@group` index.
        group: u32,
        /// `max_bind_groups` of the target.
        limit: u32,
    },
    /// naga's WGSL front end rejected the program.
    #[error("WGSL parse error in {0}")]
    Parse(Box<CompileFailure>),
    /// naga's validator rejected the module.
    #[error("WGSL validation error in {0}")]
    Validation(Box<CompileFailure>),
}

fn unknown_file_message(path: &str, included_from: Option<&SourceLine>) -> String {
    match included_from {
        Some(at) => format!("{at}: shader file {path:?} is not in the library"),
        None => format!("shader file {path:?} is not in the library"),
    }
}

/// One WGSL file of a [`ShaderLibrary`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ShaderFile {
    /// Path relative to the library root with `/` separators; the name `#include` uses.
    pub path: &'static str,
    /// The text of the file.
    pub source: &'static str,
    /// Optional device features the file needs. A program needs the features of every file
    /// it includes; validation and creation refuse it on a device without them.
    pub required_features: wgpu::Features,
}

impl ShaderFile {
    /// A file that needs no optional feature.
    pub const fn new(path: &'static str, source: &'static str) -> Self {
        Self {
            path,
            source,
            required_features: wgpu::Features::empty(),
        }
    }

    /// The same file, needing `features`.
    #[must_use]
    pub const fn requiring(mut self, features: wgpu::Features) -> Self {
        self.required_features = features;
        self
    }
}

/// A [`ShaderFile`] for `src/shader/wgsl/<path>`, embedded with `include_str!` (which resolves
/// the path relative to this file).
macro_rules! wgsl_file {
    ($path:literal) => {
        ShaderFile::new($path, include_str!(concat!("wgsl/", $path)))
    };
}

/// The WGSL files of this crate, added with `wgsl_file!("path.wgsl")` in this table. A file of
/// `src/shader/wgsl/` that is missing here fails the tests.
pub const BUILTIN_FILES: &[ShaderFile] = &[
    wgsl_file!("common/agx.wgsl"),
    wgsl_file!("common/color.wgsl"),
    wgsl_file!("common/debug_view.wgsl"),
    wgsl_file!("common/fullscreen.wgsl"),
    wgsl_file!("common/material.wgsl"),
    wgsl_file!("common/noise.wgsl"),
    wgsl_file!("common/octahedral.wgsl"),
    wgsl_file!("overlay.wgsl"),
    wgsl_file!("voxel/bounds.wgsl"),
    wgsl_file!("voxel/common.wgsl"),
    wgsl_file!("voxel/composite.wgsl"),
    wgsl_file!("voxel/cull.wgsl"),
    wgsl_file!("voxel/gbuffer.wgsl"),
    wgsl_file!("voxel/lighting.wgsl"),
    wgsl_file!("voxel/taa.wgsl"),
];

/// A set of WGSL files that include each other by path.
#[derive(Clone, Copy, Debug)]
pub struct ShaderLibrary {
    files: &'static [ShaderFile],
}

impl ShaderLibrary {
    /// A library of `files`; paths must be unique.
    pub const fn new(files: &'static [ShaderFile]) -> Self {
        Self { files }
    }

    /// The crate's own library, [`BUILTIN_FILES`].
    pub const fn builtin() -> Self {
        Self::new(BUILTIN_FILES)
    }

    /// Every file of the library.
    pub fn files(&self) -> &'static [ShaderFile] {
        self.files
    }

    /// The file at `path`.
    pub fn file(&self, path: &str) -> Option<&'static ShaderFile> {
        self.files.iter().find(|file| file.path == path)
    }

    /// The program at `path` with its includes expanded.
    ///
    /// # Errors
    /// The errors of [`preprocess`].
    pub fn preprocess(&self, program: &str) -> Result<PreprocessedSource, ShaderError> {
        preprocess(program, |path| self.file(path).map(|file| file.source))
    }

    /// The features a preprocessed program needs: those of every file it was made from.
    pub fn required_features(&self, source: &PreprocessedSource) -> wgpu::Features {
        source
            .files()
            .iter()
            .filter_map(|path| self.file(path))
            .fold(wgpu::Features::empty(), |all, file| {
                all | file.required_features
            })
    }

    /// Preprocesses `program` and validates it for `target`.
    ///
    /// # Errors
    /// The errors of [`Self::preprocess`] and [`validate`], and
    /// [`ShaderError::MissingFeatures`] when the target lacks a required feature.
    pub fn validate(
        &self,
        program: &str,
        target: &ShaderTarget,
    ) -> Result<ValidatedShader, ShaderError> {
        let source = self.preprocess(program)?;
        let missing = self.required_features(&source) - target.features;
        if !missing.is_empty() {
            return Err(ShaderError::MissingFeatures {
                program: program.to_owned(),
                missing,
            });
        }
        validate(source, target)
    }

    /// Creates the shader module of `program` on the device of `gpu`: preprocessed, validated
    /// for the device ([`ShaderTarget::of_device`]) and created inside an error scope.
    ///
    /// # Errors
    /// [`RenderError::Shader`] for errors of the source, [`RenderError::Wgpu`] when the device
    /// rejects the module anyway (for example a backend compiler failure), and
    /// [`RenderError::Device`] on a lost device.
    pub fn create_module(
        &self,
        gpu: &GpuContext,
        program: &str,
    ) -> Result<wgpu::ShaderModule, RenderError> {
        let validated = self.validate(program, &ShaderTarget::of_device(&gpu.caps))?;
        create_shader_module(gpu, program, validated.source.text())
    }
}

/// Creates a shader module from WGSL text inside an error scope, labelled `label`.
///
/// For text that does not come from a [`ShaderLibrary`]; library programs go through
/// [`ShaderLibrary::create_module`], which also maps errors to library lines.
///
/// # Errors
/// [`RenderError::Wgpu`] when wgpu rejects the module, [`RenderError::Device`] on a lost
/// device.
pub fn create_shader_module(
    gpu: &GpuContext,
    label: &str,
    wgsl: &str,
) -> Result<wgpu::ShaderModule, RenderError> {
    capture(gpu, &format!("shader module {label}"), |device| {
        device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(label),
            source: wgpu::ShaderSource::Wgsl(wgsl.into()),
        })
    })
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;
    use std::path::Path;

    use super::*;
    use crate::gpu::{AdapterProfile, GpuCaps};

    /// A desktop Vulkan device that has every optional feature the engine asks for.
    fn full_desktop_target() -> ShaderTarget {
        let profile = AdapterProfile {
            info: wgpu::AdapterInfo::new(wgpu::DeviceType::DiscreteGpu, wgpu::Backend::Vulkan),
            features: wgpu::Features::all(),
            limits: wgpu::Limits::unlimited(),
            downlevel: wgpu::DownlevelCapabilities::default(),
        };
        let caps = GpuCaps::negotiate(profile, wgpu::InstanceFlags::empty());
        ShaderTarget::of_device(&caps)
    }

    /// Every `.wgsl` file below `directory`, as library paths.
    fn wgsl_files_below(directory: &Path, prefix: &str, found: &mut BTreeSet<String>) {
        let entries = std::fs::read_dir(directory)
            .unwrap_or_else(|error| panic!("cannot list {}: {error}", directory.display()));
        for entry in entries {
            let entry = entry.unwrap();
            let name = entry.file_name().into_string().unwrap();
            let path = entry.path();
            if path.is_dir() {
                wgsl_files_below(&path, &format!("{prefix}{name}/"), found);
            } else if name.ends_with(".wgsl") {
                found.insert(format!("{prefix}{name}"));
            }
        }
    }

    #[test]
    fn every_wgsl_file_of_the_directory_is_in_the_library() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("src/shader/wgsl");
        let mut on_disk = BTreeSet::new();
        wgsl_files_below(&root, "", &mut on_disk);
        let listed: BTreeSet<String> = BUILTIN_FILES
            .iter()
            .map(|file| file.path.to_owned())
            .collect();
        assert_eq!(listed.len(), BUILTIN_FILES.len(), "a path is listed twice");
        assert_eq!(on_disk, listed);
        for file in BUILTIN_FILES {
            // Contract §2.7: LF line endings.
            assert!(
                !file.source.contains('\r'),
                "{} has CR characters",
                file.path
            );
            assert!(
                file.source.ends_with('\n'),
                "{} lacks the final newline",
                file.path
            );
        }
    }

    #[test]
    fn every_shader_validates_on_the_baseline_and_on_a_full_desktop_device() {
        let library = ShaderLibrary::builtin();
        let full = full_desktop_target();
        // The engine's whitelist is a strict superset of nothing: the full target must differ.
        assert_ne!(full.features, wgpu::Features::empty());
        assert_eq!(full.max_bind_groups, 8);
        for file in library.files() {
            let baseline = ShaderTarget::baseline().with_features(file.required_features);
            for (name, target) in [("baseline", baseline), ("full desktop", full)] {
                if let Err(error) = library.validate(file.path, &target) {
                    let detail = match &error {
                        ShaderError::Parse(failure) | ShaderError::Validation(failure) => {
                            failure.diagnostic.clone()
                        }
                        _ => String::new(),
                    };
                    panic!("{} on the {name} target: {error}\n{detail}", file.path);
                }
            }
        }
    }

    #[test]
    fn the_baseline_has_four_bind_groups_and_no_optional_capability() {
        let baseline = ShaderTarget::baseline();
        assert_eq!(baseline.max_bind_groups, 4);
        assert_eq!(baseline.features, wgpu::Features::empty());
        let capabilities = baseline.capabilities();
        assert!(!capabilities.contains(wgpu::naga::valid::Capabilities::IMMEDIATES));
        assert!(!capabilities.contains(wgpu::naga::valid::Capabilities::DUAL_SOURCE_BLENDING));
        // Immediates are a capability only with the feature, as on the device.
        let with = baseline.with_features(wgpu::Features::IMMEDIATES);
        assert!(
            with.capabilities()
                .contains(wgpu::naga::valid::Capabilities::IMMEDIATES)
        );
    }

    /// A library over leaked test files, for programs the crate does not ship.
    fn test_library(files: &[(&'static str, &'static str)]) -> ShaderLibrary {
        let files: Vec<ShaderFile> = files
            .iter()
            .map(|&(path, source)| ShaderFile::new(path, source))
            .collect();
        ShaderLibrary::new(Vec::leak(files))
    }

    #[test]
    fn naga_errors_point_into_the_included_file() {
        let library = test_library(&[
            (
                "main.wgsl",
                "#include \"lib.wgsl\"\n@compute @workgroup_size(1)\nfn main() { helper(); }\n",
            ),
            (
                "lib.wgsl",
                "// helpers\nfn helper() {\n    let x: u32 = 1.5;\n}\n",
            ),
        ]);
        let error = library
            .validate("main.wgsl", &ShaderTarget::baseline())
            .unwrap_err();
        let ShaderError::Parse(failure) = &error else {
            panic!("expected a parse error, got {error:?}");
        };
        assert_eq!(failure.program, "main.wgsl");
        assert_eq!(
            failure.at,
            Some(SourceLine {
                file: "lib.wgsl".to_owned(),
                line: 3
            })
        );
        assert!(!failure.message.is_empty());
        assert!(
            failure.diagnostic.contains("let x: u32 = 1.5"),
            "{failure:?}"
        );
        let text = error.to_string();
        assert!(
            text.starts_with("WGSL parse error in main.wgsl at lib.wgsl:3: "),
            "{text}"
        );
    }

    #[test]
    fn validation_errors_are_reported_with_their_causes() {
        // Parses, but a vertex shader must write the position.
        let library = test_library(&[(
            "bad.wgsl",
            "@vertex\nfn main() -> @location(0) vec4<f32> {\n    return vec4<f32>(1.0);\n}\n",
        )]);
        let error = library
            .validate("bad.wgsl", &ShaderTarget::baseline())
            .unwrap_err();
        assert!(matches!(error, ShaderError::Validation(_)), "{error:?}");
        assert!(
            error
                .to_string()
                .starts_with("WGSL validation error in bad.wgsl")
        );
    }

    #[test]
    fn bind_groups_beyond_the_target_are_rejected() {
        let library = test_library(&[(
            "groups.wgsl",
            "@group(5) @binding(0) var<uniform> value: vec4<f32>;\n\
             @fragment\nfn main() -> @location(0) vec4<f32> {\n    return value;\n}\n",
        )]);
        let error = library
            .validate("groups.wgsl", &ShaderTarget::baseline())
            .unwrap_err();
        assert_eq!(
            error,
            ShaderError::BindGroupOutOfRange {
                program: "groups.wgsl".to_owned(),
                variable: "value".to_owned(),
                group: 5,
                limit: 4,
            }
        );
        let roomy = ShaderTarget {
            max_bind_groups: 8,
            ..ShaderTarget::baseline()
        };
        assert!(library.validate("groups.wgsl", &roomy).is_ok());
    }

    #[test]
    fn required_features_are_checked_before_validation() {
        let files: &'static [ShaderFile] = Vec::leak(vec![
            ShaderFile::new("main.wgsl", "#include \"needs.wgsl\"\n"),
            ShaderFile::new("needs.wgsl", "fn f() {}\n")
                .requiring(wgpu::Features::DUAL_SOURCE_BLENDING),
        ]);
        let library = ShaderLibrary::new(files);
        let source = library.preprocess("main.wgsl").unwrap();
        assert_eq!(
            library.required_features(&source),
            wgpu::Features::DUAL_SOURCE_BLENDING
        );
        assert_eq!(
            library
                .validate("main.wgsl", &ShaderTarget::baseline())
                .unwrap_err(),
            ShaderError::MissingFeatures {
                program: "main.wgsl".to_owned(),
                missing: wgpu::Features::DUAL_SOURCE_BLENDING,
            }
        );
        let target = ShaderTarget::baseline().with_features(wgpu::Features::DUAL_SOURCE_BLENDING);
        assert!(library.validate("main.wgsl", &target).is_ok());
    }

    #[test]
    fn library_lookup_and_unknown_programs() {
        let library = ShaderLibrary::builtin();
        assert!(library.file("overlay.wgsl").is_some());
        assert!(library.file("missing.wgsl").is_none());
        assert_eq!(
            library.preprocess("missing.wgsl").unwrap_err().to_string(),
            "shader file \"missing.wgsl\" is not in the library"
        );
        let overlay = library.preprocess("overlay.wgsl").unwrap();
        assert_eq!(overlay.files(), ["overlay.wgsl", "common/color.wgsl"]);
    }
}
