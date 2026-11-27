//! Parsing and validation of preprocessed WGSL with `wgpu::naga`, exactly as the device does it.

use std::fmt;

use wgpu::naga;

use super::{PreprocessedSource, ShaderError, SourceLine};
use crate::gpu::GpuCaps;

/// What a shader is checked against: the features and downlevel flags of a device, from which
/// the naga capabilities follow, and its number of bind groups.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ShaderTarget {
    /// Features enabled on the device.
    pub features: wgpu::Features,
    /// Downlevel flags of the adapter.
    pub downlevel: wgpu::DownlevelFlags,
    /// `max_bind_groups` of the device: every `@group` index must be below it.
    pub max_bind_groups: u32,
}

impl ShaderTarget {
    /// The target of a real device: its enabled features and limits.
    pub fn of_device(caps: &GpuCaps) -> Self {
        Self {
            features: caps.enabled_features,
            downlevel: caps.downlevel.flags,
            max_bind_groups: caps.enabled_limits.max_bind_groups,
        }
    }

    /// The weakest device every shader must work on: a WebGPU-compliant adapter without any
    /// optional feature and with the WebGPU default of four bind groups.
    pub fn baseline() -> Self {
        Self {
            features: wgpu::Features::empty(),
            downlevel: wgpu::DownlevelFlags::compliant(),
            max_bind_groups: wgpu::Limits::default().max_bind_groups,
        }
    }

    /// The same target with `features` enabled in addition.
    #[must_use]
    pub fn with_features(mut self, features: wgpu::Features) -> Self {
        self.features |= features;
        self
    }

    /// The naga capabilities a device with this target validates shaders with.
    ///
    /// This is wgpu-core's own mapping (`features_to_naga_capabilities`, re-exported by wgpu
    /// as `wgpu::wgc`), so a shader that passes here passes the device's validation too.
    pub fn capabilities(&self) -> naga::valid::Capabilities {
        wgpu::wgc::device::features_to_naga_capabilities(self.features, self.downlevel)
    }
}

/// Where and why naga rejected a program.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CompileFailure {
    /// The program, by its library path.
    pub program: String,
    /// The library file and line of the first span naga reported, if it reported one.
    pub at: Option<SourceLine>,
    /// naga's message with its chain of causes, on one line.
    pub message: String,
    /// naga's full report over the preprocessed text, several lines with source excerpts.
    pub diagnostic: String,
}

impl fmt::Display for CompileFailure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.at {
            Some(at) => write!(f, "{} at {at}: {}", self.program, self.message),
            None => write!(f, "{}: {}", self.program, self.message),
        }
    }
}

/// A program that passed naga's validation for a [`ShaderTarget`].
#[derive(Debug)]
pub struct ValidatedShader {
    /// The preprocessed source the module was made from.
    pub source: PreprocessedSource,
    /// The parsed module.
    pub module: naga::Module,
    /// What the validator learned about it (uniformity, expression types).
    pub info: naga::valid::ModuleInfo,
}

/// Parses `source` with naga's WGSL front end and validates the module with every validation
/// flag and the capabilities of `target`, the way `wgpu::Device::create_shader_module` does;
/// also checks the bind group indices against the target.
///
/// # Errors
/// [`ShaderError::Parse`] and [`ShaderError::Validation`] with the location mapped back to the
/// library file and line, [`ShaderError::BindGroupOutOfRange`].
pub fn validate(
    source: PreprocessedSource,
    target: &ShaderTarget,
) -> Result<ValidatedShader, ShaderError> {
    let capabilities = target.capabilities();
    let text = source.text();
    let mut options = naga::front::wgsl::Options::new();
    options.capabilities = capabilities;
    let module = naga::front::wgsl::Frontend::new_with_options(options)
        .parse(text)
        .map_err(|error| {
            ShaderError::Parse(Box::new(CompileFailure {
                program: source.program().to_owned(),
                at: error
                    .location(text)
                    .and_then(|location| source.origin(location.line_number)),
                message: one_line(error.message()),
                diagnostic: error.emit_to_string_with_path(text, source.program()),
            }))
        })?;

    for (_, global) in module.global_variables.iter() {
        if let Some(binding) = &global.binding
            && binding.group >= target.max_bind_groups
        {
            return Err(ShaderError::BindGroupOutOfRange {
                program: source.program().to_owned(),
                variable: global.name.clone().unwrap_or_default(),
                group: binding.group,
                limit: target.max_bind_groups,
            });
        }
    }

    let info = naga::valid::Validator::new(naga::valid::ValidationFlags::all(), capabilities)
        .validate(&module)
        .map_err(|error| {
            ShaderError::Validation(Box::new(CompileFailure {
                program: source.program().to_owned(),
                at: error
                    .location(text)
                    .and_then(|location| source.origin(location.line_number)),
                message: one_line(&error_chain(&error)),
                diagnostic: error.emit_to_string_with_path(text, source.program()),
            }))
        })?;

    Ok(ValidatedShader {
        source,
        module,
        info,
    })
}

/// An error and all its sources, joined by `": "`.
fn error_chain(error: &dyn std::error::Error) -> String {
    let mut text = error.to_string();
    let mut cause = error.source();
    while let Some(next) = cause {
        text.push_str(": ");
        text.push_str(&next.to_string());
        cause = next.source();
    }
    text
}

/// `text` with every run of white space, line breaks included, replaced by one blank.
fn one_line(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}
