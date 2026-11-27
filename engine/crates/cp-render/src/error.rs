//! The error type of everything in this crate that talks to the GPU device.
//!
//! Pure checks have errors of their own ([`ShaderError`], [`FrameGraphError`],
//! [`TextureError`], [`MaterialError`], [`HeadlessError`], [`VoxelError`]); [`RenderError`]
//! wraps them together with what the device reports, so a caller propagates one type with `?`.

use thiserror::Error;

use crate::frame_graph::FrameGraphError;
use crate::gpu::GpuError;
use crate::headless::HeadlessError;
use crate::material::MaterialError;
use crate::shader::ShaderError;
use crate::texture::TextureError;
use crate::voxel::VoxelError;

/// The kind of error wgpu reported inside an error scope ([`crate::scope::capture`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum WgpuErrorKind {
    /// A validation error: the call or its data is wrong for this device, for example a shader
    /// the backend rejects or a texture format that cannot be rendered to.
    Validation,
    /// The device ran out of memory.
    OutOfMemory,
    /// An internal error of wgpu or the driver, for example a failing backend shader compiler.
    Internal,
}

impl WgpuErrorKind {
    /// The kind of a wgpu error value.
    pub fn of(error: &wgpu::Error) -> Self {
        match error {
            wgpu::Error::Validation { .. } => Self::Validation,
            wgpu::Error::OutOfMemory { .. } => Self::OutOfMemory,
            wgpu::Error::Internal { .. } => Self::Internal,
        }
    }

    /// Lower-case name for messages: `"validation"`, `"out-of-memory"`, `"internal"`.
    pub fn name(self) -> &'static str {
        match self {
            Self::Validation => "validation",
            Self::OutOfMemory => "out-of-memory",
            Self::Internal => "internal",
        }
    }
}

/// Why an operation of the renderer infrastructure failed.
///
/// A bad shader, a format the device cannot use or a lost device is external input (contract
/// §2.6) and arrives here, never as a panic.
#[derive(Debug, Error)]
pub enum RenderError {
    /// The device is lost, or another failure of the GPU context.
    #[error(transparent)]
    Device(#[from] GpuError),
    /// wgpu reported an error inside the error scope of `operation`.
    #[error("{operation}: wgpu reported a {} error: {message}", kind.name())]
    Wgpu {
        /// What was being done, e.g. `"shader module overlay.wgsl"`.
        operation: String,
        /// Which error filter caught it.
        kind: WgpuErrorKind,
        /// wgpu's description; may span several lines (shader diagnostics).
        message: String,
    },
    /// A shader could not be preprocessed, parsed or validated.
    #[error(transparent)]
    Shader(#[from] ShaderError),
    /// A frame graph declaration or its use in a frame is wrong.
    #[error(transparent)]
    FrameGraph(#[from] FrameGraphError),
    /// Texture data does not match its description or the device limits.
    #[error(transparent)]
    Texture(#[from] TextureError),
    /// A material table entry is out of range.
    #[error(transparent)]
    Material(#[from] MaterialError),
    /// A headless target cannot be created as asked.
    #[error(transparent)]
    Headless(#[from] HeadlessError),
    /// A voxel scene operation or renderer configuration was refused.
    #[error(transparent)]
    Voxel(#[from] VoxelError),
    /// Data could not be read back from the GPU.
    #[error("reading back {what} failed: {reason}")]
    Readback {
        /// What was read, e.g. `"headless target"`.
        what: String,
        /// Why it failed.
        reason: String,
    },
}

impl RenderError {
    /// The error for a wgpu error that the scope of `operation` caught.
    pub fn wgpu(operation: &str, error: &wgpu::Error) -> Self {
        Self::Wgpu {
            operation: operation.to_owned(),
            kind: WgpuErrorKind::of(error),
            message: error.to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wgpu_errors_name_the_operation_and_the_kind() {
        let error = RenderError::Wgpu {
            operation: "shader module overlay.wgsl".to_owned(),
            kind: WgpuErrorKind::Validation,
            message: "bad".to_owned(),
        };
        assert_eq!(
            error.to_string(),
            "shader module overlay.wgsl: wgpu reported a validation error: bad"
        );
        let kinds = [
            WgpuErrorKind::Validation,
            WgpuErrorKind::OutOfMemory,
            WgpuErrorKind::Internal,
        ];
        assert_eq!(
            kinds.map(WgpuErrorKind::name),
            ["validation", "out-of-memory", "internal"]
        );
    }

    #[test]
    fn errors_stay_small_enough_to_return_by_value() {
        // clippy's result_large_err limit; every fallible GPU call returns this type.
        assert!(
            size_of::<RenderError>() <= 128,
            "{}",
            size_of::<RenderError>()
        );
    }
}
