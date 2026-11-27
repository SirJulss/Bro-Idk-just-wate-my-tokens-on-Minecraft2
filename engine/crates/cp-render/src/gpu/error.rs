//! Errors of the GPU context.

use thiserror::Error;

use super::options::backend_list;

/// What can go wrong with a [`GpuContext`](super::GpuContext): why it could not be created,
/// and why its device stopped working afterwards.
///
/// A missing, unusable or lost GPU is external input (contract §2.6): it is reported through
/// this type and never as a panic.
#[derive(Debug, Error)]
pub enum GpuError {
    /// The `CP_BACKEND` environment variable holds something other than `vulkan` or `dx12`.
    #[error("CP_BACKEND={value:?} is not a supported backend; use \"vulkan\" or \"dx12\"")]
    InvalidBackendOverride {
        /// The rejected value exactly as found in the environment.
        value: String,
    },

    /// The `CP_DXC` environment variable names a file that does not exist.
    ///
    /// wgpu would otherwise drop the DX12 backend from the instance without an error of its
    /// own, and the context would end up on another backend or without an adapter.
    #[error("CP_DXC={path:?} is not a file; it must name dxcompiler.dll, or be unset for FXC")]
    DxcNotFound {
        /// The value exactly as found in the environment.
        path: String,
    },

    /// None of the requested backends is compiled into this build for this platform.
    #[error(
        "none of the requested backends ({}) is part of this build (compiled in: {})",
        backend_list(.requested),
        backend_list(.compiled)
    )]
    BackendNotCompiled {
        /// Backends asked for by the options or by `CP_BACKEND`.
        requested: wgpu::Backends,
        /// Backends this build can use on this platform.
        compiled: wgpu::Backends,
    },

    /// No adapter exists on the allowed backends, or none of them can present to the surface.
    #[error("{}", no_adapter_message(.backends, .surface_required))]
    NoAdapter {
        /// Backends that were searched.
        backends: wgpu::Backends,
        /// `true` when adapters had to be able to present to a surface.
        surface_required: bool,
    },

    /// [`GpuContext::new`](super::GpuContext::new) was given a surface.
    ///
    /// A wgpu surface belongs to the instance that created it, and `new` creates its own
    /// instance, so the surface can never match. Use
    /// [`GpuContext::with_surface`](super::GpuContext::with_surface): it creates the instance,
    /// lets the caller create the surface from it and picks an adapter that can present to it.
    #[error(
        "a surface only works with the instance that created it: use GpuContext::with_surface, \
         which creates the instance and lets the caller create the surface from it"
    )]
    SurfaceNeedsItsInstance,

    /// The closure given to [`GpuContext::with_surface`](super::GpuContext::with_surface)
    /// failed to create the surface.
    ///
    /// The message already contains wgpu's reason, so the reason is a plain field and not the
    /// `source()` of this error; printing an error chain would otherwise show it twice.
    #[error("the surface could not be created: {reason}")]
    SurfaceCreation {
        /// The reason given by wgpu.
        reason: wgpu::CreateSurfaceError,
    },

    /// The adapter rejected the negotiated features and limits.
    ///
    /// The message already contains wgpu's reason, so the reason is a plain field and not the
    /// `source()` of this error; printing an error chain would otherwise show it twice.
    #[error("adapter \"{adapter}\" refused to create a device: {reason}")]
    DeviceRequest {
        /// Name of the adapter that was asked.
        adapter: String,
        /// The reason given by wgpu.
        reason: wgpu::RequestDeviceError,
    },

    /// The device of the context is gone and does not come back; a new context is needed.
    ///
    /// Returned by [`GpuContext::poll`](super::GpuContext::poll) and
    /// [`GpuContext::ensure_alive`](super::GpuContext::ensure_alive) from the moment wgpu
    /// reports the loss.
    #[error("{}", device_lost_message(.reason, .message))]
    DeviceLost {
        /// `Unknown` for a loss the application did not ask for: driver reset, crash or
        /// update, GPU removed, memory pressure. `Destroyed` after `wgpu::Device::destroy`.
        reason: wgpu::DeviceLostReason,
        /// What wgpu or the driver said about it; may be empty.
        message: String,
    },

    /// [`GpuContext::poll`](super::GpuContext::poll) could not do what was asked on a device
    /// that works: the wait timed out, or it named a submission that does not exist.
    #[error("polling the GPU device failed: {reason}")]
    Poll {
        /// The reason given by wgpu.
        reason: wgpu::PollError,
    },
}

fn no_adapter_message(backends: &wgpu::Backends, surface_required: &bool) -> String {
    let suffix = if *surface_required {
        " that can present to the surface"
    } else {
        ""
    };
    format!("no GPU adapter found on {}{suffix}", backend_list(backends))
}

fn device_lost_message(reason: &wgpu::DeviceLostReason, message: &str) -> String {
    let what = match reason {
        wgpu::DeviceLostReason::Destroyed => "the GPU device was destroyed",
        wgpu::DeviceLostReason::Unknown => "the GPU device is lost",
    };
    if message.is_empty() {
        what.to_owned()
    } else {
        format!("{what}: {message}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn messages_name_the_backends_and_the_cause() {
        let error = GpuError::InvalidBackendOverride {
            value: "dx11".to_owned(),
        };
        assert_eq!(
            error.to_string(),
            "CP_BACKEND=\"dx11\" is not a supported backend; use \"vulkan\" or \"dx12\""
        );
        let error = GpuError::DxcNotFound {
            path: "C:/nowhere/dxcompiler.dll".to_owned(),
        };
        assert_eq!(
            error.to_string(),
            "CP_DXC=\"C:/nowhere/dxcompiler.dll\" is not a file; it must name dxcompiler.dll, \
             or be unset for FXC"
        );

        let error = GpuError::NoAdapter {
            backends: wgpu::Backends::VULKAN | wgpu::Backends::DX12,
            surface_required: false,
        };
        assert_eq!(error.to_string(), "no GPU adapter found on vulkan, dx12");

        let error = GpuError::NoAdapter {
            backends: wgpu::Backends::DX12,
            surface_required: true,
        };
        assert_eq!(
            error.to_string(),
            "no GPU adapter found on dx12 that can present to the surface"
        );

        let error = GpuError::BackendNotCompiled {
            requested: wgpu::Backends::GL,
            compiled: wgpu::Backends::VULKAN | wgpu::Backends::DX12,
        };
        assert_eq!(
            error.to_string(),
            "none of the requested backends (gl) is part of this build (compiled in: vulkan, dx12)"
        );
    }

    #[test]
    fn a_surface_for_new_points_to_the_constructor_that_works() {
        let message = GpuError::SurfaceNeedsItsInstance.to_string();
        assert!(message.contains("GpuContext::with_surface"), "{message}");
    }

    #[test]
    fn device_loss_messages_tell_a_loss_from_a_destroy() {
        let error = GpuError::DeviceLost {
            reason: wgpu::DeviceLostReason::Unknown,
            message: "Device is lost".to_owned(),
        };
        assert_eq!(error.to_string(), "the GPU device is lost: Device is lost");

        let error = GpuError::DeviceLost {
            reason: wgpu::DeviceLostReason::Unknown,
            message: String::new(),
        };
        assert_eq!(error.to_string(), "the GPU device is lost");

        let error = GpuError::DeviceLost {
            reason: wgpu::DeviceLostReason::Destroyed,
            message: String::new(),
        };
        assert_eq!(error.to_string(), "the GPU device was destroyed");
    }

    #[test]
    fn poll_messages_carry_the_reason_of_wgpu() {
        let error = GpuError::Poll {
            reason: wgpu::PollError::Timeout,
        };
        let message = error.to_string();
        assert!(
            message.starts_with("polling the GPU device failed: ") && message.contains("timed out"),
            "{message}"
        );
    }
}
