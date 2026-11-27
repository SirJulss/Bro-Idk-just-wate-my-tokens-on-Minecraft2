//! GPU context and capability report (contract §4.5).
//!
//! [`GpuContext`] owns the wgpu instance, adapter, device and queue. Creating it negotiates
//! with the adapter: every optional feature the engine's pipeline can use is enabled when the
//! adapter has it, and the limits the pipeline needs are raised as far as the adapter allows.
//! The outcome is recorded in [`GpuCaps`], which also holds the conclusions the renderer acts
//! on ([`GpuDerived`], [`QualityTier`]) and prints itself as Markdown or JSON.
//!
//! ```no_run
//! use cp_render::gpu::{GpuContext, GpuOptions};
//!
//! let gpu = GpuContext::new_blocking(&GpuOptions::default(), None)?;
//! let report = gpu.caps.report_markdown();
//! # let _ = report;
//! # Ok::<(), cp_render::gpu::GpuError>(())
//! ```
//!
//! With a window, the surface has to exist before the adapter is chosen, and a wgpu surface
//! belongs to the instance that created it. [`GpuContext::with_surface`] keeps that order:
//!
//! ```no_run
//! # fn open<'w>(
//! #     window: impl Into<cp_render::wgpu::SurfaceTarget<'w>>,
//! # ) -> Result<(), cp_render::gpu::GpuError> {
//! use cp_render::gpu::{GpuContext, GpuOptions};
//!
//! let options = GpuOptions::default();
//! let (gpu, surface) =
//!     GpuContext::with_surface_blocking(&options, |instance| instance.create_surface(window))?;
//! # let _ = (gpu, surface);
//! # Ok(())
//! # }
//! ```
//!
//! The same in separate steps, for callers that need the instance earlier, is
//! [`GpuContext::create_instance`], `instance.create_surface(..)` and
//! [`GpuContext::with_instance`]. The contract's constructor [`GpuContext::new`] is the
//! headless one: it cannot take a surface, because no surface can exist for an instance that
//! `new` has yet to create.
//!
//! # Backends and adapters
//!
//! Vulkan is tried before DX12. `CP_BACKEND=vulkan` or `CP_BACKEND=dx12` forces one of them
//! ([`BACKEND_ENV`]) and wins over [`GpuOptions::backends`]. A hardware adapter is preferred to
//! a software rasteriser on any backend, and an adapter that fails to open a device is skipped
//! in favour of the next one.
//!
//! The instance gets the flags of [`GpuOptions::flags`]. Its DX12 backend compiles shaders
//! with FXC unless `CP_DXC=<path of dxcompiler.dll>` ([`DXC_ENV`]) selects DXC; both choices
//! are part of the report ([`GpuCaps::instance_flags`], [`GpuCaps::dx12_shader_compiler`]).
//!
//! # Missing features
//!
//! Nothing optional is required. A feature the adapter lacks is simply not enabled, a limit the
//! adapter cannot reach is clamped, and both are visible in the report and in [`GpuDerived`],
//! where the renderer finds the fallback to take. The draw path is [`DrawPath::A`] on Vulkan
//! with `MULTI_DRAW_INDIRECT_COUNT` and `INDIRECT_FIRST_INSTANCE`, [`DrawPath::B`] everywhere
//! else.
//!
//! # Errors after creation
//!
//! wgpu errors outside an error scope are written to the log instead of panicking, which is
//! what wgpu does by default.
//!
//! A lost device (driver reset, crash or update, GPU removed, memory pressure) is put on
//! record by the context as soon as wgpu reports it, and logged once. From then on
//! [`GpuContext::is_lost`] is `true`, and [`GpuContext::ensure_alive`] and
//! [`GpuContext::poll`] return [`GpuError::DeviceLost`]. The device does not come back;
//! whoever wants to go on creates a new context.
//!
//! Engine code polls the device through [`GpuContext::poll`], never through
//! `gpu.device.poll(..)`. wgpu 30 panics in `Device::poll` when the driver reports the loss
//! there, and on Vulkan at every call after it. [`GpuContext::poll`] looks for the loss with
//! a call that does not panic, leaves a lost device alone, and contains wgpu's panic in the
//! one case that is left, a loss during a blocking wait.
//!
//! # Testing without a GPU
//!
//! The negotiation is pure: [`AdapterProfile::plan_device`] and [`GpuCaps::negotiate`] work on
//! plain data, so every fallback path can be tested with a fabricated adapter.

mod adapter;
mod caps;
mod context;
mod error;
mod features;
mod health;
mod instance;
mod json;
mod limits;
mod options;
mod report;
#[cfg(test)]
pub(crate) mod testing;

/// The probe draw of the adapter tests in `tests/`, shared with the unit tests, which can
/// build contexts the public API does not offer.
#[cfg(test)]
#[path = "../../tests/common/probe.rs"]
mod probe;

pub use caps::{
    AdapterProfile, DevicePlan, DrawPath, FeatureStatus, GpuCaps, GpuDerived, QualityTier,
};
pub use context::GpuContext;
pub use error::GpuError;
pub use instance::GpuInstance;
pub use options::{BACKEND_ENV, DXC_ENV, Dx12ShaderCompiler, GpuOptions};
