//! Error scopes around GPU calls (contract §2.6, §4.5).
//!
//! The GPU context logs every wgpu error that no error scope catches. Calls whose failure the
//! caller must see - shader modules, pipelines, textures, buffers, submissions - run inside
//! [`capture`] instead, which turns what wgpu reports into a [`RenderError`].

use futures_lite::future::block_on;

use crate::error::RenderError;
use crate::gpu::GpuContext;

/// Runs `call` on the device inside error scopes for validation, out-of-memory and internal
/// errors and returns its value, or the first error wgpu reported while it ran.
///
/// wgpu's error scopes belong to the calling thread, so only errors of calls made by `call` on
/// this thread are caught. wgpu hands out an invalid handle for a failed creation; it is
/// dropped here together with the value of `call`.
///
/// ```no_run
/// use cp_render::gpu::{GpuContext, GpuOptions};
/// use cp_render::scope::capture;
/// use cp_render::wgpu;
///
/// let gpu = GpuContext::new_blocking(&GpuOptions::default(), None)?;
/// let buffer = capture(&gpu, "upload buffer", |device| {
///     device.create_buffer(&wgpu::BufferDescriptor {
///         label: Some("upload"),
///         size: 256,
///         usage: wgpu::BufferUsages::COPY_DST,
///         mapped_at_creation: false,
///     })
/// })?;
/// # let _ = buffer;
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
///
/// # Errors
/// [`RenderError::Device`] when the device is lost before or after the call,
/// [`RenderError::Wgpu`] with `operation` and wgpu's message when a scope caught an error.
pub fn capture<T>(
    gpu: &GpuContext,
    operation: &str,
    call: impl FnOnce(&wgpu::Device) -> T,
) -> Result<T, RenderError> {
    gpu.ensure_alive()?;
    let device = &gpu.device;
    // Innermost last: an error goes to the innermost scope whose filter matches, and the
    // guards have to be popped in the opposite order of the pushes.
    let internal = device.push_error_scope(wgpu::ErrorFilter::Internal);
    let memory = device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
    let validation = device.push_error_scope(wgpu::ErrorFilter::Validation);
    let value = call(device);
    // On native backends the futures are ready at once; nothing blocks for long here.
    let validation = block_on(validation.pop());
    let memory = block_on(memory.pop());
    let internal = block_on(internal.pop());
    if let Some(error) = validation.or(memory).or(internal) {
        return Err(RenderError::wgpu(operation, &error));
    }
    // A loss reported while the call ran leaves handles that silently do nothing.
    gpu.ensure_alive()?;
    Ok(value)
}
