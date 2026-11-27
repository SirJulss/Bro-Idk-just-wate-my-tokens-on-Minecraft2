//! `cpe caps`: the capability report of the real adapter (contract 4.5).
//!
//! Creates the one headless [`GpuContext`] of the process with the default options (Vulkan
//! first, then DX12; `CP_BACKEND` and `CP_DXC` apply), and returns the report that
//! `cp-render` derives from the negotiation. The report does not depend on the time or on the
//! run, so saving it twice on one machine gives the same files.

use cp_render::gpu::{GpuContext, GpuOptions};

use crate::cli::CapsArgs;
use crate::{Failure, write_file};

/// Opens the GPU, saves the reports `args` asks for and returns the Markdown report.
///
/// # Errors
/// A [`Failure`] when no GPU device can be opened (the message is the one of
/// [`cp_render::gpu::GpuError`]) or when a report cannot be written.
pub fn run(args: &CapsArgs) -> Result<String, Failure> {
    let gpu = GpuContext::new_blocking(&GpuOptions::default(), None)
        .map_err(|error| Failure::new(format!("cannot open a GPU device: {error}")))?;
    log::info!(
        "adapter \"{}\" on {:?}, draw path {}",
        gpu.caps.adapter.name,
        gpu.caps.adapter.backend,
        gpu.caps.derived.draw_path.id()
    );
    let markdown = gpu.caps.report_markdown();
    if let Some(path) = &args.out {
        write_file(path, markdown.as_bytes())?;
    }
    if let Some(path) = &args.json {
        write_file(path, gpu.caps.report_json().as_bytes())?;
    }
    Ok(markdown)
}
