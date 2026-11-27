//! Synthetic adapters for the unit tests of the negotiation and the reports, and the log
//! output of the unit tests.

use std::sync::Once;

use super::caps::AdapterProfile;
use super::features::wanted_features;

/// Writes the log records of the unit tests to stderr, where the test harness shows them for
/// failed tests and with `--nocapture`. Library code logs through `log` only (contract §2.6);
/// without a logger, the skip lines of the adapter-gated tests (§2.7) would go nowhere.
struct StderrLogger;

static LOGGER: StderrLogger = StderrLogger;

impl log::Log for StderrLogger {
    fn enabled(&self, metadata: &log::Metadata<'_>) -> bool {
        // The crate's own lines; from wgpu and the drivers only errors.
        metadata.level() <= log::Level::Error
            || (metadata.level() <= log::Level::Info && metadata.target().starts_with("cp_render"))
    }

    // The test harness captures what the print macros write, and only that.
    #[allow(clippy::print_stderr)]
    fn log(&self, record: &log::Record<'_>) {
        if self.enabled(record.metadata()) {
            eprintln!("[{} {}] {}", record.level(), record.target(), record.args());
        }
    }

    fn flush(&self) {}
}

/// Installs [`StderrLogger`] once per test binary.
pub(crate) fn init_logging() {
    static INIT: Once = Once::new();
    INIT.call_once(|| {
        // Only this function installs a logger in the unit tests.
        if log::set_logger(&LOGGER).is_ok() {
            log::set_max_level(log::LevelFilter::Info);
        }
    });
}

/// Logs that `test` (or a part of it) is skipped and why (contract §2.7: log and return).
pub(crate) fn skip(test: &str, reason: &dyn std::fmt::Display) {
    init_logging();
    log::warn!("{test}: skipped: {reason}");
}

/// Adapter limits of a desktop GPU as wgpu 30 reports them on Vulkan (shape of a GTX 1080;
/// the exact numbers do not matter, only that they exceed what the engine asks for).
pub(crate) fn desktop_limits() -> wgpu::Limits {
    wgpu::Limits {
        max_texture_dimension_1d: 32768,
        max_texture_dimension_2d: 32768,
        max_texture_dimension_3d: 16384,
        max_texture_array_layers: 2048,
        max_bind_groups: 8,
        max_storage_buffers_per_shader_stage: 524_288,
        max_sampled_textures_per_shader_stage: 524_288,
        max_storage_textures_per_shader_stage: 524_288,
        max_color_attachment_bytes_per_sample: 128,
        max_uniform_buffer_binding_size: 65536,
        max_storage_buffer_binding_size: (1 << 31) - 4,
        max_buffer_size: u64::from(u32::MAX),
        min_uniform_buffer_offset_alignment: 64,
        min_storage_buffer_offset_alignment: 32,
        max_immediate_size: 256,
        max_buffers_and_acceleration_structures_per_shader_stage: u32::MAX,
        ..wgpu::Limits::defaults()
    }
}

/// A WebGPU-compliant adapter that offers every optional feature of the pipeline plus a few
/// the engine does not ask for.
pub(crate) fn desktop_profile(
    backend: wgpu::Backend,
    device_type: wgpu::DeviceType,
) -> AdapterProfile {
    let mut info = wgpu::AdapterInfo::new(device_type, backend);
    info.name = "Test GPU 9000".to_owned();
    info.vendor = 0x10DE;
    info.device = 0x1B80;
    info.driver = "TestDriver".to_owned();
    info.driver_info = "1.2.3".to_owned();
    AdapterProfile {
        info,
        features: wanted_features()
            | wgpu::Features::CLEAR_TEXTURE
            | wgpu::Features::POLYGON_MODE_LINE
            | wgpu::Features::SHADER_F16,
        limits: desktop_limits(),
        downlevel: wgpu::DownlevelCapabilities::default(),
    }
}

/// Instance flags of a release build: only wgpu's indirect-call validation.
pub(crate) fn indirect_validation() -> wgpu::InstanceFlags {
    wgpu::InstanceFlags::VALIDATION_INDIRECT_CALL
}

/// Instance flags with indirect-call validation switched off.
pub(crate) fn no_flags() -> wgpu::InstanceFlags {
    wgpu::InstanceFlags::empty()
}
