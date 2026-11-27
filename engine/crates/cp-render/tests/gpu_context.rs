//! Tests of `cp_render::gpu` against the real GPU of the machine, through the public API only.
//!
//! Every test skips with a log line when the machine has no GPU adapter (contract §2.7). Run
//! with `cargo test -p cp-render --test gpu_context -- --nocapture` to see the capability
//! report of the adapter; `CP_BACKEND=dx12` selects the DX12 fallback.

use std::sync::{Mutex, MutexGuard, Once, PoisonError};

use cp_render::gpu::{BACKEND_ENV, DrawPath, GpuContext, GpuError, GpuOptions};
use cp_render::wgpu;

/// The probe draw and the check of the draw-path promises, shared with the crate's unit tests.
#[path = "common/probe.rs"]
mod probe;

/// Collects the log output of the crate under test and echoes it to stderr, where the test
/// harness shows it for failed tests and with `--nocapture`.
struct TestLogger;

static LOGGER: TestLogger = TestLogger;
static RECORDS: Mutex<Vec<(log::Level, String)>> = Mutex::new(Vec::new());

/// The collected log lines. A test that failed while logging must not fail the others, so a
/// poisoned lock is used as it is.
fn records() -> MutexGuard<'static, Vec<(log::Level, String)>> {
    RECORDS.lock().unwrap_or_else(PoisonError::into_inner)
}

impl log::Log for TestLogger {
    fn enabled(&self, metadata: &log::Metadata<'_>) -> bool {
        // Everything from this crate and its tests, down to the debug lines some tests look
        // for; from wgpu and the drivers only errors, because Vulkan loaders warn at length
        // about every installed overlay layer.
        let own = ["cp_render", "gpu_context"];
        metadata.level() <= log::Level::Error
            || (metadata.level() <= log::Level::Debug
                && own
                    .iter()
                    .any(|prefix| metadata.target().starts_with(prefix)))
    }

    // The test harness captures what the print macros write, and only that; this logger is the
    // one place where the adapter tests show their output (contract §2.6 keeps libraries silent).
    #[allow(clippy::print_stderr)]
    fn log(&self, record: &log::Record<'_>) {
        if self.enabled(record.metadata()) {
            let line = format!("[{} {}] {}", record.level(), record.target(), record.args());
            eprintln!("{line}");
            records().push((record.level(), line));
        }
    }

    fn flush(&self) {}
}

fn init_logging() {
    static INIT: Once = Once::new();
    INIT.call_once(|| {
        log::set_logger(&LOGGER).expect("no other logger is installed in this test binary");
        log::set_max_level(log::LevelFilter::Debug);
    });
}

/// `true` when a log line of `level` contains every one of `needles`.
fn logged(level: log::Level, needles: &[&str]) -> bool {
    records()
        .iter()
        .any(|(line_level, line)| *line_level == level && needles.iter().all(|n| line.contains(n)))
}

/// The context of the default adapter, or `None` after a log line when the machine has no GPU
/// adapter or the build has neither backend of the contract. Anything else fails the test.
fn context_or_skip(test: &str) -> Option<GpuContext> {
    init_logging();
    match GpuContext::new_blocking(&GpuOptions::default(), None) {
        Ok(gpu) => Some(gpu),
        Err(error @ (GpuError::NoAdapter { .. } | GpuError::BackendNotCompiled { .. })) => {
            log::warn!("{test}: skipped: {error}");
            None
        }
        Err(error) => panic!("{test}: GPU context creation failed: {error}"),
    }
}

#[test]
fn creates_the_context_and_prints_the_report() {
    let Some(gpu) = context_or_skip("creates_the_context_and_prints_the_report") else {
        return;
    };
    let caps = &gpu.caps;

    // The record describes the adapter and the device that really exist.
    assert_eq!(caps.adapter, gpu.adapter.get_info());
    assert_eq!(caps.available_features, gpu.adapter.features());
    assert_eq!(caps.adapter_limits, gpu.adapter.limits());
    assert_eq!(caps.enabled_features, gpu.device.features());
    assert_eq!(caps.enabled_limits, gpu.device.limits());
    assert_eq!(caps.downlevel, gpu.adapter.get_downlevel_capabilities());

    // Negotiation invariants on real data.
    assert!(caps.available_features.contains(caps.enabled_features));
    assert!(caps.enabled_limits.check_limits(&caps.adapter_limits));
    for status in caps.optional_features() {
        assert!(
            !status.enabled || status.available,
            "{} enabled but unavailable",
            status.name
        );
        assert_eq!(
            status.enabled,
            gpu.device.features().contains(status.feature)
        );
    }

    // The backend order of the contract: nothing but Vulkan or DX12 by default.
    assert!(
        matches!(
            caps.adapter.backend,
            wgpu::Backend::Vulkan | wgpu::Backend::Dx12
        ),
        "unexpected backend {:?}",
        caps.adapter.backend
    );

    // Both reports exist; their structure is checked by the unit tests of the crate.
    let markdown = caps.report_markdown();
    assert!(markdown.starts_with("# GPU capability report\n"));
    for section in [
        "| Adapter ",
        "## Derived capabilities",
        "## Optional features",
        "## Limits",
        "## Downlevel flags",
    ] {
        assert!(markdown.contains(section), "the report lacks {section:?}");
    }
    let json = caps.report_json();
    assert!(json.starts_with("{\n") && json.ends_with("}\n"));
    log::info!("capability report of this machine:\n\n{markdown}");
}

/// The path a windowed application takes, minus the window: instance first, context second.
#[test]
fn with_instance_gives_the_same_device_as_new() {
    let test = "with_instance_gives_the_same_device_as_new";
    let Some(direct) = context_or_skip(test) else {
        return;
    };
    let options = GpuOptions::default();
    let instance = GpuContext::create_instance(&options)
        .unwrap_or_else(|error| panic!("{test}: instance creation failed: {error}"));
    let gpu = GpuContext::with_instance_blocking(&instance, &options, None)
        .unwrap_or_else(|error| panic!("{test}: context creation failed: {error}"));
    assert_eq!(gpu.caps, direct.caps);
    assert_eq!(gpu.device.features(), direct.device.features());
    assert_eq!(gpu.device.limits(), direct.device.limits());
}

#[test]
fn uncaptured_errors_are_logged_instead_of_panicking() {
    let Some(gpu) = context_or_skip("uncaptured_errors_are_logged_instead_of_panicking") else {
        return;
    };
    // Four bytes more than the device allows: a validation error without any allocation.
    // wgpu's default reaction would be a panic.
    let label = "cp-test-oversized-buffer";
    let buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: gpu.caps.enabled_limits.max_buffer_size + 4,
        usage: wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    drop(buffer);
    assert!(
        logged(
            log::Level::Error,
            &["uncaptured GPU error (validation)", label]
        ),
        "the validation error was not logged"
    );

    // The device is still usable afterwards.
    let buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("cp-test-small-buffer"),
        size: 64,
        usage: wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    gpu.queue.write_buffer(&buffer, 0, &[7; 64]);
    gpu.queue.submit([]);
    assert!(!logged(log::Level::Error, &["cp-test-small-buffer"]));
}

#[test]
fn errors_inside_an_error_scope_stay_with_the_scope() {
    let Some(gpu) = context_or_skip("errors_inside_an_error_scope_stay_with_the_scope") else {
        return;
    };
    let label = "cp-test-scoped-oversized-buffer";
    let scope = gpu.device.push_error_scope(wgpu::ErrorFilter::Validation);
    let buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: gpu.caps.enabled_limits.max_buffer_size + 4,
        usage: wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    drop(buffer);
    let error = futures_lite::future::block_on(scope.pop());
    assert!(
        matches!(error, Some(wgpu::Error::Validation { .. })),
        "got {error:?}"
    );
    assert!(
        !logged(log::Level::Error, &[label]),
        "a captured error must not be logged"
    );
}

/// `true` for the error of a device that is gone because `Device::destroy` was called.
fn is_destroyed<T>(outcome: &Result<T, GpuError>) -> bool {
    matches!(
        outcome,
        Err(GpuError::DeviceLost {
            reason: wgpu::DeviceLostReason::Destroyed,
            ..
        })
    )
}

#[test]
fn destroying_the_device_is_logged_as_information() {
    let Some(gpu) = context_or_skip("destroying_the_device_is_logged_as_information") else {
        return;
    };
    let clone = gpu.clone();
    assert!(!gpu.is_lost());
    assert!(gpu.ensure_alive().is_ok());
    let poll = gpu.poll(wgpu::PollType::Poll);
    assert!(poll.is_ok(), "polling a working device failed: {poll:?}");

    gpu.device.destroy();
    // wgpu completes the destroy, and runs the callback, when the device is maintained after
    // its queue has drained. That poll is the first to know and says so.
    let poll = gpu.poll(wgpu::PollType::Poll);
    assert!(is_destroyed(&poll), "the poll after the destroy: {poll:?}");
    assert!(
        logged(log::Level::Info, &["GPU device destroyed"]),
        "the callback did not run"
    );
    assert!(!logged(log::Level::Error, &["GPU device lost"]));

    // The state belongs to the device: every clone has it, and it stays.
    assert!(gpu.is_lost() && clone.is_lost());
    assert!(is_destroyed(&gpu.ensure_alive()));
    assert!(is_destroyed(&clone.ensure_alive()));
    for poll_type in [wgpu::PollType::Poll, wgpu::PollType::wait_indefinitely()] {
        let poll = clone.poll(poll_type);
        assert!(is_destroyed(&poll), "a later poll: {poll:?}");
    }
}

/// A renderer that has not noticed the loss yet keeps calling into the dead device, and every
/// call fails. The loss is in the log once; its consequences must not bury it frame by frame.
#[test]
fn errors_on_a_lost_device_do_not_flood_the_log() {
    let Some(gpu) = context_or_skip("errors_on_a_lost_device_do_not_flood_the_log") else {
        return;
    };
    gpu.device.destroy();
    let poll = gpu.poll(wgpu::PollType::Poll);
    assert!(is_destroyed(&poll), "{poll:?}");

    let label = "cp-test-buffer-on-a-lost-device";
    let buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: 64,
        usage: wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    gpu.queue.write_buffer(&buffer, 0, &[7; 64]);
    gpu.queue.submit([]);
    assert!(
        logged(log::Level::Debug, &["on the lost device", label]),
        "the failed call was not reported at all"
    );
    assert!(
        !logged(log::Level::Error, &[label]),
        "a failed call on the lost device was logged as an error"
    );
}

/// A window that cannot become a surface: it has no handles to offer.
struct NoWindow;

impl wgpu::rwh::HasWindowHandle for NoWindow {
    fn window_handle(&self) -> Result<wgpu::rwh::WindowHandle<'_>, wgpu::rwh::HandleError> {
        Err(wgpu::rwh::HandleError::Unavailable)
    }
}

impl wgpu::rwh::HasDisplayHandle for NoWindow {
    fn display_handle(&self) -> Result<wgpu::rwh::DisplayHandle<'_>, wgpu::rwh::HandleError> {
        Err(wgpu::rwh::HandleError::Unavailable)
    }
}

/// The windowed constructor up to the point a test can reach without a window (contract §2.7):
/// the closure gets the new instance, and its failure is the error of the constructor.
#[test]
fn with_surface_reports_a_surface_that_cannot_be_created() {
    let test = "with_surface_reports_a_surface_that_cannot_be_created";
    init_logging();
    let mut calls = 0;
    let outcome = GpuContext::with_surface_blocking(&GpuOptions::default(), |instance| {
        calls += 1;
        instance.create_surface(NoWindow)
    });
    match outcome {
        Err(GpuError::SurfaceCreation { reason }) => {
            assert_eq!(calls, 1);
            let message = GpuError::SurfaceCreation { reason }.to_string();
            assert!(
                message.starts_with("the surface could not be created: "),
                "{message}"
            );
        }
        // Without a backend there is no instance to create a surface from.
        Err(error @ GpuError::BackendNotCompiled { .. }) => {
            assert_eq!(calls, 0);
            log::warn!("{test}: skipped: {error}");
        }
        Err(error) => panic!("{test}: expected SurfaceCreation, got {error}"),
        Ok(_) => panic!("{test}: a context without a surface"),
    }
}

/// One context per backend of the contract that has an adapter on this machine, so that the
/// DX12 fallback is exercised in every test run and not only when somebody sets `CP_BACKEND`.
/// With `CP_BACKEND` set, the options cannot choose the backend; then it is the one context of
/// the forced backend.
fn contexts_per_backend(test: &str) -> Vec<GpuContext> {
    init_logging();
    if std::env::var_os(BACKEND_ENV).is_some() {
        return context_or_skip(test).into_iter().collect();
    }
    let mut contexts = Vec::new();
    for backends in [wgpu::Backends::VULKAN, wgpu::Backends::DX12] {
        let options = GpuOptions {
            backends: Some(backends),
            ..GpuOptions::default()
        };
        match GpuContext::new_blocking(&options, None) {
            Ok(gpu) => {
                let used = wgpu::Backends::from(gpu.caps.adapter.backend);
                assert_eq!(used, backends, "{test}: the option asked for {backends:?}");
                contexts.push(gpu);
            }
            Err(error @ (GpuError::NoAdapter { .. } | GpuError::BackendNotCompiled { .. })) => {
                log::warn!("{test}: {backends:?} skipped: {error}");
            }
            Err(error) => panic!("{test}: GPU context creation on {backends:?} failed: {error}"),
        }
    }
    contexts
}

#[test]
fn the_backend_option_selects_the_backend() {
    let test = "the_backend_option_selects_the_backend";
    if std::env::var_os(BACKEND_ENV).is_some() {
        init_logging();
        log::warn!("{test}: skipped, {BACKEND_ENV} overrides the option under test");
        return;
    }
    for gpu in contexts_per_backend(test) {
        let caps = &gpu.caps;
        // The same negotiation on every backend: only what the adapter lacks may differ.
        assert!(caps.available_features.contains(caps.enabled_features));
        assert!(caps.enabled_limits.check_limits(&caps.adapter_limits));
        assert_eq!(caps.enabled_features, gpu.device.features());
        assert_eq!(caps.enabled_limits, gpu.device.limits());
        log::info!(
            "{test}: {} on {}: {} tier, draw path {}",
            caps.adapter.name,
            caps.adapter.backend,
            caps.tier.id(),
            caps.derived.draw_path.id()
        );
    }
}

/// What a test skips on: an instance that lacks the requested backend has no adapter to offer,
/// and that has to arrive as `GpuError::NoAdapter`, not as a panic inside wgpu.
#[test]
fn a_backend_without_adapters_is_reported_as_no_adapter() {
    let test = "a_backend_without_adapters_is_reported_as_no_adapter";
    init_logging();
    if std::env::var_os(BACKEND_ENV).is_some() {
        log::warn!("{test}: skipped, {BACKEND_ENV} overrides the options under test");
        return;
    }
    let vulkan_only = GpuOptions {
        backends: Some(wgpu::Backends::VULKAN),
        ..GpuOptions::default()
    };
    let dx12_only = GpuOptions {
        backends: Some(wgpu::Backends::DX12),
        ..GpuOptions::default()
    };
    let instance = match GpuContext::create_instance(&vulkan_only) {
        Ok(instance) => instance,
        Err(error @ GpuError::BackendNotCompiled { .. }) => {
            log::warn!("{test}: skipped: {error}");
            return;
        }
        Err(error) => panic!("{test}: instance creation failed: {error}"),
    };
    match GpuContext::with_instance_blocking(&instance, &dx12_only, None) {
        Err(GpuError::NoAdapter {
            backends,
            surface_required,
        }) => {
            assert_eq!(backends, wgpu::Backends::DX12);
            assert!(!surface_required);
        }
        // A build without DX12 cannot ask for it in the first place.
        Err(error @ GpuError::BackendNotCompiled { .. }) => {
            log::warn!("{test}: skipped: {error}");
        }
        other => panic!("{test}: expected NoAdapter, got {other:?}"),
    }
}

/// The draw booleans are derived from reading the sources of wgpu. This test holds every
/// `true` among them, and the bucket fetch of draw path A or B, against what the GPU really
/// does on every backend; a `false` promises nothing and is only logged with the observation.
#[test]
fn indirect_draws_deliver_first_vertex_and_instance_where_promised() {
    let test = "indirect_draws_deliver_first_vertex_and_instance_where_promised";
    for gpu in contexts_per_backend(test) {
        probe::check_indirect_draw_promises(test, &gpu);
    }
}
