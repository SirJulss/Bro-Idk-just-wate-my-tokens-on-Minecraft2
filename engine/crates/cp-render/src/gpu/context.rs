//! Creation of the adapter, device and queue, and the state of the device afterwards.

use std::sync::Arc;

use super::adapter::{Candidate, first_success, rank_adapters};
use super::caps::{AdapterProfile, DevicePlan, GpuCaps};
use super::health::DeviceHealth;
use super::instance::{GpuInstance, backends_for};
use super::options::backend_list;
use super::{GpuError, GpuOptions};

/// The GPU objects every part of the renderer shares, and what the device can do.
///
/// The wgpu handles are reference-counted, so cloning the context is cheap; all clones refer
/// to the same device and share what is known about its state.
///
/// # After creation
///
/// - wgpu errors that no error scope captures are written to the log (`log::error!`, target
///   `cp_render::gpu::context`) instead of panicking.
/// - A lost device is put on record and logged once (same target): [`Self::is_lost`],
///   [`Self::ensure_alive`] and [`Self::poll`] report it from then on. A lost device does not
///   come back; the way on is a new context. The errors of the calls that still reach the lost
///   device are logged at debug level only.
/// - The device is polled through [`Self::poll`] and never through `device.poll(..)`, which
///   panics in wgpu 30 when the driver reports the loss there.
#[derive(Clone, Debug)]
pub struct GpuContext {
    /// The wgpu instance. Surfaces used with this context must be created from it.
    pub instance: wgpu::Instance,
    /// The adapter the device was created on.
    pub adapter: wgpu::Adapter,
    /// The device, with exactly the features and limits recorded in [`Self::caps`].
    ///
    /// Two things on it belong to the context: do not call `poll` on it (use [`Self::poll`]),
    /// and do not replace its device-lost callback or its handler for uncaptured errors,
    /// through which the context learns about a loss.
    pub device: wgpu::Device,
    /// The queue of [`Self::device`].
    pub queue: wgpu::Queue,
    /// What the adapter offers, what is enabled on the device and what follows from it.
    pub caps: GpuCaps,
    /// Whether the device still works; fed by the callbacks installed on [`Self::device`].
    health: Arc<DeviceHealth>,
}

impl GpuContext {
    /// Creates the wgpu instance for `opts`: the backends of
    /// [`GpuOptions::resolved_backends`] that this build contains.
    ///
    /// Only needed by callers that want the instance before the context, the first of the
    /// three steps behind [`Self::with_surface`]: this function,
    /// `instance.create_surface(..)`, then [`Self::with_instance`]. Headless callers use
    /// [`Self::new`], windowed ones normally [`Self::with_surface`].
    ///
    /// The instance gets the flags of [`GpuOptions::flags`] (by default wgpu's flags for the
    /// build type: API validation in debug builds, indirect-call validation always). Its DX12
    /// backend compiles shaders with FXC, or with DXC when `CP_DXC` names `dxcompiler.dll`
    /// ([`DXC_ENV`](super::DXC_ENV)). The returned [`GpuInstance`] remembers both, see
    /// [`GpuInstance::flags`] and [`GpuInstance::dx12_shader_compiler`].
    ///
    /// # Errors
    /// [`GpuError::InvalidBackendOverride`], [`GpuError::DxcNotFound`] or
    /// [`GpuError::BackendNotCompiled`].
    pub fn create_instance(opts: &GpuOptions) -> Result<GpuInstance, GpuError> {
        GpuInstance::new(opts)
    }

    /// Creates instance, adapter, device and queue for headless use.
    ///
    /// Picks the adapter by backend order (Vulkan, then DX12) and `opts.power`, enables every
    /// optional feature the pipeline can use and the adapter has, and raises the limits the
    /// pipeline needs as far as the adapter allows (see [`AdapterProfile::plan_device`]).
    ///
    /// `surface` must be `None`. The parameter exists because the contract lists it, but a
    /// surface cannot belong to the instance this function is about to create; for a window
    /// use [`Self::with_surface`].
    ///
    /// # Errors
    /// [`GpuError::SurfaceNeedsItsInstance`] when `surface` is `Some`, otherwise the errors of
    /// [`Self::create_instance`] and [`Self::with_instance`].
    pub async fn new(
        opts: &GpuOptions,
        surface: Option<&wgpu::Surface<'_>>,
    ) -> Result<Self, GpuError> {
        if surface.is_some() {
            return Err(GpuError::SurfaceNeedsItsInstance);
        }
        let instance = GpuInstance::new(opts)?;
        Self::with_instance(&instance, opts, None).await
    }

    /// Creates instance, surface, adapter, device and queue for a window, in the one order
    /// wgpu allows: the instance first, then the surface from that instance, then an adapter
    /// that can present to the surface.
    ///
    /// `create_surface` is called once with the new instance and returns the surface of the
    /// window, normally `|instance| instance.create_surface(window)`. The surface comes back
    /// next to the context; it has to be configured with the context's device before use.
    ///
    /// ```no_run
    /// # fn open<'w>(
    /// #     window: impl Into<cp_render::wgpu::SurfaceTarget<'w>>,
    /// # ) -> Result<(), cp_render::gpu::GpuError> {
    /// use cp_render::gpu::{GpuContext, GpuOptions};
    ///
    /// let options = GpuOptions::default();
    /// let (gpu, surface) =
    ///     GpuContext::with_surface_blocking(&options, |instance| instance.create_surface(window))?;
    /// # let _ = (gpu, surface);
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Errors
    /// [`GpuError::SurfaceCreation`] when `create_surface` fails, otherwise the errors of
    /// [`Self::create_instance`] and [`Self::with_instance`]; [`GpuError::NoAdapter`] then
    /// means that no adapter can present to the surface.
    pub async fn with_surface<'window>(
        opts: &GpuOptions,
        create_surface: impl FnOnce(
            &wgpu::Instance,
        ) -> Result<wgpu::Surface<'window>, wgpu::CreateSurfaceError>,
    ) -> Result<(Self, wgpu::Surface<'window>), GpuError> {
        let instance = GpuInstance::new(opts)?;
        let surface =
            create_surface(&instance).map_err(|reason| GpuError::SurfaceCreation { reason })?;
        let gpu = Self::with_instance(&instance, opts, Some(&surface)).await?;
        Ok((gpu, surface))
    }

    /// Creates adapter, device and queue on an existing instance.
    ///
    /// `instance` comes from [`Self::create_instance`]; its flags go into the capability
    /// record as they are:
    ///
    /// ```no_run
    /// use cp_render::gpu::{GpuContext, GpuOptions};
    ///
    /// let options = GpuOptions::default();
    /// let instance = GpuContext::create_instance(&options)?;
    /// let gpu = GpuContext::with_instance_blocking(&instance, &options, None)?;
    /// assert_eq!(gpu.caps.instance_flags, instance.flags());
    /// # Ok::<(), cp_render::gpu::GpuError>(())
    /// ```
    ///
    /// A bare `wgpu::Instance` is not accepted, because nobody can tell which flags it has
    /// (mismatched types, expected `&GpuInstance`):
    ///
    /// ```compile_fail,E0308
    /// use cp_render::gpu::{GpuContext, GpuOptions};
    ///
    /// let options = GpuOptions::default();
    /// let instance = cp_render::wgpu::Instance::default();
    /// let gpu = GpuContext::with_instance_blocking(&instance, &options, None);
    /// ```
    ///
    /// When `surface` is given, it must have been created from `instance`, and only adapters
    /// that can present to it are considered.
    ///
    /// Adapters are tried in order of preference: hardware before software, Vulkan before
    /// DX12, then `opts.power`. If the preferred adapter fails to open a device, the next one is
    /// tried, so DX12 is also the fallback for a broken Vulkan driver.
    ///
    /// # Errors
    /// [`GpuError::NoAdapter`] when no adapter qualifies, [`GpuError::DeviceRequest`] with the
    /// reason of the preferred adapter when none opens a device, and the backend errors of
    /// [`Self::create_instance`].
    pub async fn with_instance(
        instance: &GpuInstance,
        opts: &GpuOptions,
        surface: Option<&wgpu::Surface<'_>>,
    ) -> Result<Self, GpuError> {
        let backends = backends_for(opts)?;
        let mut adapters = instance.enumerate_adapters(backends).await;
        if let Some(surface) = surface {
            adapters.retain(|adapter| adapter.is_surface_supported(surface));
        }
        let candidates: Vec<Candidate> = adapters
            .iter()
            .map(|adapter| {
                let info = adapter.get_info();
                Candidate {
                    backend: info.backend,
                    device_type: info.device_type,
                }
            })
            .collect();
        let order = rank_adapters(&candidates, opts.power);

        let opened = first_success(&order, async |index| {
            let opened = open_device(&adapters[index]).await;
            if let Err(error) = &opened {
                log::warn!("{error}; trying the next adapter, if there is one");
            }
            opened
        })
        .await;
        let Some(opened) = opened else {
            log::debug!("no GPU adapter on {}", backend_list(&backends));
            return Err(GpuError::NoAdapter {
                backends,
                surface_required: surface.is_some(),
            });
        };
        let OpenedDevice {
            adapter,
            device,
            queue,
            profile,
            plan,
        } = opened?;
        // Before anything else touches the device: from here on its errors and its loss arrive
        // in the log and in `health` instead of as panics.
        let health = DeviceHealth::watch(&device);

        // The record describes the device as it is, not as it was asked for, and the instance
        // as it was created, not as one created now would be.
        let caps = GpuCaps::new(
            profile,
            instance.flags(),
            device.features(),
            device.limits(),
        )
        .with_dx12_shader_compiler(instance.dx12_shader_compiler().clone());
        if caps.enabled_features != plan.features || caps.enabled_limits != plan.limits {
            log::warn!(
                "the GPU device differs from the negotiated request; the capability record \
                 follows the device"
            );
        }
        let optional = caps.optional_features();
        log::info!(
            "GPU device ready: {} on {}, {} tier, draw path {}, {} of {} optional features, \
             DX12 shader compiler {}",
            caps.adapter.name,
            caps.adapter.backend,
            caps.tier.id(),
            caps.derived.draw_path.description(),
            optional.iter().filter(|status| status.enabled).count(),
            optional.len(),
            caps.dx12_shader_compiler.id(),
        );

        Ok(Self {
            instance: wgpu::Instance::clone(instance),
            adapter,
            device,
            queue,
            caps,
            health,
        })
    }

    /// [`Self::new`] for callers without an async runtime; blocks the calling thread.
    ///
    /// # Errors
    /// The errors of [`Self::new`].
    pub fn new_blocking(
        opts: &GpuOptions,
        surface: Option<&wgpu::Surface<'_>>,
    ) -> Result<Self, GpuError> {
        futures_lite::future::block_on(Self::new(opts, surface))
    }

    /// [`Self::with_surface`] for callers without an async runtime; blocks the calling thread.
    ///
    /// # Errors
    /// The errors of [`Self::with_surface`].
    pub fn with_surface_blocking<'window>(
        opts: &GpuOptions,
        create_surface: impl FnOnce(
            &wgpu::Instance,
        ) -> Result<wgpu::Surface<'window>, wgpu::CreateSurfaceError>,
    ) -> Result<(Self, wgpu::Surface<'window>), GpuError> {
        futures_lite::future::block_on(Self::with_surface(opts, create_surface))
    }

    /// [`Self::with_instance`] for callers without an async runtime; blocks the calling thread.
    ///
    /// # Errors
    /// The errors of [`Self::with_instance`].
    pub fn with_instance_blocking(
        instance: &GpuInstance,
        opts: &GpuOptions,
        surface: Option<&wgpu::Surface<'_>>,
    ) -> Result<Self, GpuError> {
        futures_lite::future::block_on(Self::with_instance(instance, opts, surface))
    }

    /// `true` once wgpu has reported the device as lost: after a driver reset, crash or
    /// update, a removed GPU or memory pressure, and after `wgpu::Device::destroy` has
    /// completed.
    ///
    /// wgpu reports a loss when one of its calls fails in the driver, not when the loss
    /// happens; until then this is still `false`. On Vulkan the next poll or submit is such a
    /// call. On DX12 a removed device keeps answering polls and submits as if nothing had
    /// happened, and it is the next call that creates a resource or presents a frame that
    /// finds out.
    ///
    /// Once `true` it stays `true`: the device does not come back, and every further call on
    /// it or on its resources fails without effect.
    pub fn is_lost(&self) -> bool {
        self.health.is_lost()
    }

    /// `Ok` while the device works.
    ///
    /// # Errors
    /// [`GpuError::DeviceLost`] with wgpu's reason and message once [`Self::is_lost`].
    pub fn ensure_alive(&self) -> Result<(), GpuError> {
        self.health.check()
    }

    /// Polls the device like `wgpu::Device::poll`: runs the callbacks of finished work (buffer
    /// mappings, submitted-work-done), frees what the GPU no longer uses, and blocks when
    /// `poll_type` is a wait.
    ///
    /// The difference is the lost device. In wgpu 30 `Device::poll` panics when the driver
    /// reports the loss there, and on Vulkan it keeps doing so at every further call. This
    /// function instead
    ///
    /// 1. does not touch a device that is known to be lost,
    /// 2. looks for a loss first with `Instance::poll_all`, which does not panic, so the frame
    ///    loop learns about it without a panic being raised at all, and
    /// 3. contains wgpu's panic where it cannot be avoided: when the device is lost during a
    ///    blocking wait. The process's panic hook has run by then, so wgpu's message may be
    ///    on stderr although the caller gets the error.
    ///
    /// Step 2 polls every device of the context's instance, not only this one.
    ///
    /// An `Ok` means that wgpu knows of no loss, which is less than "the device works": see
    /// [`Self::is_lost`] for when wgpu finds out on each backend. After a loss no callbacks
    /// are run here any more; a buffer mapping that was still pending gets its callback, with
    /// an error, when the buffer is dropped.
    ///
    /// # Errors
    /// [`GpuError::DeviceLost`] once the device is lost, including the call that finds out;
    /// [`GpuError::Poll`] when a wait times out or names an unknown submission.
    ///
    /// # Panics
    /// Only with a panic of a callback that wgpu runs during the poll and that has nothing to
    /// do with a device loss.
    pub fn poll(&self, poll_type: wgpu::PollType) -> Result<wgpu::PollStatus, GpuError> {
        self.health.poll(
            || {
                // Whether other queues are empty is of no interest here.
                self.instance.poll_all(false);
            },
            || self.device.poll(poll_type),
        )
    }
}

/// A device together with the adapter it was opened on and the negotiation that led to it.
struct OpenedDevice {
    adapter: wgpu::Adapter,
    device: wgpu::Device,
    queue: wgpu::Queue,
    profile: AdapterProfile,
    plan: DevicePlan,
}

/// Negotiates features and limits with `adapter` and opens the device.
async fn open_device(adapter: &wgpu::Adapter) -> Result<OpenedDevice, GpuError> {
    let profile = AdapterProfile::from_adapter(adapter);
    let plan = profile.plan_device();
    log::debug!(
        "opening GPU adapter {} ({}, {:?})",
        profile.info.name,
        profile.info.backend,
        profile.info.device_type
    );
    let descriptor = wgpu::DeviceDescriptor {
        label: Some("cubeplanet"),
        required_features: plan.features,
        required_limits: plan.limits.clone(),
        ..Default::default()
    };
    match adapter.request_device(&descriptor).await {
        Ok((device, queue)) => Ok(OpenedDevice {
            adapter: adapter.clone(),
            device,
            queue,
            profile,
            plan,
        }),
        Err(reason) => Err(GpuError::DeviceRequest {
            adapter: profile.info.name,
            reason,
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::instance::{default_instance_flags, instance_descriptor};
    use crate::gpu::json::read::parse;
    use crate::gpu::{DrawPath, probe};

    /// Logs why a test (or one backend of it) is skipped.
    fn skip(test: &str, error: &GpuError) {
        crate::gpu::testing::skip(test, error);
    }

    /// The context of the default adapter, or `None` (with a log line) when the machine has
    /// no GPU adapter or the build has neither backend of the contract. Any other failure is a
    /// test failure.
    fn context_or_skip(test: &str) -> Option<GpuContext> {
        match GpuContext::new_blocking(&GpuOptions::default(), None) {
            Ok(gpu) => Some(gpu),
            Err(error @ (GpuError::NoAdapter { .. } | GpuError::BackendNotCompiled { .. })) => {
                skip(test, &error);
                None
            }
            Err(error) => panic!("{test}: GPU context creation failed: {error}"),
        }
    }

    /// One context per backend of the contract that has an adapter here, each on an instance
    /// that `adjust` changed before wgpu created it: the instances `GpuContext::create_instance`
    /// does not make. (With `CP_BACKEND` set, both are on the forced backend.)
    fn contexts_on_adjusted_instances(
        test: &str,
        adjust: impl Fn(&mut wgpu::InstanceDescriptor),
    ) -> Vec<GpuContext> {
        let mut contexts = Vec::new();
        for backends in [wgpu::Backends::VULKAN, wgpu::Backends::DX12] {
            let opts = GpuOptions {
                backends: Some(backends),
                ..GpuOptions::default()
            };
            let mut descriptor = match instance_descriptor(&opts) {
                Ok(descriptor) => descriptor,
                Err(error @ GpuError::BackendNotCompiled { .. }) => {
                    skip(test, &error);
                    continue;
                }
                Err(error) => panic!("{test}: no instance descriptor: {error}"),
            };
            adjust(&mut descriptor);
            let instance = GpuInstance::from_descriptor(descriptor);
            match GpuContext::with_instance_blocking(&instance, &opts, None) {
                Ok(gpu) => contexts.push(gpu),
                Err(error @ GpuError::NoAdapter { .. }) => skip(test, &error),
                Err(error) => panic!("{test}: GPU context creation failed: {error}"),
            }
        }
        contexts
    }

    #[test]
    fn real_device_matches_the_pure_negotiation() {
        let Some(gpu) = context_or_skip("real_device_matches_the_pure_negotiation") else {
            return;
        };
        let profile = AdapterProfile::from_adapter(&gpu.adapter);
        let plan = profile.plan_device();
        assert_eq!(gpu.device.features(), plan.features);
        assert_eq!(gpu.device.limits(), plan.limits);
        let compiler = crate::gpu::Dx12ShaderCompiler::from_env().unwrap();
        assert_eq!(
            gpu.caps,
            GpuCaps::negotiate(profile, default_instance_flags())
                .with_dx12_shader_compiler(compiler)
        );

        assert!(
            gpu.caps
                .available_features
                .contains(gpu.caps.enabled_features)
        );
        assert!(
            gpu.caps
                .enabled_limits
                .check_limits(&gpu.caps.adapter_limits)
        );
        assert_eq!(gpu.caps.adapter, gpu.adapter.get_info());
    }

    #[test]
    fn real_adapter_report_is_valid_json() {
        let Some(gpu) = context_or_skip("real_adapter_report_is_valid_json") else {
            return;
        };
        let text = gpu.caps.report_json();
        let json = parse(&text).unwrap_or_else(|error| panic!("invalid JSON: {error}\n{text}"));
        let info = &gpu.caps.adapter;
        assert_eq!(
            json.at(&["adapter", "name"]).as_str(),
            Some(info.name.as_str())
        );
        assert_eq!(
            json.at(&["adapter", "driver"]).as_str(),
            Some(info.driver.as_str())
        );
        assert_eq!(
            json.at(&["adapter", "driver_info"]).as_str(),
            Some(info.driver_info.as_str())
        );
        assert_eq!(
            json.at(&["adapter", "backend"]).as_str(),
            Some(info.backend.to_str())
        );
        assert_eq!(json.at(&["tier"]).as_str(), Some(gpu.caps.tier.id()));
        assert_eq!(
            json.at(&["limits", "max_buffer_size", "enabled"]).as_u64(),
            Some(gpu.device.limits().max_buffer_size)
        );
    }

    /// The capability record must describe the instance the device lives on. An instance whose
    /// indirect-call validation is the other way round than `create_instance` would set it
    /// stands for every instance with flags of its own; on DX12 that flag decides whether
    /// indirect draws deliver `first_vertex` and `first_instance`.
    #[test]
    fn the_record_follows_the_flags_of_the_instance() {
        let test = "the_record_follows_the_flags_of_the_instance";
        let indirect_validation = wgpu::InstanceFlags::VALIDATION_INDIRECT_CALL;
        let flags = default_instance_flags() ^ indirect_validation;
        let contexts = contexts_on_adjusted_instances(test, |descriptor| descriptor.flags = flags);
        for gpu in contexts {
            let caps = &gpu.caps;
            let backend = caps.adapter.backend;
            assert_eq!(caps.instance_flags, flags, "{backend}");
            let profile = AdapterProfile::from_adapter(&gpu.adapter);
            let negotiated = GpuCaps::negotiate(profile, flags)
                .with_dx12_shader_compiler(caps.dx12_shader_compiler.clone());
            assert_eq!(*caps, negotiated, "{backend}");

            if backend == wgpu::Backend::Dx12 && !flags.contains(indirect_validation) {
                // Without the validation DX12 starts both builtins at zero; the bucket
                // attribute of path B does not depend on them.
                assert!(
                    !caps
                        .derived
                        .indirect_draws_respect_first_vertex_and_instance
                );
                assert_eq!(caps.derived.draw_path, DrawPath::B);
            }
            // Whatever the record promises with these flags is what the GPU does.
            probe::check_indirect_draw_promises(test, &gpu);
        }
    }

    /// A device loss that no test may cause for real is reported by wgpu-core through
    /// `Device::lose`: the device becomes invalid and the device-lost callback runs with the
    /// reason `Unknown`. The same function runs when the device exceeds the memory budget
    /// threshold of its instance, which is checked after every poll. A threshold of 0 % is
    /// therefore exceeded at once: a real loss as far as wgpu is concerned, on a GPU that is
    /// left alone.
    ///
    /// The poll itself succeeds in this case, as it does on DX12 after a removed device. A
    /// renderer could not find out that it is dead; the context has to tell it.
    #[test]
    fn a_device_that_wgpu_gives_up_is_reported_by_poll() {
        let test = "a_device_that_wgpu_gives_up_is_reported_by_poll";
        let contexts = contexts_on_adjusted_instances(test, |descriptor| {
            descriptor.memory_budget_thresholds.for_device_loss = Some(0);
        });
        for gpu in contexts {
            let backend = gpu.caps.adapter.backend;
            let clone = gpu.clone();
            assert!(!gpu.is_lost(), "{backend}: lost before anything happened");
            gpu.ensure_alive().unwrap();

            match gpu.poll(wgpu::PollType::Poll) {
                Err(GpuError::DeviceLost { reason, message }) => {
                    assert_eq!(reason, wgpu::DeviceLostReason::Unknown, "{backend}");
                    assert!(!message.is_empty(), "{backend}: wgpu names the cause");
                }
                Ok(status) if !gpu.is_lost() => {
                    // A driver without a memory budget to compare with: nothing to test.
                    let reason = format!("{backend}: no loss was provoked ({status:?})");
                    crate::gpu::testing::skip(test, &reason);
                    continue;
                }
                other => panic!("{backend}: expected the device loss, got {other:?}"),
            }

            // Everything that shares the device knows, and keeps saying so.
            assert!(gpu.is_lost() && clone.is_lost(), "{backend}");
            for context in [&gpu, &clone] {
                let alive = context.ensure_alive();
                assert!(
                    matches!(
                        alive,
                        Err(GpuError::DeviceLost {
                            reason: wgpu::DeviceLostReason::Unknown,
                            ..
                        })
                    ),
                    "{backend}: {alive:?}"
                );
                for poll_type in [wgpu::PollType::Poll, wgpu::PollType::wait_indefinitely()] {
                    let poll = context.poll(poll_type);
                    assert!(
                        matches!(poll, Err(GpuError::DeviceLost { .. })),
                        "{backend}: {poll:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn a_working_device_is_polled_like_wgpu_polls_it() {
        let Some(gpu) = context_or_skip("a_working_device_is_polled_like_wgpu_polls_it") else {
            return;
        };
        // Nothing was submitted: the queue is empty, with and without waiting.
        for poll_type in [wgpu::PollType::Poll, wgpu::PollType::wait_indefinitely()] {
            assert_eq!(gpu.poll(poll_type).unwrap(), wgpu::PollStatus::QueueEmpty);
        }

        // A mapping callback runs inside the poll that waits for it.
        let buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cp-test-poll-mapping"),
            size: 16,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let (sender, receiver) = std::sync::mpsc::channel();
        buffer
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |result| {
                // The receiver outlives the poll below.
                let _ = sender.send(result.is_ok());
            });
        let status = gpu.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        assert!(status.wait_finished(), "{status:?}");
        assert_eq!(receiver.try_recv(), Ok(true));
        buffer.unmap();

        // A submission this device has never made cannot be waited for; that is an error of
        // the poll, not a loss.
        let other = match GpuContext::new_blocking(&GpuOptions::default(), None) {
            Ok(other) => other,
            Err(error) => panic!("a second context on the same adapter failed: {error}"),
        };
        let foreign_index = other.queue.submit([]);
        let wrong = gpu.poll(wgpu::PollType::Wait {
            submission_index: Some(foreign_index),
            timeout: None,
        });
        assert!(
            matches!(
                wrong,
                Err(GpuError::Poll {
                    reason: wgpu::PollError::WrongSubmissionIndex(..)
                })
            ),
            "{wrong:?}"
        );
        assert!(!gpu.is_lost());
        gpu.ensure_alive().unwrap();
    }

    /// The poll takes two steps through wgpu, one over the instance and one over the device.
    /// Clones on several threads must be able to take them at the same time, next to a thread
    /// that submits work.
    #[test]
    fn clones_poll_the_same_device_from_several_threads() {
        let test = "clones_poll_the_same_device_from_several_threads";
        let Some(gpu) = context_or_skip(test) else {
            return;
        };
        let buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cp-test-concurrent-polls"),
            size: 64,
            usage: wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        std::thread::scope(|scope| {
            for _ in 0..4 {
                let gpu = gpu.clone();
                scope.spawn(move || {
                    for _ in 0..200 {
                        let poll = gpu.poll(wgpu::PollType::Poll);
                        assert!(poll.is_ok(), "{poll:?}");
                    }
                });
            }
            for round in 0..50_u8 {
                gpu.queue.write_buffer(&buffer, 0, &[round; 64]);
                gpu.queue.submit([]);
                let poll = gpu.poll(wgpu::PollType::wait_indefinitely());
                assert!(poll.is_ok(), "{poll:?}");
            }
        });
        let status = gpu.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        assert!(status.wait_finished(), "{status:?}");
        assert!(!gpu.is_lost());
    }

    #[test]
    fn context_and_errors_can_cross_threads() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<GpuContext>();
        assert_send_sync::<GpuInstance>();
        assert_send_sync::<GpuCaps>();
        assert_send_sync::<GpuOptions>();
        assert_send_sync::<GpuError>();
    }
}
