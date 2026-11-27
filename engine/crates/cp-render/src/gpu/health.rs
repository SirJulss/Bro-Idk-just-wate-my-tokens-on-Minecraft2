//! What a context knows about its device after creation: whether it still works.
//!
//! wgpu 30 tells a lost device to exactly one callback, exactly once, and treats a device error
//! inside `Device::poll` as fatal: it panics. Neither fits contract §2.6, where a GPU loss is
//! external input and never a panic. This module turns the callback into state that can be
//! asked at any time, and the panic into an error value.
//!
//! The facts about wgpu it relies on were read from the sources of wgpu 30.0.1 (the version is
//! pinned) and have to be checked again with every wgpu update:
//!
//! - `Device::poll` ends in `ContextWgpuCore::handle_error_fatal`, a panic, for every device
//!   error of the poll (`wgpu/src/backend/wgpu_core.rs`, `CoreDevice::poll`). On Vulkan a lost
//!   device fails the fence query of every later poll as well.
//! - Before that, wgpu-core gives the device up and calls the device-lost callback
//!   (`Device::handle_hal_error` and `Device::lose` in `wgpu-core/src/device/resource.rs`).
//! - `Instance::poll_all` maintains the devices the same way but drops their errors
//!   (`Global::poll_all_devices_of_api` in `wgpu-core/src/device/global.rs`).

use std::any::Any;
use std::panic::{AssertUnwindSafe, catch_unwind, resume_unwind};
use std::sync::{Arc, OnceLock};

use super::GpuError;

/// Log target of everything this module reports: the documented target of a context's device
/// messages (see `GpuContext`), kept the same for log filters although the code lives here.
const LOG_TARGET: &str = "cp_render::gpu::context";

/// How wgpu 30 begins the panic message of its fatal error path for the two kinds of poll:
/// `panic!("Error in {operation}: ...")` with these operation names.
const WGPU_FATAL_POLL_ERRORS: [&str; 2] = [
    "Error in Device::poll",
    "Error in Instance::poll_all_devices",
];

/// That and why a device stopped working.
#[derive(Clone, Debug, PartialEq, Eq)]
struct DeviceLoss {
    reason: wgpu::DeviceLostReason,
    message: String,
}

impl DeviceLoss {
    fn to_error(&self) -> GpuError {
        GpuError::DeviceLost {
            reason: self.reason,
            message: self.message.clone(),
        }
    }
}

/// Whether a device still works.
///
/// Shared by all clones of a context and by the callbacks installed on its device. It never
/// holds the device, so it keeps nothing alive.
#[derive(Debug, Default)]
pub(crate) struct DeviceHealth {
    /// Set once, by the first report of a loss; a lost device does not recover.
    loss: OnceLock<DeviceLoss>,
}

impl DeviceHealth {
    /// Installs the device-lost callback and the handler for uncaptured errors on `device` and
    /// returns the state they feed.
    ///
    /// Both replace wgpu's defaults for the lifetime of the device: without them a lost device
    /// goes unnoticed and an error outside an error scope is a panic.
    pub(crate) fn watch(device: &wgpu::Device) -> Arc<Self> {
        let health = Arc::new(Self::default());
        let on_loss = Arc::clone(&health);
        device.set_device_lost_callback(move |reason, message| {
            on_loss.record_loss(reason, message);
        });
        let on_error = Arc::clone(&health);
        device.on_uncaptured_error(Arc::new(move |error: wgpu::Error| {
            on_error.report_uncaptured(&error);
        }));
        health
    }

    /// `true` once a loss is on record.
    pub(crate) fn is_lost(&self) -> bool {
        self.loss.get().is_some()
    }

    /// `Ok` while the device works, [`GpuError::DeviceLost`] from the first report of a loss on.
    pub(crate) fn check(&self) -> Result<(), GpuError> {
        match self.loss.get() {
            Some(loss) => Err(loss.to_error()),
            None => Ok(()),
        }
    }

    /// Puts the loss on record and into the log. The first report stays; `false` when there
    /// was one already.
    ///
    /// Runs inside wgpu, with wgpu's locks held, when it is the device-lost callback: it must
    /// not call back into wgpu.
    fn record_loss(&self, reason: wgpu::DeviceLostReason, message: String) -> bool {
        let detail = detail(&message);
        // State first, log second: a poll on another thread that meets wgpu's panic about this
        // very loss looks for it on record.
        if self.loss.set(DeviceLoss { reason, message }).is_err() {
            return false;
        }
        match reason {
            // Requested by the application through `Device::destroy`: expected, not an error.
            wgpu::DeviceLostReason::Destroyed => {
                log::info!(target: LOG_TARGET, "GPU device destroyed{detail}");
            }
            wgpu::DeviceLostReason::Unknown => {
                log::error!(target: LOG_TARGET, "GPU device lost{detail}");
            }
        }
        true
    }

    /// Writes an error that no error scope captured to the log.
    ///
    /// A validation error caused by a driver or by content is external input (contract §2.6)
    /// and must not take the process down, which is what wgpu's default handler does.
    fn report_uncaptured(&self, error: &wgpu::Error) {
        let kind = match error {
            wgpu::Error::OutOfMemory { .. } => "out of memory",
            wgpu::Error::Validation { .. } => "validation",
            wgpu::Error::Internal { .. } => "internal",
        };
        if self.is_lost() {
            // On a lost device every call fails, frame after frame. The loss is in the log
            // already; its consequences would only bury it.
            log::debug!(target: LOG_TARGET, "GPU error ({kind}) on the lost device: {error}");
        } else {
            log::error!(target: LOG_TARGET, "uncaptured GPU error ({kind}): {error}");
        }
    }

    /// Polls a device in such a way that its loss is an error value and not a panic.
    ///
    /// `discover` is a poll that cannot fail (`Instance::poll_all`), `poll` the
    /// `Device::poll` that was asked for. Neither is called once a loss is on record.
    ///
    /// # Errors
    /// [`GpuError::DeviceLost`] as soon as the loss is known, including when this call finds
    /// it; [`GpuError::Poll`] for an error of `poll` on a device that works.
    ///
    /// # Panics
    /// With the panic of `discover` or `poll` when it has nothing to do with a device loss
    /// (see [`Self::contain`]).
    pub(crate) fn poll(
        &self,
        discover: impl FnOnce(),
        poll: impl FnOnce() -> Result<wgpu::PollStatus, wgpu::PollError>,
    ) -> Result<wgpu::PollStatus, GpuError> {
        // A device known to be lost is not touched again: on Vulkan every further
        // `Device::poll` would run into the failing fence query and from there into the panic.
        self.check()?;
        // A loss since the last poll is found here, where wgpu does not panic: the callback
        // puts it on record, and `Device::poll` is not called at all.
        self.contain(discover)?;
        self.check()?;
        // What remains for `Device::poll` to find is a loss during a blocking wait, or in the
        // moment between the two calls.
        let outcome = self.contain(poll)?;
        // Some losses come with a poll that succeeds: a `Device::destroy` that has completed,
        // or a device wgpu gives up for memory pressure.
        self.check()?;
        outcome.map_err(|reason| GpuError::Poll { reason })
    }

    /// Runs one wgpu poll and turns a panic that belongs to a device loss into the error.
    ///
    /// A panic belongs to a device loss when it is wgpu's fatal error of a poll, which wgpu
    /// only raises after it has given the device up, or when the loss was reported while the
    /// poll ran: then the panic is a consequence of it, for example a mapping callback that
    /// cannot cope with a failed mapping. Any other panic continues to unwind unchanged.
    ///
    /// wgpu raises its panic after its work is done and its locks are released, so unwinding
    /// out of it leaves wgpu consistent. The panic hook has already run at that point; with
    /// the default hook, wgpu's message is on stderr even though the caller gets an error.
    /// Containing needs panics that unwind, which is what the workspace profiles have (the
    /// panic containment the contract asks of `cp-jobs` needs it too); under `panic = "abort"`
    /// a loss that `Device::poll` finds still ends the process.
    fn contain<T>(&self, wgpu_poll: impl FnOnce() -> T) -> Result<T, GpuError> {
        let panic = match catch_unwind(AssertUnwindSafe(wgpu_poll)) {
            Ok(value) => return Ok(value),
            Err(panic) => panic,
        };
        let message = one_line(panic_message(panic.as_ref()));
        if is_fatal_poll_error_of_wgpu(&message) {
            // Normally a no-op, because wgpu told the callback first. It matters when the
            // callback is still running on another thread, or was replaced by somebody.
            self.record_loss(wgpu::DeviceLostReason::Unknown, message);
        } else if self.is_lost() {
            log::error!(
                target: LOG_TARGET,
                "a panic during the poll that found the GPU device lost was contained: {message}"
            );
        }
        match self.loss.get() {
            Some(loss) => Err(loss.to_error()),
            None => resume_unwind(panic),
        }
    }
}

/// `true` for the panic message of wgpu's fatal error path of a poll.
fn is_fatal_poll_error_of_wgpu(message: &str) -> bool {
    WGPU_FATAL_POLL_ERRORS
        .iter()
        .any(|prefix| message.starts_with(prefix))
}

/// The message of a panic payload as `panic!` creates it; a placeholder for any other payload.
fn panic_message(payload: &(dyn Any + Send)) -> &str {
    if let Some(message) = payload.downcast_ref::<String>() {
        message
    } else if let Some(message) = payload.downcast_ref::<&'static str>() {
        message
    } else {
        "a panic without a message"
    }
}

/// `text` on a single line: wgpu's messages are indented trees over several lines, which
/// neither a log line nor an error message should contain.
fn one_line(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn detail(message: &str) -> String {
    if message.is_empty() {
        String::new()
    } else {
        format!(": {message}")
    }
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;

    use super::*;
    use wgpu::{DeviceLostReason, PollError, PollStatus};

    /// What wgpu 30.0.1 panics with when the Vulkan driver answers `VK_ERROR_DEVICE_LOST` to
    /// the fence query of a poll (observed with a simulated driver answer).
    const WGPU_LOST_PANIC: &str =
        "Error in Device::poll: Validation Error\n\nCaused by:\n  Parent device is lost\n";

    fn lost_health(reason: DeviceLostReason, message: &str) -> DeviceHealth {
        let health = DeviceHealth::default();
        assert!(health.record_loss(reason, message.to_owned()));
        health
    }

    fn assert_lost(outcome: Result<PollStatus, GpuError>, reason: DeviceLostReason, text: &str) {
        match outcome {
            Err(GpuError::DeviceLost {
                reason: reported,
                message,
            }) => {
                assert_eq!(reported, reason);
                assert_eq!(message, text);
            }
            other => panic!("expected the device loss, got {other:?}"),
        }
    }

    #[test]
    fn a_new_device_is_healthy() {
        let health = DeviceHealth::default();
        assert!(!health.is_lost());
        assert!(health.check().is_ok());
    }

    #[test]
    fn a_loss_is_on_record_with_reason_and_message() {
        let health = lost_health(DeviceLostReason::Unknown, "Device is lost");
        assert!(health.is_lost());
        match health.check() {
            Err(GpuError::DeviceLost { reason, message }) => {
                assert_eq!(reason, DeviceLostReason::Unknown);
                assert_eq!(message, "Device is lost");
            }
            other => panic!("expected the device loss, got {other:?}"),
        }
    }

    #[test]
    fn the_first_report_of_a_loss_stays() {
        let health = lost_health(DeviceLostReason::Destroyed, "");
        assert!(!health.record_loss(DeviceLostReason::Unknown, "later".to_owned()));
        assert_lost(
            health.check().map(|()| PollStatus::Poll),
            DeviceLostReason::Destroyed,
            "",
        );
    }

    #[test]
    fn a_working_device_is_polled_and_its_status_passed_on() {
        for status in [
            PollStatus::QueueEmpty,
            PollStatus::WaitSucceeded,
            PollStatus::Poll,
        ] {
            let health = DeviceHealth::default();
            let step = Cell::new(0);
            let outcome = health.poll(
                || {
                    assert_eq!(step.replace(1), 0, "discovery comes first, once");
                },
                || {
                    assert_eq!(step.replace(2), 1, "the poll comes second, once");
                    Ok(match status {
                        PollStatus::QueueEmpty => PollStatus::QueueEmpty,
                        PollStatus::WaitSucceeded => PollStatus::WaitSucceeded,
                        PollStatus::Poll => PollStatus::Poll,
                    })
                },
            );
            assert_eq!(step.get(), 2);
            assert_eq!(outcome.unwrap(), status);
            assert!(!health.is_lost());
        }
    }

    #[test]
    fn a_failed_wait_on_a_working_device_keeps_the_reason_of_wgpu() {
        let health = DeviceHealth::default();
        let outcome = health.poll(|| {}, || Err(PollError::Timeout));
        assert!(
            matches!(
                outcome,
                Err(GpuError::Poll {
                    reason: PollError::Timeout
                })
            ),
            "{outcome:?}"
        );

        let outcome = health.poll(|| {}, || Err(PollError::WrongSubmissionIndex(9, 4)));
        assert!(
            matches!(
                outcome,
                Err(GpuError::Poll {
                    reason: PollError::WrongSubmissionIndex(9, 4)
                })
            ),
            "{outcome:?}"
        );
        assert!(!health.is_lost());
    }

    #[test]
    fn a_device_known_to_be_lost_is_not_touched_again() {
        for reason in [DeviceLostReason::Unknown, DeviceLostReason::Destroyed] {
            let health = lost_health(reason, "gone");
            let outcome = health.poll(
                || panic!("a lost device must not be looked at"),
                || panic!("a lost device must not be polled"),
            );
            assert_lost(outcome, reason, "gone");
        }
    }

    #[test]
    fn a_loss_found_by_the_discovery_keeps_the_device_from_being_polled() {
        let health = DeviceHealth::default();
        let outcome = health.poll(
            || {
                // What `Instance::poll_all` does on a lost Vulkan device: the device-lost
                // callback runs, the error is dropped.
                health.record_loss(DeviceLostReason::Unknown, "Device is lost".to_owned());
            },
            || panic!("wgpu would panic here: {WGPU_LOST_PANIC}"),
        );
        assert_lost(outcome, DeviceLostReason::Unknown, "Device is lost");
    }

    /// The sequence the Vulkan backend of wgpu 30 produces when a device is lost while it is
    /// polled: first the device-lost callback, then the panic. Without the containment the
    /// panic reaches the frame loop.
    #[test]
    fn the_panic_of_wgpu_about_a_lost_device_becomes_the_error() {
        let health = DeviceHealth::default();
        let outcome = health.poll(
            || {},
            || {
                health.record_loss(DeviceLostReason::Unknown, "Device is lost".to_owned());
                panic!("{WGPU_LOST_PANIC}");
            },
        );
        assert_lost(outcome, DeviceLostReason::Unknown, "Device is lost");

        // From here on the device is left alone.
        let again = health.poll(
            || panic!("looked at"),
            || panic!("polled: {WGPU_LOST_PANIC}"),
        );
        assert_lost(again, DeviceLostReason::Unknown, "Device is lost");
    }

    #[test]
    fn the_panic_of_wgpu_is_recognised_without_the_callback() {
        // A callback that somebody replaced, or one still running on another thread.
        let health = DeviceHealth::default();
        let outcome = health.poll(|| {}, || panic!("{WGPU_LOST_PANIC}"));
        assert_lost(
            outcome,
            DeviceLostReason::Unknown,
            "Error in Device::poll: Validation Error Caused by: Parent device is lost",
        );
        assert!(health.is_lost());

        // The same for the discovery, should a later wgpu make its errors fatal as well.
        let health = DeviceHealth::default();
        let outcome = health.poll(
            || panic!("Error in Instance::poll_all_devices: Parent device is lost"),
            || panic!("a lost device must not be polled"),
        );
        assert_lost(
            outcome,
            DeviceLostReason::Unknown,
            "Error in Instance::poll_all_devices: Parent device is lost",
        );
    }

    #[test]
    fn a_loss_reported_by_a_poll_that_succeeds_is_the_error() {
        // A completed `Device::destroy`, or a device given up for memory pressure.
        let health = DeviceHealth::default();
        let outcome = health.poll(
            || {},
            || {
                health.record_loss(DeviceLostReason::Destroyed, String::new());
                Ok(PollStatus::QueueEmpty)
            },
        );
        assert_lost(outcome, DeviceLostReason::Destroyed, "");

        // The loss also wins over an error of the poll itself.
        let health = DeviceHealth::default();
        let outcome = health.poll(
            || {},
            || {
                health.record_loss(DeviceLostReason::Unknown, "Out of memory".to_owned());
                Err(PollError::Timeout)
            },
        );
        assert_lost(outcome, DeviceLostReason::Unknown, "Out of memory");
    }

    #[test]
    fn a_panic_caused_by_the_loss_is_contained_with_it() {
        // A mapping callback that expects success, called by the poll that finds the loss.
        let health = DeviceHealth::default();
        let outcome = health.poll(
            || {},
            || {
                health.record_loss(DeviceLostReason::Unknown, "Device is lost".to_owned());
                panic!("map the readback buffer: BufferAsyncError");
            },
        );
        assert_lost(outcome, DeviceLostReason::Unknown, "Device is lost");
    }

    #[test]
    fn a_panic_that_is_not_about_a_loss_keeps_unwinding() {
        let health = DeviceHealth::default();
        for in_discovery in [true, false] {
            let unwound = catch_unwind(AssertUnwindSafe(|| {
                health.poll(
                    || {
                        if in_discovery {
                            panic!("a bug in a callback");
                        }
                    },
                    || panic!("a bug in a callback"),
                )
            }));
            let payload = unwound.expect_err("the panic must not be swallowed");
            assert_eq!(panic_message(payload.as_ref()), "a bug in a callback");
            assert!(!health.is_lost());
        }
    }

    #[test]
    fn only_the_poll_errors_of_wgpu_count_as_its_fatal_path() {
        assert!(is_fatal_poll_error_of_wgpu(&one_line(WGPU_LOST_PANIC)));
        assert!(is_fatal_poll_error_of_wgpu(
            "Error in Instance::poll_all_devices: Not enough memory left."
        ));
        for other in [
            "",
            "Error in RenderBundleEncoder::finish: nope",
            "Error in Surface::get_current_texture_view: nope",
            "wgpu error: Validation Error",
            "a bug in a callback mentioning Error in Device::poll",
        ] {
            assert!(!is_fatal_poll_error_of_wgpu(other), "{other:?}");
        }
    }

    #[test]
    fn panic_messages_are_read_from_both_payload_kinds() {
        let formatted = catch_unwind(|| panic!("value {}", 7)).unwrap_err();
        assert_eq!(panic_message(formatted.as_ref()), "value 7");
        let literal = catch_unwind(|| panic!("literal")).unwrap_err();
        assert_eq!(panic_message(literal.as_ref()), "literal");
        let other = catch_unwind(|| std::panic::panic_any(7_u32)).unwrap_err();
        assert_eq!(panic_message(other.as_ref()), "a panic without a message");
    }

    #[test]
    fn messages_are_folded_into_one_line() {
        assert_eq!(
            one_line(WGPU_LOST_PANIC),
            "Error in Device::poll: Validation Error Caused by: Parent device is lost"
        );
        assert_eq!(one_line("  plain  "), "plain");
        assert_eq!(one_line(""), "");
    }

    #[test]
    fn detail_is_empty_for_an_empty_message() {
        assert_eq!(detail(""), "");
        assert_eq!(detail("driver crashed"), ": driver crashed");
    }
}
