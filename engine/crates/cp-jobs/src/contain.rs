//! Keeps panics of foreign code away from the threads of the job system.
//!
//! Foreign code is everything the crate calls but does not own: job closures, the destructors
//! of the values they capture and return, panic payloads, and the installed logger.

use std::any::Any;
use std::fmt;
use std::mem;
use std::panic::{AssertUnwindSafe, catch_unwind};

use log::Level;

/// What a panic carries while it unwinds.
pub(crate) type Payload = Box<dyn Any + Send + 'static>;

/// Message reported for a panic whose payload is neither a `&str` nor a `String`.
pub(crate) const OPAQUE_PANIC: &str = "<panic payload is not a string>";

/// Target of every log record of this crate.
const LOG_TARGET: &str = "cp_jobs";

/// The message of a panic, as far as its payload is text.
pub(crate) fn panic_message(payload: &(dyn Any + Send)) -> String {
    if let Some(text) = payload.downcast_ref::<&'static str>() {
        (*text).to_owned()
    } else if let Some(text) = payload.downcast_ref::<String>() {
        text.clone()
    } else {
        OPAQUE_PANIC.to_owned()
    }
}

/// Writes a log record; all logging of this crate goes through here.
///
/// A logger may panic, for example when it cannot write its output. On a worker thread that
/// would end the worker and leave a job handle unresolved; in a destructor it could abort the
/// process. Such a panic is therefore swallowed: there is nowhere left to report it.
pub(crate) fn log_contained(level: Level, message: fmt::Arguments<'_>) {
    let logging = catch_unwind(AssertUnwindSafe(|| {
        log::log!(target: LOG_TARGET, level, "{message}");
    }));
    if let Err(payload) = logging {
        // Leaked on purpose: the destructor of the payload could panic yet again.
        mem::forget(payload);
    }
}

/// Drops `value` without letting a panic of its destructor escape; such a panic is logged.
///
/// `what` names the value in the log message.
pub(crate) fn drop_contained<T>(value: T, what: &str) {
    if let Err(payload) = catch_unwind(AssertUnwindSafe(move || drop(value))) {
        log_contained(
            Level::Error,
            format_args!(
                "{what} panicked while being dropped: {}",
                panic_message(&*payload)
            ),
        );
        // Leaked on purpose: the destructor of this second payload could panic yet again.
        mem::forget(payload);
    }
}

#[cfg(test)]
mod tests {
    use std::panic::panic_any;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::thread;

    use super::*;

    /// Sets a flag and panics when dropped.
    struct PanicsOnDrop(Arc<AtomicBool>);

    impl Drop for PanicsOnDrop {
        fn drop(&mut self) {
            self.0.store(true, Ordering::SeqCst);
            // A second panic during unwinding would abort the whole test process.
            if !thread::panicking() {
                panic!("destructor failed");
            }
        }
    }

    fn payload_of(panicking: impl FnOnce() + Send + 'static) -> Payload {
        catch_unwind(AssertUnwindSafe(panicking)).expect_err("the closure panics")
    }

    #[test]
    fn message_of_a_literal_panic() {
        let payload = payload_of(|| panic!("plain literal"));
        assert_eq!(panic_message(&*payload), "plain literal");
    }

    #[test]
    fn message_of_a_formatted_panic() {
        let payload = payload_of(|| panic!("value {} is out of range", 41 + 1));
        assert_eq!(panic_message(&*payload), "value 42 is out of range");
    }

    #[test]
    fn message_of_a_payload_that_is_not_text() {
        let payload = payload_of(|| panic_any(17_u32));
        assert_eq!(panic_message(&*payload), OPAQUE_PANIC);
    }

    #[test]
    fn drop_contained_runs_the_destructor() {
        let dropped = Arc::new(());
        drop_contained(Arc::clone(&dropped), "test value");
        assert_eq!(Arc::strong_count(&dropped), 1);
    }

    #[test]
    fn drop_contained_swallows_a_panicking_destructor() {
        let dropped = Arc::new(AtomicBool::new(false));
        drop_contained(PanicsOnDrop(Arc::clone(&dropped)), "test value");
        assert!(dropped.load(Ordering::SeqCst));
    }

    #[test]
    fn drop_contained_swallows_a_panic_payload_with_a_panicking_destructor() {
        let dropped = Arc::new(AtomicBool::new(false));
        let bomb = PanicsOnDrop(Arc::clone(&dropped));
        let payload = payload_of(move || panic_any(bomb));
        assert!(!dropped.load(Ordering::SeqCst));
        drop_contained(payload, "test payload");
        assert!(dropped.load(Ordering::SeqCst));
    }
}
