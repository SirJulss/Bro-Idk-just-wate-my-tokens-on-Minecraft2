//! A panicking job is contained, logged and reported, and its worker keeps serving.
//!
//! This module owns the logger of the test process, so the one test about a log record that
//! has nothing to do with panics lives here as well.

use std::panic::panic_any;
use std::sync::{Arc, Once};
use std::thread;

use cp_jobs::{JobError, JobSystem};
use crossbeam_channel::bounded;
use log::{Level, LevelFilter, Log, Metadata, Record};
use parking_lot::Mutex;

use crate::util::{Explosive, Gate, NORMAL, quiesce, recv, with_deadline};

/// Collects every log record of the test process.
struct CapturingLogger {
    records: Mutex<Vec<(Level, String)>>,
}

static LOGGER: CapturingLogger = CapturingLogger {
    records: Mutex::new(Vec::new()),
};

/// A record that contains this text makes the test logger panic, the way a real logger does
/// when it cannot write its output.
const LOGGER_TRAP: &str = "logger trap 7d40";

impl Log for CapturingLogger {
    fn enabled(&self, _: &Metadata<'_>) -> bool {
        true
    }

    fn log(&self, record: &Record<'_>) {
        let text = record.args().to_string();
        assert!(
            !text.contains(LOGGER_TRAP),
            "the test logger fails on purpose"
        );
        self.records.lock().push((record.level(), text));
    }

    fn flush(&self) {}
}

impl CapturingLogger {
    /// Whether an error record that contains all of `fragments` has been logged.
    fn has_error_with(&self, fragments: &[&str]) -> bool {
        self.has_record_with(Level::Error, fragments)
    }

    /// Whether a record of `wanted` level that contains all of `fragments` has been logged.
    fn has_record_with(&self, wanted: Level, fragments: &[&str]) -> bool {
        self.records.lock().iter().any(|(level, text)| {
            *level == wanted && fragments.iter().all(|fragment| text.contains(fragment))
        })
    }
}

/// Installs the capturing logger for the whole test process, once.
fn capture_logs() -> &'static CapturingLogger {
    static INSTALL: Once = Once::new();
    INSTALL.call_once(|| {
        log::set_logger(&LOGGER).expect("this test binary installs no other logger");
        log::set_max_level(LevelFilter::Trace);
    });
    &LOGGER
}

#[test]
fn panicking_job_is_reported_and_its_worker_keeps_serving() {
    with_deadline(|| {
        let system = JobSystem::new(1, "panic");
        let worker_before = system
            .spawn(NORMAL, |_| thread::current().id())
            .wait()
            .unwrap();

        let failed = system.spawn(NORMAL, |_| -> u32 { panic!("boom {}", 7) });
        assert_eq!(failed.wait(), Err(JobError::Panicked("boom 7".to_owned())));

        // The very same thread serves the next job.
        let worker_after = system
            .spawn(NORMAL, |_| thread::current().id())
            .wait()
            .unwrap();
        assert_eq!(worker_before, worker_after);
        let stats = system.stats();
        assert_eq!(
            (stats.submitted, stats.completed, stats.panicked),
            (3, 2, 1)
        );
    });
}

#[test]
fn panic_messages_of_every_payload_kind_are_reported() {
    with_deadline(|| {
        let system = JobSystem::new(2, "panic");
        let literal = system.spawn(NORMAL, |_| -> u32 { panic!("a literal message") });
        let formatted = system.spawn(NORMAL, |_| -> u32 { panic!("chunk {:?} failed", (1, -2)) });
        let opaque = system.spawn(NORMAL, |_| -> u32 { panic_any(17_u64) });

        assert_eq!(
            literal.wait(),
            Err(JobError::Panicked("a literal message".to_owned()))
        );
        assert_eq!(
            formatted.wait(),
            Err(JobError::Panicked("chunk (1, -2) failed".to_owned()))
        );
        match opaque.wait() {
            Err(JobError::Panicked(message)) => assert!(!message.is_empty()),
            other => panic!("expected a panic report, got {other:?}"),
        }
    });
}

#[test]
fn panic_is_logged_with_its_message_and_the_worker_name() {
    with_deadline(|| {
        let logger = capture_logs();
        let system = JobSystem::new(1, "logged");
        let handle = system.spawn(NORMAL, |_| -> u32 { panic!("marker 5c1f of the log test") });

        assert!(handle.wait().is_err());

        // The record is written before the handle reports the job as finished.
        assert!(logger.has_error_with(&["marker 5c1f of the log test", "logged-0"]));
    });
}

#[test]
fn panic_in_a_detached_job_is_contained_and_logged() {
    with_deadline(|| {
        let logger = capture_logs();
        let system = JobSystem::new(1, "detached");
        system.spawn_detached(NORMAL, |_| panic!("marker 9e2b of the detached test"));

        // The only worker is still there and runs the next job.
        assert_eq!(system.spawn(NORMAL, |_| 3).wait(), Ok(3));

        let stats = system.stats();
        assert_eq!(
            (stats.submitted, stats.completed, stats.panicked),
            (2, 1, 1)
        );
        assert!(logger.has_error_with(&["marker 9e2b of the detached test", "detached-0"]));
    });
}

#[test]
fn logger_that_panics_does_not_take_the_worker_down() {
    with_deadline(|| {
        capture_logs();
        let system = JobSystem::new(1, "badlog");
        let message = format!("{LOGGER_TRAP}: a job panic the logger chokes on");

        // Reporting this panic makes the logger panic in turn, on the worker thread.
        let handle = system.spawn(NORMAL, {
            let message = message.clone();
            move |_| -> u32 { panic!("{message}") }
        });

        assert_eq!(handle.wait(), Err(JobError::Panicked(message)));
        // The only worker is still there and runs the next job.
        assert_eq!(system.spawn(NORMAL, |_| 4).wait(), Ok(4));
        assert_eq!(system.stats().panicked, 1);
    });
}

#[test]
fn all_workers_survive_many_panics() {
    with_deadline(|| {
        let system = JobSystem::new(4, "panic");
        let handles: Vec<_> = (0..200)
            .map(|index| system.spawn(NORMAL, move |_| -> u32 { panic!("panic number {index}") }))
            .collect();
        for (index, handle) in handles.into_iter().enumerate() {
            assert_eq!(
                handle.wait(),
                Err(JobError::Panicked(format!("panic number {index}")))
            );
        }

        // Returns only if four workers are alive at the same time.
        quiesce(&system);
        let stats = system.stats();
        assert_eq!((stats.panicked, stats.completed), (200, 4));
    });
}

#[test]
fn panicking_job_releases_what_its_closure_owned() {
    with_deadline(|| {
        let system = JobSystem::new(1, "panic");
        let resource = Arc::new(());
        let handle = system.spawn(NORMAL, {
            let resource = Arc::clone(&resource);
            move |_| -> u32 {
                let _held = resource;
                panic!("panicking while holding a resource")
            }
        });

        assert!(handle.wait().is_err());
        assert_eq!(Arc::strong_count(&resource), 1);
    });
}

#[test]
fn result_with_a_panicking_destructor_does_not_kill_the_worker() {
    with_deadline(|| {
        let logger = capture_logs();
        let system = JobSystem::new(1, "bomb");
        let gate = Gate::close(&system);
        // Nobody keeps the handle, so the worker itself has to drop the result.
        drop(system.spawn(NORMAL, |_| Explosive("unclaimed result")));

        gate.open();
        // Returns only if the worker is still alive.
        quiesce(&system);

        let stats = system.stats();
        assert_eq!(
            (stats.completed, stats.panicked),
            (3, 0),
            "the closure itself did not panic"
        );
        assert!(logger.has_error_with(&["Explosive 'unclaimed result'"]));
    });
}

#[test]
fn closure_with_a_panicking_destructor_is_contained_when_cancelled() {
    with_deadline(|| {
        let logger = capture_logs();
        let system = JobSystem::new(1, "bomb");
        let gate = Gate::close(&system);
        let bomb = Explosive("cancelled closure");
        let handle = system.spawn(NORMAL, move |_| drop(bomb));

        // Drops the closure, and with it the bomb, on this thread.
        handle.cancel();

        assert_eq!(handle.wait(), Err(JobError::Cancelled));
        assert!(logger.has_error_with(&["Explosive 'cancelled closure'"]));
        gate.open();
        quiesce(&system);
    });
}

#[test]
fn last_handle_dropped_by_a_job_is_reported_as_a_warning() {
    with_deadline(|| {
        let logger = capture_logs();

        // The trap: a job owns a clone of the system and outlives every other handle. Nothing
        // waits for the workers then, which is worth a warning.
        let system = JobSystem::new(1, "orphan-3f7a");
        let last_clone = system.clone();
        let (started_tx, started_rx) = bounded(1);
        let (release_tx, release_rx) = bounded::<()>(1);
        let job = system.spawn(NORMAL, move |_| {
            started_tx.send(()).unwrap();
            release_rx.recv().unwrap();
            drop(last_clone);
        });
        recv(&started_rx);
        drop(system);
        assert!(!logger.has_record_with(Level::Warn, &["orphan-3f7a"]));
        release_tx.send(()).unwrap();
        assert_eq!(job.wait(), Ok(()));
        assert!(logger.has_record_with(Level::Warn, &["orphan-3f7a", "JobSpawner", "shutdown"]));

        // The way out: the job holds a spawner, and the last handle is dropped outside.
        let system = JobSystem::new(1, "tidy-91c2");
        let spawner = system.spawner();
        let job = system.spawn(NORMAL, move |_| spawner.threads());
        assert_eq!(job.wait(), Ok(1));
        drop(system);
        assert!(!logger.has_record_with(Level::Warn, &["tidy-91c2"]));
    });
}
