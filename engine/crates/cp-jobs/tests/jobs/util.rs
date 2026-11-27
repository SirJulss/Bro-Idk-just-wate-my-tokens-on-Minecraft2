//! Helpers shared by the integration tests.

use std::cell::RefCell;
use std::panic::resume_unwind;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Barrier};
use std::thread;
use std::time::Duration;

use cp_jobs::{JobClass, JobHandle, JobStats, JobSystem, Priority};
use crossbeam_channel::{Receiver, RecvTimeoutError, Sender, bounded};

/// Upper bound for anything a test waits for. It is reached only when something is broken; it
/// is generous because the machine may be busy with other builds.
pub const DEADLINE: Duration = Duration::from_secs(60);

/// A priority for tests that do not care about order.
pub const NORMAL: Priority = Priority::new(JobClass::Streaming, 0);

/// Runs `body` on its own thread and fails if it does not return within [`DEADLINE`].
///
/// A panic of `body` continues on the calling thread with its original payload, so assertions
/// and `#[should_panic]` work as usual.
pub fn with_deadline(body: impl FnOnce() + Send + 'static) {
    let (done_tx, done_rx) = bounded::<()>(1);
    let runner = thread::Builder::new()
        .name("test-body".to_owned())
        .spawn(move || {
            body();
            // Not reached when `body` panics: the sender is dropped instead and the receiver
            // sees a disconnect.
            done_tx
                .send(())
                .expect("the test thread waits for this message");
        })
        .expect("the test body thread starts");
    match done_rx.recv_timeout(DEADLINE) {
        Ok(()) | Err(RecvTimeoutError::Disconnected) => {
            if let Err(panic) = runner.join() {
                resume_unwind(panic);
            }
        }
        Err(RecvTimeoutError::Timeout) => {
            panic!("the test body did not return within {DEADLINE:?}: deadlock?")
        }
    }
}

/// Receives from `channel`, failing the test if nothing arrives before [`DEADLINE`].
pub fn recv<T>(channel: &Receiver<T>) -> T {
    channel
        .recv_timeout(DEADLINE)
        .expect("a message arrives before the deadline")
}

/// Jobs that keep every worker of a system busy until the gate is opened. While the gate is
/// closed, everything a test spawns stays in the queue.
pub struct Gate {
    release: Sender<()>,
    jobs: Vec<JobHandle<()>>,
}

impl Gate {
    /// Occupies all workers of `system`; returns once every one of them is blocked.
    pub fn close(system: &JobSystem) -> Self {
        let workers = system.threads();
        let (started_tx, started_rx) = bounded(workers);
        let (release_tx, release_rx) = bounded::<()>(0);
        let jobs = (0..workers)
            .map(|_| {
                let started = started_tx.clone();
                let release = release_rx.clone();
                system.spawn(Priority::new(JobClass::Critical, 0), move |_| {
                    started
                        .send(())
                        .expect("the gate waits for its jobs to start");
                    // Fails, and thereby returns, as soon as the gate drops its sender.
                    let _ = release.recv();
                })
            })
            .collect();
        for _ in 0..workers {
            recv(&started_rx);
        }
        Self {
            release: release_tx,
            jobs,
        }
    }

    /// Lets the workers go and returns once all of them have left the gate.
    pub fn open(self) {
        drop(self.release);
        for job in self.jobs {
            job.wait().expect("a gate job returns normally");
        }
    }
}

/// Returns once every job that was submitted before the call is done and accounted for.
///
/// It queues one job per worker behind everything else and makes these jobs meet at a barrier:
/// all workers must be inside one at the same time, and a worker takes a new job only after it
/// has completely finished the previous one. The system must not be gated or shut down, and it
/// must not have a bulk limit: the jobs are `Background` jobs, the last in queue order.
pub fn quiesce(system: &JobSystem) {
    let workers = system.threads();
    let barrier = Arc::new(Barrier::new(workers));
    let jobs: Vec<_> = (0..workers)
        .map(|_| {
            let barrier = Arc::clone(&barrier);
            system.spawn(Priority::new(JobClass::Background, u32::MAX), move |_| {
                barrier.wait();
            })
        })
        .collect();
    for job in jobs {
        job.wait().expect("a barrier job returns normally");
    }
}

/// Counts ended threads: a thread-local value is dropped when its thread ends.
struct CountsThreadEnd(Arc<AtomicUsize>);

impl Drop for CountsThreadEnd {
    fn drop(&mut self) {
        self.0.fetch_add(1, Ordering::SeqCst);
    }
}

thread_local! {
    static AT_THREAD_END: RefCell<Option<CountsThreadEnd>> = const { RefCell::new(None) };
}

/// Returns a counter that every worker thread of `system` increments when it ends. Whoever
/// claims to have joined the workers can be checked against it.
///
/// Runs one job on every worker, all at the same time; the system must not be gated or shut
/// down. The jobs are `Critical`, so a bulk limit does not keep them apart.
pub fn count_worker_thread_ends(system: &JobSystem) -> Arc<AtomicUsize> {
    let workers = system.threads();
    let ended = Arc::new(AtomicUsize::new(0));
    // One job per worker, all of them inside at the same time, so that every worker thread
    // gets its own marker.
    let all_inside = Arc::new(Barrier::new(workers));
    let installed: Vec<_> = (0..workers)
        .map(|_| {
            let ended = Arc::clone(&ended);
            let all_inside = Arc::clone(&all_inside);
            system.spawn(Priority::new(JobClass::Critical, 0), move |_| {
                all_inside.wait();
                AT_THREAD_END.with(|marker| *marker.borrow_mut() = Some(CountsThreadEnd(ended)));
            })
        })
        .collect();
    for handle in installed {
        handle.wait().expect("a marker job returns normally");
    }
    assert_eq!(ended.load(Ordering::SeqCst), 0);
    ended
}

/// A value whose destructor panics, as badly behaved user data might. The tag ends up in the
/// panic message and tells the values of different tests apart.
pub struct Explosive(pub &'static str);

impl Drop for Explosive {
    fn drop(&mut self) {
        // A second panic while a panic already unwinds would abort the whole test process.
        if !thread::panicking() {
            panic!("the destructor of Explosive '{}' panicked", self.0);
        }
    }
}

/// Asserts the invariants that hold for every snapshot of the statistics.
pub fn assert_balanced(stats: &JobStats, threads: usize) {
    let in_flight = (stats.queued + stats.running) as u64;
    assert_eq!(
        stats.submitted,
        in_flight + stats.completed + stats.cancelled + stats.panicked,
        "every job is in exactly one state: {stats:?}"
    );
    assert_eq!(
        stats.queued,
        stats.queued_by_class.iter().sum::<usize>(),
        "{stats:?}"
    );
    assert!(stats.running <= threads, "{stats:?}");
}

/// Small deterministic pseudo-random generator (SplitMix64) for reproducible test inputs.
pub struct TestRng(u64);

impl TestRng {
    pub fn new(seed: u64) -> Self {
        Self(seed)
    }

    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// A value in `0..bound`. The slight modulo bias does not matter for tests.
    pub fn below(&mut self, bound: u32) -> u32 {
        (self.next_u64() % u64::from(bound)) as u32
    }

    /// A random priority out of four classes and `keys` different keys.
    pub fn priority(&mut self, keys: u32) -> Priority {
        let class = JobClass::ALL[self.below(JobClass::COUNT as u32) as usize];
        Priority::new(class, self.below(keys))
    }
}
