//! Many small jobs from several submitting threads: nothing is lost, nothing runs twice.

use std::sync::atomic::{AtomicBool, AtomicU32, AtomicUsize, Ordering};
use std::sync::{Arc, Barrier};
use std::thread;

use cp_jobs::{JobError, JobSpawner, JobSystem};
use crossbeam_channel::{Sender, unbounded};

use crate::util::{NORMAL, TestRng, assert_balanced, quiesce, recv, with_deadline};

const WORKERS: usize = 6;
const SUBMITTERS: usize = 8;
const JOBS_PER_SUBMITTER: usize = 12_500;
const TOTAL: usize = SUBMITTERS * JOBS_PER_SUBMITTER;

/// What one submitting thread saw happen to its jobs.
#[derive(Default)]
struct Tally {
    detached: u64,
    completed: u64,
    cancelled: u64,
    panicked: u64,
}

/// Submits `JOBS_PER_SUBMITTER` jobs of every kind, waits for those with a handle and checks
/// each result against the run counter of its job.
fn submit_and_check(
    system: &JobSystem,
    submitter: usize,
    runs: &Arc<Vec<AtomicU32>>,
    detached_done: &Sender<usize>,
) -> Tally {
    let mut rng = TestRng::new(submitter as u64);
    let mut tally = Tally::default();
    let mut handles = Vec::new();
    for local in 0..JOBS_PER_SUBMITTER {
        let id = submitter * JOBS_PER_SUBMITTER + local;
        let priority = rng.priority(64);
        let runs = Arc::clone(runs);
        let kind = rng.below(16);
        if kind < 4 {
            let done = detached_done.clone();
            system.spawn_detached(priority, move |_| {
                runs[id].fetch_add(1, Ordering::Relaxed);
                done.send(id).expect("the test counts the detached jobs");
            });
            tally.detached += 1;
        } else {
            let panics = id.is_multiple_of(1009);
            let handle = system.spawn(priority, move |_| {
                runs[id].fetch_add(1, Ordering::Relaxed);
                assert!(!panics, "stress job {id} fails on purpose");
                id
            });
            let cancel_requested = kind < 6;
            if cancel_requested {
                // Races with the workers: the job may already be running or even done.
                handle.cancel();
            }
            handles.push((id, panics, cancel_requested, handle));
        }
    }

    for (id, panics, cancel_requested, handle) in handles {
        let result = handle.wait();
        let times_run = runs[id].load(Ordering::Relaxed);
        match result {
            Ok(value) => {
                assert_eq!(value, id);
                assert!(!panics, "job {id} should have panicked");
                assert_eq!(times_run, 1, "job {id} returned");
                tally.completed += 1;
            }
            Err(JobError::Cancelled) => {
                assert!(cancel_requested, "job {id} was cancelled by nobody");
                assert_eq!(times_run, 0, "job {id} ran although it was cancelled");
                tally.cancelled += 1;
            }
            Err(JobError::Panicked(message)) => {
                assert!(panics, "job {id} panicked unexpectedly: {message}");
                assert_eq!(message, format!("stress job {id} fails on purpose"));
                assert_eq!(times_run, 1, "job {id} panicked");
                tally.panicked += 1;
            }
        }
    }
    tally
}

#[test]
fn hundred_thousand_jobs_from_eight_threads() {
    with_deadline(|| {
        let system = JobSystem::new(WORKERS, "stress");
        let runs: Arc<Vec<AtomicU32>> = Arc::new((0..TOTAL).map(|_| AtomicU32::new(0)).collect());
        let (detached_tx, detached_rx) = unbounded();

        // Takes snapshots of the statistics for as long as the test runs; every single one
        // has to be consistent.
        let sampling = Arc::new(AtomicBool::new(true));
        let sampler = thread::spawn({
            let system = system.clone();
            let sampling = Arc::clone(&sampling);
            move || {
                loop {
                    assert_balanced(&system.stats(), WORKERS);
                    if !sampling.load(Ordering::Relaxed) {
                        break;
                    }
                    thread::yield_now();
                }
            }
        });

        let start = Arc::new(Barrier::new(SUBMITTERS));
        let submitters: Vec<_> = (0..SUBMITTERS)
            .map(|submitter| {
                let system = system.clone();
                let runs = Arc::clone(&runs);
                let detached_done = detached_tx.clone();
                let start = Arc::clone(&start);
                thread::spawn(move || {
                    start.wait();
                    submit_and_check(&system, submitter, &runs, &detached_done)
                })
            })
            .collect();

        let mut total = Tally::default();
        for submitter in submitters {
            let tally = submitter.join().expect("a submitting thread failed");
            total.detached += tally.detached;
            total.completed += tally.completed;
            total.cancelled += tally.cancelled;
            total.panicked += tally.panicked;
        }
        assert_eq!(
            total.detached + total.completed + total.cancelled + total.panicked,
            TOTAL as u64
        );

        // Every detached job reports in exactly once.
        let mut reported = vec![false; TOTAL];
        for _ in 0..total.detached {
            let id = recv(&detached_rx);
            assert!(!reported[id], "detached job {id} ran twice");
            reported[id] = true;
        }

        quiesce(&system);
        sampling.store(false, Ordering::Relaxed);
        sampler
            .join()
            .expect("a statistics snapshot was inconsistent");

        assert!(
            detached_rx.try_recv().is_err(),
            "a detached job ran more often than once"
        );
        let runs_in_total: u64 = runs
            .iter()
            .map(|count| u64::from(count.load(Ordering::Relaxed)))
            .sum();
        assert!(runs.iter().all(|count| count.load(Ordering::Relaxed) <= 1));
        assert_eq!(runs_in_total, TOTAL as u64 - total.cancelled);

        let stats = system.stats();
        assert_balanced(&stats, WORKERS);
        // `quiesce` added one job per worker.
        assert_eq!(stats.submitted, (TOTAL + WORKERS) as u64);
        assert_eq!(
            stats.completed,
            total.detached + total.completed + WORKERS as u64
        );
        assert_eq!(stats.cancelled, total.cancelled);
        assert_eq!(stats.panicked, total.panicked);
        assert_eq!((stats.queued, stats.running), (0, 0));
        assert!(
            total.panicked > 0,
            "the test is meant to include panicking jobs"
        );
    });
}

#[test]
fn jobs_that_spawn_jobs_all_run() {
    /// Spawns a job that counts itself and spawns two children until `depth` is used up. Every
    /// job carries a spawner for that, which makes none of them an owner of the system.
    fn grow(jobs: &JobSpawner, depth: u32, grown: &Arc<AtomicUsize>, done: &Sender<()>) {
        let spawner = jobs.clone();
        let grown = Arc::clone(grown);
        let done = done.clone();
        jobs.spawn_detached(NORMAL, move |_| {
            if depth > 0 {
                grow(&spawner, depth - 1, &grown, &done);
                grow(&spawner, depth - 1, &grown, &done);
            }
            if grown.fetch_add(1, Ordering::SeqCst) + 1 == NODES {
                done.send(()).expect("the test waits for the last node");
            }
        });
    }

    const DEPTH: u32 = 14;
    const NODES: usize = (1 << (DEPTH + 1)) - 1;

    with_deadline(|| {
        let system = JobSystem::new(4, "tree");
        let grown = Arc::new(AtomicUsize::new(0));
        let (done_tx, done_rx) = unbounded();

        grow(&system.spawner(), DEPTH, &grown, &done_tx);
        recv(&done_rx);

        // The last node has counted itself, but it may not be accounted for yet.
        quiesce(&system);
        assert_eq!(grown.load(Ordering::SeqCst), NODES);
        let stats = system.stats();
        assert_balanced(&stats, 4);
        assert_eq!(stats.completed, (NODES + 4) as u64);
        assert_eq!((stats.cancelled, stats.panicked, stats.queued), (0, 0, 0));
    });
}
