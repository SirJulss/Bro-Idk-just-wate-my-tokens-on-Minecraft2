//! The worker policy of contract §4.2: the standard system has `logical - 3` workers, at most
//! `physical - 1` bulk jobs run at once, and that limit never holds back `Critical` or
//! `Interactive` jobs.
//!
//! A job cannot see whether another worker would have taken a job it may not take, so the
//! tests let the bulk jobs report from inside: every bulk job counts itself in and out of an
//! [`Occupancy`], and most of them wait at a barrier or for a release, so that the jobs that may
//! run at once really are inside at once.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Barrier, Mutex};
use std::thread;

use cp_jobs::{CpuCores, JobClass, JobError, JobSystem, Priority, WorkerPolicy};
use crossbeam_channel::{bounded, unbounded};

use crate::util::{Gate, TestRng, assert_balanced, count_worker_thread_ends, recv, with_deadline};

/// Counts the jobs inside a section of code and remembers the most that were ever inside at
/// once. A job enters after it has started and leaves before it ends, so the peak is never
/// above the number of such jobs that really ran at the same time.
#[derive(Default)]
struct Occupancy {
    inside: AtomicUsize,
    peak: AtomicUsize,
}

impl Occupancy {
    fn enter(&self) -> Inside<'_> {
        let now = self.inside.fetch_add(1, Ordering::SeqCst) + 1;
        self.peak.fetch_max(now, Ordering::SeqCst);
        Inside(self)
    }

    fn peak(&self) -> usize {
        self.peak.load(Ordering::SeqCst)
    }
}

/// Leaves the section when dropped.
struct Inside<'a>(&'a Occupancy);

impl Drop for Inside<'_> {
    fn drop(&mut self) {
        self.0.inside.fetch_sub(1, Ordering::SeqCst);
    }
}

/// `(threads, bulk limit)` of `system`.
fn shape(system: &JobSystem) -> (usize, usize) {
    (system.threads(), system.bulk_limit())
}

/// Number of queued jobs of the bulk classes.
fn queued_bulk(system: &JobSystem) -> usize {
    let queued = system.stats().queued_by_class;
    JobClass::ALL
        .into_iter()
        .filter(|class| class.is_bulk())
        .map(|class| queued[class.index()])
        .sum()
}

/// One of the two bulk classes, alternating with `index`.
fn bulk_class(index: usize) -> JobClass {
    if index.is_multiple_of(2) {
        JobClass::Streaming
    } else {
        JobClass::Background
    }
}

/// One of the two urgent classes, alternating with `index`.
fn urgent_class(index: usize) -> JobClass {
    if index.is_multiple_of(2) {
        JobClass::Critical
    } else {
        JobClass::Interactive
    }
}

#[test]
fn standard_system_follows_the_contract_on_this_machine() {
    with_deadline(|| {
        let cores = CpuCores::detect();
        let policy = WorkerPolicy::standard(cores);
        let system = JobSystem::standard("standard");
        assert_eq!(shape(&system), (policy.threads(), policy.bulk_limit()));
        // Contract §4.2 in its own terms: `logical - 3` workers and at most `physical - 1` bulk
        // jobs at once, at least one of each and the limit within the workers.
        let threads = cores.logical().saturating_sub(3).max(1);
        assert_eq!(system.threads(), threads);
        assert_eq!(
            system.bulk_limit(),
            cores.physical().saturating_sub(1).max(1).min(threads)
        );
        assert_eq!(system.spawner().bulk_limit(), system.bulk_limit());
        for class in JobClass::ALL {
            let ran = system.spawn(Priority::new(class, 0), move |_| class).wait();
            assert_eq!(ran, Ok(class));
        }
    });
}

#[test]
fn new_sets_no_bulk_limit() {
    with_deadline(|| {
        const WORKERS: usize = 4;
        let system = JobSystem::new(WORKERS, "uncapped");
        assert_eq!(shape(&system), (WORKERS, WORKERS));
        // None of these returns before all of them are inside: every worker runs a bulk job.
        let all_inside = Arc::new(Barrier::new(WORKERS));
        let jobs: Vec<_> = (0..WORKERS)
            .map(|index| {
                let all_inside = Arc::clone(&all_inside);
                system.spawn(Priority::new(bulk_class(index), 0), move |_| {
                    all_inside.wait();
                })
            })
            .collect();
        for job in jobs {
            job.wait().expect("a bulk job returns normally");
        }
    });
}

#[test]
fn with_policy_keeps_the_limit_within_the_workers() {
    with_deadline(|| {
        let system = |threads, bulk_limit| {
            shape(&JobSystem::with_policy(
                WorkerPolicy::new(threads, bulk_limit),
                "shaped",
            ))
        };
        assert_eq!(system(3, 2), (3, 2));
        assert_eq!(system(3, 7), (3, 3));
        assert_eq!(system(3, 0), (3, 1));
        assert_eq!(system(0, 0), (1, 1));
    });
}

#[test]
fn bulk_limit_holds_under_load() {
    const WORKERS: usize = 6;
    const LIMIT: usize = 2;
    const SUBMITTERS: usize = 4;
    /// Jobs per submitter, every other one a bulk job; the bulk jobs of all submitters together
    /// are a multiple of `LIMIT`.
    const JOBS: usize = 2_000;

    with_deadline(|| {
        let system = JobSystem::with_policy(WorkerPolicy::new(WORKERS, LIMIT), "loaded");
        assert_eq!(shape(&system), (WORKERS, LIMIT));
        let bulk = Arc::new(Occupancy::default());
        let urgent = Arc::new(Occupancy::default());
        // Bulk jobs meet in groups of `LIMIT`: no bulk job returns before `LIMIT` of them are
        // inside at once, so the limit is reached again and again, however the threads are
        // scheduled. One bulk job too many would show in the peak.
        let group = Arc::new(Barrier::new(LIMIT));
        let start = Arc::new(Barrier::new(SUBMITTERS));

        let submitters: Vec<_> = (0..SUBMITTERS)
            .map(|submitter| {
                let system = system.clone();
                let bulk = Arc::clone(&bulk);
                let urgent = Arc::clone(&urgent);
                let group = Arc::clone(&group);
                let start = Arc::clone(&start);
                thread::spawn(move || {
                    let mut rng = TestRng::new(0xB01C_0000 + submitter as u64);
                    start.wait();
                    let handles: Vec<_> = (0..JOBS)
                        .map(|index| {
                            let pick = rng.below(2) as usize;
                            let class = if index.is_multiple_of(2) {
                                bulk_class(pick)
                            } else {
                                urgent_class(pick)
                            };
                            let priority = Priority::new(class, rng.below(16));
                            let bulk = Arc::clone(&bulk);
                            let urgent = Arc::clone(&urgent);
                            let group = Arc::clone(&group);
                            system.spawn(priority, move |_| {
                                if class.is_bulk() {
                                    let _inside = bulk.enter();
                                    group.wait();
                                } else {
                                    let _inside = urgent.enter();
                                }
                                class
                            })
                        })
                        .collect();
                    handles
                        .into_iter()
                        .map(|handle| handle.wait().expect("no job fails or is cancelled"))
                        .filter(|class| class.is_bulk())
                        .count()
                })
            })
            .collect();

        let bulk_jobs: usize = submitters
            .into_iter()
            .map(|submitter| submitter.join().expect("a submitting thread failed"))
            .sum();
        assert_eq!(bulk_jobs, SUBMITTERS * JOBS / 2);
        assert_eq!(bulk.peak(), LIMIT, "most bulk jobs that ran at once");
        assert!(urgent.peak() <= WORKERS);
        let stats = system.stats();
        assert_balanced(&stats, WORKERS);
        assert_eq!(
            (
                stats.completed,
                stats.cancelled,
                stats.panicked,
                stats.queued,
                stats.running
            ),
            ((SUBMITTERS * JOBS) as u64, 0, 0, 0, 0)
        );
    });
}

#[test]
fn urgent_jobs_run_while_bulk_jobs_fill_the_limit() {
    const WORKERS: usize = 5;
    const LIMIT: usize = 2;
    const BULK: usize = 12;
    const URGENT: usize = 300;

    with_deadline(|| {
        let system = JobSystem::with_policy(WorkerPolicy::new(WORKERS, LIMIT), "saturated");
        let bulk = Arc::new(Occupancy::default());
        let (started_tx, started_rx) = unbounded();
        let (release_tx, release_rx) = bounded::<()>(0);

        // Queued behind a closed gate, so that the bulk jobs that start are the first in queue
        // order: Streaming before Background, then by key.
        let gate = Gate::close(&system);
        let bulk_jobs: Vec<_> = (0..BULK)
            .map(|index| {
                let bulk = Arc::clone(&bulk);
                let started = started_tx.clone();
                let release = release_rx.clone();
                system.spawn(Priority::new(bulk_class(index), index as u32), move |_| {
                    let _inside = bulk.enter();
                    started.send(index).expect("the test counts the bulk jobs");
                    // Returns as soon as the test drops the sender.
                    let _ = release.recv();
                    index
                })
            })
            .collect();
        gate.open();

        // The limit lets two bulk jobs start; they keep their slots until they are released.
        let mut first = [recv(&started_rx), recv(&started_rx)];
        first.sort_unstable();
        assert_eq!(first, [0, 2], "the two most urgent bulk jobs start");

        // Three workers are left, and the limit holds back no urgent job: three urgent jobs run
        // at the same time, none of them returns before all three are inside.
        let free = WORKERS - LIMIT;
        let all_inside = Arc::new(Barrier::new(free));
        let meeting: Vec<_> = (0..free)
            .map(|index| {
                let all_inside = Arc::clone(&all_inside);
                system.spawn(Priority::new(urgent_class(index), 0), move |_| {
                    all_inside.wait();
                })
            })
            .collect();
        for job in meeting {
            job.wait().expect("an urgent job returns normally");
        }

        // Many more, all of them done while the two bulk jobs still fill the limit.
        let urgent: Vec<_> = (0..URGENT)
            .map(|index| {
                system.spawn(
                    Priority::new(urgent_class(index), index as u32),
                    move |_| index,
                )
            })
            .collect();
        for (index, job) in urgent.into_iter().enumerate() {
            assert_eq!(job.wait(), Ok(index));
        }

        // Meanwhile no further bulk job started: two run, the others wait in the queue.
        assert!(
            started_rx.try_recv().is_err(),
            "a bulk job started beyond the limit"
        );
        let stats = system.stats();
        assert_balanced(&stats, WORKERS);
        assert_eq!(stats.running, LIMIT);
        assert_eq!(queued_bulk(&system), BULK - LIMIT);

        // Released, all bulk jobs run, never more than `LIMIT` at once.
        drop(release_tx);
        for (index, job) in bulk_jobs.into_iter().enumerate() {
            assert_eq!(job.wait(), Ok(index));
        }
        let mut started: Vec<usize> = first.into_iter().chain(started_rx.try_iter()).collect();
        started.sort_unstable();
        assert_eq!(started, (0..BULK).collect::<Vec<_>>());
        assert_eq!(bulk.peak(), LIMIT);
    });
}

#[test]
fn bulk_jobs_start_in_queue_order_as_slots_free_up() {
    const JOBS: usize = 400;

    with_deadline(|| {
        // One slot: every bulk job starts only after the one before it has ended, on whichever
        // worker, so the bulk jobs run strictly in queue order.
        let system = JobSystem::with_policy(WorkerPolicy::new(3, 1), "one-slot");
        let gate = Gate::close(&system);
        let mut rng = TestRng::new(0x5107);
        let priorities: Vec<Priority> = (0..JOBS)
            .map(|_| Priority::new(bulk_class(rng.below(2) as usize), rng.below(8)))
            .collect();
        let log = Arc::new(Mutex::new(Vec::with_capacity(JOBS)));
        let bulk = Arc::new(Occupancy::default());
        let jobs: Vec<_> = priorities
            .iter()
            .enumerate()
            .map(|(index, &priority)| {
                let log = Arc::clone(&log);
                let bulk = Arc::clone(&bulk);
                system.spawn(priority, move |_| {
                    let _inside = bulk.enter();
                    log.lock().unwrap().push(index);
                })
            })
            .collect();
        gate.open();
        for job in jobs {
            job.wait().expect("a bulk job returns normally");
        }

        let mut expected: Vec<usize> = (0..JOBS).collect();
        // Stable, so equal priorities keep their submission order.
        expected.sort_by_key(|&index| priorities[index]);
        assert_eq!(*log.lock().unwrap(), expected);
        assert_eq!(bulk.peak(), 1);
    });
}

#[test]
fn shutdown_cancels_held_back_bulk_jobs_and_joins_the_workers_behind_the_limit() {
    const WORKERS: usize = 3;
    const HELD: usize = 20;

    with_deadline(|| {
        let system = JobSystem::with_policy(WorkerPolicy::new(WORKERS, 1), "held");
        let ended = count_worker_thread_ends(&system);
        let (started_tx, started_rx) = bounded(1);
        let (release_tx, release_rx) = bounded::<()>(0);
        let running = system.spawn(Priority::new(JobClass::Streaming, 0), move |token| {
            started_tx.send(()).unwrap();
            let _ = release_rx.recv();
            token.is_cancelled()
        });
        recv(&started_rx);

        // Held back by the limit, although two workers are idle.
        let ran = Arc::new(AtomicUsize::new(0));
        let held: Vec<_> = (0..HELD)
            .map(|index| {
                let ran = Arc::clone(&ran);
                system.spawn(Priority::new(bulk_class(index), index as u32), move |_| {
                    ran.fetch_add(1, Ordering::SeqCst);
                })
            })
            .collect();
        // The idle workers still take urgent work.
        let urgent = system.spawn(Priority::new(JobClass::Interactive, 0), |_| "urgent");
        assert_eq!(urgent.wait(), Ok("urgent"));
        assert_eq!(queued_bulk(&system), HELD);

        // The shutdown waits for the running job, so it happens on a thread of its own, which
        // reports how many workers had ended when it came back.
        let stopper = thread::spawn({
            let system = system.clone();
            let ended = Arc::clone(&ended);
            move || {
                system.shutdown();
                ended.load(Ordering::SeqCst)
            }
        });
        for job in held {
            assert_eq!(job.wait(), Err(JobError::Cancelled));
        }
        assert!(
            !stopper.is_finished(),
            "the shutdown must wait for the running bulk job"
        );

        drop(release_tx);
        assert_eq!(
            stopper.join().unwrap(),
            WORKERS,
            "a worker that waited behind the limit was not joined"
        );
        assert_eq!(running.wait(), Ok(true));
        assert_eq!(ran.load(Ordering::SeqCst), 0, "a held-back bulk job ran");
        let stats = system.stats();
        assert_balanced(&stats, WORKERS);
        assert_eq!(
            (stats.cancelled, stats.queued, stats.running),
            (HELD as u64, 0, 0)
        );
    });
}

#[test]
fn bulk_job_waiting_for_a_bulk_job_behind_the_limit_is_freed_by_the_shutdown() {
    with_deadline(|| {
        let system = JobSystem::with_policy(WorkerPolicy::new(2, 1), "nested-bulk");
        let (spawned_tx, spawned_rx) = bounded(1);
        let parent = system.spawn(Priority::new(JobClass::Streaming, 0), {
            let spawner = system.spawner();
            move |token| {
                // The child needs the one slot, which its parent keeps while it waits.
                let child = spawner.spawn(Priority::new(JobClass::Streaming, 1), |_| 7);
                spawned_tx.send(()).unwrap();
                (child.wait(), token.is_cancelled())
            }
        });
        recv(&spawned_rx);

        // The idle worker serves urgent jobs, but it does not start the child.
        let urgent = system.spawn(Priority::new(JobClass::Interactive, 0), |_| 1);
        assert_eq!(urgent.wait(), Ok(1));
        assert_eq!(queued_bulk(&system), 1);

        // The documented way out: the shutdown cancels the child and so ends the wait.
        system.shutdown();
        assert_eq!(parent.wait(), Ok((Err(JobError::Cancelled), true)));
    });
}
