//! A `JobSpawner` gives job closures the system they run on without making them its owners:
//! the system shuts down when its last `JobSystem` handle is dropped, whoever holds spawners.

use std::sync::atomic::{AtomicU32, AtomicUsize, Ordering};
use std::sync::{Arc, Barrier};
use std::thread;

use cp_jobs::{JobClass, JobError, JobSystem, Priority};
use crossbeam_channel::bounded;

use crate::util::{
    Gate, NORMAL, assert_balanced, count_worker_thread_ends, quiesce, recv, with_deadline,
};

#[test]
fn spawner_shares_the_workers_the_queue_and_the_counters_of_its_system() {
    with_deadline(|| {
        let system = JobSystem::new(2, "shared");
        let spawner = system.spawner();
        assert_eq!(spawner.threads(), 2);

        // Jobs of a spawner run on the worker threads of the system.
        let worker = spawner
            .spawn(NORMAL, |_| thread::current().name().map(str::to_owned))
            .wait()
            .unwrap();
        assert!(
            matches!(worker.as_deref(), Some("shared-0" | "shared-1")),
            "the job ran on {worker:?}"
        );
        let (ran_tx, ran_rx) = bounded(1);
        spawner
            .clone()
            .spawn_detached(NORMAL, move |_| ran_tx.send(7).unwrap());
        assert_eq!(recv(&ran_rx), 7);

        // One queue: what the spawner queues, the system counts, and the other way round.
        let gate = Gate::close(&system);
        let by_spawner = spawner.spawn(Priority::new(JobClass::Background, 3), |_| "spawner");
        let by_system = system.spawn(Priority::new(JobClass::Interactive, 3), |_| "system");
        assert_eq!((spawner.queued(), system.queued()), (2, 2));
        assert_eq!(spawner.stats(), system.stats());
        assert_eq!(spawner.stats().queued_by_class, [0, 1, 0, 1]);

        // A job of a spawner is cancelled like any other.
        by_spawner.cancel();
        assert_eq!(by_spawner.wait(), Err(JobError::Cancelled));
        assert_eq!((spawner.queued(), system.queued()), (1, 1));

        gate.open();
        assert_eq!(by_system.wait(), Ok("system"));
        quiesce(&system);
        let stats = spawner.stats();
        assert_eq!(stats, system.stats());
        assert_balanced(&stats, 2);
        // Completed: the first two jobs, the two of the gate, `by_system`, the two of `quiesce`.
        assert_eq!(
            (stats.completed, stats.cancelled, stats.panicked),
            (7, 1, 0)
        );
    });
}

#[test]
fn dropping_the_system_cancels_and_joins_although_jobs_hold_spawners() {
    with_deadline(|| {
        const QUEUED: usize = 200;
        let system = JobSystem::new(1, "spawners");
        let ended = count_worker_thread_ends(&system);

        // A running job that needs the system, here in order to call `par_for` and to queue a
        // follow-up job.
        let (started_tx, started_rx) = bounded(1);
        let (release_tx, release_rx) = bounded::<()>(0);
        let running = system.spawn(NORMAL, {
            let spawner = system.spawner();
            move |token| {
                started_tx.send(()).unwrap();
                let _ = release_rx.recv();
                let sum = AtomicUsize::new(0);
                spawner.par_for(100, 10, |index| {
                    sum.fetch_add(index, Ordering::Relaxed);
                });
                let follow_up = spawner.spawn(NORMAL, |_| ());
                (token.is_cancelled(), sum.into_inner(), follow_up.wait())
            }
        });
        recv(&started_rx);

        // Queued jobs that need the system as well, each in order to queue its follow-up job.
        let resource = Arc::new(());
        let ran = Arc::new(AtomicUsize::new(0));
        let queued: Vec<_> = (0..QUEUED)
            .map(|_| {
                let spawner = system.spawner();
                let resource = Arc::clone(&resource);
                let ran = Arc::clone(&ran);
                system.spawn(NORMAL, move |_| {
                    ran.fetch_add(1, Ordering::SeqCst);
                    let ran = Arc::clone(&ran);
                    spawner.spawn_detached(NORMAL, move |_| {
                        ran.fetch_add(1, Ordering::SeqCst);
                        drop(resource);
                    });
                })
            })
            .collect();
        assert_eq!(system.queued(), QUEUED);
        assert_eq!(Arc::strong_count(&resource), QUEUED + 1);

        // The handle of the test is the only owner, whatever the jobs hold, so this drop shuts
        // the system down. It blocks until the running job is done and therefore happens on a
        // thread of its own, which reports how many worker threads had ended when it came back.
        let dropper = thread::spawn({
            let ended = Arc::clone(&ended);
            move || {
                drop(system);
                ended.load(Ordering::SeqCst)
            }
        });

        // The queued jobs are cancelled while the worker is still busy.
        for handle in queued {
            assert_eq!(handle.wait(), Err(JobError::Cancelled));
        }
        assert!(
            !dropper.is_finished(),
            "the drop must wait for the running job"
        );

        drop(release_tx);
        assert_eq!(
            dropper.join().unwrap(),
            1,
            "the drop returned while the worker thread was still alive"
        );
        // The running job was told to stop and was allowed to finish, its loop included. What
        // it queued after the shutdown had begun was cancelled.
        assert_eq!(running.wait(), Ok((true, 4950, Err(JobError::Cancelled))));
        assert_eq!(
            ran.load(Ordering::SeqCst),
            0,
            "a queued job or a follow-up job ran during shutdown"
        );
        assert_eq!(
            Arc::strong_count(&resource),
            1,
            "a queued closure was leaked"
        );
    });
}

#[test]
fn dropping_the_system_ends_the_wait_of_jobs_for_children_that_can_no_longer_start() {
    with_deadline(|| {
        const WORKERS: usize = 2;
        let system = JobSystem::new(WORKERS, "nested");
        let all_inside = Arc::new(Barrier::new(WORKERS));
        let (spawned_tx, spawned_rx) = bounded(WORKERS);
        let parents: Vec<_> = (0..WORKERS)
            .map(|index| {
                let spawner = system.spawner();
                let all_inside = Arc::clone(&all_inside);
                let spawned = spawned_tx.clone();
                system.spawn(NORMAL, move |token| {
                    // Every worker is inside a parent before the first child is queued, so no
                    // child can ever start.
                    all_inside.wait();
                    let child = spawner.spawn(NORMAL, move |_| index);
                    spawned.send(()).unwrap();
                    (child.wait(), token.is_cancelled())
                })
            })
            .collect();
        for _ in 0..WORKERS {
            recv(&spawned_rx);
        }
        assert_eq!(system.queued(), WORKERS);

        // Returns only because the cancellation of the children wakes both parents up. With
        // clones of the system in the parents instead of spawners, it would shut nothing down.
        drop(system);

        for parent in parents {
            assert_eq!(parent.wait(), Ok((Err(JobError::Cancelled), true)));
        }
    });
}

#[test]
fn spawner_outlives_its_system_and_spawns_only_cancelled_jobs_then() {
    with_deadline(|| {
        const WORKERS: usize = 3;
        let system = JobSystem::new(WORKERS, "gone");
        let ended = count_worker_thread_ends(&system);
        let spawner = system.spawner();
        assert_eq!(spawner.spawn(NORMAL, |_| 1).wait(), Ok(1));

        // The spawner is no owner: this is the last handle, and its drop joins the workers.
        drop(system);
        assert_eq!(ended.load(Ordering::SeqCst), WORKERS);

        let resource = Arc::new(());
        let ran = Arc::new(AtomicUsize::new(0));
        let late = spawner.spawn(NORMAL, {
            let resource = Arc::clone(&resource);
            let ran = Arc::clone(&ran);
            move |_| {
                ran.fetch_add(1, Ordering::SeqCst);
                drop(resource);
            }
        });
        // Everything is settled when `spawn` returns.
        assert!(late.is_finished());
        assert!(late.token().is_cancelled());
        assert_eq!(
            Arc::strong_count(&resource),
            1,
            "the closure of a rejected job was kept"
        );
        assert_eq!(late.wait(), Err(JobError::Cancelled));

        spawner.clone().spawn_detached(NORMAL, {
            let resource = Arc::clone(&resource);
            let ran = Arc::clone(&ran);
            move |_| {
                ran.fetch_add(1, Ordering::SeqCst);
                drop(resource);
            }
        });
        assert_eq!(
            Arc::strong_count(&resource),
            1,
            "the closure of a rejected detached job was kept"
        );
        assert_eq!(ran.load(Ordering::SeqCst), 0);

        // The counters are still there and still add up.
        let stats = spawner.stats();
        assert_balanced(&stats, WORKERS);
        // Completed: one marker job per worker and the first job. Cancelled: the two late ones.
        assert_eq!(
            (stats.submitted, stats.completed, stats.cancelled),
            (WORKERS as u64 + 3, WORKERS as u64 + 1, 2)
        );
        assert_eq!((spawner.queued(), spawner.threads()), (0, WORKERS));

        // Loops never needed the workers and still visit every index exactly once.
        let visits: Vec<AtomicU32> = (0..1000).map(|_| AtomicU32::new(0)).collect();
        spawner.par_for(visits.len(), 10, |index| {
            visits[index].fetch_add(1, Ordering::Relaxed);
        });
        assert!(
            visits
                .iter()
                .all(|visit| visit.load(Ordering::Relaxed) == 1)
        );
    });
}
