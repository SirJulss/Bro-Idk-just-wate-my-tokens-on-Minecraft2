//! Dropping the system, or shutting it down explicitly, cancels queued jobs, lets running jobs
//! finish and joins the workers, without deadlocks. The explicit shutdown does so even while
//! jobs own clones of the system.

use std::sync::atomic::{AtomicU32, AtomicUsize, Ordering};
use std::sync::{Arc, Barrier};
use std::thread;

use cp_jobs::{JobClass, JobError, JobHandle, JobSystem, Priority};
use crossbeam_channel::bounded;

use crate::util::{
    Explosive, Gate, NORMAL, assert_balanced, count_worker_thread_ends, quiesce, recv,
    with_deadline,
};

#[test]
fn drop_cancels_queued_jobs_and_waits_for_running_ones() {
    with_deadline(|| {
        const WORKERS: usize = 2;
        const QUEUED: usize = 100;
        let system = JobSystem::new(WORKERS, "shutdown");

        // Two jobs that are running, and stay running until released.
        let (started_tx, started_rx) = bounded(WORKERS);
        let (release_tx, release_rx) = bounded::<()>(0);
        let returned = Arc::new(AtomicUsize::new(0));
        let running: Vec<_> = (0..WORKERS)
            .map(|_| {
                let started = started_tx.clone();
                let release = release_rx.clone();
                let returned = Arc::clone(&returned);
                system.spawn(NORMAL, move |_| {
                    started.send(()).unwrap();
                    let _ = release.recv();
                    returned.fetch_add(1, Ordering::SeqCst);
                    "ran to the end"
                })
            })
            .collect();
        for _ in 0..WORKERS {
            recv(&started_rx);
        }

        // A hundred jobs that cannot start, because both workers are busy.
        let ran = Arc::new(AtomicUsize::new(0));
        let queued: Vec<_> = (0..QUEUED)
            .map(|_| {
                let ran = Arc::clone(&ran);
                system.spawn(NORMAL, move |_| ran.fetch_add(1, Ordering::SeqCst))
            })
            .collect();
        assert_eq!(system.queued(), QUEUED);

        // The drop blocks until the running jobs are done, so it happens on a thread of its
        // own, which reports how many running jobs had returned when the drop came back.
        let dropper = thread::spawn({
            let returned = Arc::clone(&returned);
            move || {
                drop(system);
                returned.load(Ordering::SeqCst)
            }
        });

        // The queued jobs are cancelled while both workers are still busy.
        for handle in queued {
            assert_eq!(handle.wait(), Err(JobError::Cancelled));
        }
        assert!(
            !dropper.is_finished(),
            "the drop must wait for the running jobs"
        );

        drop(release_tx);
        assert_eq!(
            dropper.join().unwrap(),
            WORKERS,
            "the drop returned before the jobs did"
        );
        for handle in running {
            assert_eq!(handle.wait(), Ok("ran to the end"));
        }
        assert_eq!(
            ran.load(Ordering::SeqCst),
            0,
            "a queued job ran during shutdown"
        );
    });
}

#[test]
fn drop_tells_running_jobs_to_stop_through_their_token() {
    with_deadline(|| {
        let system = JobSystem::new(2, "shutdown");
        let (started_tx, started_rx) = bounded(2);
        let polling = system.spawn(NORMAL, {
            let started = started_tx.clone();
            move |token| {
                started.send(()).unwrap();
                while !token.is_cancelled() {
                    thread::yield_now();
                }
                "stopped early"
            }
        });
        let (seen_tx, seen_rx) = bounded(1);
        system.spawn_detached(NORMAL, move |token| {
            started_tx.send(()).unwrap();
            while !token.is_cancelled() {
                thread::yield_now();
            }
            seen_tx.send(()).unwrap();
        });
        recv(&started_rx);
        recv(&started_rx);

        // Returns only because both jobs notice the cancellation of their tokens.
        drop(system);

        assert_eq!(polling.wait(), Ok("stopped early"));
        recv(&seen_rx);
    });
}

#[test]
fn drop_joins_every_worker_thread() {
    with_deadline(|| {
        const WORKERS: usize = 4;
        let system = JobSystem::new(WORKERS, "join");
        let ended = count_worker_thread_ends(&system);
        assert_eq!(ended.load(Ordering::SeqCst), 0);

        drop(system);

        assert_eq!(
            ended.load(Ordering::SeqCst),
            WORKERS,
            "the drop returned while a worker thread was still alive"
        );
    });
}

#[test]
fn drop_runs_destructors_of_queued_closures_and_runs_none_of_them() {
    with_deadline(|| {
        let system = JobSystem::new(1, "shutdown");
        let gate = Gate::close(&system);
        let resource = Arc::new(());
        let ran = Arc::new(AtomicUsize::new(0));
        for _ in 0..10 {
            let resource = Arc::clone(&resource);
            let ran = Arc::clone(&ran);
            system.spawn_detached(NORMAL, move |_| {
                ran.fetch_add(1, Ordering::SeqCst);
                drop(resource);
            });
        }
        // Queued behind the detached jobs; tells the test when the shutdown has emptied the queue.
        let marker = system.spawn(NORMAL, |_| ());
        assert_eq!(Arc::strong_count(&resource), 11);

        let dropper = thread::spawn(move || drop(system));
        assert_eq!(marker.wait(), Err(JobError::Cancelled));
        gate.open();
        dropper.join().unwrap();

        assert_eq!(
            Arc::strong_count(&resource),
            1,
            "a queued closure was leaked"
        );
        assert_eq!(ran.load(Ordering::SeqCst), 0);
    });
}

#[test]
fn drop_contains_a_queued_closure_with_a_panicking_destructor() {
    with_deadline(|| {
        let system = JobSystem::new(1, "shutdown");
        let gate = Gate::close(&system);
        let bomb = Explosive("closure queued at shutdown");
        let first = system.spawn(NORMAL, move |_| drop(bomb));
        let second = system.spawn(NORMAL, |_| ());

        let dropper = thread::spawn(move || drop(system));
        // Both are cancelled, and the job behind the bomb is cleaned up like any other.
        assert_eq!(first.wait(), Err(JobError::Cancelled));
        assert_eq!(second.wait(), Err(JobError::Cancelled));
        gate.open();
        dropper
            .join()
            .expect("the panic of the destructor must not escape from the drop");
    });
}

#[test]
fn creating_and_dropping_idle_systems_never_hangs() {
    with_deadline(|| {
        for round in 0..200 {
            let threads = 1 + round % 8;
            let system = JobSystem::new(threads, "idle");
            assert_eq!(system.threads(), threads);
            // Workers may not even have reached their first wait yet.
            drop(system);
        }
    });
}

#[test]
fn system_lives_as_long_as_any_clone() {
    with_deadline(|| {
        let first = JobSystem::new(2, "clones");
        let second = first.clone();
        let handle = first.spawn(NORMAL, |_| 1);
        drop(first);

        assert_eq!(handle.wait(), Ok(1));
        assert_eq!(second.threads(), 2);
        assert_eq!(second.spawn(NORMAL, |_| 2).wait(), Ok(2));
    });
}

#[test]
fn dropping_the_last_clone_inside_a_job_does_not_deadlock() {
    with_deadline(|| {
        let system = JobSystem::new(2, "selfdrop");
        let last_clone = system.clone();
        let (started_tx, started_rx) = bounded(2);
        let (release_tx, release_rx) = bounded::<()>(1);

        let owner = system.spawn(NORMAL, {
            let started = started_tx.clone();
            move |_| {
                started.send(()).unwrap();
                release_rx.recv().unwrap();
                // Shuts the system down from one of its own workers, while the other worker
                // runs a job that waits for this one.
                drop(last_clone);
                7
            }
        });
        let waiter = system.spawn(NORMAL, move |_| {
            started_tx.send(()).unwrap();
            owner.wait()
        });
        recv(&started_rx);
        recv(&started_rx);

        // Not the last clone: returns at once and shuts nothing down.
        drop(system);
        release_tx.send(()).unwrap();

        assert_eq!(waiter.wait(), Ok(Ok(7)));
    });
}

#[test]
fn shutdown_from_inside_a_job_cancels_what_is_still_queued() {
    with_deadline(|| {
        let system = JobSystem::new(1, "selfdrop");
        let last_clone = system.clone();
        let (started_tx, started_rx) = bounded(1);
        let (release_tx, release_rx) = bounded::<()>(1);
        let owner = system.spawn(NORMAL, move |token| {
            started_tx.send(()).unwrap();
            release_rx.recv().unwrap();
            drop(last_clone);
            // The shutdown has reached the token of this very job as well.
            token.is_cancelled()
        });
        recv(&started_rx);
        let ran = Arc::new(AtomicUsize::new(0));
        let queued = system.spawn(NORMAL, {
            let ran = Arc::clone(&ran);
            move |_| ran.fetch_add(1, Ordering::SeqCst)
        });

        // Not the last clone: the system stays up and the queued job stays queued.
        drop(system);
        assert!(!queued.is_finished());
        release_tx.send(()).unwrap();

        assert_eq!(owner.wait(), Ok(true));
        assert_eq!(queued.wait(), Err(JobError::Cancelled));
        assert_eq!(ran.load(Ordering::SeqCst), 0);
    });
}

#[test]
fn drop_does_not_deadlock_when_a_running_job_waits_for_a_queued_one() {
    with_deadline(|| {
        let system = JobSystem::new(1, "waiting");
        let (started_tx, started_rx) = bounded(1);
        let (handle_tx, handle_rx) = bounded::<JobHandle<()>>(1);
        let waiter = system.spawn(NORMAL, move |_| {
            started_tx.send(()).unwrap();
            handle_rx.recv().unwrap().wait()
        });
        recv(&started_rx);
        // Cannot start: the only worker is busy with the job that will wait for it.
        let queued = system.spawn(NORMAL, |_| ());
        handle_tx.send(queued).unwrap();

        drop(system);

        assert_eq!(waiter.wait(), Ok(Err(JobError::Cancelled)));
    });
}

#[test]
fn handles_and_tokens_outlive_the_system() {
    with_deadline(|| {
        let system = JobSystem::new(1, "outlive");
        let finished = system.spawn(NORMAL, |_| 11);
        quiesce(&system);
        let gate = Gate::close(&system);
        let queued = system.spawn(NORMAL, |_| 12);
        let queued_token = queued.token();

        let dropper = thread::spawn(move || drop(system));
        assert_eq!(queued.wait(), Err(JobError::Cancelled));
        gate.open();
        dropper.join().unwrap();

        assert!(
            queued_token.is_cancelled(),
            "shutdown cancels the tokens of queued jobs"
        );
        let token = finished.token();
        assert!(
            !token.is_cancelled(),
            "a job that finished earlier is not touched"
        );
        // Nothing is left to cancel; this must simply do nothing.
        token.cancel();
        assert!(token.is_cancelled());
        assert_eq!(finished.wait(), Ok(11));
    });
}

#[test]
fn shutdown_cancels_and_joins_while_jobs_own_clones_of_the_system() {
    with_deadline(|| {
        const QUEUED: usize = 200;
        let system = JobSystem::new(1, "owned");
        let ended = count_worker_thread_ends(&system);

        // A running job that owns a clone of the system, here in order to call `par_for`.
        let (started_tx, started_rx) = bounded(1);
        let (release_tx, release_rx) = bounded::<()>(0);
        let running = system.spawn(NORMAL, {
            let inner = system.clone();
            move |token| {
                started_tx.send(()).unwrap();
                let _ = release_rx.recv();
                let sum = AtomicUsize::new(0);
                inner.par_for(100, 10, |index| {
                    sum.fetch_add(index, Ordering::Relaxed);
                });
                (token.is_cancelled(), sum.into_inner())
            }
        });
        recv(&started_rx);

        // Queued jobs that own clones as well, each in order to queue its follow-up job.
        let ran = Arc::new(AtomicUsize::new(0));
        let queued: Vec<_> = (0..QUEUED)
            .map(|_| {
                let inner = system.clone();
                let ran = Arc::clone(&ran);
                system.spawn(NORMAL, move |_| {
                    ran.fetch_add(1, Ordering::SeqCst);
                    let ran = Arc::clone(&ran);
                    inner.spawn_detached(NORMAL, move |_| {
                        ran.fetch_add(1, Ordering::SeqCst);
                    });
                })
            })
            .collect();
        assert_eq!(system.queued(), QUEUED);

        // Dropping the test's own handle would shut nothing down here: the jobs hold 201
        // clones. `shutdown` does not depend on that. It blocks until the running job is done,
        // so it is called on a thread of its own, which reports how many worker threads had
        // ended when the call came back.
        let stopper = thread::spawn({
            let system = system.clone();
            let ended = Arc::clone(&ended);
            move || {
                system.shutdown();
                ended.load(Ordering::SeqCst)
            }
        });

        // The queued jobs are cancelled while the worker is still busy.
        for handle in queued {
            assert_eq!(handle.wait(), Err(JobError::Cancelled));
        }
        assert!(
            !stopper.is_finished(),
            "the shutdown must wait for the running job"
        );

        drop(release_tx);
        assert_eq!(
            stopper.join().unwrap(),
            1,
            "the shutdown returned while the worker thread was still alive"
        );
        // The running job was told to stop and was allowed to finish, its loop included.
        assert_eq!(running.wait(), Ok((true, 4950)));
        assert_eq!(
            ran.load(Ordering::SeqCst),
            0,
            "a queued job or a follow-up job ran during shutdown"
        );
        let stats = system.stats();
        assert_balanced(&stats, 1);
        // Completed: the job that installed the thread-end marker, and the running job.
        assert_eq!(
            (
                stats.completed,
                stats.cancelled,
                stats.queued,
                stats.running
            ),
            (2, QUEUED as u64, 0, 0)
        );
    });
}

#[test]
fn shutdown_ends_the_wait_of_jobs_for_children_that_can_no_longer_start() {
    with_deadline(|| {
        const WORKERS: usize = 2;
        let system = JobSystem::new(WORKERS, "nested");
        let all_inside = Arc::new(Barrier::new(WORKERS));
        let (spawned_tx, spawned_rx) = bounded(WORKERS);
        let parents: Vec<_> = (0..WORKERS)
            .map(|index| {
                // A clone: every parent is an owner of the system it waits in.
                let inner = system.clone();
                let all_inside = Arc::clone(&all_inside);
                let spawned = spawned_tx.clone();
                system.spawn(NORMAL, move |token| {
                    // Every worker is inside a parent before the first child is queued, so no
                    // child can ever start.
                    all_inside.wait();
                    let child = inner.spawn(NORMAL, move |_| index);
                    spawned.send(()).unwrap();
                    (child.wait(), token.is_cancelled())
                })
            })
            .collect();
        for _ in 0..WORKERS {
            recv(&spawned_rx);
        }
        assert_eq!(system.queued(), WORKERS);

        // Returns only because the cancellation of the children wakes both parents up.
        system.shutdown();

        for parent in parents {
            assert_eq!(parent.wait(), Ok((Err(JobError::Cancelled), true)));
        }
        let stats = system.stats();
        assert_balanced(&stats, WORKERS);
        assert_eq!(
            (stats.completed, stats.cancelled),
            (WORKERS as u64, WORKERS as u64)
        );
    });
}

#[test]
fn jobs_spawned_after_the_shutdown_are_cancelled_and_never_run() {
    with_deadline(|| {
        let system = JobSystem::new(2, "late");
        assert_eq!(system.spawn(NORMAL, |_| 1).wait(), Ok(1));

        system.shutdown();

        let resource = Arc::new(());
        let ran = Arc::new(AtomicUsize::new(0));
        let late = system.clone().spawn(NORMAL, {
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

        system.spawn_detached(NORMAL, {
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

        let stats = system.stats();
        assert_balanced(&stats, 2);
        assert_eq!(
            (
                stats.submitted,
                stats.completed,
                stats.cancelled,
                stats.queued
            ),
            (3, 1, 2, 0)
        );
        assert_eq!(system.threads(), 2);

        // Loops never needed the workers and still visit every index exactly once.
        let visits: Vec<AtomicU32> = (0..1000).map(|_| AtomicU32::new(0)).collect();
        system.par_for(visits.len(), 10, |index| {
            visits[index].fetch_add(1, Ordering::Relaxed);
        });
        assert!(
            visits
                .iter()
                .all(|visit| visit.load(Ordering::Relaxed) == 1)
        );
    });
}

#[test]
fn every_caller_of_shutdown_returns_after_the_workers_have_ended() {
    with_deadline(|| {
        const WORKERS: usize = 4;
        const CALLERS: usize = 6;
        let system = JobSystem::new(WORKERS, "callers");
        let ended = count_worker_thread_ends(&system);
        // A job that keeps running while the callers arrive, so that all of them have to wait.
        let (started_tx, started_rx) = bounded(1);
        let (release_tx, release_rx) = bounded::<()>(0);
        system.spawn_detached(NORMAL, move |_| {
            started_tx.send(()).unwrap();
            let _ = release_rx.recv();
        });
        recv(&started_rx);

        let (calling_tx, calling_rx) = bounded(CALLERS);
        let callers: Vec<_> = (0..CALLERS)
            .map(|_| {
                let system = system.clone();
                let ended = Arc::clone(&ended);
                let calling = calling_tx.clone();
                thread::spawn(move || {
                    calling.send(()).unwrap();
                    system.shutdown();
                    ended.load(Ordering::SeqCst)
                })
            })
            .collect();
        for _ in 0..CALLERS {
            recv(&calling_rx);
        }
        assert!(
            callers.iter().all(|caller| !caller.is_finished()),
            "a shutdown returned while a job was still running"
        );

        drop(release_tx);
        for caller in callers {
            assert_eq!(
                caller.join().unwrap(),
                WORKERS,
                "a shutdown returned while a worker thread was still alive"
            );
        }

        // Nothing is left to do for further calls, nor for the drop at the end of the test.
        system.shutdown();
        system.shutdown();
        let stats = system.stats();
        assert_balanced(&stats, WORKERS);
        assert_eq!((stats.queued, stats.running), (0, 0));
    });
}

#[test]
fn shutdown_called_by_a_job_does_not_wait_for_that_job() {
    with_deadline(|| {
        let system = JobSystem::new(1, "inside");
        let ended = count_worker_thread_ends(&system);
        let gate = Gate::close(&system);
        let stopper = system.spawn(Priority::new(JobClass::Interactive, 0), {
            let inner = system.clone();
            move |token| {
                // Would never return if it waited for the running jobs: this is one of them.
                inner.shutdown();
                // The shutdown has reached the token of this very job as well.
                token.is_cancelled()
            }
        });
        let ran = Arc::new(AtomicUsize::new(0));
        let queued = system.spawn(Priority::new(JobClass::Background, 0), {
            let ran = Arc::clone(&ran);
            move |_| ran.fetch_add(1, Ordering::SeqCst)
        });

        gate.open();
        assert_eq!(stopper.wait(), Ok(true));
        assert_eq!(queued.wait(), Err(JobError::Cancelled));
        assert_eq!(ran.load(Ordering::SeqCst), 0);
        assert_eq!(
            system.spawn(NORMAL, |_| ()).wait(),
            Err(JobError::Cancelled),
            "the system accepted a job after its shutdown"
        );

        // The job did not join the worker it ran on; the next call from outside does.
        system.shutdown();
        assert_eq!(ended.load(Ordering::SeqCst), 1);
    });
}

#[test]
fn shutdown_from_a_par_for_body_inside_a_job_does_not_wait_for_that_job() {
    with_deadline(|| {
        const THREADS: usize = 3;
        let system = JobSystem::new(THREADS, "loop");
        let ended = count_worker_thread_ends(&system);
        let job = system.spawn(NORMAL, {
            let inner = system.clone();
            move |token| {
                // Each thread stays in its first call until all three have arrived, so the body
                // runs on the worker and on two helper threads of the loop. A helper that
                // waited for the workers would wait for this job, which waits for the helper.
                let all_inside = Barrier::new(THREADS);
                inner.par_for(THREADS, 1, |_| {
                    all_inside.wait();
                    inner.shutdown();
                });
                token.is_cancelled()
            }
        });

        assert_eq!(job.wait(), Ok(true));
        assert_eq!(
            system.spawn(NORMAL, |_| ()).wait(),
            Err(JobError::Cancelled)
        );

        // From outside the jobs, the workers are joined as usual.
        system.shutdown();
        assert_eq!(ended.load(Ordering::SeqCst), THREADS);
    });
}
