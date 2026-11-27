//! Cancellation through handles and tokens, and the rest of the `JobHandle` interface.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::thread;

use cp_jobs::{JobError, JobSystem};
use crossbeam_channel::bounded;

use crate::util::{Gate, NORMAL, quiesce, recv, with_deadline};

#[test]
fn job_cancelled_before_it_starts_never_runs() {
    with_deadline(|| {
        let system = JobSystem::new(1, "cancel");
        let gate = Gate::close(&system);
        let ran = Arc::new(AtomicBool::new(false));
        let handle = system.spawn(NORMAL, {
            let ran = Arc::clone(&ran);
            move |_| ran.store(true, Ordering::SeqCst)
        });
        assert!(!handle.is_finished());
        assert!(!handle.token().is_cancelled());
        assert_eq!(system.queued(), 1);

        handle.cancel();

        // Everything is settled when `cancel` returns, although the only worker is still busy.
        assert!(handle.is_finished());
        assert!(handle.token().is_cancelled());
        assert_eq!(system.queued(), 0);
        assert_eq!(handle.try_take(), Some(Err(JobError::Cancelled)));

        gate.open();
        quiesce(&system);
        assert!(!ran.load(Ordering::SeqCst));
        assert_eq!(system.stats().cancelled, 1);
    });
}

#[test]
fn cancelling_through_a_token_clone_is_the_same_as_through_the_handle() {
    with_deadline(|| {
        let system = JobSystem::new(1, "cancel");
        let gate = Gate::close(&system);
        let handle = system.spawn(NORMAL, |_| "ran");
        let token = handle.token();

        token.clone().cancel();

        assert!(token.is_cancelled());
        assert_eq!(system.queued(), 0);
        // Returns although no worker is free: the job is no longer anybody's business.
        assert_eq!(handle.wait(), Err(JobError::Cancelled));
        gate.open();
    });
}

#[test]
fn cancelling_a_queued_job_drops_its_closure_before_cancel_returns() {
    with_deadline(|| {
        let system = JobSystem::new(1, "cancel");
        let gate = Gate::close(&system);
        let resource = Arc::new(());
        let handle = system.spawn(NORMAL, {
            let resource = Arc::clone(&resource);
            move |_| drop(resource)
        });
        assert_eq!(Arc::strong_count(&resource), 2);

        handle.cancel();

        assert_eq!(Arc::strong_count(&resource), 1);
        gate.open();
    });
}

#[test]
fn cancelling_twice_or_after_the_fact_changes_nothing() {
    with_deadline(|| {
        let system = JobSystem::new(1, "cancel");
        let gate = Gate::close(&system);
        let cancelled = system.spawn(NORMAL, |_| 1);
        cancelled.cancel();
        cancelled.cancel();
        cancelled.token().cancel();
        assert_eq!(system.stats().cancelled, 1);
        assert_eq!(cancelled.wait(), Err(JobError::Cancelled));
        gate.open();

        let finished = system.spawn(NORMAL, |_| 5);
        quiesce(&system);
        assert!(finished.is_finished());
        finished.cancel();
        assert!(finished.token().is_cancelled());
        assert_eq!(finished.wait(), Ok(5));
        assert_eq!(system.stats().cancelled, 1);
    });
}

#[test]
fn cancelling_one_job_leaves_its_neighbours_alone() {
    with_deadline(|| {
        let system = JobSystem::new(2, "cancel");
        let gate = Gate::close(&system);
        let handles: Vec<_> = (0..20)
            .map(|index| system.spawn(NORMAL, move |_| index))
            .collect();
        for handle in handles.iter().skip(1).step_by(2) {
            handle.cancel();
        }
        assert_eq!(system.queued(), 10);
        gate.open();

        for (index, handle) in handles.into_iter().enumerate() {
            let expected = if index % 2 == 0 {
                Ok(index)
            } else {
                Err(JobError::Cancelled)
            };
            assert_eq!(handle.wait(), expected);
        }
    });
}

#[test]
fn cancelling_a_running_job_only_sets_its_token() {
    with_deadline(|| {
        let system = JobSystem::new(1, "cancel");
        let (started_tx, started_rx) = bounded(1);
        let (resume_tx, resume_rx) = bounded::<()>(1);
        let handle = system.spawn(NORMAL, move |token| {
            let before = token.is_cancelled();
            started_tx.send(()).unwrap();
            resume_rx.recv().unwrap();
            (before, token.is_cancelled())
        });
        recv(&started_rx);

        handle.cancel();

        assert!(!handle.is_finished(), "a running job is not interrupted");
        assert_eq!(handle.try_take(), None);
        resume_tx.send(()).unwrap();
        assert_eq!(handle.wait(), Ok((false, true)));
        assert_eq!(system.stats().cancelled, 0);
    });
}

#[test]
fn result_can_be_taken_exactly_once() {
    with_deadline(|| {
        let system = JobSystem::new(1, "take");
        let gate = Gate::close(&system);
        let handle = system.spawn(NORMAL, |_| String::from("result"));
        assert!(!handle.is_finished());
        assert_eq!(handle.try_take(), None);

        gate.open();
        quiesce(&system);

        assert!(handle.is_finished());
        assert_eq!(handle.try_take(), Some(Ok(String::from("result"))));
        assert_eq!(handle.try_take(), None);
        assert!(handle.is_finished());
    });
}

#[test]
#[should_panic(expected = "already taken")]
fn waiting_for_a_result_that_was_taken_panics() {
    with_deadline(|| {
        let system = JobSystem::new(1, "take");
        let handle = system.spawn(NORMAL, |_| 1);
        quiesce(&system);
        assert_eq!(handle.try_take(), Some(Ok(1)));
        let _ = handle.wait();
    });
}

#[test]
fn wait_blocks_until_the_job_has_run() {
    with_deadline(|| {
        let system = JobSystem::new(1, "wait");
        let gate = Gate::close(&system);
        let handle = system.spawn(NORMAL, |_| thread::current().name().map(str::to_owned));
        let (about_to_wait_tx, about_to_wait_rx) = bounded(1);
        let waiter = thread::spawn(move || {
            about_to_wait_tx.send(()).unwrap();
            handle.wait()
        });
        recv(&about_to_wait_rx);
        assert!(
            !waiter.is_finished(),
            "the job cannot have run behind the closed gate"
        );

        gate.open();
        assert_eq!(waiter.join().unwrap(), Ok(Some("wait-0".to_owned())));
    });
}

#[test]
fn dropping_a_handle_does_not_cancel_the_job() {
    with_deadline(|| {
        let system = JobSystem::new(1, "detach");
        let gate = Gate::close(&system);
        let (ran_tx, ran_rx) = bounded(1);
        drop(system.spawn(NORMAL, move |_| ran_tx.send(7).unwrap()));

        gate.open();
        assert_eq!(recv(&ran_rx), 7);
    });
}

#[test]
fn detached_job_runs_with_a_token_of_its_own() {
    with_deadline(|| {
        // One worker: the jobs run one after the other, so a token shared between them would
        // show up as cancelled at the start of the second job.
        let system = JobSystem::new(1, "detach");
        let (seen_tx, seen_rx) = bounded(2);
        for _ in 0..2 {
            let seen = seen_tx.clone();
            system.spawn_detached(NORMAL, move |token| {
                let before = token.is_cancelled();
                token.cancel();
                seen.send((before, token.is_cancelled())).unwrap();
            });
        }
        assert_eq!(recv(&seen_rx), (false, true));
        assert_eq!(recv(&seen_rx), (false, true));
        quiesce(&system);
        assert_eq!(system.stats().cancelled, 0);
    });
}

#[test]
fn concurrent_cancellation_either_prevents_the_run_or_loses_cleanly() {
    with_deadline(|| {
        const JOBS: usize = 20_000;
        let system = JobSystem::new(4, "race");
        let runs: Arc<Vec<AtomicUsize>> =
            Arc::new((0..JOBS).map(|_| AtomicUsize::new(0)).collect());
        let mut cancelled = 0_u64;
        for index in 0..JOBS {
            let handle = system.spawn(NORMAL, {
                let runs = Arc::clone(&runs);
                move |_| runs[index].fetch_add(1, Ordering::SeqCst)
            });
            // Races with the workers on purpose; both outcomes are legal, mixtures are not.
            handle.cancel();
            match handle.wait() {
                Ok(previous_runs) => {
                    assert_eq!(previous_runs, 0);
                    assert_eq!(runs[index].load(Ordering::SeqCst), 1);
                }
                Err(JobError::Cancelled) => {
                    cancelled += 1;
                    assert_eq!(runs[index].load(Ordering::SeqCst), 0);
                }
                Err(JobError::Panicked(message)) => panic!("job {index} panicked: {message}"),
            }
        }
        quiesce(&system);
        let stats = system.stats();
        assert_eq!(stats.cancelled, cancelled);
        assert_eq!(
            stats.completed,
            JOBS as u64 - cancelled + system.threads() as u64
        );
        let total_runs: usize = runs.iter().map(|count| count.load(Ordering::SeqCst)).sum();
        assert_eq!(total_runs as u64, JOBS as u64 - cancelled);
    });
}
