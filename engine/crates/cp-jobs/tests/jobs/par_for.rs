//! `par_for` visits every index exactly once, in parallel, and propagates panics to its caller.

use std::collections::HashSet;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Barrier, Mutex};
use std::thread;

use cp_jobs::JobSystem;

use crate::util::{NORMAL, with_deadline};

/// One visit counter per index.
fn counters(count: usize) -> Vec<AtomicU32> {
    (0..count).map(|_| AtomicU32::new(0)).collect()
}

#[test]
fn visits_every_index_exactly_once() {
    with_deadline(|| {
        let shapes = [
            (0, 1),
            (1, 1),
            (7, 1),
            (7, 3),
            (64, 64),
            (64, 65),
            (1000, 0),
            (1000, 1),
            (1000, 7),
            (4097, 64),
            (100_000, 1000),
        ];
        for threads in [1, 2, 3, 8] {
            let system = JobSystem::new(threads, "par");
            for (count, grain) in shapes {
                let visits = counters(count);
                system.par_for(count, grain, |index| {
                    visits[index].fetch_add(1, Ordering::Relaxed);
                });
                let wrong = visits
                    .iter()
                    .filter(|visit| visit.load(Ordering::Relaxed) != 1)
                    .count();
                assert_eq!(wrong, 0, "threads {threads}, count {count}, grain {grain}");
            }
        }
    });
}

#[test]
fn body_may_borrow_from_the_stack_of_the_caller() {
    with_deadline(|| {
        let system = JobSystem::new(4, "par");
        let input: Vec<u64> = (0..10_000).collect();
        let sum = AtomicU64::new(0);

        system.par_for(input.len(), 100, |index| {
            sum.fetch_add(input[index] * 2, Ordering::Relaxed);
        });

        assert_eq!(sum.into_inner(), 2 * (0..10_000_u64).sum::<u64>());
    });
}

#[test]
fn runs_on_as_many_threads_as_the_system_has_including_the_caller() {
    with_deadline(|| {
        const THREADS: usize = 4;
        let system = JobSystem::new(THREADS, "par");
        // Each thread stays in its first call until all four have arrived, so the loop can
        // only finish if four different threads take part.
        let barrier = Barrier::new(THREADS);
        let seen = Mutex::new(HashSet::new());

        system.par_for(THREADS, 1, |_| {
            barrier.wait();
            seen.lock().unwrap().insert(thread::current().id());
        });

        let seen = seen.into_inner().unwrap();
        assert_eq!(seen.len(), THREADS);
        assert!(
            seen.contains(&thread::current().id()),
            "the calling thread takes part"
        );
    });
}

#[test]
fn loop_that_fits_into_one_chunk_runs_on_the_caller_in_order() {
    with_deadline(|| {
        let system = JobSystem::new(4, "par");
        let calls = Mutex::new(Vec::new());

        system.par_for(10, 10, |index| {
            calls.lock().unwrap().push((index, thread::current().id()))
        });

        let caller = thread::current().id();
        let expected: Vec<_> = (0..10).map(|index| (index, caller)).collect();
        assert_eq!(calls.into_inner().unwrap(), expected);
    });
}

#[test]
fn panic_in_the_body_reaches_the_caller_and_leaves_the_system_usable() {
    with_deadline(|| {
        let system = JobSystem::new(4, "par");
        let visits = counters(1000);

        let outcome = catch_unwind(AssertUnwindSafe(|| {
            system.par_for(visits.len(), 10, |index| {
                visits[index].fetch_add(1, Ordering::Relaxed);
                if index == 500 {
                    panic!("index {index} failed");
                }
            });
        }));

        let payload = outcome.expect_err("the panic of the body must reach the caller");
        assert_eq!(
            payload.downcast_ref::<String>().map(String::as_str),
            Some("index 500 failed")
        );
        assert_eq!(visits[500].load(Ordering::Relaxed), 1);
        assert!(
            visits
                .iter()
                .all(|visit| visit.load(Ordering::Relaxed) <= 1)
        );

        // Neither the loop nor the workers are damaged.
        let visits = counters(1000);
        system.par_for(visits.len(), 10, |index| {
            visits[index].fetch_add(1, Ordering::Relaxed);
        });
        assert!(
            visits
                .iter()
                .all(|visit| visit.load(Ordering::Relaxed) == 1)
        );
        assert_eq!(system.spawn(NORMAL, |_| 1).wait(), Ok(1));
    });
}

#[test]
fn panic_in_an_inline_loop_reaches_the_caller_too() {
    with_deadline(|| {
        let system = JobSystem::new(1, "par");
        let outcome = catch_unwind(AssertUnwindSafe(|| {
            system.par_for(10, 1, |index| {
                assert!(index != 3, "inline index {index} failed")
            });
        }));

        let payload = outcome.expect_err("the panic of the body must reach the caller");
        assert_eq!(
            payload.downcast_ref::<String>().map(String::as_str),
            Some("inline index 3 failed")
        );
    });
}

#[test]
fn works_inside_jobs_while_every_worker_is_busy() {
    with_deadline(|| {
        const WORKERS: usize = 3;
        let system = JobSystem::new(WORKERS, "nested");
        // All workers are inside a job at the same time, and each job runs a loop of its own.
        let all_busy = Arc::new(Barrier::new(WORKERS));
        let handles: Vec<_> = (0..WORKERS as u64)
            .map(|job| {
                let inner = system.spawner();
                let all_busy = Arc::clone(&all_busy);
                system.spawn(NORMAL, move |_| {
                    all_busy.wait();
                    let sum = AtomicU64::new(0);
                    inner.par_for(1000, 10, |index| {
                        sum.fetch_add(index as u64 * (job + 1), Ordering::Relaxed);
                    });
                    sum.into_inner()
                })
            })
            .collect();

        for (job, handle) in (0..).zip(handles) {
            assert_eq!(handle.wait(), Ok(499_500 * (job + 1)));
        }
    });
}
