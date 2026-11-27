//! `threads`, `queued` and `stats` report what really happens.

use cp_jobs::{JobClass, JobStats, JobSystem, Priority};

use crate::util::{Gate, NORMAL, assert_balanced, quiesce, with_deadline};

#[test]
fn threads_reports_the_number_of_workers() {
    with_deadline(|| {
        for threads in [1, 2, 5] {
            let system = JobSystem::new(threads, "threads");
            assert_eq!(system.threads(), threads);
            // Returns only if that many workers really exist.
            quiesce(&system);
        }
    });
}

#[test]
fn zero_threads_still_gives_one_worker() {
    with_deadline(|| {
        let system = JobSystem::new(0, "zero");
        assert_eq!(system.threads(), 1);
        assert_eq!(system.spawn(NORMAL, |_| "served").wait(), Ok("served"));
    });
}

#[test]
fn statistics_follow_every_job_through_its_life() {
    with_deadline(|| {
        let system = JobSystem::new(1, "stats");
        assert_eq!(system.stats(), JobStats::default());

        // One job running (the gate), four queued behind it.
        let gate = Gate::close(&system);
        let interactive = system.spawn(Priority::new(JobClass::Interactive, 0), |_| 1);
        let failing = system.spawn(Priority::new(JobClass::Streaming, 0), |_| -> u32 {
            panic!("a job that fails on purpose")
        });
        let unwanted = system.spawn(Priority::new(JobClass::Streaming, 1), |_| 3);
        system.spawn_detached(Priority::new(JobClass::Background, 0), |_| {});

        let stats = system.stats();
        assert_balanced(&stats, 1);
        assert_eq!(
            stats,
            JobStats {
                queued: 4,
                queued_by_class: [0, 1, 2, 1],
                running: 1,
                submitted: 5,
                completed: 0,
                cancelled: 0,
                panicked: 0,
            }
        );
        assert_eq!(system.queued(), 4);

        unwanted.cancel();
        let stats = system.stats();
        assert_balanced(&stats, 1);
        assert_eq!(
            (stats.queued, stats.queued_by_class, stats.cancelled),
            (3, [0, 1, 1, 1], 1)
        );

        gate.open();
        assert_eq!(interactive.wait(), Ok(1));
        assert!(failing.wait().is_err());
        quiesce(&system);

        // Gate, interactive, detached and the job `quiesce` adds have completed.
        assert_eq!(
            system.stats(),
            JobStats {
                queued: 0,
                queued_by_class: [0; JobClass::COUNT],
                running: 0,
                submitted: 6,
                completed: 4,
                cancelled: 1,
                panicked: 1,
            }
        );
    });
}

#[test]
fn handles_and_statistics_agree() {
    with_deadline(|| {
        let system = JobSystem::new(3, "agree");
        for round in 0..2000_u64 {
            let handle = system.spawn(NORMAL, move |_| round);
            assert_eq!(handle.wait(), Ok(round));
            // The job was counted before its handle reported it as finished.
            let stats = system.stats();
            assert_eq!(
                (stats.completed, stats.running, stats.queued),
                (round + 1, 0, 0)
            );
        }
    });
}
