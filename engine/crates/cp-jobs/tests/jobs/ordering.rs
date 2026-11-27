//! Jobs start strictly by class, then by key, then in the order of their submission.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::thread::{self, ThreadId};

use cp_jobs::{JobClass, JobSystem, Priority};
use crossbeam_channel::unbounded;

use crate::util::{Gate, TestRng, quiesce, recv, with_deadline};

/// Queues one job per priority behind a closed gate on a single worker and returns the indices
/// of `priorities` in the order in which the jobs ran.
fn run_order(priorities: &[Priority]) -> Vec<usize> {
    let system = JobSystem::new(1, "order");
    let gate = Gate::close(&system);
    let (ran_tx, ran_rx) = unbounded();
    for (index, &priority) in priorities.iter().enumerate() {
        let ran = ran_tx.clone();
        system.spawn_detached(priority, move |_| {
            ran.send(index).expect("the test collects the order");
        });
    }
    assert_eq!(system.queued(), priorities.len());

    gate.open();
    priorities.iter().map(|_| recv(&ran_rx)).collect()
}

/// The order the contract demands: sorted by priority, ties in submission order.
fn expected_order(priorities: &[Priority]) -> Vec<usize> {
    let mut order: Vec<usize> = (0..priorities.len()).collect();
    // Stable, so equal priorities keep their submission order.
    order.sort_by_key(|&index| priorities[index]);
    order
}

#[test]
fn class_outranks_key() {
    with_deadline(|| {
        let priorities = [
            Priority::new(JobClass::Background, 0),
            Priority::new(JobClass::Streaming, 0),
            Priority::new(JobClass::Interactive, u32::MAX),
            Priority::new(JobClass::Critical, u32::MAX),
            Priority::new(JobClass::Critical, 0),
        ];
        assert_eq!(run_order(&priorities), [4, 3, 2, 1, 0]);
    });
}

#[test]
fn lower_key_runs_first_within_a_class() {
    with_deadline(|| {
        let keys = [5, 1, 4, u32::MAX, 0, 3, 2];
        let priorities = keys.map(|key| Priority::new(JobClass::Streaming, key));
        assert_eq!(run_order(&priorities), [4, 1, 6, 5, 2, 0, 3]);
    });
}

#[test]
fn equal_priorities_run_first_in_first_out() {
    with_deadline(|| {
        let priorities = [Priority::new(JobClass::Interactive, 7); 1000];
        assert_eq!(run_order(&priorities), (0..1000).collect::<Vec<_>>());
    });
}

#[test]
fn mixed_priorities_run_by_class_then_key_then_submission() {
    with_deadline(|| {
        let mut rng = TestRng::new(0x0D15_EA5E);
        // Few distinct keys, so that many jobs tie and the first-in-first-out rule matters.
        let priorities: Vec<Priority> = (0..800).map(|_| rng.priority(5)).collect();
        assert_eq!(run_order(&priorities), expected_order(&priorities));
    });
}

#[test]
fn every_worker_takes_jobs_in_queue_order() {
    with_deadline(|| {
        const JOBS: usize = 3000;
        let system = JobSystem::new(4, "order");
        let gate = Gate::close(&system);
        let mut rng = TestRng::new(42);
        let priorities: Vec<Priority> = (0..JOBS).map(|_| rng.priority(16)).collect();
        let log = Arc::new(Mutex::new(Vec::with_capacity(JOBS)));
        for (index, &priority) in priorities.iter().enumerate() {
            let log = Arc::clone(&log);
            system.spawn_detached(priority, move |_| {
                log.lock().unwrap().push((thread::current().id(), index));
            });
        }

        gate.open();
        quiesce(&system);

        // Jobs are taken from the queue one at a time and in queue order, and every worker
        // runs what it took before it takes the next job. Whatever the interleaving between
        // workers, each worker therefore sees an ascending sequence.
        let log = log.lock().unwrap();
        assert_eq!(log.len(), JOBS);
        let mut last_of_worker: HashMap<ThreadId, (Priority, usize)> = HashMap::new();
        for &(worker, index) in log.iter() {
            let position = (priorities[index], index);
            if let Some(previous) = last_of_worker.insert(worker, position) {
                assert!(
                    previous < position,
                    "{previous:?} ran before {position:?} on {worker:?}"
                );
            }
        }
    });
}

#[test]
fn job_spawned_by_a_running_job_overtakes_queued_lower_priority_work() {
    with_deadline(|| {
        let system = JobSystem::new(1, "order");
        let gate = Gate::close(&system);
        let (ran_tx, ran_rx) = unbounded();

        let spawner = system.spawner();
        let ran = ran_tx.clone();
        system.spawn_detached(Priority::new(JobClass::Interactive, 0), move |_| {
            ran.send("first").unwrap();
            let ran_urgent = ran.clone();
            spawner.spawn_detached(Priority::new(JobClass::Critical, 0), move |_| {
                ran_urgent.send("urgent").unwrap();
            });
        });
        let ran = ran_tx.clone();
        system.spawn_detached(Priority::new(JobClass::Background, 0), move |_| {
            ran.send("background").unwrap();
        });

        gate.open();
        let order = [recv(&ran_rx), recv(&ran_rx), recv(&ran_rx)];
        assert_eq!(order, ["first", "urgent", "background"]);
    });
}
