//! The priority queue shared by all workers of a job system, and the worker loop.
//!
//! One mutex guards the queue, the bookkeeping of running jobs and the counters; one condition
//! variable wakes idle workers. Jobs are expected to take at least 100 microseconds, so a single
//! lock is not a bottleneck and keeps every state transition easy to reason about:
//!
//! * A job is *queued* while it is in `pending`. Whoever removes it under the lock decides its
//!   fate: a worker starts it, a canceller or the shutdown reports it as cancelled. This is why
//!   a job cancelled before it starts can never run.
//! * Counters and job handles change inside the same critical section, so a snapshot of the
//!   statistics always agrees with what the handles report.
//! * User code never runs while the lock is held. Closures, their destructors and the
//!   destructors of results run after the lock is released, which lets a job use the system it
//!   runs on freely: spawn, cancel, shut it down, even drop the last `JobSystem`.
//! * The queue outlives its shutdown for as long as anybody refers to it. A job submitted to a
//!   queue that has shut down is reported as cancelled at once, so nothing is ever queued where
//!   no worker would take it.
//!
//! # The bulk limit
//!
//! At most `bulk_limit` jobs of the bulk classes ([`JobClass::is_bulk`]) run at once. The rule
//! needs no search: the bulk classes sort after all others, so a free worker takes the first
//! queued job, or nothing if that is a bulk job while the limit is reached; every job behind
//! it is then a bulk job as well.
//!
//! A queued job wakes one idle worker if it may start at once. A bulk job that the limit holds
//! back wakes nobody: it can only start after a running bulk job has ended, and the worker that
//! ran that job looks for its next job right afterwards and takes it. No job that may start
//! is therefore ever left in the queue while every worker sleeps: a worker goes to sleep only
//! when the first queued job may not start, and every job that may start was either announced
//! to a sleeping worker or is found by a worker that is awake.

use std::collections::BTreeMap;
use std::mem;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use parking_lot::{Condvar, Mutex};

use crate::cancel::CancelToken;
use crate::contain::drop_contained;
use crate::priority::{JobClass, Priority};
use crate::serving::{self, QueueId};
use crate::stats::JobStats;
use crate::task::{Outcome, Task};

/// Position of a job in the queue: priority first, then order of submission.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct QueueKey {
    priority: Priority,
    /// Unique per job and ascending in the order of the `spawn` calls.
    sequence: u64,
}

/// A job together with its token.
struct Entry {
    token: CancelToken,
    task: Box<dyn Task>,
}

/// A job that a worker has taken from the queue and is about to run.
struct Started {
    class: JobClass,
    entry: Entry,
}

pub(crate) struct JobQueue {
    /// Names this queue in the marks of the threads that run its jobs.
    id: QueueId,
    /// Most jobs of the bulk classes that run at once; at least 1.
    bulk_limit: usize,
    state: Mutex<State>,
    /// Signalled once for every queued job that may start at once (not for bulk jobs that the
    /// limit holds back), and for all workers at shutdown.
    work_available: Condvar,
    /// Source of [`QueueKey::sequence`]. Taken before the lock: two `spawn` calls that overlap
    /// in time have no defined order anyway, and calls that follow each other get ascending
    /// numbers.
    next_sequence: AtomicU64,
}

struct State {
    /// Jobs that have not started, in the order in which they will start.
    pending: BTreeMap<QueueKey, Entry>,
    /// Number of pending jobs per class, indexed by `JobClass::index`.
    pending_by_class: [usize; JobClass::COUNT],
    /// Per worker, the token of the job it is running.
    running: Vec<Option<CancelToken>>,
    /// Number of running jobs of a bulk class; at most `JobQueue::bulk_limit`.
    running_bulk: usize,
    submitted: u64,
    completed: u64,
    cancelled: u64,
    panicked: u64,
    /// Set once by `shut_down`; from then on nothing is queued and idle workers return.
    shutting_down: bool,
}

impl JobQueue {
    /// A queue for `workers` workers, which call [`work`](Self::work) with `0..workers`, and
    /// of which at most `bulk_limit` run bulk jobs at once. A limit of 0 counts as 1, one of
    /// `workers` or more has no effect.
    pub(crate) fn new(workers: usize, bulk_limit: usize) -> Self {
        Self {
            id: QueueId::unique(),
            bulk_limit: bulk_limit.max(1),
            state: Mutex::new(State {
                pending: BTreeMap::new(),
                pending_by_class: [0; JobClass::COUNT],
                running: vec![None; workers],
                running_bulk: 0,
                submitted: 0,
                completed: 0,
                cancelled: 0,
                panicked: 0,
                shutting_down: false,
            }),
            work_available: Condvar::new(),
            next_sequence: AtomicU64::new(0),
        }
    }

    /// Queues `task` and returns the token of the new job.
    ///
    /// After [`shut_down`](Self::shut_down) the job is not queued: it is reported as cancelled
    /// and its closure is dropped on the calling thread before `submit` returns.
    pub(crate) fn submit(self: &Arc<Self>, priority: Priority, task: Box<dyn Task>) -> CancelToken {
        /// What became of the job.
        enum Fate {
            /// Queued, and it may start at once.
            Startable,
            /// Queued behind the bulk limit.
            HeldBack,
            /// Not queued, because the queue has shut down.
            Rejected(Entry),
        }

        let sequence = self.next_sequence.fetch_add(1, Ordering::Relaxed);
        let key = QueueKey { priority, sequence };
        let token = CancelToken::for_job(Arc::downgrade(self), key);
        let mut entry = Entry {
            token: token.clone(),
            task,
        };

        let fate = {
            let mut state = self.state.lock();
            state.submitted += 1;
            if state.shutting_down {
                // Reached by spawning after `JobSystem::shutdown`, or through a `JobSpawner`
                // whose system is gone: typically a running job that queues its follow-up
                // while the system shuts down. No worker would ever take the job.
                state.cancelled += 1;
                entry.token.flag();
                entry.task.cancel();
                Fate::Rejected(entry)
            } else {
                state.pending_by_class[priority.class.index()] += 1;
                let previous = state.pending.insert(key, entry);
                debug_assert!(previous.is_none(), "sequence numbers are unique");
                if priority.class.is_bulk() && state.running_bulk >= self.bulk_limit {
                    Fate::HeldBack
                } else {
                    Fate::Startable
                }
            }
        };
        match fate {
            Fate::Startable => {
                self.work_available.notify_one();
            }
            // The job can start only after a running bulk job has ended, and the worker of
            // that job looks for its next job by itself; see the module documentation.
            Fate::HeldBack => {}
            Fate::Rejected(entry) => drop_contained(entry, "a job submitted during shutdown"),
        }
        token
    }

    /// Cancels the job `key` if it has not started: removes it from the queue, reports it as
    /// cancelled and drops its closure on the calling thread. Does nothing otherwise.
    pub(crate) fn cancel(&self, key: QueueKey) {
        let removed = {
            let mut state = self.state.lock();
            let mut removed = state.pending.remove(&key);
            if let Some(entry) = &mut removed {
                state.pending_by_class[key.priority.class.index()] -= 1;
                state.cancelled += 1;
                entry.task.cancel();
            }
            removed
        };
        if let Some(entry) = removed {
            drop_contained(entry, "a cancelled job");
        }
    }

    /// Stops the queue: cancels every queued job, flags the tokens of the running ones and
    /// lets each worker return from [`work`](Self::work) once its current job is done. Jobs
    /// submitted from then on are cancelled instead of queued.
    ///
    /// Returns the number of queued jobs it cancelled, whose closures are dropped on the
    /// calling thread. Only the first call does anything; every later one returns `None`.
    pub(crate) fn shut_down(&self) -> Option<usize> {
        let cancelled = {
            let mut state = self.state.lock();
            if state.shutting_down {
                return None;
            }
            state.shutting_down = true;
            for token in state.running.iter().flatten() {
                token.flag();
            }
            let mut pending = mem::take(&mut state.pending);
            state.pending_by_class = [0; JobClass::COUNT];
            state.cancelled += pending.len() as u64;
            for entry in pending.values_mut() {
                entry.token.flag();
                entry.task.cancel();
            }
            pending
        };
        self.work_available.notify_all();

        let count = cancelled.len();
        for entry in cancelled.into_values() {
            drop_contained(entry, "a job cancelled by shutdown");
        }
        Some(count)
    }

    /// Whether the calling thread is inside a job of this queue: one of its workers, or a
    /// helper thread of a `par_for` loop that one of its jobs runs.
    ///
    /// Such a thread must not wait for the jobs of this queue to end, because the job it is in
    /// is one of them.
    pub(crate) fn serves_current_thread(&self) -> bool {
        serving::served() == Some(self.id)
    }

    /// The loop of worker number `worker`: runs queued jobs in queue order until shutdown.
    ///
    /// Needs a thread of its own, which it marks as a worker of this queue for good.
    pub(crate) fn work(&self, worker: usize) {
        serving::serve(Some(self.id));
        while let Some(job) = self.next_job(worker) {
            self.run(worker, job);
        }
    }

    /// Blocks until a job may start and marks it as running on `worker`. `None` at shutdown.
    fn next_job(&self, worker: usize) -> Option<Started> {
        let mut state = self.state.lock();
        loop {
            if let Some(job) = state.start_next(worker, self.bulk_limit) {
                return Some(job);
            }
            if state.shutting_down {
                return None;
            }
            self.work_available.wait(&mut state);
        }
    }

    /// Runs `job`, which [`next_job`](Self::next_job) gave to `worker`, and books its end.
    ///
    /// The caller looks for its next job right afterwards: if `job` is a bulk job, the slot
    /// it frees under the bulk limit is taken that way, without waking anybody. What is left
    /// of `job` (an unclaimed result, for example) is dropped first, so a slow destructor
    /// delays the next bulk job that waits for the slot by as long as it takes.
    fn run(&self, worker: usize, job: Started) {
        let Started { class, mut entry } = job;
        let outcome = entry.task.run(&entry.token);
        {
            let mut state = self.state.lock();
            state.end(worker, class, outcome);
            entry.task.publish();
        }
        // If nobody holds the handle any more, this drops the result of the job.
        drop_contained(entry, "a finished job");
    }

    /// Most jobs of the bulk classes that run at once; at least 1.
    pub(crate) fn bulk_limit(&self) -> usize {
        self.bulk_limit
    }

    /// Number of jobs that have not started.
    pub(crate) fn queued(&self) -> usize {
        self.state.lock().pending.len()
    }

    /// A consistent snapshot of all counters.
    pub(crate) fn stats(&self) -> JobStats {
        let state = self.state.lock();
        JobStats {
            queued: state.pending.len(),
            queued_by_class: state.pending_by_class,
            running: state.running.iter().flatten().count(),
            submitted: state.submitted,
            completed: state.completed,
            cancelled: state.cancelled,
            panicked: state.panicked,
        }
    }
}

impl State {
    /// Takes the job that may start next and marks it as running on `worker`.
    ///
    /// That is the first job in queue order, unless it is a bulk job while `bulk_limit` bulk
    /// jobs run. Then no job may start: the bulk classes sort after all others
    /// ([`JobClass::is_bulk`]), so every job behind it is a bulk job as well.
    fn start_next(&mut self, worker: usize, bulk_limit: usize) -> Option<Started> {
        let first = self.pending.first_entry()?;
        let class = first.key().priority.class;
        let bulk = class.is_bulk();
        if bulk && self.running_bulk >= bulk_limit {
            return None;
        }
        let entry = first.remove();
        self.pending_by_class[class.index()] -= 1;
        if bulk {
            self.running_bulk += 1;
        }
        self.running[worker] = Some(entry.token.clone());
        Some(Started { class, entry })
    }

    /// Books the end of the job of `class` that ran on `worker`.
    fn end(&mut self, worker: usize, class: JobClass, outcome: Outcome) {
        self.running[worker] = None;
        if class.is_bulk() {
            self.running_bulk -= 1;
        }
        match outcome {
            Outcome::Completed => self.completed += 1,
            Outcome::Panicked => self.panicked += 1,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::thread;

    use super::*;
    use crate::handle::{JobError, Slot};
    use crate::task::{DetachedTask, HandleTask};

    const PRIORITY: Priority = Priority::new(JobClass::Streaming, 0);

    fn key(class: JobClass, key: u32, sequence: u64) -> QueueKey {
        QueueKey {
            priority: Priority::new(class, key),
            sequence,
        }
    }

    /// Queues a job that returns `value` and hands back its token and result slot.
    fn submit_value(queue: &Arc<JobQueue>, value: u32) -> (CancelToken, Arc<Slot<u32>>) {
        submit_with(queue, PRIORITY, value)
    }

    /// [`submit_value`] with a priority of the caller's choice.
    fn submit_with(
        queue: &Arc<JobQueue>,
        priority: Priority,
        value: u32,
    ) -> (CancelToken, Arc<Slot<u32>>) {
        let slot = Arc::new(Slot::new());
        let task = HandleTask::new(move |_: &CancelToken| value, Arc::clone(&slot));
        (queue.submit(priority, Box::new(task)), slot)
    }

    /// What worker `worker` would start now, taken the way a worker takes it, without blocking.
    fn start(queue: &JobQueue, worker: usize) -> Option<Started> {
        queue.state.lock().start_next(worker, queue.bulk_limit)
    }

    /// Starts the next job on `worker`, which must find one, runs it on this thread and books
    /// its end; returns the class of the job.
    fn start_and_run(queue: &JobQueue, worker: usize) -> JobClass {
        let job = start(queue, worker).expect("a job may start");
        let class = job.class;
        queue.run(worker, job);
        class
    }

    /// `(queued per class, running, running bulk jobs)` of `queue`.
    fn occupancy(queue: &JobQueue) -> ([usize; JobClass::COUNT], usize, usize) {
        let stats = queue.stats();
        let running_bulk = queue.state.lock().running_bulk;
        (stats.queued_by_class, stats.running, running_bulk)
    }

    #[test]
    fn bulk_limit_holds_back_bulk_jobs_and_nothing_else() {
        use JobClass::{Background, Critical, Interactive, Streaming};
        // Four workers, two of which may run bulk jobs at once.
        let queue = Arc::new(JobQueue::new(4, 2));
        let (_, near) = submit_with(&queue, Priority::new(Streaming, 0), 10);
        let (_, far) = submit_with(&queue, Priority::new(Streaming, 7), 11);
        let (_, idle) = submit_with(&queue, Priority::new(Background, 0), 12);
        let (_, idler) = submit_with(&queue, Priority::new(Background, 3), 13);

        // Both classes share the limit: two bulk jobs start, in queue order, then none.
        let first = start(&queue, 0).expect("the limit is not reached");
        let second = start(&queue, 1).expect("the limit is not reached");
        assert_eq!((first.class, second.class), (Streaming, Streaming));
        assert!(start(&queue, 2).is_none(), "a third bulk job started");
        assert!(start(&queue, 3).is_none(), "a third bulk job started");
        assert_eq!(occupancy(&queue), ([0, 0, 0, 2], 2, 2));

        // Urgent jobs are never held back: they start on the two idle workers.
        let (_, critical) = submit_with(&queue, Priority::new(Critical, 0), 1);
        let (_, interactive) = submit_with(&queue, Priority::new(Interactive, 0), 2);
        let urgent = start(&queue, 2).expect("an urgent job always starts");
        assert_eq!(start_and_run(&queue, 3), Interactive);
        assert_eq!(urgent.class, Critical);
        queue.run(2, urgent);
        assert_eq!(critical.try_take(), Some(Ok(1)));
        assert_eq!(interactive.try_take(), Some(Ok(2)));
        // Their end frees no bulk slot.
        assert!(start(&queue, 2).is_none());
        assert_eq!(occupancy(&queue), ([0, 0, 0, 2], 2, 2));

        // The end of a bulk job frees one slot, which goes to the next bulk job in queue order.
        queue.run(0, first);
        assert_eq!(near.try_take(), Some(Ok(10)));
        assert_eq!(start_and_run(&queue, 0), Background);
        assert_eq!(idle.try_take(), Some(Ok(12)));
        assert_eq!(occupancy(&queue), ([0, 0, 0, 1], 1, 1));
        assert_eq!(start_and_run(&queue, 2), Background);
        assert_eq!(idler.try_take(), Some(Ok(13)));
        queue.run(1, second);
        assert_eq!(far.try_take(), Some(Ok(11)));

        assert!(start(&queue, 0).is_none(), "the queue is empty");
        assert_eq!(occupancy(&queue), ([0; JobClass::COUNT], 0, 0));
        let stats = queue.stats();
        assert_eq!((stats.submitted, stats.completed), (6, 6));
    }

    #[test]
    fn urgent_job_queued_behind_held_back_bulk_jobs_starts_first() {
        use JobClass::{Background, Interactive, Streaming};
        let queue = Arc::new(JobQueue::new(2, 1));
        assert!(start(&queue, 0).is_none(), "nothing is queued yet");
        submit_with(&queue, Priority::new(Streaming, 0), 1);
        submit_with(&queue, Priority::new(Background, 0), 2);
        let bulk = start(&queue, 0).expect("the limit is not reached");
        assert!(start(&queue, 1).is_none());

        // Queued after the bulk jobs, ahead of them in queue order.
        let (_, urgent) = submit_with(&queue, Priority::new(Interactive, 9), 3);
        assert_eq!(start_and_run(&queue, 1), Interactive);
        assert_eq!(urgent.try_take(), Some(Ok(3)));
        assert!(start(&queue, 1).is_none());

        queue.run(0, bulk);
        assert_eq!(start_and_run(&queue, 1), Background);
        assert_eq!(occupancy(&queue), ([0; JobClass::COUNT], 0, 0));
    }

    #[test]
    fn queue_with_the_limit_at_its_worker_count_starts_bulk_jobs_on_every_worker() {
        let queue = Arc::new(JobQueue::new(3, 3));
        for value in 0..4 {
            submit_value(&queue, value);
        }
        let started: Vec<Started> = (0..3)
            .map(|worker| start(&queue, worker).expect("every worker may run a bulk job"))
            .collect();
        assert_eq!(occupancy(&queue), ([0, 0, 1, 0], 3, 3));
        for (worker, job) in started.into_iter().enumerate() {
            queue.run(worker, job);
        }
        assert_eq!(start_and_run(&queue, 0), JobClass::Streaming);
        assert_eq!(queue.stats().completed, 4);
    }

    #[test]
    fn bulk_limit_is_at_least_one() {
        let queue = Arc::new(JobQueue::new(2, 0));
        assert_eq!(queue.bulk_limit(), 1);
        let (_, slot) = submit_value(&queue, 5);
        assert_eq!(start_and_run(&queue, 1), JobClass::Streaming);
        assert_eq!(slot.try_take(), Some(Ok(5)));
    }

    #[test]
    fn held_back_bulk_jobs_can_be_cancelled_and_are_cancelled_by_the_shutdown() {
        let queue = Arc::new(JobQueue::new(3, 1));
        let (_, running_slot) = submit_value(&queue, 1);
        let running = start(&queue, 0).expect("the limit is not reached");
        let (held_token, held_slot) = submit_value(&queue, 2);
        let (_, last_slot) = submit_value(&queue, 3);
        assert!(start(&queue, 1).is_none());

        held_token.cancel();
        assert_eq!(held_slot.try_take(), Some(Err(JobError::Cancelled)));
        assert_eq!(occupancy(&queue), ([0, 0, 1, 0], 1, 1));

        // The shutdown cancels what the limit holds back; the running job may still end.
        assert_eq!(queue.shut_down(), Some(1));
        assert_eq!(last_slot.try_take(), Some(Err(JobError::Cancelled)));
        assert!(start(&queue, 1).is_none());
        queue.run(0, running);
        assert_eq!(running_slot.try_take(), Some(Ok(1)));
        assert_eq!(occupancy(&queue), ([0; JobClass::COUNT], 0, 0));
        let stats = queue.stats();
        assert_eq!(
            (stats.submitted, stats.completed, stats.cancelled),
            (3, 1, 2)
        );
    }

    #[test]
    fn workers_behind_the_limit_finish_the_bulk_work_and_return_at_shutdown() {
        // Real worker threads: two may run bulk jobs, so the third sleeps behind the limit for
        // most of the time. Every job must still run, and every worker must return.
        const JOBS: u32 = 300;
        let queue = Arc::new(JobQueue::new(3, 2));
        let slots: Vec<_> = (0..JOBS)
            .map(|value| {
                let class = if value.is_multiple_of(3) {
                    JobClass::Background
                } else {
                    JobClass::Streaming
                };
                submit_with(&queue, Priority::new(class, value), value).1
            })
            .collect();
        let workers: Vec<_> = (0..3)
            .map(|worker| {
                let queue = Arc::clone(&queue);
                thread::spawn(move || queue.work(worker))
            })
            .collect();

        for (value, slot) in (0..).zip(&slots) {
            assert_eq!(slot.wait(), Ok(value));
        }
        assert_eq!(queue.shut_down(), Some(0));
        for worker in workers {
            worker.join().unwrap();
        }
        assert_eq!(occupancy(&queue), ([0; JobClass::COUNT], 0, 0));
        assert_eq!(queue.stats().completed, u64::from(JOBS));
    }

    #[test]
    fn queue_keys_sort_by_class_then_key_then_sequence() {
        let ordered = [
            key(JobClass::Critical, 9, 7),
            key(JobClass::Interactive, 0, 3),
            key(JobClass::Interactive, 0, 5),
            key(JobClass::Interactive, 1, 1),
            key(JobClass::Streaming, 0, 9),
            key(JobClass::Background, 0, 0),
            key(JobClass::Background, 0, 2),
            key(JobClass::Background, u32::MAX, 4),
        ];
        assert!(ordered.is_sorted());
        assert!(ordered.windows(2).all(|pair| pair[0] != pair[1]));
    }

    #[test]
    fn job_submitted_during_shutdown_is_cancelled_instead_of_queued() {
        let queue = Arc::new(JobQueue::new(1, 1));
        assert_eq!(queue.shut_down(), Some(0));

        let (token, slot) = submit_value(&queue, 1);

        assert!(token.is_cancelled());
        assert_eq!(slot.try_take(), Some(Err(JobError::Cancelled)));
        let stats = queue.stats();
        assert_eq!((stats.submitted, stats.cancelled, stats.queued), (1, 1, 0));
    }

    #[test]
    fn closure_of_a_job_submitted_during_shutdown_is_dropped_by_submit() {
        let queue = Arc::new(JobQueue::new(1, 1));
        assert_eq!(queue.shut_down(), Some(0));
        let captured = Arc::new(());

        let detached = {
            let captured = Arc::clone(&captured);
            DetachedTask::new(move |_: &CancelToken| drop(captured))
        };
        let token = queue.submit(PRIORITY, Box::new(detached));

        assert!(token.is_cancelled());
        assert_eq!(Arc::strong_count(&captured), 1);
        let stats = queue.stats();
        assert_eq!((stats.submitted, stats.cancelled, stats.queued), (1, 1, 0));
    }

    #[test]
    fn only_the_first_shutdown_does_anything() {
        let queue = Arc::new(JobQueue::new(1, 1));
        let (_token, slot) = submit_value(&queue, 1);

        assert_eq!(queue.shut_down(), Some(1));
        assert_eq!(queue.shut_down(), None);
        assert_eq!(queue.shut_down(), None);

        assert_eq!(slot.try_take(), Some(Err(JobError::Cancelled)));
        let stats = queue.stats();
        assert_eq!((stats.submitted, stats.cancelled, stats.queued), (1, 1, 0));
    }

    #[test]
    fn shutdown_cancels_everything_that_is_queued() {
        let queue = Arc::new(JobQueue::new(1, 1));
        let (first_token, first_slot) = submit_value(&queue, 1);
        let (second_token, second_slot) = submit_value(&queue, 2);
        queue.submit(PRIORITY, Box::new(DetachedTask::new(|_: &CancelToken| {})));
        assert_eq!(queue.queued(), 3);
        assert_eq!(queue.stats().queued_by_class, [0, 0, 3, 0]);

        assert_eq!(queue.shut_down(), Some(3));

        assert!(first_token.is_cancelled() && second_token.is_cancelled());
        assert_eq!(first_slot.try_take(), Some(Err(JobError::Cancelled)));
        assert_eq!(second_slot.try_take(), Some(Err(JobError::Cancelled)));
        let stats = queue.stats();
        assert_eq!((stats.submitted, stats.cancelled, stats.queued), (3, 3, 0));
        assert_eq!(stats.queued_by_class, [0; JobClass::COUNT]);
    }

    #[test]
    fn cancel_removes_only_the_addressed_job() {
        let queue = Arc::new(JobQueue::new(1, 1));
        let (first_token, first_slot) = submit_value(&queue, 1);
        let (_second_token, second_slot) = submit_value(&queue, 2);

        first_token.cancel();
        first_token.cancel();

        assert_eq!(queue.queued(), 1);
        assert_eq!(queue.stats().cancelled, 1);
        assert_eq!(first_slot.try_take(), Some(Err(JobError::Cancelled)));
        assert_eq!(second_slot.try_take(), None);
        assert_eq!(queue.shut_down(), Some(1));
    }

    #[test]
    fn worker_drains_the_queue_and_returns_at_shutdown() {
        let queue = Arc::new(JobQueue::new(1, 1));
        let slots: Vec<_> = (0..50).map(|value| submit_value(&queue, value).1).collect();
        let worker = {
            let queue = Arc::clone(&queue);
            thread::spawn(move || queue.work(0))
        };

        // Equal priorities run first in, first out: the last job finishes after all others.
        let (last, earlier) = slots.split_last().unwrap();
        assert_eq!(last.wait(), Ok(49));
        let stats = queue.stats();
        assert_eq!(
            (
                stats.submitted,
                stats.completed,
                stats.queued,
                stats.running
            ),
            (50, 50, 0, 0)
        );
        for (value, slot) in (0..).zip(earlier) {
            assert_eq!(slot.try_take(), Some(Ok(value)));
        }

        assert_eq!(queue.shut_down(), Some(0));
        worker.join().unwrap();
    }

    #[test]
    fn queue_serves_its_own_worker_threads_only() {
        let queue = Arc::new(JobQueue::new(1, 1));
        let other = Arc::new(JobQueue::new(1, 1));
        // Asked from inside a job: does the thread that runs it serve `queue`, does it serve
        // `other`?
        let slot = Arc::new(Slot::new());
        let task = {
            let queue = Arc::clone(&queue);
            let other = Arc::clone(&other);
            HandleTask::new(
                move |_: &CancelToken| {
                    (queue.serves_current_thread(), other.serves_current_thread())
                },
                Arc::clone(&slot),
            )
        };
        queue.submit(PRIORITY, Box::new(task));
        let worker = {
            let queue = Arc::clone(&queue);
            thread::spawn(move || queue.work(0))
        };

        assert_eq!(slot.wait(), Ok((true, false)));
        assert!(!queue.serves_current_thread());

        assert_eq!(queue.shut_down(), Some(0));
        worker.join().unwrap();
    }
}
