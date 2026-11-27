//! The non-owning handle to a job system, and what all handles of one system share.

use std::fmt;
use std::sync::Arc;

use crate::cancel::CancelToken;
use crate::handle::{JobHandle, Slot};
use crate::par_for::par_for;
use crate::priority::Priority;
use crate::queue::JobQueue;
use crate::stats::JobStats;
use crate::task::{DetachedTask, HandleTask};

/// Uses a [`JobSystem`](crate::JobSystem) without owning it: spawns jobs, runs loops and reads
/// the counters, but never keeps the system alive.
///
/// Made by [`JobSystem::spawner`](crate::JobSystem::spawner). Cloning is cheap; the system and
/// all of its spawners share one queue and one set of worker threads.
///
/// # The handle for job closures
/// A job that queues follow-up jobs or calls `par_for` needs a handle to the system it runs
/// on. Let it capture a `JobSpawner`, not a clone of the [`JobSystem`](crate::JobSystem). A
/// clone is an owner: while a queued or running job holds one, dropping the system elsewhere
/// shuts nothing down. Spawners do not count. Dropping the last `JobSystem` cancels the queued
/// jobs and joins the workers, however many jobs hold a spawner.
///
/// # After the shutdown
/// A spawner may outlive its system, and it stays usable while the system shuts down:
///
/// * [`spawn`](Self::spawn) returns a handle that is already finished with
///   [`JobError::Cancelled`](crate::JobError::Cancelled), and
///   [`spawn_detached`](Self::spawn_detached) queues nothing. In both cases the closure never
///   runs and is dropped before the call returns.
/// * [`par_for`](Self::par_for) still runs the whole loop; it never used the workers.
/// * The counters stay readable.
///
/// A job that is running when the shutdown begins can therefore finish as usual. The follow-up
/// jobs it spawns are cancelled, and its own [`CancelToken`] tells it why.
#[derive(Clone)]
pub struct JobSpawner {
    shared: Arc<Shared>,
}

/// What the owning handles and the spawners of one system have in common.
struct Shared {
    /// Name of the system; the prefix of the names of its threads.
    name: Box<str>,
    /// Number of worker threads the system was started with.
    threads: usize,
    queue: Arc<JobQueue>,
}

impl JobSpawner {
    /// The spawner of the system `name`, whose `threads` worker threads serve `queue`.
    pub(crate) fn new(name: &str, threads: usize, queue: Arc<JobQueue>) -> Self {
        Self {
            shared: Arc::new(Shared {
                name: name.into(),
                threads,
                queue,
            }),
        }
    }

    /// Queues `work` and returns the handle for its result. Works like
    /// [`JobSystem::spawn`](crate::JobSystem::spawn).
    ///
    /// If the system has shut down, the job is cancelled on the spot: the returned handle is
    /// finished with [`JobError::Cancelled`](crate::JobError::Cancelled) and `work` is dropped
    /// without having run.
    #[must_use = "dropping the handle discards the result; use `spawn_detached` for such jobs"]
    pub fn spawn<T: Send + 'static>(
        &self,
        priority: Priority,
        work: impl FnOnce(&CancelToken) -> T + Send + 'static,
    ) -> JobHandle<T> {
        let slot = Arc::new(Slot::new());
        let task = HandleTask::new(work, Arc::clone(&slot));
        let token = self.shared.queue.submit(priority, Box::new(task));
        JobHandle::new(slot, token)
    }

    /// Queues `work` without a handle. Works like
    /// [`JobSystem::spawn_detached`](crate::JobSystem::spawn_detached).
    ///
    /// If the system has shut down, `work` is dropped without having run.
    pub fn spawn_detached(
        &self,
        priority: Priority,
        work: impl FnOnce(&CancelToken) + Send + 'static,
    ) {
        self.shared
            .queue
            .submit(priority, Box::new(DetachedTask::new(work)));
    }

    /// Calls `f(index)` exactly once for every index in `0..count` and returns when all calls
    /// have returned. Works like [`JobSystem::par_for`](crate::JobSystem::par_for), also after
    /// the system has shut down.
    ///
    /// # Panics
    /// If `f` panics, the first panic continues on the calling thread once the loop has ended.
    pub fn par_for(&self, count: usize, grain: usize, f: impl Fn(usize) + Sync) {
        par_for(&self.shared.name, self.shared.threads, count, grain, &f);
    }

    /// Number of worker threads the system was started with; at least 1. Does not change when
    /// the system shuts down.
    #[must_use]
    pub fn threads(&self) -> usize {
        self.shared.threads
    }

    /// Most jobs of the bulk classes ([`JobClass::is_bulk`](crate::JobClass::is_bulk)) that
    /// run at the same time; between 1 and [`threads`](Self::threads), equal to `threads` when
    /// the system has no bulk limit. See
    /// [`JobSystem::bulk_limit`](crate::JobSystem::bulk_limit).
    #[must_use]
    pub fn bulk_limit(&self) -> usize {
        self.shared.queue.bulk_limit().min(self.shared.threads)
    }

    /// Number of jobs that wait in the queue. Jobs that are running do not count.
    #[must_use]
    pub fn queued(&self) -> usize {
        self.shared.queue.queued()
    }

    /// A consistent snapshot of the counters of the system.
    #[must_use]
    pub fn stats(&self) -> JobStats {
        self.shared.queue.stats()
    }

    /// Name of the system.
    pub(crate) fn name(&self) -> &str {
        &self.shared.name
    }

    /// The queue of the system.
    pub(crate) fn queue(&self) -> &JobQueue {
        &self.shared.queue
    }
}

impl fmt::Debug for JobSpawner {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("JobSpawner")
            .field("name", &self.shared.name)
            .field("threads", &self.shared.threads)
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use std::thread;

    use super::*;
    use crate::JobClass;
    use crate::handle::JobError;

    const PRIORITY: Priority = Priority::new(JobClass::Streaming, 0);

    #[test]
    fn spawner_queues_jobs_for_the_workers_of_its_queue() {
        let queue = Arc::new(JobQueue::new(1, 1));
        let spawner = JobSpawner::new("unit", 1, Arc::clone(&queue));
        spawner.spawn_detached(PRIORITY, |_| {});
        let handle = spawner.spawn(PRIORITY, |_| 5);
        assert_eq!((spawner.queued(), spawner.threads()), (2, 1));
        assert_eq!(spawner.stats().submitted, 2);

        let worker = {
            let queue = Arc::clone(&queue);
            thread::spawn(move || queue.work(0))
        };
        assert_eq!(handle.wait(), Ok(5));
        // First in, first out: the detached job was done before the one with the handle began.
        assert_eq!(spawner.stats().completed, 2);

        assert_eq!(queue.shut_down(), Some(0));
        worker.join().unwrap();
    }

    #[test]
    fn spawner_of_a_queue_that_has_shut_down_spawns_cancelled_jobs() {
        let queue = Arc::new(JobQueue::new(2, 2));
        let spawner = JobSpawner::new("unit", 2, Arc::clone(&queue));
        assert_eq!(queue.shut_down(), Some(0));

        let handle = spawner.clone().spawn(PRIORITY, |_| 5);
        assert!(handle.is_finished() && handle.token().is_cancelled());
        assert_eq!(handle.wait(), Err(JobError::Cancelled));
        spawner.spawn_detached(PRIORITY, |_| unreachable!("the job never runs"));

        let stats = spawner.stats();
        assert_eq!((stats.submitted, stats.cancelled, stats.queued), (2, 2, 0));
        assert_eq!(spawner.threads(), 2);
    }

    #[test]
    fn debug_output_names_the_system() {
        let spawner = JobSpawner::new("debug", 3, Arc::new(JobQueue::new(3, 3)));
        assert_eq!(
            format!("{spawner:?}"),
            "JobSpawner { name: \"debug\", threads: 3, .. }"
        );
    }

    #[test]
    fn bulk_limit_never_exceeds_the_threads_that_were_started() {
        // As after an operating system that started fewer threads than the policy asked for.
        let spawner = JobSpawner::new("short", 2, Arc::new(JobQueue::new(5, 4)));
        assert_eq!((spawner.threads(), spawner.bulk_limit()), (2, 2));
        let spawner = JobSpawner::new("capped", 5, Arc::new(JobQueue::new(5, 4)));
        assert_eq!((spawner.threads(), spawner.bulk_limit()), (5, 4));
    }
}
