//! The job system: the owner of the worker threads.

use std::fmt;
use std::sync::Arc;
use std::thread::{self, JoinHandle};

use log::Level;
use parking_lot::Mutex;

use crate::cancel::CancelToken;
use crate::contain::{drop_contained, log_contained, panic_message};
use crate::handle::JobHandle;
use crate::policy::{CpuCores, WorkerPolicy};
use crate::priority::Priority;
use crate::queue::JobQueue;
use crate::spawner::JobSpawner;
use crate::stats::JobStats;

/// A fixed set of worker threads that run prioritised jobs.
///
/// Cloning is cheap; all clones refer to the same workers and the same queue, and every clone
/// is an owner of the system.
///
/// # Order
/// A free worker always takes the queued job with the smallest [`Priority`]: most urgent
/// [`JobClass`](crate::JobClass) first, then the lowest key, then the earliest `spawn` call.
/// Running jobs are never preempted, and a less urgent class waits for as long as more urgent
/// work keeps arriving.
///
/// # Bulk limit
/// A system made with [`with_policy`](Self::with_policy) or [`standard`](Self::standard) runs
/// at most [`bulk_limit`](Self::bulk_limit) jobs of the bulk classes `Streaming` and
/// `Background` ([`JobClass::is_bulk`](crate::JobClass::is_bulk)) at the same time. While that
/// many run, a free worker takes only `Critical` and `Interactive` jobs, which the limit never
/// holds back, and the queued bulk jobs start one by one, in queue order, as running ones end.
/// [`new`](Self::new) sets no limit.
///
/// A bulk job that waits for another bulk job ([`JobHandle::wait`]) keeps its slot under the
/// limit. If every slot is kept that way, the queued bulk jobs never start, however many
/// workers are idle, until the system shuts down. Bulk work that continues in another job
/// queues that job and returns.
///
/// # Panics in jobs
/// A panic inside a job is caught and logged at error level through the `log` crate. The
/// worker thread survives and keeps serving; the handle of the job reports
/// [`JobError::Panicked`](crate::JobError::Panicked). Destructors that panic while the job
/// system drops a closure or an unclaimed result are caught and logged as well.
///
/// Containment relies on unwinding. Two cases end the process, here as on any other thread: a
/// build with `panic = "abort"`, and a destructor that panics *while* its job is already
/// unwinding from a panic.
///
/// # Shutdown
/// The system shuts down when [`shutdown`](Self::shutdown) is called or when its last clone is
/// dropped, whichever happens first:
///
/// 1. Every queued job is cancelled. It never runs, its handle yields
///    [`JobError::Cancelled`](crate::JobError::Cancelled) and its closure is dropped on the
///    thread that shuts the system down.
/// 2. The tokens of the running jobs are cancelled, so that jobs which poll their token can
///    return early. No job is interrupted.
/// 3. The call, or the drop, blocks until every running job has returned and every worker
///    thread has ended.
///
/// From step 1 on the system accepts no more work: `spawn` returns handles that are already
/// cancelled and `spawn_detached` drops its closure. `par_for` and the counters keep working.
///
/// # Handles inside jobs
/// A job that queues follow-up jobs or calls `par_for` needs a handle to the system it runs
/// on. Give it a [`JobSpawner`] from [`spawner`](Self::spawner): it uses the system without
/// owning it.
///
/// A *clone* captured by a job is an owner like any other, which has two consequences:
///
/// * While such a job is queued or running, dropping the other clones shuts nothing down.
///   Nothing is cancelled and nothing is joined; the drop returns at once.
/// * When the last clone is then dropped by the job, inside the system, step 3 is skipped and
///   a warning is logged. The workers end on their own after their current job and nothing
///   waits for them: a worker that waits for the other workers could deadlock with a job that
///   waits for it.
///
/// [`shutdown`](Self::shutdown) does not depend on who holds clones. Call it where the system
/// has to end, for example before the process exits.
#[derive(Clone)]
pub struct JobSystem {
    owner: Arc<Owner>,
}

/// What all clones of a [`JobSystem`] share. Dropping it shuts the system down.
struct Owner {
    /// The handle through which every clone spawns; [`JobSystem::spawner`] hands out copies.
    spawner: JobSpawner,
    /// The worker threads that no shutdown has joined yet.
    ///
    /// Locked for the whole time a shutdown waits for them, so that every caller of
    /// [`JobSystem::shutdown`] returns only when the threads have ended, and not just the one
    /// that got hold of the join handles.
    workers: Mutex<Vec<JoinHandle<()>>>,
}

impl JobSystem {
    /// Starts `threads` worker threads named `"{name}-{index}"`. `threads == 0` counts as `1`.
    ///
    /// Every worker takes jobs of every class: the system has no bulk limit
    /// ([`WorkerPolicy::uncapped`]). The engine's job system is made by
    /// [`standard`](Self::standard).
    ///
    /// If the operating system refuses to start some of the threads, the system logs an error
    /// and runs with the threads it got; [`threads`](Self::threads) reports the real number.
    ///
    /// # Panics
    /// Panics if not a single worker thread could be started, or if `name` contains a NUL byte.
    #[must_use]
    pub fn new(threads: usize, name: &str) -> Self {
        Self::with_policy(WorkerPolicy::uncapped(threads), name)
    }

    /// The engine's job system (contract §4.2): [`WorkerPolicy::standard`] for the cores of this
    /// machine as [`CpuCores::detect`] finds them, so that `logical cores - 3` workers run and at
    /// most `physical cores - 1` of them run bulk jobs at once. Threads are named
    /// `"{name}-{index}"`.
    ///
    /// The physical cores are estimated on the assumption that logical cores come in SMT pairs;
    /// the environment variable [`CpuCores::PHYSICAL_CORES_VAR`] overrides the estimate, and
    /// [`with_policy`](Self::with_policy) takes any other policy.
    ///
    /// # Panics
    /// Panics if not a single worker thread could be started, or if `name` contains a NUL byte.
    #[must_use]
    pub fn standard(name: &str) -> Self {
        Self::with_policy(WorkerPolicy::standard(CpuCores::detect()), name)
    }

    /// Starts the [`threads`](WorkerPolicy::threads) of `policy`, named `"{name}-{index}"`, at
    /// most [`bulk_limit`](WorkerPolicy::bulk_limit) of which run bulk jobs at once (see
    /// [the type documentation](Self#bulk-limit)).
    ///
    /// If the operating system refuses to start some of the threads, the system logs an error
    /// and runs with the threads it got; [`threads`](Self::threads) reports the real number.
    ///
    /// # Panics
    /// Panics if not a single worker thread could be started, or if `name` contains a NUL byte.
    #[must_use]
    pub fn with_policy(policy: WorkerPolicy, name: &str) -> Self {
        let requested = policy.threads();
        let queue = Arc::new(JobQueue::new(requested, policy.bulk_limit()));
        let mut workers = Vec::with_capacity(requested);
        for index in 0..requested {
            let queue = Arc::clone(&queue);
            let spawned = thread::Builder::new()
                .name(format!("{name}-{index}"))
                .spawn(move || queue.work(index));
            match spawned {
                Ok(worker) => workers.push(worker),
                Err(error) => {
                    log_contained(
                        Level::Error,
                        format_args!(
                            "job system '{name}': cannot start worker {index} of {requested}: \
                             {error}"
                        ),
                    );
                    break;
                }
            }
        }
        assert!(
            !workers.is_empty(),
            "job system '{name}': the operating system did not start any worker thread"
        );
        let spawner = JobSpawner::new(name, workers.len(), queue);
        log_contained(
            Level::Debug,
            format_args!(
                "job system '{name}' runs {} worker threads, at most {} of them on bulk jobs",
                spawner.threads(),
                spawner.bulk_limit()
            ),
        );
        Self {
            owner: Arc::new(Owner {
                spawner,
                workers: Mutex::new(workers),
            }),
        }
    }

    /// Queues `work` and returns the handle for its result.
    ///
    /// The closure runs once on a worker thread, unless the job is cancelled first. It receives
    /// the job's [`CancelToken`]; long jobs should poll it and return early when it is
    /// cancelled. A panic of the closure is contained and reported through the handle.
    ///
    /// Once the system has shut down, the job is cancelled on the spot: the returned handle is
    /// finished with [`JobError::Cancelled`](crate::JobError::Cancelled) and `work` is dropped
    /// without having run.
    ///
    /// A closure that needs the system itself should capture a [`JobSpawner`], not a clone;
    /// see [the type documentation](Self#handles-inside-jobs). Use
    /// [`spawn_detached`](Self::spawn_detached) when nobody needs the result.
    #[must_use = "dropping the handle discards the result; use `spawn_detached` for such jobs"]
    pub fn spawn<T: Send + 'static>(
        &self,
        priority: Priority,
        work: impl FnOnce(&CancelToken) -> T + Send + 'static,
    ) -> JobHandle<T> {
        self.owner.spawner.spawn(priority, work)
    }

    /// Queues `work` without a handle.
    ///
    /// Like [`spawn`](Self::spawn), except that nothing reports how the job ended: a panic is
    /// only logged, and the job can be cancelled by the shutdown of the system only. The token
    /// passed to the closure tells it when that happens. Once the system has shut down, `work`
    /// is dropped without having run.
    pub fn spawn_detached(
        &self,
        priority: Priority,
        work: impl FnOnce(&CancelToken) + Send + 'static,
    ) {
        self.owner.spawner.spawn_detached(priority, work);
    }

    /// Calls `f(index)` exactly once for every index in `0..count` and returns when all calls
    /// have returned. `f` may borrow from the caller's stack.
    ///
    /// The calls are spread over at most [`threads`](Self::threads) threads, one of which is
    /// the calling thread. Each thread claims `grain` consecutive indices at a time (`0` counts
    /// as `1`); choose it so that one chunk is worth tens of microseconds. The order of the
    /// calls is unspecified. If everything fits into one chunk or the system has one thread,
    /// the loop runs on the calling thread alone, in ascending order.
    ///
    /// The loop does not go through the queue and has no priority, and the bulk limit does not
    /// apply to it. Its helpers are scoped threads started for this call, which costs some tens
    /// of microseconds per thread: use `par_for` for loops whose total work is well above that.
    /// In return it is safe to call from inside a job, even when all workers are busy, and it
    /// still runs the whole loop after the system has shut down. A job reaches it through a
    /// [`JobSpawner`].
    ///
    /// # Panics
    /// If `f` panics, the indices not yet claimed are skipped, the other threads finish the
    /// chunk they are in, and the first panic then continues on the calling thread with its
    /// original payload.
    pub fn par_for(&self, count: usize, grain: usize, f: impl Fn(usize) + Sync) {
        self.owner.spawner.par_for(count, grain, f);
    }

    /// A handle that uses this system without owning it: what a job closure should capture in
    /// order to spawn follow-up jobs or to call `par_for`. See [`JobSpawner`].
    #[must_use]
    pub fn spawner(&self) -> JobSpawner {
        self.owner.spawner.clone()
    }

    /// Shuts the system down now, whoever still holds clones or spawners of it.
    ///
    /// Does what dropping the last clone does (steps 1 to 3 of
    /// [the type documentation](Self#shutdown)): cancels every queued job, cancels the tokens
    /// of the running jobs, then blocks until these jobs have returned and all worker threads
    /// have ended. Unlike the drop it works at any time, also while jobs own clones of the
    /// system, and it ends jobs that wait for jobs which can no longer start (see
    /// [`JobHandle::wait`](crate::JobHandle::wait)).
    ///
    /// All handles stay valid afterwards: `spawn` returns handles that are already cancelled,
    /// `par_for` and the counters work as before. Calling `shutdown` again is allowed, from
    /// any thread and at any time. Every call made outside the jobs of this system returns
    /// only when the worker threads have ended, whichever call got there first.
    ///
    /// # Inside a job
    /// A job cannot wait for its own end. When a job of this system calls `shutdown`, or the
    /// body of a `par_for` loop that such a job runs, the call returns right after cancelling:
    /// the running jobs, this one included, finish in their own time and the workers end after
    /// them. A later call from outside the jobs, or the drop of the last clone there, waits
    /// for them as usual.
    ///
    /// # Deadlocks
    /// The call waits for the running jobs without a time limit. Do not make it on a thread
    /// that one of these jobs waits for, nor while holding a lock that one of them needs.
    pub fn shutdown(&self) {
        self.owner.shut_down();
    }

    /// Number of worker threads; at least 1. Does not change when the system shuts down.
    #[must_use]
    pub fn threads(&self) -> usize {
        self.owner.spawner.threads()
    }

    /// Most jobs of the bulk classes ([`JobClass::is_bulk`](crate::JobClass::is_bulk)) that
    /// run at the same time; between 1 and [`threads`](Self::threads). Equal to `threads` when
    /// the system has no bulk limit, as with [`new`](Self::new).
    #[must_use]
    pub fn bulk_limit(&self) -> usize {
        self.owner.spawner.bulk_limit()
    }

    /// Number of jobs that wait in the queue. Jobs that are running do not count.
    #[must_use]
    pub fn queued(&self) -> usize {
        self.owner.spawner.queued()
    }

    /// A consistent snapshot of the counters of this system.
    #[must_use]
    pub fn stats(&self) -> JobStats {
        self.owner.spawner.stats()
    }
}

impl fmt::Debug for JobSystem {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("JobSystem")
            .field("name", &self.owner.spawner.name())
            .field("threads", &self.threads())
            .finish_non_exhaustive()
    }
}

impl Owner {
    /// Stops the queue and, unless the calling thread is inside a job of this system, waits
    /// until every worker thread has ended. Safe to call any number of times.
    fn shut_down(&self) {
        let name = self.spawner.name();
        let queue = self.spawner.queue();
        if let Some(cancelled) = queue.shut_down() {
            log_contained(
                Level::Debug,
                format_args!("job system '{name}' shuts down, {cancelled} queued jobs cancelled"),
            );
        }

        if queue.serves_current_thread() {
            // Requested by a job of this system. Joining would wait for the job we are in, and
            // for jobs that may in turn wait for it. The workers have been told to stop and end
            // on their own; a later shutdown from outside the jobs joins them.
            return;
        }

        let mut panicked = Vec::new();
        {
            let mut workers = self.workers.lock();
            for worker in workers.drain(..) {
                let thread = worker.thread().clone();
                if let Err(payload) = worker.join() {
                    panicked.push((thread, payload));
                }
            }
        }
        // Reported without the lock: the logger and the payloads are foreign code.
        for (thread, payload) in panicked {
            // Unreachable as long as every piece of foreign code is contained.
            log_contained(
                Level::Error,
                format_args!(
                    "job system '{name}': worker thread '{}' ended with a panic: {}",
                    thread.name().unwrap_or("<unnamed>"),
                    panic_message(&*payload)
                ),
            );
            drop_contained(payload, "the payload of a worker panic");
        }
    }
}

impl Drop for Owner {
    fn drop(&mut self) {
        self.shut_down();
        if !self.workers.get_mut().is_empty() {
            // The last clone was dropped by a job of this system, so the join was skipped and
            // the join handles go away with `self`: the threads are detached.
            log_contained(
                Level::Warn,
                format_args!(
                    "job system '{}': the last JobSystem handle was dropped by one of its own \
                     jobs, nothing waits for its worker threads; let jobs capture a JobSpawner \
                     or end the system with JobSystem::shutdown",
                    self.spawner.name()
                ),
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::JobError;

    fn assert_send_sync<T: Send + Sync>() {}

    #[test]
    fn public_types_can_be_shared_between_threads() {
        assert_send_sync::<JobSystem>();
        assert_send_sync::<JobSpawner>();
        assert_send_sync::<CancelToken>();
        assert_send_sync::<JobHandle<Vec<u8>>>();
        assert_send_sync::<JobHandle<std::cell::Cell<u8>>>();
        assert_send_sync::<JobError>();
        assert_send_sync::<Priority>();
        assert_send_sync::<JobStats>();
    }

    #[test]
    fn debug_output_names_the_system() {
        let system = JobSystem::new(2, "debug");
        assert_eq!(
            format!("{system:?}"),
            "JobSystem { name: \"debug\", threads: 2, .. }"
        );
        assert_eq!(
            format!("{:?}", system.spawner()),
            "JobSpawner { name: \"debug\", threads: 2, .. }"
        );
    }

    #[test]
    fn policy_decides_threads_and_bulk_limit() {
        let capped = JobSystem::with_policy(WorkerPolicy::new(3, 2), "policy");
        assert_eq!((capped.threads(), capped.bulk_limit()), (3, 2));
        assert_eq!(capped.spawner().bulk_limit(), 2);
        assert_eq!(capped.owner.workers.lock().len(), 3);

        let uncapped = JobSystem::new(4, "uncapped");
        assert_eq!((uncapped.threads(), uncapped.bulk_limit()), (4, 4));
        assert_eq!(JobSystem::new(0, "zero").bulk_limit(), 1);
    }

    #[test]
    fn shutdown_joins_the_workers_once_and_leaves_none_for_the_drop() {
        let system = JobSystem::new(3, "unit");
        assert_eq!(system.owner.workers.lock().len(), 3);

        system.shutdown();
        assert!(system.owner.workers.lock().is_empty());
        system.clone().shutdown();
        assert!(system.owner.workers.lock().is_empty());
        assert_eq!(system.threads(), 3);
    }
}
