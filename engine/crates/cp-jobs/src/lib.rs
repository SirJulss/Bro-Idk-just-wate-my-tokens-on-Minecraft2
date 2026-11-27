//! `cp-jobs`: the job system of the Cubeplanet engine.
//!
//! Responsibility: prioritised jobs on worker threads, cancellation and panic containment.
//! It depends on no other engine crate.
//!
//! Binding contract: `engine/docs/ARCHITEKTUR.md`, section 4.2.
//!
//! # Model
//!
//! A [`JobSystem`] owns a fixed number of worker threads and one queue. Every job carries a
//! [`Priority`]; a free worker always takes the queued job of the most urgent [`JobClass`], then
//! the lowest key, then the earliest submission, as far as the bulk limit of the
//! [worker policy](#worker-policy) lets it. Jobs are never preempted, so priorities decide only
//! what starts next.
//!
//! * [`JobSystem::spawn`] returns a [`JobHandle`] that yields the closure's result,
//!   [`JobError::Cancelled`] or [`JobError::Panicked`].
//! * [`JobSystem::spawn_detached`] runs a closure whose result nobody waits for.
//! * [`JobSystem::par_for`] runs a borrowing closure over an index range and blocks until done.
//! * Every job closure receives its [`CancelToken`]. A job that is cancelled before it starts
//!   never runs; a running job is never interrupted and decides itself when to look.
//! * A panic inside a job is caught and logged, and the worker keeps serving.
//! * [`JobSystem::spawner`] returns a [`JobSpawner`], the handle for job closures: it spawns
//!   jobs and runs `par_for` like the system, but does not keep the system alive.
//! * [`JobSystem::shutdown`], or dropping the last clone of the system, cancels all queued
//!   jobs, lets the running ones finish and joins the worker threads. Jobs spawned afterwards
//!   are cancelled at once.
//! * [`JobSystem::stats`] returns a consistent [`JobStats`] snapshot.
//!
//! The system is built for jobs of 100 microseconds and more. It is deliberately simple: one
//! mutex-protected priority queue, one condition variable, no `unsafe`.
//!
//! # Worker policy
//!
//! The engine runs one job system made by [`JobSystem::standard`] (contract §4.2): it starts
//! `logical cores - 3` workers, and at most `physical cores - 1` of them run jobs of the bulk
//! classes `Streaming` and `Background` ([`JobClass::is_bulk`]) at once, so that the frame
//! thread keeps a physical core while the world streams in. `Critical` and `Interactive` jobs
//! are never held back by that bulk limit. [`WorkerPolicy::standard`] holds the arithmetic,
//! [`CpuCores::detect`] counts the cores: the standard library knows only the logical ones, so
//! the physical ones are estimated on the assumption that logical cores come in SMT pairs.
//! The environment variable `CP_PHYSICAL_CORES` ([`CpuCores::PHYSICAL_CORES_VAR`]) or
//! [`CpuCores::with_physical`] override the estimate, and [`JobSystem::with_policy`] takes any
//! policy. [`JobSystem::new`] sets no bulk limit.
//!
//! ```
//! use cp_jobs::{CpuCores, JobSystem, WorkerPolicy};
//!
//! // The engine's job system on this machine.
//! let jobs = JobSystem::standard("engine");
//! assert!(1 <= jobs.bulk_limit() && jobs.bulk_limit() <= jobs.threads());
//!
//! // The policy for a processor with eight cores and no SMT, which the estimate would halve.
//! let cores = CpuCores::assuming_smt_pairs(8).with_physical(8);
//! let policy = WorkerPolicy::standard(cores);
//! assert_eq!((policy.threads(), policy.bulk_limit()), (5, 5));
//! let jobs = JobSystem::with_policy(policy, "eight-cores");
//! assert_eq!(jobs.bulk_limit(), 5);
//! ```
//!
//! # Things to know
//!
//! * The closure of a job that is cancelled while queued is dropped on the thread that cancels
//!   it, or that shuts the system down; see [`CancelToken::cancel`].
//! * [`JobHandle::wait`] inside a job blocks a worker. If all workers wait for jobs that are
//!   still queued, nothing moves until the system is shut down.
//! * A job that queues follow-up jobs or calls `par_for` captures a [`JobSpawner`]. A captured
//!   [`JobSystem`] clone is an owner like any other: while such a job is queued or running,
//!   dropping the other clones shuts nothing down. [`JobSystem::shutdown`] ends the system
//!   whoever holds clones.
//! * [`JobSystem::par_for`] does not use the workers: it starts scoped helper threads for each
//!   call, which costs some tens of microseconds per thread. It is meant for coarse loops. The
//!   bulk limit does not apply to it.
//! * Under a bulk limit, a bulk job that waits for another bulk job keeps its slot. Once every
//!   slot is kept that way, no queued bulk job starts until the system shuts down. Streaming
//!   work that continues in another job queues that job and returns, as in the example below.
//!
//! # Example
//!
//! ```
//! use std::sync::atomic::{AtomicU64, Ordering};
//!
//! use cp_jobs::{JobClass, JobSystem, Priority};
//!
//! let jobs = JobSystem::new(4, "example");
//!
//! // A job with a result.
//! let answer = jobs.spawn(Priority::new(JobClass::Interactive, 0), |_cancel| 6 * 7);
//! assert_eq!(answer.wait(), Ok(42));
//!
//! // A long job polls its token and gives up once it is cancelled.
//! let search = jobs.spawn(Priority::new(JobClass::Background, 10), |cancel| {
//!     (0..1_000_000_u64).take_while(|_| !cancel.is_cancelled()).filter(|n| n % 7 == 0).count()
//! });
//! search.cancel();
//! // Cancelled before it started, stopped early, or already done: all three are possible here.
//! let _ = search.wait();
//!
//! // A blocking loop over borrowed data.
//! let squares: Vec<AtomicU64> = (0..1000).map(|_| AtomicU64::new(0)).collect();
//! jobs.par_for(squares.len(), 64, |i| squares[i].store((i * i) as u64, Ordering::Relaxed));
//! assert_eq!(squares[30].load(Ordering::Relaxed), 900);
//!
//! // A job that queues its follow-up job captures a spawner, not a clone of the system.
//! let spawner = jobs.spawner();
//! let generated = jobs.spawn(Priority::new(JobClass::Streaming, 0), move |_cancel| {
//!     let chunk = vec![1_u8; 32];
//!     spawner.spawn(Priority::new(JobClass::Streaming, 1), move |_cancel| chunk.len())
//! });
//! let meshed = generated.wait().expect("the first job neither panicked nor was cancelled");
//! assert_eq!(meshed.wait(), Ok(32));
//!
//! // Cancels what is still queued, waits for what is running and joins the workers. Dropping
//! // `jobs` would do the same here, because no job holds a clone of it.
//! jobs.shutdown();
//! assert!(jobs.spawn(Priority::new(JobClass::Critical, 0), |_cancel| ()).wait().is_err());
//! ```

#![warn(missing_docs)]

mod cancel;
mod contain;
mod handle;
mod par_for;
mod policy;
mod priority;
mod queue;
mod serving;
mod spawner;
mod stats;
mod system;
mod task;

pub use cancel::CancelToken;
pub use handle::{JobError, JobHandle};
pub use policy::{CpuCores, WorkerPolicy};
pub use priority::{JobClass, Priority};
pub use spawner::JobSpawner;
pub use stats::JobStats;
pub use system::JobSystem;
