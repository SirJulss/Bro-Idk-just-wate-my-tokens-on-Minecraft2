//! Counters of a job system.

use crate::priority::JobClass;

/// Snapshot of the counters of a [`JobSystem`](crate::JobSystem), taken in one step by
/// [`JobSystem::stats`](crate::JobSystem::stats).
///
/// Every job is in exactly one state, so every snapshot satisfies
/// `submitted == queued + running + completed + cancelled + panicked`.
///
/// Snapshots agree with job handles: once a handle reports its job as finished, the job is
/// counted as completed, panicked or cancelled, and once the counters show nothing queued or
/// running, every handle of an earlier job is finished. Detached jobs are counted as soon as
/// their closure has returned.
///
/// [`JobSystem::par_for`](crate::JobSystem::par_for) loops are not jobs and are not counted.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct JobStats {
    /// Jobs waiting in the queue.
    pub queued: usize,
    /// Waiting jobs per class, indexed by [`JobClass::index`]. Sums up to `queued`.
    pub queued_by_class: [usize; JobClass::COUNT],
    /// Jobs whose closure is executing right now; at most the number of worker threads.
    pub running: usize,
    /// Jobs passed to `spawn` and `spawn_detached` since the system was created, including the
    /// ones that arrived after the shutdown.
    pub submitted: u64,
    /// Jobs whose closure returned.
    pub completed: u64,
    /// Jobs that were cancelled before they started: through their token, by the shutdown, or
    /// because they were spawned after the shutdown. Their closures never ran.
    pub cancelled: u64,
    /// Jobs whose closure panicked.
    pub panicked: u64,
}
