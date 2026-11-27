//! Cooperative cancellation.

use std::fmt;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Weak};

use crate::queue::{JobQueue, QueueKey};

/// Asks a job to stop.
///
/// Every job has its own token: its closure receives a reference to it and
/// [`JobHandle::token`](crate::JobHandle::token) hands out clones. All clones share one flag,
/// and the flag never resets.
///
/// * Cancelling a job that has **not started** removes it from the queue. Its closure never
///   runs and its handle yields [`JobError::Cancelled`](crate::JobError::Cancelled).
/// * Cancelling a job that **is running** only sets the flag. Nothing interrupts the job; it
///   decides when to look at [`is_cancelled`](Self::is_cancelled) and what to return.
/// * Shutting down the [`JobSystem`](crate::JobSystem) cancels the tokens of all jobs that are
///   queued or running at that moment, and of every job that is spawned afterwards.
///
/// A token made with [`CancelToken::new`] belongs to no job. It is a plain shared flag, useful
/// for calling the body of a job directly, for example in a test.
#[derive(Clone)]
pub struct CancelToken {
    inner: Arc<Inner>,
}

struct Inner {
    cancelled: AtomicBool,
    /// Where the token's job waits; `None` for a token that belongs to no job.
    job: Option<QueuedJob>,
}

/// Enough to find a job in the queue of its system.
struct QueuedJob {
    /// Weak, so that tokens kept after the shutdown of their system hold on to nothing.
    queue: Weak<JobQueue>,
    key: QueueKey,
}

impl CancelToken {
    /// A token that belongs to no job and is not cancelled.
    #[must_use]
    pub fn new() -> Self {
        Self::with_job(None)
    }

    /// The token of the job that is queued in `queue` under `key`.
    pub(crate) fn for_job(queue: Weak<JobQueue>, key: QueueKey) -> Self {
        Self::with_job(Some(QueuedJob { queue, key }))
    }

    fn with_job(job: Option<QueuedJob>) -> Self {
        Self {
            inner: Arc::new(Inner {
                cancelled: AtomicBool::new(false),
                job,
            }),
        }
    }

    /// Requests cancellation. Calling it again, on any clone, changes nothing.
    ///
    /// If the token's job has not started, it is taken out of the queue before `cancel`
    /// returns: the closure never runs, the handle is finished with
    /// [`JobError::Cancelled`](crate::JobError::Cancelled) and the closure is dropped **on the
    /// calling thread**. Do not call `cancel` while holding a lock that the destructor of a
    /// value captured by that closure needs.
    ///
    /// If the job is running or has finished, only the flag is set.
    pub fn cancel(&self) {
        self.flag();
        if let Some(job) = &self.inner.job
            && let Some(queue) = job.queue.upgrade()
        {
            queue.cancel(job.key);
        }
    }

    /// Whether cancellation was requested, through any clone or by the shutdown of the job
    /// system. Cheap enough to poll in a loop (one atomic load).
    #[must_use]
    pub fn is_cancelled(&self) -> bool {
        self.inner.cancelled.load(Ordering::Acquire)
    }

    /// Sets the flag without touching the queue. For callers that hold the queue lock.
    pub(crate) fn flag(&self) {
        self.inner.cancelled.store(true, Ordering::Release);
    }
}

impl Default for CancelToken {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Debug for CancelToken {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CancelToken")
            .field("cancelled", &self.is_cancelled())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn free_token_starts_clear_and_stays_cancelled() {
        let token = CancelToken::new();
        assert!(!token.is_cancelled());
        token.cancel();
        assert!(token.is_cancelled());
        token.cancel();
        assert!(token.is_cancelled());
    }

    #[test]
    fn clones_share_one_flag() {
        let token = CancelToken::default();
        let clone = token.clone();
        assert!(!clone.is_cancelled());
        clone.cancel();
        assert!(token.is_cancelled());
        assert!(token.clone().is_cancelled());
    }

    #[test]
    fn separate_tokens_are_independent() {
        let first = CancelToken::new();
        let second = CancelToken::new();
        first.cancel();
        assert!(!second.is_cancelled());
    }

    #[test]
    fn debug_output_shows_the_flag() {
        let token = CancelToken::new();
        assert_eq!(format!("{token:?}"), "CancelToken { cancelled: false }");
        token.cancel();
        assert_eq!(format!("{token:?}"), "CancelToken { cancelled: true }");
    }
}
