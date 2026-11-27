//! Handles to spawned jobs and the slot their result travels through.

use std::error::Error;
use std::fmt;
use std::mem;
use std::sync::Arc;

use parking_lot::{Condvar, Mutex};

use crate::cancel::CancelToken;

/// Why a job has no result.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum JobError {
    /// The job was cancelled before it started: through its token, by the shutdown of its
    /// system, or because it was spawned after that shutdown. Its closure never ran.
    Cancelled,
    /// The job's closure panicked. Carries the panic message, or a placeholder when the panic
    /// payload was not text.
    Panicked(String),
}

impl fmt::Display for JobError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Cancelled => f.write_str("job was cancelled before it started"),
            Self::Panicked(message) => write!(f, "job panicked: {message}"),
        }
    }
}

impl Error for JobError {}

/// Receives the result of one job started with [`JobSystem::spawn`](crate::JobSystem::spawn)
/// or [`JobSpawner::spawn`](crate::JobSpawner::spawn).
///
/// A job is *finished* once its closure has returned or panicked, or once it has been cancelled
/// before it started. The handle then holds `Ok(value)`, [`JobError::Panicked`] or
/// [`JobError::Cancelled`] until the result is taken, which works exactly once.
///
/// Dropping a handle neither cancels the job nor waits for it; the result is discarded. The
/// handle stays usable after its [`JobSystem`](crate::JobSystem) is gone.
pub struct JobHandle<T> {
    slot: Arc<Slot<T>>,
    token: CancelToken,
}

impl<T> JobHandle<T> {
    pub(crate) fn new(slot: Arc<Slot<T>>, token: CancelToken) -> Self {
        Self { slot, token }
    }

    /// Whether the job is finished. Stays `true` after the result has been taken.
    #[must_use]
    pub fn is_finished(&self) -> bool {
        self.slot.is_resolved()
    }

    /// Takes the result if the job is finished. Never blocks.
    ///
    /// Returns `None` while the job is queued or running, and again after the result has been
    /// taken; [`is_finished`](Self::is_finished) tells the two apart.
    pub fn try_take(&self) -> Option<Result<T, JobError>> {
        self.slot.try_take()
    }

    /// Blocks until the job is finished and returns its result.
    ///
    /// Waiting inside a job for another job of the same system keeps a worker busy doing
    /// nothing; if every worker waits for a job that is still queued, none of them can make
    /// progress. Only the shutdown of the system ends such a wait: it cancels the queued job,
    /// and the wait returns [`JobError::Cancelled`].
    /// [`JobSystem::shutdown`](crate::JobSystem::shutdown) does that at any time. Dropping the
    /// system does it only when no clone of the system is left, so jobs that wait must not own
    /// one: they spawn through a [`JobSpawner`](crate::JobSpawner).
    ///
    /// Under a bulk limit ([`JobSystem::bulk_limit`](crate::JobSystem::bulk_limit)) the same
    /// holds for its slots: a bulk job that waits for a queued bulk job keeps a slot that the
    /// other one may need, and once every slot is kept that way no queued bulk job starts,
    /// however many workers are idle.
    ///
    /// # Panics
    /// Panics if the result was already taken with [`try_take`](Self::try_take).
    pub fn wait(self) -> Result<T, JobError> {
        self.slot.wait()
    }

    /// Requests cancellation of the job; the same as `self.token().cancel()`.
    ///
    /// See [`CancelToken::cancel`] for what that means before and after the job has started.
    pub fn cancel(&self) {
        self.token.cancel();
    }

    /// A clone of the job's token, the one its closure receives.
    #[must_use]
    pub fn token(&self) -> CancelToken {
        self.token.clone()
    }
}

impl<T> fmt::Debug for JobHandle<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("JobHandle")
            .field("finished", &self.is_finished())
            .field("cancelled", &self.token.is_cancelled())
            .finish_non_exhaustive()
    }
}

/// Where the result of a job waits for its handle.
pub(crate) struct Slot<T> {
    state: Mutex<SlotState<T>>,
    resolved: Condvar,
}

enum SlotState<T> {
    /// The job is queued or running.
    Pending,
    /// The job is finished and nobody has taken the result yet.
    Ready(Result<T, JobError>),
    /// The result has been handed out.
    Taken,
}

impl<T> Slot<T> {
    pub(crate) fn new() -> Self {
        Self {
            state: Mutex::new(SlotState::Pending),
            resolved: Condvar::new(),
        }
    }

    /// Stores the result and wakes everyone who waits for it.
    ///
    /// Must be called exactly once per slot. Runs no user code, so the job queue may be locked.
    pub(crate) fn resolve(&self, result: Result<T, JobError>) {
        let previous = mem::replace(&mut *self.state.lock(), SlotState::Ready(result));
        self.resolved.notify_all();
        debug_assert!(
            matches!(previous, SlotState::Pending),
            "a job is resolved exactly once"
        );
    }

    fn is_resolved(&self) -> bool {
        !matches!(*self.state.lock(), SlotState::Pending)
    }

    pub(crate) fn try_take(&self) -> Option<Result<T, JobError>> {
        let mut state = self.state.lock();
        match mem::replace(&mut *state, SlotState::Taken) {
            SlotState::Ready(result) => Some(result),
            unchanged => {
                *state = unchanged;
                None
            }
        }
    }

    pub(crate) fn wait(&self) -> Result<T, JobError> {
        let mut state = self.state.lock();
        while matches!(*state, SlotState::Pending) {
            self.resolved.wait(&mut state);
        }
        match mem::replace(&mut *state, SlotState::Taken) {
            SlotState::Ready(result) => result,
            SlotState::Pending | SlotState::Taken => {
                panic!("JobHandle::wait: the result was already taken with try_take")
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::mpsc;
    use std::thread;

    use super::*;

    #[test]
    fn slot_hands_its_result_out_once() {
        let slot = Slot::new();
        assert!(!slot.is_resolved());
        assert_eq!(slot.try_take(), None);

        slot.resolve(Ok(5));
        assert!(slot.is_resolved());
        assert_eq!(slot.try_take(), Some(Ok(5)));
        assert_eq!(slot.try_take(), None);
        assert!(slot.is_resolved());
    }

    #[test]
    fn wait_returns_a_result_that_is_already_there() {
        let slot: Slot<u8> = Slot::new();
        slot.resolve(Err(JobError::Cancelled));
        assert_eq!(slot.wait(), Err(JobError::Cancelled));
    }

    #[test]
    fn wait_blocks_until_the_slot_is_resolved() {
        let slot = Arc::new(Slot::new());
        let (about_to_wait_tx, about_to_wait_rx) = mpsc::channel();
        let waiter = {
            let slot = Arc::clone(&slot);
            thread::spawn(move || {
                about_to_wait_tx.send(()).unwrap();
                slot.wait()
            })
        };
        about_to_wait_rx.recv().unwrap();
        // Correct whether the waiter already sleeps or has not locked the slot yet.
        slot.resolve(Ok("late"));
        assert_eq!(waiter.join().unwrap(), Ok("late"));
    }

    #[test]
    #[should_panic(expected = "already taken")]
    fn wait_panics_when_the_result_is_gone() {
        let slot = Slot::new();
        slot.resolve(Ok(1));
        assert_eq!(slot.try_take(), Some(Ok(1)));
        let _ = slot.wait();
    }

    #[test]
    fn errors_describe_themselves() {
        assert_eq!(
            JobError::Cancelled.to_string(),
            "job was cancelled before it started"
        );
        assert_eq!(
            JobError::Panicked("boom".to_owned()).to_string(),
            "job panicked: boom"
        );
        let as_error: &dyn Error = &JobError::Cancelled;
        assert!(as_error.source().is_none());
    }
}
