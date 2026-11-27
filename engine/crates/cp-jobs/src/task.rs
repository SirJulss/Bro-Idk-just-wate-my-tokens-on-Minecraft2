//! Type-erased jobs as they sit in the queue.

use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::Arc;
use std::thread;

use log::Level;

use crate::cancel::CancelToken;
use crate::contain::{drop_contained, log_contained, panic_message};
use crate::handle::{JobError, Slot};

/// How the closure of a job ended.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Outcome {
    /// The closure returned.
    Completed,
    /// The closure panicked.
    Panicked,
}

impl Outcome {
    fn of<T>(result: &Result<T, JobError>) -> Self {
        if result.is_ok() {
            Self::Completed
        } else {
            Self::Panicked
        }
    }
}

/// A queued job with its closure and the way to report its result.
///
/// Life cycle: either [`run`](Self::run) followed by [`publish`](Self::publish), or
/// [`cancel`](Self::cancel) alone; each at most once. Then the task is dropped.
///
/// `publish` and `cancel` are called with the queue locked and therefore must not run user
/// code: no closure, no destructor of a captured value or of a result. Such destructors run
/// when the task itself is dropped, which its owner does without the lock.
pub(crate) trait Task: Send {
    /// Runs the closure, containing a panic, and keeps the result for `publish`.
    fn run(&mut self, token: &CancelToken) -> Outcome;

    /// Makes the result of `run` visible to whoever waits for it.
    fn publish(&mut self);

    /// Reports the job as cancelled. The closure stays inside the task until the task is dropped.
    fn cancel(&mut self);
}

/// A job whose result goes to a [`JobHandle`](crate::JobHandle).
pub(crate) struct HandleTask<T, F> {
    /// The closure, until it runs.
    work: Option<F>,
    /// What the closure produced, between `run` and `publish`.
    result: Option<Result<T, JobError>>,
    slot: Arc<Slot<T>>,
}

impl<T, F> HandleTask<T, F>
where
    F: FnOnce(&CancelToken) -> T,
{
    pub(crate) fn new(work: F, slot: Arc<Slot<T>>) -> Self {
        Self {
            work: Some(work),
            result: None,
            slot,
        }
    }
}

impl<T, F> Task for HandleTask<T, F>
where
    T: Send,
    F: FnOnce(&CancelToken) -> T + Send,
{
    fn run(&mut self, token: &CancelToken) -> Outcome {
        let work = self.work.take().expect("a job runs at most once");
        let result = run_contained(work, token);
        let outcome = Outcome::of(&result);
        self.result = Some(result);
        outcome
    }

    fn publish(&mut self) {
        if let Some(result) = self.result.take() {
            self.slot.resolve(result);
        }
    }

    fn cancel(&mut self) {
        self.slot.resolve(Err(JobError::Cancelled));
    }
}

/// A job nobody waits for.
pub(crate) struct DetachedTask<F> {
    /// The closure, until it runs.
    work: Option<F>,
}

impl<F> DetachedTask<F>
where
    F: FnOnce(&CancelToken),
{
    pub(crate) fn new(work: F) -> Self {
        Self { work: Some(work) }
    }
}

impl<F> Task for DetachedTask<F>
where
    F: FnOnce(&CancelToken) + Send,
{
    fn run(&mut self, token: &CancelToken) -> Outcome {
        let work = self.work.take().expect("a job runs at most once");
        Outcome::of(&run_contained(work, token))
    }

    fn publish(&mut self) {}

    fn cancel(&mut self) {}
}

/// Calls `work`. A panic is logged and returned as [`JobError::Panicked`].
fn run_contained<T>(
    work: impl FnOnce(&CancelToken) -> T,
    token: &CancelToken,
) -> Result<T, JobError> {
    // The closure is consumed by the call, so nothing it owns can be seen in a broken state
    // afterwards. State it shares with other code is that code's concern, as with any thread
    // that panics.
    catch_unwind(AssertUnwindSafe(|| work(token))).map_err(|payload| {
        let message = panic_message(&*payload);
        drop_contained(payload, "the payload of a job panic");
        let thread = thread::current();
        log_contained(
            Level::Error,
            format_args!(
                "job panicked on thread '{}': {message}",
                thread.name().unwrap_or("<unnamed>")
            ),
        );
        JobError::Panicked(message)
    })
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicBool, Ordering};

    use super::*;

    #[test]
    fn handle_task_publishes_what_the_closure_returns() {
        let slot = Arc::new(Slot::new());
        let mut task = HandleTask::new(
            |token: &CancelToken| token.is_cancelled(),
            Arc::clone(&slot),
        );

        assert_eq!(task.run(&CancelToken::new()), Outcome::Completed);
        assert_eq!(
            slot.try_take(),
            None,
            "the result is not visible before publish"
        );
        task.publish();
        assert_eq!(slot.try_take(), Some(Ok(false)));
    }

    #[test]
    fn handle_task_reports_a_panic_with_its_message() {
        let slot = Arc::new(Slot::new());
        let mut task = HandleTask::new(
            |_: &CancelToken| -> u8 { panic!("task {} failed", 3) },
            Arc::clone(&slot),
        );

        assert_eq!(task.run(&CancelToken::new()), Outcome::Panicked);
        task.publish();
        assert_eq!(
            slot.try_take(),
            Some(Err(JobError::Panicked("task 3 failed".to_owned())))
        );
    }

    #[test]
    fn cancelled_handle_task_keeps_its_closure_until_it_is_dropped() {
        let slot = Arc::new(Slot::new());
        let ran = Arc::new(AtomicBool::new(false));
        let captured = Arc::new(());
        let mut task = HandleTask::new(
            {
                let ran = Arc::clone(&ran);
                let captured = Arc::clone(&captured);
                move |_: &CancelToken| {
                    ran.store(true, Ordering::SeqCst);
                    drop(captured);
                }
            },
            Arc::clone(&slot),
        );

        task.cancel();
        assert_eq!(slot.try_take(), Some(Err(JobError::Cancelled)));
        assert_eq!(
            Arc::strong_count(&captured),
            2,
            "cancel must not drop user values"
        );
        drop(task);
        assert_eq!(Arc::strong_count(&captured), 1);
        assert!(!ran.load(Ordering::SeqCst));
    }

    #[test]
    fn detached_task_contains_a_panic() {
        let mut failing = DetachedTask::new(|_: &CancelToken| panic!("detached task failed"));
        assert_eq!(failing.run(&CancelToken::new()), Outcome::Panicked);
        failing.publish();

        let mut fine = DetachedTask::new(|_: &CancelToken| {});
        assert_eq!(fine.run(&CancelToken::new()), Outcome::Completed);
    }
}
