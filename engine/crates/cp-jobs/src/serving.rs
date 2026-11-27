//! Which threads run job code for which queue.
//!
//! A shutdown waits for the running jobs, unless it is requested from inside one of them: a job
//! cannot return before a call it makes does. "Inside a job" means on the worker thread that
//! runs the job, or on a helper thread of a `par_for` loop that the job has started. Both kinds
//! of thread carry a mark that names their queue, and every other thread carries none.

use std::cell::Cell;
use std::sync::atomic::{AtomicU64, Ordering};

/// Identity of a job queue, unique within the process.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct QueueId(u64);

impl QueueId {
    /// An identity that no other queue of this process has.
    pub(crate) fn unique() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        Self(NEXT.fetch_add(1, Ordering::Relaxed))
    }
}

thread_local! {
    /// The queue the current thread runs job code for.
    static SERVED: Cell<Option<QueueId>> = const { Cell::new(None) };
}

/// Marks the calling thread as running job code for `queue`, for the rest of its life.
///
/// Only for threads that exist for this purpose: workers, and the helpers of a `par_for` loop,
/// which take over the mark of the thread that called the loop. The mark is never removed, so
/// it still holds while the thread ends and drops its thread-local values.
pub(crate) fn serve(queue: Option<QueueId>) {
    SERVED.set(queue);
}

/// The queue the calling thread runs job code for. `None` on every thread that is neither a
/// worker nor a loop helper of a job.
pub(crate) fn served() -> Option<QueueId> {
    // `try_with`: the last handle of a job system may be dropped by the destructor of some
    // other thread-local value, at a time when thread-local storage is being torn down.
    SERVED.try_with(Cell::get).ok().flatten()
}

#[cfg(test)]
mod tests {
    use std::thread;

    use super::*;

    #[test]
    fn queue_ids_are_unique() {
        let ids: Vec<QueueId> = (0..100).map(|_| QueueId::unique()).collect();
        for (index, id) in ids.iter().enumerate() {
            assert!(!ids[..index].contains(id));
        }
    }

    #[test]
    fn mark_belongs_to_the_thread_that_set_it() {
        let queue = QueueId::unique();
        let on_marked_thread = thread::spawn(move || {
            let before = served();
            serve(Some(queue));
            (before, served())
        })
        .join()
        .unwrap();

        assert_eq!(on_marked_thread, (None, Some(queue)));
        assert_eq!(served(), None, "the mark of another thread leaked");
        assert_eq!(thread::spawn(served).join().unwrap(), None);
    }
}
