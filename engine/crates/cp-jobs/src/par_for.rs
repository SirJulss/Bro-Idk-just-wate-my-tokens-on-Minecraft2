//! Blocking parallel loops over an index range.
//!
//! The loop body may borrow from the caller's stack. Lending such a borrow to the long-lived
//! worker threads would need `unsafe`, so the helpers of a loop are scoped threads that exist
//! for the duration of the call. A side effect worth having: a loop started from inside a job
//! does not depend on a free worker and therefore cannot deadlock the job system.

use std::ops::Range;
use std::panic::{AssertUnwindSafe, catch_unwind, resume_unwind};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::thread;

use log::Level;
use parking_lot::Mutex;

use crate::contain::{Payload, drop_contained, log_contained};
use crate::serving;

/// Calls `body(index)` exactly once for every index in `0..count`, on at most `max_threads`
/// threads including the calling one, and returns when all calls have returned.
///
/// Threads claim `grain` consecutive indices at a time (`0` counts as `1`). Helper threads are
/// named `"{name}-par-{n}"`. If `body` panics, see [`JobSystem::par_for`](crate::JobSystem::par_for).
///
/// A loop that a job runs is part of that job: its helper threads count as running job code
/// for the same queue as the calling thread (see [`serving`]).
pub(crate) fn par_for<F>(name: &str, max_threads: usize, count: usize, grain: usize, body: &F)
where
    F: Fn(usize) + Sync,
{
    let grain = grain.max(1);
    let participants = max_threads.min(count.div_ceil(grain));
    if participants <= 1 {
        (0..count).for_each(body);
        return;
    }

    // Relaxed is enough for both: the cursor is only ever changed by compare-and-swap, and the
    // flag merely saves work. Joining the scope orders everything `body` did before the return.
    let cursor = AtomicUsize::new(0);
    let failed = AtomicBool::new(false);
    let first_panic: Mutex<Option<Payload>> = Mutex::new(None);

    // Captures shared references only, which makes the closure `Copy`: every thread gets its own.
    let participate = || {
        let chunks = catch_unwind(AssertUnwindSafe(|| {
            while !failed.load(Ordering::Relaxed)
                && let Some(chunk) = claim(&cursor, count, grain)
            {
                chunk.for_each(body);
            }
        }));
        if let Err(payload) = chunks {
            failed.store(true, Ordering::Relaxed);
            let mut first = first_panic.lock();
            if first.is_none() {
                *first = Some(payload);
            } else {
                drop(first);
                drop_contained(payload, "the payload of a further par_for panic");
            }
        }
    };

    // The caller waits for its helpers, so whatever must not wait for the caller's job must not
    // be done by a helper either: a shutdown requested by `body` on a helper thread would
    // otherwise wait for a job that waits for this loop. Helpers therefore take over the mark
    // of the calling thread.
    let served = serving::served();

    thread::scope(|scope| {
        for helper in 1..participants {
            // Starting a thread takes tens of microseconds. If the helpers that already run
            // have claimed everything in the meantime, one more would find nothing to do.
            if failed.load(Ordering::Relaxed) || cursor.load(Ordering::Relaxed) >= count {
                break;
            }
            let spawned = thread::Builder::new()
                .name(format!("{name}-par-{helper}"))
                .spawn_scoped(scope, move || {
                    serving::serve(served);
                    participate();
                });
            if let Err(error) = spawned {
                // Chunks are claimed dynamically, so fewer threads still visit every index.
                log_contained(
                    Level::Warn,
                    format_args!("par_for of '{name}' continues with {helper} threads: {error}"),
                );
                break;
            }
        }
        participate();
    });

    if let Some(payload) = first_panic.into_inner() {
        resume_unwind(payload);
    }
}

/// Claims the next unvisited chunk of at most `grain` indices; `None` when all are taken.
fn claim(cursor: &AtomicUsize, count: usize, grain: usize) -> Option<Range<usize>> {
    let chunk_end = |start: usize| start.saturating_add(grain).min(count);
    cursor
        .try_update(Ordering::Relaxed, Ordering::Relaxed, |start| {
            (start < count).then(|| chunk_end(start))
        })
        .ok()
        .map(|start| start..chunk_end(start))
}

#[cfg(test)]
mod tests {
    use std::sync::Barrier;

    use super::*;
    use crate::serving::QueueId;

    /// Runs a loop on `THREADS` threads at once, called from a thread that is marked as serving
    /// `caller_serves`, and returns the mark every participating thread saw in the loop body.
    fn marks_seen_in_a_loop(caller_serves: Option<QueueId>) -> Vec<Option<QueueId>> {
        const THREADS: usize = 3;
        let seen = Mutex::new(Vec::new());
        thread::scope(|scope| {
            scope.spawn(|| {
                serving::serve(caller_serves);
                // Every thread stays in its first call until all have arrived, so each index
                // is visited by a different thread: the caller and two helpers.
                let all_inside = Barrier::new(THREADS);
                par_for("test", THREADS, THREADS, 1, &|_| {
                    all_inside.wait();
                    seen.lock().push(serving::served());
                });
            });
        });
        seen.into_inner()
    }

    #[test]
    fn helper_threads_take_over_the_mark_of_the_calling_thread() {
        let queue = QueueId::unique();
        assert_eq!(marks_seen_in_a_loop(Some(queue)), [Some(queue); 3]);
    }

    #[test]
    fn helper_threads_of_an_unmarked_caller_are_unmarked() {
        assert_eq!(marks_seen_in_a_loop(None), [None; 3]);
    }

    /// Every chunk a single thread gets to claim, as `(start, end)` with `end` exclusive.
    fn all_chunks(count: usize, grain: usize) -> Vec<(usize, usize)> {
        let cursor = AtomicUsize::new(0);
        std::iter::from_fn(|| claim(&cursor, count, grain))
            .map(|chunk| (chunk.start, chunk.end))
            .collect()
    }

    #[test]
    fn chunks_tile_the_range_without_gaps_or_overlap() {
        assert_eq!(all_chunks(10, 4), [(0, 4), (4, 8), (8, 10)]);
        assert_eq!(all_chunks(8, 4), [(0, 4), (4, 8)]);
        assert_eq!(all_chunks(3, 100), [(0, 3)]);
        assert_eq!(all_chunks(5, 1), [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]);
        assert!(all_chunks(0, 4).is_empty());
    }

    #[test]
    fn claiming_does_not_overflow_near_the_end_of_the_index_space() {
        const END: usize = usize::MAX;
        assert_eq!(all_chunks(END, END - 1), [(0, END - 1), (END - 1, END)]);
        assert_eq!(all_chunks(END, END), [(0, END)]);
    }

    #[test]
    fn single_thread_runs_inline_and_in_order() {
        let visited = Mutex::new(Vec::new());
        let caller = thread::current().id();
        par_for("test", 1, 5, 2, &|index| {
            visited.lock().push((index, thread::current().id()))
        });
        assert_eq!(
            visited.into_inner(),
            (0..5).map(|index| (index, caller)).collect::<Vec<_>>()
        );
    }
}
