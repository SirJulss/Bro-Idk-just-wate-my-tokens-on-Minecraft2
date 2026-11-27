//! CPU profiling scopes (contract §2.6).
//!
//! ```
//! fn mesh_chunk() {
//!     cp_core::profile_scope!("mesh_chunk");
//!     // ... work ...
//! }
//! # cp_core::profile::set_enabled(false);
//! # mesh_chunk();
//! ```
//!
//! A scope measures the wall-clock time between its creation and the end of the enclosing
//! block and adds it to a per-thread table keyed by the scope name. [`collect`] drains the
//! tables of all threads into one sorted list, typically once per frame or per benchmark.
//!
//! Profiling is off by default. A disabled scope costs one relaxed atomic load and a branch,
//! so scopes can stay in hot paths permanently. Timings are inclusive: a scope's time contains
//! the time of the scopes nested inside it.
//!
//! Profiling data is wall-clock time and therefore not deterministic; it must never influence
//! simulation results.

use std::collections::{BTreeMap, HashMap};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};
use std::time::Instant;

/// Accumulated timings of one scope name, summed over all threads since the last [`collect`].
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct ScopeStat {
    /// The name given to [`scope`].
    pub name: &'static str,
    /// How many scopes with this name ended.
    pub calls: u64,
    /// Sum of their durations in nanoseconds.
    pub total_ns: u64,
    /// Longest single duration in nanoseconds.
    pub max_ns: u64,
}

#[derive(Clone, Copy, Default)]
struct Accumulator {
    calls: u64,
    total_ns: u64,
    max_ns: u64,
}

type ThreadTable = Arc<Mutex<HashMap<&'static str, Accumulator>>>;

static ENABLED: AtomicBool = AtomicBool::new(false);

/// The tables of all threads that ever recorded a scope. A table outlives its thread until a
/// [`collect`] that finds the thread gone has drained it, so the work of short-lived job
/// threads is not lost.
static REGISTRY: Mutex<Vec<ThreadTable>> = Mutex::new(Vec::new());

thread_local! {
    /// This thread's table. Only this thread and [`collect`] lock it, so the lock is
    /// practically never contended.
    static LOCAL: ThreadTable = {
        let table = ThreadTable::default();
        lock(&REGISTRY).push(Arc::clone(&table));
        table
    };
}

/// Locks a mutex, ignoring poisoning: the tables hold plain counters that stay consistent even
/// if a thread panicked while holding the lock.
fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}

/// Turns recording on or off for all threads. Scopes that are already open keep the decision
/// made when they started.
pub fn set_enabled(enabled: bool) {
    ENABLED.store(enabled, Ordering::Relaxed);
}

/// Whether scopes are currently being recorded.
#[inline]
#[must_use]
pub fn is_enabled() -> bool {
    ENABLED.load(Ordering::Relaxed)
}

/// Ends its scope when dropped. Created by [`scope`] or [`profile_scope!`](crate::profile_scope).
#[must_use = "the scope ends when the guard is dropped; bind it to a variable"]
#[derive(Debug)]
pub struct ScopeGuard {
    name: &'static str,
    /// `None` when profiling was disabled at the start of the scope.
    start: Option<Instant>,
}

/// Starts a scope named `name`; it ends when the returned guard is dropped.
///
/// Scopes with equal names are merged, also across threads.
#[inline]
pub fn scope(name: &'static str) -> ScopeGuard {
    let start = if is_enabled() {
        Some(Instant::now())
    } else {
        None
    };
    ScopeGuard { name, start }
}

impl Drop for ScopeGuard {
    #[inline]
    fn drop(&mut self) {
        if let Some(start) = self.start {
            record(self.name, start);
        }
    }
}

#[cold]
#[inline(never)]
fn record(name: &'static str, start: Instant) {
    let nanos = u64::try_from(start.elapsed().as_nanos()).unwrap_or(u64::MAX);
    // `try_with` fails only while the thread is being torn down; such a scope is dropped silently.
    let _ = LOCAL.try_with(|table| {
        let mut table = lock(table);
        let entry = table.entry(name).or_default();
        entry.calls += 1;
        entry.total_ns = entry.total_ns.saturating_add(nanos);
        entry.max_ns = entry.max_ns.max(nanos);
    });
}

/// Returns the statistics recorded on all threads since the previous call and resets them.
///
/// The result is sorted by name. Scopes that are still open are not included; they count
/// towards the collection during which they end. A scope that ends while a collection is
/// running is reported by that collection or by the next one, also when its thread exits
/// right afterwards; no scope is reported twice.
pub fn collect() -> Vec<ScopeStat> {
    collect_with(|_| {})
}

/// [`collect`] with a hook that runs each time one thread's table has been handled, while the
/// registry is still locked. The tests use it to let a thread record and exit at exactly that
/// point; for [`collect`] it is an empty closure that the compiler removes.
fn collect_with(mut table_done: impl FnMut(&ThreadTable)) -> Vec<ScopeStat> {
    let mut merged: BTreeMap<&'static str, Accumulator> = BTreeMap::new();
    lock(&REGISTRY).retain(|table| {
        // Whether the owning thread still exists is decided *before* its table is drained,
        // never afterwards. A table referenced only by the registry belongs to a thread that
        // has exited: nothing can be added to it any more, so draining it now gets everything
        // and it can be dropped. A thread that is still alive at this point may record more
        // scopes and exit before this pass is over; its table therefore stays registered and
        // the next collection drains what was added. (Checking after the drain would drop
        // such a table together with the scopes recorded in between.)
        let thread_alive = Arc::strong_count(table) > 1;
        for (name, stat) in lock(table).drain() {
            let entry = merged.entry(name).or_default();
            entry.calls += stat.calls;
            entry.total_ns = entry.total_ns.saturating_add(stat.total_ns);
            entry.max_ns = entry.max_ns.max(stat.max_ns);
        }
        table_done(table);
        thread_alive
    });
    merged
        .into_iter()
        .map(|(name, stat)| ScopeStat {
            name,
            calls: stat.calls,
            total_ns: stat.total_ns,
            max_ns: stat.max_ns,
        })
        .collect()
}

/// Opens a profiling scope that lasts until the end of the enclosing block.
///
/// `profile_scope!("name")` is shorthand for binding [`profile::scope`](crate::profile::scope)
/// to a hidden local variable. Available as `cp_core::profile_scope!` and as
/// `cp_core::profile::profile_scope!`.
#[macro_export]
macro_rules! profile_scope {
    ($name:expr) => {
        let _profile_scope_guard = $crate::profile::scope($name);
    };
}

pub use crate::profile_scope;

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::mpsc;
    use std::time::Duration;

    /// The profiler is global state: tests running in parallel would steal each other's
    /// statistics and switch recording on and off under each other. Every test here holds
    /// this lock from start to end.
    static SERIAL: Mutex<()> = Mutex::new(());

    fn find<'a>(stats: &'a [ScopeStat], name: &str) -> Option<&'a ScopeStat> {
        stats.iter().find(|stat| stat.name == name)
    }

    /// How many scopes called `name` the statistics contain.
    fn calls(stats: &[ScopeStat], name: &str) -> u64 {
        find(stats, name).map_or(0, |stat| stat.calls)
    }

    #[test]
    fn profiler_end_to_end() {
        let _serial = lock(&SERIAL);
        // Disabled: nothing is recorded and no clock is read.
        set_enabled(false);
        assert!(!is_enabled());
        {
            let guard = scope("test.disabled");
            assert!(guard.start.is_none());
            // The macro is reachable through the module as well as through the crate root.
            profile_scope!("test.disabled_macro");
        }
        let stats = collect();
        assert!(find(&stats, "test.disabled").is_none());
        assert!(find(&stats, "test.disabled_macro").is_none());

        // Enabled: calls, totals and maxima add up.
        set_enabled(true);
        assert!(is_enabled());
        for millis in [2, 6] {
            crate::profile_scope!("test.sleep");
            std::thread::sleep(Duration::from_millis(millis));
        }
        {
            let _outer = scope("test.outer");
            let _inner = scope("test.inner");
            std::thread::sleep(Duration::from_millis(1));
        }
        // A scope opened while enabled is recorded even if profiling is turned off meanwhile,
        // and one opened while disabled is not recorded even if it is turned on meanwhile.
        let straddling = scope("test.straddling");
        set_enabled(false);
        let late = scope("test.late");
        drop(straddling);
        set_enabled(true);
        drop(late);

        // Other threads, including ones that have already exited, are collected too.
        let workers: Vec<_> = (0..4)
            .map(|_| {
                std::thread::spawn(|| {
                    for _ in 0..25 {
                        crate::profile_scope!("test.worker");
                    }
                })
            })
            .collect();
        for worker in workers {
            worker.join().unwrap();
        }
        let live_worker = std::thread::scope(|threads| {
            threads
                .spawn(|| {
                    crate::profile_scope!("test.worker");
                })
                .join()
        });
        live_worker.unwrap();

        let stats = collect();
        let names: Vec<&str> = stats.iter().map(|stat| stat.name).collect();
        let mut sorted = names.clone();
        sorted.sort_unstable();
        assert_eq!(names, sorted, "collect() must return names in sorted order");

        // Sleeps of 2 ms and 6 ms; the bounds leave a millisecond of slack for coarse OS timers.
        let sleep = find(&stats, "test.sleep").unwrap();
        assert_eq!(sleep.calls, 2);
        assert!(sleep.max_ns >= 5_000_000, "{sleep:?}");
        assert!(sleep.total_ns >= 6_000_000, "{sleep:?}");
        assert!(sleep.max_ns < sleep.total_ns);
        let outer = find(&stats, "test.outer").unwrap();
        let inner = find(&stats, "test.inner").unwrap();
        assert_eq!((outer.calls, inner.calls), (1, 1));
        assert!(
            outer.total_ns >= inner.total_ns,
            "outer scopes include inner ones"
        );
        assert!(inner.total_ns >= 200_000, "{inner:?}");
        assert_eq!(inner.max_ns, inner.total_ns);
        assert_eq!(find(&stats, "test.straddling").unwrap().calls, 1);
        assert!(find(&stats, "test.late").is_none());
        let worker = find(&stats, "test.worker").unwrap();
        assert_eq!(worker.calls, 101);
        assert!(worker.max_ns <= worker.total_ns);

        // Collecting drains: a second collection has none of the test scopes.
        let stats = collect();
        assert!(
            stats.iter().all(|stat| !stat.name.starts_with("test.")),
            "{stats:?}"
        );
        // The tables of the exited worker threads were released.
        let registered = lock(&REGISTRY).len();
        assert!(
            registered < 5,
            "{registered} thread tables still registered"
        );

        set_enabled(false);
    }

    /// Regression test. A thread whose table has already been drained by a running collection
    /// records one more scope and exits before that collection is over. `collect` used to
    /// decide only after draining all tables which threads were gone, found this one gone and
    /// dropped its table together with the new scope.
    #[test]
    fn a_scope_recorded_during_a_collection_survives_its_thread() {
        let _serial = lock(&SERIAL);
        set_enabled(true);
        let _ = collect();

        let (table_tx, table_rx) = mpsc::channel();
        let (resume_tx, resume_rx) = mpsc::channel::<()>();
        let worker = std::thread::spawn(move || {
            drop(scope("test.late_exit"));
            // The address tells the hook below which table is this thread's.
            let table = LOCAL.with(|table| Arc::as_ptr(table).addr());
            table_tx.send(table).unwrap();
            // Wait until the collection has drained the table, then record once more and exit.
            resume_rx.recv().unwrap();
            drop(scope("test.late_exit"));
        });
        let worker_table = table_rx.recv().unwrap();

        let mut worker = Some(worker);
        let during = collect_with(|table| {
            if Arc::as_ptr(table).addr() == worker_table {
                resume_tx.send(()).unwrap();
                // Joined: the thread is gone, and so is its reference to the table.
                worker.take().unwrap().join().unwrap();
            }
        });
        assert!(worker.is_none(), "the worker's table was never visited");
        assert_eq!(calls(&during, "test.late_exit"), 1);
        let is_registered = || {
            lock(&REGISTRY)
                .iter()
                .any(|table| Arc::as_ptr(table).addr() == worker_table)
        };
        assert!(is_registered(), "the table was dropped with a scope in it");

        // The next collection reports the late scope and only then releases the table.
        let after = collect();
        assert_eq!(calls(&after, "test.late_exit"), 1, "{after:?}");
        assert!(!is_registered());
        assert_eq!(calls(&collect(), "test.late_exit"), 0);

        set_enabled(false);
    }

    /// Threads start, record and exit while another thread collects in a loop, as a per-frame
    /// `collect` does next to short-lived job threads. Every scope must be reported exactly
    /// once.
    #[test]
    fn no_scope_is_lost_while_threads_come_and_go() {
        const ROUNDS: u64 = 250;
        const THREADS: u64 = 12;
        const SCOPES: u64 = 150;
        // Threads that stay alive with a registered table make a collection take longer, so
        // more of the short-lived threads end in the middle of one.
        const BYSTANDERS: usize = 48;

        let _serial = lock(&SERIAL);
        set_enabled(true);
        let _ = collect();

        let done = AtomicBool::new(false);
        let done = &done;
        let (registered_tx, registered_rx) = mpsc::channel();
        let mut seen = std::thread::scope(|threads| {
            let bystanders: Vec<_> = (0..BYSTANDERS)
                .map(|_| {
                    let registered_tx = registered_tx.clone();
                    threads.spawn(move || {
                        drop(scope("test.bystander"));
                        registered_tx.send(()).unwrap();
                        while !done.load(Ordering::Acquire) {
                            std::thread::park();
                        }
                    })
                })
                .collect();
            for _ in 0..BYSTANDERS {
                registered_rx.recv().unwrap();
            }

            let collector = threads.spawn(move || {
                let mut seen = 0;
                while !done.load(Ordering::Acquire) {
                    seen += calls(&collect(), "test.stress");
                }
                seen
            });
            for _ in 0..ROUNDS {
                let workers: Vec<_> = (0..THREADS)
                    .map(|_| {
                        threads.spawn(|| {
                            for _ in 0..SCOPES {
                                crate::profile_scope!("test.stress");
                            }
                        })
                    })
                    .collect();
                // Joining (unlike merely leaving the thread scope) waits until a thread is
                // gone for good, including its reference to its table.
                for worker in workers {
                    worker.join().unwrap();
                }
            }
            done.store(true, Ordering::Release);
            for bystander in bystanders {
                bystander.thread().unpark();
                bystander.join().unwrap();
            }
            collector.join().unwrap()
        });
        // What was recorded after the collector's last look at a table.
        seen += calls(&collect(), "test.stress");
        assert_eq!(seen, ROUNDS * THREADS * SCOPES);
        // All those threads are gone, so that collection has released their tables too.
        let registered = lock(&REGISTRY).len();
        assert!(
            registered < 5,
            "{registered} thread tables still registered"
        );

        set_enabled(false);
    }
}
