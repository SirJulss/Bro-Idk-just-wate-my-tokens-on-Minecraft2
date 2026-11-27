//! Bounded single-flight cache of generated base chunks (contract §5.1, "Snapshots").
//!
//! The first request for a missing chunk generates it ("leader"); concurrent requests for the
//! same chunk wait for that one generation instead of starting their own. Ready chunks are
//! kept until the byte budget forces the least recently used ones out. The cache memoises a
//! pure function of the key (the base world never changes), so its state affects speed only,
//! never results.

use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Duration;

use cp_core::coords::LodKey;
use parking_lot::{Condvar, Mutex};
use rustc_hash::FxHashMap;

use crate::base::Cancelled;
use crate::chunk::VoxelChunk;

/// How often a waiting request re-checks its own cancel callback.
const CANCEL_POLL: Duration = Duration::from_millis(2);

/// The outcome of one generation, shared by its leader and its waiters.
struct Flight {
    state: Mutex<FlightState>,
    done: Condvar,
}

enum FlightState {
    Running,
    Done(Arc<VoxelChunk>),
    /// The leader was cancelled or panicked; waiters retry.
    Failed,
}

enum Entry {
    Ready {
        chunk: Arc<VoxelChunk>,
        bytes: usize,
        stamp: u64,
    },
    Pending(Arc<Flight>),
}

#[derive(Default)]
struct Inner {
    entries: FxHashMap<LodKey, Entry>,
    /// Ready entries by last use, oldest first.
    lru: BTreeMap<u64, LodKey>,
    clock: u64,
    bytes: usize,
}

/// Size of the base-chunk cache of a [`crate::VoxelWorld`] and its snapshots.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CacheUsage {
    /// Chunks kept.
    pub chunks: usize,
    /// Bytes they occupy ([`VoxelChunk::memory_bytes`]).
    pub bytes: usize,
    /// The budget in bytes.
    pub budget: usize,
}

/// Single-flight, byte-bounded cache of base chunks, shared by a world and its snapshots.
pub(crate) struct BaseCache {
    inner: Mutex<Inner>,
    budget: usize,
}

/// Removes a pending entry and wakes its waiters unless the generation finished.
struct LeaderGuard<'a> {
    cache: &'a BaseCache,
    key: LodKey,
    flight: Arc<Flight>,
    finished: bool,
}

impl LeaderGuard<'_> {
    fn finish(mut self, chunk: Arc<VoxelChunk>) {
        let bytes = chunk.memory_bytes();
        {
            let mut inner = self.cache.inner.lock();
            inner.clock += 1;
            let stamp = inner.clock;
            inner.lru.insert(stamp, self.key);
            inner.bytes += bytes;
            inner.entries.insert(
                self.key,
                Entry::Ready {
                    chunk: Arc::clone(&chunk),
                    bytes,
                    stamp,
                },
            );
            while inner.bytes > self.cache.budget {
                let Some((_, oldest)) = inner.lru.pop_first() else {
                    break;
                };
                if let Some(Entry::Ready { bytes, .. }) = inner.entries.remove(&oldest) {
                    inner.bytes -= bytes;
                }
            }
        }
        *self.flight.state.lock() = FlightState::Done(chunk);
        self.flight.done.notify_all();
        self.finished = true;
    }
}

impl Drop for LeaderGuard<'_> {
    fn drop(&mut self) {
        if self.finished {
            return;
        }
        {
            let mut inner = self.cache.inner.lock();
            let ours = matches!(
                inner.entries.get(&self.key),
                Some(Entry::Pending(flight)) if Arc::ptr_eq(flight, &self.flight)
            );
            if ours {
                inner.entries.remove(&self.key);
            }
        }
        *self.flight.state.lock() = FlightState::Failed;
        self.flight.done.notify_all();
    }
}

enum Role {
    Leader(Arc<Flight>),
    Waiter(Arc<Flight>),
}

impl BaseCache {
    /// An empty cache that keeps at most `budget` bytes of ready chunks.
    pub(crate) fn new(budget: usize) -> Self {
        Self {
            inner: Mutex::new(Inner::default()),
            budget,
        }
    }

    /// The cached chunk for `key`, or the result of `generate` (called at most once per
    /// missing key at a time, across all threads).
    ///
    /// A waiting request returns [`Cancelled`] when its own `cancel` fires; when the leader is
    /// cancelled or panics, waiters retry and one of them generates.
    pub(crate) fn get_or_generate(
        &self,
        key: LodKey,
        cancel: &dyn Fn() -> bool,
        generate: impl Fn() -> Result<VoxelChunk, Cancelled>,
    ) -> Result<Arc<VoxelChunk>, Cancelled> {
        loop {
            let role = {
                let mut inner = self.inner.lock();
                let inner = &mut *inner;
                match inner.entries.get_mut(&key) {
                    Some(Entry::Ready { chunk, stamp, .. }) => {
                        inner.clock += 1;
                        inner.lru.remove(stamp);
                        *stamp = inner.clock;
                        inner.lru.insert(inner.clock, key);
                        return Ok(Arc::clone(chunk));
                    }
                    Some(Entry::Pending(flight)) => Role::Waiter(Arc::clone(flight)),
                    None => {
                        let flight = Arc::new(Flight {
                            state: Mutex::new(FlightState::Running),
                            done: Condvar::new(),
                        });
                        inner
                            .entries
                            .insert(key, Entry::Pending(Arc::clone(&flight)));
                        Role::Leader(flight)
                    }
                }
            };
            match role {
                Role::Leader(flight) => {
                    let guard = LeaderGuard {
                        cache: self,
                        key,
                        flight,
                        finished: false,
                    };
                    // On `Err` or a panic the guard wakes the waiters, which then retry.
                    let chunk = Arc::new(generate()?);
                    guard.finish(Arc::clone(&chunk));
                    return Ok(chunk);
                }
                Role::Waiter(flight) => {
                    let mut state = flight.state.lock();
                    loop {
                        match &*state {
                            FlightState::Done(chunk) => return Ok(Arc::clone(chunk)),
                            FlightState::Failed => break,
                            FlightState::Running => {
                                if cancel() {
                                    return Err(Cancelled);
                                }
                                flight.done.wait_for(&mut state, CANCEL_POLL);
                            }
                        }
                    }
                }
            }
        }
    }

    /// Current size of the cache.
    pub(crate) fn usage(&self) -> CacheUsage {
        let inner = self.inner.lock();
        CacheUsage {
            chunks: inner.lru.len(),
            bytes: inner.bytes,
            budget: self.budget,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cp_core::ids::BlockStateId;
    use cp_core::math::IVec3;
    use std::sync::atomic::{AtomicUsize, Ordering};

    fn key(x: i32) -> LodKey {
        LodKey::lod0(IVec3::new(x, 0, 0))
    }

    #[test]
    fn hits_do_not_generate_and_the_budget_evicts_lru() {
        let per_chunk = VoxelChunk::uniform(BlockStateId(1)).memory_bytes();
        let cache = BaseCache::new(per_chunk * 2);
        let calls = AtomicUsize::new(0);
        let generate = || {
            calls.fetch_add(1, Ordering::Relaxed);
            Ok(VoxelChunk::uniform(BlockStateId(1)))
        };
        let never = || false;
        let first = cache.get_or_generate(key(0), &never, generate).unwrap();
        let again = cache.get_or_generate(key(0), &never, generate).unwrap();
        assert!(Arc::ptr_eq(&first, &again));
        assert_eq!(calls.load(Ordering::Relaxed), 1);
        cache.get_or_generate(key(1), &never, generate).unwrap();
        // Touch 0 so that 1 is the least recently used one.
        cache.get_or_generate(key(0), &never, generate).unwrap();
        cache.get_or_generate(key(2), &never, generate).unwrap();
        let usage = cache.usage();
        assert_eq!(usage.chunks, 2);
        assert_eq!(usage.bytes, per_chunk * 2);
        assert_eq!(calls.load(Ordering::Relaxed), 3);
        cache.get_or_generate(key(0), &never, generate).unwrap();
        assert_eq!(calls.load(Ordering::Relaxed), 3, "0 survived");
        cache.get_or_generate(key(1), &never, generate).unwrap();
        assert_eq!(calls.load(Ordering::Relaxed), 4, "1 was evicted");
    }

    #[test]
    fn cancelled_and_panicking_leaders_leave_no_pending_entry() {
        let cache = BaseCache::new(1 << 20);
        let never = || false;
        let result = cache.get_or_generate(key(0), &never, || Err(Cancelled));
        assert_eq!(result.unwrap_err(), Cancelled);
        let panicked = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            cache.get_or_generate(key(0), &never, || panic!("generator bug"))
        }));
        assert!(panicked.is_err());
        let chunk = cache
            .get_or_generate(key(0), &never, || Ok(VoxelChunk::uniform(BlockStateId(2))))
            .unwrap();
        assert_eq!(chunk.get(0), BlockStateId(2));
    }
}
