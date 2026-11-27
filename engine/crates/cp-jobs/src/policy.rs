//! The worker policy: how many worker threads a job system starts, and how many of them may run
//! bulk jobs at the same time (contract §4.2).

use std::env::{self, VarError};
use std::thread;

use log::Level;

use crate::contain::log_contained;

/// The processor cores a job system can use: the input of [`WorkerPolicy::standard`].
///
/// *Logical* cores are the hardware threads that can run at the same time, *physical* cores the
/// cores that carry them. With simultaneous multithreading (SMT; Intel calls it Hyper-Threading)
/// one physical core runs two logical cores, which share its execution units: two busy threads
/// on one core get far less done than two threads on two cores.
///
/// Invariant: `1 <= physical() <= logical()`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct CpuCores {
    logical: usize,
    physical: usize,
}

impl CpuCores {
    /// The environment variable that overrides the physical-core estimate of
    /// [`detect`](Self::detect): a whole number of cores, for example `CP_PHYSICAL_CORES=8` on
    /// a processor with eight cores and no SMT.
    pub const PHYSICAL_CORES_VAR: &'static str = "CP_PHYSICAL_CORES";

    /// `logical` logical cores carried by `physical` physical cores, both moved into the
    /// invariant: `logical` up to at least 1, `physical` into `1..=logical`.
    #[must_use]
    pub fn new(logical: usize, physical: usize) -> Self {
        let logical = logical.max(1);
        Self {
            logical,
            physical: physical.clamp(1, logical),
        }
    }

    /// `logical` logical cores (at least 1) that come in SMT pairs: `logical.div_ceil(2)`
    /// physical cores, where an odd one out counts as a core of its own.
    ///
    /// The assumption holds for most desktop processors, among them the baseline Ryzen 5 9600X
    /// (12 logical, 6 physical cores). It halves the real number on processors without SMT and
    /// is too low on hybrid processors whose efficiency cores run one thread each (Intel since
    /// Alder Lake). There [`with_physical`](Self::with_physical) or the environment variable
    /// [`PHYSICAL_CORES_VAR`](Self::PHYSICAL_CORES_VAR) supply the real number.
    #[must_use]
    pub fn assuming_smt_pairs(logical: usize) -> Self {
        let logical = logical.max(1);
        Self::new(logical, logical.div_ceil(2))
    }

    /// The cores of this machine that the process may use.
    ///
    /// The logical cores are [`std::thread::available_parallelism`], which also reflects limits
    /// placed on the process (CPU affinity, container quotas); 1 if it cannot tell. The standard
    /// library cannot count physical cores, so they are estimated with
    /// [`assuming_smt_pairs`](Self::assuming_smt_pairs), unless the environment variable
    /// [`PHYSICAL_CORES_VAR`](Self::PHYSICAL_CORES_VAR) holds a whole number above 0: that
    /// number replaces the estimate, clamped to the logical cores. Any other value of the
    /// variable is ignored with a warning in the log.
    #[must_use]
    pub fn detect() -> Self {
        let logical = match thread::available_parallelism() {
            Ok(count) => Some(count.get()),
            Err(error) => {
                log_contained(
                    Level::Warn,
                    format_args!("cannot count the logical processor cores, assuming 1: {error}"),
                );
                None
            }
        };
        let physical = match env::var(Self::PHYSICAL_CORES_VAR) {
            Ok(text) => Some(text),
            Err(VarError::NotPresent) => None,
            // Not a number either; reported as such below.
            Err(VarError::NotUnicode(raw)) => Some(raw.to_string_lossy().into_owned()),
        };
        let cores = Self::from_findings(logical, physical.as_deref());
        log_contained(
            Level::Debug,
            format_args!(
                "processor: {} logical cores, {} physical cores",
                cores.logical, cores.physical
            ),
        );
        cores
    }

    /// What [`detect`](Self::detect) makes of what it found: `logical` from
    /// `available_parallelism`, `physical` the text of the override variable, if it is set.
    fn from_findings(logical: Option<usize>, physical: Option<&str>) -> Self {
        let estimate = Self::assuming_smt_pairs(logical.unwrap_or(1));
        let Some(text) = physical else {
            return estimate;
        };
        match text.trim().parse::<usize>() {
            Ok(physical) if physical > 0 => {
                if physical > estimate.logical {
                    log_contained(
                        Level::Warn,
                        format_args!(
                            "{}={physical} names more physical than logical cores; using {}",
                            Self::PHYSICAL_CORES_VAR,
                            estimate.logical
                        ),
                    );
                }
                estimate.with_physical(physical)
            }
            _ => {
                log_contained(
                    Level::Warn,
                    format_args!(
                        "ignoring {}={text:?}: expected a whole number of cores above 0; \
                         estimating {} physical cores",
                        Self::PHYSICAL_CORES_VAR,
                        estimate.physical
                    ),
                );
                estimate
            }
        }
    }

    /// The same logical cores carried by `physical` physical cores, clamped into
    /// `1..=logical()`: the override of an estimate.
    #[must_use]
    pub fn with_physical(self, physical: usize) -> Self {
        Self::new(self.logical, physical)
    }

    /// Number of logical cores; at least 1.
    #[must_use]
    pub const fn logical(self) -> usize {
        self.logical
    }

    /// Number of physical cores; between 1 and [`logical`](Self::logical).
    #[must_use]
    pub const fn physical(self) -> usize {
        self.physical
    }
}

/// How many worker threads a [`JobSystem`](crate::JobSystem) starts, and how many jobs of the
/// bulk classes ([`JobClass::is_bulk`](crate::JobClass::is_bulk)) may run at the same time.
///
/// The bulk limit keeps processor time free for the threads outside the job system and keeps
/// workers free for urgent jobs: while it is reached, a free worker leaves the queued bulk
/// jobs where they are and takes only `Critical` and `Interactive` jobs, which the limit never
/// holds back. It counts jobs, not cores; which core runs which thread is up to the operating
/// system.
///
/// Invariant: `1 <= bulk_limit() <= threads()`. With `bulk_limit() == threads()` the limit has
/// no effect.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct WorkerPolicy {
    threads: usize,
    bulk_limit: usize,
}

impl WorkerPolicy {
    /// Logical cores that [`standard`](Self::standard) leaves to the threads outside the job
    /// system, such as the frame thread and the simulation thread.
    pub const RESERVED_LOGICAL_CORES: usize = 3;

    /// Physical cores that [`standard`](Self::standard) keeps free of bulk jobs, so that the
    /// frame thread keeps a physical core while the world streams in.
    pub const RESERVED_PHYSICAL_CORES: usize = 1;

    /// `threads` worker threads, at most `bulk_limit` of them on bulk jobs at once. Both are
    /// moved into the invariant: `threads` up to at least 1, `bulk_limit` into `1..=threads`.
    #[must_use]
    pub fn new(threads: usize, bulk_limit: usize) -> Self {
        let threads = threads.max(1);
        Self {
            threads,
            bulk_limit: bulk_limit.clamp(1, threads),
        }
    }

    /// `threads` worker threads (at least 1) that all take jobs of every class: no bulk limit.
    /// The policy of [`JobSystem::new`](crate::JobSystem::new).
    #[must_use]
    pub fn uncapped(threads: usize) -> Self {
        Self::new(threads, threads)
    }

    /// The engine's policy for `cores` (contract §4.2):
    /// `threads = logical - RESERVED_LOGICAL_CORES` and
    /// `bulk_limit = physical - RESERVED_PHYSICAL_CORES`, each at least 1 and the limit at most
    /// `threads`.
    ///
    /// | logical cores | physical cores (SMT pairs) | threads | bulk limit |
    /// |---|---|---|---|
    /// | 1 to 4 | 1 to 2 | 1 | 1 |
    /// | 6 | 3 | 3 | 2 |
    /// | 8 | 4 | 5 | 3 |
    /// | 12 (baseline Ryzen 5 9600X) | 6 | 9 | 5 |
    /// | 16 | 8 | 13 | 7 |
    /// | 32 | 16 | 29 | 15 |
    ///
    /// On 4 logical cores and fewer a single worker remains. It may run bulk jobs, so an urgent
    /// job can wait there for a bulk job to end.
    #[must_use]
    pub fn standard(cores: CpuCores) -> Self {
        Self::new(
            cores.logical().saturating_sub(Self::RESERVED_LOGICAL_CORES),
            cores
                .physical()
                .saturating_sub(Self::RESERVED_PHYSICAL_CORES),
        )
    }

    /// Number of worker threads; at least 1.
    #[must_use]
    pub const fn threads(self) -> usize {
        self.threads
    }

    /// Most bulk jobs that run at the same time; between 1 and [`threads`](Self::threads).
    #[must_use]
    pub const fn bulk_limit(self) -> usize {
        self.bulk_limit
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn smt_estimate_pairs_the_logical_cores() {
        // (logical, physical)
        let cases = [
            (0, 1),
            (1, 1),
            (2, 1),
            (3, 2),
            (4, 2),
            (6, 3),
            (7, 4),
            (12, 6),
            (16, 8),
            (20, 10),
            (32, 16),
            (64, 32),
            (256, 128),
        ];
        for (logical, physical) in cases {
            let cores = CpuCores::assuming_smt_pairs(logical);
            assert_eq!(
                (cores.logical(), cores.physical()),
                (logical.max(1), physical),
                "{logical} logical cores"
            );
        }
    }

    #[test]
    fn cores_are_moved_into_the_invariant() {
        assert_eq!(CpuCores::new(0, 0), CpuCores::new(1, 1));
        assert_eq!(CpuCores::new(1, 1).physical(), 1);
        assert_eq!(CpuCores::new(8, 0).physical(), 1);
        assert_eq!(CpuCores::new(8, 8).physical(), 8);
        assert_eq!(CpuCores::new(8, 9).physical(), 8);
        let estimate = CpuCores::assuming_smt_pairs(12);
        assert_eq!(estimate.with_physical(12), CpuCores::new(12, 12));
        assert_eq!(estimate.with_physical(0), CpuCores::new(12, 1));
        assert_eq!(estimate.with_physical(40), CpuCores::new(12, 12));
    }

    #[test]
    fn standard_policy_for_several_core_counts() {
        // ((logical, physical), (threads, bulk limit))
        let cases = [
            ((1, 1), (1, 1)),
            ((2, 1), (1, 1)),
            ((2, 2), (1, 1)),
            ((3, 2), (1, 1)),
            ((4, 2), (1, 1)),
            ((4, 4), (1, 1)),
            ((5, 3), (2, 2)),
            ((6, 3), (3, 2)),
            ((6, 6), (3, 3)),
            ((8, 4), (5, 3)),
            ((8, 8), (5, 5)),
            // The baseline: Ryzen 5 9600X, 6 cores, 12 threads.
            ((12, 6), (9, 5)),
            ((12, 12), (9, 9)),
            // A hybrid processor: 6 cores with SMT and 8 without.
            ((20, 14), (17, 13)),
            ((16, 8), (13, 7)),
            ((24, 12), (21, 11)),
            ((32, 16), (29, 15)),
            ((64, 32), (61, 31)),
            ((128, 64), (125, 63)),
        ];
        for ((logical, physical), expected) in cases {
            let policy = WorkerPolicy::standard(CpuCores::new(logical, physical));
            assert_eq!(
                (policy.threads(), policy.bulk_limit()),
                expected,
                "{logical} logical, {physical} physical cores"
            );
            // The formulas of contract §4.2, at least one of each, the limit within the threads.
            let threads = logical.saturating_sub(3).max(1);
            assert_eq!(policy.threads(), threads);
            assert_eq!(
                policy.bulk_limit(),
                physical.saturating_sub(1).max(1).min(threads)
            );
        }
    }

    #[test]
    fn documented_table_of_the_standard_policy() {
        // (logical cores, threads, bulk limit), physical cores estimated as SMT pairs.
        let rows = [
            (1, 1, 1),
            (2, 1, 1),
            (3, 1, 1),
            (4, 1, 1),
            (6, 3, 2),
            (8, 5, 3),
            (12, 9, 5),
            (16, 13, 7),
            (32, 29, 15),
        ];
        for (logical, threads, bulk_limit) in rows {
            let policy = WorkerPolicy::standard(CpuCores::assuming_smt_pairs(logical));
            assert_eq!(
                (policy.threads(), policy.bulk_limit()),
                (threads, bulk_limit),
                "{logical} logical cores"
            );
        }
    }

    #[test]
    fn policies_are_moved_into_the_invariant() {
        let parts = |policy: WorkerPolicy| (policy.threads(), policy.bulk_limit());
        assert_eq!(parts(WorkerPolicy::new(0, 0)), (1, 1));
        assert_eq!(parts(WorkerPolicy::new(4, 0)), (4, 1));
        assert_eq!(parts(WorkerPolicy::new(4, 2)), (4, 2));
        assert_eq!(parts(WorkerPolicy::new(4, 9)), (4, 4));
        assert_eq!(parts(WorkerPolicy::uncapped(0)), (1, 1));
        assert_eq!(parts(WorkerPolicy::uncapped(6)), (6, 6));
    }

    #[test]
    fn detection_estimates_or_takes_the_override() {
        let found = CpuCores::from_findings;
        assert_eq!(found(Some(12), None), CpuCores::new(12, 6));
        assert_eq!(found(Some(7), None), CpuCores::new(7, 4));
        // `available_parallelism` failed.
        assert_eq!(found(None, None), CpuCores::new(1, 1));
        assert_eq!(found(None, Some("4")), CpuCores::new(1, 1));
        // The override replaces the estimate.
        assert_eq!(found(Some(12), Some("12")), CpuCores::new(12, 12));
        assert_eq!(found(Some(20), Some("14")), CpuCores::new(20, 14));
        assert_eq!(found(Some(12), Some(" 8\n")), CpuCores::new(12, 8));
        assert_eq!(found(Some(12), Some("1")), CpuCores::new(12, 1));
        // A core carries at least one logical core of the process.
        assert_eq!(found(Some(4), Some("6")), CpuCores::new(4, 4));
        // Anything but a whole number above 0 leaves the estimate.
        for text in [
            "",
            " ",
            "0",
            "-2",
            "six",
            "6.0",
            "1e1",
            "6 cores",
            "99999999999999999999999",
            "\u{fffd}",
        ] {
            assert_eq!(
                found(Some(12), Some(text)),
                CpuCores::new(12, 6),
                "{text:?}"
            );
        }
    }

    #[test]
    fn detected_cores_keep_the_invariant() {
        let cores = CpuCores::detect();
        assert!(cores.logical() >= 1);
        assert!((1..=cores.logical()).contains(&cores.physical()));
        let logical = thread::available_parallelism().map_or(1, |count| count.get());
        assert_eq!(cores.logical(), logical);
        if env::var_os(CpuCores::PHYSICAL_CORES_VAR).is_none() {
            assert_eq!(cores, CpuCores::assuming_smt_pairs(logical));
        }
    }
}
