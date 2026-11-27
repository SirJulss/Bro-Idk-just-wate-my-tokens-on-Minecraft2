//! Integration tests for the guarantees of `cp-jobs` (contract section 4.2).
//!
//! The tests synchronise through channels, barriers and job handles only; none of them sleeps.
//! Every test body runs under a deadline, so a deadlock fails the test instead of hanging the
//! whole run.

mod cancellation;
mod ordering;
mod panics;
mod par_for;
mod policy;
mod shutdown;
mod spawner;
mod statistics;
mod stress;
mod util;
