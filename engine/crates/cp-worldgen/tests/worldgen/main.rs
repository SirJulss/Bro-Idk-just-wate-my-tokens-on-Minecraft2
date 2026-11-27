//! Integration tests of the `cp-worldgen` generators against the `BaseSource` contract
//! (contract §5.1, §6): pinned output, classifications that never lie, LOD consistency with
//! the normative downsample rule, pull-based features, the look of the ground and a release
//! timing benchmark.

mod classify;
mod common;
mod features;
mod golden;
mod lod;
mod terrain;
mod timing;
