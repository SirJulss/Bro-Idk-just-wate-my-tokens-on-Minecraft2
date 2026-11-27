//! Simulation time: fixed ticks and the clock that turns frame time into ticks.
//!
//! The simulation advances in fixed steps of [`TICK_DT`] seconds and counts them in [`Tick`]s;
//! simulation time is always derived from the tick count, never accumulated in floating point.
//! [`FixedClock`] lives on the client side: it converts the irregular wall-clock duration of
//! frames into whole simulation steps plus an interpolation factor for rendering.

use core::ops::{Add, AddAssign};

use serde::{Deserialize, Serialize};

/// Simulation ticks per second.
pub const TICK_HZ: u32 = 60;

/// Duration of one simulation tick in seconds (`1 / TICK_HZ`).
pub const TICK_DT: f64 = 1.0 / TICK_HZ as f64;

/// A point in simulation time, counted in ticks since the start of the simulation.
#[derive(
    Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug, Default, Serialize, Deserialize,
)]
#[repr(transparent)]
#[serde(transparent)]
pub struct Tick(pub u64);

impl Tick {
    /// The first tick.
    pub const ZERO: Self = Self(0);

    /// Simulation time of this tick in seconds (`ticks / TICK_HZ`).
    ///
    /// Exact up to 2^53 ticks (millions of years); computed with one rounding, so two ticks
    /// always compare the same way as their seconds.
    #[inline]
    #[must_use]
    pub fn seconds(self) -> f64 {
        self.0 as f64 / f64::from(TICK_HZ)
    }

    /// Number of ticks from `earlier` to `self`; zero when `earlier` is not earlier.
    #[inline]
    #[must_use]
    pub const fn since(self, earlier: Self) -> u64 {
        self.0.saturating_sub(earlier.0)
    }
}

impl Add<u64> for Tick {
    type Output = Self;

    /// The tick `ticks` steps later (saturating at the end of time).
    #[inline]
    fn add(self, ticks: u64) -> Self {
        Self(self.0.saturating_add(ticks))
    }
}

impl AddAssign<u64> for Tick {
    #[inline]
    fn add_assign(&mut self, ticks: u64) {
        *self = *self + ticks;
    }
}

/// What a frame has to do, as decided by [`FixedClock::advance`].
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct StepPlan {
    /// Number of simulation ticks to run now, `0..=max_steps`.
    pub steps: u32,
    /// Fraction of a tick that real time is ahead of the simulation after those steps, in
    /// `[0, 1)`. Render with state interpolated this far between the previous and the current
    /// tick.
    pub alpha: f64,
    /// Whole ticks of real time that were discarded because more than `max_steps` were due.
    /// Non-zero means the simulation runs slower than real time for this frame.
    pub dropped: u64,
}

/// Fixed-timestep accumulator.
///
/// Feed it the real duration of every frame; it answers how many [`TICK_DT`] steps to simulate.
/// When a frame took so long that more than `max_steps` ticks are due, the surplus is dropped
/// instead of simulated, which keeps one slow frame from causing ever slower ones.
#[derive(Clone, Debug, PartialEq)]
pub struct FixedClock {
    /// Real time not yet turned into ticks, in seconds, in `[0, TICK_DT)`.
    accumulator: f64,
    max_steps: u32,
}

impl Default for FixedClock {
    fn default() -> Self {
        Self::new(Self::DEFAULT_MAX_STEPS)
    }
}

impl FixedClock {
    /// Default limit of simulation steps per frame: real time is kept down to 12 frames per
    /// second, below that the simulation slows down.
    pub const DEFAULT_MAX_STEPS: u32 = 5;

    /// Clock that plans at most `max_steps` ticks per [`FixedClock::advance`] (at least 1).
    #[must_use]
    pub const fn new(max_steps: u32) -> Self {
        Self {
            accumulator: 0.0,
            max_steps: if max_steps == 0 { 1 } else { max_steps },
        }
    }

    /// The limit of steps per call.
    #[inline]
    #[must_use]
    pub const fn max_steps(&self) -> u32 {
        self.max_steps
    }

    /// Current interpolation factor in `[0, 1)`, as last returned in [`StepPlan::alpha`].
    #[inline]
    #[must_use]
    pub fn alpha(&self) -> f64 {
        self.accumulator / TICK_DT
    }

    /// Forgets accumulated time, for example after a pause or a loading screen.
    #[inline]
    pub fn reset(&mut self) {
        self.accumulator = 0.0;
    }

    /// Adds `real_dt` seconds of wall-clock time and plans the frame.
    ///
    /// Negative, NaN and infinite durations count as zero.
    pub fn advance(&mut self, real_dt: f64) -> StepPlan {
        if real_dt.is_finite() && real_dt > 0.0 {
            self.accumulator += real_dt;
        }
        let due = (self.accumulator / TICK_DT).floor();
        self.accumulator -= due * TICK_DT;
        // Rounding in the two lines above can leave the remainder a hair outside the tick.
        if !(self.accumulator >= 0.0 && self.accumulator < TICK_DT) {
            self.accumulator = 0.0;
        }
        // Float-to-integer casts saturate, so absurdly long frames cannot overflow.
        let due = due as u64;
        let steps = due.min(u64::from(self.max_steps)) as u32;
        StepPlan {
            steps,
            alpha: self.accumulator / TICK_DT,
            dropped: due - u64::from(steps),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tick_constants() {
        assert_eq!(TICK_HZ, 60);
        assert_eq!(TICK_DT, 1.0 / 60.0);
        assert_eq!(Tick::ZERO, Tick(0));
        assert_eq!(Tick::default(), Tick::ZERO);
        assert_eq!(Tick(60).seconds(), 1.0);
        assert_eq!(Tick(90).seconds(), 1.5);
        assert_eq!(Tick(216_000).seconds(), 3600.0);
        assert_eq!(Tick(5) + 3, Tick(8));
        assert_eq!(Tick(u64::MAX) + 1, Tick(u64::MAX));
        let mut tick = Tick(1);
        tick += 2;
        assert_eq!(tick, Tick(3));
        assert_eq!(Tick(8).since(Tick(5)), 3);
        assert_eq!(Tick(5).since(Tick(8)), 0);
        assert!(Tick(1) < Tick(2));
    }

    #[test]
    fn steady_frames_step_once() {
        let mut clock = FixedClock::default();
        assert_eq!(clock.max_steps(), FixedClock::DEFAULT_MAX_STEPS);
        for _ in 0..1000 {
            let plan = clock.advance(TICK_DT);
            assert_eq!(
                plan,
                StepPlan {
                    steps: 1,
                    alpha: 0.0,
                    dropped: 0
                }
            );
        }
    }

    #[test]
    fn fast_and_slow_frames_keep_real_time() {
        // 144 Hz display: 2.4 frames per tick. 25 Hz: 2.4 ticks per frame.
        for frame_hz in [144.0, 120.0, 75.0, 60.0, 50.0, 30.0, 25.0, 13.0] {
            let mut clock = FixedClock::default();
            let dt = 1.0 / frame_hz;
            let frames = 10_000;
            let mut total_steps = 0_u64;
            for _ in 0..frames {
                let plan = clock.advance(dt);
                assert!((0.0..1.0).contains(&plan.alpha), "{}", plan.alpha);
                assert_eq!(plan.alpha, clock.alpha());
                assert_eq!(plan.dropped, 0);
                assert!(plan.steps <= clock.max_steps());
                let per_frame = f64::from(TICK_HZ) / frame_hz;
                assert!(f64::from(plan.steps) >= per_frame.floor() - 1.0);
                assert!(f64::from(plan.steps) <= per_frame.ceil() + 1.0);
                total_steps += u64::from(plan.steps);
            }
            // Simulated time tracks real time to within one tick over the whole run.
            let real = f64::from(frames) * dt;
            let simulated = total_steps as f64 * TICK_DT + clock.alpha() * TICK_DT;
            assert!(
                (real - simulated).abs() < 1e-9,
                "{frame_hz} Hz: {real} vs {simulated}"
            );
        }
    }

    #[test]
    fn alpha_reports_the_remainder() {
        let mut clock = FixedClock::new(8);
        let plan = clock.advance(2.5 * TICK_DT);
        assert_eq!(plan.steps, 2);
        assert!((plan.alpha - 0.5).abs() < 1e-12);
        let plan = clock.advance(0.25 * TICK_DT);
        assert_eq!(plan.steps, 0);
        assert!((plan.alpha - 0.75).abs() < 1e-12);
        let plan = clock.advance(0.5 * TICK_DT);
        assert_eq!(plan.steps, 1);
        assert!((plan.alpha - 0.25).abs() < 1e-12);
        clock.advance(0.5 * TICK_DT);
        clock.reset();
        assert_eq!(clock.alpha(), 0.0);
        assert_eq!(
            clock.advance(0.0),
            StepPlan {
                steps: 0,
                alpha: 0.0,
                dropped: 0
            }
        );
    }

    #[test]
    fn long_frames_are_clamped_and_reported() {
        let mut clock = FixedClock::new(4);
        // Half a second hitch: 30 ticks due, 4 run, 26 dropped; the fraction survives.
        let plan = clock.advance(30.25 * TICK_DT);
        assert_eq!((plan.steps, plan.dropped), (4, 26));
        assert!((plan.alpha - 0.25).abs() < 1e-9);
        // The backlog is gone: the next normal frame is normal again.
        let plan = clock.advance(TICK_DT);
        assert_eq!((plan.steps, plan.dropped), (1, 0));

        assert_eq!(FixedClock::new(0).max_steps(), 1);
        let mut single = FixedClock::new(0);
        let plan = single.advance(3.0 * TICK_DT);
        assert_eq!((plan.steps, plan.dropped), (1, 2));
    }

    #[test]
    fn bad_durations_never_break_the_clock() {
        let mut clock = FixedClock::default();
        clock.advance(0.4 * TICK_DT);
        let before = clock.clone();
        for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, -1.0, -0.0, 0.0] {
            let plan = clock.advance(bad);
            assert_eq!(plan.steps, 0);
            assert_eq!(plan.dropped, 0);
            assert_eq!(clock, before);
        }
        // Astronomically long frames saturate instead of overflowing.
        for huge in [1.0e9, 1.0e30, f64::MAX] {
            let plan = clock.advance(huge);
            assert_eq!(plan.steps, clock.max_steps());
            assert!(plan.dropped > 1_000_000);
            assert!((0.0..1.0).contains(&plan.alpha));
            let plan = clock.advance(TICK_DT);
            assert!(plan.steps <= 2 && plan.dropped == 0);
        }
    }
}
