//! Deterministic elementary functions (contract §2.5, rule 1).
//!
//! The standard library forwards `f64::sin`, `f32::powf` and friends to the platform's C runtime,
//! whose results differ in the last bits between operating systems, CPUs and compiler versions.
//! Everything here is backed by the pure-Rust `libm` crate instead and returns the same bits on
//! every machine, which is what makes world generation and simulation reproducible.
//!
//! The free functions are generic over [`Real`], so `det::sin(x)` works for `f32` and `f64`.
//! Angles are in radians.
//!
//! # What deterministic code may take from `std`
//! Operations that IEEE 754 defines exactly are deterministic already and are used through the
//! standard library: `+ - * /`, `%`, the comparisons, `sqrt`, `abs`, `floor`, `ceil`, `round`,
//! `trunc`, `copysign`.
//!
//! Do **not** use these `std` methods in deterministic code:
//!
//! - `sin`, `cos`, `tan`, `exp`, `ln`, `powf` and the other transcendental methods; use the
//!   functions of this module.
//! - `mul_add` (fused or not, depending on the target) and `powi` (rounding depends on the
//!   compiler); use plain arithmetic and [`powi`].
//! - `min`, `max` and `clamp`; use [`min`], [`max`] and [`clamp`]. The `std` methods are exact
//!   for ordinary numbers, but for zeros of opposite sign (`0.0` and `-0.0`) the standard
//!   library leaves open which one is returned, and the answer does change with the operand
//!   order, the target and whether the compiler evaluates the call. The sign of a zero is not
//!   harmless: it is part of `to_bits`, so it reaches state hashes and saved data, and it
//!   decides `signum`, `copysign`, `atan2(±0, x)` and `1.0 / x`.
//!
//! The workspace's `clippy.toml` lists all of these `std` methods under `disallowed-methods`
//! (contract §3.3), so a use in a deterministic crate fails the lint. Code that must call them
//! anyway, such as the tests below that compare against `std`, does so under a scoped
//! `#[allow(clippy::disallowed_methods)]` with a comment. This module itself needs no allow:
//! it calls `libm`, never the `std` methods.
//!
//! `glam` is built with its `libm` feature (contract §2.1), so its methods that need an
//! elementary function call the same pinned `libm`; see [`crate::math`].
//!
//! NaN results carry an unspecified payload (IEEE 754 does not fix the bits of a generated
//! NaN, and CPUs differ); everything else, the sign of a zero included, is pinned.

use core::fmt::Debug;
use core::ops::{Add, Div, Mul, Neg, Sub};

use glam::{DQuat, DVec3};

mod sealed {
    pub trait Sealed {}
    impl Sealed for f32 {}
    impl Sealed for f64 {}
}

/// Generates the unary part of [`Real`]'s hidden interface, its two implementations and the
/// public free functions from one table, so the three can never drift apart.
macro_rules! real_functions {
    (
        unary {
            $( $(#[$udoc:meta])* $uname:ident / $uhidden:ident => $u64:ident, $u32:ident; )*
        }
        binary {
            $(
                $(#[$bdoc:meta])*
                $bname:ident / $bhidden:ident ($ba:ident, $bb:ident) => $b64:ident, $b32:ident;
            )*
        }
    ) => {
        /// A floating point type with deterministic elementary functions: `f32` or `f64`.
        ///
        /// The trait is sealed. Its `det_*` methods are an implementation detail; call the free
        /// functions of [`crate::det`] instead.
        pub trait Real:
            Copy
            + PartialOrd
            + Debug
            + Add<Output = Self>
            + Sub<Output = Self>
            + Mul<Output = Self>
            + Div<Output = Self>
            + Neg<Output = Self>
            + sealed::Sealed
            + 'static
        {
            /// `0.0`
            const ZERO: Self;
            /// `1.0`
            const ONE: Self;
            /// `2.0`
            const TWO: Self;
            /// `3.0`
            const THREE: Self;

            $( #[doc(hidden)] fn $uhidden(self) -> Self; )*
            $( #[doc(hidden)] fn $bhidden(self, other: Self) -> Self; )*
            #[doc(hidden)]
            fn det_sin_cos(self) -> (Self, Self);
            #[doc(hidden)]
            fn det_is_nan(self) -> bool;
            #[doc(hidden)]
            fn det_is_sign_negative(self) -> bool;
        }

        impl Real for f64 {
            const ZERO: Self = 0.0;
            const ONE: Self = 1.0;
            const TWO: Self = 2.0;
            const THREE: Self = 3.0;

            $( #[inline] fn $uhidden(self) -> Self { libm::$u64(self) } )*
            $( #[inline] fn $bhidden(self, other: Self) -> Self { libm::$b64(self, other) } )*
            #[inline]
            fn det_sin_cos(self) -> (Self, Self) {
                libm::sincos(self)
            }
            #[inline]
            fn det_is_nan(self) -> bool {
                self.is_nan()
            }
            #[inline]
            fn det_is_sign_negative(self) -> bool {
                self.is_sign_negative()
            }
        }

        impl Real for f32 {
            const ZERO: Self = 0.0;
            const ONE: Self = 1.0;
            const TWO: Self = 2.0;
            const THREE: Self = 3.0;

            $( #[inline] fn $uhidden(self) -> Self { libm::$u32(self) } )*
            $( #[inline] fn $bhidden(self, other: Self) -> Self { libm::$b32(self, other) } )*
            #[inline]
            fn det_sin_cos(self) -> (Self, Self) {
                libm::sincosf(self)
            }
            #[inline]
            fn det_is_nan(self) -> bool {
                self.is_nan()
            }
            #[inline]
            fn det_is_sign_negative(self) -> bool {
                self.is_sign_negative()
            }
        }

        $(
            $(#[$udoc])*
            #[inline]
            #[must_use]
            pub fn $uname<T: Real>(x: T) -> T {
                x.$uhidden()
            }
        )*

        $(
            $(#[$bdoc])*
            #[inline]
            #[must_use]
            pub fn $bname<T: Real>($ba: T, $bb: T) -> T {
                $ba.$bhidden($bb)
            }
        )*
    };
}

real_functions! {
    unary {
        /// Sine of `x` (radians).
        sin / det_sin => sin, sinf;
        /// Cosine of `x` (radians).
        cos / det_cos => cos, cosf;
        /// Tangent of `x` (radians).
        tan / det_tan => tan, tanf;
        /// Arcsine in radians, result in `[-π/2, π/2]`; NaN outside `[-1, 1]`.
        asin / det_asin => asin, asinf;
        /// Arccosine in radians, result in `[0, π]`; NaN outside `[-1, 1]`.
        acos / det_acos => acos, acosf;
        /// Arctangent in radians, result in `[-π/2, π/2]`.
        atan / det_atan => atan, atanf;
        /// Hyperbolic sine.
        sinh / det_sinh => sinh, sinhf;
        /// Hyperbolic cosine.
        cosh / det_cosh => cosh, coshf;
        /// Hyperbolic tangent.
        tanh / det_tanh => tanh, tanhf;
        /// `e^x`.
        exp / det_exp => exp, expf;
        /// `2^x`.
        exp2 / det_exp2 => exp2, exp2f;
        /// `e^x - 1`, accurate near zero.
        exp_m1 / det_exp_m1 => expm1, expm1f;
        /// Natural logarithm; NaN for negative `x`, `-inf` for zero.
        ln / det_ln => log, logf;
        /// `ln(1 + x)`, accurate near zero.
        ln_1p / det_ln_1p => log1p, log1pf;
        /// Base-2 logarithm.
        log2 / det_log2 => log2, log2f;
        /// Base-10 logarithm.
        log10 / det_log10 => log10, log10f;
        /// Cube root.
        cbrt / det_cbrt => cbrt, cbrtf;
    }
    binary {
        /// Four-quadrant arctangent of `y / x` in radians, result in `[-π, π]`.
        atan2 / det_atan2 (y, x) => atan2, atan2f;
        /// `x` raised to the floating point power `y`.
        powf / det_powf (x, y) => pow, powf;
        /// `sqrt(x² + y²)` without intermediate overflow or underflow.
        hypot / det_hypot (x, y) => hypot, hypotf;
    }
}

/// Sine and cosine of `x` (radians) as `(sin, cos)`.
#[inline]
#[must_use]
pub fn sin_cos<T: Real>(x: T) -> (T, T) {
    x.det_sin_cos()
}

/// `x` raised to the integer power `n` by repeated squaring.
///
/// Unlike `f64::powi`, whose rounding depends on the compiler, the multiplication order here is
/// fixed. `powi(x, 0)` is `1` for every `x`, including NaN.
#[inline]
#[must_use]
pub fn powi<T: Real>(x: T, n: i32) -> T {
    let mut base = x;
    let mut exponent = n.unsigned_abs();
    let mut result = T::ONE;
    while exponent != 0 {
        if exponent & 1 == 1 {
            result = result * base;
        }
        exponent >>= 1;
        if exponent != 0 {
            base = base * base;
        }
    }
    if n < 0 { T::ONE / result } else { result }
}

/// True when `a` comes strictly before `b` in the order of [`min`], [`max`] and [`clamp`]:
/// it is the smaller number, or the two are `-0.0` and `+0.0`. False when either is NaN.
#[inline(always)]
fn precedes<T: Real>(a: T, b: T) -> bool {
    a < b || (a == b && a.det_is_sign_negative() && !b.det_is_sign_negative())
}

/// The smaller of two numbers, with the same bits on every machine for every input.
///
/// Replaces `f64::min` / `f32::min`, whose result for `0.0` and `-0.0` is unspecified:
///
/// - `-0.0` counts as smaller than `+0.0`: `min(0.0, -0.0)` and `min(-0.0, 0.0)` are both
///   `-0.0`.
/// - NaN is ignored: if one operand is NaN the other is returned; if both are, the result is
///   NaN.
///
/// Unless both operands are NaN, the result does not depend on their order.
#[inline]
#[must_use]
pub fn min<T: Real>(a: T, b: T) -> T {
    if precedes(b, a) || a.det_is_nan() {
        b
    } else {
        a
    }
}

/// The larger of two numbers, with the same bits on every machine for every input.
///
/// Replaces `f64::max` / `f32::max`, whose result for `0.0` and `-0.0` is unspecified:
///
/// - `+0.0` counts as larger than `-0.0`: `max(0.0, -0.0)` and `max(-0.0, 0.0)` are both
///   `+0.0`.
/// - NaN is ignored: if one operand is NaN the other is returned; if both are, the result is
///   NaN.
///
/// Unless both operands are NaN, the result does not depend on their order.
#[inline]
#[must_use]
pub fn max<T: Real>(a: T, b: T) -> T {
    if precedes(a, b) || a.det_is_nan() {
        b
    } else {
        a
    }
}

/// `x` limited to the interval `[lo, hi]`, with the same bits on every machine for every
/// input.
///
/// Replaces `f64::clamp` / `f32::clamp`, whose result is unspecified when zeros of opposite
/// sign meet. Returns `lo` if `x` is below `lo`, `hi` if `x` is above `hi`, and `x` itself
/// otherwise, where `-0.0` counts as below `+0.0`. The result therefore lies inside the
/// interval down to the sign of a zero: `clamp(-0.0, 0.0, 1.0)` is `+0.0` and
/// `clamp(0.0, -1.0, -0.0)` is `-0.0`. A NaN `x` is returned as it is (unlike [`min`] and
/// [`max`], which ignore NaN). For non-NaN `x` this is `min(max(x, lo), hi)`; in particular
/// the result is `hi` for the bounds `lo = 0.0`, `hi = -0.0`.
///
/// # Panics
/// If `lo > hi` or a bound is NaN.
#[inline]
#[must_use]
#[track_caller]
pub fn clamp<T: Real>(x: T, lo: T, hi: T) -> T {
    assert!(
        lo <= hi,
        "det::clamp: lo > hi or a bound is NaN (lo = {lo:?}, hi = {hi:?})"
    );
    let raised = if precedes(x, lo) { lo } else { x };
    if precedes(hi, raised) { hi } else { raised }
}

/// Unit quaternion for a rotation of `angle` radians about `axis` (right-handed).
///
/// `axis` must be unit length. This is the deterministic replacement for
/// `DQuat::from_axis_angle`, which uses the platform's `sin`/`cos`.
#[inline]
#[must_use]
pub fn dquat_from_axis_angle(axis: DVec3, angle: f64) -> DQuat {
    let (s, c) = sin_cos(0.5 * angle);
    DQuat::from_xyzw(axis.x * s, axis.y * s, axis.z * s, c)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    // The point of this test is the comparison with the platform's `std` implementations.
    #[allow(clippy::disallowed_methods)]
    fn matches_std_closely() {
        // libm and the platform runtime may differ in the last bit, never by more than that.
        let mut x = -7.3_f64;
        while x < 7.3 {
            assert!((sin(x) - x.sin()).abs() <= 2.0 * f64::EPSILON, "sin({x})");
            assert!((cos(x) - x.cos()).abs() <= 2.0 * f64::EPSILON, "cos({x})");
            assert!(
                (atan(x) - x.atan()).abs() <= 4.0 * f64::EPSILON,
                "atan({x})"
            );
            assert!(
                (tanh(x) - x.tanh()).abs() <= 4.0 * f64::EPSILON,
                "tanh({x})"
            );
            let e = x.exp();
            assert!((exp(x) - e).abs() <= 4.0 * f64::EPSILON * e, "exp({x})");
            let (s, c) = sin_cos(x);
            assert_eq!(
                (s.to_bits(), c.to_bits()),
                (sin(x).to_bits(), cos(x).to_bits())
            );
            let xf = x as f32;
            assert!(
                (sin(xf) - xf.sin()).abs() <= 2.0 * f32::EPSILON,
                "sinf({xf})"
            );
            assert!(
                (cos(xf) - xf.cos()).abs() <= 2.0 * f32::EPSILON,
                "cosf({xf})"
            );
            x += 0.0137;
        }
        let mut x = 0.01_f64;
        while x < 50.0 {
            assert!((ln(x) - x.ln()).abs() <= 8.0 * f64::EPSILON * x.ln().abs().max(1.0));
            assert!((log2(x) - x.log2()).abs() <= 8.0 * f64::EPSILON * x.log2().abs().max(1.0));
            assert!((cbrt(x) - x.cbrt()).abs() <= 4.0 * f64::EPSILON * x.cbrt());
            let p = x.powf(1.7);
            assert!((powf(x, 1.7) - p).abs() <= 8.0 * f64::EPSILON * p);
            x *= 1.07;
        }
        assert!((atan2(1.0_f64, -1.0) - 0.75 * core::f64::consts::PI).abs() < 1e-15);
        assert!((hypot(3.0_f32, 4.0) - 5.0).abs() < 1e-6);
        assert!((exp_m1(1e-10_f64) - 1e-10).abs() < 1e-19);
        assert!((ln_1p(1e-10_f64) - 1e-10).abs() < 1e-19);
    }

    #[test]
    fn exact_values_are_pinned() {
        // Bit patterns of a sample of results. They must never change: generated worlds depend
        // on them. (They are libm's results, which are within an ulp of the true values but not
        // always the correctly rounded ones; exp(1.0), for instance, is one ulp high.)
        let actual64 = [
            sin(1.0_f64),
            cos(1.0_f64),
            tan(1.0_f64),
            asin(0.5_f64),
            acos(0.5_f64),
            atan(1.0_f64),
            atan2(1.0_f64, 2.0),
            sinh(1.0_f64),
            cosh(1.0_f64),
            tanh(1.0_f64),
            exp(1.0_f64),
            exp2(0.5_f64),
            exp_m1(0.5_f64),
            ln(2.0_f64),
            ln_1p(0.5_f64),
            log2(3.0_f64),
            log10(3.0_f64),
            cbrt(2.0_f64),
            powf(2.0_f64, 0.3),
            hypot(1.0_f64, 2.0),
            sin(1.0e6_f64),
            cos(-123.456_f64),
        ]
        .map(f64::to_bits);
        let actual32 = [
            sin(1.0_f32),
            cos(1.0_f32),
            tan(1.0_f32),
            atan2(1.0_f32, 2.0),
            exp(1.0_f32),
            ln(2.0_f32),
            powf(2.0_f32, 0.3),
            cbrt(2.0_f32),
        ]
        .map(f32::to_bits);
        assert!(
            actual64 == PINNED_F64,
            "f64 results changed: {actual64:#018x?}"
        );
        assert!(
            actual32 == PINNED_F32,
            "f32 results changed: {actual32:#010x?}"
        );
    }

    const PINNED_F64: [u64; 22] = [
        0x3fea_ed54_8f09_0cee, // sin(1)
        0x3fe1_4a28_0fb5_068c, // cos(1)
        0x3ff8_eb24_5cbe_e3a6, // tan(1)
        0x3fe0_c152_382d_7366, // asin(0.5)
        0x3ff0_c152_382d_7366, // acos(0.5)
        0x3fe9_21fb_5444_2d18, // atan(1)
        0x3fdd_ac67_0561_bb4f, // atan2(1, 2)
        0x3ff2_cd9f_c44e_b982, // sinh(1)
        0x3ff8_b075_51d9_f551, // cosh(1)
        0x3fe8_5efa_b514_f394, // tanh(1)
        0x4005_bf0a_8b14_576a, // exp(1)
        0x3ff6_a09e_667f_3bcd, // exp2(0.5)
        0x3fe4_c253_1c3c_0d38, // exp_m1(0.5)
        0x3fe6_2e42_fefa_39ef, // ln(2)
        0x3fd9_f323_ecbf_984c, // ln_1p(0.5)
        0x3ff9_5c01_a39f_bd68, // log2(3)
        0x3fde_8927_964f_d5fd, // log10(3)
        0x3ff4_28a2_f98d_728b, // cbrt(2)
        0x3ff3_b2c4_7bff_8329, // powf(2, 0.3)
        0x4001_e377_9b97_f4a8, // hypot(1, 2)
        0xbfd6_664b_2568_d867, // sin(1e6)
        0xbfe3_07e5_980a_1558, // cos(-123.456)
    ];
    const PINNED_F32: [u32; 8] = [
        0x3f57_6aa4, // sin(1)
        0x3f0a_5140, // cos(1)
        0x3fc7_5923, // tan(1)
        0x3eed_6338, // atan2(1, 2)
        0x402d_f854, // exp(1)
        0x3f31_7218, // ln(2)
        0x3f9d_9624, // powf(2, 0.3)
        0x3fa1_4518, // cbrt(2)
    ];

    #[test]
    fn powi_matches_repeated_multiplication() {
        assert_eq!(powi(2.0_f64, 10), 1024.0);
        assert_eq!(powi(2.0_f64, -2), 0.25);
        assert_eq!(powi(-3.0_f32, 3), -27.0);
        assert_eq!(powi(0.0_f64, 0), 1.0);
        assert_eq!(powi(f64::NAN, 0), 1.0);
        assert_eq!(powi(1.5_f64, 5), 1.5 * 1.5 * 1.5 * 1.5 * 1.5);
        assert_eq!(powi(2.0_f64, i32::MIN), 0.0);
    }

    /// Reference for [`min`] and [`max`] on numbers: the IEEE total order, in which `-0.0` is
    /// below `+0.0` and everything else is ordered by value.
    fn total_order_min_max(a: f64, b: f64) -> (f64, f64) {
        if a.total_cmp(&b).is_le() {
            (a, b)
        } else {
            (b, a)
        }
    }

    const SPECIAL: [f64; 16] = [
        f64::NEG_INFINITY,
        f64::MIN,
        -2.5,
        -1.0,
        -f64::MIN_POSITIVE,
        -5.0e-324,
        -0.0,
        0.0,
        5.0e-324,
        f64::MIN_POSITIVE,
        f64::EPSILON,
        1.0,
        2.5,
        1.0e300,
        f64::MAX,
        f64::INFINITY,
    ];

    #[test]
    fn min_max_fix_the_sign_of_zero() {
        let (negative, positive) = ((-0.0_f64).to_bits(), 0.0_f64.to_bits());
        assert_ne!(negative, positive);
        // Both operand orders give the same zero. (`f64::min` and `f64::max` leave it open.)
        assert_eq!(min(0.0_f64, -0.0).to_bits(), negative);
        assert_eq!(min(-0.0_f64, 0.0).to_bits(), negative);
        assert_eq!(max(0.0_f64, -0.0).to_bits(), positive);
        assert_eq!(max(-0.0_f64, 0.0).to_bits(), positive);
        assert_eq!(min(0.0_f64, 0.0).to_bits(), positive);
        assert_eq!(max(-0.0_f64, -0.0).to_bits(), negative);
        assert_eq!(min(0.0_f32, -0.0).to_bits(), (-0.0_f32).to_bits());
        assert_eq!(min(-0.0_f32, 0.0).to_bits(), (-0.0_f32).to_bits());
        assert_eq!(max(0.0_f32, -0.0).to_bits(), 0.0_f32.to_bits());
        assert_eq!(max(-0.0_f32, 0.0).to_bits(), 0.0_f32.to_bits());

        // What depends on the sign of a zero no longer depends on how the call was written:
        // "at least zero" of a value that is -0.0, for example `slope * -0.5` on flat ground.
        let slope = 0.0_f64;
        let d = slope * -0.5;
        assert!(d == 0.0 && d.is_sign_negative());
        for at_least_zero in [max(0.0, d), max(d, 0.0)] {
            assert_eq!(at_least_zero.signum(), 1.0);
            assert_eq!(atan2(at_least_zero, -1.0), core::f64::consts::PI);
            assert_eq!(1.0 / at_least_zero, f64::INFINITY);
        }
        for at_most_zero in [min(0.0, d), min(d, 0.0)] {
            assert_eq!(at_most_zero.signum(), -1.0);
            assert_eq!(atan2(at_most_zero, -1.0), -core::f64::consts::PI);
        }
    }

    #[test]
    // Compares with `f64::min` / `f64::max` where their result is specified.
    #[allow(clippy::disallowed_methods)]
    fn min_max_match_the_total_order_and_ignore_nan() {
        let nan = f64::NAN;
        for a in SPECIAL {
            for b in SPECIAL {
                let (low, high) = total_order_min_max(a, b);
                assert_eq!(min(a, b).to_bits(), low.to_bits(), "min({a:e}, {b:e})");
                assert_eq!(max(a, b).to_bits(), high.to_bits(), "max({a:e}, {b:e})");
                // Where the standard library's result is specified, it is the same.
                if a != b {
                    assert_eq!(min(a, b).to_bits(), a.min(b).to_bits());
                    assert_eq!(max(a, b).to_bits(), a.max(b).to_bits());
                }
                // The same in single precision.
                let (af, bf) = (a as f32, b as f32);
                let (low, high) = if af.total_cmp(&bf).is_le() {
                    (af, bf)
                } else {
                    (bf, af)
                };
                assert_eq!(min(af, bf).to_bits(), low.to_bits(), "min({af:e}, {bf:e})");
                assert_eq!(max(af, bf).to_bits(), high.to_bits(), "max({af:e}, {bf:e})");
            }
            // One NaN: the number wins, on either side, with its sign of zero intact.
            assert_eq!(min(a, nan).to_bits(), a.to_bits());
            assert_eq!(min(nan, a).to_bits(), a.to_bits());
            assert_eq!(max(a, nan).to_bits(), a.to_bits());
            assert_eq!(max(nan, a).to_bits(), a.to_bits());
            let af = a as f32;
            assert_eq!(min(af, f32::NAN).to_bits(), af.to_bits());
            assert_eq!(max(f32::NAN, af).to_bits(), af.to_bits());
        }
        assert!(min(nan, nan).is_nan() && max(nan, nan).is_nan());
        assert!(min(f32::NAN, f32::NAN).is_nan() && max(f32::NAN, f32::NAN).is_nan());

        // Arbitrary bit patterns, NaNs of every kind included.
        let mut rng = crate::hash::Rng::new(90, 1);
        let (mut numbers, mut nans) = (0, 0);
        for i in 0..400_000 {
            let mut a = f64::from_bits(rng.next_u64());
            let mut b = f64::from_bits(rng.next_u64());
            // Random patterns are rarely NaN and hardly ever equal; force both now and then.
            match i % 8 {
                0 => a = f64::from_bits(a.to_bits() | 0x7FF8_0000_0000_0000),
                1 => b = f64::from_bits(b.to_bits() | 0x7FF0_0000_0000_0001),
                2 => b = -a,
                3 => b = a,
                _ => {}
            }
            match (a.is_nan(), b.is_nan()) {
                (false, false) => {
                    let (low, high) = total_order_min_max(a, b);
                    assert_eq!(min(a, b).to_bits(), low.to_bits());
                    assert_eq!(max(a, b).to_bits(), high.to_bits());
                    assert_eq!(min(b, a).to_bits(), low.to_bits());
                    assert_eq!(max(b, a).to_bits(), high.to_bits());
                    numbers += 1;
                }
                (true, false) | (false, true) => {
                    let number = if a.is_nan() { b } else { a };
                    assert_eq!(min(a, b).to_bits(), number.to_bits());
                    assert_eq!(max(a, b).to_bits(), number.to_bits());
                    nans += 1;
                }
                (true, true) => {
                    assert!(min(a, b).is_nan() && max(a, b).is_nan());
                    nans += 1;
                }
            }
        }
        assert!(numbers > 200_000 && nans > 50_000, "{numbers} {nans}");
    }

    #[test]
    // Compares with `f64::clamp` where its result is specified.
    #[allow(clippy::disallowed_methods)]
    fn clamp_fixes_the_sign_of_zero_and_passes_nan() {
        let bits = f64::to_bits;
        assert_eq!(clamp(-3.0_f64, -2.0, 1.0), -2.0);
        assert_eq!(clamp(0.25_f64, -2.0, 1.0), 0.25);
        assert_eq!(clamp(2.0_f64, -2.0, 1.0), 1.0);
        assert_eq!(clamp(7.0_f32, -2.0, 1.0), 1.0);
        assert_eq!(clamp(f64::INFINITY, -2.0, 1.0), 1.0);
        assert_eq!(clamp(f64::NEG_INFINITY, -2.0, 1.0), -2.0);
        assert_eq!(clamp(5.0_f64, f64::NEG_INFINITY, f64::INFINITY), 5.0);
        assert_eq!(clamp(5.0_f64, 3.0, 3.0), 3.0);
        assert!(clamp(f64::NAN, -2.0, 1.0).is_nan());
        assert!(clamp(f32::NAN, -2.0, 1.0).is_nan());

        // The result is inside the interval down to the sign of a zero.
        assert_eq!(bits(clamp(-0.0, 0.0, 1.0)), bits(0.0));
        assert_eq!(bits(clamp(0.0, -1.0, -0.0)), bits(-0.0));
        assert_eq!(bits(clamp(-1.0, -0.0, 1.0)), bits(-0.0));
        assert_eq!(bits(clamp(1.0, -1.0, 0.0)), bits(0.0));
        assert_eq!(bits(clamp(1.0, -0.0, -0.0)), bits(-0.0));
        assert_eq!(bits(clamp(-1.0, 0.0, 0.0)), bits(0.0));
        // A zero that is inside keeps its sign.
        assert_eq!(bits(clamp(-0.0, -1.0, 1.0)), bits(-0.0));
        assert_eq!(bits(clamp(0.0, -1.0, 1.0)), bits(0.0));
        assert_eq!(bits(clamp(-0.0, -0.0, 0.0)), bits(-0.0));
        assert_eq!(bits(clamp(0.0, -0.0, 0.0)), bits(0.0));
        assert_eq!(clamp(-0.0_f32, 0.0, 1.0).to_bits(), 0.0_f32.to_bits());
        // The zeros in the wrong order are accepted as bounds (they compare equal); `hi` wins.
        for x in [-5.0, -0.0, 0.0, 5.0] {
            assert_eq!(bits(clamp(x, 0.0, -0.0)), bits(-0.0), "{x}");
        }

        // Defined as min(max(x, lo), hi), which the total order pins down completely.
        for x in SPECIAL {
            for lo in SPECIAL {
                for hi in SPECIAL {
                    if lo <= hi {
                        let (_, raised) = total_order_min_max(x, lo);
                        let (expected, _) = total_order_min_max(raised, hi);
                        let actual = clamp(x, lo, hi);
                        assert_eq!(bits(actual), bits(expected), "clamp({x:e}, {lo:e}, {hi:e})");
                        assert_eq!(bits(actual), bits(min(max(x, lo), hi)));
                        // Where the standard library's result is specified, it is the same.
                        if actual != 0.0 {
                            assert_eq!(bits(actual), bits(x.clamp(lo, hi)));
                        }
                    }
                }
            }
        }
    }

    #[test]
    #[should_panic(expected = "det::clamp: lo > hi")]
    fn clamp_rejects_reversed_bounds() {
        let _ = clamp(0.0_f64, 1.0, -1.0);
    }

    #[test]
    #[should_panic(expected = "det::clamp: lo > hi or a bound is NaN")]
    fn clamp_rejects_nan_bounds() {
        let _ = clamp(0.0_f32, f32::NAN, 1.0);
    }

    /// Running hash of results. NaNs count as one value because their payload is unspecified
    /// (see the module documentation); every other bit pattern counts, zero signs included.
    #[derive(Default)]
    struct Digest(u64);

    impl Digest {
        fn f64(&mut self, x: f64) {
            let bits = if x.is_nan() {
                0x7FF8_0000_0000_0000
            } else {
                x.to_bits()
            };
            self.0 = crate::hash::hash_combine(self.0, bits);
        }

        fn f32(&mut self, x: f32) {
            let bits = if x.is_nan() { 0x7FC0_0000 } else { x.to_bits() };
            self.0 = crate::hash::hash_combine(self.0, u64::from(bits));
        }
    }

    /// Inputs of the golden test: special values, a dense linear sweep where the functions
    /// are most interesting, and a logarithmic sweep over the whole range, both signs.
    fn golden_inputs(log_factor: f64) -> Vec<f64> {
        use core::f64::consts::{FRAC_PI_2, PI};
        let mut inputs = vec![
            0.0,
            -0.0,
            5.0e-324,
            -5.0e-324,
            f64::MIN_POSITIVE,
            -f64::MIN_POSITIVE,
            f64::EPSILON,
            0.5,
            -0.5,
            1.0,
            -1.0,
            2.0,
            -2.0,
            PI,
            -PI,
            FRAC_PI_2,
            1.0e10,
            -1.0e10,
            1.0e300,
            -1.0e300,
            f64::MAX,
            f64::MIN,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::NAN,
        ];
        if log_factor < 100.0 {
            inputs.extend((-1000..=1000).map(|i| f64::from(i) * 0.0137));
        } else {
            inputs.extend((-10..=10).map(|i| f64::from(i) * 0.7));
        }
        let mut magnitude = 1.0e-300;
        while magnitude < 1.0e300 {
            inputs.extend([magnitude, -magnitude]);
            magnitude *= log_factor;
        }
        inputs
    }

    /// Golden test over every function of this module (contract §4.1): a grid of inputs per
    /// function, hashed. A change of any result bit, for example through a `libm` update or a
    /// changed formula, changes a hash. Pinned on x86_64 Windows; the results are pure IEEE
    /// arithmetic in `libm`, so every 64-bit target must agree.
    #[test]
    fn golden_hash_of_every_function() {
        const UNARY: [&str; 17] = [
            "sin", "cos", "tan", "asin", "acos", "atan", "sinh", "cosh", "tanh", "exp", "exp2",
            "exp_m1", "ln", "ln_1p", "log2", "log10", "cbrt",
        ];
        let unary64: [fn(f64) -> f64; 17] = [
            sin, cos, tan, asin, acos, atan, sinh, cosh, tanh, exp, exp2, exp_m1, ln, ln_1p, log2,
            log10, cbrt,
        ];
        let unary32: [fn(f32) -> f32; 17] = [
            sin, cos, tan, asin, acos, atan, sinh, cosh, tanh, exp, exp2, exp_m1, ln, ln_1p, log2,
            log10, cbrt,
        ];
        const BINARY: [&str; 5] = ["atan2", "powf", "hypot", "min", "max"];
        let binary64: [fn(f64, f64) -> f64; 5] = [atan2, powf, hypot, min, max];
        let binary32: [fn(f32, f32) -> f32; 5] = [atan2, powf, hypot, min, max];

        let dense = golden_inputs(1.9);
        let coarse = golden_inputs(1.0e25);
        let mut actual: Vec<(&str, u64, u64)> = Vec::new();
        for (index, name) in UNARY.into_iter().enumerate() {
            let (mut d64, mut d32) = (Digest::default(), Digest::default());
            for &x in &dense {
                d64.f64(unary64[index](x));
                d32.f32(unary32[index](x as f32));
            }
            actual.push((name, d64.0, d32.0));
        }
        let (mut d64, mut d32) = (Digest::default(), Digest::default());
        for &x in &dense {
            let (s, c) = sin_cos(x);
            d64.f64(s);
            d64.f64(c);
            let (s, c) = sin_cos(x as f32);
            d32.f32(s);
            d32.f32(c);
        }
        actual.push(("sin_cos", d64.0, d32.0));
        for (index, name) in BINARY.into_iter().enumerate() {
            let (mut d64, mut d32) = (Digest::default(), Digest::default());
            for &a in &coarse {
                for &b in &coarse {
                    d64.f64(binary64[index](a, b));
                    d32.f32(binary32[index](a as f32, b as f32));
                }
            }
            actual.push((name, d64.0, d32.0));
        }
        let bounds = [
            f64::NEG_INFINITY,
            -2.5,
            -1.0,
            -0.0,
            0.0,
            0.5,
            1.0,
            1.0e300,
            f64::INFINITY,
        ];
        let (mut d64, mut d32) = (Digest::default(), Digest::default());
        for &x in &coarse {
            for lo in bounds {
                for hi in bounds.into_iter().filter(|&hi| lo <= hi) {
                    d64.f64(clamp(x, lo, hi));
                    d32.f32(clamp(x as f32, lo as f32, hi as f32));
                }
            }
        }
        actual.push(("clamp", d64.0, d32.0));
        let (mut d64, mut d32) = (Digest::default(), Digest::default());
        let exponents = (-40..=40).chain([1000, -1000, i32::MAX, i32::MIN]);
        for n in exponents {
            for &x in &coarse {
                d64.f64(powi(x, n));
                d32.f32(powi(x as f32, n));
            }
        }
        actual.push(("powi", d64.0, d32.0));
        let mut d64 = Digest::default();
        let third = (1.0_f64 / 3.0).sqrt();
        let axes = [
            DVec3::X,
            DVec3::Y,
            DVec3::Z,
            DVec3::new(0.6, 0.0, -0.8),
            DVec3::splat(third),
        ];
        for axis in axes {
            for i in -500..=500 {
                let q = dquat_from_axis_angle(axis, f64::from(i) * 0.0137);
                for component in q.to_array() {
                    d64.f64(component);
                }
            }
        }
        actual.push(("dquat_from_axis_angle", d64.0, 0));

        let listing: String = actual
            .iter()
            .map(|(name, h64, h32)| format!("(\"{name}\", {h64:#018x}, {h32:#018x}),\n"))
            .collect();
        assert!(
            actual == GOLDEN,
            "det results changed; new table:\n{listing}"
        );
    }

    /// `(function, hash of the f64 results, hash of the f32 results)`. Generated worlds depend
    /// on these results: never re-pin without a recorded reason.
    const GOLDEN: &[(&str, u64, u64)] = &[
        ("sin", 0x40ae_84ab_e6db_3371, 0xceba_34be_54d0_0a62),
        ("cos", 0xf410_19d7_e675_3a0a, 0x2056_cb82_8492_4272),
        ("tan", 0x994e_88c1_0e16_0447, 0xb52d_458e_81f1_3eb5),
        ("asin", 0xdc5b_06d7_f48c_39d7, 0xf95c_d957_a696_2a3c),
        ("acos", 0x0b49_4de5_e9d4_8345, 0x6a21_35e2_1102_7db5),
        ("atan", 0xd175_ac1a_723f_12fd, 0x9be5_7c23_8659_46d2),
        ("sinh", 0x0cb5_dcd1_9505_c311, 0xffec_f71c_ffbb_9f6d),
        ("cosh", 0xff1d_8c29_c98b_7f24, 0x1983_d493_c90f_3ce6),
        ("tanh", 0x7811_54d0_f8c2_296d, 0xec63_2201_0109_0b1b),
        ("exp", 0x721b_7020_eb21_3b9e, 0x42f0_d8ea_79ee_e530),
        ("exp2", 0x5059_12e3_5fa7_5072, 0xa791_b59c_d8f7_9efd),
        ("exp_m1", 0x7a34_c55b_dfc8_b088, 0x68b3_10e0_764a_5916),
        ("ln", 0x56b3_9d1c_8704_f7a2, 0x0123_8541_f70b_eea2),
        ("ln_1p", 0xa274_4ab7_3c55_4ee6, 0x80c6_f8d5_bcfc_b25c),
        ("log2", 0x8f50_1aa8_b45c_6b21, 0x1bc2_5473_cc8d_3235),
        ("log10", 0xda55_fedf_dc94_8270, 0x5584_bfb4_136e_fde2),
        ("cbrt", 0xb43a_86b2_25e4_e72e, 0xf050_d439_d4ce_7041),
        ("sin_cos", 0x756d_5890_bf9b_9d17, 0x9d44_d2b7_9455_0ad1),
        ("atan2", 0xb57f_d16f_d24d_d9b8, 0x81f8_9f2f_56ec_1ce3),
        ("powf", 0x2629_53d9_9eb5_8ebd, 0x9a0c_516c_b786_57eb),
        ("hypot", 0x645f_5bb4_2fa0_1bce, 0x4611_651e_e3bc_0024),
        ("min", 0xecb3_3f9b_def8_85f9, 0x0e66_74a8_c876_f659),
        ("max", 0x1465_90e0_e87c_c5bc, 0xbe9e_dba7_a88c_f5b6),
        ("clamp", 0x0263_c10c_474e_ee87, 0xf676_19c6_9b5a_5e2b),
        ("powi", 0x4804_9f92_9ff1_07f0, 0xe8e9_e3d0_41b5_94e8),
        ("dquat_from_axis_angle", 0x0556_9455_3b32_dca6, 0),
    ];

    #[test]
    fn axis_angle_quaternion_rotates_right_handed() {
        let q = dquat_from_axis_angle(DVec3::Z, core::f64::consts::FRAC_PI_2);
        let v = q * DVec3::X;
        assert!((v - DVec3::Y).length() < 1e-15, "{v:?}");
        assert!((q.length() - 1.0).abs() < 1e-15);
        let reference = DQuat::from_axis_angle(DVec3::new(0.6, 0.0, 0.8), 1.234);
        let q = dquat_from_axis_angle(DVec3::new(0.6, 0.0, 0.8), 1.234);
        assert!(q.abs_diff_eq(reference, 1e-15));
    }
}
