//! Kepler orbits: classical elements, Kepler's equation, position and velocity.
//!
//! An orbit is given relative to the parent's centre in the parent's *inertial* axes, which are
//! the system axes (no frame of the model rotates its children's orbits). The reference plane is
//! the system's XZ plane with north `+Y`; prograde motion runs from `+X` towards `-Z`
//! (counter-clockwise seen from `+Y`). Angles are radians, distances metres, times seconds
//! since the epoch.

use core::f64::consts::{PI, TAU};

use cp_core::det;
use cp_core::math::DVec3;

/// Classical (Keplerian) elements of an orbit around the parent, valid at the epoch `t = 0`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OrbitElements {
    /// Semi-major axis `a` in metres; `> 0`.
    pub semi_major_axis: f64,
    /// Eccentricity `e`; `0 <= e < 1` (closed orbits only).
    pub eccentricity: f64,
    /// Inclination `i` of the orbit plane against the reference plane (system XZ), radians in
    /// `[0, π]`. Above `π/2` the orbit is retrograde.
    pub inclination: f64,
    /// Longitude of the ascending node `Ω`, radians: the angle in the reference plane from `+X`
    /// to the point where the orbit crosses it northwards, measured prograde (towards `-Z`).
    pub ascending_node: f64,
    /// Argument of periapsis `ω`, radians: the angle in the orbit plane from the ascending node
    /// to the periapsis, measured in the direction of motion.
    pub periapsis_argument: f64,
    /// Mean anomaly `M0` at the epoch, radians.
    pub mean_anomaly_at_epoch: f64,
}

/// Position and velocity relative to the parent's centre, in system axes.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct OrbitState {
    /// Metres.
    pub position: DVec3,
    /// Metres per second.
    pub velocity: DVec3,
}

/// A Kepler orbit with its gravitational parameter: everything needed to evaluate it.
///
/// Evaluation is analytic (no integration), so a state at any time costs about the same and is
/// identical on every machine (contract §2.5): Kepler's equation is solved by
/// [`solve_kepler`], a safeguarded Halley iteration on `cp_core::det` functions that needs at
/// most three evaluations.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct KeplerOrbit {
    elements: OrbitElements,
    mu: f64,
    mean_motion: f64,
    semi_minor_axis: f64,
    /// Unit vector from the parent towards the periapsis (system axes).
    periapsis_direction: DVec3,
    /// Unit vector 90° past the periapsis in the direction of motion.
    ahead_direction: DVec3,
}

/// Classical astronomical axes (reference plane XY, north `+Z`) → system axes (reference plane
/// XZ, north `+Y`): `(x, y, z) ↦ (x, z, -y)`, a proper rotation.
fn astronomical_to_system(x: f64, y: f64, z: f64) -> DVec3 {
    DVec3::new(x, z, -y)
}

impl KeplerOrbit {
    /// The orbit with `elements` around a parent with gravitational parameter `mu` (m³/s²).
    ///
    /// For the relative orbit of two bodies `mu` is the sum of both parameters (Kepler's third
    /// law for two bodies). The elements must be valid (`a > 0`, `0 <= e < 1`, finite angles);
    /// [`crate::SystemDef::new`] checks them for loaded systems.
    #[must_use]
    pub fn new(elements: OrbitElements, mu: f64) -> Self {
        let (sin_node, cos_node) = det::sin_cos(elements.ascending_node);
        let (sin_peri, cos_peri) = det::sin_cos(elements.periapsis_argument);
        let (sin_incl, cos_incl) = det::sin_cos(elements.inclination);
        // The rotation R_z(Ω) R_x(i) R_z(ω) of the perifocal frame, column by column.
        let periapsis_direction = astronomical_to_system(
            cos_node * cos_peri - sin_node * sin_peri * cos_incl,
            sin_node * cos_peri + cos_node * sin_peri * cos_incl,
            sin_peri * sin_incl,
        );
        let ahead_direction = astronomical_to_system(
            -cos_node * sin_peri - sin_node * cos_peri * cos_incl,
            -sin_node * sin_peri + cos_node * cos_peri * cos_incl,
            cos_peri * sin_incl,
        );
        let a = elements.semi_major_axis;
        let e = elements.eccentricity;
        Self {
            elements,
            mu,
            mean_motion: (mu / (a * a * a)).sqrt(),
            semi_minor_axis: a * (1.0 - e * e).sqrt(),
            periapsis_direction,
            ahead_direction,
        }
    }

    /// The elements the orbit was built from.
    #[inline]
    #[must_use]
    pub fn elements(&self) -> &OrbitElements {
        &self.elements
    }

    /// Gravitational parameter of the orbit in m³/s².
    #[inline]
    #[must_use]
    pub fn mu(&self) -> f64 {
        self.mu
    }

    /// Mean motion `n = sqrt(mu / a³)` in radians per second.
    #[inline]
    #[must_use]
    pub fn mean_motion(&self) -> f64 {
        self.mean_motion
    }

    /// Orbital period `2π / n` in seconds.
    #[inline]
    #[must_use]
    pub fn period(&self) -> f64 {
        TAU / self.mean_motion
    }

    /// Semi-minor axis `b = a sqrt(1 - e²)` in metres.
    #[inline]
    #[must_use]
    pub fn semi_minor_axis(&self) -> f64 {
        self.semi_minor_axis
    }

    /// Closest distance to the parent's centre, `a (1 - e)`, metres.
    #[inline]
    #[must_use]
    pub fn periapsis_distance(&self) -> f64 {
        self.elements.semi_major_axis * (1.0 - self.elements.eccentricity)
    }

    /// Largest distance from the parent's centre, `a (1 + e)`, metres.
    #[inline]
    #[must_use]
    pub fn apoapsis_distance(&self) -> f64 {
        self.elements.semi_major_axis * (1.0 + self.elements.eccentricity)
    }

    /// Unit vector from the parent's centre towards the periapsis (system axes).
    #[inline]
    #[must_use]
    pub fn periapsis_direction(&self) -> DVec3 {
        self.periapsis_direction
    }

    /// Unit normal of the orbit plane along the angular momentum (system axes): `+Y` for a
    /// prograde orbit in the reference plane.
    #[inline]
    #[must_use]
    pub fn normal(&self) -> DVec3 {
        self.periapsis_direction.cross(self.ahead_direction)
    }

    /// Mean anomaly at time `t` (seconds since the epoch), reduced to `[-π, π)`.
    #[must_use]
    pub fn mean_anomaly(&self, t: f64) -> f64 {
        let m = self.elements.mean_anomaly_at_epoch + self.mean_motion * t;
        (m + PI).rem_euclid(TAU) - PI
    }

    /// Mean longitude `Ω + ω + M(t)` at time `t`, reduced to `[0, 2π)`: the direction the body
    /// would have on a circular orbit in the reference plane.
    #[must_use]
    pub fn mean_longitude(&self, t: f64) -> f64 {
        let e = &self.elements;
        (e.ascending_node + e.periapsis_argument + e.mean_anomaly_at_epoch + self.mean_motion * t)
            .rem_euclid(TAU)
    }

    /// Eccentric anomaly at time `t`, radians.
    #[must_use]
    pub fn eccentric_anomaly(&self, t: f64) -> f64 {
        solve_kepler(self.mean_anomaly(t), self.elements.eccentricity)
    }

    /// Position and velocity relative to the parent's centre at time `t` (seconds since the
    /// epoch), system axes.
    #[must_use]
    pub fn state(&self, t: f64) -> OrbitState {
        let a = self.elements.semi_major_axis;
        let e = self.elements.eccentricity;
        let (sin_e, cos_e) = det::sin_cos(self.eccentric_anomaly(t));
        let along = a * (cos_e - e);
        let across = self.semi_minor_axis * sin_e;
        // dE/dt = n / (1 - e cos E).
        let rate = self.mean_motion / (1.0 - e * cos_e);
        OrbitState {
            position: self.periapsis_direction * along + self.ahead_direction * across,
            velocity: self.periapsis_direction * (-a * sin_e * rate)
                + self.ahead_direction * (self.semi_minor_axis * cos_e * rate),
        }
    }
}

/// Most evaluations of Kepler's equation [`solve_kepler`] makes: a safety net that ends the
/// iteration should the bisection fallback ever take over. From [`kepler_start`] the iteration
/// needs at most three for every eccentricity below 1 and never bisects (see
/// `kepler_solver_needs_at_most_three_evaluations`).
const KEPLER_MAX_EVALUATIONS: u32 = 100;

/// Solves Kepler's equation `E - e sin E = M` for the eccentric anomaly `E` (radians).
///
/// `mean_anomaly` is in radians and need not be reduced: the result solves the equation for the
/// value given. `0 <= eccentricity < 1`; any other eccentricity (or NaN) gives NaN. A mean
/// anomaly of zero (of either sign) or a non-finite one is returned unchanged.
///
/// Method: Mikkola's cubic approximation as the starting value (within 4e-3 rad of the root),
/// then Halley steps. The function `f(E) = E - e sin E - M` is increasing and its root lies in
/// `[M - e, M + e]`. Every step stays inside that bracket, which shrinks to the points
/// evaluated so far: a step past an evaluated point is replaced by bisection, so the iteration
/// converges for every input, while a step onto or past one of the two initial ends evaluates
/// that end (near `E = ±π/2` the root lies within rounding of it). The iteration stops as soon
/// as the residual `f` is within the rounding error of its own evaluation or the Newton
/// correction is below two ulps, and returns that point with the Newton correction applied.
/// That takes at most three evaluations of `sin_cos` (two Halley steps and the check), for
/// small eccentricities usually two, so the cost is bounded. The residual of the result is
/// about one ulp of `max(1, |E|)`.
///
/// Deterministic (contract §2.5): only `cp_core::det` functions and exact IEEE operations, and
/// the result is odd in the mean anomaly bit for bit (`E(-M) = -E(M)` for `|M| <= π`).
#[must_use]
pub fn solve_kepler(mean_anomaly: f64, eccentricity: f64) -> f64 {
    solve_kepler_counted(mean_anomaly, eccentricity).0
}

/// [`solve_kepler`] and the number of evaluations of Kepler's equation it made.
fn solve_kepler_counted(mean_anomaly: f64, eccentricity: f64) -> (f64, u32) {
    let (m, e) = (mean_anomaly, eccentricity);
    if !(0.0..1.0).contains(&e) {
        return (f64::NAN, 0);
    }
    // E(0) = 0 exactly; returning `m` keeps the sign of a zero.
    if e == 0.0 || m == 0.0 || !m.is_finite() {
        return (m, 0);
    }
    // f(M - e) = -e (1 + sin(M - e)) <= 0 <= e (1 - sin(M + e)) = f(M + e). The two initial
    // ends are analytic bounds, not evaluated points: when the root lies within rounding of one
    // of them (E near ±π/2), a step onto that end is taken rather than rejected.
    let (mut lo, mut hi) = (m - e, m + e);
    let (mut lo_evaluated, mut hi_evaluated) = (false, false);
    let mut x = det::clamp(kepler_start(m, e), lo, hi);
    for evaluations in 1..=KEPLER_MAX_EVALUATIONS {
        let (sin_x, cos_x) = det::sin_cos(x);
        let f = x - e * sin_x - m;
        if f == 0.0 {
            return (x, evaluations);
        }
        // f' = 1 - e cos x lies in [1 - e, 1 + e], so it is positive.
        let slope = 1.0 - e * cos_x;
        let newton = f / slope;
        // A bound of the rounding error of `f`: each of its three terms and two subtractions
        // contributes at most an ulp. The floor stands for the absolute rounding error of
        // subnormal residuals.
        let noise =
            2.0 * f64::EPSILON * (x.abs() + (e * sin_x).abs() + m.abs()) + f64::MIN_POSITIVE;
        if f.abs() <= noise || newton.abs() <= 2.0 * f64::EPSILON * x.abs() {
            return (det::clamp(x - newton, lo, hi), evaluations);
        }
        if f > 0.0 {
            hi = x;
            hi_evaluated = true;
        } else {
            lo = x;
            lo_evaluated = true;
        }
        // Halley's step with f'' = e sin x. Its denominator is kept above half the slope, so
        // the step is never more than twice Newton's.
        let denominator = slope - 0.5 * f * (e * sin_x) / slope;
        let step = if denominator > 0.5 * slope {
            f / denominator
        } else {
            newton
        };
        let halley = x - step;
        let next = if halley > lo && halley < hi {
            halley
        } else if halley <= lo && !lo_evaluated {
            lo
        } else if halley >= hi && !hi_evaluated {
            hi
        } else {
            0.5 * (lo + hi)
        };
        // A step that does not move (bisection between two neighbouring floats) leaves
        // nothing to improve.
        if next == x {
            return (x, evaluations);
        }
        x = next;
    }
    (x, KEPLER_MAX_EVALUATIONS)
}

/// Starting value for [`solve_kepler`]: Mikkola's cubic approximation (S. Mikkola, "A cubic
/// approximation for Kepler's equation", Celestial Mechanics 40, 1987), within 4e-3 rad of the
/// root for every `0 < e < 1`, and odd in `m` for `|m| <= π`.
fn kepler_start(m: f64, e: f64) -> f64 {
    // The approximation is built for anomalies in [-π, π]; E(M + 2πk) = E(M) + 2πk.
    if m.abs() > PI {
        let reduced = (m + PI).rem_euclid(TAU) - PI;
        return kepler_start(reduced, e) + (m - reduced);
    }
    let scale = 4.0 * e + 0.5;
    let alpha = (1.0 - e) / scale;
    let beta = 0.5 * m / scale;
    // s solves the cubic s³ + 3αs = 2β: with z³ = |β| + sqrt(β² + α³) it is z - α/z, written
    // as 2βz² / (z⁴ + αz² + α²) because z² - α cancels for small |β|.
    let z = det::cbrt(beta.abs() + (beta * beta + alpha * alpha * alpha).sqrt());
    let z2 = z * z;
    let s = 2.0 * beta * z2 / (z2 * z2 + alpha * z2 + alpha * alpha);
    // Mikkola's correction of the cubic's largest error term.
    let s2 = s * s;
    let s = s - 0.078 * s2 * s2 * s / (1.0 + e);
    // E = M + e (3s - 4s³), where 3s - 4s³ approximates sin E.
    m + e * s * (3.0 - 4.0 * s * s)
}

/// Largest amount by which the true anomaly runs ahead of the mean anomaly on an orbit of this
/// eccentricity (the maximum of the equation of the centre `ν - M`), radians: about `2e` for
/// small eccentricities (0.0400 rad, 2.29°, at `e = 0.02`), 1.03 rad at `e = 0.5`, π/2 at about
/// `e = 0.72`. It bounds the libration in longitude of a tidally locked body.
///
/// `0 <= eccentricity < 1`.
pub(crate) fn max_equation_of_centre(eccentricity: f64) -> f64 {
    let e = eccentricity;
    if e <= 0.0 {
        return 0.0;
    }
    // ν - M is largest where dν/dM = (1 + e cos ν)² / (1 - e²)^(3/2) = 1, i.e. where
    // 1 + e cos ν = (1 - e²)^(3/4).
    let one_minus_e2 = (1.0 - e) * (1.0 + e);
    let root = det::powf(one_minus_e2, 0.75);
    let cos_nu = (root - 1.0) / e;
    let sin_nu = (1.0 - cos_nu * cos_nu).sqrt();
    let nu = det::atan2(sin_nu, cos_nu);
    // The eccentric anomaly of that point, from cos E = (e + cos ν) / (1 + e cos ν) and
    // sin E = sqrt(1 - e²) sin ν / (1 + e cos ν).
    let sin_e = one_minus_e2.sqrt() * sin_nu / root;
    let cos_e = (e + cos_nu) / root;
    let eccentric = det::atan2(sin_e, cos_e);
    nu - (eccentric - e * sin_e)
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::f64::consts::FRAC_PI_2;
    use cp_core::hash::Rng;

    /// Bisection with many more steps than needed: slow and obviously correct.
    fn kepler_reference(m: f64, e: f64) -> f64 {
        let (mut lo, mut hi) = (m - e - 1e-9, m + e + 1e-9);
        for _ in 0..200 {
            let mid = 0.5 * (lo + hi);
            if mid - e * det::sin(mid) - m > 0.0 {
                hi = mid;
            } else {
                lo = mid;
            }
        }
        0.5 * (lo + hi)
    }

    fn elements(a: f64, e: f64, i: f64, node: f64, peri: f64, m0: f64) -> OrbitElements {
        OrbitElements {
            semi_major_axis: a,
            eccentricity: e,
            inclination: i,
            ascending_node: node,
            periapsis_argument: peri,
            mean_anomaly_at_epoch: m0,
        }
    }

    #[test]
    fn kepler_solver_matches_bisection() {
        let eccentricities = [0.0, 1e-6, 0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.97, 0.99, 0.999];
        for &e in &eccentricities {
            for step in -360..=360 {
                let m = f64::from(step) * PI / 360.0;
                let m = if m >= PI { m - TAU } else { m };
                let solved = solve_kepler(m, e);
                let reference = kepler_reference(m, e);
                assert!(
                    (solved - reference).abs() <= 1e-13,
                    "M {m} e {e}: {solved} vs {reference}"
                );
                // A few ulps of the anomaly (the residual itself cancels to that noise).
                let residual = solved - e * det::sin(solved) - m;
                assert!(
                    residual.abs() <= 8.0 * f64::EPSILON * det::max(1.0, solved.abs()),
                    "M {m} e {e}: residual {residual:e}"
                );
            }
        }
        // Random inputs, including near-parabolic ones.
        let mut rng = Rng::new(17, 4);
        for _ in 0..20_000 {
            let m = rng.range_f64(-PI, PI);
            let e = 1.0 - rng.f64() * rng.f64();
            let e = if e >= 1.0 { 0.999_999 } else { e };
            let solved = solve_kepler(m, e);
            let residual = solved - e * det::sin(solved) - m;
            assert!(
                residual.abs() <= 8.0 * f64::EPSILON * det::max(1.0, solved.abs()),
                "M {m} e {e}: residual {residual:e}"
            );
        }
        // Deterministic: bit for bit the same result for the same input.
        assert_eq!(
            solve_kepler(1.234, 0.567).to_bits(),
            solve_kepler(1.234, 0.567).to_bits()
        );
        assert_eq!(solve_kepler(0.5, 0.0), 0.5);
    }

    /// Solves with the evaluation count and checks the count and the residual.
    fn solve_checked(m: f64, e: f64) -> u32 {
        let (solved, evaluations) = solve_kepler_counted(m, e);
        assert!(evaluations <= 3, "M {m:e} e {e}: {evaluations} evaluations");
        let residual = solved - e * det::sin(solved) - m;
        assert!(
            residual.abs() <= 2.0 * f64::EPSILON * det::max(1.0, solved.abs()),
            "M {m:e} e {e}: residual {residual:e}"
        );
        evaluations
    }

    #[test]
    fn kepler_solver_needs_at_most_three_evaluations() {
        // A dense grid of eccentricities up to the parabolic limit and of mean anomalies.
        let mut eccentricities: Vec<f64> = (0..1000).map(|k| f64::from(k) * 0.001).collect();
        eccentricities.extend([
            0.9995,
            0.9999,
            0.999_999,
            1.0 - 1e-9,
            1.0 - f64::EPSILON / 2.0,
        ]);
        for &e in &eccentricities {
            for k in 0..1024 {
                solve_checked(-PI + TAU * (f64::from(k) + 0.5) / 1024.0, e);
            }
            for m in [PI, -PI, PI - 1e-12, 1e-6, -1e-12, 1e-300, f64::from_bits(1)] {
                solve_checked(m, e);
            }
        }
        // Roots within rounding of an end of the initial bracket [M - e, M + e], which is where
        // E = ±π/2 + 2πk; with large anomalies the window of such roots is wide.
        let mut rng = Rng::new(19, 3);
        for k in 0..60_000 {
            let e = rng.range_f64(1e-3, 0.999);
            let turns = if k % 2 == 0 {
                0.0
            } else {
                (rng.f64() * 2.0e5).floor()
            };
            let sign = if k % 4 < 2 { 1.0 } else { -1.0 };
            let root = sign * FRAC_PI_2 + TAU * turns + rng.range_f64(-5e-7, 5e-7);
            solve_checked(root - e * det::sin(root), e);
        }
        // Unreduced anomalies (mean motion times seconds of play) and tiny ones.
        for _ in 0..30_000 {
            let e = rng.range_f64(0.0, 0.999);
            solve_checked(rng.range_f64(-1.0e6, 1.0e6), e);
            solve_checked(rng.range_f64(-1e-9, 1e-9), e);
            solve_checked(rng.range_f64(-PI, PI), 1.0 - rng.f64() * rng.f64() * 1e-3);
        }
        // The eccentricities of the home system's orbits and of Hohmann transfers between its
        // planets (Heimat-Cinder 0.46, Heimat-Relikt 0.41): two evaluations for most inputs.
        for (e, mean_at_most) in [(0.02, 2.1), (0.08, 2.4), (0.41, 2.9), (0.46, 2.9)] {
            let total: u32 = (0..10_000)
                .map(|k| solve_checked(-PI + TAU * (f64::from(k) + 0.5) / 10_000.0, e))
                .sum();
            let mean = f64::from(total) / 10_000.0;
            assert!(mean <= mean_at_most, "e {e}: {mean} evaluations on average");
        }
    }

    #[test]
    fn kepler_solver_regressions() {
        // The former Newton iteration rejected its own converged step against the bracket and
        // bisected down to the last bit: 34 evaluations here, up to 56 elsewhere.
        assert!(solve_checked(0.5 * PI, 0.5) <= 3);
        // Roots at an end of the bracket [M - e, M + e]: at E = ±π/2 + 2πk the sine rounds to
        // ±1, so the root is M ± e up to rounding and a step lands on (or just past) that end.
        // It must be taken, not rejected in favour of bisection.
        let e = 0.7;
        for root in [
            FRAC_PI_2 + TAU * 61_250.0,
            -FRAC_PI_2 - TAU * 61_250.0,
            FRAC_PI_2,
        ] {
            let m = root - e * det::sin(root);
            assert!(solve_checked(m, e) <= 3, "root {root}");
            let solved = solve_kepler(m, e);
            assert!(
                (solved - root).abs() <= 2.0 * f64::EPSILON * root.abs(),
                "{solved} vs {root}"
            );
        }
    }

    #[test]
    fn kepler_solver_is_odd_and_rejects_invalid_eccentricities() {
        let mut rng = Rng::new(20, 1);
        for _ in 0..20_000 {
            let e = rng.range_f64(0.0, 1.0);
            let m = rng.range_f64(-PI, PI);
            assert_eq!(
                solve_kepler(-m, e).to_bits(),
                (-solve_kepler(m, e)).to_bits(),
                "M {m} e {e}"
            );
        }
        assert!((solve_kepler(PI, 0.3) - PI).abs() <= 2.0 * f64::EPSILON * PI);
        assert_eq!(solve_kepler(0.0, 0.5).to_bits(), 0.0_f64.to_bits());
        assert_eq!(solve_kepler(-0.0, 0.5).to_bits(), (-0.0_f64).to_bits());
        assert_eq!(solve_kepler(f64::INFINITY, 0.5), f64::INFINITY);
        assert!(solve_kepler(f64::NAN, 0.5).is_nan());
        for e in [-0.1, 1.0, 1.5, f64::NAN, f64::INFINITY] {
            assert!(solve_kepler(1.0, e).is_nan(), "e {e}");
        }
        // The starting value is within 4e-3 rad of the root everywhere.
        let eccentricities = (1..100).map(|k| f64::from(k) * 0.01);
        for e in eccentricities.chain([0.999, 0.999_999]) {
            for k in 0..256 {
                let m = -PI + TAU * (f64::from(k) + 0.5) / 256.0;
                let error = kepler_start(m, e) - kepler_reference(m, e);
                assert!(error.abs() < 4e-3, "M {m} e {e}: {error:e}");
            }
        }
    }

    #[test]
    fn max_equation_of_centre_matches_brute_force() {
        assert_eq!(max_equation_of_centre(0.0), 0.0);
        for e in [1e-4_f64, 0.02, 0.08, 0.2, 0.5, 0.72, 0.9, 0.99] {
            // ν - M sampled densely around the orbit; ν from E by the half-angle formula.
            let (scale_sin, scale_cos) = ((1.0 + e).sqrt(), (1.0 - e).sqrt());
            let mut sampled = 0.0_f64;
            for k in 0..100_000 {
                let m = TAU * f64::from(k) / 100_000.0;
                let (sin_half, cos_half) = det::sin_cos(0.5 * solve_kepler(m, e));
                let nu = 2.0 * det::atan2(scale_sin * sin_half, scale_cos * cos_half);
                sampled = det::max(sampled, nu - m);
            }
            let analytic = max_equation_of_centre(e);
            assert!(analytic >= sampled - 1e-12, "e {e}: {analytic} < {sampled}");
            assert!(
                analytic - sampled < 1e-6 * e,
                "e {e}: {analytic} vs {sampled}"
            );
        }
        // The values the documentation quotes.
        assert!((max_equation_of_centre(0.02) - 0.0400).abs() < 5e-5);
        assert!((max_equation_of_centre(0.5) - 1.03).abs() < 5e-3);
        assert!(max_equation_of_centre(0.71) < FRAC_PI_2);
        assert!(max_equation_of_centre(0.73) > FRAC_PI_2);
    }

    #[test]
    fn circular_orbit_in_the_reference_plane_is_prograde_about_y() {
        let mu = 4.0e9;
        let orbit = KeplerOrbit::new(elements(40_000.0, 0.0, 0.0, 0.0, 0.0, 0.0), mu);
        let start = orbit.state(0.0);
        assert!((start.position - DVec3::new(40_000.0, 0.0, 0.0)).length() < 1e-9);
        let speed = (mu / 40_000.0).sqrt();
        // Prograde: from +X towards -Z.
        assert!((start.velocity - DVec3::new(0.0, 0.0, -speed)).length() < 1e-9);
        assert!((orbit.normal() - DVec3::Y).length() < 1e-15);
        let quarter = orbit.state(0.25 * orbit.period());
        assert!(
            (quarter.position - DVec3::new(0.0, 0.0, -40_000.0)).length() < 1e-6,
            "{:?}",
            quarter.position
        );
        let period = TAU * (det::powi(40_000.0_f64, 3) / mu).sqrt();
        assert!((orbit.period() - period).abs() < 1e-9 * period);
        for k in 0..100 {
            let state = orbit.state(f64::from(k) * 37.0);
            assert!((state.position.length() - 40_000.0).abs() < 1e-8);
            assert!((state.velocity.length() - speed).abs() < 1e-9);
            assert!(state.position.dot(state.velocity).abs() < 1e-6);
        }
    }

    #[test]
    fn node_and_inclination_follow_the_documented_convention() {
        // A polar orbit whose ascending node is +X: it passes the periapsis at the node and moves
        // north (+Y) from there.
        let orbit = KeplerOrbit::new(elements(1.0e6, 0.0, 0.5 * PI, 0.0, 0.0, 0.0), 1.0e10);
        let start = orbit.state(0.0);
        assert!((start.position.normalize() - DVec3::X).length() < 1e-12);
        assert!(start.velocity.normalize().dot(DVec3::Y) > 0.999_999);
        // Ascending node at 90°: the node lies at -Z (prograde from +X).
        let orbit = KeplerOrbit::new(elements(1.0e6, 0.0, 0.3, 0.5 * PI, 0.0, 0.0), 1.0e10);
        let start = orbit.state(0.0);
        assert!((start.position.normalize() - DVec3::NEG_Z).length() < 1e-12);
        assert!(start.velocity.y > 0.0);
        let normal = orbit.normal();
        assert!((normal.dot(DVec3::Y) - det::cos(0.3_f64)).abs() < 1e-12);
        // Retrograde orbits turn the other way.
        let retro = KeplerOrbit::new(elements(1.0e6, 0.0, PI, 0.0, 0.0, 0.0), 1.0e10);
        assert!((retro.normal() - DVec3::NEG_Y).length() < 1e-12);
        // The periapsis lies ω past the node in the direction of motion.
        let orbit = KeplerOrbit::new(elements(1.0e6, 0.4, 0.0, 0.0, 0.5 * PI, 0.0), 1.0e10);
        assert!((orbit.periapsis_direction() - DVec3::NEG_Z).length() < 1e-12);
        let start = orbit.state(0.0);
        assert!((start.position.length() - orbit.periapsis_distance()).abs() < 1e-6);
        let half = orbit.state(0.5 * orbit.period());
        assert!((half.position.length() - orbit.apoapsis_distance()).abs() < 1e-6);
    }

    #[test]
    fn velocity_is_the_time_derivative_of_position() {
        let mut rng = Rng::new(18, 1);
        for _ in 0..500 {
            let a = rng.range_f64(1.0e4, 2.0e7);
            let e = rng.range_f64(0.0, 0.95);
            let orbit = KeplerOrbit::new(
                elements(
                    a,
                    e,
                    rng.range_f64(0.0, PI),
                    rng.range_f64(-PI, PI),
                    rng.range_f64(-PI, PI),
                    rng.range_f64(-PI, PI),
                ),
                rng.range_f64(1.0e6, 1.0e12),
            );
            let t = rng.range_f64(-1.0e6, 1.0e6);
            // Central difference with a step that is small against the period.
            let h = orbit.period() * 1e-7;
            let numeric = (orbit.state(t + h).position - orbit.state(t - h).position) / (2.0 * h);
            let analytic = orbit.state(t).velocity;
            assert!(
                (numeric - analytic).length() <= 1e-6 * analytic.length(),
                "a {a} e {e}: {numeric:?} vs {analytic:?}"
            );
        }
    }

    #[test]
    fn analytic_states_conserve_energy_and_angular_momentum() {
        let mu = 7.0e11;
        let orbit = KeplerOrbit::new(elements(8.0e6, 0.3, 0.2, 1.0, 2.0, 0.5), mu);
        let first = orbit.state(0.0);
        let momentum = first.position.cross(first.velocity);
        for k in 0..200 {
            let state = orbit.state(f64::from(k) * 1234.5);
            let r = state.position.length();
            // Vis-viva: v² = mu (2/r - 1/a).
            let vis_viva = mu * (2.0 / r - 1.0 / 8.0e6);
            assert!((state.velocity.length_squared() - vis_viva).abs() <= 1e-10 * vis_viva);
            let h = state.position.cross(state.velocity);
            assert!((h - momentum).length() <= 1e-10 * momentum.length());
            assert!(r >= orbit.periapsis_distance() * (1.0 - 1e-12));
            assert!(r <= orbit.apoapsis_distance() * (1.0 + 1e-12));
        }
        // The angular momentum is along the documented normal.
        assert!((momentum.normalize() - orbit.normal()).length() < 1e-12);
    }

    #[test]
    fn long_times_stay_on_the_orbit() {
        let orbit = KeplerOrbit::new(elements(120_000.0, 0.02, 0.1, 1.5, 0.0, 4.7), 4.1e9);
        let period = orbit.period();
        // A billion seconds later (thirty years of play), whole periods away from t = 1000 s.
        let turns = (1.0e9 / period).floor();
        let near = orbit.state(1000.0);
        let far = orbit.state(1000.0 + turns * period);
        assert!(
            (near.position - far.position).length() < 1e-3,
            "{:?} vs {:?}",
            near.position,
            far.position
        );
        let m = orbit.mean_anomaly(5.0e11);
        assert!((-PI..PI).contains(&m));
    }
}
