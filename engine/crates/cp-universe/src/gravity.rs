//! Gravity of a body in its body frame: the cube-to-sphere blended potential of cube bodies and
//! the field of a uniform sphere for stars.
//!
//! The cube model is a port of the prototype's tested `src/physics/mod.rs` (`Gravity::field`)
//! with every planet-specific constant replaced by a parameter of the body: the half edge `R`,
//! the surface gravity `g`, `mu = g R²`, and five shape lengths ([`CubeGravityShape`]) that
//! default to the prototype's large-planet values as fractions of `R`.
//!
//! # The cube model
//! The potential is a radial profile `V(ρ)` of one blended distance `ρ(p)`:
//!
//! - near the surface `ρ` is a *rounded cube distance*: the maximum of `|x|, |y|, |z|` with
//!   edges and corners rounded over `edge_rounding` (a C¹ smooth maximum, averaged over the
//!   three pairings so that it is symmetric under axis permutations). Its level sets are cubes,
//!   so gravity is exactly perpendicular to each face away from the edges;
//! - from `outer_start` above the reference surface `ρ` blends (quintic smootherstep over
//!   `outer_width`) into the distance `r` from the centre, so far away the field is spherical;
//! - from `inner_start` below the surface `ρ` blends over `inner_width` into `r - shift`, a
//!   sphere aligned so that the force keeps pointing inwards through the blend;
//! - `V(ρ)` has the constant slope `g` from the core to `outer_start` above the surface, a cubic
//!   blend of the slope from `g` to `mu / ρ²` over the outer band, `-mu / ρ` beyond it, and a
//!   linear core (uniform density) inside `core = R - inner_start - inner_width`.
//!
//! The acceleration is `-V'(ρ) ∇ρ`, carried exactly by forward-mode differentiation, so it is
//! the negative gradient of the potential everywhere (the field is conservative) and both are
//! continuous. At a face centre, from `inner_start` below to `outer_start` above the reference
//! surface, the acceleration is exactly `g` towards the centre; beyond the outer band it is
//! exactly Kepler's `-mu p / r³`.
//!
//! # Properties of the model (measured for Home with the default shape)
//! - In the shell from `inner_start` below to `outer_start` above the reference surface
//!   gravity never exceeds `g`.
//! - Where `ρ` changes from the cube distance to a sphere, it changes faster than the distance
//!   itself, which steepens the field: up to 2.9 g in the inner band about 9 km below the face
//!   centres, and up to about 2.7 g in the outer band 1 to 8 km above edges and corners.
//! - Consequently circular orbits that pass over edges survive only from about 39 km from
//!   the centre, over corners from about 48 km (the field is Kepler's in every direction from
//!   [`CubeGravity::kepler_radius`], 48.8 km). A wider outer band lowers the peak but moves
//!   the lowest orbit further out (1.0 R: 1.5 g, orbits from 54 km over edges).

use cp_core::det;
use cp_core::math::DVec3;

use crate::body::{BodyDef, BodyShape};
use crate::error::GravityError;

/// Potential and acceleration at one point.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct FieldSample {
    /// Gravitational potential in J/kg (m²/s²); zero at infinity, negative everywhere.
    pub potential: f64,
    /// Gravitational acceleration in m/s², in the coordinates of the position: the negative
    /// gradient of `potential`.
    pub acceleration: DVec3,
}

/// The gravity field of one body, evaluated in its body frame (origin at the body centre,
/// metres). Pure data: no global state, identical on every machine.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum GravityField {
    /// A planet or moon.
    Cube(CubeGravity),
    /// A star.
    Sphere(SphereGravity),
}

impl GravityField {
    /// The field of `body`: the blended cube potential with the default
    /// [`CubeGravityShape::for_half_edge`] for planets and moons, a uniform sphere for stars.
    ///
    /// # Errors
    /// When the body's radius or surface gravity is not positive and finite. Bodies of a
    /// [`crate::SystemDef`] are valid; [`crate::SystemDef::gravity`] cannot fail.
    pub fn for_body(body: &BodyDef) -> Result<Self, GravityError> {
        match &body.shape {
            BodyShape::Sphere { radius } => {
                SphereGravity::new(*radius, body.surface_gravity).map(Self::Sphere)
            }
            BodyShape::Cube(cube) => {
                let half_edge = cube.half_edge();
                CubeGravity::new(
                    half_edge,
                    body.surface_gravity,
                    CubeGravityShape::for_half_edge(half_edge),
                )
                .map(Self::Cube)
            }
        }
    }

    /// Potential and acceleration at `position` (body frame, metres from the centre).
    #[inline]
    #[must_use]
    pub fn field(&self, position: DVec3) -> FieldSample {
        match self {
            Self::Cube(cube) => cube.field(position),
            Self::Sphere(sphere) => sphere.field(position),
        }
    }

    /// Gravitational parameter in m³/s².
    #[must_use]
    pub fn mu(&self) -> f64 {
        match self {
            Self::Cube(cube) => cube.mu(),
            Self::Sphere(sphere) => sphere.mu(),
        }
    }
}

/// The lengths that shape the cube field, in metres (names of the prototype in brackets).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CubeGravityShape {
    /// Width over which edges and corners of the cube distance are rounded (`edge_width`).
    pub edge_rounding: f64,
    /// Height above the reference surface where the blend to the spherical field starts
    /// (`space_start`); below it gravity is exactly perpendicular to the faces.
    pub outer_start: f64,
    /// Height range of the outer blend (`space_width`); beyond it the field is Kepler's.
    pub outer_width: f64,
    /// Depth below the reference surface where the blend to the inner sphere starts
    /// (`deep_start`).
    pub inner_start: f64,
    /// Depth range of the inner blend (`deep_width`).
    pub inner_width: f64,
}

impl CubeGravityShape {
    /// The prototype's large-planet parameters as fractions of the half edge `R`:
    /// `edge_rounding = 3R/256`, `outer_start = R/20`, `outer_width = 0.325 R`,
    /// `inner_start = R/8`, `inner_width = 0.625 R`. For Home (`R = 20 480 m`) these are the
    /// prototype's values exactly: 240, 1024, 6656, 2560 and 12 800 m.
    #[must_use]
    pub fn for_half_edge(half_edge: f64) -> Self {
        Self {
            edge_rounding: half_edge * (3.0 / 256.0),
            outer_start: half_edge / 20.0,
            outer_width: half_edge * 0.325,
            inner_start: half_edge / 8.0,
            inner_width: half_edge * 0.625,
        }
    }
}

/// A value with its gradient (forward-mode differentiation).
#[derive(Clone, Copy, Debug)]
struct Dual {
    value: f64,
    grad: DVec3,
}

impl Dual {
    fn negated(self) -> Self {
        Self {
            value: -self.value,
            grad: -self.grad,
        }
    }

    /// `self + (other - self) * weight`.
    fn mix(self, other: Self, weight: Self) -> Self {
        Self {
            value: self.value + (other.value - self.value) * weight.value,
            grad: self.grad * (1.0 - weight.value)
                + other.grad * weight.value
                + weight.grad * (other.value - self.value),
        }
    }

    /// Quintic smootherstep (C²) of `(self - start) / width`: 0 below `start`, 1 from
    /// `start + width` on.
    fn ramp(self, start: f64, width: f64) -> Self {
        let t = det::clamp((self.value - start) / width, 0.0, 1.0);
        Self {
            value: t * t * t * (10.0 + t * (-15.0 + 6.0 * t)),
            grad: self.grad * (30.0 * t * t * (1.0 - t) * (1.0 - t) / width),
        }
    }
}

/// A C¹ maximum of `a` and `b` that rounds the crease over `|a - b| < width` by a quadratic.
/// It is symmetric in its arguments and exceeds the true maximum by at most `width / 4`.
fn rounded_max(a: Dual, b: Dual, width: f64) -> Dual {
    let d = a.value - b.value;
    if d >= width {
        return a;
    }
    if d <= -width {
        return b;
    }
    let t = (d / width + 1.0) * 0.5;
    Dual {
        value: b.value + width * t * t,
        grad: a.grad * t + b.grad * (1.0 - t),
    }
}

/// The blended gravity field of a cube body.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CubeGravity {
    half_edge: f64,
    surface_gravity: f64,
    mu: f64,
    shape: CubeGravityShape,
    /// Offset of the inner sphere, `(√3 - 1)(R - inner_start)`.
    shift: f64,
    /// Radius of the linear core, `R - inner_start - inner_width`.
    core: f64,
    /// Start of the outer band in `ρ`, `R + outer_start`.
    anchor: f64,
    /// End of the outer band in `ρ`, `anchor + outer_width`.
    end: f64,
    /// Kepler slope `mu / end²` at the end of the band.
    far_slope: f64,
    /// Potential at `anchor`.
    potential_at_anchor: f64,
}

impl CubeGravity {
    /// The field of a cube with half edge `half_edge` (m) and gravity `surface_gravity` (m/s²)
    /// at the face centres, shaped by `shape`; `mu = g R²`.
    ///
    /// # Errors
    /// When a value is not finite, the half edge, the surface gravity, the edge rounding or a
    /// blend width is not positive, a blend start is negative, or the inner blend and the edge
    /// rounding do not leave room for the core (`inner_start + inner_width + edge_rounding`
    /// must stay below the half edge).
    pub fn new(
        half_edge: f64,
        surface_gravity: f64,
        shape: CubeGravityShape,
    ) -> Result<Self, GravityError> {
        let values = [
            half_edge,
            surface_gravity,
            shape.edge_rounding,
            shape.outer_start,
            shape.outer_width,
            shape.inner_start,
            shape.inner_width,
        ];
        if values.iter().any(|value| !value.is_finite()) {
            return Err(GravityError(format!(
                "every value must be finite: {values:?}"
            )));
        }
        let positive = [
            half_edge,
            surface_gravity,
            shape.edge_rounding,
            shape.outer_width,
            shape.inner_width,
        ];
        if positive.iter().any(|&value| value <= 0.0)
            || shape.outer_start < 0.0
            || shape.inner_start < 0.0
        {
            return Err(GravityError(format!(
                "lengths and the surface gravity must be positive, blend starts at least 0: \
                 {values:?}"
            )));
        }
        if shape.inner_start + shape.inner_width + shape.edge_rounding >= half_edge {
            return Err(GravityError(format!(
                "no room for the core: inner start {} + inner width {} + edge rounding {} must \
                 stay below the half edge {half_edge}",
                shape.inner_start, shape.inner_width, shape.edge_rounding
            )));
        }
        let mu = surface_gravity * half_edge * half_edge;
        let anchor = half_edge + shape.outer_start;
        let end = anchor + shape.outer_width;
        let far_slope = mu / (end * end);
        // The slope blends from g to far_slope with a smoothstep, whose mean over the band is
        // one half; integrating it back from the Kepler potential at `end` fixes the constant.
        let potential_at_anchor =
            -mu / end - shape.outer_width * (surface_gravity + far_slope) * 0.5;
        Ok(Self {
            half_edge,
            surface_gravity,
            mu,
            shape,
            shift: (3.0_f64.sqrt() - 1.0) * (half_edge - shape.inner_start),
            core: half_edge - shape.inner_start - shape.inner_width,
            anchor,
            end,
            far_slope,
            potential_at_anchor,
        })
    }

    /// Half edge `R` in metres.
    #[must_use]
    pub fn half_edge(&self) -> f64 {
        self.half_edge
    }

    /// Gravity at the face centres in m/s².
    #[must_use]
    pub fn surface_gravity(&self) -> f64 {
        self.surface_gravity
    }

    /// Gravitational parameter `g R²` in m³/s².
    #[must_use]
    pub fn mu(&self) -> f64 {
        self.mu
    }

    /// The shape lengths.
    #[must_use]
    pub fn shape(&self) -> &CubeGravityShape {
        &self.shape
    }

    /// Distance from the centre beyond which the field is exactly Kepler's in every direction,
    /// metres: `√3 (R + outer_start + outer_width)`. The outer band ends where the rounded cube
    /// distance reaches `R + outer_start + outer_width`, and the rounded cube distance is at
    /// least the largest coordinate, which is at least `r / √3` (above a corner). Above a face
    /// centre the field is Kepler's from `R + outer_start + outer_width` on.
    #[must_use]
    pub fn kepler_radius(&self) -> f64 {
        3.0_f64.sqrt() * self.end
    }

    /// The rounded cube distance and its gradient.
    fn cube_distance(&self, p: DVec3) -> Dual {
        let width = self.shape.edge_rounding;
        let axes = [DVec3::X, DVec3::Y, DVec3::Z];
        let components: [Dual; 3] = core::array::from_fn(|axis| {
            let coordinate = p[axis];
            let sign = if coordinate == 0.0 {
                0.0
            } else {
                coordinate.signum()
            };
            Dual {
                value: coordinate.abs(),
                grad: axes[axis] * sign,
            }
        });
        // Averaging the three pairings keeps the corners symmetric under axis permutations.
        // (Summing before dividing keeps a face-centre distance exact.)
        let mut value = 0.0;
        let mut grad = DVec3::ZERO;
        for first in 0..3 {
            let pair = rounded_max(components[first], components[(first + 1) % 3], width);
            let all = rounded_max(pair, components[(first + 2) % 3], width);
            value += all.value;
            grad += all.grad;
        }
        Dual {
            value: value / 3.0,
            grad: grad / 3.0,
        }
    }

    /// The radial profile: potential `V(ρ)` and slope `V'(ρ) > 0`.
    fn profile(&self, rho: f64) -> (f64, f64) {
        let g = self.surface_gravity;
        if rho >= self.end {
            return (-self.mu / rho, self.mu / (rho * rho));
        }
        if rho > self.anchor {
            let width = self.shape.outer_width;
            let t = (rho - self.anchor) / width;
            let smooth = t * t * (3.0 - 2.0 * t);
            let slope = g + (self.far_slope - g) * smooth;
            let integral = g * t + (self.far_slope - g) * (t * t * t - 0.5 * t * t * t * t);
            return (self.potential_at_anchor + width * integral, slope);
        }
        let inner_limit = self.core - self.shift;
        if rho >= inner_limit {
            return (self.potential_at_anchor + g * (rho - self.anchor), g);
        }
        // Linear core (uniform density): with ρ = r - shift the slope is g r / core.
        let radius = rho + self.shift;
        (
            self.potential_at_anchor
                + g * (inner_limit - self.anchor)
                + g * (radius * radius - self.core * self.core) / (2.0 * self.core),
            g * radius / self.core,
        )
    }

    /// Potential and acceleration at `position` (body frame, metres from the cube centre).
    #[must_use]
    pub fn field(&self, position: DVec3) -> FieldSample {
        let r = position.length();
        let cube = self.cube_distance(position);
        let height = Dual {
            value: cube.value - self.half_edge,
            grad: cube.grad,
        };
        let outer = height.ramp(self.shape.outer_start, self.shape.outer_width);
        let inner = height
            .negated()
            .ramp(self.shape.inner_start, self.shape.inner_width);
        let radial = position.normalize_or_zero();
        let spherical = Dual {
            value: r,
            grad: radial,
        };
        // Inner equipotentials are aligned with the corners so that even a narrow inner blend
        // keeps the force pointing inwards.
        let inner_sphere = Dual {
            value: r - self.shift,
            grad: radial,
        };
        let rho = if outer.value >= 1.0 {
            spherical
        } else if inner.value >= 1.0 {
            inner_sphere
        } else if height.value < 0.0 {
            cube.mix(inner_sphere, inner)
        } else {
            cube.mix(spherical, outer)
        };
        let (potential, slope) = self.profile(rho.value);
        FieldSample {
            potential,
            acceleration: rho.grad * -slope,
        }
    }
}

/// The field of a uniform-density sphere (stars): Kepler's outside, linear inside.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SphereGravity {
    radius: f64,
    mu: f64,
}

impl SphereGravity {
    /// A sphere of `radius` metres with `surface_gravity` m/s² on its surface; `mu = g R²`.
    ///
    /// # Errors
    /// When either value is not positive and finite.
    pub fn new(radius: f64, surface_gravity: f64) -> Result<Self, GravityError> {
        if !(radius.is_finite() && surface_gravity.is_finite()) || radius <= 0.0 {
            return Err(GravityError(format!(
                "the radius must be positive and finite, found {radius}"
            )));
        }
        if surface_gravity <= 0.0 {
            return Err(GravityError(format!(
                "the surface gravity must be positive, found {surface_gravity}"
            )));
        }
        Ok(Self {
            radius,
            mu: surface_gravity * radius * radius,
        })
    }

    /// Radius in metres.
    #[must_use]
    pub fn radius(&self) -> f64 {
        self.radius
    }

    /// Gravitational parameter in m³/s².
    #[must_use]
    pub fn mu(&self) -> f64 {
        self.mu
    }

    /// Potential and acceleration at `position` (metres from the centre).
    #[must_use]
    pub fn field(&self, position: DVec3) -> FieldSample {
        let r = position.length();
        if r >= self.radius {
            FieldSample {
                potential: -self.mu / r,
                acceleration: position * (-self.mu / (r * r * r)),
            }
        } else {
            let cube = self.radius * self.radius * self.radius;
            FieldSample {
                potential: -self.mu * (3.0 * self.radius * self.radius - r * r) / (2.0 * cube),
                acceleration: position * (-self.mu / cube),
            }
        }
    }
}

#[cfg(test)]
mod tests;
