//! Maths: the `glam` re-export plus what `glam` lacks.
//!
//! - integer floor division for negative coordinates,
//! - axis-aligned boxes for the three numeric domains of the engine ([`IAabb`] voxels,
//!   [`DAabb`] world metres, [`Aabb`] camera-relative metres),
//! - the view frustum of the reversed infinite projection (contract §2.4),
//! - 3D Morton codes,
//! - scalar interpolation helpers for `f32` and `f64`.
//!
//! Units follow contract §2.1: metres, seconds, radians.
//!
//! # Determinism
//! Everything *defined* in this module is deterministic (contract §2.5). The re-exported
//! `glam` API is deterministic as well:
//!
//! - what gets by with `+ - * /`, comparisons and `sqrt` is exact IEEE arithmetic:
//!   constructors from components, sums and products, `dot`, `cross`, `length`, `normalize`,
//!   `inverse`, `from_quat`, and the component-wise `min` / `max` / `clamp` of the `f64` and
//!   integer vectors (single comparisons; for a NaN or for zeros of opposite sign they return
//!   their second operand, which a test pins);
//! - what needs an elementary function (`from_axis_angle`, `from_rotation_x/y/z`,
//!   `from_euler`, `from_angle`, `angle_between`, `slerp`, `to_euler`, the component-wise
//!   `exp`, `powf`, `sin`, `cos`, …) goes through `libm`: the workspace builds `glam` with its
//!   `libm` feature (contract §2.1), which replaces its `std` maths backend, and `glam`
//!   resolves to the same exactly pinned `libm` as [`crate::det`]. A test checks that `glam`'s
//!   rotations are bit for bit the [`crate::det`] ones, so losing the feature is noticed.
//!
//! The scalar `f64` / `f32` methods are a different matter: `f64::sin`, `f64::max` and
//! friends call the platform or leave the sign of zero open. Deterministic crates use
//! [`crate::det`] for them ([`det::sin_cos`], [`det::atan2`], [`det::max`], …;
//! [`det::dquat_from_axis_angle`] for rotations), which `clippy.toml` enforces. Use
//! [`perspective_infinite_reverse_rh`] for the projection (contract §2.4).
//!
//! ```
//! use cp_core::math::{DAabb, DVec3, Frustum, IAabb, IVec3, Mat3, div_floor, mod_floor};
//!
//! // Floor semantics for negative coordinates.
//! assert_eq!((div_floor(-33, 32), mod_floor(-33, 32)), (-2, 31));
//!
//! // A chunk's voxels as a half-open box, culled camera-relative.
//! let chunk = IAabb::from_min_size(IVec3::new(-64, 0, 32), IVec3::splat(32));
//! let camera = DVec3::new(-48.0, 16.0, 200.0);
//! let relative = chunk.as_daabb().relative_to(camera);
//! // Camera looking along -Z (identity view rotation), 70° vertical field of view, 16:9.
//! let frustum = Frustum::from_rotation_fov(Mat3::IDENTITY, 70.0_f32.to_radians(), 16.0 / 9.0);
//! assert!(frustum.intersects_aabb(relative.center(), relative.half_extent()));
//! let behind = DAabb::new(DVec3::new(-64.0, 0.0, 300.0), DVec3::new(-32.0, 32.0, 332.0));
//! let behind = behind.relative_to(camera);
//! assert!(!frustum.intersects_aabb(behind.center(), behind.half_extent()));
//! ```

pub use glam::*;

use serde::{Deserialize, Serialize};

use crate::det::{self, Real};

// ---------------------------------------------------------------------------------------------
// Integer floor maths
// ---------------------------------------------------------------------------------------------

/// Integer division rounding towards negative infinity: `floor(a / b)`.
///
/// `div_floor(-1, 32) == -1` where `-1 / 32 == 0`. For powers of two this equals `a >> log2(b)`.
///
/// # Panics
/// If `b == 0` or on overflow (`a == i32::MIN && b == -1`).
#[inline]
#[must_use]
pub const fn div_floor(a: i32, b: i32) -> i32 {
    let quotient = a / b;
    let remainder = a % b;
    if remainder != 0 && ((remainder < 0) != (b < 0)) {
        quotient - 1
    } else {
        quotient
    }
}

/// Remainder of [`div_floor`]: `a - b * div_floor(a, b)`, which has the sign of `b`.
///
/// `mod_floor(-1, 32) == 31` where `-1 % 32 == -1`. For powers of two this equals `a & (b - 1)`.
///
/// # Panics
/// If `b == 0` or on overflow (`a == i32::MIN && b == -1`).
#[inline]
#[must_use]
pub const fn mod_floor(a: i32, b: i32) -> i32 {
    let remainder = a % b;
    if remainder != 0 && ((remainder < 0) != (b < 0)) {
        remainder + b
    } else {
        remainder
    }
}

/// Component-wise [`div_floor`] of a vector by a scalar.
///
/// # Panics
/// Like [`div_floor`].
#[inline]
#[must_use]
pub const fn div_floor_ivec3(a: IVec3, b: i32) -> IVec3 {
    IVec3::new(div_floor(a.x, b), div_floor(a.y, b), div_floor(a.z, b))
}

/// Component-wise [`mod_floor`] of a vector by a scalar.
///
/// # Panics
/// Like [`mod_floor`].
#[inline]
#[must_use]
pub const fn mod_floor_ivec3(a: IVec3, b: i32) -> IVec3 {
    IVec3::new(mod_floor(a.x, b), mod_floor(a.y, b), mod_floor(a.z, b))
}

// ---------------------------------------------------------------------------------------------
// Boxes
// ---------------------------------------------------------------------------------------------

/// Axis-aligned box of voxels, **half-open**: it contains the integer points `p` with
/// `min <= p < max` on every axis.
///
/// Because voxel `(x, y, z)` spans `[x, x+1)` m, the box covers exactly the metres
/// `[min, max]` (see [`IAabb::as_daabb`]). A box is empty when `max <= min` on any axis.
/// Arithmetic is unchecked `i32`; keep coordinates inside the world limits of
/// [`crate::coords`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct IAabb {
    /// Inclusive minimum corner.
    pub min: IVec3,
    /// Exclusive maximum corner.
    pub max: IVec3,
}

impl IAabb {
    /// An empty box (`min == max == 0`). It is the neutral element of [`IAabb::union`].
    pub const EMPTY: Self = Self {
        min: IVec3::ZERO,
        max: IVec3::ZERO,
    };

    /// Box from its inclusive minimum and exclusive maximum corner.
    #[inline]
    #[must_use]
    pub const fn new(min: IVec3, max: IVec3) -> Self {
        Self { min, max }
    }

    /// Box from its minimum corner and its size in voxels.
    #[inline]
    #[must_use]
    pub fn from_min_size(min: IVec3, size: IVec3) -> Self {
        Self {
            min,
            max: min + size,
        }
    }

    /// True when the box contains no voxel.
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.max.cmple(self.min).any()
    }

    /// True when `point` lies inside (`min <= point < max`).
    #[inline]
    #[must_use]
    pub fn contains(&self, point: IVec3) -> bool {
        point.cmpge(self.min).all() && point.cmplt(self.max).all()
    }

    /// True when the two boxes share at least one voxel. Empty boxes intersect nothing.
    #[inline]
    #[must_use]
    pub fn intersects(&self, other: &Self) -> bool {
        self.min.max(other.min).cmplt(self.max.min(other.max)).all()
    }

    /// Smallest box containing both boxes. An empty operand is ignored.
    #[inline]
    #[must_use]
    pub fn union(&self, other: &Self) -> Self {
        if self.is_empty() {
            *other
        } else if other.is_empty() {
            *self
        } else {
            Self {
                min: self.min.min(other.min),
                max: self.max.max(other.max),
            }
        }
    }

    /// The voxels common to both boxes, or `None` when there are none.
    #[inline]
    #[must_use]
    pub fn intersection(&self, other: &Self) -> Option<Self> {
        let min = self.min.max(other.min);
        let max = self.max.min(other.max);
        min.cmplt(max).all().then_some(Self { min, max })
    }

    /// Box grown by `margin` voxels on all six sides (shrunk for a negative margin).
    #[inline]
    #[must_use]
    pub fn expand(&self, margin: i32) -> Self {
        Self {
            min: self.min - margin,
            max: self.max + margin,
        }
    }

    /// Edge lengths in voxels (`max - min`). Components `<= 0` mean the box is empty.
    #[inline]
    #[must_use]
    pub fn size(&self) -> IVec3 {
        self.max - self.min
    }

    /// Geometric centre in metres. Half-integer for odd sizes, hence `f64`.
    #[inline]
    #[must_use]
    pub fn center(&self) -> DVec3 {
        (self.min.as_dvec3() + self.max.as_dvec3()) * 0.5
    }

    /// The same region as a box of metres: `[min, max]` as `f64`.
    #[inline]
    #[must_use]
    pub fn as_daabb(&self) -> DAabb {
        DAabb {
            min: self.min.as_dvec3(),
            max: self.max.as_dvec3(),
        }
    }
}

macro_rules! float_aabb {
    ($(#[$doc:meta])* $name:ident, $vec:ident, $scalar:ident) => {
        $(#[$doc])*
        #[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
        pub struct $name {
            /// Minimum corner (inclusive).
            pub min: $vec,
            /// Maximum corner (inclusive).
            pub max: $vec,
        }

        impl $name {
            /// The empty box (`min = +inf`, `max = -inf`), neutral element of `union`.
            pub const EMPTY: Self = Self { min: $vec::INFINITY, max: $vec::NEG_INFINITY };

            /// Box from its minimum and maximum corner.
            #[inline]
            #[must_use]
            pub const fn new(min: $vec, max: $vec) -> Self {
                Self { min, max }
            }

            /// Box from its centre and half edge lengths (`half_extent >= 0`).
            #[inline]
            #[must_use]
            pub fn from_center_half_extent(center: $vec, half_extent: $vec) -> Self {
                Self { min: center - half_extent, max: center + half_extent }
            }

            /// True when the box contains no point: `max < min` on some axis, or a NaN corner.
            #[inline]
            #[must_use]
            pub fn is_empty(&self) -> bool {
                !self.min.cmple(self.max).all()
            }

            /// True when `point` lies inside or on the boundary.
            #[inline]
            #[must_use]
            pub fn contains(&self, point: $vec) -> bool {
                point.cmpge(self.min).all() && point.cmple(self.max).all()
            }

            /// True when the two closed boxes share at least one point; touching counts.
            /// Empty boxes intersect nothing.
            #[inline]
            #[must_use]
            pub fn intersects(&self, other: &Self) -> bool {
                self.min.max(other.min).cmple(self.max.min(other.max)).all()
            }

            /// Smallest box containing both boxes. An empty operand is ignored.
            #[inline]
            #[must_use]
            pub fn union(&self, other: &Self) -> Self {
                if self.is_empty() {
                    *other
                } else if other.is_empty() {
                    *self
                } else {
                    Self { min: self.min.min(other.min), max: self.max.max(other.max) }
                }
            }

            /// The points common to both boxes, or `None` when there are none. Boxes that
            /// only touch yield a box without volume.
            #[inline]
            #[must_use]
            pub fn intersection(&self, other: &Self) -> Option<Self> {
                let min = self.min.max(other.min);
                let max = self.max.min(other.max);
                min.cmple(max).all().then_some(Self { min, max })
            }

            /// Box grown by `margin` metres on all six sides (shrunk for a negative margin).
            #[inline]
            #[must_use]
            pub fn expand(&self, margin: $scalar) -> Self {
                Self { min: self.min - margin, max: self.max + margin }
            }

            /// Edge lengths in metres (`max - min`).
            #[inline]
            #[must_use]
            pub fn size(&self) -> $vec {
                self.max - self.min
            }

            /// Centre point.
            #[inline]
            #[must_use]
            pub fn center(&self) -> $vec {
                (self.min + self.max) * 0.5
            }

            /// Half edge lengths in metres, the form [`Frustum::intersects_aabb`] takes.
            #[inline]
            #[must_use]
            pub fn half_extent(&self) -> $vec {
                (self.max - self.min) * 0.5
            }
        }
    };
}

float_aabb!(
    /// Axis-aligned box in `f64` metres (body, system or galaxy-local space), **closed**:
    /// it contains the points `p` with `min <= p <= max`.
    DAabb,
    DVec3,
    f64
);

float_aabb!(
    /// Axis-aligned box in `f32` metres, **closed**. For camera-relative render space and small
    /// local offsets only; world positions never enter an `f32` (contract §2.1).
    Aabb,
    Vec3,
    f32
);

impl DAabb {
    /// The box relative to `origin`, narrowed to `f32` (for example camera-relative).
    ///
    /// The subtraction happens in `f64`, so precision is lost only in the final small offsets.
    #[inline]
    #[must_use]
    pub fn relative_to(&self, origin: DVec3) -> Aabb {
        Aabb {
            min: (self.min - origin).as_vec3(),
            max: (self.max - origin).as_vec3(),
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Projection and frustum
// ---------------------------------------------------------------------------------------------

/// Distance of the near plane in metres (contract §2.4).
pub const Z_NEAR: f32 = 0.1;

/// The engine's projection (contract §2.4): right-handed view space looking along `-Z`,
/// infinite far plane, reversed depth in `[0, 1]` (`z_near` maps to depth 1, infinity to 0).
///
/// Same matrix as `glam::camera::rh::proj::directx::perspective_infinite_reverse` (formerly
/// `Mat4::perspective_infinite_reverse_rh`), but built with the deterministic tangent.
/// `fov_y` is the full vertical field of view in radians (`0 < fov_y < π`), `aspect` is
/// width / height, `z_near > 0` is in metres; pass [`Z_NEAR`].
#[inline]
#[must_use]
pub fn perspective_infinite_reverse_rh(fov_y: f32, aspect: f32, z_near: f32) -> Mat4 {
    let focal = 1.0 / det::tan(0.5 * fov_y);
    Mat4::from_cols(
        Vec4::new(focal / aspect, 0.0, 0.0, 0.0),
        Vec4::new(0.0, focal, 0.0, 0.0),
        Vec4::new(0.0, 0.0, 0.0, -1.0),
        Vec4::new(0.0, 0.0, z_near, 0.0),
    )
}

/// Oriented plane `normal · p + d = 0` in `f32` metres.
///
/// `normal` is unit length and points to the positive ("inside") half-space, so
/// [`Plane::signed_distance`] is in metres.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Plane {
    /// Unit normal.
    pub normal: Vec3,
    /// Offset: the plane's signed distance from the origin is `-d`.
    pub d: f32,
}

impl Plane {
    /// Plane from a unit normal and offset.
    #[inline]
    #[must_use]
    pub const fn new(normal: Vec3, d: f32) -> Self {
        Self { normal, d }
    }

    /// Plane through `point` with unit `normal`.
    #[inline]
    #[must_use]
    pub fn from_point_normal(point: Vec3, normal: Vec3) -> Self {
        Self {
            normal,
            d: -normal.dot(point),
        }
    }

    /// Signed distance of `point` in metres; positive on the side the normal points to.
    #[inline]
    #[must_use]
    pub fn signed_distance(&self, point: Vec3) -> f32 {
        self.normal.dot(point) + self.d
    }
}

/// View frustum of the reversed infinite projection in render space (contract §2.2, §2.4).
///
/// Five planes with inward normals; the camera sits at the origin and there is no far plane.
/// Used to cull camera-relative bounds.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Frustum {
    /// Planes in the order [`Frustum::LEFT`], [`Frustum::RIGHT`], [`Frustum::BOTTOM`],
    /// [`Frustum::TOP`], [`Frustum::NEAR`]. A point is inside when all five signed distances
    /// are `>= 0`.
    pub planes: [Plane; 5],
}

impl Frustum {
    /// Index of the left plane in [`Frustum::planes`].
    pub const LEFT: usize = 0;
    /// Index of the right plane.
    pub const RIGHT: usize = 1;
    /// Index of the bottom plane.
    pub const BOTTOM: usize = 2;
    /// Index of the top plane.
    pub const TOP: usize = 3;
    /// Index of the near plane.
    pub const NEAR: usize = 4;

    /// Frustum of a camera at the origin with the near plane at [`Z_NEAR`].
    ///
    /// `view_rot` is the rotation part of the view matrix: it maps render-space directions to
    /// view space (`+X` right, `+Y` up, `-Z` forward), so its rows are the camera's right, up and
    /// back axes in render space. For a camera orientation `q` (view → render) pass
    /// `Mat3::from_quat(q).transpose()`. `fov_y` is the full vertical field of view in radians
    /// (`0 < fov_y < π`), `aspect` is width / height.
    ///
    /// The planes match [`perspective_infinite_reverse_rh`]`(fov_y, aspect, Z_NEAR) * view`.
    #[inline]
    #[must_use]
    pub fn from_rotation_fov(view_rot: Mat3, fov_y: f32, aspect: f32) -> Self {
        Self::from_rotation_fov_near(view_rot, fov_y, aspect, Z_NEAR)
    }

    /// Like [`Frustum::from_rotation_fov`] with an explicit near distance in metres
    /// (`z_near >= 0`; zero puts the near plane through the camera).
    #[must_use]
    pub fn from_rotation_fov_near(view_rot: Mat3, fov_y: f32, aspect: f32, z_near: f32) -> Self {
        let tan_y = det::tan(0.5 * fov_y);
        let tan_x = tan_y * aspect;
        // `view_rot` is orthonormal, so its transpose maps view-space normals to render space.
        let to_render = view_rot.transpose();
        let side = |x: f32, y: f32, z: f32| Plane {
            normal: to_render * Vec3::new(x, y, z).normalize(),
            d: 0.0,
        };
        Self {
            planes: [
                // A view-space point is inside the left plane when x >= tan_x * z (z < 0).
                side(1.0, 0.0, -tan_x),
                side(-1.0, 0.0, -tan_x),
                side(0.0, 1.0, -tan_y),
                side(0.0, -1.0, -tan_y),
                Plane {
                    normal: to_render * Vec3::NEG_Z,
                    d: -z_near,
                },
            ],
        }
    }

    /// True when `point` (render space, metres) is inside or on the boundary.
    #[inline]
    #[must_use]
    pub fn contains_point(&self, point: Vec3) -> bool {
        self.planes
            .iter()
            .all(|plane| plane.signed_distance(point) >= 0.0)
    }

    /// Conservative sphere test: `false` only when the sphere is entirely outside one plane.
    #[inline]
    #[must_use]
    pub fn intersects_sphere(&self, center: Vec3, radius: f32) -> bool {
        self.planes
            .iter()
            .all(|plane| plane.signed_distance(center) >= -radius)
    }

    /// Conservative box test for the box `center ± half_extent` (render space, metres,
    /// `half_extent >= 0`).
    ///
    /// Returns `false` only when the box lies entirely outside one of the five planes, so a
    /// visible box is never culled. A box outside the frustum near one of its edges or its apex
    /// may still be reported as intersecting.
    #[inline]
    #[must_use]
    pub fn intersects_aabb(&self, center: Vec3, half_extent: Vec3) -> bool {
        self.planes.iter().all(|plane| {
            // Projection radius of the box onto the plane normal.
            let radius = plane.normal.abs().dot(half_extent);
            plane.signed_distance(center) >= -radius
        })
    }
}

// ---------------------------------------------------------------------------------------------
// Morton codes
// ---------------------------------------------------------------------------------------------

/// Number of bits per coordinate in a 3D Morton code.
pub const MORTON3_BITS: u32 = 21;

/// Spreads the low 21 bits of `v` so that two zero bits separate each of them.
#[inline]
const fn morton3_spread(v: u32) -> u64 {
    let mut x = v as u64 & 0x1F_FFFF;
    x = (x | (x << 32)) & 0x001F_0000_0000_FFFF;
    x = (x | (x << 16)) & 0x001F_0000_FF00_00FF;
    x = (x | (x << 8)) & 0x100F_00F0_0F00_F00F;
    x = (x | (x << 4)) & 0x10C3_0C30_C30C_30C3;
    x = (x | (x << 2)) & 0x1249_2492_4924_9249;
    x
}

/// Inverse of [`morton3_spread`]: collects every third bit.
#[inline]
const fn morton3_compact(v: u64) -> u32 {
    let mut x = v & 0x1249_2492_4924_9249;
    x = (x | (x >> 2)) & 0x10C3_0C30_C30C_30C3;
    x = (x | (x >> 4)) & 0x100F_00F0_0F00_F00F;
    x = (x | (x >> 8)) & 0x001F_0000_FF00_00FF;
    x = (x | (x >> 16)) & 0x001F_0000_0000_FFFF;
    x = (x | (x >> 32)) & 0x1F_FFFF;
    x as u32
}

/// Interleaves three 21-bit coordinates into a 63-bit Morton (Z-order) code.
///
/// Bit `i` of `x`, `y`, `z` lands in bit `3i`, `3i + 1`, `3i + 2`. Only the low
/// [`MORTON3_BITS`] bits of each coordinate are used; higher bits are ignored. Offset signed
/// coordinates into the non-negative range first.
#[inline]
#[must_use]
pub const fn morton3_encode(x: u32, y: u32, z: u32) -> u64 {
    morton3_spread(x) | (morton3_spread(y) << 1) | (morton3_spread(z) << 2)
}

/// Inverse of [`morton3_encode`]: returns `[x, y, z]`. Bit 63 of `code` is ignored.
#[inline]
#[must_use]
pub const fn morton3_decode(code: u64) -> [u32; 3] {
    [
        morton3_compact(code),
        morton3_compact(code >> 1),
        morton3_compact(code >> 2),
    ]
}

// ---------------------------------------------------------------------------------------------
// Scalar helpers
// ---------------------------------------------------------------------------------------------

/// Linear interpolation `a + (b - a) * t`, not clamped.
///
/// Exact at `t = 0`, monotonic in `t`; at `t = 1` the result can differ from `b` by rounding.
#[inline]
#[must_use]
pub fn lerp<T: Real>(a: T, b: T, t: T) -> T {
    a + (b - a) * t
}

/// Clamps `x` to `[0, 1]`. NaN becomes `0`.
#[inline]
#[must_use]
pub fn saturate<T: Real>(x: T) -> T {
    // Written so that every comparison with NaN falls through to zero.
    if x > T::ZERO {
        if x < T::ONE { x } else { T::ONE }
    } else {
        T::ZERO
    }
}

/// Hermite step: `0` for `x <= edge0`, `1` for `x >= edge1`, smooth (C1) in between.
///
/// With `edge0 > edge1` the step is mirrored; with `edge0 == edge1` it degenerates to a hard
/// step (`0` at the edge itself).
#[inline]
#[must_use]
pub fn smoothstep<T: Real>(edge0: T, edge1: T, x: T) -> T {
    let t = saturate((x - edge0) / (edge1 - edge0));
    t * t * (T::THREE - T::TWO * t)
}

/// Maps `x` linearly from `[in_min, in_max]` to `[out_min, out_max]`, not clamped.
///
/// `in_min == in_max` divides by zero and yields an infinity or NaN.
#[inline]
#[must_use]
pub fn remap<T: Real>(x: T, in_min: T, in_max: T, out_min: T, out_max: T) -> T {
    out_min + (x - in_min) * (out_max - out_min) / (in_max - in_min)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hash::Rng;

    #[test]
    fn floor_division_on_negatives() {
        assert_eq!(div_floor(7, 2), 3);
        assert_eq!(div_floor(-7, 2), -4);
        assert_eq!(div_floor(7, -2), -4);
        assert_eq!(div_floor(-7, -2), 3);
        assert_eq!(mod_floor(7, 2), 1);
        assert_eq!(mod_floor(-7, 2), 1);
        assert_eq!(mod_floor(7, -2), -1);
        assert_eq!(mod_floor(-7, -2), -1);
        assert_eq!(div_floor(-1, 32), -1);
        assert_eq!(mod_floor(-1, 32), 31);
        assert_eq!(div_floor(-32, 32), -1);
        assert_eq!(mod_floor(-32, 32), 0);
        assert_eq!(div_floor(-33, 32), -2);
        assert_eq!(mod_floor(-33, 32), 31);
        assert_eq!(div_floor(i32::MIN, 32), -(1 << 26));
        assert_eq!(mod_floor(i32::MIN, 32), 0);
        assert_eq!(div_floor(i32::MAX, 32), (1 << 26) - 1);
        assert_eq!(mod_floor(i32::MAX, 32), 31);
        assert_eq!(div_floor(i32::MIN, i32::MAX), -2);
        assert_eq!(mod_floor(i32::MIN, i32::MAX), i32::MAX - 1);
    }

    #[test]
    fn floor_division_matches_reference() {
        // Reference: exact rational floor in i64, obviously correct.
        fn reference(a: i64, b: i64) -> (i64, i64) {
            // For b > 0 the floor is the largest q with q * b <= a, for b < 0 the largest q with
            // q * b >= a. Start from the truncated quotient and step to it.
            let too_big = |q: i64| if b > 0 { q * b > a } else { q * b < a };
            let mut q = a / b;
            while too_big(q) {
                q -= 1;
            }
            while !too_big(q + 1) {
                q += 1;
            }
            (q, a - q * b)
        }
        for a in -70..=70 {
            for b in (-9..=9).filter(|&b| b != 0) {
                let (q, r) = reference(i64::from(a), i64::from(b));
                assert_eq!(i64::from(div_floor(a, b)), q, "{a} / {b}");
                assert_eq!(i64::from(mod_floor(a, b)), r, "{a} % {b}");
                // Defining identities.
                assert_eq!(div_floor(a, b) * b + mod_floor(a, b), a);
                assert!(r == 0 || (r < 0) == (b < 0));
                assert!(r.abs() < i64::from(b).abs());
            }
            // Powers of two agree with shifts and masks (contract §2.2).
            for bits in 0..8 {
                assert_eq!(div_floor(a, 1 << bits), a >> bits);
                assert_eq!(mod_floor(a, 1 << bits), a & ((1 << bits) - 1));
            }
        }
        let mut rng = Rng::new(11, 1);
        for _ in 0..20_000 {
            // i32::MIN / -1 overflows by definition; keep it out of the random sweep.
            let a = (rng.next_u32() as i32).max(i32::MIN + 1);
            let b = (rng.next_u32() as i32 >> rng.range_u32(0, 31)).max(1);
            let (q, r) = reference(i64::from(a), i64::from(b));
            assert_eq!(i64::from(div_floor(a, b)), q);
            assert_eq!(i64::from(mod_floor(a, b)), r);
            let (q, r) = reference(i64::from(a), -i64::from(b));
            assert_eq!(i64::from(div_floor(a, -b)), q);
            assert_eq!(i64::from(mod_floor(a, -b)), r);
        }
    }

    #[test]
    fn floor_division_of_vectors() {
        let v = IVec3::new(-1, 31, -33);
        assert_eq!(div_floor_ivec3(v, 32), IVec3::new(-1, 0, -2));
        assert_eq!(mod_floor_ivec3(v, 32), IVec3::new(31, 31, 31));
        assert_eq!(div_floor_ivec3(v, 32) * 32 + mod_floor_ivec3(v, 32), v);
        assert_eq!(div_floor_ivec3(v, 32), v >> 5);
    }

    #[test]
    fn iaabb_is_half_open() {
        let a = IAabb::new(IVec3::new(-2, 0, 1), IVec3::new(3, 4, 2));
        assert!(a.contains(IVec3::new(-2, 0, 1)));
        assert!(a.contains(IVec3::new(2, 3, 1)));
        assert!(!a.contains(IVec3::new(3, 3, 1)));
        assert!(!a.contains(IVec3::new(2, 4, 1)));
        assert!(!a.contains(IVec3::new(2, 3, 2)));
        assert!(!a.contains(IVec3::new(-3, 0, 1)));
        assert_eq!(a.size(), IVec3::new(5, 4, 1));
        assert_eq!(a.center(), DVec3::new(0.5, 2.0, 1.5));
        assert_eq!(
            a,
            IAabb::from_min_size(IVec3::new(-2, 0, 1), IVec3::new(5, 4, 1))
        );
        assert!(!a.is_empty());
        assert!(IAabb::EMPTY.is_empty());
        assert!(IAabb::new(IVec3::ZERO, IVec3::new(4, 0, 4)).is_empty());
        assert_eq!(
            a.expand(2),
            IAabb::new(IVec3::new(-4, -2, -1), IVec3::new(5, 6, 4))
        );
        assert_eq!(a.expand(2).expand(-2), a);
        assert_eq!(
            a.as_daabb(),
            DAabb::new(DVec3::new(-2.0, 0.0, 1.0), DVec3::new(3.0, 4.0, 2.0))
        );

        // Boxes that share only a face do not intersect.
        let touching = IAabb::new(IVec3::new(3, 0, 1), IVec3::new(5, 4, 2));
        assert!(!a.intersects(&touching));
        assert_eq!(a.intersection(&touching), None);
        let overlapping = IAabb::new(IVec3::new(2, -5, -5), IVec3::new(9, 1, 9));
        assert!(a.intersects(&overlapping) && overlapping.intersects(&a));
        assert_eq!(
            a.intersection(&overlapping),
            Some(IAabb::new(IVec3::new(2, 0, 1), IVec3::new(3, 1, 2)))
        );
        assert_eq!(
            a.union(&overlapping),
            IAabb::new(IVec3::new(-2, -5, -5), IVec3::new(9, 4, 9))
        );
        // Empty boxes are neutral for union and intersect nothing, wherever they sit.
        let empty_far = IAabb::new(IVec3::splat(100), IVec3::splat(100));
        assert_eq!(a.union(&empty_far), a);
        assert_eq!(empty_far.union(&a), a);
        assert_eq!(IAabb::EMPTY.union(&a), a);
        let empty_inside = IAabb::new(IVec3::new(0, 1, 1), IVec3::new(0, 2, 2));
        assert!(!a.intersects(&empty_inside) && !empty_inside.intersects(&a));
    }

    #[test]
    fn iaabb_operations_match_point_sets() {
        // Brute force over all voxels of a small universe.
        let mut rng = Rng::new(21, 2);
        let random_box = |rng: &mut Rng| {
            let min = IVec3::new(
                rng.range_u32(0, 6) as i32 - 3,
                rng.range_u32(0, 6) as i32 - 3,
                rng.range_u32(0, 6) as i32 - 3,
            );
            let size = IVec3::new(
                rng.range_u32(0, 5) as i32,
                rng.range_u32(0, 5) as i32,
                rng.range_u32(0, 5) as i32,
            );
            IAabb::from_min_size(min, size)
        };
        for _ in 0..3000 {
            let a = random_box(&mut rng);
            let b = random_box(&mut rng);
            let union = a.union(&b);
            let intersection = a.intersection(&b);
            let (mut any_a, mut any_common) = (false, false);
            for x in -5..8 {
                for y in -5..8 {
                    for z in -5..8 {
                        let p = IVec3::new(x, y, z);
                        let (in_a, in_b) = (a.contains(p), b.contains(p));
                        any_a |= in_a;
                        any_common |= in_a && in_b;
                        if in_a || in_b {
                            assert!(union.contains(p));
                        }
                        let in_intersection = intersection.is_some_and(|i| i.contains(p));
                        assert_eq!(in_intersection, in_a && in_b);
                    }
                }
            }
            assert_eq!(a.is_empty(), !any_a);
            assert_eq!(a.intersects(&b), any_common);
            assert_eq!(intersection.is_some(), any_common);
            if !a.is_empty() && !b.is_empty() {
                // The union is tight: every face of it touches one of the operands.
                assert_eq!(union.min, a.min.min(b.min));
                assert_eq!(union.max, a.max.max(b.max));
            }
        }
    }

    #[test]
    fn float_aabbs_are_closed() {
        let a = DAabb::new(DVec3::new(-1.0, 0.0, 2.0), DVec3::new(1.0, 4.0, 3.0));
        assert!(a.contains(a.min) && a.contains(a.max) && a.contains(a.center()));
        assert!(!a.contains(DVec3::new(1.0 + 1e-12, 1.0, 2.5)));
        assert_eq!(a.size(), DVec3::new(2.0, 4.0, 1.0));
        assert_eq!(a.center(), DVec3::new(0.0, 2.0, 2.5));
        assert_eq!(a.half_extent(), DVec3::new(1.0, 2.0, 0.5));
        assert_eq!(
            DAabb::from_center_half_extent(a.center(), a.half_extent()),
            a
        );
        assert_eq!(
            a.expand(0.5),
            DAabb::new(DVec3::new(-1.5, -0.5, 1.5), DVec3::new(1.5, 4.5, 3.5))
        );

        let touching = DAabb::new(DVec3::new(1.0, 0.0, 2.0), DVec3::new(2.0, 1.0, 3.0));
        assert!(a.intersects(&touching));
        let shared = a.intersection(&touching).unwrap();
        assert_eq!(shared.size().x, 0.0);
        let apart = DAabb::new(DVec3::new(1.5, 0.0, 2.0), DVec3::new(2.0, 1.0, 3.0));
        assert!(!a.intersects(&apart) && a.intersection(&apart).is_none());
        assert_eq!(
            a.union(&apart),
            DAabb::new(a.min, DVec3::new(2.0, 4.0, 3.0))
        );

        assert!(DAabb::EMPTY.is_empty() && !a.is_empty());
        assert_eq!(DAabb::EMPTY.union(&a), a);
        assert_eq!(a.union(&DAabb::EMPTY), a);
        assert!(!a.intersects(&DAabb::EMPTY) && !DAabb::EMPTY.intersects(&a));
        assert!(!DAabb::EMPTY.contains(DVec3::ZERO));
        assert!(DAabb::new(DVec3::splat(f64::NAN), DVec3::ONE).is_empty());
        // A single point is a valid, non-empty box.
        let point = DAabb::new(DVec3::ONE, DVec3::ONE);
        assert!(!point.is_empty() && point.contains(DVec3::ONE));

        // The f32 variant behaves the same.
        let b = Aabb::new(Vec3::ZERO, Vec3::splat(2.0));
        assert!(b.contains(Vec3::splat(2.0)) && !b.contains(Vec3::new(2.1, 1.0, 1.0)));
        assert_eq!(b.center(), Vec3::ONE);
        assert_eq!(b.half_extent(), Vec3::ONE);
        assert!(b.intersects(&Aabb::new(Vec3::splat(2.0), Vec3::splat(3.0))));
        assert!(Aabb::EMPTY.is_empty());
        assert_eq!(Aabb::EMPTY.union(&b), b);

        // Camera-relative narrowing subtracts in f64 first.
        let far = DAabb::new(DVec3::splat(1.0e9), DVec3::splat(1.0e9 + 32.0));
        let relative = far.relative_to(DVec3::splat(1.0e9 - 0.25));
        assert_eq!(relative, Aabb::new(Vec3::splat(0.25), Vec3::splat(32.25)));
    }

    #[test]
    fn glam_vector_min_max_are_single_comparisons() {
        // The boxes above are built on the component-wise `min`/`max` of `glam`, and the
        // module documentation calls them deterministic. That holds because the pinned `glam`
        // implements them for `f64` vectors as one comparison per component
        // (`if a < b { a } else { b }` and its mirror image) instead of `f64::min`/`f64::max`,
        // whose result for zeros of opposite sign is unspecified. This pins that behaviour:
        // whenever the comparison is false (equal values, mixed zeros, a NaN) the second
        // operand is returned.
        let bits = |v: DVec3| v.to_array().map(f64::to_bits);
        let first = DVec3::new(0.0, -0.0, 2.0);
        let second = DVec3::new(-0.0, 0.0, 2.0);
        assert_eq!(bits(first.min(second)), bits(second));
        assert_eq!(bits(first.max(second)), bits(second));
        assert_eq!(bits(second.min(first)), bits(first));
        assert_eq!(bits(second.max(first)), bits(first));
        let numbers = DVec3::new(1.0, -4.0, 7.5);
        let with_nan = DVec3::new(f64::NAN, -5.0, 9.0);
        // A NaN in the first operand yields the second operand's component, a NaN in the
        // second operand yields that NaN.
        assert_eq!(with_nan.min(numbers), DVec3::new(1.0, -5.0, 7.5));
        assert_eq!(with_nan.max(numbers), DVec3::new(1.0, -4.0, 9.0));
        assert!(numbers.min(with_nan).x.is_nan() && numbers.max(with_nan).x.is_nan());
        // The same in two and four dimensions, and for the boxes built on them.
        assert!(
            DVec2::splat(0.0)
                .max(DVec2::splat(-0.0))
                .x
                .is_sign_negative()
        );
        assert!(
            DVec4::splat(-0.0)
                .min(DVec4::splat(0.0))
                .w
                .is_sign_positive()
        );
        let a = DAabb::new(DVec3::splat(0.0), DVec3::splat(1.0));
        let b = DAabb::new(DVec3::splat(-0.0), DVec3::splat(1.0));
        assert_eq!(bits(a.union(&b).min), bits(b.min));
        assert_eq!(bits(b.union(&a).min), bits(a.min));
    }

    #[test]
    fn glam_uses_the_pinned_libm() {
        // Contract 2.1: glam is built with its `libm` feature, so its trigonometry is the
        // pinned libm of `det`, bit for bit. (With glam's std backend the E0 review measured
        // about 3 % of angles differing in the last bit on Windows; this test would see it.)
        let mut rng = Rng::new(2026, 21);
        let third = (1.0_f64 / 3.0).sqrt();
        let axes = [DVec3::X, DVec3::new(0.6, 0.0, -0.8), DVec3::splat(third)];
        let bits4 = |q: DQuat| q.to_array().map(f64::to_bits);
        for _ in 0..20_000 {
            let angle = rng.range_f64(-10.0, 10.0);
            for axis in axes {
                let ours = det::dquat_from_axis_angle(axis, angle);
                assert_eq!(
                    bits4(DQuat::from_axis_angle(axis, angle)),
                    bits4(ours),
                    "{angle}"
                );
            }
            let (sin, cos) = det::sin_cos(angle);
            let unit = DVec2::from_angle(angle);
            assert_eq!(
                [unit.x.to_bits(), unit.y.to_bits()],
                [cos.to_bits(), sin.to_bits()]
            );
            let angle = angle as f32;
            let (sin, cos) = det::sin_cos(angle);
            let unit = Vec2::from_angle(angle);
            assert_eq!(
                [unit.x.to_bits(), unit.y.to_bits()],
                [cos.to_bits(), sin.to_bits()]
            );
        }
    }

    /// Clip-space reference: is `p` inside the clip volume of the engine's projection?
    fn inside_clip_volume(view_proj: &DMat4, p: DVec3) -> bool {
        let clip = *view_proj * p.extend(1.0);
        clip.x.abs() <= clip.w && clip.y.abs() <= clip.w && clip.z >= 0.0 && clip.z <= clip.w
    }

    fn random_rotation(rng: &mut Rng) -> DQuat {
        // Uniformly random unit quaternion from four Gaussian-free rejection samples.
        loop {
            let q = DVec4::new(
                rng.range_f64(-1.0, 1.0),
                rng.range_f64(-1.0, 1.0),
                rng.range_f64(-1.0, 1.0),
                rng.range_f64(-1.0, 1.0),
            );
            let length_squared = q.length_squared();
            if length_squared > 0.01 && length_squared <= 1.0 {
                let q = q / length_squared.sqrt();
                return DQuat::from_xyzw(q.x, q.y, q.z, q.w);
            }
        }
    }

    #[test]
    fn projection_matches_glam() {
        for (fov, aspect) in [(1.0_f32, 16.0 / 9.0), (0.4, 1.0), (2.2, 0.6)] {
            let ours = perspective_infinite_reverse_rh(fov, aspect, Z_NEAR);
            let reference =
                camera::rh::proj::directx::perspective_infinite_reverse(fov, aspect, Z_NEAR);
            // The only difference is libm's tangent against the platform's: a few ulps.
            assert!(
                ours.abs_diff_eq(reference, 1e-5),
                "{ours:?} vs {reference:?}"
            );
        }
        // Depth is 1 at the near plane and falls towards 0 with distance.
        let projection = perspective_infinite_reverse_rh(1.0, 1.5, Z_NEAR);
        let near = projection * Vec4::new(0.0, 0.0, -Z_NEAR, 1.0);
        assert!((near.z / near.w - 1.0).abs() < 1e-6);
        let far = projection * Vec4::new(0.0, 0.0, -1.0e6, 1.0);
        assert!(far.z / far.w > 0.0 && far.z / far.w < 1e-6);
    }

    #[test]
    fn frustum_planes_are_unit_and_ordered() {
        let frustum = Frustum::from_rotation_fov(Mat3::IDENTITY, 1.2, 1.5);
        for plane in frustum.planes {
            assert!((plane.normal.length() - 1.0).abs() < 1e-6);
        }
        // With the identity view rotation the camera looks along -Z.
        assert!(frustum.planes[Frustum::LEFT].normal.x > 0.0);
        assert!(frustum.planes[Frustum::RIGHT].normal.x < 0.0);
        assert!(frustum.planes[Frustum::BOTTOM].normal.y > 0.0);
        assert!(frustum.planes[Frustum::TOP].normal.y < 0.0);
        assert_eq!(
            frustum.planes[Frustum::NEAR],
            Plane::new(Vec3::NEG_Z, -Z_NEAR)
        );
        assert!(frustum.contains_point(Vec3::new(0.0, 0.0, -1.0)));
        assert!(!frustum.contains_point(Vec3::new(0.0, 0.0, 1.0)));
        assert!(!frustum.contains_point(Vec3::new(0.0, 0.0, -0.05)));
        assert!(frustum.contains_point(Vec3::new(0.0, 0.0, -1.0e12)));
        // The horizontal half angle is wider than the vertical one for aspect > 1.
        let tan_y = det::tan(0.6_f32);
        assert!(frustum.contains_point(Vec3::new(1.4 * tan_y, 0.0, -1.0)));
        assert!(!frustum.contains_point(Vec3::new(0.0, 1.4 * tan_y, -1.0)));
        // Spheres: straddling a plane counts, clearly behind does not.
        assert!(frustum.intersects_sphere(Vec3::new(0.0, 0.0, 0.5), 0.7));
        assert!(!frustum.intersects_sphere(Vec3::new(0.0, 0.0, 0.5), 0.5));
        let plane = Plane::from_point_normal(Vec3::new(0.0, 3.0, 0.0), Vec3::Y);
        assert_eq!(plane.signed_distance(Vec3::new(5.0, 4.5, -2.0)), 1.5);
    }

    #[test]
    fn frustum_matches_brute_force() {
        let mut rng = Rng::new(2026, 11);
        let (mut inside_cases, mut culled_cases, mut conservative_cases) = (0, 0, 0);
        for case in 0..4000 {
            let fov_y = rng.range_f64(0.3, 2.4) as f32;
            let aspect = rng.range_f64(0.5, 2.5) as f32;
            let camera_orientation = random_rotation(&mut rng); // view -> render
            let view_rot_d = DMat3::from_quat(camera_orientation).transpose();
            let view_rot = view_rot_d.as_mat3();
            let frustum = Frustum::from_rotation_fov(view_rot, fov_y, aspect);

            // Independent reference: the projection matrix applied to points in f64.
            let projection = perspective_infinite_reverse_rh(fov_y, aspect, Z_NEAR).as_dmat4();
            let view_proj = projection * DMat4::from_mat3(view_rot_d);

            // Boxes at very different scales, some around the camera, some far away.
            let scale = [0.05, 1.0, 30.0, 2000.0][case % 4];
            let center = DVec3::new(
                rng.range_f64(-4.0, 4.0),
                rng.range_f64(-4.0, 4.0),
                rng.range_f64(-4.0, 4.0),
            ) * scale;
            let half = DVec3::new(
                rng.range_f64(0.02, 1.5),
                rng.range_f64(0.02, 1.5),
                rng.range_f64(0.02, 1.5),
            ) * scale;
            let result = frustum.intersects_aabb(center.as_vec3(), half.as_vec3());

            // (1) Point sampling: any sample inside the clip volume means the box is visible.
            const N: i32 = 12;
            let mut any_sample_inside = false;
            'samples: for ix in 0..=N {
                for iy in 0..=N {
                    for iz in 0..=N {
                        let t = DVec3::new(f64::from(ix), f64::from(iy), f64::from(iz))
                            / f64::from(N)
                            * 2.0
                            - 1.0;
                        // Stay a hair inside the box so f32 rounding of its faces cannot matter.
                        let p = center + half * t * (1.0 - 1e-4);
                        if inside_clip_volume(&view_proj, p) {
                            any_sample_inside = true;
                            break 'samples;
                        }
                    }
                }
            }
            if any_sample_inside {
                assert!(result, "case {case}: visible box was culled");
                inside_cases += 1;
            }

            // (2) Corner reference with planes taken from the matrix rows (Gribb-Hartmann):
            // the box is culled exactly when all eight corners are outside the same plane.
            let rows = [
                view_proj.row(0),
                view_proj.row(1),
                view_proj.row(2),
                view_proj.row(3),
            ];
            let reference_planes = [
                rows[3] + rows[0],
                rows[3] - rows[0],
                rows[3] + rows[1],
                rows[3] - rows[1],
                rows[3] - rows[2],
            ];
            let mut separated = false;
            let mut borderline = false;
            for plane in reference_planes {
                let normal_length = plane.truncate().length();
                let mut max_distance = f64::NEG_INFINITY;
                for corner in 0..8 {
                    let sign = DVec3::new(
                        if corner & 1 == 0 { -1.0 } else { 1.0 },
                        if corner & 2 == 0 { -1.0 } else { 1.0 },
                        if corner & 4 == 0 { -1.0 } else { 1.0 },
                    );
                    let p = center + half * sign;
                    max_distance = det::max(max_distance, plane.dot(p.extend(1.0)) / normal_length);
                }
                // Ignore boxes whose decisive corner is within f32 noise of a plane.
                let tolerance = 1e-4 * (center.length() + half.length()) + 1e-5;
                if max_distance.abs() <= tolerance {
                    borderline = true;
                }
                if max_distance < 0.0 {
                    separated = true;
                }
            }
            if !borderline {
                assert_eq!(
                    result, !separated,
                    "case {case}: disagrees with the corner reference"
                );
                if separated {
                    culled_cases += 1;
                    assert!(!any_sample_inside);
                } else if !any_sample_inside {
                    conservative_cases += 1;
                }
            }
        }
        // The case mix must exercise both outcomes; conservative positives stay a small share.
        assert!(inside_cases > 800, "{inside_cases}");
        assert!(culled_cases > 800, "{culled_cases}");
        assert!(conservative_cases < 200, "{conservative_cases}");
    }

    #[test]
    fn frustum_never_culls_what_it_contains() {
        // Any box around a point inside the frustum intersects; a box behind the camera does not.
        let mut rng = Rng::new(8, 8);
        for _ in 0..2000 {
            let orientation = random_rotation(&mut rng);
            let view_rot = DMat3::from_quat(orientation).transpose().as_mat3();
            let frustum = Frustum::from_rotation_fov(view_rot, 1.1, 1.6);
            let forward = (orientation * DVec3::NEG_Z).as_vec3();
            let distance = rng.range_f64(0.2, 5000.0) as f32;
            let half = Vec3::splat(rng.range_f64(0.001, 3.0) as f32);
            assert!(frustum.contains_point(forward * distance));
            assert!(frustum.intersects_aabb(forward * distance, half));
            assert!(frustum.intersects_sphere(forward * distance, half.x));
            let behind = -forward * (distance + 6.0);
            assert!(!frustum.intersects_aabb(behind, half));
            assert!(!frustum.intersects_sphere(behind, half.x));
        }
    }

    #[test]
    fn morton_round_trip_and_bit_layout() {
        fn reference(x: u32, y: u32, z: u32) -> u64 {
            let mut code = 0_u64;
            for bit in 0..MORTON3_BITS {
                code |= u64::from((x >> bit) & 1) << (3 * bit);
                code |= u64::from((y >> bit) & 1) << (3 * bit + 1);
                code |= u64::from((z >> bit) & 1) << (3 * bit + 2);
            }
            code
        }
        assert_eq!(morton3_encode(0, 0, 0), 0);
        assert_eq!(morton3_encode(1, 0, 0), 0b001);
        assert_eq!(morton3_encode(0, 1, 0), 0b010);
        assert_eq!(morton3_encode(0, 0, 1), 0b100);
        assert_eq!(morton3_encode(0b11, 0b01, 0b10), 0b101_011);
        let max = (1 << MORTON3_BITS) - 1;
        assert_eq!(morton3_encode(max, max, max), (1 << 63) - 1);
        assert_eq!(morton3_decode((1 << 63) - 1), [max, max, max]);
        assert_eq!(morton3_decode(u64::MAX), [max, max, max]);
        // Bits above 21 are ignored.
        assert_eq!(morton3_encode(u32::MAX, 0, 0), morton3_encode(max, 0, 0));

        // Every single bit of every coordinate, and all small coordinates exhaustively.
        for bit in 0..MORTON3_BITS {
            for axis in 0..3 {
                let mut p = [0_u32; 3];
                p[axis] = 1 << bit;
                let code = morton3_encode(p[0], p[1], p[2]);
                assert_eq!(code, 1 << (3 * bit + axis as u32));
                assert_eq!(morton3_decode(code), p);
            }
        }
        for x in 0..32 {
            for y in 0..32 {
                for z in 0..32 {
                    let code = morton3_encode(x, y, z);
                    assert_eq!(code, reference(x, y, z));
                    assert_eq!(morton3_decode(code), [x, y, z]);
                }
            }
        }
        let mut rng = Rng::new(4, 4);
        for _ in 0..100_000 {
            let p = [
                rng.range_u32(0, max + 1),
                rng.range_u32(0, max + 1),
                rng.range_u32(0, max + 1),
            ];
            let code = morton3_encode(p[0], p[1], p[2]);
            assert_eq!(code, reference(p[0], p[1], p[2]));
            assert_eq!(morton3_decode(code), p);
        }
        // Z-order: the code is monotonic in each coordinate.
        assert!(morton3_encode(5, 9, 2) < morton3_encode(6, 9, 2));
        assert!(morton3_encode(5, 9, 2) < morton3_encode(5, 10, 2));
        assert!(morton3_encode(5, 9, 2) < morton3_encode(5, 9, 3));
    }

    #[test]
    fn scalar_helpers() {
        assert_eq!(lerp(2.0_f64, 6.0, 0.0), 2.0);
        assert_eq!(lerp(2.0_f64, 6.0, 0.25), 3.0);
        assert_eq!(lerp(2.0_f32, 6.0, 1.0), 6.0);
        assert_eq!(lerp(2.0_f64, 6.0, 1.5), 8.0);

        assert_eq!(saturate(-0.5_f64), 0.0);
        assert_eq!(saturate(0.25_f64), 0.25);
        assert_eq!(saturate(7.0_f32), 1.0);
        assert_eq!(saturate(f64::NAN), 0.0);
        assert_eq!(saturate(f32::INFINITY), 1.0);
        assert_eq!(saturate(f32::NEG_INFINITY), 0.0);

        assert_eq!(smoothstep(0.0_f64, 1.0, -1.0), 0.0);
        assert_eq!(smoothstep(0.0_f64, 1.0, 0.5), 0.5);
        assert_eq!(smoothstep(0.0_f64, 1.0, 2.0), 1.0);
        assert_eq!(smoothstep(10.0_f32, 20.0, 12.5), 0.15625);
        // Mirrored and degenerate edges.
        assert_eq!(
            smoothstep(1.0_f64, 0.0, 0.25),
            smoothstep(0.0_f64, 1.0, 0.75)
        );
        assert_eq!(smoothstep(3.0_f64, 3.0, 2.0), 0.0);
        assert_eq!(smoothstep(3.0_f64, 3.0, 3.0), 0.0);
        assert_eq!(smoothstep(3.0_f64, 3.0, 4.0), 1.0);
        // Monotonic with zero slope at both ends.
        let mut previous = 0.0;
        for i in 0..=1000 {
            let value = smoothstep(0.0, 1.0, f64::from(i) / 1000.0);
            assert!(value >= previous);
            previous = value;
        }
        assert!(smoothstep(0.0_f64, 1.0, 1e-4) < 1e-7);
        assert!(1.0 - smoothstep(0.0_f64, 1.0, 1.0 - 1e-4) < 1e-7);

        assert_eq!(remap(5.0_f64, 0.0, 10.0, 100.0, 200.0), 150.0);
        assert_eq!(remap(-5.0_f32, 0.0, 10.0, 100.0, 200.0), 50.0);
        assert_eq!(remap(2.0_f64, 4.0, 0.0, 0.0, 1.0), 0.5);
        assert!(!remap(1.0_f64, 2.0, 2.0, 0.0, 1.0).is_finite());
    }
}
