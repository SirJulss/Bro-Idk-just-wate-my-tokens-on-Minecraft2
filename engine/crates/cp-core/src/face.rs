//! Cube faces, quad tangent frames, block orientation and texture rotation (contract §2.3).
//!
//! Everything in this module is a wire contract shared with the mesher, the content pipeline
//! and the GPU shaders.
//!
//! # Faces and tangent axes
//! Face indices are `0:+X 1:-X 2:+Y 3:-Y 4:+Z 5:-Z`; `axis = index >> 1`,
//! `negative = index & 1`. A face with normal axis `n` has the tangent axes
//! `u_axis = (n + 1) % 3` and `v_axis = (n + 2) % 3`, so `u × v = +n` (the positive axis
//! direction) for both faces of an axis: X→(Y, Z), Y→(Z, X), Z→(X, Y).
//!
//! # Texture mapping of a quad
//! A quad stores `uv_rot ∈ 0..4`. Let `(u, v)` be a point's coordinates in metres along the
//! face's `u_axis` and `v_axis`, and let `(a, b)` be the *texturing frame* of contract §2.3:
//!
//! - `(a, b) = (u, v)` on positive faces,
//! - `(a, b) = (v, u)` on negative faces,
//!
//! so that `a × b` is the outward normal on every face. The texture coordinates `(s, t)`
//! (`s` = image right, `t` = image up, in metres, one tile per metre at every LOD) are `(a, b)`
//! rotated by `uv_rot × 90°`:
//!
//! | `uv_rot` | `(s, t)` | on `+X +Y +Z` | on `-X -Y -Z` |
//! |---|---|---|---|
//! | 0 | `( a,  b)` | `( u,  v)` | `( v,  u)` |
//! | 1 | `( b, -a)` | `( v, -u)` | `( u, -v)` |
//! | 2 | `(-a, -b)` | `(-u, -v)` | `(-v, -u)` |
//! | 3 | `(-b,  a)` | `(-v,  u)` | `(-u,  v)` |
//!
//! Sampling uses `(fract(s), 1 - fract(t))` because image row 0 is the top.
//!
//! [`Face::uv_to_tex`] is the normative implementation of this rule (contract v2 §2.3 names
//! it). [`UV_TO_TEX`] is the same rule as a table of 2×2 matrices, for code that cannot call a
//! Rust function: a shader either embeds that table or computes `ab = negative ? uv.yx : uv`
//! followed by the rotation, and in both cases its mapping is to be tested against
//! [`UV_TO_TEX`] for all 6 faces × 4 rotations.
//!
//! Why negative faces swap: there `u × v` points *into* the block, so the `(u, v)` frame is
//! mirrored when seen from outside, while the texture axes of every block face are right-handed
//! about the outward normal (`tex_u × tex_v = n`). No rotation turns a mirrored frame into a
//! right-handed one, so negative faces need one fixed mirror first, and the swap `(v, u)` is
//! that mirror. (Contract v1 printed the rotation of `(u, v)` for all faces, which has no
//! solution on negative faces; v2 adopted this rule. A test keeps the argument checked.)
//!
//! # Orientation
//! [`Orientation`] is one of the 24 rotations of a block, `{ up, turn }`; see its
//! documentation for the exact rotation and [`Orientation::uv_rot`] for the texture rotation it
//! implies on every world face.

use core::ops::Neg;

use crate::math::{DVec3, IVec3};

mod orientation;

pub use orientation::{InvalidOrientation, Orientation};

const AXIS_UNITS: [IVec3; 3] = [IVec3::X, IVec3::Y, IVec3::Z];

/// The texture mapping of [`Face::uv_to_tex`] as data, for shaders and any other code that
/// cannot call it.
///
/// `UV_TO_TEX[face.index()][uv_rot]` is the matrix `[[su, sv], [tu, tv]]` with
///
/// ```text
/// s = su·u + sv·v
/// t = tu·u + tv·v
/// ```
///
/// where `(u, v)` are a point's coordinates along the face's `u_axis` and `v_axis` and
/// `(s, t)` its texture coordinates (image right, image up), see the
/// [module documentation](self). Every matrix is a signed permutation: its entries are `0`,
/// `1` or `-1`, with one non-zero entry per row and column. Its determinant is `+1` on
/// positive faces (a rotation) and `-1` on negative faces (the swap and a rotation).
///
/// The 24 matrices are pinned by a test. A shader's texture mapping has to reproduce all of
/// them.
pub const UV_TO_TEX: [[[[i8; 2]; 2]; 4]; 6] = {
    let mut table = [[[[0; 2]; 2]; 4]; 6];
    let mut face = 0;
    while face < 6 {
        let mut uv_rot = 0;
        while uv_rot < 4 {
            table[face][uv_rot] = uv_to_tex_matrix(face & 1 == 1, uv_rot);
            uv_rot += 1;
        }
        face += 1;
    }
    table
};

/// One entry of [`UV_TO_TEX`]: the rows are `s` and `t` written in terms of `(u, v)`.
const fn uv_to_tex_matrix(negative: bool, uv_rot: usize) -> [[i8; 2]; 2] {
    const fn opposite(row: [i8; 2]) -> [i8; 2] {
        [-row[0], -row[1]]
    }
    // The texturing frame (a, b): (u, v) on positive faces, (v, u) on negative faces.
    let (a, b) = if negative {
        ([0, 1], [1, 0])
    } else {
        ([1, 0], [0, 1])
    };
    match uv_rot & 3 {
        0 => [a, b],
        1 => [b, opposite(a)],
        2 => [opposite(a), opposite(b)],
        _ => [opposite(b), a],
    }
}

/// One of the six faces of a cube, named by its outward normal.
///
/// The discriminants are the wire indices of contract §2.3 and are ordered by them.
#[derive(
    Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug, serde::Serialize, serde::Deserialize,
)]
#[repr(u8)]
pub enum Face {
    /// `+X`, index 0.
    PosX = 0,
    /// `-X`, index 1.
    NegX = 1,
    /// `+Y`, index 2.
    PosY = 2,
    /// `-Y`, index 3.
    NegY = 3,
    /// `+Z`, index 4.
    PosZ = 4,
    /// `-Z`, index 5.
    NegZ = 5,
}

impl Face {
    /// All faces in index order.
    pub const ALL: [Self; 6] = [
        Self::PosX,
        Self::NegX,
        Self::PosY,
        Self::NegY,
        Self::PosZ,
        Self::NegZ,
    ];

    /// Wire index `0..6`.
    #[inline]
    #[must_use]
    pub const fn index(self) -> usize {
        self as usize
    }

    /// Face for a wire index, `None` for indices `>= 6`.
    #[inline]
    #[must_use]
    pub const fn from_index(index: usize) -> Option<Self> {
        match index {
            0 => Some(Self::PosX),
            1 => Some(Self::NegX),
            2 => Some(Self::PosY),
            3 => Some(Self::NegY),
            4 => Some(Self::PosZ),
            5 => Some(Self::NegZ),
            _ => None,
        }
    }

    /// Face of `axis` (`0:X 1:Y 2:Z`, taken modulo 3) on the negative or positive side.
    #[inline]
    #[must_use]
    pub const fn from_axis(axis: usize, negative: bool) -> Self {
        Self::ALL[(axis % 3) * 2 + negative as usize]
    }

    /// Face whose normal is the unit axis vector `normal`; `None` for any other vector.
    #[inline]
    #[must_use]
    pub const fn from_normal(normal: IVec3) -> Option<Self> {
        match (normal.x, normal.y, normal.z) {
            (1, 0, 0) => Some(Self::PosX),
            (-1, 0, 0) => Some(Self::NegX),
            (0, 1, 0) => Some(Self::PosY),
            (0, -1, 0) => Some(Self::NegY),
            (0, 0, 1) => Some(Self::PosZ),
            (0, 0, -1) => Some(Self::NegZ),
            _ => None,
        }
    }

    /// Face whose normal is closest to `direction`: the axis of the component with the largest
    /// magnitude, with that component's sign (contract §2.3).
    ///
    /// Ties go to the smaller axis index (X before Y before Z), and a zero component, `-0.0`
    /// included, counts as positive; so the zero vector yields `+X`. The result for vectors
    /// with NaN components is unspecified. This is how "which face of the cube planet am I on"
    /// is answered from `-gravity`.
    #[inline]
    #[must_use]
    pub fn from_dominant_axis(direction: DVec3) -> Self {
        let magnitude = direction.abs();
        let axis = if magnitude.x >= magnitude.y && magnitude.x >= magnitude.z {
            0
        } else if magnitude.y >= magnitude.z {
            1
        } else {
            2
        };
        // `-0.0 < 0.0` is false: a zero of either sign is positive.
        Self::from_axis(axis, direction[axis] < 0.0)
    }

    /// Normal axis: `0:X 1:Y 2:Z` (`index >> 1`).
    #[inline]
    #[must_use]
    pub const fn axis(self) -> usize {
        self as usize >> 1
    }

    /// True for `-X`, `-Y`, `-Z` (`index & 1`).
    #[inline]
    #[must_use]
    pub const fn is_negative(self) -> bool {
        self as usize & 1 == 1
    }

    /// Outward unit normal.
    #[inline]
    #[must_use]
    pub const fn normal(self) -> IVec3 {
        match self {
            Self::PosX => IVec3::X,
            Self::NegX => IVec3::NEG_X,
            Self::PosY => IVec3::Y,
            Self::NegY => IVec3::NEG_Y,
            Self::PosZ => IVec3::Z,
            Self::NegZ => IVec3::NEG_Z,
        }
    }

    /// First tangent axis, along which a quad's width runs: `(axis + 1) % 3`.
    #[inline]
    #[must_use]
    pub const fn u_axis(self) -> usize {
        (self.axis() + 1) % 3
    }

    /// Second tangent axis, along which a quad's height runs: `(axis + 2) % 3`.
    #[inline]
    #[must_use]
    pub const fn v_axis(self) -> usize {
        (self.axis() + 2) % 3
    }

    /// Unit vector of [`Face::u_axis`] (always the positive direction).
    #[inline]
    #[must_use]
    pub const fn u_dir(self) -> IVec3 {
        AXIS_UNITS[self.u_axis()]
    }

    /// Unit vector of [`Face::v_axis`] (always the positive direction).
    #[inline]
    #[must_use]
    pub const fn v_dir(self) -> IVec3 {
        AXIS_UNITS[self.v_axis()]
    }

    /// The face on the other side of the same axis (`index ^ 1`).
    #[inline]
    #[must_use]
    pub const fn opposite(self) -> Self {
        Self::ALL[self as usize ^ 1]
    }

    /// The texturing frame `(a, b)` of contract §2.3 as directions, with `a × b = normal()`:
    /// `(u_dir, v_dir)` on positive faces, `(v_dir, u_dir)` on negative faces.
    #[inline]
    #[must_use]
    pub const fn outward_tangents(self) -> (IVec3, IVec3) {
        if self.is_negative() {
            (self.v_dir(), self.u_dir())
        } else {
            (self.u_dir(), self.v_dir())
        }
    }

    /// Texture axes `(tex_u, tex_v)` of this face of an unrotated block (image right, image
    /// up), with `tex_u × tex_v = normal()`.
    ///
    /// Side faces (±X, ±Z) have `tex_v = +Y` and `tex_u = tex_v × n`; the top (+Y) has
    /// `tex_u = +X, tex_v = -Z`; the bottom (-Y) has `tex_u = +X, tex_v = +Z`.
    #[inline]
    #[must_use]
    pub const fn local_tex_axes(self) -> (IVec3, IVec3) {
        match self {
            Self::PosX => (IVec3::NEG_Z, IVec3::Y),
            Self::NegX => (IVec3::Z, IVec3::Y),
            Self::PosY => (IVec3::X, IVec3::NEG_Z),
            Self::NegY => (IVec3::X, IVec3::Z),
            Self::PosZ => (IVec3::X, IVec3::Y),
            Self::NegZ => (IVec3::NEG_X, IVec3::Y),
        }
    }

    /// Directions `(tex_u, tex_v)` in which the texture of a quad on this face runs when the
    /// quad carries `uv_rot` (only the low two bits are used): the
    /// [texturing frame](Face::outward_tangents) rotated by `uv_rot × 90°`.
    #[inline]
    #[must_use]
    pub fn quad_tex_axes(self, uv_rot: u8) -> (IVec3, IVec3) {
        let (a, b) = self.outward_tangents();
        match uv_rot & 3 {
            0 => (a, b),
            1 => (b, -a),
            2 => (-a, -b),
            _ => (-b, a),
        }
    }

    /// Texture coordinates `[s, t]` of a point of a quad on this face; the normative form of
    /// the texture rule of contract §2.3.
    ///
    /// `u` and `v` are the point's coordinates along [`Face::u_axis`] and [`Face::v_axis`]
    /// (for texture lookups in metres of grid space, so that tiles line up across quads and
    /// chunks); `uv_rot` is the quad's rotation (only the low two bits are used). `s` grows to
    /// the right of the image and `t` upwards, in the units of `u` and `v`. See the
    /// [module documentation](self) for the rule; [`UV_TO_TEX`] holds the same mapping as a
    /// table.
    #[inline]
    #[must_use]
    pub fn uv_to_tex<T: Copy + Neg<Output = T>>(self, uv_rot: u8, u: T, v: T) -> [T; 2] {
        let (a, b) = if self.is_negative() { (v, u) } else { (u, v) };
        match uv_rot & 3 {
            0 => [a, b],
            1 => [b, -a],
            2 => [-a, -b],
            _ => [-b, a],
        }
    }
}

/// Face for a vector known to be a unit axis vector.
const fn face_of_axis_vector(v: IVec3) -> Face {
    if v.x > 0 {
        Face::PosX
    } else if v.x < 0 {
        Face::NegX
    } else if v.y > 0 {
        Face::PosY
    } else if v.y < 0 {
        Face::NegY
    } else if v.z > 0 {
        Face::PosZ
    } else {
        Face::NegZ
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn indices_axes_and_normals() {
        let expected = [
            (Face::PosX, 0, false, IVec3::new(1, 0, 0)),
            (Face::NegX, 0, true, IVec3::new(-1, 0, 0)),
            (Face::PosY, 1, false, IVec3::new(0, 1, 0)),
            (Face::NegY, 1, true, IVec3::new(0, -1, 0)),
            (Face::PosZ, 2, false, IVec3::new(0, 0, 1)),
            (Face::NegZ, 2, true, IVec3::new(0, 0, -1)),
        ];
        for (index, (face, axis, negative, normal)) in expected.into_iter().enumerate() {
            assert_eq!(Face::ALL[index], face);
            assert_eq!(face.index(), index);
            assert_eq!(face as u8, index as u8);
            assert_eq!(Face::from_index(index), Some(face));
            assert_eq!(face.axis(), axis);
            assert_eq!(face.axis(), index >> 1);
            assert_eq!(face.is_negative(), negative);
            assert_eq!(face.is_negative(), index & 1 == 1);
            assert_eq!(face.normal(), normal);
            assert_eq!(Face::from_normal(normal), Some(face));
            assert_eq!(Face::from_axis(axis, negative), face);
            assert_eq!(face.opposite().normal(), -normal);
            assert_eq!(face.opposite().opposite(), face);
            assert_eq!(face.opposite().index(), index ^ 1);
            assert_eq!(face_of_axis_vector(normal), face);
        }
        assert_eq!(Face::from_index(6), None);
        assert_eq!(Face::from_index(usize::MAX), None);
        assert_eq!(Face::from_normal(IVec3::ZERO), None);
        assert_eq!(Face::from_normal(IVec3::new(1, 1, 0)), None);
        assert_eq!(Face::from_normal(IVec3::new(0, 2, 0)), None);
        assert!(Face::PosX < Face::NegX && Face::NegX < Face::PosY && Face::PosZ < Face::NegZ);
    }

    #[test]
    fn tangent_axes_are_right_handed_about_the_positive_axis() {
        // X -> (Y, Z), Y -> (Z, X), Z -> (X, Y); negative faces use the same axes.
        let expected = [(1, 2), (1, 2), (2, 0), (2, 0), (0, 1), (0, 1)];
        for face in Face::ALL {
            assert_eq!((face.u_axis(), face.v_axis()), expected[face.index()]);
            assert_eq!(face.u_axis(), (face.axis() + 1) % 3);
            assert_eq!(face.v_axis(), (face.axis() + 2) % 3);
            assert_eq!(face.u_dir()[face.u_axis()], 1);
            assert_eq!(face.v_dir()[face.v_axis()], 1);
            assert_eq!(face.u_dir().abs().element_sum(), 1);
            assert_eq!(face.v_dir().abs().element_sum(), 1);
            // u x v = +n, the positive direction of the normal axis, on both faces of an axis.
            let positive_normal = Face::from_axis(face.axis(), false).normal();
            assert_eq!(face.u_dir().cross(face.v_dir()), positive_normal);
            assert_eq!(face.opposite().u_dir(), face.u_dir());
            assert_eq!(face.opposite().v_dir(), face.v_dir());
            // The texturing frame spans the same plane with a x b = outward normal.
            let (a, b) = face.outward_tangents();
            assert_eq!(a.cross(b), face.normal());
            assert_eq!(a.dot(face.normal()), 0);
            assert_eq!(b.dot(face.normal()), 0);
        }
    }

    /// The tie rule of contract §2.3, written as a scan instead of comparisons: the first axis
    /// whose magnitude is not exceeded by any other, with the sign of its component, where a
    /// zero of either sign is positive.
    fn dominant_axis_reference(direction: DVec3) -> Face {
        let components = direction.to_array();
        let mut best = 0;
        for axis in 1..3 {
            if components[axis].abs() > components[best].abs() {
                best = axis;
            }
        }
        let negative = components[best] < 0.0;
        Face::ALL[2 * best + usize::from(negative)]
    }

    #[test]
    fn dominant_axis() {
        assert_eq!(
            Face::from_dominant_axis(DVec3::new(3.0, -2.0, 1.0)),
            Face::PosX
        );
        assert_eq!(
            Face::from_dominant_axis(DVec3::new(-3.0, 2.0, 1.0)),
            Face::NegX
        );
        assert_eq!(
            Face::from_dominant_axis(DVec3::new(0.1, 2.0, -1.0)),
            Face::PosY
        );
        assert_eq!(
            Face::from_dominant_axis(DVec3::new(0.1, -2.0, 1.0)),
            Face::NegY
        );
        assert_eq!(
            Face::from_dominant_axis(DVec3::new(0.1, 0.2, 0.3)),
            Face::PosZ
        );
        assert_eq!(
            Face::from_dominant_axis(DVec3::new(0.1, 0.2, -0.3)),
            Face::NegZ
        );
        // Ties go to the smaller axis index.
        assert_eq!(
            Face::from_dominant_axis(DVec3::new(1.0, 1.0, 1.0)),
            Face::PosX
        );
        assert_eq!(
            Face::from_dominant_axis(DVec3::new(-1.0, 1.0, 1.0)),
            Face::NegX
        );
        assert_eq!(
            Face::from_dominant_axis(DVec3::new(0.5, -1.0, 1.0)),
            Face::NegY
        );
        assert_eq!(
            Face::from_dominant_axis(DVec3::new(0.5, 1.0, -1.0)),
            Face::PosY
        );
        assert_eq!(
            Face::from_dominant_axis(DVec3::new(-2.0, 0.0, -2.0)),
            Face::NegX
        );
        // A zero component counts as positive, also with a negative sign.
        assert_eq!(Face::from_dominant_axis(DVec3::ZERO), Face::PosX);
        assert_eq!(Face::from_dominant_axis(DVec3::splat(-0.0)), Face::PosX);
        assert_eq!(
            Face::from_dominant_axis(DVec3::new(-0.0, 0.0, -0.0)),
            Face::PosX
        );
        for face in Face::ALL {
            let n = face.normal().as_dvec3();
            assert_eq!(Face::from_dominant_axis(n * 1.0e-300), face);
            assert_eq!(Face::from_dominant_axis(n * 1.0e300), face);
            assert_eq!(Face::from_dominant_axis(n * 5.0e-324), face);
            assert_eq!(
                Face::from_dominant_axis(n + DVec3::new(0.3, -0.3, 0.3)),
                face
            );
        }
        // Every combination of a few magnitudes and both zeros, against the reference.
        let values = [-2.0, -1.0, -0.5, -0.0, 0.0, 0.5, 1.0, 2.0];
        for x in values {
            for y in values {
                for z in values {
                    let direction = DVec3::new(x, y, z);
                    assert_eq!(
                        Face::from_dominant_axis(direction),
                        dominant_axis_reference(direction),
                        "{direction:?}"
                    );
                }
            }
        }
    }

    /// Texture axes derived directly from the wording of contract §2.3.
    fn tex_axes_from_rule(face: Face) -> (IVec3, IVec3) {
        let n = face.normal();
        match face {
            Face::PosY => (IVec3::X, IVec3::NEG_Z),
            Face::NegY => (IVec3::X, IVec3::Z),
            _ => {
                let tex_v = IVec3::Y;
                (tex_v.cross(n), tex_v)
            }
        }
    }

    #[test]
    fn local_texture_axes_follow_the_rule() {
        for face in Face::ALL {
            let (tex_u, tex_v) = face.local_tex_axes();
            assert_eq!((tex_u, tex_v), tex_axes_from_rule(face));
            assert_eq!(
                tex_u.cross(tex_v),
                face.normal(),
                "{face:?}: tex_u x tex_v must be n"
            );
            assert_eq!(tex_u.dot(face.normal()), 0);
            assert_eq!(tex_v.dot(face.normal()), 0);
        }
    }

    /// The rotation list of contract §2.3 applied to a pair of axes:
    /// `0:(a,b) 1:(b,-a) 2:(-a,-b) 3:(-b,a)`.
    fn rotate_pair(pair: (IVec3, IVec3), uv_rot: u8) -> (IVec3, IVec3) {
        let (a, b) = pair;
        match uv_rot {
            0 => (a, b),
            1 => (b, -a),
            2 => (-a, -b),
            3 => (-b, a),
            _ => unreachable!(),
        }
    }

    #[test]
    fn texturing_frame_follows_the_contract() {
        // Contract §2.3: positive faces (a, b) = (u, v), negative faces (a, b) = (v, u), then
        // the rotation list.
        for face in Face::ALL {
            let frame = if face.is_negative() {
                (face.v_dir(), face.u_dir())
            } else {
                (face.u_dir(), face.v_dir())
            };
            assert_eq!(face.outward_tangents(), frame);
            for uv_rot in 0..4 {
                assert_eq!(face.quad_tex_axes(uv_rot), rotate_pair(frame, uv_rot));
                assert_eq!(face.quad_tex_axes(uv_rot + 4), rotate_pair(frame, uv_rot));
            }
        }
    }

    #[test]
    fn uv_to_tex_agrees_with_the_texture_axes() {
        // For points p in the face plane, (s, t) = (p . tex_u, p . tex_v).
        for face in Face::ALL {
            for uv_rot in 0..4 {
                let (tex_u, tex_v) = face.quad_tex_axes(uv_rot);
                assert_eq!(tex_u.cross(tex_v), face.normal());
                for (u, v) in [(1, 0), (0, 1), (3, 7), (-2, 5)] {
                    let p = face.u_dir() * u + face.v_dir() * v;
                    assert_eq!(face.uv_to_tex(uv_rot, u, v), [p.dot(tex_u), p.dot(tex_v)]);
                    // Generic over the scalar type.
                    let [s, t] = face.uv_to_tex(uv_rot, f64::from(u), f64::from(v));
                    assert_eq!([s, t], [f64::from(p.dot(tex_u)), f64::from(p.dot(tex_v))]);
                }
                assert_eq!(
                    face.uv_to_tex(uv_rot + 4, 3, 7),
                    face.uv_to_tex(uv_rot, 3, 7)
                );
            }
        }
        // Positive faces rotate (u, v).
        assert_eq!(Face::PosZ.uv_to_tex(0, 3, 7), [3, 7]);
        assert_eq!(Face::PosZ.uv_to_tex(1, 3, 7), [7, -3]);
        assert_eq!(Face::PosZ.uv_to_tex(2, 3, 7), [-3, -7]);
        assert_eq!(Face::PosZ.uv_to_tex(3, 3, 7), [-7, 3]);
        // Negative faces rotate the swapped pair (v, u).
        assert_eq!(Face::NegZ.uv_to_tex(0, 3, 7), [7, 3]);
        assert_eq!(Face::NegZ.uv_to_tex(1, 3, 7), [3, -7]);
        assert_eq!(Face::NegZ.uv_to_tex(2, 3, 7), [-7, -3]);
        assert_eq!(Face::NegZ.uv_to_tex(3, 3, 7), [-3, 7]);
    }

    #[test]
    fn negative_faces_need_the_swapped_pair() {
        // Why the texturing frame swaps on negative faces: enumerate every frame of unit axes
        // that is right-handed about the outward normal (texture axes of block faces always
        // are) and see which rule reaches them.
        for face in Face::ALL {
            let normal = face.normal();
            let quad_axes = (face.u_dir(), face.v_dir());
            let in_plane: Vec<IVec3> = AXIS_UNITS
                .into_iter()
                .flat_map(|axis| [axis, -axis])
                .filter(|axis| axis.dot(normal) == 0)
                .collect();
            let mut right_handed = Vec::new();
            for &tex_u in &in_plane {
                for &tex_v in &in_plane {
                    if tex_u.cross(tex_v) == normal {
                        right_handed.push((tex_u, tex_v));
                    }
                }
            }
            assert_eq!(right_handed.len(), 4, "{face:?}");

            for frame in right_handed {
                let unswapped: Vec<u8> = (0..4)
                    .filter(|&rot| rotate_pair(quad_axes, rot) == frame)
                    .collect();
                let implemented: Vec<u8> = (0..4)
                    .filter(|&rot| face.quad_tex_axes(rot) == frame)
                    .collect();
                // The rule reaches every right-handed frame with exactly one rotation.
                assert_eq!(implemented.len(), 1, "{face:?} {frame:?}");
                if face.is_negative() {
                    // Rotating (u, v) itself reaches none of them: its frames are mirrored.
                    assert!(unswapped.is_empty(), "{face:?} {frame:?} {unswapped:?}");
                } else {
                    assert_eq!(unswapped, implemented, "{face:?} {frame:?}");
                }
            }
            for rot in 0..4 {
                let (tex_u, tex_v) = rotate_pair(quad_axes, rot);
                let handedness = tex_u.cross(tex_v).dot(normal);
                assert_eq!(handedness, if face.is_negative() { -1 } else { 1 });
            }
        }
    }

    #[test]
    fn uv_to_tex_table_is_pinned_and_equals_the_function() {
        // Written out from the table in the module documentation, not computed: rows are
        // (s, t), columns are (u, v). Wire contract with the shaders; never change it.
        const POSITIVE: [[[i8; 2]; 2]; 4] = [
            [[1, 0], [0, 1]],   // ( u,  v)
            [[0, 1], [-1, 0]],  // ( v, -u)
            [[-1, 0], [0, -1]], // (-u, -v)
            [[0, -1], [1, 0]],  // (-v,  u)
        ];
        const NEGATIVE: [[[i8; 2]; 2]; 4] = [
            [[0, 1], [1, 0]],   // ( v,  u)
            [[1, 0], [0, -1]],  // ( u, -v)
            [[0, -1], [-1, 0]], // (-v, -u)
            [[-1, 0], [0, 1]],  // (-u,  v)
        ];
        assert_eq!(
            UV_TO_TEX,
            [POSITIVE, NEGATIVE, POSITIVE, NEGATIVE, POSITIVE, NEGATIVE]
        );

        for face in Face::ALL {
            for uv_rot in 0..4_u8 {
                let [[su, sv], [tu, tv]] =
                    UV_TO_TEX[face.index()][usize::from(uv_rot)].map(|row| row.map(i32::from));
                // The table and the function are the same mapping.
                for (u, v) in [(1, 0), (0, 1), (3, 7), (-2, 5)] {
                    assert_eq!(
                        [su * u + sv * v, tu * u + tv * v],
                        face.uv_to_tex(uv_rot, u, v),
                        "{face:?} uv_rot {uv_rot}"
                    );
                }
                // Its rows are the texture axes, written in the quad's (u, v) frame.
                let (tex_u, tex_v) = face.quad_tex_axes(uv_rot);
                assert_eq!(tex_u, face.u_dir() * su + face.v_dir() * sv);
                assert_eq!(tex_v, face.u_dir() * tu + face.v_dir() * tv);
                // A rotation on positive faces; mirrored once on negative faces.
                let determinant = su * tv - sv * tu;
                assert_eq!(determinant, if face.is_negative() { -1 } else { 1 });
            }
        }
    }
}
