//! The 24 rotations of a block (contract §2.3).

use core::cmp::Ordering;

use serde::{Deserialize, Serialize};

use super::{Face, face_of_axis_vector};
use crate::math::IVec3;

/// Number of orientations: the rotation group of the cube.
const COUNT: usize = 24;

/// One of the 24 rotations of a cube: how a block is turned when placed (contract §2.3).
///
/// `up` is the world face that the block's top (local `+Y`) points to; `turn ∈ 0..4` counts
/// quarter turns about that world up direction, counter-clockwise seen from the tip of the up
/// vector (the right-hand rule). The rotation local → world is `R(up_world, turn × 90°) ∘
/// T(up)` with
///
/// | up | `(x, y, z)` local → world by `T(up)` |
/// |---|---|
/// | +Y | `( x,  y,  z)` |
/// | -Y | `( x, -y, -z)` |
/// | +X | `( y, -x,  z)` |
/// | -X | `(-y,  x,  z)` |
/// | +Z | `( x, -z,  y)` |
/// | -Z | `( x,  z, -y)` |
///
/// so the six orientations with `turn = 0` are exactly `T`. A block's **front** is its local
/// `+Z` face ([`Orientation::front`]). Blocks with six states (`up6`) use `turn = 0`; blocks
/// with three states (`axis3`) use `up = +a` for the axis `a`.
///
/// The wire index is `up.index() + 6 * turn` ([`Orientation::index`]), so
/// `Orientation::ALL[..6]` are the six `turn = 0` orientations in face order. Orientations are
/// ordered by index and serialise as their index (a `u8` in `0..24`).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Serialize, Deserialize)]
#[serde(try_from = "u8", into = "u8")]
pub struct Orientation {
    /// World face of the block's local `+Y`.
    up: Face,
    /// Quarter turns about the world up direction, `0..4`.
    turn: u8,
}

/// A number that is not the [index](Orientation::index) of an orientation (`>= 24`).
#[derive(Clone, Copy, PartialEq, Eq, Debug, thiserror::Error)]
#[error("orientation index {0} is invalid (must be 0..24)")]
pub struct InvalidOrientation(pub u8);

impl Default for Orientation {
    fn default() -> Self {
        Self::IDENTITY
    }
}

impl Ord for Orientation {
    fn cmp(&self, other: &Self) -> Ordering {
        self.index().cmp(&other.index())
    }
}

impl PartialOrd for Orientation {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl TryFrom<u8> for Orientation {
    type Error = InvalidOrientation;

    fn try_from(index: u8) -> Result<Self, Self::Error> {
        Self::from_index(usize::from(index)).ok_or(InvalidOrientation(index))
    }
}

impl From<Orientation> for u8 {
    fn from(orientation: Orientation) -> Self {
        // At most 23.
        orientation.index() as u8
    }
}

impl Orientation {
    /// The unrotated orientation: `up = +Y`, `turn = 0` (index 2).
    pub const IDENTITY: Self = Self {
        up: Face::PosY,
        turn: 0,
    };

    /// All 24 orientations in index order (`up.index() + 6 * turn`).
    pub const ALL: [Self; COUNT] = {
        let mut all = [Self::IDENTITY; COUNT];
        let mut index = 0;
        while index < COUNT {
            all[index] = Self {
                up: Face::ALL[index % 6],
                turn: (index / 6) as u8,
            };
            index += 1;
        }
        all
    };

    /// The orientation with the given up face and turn.
    ///
    /// # Panics
    /// If `turn >= 4`.
    #[inline]
    #[must_use]
    #[track_caller]
    pub const fn new(up: Face, turn: u8) -> Self {
        assert!(turn < 4, "Orientation::new: turn must be 0..4");
        Self { up, turn }
    }

    /// The orientation with up face `up` and no turn (`turn = 0`): the rotation `T(up)`.
    #[inline]
    #[must_use]
    pub const fn from_up(up: Face) -> Self {
        Self { up, turn: 0 }
    }

    /// The orientation with wire index `index`, `None` for indices `>= 24`.
    #[inline]
    #[must_use]
    pub const fn from_index(index: usize) -> Option<Self> {
        if index < COUNT {
            Some(Self::ALL[index])
        } else {
            None
        }
    }

    /// Wire index `up.index() + 6 * turn`, in `0..24`.
    #[inline]
    #[must_use]
    pub const fn index(self) -> usize {
        self.up.index() + 6 * self.turn as usize
    }

    /// The world face the block's local `+Y` (its top) points to.
    #[inline]
    #[must_use]
    pub const fn up(self) -> Face {
        self.up
    }

    /// Quarter turns about the world up direction, `0..4`.
    #[inline]
    #[must_use]
    pub const fn turn(self) -> u8 {
        self.turn
    }

    /// The world face the block's front (local `+Z`) points to.
    #[inline]
    #[must_use]
    pub const fn front(self) -> Face {
        self.world_face(Face::PosZ)
    }

    /// The orientation whose top points to `up` and whose front points to `front`, or `None`
    /// when the two faces lie on the same axis (no rotation does that).
    #[must_use]
    pub const fn from_up_front(up: Face, front: Face) -> Option<Self> {
        if up.axis() == front.axis() {
            return None;
        }
        let mut turn = 0;
        while turn < 4 {
            let orientation = Self { up, turn };
            if orientation.front().index() == front.index() {
                return Some(orientation);
            }
            turn += 1;
        }
        // Unreachable: the four turns carry the front through all four faces off the up axis.
        None
    }

    /// Rotates a block-local vector into world (grid) axes.
    ///
    /// Exact for components in `-i32::MAX..=i32::MAX` (`i32::MIN` wraps).
    #[inline]
    #[must_use]
    pub const fn rotate_vec(self, v: IVec3) -> IVec3 {
        let [x, y, z] = COLUMNS[self.index()];
        let v = [v.x, v.y, v.z];
        IVec3::new(
            dot([x[0], y[0], z[0]], v),
            dot([x[1], y[1], z[1]], v),
            dot([x[2], y[2], z[2]], v),
        )
    }

    /// Rotates a world (grid) vector into block-local axes; the inverse of
    /// [`Orientation::rotate_vec`].
    #[inline]
    #[must_use]
    pub const fn inverse_rotate_vec(self, v: IVec3) -> IVec3 {
        let [x, y, z] = COLUMNS[self.index()];
        let v = [v.x, v.y, v.z];
        IVec3::new(dot(x, v), dot(y, v), dot(z, v))
    }

    /// The world face that the block's local face `local` ends up on.
    #[inline]
    #[must_use]
    pub const fn world_face(self, local: Face) -> Face {
        face_of_axis_vector(self.rotate_vec(local.normal()))
    }

    /// The block-local face that shows on the world face `world`.
    #[inline]
    #[must_use]
    pub const fn local_face(self, world: Face) -> Face {
        face_of_axis_vector(self.inverse_rotate_vec(world.normal()))
    }

    /// `self ∘ other`: the rotation that applies `other` first and then `self`, so that
    /// `a.compose(b).rotate_vec(v) == a.rotate_vec(b.rotate_vec(v))`.
    #[inline]
    #[must_use]
    pub const fn compose(self, other: Self) -> Self {
        Self::ALL[COMPOSE[self.index()][other.index()] as usize]
    }

    /// The opposite rotation: `o.inverse().compose(o) == o.compose(o.inverse()) == IDENTITY`.
    #[inline]
    #[must_use]
    pub const fn inverse(self) -> Self {
        Self::ALL[INVERSE[self.index()] as usize]
    }

    /// The `uv_rot` (`0..4`) a quad on the world face `world` needs so that the texture of the
    /// block-local face showing there stays attached to the rotated block.
    ///
    /// Derived geometrically: the unique value for which [`Face::quad_tex_axes`]`(world,
    /// uv_rot)` equals the [texture axes](Face::local_tex_axes) of
    /// [`Orientation::local_face`]`(world)` rotated into the world by this orientation. Both
    /// frames are right-handed about the world normal, so the image of `tex_u` decides. The
    /// 24 × 6 values are evaluated at compile time and pinned by a test; the six `turn = 0`
    /// rows (by up face; columns by world face) are
    ///
    /// | up \ world face | +X | -X | +Y | -Y | +Z | -Z |
    /// |---|---|---|---|---|---|---|
    /// | +X | 2 | 3 | 0 | 3 | 3 | 0 |
    /// | -X | 0 | 1 | 2 | 1 | 1 | 2 |
    /// | +Y | 3 | 0 | 1 | 0 | 0 | 3 |
    /// | -Y | 1 | 2 | 1 | 0 | 2 | 1 |
    /// | +Z | 0 | 3 | 3 | 0 | 0 | 1 |
    /// | -Z | 2 | 1 | 1 | 2 | 0 | 1 |
    #[inline]
    #[must_use]
    pub const fn uv_rot(self, world: Face) -> u8 {
        UV_ROT[self.index()][world.index()]
    }
}

/// Dot product with wrapping arithmetic (the entries of a rotation are `0` or `±1`).
const fn dot(a: [i32; 3], b: [i32; 3]) -> i32 {
    a[0].wrapping_mul(b[0])
        .wrapping_add(a[1].wrapping_mul(b[1]))
        .wrapping_add(a[2].wrapping_mul(b[2]))
}

/// `T(up)` of contract §2.3 applied to `v`.
const fn t_up(up: Face, v: [i32; 3]) -> [i32; 3] {
    let [x, y, z] = v;
    match up {
        Face::PosY => [x, y, z],
        Face::NegY => [x, -y, -z],
        Face::PosX => [y, -x, z],
        Face::NegX => [-y, x, z],
        Face::PosZ => [x, -z, y],
        Face::NegZ => [x, z, -y],
    }
}

/// A quarter turn of `v` about the unit axis `n`, counter-clockwise seen from the tip of `n`:
/// `n × v + n (n · v)`.
const fn quarter_turn(n: [i32; 3], v: [i32; 3]) -> [i32; 3] {
    let along = dot(n, v);
    [
        n[1] * v[2] - n[2] * v[1] + n[0] * along,
        n[2] * v[0] - n[0] * v[2] + n[1] * along,
        n[0] * v[1] - n[1] * v[0] + n[2] * along,
    ]
}

/// Images of the local unit axes `X`, `Y`, `Z` (the columns of the rotation matrix) by index:
/// `R(up_world, turn × 90°) ∘ T(up)`.
const COLUMNS: [[[i32; 3]; 3]; COUNT] = {
    let mut table = [[[0; 3]; 3]; COUNT];
    let mut index = 0;
    while index < COUNT {
        let Orientation { up, turn } = Orientation::ALL[index];
        let normal = up.normal();
        let axis = [normal.x, normal.y, normal.z];
        let mut columns = [
            t_up(up, [1, 0, 0]),
            t_up(up, [0, 1, 0]),
            t_up(up, [0, 0, 1]),
        ];
        let mut quarter = 0;
        while quarter < turn {
            columns = [
                quarter_turn(axis, columns[0]),
                quarter_turn(axis, columns[1]),
                quarter_turn(axis, columns[2]),
            ];
            quarter += 1;
        }
        table[index] = columns;
        index += 1;
    }
    table
};

/// The index of the orientation with the given columns. Every rotation of the cube has one.
const fn index_of_columns(columns: [[i32; 3]; 3]) -> usize {
    let mut index = 0;
    while index < COUNT {
        let candidate = COLUMNS[index];
        let mut equal = true;
        let mut column = 0;
        while column < 3 {
            let mut row = 0;
            while row < 3 {
                equal &= candidate[column][row] == columns[column][row];
                row += 1;
            }
            column += 1;
        }
        if equal {
            return index;
        }
        index += 1;
    }
    panic!("not a rotation of the cube");
}

/// `COMPOSE[a][b]` is the index of `ALL[a] ∘ ALL[b]`.
const COMPOSE: [[u8; COUNT]; COUNT] = {
    let mut table = [[0; COUNT]; COUNT];
    let mut a = 0;
    while a < COUNT {
        let mut b = 0;
        while b < COUNT {
            let outer = Orientation::ALL[a];
            let mut columns = COLUMNS[b];
            let mut column = 0;
            while column < 3 {
                let [x, y, z] = columns[column];
                let image = outer.rotate_vec(IVec3::new(x, y, z));
                columns[column] = [image.x, image.y, image.z];
                column += 1;
            }
            table[a][b] = index_of_columns(columns) as u8;
            b += 1;
        }
        a += 1;
    }
    table
};

/// `INVERSE[a]` is the index of the inverse of `ALL[a]` (the transposed matrix).
const INVERSE: [u8; COUNT] = {
    let mut table = [0; COUNT];
    let mut index = 0;
    while index < COUNT {
        let columns = COLUMNS[index];
        let mut transposed = [[0; 3]; 3];
        let mut column = 0;
        while column < 3 {
            let mut row = 0;
            while row < 3 {
                transposed[column][row] = columns[row][column];
                row += 1;
            }
            column += 1;
        }
        table[index] = index_of_columns(transposed) as u8;
        index += 1;
    }
    table
};

/// The geometric definition of [`Orientation::uv_rot`].
const fn derive_uv_rot(orientation: Orientation, world: Face) -> u8 {
    let (local_tex_u, _) = orientation.local_face(world).local_tex_axes();
    let tex_u = orientation.rotate_vec(local_tex_u);
    let (a, b) = world.outward_tangents();
    let tex_u = [tex_u.x, tex_u.y, tex_u.z];
    // `tex_u` is one of a, b, -a, -b, the first axis of the rotations 0, 1, 2, 3 of (a, b).
    match (dot(tex_u, [a.x, a.y, a.z]), dot(tex_u, [b.x, b.y, b.z])) {
        (1, _) => 0,
        (_, 1) => 1,
        (-1, _) => 2,
        _ => 3,
    }
}

/// `UV_ROT[orientation][world face]`, see [`Orientation::uv_rot`].
const UV_ROT: [[u8; 6]; COUNT] = {
    let mut table = [[0; 6]; COUNT];
    let mut index = 0;
    while index < COUNT {
        let mut face = 0;
        while face < 6 {
            table[index][face] = derive_uv_rot(Orientation::ALL[index], Face::ALL[face]);
            face += 1;
        }
        index += 1;
    }
    table
};

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    const AXES: [IVec3; 3] = [IVec3::X, IVec3::Y, IVec3::Z];

    /// Matrix columns (images of X, Y, Z) of an orientation, through the public API only.
    fn columns(orientation: Orientation) -> [IVec3; 3] {
        AXES.map(|axis| orientation.rotate_vec(axis))
    }

    #[test]
    fn indices_constructors_and_order() {
        assert_eq!(Orientation::ALL.len(), 24);
        assert_eq!(Orientation::default(), Orientation::IDENTITY);
        assert_eq!(Orientation::IDENTITY, Orientation::new(Face::PosY, 0));
        assert_eq!(Orientation::IDENTITY.index(), 2);
        for (index, orientation) in Orientation::ALL.into_iter().enumerate() {
            let (up, turn) = (Face::ALL[index % 6], (index / 6) as u8);
            assert_eq!((orientation.up(), orientation.turn()), (up, turn));
            assert_eq!(orientation, Orientation::new(up, turn));
            assert_eq!(orientation.index(), up.index() + 6 * usize::from(turn));
            assert_eq!(Orientation::from_index(index), Some(orientation));
            assert_eq!(u8::from(orientation), index as u8);
            assert_eq!(Orientation::try_from(index as u8), Ok(orientation));
        }
        for face in Face::ALL {
            assert_eq!(Orientation::from_up(face), Orientation::new(face, 0));
            assert_eq!(Orientation::ALL[face.index()], Orientation::from_up(face));
        }
        assert_eq!(Orientation::from_index(24), None);
        assert_eq!(Orientation::try_from(24_u8), Err(InvalidOrientation(24)));
        assert_eq!(Orientation::try_from(255_u8), Err(InvalidOrientation(255)));
        assert!(InvalidOrientation(30).to_string().contains("30"));
        // Ordered by index.
        let mut sorted = Orientation::ALL;
        sorted.reverse();
        sorted.sort();
        assert_eq!(sorted, Orientation::ALL);
        assert!(Orientation::new(Face::NegZ, 0) < Orientation::new(Face::PosX, 1));
        let distinct: BTreeSet<Orientation> = Orientation::ALL.into_iter().collect();
        assert_eq!(distinct.len(), 24);
    }

    #[test]
    #[should_panic(expected = "turn must be 0..4")]
    fn new_rejects_a_fifth_turn() {
        let _ = Orientation::new(Face::PosY, 4);
    }

    /// `T(up)` literally from the table of contract §2.3.
    fn contract_t(up: Face, v: IVec3) -> IVec3 {
        let IVec3 { x, y, z } = v;
        match up {
            Face::PosY => IVec3::new(x, y, z),
            Face::NegY => IVec3::new(x, -y, -z),
            Face::PosX => IVec3::new(y, -x, z),
            Face::NegX => IVec3::new(-y, x, z),
            Face::PosZ => IVec3::new(x, -z, y),
            Face::NegZ => IVec3::new(x, z, -y),
        }
    }

    /// Rodrigues' formula for a rotation by `quarters × 90°` about the unit axis `n`
    /// (right-hand rule, i.e. counter-clockwise seen from the tip of `n`).
    fn rodrigues(n: IVec3, quarters: u8, v: IVec3) -> IVec3 {
        let (cos, sin) = [(1, 0), (0, 1), (-1, 0), (0, -1)][usize::from(quarters % 4)];
        v * cos + n.cross(v) * sin + n * (n.dot(v) * (1 - cos))
    }

    #[test]
    fn rotations_follow_the_contract() {
        for orientation in Orientation::ALL {
            let up = orientation.up();
            for x in -2..=2 {
                for y in -2..=2 {
                    for z in -2..=2 {
                        let v = IVec3::new(x, y, z);
                        // R(up_world, turn x 90 deg) o T(up).
                        let expected =
                            rodrigues(up.normal(), orientation.turn(), contract_t(up, v));
                        let rotated = orientation.rotate_vec(v);
                        assert_eq!(rotated, expected, "{orientation:?} {v}");
                        assert_eq!(orientation.inverse_rotate_vec(rotated), v);
                        assert_eq!(orientation.rotate_vec(orientation.inverse_rotate_vec(v)), v);
                    }
                }
            }
            // Orthonormal columns with determinant +1: a rotation, not a reflection.
            let [cx, cy, cz] = columns(orientation);
            for (i, a) in [cx, cy, cz].iter().enumerate() {
                for (j, b) in [cx, cy, cz].iter().enumerate() {
                    assert_eq!(a.dot(*b), i32::from(i == j), "{orientation:?}");
                }
            }
            assert_eq!(cx.dot(cy.cross(cz)), 1, "{orientation:?}");
            // The top points to `up` whatever the turn.
            assert_eq!(orientation.rotate_vec(IVec3::Y), up.normal());
            assert_eq!(orientation.world_face(Face::PosY), up);
            assert_eq!(orientation.local_face(up), Face::PosY);
        }
        // Large components stay exact: a rotation permutes them and flips signs.
        let big = IVec3::new(i32::MAX, -i32::MAX, 123_456_789);
        let magnitudes = |v: IVec3| {
            let mut magnitudes = v.to_array().map(i32::unsigned_abs);
            magnitudes.sort_unstable();
            magnitudes
        };
        for orientation in Orientation::ALL {
            let rotated = orientation.rotate_vec(big);
            assert_eq!(magnitudes(rotated), magnitudes(big));
            assert_eq!(orientation.inverse_rotate_vec(rotated), big);
        }
    }

    #[test]
    fn turn_zero_is_the_previous_table() {
        // The six turn-0 rotations are unchanged from E0 (contract §2.3, table T).
        let (x, y, z) = (2, 3, 5);
        let table = [
            (Face::PosY, IVec3::new(x, y, z)),
            (Face::NegY, IVec3::new(x, -y, -z)),
            (Face::PosX, IVec3::new(y, -x, z)),
            (Face::NegX, IVec3::new(-y, x, z)),
            (Face::PosZ, IVec3::new(x, -z, y)),
            (Face::NegZ, IVec3::new(x, z, -y)),
        ];
        for (up, world) in table {
            assert_eq!(
                Orientation::from_up(up).rotate_vec(IVec3::new(x, y, z)),
                world,
                "up {up:?}"
            );
        }
    }

    #[test]
    fn turns_are_counter_clockwise_seen_from_the_tip_of_up() {
        // On flat ground (up = +Y) the front walks +Z, +X, -Z, -X: seen from above with +X to
        // the right and -Z up the screen, that is counter-clockwise.
        let fronts: Vec<Face> = (0..4)
            .map(|turn| Orientation::new(Face::PosY, turn).front())
            .collect();
        assert_eq!(fronts, [Face::PosZ, Face::PosX, Face::NegZ, Face::NegX]);
        for up in Face::ALL {
            let n = up.normal();
            for turn in 0..4 {
                let orientation = Orientation::new(up, turn);
                let next = Orientation::new(up, (turn + 1) % 4);
                // One more quarter turn: every world vector v becomes n x v + n (n . v).
                for axis in AXES {
                    let v = orientation.rotate_vec(axis);
                    assert_eq!(next.rotate_vec(axis), n.cross(v) + n * n.dot(v));
                }
                assert_eq!(next.front().normal(), n.cross(orientation.front().normal()));
            }
        }
    }

    /// The 24 proper rotations of the cube, enumerated without the orientation code: every
    /// signed permutation matrix with determinant +1, as columns.
    fn all_cube_rotations() -> Vec<[IVec3; 3]> {
        let permutations = [
            [0, 1, 2],
            [0, 2, 1],
            [1, 0, 2],
            [1, 2, 0],
            [2, 0, 1],
            [2, 1, 0],
        ];
        let mut rotations = Vec::new();
        for permutation in permutations {
            for signs in 0..8 {
                let columns: [IVec3; 3] = core::array::from_fn(|column| {
                    let sign = if signs >> column & 1 == 1 { -1 } else { 1 };
                    AXES[permutation[column]] * sign
                });
                if columns[0].dot(columns[1].cross(columns[2])) == 1 {
                    rotations.push(columns);
                }
            }
        }
        rotations
    }

    #[test]
    fn the_24_orientations_are_the_24_rotations_of_the_cube() {
        let rotations = all_cube_rotations();
        assert_eq!(rotations.len(), 24);
        for rotation in rotations {
            let matching: Vec<Orientation> = Orientation::ALL
                .into_iter()
                .filter(|&orientation| columns(orientation) == rotation)
                .collect();
            assert_eq!(matching.len(), 1, "{rotation:?}");
        }
    }

    #[test]
    fn face_mapping_is_a_bijection_that_preserves_opposites() {
        for orientation in Orientation::ALL {
            let mut hit = [false; 6];
            for local in Face::ALL {
                let world = orientation.world_face(local);
                assert!(
                    !hit[world.index()],
                    "{orientation:?} maps two faces to {world:?}"
                );
                hit[world.index()] = true;
                assert_eq!(world.normal(), orientation.rotate_vec(local.normal()));
                assert_eq!(orientation.local_face(world), local);
                assert_eq!(orientation.world_face(orientation.local_face(local)), local);
                assert_eq!(orientation.world_face(local.opposite()), world.opposite());
                assert_eq!(orientation.local_face(world.opposite()), local.opposite());
            }
            assert_eq!(hit, [true; 6]);
            let up = orientation.up();
            assert_eq!(orientation.world_face(Face::NegY), up.opposite());
            for side in [Face::PosX, Face::NegX, Face::PosZ, Face::NegZ] {
                assert_ne!(orientation.world_face(side).axis(), up.axis());
            }
        }
    }

    #[test]
    fn compose_and_inverse_form_the_rotation_group() {
        let vectors = [
            IVec3::new(1, 2, 3),
            IVec3::new(-4, 0, 7),
            IVec3::new(0, -1, 0),
        ];
        for a in Orientation::ALL {
            assert_eq!(a.compose(Orientation::IDENTITY), a);
            assert_eq!(Orientation::IDENTITY.compose(a), a);
            assert_eq!(a.compose(a.inverse()), Orientation::IDENTITY);
            assert_eq!(a.inverse().compose(a), Orientation::IDENTITY);
            assert_eq!(a.inverse().inverse(), a);
            for v in vectors {
                assert_eq!(a.inverse().rotate_vec(v), a.inverse_rotate_vec(v));
            }
            let mut products = BTreeSet::new();
            for b in Orientation::ALL {
                let ab = a.compose(b);
                for v in vectors {
                    assert_eq!(
                        ab.rotate_vec(v),
                        a.rotate_vec(b.rotate_vec(v)),
                        "{a:?} {b:?}"
                    );
                }
                for local in Face::ALL {
                    assert_eq!(ab.world_face(local), a.world_face(b.world_face(local)));
                }
                products.insert(ab);
                for c in [
                    Orientation::ALL[7],
                    Orientation::ALL[16],
                    Orientation::ALL[23],
                ] {
                    assert_eq!(ab.compose(c), a.compose(b.compose(c)));
                }
            }
            // Composing with a is a permutation of the group.
            assert_eq!(products.len(), 24);
        }
        // Turns about the same axis add up.
        for t in 0..4 {
            for s in 0..4 {
                let turned_again =
                    Orientation::new(Face::PosY, s).compose(Orientation::new(Face::PosY, t));
                assert_eq!(turned_again, Orientation::new(Face::PosY, (s + t) % 4));
            }
        }
        // Turning about the world up after T(up) is turning about the local up before it:
        // R(up_world, s) o T(up) = T(up) o R(+Y, s).
        for up in Face::ALL {
            for s in 0..4 {
                assert_eq!(
                    Orientation::new(up, s),
                    Orientation::from_up(up).compose(Orientation::new(Face::PosY, s))
                );
            }
        }
    }

    #[test]
    fn from_up_front_finds_the_one_orientation() {
        let mut seen = BTreeSet::new();
        for up in Face::ALL {
            for front in Face::ALL {
                let found = Orientation::from_up_front(up, front);
                if up.axis() == front.axis() {
                    assert_eq!(found, None, "{up:?} {front:?}");
                    continue;
                }
                let orientation = found.expect("non-parallel faces have an orientation");
                assert_eq!(orientation.up(), up);
                assert_eq!(orientation.front(), front);
                assert_eq!(orientation.world_face(Face::PosZ), front);
                assert!(seen.insert(orientation));
            }
        }
        assert_eq!(seen.len(), 24);
        for orientation in Orientation::ALL {
            assert_eq!(
                Orientation::from_up_front(orientation.up(), orientation.front()),
                Some(orientation)
            );
        }
    }

    /// The `uv_rot` of one world face, derived by brute force from the rotation's columns
    /// alone: try all four values and keep those for which moving across the world face moves
    /// through the image exactly as on the unrotated block's face. Uses [`Face::uv_to_tex`]
    /// (normative) and the texture axes as worded in contract §2.3; nothing of the
    /// orientation code.
    fn brute_force_uv_rot(rotation: [IVec3; 3], world: Face) -> Vec<u8> {
        let rotate = |v: IVec3| rotation[0] * v.x + rotation[1] * v.y + rotation[2] * v.z;
        let local: Vec<Face> = Face::ALL
            .into_iter()
            .filter(|face| rotate(face.normal()) == world.normal())
            .collect();
        assert_eq!(local.len(), 1);
        let local = local[0];
        let (tex_u, tex_v) = match local {
            Face::PosY => (IVec3::X, IVec3::NEG_Z),
            Face::NegY => (IVec3::X, IVec3::Z),
            side => (IVec3::Y.cross(side.normal()), IVec3::Y),
        };
        let (world_u, world_v) = (rotate(tex_u), rotate(tex_v));
        (0..4)
            .filter(|&uv_rot| {
                [(1, 0), (0, 1), (2, -3)].into_iter().all(|(u, v)| {
                    let step = world.u_dir() * u + world.v_dir() * v;
                    world.uv_to_tex(uv_rot, u, v) == [step.dot(world_u), step.dot(world_v)]
                })
            })
            .collect()
    }

    #[test]
    fn uv_rot_matches_a_brute_force_derivation() {
        for rotation in all_cube_rotations() {
            let orientation = Orientation::ALL
                .into_iter()
                .find(|&orientation| columns(orientation) == rotation)
                .expect("every rotation is an orientation");
            for world in Face::ALL {
                assert_eq!(
                    brute_force_uv_rot(rotation, world),
                    [orientation.uv_rot(world)],
                    "{orientation:?} {world:?}"
                );
            }
        }
    }

    #[test]
    fn uv_rot_matches_rotated_texture_axes() {
        for orientation in Orientation::ALL {
            for world in Face::ALL {
                let local = orientation.local_face(world);
                let (local_u, local_v) = local.local_tex_axes();
                let expected = (
                    orientation.rotate_vec(local_u),
                    orientation.rotate_vec(local_v),
                );
                assert_eq!(expected.0.cross(expected.1), world.normal());
                let uv_rot = orientation.uv_rot(world);
                assert!(uv_rot < 4);
                assert_eq!(
                    world.quad_tex_axes(uv_rot),
                    expected,
                    "{orientation:?} {world:?}"
                );
                assert_eq!(derive_uv_rot(orientation, world), uv_rot);
            }
        }
    }

    #[test]
    fn uv_rot_table_is_pinned() {
        // Rows: orientation index (up.index() + 6 * turn); columns: world face in face order.
        // Wire contract with the content pipeline and the shaders; never change these numbers.
        // The turn-0 rows are the E0 table. All rows were also recomputed by an independent
        // script from the text of contract §2.3 before being pinned.
        const PINNED: [[u8; 6]; 24] = [
            [2, 3, 0, 3, 3, 0], //  0: up +X, turn 0, front +Z
            [0, 1, 2, 1, 1, 2], //  1: up -X, turn 0, front +Z
            [3, 0, 1, 0, 0, 3], //  2: up +Y, turn 0, front +Z (identity)
            [1, 2, 1, 0, 2, 1], //  3: up -Y, turn 0, front -Z
            [0, 3, 3, 0, 0, 1], //  4: up +Z, turn 0, front -Y
            [2, 1, 1, 2, 0, 1], //  5: up -Z, turn 0, front +Y
            [3, 2, 0, 3, 3, 0], //  6: up +X, turn 1, front -Y
            [3, 2, 2, 1, 1, 2], //  7: up -X, turn 1, front +Y
            [3, 0, 2, 3, 0, 3], //  8: up +Y, turn 1, front +X
            [1, 2, 0, 1, 2, 1], //  9: up -Y, turn 1, front +X
            [0, 3, 3, 0, 1, 0], // 10: up +Z, turn 1, front +X
            [2, 1, 1, 2, 3, 2], // 11: up -Z, turn 1, front +X
            [0, 1, 0, 3, 3, 0], // 12: up +X, turn 2, front -Z
            [2, 3, 2, 1, 1, 2], // 13: up -X, turn 2, front -Z
            [3, 0, 3, 2, 0, 3], // 14: up +Y, turn 2, front -Z
            [1, 2, 3, 2, 2, 1], // 15: up -Y, turn 2, front +Z
            [0, 3, 3, 0, 2, 3], // 16: up +Z, turn 2, front +Y
            [2, 1, 1, 2, 2, 3], // 17: up -Z, turn 2, front -Y
            [1, 0, 0, 3, 3, 0], // 18: up +X, turn 3, front +Y
            [1, 0, 2, 1, 1, 2], // 19: up -X, turn 3, front -Y
            [3, 0, 0, 1, 0, 3], // 20: up +Y, turn 3, front -X
            [1, 2, 2, 3, 2, 1], // 21: up -Y, turn 3, front -X
            [0, 3, 3, 0, 3, 2], // 22: up +Z, turn 3, front -X
            [2, 1, 1, 2, 1, 0], // 23: up -Z, turn 3, front -X
        ];
        let table: [[u8; 6]; 24] = core::array::from_fn(|index| {
            Face::ALL.map(|world| Orientation::ALL[index].uv_rot(world))
        });
        assert_eq!(table, PINNED);
        let fronts: Vec<usize> = Orientation::ALL
            .into_iter()
            .map(|orientation| orientation.front().index())
            .collect();
        assert_eq!(
            fronts,
            [
                4, 4, 4, 5, 3, 2, 3, 2, 0, 0, 0, 0, 5, 5, 5, 4, 2, 3, 2, 3, 1, 1, 1, 1
            ]
        );
    }

    #[test]
    fn textures_stay_upright_on_rotated_blocks() {
        // A grass block in any orientation: on each of its four side faces the texture's up
        // direction (tex_v) points along the block's up, so the grass edge is at the top.
        for orientation in Orientation::ALL {
            let up = orientation.up();
            for world in Face::ALL {
                let (tex_u, tex_v) = world.quad_tex_axes(orientation.uv_rot(world));
                assert_eq!(tex_u.cross(tex_v), world.normal());
                if world.axis() != up.axis() {
                    assert_eq!(tex_v, up.normal(), "{orientation:?} {world:?}");
                }
            }
        }
    }
}
