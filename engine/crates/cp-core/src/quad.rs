//! The packed quad: the one mesh primitive of the engine (contract §2.3, §4.1, v2 layout).
//!
//! A quad is an axis-aligned rectangle of `w × h` cells on one face of a chunk cell. Eight bytes
//! describe it; the vertex shader expands them into four vertices and two triangles ("vertex
//! pulling", §5.3).
//!
//! # Bit layout (wire contract with the shaders)
//! ```text
//! w0: bits  0–4   x        min cell, 0..32
//!     bits  5–9   y
//!     bits 10–14  z
//!     bits 15–19  w - 1    width  in cells along the face's u axis, w in 1..=32
//!     bits 20–24  h - 1    height in cells along the face's v axis, h in 1..=32
//!     bits 25–27  face     0:+X 1:-X 2:+Y 3:-Y 4:+Z 5:-Z
//!     bits 28–31  reserved, zero
//! w1: bits  0–15  material
//!     bits 16–17  uv_rot   0..4, see `face`
//!     bit  18     authored (the cells carry a player edit)
//!     bits 19–30  class payload, 0 for cube and cutout quads
//!     bit  31     reserved, zero
//! ```
//!
//! # Geometry and drawing order
//! The quad lies in the plane `n = cell_n + 1` for positive faces and `n = cell_n` for negative
//! faces (`n` = the face's normal axis). In the plane it spans `[cell_u, cell_u + w] ×
//! [cell_v, cell_v + h]` with the corners `c0 = (0, 0)`, `c1 = (w, 0)`, `c2 = (w, h)`,
//! `c3 = (0, h)` in `(u, v)`. All coordinates are in cells of the chunk; multiply by the
//! chunk's cell size for metres.
//!
//! Every quad is drawn indexed with [`QUAD_VERTICES`] vertices and the static index pattern
//! [`QUAD_INDEX_PATTERN`] `[0, 1, 2, 0, 2, 3]`. Vertex `k` of a quad is corner
//! `CORNER_MAP[k]`, with [`CORNER_MAP_POSITIVE`] `[0, 1, 2, 3]` and [`CORNER_MAP_NEGATIVE`]
//! `[0, 3, 2, 1]`, so both triangles wind counter-clockwise seen from outside the block on
//! every face. Back faces are culled.
//!
//! ```
//! use cp_core::face::Face;
//! use cp_core::ids::MaterialId;
//! use cp_core::math::IVec3;
//! use cp_core::quad::PackedQuad;
//!
//! // The top faces of the cells (4..7, 9, 2..4): 3 cells along X, 2 along Z.
//! // On a Y face the width runs along Z and the height along X.
//! let quad = PackedQuad::new([4, 9, 2], 2, 3, Face::PosY, MaterialId(7), 0, false)?;
//! assert_eq!(quad.corners()[0], IVec3::new(4, 10, 2));
//! assert_eq!(quad.corners()[2], IVec3::new(7, 10, 4));
//! assert_eq!(quad.covered_cells().count(), 6);
//! // A rectangle that leaves the chunk is rejected.
//! assert!(PackedQuad::new([4, 9, 31], 2, 3, Face::PosY, MaterialId(7), 0, false).is_err());
//! # Ok::<(), cp_core::quad::QuadError>(())
//! ```

use bytemuck::{Pod, Zeroable};

use crate::coords::CHUNK_SIZE;
use crate::face::Face;
use crate::ids::MaterialId;
use crate::math::IVec3;

/// Vertices per quad in the indexed draw.
pub const QUAD_VERTICES: u32 = 4;
/// Indices per quad in the indexed draw (two triangles).
pub const QUAD_INDICES: u32 = 6;
/// The index pattern of one quad, relative to its first vertex: triangles `(v0, v1, v2)` and
/// `(v0, v2, v3)`. The static index buffer repeats it with a base of `QUAD_VERTICES` per quad.
pub const QUAD_INDEX_PATTERN: [u32; QUAD_INDICES as usize] = [0, 1, 2, 0, 2, 3];

/// Corner index (into [`PackedQuad::corners`]) of each of the four vertices of a quad on a
/// positive face.
pub const CORNER_MAP_POSITIVE: [usize; QUAD_VERTICES as usize] = [0, 1, 2, 3];
/// Corner index of each of the four vertices of a quad on a negative face. Reversing the
/// corner order there keeps the triangles counter-clockwise seen from outside.
pub const CORNER_MAP_NEGATIVE: [usize; QUAD_VERTICES as usize] = [0, 3, 2, 1];
/// Both corner maps, indexed by the face sign (`face.index() & 1`), as the vertex shader
/// looks them up.
pub const CORNER_MAP: [[usize; QUAD_VERTICES as usize]; 2] =
    [CORNER_MAP_POSITIVE, CORNER_MAP_NEGATIVE];

/// The corner map for `face`.
#[inline]
#[must_use]
pub const fn corner_map(face: Face) -> [usize; QUAD_VERTICES as usize] {
    CORNER_MAP[face.is_negative() as usize]
}

/// Largest class payload (12 bits).
pub const MAX_PAYLOAD: u16 = 0xFFF;

/// Chunk edge in cells as the type of the packed fields.
const CHUNK_EDGE: u8 = CHUNK_SIZE as u8;

const COORD_MASK: u32 = 0x1F;
const Y_SHIFT: u32 = 5;
const Z_SHIFT: u32 = 10;
const W_SHIFT: u32 = 15;
const H_SHIFT: u32 = 20;
const FACE_SHIFT: u32 = 25;
const FACE_MASK: u32 = 0x7;
const W0_RESERVED: u32 = 0xF000_0000;
const MATERIAL_MASK: u32 = 0xFFFF;
const UV_ROT_SHIFT: u32 = 16;
const UV_ROT_MASK: u32 = 0x3;
const AUTHORED_BIT: u32 = 1 << 18;
const PAYLOAD_SHIFT: u32 = 19;
const PAYLOAD_MASK: u32 = MAX_PAYLOAD as u32;
const W1_RESERVED: u32 = 0x8000_0000;

/// Why a quad description is not a valid [`PackedQuad`].
#[derive(Clone, Copy, PartialEq, Eq, Debug, thiserror::Error)]
pub enum QuadError {
    /// A cell coordinate is not inside the chunk.
    #[error("quad cell {cell:?} is outside the chunk (each coordinate must be < 32)")]
    CellOutOfRange {
        /// The offending min cell.
        cell: [u8; 3],
    },
    /// Width or height is zero or larger than a chunk.
    #[error("quad size {w}x{h} is invalid (width and height must be 1..=32)")]
    SizeOutOfRange {
        /// Requested width in cells.
        w: u8,
        /// Requested height in cells.
        h: u8,
    },
    /// The rectangle starts inside the chunk but reaches beyond its far side.
    #[error("quad at {cell:?} with size {w}x{h} on face {face:?} extends beyond the chunk")]
    ExtendsBeyondChunk {
        /// Min cell.
        cell: [u8; 3],
        /// Width in cells.
        w: u8,
        /// Height in cells.
        h: u8,
        /// Face of the quad (it determines the axes of `w` and `h`).
        face: Face,
    },
    /// The texture rotation is not one of the four quarter turns.
    #[error("uv_rot {uv_rot} is invalid (must be 0..4)")]
    UvRotOutOfRange {
        /// The offending value.
        uv_rot: u8,
    },
    /// The class payload does not fit into its 12 bits.
    #[error("quad payload {payload} is invalid (must be at most 4095)")]
    PayloadOutOfRange {
        /// The offending value.
        payload: u16,
    },
    /// The face field of packed words holds 6 or 7.
    #[error("packed quad has invalid face bits {bits}")]
    InvalidFace {
        /// The raw 3-bit value.
        bits: u8,
    },
    /// Reserved bits of packed words are not zero.
    #[error("packed quad has reserved bits set (w0 = {w0:#010x}, w1 = {w1:#010x})")]
    ReservedBits {
        /// First word.
        w0: u32,
        /// Second word.
        w1: u32,
    },
}

/// A quad packed into two `u32` words (8 bytes), see the [module documentation](self) for the
/// layout, geometry and drawing order.
///
/// Values made by [`PackedQuad::new`] and [`PackedQuad::with_payload`] are always valid.
/// Words from elsewhere (a cache file, a byte cast) must pass [`PackedQuad::validate`] before
/// the accessors are used. Ordered by `w0`, then `w1`.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug, Pod, Zeroable)]
#[repr(C)]
pub struct PackedQuad {
    /// Cell, size and face.
    pub w0: u32,
    /// Material, texture rotation, authored flag and class payload.
    pub w1: u32,
}

impl PackedQuad {
    /// Packs a quad with class payload 0 (cube and cutout quads; see
    /// [`PackedQuad::with_payload`] for the others).
    ///
    /// `cell` is the min cell `[x, y, z]`, `w` and `h` the size in cells along the face's
    /// `u_axis` and `v_axis`, `uv_rot` the texture rotation in quarter turns and `authored`
    /// marks quads of cells that carry a player edit (they keep their material colour at a
    /// distance, contract §8).
    ///
    /// # Errors
    /// When a cell coordinate is `>= 32`, a size is outside `1..=32`, the rectangle leaves the
    /// chunk, or `uv_rot >= 4`.
    #[inline]
    pub const fn new(
        cell: [u8; 3],
        w: u8,
        h: u8,
        face: Face,
        material: MaterialId,
        uv_rot: u8,
        authored: bool,
    ) -> Result<Self, QuadError> {
        if cell[0] >= CHUNK_EDGE || cell[1] >= CHUNK_EDGE || cell[2] >= CHUNK_EDGE {
            return Err(QuadError::CellOutOfRange { cell });
        }
        if w == 0 || w > CHUNK_EDGE || h == 0 || h > CHUNK_EDGE {
            return Err(QuadError::SizeOutOfRange { w, h });
        }
        if cell[face.u_axis()] + w > CHUNK_EDGE || cell[face.v_axis()] + h > CHUNK_EDGE {
            return Err(QuadError::ExtendsBeyondChunk { cell, w, h, face });
        }
        if uv_rot > UV_ROT_MASK as u8 {
            return Err(QuadError::UvRotOutOfRange { uv_rot });
        }
        Ok(Self {
            w0: cell[0] as u32
                | (cell[1] as u32) << Y_SHIFT
                | (cell[2] as u32) << Z_SHIFT
                | ((w - 1) as u32) << W_SHIFT
                | ((h - 1) as u32) << H_SHIFT
                | (face as u32) << FACE_SHIFT,
            w1: material.0 as u32
                | (uv_rot as u32) << UV_ROT_SHIFT
                | if authored { AUTHORED_BIT } else { 0 },
        })
    }

    /// The same quad with the class payload replaced by `payload` (`0..=MAX_PAYLOAD`).
    ///
    /// The payload carries class-specific data of non-cube quads; cube and cutout quads keep
    /// 0 (contract §4.1).
    ///
    /// # Errors
    /// When `payload > MAX_PAYLOAD`.
    #[inline]
    pub const fn with_payload(self, payload: u16) -> Result<Self, QuadError> {
        if payload > MAX_PAYLOAD {
            return Err(QuadError::PayloadOutOfRange { payload });
        }
        Ok(Self {
            w0: self.w0,
            w1: (self.w1 & !(PAYLOAD_MASK << PAYLOAD_SHIFT)) | (payload as u32) << PAYLOAD_SHIFT,
        })
    }

    /// Checks words that did not come from [`PackedQuad::new`].
    ///
    /// # Errors
    /// When the face bits hold 6 or 7, a reserved bit is set, or the rectangle leaves the chunk.
    pub const fn validate(self) -> Result<(), QuadError> {
        let face_bits = ((self.w0 >> FACE_SHIFT) & FACE_MASK) as u8;
        let Some(face) = Face::from_index(face_bits as usize) else {
            return Err(QuadError::InvalidFace { bits: face_bits });
        };
        if self.w0 & W0_RESERVED != 0 || self.w1 & W1_RESERVED != 0 {
            return Err(QuadError::ReservedBits {
                w0: self.w0,
                w1: self.w1,
            });
        }
        let cell = self.cell();
        let (w, h) = (self.width(), self.height());
        if cell[face.u_axis()] + w > CHUNK_EDGE || cell[face.v_axis()] + h > CHUNK_EDGE {
            return Err(QuadError::ExtendsBeyondChunk { cell, w, h, face });
        }
        Ok(())
    }

    /// Min cell x, `0..32`.
    #[inline]
    #[must_use]
    pub const fn x(self) -> u8 {
        (self.w0 & COORD_MASK) as u8
    }

    /// Min cell y, `0..32`.
    #[inline]
    #[must_use]
    pub const fn y(self) -> u8 {
        ((self.w0 >> Y_SHIFT) & COORD_MASK) as u8
    }

    /// Min cell z, `0..32`.
    #[inline]
    #[must_use]
    pub const fn z(self) -> u8 {
        ((self.w0 >> Z_SHIFT) & COORD_MASK) as u8
    }

    /// Min cell `[x, y, z]`.
    #[inline]
    #[must_use]
    pub const fn cell(self) -> [u8; 3] {
        [self.x(), self.y(), self.z()]
    }

    /// Width in cells along the face's `u_axis`, `1..=32`.
    #[inline]
    #[must_use]
    pub const fn width(self) -> u8 {
        ((self.w0 >> W_SHIFT) & COORD_MASK) as u8 + 1
    }

    /// Height in cells along the face's `v_axis`, `1..=32`.
    #[inline]
    #[must_use]
    pub const fn height(self) -> u8 {
        ((self.w0 >> H_SHIFT) & COORD_MASK) as u8 + 1
    }

    /// The face the quad lies on.
    ///
    /// # Panics
    /// If the face bits hold 6 or 7, which no validated quad does.
    #[inline]
    #[must_use]
    pub const fn face(self) -> Face {
        match Face::from_index(((self.w0 >> FACE_SHIFT) & FACE_MASK) as usize) {
            Some(face) => face,
            None => panic!("PackedQuad with invalid face bits; validate untrusted quads first"),
        }
    }

    /// Surface material.
    #[inline]
    #[must_use]
    pub const fn material(self) -> MaterialId {
        MaterialId((self.w1 & MATERIAL_MASK) as u16)
    }

    /// Texture rotation in quarter turns, `0..4` (see [`Face::uv_to_tex`]).
    #[inline]
    #[must_use]
    pub const fn uv_rot(self) -> u8 {
        ((self.w1 >> UV_ROT_SHIFT) & UV_ROT_MASK) as u8
    }

    /// True for quads of cells that carry a player edit.
    #[inline]
    #[must_use]
    pub const fn is_authored(self) -> bool {
        self.w1 & AUTHORED_BIT != 0
    }

    /// The class payload, `0..=MAX_PAYLOAD`; 0 for cube and cutout quads.
    #[inline]
    #[must_use]
    pub const fn payload(self) -> u16 {
        ((self.w1 >> PAYLOAD_SHIFT) & PAYLOAD_MASK) as u16
    }

    /// The four corners `[c0, c1, c2, c3]` as lattice points of the chunk's cell grid
    /// (each coordinate in `0..=32`): `c0` is the min corner in the quad's plane, `c1 = c0 + w·u`,
    /// `c2 = c0 + w·u + h·v`, `c3 = c0 + h·v`.
    ///
    /// # Panics
    /// Like [`PackedQuad::face`].
    #[inline]
    #[must_use]
    pub fn corners(self) -> [IVec3; 4] {
        let face = self.face();
        let mut c0 = IVec3::new(self.x().into(), self.y().into(), self.z().into());
        if !face.is_negative() {
            c0 += face.normal();
        }
        let across = face.u_dir() * i32::from(self.width());
        let up = face.v_dir() * i32::from(self.height());
        [c0, c0 + across, c0 + across + up, c0 + up]
    }

    /// The four vertices in vertex order (vertex `k` is corner [`corner_map`]`(face)[k]`), as
    /// lattice points like [`PackedQuad::corners`]. [`QUAD_INDEX_PATTERN`] turns them into two
    /// counter-clockwise triangles seen from outside.
    ///
    /// # Panics
    /// Like [`PackedQuad::face`].
    #[inline]
    #[must_use]
    pub fn vertices(self) -> [IVec3; QUAD_VERTICES as usize] {
        let corners = self.corners();
        corner_map(self.face()).map(|corner| corners[corner])
    }

    /// The `w × h` cells whose face this quad covers, `[x, y, z]` each, `v` outer and `u`
    /// inner.
    ///
    /// # Panics
    /// Like [`PackedQuad::face`].
    #[inline]
    pub fn covered_cells(self) -> impl Iterator<Item = [u8; 3]> {
        let face = self.face();
        let (u_axis, v_axis) = (face.u_axis(), face.v_axis());
        let (w, h) = (self.width(), self.height());
        let origin = self.cell();
        (0..h).flat_map(move |dv| {
            (0..w).map(move |du| {
                let mut cell = origin;
                cell[u_axis] += du;
                cell[v_axis] += dv;
                cell
            })
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    /// The layout of the contract, written out independently of the implementation.
    #[allow(clippy::too_many_arguments)]
    fn reference_words(
        x: u32,
        y: u32,
        z: u32,
        w: u32,
        h: u32,
        face: u32,
        material: u32,
        uv_rot: u32,
        authored: u32,
        payload: u32,
    ) -> (u32, u32) {
        (
            x | y << 5 | z << 10 | (w - 1) << 15 | (h - 1) << 20 | face << 25,
            material | uv_rot << 16 | authored << 18 | payload << 19,
        )
    }

    #[test]
    fn layout_is_eight_bytes_of_plain_data() {
        assert_eq!(size_of::<PackedQuad>(), 8);
        assert_eq!(align_of::<PackedQuad>(), 4);
        let quad =
            PackedQuad::new([1, 2, 3], 4, 5, Face::NegY, MaterialId(0x1234), 2, true).unwrap();
        assert_eq!(quad.w0, 1 | 2 << 5 | 3 << 10 | 3 << 15 | 4 << 20 | 3 << 25);
        // Golden words of the v2 layout (E0 had the border-wall bit 28 set here instead of the
        // authored bit 18).
        assert_eq!(quad.w0, 0x0641_8C41);
        assert_eq!(quad.w1, 0x0006_1234);
        let with_payload = quad.with_payload(0xABC).unwrap();
        assert_eq!(with_payload.w0, quad.w0);
        assert_eq!(with_payload.w1, 0x55E6_1234);
        // Little-endian byte image as the GPU sees it.
        let expected_bytes: [u8; 8] = [0x41, 0x8C, 0x41, 0x06, 0x34, 0x12, 0x06, 0x00];
        if cfg!(target_endian = "little") {
            assert_eq!(bytemuck::bytes_of(&quad), &expected_bytes);
        }
        assert_eq!(bytemuck::cast::<_, [u32; 2]>(quad), [quad.w0, quad.w1]);
        assert_eq!(PackedQuad::zeroed().validate(), Ok(()));
    }

    #[test]
    fn round_trip_over_the_full_geometry_range() {
        // Every valid (cell, w, h, face): 53.5 million quads. The remaining fields cycle
        // through their full ranges so that every value of every field is combined with
        // many geometries.
        let mut count = 0_u64;
        let mut counter = 0_u32;
        let mut materials_seen = vec![false; 1 << 16];
        let mut payloads_seen = vec![false; usize::from(MAX_PAYLOAD) + 1];
        let mut attributes_seen = [false; 8];
        for face in Face::ALL {
            let (u_axis, v_axis, n_axis) = (face.u_axis(), face.v_axis(), face.axis());
            for cn in 0..32_u8 {
                for cu in 0..32_u8 {
                    for cv in 0..32_u8 {
                        let mut cell = [0_u8; 3];
                        cell[n_axis] = cn;
                        cell[u_axis] = cu;
                        cell[v_axis] = cv;
                        for w in 1..=32 - cu {
                            for h in 1..=32 - cv {
                                counter =
                                    counter.wrapping_mul(0x9E37_79B1).wrapping_add(0x7F4A_7C15);
                                let material = MaterialId((counter >> 16) as u16);
                                let uv_rot = ((counter >> 8) & 3) as u8;
                                let authored = counter & 0x80 != 0;
                                let payload = ((counter >> 4) & 0xFFF) as u16;
                                materials_seen[material.index()] = true;
                                payloads_seen[usize::from(payload)] = true;
                                attributes_seen[usize::from(uv_rot) * 2 + usize::from(authored)] =
                                    true;
                                let quad =
                                    PackedQuad::new(cell, w, h, face, material, uv_rot, authored)
                                        .unwrap()
                                        .with_payload(payload)
                                        .unwrap();
                                let expected = reference_words(
                                    cell[0].into(),
                                    cell[1].into(),
                                    cell[2].into(),
                                    w.into(),
                                    h.into(),
                                    face.index() as u32,
                                    material.0.into(),
                                    uv_rot.into(),
                                    authored.into(),
                                    payload.into(),
                                );
                                assert_eq!((quad.w0, quad.w1), expected);
                                let decoded = (
                                    quad.cell(),
                                    quad.width(),
                                    quad.height(),
                                    quad.face(),
                                    quad.material(),
                                    quad.uv_rot(),
                                    quad.is_authored(),
                                    quad.payload(),
                                );
                                assert_eq!(
                                    decoded,
                                    (cell, w, h, face, material, uv_rot, authored, payload)
                                );
                                count += 1;
                            }
                        }
                    }
                }
            }
        }
        // 6 faces * 32 planes * (sum over c of (32 - c))^2 rectangles.
        assert_eq!(count, 6 * 32 * 528 * 528);
        assert!(
            materials_seen.iter().all(|&seen| seen),
            "not every material value was used"
        );
        assert!(
            payloads_seen.iter().all(|&seen| seen),
            "not every payload value was used"
        );
        assert_eq!(attributes_seen, [true; 8]);
    }

    #[test]
    fn round_trip_over_the_full_attribute_range() {
        // Every material with every rotation, flag and face, on two extreme geometries; every
        // payload on one.
        for (cell, w, h) in [
            ([0_u8, 0, 0], 1_u8, 1_u8),
            ([31, 31, 31], 1, 1),
            ([0, 0, 0], 32, 32),
        ] {
            for material in 0..=u16::MAX {
                for uv_rot in 0..4_u8 {
                    for authored in [false, true] {
                        for face in Face::ALL {
                            let mut origin = cell;
                            if w == 32 {
                                origin[face.axis()] = 17;
                            }
                            let material = MaterialId(material);
                            let quad =
                                PackedQuad::new(origin, w, h, face, material, uv_rot, authored)
                                    .unwrap();
                            assert_eq!(quad.validate(), Ok(()));
                            assert_eq!(quad.cell(), origin);
                            assert_eq!(
                                (quad.x(), quad.y(), quad.z()),
                                (origin[0], origin[1], origin[2])
                            );
                            assert_eq!((quad.width(), quad.height()), (w, h));
                            assert_eq!(quad.face(), face);
                            assert_eq!(quad.material(), material);
                            assert_eq!(quad.uv_rot(), uv_rot);
                            assert_eq!(quad.is_authored(), authored);
                            assert_eq!(quad.payload(), 0);
                            assert_eq!(quad.w0 & W0_RESERVED, 0);
                            assert_eq!(quad.w1 & W1_RESERVED, 0);
                        }
                    }
                }
            }
        }
        let base =
            PackedQuad::new([5, 6, 7], 3, 2, Face::NegX, MaterialId(0xFFFF), 3, true).unwrap();
        for payload in 0..=MAX_PAYLOAD {
            let quad = base.with_payload(payload).unwrap();
            assert_eq!(quad.validate(), Ok(()));
            assert_eq!(quad.payload(), payload);
            assert_eq!(quad.w0, base.w0);
            assert_eq!(
                (quad.material(), quad.uv_rot(), quad.is_authored()),
                (base.material(), base.uv_rot(), base.is_authored())
            );
            // Replacing the payload again leaves no trace of the previous one.
            assert_eq!(quad.with_payload(0), Ok(base));
            assert_eq!(quad.w1 & W1_RESERVED, 0);
        }
    }

    #[test]
    fn fields_do_not_overlap() {
        // Changing one field changes only that field's bits.
        let new = |cell, w, h, face, material, uv_rot, authored| {
            PackedQuad::new(cell, w, h, face, MaterialId(material), uv_rot, authored).unwrap()
        };
        let base = new([0, 0, 0], 1, 1, Face::PosX, 0, 0, false);
        assert_eq!((base.w0, base.w1), (0, 0));
        let variants = [
            (new([31, 0, 0], 1, 1, Face::PosX, 0, 0, false), 0x1F, 0),
            (new([0, 31, 0], 1, 1, Face::PosX, 0, 0, false), 0x1F << 5, 0),
            (
                new([0, 0, 31], 1, 1, Face::PosX, 0, 0, false),
                0x1F << 10,
                0,
            ),
            (
                new([0, 0, 0], 32, 1, Face::PosX, 0, 0, false),
                0x1F << 15,
                0,
            ),
            (
                new([0, 0, 0], 1, 32, Face::PosX, 0, 0, false),
                0x1F << 20,
                0,
            ),
            (new([0, 0, 0], 1, 1, Face::NegZ, 0, 0, false), 5 << 25, 0),
            (
                new([0, 0, 0], 1, 1, Face::PosX, u16::MAX, 0, false),
                0,
                0xFFFF,
            ),
            (new([0, 0, 0], 1, 1, Face::PosX, 0, 3, false), 0, 3 << 16),
            (new([0, 0, 0], 1, 1, Face::PosX, 0, 0, true), 0, 1 << 18),
            (base.with_payload(MAX_PAYLOAD).unwrap(), 0, 0xFFF << 19),
        ];
        let (mut seen0, mut seen1) = (0_u32, 0_u32);
        for (quad, w0, w1) in variants {
            assert_eq!((quad.w0, quad.w1), (w0, w1));
            assert_eq!((seen0 & w0, seen1 & w1), (0, 0));
            seen0 |= w0;
            seen1 |= w1;
        }
        // Together the fields use exactly the non-reserved bits (face needs 0b111 for the mask).
        assert_eq!(seen0 | 0b111 << 25, !W0_RESERVED);
        assert_eq!(seen1, !W1_RESERVED);
        assert_eq!(W0_RESERVED, 0xF000_0000);
        assert_eq!(W1_RESERVED, 0x8000_0000);
    }

    #[test]
    fn out_of_range_input_is_rejected() {
        let material = MaterialId(1);
        let new =
            |cell, w, h, face, uv_rot| PackedQuad::new(cell, w, h, face, material, uv_rot, false);
        for bad in [32_u8, 33, 64, 255] {
            for axis in 0..3 {
                let mut cell = [0_u8; 3];
                cell[axis] = bad;
                assert_eq!(
                    new(cell, 1, 1, Face::PosX, 0),
                    Err(QuadError::CellOutOfRange { cell })
                );
            }
            let size_error = |w, h| Err(QuadError::SizeOutOfRange { w, h });
            if bad > 32 {
                assert_eq!(new([0; 3], bad, 1, Face::PosX, 0), size_error(bad, 1));
                assert_eq!(new([0; 3], 1, bad, Face::PosX, 0), size_error(1, bad));
            }
        }
        assert_eq!(
            new([0; 3], 0, 1, Face::PosY, 0),
            Err(QuadError::SizeOutOfRange { w: 0, h: 1 })
        );
        assert_eq!(
            new([0; 3], 1, 0, Face::PosY, 0),
            Err(QuadError::SizeOutOfRange { w: 1, h: 0 })
        );
        for uv_rot in 4..=u8::MAX {
            assert_eq!(
                new([0; 3], 1, 1, Face::PosZ, uv_rot),
                Err(QuadError::UvRotOutOfRange { uv_rot })
            );
        }
        let quad = new([0; 3], 1, 1, Face::PosZ, 0).unwrap();
        for payload in [MAX_PAYLOAD + 1, 0x1000, 0x8000, u16::MAX] {
            assert_eq!(
                quad.with_payload(payload),
                Err(QuadError::PayloadOutOfRange { payload })
            );
        }
        // Rectangles that start inside but leave the chunk, on the axes of each face.
        for face in Face::ALL {
            let mut cell = [5_u8; 3];
            cell[face.u_axis()] = 20;
            let error = Err(QuadError::ExtendsBeyondChunk {
                cell,
                w: 13,
                h: 1,
                face,
            });
            assert_eq!(new(cell, 13, 1, face, 0), error);
            assert!(new(cell, 12, 1, face, 0).is_ok());
            let mut cell = [5_u8; 3];
            cell[face.v_axis()] = 31;
            let error = Err(QuadError::ExtendsBeyondChunk {
                cell,
                w: 1,
                h: 2,
                face,
            });
            assert_eq!(new(cell, 1, 2, face, 0), error);
            assert!(new(cell, 1, 1, face, 0).is_ok());
            // The normal axis does not limit the size.
            let mut cell = [0_u8; 3];
            cell[face.axis()] = 31;
            assert!(new(cell, 32, 32, face, 0).is_ok());
        }
        // Errors have readable messages.
        let message = new([40, 0, 0], 1, 1, Face::PosX, 0)
            .unwrap_err()
            .to_string();
        assert!(message.contains("[40, 0, 0]"), "{message}");
    }

    #[test]
    fn untrusted_words_are_validated() {
        let good = PackedQuad::new([3, 4, 5], 6, 7, Face::PosZ, MaterialId(9), 1, true)
            .unwrap()
            .with_payload(0x5A5)
            .unwrap();
        assert_eq!(good.validate(), Ok(()));
        for bits in [6_u32, 7] {
            let bad = PackedQuad {
                w0: (good.w0 & !(7 << 25)) | bits << 25,
                w1: good.w1,
            };
            assert_eq!(
                bad.validate(),
                Err(QuadError::InvalidFace { bits: bits as u8 })
            );
        }
        for bit in 28..32 {
            let bad = PackedQuad {
                w0: good.w0 | 1 << bit,
                w1: good.w1,
            };
            assert_eq!(
                bad.validate(),
                Err(QuadError::ReservedBits {
                    w0: bad.w0,
                    w1: bad.w1
                })
            );
        }
        let bad = PackedQuad {
            w0: good.w0,
            w1: good.w1 | 1 << 31,
        };
        assert_eq!(
            bad.validate(),
            Err(QuadError::ReservedBits {
                w0: bad.w0,
                w1: bad.w1
            })
        );
        // Every other bit of w1 is a field.
        for bit in 0..31 {
            let flipped = PackedQuad {
                w0: good.w0,
                w1: good.w1 ^ 1 << bit,
            };
            assert_eq!(flipped.validate(), Ok(()), "bit {bit}");
        }
        // x = 30 with width 6 on +Z (u axis = X) leaves the chunk.
        let bad = PackedQuad {
            w0: (good.w0 & !0x1F) | 30,
            w1: good.w1,
        };
        assert!(matches!(
            bad.validate(),
            Err(QuadError::ExtendsBeyondChunk { .. })
        ));
        // Every word pair that validates decodes and re-encodes to itself.
        let mut state = 0x1234_5678_u32;
        let mut valid = 0;
        for _ in 0..2_000_000 {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let w0 = state;
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let quad = PackedQuad {
                w0: w0 & !W0_RESERVED,
                w1: state & !W1_RESERVED,
            };
            if quad.validate().is_ok() {
                valid += 1;
                let again = PackedQuad::new(
                    quad.cell(),
                    quad.width(),
                    quad.height(),
                    quad.face(),
                    quad.material(),
                    quad.uv_rot(),
                    quad.is_authored(),
                )
                .and_then(|again| again.with_payload(quad.payload()));
                assert_eq!(again, Ok(quad));
            }
        }
        assert!(valid > 100_000, "{valid}");
    }

    #[test]
    #[should_panic(expected = "invalid face bits")]
    fn face_accessor_panics_on_corrupt_bits() {
        let _ = PackedQuad { w0: 7 << 25, w1: 0 }.face();
    }

    #[test]
    fn corners_follow_the_contract() {
        // Positive faces sit at cell_n + 1, negative faces at cell_n.
        let quad = |face| PackedQuad::new([4, 5, 6], 2, 3, face, MaterialId(0), 0, false).unwrap();
        let v = IVec3::new;
        assert_eq!(
            quad(Face::PosX).corners(),
            [v(5, 5, 6), v(5, 7, 6), v(5, 7, 9), v(5, 5, 9)]
        );
        assert_eq!(
            quad(Face::NegX).corners(),
            [v(4, 5, 6), v(4, 7, 6), v(4, 7, 9), v(4, 5, 9)]
        );
        assert_eq!(
            quad(Face::PosY).corners(),
            [v(4, 6, 6), v(4, 6, 8), v(7, 6, 8), v(7, 6, 6)]
        );
        assert_eq!(
            quad(Face::NegY).corners(),
            [v(4, 5, 6), v(4, 5, 8), v(7, 5, 8), v(7, 5, 6)]
        );
        assert_eq!(
            quad(Face::PosZ).corners(),
            [v(4, 5, 7), v(6, 5, 7), v(6, 8, 7), v(4, 8, 7)]
        );
        assert_eq!(
            quad(Face::NegZ).corners(),
            [v(4, 5, 6), v(6, 5, 6), v(6, 8, 6), v(4, 8, 6)]
        );
    }

    /// The triangles of a quad as drawn: the index pattern applied to the corner map.
    fn drawn_triangles(map: [usize; 4]) -> [[usize; 3]; 2] {
        let pattern = QUAD_INDEX_PATTERN.map(|index| map[index as usize]);
        [
            [pattern[0], pattern[1], pattern[2]],
            [pattern[3], pattern[4], pattern[5]],
        ]
    }

    /// A triangle with its corners rotated so the smallest comes first; two triangles are the
    /// same triangle with the same winding exactly when their normal forms are equal.
    fn normal_form(triangle: [usize; 3]) -> [usize; 3] {
        let first = (0..3).min_by_key(|&i| triangle[i]).unwrap();
        [0, 1, 2].map(|k| triangle[(first + k) % 3])
    }

    #[test]
    fn corner_maps_draw_the_triangles_of_the_old_six_vertex_tables() {
        assert_eq!(QUAD_INDEX_PATTERN, [0, 1, 2, 0, 2, 3]);
        assert_eq!((QUAD_VERTICES, QUAD_INDICES), (4, 6));
        assert_eq!(CORNER_MAP_POSITIVE, [0, 1, 2, 3]);
        assert_eq!(CORNER_MAP_NEGATIVE, [0, 3, 2, 1]);
        assert_eq!(CORNER_MAP, [CORNER_MAP_POSITIVE, CORNER_MAP_NEGATIVE]);
        // The E0 tables: six vertices per quad, given as corners.
        const OLD_POSITIVE: [usize; 6] = [0, 1, 2, 0, 2, 3];
        const OLD_NEGATIVE: [usize; 6] = [0, 2, 1, 0, 3, 2];
        for (map, old) in [
            (CORNER_MAP_POSITIVE, OLD_POSITIVE),
            (CORNER_MAP_NEGATIVE, OLD_NEGATIVE),
        ] {
            let old_triangles = [[old[0], old[1], old[2]], [old[3], old[4], old[5]]];
            let mut expected: Vec<[usize; 3]> = old_triangles.map(normal_form).to_vec();
            let mut actual: Vec<[usize; 3]> = drawn_triangles(map).map(normal_form).to_vec();
            expected.sort_unstable();
            actual.sort_unstable();
            // The same two triangles with the same winding (their order may differ).
            assert_eq!(actual, expected, "{map:?}");
        }
        for face in Face::ALL {
            assert_eq!(corner_map(face), CORNER_MAP[face.index() & 1]);
        }
    }

    #[test]
    fn vertices_and_winding_for_every_face_and_size() {
        for face in Face::ALL {
            let (u_axis, v_axis, n_axis) = (face.u_axis(), face.v_axis(), face.axis());
            for (cell, w, h) in [
                ([0_u8, 0, 0], 1_u8, 1_u8),
                ([3, 7, 11], 5, 2),
                ([0, 0, 0], 32, 32),
                ([31, 31, 31], 1, 1),
            ] {
                let quad = PackedQuad::new(cell, w, h, face, MaterialId(7), 0, false).unwrap();
                let corners = quad.corners();
                let plane = i32::from(cell[n_axis]) + i32::from(!face.is_negative());
                let (u0, v0) = (i32::from(cell[u_axis]), i32::from(cell[v_axis]));
                let expected_uv = [
                    (0, 0),
                    (i32::from(w), 0),
                    (i32::from(w), i32::from(h)),
                    (0, i32::from(h)),
                ];
                for (corner, (du, dv)) in corners.iter().zip(expected_uv) {
                    assert_eq!(corner[n_axis], plane);
                    assert_eq!((corner[u_axis], corner[v_axis]), (u0 + du, v0 + dv));
                    assert!(
                        corner.cmpge(IVec3::ZERO).all() && corner.cmple(IVec3::splat(32)).all()
                    );
                }
                // Vertex k is corner CORNER_MAP[k].
                let vertices = quad.vertices();
                assert_eq!(vertices, corner_map(face).map(|corner| corners[corner]));
                let distinct: HashSet<IVec3> = vertices.into_iter().collect();
                assert_eq!(distinct.len(), 4);
                // Both triangles are counter-clockwise seen from outside: their geometric
                // normal points along the outward face normal, with twice the triangle area.
                let mut doubled_area = 0;
                for triangle in QUAD_INDEX_PATTERN.chunks(3) {
                    let [a, b, c] = [0, 1, 2].map(|k| vertices[triangle[k] as usize]);
                    let normal = (b - a).cross(c - a);
                    let along = normal.dot(face.normal());
                    assert!(along > 0, "{face:?}: triangle winds clockwise");
                    assert_eq!(normal, face.normal() * along);
                    doubled_area += along;
                }
                assert_eq!(doubled_area, 2 * i32::from(w) * i32::from(h));
            }
        }
    }

    #[test]
    fn covered_cells_are_the_rectangle_behind_the_quad() {
        for face in Face::ALL {
            let quad = PackedQuad::new([2, 3, 4], 5, 6, face, MaterialId(0), 0, false).unwrap();
            let cells: Vec<[u8; 3]> = quad.covered_cells().collect();
            assert_eq!(cells.len(), 30);
            let unique: HashSet<[u8; 3]> = cells.iter().copied().collect();
            assert_eq!(unique.len(), 30);
            assert_eq!(cells[0], [2, 3, 4]);
            // u runs fastest.
            let mut second = [2_u8, 3, 4];
            second[face.u_axis()] += 1;
            assert_eq!(cells[1], second);
            let corners = quad.corners();
            for cell in &cells {
                assert_eq!(cell[face.axis()], [2, 3, 4][face.axis()]);
                assert!((0..5).contains(&(cell[face.u_axis()] - [2, 3, 4][face.u_axis()])));
                assert!((0..6).contains(&(cell[face.v_axis()] - [2, 3, 4][face.v_axis()])));
                // The quad's plane is the cell's face: the cell centre projects into the quad and
                // lies half a cell behind the plane.
                let twice_centre =
                    IVec3::new(cell[0].into(), cell[1].into(), cell[2].into()) * 2 + 1;
                let offset = twice_centre - corners[0] * 2;
                assert_eq!(offset.dot(face.normal()), -1);
                assert!((0..10).contains(&offset.dot(face.u_dir())));
                assert!((0..12).contains(&offset.dot(face.v_dir())));
            }
        }
        let single =
            PackedQuad::new([31, 0, 31], 1, 1, Face::NegY, MaterialId(0), 0, false).unwrap();
        assert_eq!(single.covered_cells().collect::<Vec<_>>(), [[31, 0, 31]]);
        let full = PackedQuad::new([0, 9, 0], 32, 32, Face::PosY, MaterialId(0), 0, false).unwrap();
        assert_eq!(full.covered_cells().count(), 1024);
        assert!(full.covered_cells().all(|cell| cell[1] == 9));
    }
}
