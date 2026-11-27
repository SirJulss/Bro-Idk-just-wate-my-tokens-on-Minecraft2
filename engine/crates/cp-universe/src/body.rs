//! Body definitions: a star, planet or moon as data (contract §2.2, §6).
//!
//! A [`BodyDef`] holds what a content author writes; derived quantities (gravitational
//! parameter, octree root level, headroom of the voxel lattice) are methods. Bodies live in a
//! [`crate::SystemDef`], which checks every rule and assigns the [`BodyId`]s.
//!
//! # The voxel lattice of a cube body
//! The body frame has its origin at the cube centre; the reference surface of each face lies at
//! the half edge `R` from it. The voxel lattice is shifted against the centre by
//! [`CubeDef::grid_offset`] `o` on every axis (contract §2.2): *lattice = body + o*. The body
//! point `p` (metres) lies in voxel `floor(p + o)`, voxel `v` spans `[v - o, v - o + 1)` m of the
//! body frame, and the cube centre is the lattice point `(o, o, o)`. Gravity, frames and saves
//! keep using the cube centre. The octree's root set is the eight chunks with `pos ∈ {-1, 0}³`
//! at [`CubeDef::root_level`], i.e. the lattice box `[-E, E)³` with `E` =
//! [`CubeDef::root_extent`].

use core::f64::consts::PI;

use cp_core::coords::{CHUNK_SIZE, MAX_LOD_LEVEL};
use cp_core::det;
use cp_core::ids::BodyId;
use cp_core::math::{DVec3, IVec3};

use crate::orbit::OrbitElements;

/// What a body is. Stars are spheres; planets and moons are cubes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum BodyKind {
    /// The root of a system: a sphere that orbits nothing.
    Star,
    /// A cube body that orbits a star.
    Planet,
    /// A cube body that orbits a planet.
    Moon,
}

impl BodyKind {
    /// Every kind, in declaration order.
    pub const ALL: [Self; 3] = [Self::Star, Self::Planet, Self::Moon];

    /// The name used in system files: `star`, `planet`, `moon`.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Star => "star",
            Self::Planet => "planet",
            Self::Moon => "moon",
        }
    }

    /// The kind with this [`BodyKind::name`].
    #[must_use]
    pub fn from_name(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|kind| kind.name() == name)
    }
}

/// How a body turns about its spin axis (body `+Y`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Rotation {
    /// Uniform rotation.
    Uniform {
        /// Sidereal rotation period in seconds (one turn against the stars), `> 0`.
        period: f64,
        /// Rotation angle about the spin axis at the epoch, radians.
        phase: f64,
    },
    /// Tidally locked to the parent: the body keeps its `-X` face towards its parent on an orbit
    /// of any inclination, prograde or retrograde.
    ///
    /// The rotation is built in the orbit's own frame: the spin axis is the orbit normal,
    /// tilted by [`BodyDef::axial_tilt`] about the line of nodes towards the orbit's
    /// northernmost point (which is towards the system's `+Y`), the period is the orbital
    /// period, and the rotation angle is the argument of latitude with the mean anomaly,
    /// `ω + M(t)`, counted from the ascending node. On a circular orbit with no tilt `-X` points
    /// exactly at the parent. Otherwise the direction to the parent librates about the `-X`
    /// face normal: by up to the axial tilt in latitude and by up to the largest equation of
    /// the centre `ν - M` in longitude (about `2e` radians; 2.3° at `e = 0.02`).
    /// [`crate::SystemDef::new`] requires the two together to stay below 90°, so the parent
    /// always stands in front of the `-X` face, seen from the body's centre.
    Synchronous,
}

/// The cube-specific part of a planet or moon.
#[derive(Clone, Debug, PartialEq)]
pub struct CubeDef {
    /// Edge length of the cube in metres; even and at least 2 (contract §2.2).
    pub edge: u32,
    /// Height of the ocean surface above the reference surface in metres, negative below it.
    /// On bodies without liquid it is the zero-elevation datum the terrain is built around.
    pub sea_level: f64,
    /// Shift of the voxel lattice against the cube centre, voxels, the same on all axes (see
    /// the module documentation).
    pub grid_offset: i32,
    /// Height of the top of the atmosphere above the reference surface in metres; `0` for an
    /// airless body.
    pub atmosphere_height: f64,
    /// Key of the terrain generator (a `cp_voxel::BaseSource` implementation),
    /// `namespace:name`.
    pub generator: String,
    /// Seed of the terrain generator.
    pub generator_seed: u64,
}

/// The levels on which the mean surface must stay clear of chunk boundaries (contract §2.2),
/// as far as they exist below a body's root level.
pub const CLEARANCE_LEVELS: core::ops::RangeInclusive<u8> = 4..=8;

/// Smallest [`surface_clearance`] a body may have: the mean surface stays at least an eighth of
/// a chunk away from every chunk boundary on the [`CLEARANCE_LEVELS`].
pub const MIN_SURFACE_CLEARANCE: f64 = 0.125;

/// Octree root level of a cube with this edge (metres): the smallest `L` with
/// `32 * 2^L >= edge / 2` (contract §2.2). Home, 40 960 m: 10.
#[must_use]
pub fn root_level(edge: u32) -> u8 {
    let half = u64::from(edge / 2);
    let mut level = 0_u8;
    while ((CHUNK_SIZE as u64) << level) < half {
        level += 1;
    }
    level
}

/// Half the extent of the root set in metres, `32 * 2^level`: the root chunks cover the lattice
/// box `[-E, E)³`.
#[must_use]
pub fn root_extent(level: u8) -> f64 {
    f64::from(CHUNK_SIZE) * det::powi(2.0, i32::from(level))
}

/// How far the mean surface (reference surface plus `sea_level`) of a cube with this edge and
/// grid offset stays from chunk boundaries.
///
/// The result is the smallest distance to a chunk boundary, as a fraction of the chunk extent
/// of that level (`0` = on a boundary, `0.5` = in the middle), over the faces of both signs and
/// the [`CLEARANCE_LEVELS`] up to the body's root level. `0.5` when no such level exists.
#[must_use]
pub fn surface_clearance(edge: u32, sea_level: f64, grid_offset: i32) -> f64 {
    let top = root_level(edge);
    let surface = f64::from(edge / 2) + sea_level;
    let offset = f64::from(grid_offset);
    let mut clearance = 0.5_f64;
    for level in CLEARANCE_LEVELS.filter(|&level| level <= top) {
        let extent = root_extent(level);
        for lattice in [surface + offset, offset - surface] {
            let within = lattice.rem_euclid(extent);
            let distance = det::min(within, extent - within) / extent;
            clearance = det::min(clearance, distance);
        }
    }
    clearance
}

/// The grid offset in `[-max_abs_offset, max_abs_offset]` with the largest
/// [`surface_clearance`]; ties go to the smaller magnitude, then to the positive offset.
#[must_use]
pub fn best_grid_offset(edge: u32, sea_level: f64, max_abs_offset: u32) -> i32 {
    let limit = i32::try_from(max_abs_offset).unwrap_or(i32::MAX);
    let mut best = (0_i32, surface_clearance(edge, sea_level, 0));
    for magnitude in 1..=limit {
        for offset in [magnitude, -magnitude] {
            let clearance = surface_clearance(edge, sea_level, offset);
            if clearance > best.1 {
                best = (offset, clearance);
            }
        }
    }
    best.0
}

impl CubeDef {
    /// Half the edge, `R`, metres: the distance of each face's reference surface from the
    /// centre.
    #[inline]
    #[must_use]
    pub fn half_edge(&self) -> f64 {
        f64::from(self.edge) * 0.5
    }

    /// The octree root level (contract §2.2); see [`root_level`].
    #[inline]
    #[must_use]
    pub fn root_level(&self) -> u8 {
        root_level(self.edge)
    }

    /// Half the extent of the root set in metres; see [`root_extent`].
    #[inline]
    #[must_use]
    pub fn root_extent(&self) -> f64 {
        root_extent(self.root_level())
    }

    /// Room of the voxel lattice above the reference surface on the most constrained face, in
    /// metres: root extent minus half edge minus the magnitude of the grid offset. Terrain and
    /// buildings above it are outside the body's octree.
    #[must_use]
    pub fn headroom(&self) -> f64 {
        self.root_extent() - self.half_edge() - f64::from(self.grid_offset.unsigned_abs())
    }

    /// [`surface_clearance`] of this cube.
    #[must_use]
    pub fn surface_clearance(&self) -> f64 {
        surface_clearance(self.edge, self.sea_level, self.grid_offset)
    }

    /// Body-frame point (metres from the cube centre) → lattice coordinates.
    #[inline]
    #[must_use]
    pub fn body_to_lattice(&self, point: DVec3) -> DVec3 {
        point + DVec3::splat(f64::from(self.grid_offset))
    }

    /// Lattice coordinates → body-frame point.
    #[inline]
    #[must_use]
    pub fn lattice_to_body(&self, lattice: DVec3) -> DVec3 {
        lattice - DVec3::splat(f64::from(self.grid_offset))
    }

    /// The voxel that contains the body-frame point `point`: `floor(point + grid_offset)`.
    /// Coordinates saturate at the `i32` limits.
    #[inline]
    #[must_use]
    pub fn voxel_at(&self, point: DVec3) -> IVec3 {
        self.body_to_lattice(point).floor().as_ivec3()
    }

    /// Body-frame position of the centre of voxel `voxel`.
    #[inline]
    #[must_use]
    pub fn voxel_center(&self, voxel: IVec3) -> DVec3 {
        self.lattice_to_body(voxel.as_dvec3() + DVec3::splat(0.5))
    }
}

/// The geometry of a body.
#[derive(Clone, Debug, PartialEq)]
pub enum BodyShape {
    /// A star: a sphere.
    Sphere {
        /// Radius in metres, `> 0`.
        radius: f64,
    },
    /// A planet or moon: a cube with six walkable faces.
    Cube(CubeDef),
}

/// One body of a star system, as written in its system file.
///
/// Persistent data names bodies by [`BodyDef::key`]; [`BodyId`]s are indices into the system's
/// body list and valid for one session.
#[derive(Clone, Debug, PartialEq)]
pub struct BodyDef {
    /// Stable key `namespace:name` (the content key grammar, contract §4.4), e.g.
    /// `cubeplanet:heimat`.
    pub key: String,
    /// Name shown to players.
    pub display_name: String,
    /// Star, planet or moon.
    pub kind: BodyKind,
    /// The body this one orbits; `None` only for the star at the root. Parents come before
    /// their children in the system's body list.
    pub parent: Option<BodyId>,
    /// Sphere (stars) or cube (planets, moons).
    pub shape: BodyShape,
    /// Gravity at the reference surface in m/s², `> 0`: at the face centres of a cube, on the
    /// surface of a sphere. Defines the gravitational parameter, see [`BodyDef::mu`].
    pub surface_gravity: f64,
    /// Rotation about the spin axis.
    pub rotation: Rotation,
    /// Angle of the spin axis (body `+Y`, the north pole) against a reference axis, radians in
    /// `[0, π]`. For a uniform rotation the reference is the system's `+Y` axis and the north
    /// pole leans towards system `-X` (a rotation about `+Z`). For a tidally locked body
    /// ([`Rotation::Synchronous`]) it is the orbit normal, and the pole leans towards the
    /// orbit's northernmost point (a rotation about the line of nodes); the Moon's would be
    /// 6.7°, its inclination plus the 1.5° by which its pole passes the ecliptic's.
    pub axial_tilt: f64,
    /// Orbit around the parent at the epoch; `Some` exactly when `parent` is.
    pub orbit: Option<OrbitElements>,
}

impl BodyDef {
    /// The radius that defines the gravitational parameter: the half edge of a cube, the radius
    /// of a sphere. Metres.
    #[must_use]
    pub fn reference_radius(&self) -> f64 {
        match &self.shape {
            BodyShape::Sphere { radius } => *radius,
            BodyShape::Cube(cube) => cube.half_edge(),
        }
    }

    /// Gravitational parameter `mu = g R²` in m³/s² with `g` the surface gravity and `R` the
    /// [`BodyDef::reference_radius`]: far away a body pulls like a point of this `mu`.
    #[must_use]
    pub fn mu(&self) -> f64 {
        let radius = self.reference_radius();
        self.surface_gravity * radius * radius
    }

    /// Radius of the smallest sphere around the centre that contains the reference body: the
    /// half diagonal `√3 R` of a cube, the radius of a sphere. Metres.
    #[must_use]
    pub fn bounding_radius(&self) -> f64 {
        match &self.shape {
            BodyShape::Sphere { radius } => *radius,
            BodyShape::Cube(cube) => cube.half_edge() * 3.0_f64.sqrt(),
        }
    }

    /// Height of the atmosphere above the reference surface in metres (`0` for stars and
    /// airless bodies).
    #[must_use]
    pub fn atmosphere_height(&self) -> f64 {
        self.cube().map_or(0.0, |cube| cube.atmosphere_height)
    }

    /// The cube-specific part, for planets and moons.
    #[must_use]
    pub fn cube(&self) -> Option<&CubeDef> {
        match &self.shape {
            BodyShape::Cube(cube) => Some(cube),
            BodyShape::Sphere { .. } => None,
        }
    }

    /// `BodyDef.root_level` of contract §2.2 for cube bodies; `None` for stars, which have no
    /// voxel grid.
    #[must_use]
    pub fn root_level(&self) -> Option<u8> {
        self.cube().map(CubeDef::root_level)
    }
}

/// Degrees → radians, as written in system files.
pub(crate) fn radians(degrees: f64) -> f64 {
    degrees * (PI / 180.0)
}

/// Largest root level the octree supports (contract §2.2).
pub(crate) const MAX_ROOT_LEVEL: u8 = MAX_LOD_LEVEL;

#[cfg(test)]
mod tests {
    use super::*;

    fn cube(edge: u32, sea_level: f64, grid_offset: i32) -> CubeDef {
        CubeDef {
            edge,
            sea_level,
            grid_offset,
            atmosphere_height: 0.0,
            generator: "test:flat".to_owned(),
            generator_seed: 1,
        }
    }

    /// Reference: the contract's definition taken literally.
    fn root_level_reference(edge: u32) -> u8 {
        (0..=40_u8)
            .find(|&level| 32.0 * (1_u64 << level) as f64 >= f64::from(edge) / 2.0)
            .expect("small edges")
    }

    #[test]
    fn root_level_follows_the_contract() {
        assert_eq!(root_level(40_960), 10);
        assert_eq!(root_level(2), 0);
        assert_eq!(root_level(64), 0);
        assert_eq!(root_level(66), 1);
        assert_eq!(root_level(8_192), 7);
        assert_eq!(root_level(8_194), 8);
        for edge in (2..200_000).step_by(2).chain([u32::MAX - 1]) {
            assert_eq!(root_level(edge), root_level_reference(edge), "edge {edge}");
        }
        assert_eq!(root_extent(10), 32_768.0);
        assert_eq!(cube(40_960, 0.0, 0).root_extent(), 32_768.0);
    }

    /// Reference: distance to the nearest multiple of the chunk extent, by brute force.
    fn clearance_reference(edge: u32, sea_level: f64, offset: i32) -> f64 {
        let top = root_level(edge);
        let surface = f64::from(edge / 2) + sea_level;
        let mut worst = 0.5_f64;
        for level in 4..=8_u8 {
            if level > top {
                continue;
            }
            let extent = 32.0 * f64::from(1_u32 << level);
            for lattice in [surface + f64::from(offset), f64::from(offset) - surface] {
                let nearest = (lattice / extent).round() * extent;
                worst = det::min(worst, (lattice - nearest).abs() / extent);
            }
        }
        worst
    }

    #[test]
    fn clearance_matches_brute_force() {
        for edge in [5_500_u32, 11_000, 18_000, 40_960, 40_962, 100] {
            for sea_level in [0.0, 12.0, -250.5] {
                for offset in -3000..3000 {
                    let ours = surface_clearance(edge, sea_level, offset);
                    let reference = clearance_reference(edge, sea_level, offset);
                    assert!(
                        (ours - reference).abs() < 1e-15,
                        "{edge} {sea_level} {offset}"
                    );
                }
            }
        }
        // The sea level raises the mean surface of every face, so it moves the surfaces of
        // opposite faces apart, unlike the offset, which shifts both the same way.
        assert_eq!(
            surface_clearance(40_960, 1.0, 1365),
            clearance_reference(40_960, 1.0, 1365)
        );
        // Without a level in 4..=8 below the root there is nothing to clear.
        assert_eq!(surface_clearance(100, 0.0, 0), 0.5);
    }

    #[test]
    fn home_needs_a_grid_offset() {
        // Without an offset the faces of Home (half edge 20 480 m = 40 · 512 m) lie on chunk
        // boundaries at levels 4 to 7.
        assert_eq!(surface_clearance(40_960, 0.0, 0), 0.0);
        // 1365 = 0b101_0101_0101 keeps them a third of a chunk away on every level.
        let best = best_grid_offset(40_960, 0.0, 4096);
        assert_eq!(best, 1365);
        let clearance = surface_clearance(40_960, 0.0, best);
        assert!((clearance - 1.0 / 3.0).abs() < 1e-3, "{clearance}");
        // Ties go to the smaller magnitude: -1365 is as good but later.
        assert_eq!(
            surface_clearance(40_960, 0.0, -1365),
            surface_clearance(40_960, 0.0, 1365)
        );
        assert_eq!(best_grid_offset(40_960, 0.0, 0), 0);
    }

    #[test]
    fn lattice_offsets_shift_voxels_against_the_centre() {
        let home = cube(40_960, 0.0, 1365);
        assert_eq!(home.half_edge(), 20_480.0);
        assert_eq!(home.root_level(), 10);
        assert_eq!(home.headroom(), 32_768.0 - 20_480.0 - 1365.0);
        // The cube centre is the lattice point (o, o, o).
        assert_eq!(home.voxel_at(DVec3::ZERO), IVec3::splat(1365));
        assert_eq!(home.voxel_at(DVec3::splat(-0.25)), IVec3::splat(1364));
        assert_eq!(
            home.voxel_at(DVec3::new(-1365.0, -1366.0, 0.999)),
            IVec3::new(0, -1, 1365)
        );
        // The +Y reference surface is the bottom of voxel row y = R + o.
        assert_eq!(home.voxel_at(DVec3::new(0.0, 20_480.0, 0.0)).y, 21_845);
        assert_eq!(home.voxel_at(DVec3::new(0.0, 20_479.999, 0.0)).y, 21_844);
        let voxel = IVec3::new(-7, 21_844, 300);
        let center = home.voxel_center(voxel);
        assert_eq!(home.voxel_at(center), voxel);
        assert_eq!(home.body_to_lattice(home.lattice_to_body(center)), center);
        let negative = cube(40_960, 0.0, -1365);
        assert_eq!(negative.voxel_at(DVec3::ZERO), IVec3::splat(-1365));
        assert_eq!(negative.headroom(), home.headroom());
    }

    #[test]
    fn derived_quantities() {
        let home = BodyDef {
            key: "cubeplanet:heimat".to_owned(),
            display_name: "Heimat".to_owned(),
            kind: BodyKind::Planet,
            parent: Some(BodyId(0)),
            shape: BodyShape::Cube(cube(40_960, 0.0, 1365)),
            surface_gravity: 9.81,
            rotation: Rotation::Uniform {
                period: 6912.0,
                phase: 0.0,
            },
            axial_tilt: radians(23.5),
            orbit: None,
        };
        assert_eq!(home.reference_radius(), 20_480.0);
        assert_eq!(home.mu(), 9.81 * 20_480.0 * 20_480.0);
        assert!((home.bounding_radius() - 35_472.40).abs() < 0.01);
        assert_eq!(home.root_level(), Some(10));
        assert_eq!(home.atmosphere_height(), 0.0);
        let star = BodyDef {
            shape: BodyShape::Sphere { radius: 70_000.0 },
            kind: BodyKind::Star,
            ..home
        };
        assert_eq!(star.reference_radius(), 70_000.0);
        assert_eq!(star.bounding_radius(), 70_000.0);
        assert_eq!(star.root_level(), None);
        assert!(star.cube().is_none());
        for kind in BodyKind::ALL {
            assert_eq!(BodyKind::from_name(kind.name()), Some(kind));
        }
        assert_eq!(BodyKind::from_name("comet"), None);
        assert!((radians(180.0) - PI).abs() < 1e-15);
    }
}
