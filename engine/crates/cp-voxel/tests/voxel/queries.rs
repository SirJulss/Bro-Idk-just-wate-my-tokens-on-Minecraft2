//! Raycast and box overlap against brute-force references.

use std::sync::Arc;

use cp_core::block::BlockFlags;
use cp_core::coords::{CHUNK_VOLUME, LodKey, index_to_cell};
use cp_core::det;
use cp_core::face::Face;
use cp_core::hash::{Rng, hash_ivec3};
use cp_core::ids::BlockStateId;
use cp_core::math::{DAabb, DVec3, IVec3};
use cp_voxel::{CellEdit, VoxelSnapshot, VoxelTables, VoxelWorld, WorldConfig};

use crate::common::{AIR, FnBase, HillsBase, PLANK, STONE, WATER, range_i32, rng, tables};

/// A dense copy of the level-0 composite cells of a box, read chunk by chunk.
struct Dense {
    min: IVec3,
    size: IVec3,
    cells: Vec<BlockStateId>,
}

impl Dense {
    fn new(snapshot: &VoxelSnapshot, min: IVec3, max: IVec3) -> Self {
        let size = max - min;
        let mut cells = vec![AIR; (size.x * size.y * size.z) as usize];
        let (first, last) = (min >> 5_u32, (max - 1) >> 5_u32);
        for cy in first.y..=last.y {
            for cz in first.z..=last.z {
                for cx in first.x..=last.x {
                    let key = LodKey::lod0(IVec3::new(cx, cy, cz));
                    let chunk = snapshot.sample(key);
                    for index in 0..CHUNK_VOLUME {
                        let local = IVec3::from_array(index_to_cell(index).map(i32::from));
                        let cell = key.pos * 32 + local - min;
                        if cell.cmpge(IVec3::ZERO).all() && cell.cmplt(size).all() {
                            let at = cell.x + cell.z * size.x + cell.y * size.x * size.z;
                            cells[at as usize] = chunk.get(index);
                        }
                    }
                }
            }
        }
        Self { min, size, cells }
    }

    fn get(&self, cell: IVec3) -> BlockStateId {
        let local = cell - self.min;
        assert!(local.cmpge(IVec3::ZERO).all() && local.cmplt(self.size).all());
        self.cells[(local.x + local.z * self.size.x + local.y * self.size.x * self.size.z) as usize]
    }
}

/// The first non-air cell whose interior the segment passes through, by testing every cell
/// of the segment's bounding box with the slab method: `(cell, entry axis, distance)`.
fn brute_raycast(
    dense: &Dense,
    origin: DVec3,
    dir: DVec3,
    max_dist: f64,
) -> Option<(IVec3, Option<usize>, f64)> {
    let dir = dir / dir.length();
    let end = origin + dir * max_dist;
    let lo = origin.min(end).floor().as_ivec3() - 1;
    let hi = origin.max(end).floor().as_ivec3() + 1;
    let mut best: Option<(IVec3, Option<usize>, f64)> = None;
    for y in lo.y..=hi.y {
        for z in lo.z..=hi.z {
            for x in lo.x..=hi.x {
                let cell = IVec3::new(x, y, z);
                if dense.get(cell).is_air() {
                    continue;
                }
                let mut enter = f64::NEG_INFINITY;
                let mut exit = f64::INFINITY;
                let mut axis_in = None;
                let mut inside = true;
                for axis in 0..3 {
                    let (o, d, c) = (origin[axis], dir[axis], f64::from(cell[axis]));
                    if d == 0.0 {
                        inside &= c <= o && o < c + 1.0;
                        continue;
                    }
                    let (a, b) = ((c - o) / d, (c + 1.0 - o) / d);
                    let (near, far) = if d > 0.0 { (a, b) } else { (b, a) };
                    if near > enter {
                        enter = near;
                        axis_in = Some(axis);
                    }
                    exit = det::min(exit, far);
                }
                if !inside || enter >= exit || exit <= 0.0 || enter > max_dist {
                    continue;
                }
                let distance = det::max(enter, 0.0);
                if best.is_none_or(|(_, _, d)| distance < d) {
                    best = Some((cell, if enter > 0.0 { axis_in } else { None }, distance));
                }
            }
        }
    }
    best
}

fn random_direction(rng: &mut Rng) -> DVec3 {
    loop {
        let d = DVec3::new(
            rng.range_f64(-1.0, 1.0),
            rng.range_f64(-1.0, 1.0),
            rng.range_f64(-1.0, 1.0),
        );
        if d.length() > 0.1 {
            return d;
        }
    }
}

fn check_ray(
    snapshot: &VoxelSnapshot,
    dense: &Dense,
    origin: DVec3,
    dir: DVec3,
    max_dist: f64,
) -> bool {
    let hit = snapshot.raycast(origin, dir, max_dist);
    let brute = brute_raycast(dense, origin, dir, max_dist);
    match (hit, brute) {
        (None, None) => false,
        (Some(hit), Some((cell, axis, distance))) => {
            assert_eq!(hit.cell, cell, "origin {origin:?} dir {dir:?}");
            assert_eq!(hit.distance, distance, "origin {origin:?} dir {dir:?}");
            assert_eq!(hit.state, dense.get(cell));
            if let Some(axis) = axis {
                let d = dir[axis];
                assert_eq!(
                    hit.face,
                    Face::from_axis(axis, d > 0.0),
                    "origin {origin:?}"
                );
            }
            true
        }
        (hit, brute) => panic!("origin {origin:?} dir {dir:?}: dda {hit:?} brute {brute:?}"),
    }
}

/// Sparse random blocks plus edits, around the origin (negative coordinates included).
fn sparse_world() -> VoxelWorld {
    let base = Arc::new(FnBase::new("test:sparse", |_, cell: IVec3| {
        match hash_ivec3(3, cell) % 23 {
            0 => STONE,
            1 => WATER,
            _ => AIR,
        }
    }));
    let mut world = VoxelWorld::new(
        base,
        tables(),
        WorldConfig {
            top_level: 2,
            ..WorldConfig::default()
        },
    );
    let mut rng = rng(9);
    let mut edits = Vec::new();
    for _ in 0..30 {
        let min = IVec3::new(
            range_i32(&mut rng, -60, 50),
            range_i32(&mut rng, -60, 50),
            range_i32(&mut rng, -60, 50),
        );
        let state = if rng.range_u32(0, 2) == 0 { AIR } else { PLANK };
        for y in 0..4 {
            for z in 0..4 {
                for x in 0..4 {
                    edits.push(CellEdit {
                        cell: min + IVec3::new(x, y, z),
                        state,
                    });
                }
            }
        }
    }
    world.apply(&edits);
    world
}

#[test]
fn raycast_agrees_with_a_brute_force_walk() {
    let world = sparse_world();
    let snapshot = world.snapshot();
    let dense = Dense::new(&snapshot, IVec3::splat(-72), IVec3::splat(72));
    let mut rng = rng(10);
    let mut hits = 0;
    let mut rays = 0;
    for i in 0..1500 {
        let origin = match i % 5 {
            // Integer origins: corners of cells, many on chunk borders.
            0 => DVec3::new(
                f64::from(range_i32(&mut rng, -40, 40)),
                f64::from(range_i32(&mut rng, -40, 40)),
                f64::from(range_i32(&mut rng, -40, 40)),
            ),
            // On chunk boundary planes.
            1 => DVec3::new(
                f64::from(32 * range_i32(&mut rng, -1, 2)),
                rng.range_f64(-40.0, 40.0),
                rng.range_f64(-40.0, 40.0),
            ),
            _ => DVec3::new(
                rng.range_f64(-40.0, 40.0),
                rng.range_f64(-40.0, 40.0),
                rng.range_f64(-40.0, 40.0),
            ),
        };
        let max_dist = rng.range_f64(0.0, 28.0);
        let dir = random_direction(&mut rng);
        rays += 1;
        if check_ray(&snapshot, &dense, origin, dir, max_dist) {
            hits += 1;
        }
    }
    // Axis-aligned rays, also starting on cell and chunk boundaries.
    for i in 0..300 {
        let face = Face::ALL[i % 6];
        let origin = if i % 2 == 0 {
            DVec3::new(
                f64::from(range_i32(&mut rng, -40, 40)),
                f64::from(range_i32(&mut rng, -40, 40)),
                f64::from(range_i32(&mut rng, -40, 40)),
            )
        } else {
            DVec3::new(
                rng.range_f64(-40.0, 40.0),
                -32.0,
                rng.range_f64(-40.0, 40.0),
            )
        };
        rays += 1;
        if check_ray(
            &snapshot,
            &dense,
            origin,
            face.normal().as_dvec3() * 3.0,
            25.0,
        ) {
            hits += 1;
        }
    }
    assert!(hits > rays / 3 && hits < rays, "{hits} hits of {rays} rays");
}

#[test]
fn raycast_details() {
    let base = Arc::new(FnBase::new("test:floor", |_, cell: IVec3| {
        if cell.y < -1 { STONE } else { AIR }
    }));
    let world = VoxelWorld::new(base, tables(), WorldConfig::default());
    let snapshot = world.snapshot();
    // Straight down from above the floor, in a negative chunk.
    let hit = snapshot
        .raycast(
            DVec3::new(-33.5, 4.0, -0.25),
            DVec3::new(0.0, -2.0, 0.0),
            10.0,
        )
        .unwrap();
    assert_eq!(hit.cell, IVec3::new(-34, -2, -1));
    assert_eq!(hit.face, Face::PosY);
    assert_eq!(hit.distance, 5.0);
    assert_eq!(hit.state, STONE);
    // Exactly at the range limit still hits; just below it does not.
    assert!(
        snapshot
            .raycast(DVec3::new(0.5, 4.0, 0.5), DVec3::NEG_Y, 5.0)
            .is_some()
    );
    assert!(
        snapshot
            .raycast(DVec3::new(0.5, 4.0, 0.5), DVec3::NEG_Y, 4.999)
            .is_none()
    );
    // Starting inside a solid cell hits it at distance 0.
    let inside = snapshot
        .raycast(DVec3::new(0.5, -3.5, 0.5), DVec3::X, 3.0)
        .unwrap();
    assert_eq!((inside.cell, inside.distance), (IVec3::new(0, -4, 0), 0.0));
    assert_eq!(inside.face, Face::NegX);
    // An origin on the floor surface looking down starts in the stone below.
    let on = snapshot
        .raycast(DVec3::new(0.5, -1.0, 0.5), DVec3::NEG_Y, 3.0)
        .unwrap();
    assert_eq!((on.cell, on.distance), (IVec3::new(0, -2, 0), 0.0));
    // Looking up from the same point sees nothing.
    assert!(
        snapshot
            .raycast(DVec3::new(0.5, -1.0, 0.5), DVec3::Y, 30.0)
            .is_none()
    );
    // Degenerate input.
    assert!(snapshot.raycast(DVec3::ZERO, DVec3::ZERO, 10.0).is_none());
    assert!(
        snapshot
            .raycast(DVec3::ZERO, DVec3::NEG_Y, f64::NAN)
            .is_none()
    );
    assert!(
        snapshot
            .raycast(DVec3::splat(f64::NAN), DVec3::NEG_Y, 10.0)
            .is_none()
    );
    // A filter that skips stone finds nothing.
    assert!(
        snapshot
            .raycast_with(DVec3::new(0.5, 4.0, 0.5), DVec3::NEG_Y, 20.0, |s| s
                == WATER)
            .is_none()
    );
}

#[test]
fn overlaps_agrees_with_a_brute_force_scan() {
    let mut world = VoxelWorld::new(
        Arc::new(HillsBase::new()),
        tables(),
        WorldConfig {
            top_level: 2,
            ..WorldConfig::default()
        },
    );
    world.apply(
        &(0..200)
            .map(|i| CellEdit {
                cell: IVec3::new(i % 13 - 6, 30 + i / 13, -3),
                state: PLANK,
            })
            .collect::<Vec<_>>(),
    );
    let snapshot = world.snapshot();
    let dense = Dense::new(&snapshot, IVec3::new(-72, -96, -72), IVec3::new(72, 96, 72));
    let tables: VoxelTables = tables();
    let mut rng = rng(12);
    let flag_sets = [
        BlockFlags::SOLID,
        BlockFlags::OPAQUE,
        BlockFlags::FLUID,
        BlockFlags::SOLID | BlockFlags::OPAQUE,
        BlockFlags::CUTOUT,
        BlockFlags::STRUCTURAL,
        BlockFlags::empty(),
    ];
    let mut positives = 0;
    for i in 0..3000 {
        let integral = i % 3 == 0;
        let coordinate = |rng: &mut Rng, lo: f64, hi: f64| {
            let value = rng.range_f64(lo, hi);
            if integral { value.floor() } else { value }
        };
        let min = DVec3::new(
            coordinate(&mut rng, -60.0, 50.0),
            coordinate(&mut rng, -80.0, 70.0),
            coordinate(&mut rng, -60.0, 50.0),
        );
        let size = DVec3::new(
            coordinate(&mut rng, 0.0, 8.0),
            coordinate(&mut rng, 0.0, 8.0),
            coordinate(&mut rng, 0.0, 8.0),
        );
        let aabb = DAabb::new(min, min + size);
        let flags = flag_sets[i % flag_sets.len()];
        let lo = aabb.min.floor().as_ivec3();
        let hi = aabb.max.ceil().as_ivec3();
        let mut expected = false;
        for y in lo.y..hi.y {
            for z in lo.z..hi.z {
                for x in lo.x..hi.x {
                    expected |= tables
                        .flags_of(dense.get(IVec3::new(x, y, z)))
                        .contains(flags);
                }
            }
        }
        assert_eq!(
            snapshot.overlaps(aabb, flags),
            expected,
            "{aabb:?} {flags:?}"
        );
        positives += usize::from(expected);
    }
    assert!(positives > 300 && positives < 2700, "{positives} overlaps");
    // A box resting on the ground touches but does not overlap it.
    let ground = IVec3::new(0, 0, 0);
    let top = (-96..96)
        .rev()
        .find(|&y| {
            tables
                .flags_of(dense.get(ground.with_y(y)))
                .contains(BlockFlags::SOLID)
        })
        .expect("ground below the origin");
    let standing = DAabb::new(
        DVec3::new(0.2, f64::from(top + 1), 0.2),
        DVec3::new(0.8, f64::from(top + 3), 0.8),
    );
    assert!(!snapshot.overlaps(standing, BlockFlags::SOLID));
    let sunk = DAabb::new(standing.min - DVec3::Y * 0.01, standing.max);
    assert!(snapshot.overlaps(sunk, BlockFlags::SOLID));
    // Huge and degenerate boxes.
    assert!(snapshot.overlaps(
        DAabb::new(DVec3::splat(-1.0e12), DVec3::splat(1.0e12)),
        BlockFlags::OPAQUE
    ));
    assert!(!snapshot.overlaps(DAabb::EMPTY, BlockFlags::empty()));
    assert!(!snapshot.overlaps(
        DAabb::new(DVec3::splat(f64::NAN), DVec3::ONE),
        BlockFlags::empty()
    ));
    // Sky chunks are uniform air (the base fills them on its all-air fast path).
    assert!(!snapshot.overlaps(
        DAabb::new(
            DVec3::new(-500.0, 200.0, -500.0),
            DVec3::new(500.0, 900.0, 500.0)
        ),
        BlockFlags::SOLID
    ));
    // Point reads agree with the dense copy.
    for _ in 0..2000 {
        let cell = IVec3::new(
            range_i32(&mut rng, -72, 72),
            range_i32(&mut rng, -96, 96),
            range_i32(&mut rng, -72, 72),
        );
        assert_eq!(snapshot.cell_state(cell), dense.get(cell));
    }
}
