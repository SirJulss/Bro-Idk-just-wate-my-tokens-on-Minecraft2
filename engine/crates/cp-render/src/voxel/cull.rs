//! Culling of the draw items on the CPU, for draw path B and the direct path (contract §5.3).
//!
//! Every function repeats its namesake in `voxel/common.wgsl` and `voxel/cull.wgsl` operation
//! for operation in `f32`, and [`DrawList::build`] produces the draws in the order of the GPU
//! compaction: draw class, then LOD level (level 0 first), then item index, then range slot.
//! Both sides compare exactly computed values in the face test; the frustum test may round
//! differently (the GPU can fuse multiply-adds), which its margin makes harmless: whatever
//! either side culls lies outside the frustum in exact arithmetic, so both draw the same
//! pixels.

use cp_core::math::{IVec3, Mat4, Vec3, Vec4};
use wgpu::util::DrawIndexedIndirectArgs;

use super::layout::{
    CULL_MARGIN, CellBounds, DRAW_CLASSES, GpuDrawItem, GpuObject, LOD_LEVELS, MeshLayer,
    RANGES_PER_LAYER, SEGMENTS,
};

/// An axis-aligned box in local space (grid axes, anchor-relative), metres.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct LocalBounds {
    pub(crate) lo: Vec3,
    pub(crate) hi: Vec3,
}

/// The lattice box of the item's cells relative to `anchor` (WGSL `item_bounds`). Exact while
/// every coordinate stays below 2^24 m.
pub(crate) fn item_bounds(item: &GpuDrawItem, anchor: [i32; 3]) -> LocalBounds {
    let cells = CellBounds::unpack(item.cell_bounds);
    let size = IVec3::splat(item.cell_size());
    let base = IVec3::from_array(item.origin).wrapping_sub(IVec3::from_array(anchor));
    let lo = IVec3::from_array(cells.min.map(i32::from));
    let hi = IVec3::from_array(cells.max.map(i32::from)) + IVec3::ONE;
    LocalBounds {
        lo: base.wrapping_add(lo.wrapping_mul(size)).as_vec3(),
        hi: base.wrapping_add(hi.wrapping_mul(size)).as_vec3(),
    }
}

/// WGSL `outside_plane`.
fn outside_plane(plane: Vec4, center: Vec3, extent: Vec3) -> bool {
    let normal = plane.truncate();
    let distance = normal.dot(center) + plane.w;
    let radius = normal.abs().dot(extent);
    let scale = plane.w.abs() + normal.abs().dot(center.abs()) + radius;
    distance + radius < -CULL_MARGIN * scale
}

/// WGSL `bounds_in_frustum`: false only when the box lies outside the left, right, bottom,
/// top or near plane of `local_to_clip` by more than the margin.
pub(crate) fn bounds_in_frustum(local_to_clip: &Mat4, bounds: &LocalBounds) -> bool {
    let center = (bounds.lo + bounds.hi) * 0.5;
    let extent = (bounds.hi - bounds.lo) * 0.5;
    let [r0, r1, r2, r3] = [0, 1, 2, 3].map(|row| local_to_clip.row(row));
    let outside = outside_plane(r3 + r0, center, extent)
        || outside_plane(r3 - r0, center, extent)
        || outside_plane(r3 + r1, center, extent)
        || outside_plane(r3 - r1, center, extent)
        || outside_plane(r3 - r2, center, extent);
    !outside
}

/// WGSL `faces_camera`: whether a face of storage slot `face_slot` (`0..6`) in `bounds` can
/// face a camera at `eye`.
fn faces_camera(face_slot: usize, bounds: &LocalBounds, cell_size: f32, eye: Vec3) -> bool {
    let axis = face_slot % 3;
    if face_slot < 3 {
        eye[axis] > bounds.lo[axis] + cell_size
    } else {
        eye[axis] < bounds.hi[axis] - cell_size
    }
}

/// WGSL `slot_masks`: `(visible, nonempty)` bit masks over the twelve range slots. Items of the
/// cutout layer are drawn from both sides, so every slot of theirs counts as facing the camera.
pub(crate) fn slot_masks(item: &GpuDrawItem, object: &GpuObject) -> (u32, u32) {
    let mut nonempty = 0;
    for slot in 0..RANGES_PER_LAYER {
        if item.slot_count(slot) > 0 {
            nonempty |= 1 << slot;
        }
    }
    let bounds = item_bounds(item, object.anchor);
    let local_to_clip = Mat4::from_cols_array_2d(&object.local_to_clip);
    if !bounds_in_frustum(&local_to_clip, &bounds) {
        return (0, nonempty);
    }
    let cell_size = item.cell_size() as f32;
    let walls = u32::from(item.walls().bits());
    let eye = Vec3::from_array(object.camera_local);
    let two_sided = item.layer_index() == MeshLayer::Cutout as u32;
    let mut visible = 0;
    for slot in 0..RANGES_PER_LAYER {
        let face_slot = slot % 6;
        let enabled = slot < 6 || (walls >> face_slot) & 1 == 1;
        let filled = (nonempty >> slot) & 1 == 1;
        let facing = two_sided || faces_camera(face_slot, &bounds, cell_size, eye);
        if filled && enabled && facing {
            visible |= 1 << slot;
        }
    }
    (visible, nonempty)
}

/// WGSL `count_runs`: the draws of an item.
pub(crate) fn count_runs(visible: u32, nonempty: u32) -> u32 {
    let mut runs = 0;
    let mut open = false;
    for slot in 0..RANGES_PER_LAYER {
        if (visible >> slot) & 1 == 1 {
            if !open {
                runs += 1;
                open = true;
            }
        } else if (nonempty >> slot) & 1 == 1 {
            open = false;
        }
    }
    runs
}

/// WGSL `visible_quads`.
pub(crate) fn visible_quads(item: &GpuDrawItem, visible: u32) -> u32 {
    (0..RANGES_PER_LAYER)
        .filter(|slot| (visible >> slot) & 1 == 1)
        .map(|slot| item.slot_count(slot))
        .sum()
}

/// WGSL `write_runs`: calls `emit(first, quads)` for every run, `first` relative to the
/// item's `quad_base`.
pub(crate) fn for_each_run(item: &GpuDrawItem, visible: u32, mut emit: impl FnMut(u32, u32)) {
    let mut offset = 0;
    let mut start = 0;
    let mut quads = 0;
    let mut open = false;
    for slot in 0..RANGES_PER_LAYER {
        let count = item.slot_count(slot);
        if (visible >> slot) & 1 == 1 {
            if !open {
                open = true;
                start = offset;
                quads = 0;
            }
            quads += count;
        } else if count > 0 && open {
            emit(start, quads);
            open = false;
        }
        offset += count;
    }
    if open {
        emit(start, quads);
    }
}

/// The compacted draws of one frame, as path A writes them to the GPU: per draw class the
/// indirect arguments and the buckets `(draw item, first quad)` of its slots. The draw at
/// position `k` of class `c` uses draw slot `c * class_capacity + k` as `first_instance`.
#[derive(Debug, Default)]
pub(crate) struct DrawList {
    pub(crate) args: [Vec<DrawIndexedIndirectArgs>; DRAW_CLASSES as usize],
    pub(crate) buckets: [Vec<[u32; 2]>; DRAW_CLASSES as usize],
    /// Quads in the visible ranges.
    pub(crate) visible_quads: u64,
    /// Per item: visible mask, draws, segment.
    states: Vec<(u32, u32, u32)>,
}

impl DrawList {
    /// Culls `items` against `objects` (indexed by `item.object`) and compacts the draws.
    /// Items whose object is missing draw nothing.
    pub(crate) fn build(
        &mut self,
        items: &[GpuDrawItem],
        objects: &[GpuObject],
        class_capacity: u32,
    ) {
        self.visible_quads = 0;
        self.states.clear();
        let mut segment_draws = [0_u32; SEGMENTS as usize];
        for item in items {
            let state = match objects.get(item.object as usize) {
                Some(object) => {
                    let (visible, nonempty) = slot_masks(item, object);
                    let draws = count_runs(visible, nonempty);
                    if draws > 0 {
                        self.visible_quads += u64::from(visible_quads(item, visible));
                    }
                    (visible, draws, item.segment())
                }
                None => (0, 0, item.segment()),
            };
            segment_draws[state.2 as usize] += state.1;
            self.states.push(state);
        }
        // Segment bases relative to the class region: lower levels first.
        let mut cursor = [0_u32; SEGMENTS as usize];
        for class in 0..DRAW_CLASSES {
            let mut base = 0;
            for level in 0..LOD_LEVELS {
                let segment = (class * LOD_LEVELS + level) as usize;
                cursor[segment] = base;
                base += segment_draws[segment];
            }
            let draws = base as usize;
            self.args[class as usize].clear();
            self.args[class as usize].resize(draws, DrawIndexedIndirectArgs::default());
            self.buckets[class as usize].clear();
            self.buckets[class as usize].resize(draws, [0, 0]);
        }
        for (index, (item, &(visible, draws, segment))) in
            items.iter().zip(&self.states).enumerate()
        {
            if draws == 0 {
                continue;
            }
            let class = segment / LOD_LEVELS;
            let args = &mut self.args[class as usize];
            let buckets = &mut self.buckets[class as usize];
            let position = &mut cursor[segment as usize];
            for_each_run(item, visible, |first, quads| {
                let at = *position as usize;
                args[at] = DrawIndexedIndirectArgs {
                    index_count: quads * 6,
                    instance_count: 1,
                    first_index: 0,
                    base_vertex: 0,
                    first_instance: class * class_capacity + *position,
                };
                // Item indices fit u32: the table holds at most max_draw_items.
                buckets[at] = [index as u32, item.quad_base + first];
                *position += 1;
            });
        }
    }

    /// Draws of `class`.
    pub(crate) fn draws(&self, class: u32) -> u32 {
        // At most MAX_ITEM_DRAWS per item.
        self.args[class as usize].len() as u32
    }
}

#[cfg(test)]
mod tests {
    use cp_core::det;
    use cp_core::math::{DMat4, DVec3, DVec4, IAabb};

    use super::*;
    use crate::voxel::camera::{Camera, FrameCamera, GridPose, compose_object};
    use crate::voxel::layout::{
        FACE_STORAGE_ORDER, Fade, GpuDrawItem, MAX_ITEM_DRAWS, MeshLayer, WallMask,
    };

    /// A small xorshift generator; the tests need varied but repeatable data.
    struct Random(u64);

    impl Random {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }

        fn below(&mut self, n: u64) -> u64 {
            self.next() % n
        }
    }

    fn random_item(random: &mut Random, object: u32) -> GpuDrawItem {
        let level = random.below(3) as u8;
        let extent = 32 << level;
        let origin = [0, 1, 2].map(|_| (random.below(16) as i32 - 8) * extent);
        let a = [0, 1, 2].map(|_| random.below(32) as u8);
        let b = [0, 1, 2].map(|_| random.below(32) as u8);
        let bounds = CellBounds {
            min: [0, 1, 2].map(|axis| a[axis].min(b[axis])),
            max: [0, 1, 2].map(|axis| a[axis].max(b[axis])),
        };
        let mut counts = [0_u32; 6];
        for count in &mut counts {
            let regular = if random.below(3) == 0 {
                0
            } else {
                random.below(40) as u32
            };
            let walls = if random.below(2) == 0 {
                0
            } else {
                random.below(9) as u32
            };
            *count = regular | walls << 16;
        }
        let layer = MeshLayer::from_index(random.below(3) as usize).unwrap();
        let fade = if random.below(5) == 0 {
            Fade::new(0.5, false)
        } else {
            Fade::FULL
        };
        GpuDrawItem {
            origin,
            lod_flags: GpuDrawItem::pack_lod_flags(
                level,
                layer,
                WallMask::from_bits(random.below(64) as u8),
            ),
            cell_bounds: bounds.pack(),
            quad_base: random.below(1 << 20) as u32,
            counts,
            object,
            fade: fade.pack(),
            reserved: [0; 2],
        }
    }

    fn test_object(eye: DVec3, target: DVec3) -> GpuObject {
        let camera = FrameCamera {
            camera: Camera::look_at(eye, target, DVec3::Y, 1.1),
            aspect: 1.5,
            jitter: [0.0, 0.0],
        };
        let bounds = IAabb::new(
            cp_core::math::IVec3::splat(-1024),
            cp_core::math::IVec3::splat(1024),
        );
        compose_object(
            &bounds,
            &GridPose::IDENTITY,
            &GridPose::IDENTITY,
            &camera,
            &camera,
        )
    }

    /// The reference frustum decision in f64 with the eight corners of the box, or `None` when
    /// the box is within a small tolerance of a plane (where f32 rounding may decide either
    /// way).
    fn reference_in_frustum(object: &GpuObject, bounds: &LocalBounds) -> Option<bool> {
        let m = Mat4::from_cols_array_2d(&object.local_to_clip).as_dmat4();
        let rows: [DVec4; 4] = [0, 1, 2, 3].map(|row| m.row(row));
        let planes = [
            rows[3] + rows[0],
            rows[3] - rows[0],
            rows[3] + rows[1],
            rows[3] - rows[1],
            rows[3] - rows[2],
        ];
        let lo = bounds.lo.as_dvec3();
        let hi = bounds.hi.as_dvec3();
        let mut inside = true;
        for plane in planes {
            let mut best = f64::NEG_INFINITY;
            let mut magnitude: f64 = 0.0;
            for corner in 0..8 {
                let p = DVec3::new(
                    if corner & 1 == 0 { lo.x } else { hi.x },
                    if corner & 2 == 0 { lo.y } else { hi.y },
                    if corner & 4 == 0 { lo.z } else { hi.z },
                );
                let value = plane.truncate().dot(p) + plane.w;
                best = det::max(best, value);
                magnitude = det::max(
                    magnitude,
                    plane.truncate().abs().dot(p.abs()) + plane.w.abs(),
                );
            }
            if best.abs() < 1.0e-3 * magnitude {
                return None;
            }
            if best < 0.0 {
                inside = false;
            }
        }
        Some(inside)
    }

    #[test]
    fn slot_masks_match_a_reference_of_the_rules() {
        let mut random = Random(0x9E37_79B9_7F4A_7C15);
        let mut compared = 0;
        for round in 0..200 {
            let eye = DVec3::new(
                random.below(400) as f64 - 200.25,
                random.below(200) as f64 - 50.5,
                random.below(400) as f64 - 200.75,
            );
            let target = eye
                + DVec3::new(
                    random.below(9) as f64 - 4.0,
                    -1.0,
                    random.below(9) as f64 - 4.5,
                );
            let object = test_object(eye, target);
            for _ in 0..200 {
                let item = random_item(&mut random, 0);
                let (visible, nonempty) = slot_masks(&item, &object);
                let bounds = item_bounds(&item, object.anchor);
                // Nonempty: exactly the slots with quads.
                for slot in 0..RANGES_PER_LAYER {
                    assert_eq!((nonempty >> slot) & 1 == 1, item.slot_count(slot) > 0);
                }
                assert_eq!(visible & !nonempty, 0);
                let Some(in_frustum) = reference_in_frustum(&object, &bounds) else {
                    continue;
                };
                compared += 1;
                if !in_frustum {
                    assert_eq!(visible, 0, "round {round}: {item:?} is outside the frustum");
                    continue;
                }
                // The face rule, from the planes of the quads in the box.
                let eye_local = DVec3::from_array(object.camera_local.map(f64::from));
                let size = f64::from(item.cell_size());
                let cells = CellBounds::unpack(item.cell_bounds);
                let base = DVec3::from_array(
                    [0, 1, 2].map(|axis| f64::from(item.origin[axis] - object.anchor[axis])),
                );
                for slot in 0..RANGES_PER_LAYER {
                    let face = FACE_STORAGE_ORDER[slot % 6];
                    let axis = face.axis();
                    let wall = slot >= 6;
                    let has_quads = item.slot_count(slot) > 0;
                    let enabled = !wall || item.walls().contains(face);
                    // Positive faces: the smallest plane is at min cell + 1; negative faces:
                    // the largest is at max cell. The cutout layer is drawn from both sides.
                    let two_sided = item.layer_index() == MeshLayer::Cutout as u32;
                    let facing = two_sided
                        || if face.is_negative() {
                            eye_local[axis] < base[axis] + f64::from(cells.max[axis]) * size
                        } else {
                            eye_local[axis] > base[axis] + (f64::from(cells.min[axis]) + 1.0) * size
                        };
                    assert_eq!(
                        (visible >> slot) & 1 == 1,
                        has_quads && enabled && facing,
                        "round {round} slot {slot}: {item:?}"
                    );
                }
            }
        }
        assert!(compared > 20_000, "{compared}");
    }

    /// Runs found the slow way: split the slot list at non-empty invisible slots.
    fn reference_runs(item: &GpuDrawItem, visible: u32) -> Vec<(u32, u32)> {
        let mut runs = Vec::new();
        let mut current: Option<(u32, u32)> = None;
        let mut offset = 0;
        for slot in 0..RANGES_PER_LAYER {
            let count = item.slot_count(slot);
            let is_visible = (visible >> slot) & 1 == 1;
            if is_visible {
                let run = current.get_or_insert((offset, 0));
                run.1 += count;
            } else if count > 0 {
                runs.extend(current.take());
            }
            offset += count;
        }
        runs.extend(current);
        runs
    }

    #[test]
    fn runs_cover_exactly_the_visible_quads() {
        let mut random = Random(77);
        for _ in 0..20_000 {
            let item = random_item(&mut random, 0);
            let nonempty = (0..RANGES_PER_LAYER)
                .filter(|&slot| item.slot_count(slot) > 0)
                .fold(0, |mask, slot| mask | 1 << slot);
            let visible = random.next() as u32 & nonempty;
            let mut runs = Vec::new();
            for_each_run(&item, visible, |first, quads| runs.push((first, quads)));
            assert_eq!(runs, reference_runs(&item, visible));
            assert_eq!(runs.len() as u32, count_runs(visible, nonempty));
            assert!(runs.len() as u32 <= MAX_ITEM_DRAWS);
            let quads: u32 = runs.iter().map(|run| run.1).sum();
            assert_eq!(quads, visible_quads(&item, visible));
        }
        // Empty slots between visible ones do not split a run.
        let item = GpuDrawItem {
            counts: [3, 0, 4, 0, 0, 0],
            ..GpuDrawItem::default()
        };
        let mut runs = Vec::new();
        for_each_run(&item, 0b101, |first, quads| runs.push((first, quads)));
        assert_eq!(runs, [(0, 7)]);
    }

    #[test]
    fn draws_are_ordered_by_class_level_item_and_slot() {
        let mut random = Random(4242);
        let object = test_object(DVec3::new(0.5, 40.25, 0.75), DVec3::new(10.0, 0.0, -30.0));
        let items: Vec<GpuDrawItem> = (0..3000).map(|_| random_item(&mut random, 0)).collect();
        let capacity = MAX_ITEM_DRAWS * items.len() as u32;
        let mut list = DrawList::default();
        list.build(&items, &[object], capacity);
        // The order the GPU compaction defines, built by sorting.
        let mut expected: Vec<(u32, u32, usize, u32, u32)> = Vec::new();
        let mut quads = 0_u64;
        for (index, item) in items.iter().enumerate() {
            let (visible, _) = slot_masks(item, &object);
            for_each_run(item, visible, |first, count| {
                expected.push((
                    item.draw_class(),
                    u32::from(item.level()),
                    index,
                    first,
                    count,
                ));
                quads += u64::from(count);
            });
        }
        expected.sort_by_key(|&(class, level, index, first, _)| (class, level, index, first));
        let mut got = Vec::new();
        for class in 0..DRAW_CLASSES {
            for (position, (args, bucket)) in list.args[class as usize]
                .iter()
                .zip(&list.buckets[class as usize])
                .enumerate()
            {
                assert_eq!(args.first_instance, class * capacity + position as u32);
                assert_eq!(
                    (args.instance_count, args.first_index, args.base_vertex),
                    (1, 0, 0)
                );
                let item = &items[bucket[0] as usize];
                got.push((
                    class,
                    u32::from(item.level()),
                    bucket[0] as usize,
                    bucket[1] - item.quad_base,
                    args.index_count / 6,
                ));
            }
        }
        assert_eq!(got, expected);
        assert_eq!(list.visible_quads, quads);
        assert!(!got.is_empty());
        // A missing object makes its items invisible instead of panicking.
        list.build(&items, &[], capacity);
        assert_eq!(list.draws(0) + list.draws(1), 0);
    }

    #[test]
    fn the_frustum_planes_are_the_projection_s() {
        let object = test_object(DVec3::new(0.5, 2.0, 0.5), DVec3::new(0.5, 2.0, -10.0));
        let m = Mat4::from_cols_array_2d(&object.local_to_clip);
        let cube = |center: Vec3| LocalBounds {
            lo: center - Vec3::splat(0.5),
            hi: center + Vec3::splat(0.5),
        };
        // In front of the camera (local space is anchor-relative: the camera sits near 0).
        assert!(bounds_in_frustum(&m, &cube(Vec3::new(0.0, 0.0, -20.0))));
        // Behind it, far to the side, below the near plane distance.
        assert!(!bounds_in_frustum(&m, &cube(Vec3::new(0.0, 0.0, 20.0))));
        assert!(!bounds_in_frustum(&m, &cube(Vec3::new(500.0, 0.0, -20.0))));
        assert!(!bounds_in_frustum(&m, &cube(Vec3::new(0.0, -500.0, -20.0))));
        let reference = DMat4::from_cols_array_2d(&object.local_to_clip.map(|c| c.map(f64::from)));
        let clip = reference * DVec4::new(0.0, 0.0, -20.0, 1.0);
        assert!(clip.w > 0.0);
    }
}
