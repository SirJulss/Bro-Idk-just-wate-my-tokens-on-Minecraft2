// Draw path A: GPU culling of the draw items with deterministic compaction (contract Â§5.3).
//
// Three dispatches in one compute pass, separated by wgpu's storage barriers:
//
// 1. cull_count, one thread per item: frustum and face tests give the visible range slots and
//    the number of draws (runs of visible slots); per block of 256 items and per segment
//    (draw class, LOD level) the draws are summed into scratch.
// 2. cull_scan, one workgroup per segment: exclusive prefix sum of the block sums over the
//    blocks, and the segment total.
// 3. cull_scatter, one thread per item: the draw slot of the item's first run is its class
//    region + the totals of the earlier segments of its class (lower levels first) + the block
//    offset + the draws of the earlier items of its block in the same segment. Each run
//    becomes DrawIndexedIndirectArgs { 6 * quads, 1, 0, 0, slot } and buckets[slot] =
//    (item, first quad). Workgroup 0 writes the draw count of each class.
//
// The order of the draws is therefore a pure function of the item order: class, then level
// (near to far), then item index, then range slot. Path B computes the same order on the CPU.

#include "voxel/common.wgsl"

@group(0) @binding(0) var<uniform> frame: Frame;
@group(0) @binding(1) var<storage, read> items: array<DrawItem>;
@group(0) @binding(2) var<storage, read> objects: array<Object>;
// [0, item count): item state, visible slots in bits 0-11, draws in 12-15, segment in 16-21.
// From frame.scratch.x: block sums, one row of frame.cull.z blocks per segment.
// From frame.scratch.y: one total per segment.
@group(0) @binding(3) var<storage, read_write> scratch: array<u32>;
// DrawIndexedIndirectArgs, five words each; class c owns slots [c, c + 1) * frame.cull.w.
@group(0) @binding(4) var<storage, read_write> args: array<u32>;
// (draw item, first quad) per draw slot; the vertex shader reads it as an instance attribute.
@group(0) @binding(5) var<storage, read_write> buckets: array<vec2<u32>>;
// Draws of class 0 and 1 (the count buffer of the indirect draws), visible quads, reserved.
@group(0) @binding(6) var<storage, read_write> counters: array<atomic<u32>>;

var<workgroup> group_draws: array<atomic<u32>, SEGMENTS>;
var<workgroup> group_quads: atomic<u32>;
var<workgroup> group_keys: array<u32, CULL_GROUP>;
var<workgroup> group_bases: array<u32, SEGMENTS>;
var<workgroup> group_scan: array<u32, CULL_GROUP>;

@compute @workgroup_size(CULL_GROUP)
fn cull_count(
    @builtin(global_invocation_id) global: vec3<u32>,
    @builtin(local_invocation_index) local: u32,
    @builtin(workgroup_id) group: vec3<u32>,
) {
    if local < SEGMENTS {
        atomicStore(&group_draws[local], 0u);
    }
    if local == 0u {
        atomicStore(&group_quads, 0u);
    }
    workgroupBarrier();
    let index = global.x;
    if index < frame.cull.x {
        let item = items[index];
        let masks = slot_masks(item, objects[item.object]);
        let draws = count_runs(masks.x, masks.y);
        let segment = item_segment(item);
        scratch[index] = masks.x | (draws << 12u) | (segment << 16u);
        if draws > 0u {
            atomicAdd(&group_draws[segment], draws);
            atomicAdd(&group_quads, visible_quads(item, masks.x));
        }
    }
    workgroupBarrier();
    if local < SEGMENTS {
        let row = frame.scratch.x + local * frame.cull.z;
        scratch[row + group.x] = atomicLoad(&group_draws[local]);
    }
    if local == 0u {
        atomicAdd(&counters[COUNTER_QUADS], atomicLoad(&group_quads));
    }
}

@compute @workgroup_size(CULL_GROUP)
fn cull_scan(
    @builtin(local_invocation_index) local: u32,
    @builtin(workgroup_id) group: vec3<u32>,
) {
    let segment = group.x;
    let blocks = frame.cull.y;
    let row = frame.scratch.x + segment * frame.cull.z;
    var carry = 0u;
    for (var start = 0u; start < blocks; start += CULL_GROUP) {
        let block = start + local;
        var value = 0u;
        if block < blocks {
            value = scratch[row + block];
        }
        group_scan[local] = value;
        workgroupBarrier();
        // Hillis-Steele inclusive scan over the 256 entries.
        for (var offset = 1u; offset < CULL_GROUP; offset <<= 1u) {
            var add = 0u;
            if local >= offset {
                add = group_scan[local - offset];
            }
            workgroupBarrier();
            group_scan[local] += add;
            workgroupBarrier();
        }
        if block < blocks {
            scratch[row + block] = carry + group_scan[local] - value;
        }
        carry += group_scan[CULL_GROUP - 1u];
        workgroupBarrier();
    }
    if local == 0u {
        scratch[frame.scratch.y + segment] = carry;
    }
}

// Writes one draw: `quads` quads from pool index `first_quad` of item `item` into slot
// `position`.
fn write_draw(position: u32, item: u32, first_quad: u32, quads: u32) {
    let word = position * 5u;
    args[word] = quads * 6u;
    args[word + 1u] = 1u;
    args[word + 2u] = 0u;
    args[word + 3u] = 0u;
    args[word + 4u] = position;
    buckets[position] = vec2<u32>(item, first_quad);
}

// Writes the runs of `visible` slots of item `index` into consecutive slots from `first`.
fn write_runs(index: u32, item: DrawItem, visible: u32, first: u32) {
    var position = first;
    var offset = 0u;
    var start = 0u;
    var quads = 0u;
    var open = false;
    for (var slot = 0u; slot < SLOTS; slot += 1u) {
        let count = slot_quads(item.counts, slot);
        if ((visible >> slot) & 1u) == 1u {
            if !open {
                open = true;
                start = offset;
                quads = 0u;
            }
            quads += count;
        } else if count > 0u && open {
            write_draw(position, index, item.quad_base + start, quads);
            position += 1u;
            open = false;
        }
        offset += count;
    }
    if open {
        write_draw(position, index, item.quad_base + start, quads);
    }
}

@compute @workgroup_size(CULL_GROUP)
fn cull_scatter(
    @builtin(global_invocation_id) global: vec3<u32>,
    @builtin(local_invocation_index) local: u32,
    @builtin(workgroup_id) group: vec3<u32>,
) {
    let totals = frame.scratch.y;
    if local < SEGMENTS {
        let draw_class = local / LOD_LEVELS;
        var base = draw_class * frame.cull.w;
        for (var segment = draw_class * LOD_LEVELS; segment < local; segment += 1u) {
            base += scratch[totals + segment];
        }
        group_bases[local] = base;
    }
    let index = global.x;
    let in_range = index < frame.cull.x;
    var state = 0u;
    if in_range {
        state = scratch[index];
    }
    let draws = (state >> 12u) & 15u;
    let segment = state >> 16u;
    group_keys[local] = (segment << 8u) | draws;
    workgroupBarrier();
    if group.x == 0u && local < DRAW_CLASSES {
        var total = 0u;
        for (var level = 0u; level < LOD_LEVELS; level += 1u) {
            total += scratch[totals + local * LOD_LEVELS + level];
        }
        atomicStore(&counters[local], total);
    }
    if !in_range || draws == 0u {
        return;
    }
    let row = frame.scratch.x + segment * frame.cull.z;
    var position = group_bases[segment] + scratch[row + group.x];
    for (var other = 0u; other < local; other += 1u) {
        let key = group_keys[other];
        if (key >> 8u) == segment {
            position += key & 255u;
        }
    }
    write_runs(index, items[index], state & 0xfffu, position);
}
