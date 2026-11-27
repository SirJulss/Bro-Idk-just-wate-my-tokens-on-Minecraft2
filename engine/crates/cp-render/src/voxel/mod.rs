//! The GPU-driven voxel pipeline (contract §5.3, milestone E2).
//!
//! # Data on the GPU
//! - The **quad pool** ([`VoxelScene`]): one storage buffer of [`PackedQuad`]s of a size fixed
//!   at start-up (per quality tier, default 128 MiB), allocated per chunk in whole units of
//!   [`POOL_UNIT_QUADS`] quads, all layers of a chunk in one range. Meshes arrive through
//!   [`ChunkUpload`] (the layout of cp-mesh: layers, then twelve range slots per layer), or as
//!   a [`PreparedChunk`] checked on a worker thread, and are copied in within a per-frame
//!   upload budget.
//! - The **static index buffer**: `QUAD_INDEX_PATTERN` repeated for the largest possible draw
//!   ([`MAX_ITEM_QUADS`]), so every draw uses `first_index = 0` and `base_vertex = 0`.
//! - **Draw items** ([`GpuDrawItem`], 64 bytes): one per non-empty layer of a resident chunk.
//! - **Objects** ([`GpuObject`]): one per drawn grid, with an integer anchor voxel next to the
//!   camera and matrices composed in `f64` ([`Camera`], [`GridPose`]); vertex positions are
//!   `origin - anchor + lattice · cell_size` in `i32`, exact in `f32`, so a chunk 3,000 km out
//!   renders like one at the origin. After the transform every corner moves outward along the
//!   edges of its quad by [`EDGE_OVERLAP_PX`] on screen, so that the T-junctions of greedy
//!   meshes leave no cracks (a recorded deviation from exact lattice positions).
//!
//! # Drawing
//! A draw is a run of visible range slots of one item (empty slots join their neighbours, a
//! non-empty invisible slot ends a run). Every draw has one bucket `(draw item, first quad)`;
//! the vertex shader reads it as an instance-step `Uint32x2` attribute addressed by the draw's
//! `first_instance` and never uses `@builtin(instance_index)`.
//!
//! - [`DrawMode::A`]: a compute pass culls (frustum, face visibility per range, wall ranges
//!   only with the face in the item's [`WallMask`]) and compacts deterministically (count,
//!   prefix sum, scatter into per-level segments, level 0 first); one
//!   `multi_draw_indexed_indirect_count` per draw class.
//! - [`DrawMode::B`]: the same cull on the CPU (a mirror of the WGSL, operation for
//!   operation), one `multi_draw_indexed_indirect` per draw class.
//! - [`DrawMode::Direct`]: the CPU's list, one `draw_indexed` per bucket.
//!
//! The draw classes are the two pipelines: opaque (opaque and fluid layers; fluids are opaque
//! until the water pass exists) and cutout (alpha test, and every item that is cross-fading
//! with a dither pattern). All modes issue the same draws in the same order: class, level,
//! item, slot, so their images are identical.
//!
//! # The frame
//! `cull` (path A) → `gbuffer` (albedo `Rgba8UnormSrgb`, normal + roughness `Rgb10a2Unorm`
//! octahedral, material `Rgba8Unorm`, `Depth32Float` reversed Z; pixel-art sampling with
//! texel snapping, an analytic one-pixel edge ramp and `textureSampleGrad`) → `overdraw`
//! (overdraw view) → `lighting` (sun, hemispheric sky ambient, height fog, gradient sky into
//! `Rgba16Float`) → `taa` (temporal anti-aliasing: every frame is jittered by a Halton
//! sub-pixel offset and blended into a history reprojected from depth and the last camera,
//! clipped to the current neighbourhood; renderers configured without it skip the pass) →
//! `composite` (AgX into the output, or a debug view) → `bounds` (chunk-bounds view) →
//! `overlay`. Every pipeline is created by [`VoxelRenderer::new`]
//! ([`PIPELINE_COUNT`]); timings per pass come from the frame graph, counts per frame from
//! [`FrameStats`].
//!
//! [`PackedQuad`]: cp_core::quad::PackedQuad

mod calibration;
mod camera;
mod cull;
mod ground;
mod layout;
mod pipelines;
mod pool;
mod renderer;
mod scene;
mod state;
mod stats;
mod taa;
#[cfg(test)]
mod tests;

pub use calibration::{CalibrationCounts, CalibrationParams, CalibrationScene};
pub use camera::{
    ANCHOR_MARGIN_VOXELS, CLOUD_ANCHOR_PERIOD, Camera, GridPose, Lighting, anchor_voxel,
    cloud_anchor,
};
pub use ground::GroundMap;
pub use layout::{
    CULL_MARGIN, CULL_WORKGROUP, CellBounds, DRAW_CLASSES, EDGE_OVERLAP_PX, FACE_STORAGE_ORDER,
    Fade, GpuDrawItem, GpuFrame, GpuObject, LOD_LEVELS, MAX_ITEM_DRAWS, MAX_ITEM_QUADS,
    MAX_REGULAR_RANGE_QUADS, MAX_WALL_RANGE_QUADS, MESH_LAYERS, MeshLayer, OVERLAP_SLIDE_MAX_PX,
    OVERLAP_STRETCH_MAX, POOL_UNIT_QUADS, QUAD_BYTES, RANGES_PER_LAYER, SEGMENTS, TEXTURED_LEVELS,
    WallMask, storage_slot,
};
pub use pipelines::{
    ALBEDO_FORMAT, DEPTH_FORMAT, HDR_FORMAT, MATERIAL_FORMAT, NORMAL_FORMAT, OVERDRAW_FORMAT,
    PASS_BOUNDS, PASS_COMPOSITE, PASS_CULL, PASS_GBUFFER, PASS_LIGHTING, PASS_OVERDRAW,
    PASS_OVERLAY, PASS_TAA, PIPELINE_COUNT,
};
pub use pool::{PoolRange, RangeAllocator};
pub use renderer::{DrawMode, FrameInput, UnsupportedDrawMode, VoxelRenderer, VoxelRendererConfig};
pub use scene::{
    ChunkId, ChunkUpload, LayerCounts, ObjectDesc, ObjectId, PreparedChunk, RefusedChunk,
    SceneLimits, SceneStats, UploadReport, VoxelError, VoxelScene,
};
pub use stats::{FrameStats, VisibleCounts};
pub use taa::{
    GpuTaa, HISTORY_FORMAT, JITTER_PHASES, MIN_BLEND, STILL_BLEND, jitter_ndc, jitter_pixels,
};
