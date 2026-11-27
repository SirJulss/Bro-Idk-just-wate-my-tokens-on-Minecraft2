# E3 – Streaming and LOD (contract)

**Status:** binding for milestone E3 once the lead accepts it. Read together with `ARCHITEKTUR.md`
v3 (the "contract"); where this file needs v3 to change, §12 lists the proposed text and v3 stays
authoritative until the lead applies it. Synthesised from three independent designs and two judge
verdicts; the base is the design "zukunft" (highest summed judge total, 91), with the grafts,
fixes and corrections recorded in §13.

**Conventions.**
- Units: m, s, ms, px, bytes. "px" without qualifier = pixels of a 1080-line reference viewport.
  "Lattice m" = grid-local level-0 voxel coordinates as `f64` metres; for a body, lattice = body +
  `grid_offset` on every axis (`cp-universe/src/body.rs:8-15`).
- *Shall* is binding, *may* is allowed. **(A)** marks an assumption decided without measurement:
  the first E3 bench confirms or corrects it, and a failed assumption is reported, never silently
  changed. "Derived" marks numbers computed here from cited measurements.
- `crate/path:line` is relative to `engine/crates/`; `docs/…` = `engine/docs/`; `src/render/lod.rs`
  = the frozen prototype (quarry only). Line numbers were read on 2026-11-27 evening. The E1/E2
  beauty loop is still editing cp-render, cp-app, cp-worldgen and content/base (several cited lines
  moved while this file was written), so builders locate code by the quoted names and re-read every
  file before changing it.

---

## 1. Scope and non-goals

### 1.1 Scope
E3 makes the flight block → orbit → block seamless on Heimat (edge 40,960 m, root level 10,
`grid_offset` 1365; `content/base/systems/home.toml:55-81`) within the budgets of contract §6, §7
and §9. It builds:

1. **cp-stream** (empty today, `cp-stream/src/lib.rs:1-5`): render residency for any number of
   grids and views (LOD selection under hard caps, GPU-timed dithered transitions, exact wall
   masks, core occlusion, governor), production (own priority queue drained by pump jobs, family and
   single mesh jobs, edit groups), caches (class, memory mesh, disk mesh) and simulation residency
   (anchors, pins, counted blocking reads; `docs/PHYSIK.md:417-419`).
2. **cp-voxel**: an additive planning, sampling and residency API (§3.2).
3. **cp-mesh**: a `ChunkMesh` codec for the disk cache (§5.5).
4. **cp-save**: a minimal generic pack store (§7.9); the crate is a stub today
   (`cp-save/src/lib.rs:1-6`).
5. **cp-worldgen**: the cube test body `cubeplanet:cube_hills` and the analytic
   `cubeplanet:cube_flat` (§4). Every existing generator is an infinite `+Y` plane
   (`cp-worldgen/src/lib.rs:14`).
6. **cp-render**: GPU-timed fades, culling of hidden items, upload lanes and groups, coalesced item
   writes, two-phase Hi-Z on path A, integer fragment positions, six-face shading with face ground
   maps, interim orbit lighting, relative TAA cut, debug views, depth read-back, counters, memory
   report (§6).
7. **cp-app**: the cube scene on Heimat (first consumer of cp-universe), the generator factory, the
   stream glue, routes, streaming modes of `cpe run/bench/shot`, `cpe probe`, the E3 showcase and
   one acceptance command (§8).
8. **content**: the eight cube bodies of `home.toml` name `cubeplanet:cube_hills` (§4.10).

### 1.2 Non-goals
- Terrain 2.0, macro maps, stamps (E6); `cube_hills` is a test generator.
- Atmosphere, limb glow, night, shadows (E5). E3 has an interim space factor (§6.9) and fixes the
  shadow switch rule for E5 (§6.3).
- Ocean surface clipmap, translucent water, fluid channel (E7).
- Other bodies in the sky, SOI changes, map camera, star points (E4). The cp-stream API supports
  many grids and frustum-limited views (I1, I2); E3 tests them with synthetic grids only.
- Moving constructs and a velocity buffer (P2/E8); `Streamer::set_roots` exists for them.
- Occlusion on path B and direct draws (§6.6), nested fades (§13 D5), `memmap2` (§7.9).
- Changes to the E2 scenes (hills, structures, calib): they keep the static planner
  (`cp-app/src/scene/lod.rs`) and must render unchanged (gate A18).
- Network replication and persistence of streaming state.

### 1.3 Preconditions and parallel work
- The E3 build starts after the E1/E2 acceptance and runs in parallel with E4 and P1
  (`docs/STAND.md:86`, `101-108`). File ownership (§10) keeps the three apart.
- Two facts bind the plan: since the end of E2 the hills generation grew from 0.561–0.706 ms mean
  per chunk (`docs/bench/e2-hills.json:787-1077`) to about 3.7 ms at scene start, and the G-buffer
  pass to 1.77 ms at 1 M triangles (`docs/STAND.md:62-67`). Generators and shading get more
  expensive in beauty loops; E3's throughput and frame-time gates hold after every beauty fix
  (contract §9, `ARCHITEKTUR.md:645`).

---

## 2. Terms and invariants

### 2.1 Terms

| Term | Meaning |
|---|---|
| grid | a voxel grid (`GridId`, contract §2.2) with one `VoxelWorld` and its snapshots (`cp-voxel/src/world.rs:34-43`) |
| root set | `RootSet { level, min, max }` (inclusive chunk positions). A body: `level = root_level`, `pos ∈ {-1,0}³` (contract §2.2); Heimat: level 10, lattice `[-32768, 32768)³` |
| node | one `LodKey` of a grid's octree below its root set |
| leaf / split | a node drawn as itself / represented by its eight children |
| cut | the drawn leaves plus the void nodes; together they cover the root set |
| drawn | in the current cut with a display that keeps at least one dither threshold now |
| resident | held by the render sink: hidden, pending upload, drawn, fading or dormant-resident |
| hidden | resident with `FadeCmd::Hidden`; keeps no threshold and is culled (§6.4) |
| void | needs no mesh: class Empty, a mesh with 0 uploaded quads, or dormant with an empty mask |
| dormant | class Opaque with no regular quads (walls only, or not built yet); resident only while its wall mask is non-empty (§5.4) |
| family | one transition: parent P and its eight children; *split* (children appear) or *merge* (P appears) |
| ramp | the GPU-timed fade of a family (§6.3) |
| view kinds | *Primary* (the player camera, exactly one per frame), *Omni* (other cameras that need LOD in every direction: other players, observers), *Frustum* (telescope, map camera: refine only inside their frustum) |
| `f_v` | focal length of view v: `min(H_v, 1080) / 2 / tan(fov_y / 2)` px; 935.3 px at 60° |
| `e(N)` | normalised error of node N (§7.2): split iff `e > 1`, merge iff `e < 1/1.15` |
| bias `b` | lod_bias in `[1.0, 1.5]` (contract §7 governors) |
| in-frustum | a drawn chunk whose voxel box meets the Primary view's frustum |
| quick class | O(1) class from the source's declared shape (§3.2) |
| owner face | the face whose chart generates a cell: dominant axis of (cell centre − cube centre) (§4.2) |

### 2.2 Invariants (binding)
- **I1 Grids.** cp-stream streams any number of grids. Each has its own root set and its own views
  in its own lattice frame; all grids share one budget (§7.5). E4 bodies and E8 constructs need no
  API change.
- **I2 Views.** A frame has any number of views, exactly one Primary. Primary and Omni views refine
  in every direction: while no cap binds, turning a camera never changes the cut. Frustum views add
  demand only inside their frustum widened by 5 %, never outside it.
- **I3 Pure meshes.** A chunk mesh is a pure function of `GeneratorIdentity`, the grid, the
  `LodKey`, the override data of the chunk and of its six face-halo slices, the content tables and
  `MESH_FORMAT_VERSION`/`MESHER_VERSION` (`cp-mesh/src/lib.rs:89-92`, `cp-voxel/src/stamp.rs:1-9`).
  The LOD structure never enters a mesh. Every built `ChunkMesh` keeps all six wall ranges of all
  three layers (`cp-mesh/src/mesher.rs:305-338`); what is uploaded follows §5.3. In memory a mesh
  is valid while `VoxelSnapshot::revision(key)` is unchanged (`cp-voxel/src/snapshot.rs:155-166`),
  on disk while its persistent key (§5.6) is unchanged.
- **I4 Residency is speed and looks.** Render residency changes what is drawn and how fast;
  simulation residency changes only speed. cp-stream never answers a simulation query; simulation
  reads go through cp-voxel, whose results never depend on cache state (`cp-voxel/src/cache.rs:5-7`).
- **I5 Replica-agnostic.** cp-stream consumes per-grid snapshots plus that grid's edit receipts in
  order and cannot tell the in-process server's world from a cp-net replica.
- **I6 Declared shape.** The frame thread never evaluates a generator field. It plans with O(1)
  declared shapes (§3.2) and the class results of jobs.
- **I7 Hard caps.** In every frame: resident ≤ 5,500 and in-frustum ≤ 1,400 chunks, summed over all
  grids (§7.5). τ is a target, not a guarantee.
- **I8 No waits.** In Realtime mode the frame thread never blocks on generation, meshing, disk or a
  lock held while foreign code runs (contract §1 principle 4). Only the tool modes Settle and
  Deterministic block (§7.12).
- **I9 Exact coverage.** In every frame every point of every root box lies in a void node, or in
  exactly one drawn Full leaf, or in the parent and exactly one child of one fading family whose
  displays partition the dither thresholds exactly (§6.3). No drawn chunk is removed while it keeps
  a threshold.
- **I10 Exact walls.** Every drawn chunk's wall mask equals the rule of §5.3 in every frame.
- **I11 Atomic edits.** All meshes of one edit receipt become visible in the same frame, at most
  2 frames after the receipt (contract §9 targets, `ARCHITEKTUR.md:647-650`).
- **I12 Reproducible tools.** In Deterministic mode (§7.12) the sink commands and the rendered
  images are a pure function of content, generators, route and frame index, on one machine and
  driver.

### 2.3 Numbers (binding defaults)

| Item | Value |
|---|---|
| τ | 6.0 px on every quality tier (contract §6) |
| level-1 floor | the level-0/1 split stays ≥ 240 m and its merge ≥ 276 m at any bias (§7.2) |
| hysteresis | split at `e > 1`, merge at `e < 1/1.15` |
| look-ahead | `0.5 s` of velocity, at most 2,000 m |
| silhouette gains | ×1.5 in the cube-edge silhouette band; ×1.5 on skyline nodes below 5 km altitude |
| bias | `[1.0, 1.5]` |
| caps | resident 5,500; in-frustum 1,400; new splits only below 80 %; forced merges only above 100 % |
| fades | 250 ms; 66 ms when lagging or forced; 0 in Settle mode and in a TAA-cut frame; 16 levels |
| transitions | ≤ 128 open families; ≤ 32 starts per frame |
| frame-thread caps | ≤ 6,000 node visits; ≤ 128 results or 0.25 ms; ≤ 512 sink operations; ≤ 4 MiB of new meshes per frame |
| workers | `bulk_limit + 2` render pump tokens (Streaming, key 1); ≤ 2 simulation pumps (Streaming, key 0); edit groups Interactive |
| caches | class ≤ 2^20 entries; memory mesh cache 256 MiB; disk mesh cache for levels ≥ 3, 1 GiB in packs of 64 MiB |
| simulation | ≤ 2,048 level-0 chunks per anchor |
| node lifetime | freed 600 frames after it was last wanted, drawn, requested or referenced |

Level distances at τ = 6, b = 1, 60° vertical FOV: `D_L = 2^L · 935.3 / 6 = 155.9 · 2^L` m; a
level-L node splits below `D_L` and its children merge above `1.15 · D_L` (derived).

| L | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| `D_L` (m) | 312 | 624 | 1,247 | 2,494 | 4,988 | 9,977 | 19,953 | 39,906 | 79,813 | 159,626 |

From the 40 km orbit (19,520 m above the +Z face centre) the nearest surface is drawn at level 6.
At 321 m/s (5.35 m per 60 Hz frame) a node on the flight line passes from the level-2/1 switch
(624 m) to the level-1/0 switch (312 m) in 0.97 s (derived).

---

## 3. cp-stream API

### 3.1 Crate rules
- Internal dependencies exactly as contract §3.1: cp-core, cp-voxel, cp-mesh, cp-jobs, cp-save;
  external: parking_lot, crossbeam-channel, rustc-hash, log (contract §3.2). Today the manifest
  lists cp-worldgen and lacks cp-save (`cp-stream/Cargo.toml:10-20`): cp-worldgen moves to
  `[dev-dependencies]` (Heimat tests only), cp-save is added. thiserror is not in the list, so
  cp-stream errors implement `std::error::Error` by hand.
- cp-stream never names cp-render types (its dependency row, `ARCHITEKTUR.md:190`; cp-app glues,
  `ARCHITEKTUR.md:196`); it calls the injected traits of §3.4.
- cp-stream is not a deterministic crate (contract §2.5), but it follows the arithmetic rules so that
  Deterministic mode reproduces: `min`/`max`/`clamp` and transcendental functions through
  `cp_core::det` (`engine/clippy.toml:12-75` applies to every crate), no result depends on hash-map
  iteration order.
- Profile scopes: `cp_stream::select`, `cp_stream::apply`, `cp_stream::walls`, `cp_stream::publish`.

### 3.2 Inputs from cp-voxel (WP-A; additive, no existing signature or behaviour changes)

```rust
// base.rs
/// Conservative O(1) shape of a base world, valid at every level. Lattice voxels, half-open.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SourceShape {
    /// Every cell of every level whose centre lies inside `core` is OPAQUE.
    pub core: IAabb,
    /// Every cell of every level whose centre lies outside `hull` is AIR.
    pub hull: IAabb,
}
impl SourceShape {
    pub const ALL_AIR: Self;                                    // empty core and hull
    pub fn class_of(&self, key: LodKey) -> Option<NodeClass>;   // Empty: no cell centre in hull;
                                                                // Opaque: every cell centre in core
    pub fn subtree_empty(&self, key: LodKey) -> bool;           // key.aabb_voxels() misses hull
}
pub trait BaseSource: Send + Sync {
    /* identity, fill, classify, subtree_empty unchanged (base.rs:78-110) */
    fn shape(&self) -> Option<SourceShape> { None }            // O(1)
}

// plan.rs (new)
impl VoxelSnapshot {
    pub fn shape(&self) -> Option<SourceShape>;
    pub fn edited_within(&self, key: LodKey) -> bool;
    pub fn quick_class(&self, key: LodKey) -> Option<NodeClass>;   // shape class unless edited_within
    pub fn quick_subtree_empty(&self, key: LodKey) -> bool;        // false = unknown
}

// sample.rs (new)
/// Returned by a cancelled sample: the caller's buffer comes back.
pub struct SampleCancelled { pub cells: Box<[BlockStateId]> }
/// Per-thread scratch of the family sample: the 66³ region and spare padded buffers.
pub struct FamilyScratch { /* … */ }
impl FamilyScratch {
    pub fn new() -> Self;
    pub fn recycle(&mut self, padded: PaddedChunk);   // takes the cells back (PaddedChunk::into_parts)
}
impl VoxelSnapshot {
    /// `try_sample_padded` into a caller buffer of exactly PADDED_VOLUME cells (panics otherwise).
    pub fn try_sample_padded_with(&self, key: LodKey, cancel: &dyn Fn() -> bool,
                                  cells: Box<[BlockStateId]>) -> Result<PaddedChunk, SampleCancelled>;
    /// The eight padded children of `parent` (level >= 1) from one fill of the family region.
    pub fn try_sample_padded_family(&self, parent: LodKey, cancel: &dyn Fn() -> bool,
                                    scratch: &mut FamilyScratch) -> Result<[PaddedChunk; 8], Cancelled>;
}

// residency.rs (new)
/// RAII pin: while alive, the cached base chunk is never evicted.
pub struct BasePin { /* key, Arc<VoxelChunk> */ }
impl BasePin { pub fn key(&self) -> LodKey; pub fn chunk(&self) -> &Arc<VoxelChunk>; }
impl VoxelSnapshot {
    pub fn try_pin(&self, key: LodKey, cancel: &dyn Fn() -> bool) -> Result<BasePin, Cancelled>;
}
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ReadStats { pub hits: u64, pub generated: u64, pub waited: u64, pub blocked_ns: u64 }
/// Base-cache reads of the calling thread since its previous call.
pub fn take_thread_read_stats() -> ReadStats;

// cache.rs: CacheUsage gains `hits`, `misses`, `pinned_chunks`, `pinned_bytes`.
// world.rs
impl WorldConfig { pub fn for_root_level(root_level: u8) -> Self; }   // top_level = root_level, 128 MiB
```

Rules:
- `shape` never contradicts `classify`, `subtree_empty` or `fill` (tested on 5,000 random keys of a
  test source with a known core and hull).
- `edited_within(key)` = `has_layer_within(key, min(key.level, top_level))`
  (`cp-voxel/src/layers.rs:138-167`). Every level-0 override leaves a layer on every pyramid level up
  to the top level (`cp-voxel/src/snapshot.rs:79-81`), so one level suffices.
- `try_sample_padded_with` and `try_sample_padded_family` produce exactly the cells and authored
  masks of `try_sample_padded` (`cp-voxel/src/snapshot.rs:131-153`) for every key, including keys
  whose own layer or face-neighbour layers carry overrides. The family variant does one
  `base.fill` over the cells `[64·P.pos − 1, 64·P.pos + 65)³` of level `P.level − 1` (66³ =
  287,496 cells, 1,149,984 B), which is valid because fill output never depends on the region asked
  for (`cp-voxel/src/base.rs:75-76`); it copies the eight 34³ blocks at offsets `32·(dx, dy, dz)` and
  applies the overlays exactly like `try_sample_padded`: the private `overlay_interior` and
  `overlay_face` (`cp-voxel/src/snapshot.rs:216-266`) become `pub(crate)`, nothing else in
  snapshot.rs changes. Today every padded sample allocates a new 157,216 B buffer
  (`cp-voxel/src/snapshot.rs:138`).
- Pins: LRU eviction skips pinned entries; only pinned bytes may exceed the budget (today the oldest
  entry is evicted unconditionally, `cp-voxel/src/cache.rs:96-103`).
- `ReadStats` are thread-local counters incremented in `BaseCache::get_or_generate`
  (`cp-voxel/src/cache.rs:150-211`): a hit → `hits`; a leader generation → `generated` and its time
  into `blocked_ns`; a waiter → `waited` and its wait into `blocked_ns`. Counting per thread keeps
  prefetch work on worker threads out of the simulation's "physics waited" count.
- A body's world shall use `WorldConfig::for_root_level(root_level)` (Heimat: 10); the default
  (top level 20, `cp-voxel/src/world.rs:24-31`) stays for the plane worlds.
- WP-A touches neither `edit.rs`, `query.rs`, `chunk_data.rs` nor `padded.rs`, which P1 step 0
  changes (`docs/PHYSIK.md:346-416`). P1's `bounds(grid)` (`docs/PHYSIK.md:386`) is a different item;
  the names `SourceShape`/`shape` avoid a clash.

### 3.3 Public types

```rust
pub struct StreamConfig {
    pub tau_px: f64,                    // 6.0
    pub reference_height_px: u32,       // 1080
    pub level1_floor_m: f64,            // 240.0
    pub hysteresis: f64,                // 0.15
    pub lookahead_s: f64,               // 0.5
    pub lookahead_max_m: f64,           // 2_000.0
    pub silhouette_gain: f64,           // 1.5 (cube-edge band)
    pub skyline_gain: f64,              // 1.5 (1.0 disables)
    pub skyline_max_altitude_m: f64,    // 5_000.0
    pub fade_ms: u16,                   // 250
    pub fast_fade_ms: u16,              // 66
    pub max_resident: u32,              // 5_500
    pub max_in_frustum: u32,            // 1_400
    pub select_fill: f64,               // 0.8
    pub max_open_families: u32,         // 128
    pub max_new_transitions: u32,       // 32 per frame
    pub max_visits: u32,                // 6_000 per frame
    pub max_results: u32,               // 128 per frame
    pub max_result_ms: f64,             // 0.25
    pub max_sink_ops: u32,              // 512 per frame
    pub upload_bytes_per_frame: u64,    // 4 << 20
    pub mem_cache_bytes: usize,         // 256 << 20
    pub class_cache_entries: usize,     // 1 << 20
    pub disk: Option<DiskCacheConfig>,
    pub pump_slice_ms: f64,             // 2.0
    pub node_ttl_frames: u32,           // 600
    pub mode: StreamMode,               // Realtime
}
pub struct DiskCacheConfig { pub dir: PathBuf, pub budget_bytes: u64 /* 1 << 30 */,
                             pub pack_bytes: u64 /* 64 << 20 */, pub min_level: u8 /* 3 */ }
pub enum StreamMode { Realtime, Settle, Deterministic { latency_frames: u32 /* 4 */ }, Inline }
pub struct StreamTables { pub flags: Arc<[BlockFlags]>, pub render: Arc<[BlockRender]>,
                          pub cull_group: Arc<[u16]>, pub state_keys: Arc<[Box<str>]>,
                          pub content_hash: u64 }
pub struct RootSet { pub level: u8, pub min: IVec3, pub max: IVec3 }   // inclusive
impl RootSet { pub fn body(root_level: u8) -> Self; }                  // pos ∈ {-1, 0}³
pub struct CubeFrame { pub centre: DVec3 /* lattice m */, pub half_edge: f64 /* m */ }
pub struct GridDesc { pub grid: GridId, pub roots: RootSet, pub cache_salt: Option<u64>,
                      pub cube: Option<CubeFrame> }
pub struct ViewId(pub u32);
pub enum ViewKind { Primary, Omni, Frustum }
pub struct ViewDesc { pub id: ViewId, pub kind: ViewKind, pub fov_y: f64, pub aspect: f64,
                      pub viewport_px: [u32; 2], pub lod_bias: f64, pub priority_weight: f64 }
pub struct GridView { pub view: ViewId, pub eye: DVec3, pub view_to_grid: DQuat,
                      pub velocity: DVec3, pub up: DVec3, pub draw: bool }
                      // lattice m and m/s; up = −gravity at the eye, unit; view space looks along −Z
pub struct GridFrame<'a> { pub grid: GridId, pub snapshot: &'a Arc<VoxelSnapshot>,
                           pub receipts: &'a [EditReceipt], pub views: &'a [GridView] }
pub struct StreamFrame<'a> { pub frame: u64, pub clock_ms: u32, pub taa_cut: bool,
                             pub views: &'a [ViewDesc], pub grids: &'a [GridFrame<'a>] }
```

Rules:
- `cache_salt` = `cp_core::hash::hash_str(body key)` for bodies: a `BodyId` is a session-local list
  index and never enters a persistent hash (contract §2.5 rule 5). `None` for constructs (no disk
  cache).
- `clock_ms` is the value cp-app passes to `VoxelRenderer::set_clock_ms` for the same frame (§6.3).
  `taa_cut` is true when the renderer resets its TAA history in this frame (§6.9).
- `receipts` are the receipts produced since the previous frame, in apply order.

### 3.4 Backend and sink traits

```rust
/// Worker side; cp-app implements it with PreparedChunk::from_vec.
pub trait MeshBackend: Send + Sync + 'static {
    type Mesh: Clone + Send + 'static;
    /// `mesh` holds every wall range; the backend uploads exactly the ranges of §5.3.
    fn prepare(&self, key: ChunkKey, mesh: &ChunkMesh) -> Result<Self::Mesh, String>;
    fn bytes(mesh: &Self::Mesh) -> u64;
}
/// Frame thread; cp-app implements it with VoxelScene.
pub trait RenderSink<M> {
    fn insert(&mut self, key: ChunkKey, mesh: M, display: Display) -> Result<SinkChunk, Refusal<M>>;
    fn replace(&mut self, id: SinkChunk, mesh: M) -> Result<(), Refusal<M>>;   // keeps the display
    fn remove(&mut self, id: SinkChunk);
    fn set_display(&mut self, id: SinkChunk, display: Display);
    fn apply_group(&mut self, group: SinkGroup<M>) -> Result<Vec<SinkChunk>, RefusedGroup<M>>;
    fn is_resident(&self, id: SinkChunk) -> bool;
    fn capacity(&self) -> SinkCapacity;
}
pub struct SinkChunk(pub u64);
pub struct Display { pub walls: WallBits, pub fade: FadeCmd, pub debug: u8 }
/// Bit k = face cp_mesh::FACE_STORAGE_ORDER[k] = +X, +Y, +Z, −X, −Y, −Z; NOT Face::index order.
pub struct WallBits(pub u8);
pub enum FadeRole { Child, Parent }
pub enum FadeRamp { Up, Down }
pub enum FadeCmd {
    Full,
    Hidden,
    Ramp { role: FadeRole, ramp: FadeRamp, start_level: u8, start_ms: u32, duration_ms: u16 },
}
pub struct SinkGroup<M> { pub inserts: Vec<(ChunkKey, M, Display)>, pub replaces: Vec<(SinkChunk, M)>,
                          pub removes: Vec<SinkChunk>, pub displays: Vec<(SinkChunk, Display)> }
pub struct Refusal<M> { pub mesh: M, pub reason: RefusalReason }
pub struct RefusedGroup<M> { pub group: SinkGroup<M>, pub reason: RefusalReason }
pub enum RefusalReason { PoolFull, ItemsFull, Other(String) }
pub struct SinkCapacity { pub pool_free_quads: u64, pub pool_largest_free_quads: u64,
                          pub items_free: u32, pub pending_bytes: u64 }
```

Rules:
- `WallBits` map 1:1 onto `WallMask::from_bits` (`cp-render/src/voxel/layout.rs:162-179`, the same
  storage-slot order as `cp-mesh/src/mesh.rs:13-23`). A cp-app test checks every face.
- `apply_group` is atomic: every member is reserved or none (a refusal hands the group back
  unchanged); its uploads complete in one flush, and its removes and display changes take effect in
  that same flush (§6.5).
- `Display.debug` is the LodLag heat byte (§6.10), written only while that debug view is active.

### 3.5 Streamer

```rust
pub struct Streamer<B: MeshBackend> { /* … */ }
impl<B: MeshBackend> Streamer<B> {
    pub fn new(config: StreamConfig, jobs: JobSpawner, backend: Arc<B>, tables: StreamTables)
        -> Result<Self, StreamError>;
    pub fn add_grid(&mut self, desc: GridDesc, snapshot: &Arc<VoxelSnapshot>) -> Result<(), StreamError>;
    pub fn set_roots(&mut self, grid: GridId, roots: RootSet) -> Result<(), StreamError>;
    pub fn remove_grid(&mut self, grid: GridId, sink: &mut dyn RenderSink<B::Mesh>);
    /// Frame thread, once per frame, before VoxelRenderer::render. Every sink call happens here.
    pub fn update(&mut self, frame: &StreamFrame<'_>, sink: &mut dyn RenderSink<B::Mesh>) -> FrameReport;
    pub fn set_mode(&mut self, mode: StreamMode);
    pub fn is_settled(&self) -> bool;
    pub fn drawn(&self, grid: GridId) -> Vec<DrawnChunk>;            // key, display, resident
    pub fn fading_families(&self) -> Vec<FamilyInfo>;                // id, grid, parent, kind, level, members
    pub fn force_cut(&mut self, family: FamilyId, cut: ForcedCut);   // Deterministic mode only (§8.7)
    pub fn validate(&self) -> Result<(), InvariantError>;            // §7.1 step 10
    pub fn stats(&self) -> &StreamStats;
    pub fn shutdown(&mut self, sink: &mut dyn RenderSink<B::Mesh>);
}
pub enum ForcedCut { Parent, Children }
pub enum StreamError { UnknownGrid(GridId), DuplicateGrid(GridId), InvalidRoots, Disk(String) }
```

`FrameReport` (per frame): frame, select_us, apply_us, walls_us, publish_us, visits, nodes, drawn,
resident, hidden, dormant_resident, in_frustum, open_families, new_transitions, fast_fades,
instant_switches, queued, running, results, stale, sink_ops, upload_bytes, bias, budget_limited,
lag_share, lag_max, settled. `StreamStats` (cumulative): completed, non_empty, void_quick, void_job,
dormant, cancelled, stale, failed, mem_hits, disk_hits, disk_misses, disk_corrupt,
disk_writes_dropped, job latency p50/p99, refine latency p50/p99/max, chunks_per_s (1 s window),
budget_limited_frames.

### 3.6 SimResidency

```rust
pub struct AnchorId(pub u64);
pub struct AnchorDesc { pub id: AnchorId, pub grid: GridId,
                        pub chunks: IAabb /* level-0 chunk positions, half-open */, pub priority: u32 }
pub struct SimResidency { /* … */ }
impl SimResidency {
    pub fn new(jobs: JobSpawner, max_chunks_per_anchor: u32 /* 2048 */) -> Self;
    pub fn set_anchor(&mut self, anchor: AnchorDesc);
    pub fn remove_anchor(&mut self, id: AnchorId);
    pub fn update(&mut self, grids: &[(GridId, &Arc<VoxelSnapshot>)]);   // at tick boundaries
    pub fn record_reads(&mut self, reads: ReadStats);                       // once per tick, sim thread
    pub fn status(&self, id: AnchorId) -> Option<AnchorStatus>;  // wanted, pinned, pending, skipped_empty, truncated
    pub fn stats(&self) -> SimStats;   // generated and waited per tick (last, max, total), blocked_ms
}
```
`SimResidency` is independent of `Streamer`; a dedicated server runs it alone. Behaviour: §7.11.

### 3.7 Governor

```rust
pub struct Governor { /* target_ms, bias */ }
impl Governor {
    pub fn new(target_ms: f64 /* 16.6 */) -> Self;
    pub fn observe(&mut self, gpu_frame_ms: f64) -> f64;   // returns the bias
}
```
Rule (prototype `src/render/lod.rs:581-587`): bias × 1.018 while the GPU frame time exceeds
0.88 · target, × 0.997 while it is below 0.70 · target, clamped to `[1.0, 1.5]`. cp-app feeds
`FrameTimings::gpu_total_ms()` of the latest timed frame (`cp-render/src/frame_graph/timing.rs:39-59`)
and writes the result into the Primary view's `lod_bias`. Off in Deterministic mode (bias 1.0).

### 3.8 Internal seam between WP-D and WP-E
WP-D owns `src/seam.rs` and tests against a mock in `tests/stream/mock.rs`; WP-E implements it in
`src/produce/`.

```rust
pub(crate) trait Produce<M: Clone + Send + 'static> {
    fn set_wanted(&mut self, epoch: u64, wanted: &mut Vec<BuildRequest>);   // replaces the wanted set
    fn request_group(&mut self, group: u64, members: Vec<BuildRequest>);    // Interactive edit group
    fn poll(&mut self, out: &mut Vec<BuildResult<M>>, max: usize);          // never blocks (Realtime)
    fn cached(&mut self, key: ChunkKey, revision: u64) -> Option<CachedMesh<M>>;
    fn keep(&mut self, key: ChunkKey, entry: CachedMesh<M>);
    fn invalidate(&mut self, key: ChunkKey);
    fn defer_drop(&mut self, item: Box<dyn Send>);                          // housekeeping thread
    fn stats(&self) -> ProduceStats;
    fn shutdown(&mut self);
}
pub(crate) enum BuildKind { Family { parent: LodKey }, Single { key: LodKey } }
pub(crate) struct BuildRequest { pub grid: GridId, pub kind: BuildKind, pub snapshot: Arc<VoxelSnapshot>,
    pub revisions: [u64; 8] /* Single: [0] */, pub band: u8, pub key: u32, pub disk: bool,
    pub publish_frame: u64 }
pub(crate) enum BuildOutcome<M> {
    Void { class: NodeClass, subtree_empty: bool },
    Mesh { class: NodeClass, mesh: M, summary: MeshSummary },
    Failed(String),
}
pub(crate) struct BuildResult<M> { pub grid: GridId, pub group: Option<u64>,
    pub items: Vec<(LodKey, u64 /* revision */, BuildOutcome<M>)>, pub ms: BuildMs, pub source: BuildSource }
pub(crate) struct MeshSummary { pub quads: u32, pub regular_quads: u32,
    pub wall_faces: u8 /* WallBits order: faces with uploaded wall quads */, pub bounds: Option<[[u8; 3]; 2]> }
pub(crate) struct CachedMesh<M> { pub revision: u64, pub class: NodeClass, pub mesh: M,
    pub summary: MeshSummary, pub bytes: u64 }
```

---

## 4. Generation at coarse levels and the 41 km test body

### 4.1 Coarse levels are generated directly (existing)
`BaseSource::fill` produces level-L cells directly from the band-limited field at the cell centres,
for any level up to `MAX_LOD_LEVEL` (`cp-voxel/src/base.rs:68-98`); an octave is kept at level L
while its wavelength spans at least `BAND_LIMIT_CELLS = 4` cells (`cp-worldgen/src/field.rs:7-14,
121-132`). Unedited terrain never needs a level-0 pyramid; base chunks are one fill each
(`cp-voxel/src/snapshot.rs:41-59`), edits use the synchronous pyramid (`cp-voxel/src/edit.rs:1-12`).
LOD consistency is the measured rule of contract §5.1.

### 4.2 The cube frame (shared by every cube source)
Input from cp-app, because cp-worldgen may not depend on cp-universe (contract §3.1):
`CubeShape { edge: u32, grid_offset: i32, sea_level: f64 }`, filled from `CubeDef`
(`cp-universe/src/body.rs:84-103`). `R = edge / 2`, `o = grid_offset`, `E` = root extent
(`body.rs:125-130`).

1. **Cells.** Cell `k` of level L has its centre at `c = (k + ½) · 2^L` (exact in f64) and the
   body-relative position `q = c − o` on every axis.
2. **Owner face.** `F = Face::from_dominant_axis(q)`: ties go to the smaller axis, a zero component
   counts as positive (`cp-core/src/face.rs:188-208`). The same rule on every level.
3. **Chart of F** (normal axis n, sign s; tangents `u = (n+1) % 3`, `v = (n+2) % 3`, contract §2.3):
   column `(a, b) = (q_u, q_v)` m; height index `h = k_n` for s = +, `h = −k_n − 1` for s = −;
   height of the cell centre above F's reference plane `t = (h + ½) · 2^L − Z_F` with
   `Z_F = R + o` (s = +) or `R − o` (s = −). `Z_F` is an integer (Heimat: 21,845 and 19,115), so
   every level keeps the grid's own cell lattice, and `t = s · q_n − R` exactly.
4. **Clamped columns.** The column point on the reference cube is
   `P = s·R·e_n + clamp(a, −R, R)·e_u + clamp(b, −R, R)·e_v` (body m): columns beyond an edge use
   the edge point, so two faces evaluate the same P on their common edge and three faces the same P
   at a corner. Edge distance `δ = R − max(|clamp a|, |clamp b|) ≥ 0`.
5. **Point rule.** A cell is ground iff `t < H'_L(P)` (§4.3); otherwise water iff
   `t < sea_level`; otherwise air. Features come on top (§4.5).
6. **Fill.** Partition the region's cells by owner (one face for interior regions, two at edges,
   three at corners). Per owner, evaluate the columns of the owned cells' footprint plus a ring of
   one column (for slopes) column-first in face-local layout (x → a, z → b, y → h, mirrored on
   negative faces), then write the owned cells into the lattice region. Poll `cancel` once per
   column (contract §6 cp-worldgen).

### 4.3 `cubeplanet:cube_hills`: the height field
`H_L(P)`, metres above the reference surface, is a sum of 3D channels of `cp_core::noise::Fbm`
(lacunarity 2, gain 0.5, one salt per channel; `cp-core/src/noise/fbm.rs:88-149, 233`) evaluated at
P (optionally domain-warped), band-limited per level by a new `BandLimitedNoise3` with exactly the
per-level octave arithmetic of the 2D `BandLimitedNoise` (`cp-worldgen/src/field.rs:48-144`, which
only samples `(x, z)`, `field.rs:94`). Defaults; every amplitude is multiplied by `relief_scale`:

| channel | shape | first wavelength | octaves | contribution |
|---|---|---|---|---|
| continents | plain | 8,192 m | 3 | `40·c − 8` m |
| hills | plain, warped | 512 m | 6 | `24·c` m |
| mountains | ridged, warped | 3,072 m | 6 | `900·m·r²` m, `r = (ridged + 1)/2` |
| mountain mask `m` | plain | 9,000 m | 2 | `smoothstep(0.10, 0.45, c)` (≈ 25 % cover (A)) |
| detail | plain | 48 m | 2 | `1.5·c` m |
| warp of P | `domain_warp3` | 1,500 m | 2 | ≤ 300 m, identical on every level |

- `relief_scale = min(1, edge / 40,960)`, set by cp-app: Heimat 1.0, Kiesel 0.134. Small bodies have
  little room: Kiesel has 1,346 m between its faces and its root box (derived from
  `cp-universe/src/body.rs:195-201` and `home.toml:102-119`).
- Analytic per-level bounds `H_lo(L) ≤ H_L ≤ H_hi(L)`; at relief 1 about −74 m and +958 m (A).
- **Land bridge:** `H'_L = H_L + w(δ) · max(0, −H_L)` with `w = 1` for `δ ≤ 128 m`, falling by
  smoothstep to 0 at `δ = 512 m`. A config with `H_lo < −128 m` is rejected.
- **Watertight by construction** (a tested property, §4.9). On the diagonal ownership plane between
  faces X and Y both charts give the same predicate: (i) beyond the edge both evaluate the same
  clamped P and `t_X = t_Y`; (ii) inside the cube within 128 m of the edge `δ = −t` and
  `H' ≥ 0 > t`, so both say ground; (iii) deeper, `t < −128 m ≤ H_lo ≤ H'`, ground again. Corners
  follow pairwise. Water never crosses an edge (`H' ≥ 0` within 128 m). Beyond an edge the ground is
  the square prism `max(q_X, q_Y) − R < H'(edge point)`: edges and corners are exact 90° crests.
- **No settling.** The E2 hills settle heights with two 3×3 median passes (`cp-worldgen/src/terrain.rs:908-981`),
  which would break bit-equality across edges. Slopes come from neighbouring columns of the same
  level and chart (clamped columns beyond edges).

### 4.4 Columns, water, layers
- Layers from the top: grass over dirt (to 4 m) over stone; sand where `|H'| ≤ 2.5 m` next to
  water; gravel floor under water deeper than 8 m; snow above `520 ± 60 m` (3D noise) on slopes
  < 0.8; bare stone or slate on slopes > 1.0. Every input is continuous in 3D at P, so no material
  seam follows an edge. The top ground cell of a column always takes the surface layer (the rule of
  `cp-worldgen/src/lib.rs:23-25`).
- Water: open cells with `t < sea_level` (Heimat 0.0, `home.toml:66`).
- States: `CubePalettes { faces: [TerrainPalette; 6] }` in `Face::index` order. cp-app builds them
  with `Registry::rotated(state, Orientation::from_up(face))` (`cp-content/src/registry.rs:179`,
  `cp-core/src/face/orientation.rs:119`): grass is `up6` (`content/base/blocks/terrain.toml:40-47`),
  logs `axis3`. All six palette key hashes enter the identity.

### 4.5 Features
- **Trees.** A placer of its own per face on a jittered 8 m grid of face-local voxel columns
  (`TREE_GRID`, `cp-worldgen/src/trees.rs:77`); density and species from 3D noise at P; the trunk
  stands on the cube's own level-0 ground. The E2 `TreePlacer` stands trunks on the hills Terrain's
  settled field (`trees.rs:1183`, `terrain.settled_height(0, tx, tz)`) and is not reused. No trees
  within 32 m of an edge (`δ < 32 m`: `TREE_MAX_RADIUS` 6 m + 2^`TREE_CANOPY_MAX_LEVEL` 8 m + margin,
  `trees.rs:80-86`), in water, above the snow line or on slopes > 0.6. Trees are drawn with the
  existing shapes and the `LANDSCAPE_TREES` style (`trees.rs:393`: rasterised to level 1, canopy
  stand-in to level 3) through a face-frame canvas (axis permutation, mirrored height on negative
  faces); only cells owned by the face are written. The private shape constructors (`fn shape`,
  `trees.rs:808`) become `pub(crate)`; output-neutral.
- **Far canopy** (levels 4 to root): columns with wood ≥ 0.45 where trees are allowed rise by
  `12 m · smoothstep(0.45, 0.8, wood)` with the species' leaves on top, so forests stay forests at
  every distance; it agrees with the level-3 canopy within one coarse cell (tested, §4.9).
- **Landmarks** (`landmarks: true`, Heimat only), placed on +Z relative to the start point S:
  towers of 8 × 8 m, heights 64, 96, 128, 192, 256 and 256 m at 0.4, 1, 2, 4, 8 and 12 km along
  +u; fins of 2 × 128 × 96 m at 1.5 and 6 km along +v; a 120 × 120 m plaza 200 m along −u with the
  E1 structure set (pillar, one-thick wall, beam, hollow cube, tree; the geometry of
  `StructureShape`). Exact at level 0; above it a cell is landmark iff the landmark's voxel box
  meets the cell (contract §8 "slightly inflated"). Towers and fins are stone; the structure set is
  logs, as `StructureField` uses (`cp-worldgen/src/structures.rs:259`).

### 4.6 Planning
- **`shape()`:** core = `{‖q‖∞ < R + min_L H_lo(L) − 1 m}`; hull =
  `{‖q‖∞ < R + max(H_hi + 26 m, landmark tops, sea_level) + 1 m}` (26 m covers trees, canopy and
  margin); both rounded to whole voxels conservatively.
- **`height_interval(level, face, a, b) → (lo, hi)`**, conservative, O(octaves): per channel the
  value at the footprint centre ± `min(range, G · radius · frequency)`, with a Lipschitz constant G
  of the octave noise (warp included) derived from the noise construction and pinned by a sampled
  test with a safety factor of 1.5 (A); plus the full amplitude of the dropped octaves; then the
  land bridge and the mask product in interval arithmetic.
- **`classify`:** (1) the shape; (2) the interval: Empty if every owner's cell centres lie above
  `hi` plus feature tops; Opaque if every centre lies below `lo − 8 m` and no water or shallows can
  occur; (3) for L ≤ 5 an exact column scan with early exit, as the hills scan, which stops at the
  first column that proves Mixed (`cp-worldgen/src/hills.rs:284-304`); (4) otherwise Mixed.
- **`subtree_empty`:** the hull; then the level-0 interval (all octaves) plus feature tops; else
  false.
- **`fill` fast paths** (the trait's cost rule, `cp-voxel/src/base.rs:87-91`): per region and owner,
  all air when every cell centre lies above the interval's `hi` plus features, all stone when every
  centre lies below `lo − 8 m`. They make the air and deep families of the family path nearly free.

### 4.7 API, identity, validation

```rust
pub struct CubeShape { pub edge: u32, pub grid_offset: i32, pub sea_level: f64 }
#[derive(Clone, Debug, PartialEq, serde::Deserialize)]
pub struct CubeHillsConfig { /* relief_scale 1.0; the channels of §4.3; land_bridge_inner_m 128.0,
    land_bridge_outer_m 512.0; snow_line_m 520.0; tree_density 0.9; tree_edge_margin_m 32.0;
    far_canopy true; landmarks false */ }
pub struct CubePalettes { pub faces: [TerrainPalette; 6] }
pub struct CubeHillsSource { /* … */ }
impl CubeHillsSource {
    pub const KEY: &'static str = "cubeplanet:cube_hills";
    pub const VERSION: u32 = 1;
    pub fn new(seed: u64, shape: CubeShape, config: CubeHillsConfig, palettes: CubePalettes,
               flags: &[BlockFlags]) -> Result<Self, WorldgenError>;
    pub fn surface_height(&self, face: Face, a: f64, b: f64) -> f64;   // level 0, H' in m
    pub fn face_of(&self, lattice: DVec3) -> Face;
    pub fn start_point(&self) -> (Face, f64, f64);  // land column nearest the +Z centre with
                                                    // 4 m ≤ H' ≤ 40 m and slope < 0.25
    pub fn tint(&self, face: Face, a: f64, b: f64) -> [u8; 4];         // ground-map channels (§6.8)
    pub fn landmarks(&self) -> &[Landmark];
    pub fn height_interval(&self, level: u8, face: Face, a: [f64; 2], b: [f64; 2]) -> (f64, f64);
}
impl BaseSource for CubeHillsSource { /* fill, classify, subtree_empty, identity, shape */ }
```

- **Identity:** `{ key, VERSION, seed = CubeDef.generator_seed, macro_hash }`; `macro_hash` =
  `ConfigHasher` over edge, `grid_offset`, `sea_level.to_bits()`, every config field in declaration
  order and the six palette key hashes (current practice for parameters,
  `cp-worldgen/src/lib.rs:38-43`). The disk key uses `stable_hash` (`cp-voxel/src/base.rs:36-48`).
- **Validation** (`WorldgenError::InvalidConfig`): `H_lo < −128 m`; or a hull that does not fit
  the root set, `R + |o| + hull top ≥ E`. Today the fit check of cp-universe covers the atmosphere
  only (`cp-universe/src/system.rs:603-616`).

### 4.8 `cubeplanet:cube_flat` (VERSION 1)
Every face is flat at `H = 3.5 m` (grass top cell, dirt to 4 m, stone below), with no water, no
features and the frame and ownership of §4.2. `classify`, `subtree_empty` and `shape` are exact; the
point-rule surface of every level is analytic (the boundary above the last cell centre below
`Z_F + 3.5`). It validates the pop probe (§8.7).

### 4.9 Tests (WP-B)
- **Goldens:** canonical hashes (contract §2.5 rule 5) at 14 points (6 face centres, 3 edge
  midpoints, 2 corners, S, the top of the 256 m tower, a sea chunk) on levels 0..=10 for Heimat's
  shape and seed 14159; non-uniform content asserted on levels 0..=8; `stable_hash` pinned. The plane
  helper that demands a non-uniform chunk on every level 0..=20
  (`cp-worldgen/tests/worldgen/golden.rs:255-282`) does not apply to bounded bodies.
- **LOD consistency** per region type (face interior, edge, corner), levels 1..=10, seeds 14159 and
  42, at least 14 parents each: surface offset ≤ 1 coarse cell along the owner normal, differing
  cells ≤ 7 %, class mismatch ≤ 3.5 % worst and ≤ 1.2 % mean (the plane limits,
  `cp-worldgen/tests/worldgen/lod.rs:46-50`); the far canopy at the level-3/4 border included.
- **Seams:** `H'` bit-equal from both owners at 10,000 random edge points on every level; equal from
  all three at the 8 corners; 0 enclosed air pockets (flood fill from above the hull) in 24 boxes
  straddling edges and corners on levels 0–6; no water cell in a wedge beyond an edge.
- **Conservative planning:** `classify`, `subtree_empty`, `height_interval` and `shape` never
  contradict `fill` on 2,000 random keys (edges, corners, core, outside).
- **Features** independent of order, threads and the region asked for; a family region (66³)
  sliced equals eight padded fills.
- **Timing** (release, `#[ignore]`, run in WP-B's first three days): padded fill ≤ 3.0 ms mean and
  ≤ 6.0 ms p99 over 300 surface chunks of levels 0–6; family fill ≤ 12 ms mean (≤ 1.5 ms per child);
  meshing ≤ 0.2 ms mean.
- The existing flat, hills and structure-field goldens stay green.

### 4.10 Generator factory and content
- cp-app `generators.rs`: `base_source(body: &BodyDef, registry: &Registry, flags: &[BlockFlags])
  -> Result<Arc<dyn BaseSource>, Failure>` maps `CubeDef.generator` (`cp-universe/src/body.rs:98-102`;
  cp-universe checks only the key syntax, `system.rs:600-601`): `cubeplanet:cube_hills` →
  `CubeHillsSource` (seed `generator_seed`, `relief_scale = min(1, edge/40,960)`, landmarks for
  `cubeplanet:heimat` only, six rotated palettes), `cubeplanet:cube_flat` → `CubeFlatSource`. A plane
  generator (`cubeplanet:flat`, `cubeplanet:hills`, `cubeplanet:structure_field`) on a cube body, or
  an unknown key, is an error.
- `content/base/systems/home.toml`: the eight cube bodies change `generator = "cubeplanet:hills"`
  (`home.toml:74, 119, 169, 201, 233, 264, 295, 326`) to `"cubeplanet:cube_hills"`; the comment at
  `home.toml:29-30` is updated. The E2 scenes keep building `HillsSource` directly.

---

## 5. Meshing at LOD (walls, aprons, cache keys)

### 5.1 Inputs and purity
One padded 34³ chunk at the chunk's own level is the mesher input (`cp-mesh/src/lib.rs:38-51`):
face-halo cells are the same-level neighbours' cells, edge and corner halo cells are never read
(`cp-voxel/src/padded.rs:29-34`). A mesh therefore never depends on a neighbour's level; an LOD
change only changes displays and wall masks (I3). Each worker thread keeps one `Mesher` and one
`ChunkMesh` (`thread_local`, as `cp-app/src/scene/build.rs:32`; `Mesher::mesh_into` reuses
allocations, `cp-mesh/src/mesher.rs:238-274`).

### 5.2 Single and family sampling (aprons)
- **Single** (roots, merge parents, edit re-meshes, families with at most 2 children to build):
  `classify` first (cheap for surface chunks, whose scan stops at the first column that proves
  Mixed; for Empty chunks it replaces the fill), Empty → Void; otherwise `try_sample_padded_with`
  with the worker's buffer, then mesh.
- **Family** (every other split): children whose quick class is Empty or whose
  `quick_subtree_empty` holds need nothing; if all eight are quick-void no job runs. Otherwise
  `try_sample_padded_family` (§3.2) gives all eight padded children from one fill; each child's class
  comes from its interior cells (all air → Empty, all `OPAQUE` → Opaque, else Mixed). The family
  path is exact with edits too, because the overlays stay cp-voxel's.
- **Cost** (derived, (A)): a family evaluates `(66 + 2)² = 4,624` columns instead of
  `8 · (34 + 2)² = 10,368` for eight padded fills. At about 1 µs per 3D column (A) and about four
  non-empty children per surface family this is about 1.5 ms of worker time per non-empty chunk
  (§7.13).

### 5.3 Upload policy and wall masks
- **Uploaded:** the six regular ranges of every layer and the six wall ranges of the opaque and
  fluid layers. Cutout wall ranges are never uploaded: a leaf sheet on a chunk border inside a crown
  reads as an artefact, a gap in porous foliage does not. The cutout layer exists on levels 0–1
  only (`cp-mesh/src/mesh.rs:9-11`). Today cp-app drops every wall range outside the requested mask
  (`upload_parts`, `cp-app/src/scene/build.rs:262-282`); streamed chunks use this policy instead, so
  a mask change is an item rebuild in the same frame (`set_wall_mask`,
  `cp-render/src/voxel/state.rs:548-552`), never a re-upload. The E2 scenes keep `upload_parts`.
- **Exact mask rule (I10).** Bit f (WallBits order) of a drawn chunk C at level L is ON iff
  1. C's uploaded mesh has wall quads on f, and
  2. some node M of the cut (drawn, or void) shares a face area of positive measure with C across
     f — C's same-level neighbour `K = (L, C.pos + f.normal())`, K's ancestor in the cut, or K's
     descendants in the cut on the shared face — such that
     (a) `level(M) ≠ L`, (b) M's class is not Opaque, and (c) M's display coexists with C's: they
     are not the parent and a child of one fading family. A void node counts with the role of the
     family it belongs to, and as Full otherwise.

  Otherwise the bit is OFF. Space outside the root set holds no node and never turns a bit on (the
  hull fits inside the root set, §4.7).
- **Why exact.** An unneeded wall is not always hidden: the vertex shader moves every quad corner
  0.05 px outward along both edges (`EDGE_OVERLAP_PX`, `cp-render/src/voxel/layout.rs:86-94`;
  `edge_push` in gbuffer.wgsl `vs_voxel`), so a wall between two same-level chunks can poke a sliver
  above their shared top faces; a missing wall is a crack. The rule is watertight for any level
  difference: where C is solid and M is not, C's wall closes the boundary; where M is solid and C is
  not, M's wall does. Same-level neighbours never need walls, because C's halo holds their cells.
- **Edits need no wall case.** All dirty chunks of one receipt, including the face neighbours whose
  halo changed (`cp-voxel/src/edit.rs:64-74`), are replaced in one group (§7.8), so no drawn pair
  ever has mismatched halos. The v3 clause "or the face between an edited and an unedited chunk of
  the same level" (`ARCHITEKTUR.md:466-467`, also the `WallMask` doc, `layout.rs:162-164`) is dropped
  (§12).
- **Recomputation:** in the frame of every change, for each drawn chunk face-adjacent to a node
  whose display, class, drawn state or residency changed. `validate()` compares every mask with a
  brute-force recomputation.

### 5.4 Mesh outcome classes
| class | condition | treatment |
|---|---|---|
| Empty | no non-air interior cell | void; may split unless subtree-empty |
| Void | cells, but 0 uploaded quads | void; may split |
| Dormant | class Opaque and 0 regular quads with ≥ 1 uploaded wall quad; or quick-Opaque from `shape.core` with the chunk widened by one cell inside the core and not edited (its mesh is built lazily) | resident only while its mask is non-empty; readiness waits for that residency (§7.6) |
| Meshed | ≥ 1 regular quad | resident before it is revealed |
| Failed | `MeshError` or a contained panic | treated as Void; retried after 1 s, doubling to 30 s; must be 0 in the gates |

`NodeClass::Opaque` alone never makes a chunk ready: an Opaque chunk has regular faces wherever a
same-level neighbour's border cell is not opaque, and wall faces at LOD borders
(`cp-voxel/src/base.rs:51-60`, `cp-mesh/src/mesher.rs:305-338`); E2 skips only Empty chunks for the
same reason (`cp-app/src/scene/build.rs:214-225`).

### 5.5 Mesh codec (cp-mesh, WP-A)
```rust
pub const MESH_CODEC_VERSION: u32 = 1;
impl ChunkMesh {
    pub fn encode(&self, out: &mut Vec<u8>);
    pub fn decode(bytes: &[u8]) -> Result<ChunkMesh, MeshDecodeError>;
}
```
Layout (little-endian): `MESH_FORMAT_VERSION` u32 (3, `cp-mesh/src/lib.rs:127`),
`MESH_CODEC_VERSION` u32, quad count u32, the quads as `(w0, w1)` u32 pairs, then per layer `first`
u32, `counts [u32; 12]` and three boxes (present u8 + min `[u8; 3]` + max `[u8; 3]`). `decode`
checks both versions, `PackedQuad::validate` for every quad, each quad's face against its range
slot, the counts against the quad count and contiguity, and every box against the box recomputed
from the quads. Any mismatch is an error, never a panic. `MeshDecodeError` implements
`std::error::Error` by hand (cp-mesh depends only on cp-core and bytemuck, `cp-mesh/Cargo.toml`).

### 5.6 Cache keys
- **Memory:** `(ChunkKey, revision)` (`cp-voxel/src/snapshot.rs:155-166`).
- **Disk** (levels ≥ 3; levels 0–2 are always rebuilt, contract §6): a 64-byte key, stored and
  compared verbatim, little-endian:

| bytes | field |
|---|---|
| 0..8 | `GeneratorIdentity::stable_hash` (`cp-voxel/src/base.rs:36-48`) |
| 8..16 | grid salt = `hash_str(body key)` |
| 16 | level u8; bytes 17..20 zero |
| 20..32 | pos x, y, z as i32 |
| 32..40 | `content_hash` (`Registry::content_hash`, `cp-content/src/registry.rs:315`) |
| 40..44 | `MESH_FORMAT_VERSION` |
| 44..48 | `MESHER_VERSION` (`cp-mesh/src/lib.rs:133`) |
| 48..52 | `APRON_STAMP_VERSION` (`cp-voxel/src/stamp.rs:20-22`) |
| 52..56 | `MESH_CODEC_VERSION` |
| 56..64 | `apron_stamp(key, state_keys)` (`cp-voxel/src/stamp.rs:34-124`) |

  `revision()` is session-local and never enters it (`cp-voxel/src/snapshot.rs:160-162`). The value
  is `ChunkMesh::encode`. From E6 the apron stamp also covers stamps (an `APRON_STAMP_VERSION`
  bump); the layout stays. Grids without a salt never use the disk cache.

---

## 6. Render integration (cp-render, WP-F)

### 6.1 Compatibility rules
- E4 and P1 build in parallel and existing struct literals must keep compiling: no new field in
  `FrameInput` (literals at `cp-app/src/bench.rs:367`, `run.rs:489`, `shot.rs:143` and in cp-render
  tests), `Lighting` (`cp-app/src/scene/sky.rs:169`), `ObjectDesc` (`cp-app/src/run.rs:657`,
  `stage.rs:154`, `scene/structures.rs:111`) or `VoxelRendererConfig` (cp-render tests). New inputs
  are setters on `VoxelRenderer` (per frame) or `VoxelScene` (per object and chunk), new enums and
  new structs.
- `FrameStats`, `SceneStats`, `UploadReport` and `VisibleCounts` may gain fields: only cp-render
  constructs them (`cp-render/src/voxel/renderer.rs:790-796`, `state.rs:792-805`, `stats.rs`).
- Defaults keep every E2 image: legacy up mode, `Fade::FULL`, space 0, cloud opacity 1, TAA cut
  distance 50 m, plane ground map, Hi-Z off until gate A8 decides.
- `DebugView` gains variants 8–13; cp-app uses only `DebugView::ALL` and `from_id`
  (`cp-app/src/cli.rs:597, 813-816, 1120`).
- Every pipeline is created at start-up; the total stays below 64 (11 today,
  `cp-render/src/voxel/pipelines.rs:49`).

### 6.2 Draw item and frame words (frozen on day 0)
`GpuDrawItem`, 64 bytes (`cp-render/src/voxel/layout.rs:371-400`), mirrored in `common.wgsl`; the
layout tests are extended:

| word | bits | meaning |
|---|---|---|
| `lod_flags` | 0–4, 5–6, 8–13 | level, layer, wall mask (unchanged) |
| `lod_flags` | 14–16 | up mode: 0 legacy +Y, 1–6 = `Face::index() + 1`, 7 per cell |
| `lod_flags` | 7, 17–31 | 0 (reserved) |
| `fade` | 0–4 | start level `s0`, 0..=16 |
| `fade` | 5 | role: 0 child, 1 parent |
| `fade` | 6–7 | ramp: 0 static, 1 up, 2 down |
| `fade` | 8–15 | debug byte (LodLag heat); 0 unless that view is on |
| `fade` | 16–31 | ramp duration in ms (0 when static) |
| `reserved[0]` | 0–31 | ramp start, frame clock ms (u32, wrapping) |
| `reserved[1]` | 0–31 | resident since, frame clock ms (written by cp-render when the mesh completes) |

Today `fade` holds coverage in bits 0–7 and the complement in bit 8 (`layout.rs:286-291`,
`common.wgsl:22-23`), and `reserved` is zero (`layout.rs:396-397`).

`GpuObject` (224 B, `layout.rs:478-506`): `reserved` (`layout.rs:502-503`) = the lattice centre `o`
as i32 bits for planet objects, 0 otherwise; `flags` stays reserved for velocity.

`GpuFrame` (464 B, `layout.rs:508-567`): `scratch[3]` (zero today, `layout.rs:540-542`) = frame clock
ms; `sun_color.w` (zero, `layout.rs:522-523`) = cloud opacity (default 1.0); `sky_sunset.w` (zero,
`layout.rs:549-550`) = space factor; `mode[3]` (zero, `layout.rs:543-544`) = ground-map mode (0 plane,
1 faces); `ground_map` (`layout.rs:558-560`) in face mode = `(R, 1/(2R), texel m, side texels)`.

Counter buffer: `COUNTER_WORDS` 4 → 8 (`layout.rs:119-124`): [0] early class-0 draws, [1] early
class-1 draws, [2] visible quads (`COUNTER_QUADS` stays 2), [3] in-frustum items, [4] late class-0
draws, [5] late class-1 draws, [6] Hi-Z-culled items, [7] reserved. The clear and the 4-slot
read-back ring (`cp-render/src/voxel/stats.rs:69`) grow with it.

### 6.3 Fades: GPU-timed integer ramps on the 16 dither thresholds
- **Threshold** `b = dither_threshold(pixel) ∈ 0..15`: the existing 4 × 4 Bayer matrix, rotated by
  the frame index while TAA accumulates and fixed otherwise (gbuffer.wgsl `dither_threshold`). Its
  16-frame rotation matches the 16 jitter phases (`cp-render/src/voxel/taa.rs:23`), so TAA averages
  the pattern into a cross-dissolve.
- **Level** of an item at frame clock T: static → `s0`; ramp: `elapsed = T − start` (wrapping u32;
  values ≥ 2^31 count as 0), `steps = min(16, ⌊16 · min(elapsed, duration) / duration⌋)`
  (duration 0 → 16); up → `min(16, s0 + steps)`; down → `s0 − min(s0, steps)`.
- **Keep rule:** a child keeps `b < level`, a parent keeps `b ≥ level`. FULL = child, static, 16.
  HIDDEN = child, static, 0 (a parent at static 16 is hidden too).
- **Exactness:** parent and children of a family carry identical `(s0, ramp, start, duration)` and
  opposite roles, so every threshold belongs to exactly one of them in every frame, in integers.
  Pairs built today with `Fade::new(c)` and `Fade::new(1 − c)` are not exact: it rounds `c · 255`
  half away from zero (`layout.rs:242-263`), and `c = 7.5/255` gives 8 and 248, which both keep
  threshold 0 under the current rule `t = (b + 0.5)/16` (gbuffer.wgsl `fade_keeps`).
- **Families:** split → children `Ramp { Child, Up, 0, t0, d }`, parent `Ramp { Parent, Up, 0, t0, d }`;
  merge → both `Down` from 16. Reversal at clock T: `s0 = level(T)`, opposite ramp, `start = T`,
  written for every member in one frame. End: in the frame where the level reaches 16 (up) or 0
  (down), the CPU sets the survivors FULL and removes the others.
- **CPU cost:** two writes per member per transition, instead of an item rebuild per frame
  (`set_fade` → `rebuild_items`, `state.rs:554-558, 736-759`).
- The vertex shader evaluates the item's level once and passes `(level, role)` flat; the fragment
  shader compares integers.
- **Shadow views (E5):** a member is drawn into a shadow map iff it keeps threshold 8 (child: level
  > 8; parent: level ≤ 8): exact, without dither (contract §6 "shadow passes switch without dither").
- **Rust API** (`layout.rs`): `FadeRole`, `FadeRamp`; `Fade::FULL`, `Fade::HIDDEN`;
  `Fade::level(role, level) -> Fade`; `Fade::ramp(role, ramp, start_level, start_ms, duration_ms)
  -> Fade`; `Fade::level_at(self, clock_ms: u32) -> u8`; `Fade::keeps(self, threshold: u8,
  clock_ms: u32) -> bool`; `Fade::pack(self) -> [u32; 2]` (fade word, start). `Fade::new(coverage,
  complement)` stays as a constructor: static level `round(16c)` as child, or `16 − round(16c)` as
  parent.
- `VoxelRenderer::set_clock_ms(u32)` sets the frame clock before `render` (default 0).

### 6.4 Draw classes and hidden items
An item belongs to draw class 1 (the cutout pipeline, which can discard) iff it is in the cutout
layer, or its ramp is not static, or it is static and not full (today: coverage ≠ 255,
`common.wgsl:121-125`, `layout.rs:438-448`). Items that keep no threshold at the frame clock produce
no draws on paths A, B and direct (GPU `slot_masks`, `common.wgsl:213-235`; CPU mirror
`cp-render/src/voxel/cull.rs:76-103`): hidden children cost no rasterisation.

### 6.5 Upload lanes, groups, item writes
- `VoxelScene::insert_prepared_in(chunk, Lane)` with `Lane::{Interactive, Streaming}`
  (`insert_prepared` = Streaming lane) and `VoxelScene::apply_group(group: UploadGroup)
  -> Result<Vec<ChunkId>, RefusedGroup>` with inserts (prepared chunk + walls, fade, up mode),
  replaces, removes and display changes: every reservation or none; the group's uploads complete in
  one flush whatever the budget; its removes and display changes apply in that flush.
- `take_uploads` (`state.rs:616-655`: FIFO, at least one mesh per call) serves the Interactive lane
  first, never splits a group, and completes at least one group or chunk per flush.
- `take_dirty_items` (`state.rs:765-786`) merges runs separated by at most 16 clean items;
  `write_items` (`cp-render/src/voxel/scene.rs:656-667`, one `write_buffer` per run) then writes
  fewer, larger ranges.
- New setters, stored on the chunk and baked into its items when its upload completes, like walls
  and fade today (`state.rs:461-469, 658-697`): `set_up_mode(id, UpMode)`, `set_debug_byte(id, u8)`;
  per object `set_object_lattice_centre(object, i32)`.

### 6.6 Hi-Z two-phase culling (path A; `ARCHITEKTUR.md:474-475`)
- **Pyramid:** a persistent `R32Float` texture outside the transient pool (the pattern of the TAA
  history, `cp-render/src/voxel/taa.rs:137`), recreated on resize; mip 0 = ⌈W/2⌉ × ⌈H/2⌉
  (960 × 540 at 1080p), each texel the min of 2 × 2 depth texels (3 at odd borders), mips down to
  1 × 1, one dispatch per mip through its own single-mip storage view (16 storage textures per
  stage are already enabled, `cp-render/src/gpu/limits.rs:47`).
- **Passes** in declared order, each at most once per frame (`cp-render/src/frame_graph/mod.rs:11-13`):
  `cull` (early) → `gbuffer` (early) → `hiz` → `cull_late` → `gbuffer_late` (`LoadOp::Load`) →
  overdraw → lighting → taa → composite → bounds → overlay. Phase 2 has its own argument, bucket and
  count regions.
- **History:** one u32 per item index. Phase 1 draws the in-frustum items whose history says
  visible; phase 2 tests every in-frustum item not drawn in phase 1 against the pyramid, draws the
  visible ones and writes the history of every in-frustum item. Item indices move on swap-remove
  (`state.rs:712-733`); a stale entry only moves an item between the phases, because phase 2 tests
  everything phase 1 skipped against the depth phase 1 wrote.
- **Box test:** project the 8 corners with `local_to_clip`; any corner with `w ≤ Z_NEAR` → visible;
  otherwise the screen rectangle, dilated by one texel, selects the mip at which it spans at most
  2 texels; the item is occluded iff `d_near · (1 + 1e-4) < min(those texels)` (reversed Z, contract
  §2.4). Dithered and alpha-tested pixels leave far depth behind, which keeps the test conservative.
- After a resize, a TAA cut or a debug-view change, and on the first frame, every history entry is
  reset to visible.
- **Paths B and direct** cull by frustum only. The v3 "optional occlusion from last frame's read-back
  Hi-Z" (`ARCHITEKTUR.md:461`) is declined in E3: a one-phase test against old depth drops newly
  visible chunks for frames.
- **Default:** `VoxelRenderer::set_hiz(bool)`, off until gate A8 shows GPU P95 with Hi-Z at most the
  P95 without it on every E3 route; then the integrator flips the default in `renderer.rs`. Two
  phases double the count-draw cost per `max_count` slot (about 7.5 ns per slot on the GTX 1080,
  `renderer.rs:638-650`).
- The Holes view and every probe run with Hi-Z off.

### 6.7 Precision
- **Defect:** `vs_voxel` passes the absolute grid position as f32 (`out.grid = vec3<f32>(grid)`,
  gbuffer.wgsl line 215 at reading). After interpolation, floor-based cell hashes
  (`floor(inside)`, line 540) and the 1/16 m ground-cover texels work with an f32 spacing of 1.95 mm
  for `|x| ∈ [16,384, 32,768)` m — Heimat's side faces and every point near its edges (lattice
  coordinates up to ±21,845) — while one pixel at 1 m covers about 1.07 mm at 1080p and 60°. Cells
  near edges flicker with the TAA jitter. This breaks "absolute positions never enter an f32"
  (`ARCHITEKTUR.md:126`).
- **Fix:** the vertex shader outputs a flat `vec3<i32>` base = item origin + quad min cell · cell
  size and a `vec3<f32>` offset from it (at most `32 · 2^L` m, the edge push included). Every floor,
  cell hash, texel index and owner-face test uses integer arithmetic on the base plus the floor of
  the offset; smooth noise inputs use `f32(base − o) + offset` (o = the object's lattice centre,
  magnitudes ≤ 2^15 m). Vertex positions stay integer offsets from the anchor (gbuffer.wgsl
  `let local = (item.origin - object.anchor) + lattice * cell_size`, line 145;
  `cp-render/src/voxel/camera.rs:279-302`). E2 scenes (o = 0, small coordinates) render unchanged
  (gate A18).

### 6.8 Six-face shading and face ground maps
- **Today** the G-buffer pass assumes a +Y ground plane: `land_at`, `tint_variation` and
  `foliage_shift` take `v.grid.xz` (gbuffer.wgsl lines 527, 532, 536), ground cover is laid only on
  face 2 (+Y, line 566), and the grass fringe goes on every face whose axis is not Y
  (`(face >> 1u) != 1u`, line 573). On the +Z start face grass tops would get the side fringe and
  ground cover would land on cliffs.
- **Up mode** per item (`lod_flags` 14–16): Legacy (the E2 behaviour), `Face(F)`, or `PerCell`: up
  face = `from_dominant_axis(cell − o)` in integers with the tie rule of contract §2.3. The plane
  coordinates are the up face's two tangent axes in ascending order: `(x, z)` for ±Y (E2
  unchanged), `(y, z)` for ±X, `(x, y)` for ±Z. Ground cover goes only on faces equal to the up
  face, the grass fringe only on faces whose axis differs from the up axis.
- cp-app picks the mode per chunk: `Face(owner)` when all 8 corners of the chunk box, widened by one
  cell, have the same owner; otherwise `PerCell`.
- **Face ground maps:** `FaceGroundMaps { half_edge_m: f64, texel_m: f64 /* 64 */, size: u32 /* 640 */,
  texels: [Vec<[u8; 4]>; 6], relief: [Vec<[u8; 4]>; 6] /* may be empty */ }`, with the channels of
  `GroundMap` (`cp-render/src/voxel/ground.rs:19-45`), one layer per face, texel (i, j) covering
  face-local a, b from −R. `VoxelRenderer::set_face_ground_maps(gpu, &maps)` binds them as a
  `texture_2d_array` (the plane `GroundMap` becomes a one-layer array; mode in `GpuFrame.mode[3]`).
  Planet items sample layer = up face at `uv = ((a + R)/(2R), (b + R)/(2R))`. 6 × 640² RGBA8 =
  9.8 MB, plus relief. E3 requires the tint channels; relief on faces is optional (beauty loop).

### 6.9 Interim orbit lighting and TAA cut (until E5)
- `VoxelRenderer::set_space(space: f32, cloud_opacity: f32)` → `sky_sunset.w` and `sun_color.w`:
  sky, horizon band, clouds and fog scale by `1 − space`, the hemisphere sky light by
  `1 − 0.8 · space`, the clouds by `cloud_opacity`; the sun disc stays. Today the cloud layer sits a
  fixed `CLOUD_HEIGHT = 1600 m` above the camera (`lighting.wgsl:246`) and fog is measured along one
  up direction (`cp-render/src/voxel/camera.rs:104-150`).
- `VoxelRenderer::set_taa_cut_distance(m: f64)` replaces the constant in `is_cut`
  (`CUT_DISTANCE = 50 m`, `taa.rs:39, 114-120`); cp-app passes `max(50 m, 0.1 · altitude)`. With the
  free-fly speed `10 · (1 + h/10)` m/s (`cp-app/src/run.rs:47-49, 548`) every frame above about 3 km
  is a cut today, which leaves fades as raw dither.

### 6.10 Debug views and probe support
Appended to `DebugView` (`cp-render/src/debug_view.rs:18-49`), stable numbers:

| # | view | shows |
|---|---|---|
| 8 | Fade | children green, parents magenta, brightness = level/16; static items white |
| 9 | Residency | age since resident (`reserved[1]`): < 0.25 s red, < 1 s yellow, then the LOD colour dimmed |
| 10 | Walls | wall quads orange, regular quads grey |
| 11 | Occlusion | phase-2 draws cyan; the overlay shows the Hi-Z-culled count |
| 12 | LodLag | debug byte = `min(255, 64 · e_eye)`: green ≤ 64, yellow ≤ 80, red ≥ 96 |
| 13 | Holes | path A, Hi-Z off: every opaque and fluid item goes through the two-sided pipeline; the GPU cull and its CPU mirror keep their back-facing slots (`two_sided` = cutout layer ∨ Holes view; `common.wgsl:226`, `cull.rs:91`); back-facing quads get no edge push; a back face that wins the depth test is written pure magenta `(1, 0, 1)`, everything else as unlit albedo |

LodLevel (4) hatches fading items. Depth read-back for the probes:
`VoxelRenderer::request_depth_readback()` before `render`, then `take_depth(gpu) ->
Result<Option<DepthImage { width, height, depth: Vec<f32> }>, RenderError>` (blocking, tools
only); the depth target gains `COPY_SRC`. `HeadlessTarget` reads colour only today
(`cp-render/src/headless.rs:209-282`).

### 6.11 Counters, stats, memory report
`FrameStats` gains early and late draws, `in_frustum_items`, `hiz_culled_items`, `upload_ms` (CPU
time of the flush and the item writes) and `item_write_bytes`. `VoxelRenderer::memory() ->
MemoryReport { pool, items, objects, targets, history, hiz, textures, ground_maps, staging, total }`
in bytes, from the created resources.

### 6.12 Quad pool
Unchanged: 128 / 96 / 32 MiB by tier, units of 256 quads, never reallocated
(`renderer.rs:132-160`, `scene.rs:727-731`). Estimate (A): 5,500 resident × about 450 quads (walls
about 15 % (A)) ≈ 2.5 M quads ≈ 19 MiB = 15 % of 128 MiB; fades add at most 128 × 9 chunks.
cp-stream's admission keeps 10 % of the pool free (§7.5), so the Low tier (32 MiB) is pool-limited
rather than refused.

---

## 7. Scheduling, budgets, caches (memory, disk), eviction

### 7.1 The frame-thread update
`Streamer::update` runs these steps in order:
1. Receipts of every grid, in order (§7.8).
2. Drain results: at most 128 or 0.25 ms. A result is accepted iff its node still wants that key at
   that revision (`snapshot.revision(key)` equal); otherwise it goes into the memory cache under its
   own revision (stale).
3. Residency feedback: poll `is_resident` for nodes waiting for it; a refusal sends the mesh to the
   memory cache, sets `budget_limited` and retries next frame.
4. Incremental walk, at most 6,000 visits (§7.5).
5. Admission and forced merges (§7.5).
6. Transitions: finish ended ramps, start ready families (at most 32 per frame and 128 open),
   reversals and cancellations (§7.6).
7. Wall masks of dirty nodes (§5.3).
8. Sink operations, at most 512; new inserts at most 4 MiB of meshes per frame, families closest to
   completion first.
9. Publish requests (§7.7).
10. Statistics; `validate()` in tests, in Deterministic mode and in `--check` runs: coverage (I9),
    drawn ⇒ ready, masks (I10), at most one open family per node with identical ramp parameters,
    caps (I7), no drawn chunk removed while it keeps a threshold.

Budgets (P99, contract §7, `ARCHITEKTUR.md:584-587`): selection (steps 4, 5, 9) ≤ 0.5 ms; draw items
and uploads (steps 2, 3, 6, 7, 8 plus the scene flush and item writes in `render`) ≤ 0.8 ms. No
allocation or drop above 64 KiB on the frame thread: old snapshots, freed subtrees, evicted cache
entries and cancelled requests go to the housekeeping thread (§7.7).

### 7.2 LOD metric
For node N (level L, voxel box `B_N` in lattice m) and every view v with `draw = true` for N's grid:
- `f_v = min(H_v, 1080) / 2 / tan(fov_y / 2)` (`det::tan`): 935.3 px at 1080p and 60°, 30,940 px for
  a 2° telescope. The cap at 1,080 keeps the budgets independent of the window size.
- `d_v = max(0.5 m, min(dist(eye_v, B_N), dist(eye_v + clamp_len(velocity_v · 0.5 s, 2,000 m), B_N)))`,
  Euclidean distance to the closed box, 0 inside.
- `inside_v(N)`: Frustum views — `B_N` meets the view's frustum widened by 5 %; Primary and Omni —
  always.
- `σ_v(N) = 1.5` when N is in the cube-edge silhouette band or is a skyline node (below), else 1;
  Primary and Omni views only.
- Bias: `b = clamp(max(view lod_bias, pressure bias), 1.0, 1.5)`; for L = 1,
  `b₁ = min(b, 2 · f_v / (240 m · τ))` (1.299 at τ = 6 and 60°), so the level-0/1 split stays ≥ 240 m
  and its merge ≥ 276 m: no LOD fade below 240 m, where GTAO, contact shadows and SSR run
  (`ARCHITEKTUR.md:593`).
- `e_v(N) = 2^L · f_v · σ_v(N) / (τ · b · d_v)` if `inside_v(N)`, else 0; `e(N) = max_v e_v(N)`.
- Split if `e(N) > 1` (L ≥ 1); merge N's children if `e(N) < 1/1.15`; otherwise hold.
- **Cube-edge silhouette band:** with the grid's `CubeFrame` (centre c, half edge R) and eye e, a
  face F is eye-side iff `s · (e_n − c_n) > R`; a silhouette edge joins an eye-side and a
  non-eye-side face; N is in the band iff `B_N` meets the box of half-width `max(4 · 2^L, 256)` m
  around such an edge segment. From the ground these are the four edges of the current face, from
  orbit the limb.
- **Skyline nodes** (eye altitude above the reference surface below 5 km): every 8th frame, 512
  azimuth bins around the view's `up` collect the highest elevation angle of the top corner of each
  drawn non-void leaf within 20 km (from its mesh bounds); a node whose top elevation lies within two
  cell angles (`2 · 2^L / d`) of its bin's maximum is a skyline node. All directions count.
- The prototype's grazing factor (× 1.45 for `|cos| < 0.38`, `src/render/lod.rs:212-241`) is not
  used: near the ground it applies to almost every visible node and roughly doubles in-frustum
  demand.

### 7.3 Node classes and never-split rules
- Classes come from job results (exact, from cells), from `quick_class` and `quick_subtree_empty`
  (§3.2), from `EditReceipt::class_changes` (`cp-voxel/src/edit.rs:75-76`) and from the class cache
  (§7.9). `classify` never runs on the frame thread: it may cost as much as a fill
  (`cp-voxel/src/base.rs:99-107`, `snapshot.rs:65-69`).
- A node never splits when: L = 0; it is subtree-empty (quick or exact); its box lies inside
  `shape.core` and `!edited_within` (every descendant is opaque); it is **buried** — class Opaque,
  all six same-level face neighbours Opaque, no override layer in its 27-neighbourhood at its level
  (`VoxelSnapshot::override_layer`, `snapshot.rs:174-178`) and no eye within its box widened by 32
  cells of its level (a 32-cell opaque shell exceeds the ≤ 1 coarse cell LOD-consistency offset many
  times, so its children are all opaque); or it is occluded (§7.4).
- Empty nodes may split: finer levels keep finer octaves and features.
- Dormant nodes (§5.4) are void while their mask is empty; their mesh is requested as soon as a mask
  bit would turn ON.

### 7.4 Core occlusion (`ARCHITEKTUR.md:519-520`)
- **Occluder** per grid: `C_g` = `shape.core` shrunk by `2^(root_level − 1)` m on every side (512 m
  on Heimat): a point of the shrunk box lies at least half a root cell inside the core, so the cells
  of every drawn level that contain it are opaque. Then, for every level-0 override chunk inside
  `C_g`, its nearest face moves inward until `C_g` excludes that chunk widened by
  `2^(root_level − 1)` m (ties by `Face::index`). Computed from `overrides()` at `add_grid`
  (`snapshot.rs:180-185`) and kept up to date from receipts.
- **Test** (f64 slab test): N (L ≥ 2) is occluded for a view iff the eye lies outside `C_g` and every
  segment from the eye to the 8 corners of `B_N` meets `C_g`. The shadow of a convex box seen from a
  point is convex, so the corners decide for the whole box.
- **Hysteresis:** a leaf does not split while `B_N` is occluded; a split node merges for occlusion
  only once `B_N` grown by 10 % of its extent is occluded. Rechecked at least every 64 m of eye
  travel. Frustum views never occlude. E4 adds other grids' oriented boxes to the same test.

### 7.5 Selection under hard caps
- **Incremental walk.** An odometer Ω accumulates each frame the maximum over views of
  `max(|Δeye|, |Δ(eye + look-ahead)|)`. After a node is evaluated, `next_check = Ω + slack`, slack =
  the smallest distance (m) to its split or merge threshold, capped at 64 m (occlusion recheck),
  infinite for never-split void nodes until a class or edit change; `subtree_next` = minimum over the
  subtree; the walk descends only where `subtree_next ≤ Ω`. Silhouette-band and skyline nodes have
  slack 0. A full walk follows a bias change, a view added or removed, an FOV change or the eye
  crossing a face plane. At most 6,000 visits per frame; an interrupted walk resumes in the same
  order next frame (grids by `GridId`, depth-first in child order). In tests and in Deterministic
  mode every 64th frame's incremental cut must equal a full walk.
- **Admission.** Nodes that want to split are admitted in descending priority
  `p(N) = max_v(priority_weight_v · k_v · e_v(N))` (k = 1 inside a Primary or Frustum view's frustum,
  0.5 outside) while: projected resident ≤ 0.8 · 5,500 = 4,400 (each admitted split adds its
  non-void children; its parent leaves when the fade ends); projected in-frustum =
  `max(in-frustum + new in-frustum children, 0.25 · projected resident) ≤ 0.8 · 1,400 = 1,120`
  (the second term (A) keeps turning the camera from exceeding the cap); the sink keeps ≥ 10 % of
  the pool and ≥ 1,024 items free. A refused split sets `budget_limited`.
- **Cap hysteresis.** Existing splits stay until a cap reaches 100 %. Above 100 %, the
  lowest-priority split nodes whose children are all leaves (for the in-frustum cap: those with
  children in the frustum) merge with fast fades until both counts are at most 100 %.
- **Pressure levers,** applied in this order after splits have been refused for 1 s, one step per
  0.5 s: (1) skyline gain off, (2) silhouette gain off, (3) pressure bias × 1.05 up to 1.5. Relaxed
  in reverse after 2 s below 70 % of both caps. Drawn coverage is removed only by forced merges
  above 100 %.
- **Counts:** resident = chunks held by the sink; in-frustum = drawn chunks whose box meets the
  Primary frustum (f32 `cp_core::math::Frustum` on camera-relative boxes,
  `cp-core/src/math.rs:362-379, 445-539`), found by a hierarchical walk in ≤ 0.1 ms.
- A grid none of whose views has `draw = true` keeps only its roots.

### 7.6 Transitions and readiness
- **Split(P)** — P is a drawn Full leaf that wants to split and is admitted: request the family
  (§7.7); insert meshed children `Hidden` as their meshes arrive (paced, §7.1). Ready when every
  meshed child is resident, every dormant node (child or neighbour) whose mask is non-empty in the
  fade configuration is resident with that mask, and the family limits allow. Then, in one frame,
  the ramps start (§6.3) and the masks of the fade configuration are set on the members and their
  face-adjacent drawn nodes. At the end the children become Full, P leaves the sink (its mesh goes
  to the memory cache) and the masks are recomputed.
- **Merge(P)** — P is split, all its children are drawn Full leaves or void, and P wants to merge:
  P's mesh comes from the memory cache, the disk cache or a Single job; P is inserted hidden (parent
  role, static 16); ready when it is resident (plus dormant walls). The ramps run down from 16; at
  the end P is Full and the children leave.
- **Duration:** 250 ms; 66 ms (fast) when the family lags (`e_eye > 1.5` at the current eye for any
  member, or the desired leaves lie at least 2 levels below P) or when the merge is forced; 0
  (instant) in Settle mode and in a frame with `taa_cut`.
- A node belongs to at most one open family; a transition whose region intersects an open family
  waits for it (no nesting).
- **Reversal:** when the wanted state flips during a ramp, all members reverse from the current
  level (§6.3). **Cancellation** before the start removes hidden members; their meshes stay cached.
- **Bootstrap:** a node with no drawn ancestor and no drawn descendant (roots, new grids) is shown
  Full as soon as it is ready, without a fade. cp-app hides start-up behind Settle mode.

### 7.7 Requests, priorities, pumps and jobs
- **Own priorities.** cp-jobs fixes a job's queue key at submit (`cp-jobs/src/queue.rs:47-53,
  127-178`), so priorities live in cp-stream. Every frame the frame thread recomputes the priority of
  every wanted, unstarted request, selects the K most urgent (`select_nth_unstable`, O(n);
  `K = max(32, ⌈1.5 · T · 16.7 ms / unit_ms⌉)` with `unit_ms` the EWMA of job time; K = 64 in
  Deterministic mode) and swaps them into the shared queue (one `parking_lot::Mutex`, O(K)).
- **Order:** `(band, key, sequence)`. Band 0: due now without look-ahead (`e_eye > 1`) inside a
  Primary or Frustum frustum; 1: due with look-ahead inside a frustum; 2: due elsewhere; 3: merges
  and prefetch (families of children of admitted splits with `e > 0.7`; results go to the memory
  cache only). `key = ⌊2^24 / (1 + p)⌋`. The eight children of one parent are one request.
- **Pumps.** `T = bulk_limit + 2` token jobs of class Streaming, key 1, spawned detached through the
  `JobSpawner`; a detached job can only be cancelled by shutdown (`cp-jobs/src/system.rs:210-222`), so
  tokens carry no request in their closure. A token takes requests (`Pending → Taken` by CAS) and
  runs them; after ≥ 2 ms it spawns its successor and returns; it also returns when the queue is
  empty, and the frame thread spawns tokens when work appears. Tokens never wait for other jobs: a
  bulk job that waits for a bulk job keeps its slot (`cp-jobs/src/lib.rs:74-76`). The standard
  policy gives the baseline 9 workers and a bulk limit of 5 (`cp-jobs/src/policy.rs:194-218`).
- **Cancel.** An unwanted Pending request leaves the queue and never runs; a Taken one gets its flag
  set, and the generator polls it per column (`&|| request.cancelled()`,
  `cp-voxel/src/base.rs:81-86`). cp-stream never cancels a cp-jobs job from the frame thread,
  because cancelling a queued job drops its closure on the cancelling thread
  (`cp-jobs/src/queue.rs:180-196`).
- **Family job:** quick-void children skipped; disk lookup for L ≥ 3 for every needed child (all
  hits → done); `try_sample_padded_family`; per child class from cells, `mesh_into` with the
  thread-local Mesher, summary, `MeshBackend::prepare`; codec bytes of levels ≥ 3 to the
  housekeeping queue. **Single job:** disk lookup (L ≥ 3); `classify`; Empty → Void with
  `snapshot.subtree_empty`; else `try_sample_padded_with`, mesh, prepare. A reply guard sends
  `Failed` when a job ends by panic or cancel (the pattern of `cp-app/src/scene/build.rs:69-93`).
- **Per worker thread** (`thread_local`): Mesher, ChunkMesh, padded buffer, `FamilyScratch`, codec
  buffer. No per-chunk allocation except the prepared mesh's quad `Vec`.
- **Results** travel over a crossbeam channel. Back-pressure: nothing is published while more than
  256 results are undrained or more than 16 MiB of uploads are pending in the sink.
- **Housekeeping thread** (one, owned by cp-stream): disk writes (bounded queue of 4,096 records;
  overflow drops and counts) and deferred drops. It is not a cp-jobs Background job: cp-jobs never
  starts a Background job while Streaming work is queued under the shared bulk limit
  (`cp-jobs/src/queue.rs:19-31, 314-328`).
- **Edit groups:** `JobClass::Interactive`, at most 8 keys per job, spawned with kept `JobHandle`s.
- **Settle mode:** requests run as Interactive jobs on all workers, as E2's loading builds do today
  (`cp-app/src/scene/build.rs:399-410`).

### 7.8 Edits
Receipts are applied in order before the selection. Each dirty key (`cp-voxel/src/edit.rs:64-74`)
loses its memory-cache entry; class changes update the classes and the occluder; dirty keys that
are drawn or resident (except hidden family members, which are requested again) are rebuilt as one
edit group per frame (all receipts of that frame) at Interactive priority. When every member is
built, the group's replaces, inserts (chunks that became non-void) and removes (chunks that became
void) go to `RenderSink::apply_group` in one call and become visible in the same frame (I11).
Target: up to 64 dirty drawn chunks visible at most 2 frames after the receipt (A: about
64 × 0.7 ms on the 4 workers outside the bulk limit ≈ 11 ms). Dirty keys that are not drawn are
rebuilt when next needed.

### 7.9 Caches and the pack store
- **Class cache:** `ChunkKey → { class, subtree_empty, summary, revision }`, LRU of at most 2^20
  entries, invalidated by receipts.
- **Memory mesh cache:** `(ChunkKey, revision) → CachedMesh` (a `B::Mesh` clone and its summary),
  LRU by bytes, 256 MiB, fed by every accepted result and every mesh that leaves the sink. A hit is
  re-inserted by clone (a few KB of memcpy; `PreparedChunk` is `Clone`,
  `cp-render/src/voxel/scene.rs:166-167`) on the frame thread within the sink-op cap.
- **Disk mesh cache:** a `PackStore` under `DiskCacheConfig.dir`, levels ≥ 3, key of §5.6, value =
  codec bytes; looked up in jobs only, written by the housekeeping thread; a hit is decoded,
  validated and prepared on the worker; any decode or validation error counts as a miss and marks
  the record dead. Default directory (cp-app): `CP_CACHE_DIR`, else
  `%LOCALAPPDATA%\Cubeplanet\cache\meshes` on Windows and `$XDG_CACHE_HOME/cubeplanet/meshes` (or
  `~/.cache/cubeplanet/meshes`) elsewhere — never inside the repository. Benches use a fresh
  directory under `std::env::temp_dir()`, deleted at exit unless `--warm-cache <dir>` is given.
- **PackStore** (cp-save, WP-C). std only: cp-save declares flate2, serde, thiserror and toml, no
  parking_lot, rustc-hash or memmap2 (`cp-save/Cargo.toml:10-17`).

```rust
pub struct PackConfig { pub budget_bytes: u64 /* 1 << 30 */, pub pack_bytes: u64 /* 64 << 20 */,
                        pub refresh_age: u32 /* 4 */ }
pub struct PackStore { /* Send + Sync */ }
impl PackStore {
    pub fn open(dir: &Path, config: PackConfig) -> Result<PackStore, SaveError>;   // writer; locks dir/LOCK
    pub fn open_read_only(dir: &Path) -> Result<PackStore, SaveError>;
    pub fn get(&self, key: &[u8]) -> Result<Option<Vec<u8>>, SaveError>;  // key ≤ 64 B, compared verbatim
    pub fn put(&self, key: &[u8], value: &[u8]) -> Result<(), SaveError>; // append; newest wins; ≤ 4 MiB
    pub fn flush(&self) -> Result<(), SaveError>;
    pub fn usage(&self) -> PackUsage;   // packs, bytes, records, hits, misses, corrupt, evicted packs
}
```
  - Files `pack-{generation:010}.cpk` (a counter, never a clock); immutable once sealed, never
    truncated or rewritten (contract §6 cp-save). Header: `CPPACK01`, format u32, namespace
    `[u8; 4]` = `MESH`. Record: `CPR1`, key length u16, value length u32, key, value, FNV-1a 64
    (`cp_core::hash::hash_bytes`) of key‖value. Sealing appends an index (per record: FNV-1a of the
    key u64, offset u64, length u32) and a footer (`CPI1`, count u32, index hash u64, index offset
    u64).
  - Open: sealed packs load their index; an unsealed pack is scanned record by record up to the
    first bad length or hash and stays unsealed (rescanned on every open); new writes go to a new
    pack.
  - Reads are positioned (`std::os::windows::fs::FileExt::seek_read`,
    `std::os::unix::fs::FileExt::read_at`): no `unsafe` (`unsafe_code = "deny"`,
    `engine/Cargo.toml:60-61`), no memmap2.
  - Eviction: while the total exceeds the budget, delete the oldest sealed pack. A hit in a pack
    more than `refresh_age` generations old is put again into the current pack (LRU by generation).
  - A second writer gets `SaveError::Locked` (via `std::fs::File::try_lock`) and opens read-only.
    Corrupt records are misses, logged once per pack. `BTreeMap` and `std::sync::Mutex`; equal put
    sequences give byte-identical files (cp-save is deterministic, contract §2.5).

### 7.10 Eviction
| What | Where | Rule |
|---|---|---|
| drawn and hidden meshes | quad pool | removed when a family ends or is cancelled, or a dormant mask empties; the mesh moves to the memory cache |
| memory meshes | 256 MiB | LRU by bytes |
| classes | 2^20 entries | LRU |
| disk records | 1 GiB | whole packs, oldest generation first; refresh on hit |
| nodes | arena | freed 600 frames after they were last wanted, drawn, requested or referenced |
| base chunks | cp-voxel `BaseCache`, 128 MiB (`cp-voxel/src/world.rs:24-31`) | its LRU, pinned chunks excepted |

There is no GPU grace period: a chunk leaves the pool when its family ends, so resident counts stay
honest.

### 7.11 Simulation residency (`docs/PHYSIK.md:417-419`)
- An anchor is a box of level-0 chunk positions with a priority; physics supplies one per active
  body (its AABB swept over 2 s plus 32 m). Keys are taken nearest to the box centre first, Empty
  ones (`quick_class`, `quick_subtree_empty`) skipped, at most 2,048 per anchor; the status reports
  `truncated` when more were wanted.
- **Prefetch:** at most 2 SimResidency pump jobs of class Streaming with key 0 (they start before
  the render tokens' key 1 whenever a bulk slot frees), batches of 8 keys, `snapshot.try_pin(key,
  cancel)`; pins are held while an anchor covers the key. Not Critical: Critical and Interactive are
  never held back by the bulk limit (`cp-jobs/src/lib.rs:35-39`), so an uncapped Critical prefetch
  could occupy all 9 workers.
- A simulation read that misses generates on the reading thread at once through cp-voxel's
  single-flight cache (`cp-voxel/src/cache.rs:145-211`), never queued: this is v3's "generate on miss
  in class Critical" (`ARCHITEKTUR.md:515-516`). The simulation thread passes
  `take_thread_read_stats()` to `record_reads` once per tick; `generated + waited` is the
  "physics waited" count.
- cp-stream never answers simulation queries (I4).

### 7.12 Modes
| Mode | Used by | Behaviour |
|---|---|---|
| Realtime | `cpe run`, real-time benches | never blocks; governor on |
| Settle | start-up, teleports, fixed-camera shots | jobs Interactive; transitions instant; `is_settled()` = no request queued or running, no ramp running, wanted cut = drawn cut, no pending sink uploads |
| Deterministic `{ latency_frames: 4 }` | route shots, probes, determinism gates | results of requests published at frame n are applied exactly at frame n + 4 (`update` blocks until they exist), in `(publish frame, GridId, LodKey)` order; no cancellation (unwanted results go to the cache); K = 64; bias 1.0; governor off; `clock_ms = ⌊frame · 1000 / 60⌋`; fixed upload budget |
| Inline | unit tests | every job runs inside `update` |

### 7.13 Budget arithmetic (derived)
| Quantity | Value |
|---|---|
| focal length | 935.3 px at 1080p and 60° |
| level-0 region | ≤ 312 m at b = 1; ≥ 240 m at any bias |
| eye-level counts (flat-plane model: 224 chunk columns per level ring, 1.3 chunks per column (A)) | 2,400–2,700 resident steady, ≤ 3,500 with both hysteresis bands full; 650–900 in a 91.5° × 60° frustum |
| demand at 321 m/s, 50 m above ground | about 600 non-empty chunks/s (the level-0 band, 2 × 312 m wide, sweeps 200,000 m²/s) |
| capacity | about 3,300 non-empty chunks/s on 5 bulk workers at about 1 µs per 3D column (A); E2 measured about 10,200 per second on 9 Interactive workers with 0.561–0.706 ms generation (`docs/bench/e2-hills.json:787-1077`), no longer the hills' cost (about 3.7 ms, `docs/STAND.md:64-67`) |
| uploads | 600 chunks/s × about 3.6 KB ≈ 2.2 MB/s ≈ 36 KB per frame (0.9 % of 4 MiB) |
| quad pool | about 2.5 M quads, 15 % of 128 MiB (§6.12) |
| CPU memory | mesh cache 256 MiB + base cache 128 MiB + about 2.5 MB of scratch per worker + nodes about 5 MB |
| tracked VRAM | pool 128 MiB, items 4 MiB (65,536 × 64 B, `cp-app/src/stage.rs:26`), indirect and buckets about 22 MB, targets about 87 MB (both TAA history textures included), Hi-Z about 2.8 MB, history 0.25 MB, face maps 10–20 MB, block textures ≤ 50 MB: about 0.33 GB of the 3 GB limit |

---

## 8. cpe tools: scene, flight route, benchmark output, beauty shots

### 8.1 The cube scene runtime (WP-G)
- Loads the systems with `cp_universe::load_systems` (`cp-universe/src/schema.rs:564`) from
  `content/base/systems`, finds `cubeplanet:heimat` (`system.rs:241`) and takes its `CubeDef`. This
  makes cp-app the first consumer of cp-universe (declared in `cp-app/Cargo.toml:24`).
- Generator from the factory (§4.10); world
  `VoxelWorld::new(source, tables, WorldConfig::for_root_level(10))`.
- Render object: bounds = the root box `[-32768, 32768)³` (lattice); pose = identity rotation and
  translation `(−o, −o, −o)`, so that the render reference frame is the body frame (lattice = body +
  o, `cp-universe/src/body.rs:10-12`); `set_object_lattice_centre(object, o)`.
- Face ground maps from `CubeHillsSource::tint` at 64 m texels, built with `par_for` at start-up in
  at most 1 s.
- `Streamer` with `RenderSink` and `PreparedBackend` (§8.2), one Primary view, and
  `GridDesc { grid: GridId::Body(heimat), roots: RootSet::body(10),
  cache_salt: Some(hash_str("cubeplanet:heimat")), cube: Some(CubeFrame { centre: (o, o, o),
  half_edge: R }) }`.
- Camera in the body frame (f64); `eye_lattice = body_to_lattice(eye)` (`body.rs:209-214`); up =
  −`GravityField::field(p).acceleration`, normalised (`cp-universe/src/gravity.rs:94`;
  `SystemDef::gravity`, `system.rs:307`), which rolls the view over at edges; the sun from
  `SystemDef::body_frame` (`system.rs:345`) at the route clock; sky colours from the existing
  `daylight` helper (`cp-app/src/scene/sky.rs:129`).
- Interim lighting: altitude = `‖q‖∞ − R`; fog density × `(1 − smoothstep(2 km, 8 km, altitude))`,
  fog falloff 0 above 2 km, horizon fade off; space = `smoothstep(0.5, 1.0, altitude /
  atmosphere_height)` (Heimat 8,000 m, `home.toml:73`); cloud opacity =
  `1 − smoothstep(2 km, 4 km, altitude)`; TAA cut distance `max(50 m, 0.1 · altitude)`.
- `--generator cube_flat` swaps the source for the probe validation.
- Edits: the `edits` route applies scripted `CellEdit`s through `VoxelWorld::apply`
  (`cp-voxel/src/world.rs:59-63`) and passes the receipts in the next `StreamFrame`.

### 8.2 Glue (`stream_glue.rs`)
- `PreparedBackend: MeshBackend<Mesh = PreparedChunk>` uploads the ranges of §5.3 (all regular
  ranges, opaque and fluid walls, no cutout walls) through `LayerCounts::from_slots`
  (`cp-render/src/voxel/scene.rs:93-101`) and `PreparedChunk::from_vec(quads, layers,
  key.origin_voxel(), key.level, object)` (`scene.rs:186`) on the worker.
- `RenderSink` keeps the `SinkChunk → ChunkId` map; `insert` = `insert_prepared` plus `set_fade`,
  `set_wall_mask` and `set_up_mode`, all stored on the pending chunk and baked in on completion
  (`state.rs:658-697`); `WallBits → WallMask::from_bits`; `FadeCmd → Fade`; groups →
  `VoxelScene::apply_group`; refusals from `RefusedChunk` (`scene.rs:229-236`). Up mode as §6.8.
- The governor (§3.7) is fed from `latest_timings`.

### 8.3 Routes
Deterministic functions of time, sampled at 60 Hz camera time, in the body frame: Catmull-Rom
splines through keyframes; terrain-following segments smooth the ground height over 2 s and keep
≥ 30 m clearance over landmarks; speed ≤ 321 m/s below 2 km altitude and ≤ 1,500 m/s above.

| Route | Content | Length |
|---|---|---|
| `flight` | 10 s standing at S (eye 1.7 m) → 20 s accelerating to 321 m/s at 50 m above ground → 60 s low pass along +u to the +Z/+X edge → crossing the edge at 300 m → climb (≤ 321 m/s below 2 km) to r = 40 km from the centre → 60 s on the circular 40 km orbit at 321 m/s (the circular speed, `home.toml:63`), camera 25° below the horizon at the limb → descent to a point 2 km from S on +Z at `min(321, 10 · (1 + h/10))` m/s below 2 km → 10 s standing | ≤ 400 s |
| `lowpass` | 10 s to reach 321 m/s, then 60 s at 321 m/s, 50 m above ground, straight along +u from S | 70 s |
| `descent` | from r = 40 km over the +Z face centre down to S | ≈ 150 s |
| `edge` | 200 m/s, 100 m above ground, 60 s along the +Z/+X edge, then over the +X/+Y/+Z corner | 90 s |
| `edits` | standing at S; from t = 5 s one 3 × 3 × 3 dig per second at chunk corners within 30 m (20 digs); at t = 25 s a 2 × 2 × 64 m tower and a 40 × 1 × 10 m one-block wall of a STRUCTURAL state within 60 m | 30 s |

### 8.4 `cpe run --scene cube`
Realtime mode after Settle behind a loading screen; free fly at `10 · (1 + h/10)` m/s (h = altitude
above the level-0 ground along −gravity, from `surface_height`), × 6 with Shift, at most 3,000 m/s;
`--route <name>` plays a route; the debug-view cycle includes views 8–13; HUD: resident,
in-frustum, queued, chunks/s, upload KB per frame, bias, lag, open fades, pool use.

### 8.5 `cpe bench --scene cube` (schema 2)
Options: `--route <name> | --throughput | --acceptance`, `[--hiz on|off] [--path a|b|direct]
[--check] [--warm-cache <dir>] [--frames n] --out <json>`.
- **Paced real time:** frame n starts at `t0 + n · 16.667 ms`; at most 2 frames in flight; no
  per-frame GPU wait (today `Bench::measure` waits after every frame, `cp-app/src/bench.rs:375`);
  `frame_ms = max(CPU wall time, GPU total of the same frame index)`; 60 warm-up frames after
  Settle. The E2 static bench stays as it is.
- **Schema 2** (`REPORT_SCHEMA` 1 → 2, `bench.rs:39`): `frames` {count, frame_ms, cpu_frame_ms,
  gpu_total_ms as mean/p50/p95/p99/min/max, over_16_6, over_20}; `gpu_ms` per pass (cull, gbuffer,
  hiz, cull_late, gbuffer_late, lighting, taa, composite); `cpu` {select, apply, walls, publish,
  flush_uploads, write_items} µs summaries; `stream` {resident max/mean, in_frustum max/mean,
  drawn_by_level, chunks_per_s {mean, saturated_min}, completed, non_empty, cancelled, stale,
  failed, mem and disk hits, lag {share p99/max, e_eye max, refine latency p99/max}, fast_fades,
  instant_switches, bias, budget_limited_frames, open_families max}; `uploads` {mb_per_s,
  kb_per_frame max, pending_bytes max}; `pool` {quads max, largest_free min, refusals, wall_share};
  `hiz` {phase1_items, phase2_items, culled_share}; `vram` (MemoryReport); `generation` {fill_ms,
  family_ms_per_child, mesh_ms, classify_ms}; a timeline every 0.5 s (t, altitude, speed, frame_ms
  max, resident, chunks/s); per-segment summaries; config, machine and adapter as in schema 1.
- **`--check`:** an untimed replay in Deterministic mode with `validate()` every frame and the
  background-hole check: every 30th frame the full-resolution depth is read back; depth-0 pixels not
  connected to the image border through depth-0 pixels are holes.
- **`--throughput`:** five teleports (eye level at S, 300 m, 3 km, 12 km, 40 km), disk cache off,
  Realtime mode (bulk limit 5), each measured until settled. Chunks/s = meshes with at least one
  uploaded quad accepted per second, over saturated 1 s windows (the request queue never empty in
  the window).
- **`--acceptance`:** runs the automatable gates of §9, prints a pass/fail table and writes
  `engine/docs/bench/e3/acceptance.json` (with `e3-<route>.json` and `e3-silhouette.json`).

### 8.6 `cpe shot --scene cube` and the E3 showcase
- `--camera n`: Settle until settled, then 32 TAA frames (`cp-app/src/shot.rs:34`).
- `--route r --time t`: Settle at the camera of `t − 8 s`, then Deterministic mode over 8 s of route
  to t; the last 24 history frames are rendered at 30 fps camera steps (`shot.rs:38-42`), each
  advancing the stream by two 60 Hz steps. Motion pairs are taken at t and `t + 1/30 s`.
- **Showcase** (fixed in `scene::cube::SHOWCASE`, kept under `target/beauty/`, same cameras and sun
  every round, in addition to the E2 showcase):

| # | Name | View |
|---|---|---|
| 0 | eye-noon | 1.7 m at S, sun 45°, plaza and hills, level 0–2 borders in view |
| 1 | eye-evening | the same, sun 6° |
| 2 | coast | 30 m over a coast, sea to the horizon |
| 3 | forest | 120 m over a forest edge, 1–5 km of forest (canopy and far canopy) |
| 4 | mountains | 400 m, towards a range 3–12 km away (skylines) |
| 5 | edge-walk | 10 m above ground, 25 m from the +Z/+X edge, looking over it |
| 6 | edge-air | 300 m above the +Z/+X edge |
| 7 | corner-air | 1 km above the +X/+Y/+Z corner, along an edge |
| 8 | alt-2km | 2 km above +Z |
| 9 | alt-8km | 8 km above the +X edge, two faces |
| 10 | orbit-face | 40 km over +Z, 60° down |
| 11 | orbit-limb | 40 km, the +X limb against space |
| 12 | orbit-corner | 60 km from the centre along (1, 1, 1), three faces |
| 13 | far-face-y | eye level on the −Y face |
| 14 | far-face-x | 30 m above the +X face |
| 15 | towers | 300 m, the landmark towers 2–12 km away |
| 16, 17 | lowpass-a0/a1 | `lowpass` at t = 30.0 / 30.0333 s |
| 18, 19 | descent-b0/b1 | `descent` at t = 120.0 / 120.0333 s |

- **Judging focus** (contract §9 beauty review, `ARCHITEKTUR.md:633-645`): all 14 criteria are
  scored; the E3 focus is popping and fades in the motion pairs, cracks and seams at level borders,
  shimmer of far cells, cube edges and corners (5–7), silhouettes and the limb (4, 11, 12, 15).
  Atmosphere, limb glow and night belong to E5, and the judges are told so. The fixer stays inside
  E3's scope (LOD, streaming, the cube test generator, cube-aware shading); the performance gates
  hold after every fix.

### 8.7 `cpe probe`
- **`holes`** (default route `flight`): Deterministic replay, path A, Hi-Z off, the Holes view on
  every 30th frame plus the background-hole check; reports frames, magenta pixels, the worst frame
  and its pixels.
- **`pop`:** Deterministic replay; at sampled frames, for up to 4 in-frustum fading families, render
  depth with the parent cut and with the children cut (`Streamer::force_cut`, TAA and jitter off);
  per screen column where the family is the topmost geometry, the skyline displacement
  `|y_P − y_C|` in px and in coarse cells (÷ `2^Lp · f / d_P`); the area that changes between sky and
  geometry. On `--generator cube_flat` the displacements must equal the analytic point-rule steps.
- **`silhouette`:** 12 fixed cameras (4 at eye level towards horizons 1–20 km away; 500 m, 2 km,
  8 km and 15 km; 2 across edges; 2 at the orbit limb), each settled at bias 1.00 and at 1.15 (so
  exactly the nodes in the hysteresis band switch); per image column the distance between the
  topmost geometry rows; p50, p95 and max px per camera.

---

## 9. Acceptance (automatable, with thresholds)

Release builds on the baseline (Ryzen 5 9600X, GTX 1080, Vulkan, 1080p, path A unless stated), with
no other build running (contract §10 rule 8).

| # | Gate | Method | Threshold |
|---|---|---|---|
| A1 | seamless | `cpe probe holes --route flight`; `cpe bench --route flight --check` | 0 magenta pixels; 0 enclosed background pixels; `validate()` without violation in every frame. A1 binds only after WP-F's test shows 0 magenta pixels on a watertight scene and > 0 on a deliberately cracked one |
| A2 | chunk budgets | real-time bench on `flight`, `lowpass`, `descent`, `edge` | resident ≤ 5,500 and in-frustum ≤ 1,400 in every frame; `budget_limited` in ≤ 2 % of frames; 0 sink refusals; bias ≤ 1.2 in ≥ 90 % of frames |
| A3 | throughput | `--throughput` | non-empty chunks accepted per second: mean ≥ 1,500 over the saturated windows, every saturated window ≥ 1,200 |
| A4 | no LOD lag at 321 m/s | real-time `lowpass` | lag share (drawn non-void leaves in the Primary frustum with `e_eye > 1.15`, at the current eye, no look-ahead, σ = 1; a fading parent counts until its level passes 8) P99 ≤ 1 % and max ≤ 3 %; `e_eye` ≤ 1.5 in every frame; refine latency (first frame with `e_eye > 1` until the fade ends) P99 ≤ 250 ms, max ≤ 500 ms; fast fades ≤ 2 % of transitions; 0 instant switches |
| A5 | frame time while streaming | real-time `descent` and `flight` | P99 `frame_ms` < 16.6 ms; no frame > 20 ms |
| A6 | frame-thread CPU | profile scopes on `flight` | selection P99 ≤ 0.5 ms; apply + scene flush + item writes P99 ≤ 0.8 ms; CPU frame P99 ≤ 4 ms |
| A7 | visibility GPU | `flight`, path A | `cull + hiz + cull_late` P99 ≤ 0.25 ms; G-buffer (both phases) P99 reported against contract §7's 1.9 ms — exceeding it is a recorded deviation for the lead, not a fail by itself (the E2 beauty loop already raised it, `docs/STAND.md:62-63`) |
| A8 | Hi-Z | 8 settled cube shots and 100 route frames with Hi-Z on and off; cp-render test | ≤ 0.005 % differing pixels and no differing 2 × 2 block; Hi-Z-culled items drawn against the final depth give 0 passing fragments over 20 frames; Hi-Z becomes the default iff its GPU P95 ≤ the P95 without it on every route |
| A9 | exact fades | cp-render GPU test; CPU partition test; route frames | overdraw exactly 1 for parent/child pairs at all 17 levels × 16 rotations on paths A, B and direct; every threshold owned by exactly one member for all `(s0, ramp, elapsed)` including reversal; per-frame luminance step after TAA ≤ 25 % of the family's total change in 99 % of families |
| A10 | silhouette deviation | `cpe probe pop` (≥ 200 families, levels 1–8); `cpe probe silhouette` | skyline displacement P95 ≤ 0.5 coarse cell, max ≤ 1.0 coarse cell; `cube_flat` equals the analytic steps within ±0.5 px; bias probe p95 ≤ 6 px and max ≤ 12 px per camera |
| A11 | edits | `edits` route | 100 % of receipts complete in one frame, at most 2 frames after the receipt |
| A12 | generator | cp-worldgen cube tests (§4.9) | all green, timing included |
| A13 | precision and six faces | cube shot at lattice offset 21,845 vs an identical copy at the origin; static camera 1.5 m over the +Z/+X edge; cp-render six-face test | identical images; temporal standard deviation ≤ 1/255 in ≥ 99.9 % of pixels over 32 frames after 64 warm-up frames; every face renders like the rotated +Y reference (texture, tint, ground cover, grass fringe) |
| A14 | disk mesh cache | a second `flight` run in a new process; cp-save and cp-stream tests | ≥ 90 % disk hits among level ≥ 3 builds; 200 random entries byte-equal to fresh builds; total ≤ 1 GiB plus one pack; torn-tail, corruption and lock tests green |
| A15 | simulation residency | cp-stream test | an anchor of ≤ 2,048 chunks pinned within 2 s; a reader thread over 64 columns then reports `generated = 0`; without prefetch > 0; Empty chunks skipped; truncation reported |
| A16 | determinism | route shots twice; Inline runs of `lowpass` (600 frames) twice; walk check | identical PNG bytes; identical sink command logs; incremental cut = full walk at every 64th frame |
| A17 | budget forecast | WP-D Heimat test: 40 flight poses and the 5 throughput places, selection only, classes from the real `CubeHillsSource` | desired drawn leaves ≤ 4,400 and in-frustum ≤ 1,120 at τ = 6 |
| A18 | E2 unchanged | hills cameras 0–7 and structures 0–2 against pre-E3 images; calib bench | ≤ 0.05 % differing pixels; G-buffer ≤ 2.0 ms at 1.0 M triangles, cull ≤ 0.1 ms |
| A19 | memory | `MemoryReport` over `flight` | VRAM < 3 GB; pool peak ≤ 60 % of 128 MiB |
| A20 | beauty | E3 showcase and E2 showcase, 3 judges, 14 criteria (contract §9) | every judge averages ≥ 8.0, no criterion below 7.0; A1–A19 still pass after every fix |

`check.ps1` is green.

---

## 10. Work packages and build order

### 10.1 Packages
Every path is relative to `engine/`; ownership is disjoint. A package edits only its paths and
requests changes elsewhere from their owner (contract §10 rule 1). Paths not listed stay free for
E4 and P1.

**WP-A — cp-voxel and cp-mesh (lands first, target ≤ 2 days).**
- Owns: `crates/cp-voxel/src/base.rs`, `cache.rs`, `world.rs`, `lib.rs`, `snapshot.rs` (only:
  `overlay_interior` and `overlay_face` become `pub(crate)`), new `plan.rs`, `residency.rs`,
  `sample.rs`; `crates/cp-voxel/tests/voxel/main.rs`, new `tests/voxel/plan.rs`, `residency.rs`,
  `sample.rs`; `crates/cp-mesh/src/lib.rs`, new `src/codec.rs`, `src/tests/mod.rs`, new
  `src/tests/codec.rs`.
- Depends on: nothing.
- Deliverables: §3.2 and §5.5.
- Tests: shape consistency (5,000 keys); `quick_class` None for every key with an override layer
  inside (levels 0..top_level); pins survive LRU pressure (1 MiB budget, 10,000 generated chunks) and
  become evictable when dropped; `ReadStats` counts leader and waiter on 2 threads and resets;
  `try_sample_padded_with` and `try_sample_padded_family` equal `try_sample_padded` (cells and
  authored masks) on 1,000 keys including edited and halo-edited ones; the buffer comes back on
  cancel; codec round trip of 200 terrain meshes bit-equal, and bit flips, truncation, wrong
  versions, a quad on a wrong range and inconsistent boxes give errors, never panics; existing
  cp-voxel and cp-mesh tests and goldens unchanged.

**WP-B — cp-worldgen.**
- Owns: `crates/cp-worldgen/**`: new `src/cube/{mod.rs, shape.rs, noise3.rs, field.rs, bounds.rs,
  surface.rs, trees.rs, landmarks.rs, source.rs, flat.rs, tint.rs}`; `src/lib.rs`; existing files
  only for output-neutral visibility changes (`trees.rs` shape constructors); `tests/worldgen/main.rs`,
  `common.rs`, new `cube_golden.rs`, `cube_lod.rs`, `cube_seams.rs`, `cube_classify.rs`,
  `cube_features.rs`, `cube_timing.rs`.
- Depends on: WP-A (`SourceShape`; until it lands, an inherent `source_shape()`).
- Deliverables: §4.2–§4.9.
- Tests: §4.9. The timing test runs in the first three days; if a family exceeds 1.5 ms per child,
  octaves are reduced (VERSION stays 1 until the first golden pin) before WP-G integrates.

**WP-C — cp-save.**
- Owns: `crates/cp-save/**` (`Cargo.toml` unchanged, `src/lib.rs`, new `src/pack.rs`, `record.rs`,
  `index.rs`, `io.rs`, `error.rs`, new `tests/pack.rs`).
- Depends on: nothing.
- Deliverables: the PackStore of §7.9.
- Tests: round trip of 10,000 records (keys 1–64 B, values 0–256 KiB), newest wins, reopen restores
  all; budget eviction (64 KiB packs, 1 MiB budget) and refresh of hot keys; truncation at every
  byte offset of the last record loses only that record; a flipped byte in key, value or index is a
  miss, never a panic; a second writer gets `Locked`; 8 readers and 1 writer for 5 s; identical put
  sequences give byte-identical files; no `unsafe`.

**WP-D — cp-stream: selection, transitions, walls, sink protocol.**
- Owns: `crates/cp-stream/Cargo.toml`, `src/lib.rs`, `config.rs`, `ids.rs`, `view.rs`, `grid.rs`,
  `metric.rs`, `octree.rs`, `select.rs`, `transition.rs`, `walls.rs`, `occlusion.rs`,
  `silhouette.rs`, `sink.rs`, `seam.rs`, `governor.rs`, `streamer.rs`, `stats.rs`, `check.rs`,
  `diag.rs`; `tests/stream/main.rs` (with the module lines of WP-E's test files), `mock.rs`,
  `select.rs`, `transitions.rs`, `walls.rs`, `occlusion.rs`, `budgets.rs`, `views.rs`, `edits.rs`,
  `determinism.rs`, `heimat.rs`.
- Depends on: WP-A (signatures of §3.2).
- Deliverables: §3.1, §3.3–§3.5, §3.7, §3.8 (trait), §5.3 (masks), §7.1–§7.6, §7.8, §7.12; module
  declarations for WP-E's files.
- Tests (mock producer and analytic test grids): every selected leaf has `e ≤ 1` unless budget-
  limited, caps hold under a τ = 1 px stress test; no transition during 600 frames of ±1 m
  oscillation; the look-ahead splits 0.5 s before arrival; far-side nodes of a cube are not refined
  from its surface and occlusion is off with the eye inside the occluder; a 2° Frustum view refines
  only inside its cone; two grids share the caps by priority; `draw = false` keeps only roots; a
  parent stays drawn until its children and their walls are resident (random residency delays);
  ramps never overlap or gap a threshold, reversal is continuous; masks equal a brute-force
  recomputation every frame; edit groups arrive in one `apply_group`; `validate()` holds over
  10,000 frames of random flight with random job delays; Inline runs give identical command logs;
  the incremental walk equals a full walk on 1,000 random camera paths; the governor rule; Heimat
  forecast (A17); selection ≤ 0.5 ms P99 for 6,000 active nodes in release.

**WP-E — cp-stream: production, caches, simulation residency.**
- Owns: `crates/cp-stream/src/produce/{mod.rs, queue.rs, pump.rs, job.rs, mem_cache.rs,
  class_cache.rs, disk_cache.rs, housekeeping.rs}`, `src/sim.rs`; `tests/stream/produce.rs`,
  `pump.rs`, `disk_cache.rs`, `sim.rs`.
- Depends on: WP-A, WP-C, WP-D (seam).
- Deliverables: §3.6, §3.8 (implementation), §5.2, §5.6, §7.7, §7.9 (caches), §7.10, §7.11.
- Tests: queue order (band, key, sequence) with siblings together; never more than
  `bulk_limit + 2` tokens, a slice ends within 2 ms plus one job; Pending cancels never run, Taken
  ones stop within one column; job output equals the E2 pipeline (classify, padded sample, mesh) on
  300 `HillsSource` keys (dev-dependency) including edited ones; family jobs equal 8 single jobs;
  memory cache LRU and revision misses; the disk key layout pinned by a golden, a one-byte key
  difference and a corrupt value are misses, levels < 3 and salt-less grids never write; writes
  never block the frame thread and overflow drops are counted; SimResidency on the hills plane:
  2,048 chunks pinned within 2 s, quick-Empty skipped, truncation reported, `generated = 0` after
  prefetch, pins released with the anchor.

**WP-F — cp-render.**
- Owns: `crates/cp-render/src/voxel/{layout.rs, state.rs, scene.rs, renderer.rs, cull.rs,
  pipelines.rs, stats.rs, taa.rs, ground.rs, mod.rs, tests.rs}`, new
  `src/voxel/{hiz.rs, memory.rs, depth.rs}`; `src/debug_view.rs`; `src/shader/mod.rs`;
  `src/shader/wgsl/common/debug_view.wgsl`; `src/shader/wgsl/voxel/{common.wgsl, cull.wgsl,
  gbuffer.wgsl, lighting.wgsl, composite.wgsl}`, new `src/shader/wgsl/voxel/hiz.wgsl`;
  `tests/{voxel_pipeline.rs, voxel_lighting.rs, voxel_bench.rs}`, new
  `tests/{voxel_streaming.rs, voxel_hiz.rs}`. Not owned (free for E4): `src/gpu/**`,
  `src/frame_graph/**`, `src/overlay/**`, `src/headless.rs`, `src/material.rs`, `src/texture.rs`,
  `src/lib.rs`, `src/voxel/{camera.rs, calibration.rs, pool.rs}`, the other WGSL files.
- Depends on: nothing.
- Deliverables: §6.
- Tests: layout and WGSL mirrors; for all 17 levels × 16 rotations a headless parent/child pair
  covers every pixel exactly once on A, B and direct; `Fade::level_at` equals the shader; hidden
  items draw nothing; a group lands in one flush whatever the budget, the Interactive lane first;
  coalesced writes leave the same buffer contents; Hi-Z pyramid equals a CPU min reference at odd
  sizes (1919 × 1079); Hi-Z images on/off within the A8 tolerance on a wall-occluder scene and under
  swap-remove churn; culled items give 0 fragments; precision scene at offset 21,845 identical to the
  origin; six-face test; Holes view 0 magenta on a watertight mesh and > 0 on a cracked one; TAA cut
  rule; depth read-back of a known quad; `memory()` sums the resources; pipelines < 64 created at
  start-up on every backend; existing pipeline tests unchanged with legacy defaults.

**WP-G — cp-app runtime and content.**
- Owns: `crates/cp-app/src/stream_glue.rs` (new), `src/generators.rs` (new),
  `src/scene/cube.rs` (new), `src/scene/cube/{route.rs, light.rs, showcase.rs}` (new),
  `src/scene/mod.rs`, `src/run.rs`, `src/stage.rs`; `content/base/systems/home.toml`.
- Depends on: WP-B, WP-D, WP-E, WP-F.
- Deliverables: §4.10, §8.1–§8.4, and the runtime API used by WP-H (frozen on day 0):
  `scene::cube::CubeWorld::{load, frame, settle, lighting, streamer, apply_edits, route}`,
  `RouteName::{Flight, Lowpass, Descent, Edge, Edits}`, `Route::camera(t)`, `SHOWCASE`.
- Tests: routes deterministic, C¹, speed and clearance limits hold, eye ≥ 1.7 m above the level-0
  ground; up = −gravity and rolls over at edges; the factory maps keys and rejects plane generators
  on cube bodies and unknown keys; grass `up` equals the face normal on every face's palette;
  `home.toml` loads and passes `SystemDef` validation with `cube_hills`; `WallBits → WallMask` per
  face; `FadeCmd → Fade`; the cp-stream ramp formula equals `Fade::level_at` (this crate sees both);
  GPU-gated: 120 frames of `lowpass` stream without refusal.

**WP-H — cp-app measurement.**
- Owns: `crates/cp-app/src/lib.rs`, `src/cli.rs`, `src/bench.rs`, new `src/bench/{stream.rs,
  schema2.rs, throughput.rs, acceptance.rs}`, `src/shot.rs`, new `src/shot/replay.rs`, new
  `src/probe.rs` and `src/probe/{holes.rs, pop.rs, silhouette.rs}`, `src/metrics.rs`, `src/json.rs`,
  `src/stats.rs`, `tests/cpe.rs`; `docs/bench/e3/` (outputs).
- Depends on: WP-G (runtime API), WP-F (depth read-back, counters), WP-D (diagnostics).
- Deliverables: §8.5–§8.7.
- Tests: parsing of every new option and its errors; schema-2 fields present and typed; the
  silhouette metric on synthetic masks measures a known shift; the crack counters on synthetic
  images; percentile helper against a reference; acceptance evaluation on recorded pass and fail
  fixtures; GPU-gated: `bench --scene cube --route lowpass --frames 600` writes a valid report, two
  shots at `--time 42` give identical PNG bytes, `probe holes` reports 0 normally and > 0 with walls
  disabled through a test switch.

### 10.2 Build order
1. **Day 0:** the signatures of §3.2–§3.8, §6.2 and §8 (WP-G runtime API) are frozen as written
   here. Packages build against mocks in their own tests until the real code lands; no placeholder
   that pretends to work is committed (contract §10 rule 9). On day 0 WP-G adds `SceneName::Cube`
   (`cp-app/src/scene/mod.rs:28`) and WP-H adds its arms in `bench.rs` and `shot.rs` in the same
   integration batch, because those matches are exhaustive (`bench.rs:114-116`, `shot.rs:208-259`).
2. **WP-A first** (≤ 2 days), before P1 step 0 edits `base.rs`, `cache.rs`, `world.rs`, `lib.rs`
   or `snapshot.rs`; P1 rebases on it.
3. **WP-B, WP-C, WP-F** start on day 0; **WP-D** on day 0 against WP-A's signatures; **WP-E** once
   WP-A and WP-C land.
4. **WP-G** once WP-B, WP-D, WP-E and WP-F have landed; **WP-H** after WP-G's runtime API.
5. **Integration:** the integrator runs `check.ps1`, flips the Hi-Z default per A8, applies the
   accepted §12 changes and the workspace profile change (cp-stream to `opt-level = 3` in dev,
   `engine/Cargo.toml:83-94`), then runs §9 on an idle machine and starts the E3 beauty review.

---

## 11. Risks and mitigations

| # | Risk | Mitigation |
|---|---|---|
| R1 | τ = 6 is coarser than the reviewed E2 shots, which drew level 0 to 512 m (`cp-app/src/scene/hills.rs:167-168`); level 1 starts at 312 m and untextured levels (`TEXTURED_LEVELS = 2`, `layout.rs:79`) at 624 m. Beauty points may drop | far canopy, face ground maps, silhouette and skyline gains; the beauty fixer may lower τ to 5 only if A2 and A17 pass at 5 with ≤ 90 % of the caps (a recorded decision) |
| R2 | generator cost: 3D channels cost about 25 octave evaluations per level-0 column; the E2 hills grew to about 3.7 ms per chunk in their beauty loop (`docs/STAND.md:64-67`); A3 depends on it | family fill (2.24× fewer columns), interval fast paths, band limits, the timing test in WP-B's first three days, the disk cache for levels ≥ 3; fallbacks: fewer octaves, vertical family pairs sharing columns, mask-gated mountain evaluation |
| R3 | wall quads were never measured; keeping opaque and fluid walls may grow the pool and uploads | the bench reports `wall_share`; fallback without contract change: upload walls only for faces that border another level at insert time and re-upload when that changes |
| R4 | LOD lag at 321 m/s (a parent stays until its children are ready; fades take 250 ms) | 0.5 s look-ahead, band 0, atomic family results, prefetch, fast fades when lagging; A4 |
| R5 | frame spikes above 20 ms (large drops, sink bursts, staging growth) | §7.1 caps, the housekeeping thread, no allocation or drop above 64 KiB on the frame thread, coalesced item writes, start-up pipelines; A5 and A6 |
| R6 | two-phase Hi-Z doubles the count-draw cost and may not pay on open terrain; a bug could cull visible chunks | default decided by A8; conservative test with margin; tolerance gate and zero-fragment test; runtime switch; paths B and direct without occlusion |
| R7 | fading items use the two-sided cutout pipeline with discard and no early-Z (gbuffer.wgsl `fs_cutout`) | hidden items culled, ≤ 128 open families, fast fades only when lagging; G-buffer time reported (A7) |
| R8 | point-rule quantisation: Heimat's +Z plane at lattice 21,845 (odd) puts the ground top of flat land at 0, −1, −1, +3, −5, +11, −21, +43, −85, +171, −341 m on levels 0–10 (derived), so each L/L+1 border steps by `2^L` m; fluid walls show water curtains | walls close every step; the pop probe and the coast shot measure it; E7's ocean surface hides coarse water |
| R9 | parallel work: P1 step 0 changes cp-voxel; E4 needs cp-render and cp-app; the beauty loop edits cp-render, cp-app, cp-worldgen and content now | WP-A small and first; exact file lists with free paths for E4; setters instead of struct fields; E3 starts after the E1/E2 acceptance; every builder re-reads files first |
| R10 | holes-probe false results (water surfaces, camera inside geometry) | back-facing slots kept and no edge push in the Holes view; the background flood fill as a second probe; routes above water with clearance |
| R11 | disk-cache corruption, collisions, concurrency | verbatim 64-byte keys, FNV-1a per record, codec validation, the LOCK file and read-only fallback; corrupt = miss |
| R12 | incremental-walk bugs skip a node that should change | full-walk comparison every 64th frame in tests and Deterministic mode; `validate()` every frame in `--check` |
| R13 | cp-jobs behaviour: no reprioritisation, bulk waits, Background starvation, closures dropped by the canceller | pump model, housekeeping thread, no cp-jobs cancellation from the frame thread |
| R14 | the first write after each snapshot copies the region map (O(#regions), `cp-voxel/src/layers.rs:111-119`) | small in E3; measured on the `edits` route |
| R15 | first integration of cp-universe: session-local `BodyId`, lattice vs body frame | salt from the body key; object pose translation −o; unit tests in WP-G |
| R16 | interim orbit lighting (one up, camera-anchored clouds) may score low from orbit | space factor and cloud opacity; the judges are told the E5 scope |
| R17 | face maps without relief weaken far relief shading on faces compared with the E2 plane | relief layers are optional in §6.8 and in the beauty loop's scope |
| R18 | the precision and up-mode changes touch every shader path | legacy defaults; A18 protects the E2 images |
| R19 | Deterministic mode (fixed latency) differs from real time, so shots may hide lag | shots judge looks; real-time benches measure lag and frame time; both are required |
| R20 | `File::try_lock` needs Rust ≥ 1.89 (A: the workspace's stable toolchain is newer; `clippy.toml:10` names clippy 0.1.99) | fallback: `create_new` on `dir/LOCK` holding the process id, with stale-lock detection |

---

## 12. Proposed changes to ARCHITEKTUR.md (not applied; the orchestrator decides)

1. **§6 cp-stream** (`ARCHITEKTUR.md:513-525`): replace the bullet by a summary of this file and a
   reference to `docs/STREAMING.md`: two residency sets (`Streamer`, `SimResidency`); views Primary,
   Omni and Frustum with their own focal length (capped at 1,080 px); the metric with 0.5 s
   look-ahead, silhouette-band and skyline gains and the level-1 floor; hard caps over all grids by
   priority-ordered admission; GPU-timed fades; the exact wall rule; core occlusion shrunk around
   edits; own priority queue and pump jobs; class, memory and disk caches; atomic edit groups.
2. **§6 cp-stream readiness:** "Empty/Opaque children count as ready at once" → "Empty children
   count as ready at once. Opaque children are meshed; a walls-only (dormant) chunk is resident only
   while one of its walls is drawn. A transition waits for its children and for every chunk whose
   walls it switches on."
3. **§6 cp-stream mesh-cache key:** "hash(GeneratorIdentity, GridId, LodKey, content_hash, …, apron
   revision of the chunk and its six neighbours)" → the 64-byte verbatim key of §5.6 (stable hash,
   persistent grid salt from the body key, LodKey, content hash, mesher and format versions,
   `APRON_STAMP_VERSION`, `MESH_CODEC_VERSION`, `apron_stamp`).
4. **§6 cp-stream "one chunk cache per grid":** → per grid an octree of nodes plus entries in one
   shared class cache, one 256 MiB memory mesh cache and the disk cache.
5. **§6 cp-stream simulation reads:** "generate on miss in class Critical" → "a read that misses
   generates on the reading thread at once; cp-stream prefetches anchor boxes in Streaming, pins
   them, and the simulation counts its own blocking reads per thread".
6. **§5.3 walls** (`ARCHITEKTUR.md:466-467`): wall ranges of the opaque and fluid layers are always
   uploaded, cutout wall ranges never; the mask follows the exact rule of §5.3; delete "or the face
   between an edited and an unedited chunk of the same level".
7. **§5.3 Hi-Z and path B** (`ARCHITEKTUR.md:461, 474-475`): specify §6.6; path B and direct draws
   have no occlusion in E3; Hi-Z is the default only if it does not raise GPU P95.
8. **§5.3 draw item:** the `fade` word and `reserved[0..2]` as in §6.2; items that keep no threshold
   are culled; upload lanes and atomic groups; coalesced item writes.
9. **§2.4 and §5.3 precision:** add "Fragment stages take cell, texel, hash and owner-face indices
   from a flat integer base plus a small f32 offset; no absolute position is interpolated in f32."
10. **§2.3, §5.3 and §8 six faces:** per-item up mode; face-planar tint, ground cover and grass
    fringe; per-face ground maps as the interim for macro-map tinting.
11. **§5.1 cp-voxel:** add `SourceShape`, `BaseSource::shape`, the snapshot planning, sampling and
    pinning API, `ReadStats`, `WorldConfig::for_root_level`; a body's world uses `top_level =
    root_level`.
12. **§5.1 `GeneratorIdentity.macro_hash`:** "Hash of every input besides key, version and seed that
    decides the output: parameters, palettes, body shape and baked macro maps" (the code already
    stores parameter and palette hashes, `cp-worldgen/src/lib.rs:38-43`, while
    `cp-voxel/src/base.rs:32` says "baked macro maps").
13. **§5.2 cp-mesh:** add the `ChunkMesh` codec with `MESH_CODEC_VERSION`.
14. **§6 cp-worldgen:** add `cubeplanet:cube_hills` and `cubeplanet:cube_flat` with the cube-frame
    rules of §4.2 (dominant-axis ownership, integer `Z_F`, clamped columns, land bridge); generators
    of cube bodies are bit-continuous across edges; goldens of bounded bodies cover levels
    0..=root_level; relief must fit the root set (validated).
15. **§6 cp-save:** specify the pack store of §7.9 (positioned reads; memmap2 stays unused until
    approved).
16. **§3.1 / §3.2:** cp-stream depends on core, voxel, mesh, jobs and save, with cp-worldgen as a
    dev-dependency only; cp-save's pack store is std-only.
17. **§3.3 profiles** (`ARCHITEKTUR.md:229-230`): add cp-stream to the crates built with
    `opt-level = 3` in dev.
18. **§4.2 cp-jobs usage:** streaming keeps its own priority queue drained by at most
    `bulk_limit + 2` pump jobs; simulation prefetch is Streaming key 0; housekeeping runs on a
    dedicated thread because Background jobs starve under Streaming load.
19. **§7:** keep "GTAO, contact shadows and SSR only below 240 m where no LOD fade occurs"
    (`ARCHITEKTUR.md:593`); the LOD level-1 floor of §7.2 guarantees it at every bias. Add: the TAA
    cut distance is a frame input (cp-app passes `max(50 m, 0.1 · altitude)`); until E5 a space
    factor fades sky, clouds and fog above the atmosphere.
20. **§8 scale ladder** (`ARCHITEKTUR.md:604`): "far cells are 3–6 px" → "far cells are at most 6 px
    (at most 4 px on silhouette and skyline nodes)".
21. **§9 E3 row** (`ARCHITEKTUR.md:622`): replace by gates A1–A20 with their definitions
    (seamless, LOD lag, silhouette deviation, throughput as non-empty chunks per second, budgets in
    every frame).
22. **§2.2 spaces table** (`ARCHITEKTUR.md:57`): "Body: … voxel (x, y, z) spans `[x − o, x − o + 1)`
    m of the body frame; lattice = body + o" (`cp-universe/src/body.rs:8-15`).
23. **§5.4 cpe:** add `cpe probe`, bench schema 2 with paced frames, the cube scene, its routes and
    shots that settle or replay a route deterministically.
24. **content/base/systems/home.toml:** every cube body names `cubeplanet:cube_hills`.
25. **§11 decision log:** the entries D2–D30 below.

---

## 13. Decision log

| # | Decision | Reason | Rejected |
|---|---|---|---|
| D1 | Base design "zukunft" | highest summed judge total (47 + 44 = 91; "durchsatz" 90, "bild" 88); grid- and view-agnostic, replica-agnostic, parallel-buildable | — |
| D2 | Fades are GPU-timed integer ramps on the 16 Bayer thresholds (from "durchsatz") | exact partition in integers; 2 CPU writes per member per transition instead of a rebuild per frame (`state.rs:554-558`); exact reversal | per-frame coverage writes ("zukunft"); CPU-driven nested bands ("bild"): writes for every fading item every frame; float complement pairs: not exact at .5 boundaries (`layout.rs:242-263`) |
| D3 | τ = 6 px on every tier plus the 240 m level-1 floor (from "durchsatz") | the binding contract value and its budgets; keeps §7's fade-free 240 m at every bias and tier | τ = 5 for High ("bild": thin budget margin, Medium/Low still violate §7 at bias 1.5); shrinking the §7 range with bias ("zukunft") |
| D4 | Family fill through a cp-voxel API with exact overlays (from "durchsatz", extended) | 2.24× fewer column evaluations; atomic family results; overlay semantics stay in cp-voxel, so edits need no special case | a "no overrides" precondition; eight independent padded fills |
| D5 | No nested fades; a transition waits for open families; 66 ms fast fades when lagging; instant switches only in Settle mode or a TAA-cut frame | near the ground at ≤ 321 m/s successive level switches lie ≥ 0.49 s apart against 0.25 s fades (derived); a fast fade still dithers | instant catch-up switches at ≥ 2 levels behind ("zukunft": a pop); nested bands (D2) |
| D6 | Exact wall masks; opaque and fluid wall ranges uploaded, cutout never (from "bild") | a mask change needs no re-upload; unneeded walls can show 0.05 px slivers (`layout.rs:86-94`); leaf sheets read as artefacts | generous masks ("durchsatz", "zukunft"); walls of all layers; re-upload on mask change |
| D7 | Dormant walls-only chunks resident only while a wall is drawn (from "durchsatz"); readiness includes every chunk whose walls switch on (from "bild") | saves pool and items for underground chunks; no hole when a coarser neighbour shows air | "Opaque children count as ready at once" (holes); a static sealed-by-margin rule (wrong next to coarser neighbours) |
| D8 | Never-split rules: core box, buried with a 32-cell shell and override check, occlusion with an edit-shrunk occluder (from "zukunft") | conservative at every drawn level; player excavations stay visible | an analytic "Deep" that ignores overrides ("durchsatz") |
| D9 | Views Primary, Omni and Frustum with their own focal length (from "zukunft"); direction-independent LOD for Primary and Omni unless a cap binds | the owner's telescope and map camera (`docs/STAND.md:140-142`); turning never pops | the grazing factor ×1.45 (`src/render/lod.rs:212-241`); in-frustum-only gains and an off-frustum lever ("bild") |
| D10 | Cube-edge silhouette band and skyline gains, direction-independent (from "bild") | silhouettes and the limb are where pops are seen; thin node sets | no silhouette refinement |
| D11 | Hard caps by priority-ordered admission: new splits below 80 %, kept to 100 %, forced merges above; pressure levers as slow levers | budgets hold without flapping at the boundary | a bias governor alone with a failure flag ("durchsatz") |
| D12 | Incremental walk with odometer and slack, checked against a full walk (from "durchsatz") | keeps selection under 0.5 ms with thousands of nodes | a full best-first heap every frame ("zukunft") |
| D13 | cp-stream-owned priorities, K most urgent published per frame, `bulk_limit + 2` Streaming tokens | cp-jobs fixes queue keys at submit (`queue.rs:127-178`) and bulk jobs must not wait on bulk jobs (`lib.rs:74-76`) | one cp-jobs job per chunk (stale queue at 321 m/s) |
| D14 | Dedicated housekeeping thread for disk writes and deferred drops (from "bild") | Background jobs never start while Streaming work is queued (`queue.rs:19-31, 314-328`) | Background jobs ("durchsatz", "zukunft") |
| D15 | Simulation prefetch in Streaming key 0, ≤ 2 pumps, pins, thread-local read stats (from "zukunft") | bounded worker use; exact "physics waited" counting | uncapped Critical prefetch and global cache deltas ("bild": counts its own prefetch misses) |
| D16 | Disk cache: 64-byte verbatim key; std-only PackStore with positioned reads, LOCK, FNV-1a records, per-user directory | no false hits; no `unsafe`; cp-save's declared dependencies (`cp-save/Cargo.toml:10-17`); no cache inside the repository | hashed keys only; memmap2; parking_lot in cp-save; `<cwd>/cache/meshes` |
| D17 | `ChunkMesh` codec in cp-mesh (from "zukunft") | the crate that defines the format validates it | ad-hoc record formats in cp-stream |
| D18 | Cube field from own 3D channels at the reference-surface point, clamped columns, land bridge, no settling, own face tree placer (from "zukunft") | watertight and bit-continuous at edges and corners by construction; independent of the E2 Terrain, which the beauty loop is still changing | per-face 2D Terrains blended at edges ("durchsatz": settling breaks exactness, trees stand on the wrong field, `trees.rs:1183`); reusing E2 Terrain internals ("bild") |
| D19 | Relief scaled by body size and validated against the root set | Kiesel has 1,346 m of headroom; 1.4 km mountains would not fit | one relief for every body |
| D20 | Far canopy stand-in (from "bild") and per-face ground maps (from "zukunft") | forests keep volume and tint at every distance on all six faces | an x-z ground map on a cube; forests that vanish above level 3 |
| D21 | Up mode per item with an integer per-cell owner near edges (from "bild") | fixes ground cover and the grass fringe on all faces without interpolated f32 | a per-object planet frame from interpolated positions ("zukunft") |
| D22 | Integer fragment bases (all designs; "bild"'s formulation) | the current f32 grid position flickers at 1.95 mm spacing (`ARCHITEKTUR.md:126`) | — |
| D23 | Hi-Z history per item index (from "durchsatz"); default decided by measurement; path B without occlusion (from "bild", "zukunft") | stale entries only move items between phases, so no draw-item bits are needed; Hi-Z must pay for its doubled count-draw cost | chunk-slot owners in draw-item words; read-back occlusion on path B |
| D24 | Setters instead of new struct fields | existing struct literals in cp-app and tests keep compiling while E4 and P1 build in parallel | new fields in `FrameInput`, `Lighting`, `ObjectDesc` |
| D25 | Deterministic mode with a fixed 4-frame result latency (from "bild") | reproducible shots and probes with a realistic latency | lockstep ("durchsatz": hides latency); paced replay ("zukunft": not reproducible) |
| D26 | Probes: holes view with the cull fix and the background flood fill; the pop probe validated on `cube_flat`; the bias probe | measures "seamless" and "silhouette deviation" in pixels and validates the probes themselves | a holes view that keeps back-face culling (it would hide cracks) |
| D27 | No GPU grace period (from "zukunft", "durchsatz") | chunks that keep pool space count as resident; a 2 s grace adds about 1,200 at 321 m/s | the 2 s grace ("bild") |
| D28 | Eight work packages (the seven of "zukunft" with cp-app split into runtime and measurement) | parallel building with disjoint files | one package for all of cp-app |
| D29 | cp-core stays unchanged in E3; helpers such as `LodKey` neighbours and box distances are private to cp-stream | fewer shared files with E4 and P1 | new cp-core helpers ("bild") |
| D30 | The E2 scenes keep the static planner; gate A18 protects their images | E2 is accepted on those images | moving the hills scene to cp-stream in E3 |
