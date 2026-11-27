# Cubeplanet Engine – Architecture Contract (v3)

**Status:** binding for all work under `engine/`. Maintained by the lead session. Implement against this
document; if something here is wrong, contradictory or impossible, **report it** (do not silently
deviate). v2 (2026-11-26) incorporates three independent contract reviews (wgpu 30.0.1 reality,
frame budget on the baseline GPU, data model) and the decisions taken while building E0; v3 adds the
physics engine, specified in the companion document `PHYSIK.md`. §11 lists the changes.

The game design lives in `../../docs/SPIELPLAN.md` (German). The old prototype in `../../src` is a
quarry for ideas and tested maths only; nothing in `engine/` may depend on it.

---

## 1. Goals and principles

The engine must render and simulate a hand-made solar system of eight literal **cube-shaped** bodies
(edge 4–41 km, 1 m blocks, six walkable faces) and later procedurally generated neighbouring systems —
seamlessly from a single block up to the galaxy, at 60 fps with all effects on a GTX 1080 at 1080p.

1. **Everything has levels of detail** – geometry, simulation and storage. No cost may scale with
   total world size, only with what is visible or active.
2. **No baked-in planet.** Bodies, systems, blocks and materials are data.
3. **Headless, deterministic simulation.** The client is a view.
4. **GPU-driven rendering.** The frame thread never waits for generation, meshing or disk.
5. **Measurable from day one.** Benchmarks, per-pass GPU timings and golden images are engine features.
6. **One voxel path for all cube-shaped cells.** Terrain, buildings and moving constructs share
   storage, mesh format and draw path. Non-cube and animated blocks (machines, stairs, levers) use an
   instanced model path that is reserved now (`RenderClass::Model`) and built with the machines.
7. **Multiplayer-ready from day one.** Single-player runs as a local server with an in-process
   client. The simulation is authoritative; a client only sends commands and receives state, edit
   receipts and events. No client or renderer code mutates world or simulation state directly, and
   every system must work with several players (several anchors, several cameras, per-player
   inventories and permissions). Co-op itself ships later (§6 `cp-net`).

Baseline hardware: Ryzen 5 9600X (6C/12T), 32 GB RAM, GTX 1080 (8 GB), Windows 11, **Vulkan**.
DX12 is a functional fallback; performance targets and golden images are defined on Vulkan.
No hardware ray tracing and no mesh shaders on the baseline.

---

## 2. Conventions

### 2.1 Units and numeric types
- Metres, seconds, kilograms, radians.
- `f64` for every simulation/world position and time. `f32` only for camera-relative rendering data
  and small local offsets. Voxel coordinates are `i32`.
- Maths library: `glam` (re-exported by `cp_core::math`), built with the `libm` feature so its
  transcendental functions are platform-independent.
- Deterministic crates (§2.5) use the `f64` glam types (`DVec2/3`, `DQuat`, `DMat3/4`, `IVec3`) and
  scalar `Vec3` only. `Vec3A`, `Vec4`, `Quat`, `Mat4`, `Affine3A` are for rendering.

### 2.2 Spaces and keys
| Space | Type | Meaning |
|---|---|---|
| Galaxy | `GalaxyPos { sector: [i64; 3], local: DVec3 }` | sectors of 2^40 m |
| System | `DVec3` | inertial frame, origin at the system barycentre |
| Body | `DVec3` / `IVec3` | body-fixed rotating frame, origin at the cube centre; voxel `(x,y,z)` spans `[x, x+1)` m |
| Construct | `DVec3` / `IVec3` | local grid of a moving construct; pose relative to its parent frame |
| Chunk | `LodKey { level: u8, pos: IVec3 }` | 32³ cells, cell size `2^level` m, min corner at `pos * (32 << level)` voxels |
| Render | `Vec3` | origin near the camera, axes of the camera's reference frame |

- `GridId { Body(BodyId), Construct(EntityId) }` names a voxel grid. `ChunkKey { grid: GridId, lod:
  LodKey }`, `CellAddress { grid: GridId, cell: IVec3 }`. `FrameId { System(SystemId), Body(BodyId),
  Construct(EntityId) }`.
- Right-handed everywhere. There is no global "up": up is the local direction of `-gravity`.
- `CHUNK_BITS = 5`, `CHUNK_SIZE = 32`, `CHUNK_VOLUME = 32768`, `MAX_LOD_LEVEL = 20`,
  `VOXEL_LIMIT = 2^30`: valid keys lie entirely inside `[-2^30, 2^30)` voxels on every axis.
- Cell index inside a chunk: `index = x + z * 32 + y * 1024`. Child index of the eight children of a
  key: `dx + 2 * dz + 4 * dy` (same axis ranking).
- Floor semantics: `chunk = voxel >> 5`, `local = voxel & 31`.
- **Octree roots.** `BodyDef.root_level` = smallest `L` with `32 * 2^L >= half edge`. A body's root
  set is the eight chunks with `pos ∈ {-1, 0}³` at `root_level` (Home, 40.96 km: level 10). Body
  edges are an even number of metres.
- **Lattice offset.** `BodyDef.grid_offset: i32` (same on all axes) shifts the voxel lattice against
  the cube centre so that the mean surface of each face does not lie on a chunk boundary at levels
  4–8. Gravity, frames and saves keep using the cube centre.

### 2.3 Faces, quads, orientation, texturing
- `Face` indices: `0:+X 1:-X 2:+Y 3:-Y 4:+Z 5:-Z`. `axis = index >> 1`, `negative = index & 1`.
- Tangent axes of a face with normal axis `n`: `u_axis = (n + 1) % 3`, `v_axis = (n + 2) % 3`
  (`u × v = +n`): X→(Y,Z), Y→(Z,X), Z→(X,Y). Negative faces use the same axes.
- A quad has a min cell `(x,y,z)`, width `w` along `u`, height `h` along `v`. Its plane is at
  `cell_n + 1` for positive faces and `cell_n` for negative faces. Corners in `(u,v)`:
  `c0=(0,0) c1=(w,0) c2=(w,h) c3=(0,h)`.
- **Drawing order.** Quads are drawn indexed with 4 vertices each and one static index pattern
  `[0,1,2, 0,2,3]` per quad. Vertex `k` of a quad is corner `CORNER_MAP[k]` with
  `CORNER_MAP_POSITIVE = [0,1,2,3]` and `CORNER_MAP_NEGATIVE = [0,3,2,1]`, which makes every
  triangle counter-clockwise seen from outside. Back faces are culled.
- `from_dominant_axis`: ties go to the smaller axis index; a zero component counts as positive.
- **Orientation (24).** `Orientation { up: Face, turn: u8 }`, `turn ∈ 0..4`,
  `index = up.index() + 6 * turn`. Rotation local→world = `R(up_world, turn × 90°) ∘ T(up)`, where
  `R` turns counter-clockwise seen from the tip of the world up vector and `T` is:

  | up | (x, y, z) local → world |
  |---|---|
  | +Y | ( x,  y,  z) |
  | -Y | ( x, -y, -z) |
  | +X | ( y, -x,  z) |
  | -X | (-y,  x,  z) |
  | +Z | ( x, -z,  y) |
  | -Z | ( x,  z, -y) |

  A block's **front** is its local `+Z` face. API: `ALL: [Orientation; 24]`, `IDENTITY`,
  `rotate_vec`, `world_face(local)`, `local_face(world)`, `compose`, `inverse`,
  `from_up_front(up, front) -> Option<Orientation>`, `uv_rot(world_face)`. `up6` blocks use `turn = 0`;
  `axis3` blocks use `Up(+a)` for axis `a`.
- **Texture axes of a block-local face** (image right = `tex_u`, image up = `tex_v`,
  `tex_u × tex_v = n`): sides (±X, ±Z) `tex_v = +Y, tex_u = tex_v × n`; top `tex_u = +X, tex_v = -Z`;
  bottom `tex_u = +X, tex_v = +Z`.
- **Texturing frame of a quad** (normative implementation: `Face::uv_to_tex`): positive faces
  `(a, b) = (u, v)`, negative faces `(a, b) = (v, u)`, so `a × b` is the outward normal. Texture
  coordinates `(s, t)` in metres = `0:(a,b) 1:(b,-a) 2:(-a,-b) 3:(-b,a)` for `uv_rot = 0..3`.
  `Orientation::uv_rot(world_face)` is the unique value that maps the rotated block-local texture
  axes onto this frame. Sampling uses `(fract(s), 1 - fract(t))` because image row 0 is the top.
  One texture tile per metre at every LOD.

### 2.4 Projection and depth
- View space right-handed, camera looks along `-Z`.
- Projection: `cp_core::math::perspective_infinite_reverse_rh(fov_y, aspect, z_near)` with
  `Z_NEAR = 0.1` (identical to glam's `camera::rh::proj::directx::perspective_infinite_reverse`; never
  the `vulkan` variant, never the deprecated `Mat4::perspective_infinite_reverse_rh`). WebGPU NDC:
  x right, y up, z in [0,1]. Depth `d = z_near / -z_view` (1 at the near plane, 0 at infinity).
- `Depth32Float`, cleared to `0.0`, compare `Greater`; shadow maps use the same convention with
  comparison `GreaterEqual`; Hi-Z reduces with `min` (farthest).
- Texture origin top-left: `uv = ndc.xy * vec2(0.5, -0.5) + 0.5`.
- All vertex positions reach the GPU camera-relative (§5.3). Absolute positions never enter an `f32`.

### 2.5 Determinism rules
Apply to `cp-core`, `cp-voxel`, `cp-worldgen`, `cp-universe`, `cp-physics`, `cp-sim`, `cp-save`,
`cp-content`:
1. Transcendental maths only through `cp_core::det` (backed by an exactly pinned `libm`). Forbidden
   in these crates and enforced with clippy `disallowed-methods`: `f64::{sin, cos, tan, asin, acos,
   atan, atan2, sinh, cosh, tanh, exp, exp2, ln, log2, log10, powf, powi, cbrt, hypot, mul_add}` and
   the `f32` equivalents. `min`/`max`/`clamp` only through `det::{min, max, clamp}` (defined
   behaviour for ±0 and NaN). `+ - * /` and `sqrt` are IEEE-exact and allowed.
2. No result may depend on `HashMap` iteration order, thread scheduling, wall-clock time, pointer
   values or `usize` width (64-bit targets only). Iterate sorted or use `BTreeMap`; apply job results
   in key order.
3. Randomness only from seeded `cp_core::hash::Rng` / hash functions.
4. Serialized and hashed data use fixed-width little-endian integers and explicit field order.
5. **Canonical form.** Equality and hashes of voxel data use the canonical encoding: cells in index
   order, string palette in order of first occurrence, minimal bit width, uniform chunks collapsed.
   Session-local numeric ids never enter a hash.
6. Generator output for a given `(GeneratorIdentity, LodKey)` is identical on every machine.
   Macro maps are baked artefacts identified by their hash (§6).

Rendering and the client may be non-deterministic.

### 2.6 Errors, logging, profiling
- Libraries return `Result<T, E>` with `thiserror` enums. Bad external input (files, GPU loss,
  content packs) is never a panic. Panics are for violated internal invariants only.
- No `unwrap()` outside tests. `expect("reason")` only where failure is a programming error.
- Logging through the `log` crate; only `cp-app` prints (clippy `print_stdout` and `print_stderr`).
- wgpu's default uncaptured-error handler panics and must never stay active (§4.5).
- CPU profiling through `cp_core::profile` scopes; GPU timing through timestamp queries per pass.

### 2.7 Tests
- Every crate has unit tests for its stated guarantees. Hot algorithms are checked against a slow,
  obviously-correct reference implementation.
- Tests must not need a window. Tests that need a GPU adapter skip cleanly (log + return) when none
  is available.
- Never weaken or delete a failing test to get green; fix the cause or report it. A pinned golden
  value may be re-pinned only when the change is intended and the reason is stated in the report.
- Source files use LF line endings.

---

## 3. Workspace

### 3.1 Layout
```
engine/
  Cargo.toml  rustfmt.toml  clippy.toml  check.ps1  check.sh
  crates/<crate>/{Cargo.toml, src/, tests/}
  content/base/{pack.toml, blocks/, materials/, textures/, systems/}
  docs/{ARCHITEKTUR.md, gpu/, img/, bench/}
```

| Crate | Responsibility | Internal deps |
|---|---|---|
| `cp-core` | maths, ids, keys, coordinates, faces, orientation, time, hashing/RNG, noise, frames, block/quad wire types, profiling | – |
| `cp-jobs` | prioritised job system, cancellation, panic containment | – |
| `cp-proctex` | procedural 16 px PBR textures (own art, release-safe) | core |
| `cp-content` | data registry: blocks, states, materials, textures, derived tables; content packs | core |
| `cp-voxel` | chunk storage, edit overrides, LOD pyramid, `BaseSource` trait, snapshots, queries | core |
| `cp-mesh` | binary greedy meshing of padded chunks → packed quads | core, voxel |
| `cp-worldgen` | `BaseSource` implementations: test generators (E1), Terrain 2.0 (E6) | core, voxel, jobs |
| `cp-universe` | bodies, orbits, rotation, gravity potential, atmosphere, SOI table, Kepler conics, star catalogue | core |
| `cp-physics` | rigid bodies of voxel grids, collision, contact and joint solver, prescribed motion, forces, characters and items, stages, physics save section (`PHYSIK.md`) | core, voxel, universe, jobs |
| `cp-stream` | LOD selection, residency, streaming scheduler, mesh cache, upload budgets | core, voxel, mesh, jobs, save |
| `cp-render` | wgpu renderer, frame graph, all shaders | core |
| `cp-sim` | simulation kernel: commands, tick, sites, machines and networks, joint/train/route schedules, player data; drives `cp-physics` | core, voxel, universe, physics, jobs |
| `cp-save` | manifest + generation files, pack store, sections, migrations | core, voxel |
| `cp-app` | binary `cpe`: window, input, debug views, headless shots, benchmarks, tools | all |

`cp-render` knows only `cp-core` wire types; `cp-app` glues content → textures → renderer.
`cp-voxel`, `cp-mesh` and `cp-physics` take plain tables (§4.4, `BlockTables::physics` from P1),
not the registry.

### 3.2 Dependencies (offline-first, pinned by `Cargo.lock`)
Only crates already in the local cargo cache unless the lead approves otherwise:
`glam 0.33` (features `bytemuck`, `serde`, `libm`), `bytemuck 1` (`derive`), `serde 1` (`derive`),
`toml 1`, `thiserror 2`, `bitflags 2`, `log 0.4`, `libm =0.2.16`, `parking_lot 0.12`,
`crossbeam-channel 0.5`, `rustc-hash 2`, `smallvec 1`, `arrayvec 0.7`, `memmap2 0.9`, `flate2 1`,
`png 0.18`, `futures-lite 2`, `wgpu =30.0.1` (default-features off; `std`, `wgsl`, `vulkan`, `dx12`,
`metal`), `winit 0.30`. naga is used through `wgpu::naga` (no extra dependency).
Not available offline: rayon, criterion, zstd, lz4, noise, pollster, egui.

Per-crate external deps: core: glam, bytemuck, serde, libm, bitflags, thiserror · jobs: parking_lot,
crossbeam-channel, log · proctex: – (dev: png) · content: serde, toml, thiserror, log, rustc-hash ·
voxel: bytemuck, rustc-hash, parking_lot, thiserror, smallvec · mesh: bytemuck, smallvec ·
worldgen: serde, thiserror, log · universe: serde, thiserror · physics: smallvec, arrayvec,
thiserror, log · stream: parking_lot,
crossbeam-channel, rustc-hash, log · render: wgpu, bytemuck, log, thiserror, parking_lot,
futures-lite, arrayvec · sim: log, thiserror, smallvec · save: flate2, memmap2, serde, toml,
thiserror · app: winit, png, log, futures-lite, serde, toml.

### 3.3 Lints, formatting, profiles
- Edition 2024, stable toolchain. `rustfmt.toml`: `edition = "2024"`, `max_width = 100`,
  `newline_style = "Unix"`.
- `[workspace.lints]`: `unsafe_code = "deny"`, clippy `all = warn` (priority -1), `dbg_macro`/`todo`/
  `unimplemented` = deny, `unwrap_used`, `print_stdout`, `print_stderr` = warn (allowed in `cp-app`).
  `clippy.toml`: `allow-unwrap-in-tests`, `allow-expect-in-tests`, `disallowed-methods` per §2.5
  (`cp_core::det` and non-deterministic crates opt out with a scoped `#[allow]` and a comment).
  `unsafe` needs lead approval and a `// SAFETY:` comment.
- From P1, clippy also disallows glam's `DVec2/DVec3::{min, max, clamp, min_element, max_element}`
  (they call `f64::min/max`); `cp_core::det` gains `vmin/vmax/vclamp`.
- Profiles: `dev` → `opt-level = 1`, `debug = "line-tables-only"`; dependencies and the hot crates
  (`cp-core`, `cp-voxel`, `cp-mesh`, `cp-worldgen`, `cp-physics`) → `opt-level = 3` in dev; `release` →
  `opt-level = 3`, `lto = "thin"`, `debug = "line-tables-only"`. Benchmarks use release builds only.

### 3.4 Commands
- Whole workspace (lead / integrator / acceptance only):
  `powershell -NoProfile -ExecutionPolicy Bypass -File check.ps1` (= `cargo fmt --all -- --check`,
  `cargo clippy --workspace --all-targets -- -D warnings`, `cargo test --workspace`).
- Per crate (everyone else): `cargo fmt -p <crate>`, `cargo clippy -p <crate> --all-targets -- -D warnings`,
  `cargo test -p <crate>`.

---

## 4. E0 crates (built; v2 deltas marked **Δ**)

### 4.1 `cp-core`
Modules: `math`, `ids`, `coords`, `face`, `block`, `quad`, `time`, `hash`, `det`, `noise`, `frames`,
`profile` as built in E0 (names and decisions recorded in the E0 report: `LodKey::parent`/
`ancestor_at` return `Option`, `children` requires level > 0, `PackedQuad::new` returns `Result`,
`IAabb` half-open, `DAabb`/`Aabb` closed, `math::perspective_infinite_reverse_rh`, `math::Z_NEAR`).

**Δ v2 (implement before E0 is accepted):**
- `ids`: `BlockStateId::UNCHANGED = u32::MAX` (reserved sentinel, never a registry state);
  `GridId`, `CellAddress`, `FrameId::Construct` per §2.2; `ChunkKey { grid, lod }`.
- `face`: 24-element `Orientation` per §2.3 (struct `{ up, turn }`), `uv_rot` for all 24 × 6 derived
  geometrically and tested against a brute-force derivation; tie rule of `from_dominant_axis`.
- `block`: `BlockFlags: u16 { SOLID, OPAQUE, CUTOUT, FLUID, EMISSIVE, ENTITY, STRUCTURAL }`
  (`CULL_SELF` removed; `CUTOUT`/`FLUID` are derived from the render class by cp-content);
  `RenderClass { None, Cube, Cutout, Fluid, Model }` (`Model` reserved: the mesher emits nothing for it
  yet); `LodClass { Empty, Fluid, Solid }`, `LodInfo { class: LodClass, proxy: BlockStateId }`.
- `quad`: `PackedQuad` v2 —
  `w0`: bits 0–4 x, 5–9 y, 10–14 z, 15–19 `w-1`, 20–24 `h-1`, 25–27 face, 28–31 reserved (0);
  `w1`: bits 0–15 material, 16–17 `uv_rot`, 18 `authored`, 19–30 class payload (0 for cube and
  cutout), 31 reserved (0). The border-wall bit is gone (walls are separate ranges, §5.2).
  `CORNER_MAP_POSITIVE/NEGATIVE`, `QUAD_INDEX_PATTERN = [0,1,2,0,2,3]`, `QUAD_VERTICES = 4`,
  `QUAD_INDICES = 6` replace the 6-vertex tables; `corners()` unchanged.
- `det`: `min`, `max`, `clamp` with documented ±0/NaN behaviour; `powi` with fixed multiplication
  order; a golden test hashes a grid of every `det` function.
- Resolve the open review findings of E0 (profile records of exiting threads, `GalaxyPos::normalized`
  for negative subnormal offsets, glam trigonometry via the `libm` feature).

### 4.2 `cp-jobs`
As built in E0, including `JobSystem::shutdown(&self)` (cancels queued jobs, joins workers, works
while clones exist). Worker policy for streaming: `threads = logical cores - 3`; at most
`physical cores - 1` jobs of class `Streaming` or `Background` run at once so the frame thread keeps
a physical core.

### 4.3 `cp-proctex`
As built in E0 (recipes, presets per species/ore, tint-neutral recipes documented, contact sheet).
Quality bar unchanged: seamless tiling, limited harmonious palette, one art direction, clustered
pixels, readable at 16 px. **Δ:** every ore must be distinguishable from every other ore at a glance.

### 4.4 `cp-content`
```rust
impl Registry {
    pub fn load(pack_dirs: &[&Path]) -> Result<Registry, ContentError>;  // later packs override
    pub fn builtin_minimal() -> Registry;
    pub fn state(&self, key: &str) -> Option<BlockStateId>;  pub fn state_key(&self, id) -> &str;
    pub fn state_count(&self) -> usize;
    pub fn tables(&self) -> &BlockTables;               // Δ: one bundle of Arc'd per-state tables
    pub fn rotated(&self, id: BlockStateId, o: Orientation) -> BlockStateId;   // Δ
    pub fn blocks(&self) -> &[BlockDef];  pub fn materials(&self) -> &[MaterialDef];
    pub fn material(&self, key: &str) -> Option<MaterialId>;  pub fn textures(&self) -> &[TextureDef];
    pub fn content_hash(&self) -> u64;
}
pub struct BlockTables {                     // indexed by BlockStateId, immutable after load
    pub flags: Arc<[BlockFlags]>, pub render: Arc<[BlockRender]>,
    pub cull_group: Arc<[u16]>, pub lod: Arc<[LodInfo]>, pub material_color: Arc<[[f32; 3]]>,
}
```
- **Key grammar** (fixed forever once saves exist): `namespace:name[prop=value,…]`; names
  `[a-z0-9_]`, values `[a-z0-9_+-]`, properties sorted by name, **only non-default values are
  written**, the bare key is the default state. Orientation kinds: `none`; `up6` → `up ∈
  {+x,-x,+y,-y,+z,-z}` (default `+y`); `axis3` → `axis ∈ {x,y,z}` (default `y`); `rot24` → `up`
  (default `+y`) and `turn ∈ {0,1,2,3}` (default `0`). `cubeplanet:air` is state 0. Ids are assigned
  by sorted state key, so comparing ids compares keys; ids are session-local.
- **Derived tables:** `flags` (SOLID/OPAQUE/EMISSIVE/STRUCTURAL from the block def; CUTOUT/FLUID
  from the render class; ENTITY reserved), `render` (per world face, with `uv_rot`), `cull_group`
  (`cull_group = "<name>"` in the block def, groups numbered by sorted name, 0 = none), `lod`
  (`class` from the block: opaque/cutout solids → Solid, fluids → Fluid, air → Empty; `proxy` =
  the state itself for cube-shaped states, `lod_proxy = "<state key>"` for others), and
  `material_color` (linear average albedo of the face materials, filled by cp-app from cp-proctex
  through `Registry::set_material_colors`).
- `rotated(state, o)`: the state whose orientation is `o ∘ orientation(state)` projected onto the
  block's orientation kind (`up6` drops the turn, `axis3` keeps the axis, `none` is unchanged).
- Block defs may carry an `[ext.<namespace>]` table that the engine stores without interpreting it.
- Hot reload may change textures and material parameters only; anything that adds, removes or
  renames states requires a restart (remap comes with the save format).
- Base pack: as built in E0; ice is opaque until a translucent layer exists; logs are `axis3` and
  `STRUCTURAL`; leaves `cull_group = "leaves"`; water and lava have their own cull groups.

### 4.5 GPU context and capability report (`cp-render::gpu`, built in E0)
- `GpuContext::create_instance(&GpuOptions) -> Result<GpuInstance, GpuError>` creates the one
  instance of the process (`CP_BACKEND=vulkan|dx12` restricts the backends; flags from
  `GpuOptions.flags`, default `InstanceFlags::from_build_config()`; the effective flags appear in the
  report). Constructors: `GpuContext::new` (headless), `with_surface` (creates instance, surface and
  device in one call), `with_instance` (surface created by the caller from that instance), each with
  a `_blocking` variant.
- Adapters: Vulkan first, then DX12; discrete GPU wins unless `PowerPreference::LowPower`; adapters
  that cannot present to the surface are skipped.
- Features: a **whitelist** intersected with the adapter: `MULTI_DRAW_INDIRECT_COUNT`,
  `INDIRECT_FIRST_INSTANCE`, `TIMESTAMP_QUERY`, `TIMESTAMP_QUERY_INSIDE_ENCODERS`,
  `TIMESTAMP_QUERY_INSIDE_PASSES`, `RG11B10UFLOAT_RENDERABLE`, `FLOAT32_FILTERABLE`,
  `TEXTURE_FORMAT_16BIT_NORM`, `TEXTURE_COMPRESSION_BC`, `DEPTH_CLIP_CONTROL`, `DUAL_SOURCE_BLENDING`,
  `IMMEDIATES`, `TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES`. `PIPELINE_CACHE` is not requested
  (its creation is `unsafe`).
- Limits start from `Limits::default()`; raised to `min(needed, adapter)`: `max_buffer_size`,
  `max_storage_buffer_binding_size` (≥ quad pool), `max_texture_array_layers` (adapter maximum),
  `max_immediate_size` (≤ 128), `max_sampled_textures_per_shader_stage`,
  `max_storage_textures_per_shader_stage`, `max_color_attachment_bytes_per_sample`, `max_bind_groups`.
  No buffer above 4 GiB.
- Error handling: `on_uncaptured_error` and the device-lost callback are installed by every
  constructor; shader and pipeline creation runs inside a validation error scope so a bad shader is an
  `Err`. DX12 uses `Dx12Compiler::Fxc` unless `CP_DXC=<path to dxcompiler.dll>` selects DynamicDxc;
  the choice is reported.
- `GpuCaps` derives the draw path (§5.3): **A** = Vulkan with `MULTI_DRAW_INDIRECT_COUNT` and
  `INDIRECT_FIRST_INSTANCE`; **B** = everything else. `cpe caps [--out f.md] [--json f.json]`.

---

## 5. E1–E2 crates

### 5.1 `cp-voxel` (E1)
**Chunk data**
- `ChunkData = Uniform(BlockStateId) | Paletted { palette, bits: 1|2|4|8|16, data: Box<[u64]> }`
  with `get/set/fill/from_fn/decode_into/compact/canonicalize/memory_bytes`, palette growth and
  collapse to `Uniform`. `PartialEq` is semantic (cell by cell). `write_canonical(out, key_of)`
  produces the canonical encoding of §2.5.
- `CellMask` = 32768 bits (`[u64; 512]`), bit = cell index.
- `VoxelChunk { blocks: ChunkData, authored: Option<Box<CellMask>> }` – `authored` marks cells that
  carry a player override at this level. (A dynamic fluid channel is added with the fluid milestone;
  until then fluids are ordinary states.)
- `PaddedChunk`: `34³` states, cell `(x,y,z)` of the chunk at padded index
  `(x+1) + (z+1)*34 + (y+1)*1156`; face-halo cells are exact copies of the neighbour cells at the
  same level, edge and corner halo cells are unspecified.

**Base source (generator interface, implemented by cp-worldgen)**
```rust
pub struct GeneratorIdentity { pub key: String, pub version: u32, pub seed: u64, pub macro_hash: u64 }
pub enum NodeClass { Empty, Opaque, Mixed }
pub trait BaseSource: Send + Sync {
    fn identity(&self) -> &GeneratorIdentity;
    /// Fills `out` with the cells of `region` (cells of `level`, x fastest, then z, then y).
    fn fill(&self, level: u8, region: IAabb, out: &mut [BlockStateId], cancel: &dyn Fn() -> bool)
        -> Result<(), Cancelled>;
    /// Exact-conservative for THIS chunk at its level: Empty ⇒ every cell is air,
    /// Opaque ⇒ every cell is OPAQUE. Mixed is always allowed.
    fn classify(&self, key: LodKey) -> NodeClass;
    /// Conservative: true only if no cell of any descendant at any level is non-air.
    fn subtree_empty(&self, key: LodKey) -> bool;
}
```
Rules: a cell at level `L` is solid iff the generator's band-limited field at the cell centre says
so (the same rule on every level, no bias); features (trees, boulders) are placed independently of
the level and evaluated pull-based within a declared maximum radius; every feature class names the
coarsest level at which it is still rasterised and its stand-in above it. LOD consistency is
measured: for random keys, `fill(parent)` against the downsampled children – surface offset ≤ 1
coarse cell, differing-cell fraction below a pinned threshold.

**Edits = sparse overrides**
- `EditStore` per grid stores override layers, never whole copies. Level 0: per touched chunk a
  `ChunkData` whose cells are `UNCHANGED` or an explicit state (`AIR` = dug). `composite(L0) =
  override.or(base)`.
- **Pyramid** (derived, rebuildable): level `L+1` overrides are a pure function of the eight level-`L`
  composites and the base: a parent cell carries an override only if at least one of its children
  carries one and the downsampled result differs from `base(L+1)`. `composite(L) = O_L.or(base(L))`.
  The simulation never reads levels above 0.
- `apply(&mut self, edits: &[CellEdit], base: &dyn BaseSource) -> EditReceipt` (`CellEdit { cell,
  state }`, `UNCHANGED` reverts a cell). The receipt lists every dirty `LodKey` (level-0 chunk,
  face neighbours whose halo contains the cell, every affected pyramid chunk) and class changes.
  In E1 the pyramid is updated synchronously.
- **Downsample rule** (normative; input = 8 composites, child index `dx + 2dz + 4dy`, tables
  `lod`, `flags`): per parent cell count the classes of its 8 children. `structural` = number of
  Solid children that are `STRUCTURAL` or authored. The parent is **Solid** if `solid ≥ 4`, or
  `structural ≥ 1` when the parent level ≤ 2, or `structural ≥ 2` above; else **Fluid** if
  `solid + fluid ≥ 4`; else **Empty**. Material: the most frequent `proxy` among Solid children that
  touch a non-Solid child within the 2×2×2 group (all Solid children if none touches); fluid parents
  take the most frequent fluid state; ties → smallest id. A parent cell is authored iff any child is.

**Snapshots**
- `VoxelWorld` (per grid) owns base, `EditStore` and a single-flight cache of generated base chunks
  (bounded by bytes). `snapshot() -> VoxelSnapshot` is O(1): a persistent map grid → region (16³
  chunks) index → `Arc` chunk. The first write after a snapshot copies one region index and the
  written chunk, never a region's chunk data.
- `VoxelSnapshot`: `classify(key)` (base ⊕ overrides), `sample(key) -> Arc<VoxelChunk>`,
  `sample_padded(key) -> PaddedChunk`, `revision(key) -> u64` (changes iff the chunk or its 1-cell
  halo changed). `VoxelWorld` is mutated only by its owner (the simulation thread) at tick boundaries.
- Queries on a snapshot at level 0: `raycast(origin, dir, max_dist) -> Option<RayHit { cell, face,
  state, distance }>` (DDA, exact at chunk borders and negative coordinates) and
  `overlaps(aabb, flags) -> bool`.

### 5.2 `cp-mesh` (E1)
- Input: one `PaddedChunk`, the chunk's `authored` mask, its level and `MeshTables { flags, render,
  cull_group, lod }`. No other chunk is read.
- Face rule: a face of cell A towards B is emitted iff A is renderable (Cube, Cutout or Fluid) and B
  does not hide it. B hides A iff B is `OPAQUE`, or `cull_group(A) != 0 && cull_group(A) ==
  cull_group(B)`. `Model` cells emit nothing yet.
- Cutout is a class of levels 0 and 1 only; from level 2 cutout states are meshed into the opaque
  layer (the renderer draws them with their material colour).
- Binary greedy meshing over `u64` bit columns; merge key `(material, uv_rot, authored)`; payload 0.
- **Walls:** for each of the six chunk boundaries, the boundary faces that would exist if the halo
  were air, but were culled by the real halo, are emitted as wall quads.
- Output `ChunkMesh { quads: Vec<PackedQuad>, layers: [LayerRanges; 3] }` for opaque, cutout, fluid.
  Within a layer: six regular face ranges in storage order `+X, +Y, +Z, -X, -Y, -Z`, then six wall
  ranges in the same order; quads inside a range sorted by (plane along the normal, v, u). Each layer
  records its tight cell bounds. `MESH_FORMAT_VERSION` and `MESHER_VERSION` constants.
- Works identically for every level and for construct grids. Reference test: the greedy output
  covers exactly the face set of a naive one-face-per-cell mesher, as a set of
  `(cell, face, material, uv_rot, authored)`; wall set checked the same way. Target ≤ 0.2 ms per chunk
  (release, typical terrain).

### 5.3 `cp-render` (E2)
**Data on the GPU**
- *Quad pool*: one storage buffer of `PackedQuad`, size fixed at start-up (quality tier, default
  128 MiB, ≤ `max_storage_buffer_binding_size`), range-allocated per chunk (all layers contiguous,
  size classes of 256 quads). Created and bound once; never reallocated in game.
- *Static index buffer*: `QUAD_INDEX_PATTERN` repeated for the largest range (u32).
- *Draw items* (64 bytes): `origin: vec3<i32>` (level-0 voxels, grid-local), `lod_flags` (level,
  layer, wall mask), packed cell AABB, `quad_base`, six regular counts and six wall counts (as six
  `u32` in WGSL), `object`, `fade`.
- *Objects*: one per grid that is drawn: integer `anchor` (grid-local voxel), `local_to_clip`,
  `prev_local_to_clip`, `local_to_render`, composed in `f64` on the CPU. The anchor is the voxel that
  contains the camera when the camera is inside the grid's bounds (expanded by 1 km), otherwise the
  nearest voxel of those bounds. Vertex position = `vec3<f32>(origin - anchor + cell * cell_size)`
  computed in `i32` first, then `local_to_clip` (which carries the fractional camera offset).

**Drawing** (one shader set for both paths)
- Path A: a compute pass culls draw items (frustum, face visibility, later Hi-Z) and writes, per
  visible face range, `DrawIndexedIndirectArgs { index_count: 6·n, instance_count: 1, first_index: 0,
  base_vertex: 0, first_instance: slot }` and `buckets[slot] = (draw_item, first_quad)`; compaction by
  count, prefix sum and scatter into per-level segments (level 0 first: deterministic and near to
  far); one `multi_draw_indexed_indirect_count` per view.
- Path B: the same culling on the CPU into a compact indirect buffer with a known count;
  `multi_draw_indexed_indirect`; optional occlusion from last frame's read-back Hi-Z.
- The vertex shader takes `buckets` as vertex buffer 0 (`VertexStepMode::Instance`, `Uint32x2`,
  addressed by `first_instance`). It never uses `@builtin(instance_index)` and never depends on a
  non-zero `base_vertex`/`first_vertex` (DX12 does not add them in count draws).
  `quad = quads[bucket.y + (vertex_index >> 2)]`, corner = `CORNER_MAP[face sign][vertex_index & 3]`.
- Wall ranges are emitted only when the item's wall mask has the face (LOD boundary, or the face
  between an edited and an unedited chunk of the same level).
- **G-buffer:** albedo `Rgba8UnormSrgb`, normal + roughness `Rgb10a2Unorm` (octahedral), material
  `Rgba8Unorm` (incl. emissive strength), depth `Depth32Float`. Only moving objects write velocity
  (`Rg16Float`); static motion is reconstructed from depth.
- **Pixel-art sampling:** one sampler, linear mag/min/mip, `Repeat`, anisotropy from the tier; the
  fragment shader snaps UVs to texel centres with an analytic one-pixel edge ramp and samples with
  `textureSampleGrad` using the gradients of the unsnapped UV.
- **Hi-Z** (E3): `R32Float` pyramid built by compute from the depth buffer (`textureLoad`, `min`),
  each mip through its own single-mip storage view; two-phase culling.

**No shader stutter:** every render and compute pipeline is created at start-up (behind a loading
screen if needed), never inside the frame loop. Materials are data (texture arrays + material table),
not shader variants, so the pipeline count stays small (target: < 64 pipelines in total); a pipeline
that would only be needed later is still created up front. The renderer reports pipeline count and
start-up compile time.

**Infrastructure:** frame graph (declared passes, transient texture pool, resize), WGSL with a small
`#include` preprocessor, every module validated in unit tests with `wgpu::naga` using the device's
capability set, an adapter-gated test that creates every pipeline on each available backend,
headless render target with read-back, debug views (albedo, normal, depth, LOD level, overdraw, chunk
bounds, draw path), own bitmap-font overlay, GPU timing: two timestamp indices per pass in one query
set (≤ 4096), resolved at 256-byte offsets into a read-back ring.

### 5.4 `cp-app` (E2)
Binary `cpe`. Subcommands: `caps`, `proctex-sheet`, `content-check` (E0); `run` (window), `shot
--scene <name> --out <png>`, `bench --scene <name> --frames N --out <json>` (E2). Free-fly camera
whose speed scales with altitude. Scenes are deterministic camera scripts.

---

## 6. Later crates *(interface level)*

- **`cp-worldgen`** – implements `BaseSource`. E1: deterministic test generators (flat, rolling
  hills from `cp_core::noise`, a structure test field) for meshing and rendering. E6 (Terrain 2.0):
  planet history → climate per face → 12–20 regions per face → stream-power erosion and hydrology →
  biomes → landmarks with a spacing rule → strata, ores, caves → detail. Generation is column-first;
  3D fields are sampled on a 4 m lattice or built from analytic shapes; generators poll `cancel` per
  column. **Macro maps** per face (16 m texels for a 41 km body: height, flow, climate, region,
  albedo, normal) are baked artefacts with a hash, cached on disk, shipped for the hand-made system.
  **Stamps** (analytic CSG edits such as craters, evaluated at every level) sit between base and
  overrides. A save pins per body its `GeneratorIdentity` and the content hash; released generator
  versions stay runnable; every change of generator output bumps `version` (enforced by golden
  hashes over fixed keys at all levels).
- **`cp-universe`** – `SystemDef`/`BodyDef` (edge, `root_level`, `grid_offset`, sea level, generator)
  from TOML; analytic ephemeris `body_frame(body, t)`; gravity potential blending cube → sphere
  (port of the prototype's `src/physics/mod.rs`); procedural star catalogue in galaxy coordinates.
- **`cp-stream`** – one chunk cache per grid with two residency sets: **render** (LOD selection) and
  **simulation** (level-0 boxes around registered anchors: players, dynamic constructs, sites with
  voxel needs). Simulation reads are blocking and deterministic (generate on miss in class
  `Critical`); the frame thread never blocks. LOD metric: level `L` is acceptable when
  `cell_size · focal_px / distance_to_chunk_box ≤ τ`, `τ = 6 px` at 1080p / 60° vertical FOV, scaled
  by the `lod_bias` governor; 15 % hysteresis; a parent stays until its children and their walls are
  ready; children of one parent share a priority key; Empty/Opaque children count as ready at once;
  nodes occluded by the body's conservative core box are not refined. Budgets on a 41 km body:
  ≤ 5,500 resident and ≤ 1,400 in-frustum chunks. Dithered cross-fade 250 ms with a per-frame
  rotating pattern (parent uses the complement); shadow passes switch without dither. Mesh cache:
  levels ≥ 3 in pack files (cp-save pack store, LRU, default 1 GiB), key = hash(GeneratorIdentity,
  GridId, LodKey, content_hash, MESH_FORMAT_VERSION, MESHER_VERSION, apron revision of the chunk and
  its six neighbours); levels 0–2 are always rebuilt.
- **`cp-physics`** – binding specification in `PHYSIK.md`. In short: one physics scene per SOI body
  (its rotating frame) plus a system scene; contacts only from admissible surface features of the
  voxel geometry (no seam catching); TGS-Soft solver with 4 substeps at 240 Hz; mechanisms (bearings,
  pistons, elevators, trains, routes) are prescribed integer motion owned by sites, never stiff
  constraints; exact integer mass sums; stages (Active, FreeFlight, Asleep, Passenger, Rails,
  Prescribed, Anchored, Frozen) so that cost follows what moves and touches; Minecraft-parity
  characters with roll-over at cube edges; no block damage. Coupled to sites through tick-boundary
  messages and the `MotionBoard`; not partitioned by site.
- **`cp-sim`** – command queue in, events out; fixed 60 Hz tick. Character dimensions, eye height,
  step-up height, reach and movement constants are data in the content pack (`player.toml`), never
  code constants, so the player-to-block scale can be tuned without touching code (voxels stay 1 m). **Site** = one grid. Sites interact
  only through messages stamped with an arrival tick ≥ send tick + 1, applied at tick boundaries
  ordered by (arrival tick, source site, sequence), so sites may run in parallel. Machine ids are
  site-local; `EntityId`s of constructs, characters and items are save-wide (lowest free slot first).
  Physics runs in parallel with the sites (`PHYSIK.md`, Tick).
  One simulation model at every LOD: machines, buffers and networks are integer state machines with an
  exactly additive `advance(n_ticks)`; far sites take larger steps and skip to the next event. Belts
  are FIFO queues with latency and capacity; item positions are presentation only. Physics follows
  the same rule through exact stages; its single recorded exception is `Frozen` (undriven bodies
  outside every player bubble). Required tests: conservation across LOD changes, equality of full
  and skipping runs at sync points.
- **`cp-net`** (co-op milestone, after the first playable loop) – server-authoritative co-op: the
  host's cp-sim is the truth (dedicated server possible later). Clients send tick-stamped commands and
  receive what is near them (interest management per grid and distance): entity snapshots and deltas,
  voxel edits as edit receipts carrying override payloads, events. Generated terrain is never sent:
  clients regenerate the base deterministically (§2.5) and receive only overrides and stamps. Each
  client predicts its own player body and the construct it pilots and reconciles with the server. Joining requires the same
  content hash and the same GeneratorIdentity per body. Lockstep is not used (floating-point physics
  across CPUs, late joining into large worlds). Until cp-net exists, cp-app connects its client to an
  in-process server through the same command/receipt interface.
- **`cp-save`** – a save is a directory of immutable generation-numbered files plus `manifest.toml`
  (every live file with length and hash, format version, tick, GeneratorIdentity per body, content
  hash). Commit = write new files, fsync, atomically replace the manifest. Region files are
  append-only chunk logs with an index footer; one string palette per file; unknown keys load as
  `cubeplanet:missing` and keep their original key. Stored per grid: level-0 overrides, pyramid
  levels (rebuildable), stamps. Named sections let cp-sim and cp-universe serialise themselves into
  the same commit. Mapped files are never truncated or replaced in place. Also provides the generic
  pack store used by the mesh cache.

---

## 7. Frame anatomy and budget

**Benchmark scenes** (every figure is the worst frame of the scene, including staggered work):
(1) low flight over forest at noon, (2) coast with ocean, clouds and rain, (3) orbit with limb.

| Pass | Technique | Budget |
|---|---|---|
| Visibility | cull, depth pyramid, shadow-view culls | 0.45 ms |
| G-buffer | indexed packed quads, near-to-far, texture arrays | 1.9 ms |
| Shadows | 3 cascades 2048² to 384 m (cascade 2 every second frame), rotated PCF resolved by TAA, contact shadows ≤ 50 m; far shadows: one tap into a sun-swept shadow-height map | 2.0 ms |
| Lighting | deferred PBR, light clipmap (RGBA8, 4 levels of 128³ at 1/2/4/8 m: block light + sky visibility), half-resolution GTAO ≤ 240 m, cloud shadows | 1.6 ms |
| Atmosphere + volumetrics | one froxel volume 160×90×64 (exponential slices to ~8 km) for in-scattering, fog and shafts; sky-view table; half-resolution march only for limb/orbit | 1.5 ms |
| Clouds | ≤ 130k rays per frame (4×4 interleaved half resolution), ≤ 64 steps with empty-space skipping, optical-depth volume for sun light, reprojection; 2.5D shell from orbit | 1.4 ms |
| Water | forward pass after lighting, nearest surface only, normal maps + analytic foam, half-resolution refraction, caustics lookup; large analytic waves (Thalassa's kilometre-high tidal waves, owner requirement) need a displaced ocean surface mesh (clipmap) that the water design (before E7) specifies | 1.0 ms |
| Reflections | screen space, half resolution, ≤ 240 m | 0.7 ms |
| Post | TAA + upscaling, bloom, auto exposure, AgX, sharpening | 1.45 ms |

Sum 12.0 ms + 2.0 ms named reserve (UI, particles, debug, unlisted passes) + 2.6 ms headroom to
16.6 ms. Frame-thread CPU budget 4 ms (LOD selection 0.5, draw items and uploads 0.8, encoding and
submit 2.0, rest 0.7). **Governors** driven by measured pass times, in this order: shadow range,
LOD bias (τ × 1.0–1.5, also lowers cascade rates), render scale 70–100 % per axis with TAA upscaling.

Formats: HDR scene colour and TAA history `Rgba16Float`; `Rg11b10Ufloat` only for attachment-only
intermediates without alpha (bloom, half-resolution reflections). Light clipmap `Rgba8Unorm` (the
only 8-bit format guaranteed writable from compute); noise volumes generated on the CPU and uploaded;
froxels `Rgba16Float`, ping-pong for temporal accumulation; macro height maps `R16Unorm` (fallback
`R32Float`). GTAO, contact shadows and SSR only below 240 m where no LOD fade occurs.

**VRAM plan** (limit 3 GB): render targets ≤ 400 MB, quad pool 128 MiB, macro maps ≤ 300 MB (only
mips ≤ 512² for bodies other than the current one), clipmaps ≤ 100 MB, block textures ≤ 50 MB,
rest ≤ 250 MB.

## 8. Scale ladder

| Distance | Representation |
|---|---|
| 0–300 m | exact blocks, full textures, light clipmap, detail vegetation |
| to the horizon | levels 1 … root_level of the same octree (far cells are 3–6 px); generated far surfaces tinted from macro maps, authored quads keep their material colour |
| another body in the sky | its own octree, coarsest levels only (at least 16 cells across), own atmosphere, phases, eclipses |
| body below 2 px | emissive point with correct brightness |
| other stars | points from the star catalogue at their real positions |

Each body selects LOD by projected cell size, limited by core occlusion and the governor. A map
camera zooms without a cut from block to planet, system and galaxy. Structures stay visible at a
distance conservatively (slightly inflated), not exactly. Far shadows do not include player
buildings until a ray-traced tier exists.

## 9. Milestones and acceptance

| # | Milestone | Acceptance |
|---|---|---|
| E0 | scaffold, core, jobs, proctex, content, GPU caps | check green; caps report of the real adapter; texture contact sheet; base pack loads and cross-checks against proctex |
| E1 | voxel core, overrides + pyramid, snapshots, greedy mesher, test generators | mesh equals naive reference (regular and walls); structure set (pillar, one-thick wall, beam, hollow cube, tree) keeps a connected representation at levels 1–3; ≤ 0.2 ms meshing per chunk |
| P1 | physics core, headless, after E1 and in parallel with E2–E7 (`PHYSIK.md`): step 0 physics-facing cp-voxel API, `cp-physics`, `BlockPhysics` tables, gravity potential port, exact mass, collision caches, broad- and narrowphase, CCD, TGS-Soft, islands, sleep, free flight, passengers, determinism harness, `cpe physics-lab`, `cpe bench-physics` | physics tests 1–10; budgets B3, B4, B5; gate after two weeks: tests 1, 2, 4, 5 and B3 pass, otherwise the narrowphase switches to per-cell feature SAT under the same admissibility rule |
| E2 | renderer core: GPU-driven voxel pipeline (paths A and B), G-buffer, simple sun/sky lighting, debug views, timings, `cpe run/shot/bench` | test world of ≥ 4,000 chunks; headless PNG; identical images via path A, path B and direct draws; calibration bench: G-buffer ≤ 2.0 ms at 1.0 M visible triangles, cull ≤ 0.1 ms, ns per indirect draw reported |
| E3 | streaming + LOD | seamless flight block → orbit on a 41 km cube; chunk budgets of §6; ≥ 1,500 chunks/s throughput; no LOD lag at 321 m/s; silhouette deviation at LOD borders measured |
| E4 | universe | see moon and planets from the surface, fly there without loading; zoom to galaxy; `Conic`, SOI table, atmosphere density and r_K tested |
| E5 | light + atmosphere | golden images noon/dusk/night/cave/orbit; passes within §7 |
| E6 | terrain 2.0 | repetition test and landmark spacing test pass |
| E7 | clouds, water, volumetrics, reflections, fluid channel | golden images coast/waterfall/storm sky/planet from orbit |
| E8 | sim core (commands, tick, sites, messages), characters and items from `cp-physics`, riding on prescribed test platforms, edits with receipts to physics, saves incl. the physics section | walk, dig, build on all six faces; physics tests 11–13; B6; state hash identical across save/resume; save survives restart and crash |
| E9 | weather + detail vegetation | snowstorm and rain scenes within budget |
| P2 | contraptions, ground and water vehicles (before game step "first loop"): `MotionBoard`, `swept_clear`, assembly, docking, splits, passive joints, wheels, buoyancy, drag panels, build report | physics tests 14–19; B1 without routes; B2 |
| P3 | trains, flight, routes (before "steam and electricity"): train consists, lifting parts, propellers, balloons, routes, weather wind and tornado field | physics tests 20–21; B1 with routes |
| P4 | space (before "spaceflight"): rails, timer wheel, SOI changes, system scene, rockets, stations, floating movement, docking in orbit | physics tests 22–23; B5 complete; B6 descent counter = 0 |

**Beauty review.** Every milestone that changes what is rendered (E2 onwards) ends with a beauty
review before acceptance: a fixed showcase set of shots (same cameras, sun positions and sizes each
time, including distant places for the variety check and frame pairs 1/30 s apart for motion
stability, kept under `target/beauty/`), judged independently by an art director (mood, colour,
composition, light), a world designer (variety, naturalness, scale) and a technical artist
(artefacts, filtering, precision, LOD, colour space, temporal stability). Each judge scores 14
criteria from 1 to 10: composition and readability, colour harmony, lighting, sky and time of day,
atmosphere and depth, scale and vastness, **variety**, naturalness, texture quality, close-up detail,
style cohesion, temporal stability, freedom from artefacts, overall. A fixer implements the
highest-priority actions within the milestone's scope and changes strategy when scores stagnate.
**Owner's bar:** the loop continues until every judge averages ≥ 8.0 with no criterion below 7.0
(round cap 10 as a safety net; the usage guard may pause it, and it resumes after the reset).
Performance gates must still hold after every beauty fix.

Performance targets (measured in release builds, not promised): descent orbit → ground P99
< 16.6 ms and no frame > 20 ms while streaming; meshing ≤ 0.2 ms and generation ≤ 3 ms per chunk on
average; block edit visible within 2 frames (all meshes of one edit receipt appear in the same frame);
VRAM < 3 GB.

## 10. Working rules (for every contributor and agent)

1. **Ownership.** Edit only the paths assigned to you. Need a change elsewhere? Report it as a
   request; do not make it.
2. **Cargo.** Crate-scoped commands only (§3.4). Never `cargo fmt --all` (except the integrator),
   `cargo clean` or `cargo update`. Other builds share the target directory; waiting for the build
   lock is normal.
3. **Dependencies.** Use only what your crate's `Cargo.toml` already declares. Otherwise request.
4. **Git.** Do not commit, branch, stash, reset or push. The lead commits after acceptance.
5. **No questions to the user.** Decide, and record every deviation from this contract.
6. **Honest reporting.** Report real command output. A red check is reported as red.
7. **Tests.** New behaviour comes with tests. GPU-dependent tests skip without an adapter.
8. **Measurements.** Timings taken while other agents are building are not reliable; say so.
9. **Scope.** Build what the contract asks. No speculative features, no placeholder stubs that
   pretend to work.
10. **Style.** Idiomatic Rust, small modules, doc comments on public items stating units and
    invariants. Comments explain why, not what. LF line endings.

## 11. Decision log

- 2026-11-26 Full engine rewrite in `engine/`; prototype frozen at tag `prototyp-v0`.
- 2026-11-26 Blocky LOD at every level (voxel mips), one packed-quad format; no smooth iso-surface path.
- 2026-11-26 Reversed infinite depth, camera-relative `f32`, `f64` frames; no scaled-space pass.
- 2026-11-26 Deferred renderer with compact G-buffer; TAA instead of MSAA.
- 2026-11-26 No external engine framework, ECS or UI crate; own thin layers on wgpu + winit.
- 2026-11-26 Own procedural textures as the default art; third-party art only as marked placeholders.
- 2026-11-26 60 Hz simulation tick (prototype used 120 Hz).
- v2 2026-11-26 Texturing frame on negative faces `(v, u)` (v1 text was unsatisfiable); E0 code is normative.
- v2 24 orientations now (machines need a front on every planet face); key grammar fixed.
- v2 Edits are sparse overrides with an `UNCHANGED` sentinel and a derived pyramid (v1 whole-chunk
  copies would freeze old terrain and show 32 m squares at a distance).
- v2 Generator interface `BaseSource` lives in cp-voxel; meshing reads one padded 34³ chunk.
- v2 Downsample keeps structural and authored cells conservatively; cutout only on levels 0–1.
- v2 `cull_group` table replaces `CULL_SELF`; `BlockFlags` widened to `u16`.
- v2 Indexed 4-vertex quads; walls as separate ranges; draw paths A (Vulkan count) and B (CPU cull);
  bucket data through an instance-step vertex attribute because DX12 count draws ignore
  `first_instance` in `instance_index`.
- v2 Frame budget restated as worst-frame budget with reserve, CPU budget, governors and VRAM plan.
- v2 Sites with message passing and one exactly additive simulation model at every LOD.
- v2 Save = manifest + immutable generation files.
- v2 Beauty review with two independent judges is part of every rendering milestone (owner's request).
- v2 Multiplayer-ready from day one (owner's request): server-authoritative co-op, single-player as an
  integrated local server, only overrides travel over the network; lockstep rejected.
- v3 Physics engine (`PHYSIK.md`), synthesised from three independent designs: one scene per SOI body
  plus a system scene, coupled to sites by tick-boundary messages and the `MotionBoard`; TGS-Soft with
  4 substeps; contacts only from admissible surface features (per-voxel box SAT rejected: seam
  catching, 17 M pair tests); mechanisms as prescribed integer motion (reduced-coordinate multibody
  rejected: implementation risk); exact integer mass (f64 per-chunk moments rejected: edit-order
  dependent); stages with `Frozen` as the one recorded LOD deviation; characters with 20 Hz Java
  locomotion phases, 60 Hz sweeps and roll-over at cube edges; co-op stays server-authoritative.
  The physics-facing cp-voxel API is P1 step 0 instead of E1 (E1 was already running).
- v2 Owner confirmed 1 m world blocks (finer world grids multiply per-block costs on planet scales).
  Fine detail comes from shapes (slabs, stairs, slopes), the model path and later "Feinbau": a single
  block carved into 8×8×8 sub-cells, stored per block as entity data, rendered through the model path.
