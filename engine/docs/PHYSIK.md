# Cubeplanet Engine – Physics Contract (v3)

**Status:** binding for cp-physics and for every crate that feeds it (cp-voxel, cp-universe, cp-sim,
cp-content, cp-stream, cp-save, cp-net). Part of the architecture contract ARCHITEKTUR.md (§6
refers here). Synthesised on 2026-11-26 from three independent designs (gameplay-first, scale-first,
robustness-first) by a chief-architect review; the base is the robustness-first design. Implement
against this document; report contradictions instead of deviating silently.

Milestones: **P1** physics core (headless, starts after E1, runs in parallel with E2–E7; step 0 is
the physics-facing cp-voxel API at the end of this file), **P2** contraptions, ground and water
vehicles, **P3** trains, flight, routes, **P4** space. Characters arrive with E8.

### `cp-physics` *(contract v3; milestones P1–P4)*

Rigid-body physics of voxel grids, characters and items for every body of the universe. It is deterministic (§2.5) and headless. Its cost grows with what moves and touches, never with the number of constructs or the size of the world (§1.1).

**Principles**
1. **Scenes.** There is one physics scene per SOI body, in that body's rotating frame (`FrameId::Body`), and one per system (`FrameId::System`) for bodies outside every SOI. Bodies interact only within their scene. Physics is not partitioned by site.
2. **Contacts.** Contacts come only from admissible surface features of the voxel geometry. Covered faces and internal edges never produce a normal.
3. **Mechanisms are prescribed.** Bearings, pistons, gantries, elevators, train cars and routed vehicles have integer joint state owned by the driving site. Physics places them analytically and never solves them as stiff constraints. Only passive couplings are solver joints.
4. **Forces.** Every force is evaluated once per tick from data aggregated on edit. Cost per tick is O(parts + wet runs + contacts + characters), never O(blocks).
5. **Stages.** Every body is in exactly one stage. Each stage is exact for its class, and changes happen at tick boundaries with exact state conversion. `Frozen` is the only approximation across LOD and never applies to anything a site drives.
6. **No damage.** Blocks and joints never take damage or break; impacts are events.

**Crate and ownership**
- `cp-physics` is deterministic and a hot crate.
  - Internal deps: `cp-core`, `cp-voxel`, `cp-universe`, `cp-jobs`. External: `smallvec`, `arrayvec`, `thiserror`, `log`.
  - It takes plain tables (`BlockTables::physics`) and snapshots, never the registry. It knows no machines and no sites.
- `cp-physics` owns: bodies, mass properties, collision caches, broadphase, narrowphase, CCD, solver and joints, forces, characters and items, stages, rails, and its save section.
- `cp-sim` owns: the tick, sites, machines and networks, joint/train/route schedules, the non-physical state of players, and the `WindField`.
- **Ids.** `EntityId`s of constructs, characters and items are save-wide: one deterministic allocator hands out the lowest free slot first. Machine ids stay site-local. `MotionId = (GridId, u32)`.
- The body grid is placed in its scene with the `grid_offset` translation (§2.2).

**Tick** (60 Hz, run by `cp-sim`)
1. **Input, serial:**
   - Apply commands with arrival tick T, then voxel edits.
   - `apply_receipts`: wake bodies, update mass sums, invalidate caches, and re-check statically every motion whose swept volume a receipt touches. A block placed into a swept volume blocks that motion in this tick; the revocation reaches the site in this phase.
2. **Parallel:** sites advance T → T+1 while `step` runs. Both read state T (snapshots after step 1, `MotionBoard` T, messages with arrival T); each writes only its own state.
3. **Commit, serial:** apply job results in key order; publish snapshots and the `PoseBoard` for T+1; deliver messages and events with arrival T+1.

Sites and physics therefore see each other one tick late, as sites see each other.
- **Into physics:** controls, thrust per thruster group, shaft power and rpm, the mass of tanks, cargo and gas, anchor state.
- **Out of physics:** poses, motion load and blocking, impacts, landings, stage and SOI changes, splits.

**Data model** (f64 state in the scene root frame unless stated)
```rust
pub enum Stage { Active, FreeFlight, Asleep, Passenger { carrier: EntityId },
                 Rails(Conic), Prescribed(MotionId), Anchored, Frozen }
pub struct Body {
    pub id: EntityId, pub scene: FrameId, pub stage: Stage,
    pub com: DVec3, pub q: DQuat,           // body axes → scene axes, unit, canonical w >= 0
    pub v: DVec3, pub l: DVec3,             // com velocity; angular momentum about com, scene axes
    pub grids: SmallVec<[GridAttach; 1]>,   // >= 2 after docking, each with a fixed RelativeTransform
    pub slots: SmallVec<[MassSlot; 4]>,     // tanks, cargo, gas, passengers; <= 256
    pub sleep_ticks: u16,
}
pub struct MassSums { pub m: i64, pub s1: [i128; 3], pub s2: [i128; 6] }
// per grid, exact; unit u = 1/32 m in grid coordinates: m in grams, s1 = Σ m·p,
// s2 = Σ (m·p·pᵀ + j) as xx yy zz xy xz yz
pub struct MassSlot { pub at: IVec3 /* u, grid coordinates */, pub grams: i64 }
```
- **Mass.**
  - Each receipt cell and each slot change updates the sums by one O(1) integer step. The sums do not depend on the order of updates and equal a full recomputation.
  - m, com, I and I⁻¹ are derived in f64 when they change; ω = I_w⁻¹·(L − L_int).
  - A mass change keeps the grid's velocity field: v += ω × Δcom, ω is kept, L is recomputed.
  - Splits and merges conserve linear and angular momentum.
- **`BlockPhysics`.** Element type in `cp_core::block`, stored in `BlockTables::physics` and derived by cp-content from block definitions:
  - `mass_g: u32`: density × shape volume unless overridden.
  - `com: [u8; 3]`: centre of mass in u from the cell's min corner.
  - `j: [i64; 6]`: second moment about com in g·u², rounded once in the table.
  - `boxes: ArrayVec<[u8; 6], 4>`: collision boxes in 1/16 m, up to 24/16 m high. The part above the cell applies to characters and items only.
  - `coverage: [u16; 6]`: 4×4 bits per face.
  - `friction`, `restitution`: × 100. `slip`: character block friction × 100 (60 by default, 98 for ice).
  - `displaced: u8`: m³ × 255. `fluid`: kind and fill height in 1/16 m.
  - Flags: `SEALS`, `CLIMBABLE`, `ANCHOR`.
  - Air is all zero. `OPAQUE` implies a full cube.
- **`ConstructShape`.** Per grid, rebuilt per dirty 4³ brick in the tick of the edit:
  - probe vertices with their normal cones, convex exposed edges, brick bounds, and 26 support points;
  - six drag panels (outside-air faces per face direction: area, centroid, second moment);
  - part lists: thrusters, gyros, lifting parts, rotors, wheels, seats, anchors.
  - Flood-fill data comes from jobs (see Forces).
- **`CollisionChunk` cache.** Key `(GridId, level-0 key, revision)`.
  - Contents: solid and partial masks (4 KiB each), a surface mask built from the padded chunk, and a 512-bit mask of 4³ bricks.
  - LRU limit 64 MB. Prewarmed in `Background` jobs for bodies that approach a bubble.
  - Caches never change results.
- **`Manifold`.** One per (pair, cluster), with at most 4 points `{ feature: u64, anchors relative to both coms, normal, separation, impulses n/t1/t2 }`, and at most 16 points per pair. A manifold is removed only when the narrowphase emits no point for it.

**Stages**
| Stage | Entry | Exit | Cost per tick |
|---|---|---|---|
| Active | default inside a bubble | see the other rows | full |
| FreeFlight | Active, no joint, no contact candidate within ‖v‖·2 s + 32 m | candidate found, command | ≈ 0.3 µs |
| Asleep | island point speed ‖v‖ + ‖ω‖·r < 0.05 m/s for 30 ticks, no actuator | contact by an awake body, receipt inside fat AABB + 1 cell, command, wind or current above the stored tolerance | 0 |
| Passenger | as Asleep, but all contacts are with one moving carrier | ‖a_t‖ > μ·‖a_n‖, or a_n pointing off the deck (carrier-frame acceleration at the body), other contact, receipt, command | O(1); mass counts in the carrier |
| Rails | 60 ticks without contact, joint or thrust; conic outside the atmosphere with periapsis ≥ r_K; stationary spin | command; SOI exit, atmosphere entry or r < r_K (tick found by 64-step bisection, scheduled); overlap with the index of 1 s swept AABBs of awake bodies (rebuilt every 60 ticks) | 0 |
| Prescribed | grid driven by a `Motion` | its site releases it | 0 outside bubbles; inside, the contacts it pushes |
| Anchored | anchor part engaged, or staff lock | released | 0 (static collider) |
| Frozen | outside every bubble after 30 s of grace (≤ 32 bodies in grace, oldest first), eligible for no other stage | a bubble reaches it | 0; its time stops |

- **Bubbles.** Every player character has one. A body enters at 1,024 m and leaves at 1,280 m, measured to its AABB in the scene frame. Outside every bubble, a FreeFlight body that gets a contact candidate freezes, and prescribed grids are not placed.
- **r_K** = √3 · (half edge + space_start + space_width), from cp-universe. Beyond it the potential is exactly −μ/r; with the prototype's parameters for Home it is 48.8 km.
- **Rails** use `cp_universe::Conic::state_at(tick)` with a fixed number of Newton iterations, so the pose is a function of the absolute tick. Scheduled events live in a hierarchical timer wheel keyed by (tick, EntityId), with levels of 64, 4,096 and 262,144 ticks.
- **SOI.** Membership is evaluated per island, by its com, at tick boundaries. Conversion through the ephemeris and `BodyFrame` is exact.

**Time stepping**
- Forces are computed once per tick and held over the substeps. There are no speed-dependent substeps.
- **Islands with contacts or joints:** TGS-Soft with 4 substeps, h = 1/240 s. Each substep:
  1. v += h·(g + fictitious + F/m), with drag implicit (backward Euler on the 3×3 damping matrix); L += h·τ; ω = I_w⁻¹·(L − L_int).
  2. Warm start, then 1 solve iteration.
  3. x += h·v and q ← exp(½·ω·h)·q, computed exactly through `det::sin_cos`; normalise; canonicalise to w ≥ 0.
  4. 1 relax iteration.

  After the substeps come restitution and impulse storage.
- **Unconstrained bodies:** one kick-drift-kick step per tick in the rotating frame with exact frame transport (port of the prototype's `Gravity::step` and its attitude transport).
- **Fictitious forces:** Coriolis and centrifugal terms always apply in body scenes. Tides and third bodies are omitted (patched conics).
- **Guard rails:**
  - ‖ω‖ ≤ 50 rad/s for dynamic bodies.
  - A non-finite value restores the body's state of tick T with v = ω = 0, puts it to sleep and emits `Quarantined`.
  - Penetration > 0.5 m lasting 30 ticks is resolved positionally, 0.2 m per tick, without adding velocity.

**Collision**
- **Broadphase** per scene:
  - Two dynamic AABB trees: one for awake bodies, one for asleep, kinematic and anchored bodies. Fat margin 0.1 m + 2·‖v‖·Δt. Rotating prescribed grids use the AABB of a full turn.
  - Terrain is a query, not a proxy: a body's AABB is tested against level-0 classes (`region_may_collide`), then against brick masks.
  - Static and kinematic bodies never join islands. Pairs are sorted by (EntityId, EntityId).
- **Narrowphase, grid A against grid B.** The same algorithm serves construct–construct and construct–terrain. It runs in the integer frame of the grid that owns the target face.
  1. **Candidate brick pairs:** A's surface bricks, as OBBs, against B's brick mask.
  2. **Vertex probes:**
     - A's exposed lattice vertices (1 m spacing; box corners in partial cells) are tested against B's solid cells within the speculative margin, then B's against A.
     - The normal cone of a lattice vertex is a function of the 2×2×2 occupancy around it (a 256-entry table).
  3. **Edge probes:** A's convex exposed edges against B's, segment against segment, inside candidate brick pairs.
  4. **Admissibility (normative).** This rule is what removes internal edges.
     - A probe vertex carries its outward normal cone: {n} on a flat face, the arc between two normals on a convex edge, three normals at a convex corner. Concave vertices and edges are not probes.
     - A probe may pair only with a target face whose normal n satisfies −n ∈ cone (widened by 2°), and only if the probe projects into that face.
     - An edge pair is admissible only if the normal arcs of the two edges intersect.
     - A penetrating probe takes the shallowest admissible exposed face of its cell. Ties go to the face whose plane it crossed since tick T, then to the lower face index.
     - Covered faces are never targets. The rule applies equally to penetrating and to speculative probes.
  5. **Reduction** per cluster (owner grid, face direction); clusters of the two owners whose normals agree within 2° merge. Keep the deepest point, the point farthest from it, then two points that maximise the area. Ties are broken by feature id.
- **Partial cells** use their ≤ 4 boxes. A box face is exposed unless the neighbour's `coverage` hides it.
- **CCD.**
  - Speculative margin: 0.02 m + ‖v_rel‖·Δt, at most 1 m.
  - A body that moves more than 0.5 m in a tick also casts voxel-DDA rays from its 26 support points and its com along the relative motion. It is clamped to TOI − 0.02 m and keeps its velocity.
- **Material pairing:** μ = √(μa·μb); restitution = max(ea, eb), applied above 1 m/s.
- **Fallback** if the P1 gate fails: box SAT per candidate cell pair, restricted by the same admissibility rule.

**Solver**
- **TGS-Soft** (Box2D v3 "soft step"):
  - Contacts: 30 Hz, ζ = 10, push-out ≤ 3 m/s, linear slop 5 mm.
  - Joints: 60 Hz, ζ = 2. P1 may raise ζ to 5 against test 16 and must record the change.
  - 2 friction rows per contact point; fixed iteration counts.
- **Order.** Constraints are sorted by (pair, feature), islands by their smallest EntityId. Islands with more than 1,024 constraints use a deterministic greedy graph colouring at every thread count.
- **Passive joints:**
  - revolute: free bearing or hinge, with limits and brake torque;
  - prismatic;
  - spherical: towbar, coupling;
  - distance: rope with min/max length and spring; a winch is a length motor with rated force;
  - wheel.

  A motor is a target speed with a maximum force or torque; above it, it slips.
- **Wheel** (no wheel bodies):
  - 1–3 suspension rays; spring tuned to the sprung mass (2 Hz, ζ = 0.5, travel 0.3 m).
  - Longitudinal and lateral rows, bounded by μ × the normal impulse of the same substep.
  - Drive torque min(P/ω, grip) from the shaft message; rolling resistance 0.015.
  - The reaction acts on the body that was hit.

**Prescribed motion**
```rust
pub struct Motion {
    pub id: MotionId, pub child: GridId, pub parent: GridId,   // parent: body grid or construct grid
    pub at_t: RelativeTransform, pub at_t1: RelativeTransform,  // child -> parent, from integer joint state
    pub twist: Twist,                                           // analytic v and ω relative to the parent
    pub capacity: f64, pub axis: DVec3,                         // N·m or N along `axis`
    pub riders: bool,
}
pub struct MotionBoard(Vec<Motion>); // sorted by id; published by sites with their state for tick T
```
- **Joint coordinates** are integers owned by sites: angle in 2⁻³² turn, linear position in 1/4096 m, and an increment per tick.
  - They are exactly additive (`advance(n)`).
  - `at_t1` and `twist` are computed from them through `det::sin_cos`, never by differencing poses.
- **Committing a move.** A site commits the next increment only after `swept_clear` succeeds. The query checks the parent body's grids, static grids, and the tick-T poses of other prescribed grids.
- **Rooted on a body grid:**
  - The child is kinematic (infinite mass) and pushes dynamic bodies and characters.
  - A body or character pinned against static geometry blocks it with `MotionBlocked { id, safe }` (arrival T+1; overshoot ≤ 2 increments).
  - Passive joints (ropes, hinges, couplings) may connect dynamic bodies to the child. Their reactions and the contacts on the child are reported as `MotionLoad` about `axis`, exactly as for construct-rooted motion, so a motor feels what hangs on it (owner requirement 2026-11-26: swing carousels with free-hanging blocks must work through real centripetal dynamics; acceptance test 24).
- **Rooted on a construct:**
  - The child joins the parent body. Mass properties include it at its current pose; its tensor is rotated each tick.
  - L_int = Σ (I_child·ω_rel + m·r × v_rel). Its contacts carry its surface velocity.
  - The reaction about `axis` is reported every tick as `MotionLoad`, for the stress model. Above `capacity` for 3 ticks → `MotionBlocked`.
- **Riders:** `riders = false` for grids that rotate about an axis more than 45° from gravity. They push characters but never carry them.
- **Trains and routes.** Train cars (track state in cp-sim) and routed vehicles are prescribed the same way.
- **Docking and welding** merge grids into one body; there is no weld constraint.

**Forces** (each once per tick)
- **Gravity:** the SOI body's potential from cp-universe, at the com. Bodies with a bounding radius > 16 m add the gradient torque from the exact second moment and a finite-difference gradient (3 samples, 1 m apart).
- **Thrust:**
  - Thrusters sharing an axis and throttle group act as one: force plus torque.
  - Mass flow ṁ = F/(Isp·g₀), with Isp per part (default 300 s). Fuel is integer grams in slots.
  - Gyros apply a bounded torque. There is no energy limiter.
- **Aero:**
  - Six drag panels with Newtonian pressure F = −½·ρ·1.05·A·max(0, v_rel·n)²·n at the panel centroid, plus rotational damping from the panel's second moment.
  - Lift comes only from lifting parts: wings, control surfaces, sails, and rudders and keels in water. They merge into coplanar panels and use the prototype polar: C_L = 5.5·α up to 14°, continuous stall to 0 at 90°, C_D = 0.025 + 0.065·C_L² + 1.4·sin²α.
  - Propellers and rotors use the prototype's power-bounded actuator disk with reaction torque; their spin enters L_int.
  - Air density comes from cp-universe.
- **Buoyancy:**
  - Each construct caches runs of displacing cells (solid, or sealed air) along the grid axis closest to the water normal.
  - Each run carries a spill level, found by a priority flood through unsealed faces from outside water. An open hull therefore floats until water tops the gunwale, and a hole below the waterline floods.
  - Each tick, a water plane fitted to ≤ 9 `FluidSampler` samples clips every run. That gives the exact volume, centroid and righting moment.
  - Water drag uses the panels with water density and the sampled current.
- **Balloons:** gas pockets. Lift ρ_air·V·g acts at the pocket centroid; the gas temperature follows the prototype's heat model (exact exponential).
- **Wind:** `WindField::sample(scene, point, tick)` at panels and parts; deterministic. A tornado is a Rankine vortex with updraft on a seeded track. On a supported body the horizontal wind force is capped at 0.5 × weight. Anchored bodies are immune.
- **Fluids:** `FluidSampler::sample(scene, point) -> Option<FluidSample { surface, normal, velocity, density }>`. Until the E7 fluid channel arrives, it reads FLUID cells (fill heights, flow from levels). Coupling is one-way.
- **Derived-data jobs** compute outside air, spill levels, gas pockets and connectivity after edits. Spill levels are recomputed when the tilt to the water normal changes by more than 5° or the waterline by more than 0.25 m.
  - Jobs are pure functions of saved state and commit exactly at issue tick + 4. A late job is waited for.
  - An edit to an input grid re-issues the job in that tick.

**Characters and items**
- **Controller:** kinematic, not a solver body.
  - Box 0.6 × 1.8 × 0.6 m (1.5 m high when sneaking); eyes at 1.62 m (1.27 m when sneaking).
  - The box is axis-aligned in the character's reference grid: the grid it stands on or last stood on, otherwise the SOI body's grid. Its state is in that grid's coordinates.
  - Collision uses `BlockPhysics::boxes`.
- **Apparent gravity:** g_app = g(x) − a_com of the reference grid's body, averaged over 6 ticks and limited to 3 g; no rotational terms.
  - The down axis D is one of the six grid axes. The character falls along D with ‖g_app‖.
  - Tangential parts are ignored, so cube-edge bands and accelerating carriers never push characters sideways.
- **Changing D.** D changes only while the character is not supported along D:
  - (a) Roll-over at a convex edge onto the adjacent face, if that face's axis is at least as dominant at the feet and the standing box fits there.
  - (b) In the air, when another axis is at least 3° better aligned with g_app.
  - Every change passes through a 0.6 m cube at the feet, so it cannot jam. The camera roll eases over 0.3 s.
  - Pure angular hysteresis is not allowed: near an edge the field turns 1/edge_width rad per metre (0.24°/m on Home), so 3–10° would let the character fall 12–42 m down the next face.
  - ‖g_app‖ < 0.5 m/s² → floating movement (P4).
- **Locomotion** runs in 20 Hz phases with the Java model. Per 50 ms, in blocks:
  - jump 0.42; gravity 0.08 × ‖g_app‖/9.81; vertical drag 0.98; horizontal drag slip × 0.91;
  - walk 0.1, sprint 0.13, air 0.02 (sprinting 0.026); sneak × 0.3; sprint jump +0.2;
  - water: acceleration 0.02, drag 0.8 (sprinting 0.9), vertical 0.8, swim 0.04, sink 0.005, current 0.014.

  Result: walking 4.317 m/s, sprinting 5.612 m/s, jump apex 1.2522 m.
- **Sweeps** run every 60 Hz tick:
  - Order: along D first, then the larger, then the smaller tangential component. Skin 1 mm.
  - Step-up 0.6 m (up, forward, down); sneak edge guard; swimming; climbing.
- **Riding:**
  - A character attaches to a grid after 2 ticks of support on it.
  - It detaches after 30 ticks unsupported outside the grid's AABB + 2 m, or when g_app has been more than 60° from D for 10 ticks.
  - Pose, velocity and look transfer exactly through `BodyFrame`, with no impulse. Jumps stay in the reference frame.
  - Seats fix the character in the seat's grid. Dismounting box-casts 6 candidate places in that grid; relative velocity is 0.
  - `riding.centrifugal` in `player.toml` (default 0) scales a horizontal centrifugal acceleration ω²·r of a rotating reference grid, opposed by the character's ground friction: 0 keeps standing characters carried, 1 is physical. Seated characters always move with their seat, so a seat on a swinging chair swings out with it.
- **One-way coupling:**
  - The character's weight (80 kg × ‖g_app‖) acts at its foot point.
  - Walking into a body under 2 t pushes it with at most 300 N.
  - Other grids that overlap the character push it out along admissible normals at ≤ 3 m/s. A crushed character moves to the nearest free cell.
  - Fall damage is an event.
- **Items** are 0.25 m boxes with the same sweeps and gravity. They fall asleep after 10 grounded ticks; on moving grids they become passengers.
- **Animals** use the character controller, driven by AI commands.

**Routes** (P3)
- Owned by cp-sim: a deterministic function tick → pose in a body frame. Segments are cubic Hermite curves; coasts are conics.
- A route is either recorded from a piloted run, or compiled from waypoints with the vehicle's point-mass model (duration, fuel, feasibility; recompiled on edit).
- Near and far, the vehicle is prescribed. A failed `swept_clear` holds it and raises `RouteBlocked`.
- When a player takes the controls, the vehicle becomes Active at a tick boundary with the route's analytic velocity.

**Determinism, saves, co-op**
- **Determinism:** §2.5 applies, plus:
  - f64 and scalar glam types only. glam's vector `min`, `max`, `clamp`, `min_element` and `max_element` are disallowed because they call `f64::min`/`max`.
  - Fixed iteration counts; the sorted orders defined above.
  - Jobs write disjoint slots, and results are applied in key order. No float atomics.
- **Save section `physics`** holds everything the next tick reads: bodies with stage and counters, manifolds with impulses, joints, characters with their 20 Hz phase, items, the timer wheel, and pending job inputs.
- **State hash:** `state_hash()` (fixed field order, little-endian) is compared every 60 ticks in tests and in co-op diagnostics.
- **Co-op**, as in §6 cp-net: the host is authoritative. A client predicts its own character (a pure function of snapshot and inputs) and the construct it pilots, with rollback of up to 8 ticks and 100 ms correction blending.

**Threading**
- The sim thread plus at most 3 `Critical` workers.
- In parallel: forces per body, narrowphase per pair, solve per island (largest first), characters, derived-data jobs.
- Serial: receipts, broadphase, island building, stages, events.
- Work packets own their data: `VoxelSnapshot` clones; island state is moved in and out. `JobSystem::par_for` is not used per tick, because it starts threads on every call.

**Interface**
```rust
impl PhysicsWorld {
    pub fn new(config: PhysicsConfig) -> Self;
    pub fn apply(&mut self, view: &WorldView, commands: &[PhysicsCommand]);          // sorted
    pub fn apply_receipts(&mut self, view: &WorldView, receipts: &[EditReceipt]) -> Vec<PhysicsEvent>;
    pub fn step(&mut self, view: &WorldView, jobs: &JobSpawner) -> StepOutput;        // events, messages
    pub fn poses(&self) -> &PoseBoard;        // per grid, character, item: FrameId, f64 pose at T and T+1
    pub fn residency(&self) -> &[SimAnchor];  // level-0 boxes for cp-stream: swept AABB over 2 s + 32 m
    pub fn report(&self, body: EntityId) -> Option<BuildReport>; // mass, com, inertia, buoyancy centre,
                                              // draft, lift and drag centres, thrust/weight, Δv
    pub fn write_section(&self, out: &mut Vec<u8>);
    pub fn read_section(bytes: &[u8]) -> Result<Self, PhysicsError>;
    pub fn state_hash(&self) -> u64;
}
pub struct WorldView<'a> {
    pub tick: Tick, pub voxels: &'a VoxelSnapshot, pub tables: &'a [BlockPhysics],
    pub universe: &'a dyn UniverseView, pub motions: &'a MotionBoard,
    pub wind: &'a dyn WindField, pub fluid: &'a dyn FluidSampler,
}
pub fn swept_clear(view: &WorldView, motion: &Motion, to: &RelativeTransform) -> Result<(), Blocked>;
pub fn raycast(view: &WorldView, world: &PhysicsWorld, scene: FrameId, ray: DRay, filter: HitFilter)
    -> Option<Hit>;
// PhysicsCommand: Spawn, Assemble, Disassemble, Dock, Undock, Control, Drive, MassSlot, Character,
//                 Seat, Anchor, TakeControl
// PhysicsEvent:   Impact, Landed, MotionBlocked, MotionLoad, StageChanged, SoiChanged, Split,
//                 Quarantined, RouteBlocked
```

**Budgets** (release build, 9600X, sim thread + 3 workers, nothing else running)
| Scene | Content | Target |
|---|---|---|
| B1 harbour | 300 awake constructs of 500 blocks, one ship of 20,000 blocks, ≈ 1,500 contact points, 200 items, 4 characters; 10,000 asleep, 2,000 on rails, 1,000 prescribed | ≤ 2.5 ms wall P99, no tick > 4 ms, ≤ 6 ms CPU |
| B2 | 100 awake vehicles of 500 blocks in ground contact | ≤ 3 ms on 1 thread, ≤ 1 ms on 4 |
| B3 | two hulls of 20,000 blocks with 100 m² of contact; two of 50,000 blocks with 400 m² | contact generation ≤ 0.2 ms and ≤ 0.5 ms, 1 thread |
| B4 | pile of 1,000 crates | ≤ 3 ms at peak, all asleep within 10 s |
| B5 | 10,000 asleep, 2,000 on rails, 1,000 in free flight | ≤ 0.4 ms; ≤ 5 ns per asleep body |
| B6 | one character; descent from orbit to the ground at 300 m/s | ≤ 20 µs; counter "physics waited for generation" = 0 |

**Acceptance tests**
- Every test drives `PhysicsWorld::step` (from E8 on also `Sim::tick`) on real snapshots, from scenario files that `cpe physics-lab` replays. There are no test-only code paths.
- Gravity is a uniform 9.81 m/s² unless stated otherwise.

| # | Test | Target | Gate |
|---|---|---|---|
| 1 | Seam slide: 8×1×4 m hull, μ = 0, at 5 and 20 m/s across ≥ 100 block and chunk seams, yaw 0°, 17°, 30° and 45°, started 1 mm sunk | speed loss < 0.5 %; normal speed < 0.01 m/s; yaw drift < 0.1° | P1 |
| 2 | As test 1, with μ = 0.5 from 10 m/s | stops after 10.2 m ± 2 %; normal impulse per tick within ±5 % of m·g·Δt | P1 |
| 3 | Crate on a 20° slope, μ = 0.5 | creeps < 1 cm per minute | P1 |
| 4 | 10-crate column; 15-box pyramid; mass ratios 1:20, 1:100 and 1:1,000; 1 kg on a 10⁷ kg station | drift < 1 cm in 60 s; asleep within 2 s; jitter < 1 mm/s | P1 |
| 5 | Narrowphase against a brute-force enumeration of all feature pairs under the same admissibility rule, 100,000 random poses, partial blocks included | identical contact sets; separations within 1e-9 m; covered faces never emit | P1 |
| 6 | Bodies of 1 m and 5 m at 1,000 m/s against a 1-block wall, 1,000 random angles each | no pass-through | P1 |
| 7 | 5,000-block construct dropped obliquely from 200 m | after contact never faster than 2 × impact speed; at rest within 5 s; penetration < 0.5 m; no NaN | P1 |
| 8 | Mass sums after random edits, cargo and fuel changes, compared with full recomputation | identical integers; bit-identical f64 properties | P1 |
| 9 | 10 minutes of B1 with 1, 2, 4 and 9 workers, dev and release builds, save and resume at tick 18,000 | identical state hash every tick | P1 (B1 grows with P2–P4) |
| 10 | Gravity port: −∇Φ against central differences, across all cube edges and corners | agreement < 1e-6 relative; no jump > 1e-9 relative | P1 |
| 11 | Character against the 20 Hz reference (ported prototype tests) | identical displacement per phase (≤ 1e-12 m); steps up 0.5 m, 1 m needs a jump; 0 falls in 1,000 sneak walks along edges | E8 |
| 12 | 10,000 random runs (walk, jump, sneak) across all 12 cube edges and 8 corners | never inside a block; never stuck > 1 s; no fall > 1 m at an edge whose faces have equal height | E8 |
| 13 | Riding: turntable at 32 rpm, r = 5 m, 10 min; 20 jumps on a ship yawing at 0.5 rad/s; 100 seat cycles on a 16 rpm rotor on a ship at 20 m/s; 1,000 random boardings of a yawing ship | drift < 1 cm; every landing on deck ± 0.1 m; never inside a block, relative speed < 0.5 m/s; no fall-through | E8 |
| 14 | Bearings at 0.1–256 rpm and trains on S-curves, 1 h | pose error vs analytic < 1e-9 m; ω error < 1e-9 rad/s every tick; w ≥ 0 | P2 |
| 15 | Free station whose rotor spins up | angular momentum error < 1e-9 relative | P2 |
| 16 | Free 2 t rotor on a 50 t ship pitching ±10° at 0.2 Hz, 10 min; rope holding a static 10 t load | anchor error RMS < 0.5 mm, max < 3 mm; rope stretch < 0.5 % | P2 |
| 17 | Piston pushes a character against a wall; a block is placed into a windmill's swept volume | piston blocked, penetration < 5 cm; bearing blocked in the same tick | P2 |
| 18 | Closed 10×4×3 m hull of 300 kg/m³; holes below and above the waterline; open boat loaded until water reaches the gunwale; top-heavy boat | draft 0.9 m ± 2 cm; floods only through the lower hole; sinks only past the gunwale; capsizes | P2 |
| 19 | 100 kg cart on a 200 t ship rolling ±10° at 0.1 Hz, then at rest | does not tip; jitter < 1 mm/s at rest | P2 |
| 24 | Swing carousel: a prescribed turntable at 2–20 rpm with four 3 m ropes, each holding a free 100 kg block at radius R = 4 m; a loose crate (μ = 0.5) on the deck; spin-up, 10 min steady, stop | steady rope angle θ satisfies tan θ = ω²·(R + L·sin θ)/g within 1 %; the crate slides off once ω²·r > μ·g; the motor's `MotionLoad` includes the rope reactions; after stopping, the blocks swing down and come to rest; no energy gain | P2 |
| 20 | Rails, prescribed motion and routes advanced in event jumps vs tick by tick | identical at sync points | P3 |
| 21 | Flight lab: trimmed glider; airship; propeller test stand | glider stable for 60 s at 30 m/s and recovers from a 20° stall; airship lift = displaced air ± 1 %; propeller static thrust = prototype formula ± 1 % | P3 |
| 22 | Active KDK, 100 orbits in the Kepler region; 1 h through the cube zone; rails, 100 periods; switching active ↔ rails every 100 s for 10 orbits; SOI change Home → Kiesel | ΔE/E < 1e-7; Jacobi constant within 1e-5, no trend; back to start within 1e-6 m; deviation < 1 m; change within ±1 tick, jump < 1 mm and 1 mm/s | P4 |
| 23 | Rocket with Δv = 2·v_esc launched vertically (regression test for the prototype's energy limiter) | leaves the SOI | P4 |

---

## P1 step 0: physics-facing cp-voxel API

E1 built cp-voxel as specified in ARCHITEKTUR.md §5.1. Before the physics core starts, cp-voxel gains
the following API. Existing E1 signatures keep compiling (new functions or wrappers, never silent
behaviour changes); cp-app and cp-worldgen must still build.

The answer is not "nothing beyond §5.1". Physics needs seven items from E1. Four of them change public signatures or invariants that cp-mesh, cp-stream and cp-save will also consume, so adding them after E1 would break those crates: items 1, 3, 4 and 6. The other three (2, 5, 7) only add API, but P1 starts right after E1 and needs them on day one.

**1. Per-cell level-0 changes in every receipt (breaking)**
```rust
pub struct EditReceipt { pub grid: GridId, pub cells: Vec<CellChange>, /* existing: dirty keys, class changes */ }
pub struct CellChange { pub cell: IVec3, pub before: BlockStateId, pub after: BlockStateId }
```
- `before` and `after` are composite states (base ⊕ override), listed in application order. A revert to `UNCHANGED` reports the base state; `UNCHANGED` itself never appears.
- Consumers:
  - O(1) integer mass sums;
  - waking bodies;
  - collision-cache invalidation;
  - swept-volume re-checks of prescribed motion in the same tick;
  - connectivity and split jobs;
  - later, network sync.
- Test: the list equals a cell-by-cell diff of the composites before and after.

**2. Palette-aware masks, shared with the mesher (additive)**
- `ChunkData::mask_where(&self, table: &[bool]) -> CellMask`
  - `table` is indexed by `BlockStateId` and evaluated once per palette entry.
  - `Uniform` chunks take O(1); a paletted chunk takes ≤ 5 µs.
- `PaddedChunk::mask_where(&self, table: &[bool]) -> PaddedMask` (34³ bits), for exposed faces across chunk borders.
- The binary greedy mesher's opacity columns use the same code.
- Test: equals a per-cell flag test.

**3. One snapshot for all grids (invariant)**
- `VoxelWorld` may stay per grid as the mutable owner. The simulation, however, needs one `VoxelSnapshot` that covers every grid, body and construct alike: a persistent map from `GridId` to grid snapshot.
- It clones in O(1), is `Send + Sync` and immutable, and its results never depend on cache state (generate-on-miss is deterministic).
- Every query takes a `GridId`.

**4. Construct grids (invariant)**
- `EmptySource`: a `BaseSource` that is all air. `classify` returns `Empty`, `subtree_empty` returns true, `fill` writes `AIR`.
- A grid without a generator costs ≤ 1 KB plus its chunk data, so 5,000 grids fit in ≤ 5 MB.
- Queries:
  - `bounds(grid) -> Option<IAabb>`: tight over non-air cells, maintained per edit;
  - `non_air_count(grid)`;
  - `chunks(grid)`: the level-0 keys with non-air cells, in sorted order.
- Negative cell coordinates are supported everywhere.

**5. Atomic cell transfer between grids (additive; may be built on `apply`)**
```rust
pub fn transfer(from: &mut VoxelWorld, to: &mut VoxelWorld, cells: &[IVec3], map: GridMap,
                rotate: &dyn Fn(BlockStateId, Orientation) -> BlockStateId) -> (EditReceipt, EditReceipt);
pub struct GridMap { pub orientation: Orientation, pub offset: IVec3 }
```
- Either both receipts happen or neither does. Cells are handled in a deterministic order.
- Used for contraption assembly and disassembly, construct splits and grid merges.
- cp-voxel never sees the registry; the caller passes `Registry::rotated` as `rotate`.

**6. Flag-filtered queries (breaking against the current §5.1 signature)**
- `raycast(grid, origin, dir, max_dist, flags: BlockFlags) -> Option<RayHit>`.
- `ray_cells(grid, origin, dir, max_dist, visit: impl FnMut(IVec3, Face, f64 /*entry*/, f64 /*exit*/) -> ControlFlow<()>)`. This is an exact DDA visitor: physics tests partial shapes per cell itself.
- `overlaps(grid, aabb, flags)`.
- `for_each_cell(grid, IAabb, flags, f)`, in index order.

**7. Fast level-0 reads and skipping empty space (additive)**
- `state(grid, cell) -> BlockStateId`: ≤ 25 ns on a resident chunk.
- `states_in(grid, IAabb, &mut [BlockStateId])`: in index order.
- `region_may_collide(grid, IAabb, solid: &[bool]) -> bool`: conservative and O(log n), using `classify` and `subtree_empty` plus an index of edited chunks. It lets large or high-flying bodies skip thousands of chunk keys; ocean chunks without SOLID cells take O(1).

**Keep as already contracted**
- `revision(grid, key)`: O(1), monotonic, changes when the 1-cell halo changes.
- The simulation reads level 0 only.
- The snapshot is mutated only at tick boundaries.

**cp-stream, later (E3) but fixed now**
- Simulation anchors are boxes of level-0 chunks with a priority. Physics supplies one per active body: its AABB swept over 2 s + 32 m, at most 2,048 chunks; `Empty` chunks are skipped.
- cp-stream counts blocking simulation reads, so the "physics waited" gate can be measured.
