//! Generation and meshing of planned chunks on the engine's job system, and their way into the
//! renderer's scene.
//!
//! One job per chunk: classify the chunk through the snapshot (an `Empty` chunk is done), sample
//! it with its one-cell halo, mesh it with the worker thread's own [`Mesher`], drop the wall
//! ranges the chunk will never draw, and check the result into a [`PreparedChunk`] (the checks
//! of cp-render run on the worker, contract §5.3). The frame thread only reserves pool space and
//! queues the quads ([`VoxelScene::insert_prepared`]).
//!
//! Results come back over a channel. Every job owns a [`Reply`] that sends an error when the job
//! ends without a result (a contained panic, a cancelled job), so a batch always completes.

use std::cell::RefCell;
use std::sync::Arc;
use std::sync::mpsc::{Receiver, Sender, TryRecvError, channel};
use std::time::{Duration, Instant};

use cp_content::BlockTables;
use cp_core::coords::LodKey;
use cp_core::quad::PackedQuad;
use cp_jobs::{JobClass, JobSystem, Priority};
use cp_mesh::{
    ChunkMesh, FACE_STORAGE_ORDER, Layer, MeshInput, Mesher, RANGES_PER_LAYER, WALL_RANGE_OFFSET,
};
use cp_render::voxel::{ChunkId, LayerCounts, ObjectId, PreparedChunk, VoxelScene, WallMask};
use cp_voxel::{NodeClass, VoxelSnapshot};

use crate::Failure;
use crate::assets::mesh_tables;
use crate::stats::Summary;

thread_local! {
    /// One mesher per worker thread (cp-mesh: reuse its scratch buffers).
    static MESHER: RefCell<Mesher> = RefCell::new(Mesher::new());
}

/// One chunk to generate, mesh and add to an object.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ChunkRequest {
    /// The chunk.
    pub key: LodKey,
    /// The wall ranges it draws; the others are dropped from its mesh.
    pub walls: WallMask,
    /// The object (grid) it belongs to.
    pub object: ObjectId,
    /// Distance from the camera (metres): nearer chunks are built first.
    pub distance: f64,
}

/// A chunk after its job.
#[derive(Debug)]
pub struct BuiltChunk {
    /// What was asked for.
    pub request: ChunkRequest,
    /// The class of the composite chunk.
    pub class: NodeClass,
    /// The checked mesh, `None` when the chunk draws nothing.
    pub prepared: Option<PreparedChunk>,
    /// Quads of the mesh as uploaded (walls it does not draw removed).
    pub quads: usize,
    /// Milliseconds of classification and generation (sampling with the halo).
    pub generate_ms: f64,
    /// Milliseconds of meshing, wall removal and upload checks.
    pub mesh_ms: f64,
}

type JobResult = Result<BuiltChunk, String>;

/// Sends the job's result, or an error when the job ends without one.
struct Reply {
    sender: Sender<JobResult>,
    key: LodKey,
    sent: bool,
}

impl Reply {
    fn send(mut self, result: JobResult) {
        self.sent = true;
        // A receiver that is gone has given up on the batch; nothing left to tell.
        let _ = self.sender.send(result);
    }
}

impl Drop for Reply {
    fn drop(&mut self) {
        if !self.sent {
            let _ = self.sender.send(Err(format!(
                "the build job of chunk {:?} ended without a result (panic or cancellation)",
                self.key
            )));
        }
    }
}

/// The chunks of one plan on their way through the job system.
#[derive(Debug)]
pub struct BuildBatch {
    receiver: Receiver<JobResult>,
    total: usize,
    received: usize,
    started: Instant,
}

impl BuildBatch {
    /// Queues one job per request on `jobs`, nearest first (priority key = distance in m).
    pub fn spawn(
        jobs: &JobSystem,
        class: JobClass,
        snapshot: &Arc<VoxelSnapshot>,
        tables: &Arc<BlockTables>,
        requests: &[ChunkRequest],
    ) -> Self {
        let (sender, receiver) = channel();
        for &request in requests {
            let reply = Reply {
                sender: sender.clone(),
                key: request.key,
                sent: false,
            };
            let snapshot = Arc::clone(snapshot);
            let tables = Arc::clone(tables);
            // Distances beyond u32 metres all count as the farthest.
            let key = request.distance.clamp(0.0, f64::from(u32::MAX)) as u32;
            jobs.spawn_detached(Priority::new(class, key), move |cancel| {
                let result = build_chunk(&snapshot, &tables, request, &|| cancel.is_cancelled());
                reply.send(result);
            });
        }
        Self {
            receiver,
            total: requests.len(),
            received: 0,
            started: Instant::now(),
        }
    }

    /// Requests in the batch.
    pub fn total(&self) -> usize {
        self.total
    }

    /// Results taken so far.
    pub fn received(&self) -> usize {
        self.received
    }

    /// True once every result has been taken.
    pub fn is_done(&self) -> bool {
        self.received >= self.total
    }

    /// Wall-clock time since the batch was queued.
    pub fn elapsed(&self) -> Duration {
        self.started.elapsed()
    }

    /// Moves the results that have arrived into `out` without waiting; returns how many.
    ///
    /// # Errors
    /// A [`Failure`] for a job that failed (meshing error, panic) and when the job system is
    /// gone before every result arrived.
    pub fn poll(&mut self, out: &mut Vec<BuiltChunk>) -> Result<usize, Failure> {
        let mut taken = 0;
        while self.received < self.total {
            match self.receiver.try_recv() {
                Ok(result) => {
                    self.received += 1;
                    taken += 1;
                    out.push(result.map_err(Failure::new)?);
                }
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => {
                    return Err(Failure::new(
                        "the job system stopped before every chunk was built",
                    ));
                }
            }
        }
        Ok(taken)
    }

    /// Waits up to `timeout` for the next result and moves everything that has arrived into
    /// `out`.
    ///
    /// # Errors
    /// Those of [`Self::poll`].
    pub fn wait(&mut self, timeout: Duration, out: &mut Vec<BuiltChunk>) -> Result<usize, Failure> {
        if self.received >= self.total {
            return Ok(0);
        }
        match self.receiver.recv_timeout(timeout) {
            Ok(result) => {
                self.received += 1;
                out.push(result.map_err(Failure::new)?);
                Ok(1 + self.poll(out)?)
            }
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => Ok(0),
            Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => Err(Failure::new(
                "the job system stopped before every chunk was built",
            )),
        }
    }
}

/// The work of one job.
fn build_chunk(
    snapshot: &VoxelSnapshot,
    tables: &BlockTables,
    request: ChunkRequest,
    cancel: &dyn Fn() -> bool,
) -> JobResult {
    let started = Instant::now();
    let key = request.key;
    let class = snapshot.classify(key);
    let empty = |generate_ms| BuiltChunk {
        request,
        class,
        prepared: None,
        quads: 0,
        generate_ms,
        mesh_ms: 0.0,
    };
    if class == NodeClass::Empty {
        return Ok(empty(ms_since(started)));
    }
    let padded = snapshot
        .try_sample_padded(key, cancel)
        .map_err(|_| format!("chunk {key:?}: generation was cancelled"))?;
    let generate_ms = ms_since(started);
    let meshed = Instant::now();
    let input = MeshInput {
        cells: padded.cells(),
        authored: padded.authored(),
        level: padded.level(),
        tables: mesh_tables(tables),
    };
    let mesh = MESHER
        .with(|mesher| mesher.borrow_mut().mesh(&input))
        .map_err(|error| format!("chunk {key:?}: {error}"))?;
    let (quads, layers) = upload_parts(&mesh, request.walls);
    if quads.is_empty() {
        return Ok(empty(generate_ms));
    }
    let count = quads.len();
    let prepared =
        PreparedChunk::from_vec(quads, layers, key.origin_voxel(), key.level, request.object)
            .map_err(|error| format!("chunk {key:?}: {error}"))?;
    Ok(BuiltChunk {
        request,
        class,
        prepared: Some(prepared),
        quads: count,
        generate_ms,
        mesh_ms: ms_since(meshed),
    })
}

fn ms_since(start: Instant) -> f64 {
    start.elapsed().as_secs_f64() * 1.0e3
}

/// The quads and range counts of `mesh` in the layout of cp-render's uploads, without the wall
/// ranges of faces outside `walls` (the chunk never draws them).
pub fn upload_parts(mesh: &ChunkMesh, walls: WallMask) -> (Vec<PackedQuad>, [LayerCounts; 3]) {
    let mut quads = Vec::with_capacity(mesh.quads.len());
    let mut layers = [LayerCounts::EMPTY; 3];
    for layer in Layer::ALL {
        let ranges = mesh.layer(layer);
        let mut slots = [0_u32; RANGES_PER_LAYER];
        for (slot, count) in slots.iter_mut().enumerate() {
            let drawn = slot < WALL_RANGE_OFFSET
                || walls.contains(FACE_STORAGE_ORDER[slot - WALL_RANGE_OFFSET]);
            if drawn {
                let range = ranges.slot(slot);
                *count = range.len() as u32;
                quads.extend_from_slice(&mesh.quads[range]);
            }
        }
        layers[layer.index()] = LayerCounts::from_slots(slots);
    }
    (quads, layers)
}

/// What a batch made, for logs and benchmarks.
#[derive(Clone, Debug, Default)]
pub struct BuildReport {
    /// Chunks asked for.
    pub requested: usize,
    /// Chunks the snapshot classified `Empty` (never generated).
    pub empty_class: usize,
    /// Chunks that were generated but draw nothing.
    pub empty_mesh: usize,
    /// Chunks with quads, added to the scene.
    pub drawn: usize,
    /// Chunks with quads per LOD level, index = level.
    pub drawn_per_level: Vec<usize>,
    /// Quads uploaded.
    pub quads: u64,
    /// Milliseconds of generation per generated chunk.
    pub generate_ms: Vec<f64>,
    /// Milliseconds of meshing per meshed chunk.
    pub mesh_ms: Vec<f64>,
    /// Wall-clock seconds from the first job to the last result.
    pub wall_seconds: f64,
}

impl BuildReport {
    /// Counts one built chunk.
    pub fn add(&mut self, built: &BuiltChunk) {
        self.requested += 1;
        if built.class == NodeClass::Empty {
            self.empty_class += 1;
            return;
        }
        self.generate_ms.push(built.generate_ms);
        if built.prepared.is_none() {
            self.empty_mesh += 1;
            return;
        }
        self.mesh_ms.push(built.mesh_ms);
        self.drawn += 1;
        let level = usize::from(built.request.key.level);
        if self.drawn_per_level.len() <= level {
            self.drawn_per_level.resize(level + 1, 0);
        }
        self.drawn_per_level[level] += 1;
        self.quads += built.quads as u64;
    }

    /// Chunks with quads on level 0.
    pub fn drawn_level0(&self) -> usize {
        self.drawn_per_level.first().copied().unwrap_or(0)
    }

    /// Generation times.
    pub fn generate_summary(&self) -> Summary {
        Summary::of(&self.generate_ms)
    }

    /// Meshing times.
    pub fn mesh_summary(&self) -> Summary {
        Summary::of(&self.mesh_ms)
    }

    /// One log line.
    pub fn line(&self) -> String {
        let generate = self.generate_summary();
        let mesh = self.mesh_summary();
        format!(
            "{} chunks planned, {} empty by class, {} empty meshes, {} drawn ({} on level 0, per \
             level {:?}), {:.2} M quads; generation {:.3} ms mean / {:.3} ms p95, meshing {:.3} ms \
             mean / {:.3} ms p95; {:.2} s wall clock ({:.0} chunks/s)",
            self.requested,
            self.empty_class,
            self.empty_mesh,
            self.drawn,
            self.drawn_level0(),
            self.drawn_per_level,
            self.quads as f64 / 1.0e6,
            generate.mean,
            generate.p95,
            mesh.mean,
            mesh.p95,
            self.wall_seconds,
            self.requested as f64 / self.wall_seconds.max(1.0e-9)
        )
    }
}

/// Adds a built chunk to the scene and sets its walls; `None` for a chunk without quads.
///
/// # Errors
/// A [`Failure`] when the scene refuses the chunk (pool or draw item table full).
pub fn insert(scene: &mut VoxelScene, built: BuiltChunk) -> Result<Option<ChunkId>, Failure> {
    let Some(prepared) = built.prepared else {
        return Ok(None);
    };
    let key = built.request.key;
    let id = scene.insert_prepared(prepared).map_err(|refused| {
        Failure::new(format!(
            "the renderer refused chunk {key:?}: {} (scene: {:?})",
            refused.error,
            scene.stats()
        ))
    })?;
    if !built.request.walls.is_empty() {
        scene
            .set_wall_mask(id, built.request.walls)
            .map_err(|error| Failure::new(format!("chunk {key:?}: {error}")))?;
    }
    Ok(Some(id))
}

/// Builds every request and adds the results to `scene`, logging a progress line about twice a
/// second; the ids of the added chunks go to `added`. Blocks until the batch is done.
///
/// # Errors
/// A [`Failure`] when a job fails or the scene refuses a chunk.
pub fn build_blocking(
    jobs: &JobSystem,
    snapshot: &Arc<VoxelSnapshot>,
    tables: &Arc<BlockTables>,
    requests: &[ChunkRequest],
    scene: &mut VoxelScene,
    label: &str,
    added: &mut Vec<ChunkId>,
) -> Result<BuildReport, Failure> {
    // The user waits for these chunks: all workers may build them (the bulk limit of contract
    // §4.2 keeps a core free for the frame thread while one runs; here nothing does).
    let mut batch = BuildBatch::spawn(jobs, JobClass::Interactive, snapshot, tables, requests);
    let mut report = BuildReport::default();
    let mut arrived = Vec::new();
    let mut last_line = Instant::now();
    while !batch.is_done() {
        batch.wait(Duration::from_millis(100), &mut arrived)?;
        for built in arrived.drain(..) {
            report.add(&built);
            added.extend(insert(scene, built)?);
        }
        if last_line.elapsed() >= Duration::from_millis(500) || batch.is_done() {
            last_line = Instant::now();
            log::info!(
                "{label}: {}/{} chunks ({:.0} %), {} drawn, {:.1} s",
                batch.received(),
                batch.total(),
                100.0 * batch.received() as f64 / batch.total().max(1) as f64,
                report.drawn,
                batch.elapsed().as_secs_f64()
            );
        }
    }
    report.wall_seconds = batch.elapsed().as_secs_f64();
    log::info!("{label}: {}", report.line());
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;
    use cp_core::face::Face;
    use cp_core::ids::{BlockStateId, MaterialId};
    use cp_mesh::{MeshTables, PADDED_VOLUME, padded_index};

    #[test]
    fn unused_wall_ranges_are_dropped_from_the_upload() {
        use cp_core::block::{BlockFlags, BlockRender, RenderClass};
        use cp_core::face::Orientation;
        let flags = [BlockFlags::empty(), BlockFlags::SOLID | BlockFlags::OPAQUE];
        let stone =
            BlockRender::oriented(RenderClass::Cube, [MaterialId(3); 6], Orientation::IDENTITY);
        let render = [BlockRender::NONE, stone];
        let cull_group = [0, 0];
        let tables = MeshTables {
            flags: &flags,
            render: &render,
            cull_group: &cull_group,
        };
        // Stone below y = 4 in the chunk and in its halo: walls on all four sides and the
        // bottom, a regular top face.
        let mut cells = vec![BlockStateId::AIR; PADDED_VOLUME];
        for y in -1..4 {
            for z in -1..=32 {
                for x in -1..=32 {
                    cells[padded_index(x, y, z)] = BlockStateId(1);
                }
            }
        }
        let input = MeshInput {
            cells: &cells,
            authored: None,
            level: 0,
            tables,
        };
        let mesh = Mesher::new().mesh(&input).expect("meshes");
        assert!(mesh.wall_len() > 0);
        let (all, all_layers) = upload_parts(&mesh, WallMask::ALL);
        assert_eq!(all.len(), mesh.quads.len());
        let (none, none_layers) = upload_parts(&mesh, WallMask::NONE);
        assert_eq!(none.len(), mesh.regular_len());
        assert!(none_layers.iter().all(|layer| layer.walls == [0; 6]));
        let (east, east_layers) = upload_parts(&mesh, WallMask::NONE.with(Face::PosX));
        assert_eq!(
            east.len(),
            mesh.regular_len() + mesh.wall_quads(Layer::Opaque, Face::PosX).len()
        );
        assert_eq!(
            east_layers[0].walls[0],
            mesh.wall_quads(Layer::Opaque, Face::PosX).len() as u32
        );
        // The counts add up to the quads in every variant, and every quad of a slot lies on
        // that slot's face (what cp-render checks when the chunk is prepared).
        for (quads, layers) in [(all, all_layers), (none, none_layers), (east, east_layers)] {
            let total: u64 = layers.iter().map(LayerCounts::total).sum();
            assert_eq!(total, quads.len() as u64);
            let mut next = 0;
            for layer in layers {
                for (slot, &count) in layer.slots().iter().enumerate() {
                    let face = FACE_STORAGE_ORDER[slot % 6];
                    for quad in &quads[next..next + count as usize] {
                        assert_eq!(quad.face(), face);
                    }
                    next += count as usize;
                }
            }
        }
    }
}
