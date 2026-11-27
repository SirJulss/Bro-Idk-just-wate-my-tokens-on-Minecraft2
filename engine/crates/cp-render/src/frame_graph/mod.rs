//! The frame graph: declared passes over transient textures, a pool that keeps those textures
//! from frame to frame, and GPU timings per pass (contract §5.3 "Infrastructure", §7).
//!
//! A [`FrameGraphDesc`] declares, in execution order, the render and compute passes of a frame
//! and the transient textures they read and write (format and size relative to the output).
//! [`FrameGraph::new`] checks the declaration once. Each frame,
//!
//! 1. [`FrameGraph::begin_frame`] binds a texture to every transient, taking it from the pool
//!    when one of the same format, size and usage exists (after a [`FrameGraph::resize`], new
//!    ones are created and the old ones dropped), and opens a command encoder;
//! 2. the caller records the passes in their declared order through
//!    [`Frame::begin_render_pass`] and [`Frame::begin_compute_pass`], which add the timestamp
//!    writes; a pass may be skipped, none may run twice or out of order;
//! 3. [`Frame::submit`] resolves the timestamps, submits and starts their read-back.
//!
//! Timings of a frame arrive a few frames later ([`FrameGraph::latest_timings`]); the CPU
//! encode time of the last frame is there at once ([`FrameGraph::last_cpu_encode_ms`]). On a
//! device without `TIMESTAMP_QUERY` no query exists and the GPU part of the timings stays
//! empty.
//!
//! ```no_run
//! use cp_render::frame_graph::{FrameGraph, FrameGraphDesc, TransientDesc, TransientSize};
//! use cp_render::gpu::{GpuContext, GpuOptions};
//! use cp_render::wgpu;
//!
//! let gpu = GpuContext::new_blocking(&GpuOptions::default(), None)?;
//! let mut desc = FrameGraphDesc::new();
//! let hdr = desc.transient(TransientDesc::new(
//!     "hdr",
//!     wgpu::TextureFormat::Rgba16Float,
//!     TransientSize::FULL,
//! ));
//! let scene = desc.render_pass("scene", &[], &[hdr]);
//! let mut graph = FrameGraph::new(&gpu, desc, (1920, 1080))?;
//!
//! let mut frame = graph.begin_frame(&gpu)?;
//! let view = frame.view(hdr);
//! let pass = frame.begin_render_pass(scene, &wgpu::RenderPassDescriptor {
//!     color_attachments: &[Some(wgpu::RenderPassColorAttachment {
//!         view: &view,
//!         depth_slice: None,
//!         resolve_target: None,
//!         ops: wgpu::Operations {
//!             load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
//!             store: wgpu::StoreOp::Store,
//!         },
//!     })],
//!     ..Default::default()
//! })?;
//! drop(pass);
//! frame.submit()?;
//! if let Some(timings) = graph.latest_timings() {
//!     let _ = timings.gpu_total_ms();
//! }
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

mod desc;
mod pool;
mod timing;

use std::time::Instant;

pub use desc::{
    FrameGraphDesc, FrameGraphError, MAX_PASSES, PassDesc, PassId, PassKind, TransientDesc,
    TransientId, TransientSize,
};
pub use pool::PoolStats;
pub use timing::{FrameTimings, PassTiming};

use crate::error::RenderError;
use crate::gpu::GpuContext;
use crate::scope::capture;
use crate::texture::full_mip_level_count;
use pool::{PooledTexture, TextureKey, TransientPool};
use timing::GpuTimer;

/// A checked frame graph with its texture pool and timer.
#[derive(Debug)]
pub struct FrameGraph {
    desc: FrameGraphDesc,
    usages: Vec<wgpu::TextureUsages>,
    output: (u32, u32),
    pool: TransientPool,
    /// The textures of the current or last frame, by transient.
    bound: Vec<PooledTexture>,
    timer: GpuTimer,
    frames: u64,
    last_cpu_encode_ms: Option<f64>,
}

impl FrameGraph {
    /// Checks `desc` and prepares the graph for an output of `output_size` pixels (each side
    /// at least 1). No texture exists before the first frame.
    ///
    /// # Errors
    /// [`RenderError::FrameGraph`] for a wrong declaration, [`RenderError::Wgpu`] when the
    /// timestamp queries cannot be created.
    pub fn new(
        gpu: &GpuContext,
        desc: FrameGraphDesc,
        output_size: (u32, u32),
    ) -> Result<Self, RenderError> {
        desc.validate()?;
        let usages = (0..desc.transients().len())
            .map(|index| desc.usage(TransientId(index)))
            .collect();
        let timer = GpuTimer::new(gpu, desc.passes().len())?;
        Ok(Self {
            desc,
            usages,
            output: clamp_size(output_size),
            pool: TransientPool::default(),
            bound: Vec::new(),
            timer,
            frames: 0,
            last_cpu_encode_ms: None,
        })
    }

    /// The declaration.
    pub fn desc(&self) -> &FrameGraphDesc {
        &self.desc
    }

    /// The output size in pixels that transient sizes are relative to.
    pub fn output_size(&self) -> (u32, u32) {
        self.output
    }

    /// Sets the output size (each side at least 1). Transients whose size changes get new
    /// textures in the next frame; the old ones leave the pool then.
    pub fn resize(&mut self, width: u32, height: u32) {
        self.output = clamp_size((width, height));
    }

    /// The size of transient `id` at the current output size.
    ///
    /// # Panics
    /// When `id` is out of range. Ids are positions: an id of another graph that is in range
    /// is not detected and names the transient at that position.
    pub fn transient_size(&self, id: TransientId) -> (u32, u32) {
        self.desc.transients()[id.0].size.resolve(self.output)
    }

    /// The usages the texture of transient `id` has (see [`FrameGraphDesc::usage`]).
    ///
    /// # Panics
    /// When `id` is out of range. Ids are positions: an id of another graph that is in range
    /// is not detected and names the transient at that position.
    pub fn transient_usage(&self, id: TransientId) -> wgpu::TextureUsages {
        self.usages[id.0]
    }

    /// Whether the device writes timestamps, so that [`Self::latest_timings`] gets GPU times.
    pub fn gpu_timing_enabled(&self) -> bool {
        self.timer.is_enabled()
    }

    /// The latest frame whose GPU timings have arrived; `None` before the first one and on a
    /// device without timestamps.
    pub fn latest_timings(&self) -> Option<&FrameTimings> {
        self.timer.latest()
    }

    /// Takes the timings that have arrived since the last call, without waiting. Called by
    /// [`Self::begin_frame`]; useful after a wait for the GPU.
    pub fn collect_timings(&mut self) {
        self.timer.collect();
    }

    /// CPU time of the last submitted frame from [`Self::begin_frame`] to its finished command
    /// buffer, in milliseconds.
    pub fn last_cpu_encode_ms(&self) -> Option<f64> {
        self.last_cpu_encode_ms
    }

    /// What the texture pool holds.
    pub fn pool_stats(&self) -> PoolStats {
        self.pool.stats()
    }

    /// Frames begun so far; the index of the next frame.
    pub fn frames_begun(&self) -> u64 {
        self.frames
    }

    /// Begins a frame: binds a texture to every transient and opens the command encoder.
    ///
    /// # Errors
    /// [`RenderError::Device`] on a lost device, [`RenderError::Wgpu`] when a transient
    /// texture cannot be created (for example a format without the usage it needs).
    pub fn begin_frame<'g>(&'g mut self, gpu: &'g GpuContext) -> Result<Frame<'g>, RenderError> {
        let started = Instant::now();
        gpu.ensure_alive()?;
        self.timer.collect();
        let index = self.frames;
        self.frames += 1;
        self.bind_transients(gpu, index)?;
        let slot = self.timer.acquire();
        let encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("frame graph"),
            });
        Ok(Frame {
            graph: self,
            gpu,
            encoder,
            index,
            next_pass: 0,
            recorded: Vec::new(),
            slot,
            started,
        })
    }

    fn bind_transients(&mut self, gpu: &GpuContext, frame: u64) -> Result<(), RenderError> {
        let mut bound = Vec::with_capacity(self.desc.transients().len());
        for (transient, &usage) in self.desc.transients().iter().zip(&self.usages) {
            let (width, height) = transient.size.resolve(self.output);
            let key = TextureKey {
                format: transient.format,
                width,
                height,
                mip_level_count: transient
                    .mip_level_count
                    .min(full_mip_level_count(width, height)),
                usage,
            };
            bound.push(self.pool.acquire(gpu, &key, &transient.name, frame)?);
        }
        self.pool.evict_unused(frame);
        self.bound = bound;
        Ok(())
    }
}

fn clamp_size((width, height): (u32, u32)) -> (u32, u32) {
    (width.max(1), height.max(1))
}

/// One frame being recorded; ends with [`Frame::submit`]. Dropping it discards the frame.
#[derive(Debug)]
pub struct Frame<'g> {
    graph: &'g mut FrameGraph,
    gpu: &'g GpuContext,
    encoder: wgpu::CommandEncoder,
    index: u64,
    /// Index of the first pass that may still begin.
    next_pass: usize,
    /// The passes begun, in order; pass `k` of it wrote timestamps `2k` and `2k + 1`.
    recorded: Vec<PassId>,
    /// The timer's ring slot of this frame; `None` without timestamps or with a full ring.
    slot: Option<usize>,
    started: Instant,
}

impl Frame<'_> {
    /// The frame's number, counted from 0.
    pub fn index(&self) -> u64 {
        self.index
    }

    /// The output size of the frame in pixels.
    pub fn output_size(&self) -> (u32, u32) {
        self.graph.output
    }

    /// The texture of transient `id` in this frame (a cheap handle clone).
    ///
    /// # Panics
    /// When `id` is out of range. Ids are positions: an id of another graph that is in range
    /// is not detected and names the transient at that position.
    pub fn texture(&self, id: TransientId) -> wgpu::Texture {
        self.graph.bound[id.0].texture.clone()
    }

    /// A view over all mip levels of transient `id` in this frame (a cheap handle clone).
    ///
    /// # Panics
    /// When `id` is out of range. Ids are positions: an id of another graph that is in range
    /// is not detected and names the transient at that position.
    pub fn view(&self, id: TransientId) -> wgpu::TextureView {
        self.graph.bound[id.0].view.clone()
    }

    /// The size of transient `id` in this frame.
    ///
    /// # Panics
    /// When `id` is out of range. Ids are positions: an id of another graph that is in range
    /// is not detected and names the transient at that position.
    pub fn transient_size(&self, id: TransientId) -> (u32, u32) {
        self.graph.transient_size(id)
    }

    /// The command encoder, for copies and clears between passes.
    pub fn encoder(&mut self) -> &mut wgpu::CommandEncoder {
        &mut self.encoder
    }

    /// Begins render pass `pass` with the attachments of `desc`. The graph sets the
    /// timestamp writes (whatever `desc` has there is replaced) and, when `desc` has none, the
    /// label.
    ///
    /// # Errors
    /// [`FrameGraphError::UnknownPass`], [`FrameGraphError::WrongPassKind`] and
    /// [`FrameGraphError::OutOfOrder`]; nothing is recorded then.
    pub fn begin_render_pass(
        &mut self,
        pass: PassId,
        desc: &wgpu::RenderPassDescriptor<'_>,
    ) -> Result<wgpu::RenderPass<'_>, FrameGraphError> {
        let query = self.enter(pass, PassKind::Render)?;
        let name = self.graph.desc.passes()[pass.0].name.as_str();
        let timestamp_writes = query
            .zip(self.graph.timer.query_set())
            .map(|(first, query_set)| wgpu::RenderPassTimestampWrites {
                query_set,
                beginning_of_pass_write_index: Some(first),
                end_of_pass_write_index: Some(first + 1),
            });
        let desc = wgpu::RenderPassDescriptor {
            label: desc.label.or(Some(name)),
            timestamp_writes,
            ..desc.clone()
        };
        Ok(self.encoder.begin_render_pass(&desc))
    }

    /// Begins compute pass `pass`, labelled with its name, with the timestamp writes.
    ///
    /// # Errors
    /// [`FrameGraphError::UnknownPass`], [`FrameGraphError::WrongPassKind`] and
    /// [`FrameGraphError::OutOfOrder`]; nothing is recorded then.
    pub fn begin_compute_pass(
        &mut self,
        pass: PassId,
    ) -> Result<wgpu::ComputePass<'_>, FrameGraphError> {
        let query = self.enter(pass, PassKind::Compute)?;
        let name = self.graph.desc.passes()[pass.0].name.as_str();
        let timestamp_writes = query
            .zip(self.graph.timer.query_set())
            .map(|(first, query_set)| wgpu::ComputePassTimestampWrites {
                query_set,
                beginning_of_pass_write_index: Some(first),
                end_of_pass_write_index: Some(first + 1),
            });
        Ok(self
            .encoder
            .begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(name),
                timestamp_writes,
            }))
    }

    /// Checks that `pass` may begin now and returns the first of its two timestamp indices
    /// when the frame is timed.
    fn enter(&mut self, pass: PassId, kind: PassKind) -> Result<Option<u32>, FrameGraphError> {
        let passes = self.graph.desc.passes();
        let Some(declared) = passes.get(pass.0) else {
            return Err(FrameGraphError::UnknownPass { index: pass.0 });
        };
        if declared.kind != kind {
            return Err(FrameGraphError::WrongPassKind {
                pass: declared.name.clone(),
                declared: declared.kind,
                begun: kind,
            });
        }
        if pass.0 < self.next_pass {
            return Err(FrameGraphError::OutOfOrder {
                pass: declared.name.clone(),
                previous: passes[self.next_pass - 1].name.clone(),
            });
        }
        self.next_pass = pass.0 + 1;
        let first = u32::try_from(2 * self.recorded.len()).unwrap_or(u32::MAX);
        self.recorded.push(pass);
        Ok(self.slot.map(|_| first))
    }

    /// Ends the frame: resolves the timestamps, finishes the command buffer and submits it.
    ///
    /// # Errors
    /// [`RenderError::Wgpu`] when wgpu rejects the recorded commands (the frame is lost),
    /// [`RenderError::Device`] on a lost device.
    pub fn submit(self) -> Result<wgpu::SubmissionIndex, RenderError> {
        let Frame {
            graph,
            gpu,
            mut encoder,
            index,
            recorded,
            slot,
            started,
            ..
        } = self;
        let timed_slot = slot.filter(|_| !recorded.is_empty());
        if let Some(slot) = timed_slot {
            graph
                .timer
                .encode_resolve(&mut encoder, slot, recorded.len());
        }
        let submitted = capture(gpu, "frame graph submission", |_| {
            let commands = encoder.finish();
            let encode_ms = started.elapsed().as_secs_f64() * 1.0e3;
            (gpu.queue.submit([commands]), encode_ms)
        });
        let (submission, encode_ms) = match submitted {
            Ok(submitted) => submitted,
            Err(error) => {
                if let Some(slot) = slot {
                    graph.timer.release(slot);
                }
                return Err(error);
            }
        };
        graph.last_cpu_encode_ms = Some(encode_ms);
        match (slot, timed_slot) {
            (_, Some(slot)) => {
                let passes = recorded
                    .iter()
                    .map(|&pass| (pass, graph.desc.passes()[pass.0].name.clone()))
                    .collect();
                graph.timer.submitted(slot, index, passes, encode_ms);
            }
            (Some(slot), None) => graph.timer.release(slot),
            (None, None) => {}
        }
        Ok(submission)
    }
}
