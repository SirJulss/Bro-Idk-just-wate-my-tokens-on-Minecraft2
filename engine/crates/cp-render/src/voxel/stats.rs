//! Per-frame statistics of the voxel renderer, and the read-back of the GPU counters of draw
//! path A, whose visible draws only the GPU knows.

use std::sync::Arc;
use std::sync::atomic::{AtomicU8, Ordering};

use super::layout::{COUNTER_QUADS, COUNTER_WORDS};
use super::renderer::DrawMode;
use super::scene::UploadReport;
use crate::error::RenderError;
use crate::gpu::GpuContext;
use crate::scope::capture;

/// Draws and quads that passed culling in one frame.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct VisibleCounts {
    /// The frame they were counted in (path A reports a few frames late).
    pub frame: u64,
    /// Draws (visible buckets) of the opaque and the cutout pipeline.
    pub draws: [u32; 2],
    /// Quads in the visible ranges.
    pub quads: u64,
}

impl VisibleCounts {
    /// Draws of both pipelines.
    #[must_use]
    pub fn total_draws(&self) -> u32 {
        self.draws[0] + self.draws[1]
    }

    /// Triangles submitted: two per quad.
    #[must_use]
    pub fn triangles(&self) -> u64 {
        2 * self.quads
    }
}

/// What one call of [`VoxelRenderer::render`](super::VoxelRenderer::render) did.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FrameStats {
    /// The frame's index (counted by the frame graph from 0).
    pub frame: u64,
    /// The draw path used.
    pub mode: DrawMode,
    /// Live objects.
    pub objects: u32,
    /// Live chunks.
    pub chunks: u32,
    /// Draw items the cull looked at.
    pub draw_items: u32,
    /// Visible draws and quads: of this frame for path B and direct draws, of the latest frame
    /// whose GPU counters have arrived for path A (`None` until the first one arrives).
    pub visible: Option<VisibleCounts>,
    /// CPU time of the cull of path B and direct draws, milliseconds.
    pub cpu_cull_ms: Option<f64>,
    /// CPU time of the whole call, milliseconds.
    pub cpu_frame_ms: f64,
    /// The uploads of this frame.
    pub uploads: UploadReport,
}

const IDLE: u8 = 0;
const MAPPING: u8 = 1;
const READY: u8 = 2;
const FAILED: u8 = 3;

/// Frames whose counters can be in flight at once.
const SLOTS: usize = 4;

#[derive(Debug)]
struct Slot {
    buffer: wgpu::Buffer,
    state: Arc<AtomicU8>,
    /// The frame whose counters the slot holds while it is in flight.
    frame: Option<u64>,
}

/// A ring of small read-back buffers for the counter buffer of path A.
#[derive(Debug)]
pub(crate) struct CountersReadback {
    slots: Vec<Slot>,
    latest: Option<VisibleCounts>,
}

const COUNTER_BYTES: u64 = (COUNTER_WORDS * 4) as u64;

impl CountersReadback {
    pub(crate) fn new(gpu: &GpuContext) -> Result<Self, RenderError> {
        let buffers = capture(gpu, "voxel counter read-back", |device| {
            (0..SLOTS)
                .map(|_| {
                    device.create_buffer(&wgpu::BufferDescriptor {
                        label: Some("voxel counter read-back"),
                        size: COUNTER_BYTES,
                        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                        mapped_at_creation: false,
                    })
                })
                .collect::<Vec<_>>()
        })?;
        Ok(Self {
            slots: buffers
                .into_iter()
                .map(|buffer| Slot {
                    buffer,
                    state: Arc::new(AtomicU8::new(IDLE)),
                    frame: None,
                })
                .collect(),
            latest: None,
        })
    }

    /// A slot that no frame uses, or `None` when all are in flight (no counters this frame).
    pub(crate) fn acquire(&self) -> Option<usize> {
        self.slots.iter().position(|slot| slot.frame.is_none())
    }

    /// Records the copy of the counters into `slot`.
    pub(crate) fn encode(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        counters: &wgpu::Buffer,
        slot: usize,
    ) {
        encoder.copy_buffer_to_buffer(counters, 0, &self.slots[slot].buffer, 0, COUNTER_BYTES);
    }

    /// Starts mapping `slot` after the submission of `frame`.
    pub(crate) fn submitted(&mut self, slot: usize, frame: u64) {
        let slot = &mut self.slots[slot];
        slot.frame = Some(frame);
        slot.state.store(MAPPING, Ordering::Release);
        let state = Arc::clone(&slot.state);
        slot.buffer
            .map_async(wgpu::MapMode::Read, .., move |result| {
                state.store(
                    if result.is_ok() { READY } else { FAILED },
                    Ordering::Release,
                );
            });
    }

    /// Takes every mapped slot's counters without waiting.
    pub(crate) fn collect(&mut self) {
        for slot in &mut self.slots {
            let Some(frame) = slot.frame else {
                continue;
            };
            match slot.state.load(Ordering::Acquire) {
                READY => {
                    let words: Option<[u32; COUNTER_WORDS]> = match slot.buffer.get_mapped_range(..)
                    {
                        Ok(bytes) => {
                            let mut words = [0; COUNTER_WORDS];
                            for (word, chunk) in words.iter_mut().zip(bytes.as_chunks::<4>().0) {
                                *word = u32::from_le_bytes(*chunk);
                            }
                            Some(words)
                        }
                        Err(error) => {
                            log::debug!("voxel counters of frame {frame}: {error}");
                            None
                        }
                    };
                    slot.buffer.unmap();
                    slot.frame = None;
                    slot.state.store(IDLE, Ordering::Release);
                    if let Some(words) = words
                        && self.latest.is_none_or(|latest| latest.frame < frame)
                    {
                        self.latest = Some(VisibleCounts {
                            frame,
                            draws: [words[0], words[1]],
                            quads: u64::from(words[COUNTER_QUADS]),
                        });
                    }
                }
                FAILED => {
                    log::debug!("voxel counters of frame {frame} could not be read back");
                    slot.frame = None;
                    slot.state.store(IDLE, Ordering::Release);
                }
                _ => {}
            }
        }
    }

    /// The counters of the newest frame that arrived.
    pub(crate) fn latest(&self) -> Option<VisibleCounts> {
        self.latest
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn visible_counts_add_up() {
        let counts = VisibleCounts {
            frame: 3,
            draws: [10, 4],
            quads: 500,
        };
        assert_eq!(counts.total_draws(), 14);
        assert_eq!(counts.triangles(), 1000);
    }
}
