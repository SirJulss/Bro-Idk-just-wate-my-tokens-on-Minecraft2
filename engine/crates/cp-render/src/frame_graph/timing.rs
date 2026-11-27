//! GPU time per pass from timestamp queries (contract §5.3 "Infrastructure", §2.6).
//!
//! One query set holds two timestamps per pass, written at the beginning and the end of the
//! pass. At the end of a frame the written queries are resolved into one region of a resolve
//! buffer (regions at 256-byte offsets, one per ring slot) and copied into that slot's
//! read-back buffer, which is mapped after the submission. Results arrive a few frames later
//! and never block the frame. Without `TIMESTAMP_QUERY` nothing is created and every
//! operation does nothing.

use std::sync::Arc;
use std::sync::atomic::{AtomicU8, Ordering};

use super::PassId;
use crate::error::RenderError;
use crate::gpu::GpuContext;
use crate::scope::capture;

/// Frames whose timings can be in flight at once.
const RING_SLOTS: usize = 4;

/// States of a slot's mapping, set by the map callback.
const MAP_WAITING: u8 = 0;
const MAP_READY: u8 = 1;
const MAP_FAILED: u8 = 2;

/// The GPU time of one pass of one frame.
#[derive(Clone, Debug, PartialEq)]
pub struct PassTiming {
    /// The pass.
    pub pass: PassId,
    /// Its declared name.
    pub name: String,
    /// Time from the beginning to the end of the pass on the GPU, in milliseconds; `>= 0`.
    pub gpu_ms: f64,
}

/// The measured times of one frame.
#[derive(Clone, Debug, PartialEq)]
pub struct FrameTimings {
    /// The frame, counted from 0 by `FrameGraph::begin_frame`.
    pub frame: u64,
    /// CPU time from the beginning of the frame to its finished command buffer, in
    /// milliseconds: transient binding, pass recording and the query resolve.
    pub cpu_encode_ms: f64,
    /// One entry per pass the frame recorded, in execution order.
    pub passes: Vec<PassTiming>,
}

impl FrameTimings {
    /// Sum of the pass times in milliseconds (gaps between passes are not included).
    pub fn gpu_total_ms(&self) -> f64 {
        self.passes.iter().map(|pass| pass.gpu_ms).sum()
    }

    /// The time of the pass called `name`.
    pub fn pass(&self, name: &str) -> Option<&PassTiming> {
        self.passes.iter().find(|pass| pass.name == name)
    }
}

/// Durations in milliseconds of `(begin, end)` tick pairs, with `period_ns` nanoseconds per
/// tick. A pair whose end lies before its begin (timestamps of different engines or a
/// wrapped counter) counts as zero.
pub(crate) fn durations_ms(ticks: &[u64], period_ns: f64) -> Vec<f64> {
    let (pairs, _) = ticks.as_chunks::<2>();
    pairs
        .iter()
        .map(|&[begin, end]| end.saturating_sub(begin) as f64 * period_ns / 1.0e6)
        .collect()
}

#[derive(Debug)]
enum Phase {
    Free,
    /// Taken by the frame being recorded.
    Acquired,
    /// Submitted and mapping.
    Pending {
        frame: u64,
        passes: Vec<(PassId, String)>,
        cpu_encode_ms: f64,
        bytes: u64,
    },
}

#[derive(Debug)]
struct Slot {
    readback: wgpu::Buffer,
    map_state: Arc<AtomicU8>,
    phase: Phase,
}

#[derive(Debug)]
struct Resources {
    query_set: wgpu::QuerySet,
    resolve: wgpu::Buffer,
    /// Bytes of one slot's region in the resolve buffer, a multiple of 256.
    stride: u64,
    /// Nanoseconds per timestamp tick.
    period_ns: f64,
    slots: Vec<Slot>,
}

/// The timestamp queries of a frame graph.
#[derive(Debug)]
pub(crate) struct GpuTimer {
    resources: Option<Resources>,
    latest: Option<FrameTimings>,
}

impl GpuTimer {
    /// A timer for up to `passes` passes per frame; inert without `TIMESTAMP_QUERY`.
    pub(crate) fn new(gpu: &GpuContext, passes: usize) -> Result<Self, RenderError> {
        if !gpu.caps.derived.can_timestamp_passes || passes == 0 {
            return Ok(Self {
                resources: None,
                latest: None,
            });
        }
        let queries = u32::try_from(2 * passes).unwrap_or(u32::MAX);
        let stride = (u64::from(queries) * u64::from(wgpu::QUERY_SIZE))
            .next_multiple_of(wgpu::QUERY_RESOLVE_BUFFER_ALIGNMENT);
        let (query_set, resolve, slots) = capture(gpu, "frame graph timestamps", |device| {
            let query_set = device.create_query_set(&wgpu::QuerySetDescriptor {
                label: Some("frame graph timestamps"),
                ty: wgpu::QueryType::Timestamp,
                count: queries,
            });
            let resolve = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("frame graph timestamp resolve"),
                size: stride * RING_SLOTS as u64,
                usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            let slots: Vec<Slot> = (0..RING_SLOTS)
                .map(|_| Slot {
                    readback: device.create_buffer(&wgpu::BufferDescriptor {
                        label: Some("frame graph timestamp read-back"),
                        size: stride,
                        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                        mapped_at_creation: false,
                    }),
                    map_state: Arc::new(AtomicU8::new(MAP_WAITING)),
                    phase: Phase::Free,
                })
                .collect();
            (query_set, resolve, slots)
        })?;
        Ok(Self {
            resources: Some(Resources {
                query_set,
                resolve,
                stride,
                period_ns: f64::from(gpu.queue.get_timestamp_period()),
                slots,
            }),
            latest: None,
        })
    }

    /// Whether the device can time passes.
    pub(crate) fn is_enabled(&self) -> bool {
        self.resources.is_some()
    }

    /// The query set the passes write to.
    pub(crate) fn query_set(&self) -> Option<&wgpu::QuerySet> {
        self.resources
            .as_ref()
            .map(|resources| &resources.query_set)
    }

    /// The latest frame whose timings have arrived.
    pub(crate) fn latest(&self) -> Option<&FrameTimings> {
        self.latest.as_ref()
    }

    /// A free ring slot for the frame being begun, or `None` (no timing this frame) when all
    /// are in flight. A slot still taken by an earlier frame belonged to a frame that was
    /// dropped without being submitted and is free again.
    pub(crate) fn acquire(&mut self) -> Option<usize> {
        let resources = self.resources.as_mut()?;
        for slot in &mut resources.slots {
            if matches!(slot.phase, Phase::Acquired) {
                slot.phase = Phase::Free;
            }
        }
        let index = resources
            .slots
            .iter()
            .position(|slot| matches!(slot.phase, Phase::Free))?;
        resources.slots[index].phase = Phase::Acquired;
        Some(index)
    }

    /// Frees `slot` without results (the frame recorded no pass or failed to submit).
    pub(crate) fn release(&mut self, slot: usize) {
        if let Some(resources) = &mut self.resources {
            resources.slots[slot].phase = Phase::Free;
        }
    }

    /// Records the resolve of the first `passes` passes' queries into `slot`.
    pub(crate) fn encode_resolve(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        slot: usize,
        passes: usize,
    ) {
        let Some(resources) = &self.resources else {
            return;
        };
        let queries = u32::try_from(2 * passes).unwrap_or(u32::MAX);
        let offset = slot as u64 * resources.stride;
        encoder.resolve_query_set(&resources.query_set, 0..queries, &resources.resolve, offset);
        encoder.copy_buffer_to_buffer(
            &resources.resolve,
            offset,
            &resources.slots[slot].readback,
            0,
            u64::from(queries) * u64::from(wgpu::QUERY_SIZE),
        );
    }

    /// Starts reading `slot` back after the submission of `frame`, which recorded `passes`.
    pub(crate) fn submitted(
        &mut self,
        slot: usize,
        frame: u64,
        passes: Vec<(PassId, String)>,
        cpu_encode_ms: f64,
    ) {
        let Some(resources) = &mut self.resources else {
            return;
        };
        let bytes = passes.len() as u64 * 2 * u64::from(wgpu::QUERY_SIZE);
        let slot = &mut resources.slots[slot];
        slot.map_state.store(MAP_WAITING, Ordering::Release);
        let state = Arc::clone(&slot.map_state);
        slot.readback
            .map_async(wgpu::MapMode::Read, 0..bytes, move |result| {
                let value = if result.is_ok() {
                    MAP_READY
                } else {
                    MAP_FAILED
                };
                state.store(value, Ordering::Release);
            });
        slot.phase = Phase::Pending {
            frame,
            passes,
            cpu_encode_ms,
            bytes,
        };
    }

    /// Takes the results of every slot whose mapping has completed; never waits. The map
    /// callbacks run when the device is polled or work is submitted.
    pub(crate) fn collect(&mut self) {
        let Some(resources) = &mut self.resources else {
            return;
        };
        for slot in &mut resources.slots {
            if !matches!(slot.phase, Phase::Pending { .. }) {
                continue;
            }
            let state = slot.map_state.load(Ordering::Acquire);
            if state == MAP_WAITING {
                continue;
            }
            let Phase::Pending {
                frame,
                passes,
                cpu_encode_ms,
                bytes,
            } = std::mem::replace(&mut slot.phase, Phase::Free)
            else {
                continue;
            };
            if state == MAP_FAILED {
                log::debug!("timestamps of frame {frame} could not be read back");
                continue;
            }
            let ticks: Option<Vec<u64>> = match slot.readback.get_mapped_range(0..bytes) {
                Ok(mapped) => Some(
                    mapped
                        .as_chunks::<8>()
                        .0
                        .iter()
                        .map(|word| u64::from_le_bytes(*word))
                        .collect(),
                ),
                Err(error) => {
                    log::debug!("timestamps of frame {frame}: {error}");
                    None
                }
            };
            slot.readback.unmap();
            let Some(ticks) = ticks else {
                continue;
            };
            let newer = self
                .latest
                .as_ref()
                .is_none_or(|latest| latest.frame < frame);
            if newer {
                let durations = durations_ms(&ticks, resources.period_ns);
                self.latest = Some(FrameTimings {
                    frame,
                    cpu_encode_ms,
                    passes: passes
                        .into_iter()
                        .zip(durations)
                        .map(|((pass, name), gpu_ms)| PassTiming { pass, name, gpu_ms })
                        .collect(),
                });
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ticks_become_non_negative_milliseconds() {
        let ticks = [1_000, 3_500_000, 10, 10, 9_000, 8_000];
        assert_eq!(durations_ms(&ticks, 1.0), [3.499, 0.0, 0.0]);
        // A period of 2.5 ns per tick (some AMD and Intel GPUs).
        assert_eq!(durations_ms(&[0, 400_000], 2.5), [1.0]);
        // A trailing unpaired value is ignored.
        assert_eq!(durations_ms(&[5, 1_000_005, 7], 1.0), [1.0]);
        assert!(durations_ms(&[], 1.0).is_empty());
    }

    #[test]
    fn frame_totals_sum_the_passes() {
        let timings = FrameTimings {
            frame: 3,
            cpu_encode_ms: 0.25,
            passes: vec![
                PassTiming {
                    pass: PassId(0),
                    name: "a".to_owned(),
                    gpu_ms: 0.5,
                },
                PassTiming {
                    pass: PassId(1),
                    name: "b".to_owned(),
                    gpu_ms: 1.25,
                },
            ],
        };
        assert_eq!(timings.gpu_total_ms(), 1.75);
        assert_eq!(timings.pass("b").map(|pass| pass.pass), Some(PassId(1)));
        assert!(timings.pass("c").is_none());
    }
}
