//! Probe draws that show what a vertex shader sees of the arguments of a draw call, and the
//! check built on them: every draw-path promise of a capability record against the real GPU.
//!
//! Two probes exist. The builtin probe reads `@builtin(vertex_index)` and
//! `@builtin(instance_index)` of a non-indexed draw (the diagnostic booleans of `GpuDerived`).
//! The bucket probe is the mechanism of contract §5.3: an indexed draw whose vertex shader
//! reads an instance-step vertex attribute, addressed by the `first_instance` of the draw.
//!
//! This file is a module of two test crates: of the adapter tests in `tests/gpu_context.rs`,
//! which use the public API only, and of the crate's unit tests (`gpu::probe`), which can build
//! contexts on instances the public API does not offer. Both have `DrawPath` and `GpuContext`
//! in the parent module.

use super::{DrawPath, GpuContext};

/// Draw arguments of the probes. The first vertex is a multiple of three so that the shader
/// can tell the triangle corner from `vertex_index` whether or not the offset is included.
const PROBE_FIRST_VERTEX: u32 = 300;
const PROBE_FIRST_INSTANCE: u32 = 7;

/// What the builtin probe writes when it sees both draw arguments in its builtins.
const PROBE_EXPECTED: [u32; 4] = [PROBE_FIRST_VERTEX, PROBE_FIRST_INSTANCE, 1, 1];

/// Entries of the bucket buffer; entry `i` holds `bucket(i)`.
const BUCKET_SLOTS: u32 = 16;

/// The bucket stored in entry `slot`: distinct per slot, never zero.
const fn bucket(slot: u32) -> [u32; 2] {
    [1000 + slot, 2000 + slot]
}

/// What the bucket probe writes when the attribute was fetched at `PROBE_FIRST_INSTANCE`.
const BUCKET_EXPECTED: [u32; 4] = [
    bucket(PROBE_FIRST_INSTANCE)[0],
    bucket(PROBE_FIRST_INSTANCE)[1],
    1,
    1,
];

/// One triangle over the whole target; every pixel records the vertex index of the first
/// corner and the instance index as the vertex shader saw them.
const PROBE_SHADER: &str = "
struct Corner {
    @builtin(position) position: vec4<f32>,
    @location(0) @interpolate(flat) seen: vec2<u32>,
}

@vertex
fn vertex(@builtin(vertex_index) vertex: u32, @builtin(instance_index) instance: u32) -> Corner {
    let corner = vertex % 3u;
    let xy = vec2<f32>(f32(corner & 1u), f32(corner >> 1u)) * 4.0 - 1.0;
    return Corner(vec4<f32>(xy, 0.0, 1.0), vec2<u32>(vertex - corner, instance));
}

@fragment
fn fragment(corner: Corner) -> @location(0) vec4<u32> {
    return vec4<u32>(corner.seen, 1u, 1u);
}
";

/// The same triangle, drawn indexed; every pixel records the bucket attribute of the instance.
/// Neither builtin offset is used, as in the voxel renderer.
const BUCKET_SHADER: &str = "
struct Corner {
    @builtin(position) position: vec4<f32>,
    @location(0) @interpolate(flat) bucket: vec2<u32>,
}

@vertex
fn vertex(@builtin(vertex_index) vertex: u32, @location(0) bucket: vec2<u32>) -> Corner {
    let corner = vertex % 3u;
    let xy = vec2<f32>(f32(corner & 1u), f32(corner >> 1u)) * 4.0 - 1.0;
    return Corner(vec4<f32>(xy, 0.0, 1.0), bucket);
}

@fragment
fn fragment(corner: Corner) -> @location(0) vec4<u32> {
    return vec4<u32>(corner.bucket, 1u, 1u);
}
";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ProbeDraw {
    /// Builtin probe, `draw`.
    Direct,
    /// Builtin probe, `multi_draw_indirect`.
    Indirect,
    /// Builtin probe, `multi_draw_indirect_count`.
    IndirectCount,
    /// Bucket probe, `draw_indexed`.
    BucketDirect,
    /// Bucket probe, `multi_draw_indexed_indirect` (draw path B).
    BucketIndirect,
    /// Bucket probe, `multi_draw_indexed_indirect_count` (draw path A).
    BucketIndirectCount,
}

impl ProbeDraw {
    fn reads_bucket(self) -> bool {
        matches!(
            self,
            Self::BucketDirect | Self::BucketIndirect | Self::BucketIndirectCount
        )
    }
}

/// Draws the probe triangle with `draw` into a 1x1 target and returns its texel: for the
/// builtin probe `[first vertex seen, instance seen, 1, 1]`, for the bucket probe
/// `[bucket.x, bucket.y, 1, 1]`, or zeros when nothing was drawn.
fn probe_draw(gpu: &GpuContext, draw: ProbeDraw) -> [u32; 4] {
    use wgpu::util::DeviceExt;

    let device = &gpu.device;
    // Errors do not panic in this context, so they have to be collected to be noticed.
    let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);

    let source = if draw.reads_bucket() {
        BUCKET_SHADER
    } else {
        PROBE_SHADER
    };
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("cp-test-probe"),
        source: wgpu::ShaderSource::Wgsl(source.into()),
    });
    let bucket_attributes = wgpu::vertex_attr_array![0 => Uint32x2];
    // Vertex buffer 0, as in the voxel renderer: Uint32x2, one entry per instance.
    let bucket_layout = [Some(wgpu::VertexBufferLayout {
        array_stride: 8,
        step_mode: wgpu::VertexStepMode::Instance,
        attributes: &bucket_attributes,
    })];
    let vertex_buffers: &[Option<wgpu::VertexBufferLayout<'_>>] = if draw.reads_bucket() {
        &bucket_layout
    } else {
        &[]
    };
    let format = wgpu::TextureFormat::Rgba32Uint;
    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("cp-test-probe"),
        layout: None,
        vertex: wgpu::VertexState {
            module: &module,
            entry_point: Some("vertex"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            buffers: vertex_buffers,
        },
        fragment: Some(wgpu::FragmentState {
            module: &module,
            entry_point: Some("fragment"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            targets: &[Some(format.into())],
        }),
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        multiview_mask: None,
        cache: None,
    });

    let size = wgpu::Extent3d {
        width: 1,
        height: 1,
        depth_or_array_layers: 1,
    };
    let target = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("cp-test-probe-target"),
        size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let view = target.create_view(&wgpu::TextureViewDescriptor::default());

    let arguments = wgpu::util::DrawIndirectArgs {
        vertex_count: 3,
        instance_count: 1,
        first_vertex: PROBE_FIRST_VERTEX,
        first_instance: PROBE_FIRST_INSTANCE,
    };
    let indirect = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("cp-test-probe-arguments"),
        contents: arguments.as_bytes(),
        usage: wgpu::BufferUsages::INDIRECT,
    });
    // The arguments of the voxel renderer: no index or vertex offset, the bucket slot as first
    // instance.
    let indexed_arguments = wgpu::util::DrawIndexedIndirectArgs {
        index_count: 3,
        instance_count: 1,
        first_index: 0,
        base_vertex: 0,
        first_instance: PROBE_FIRST_INSTANCE,
    };
    let indexed_indirect = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("cp-test-probe-indexed-arguments"),
        contents: indexed_arguments.as_bytes(),
        usage: wgpu::BufferUsages::INDIRECT,
    });
    let count = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("cp-test-probe-count"),
        contents: &1u32.to_le_bytes(),
        usage: wgpu::BufferUsages::INDIRECT,
    });
    let indices: Vec<u8> = [0u32, 1, 2]
        .iter()
        .flat_map(|index| index.to_le_bytes())
        .collect();
    let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("cp-test-probe-indices"),
        contents: &indices,
        usage: wgpu::BufferUsages::INDEX,
    });
    let buckets: Vec<u8> = (0..BUCKET_SLOTS)
        .flat_map(bucket)
        .flat_map(u32::to_le_bytes)
        .collect();
    let bucket_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("cp-test-probe-buckets"),
        contents: &buckets,
        usage: wgpu::BufferUsages::VERTEX,
    });
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("cp-test-probe-readback"),
        size: 16,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("cp-test-probe"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &view,
                resolve_target: None,
                depth_slice: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            ..Default::default()
        });
        pass.set_pipeline(&pipeline);
        if draw.reads_bucket() {
            pass.set_vertex_buffer(0, bucket_buffer.slice(..));
            pass.set_index_buffer(index_buffer.slice(..), wgpu::IndexFormat::Uint32);
        }
        match draw {
            ProbeDraw::Direct => pass.draw(
                PROBE_FIRST_VERTEX..PROBE_FIRST_VERTEX + 3,
                PROBE_FIRST_INSTANCE..PROBE_FIRST_INSTANCE + 1,
            ),
            ProbeDraw::Indirect => pass.multi_draw_indirect(&indirect, 0, 1),
            ProbeDraw::IndirectCount => pass.multi_draw_indirect_count(&indirect, 0, &count, 0, 1),
            ProbeDraw::BucketDirect => {
                pass.draw_indexed(0..3, 0, PROBE_FIRST_INSTANCE..PROBE_FIRST_INSTANCE + 1);
            }
            ProbeDraw::BucketIndirect => pass.multi_draw_indexed_indirect(&indexed_indirect, 0, 1),
            ProbeDraw::BucketIndirectCount => {
                pass.multi_draw_indexed_indirect_count(&indexed_indirect, 0, &count, 0, 1);
            }
        }
    }
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture: &target,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &readback,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: None,
                rows_per_image: None,
            },
        },
        size,
    );
    gpu.queue.submit([encoder.finish()]);

    readback.slice(..).map_async(wgpu::MapMode::Read, |result| {
        result.expect("map the probe readback buffer");
    });
    // Through the context, as all engine code polls: a device lost here is an error, not a
    // panic inside wgpu.
    gpu.poll(wgpu::PollType::wait_indefinitely())
        .expect("wait for the probe to finish");
    let mut texel = [0u32; 4];
    {
        let bytes = readback
            .slice(..)
            .get_mapped_range()
            .expect("the readback buffer is mapped");
        let (words, rest) = bytes.as_chunks::<4>();
        assert!(
            words.len() == texel.len() && rest.is_empty(),
            "one Rgba32Uint texel"
        );
        for (word, bytes) in texel.iter_mut().zip(words) {
            *word = u32::from_le_bytes(*bytes);
        }
    }
    readback.unmap();

    let error = futures_lite::future::block_on(scope.pop());
    assert!(
        error.is_none(),
        "the {draw:?} probe caused a validation error: {error:?}"
    );
    texel
}

/// Holds the draw path of `gpu.caps` and every `true` among its draw booleans against what the
/// GPU really does; a `false` promises nothing and is only logged together with the
/// observation.
///
/// The booleans are derived from reading the sources of wgpu, which is why they are checked
/// here on every adapter the tests can get hold of.
pub(super) fn check_indirect_draw_promises(test: &str, gpu: &GpuContext) {
    let derived = gpu.caps.derived;
    let backend = gpu.caps.adapter.backend;

    // Controls: direct draws deliver both arguments, and the bucket of the first instance, on
    // every backend. They prove the probes.
    let seen = probe_draw(gpu, ProbeDraw::Direct);
    assert_eq!(seen, PROBE_EXPECTED, "direct draw on {backend}");
    let seen = probe_draw(gpu, ProbeDraw::BucketDirect);
    assert_eq!(seen, BUCKET_EXPECTED, "direct indexed draw on {backend}");

    if !gpu
        .caps
        .enabled_features
        .contains(wgpu::Features::INDIRECT_FIRST_INSTANCE)
    {
        // A non-zero first_instance is not allowed in indirect arguments on this device.
        assert_eq!(derived.draw_path, DrawPath::B);
        assert!(!derived.can_indirect_first_instance);
        log::warn!("{test}: {backend}: indirect draws skipped, no INDIRECT_FIRST_INSTANCE");
        return;
    }

    let seen = probe_draw(gpu, ProbeDraw::Indirect);
    let promised = derived.indirect_draws_respect_first_vertex_and_instance;
    log::info!("{backend}: multi_draw_indirect shader saw {seen:?}, promised: {promised}");
    if promised {
        assert_eq!(seen, PROBE_EXPECTED, "multi_draw_indirect on {backend}");
    }

    // Draw path B: the bucket of first_instance reaches the shader through the attribute.
    let seen = probe_draw(gpu, ProbeDraw::BucketIndirect);
    let promised = derived.can_indirect_first_instance;
    log::info!("{backend}: multi_draw_indexed_indirect read bucket {seen:?}, promised: {promised}");
    if promised {
        assert_eq!(
            seen, BUCKET_EXPECTED,
            "multi_draw_indexed_indirect (path B) on {backend}"
        );
    }

    if derived.can_multi_draw_indirect_count {
        let seen = probe_draw(gpu, ProbeDraw::IndirectCount);
        let promised = derived.indirect_count_draws_respect_first_vertex_and_instance;
        log::info!(
            "{backend}: multi_draw_indirect_count shader saw {seen:?}, promised: {promised}"
        );
        if promised {
            assert_eq!(
                seen, PROBE_EXPECTED,
                "multi_draw_indirect_count on {backend}"
            );
        }

        // Draw path A: the same with the count draw.
        let seen = probe_draw(gpu, ProbeDraw::BucketIndirectCount);
        let promised = derived.draw_path == DrawPath::A;
        log::info!(
            "{backend}: multi_draw_indexed_indirect_count read bucket {seen:?}, \
             promised (path A): {promised}"
        );
        if promised {
            assert_eq!(
                seen, BUCKET_EXPECTED,
                "multi_draw_indexed_indirect_count (path A) on {backend}"
            );
        }
    } else {
        assert!(!derived.indirect_count_draws_respect_first_vertex_and_instance);
        assert_eq!(derived.draw_path, DrawPath::B, "{backend}");
    }

    // The path follows the rule of contract §4.5 on a real device.
    let path_a = backend == wgpu::Backend::Vulkan
        && derived.can_multi_draw_indirect_count
        && derived.can_indirect_first_instance;
    let expected = if path_a { DrawPath::A } else { DrawPath::B };
    assert_eq!(derived.draw_path, expected, "draw path on {backend}");
}
