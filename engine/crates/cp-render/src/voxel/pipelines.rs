//! What the voxel renderer creates once at start-up: the bind group layouts and every pipeline,
//! the per-frame buffers, the static index buffer and the frame graph declaration (contract
//! §5.3 "No shader stutter").

use cp_core::quad::{QUAD_INDEX_PATTERN, QUAD_VERTICES};

use super::layout::{
    COUNTER_WORDS, CULL_WORKGROUP, DRAW_CLASSES, GpuFrame, MAX_ITEM_QUADS, SEGMENTS,
};
use super::taa::{GpuTaa, HISTORY_FORMAT};
use crate::error::RenderError;
use crate::frame_graph::{FrameGraphDesc, PassId, TransientDesc, TransientId, TransientSize};
use crate::gpu::GpuContext;
use crate::scope::capture;

/// G-buffer albedo: sRGB colour, alpha unused (contract §5.3).
pub const ALBEDO_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8UnormSrgb;
/// G-buffer normal (octahedral, rg) and roughness (b).
pub const NORMAL_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgb10a2Unorm;
/// G-buffer material: metallic (r), emissive strength `e / (1 + e)` (g).
pub const MATERIAL_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;
/// Reversed-Z depth, cleared to 0, compare `Greater`.
pub const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;
/// Scene-linear colour of the lighting pass (contract §7).
pub const HDR_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
/// Fragment counts of the overdraw view.
pub const OVERDRAW_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R16Float;

/// Name of the cull pass (draw path A) in the frame graph and its timings.
pub const PASS_CULL: &str = "cull";
/// Name of the G-buffer pass.
pub const PASS_GBUFFER: &str = "gbuffer";
/// Name of the overdraw pass (overdraw view only).
pub const PASS_OVERDRAW: &str = "overdraw";
/// Name of the lighting pass.
pub const PASS_LIGHTING: &str = "lighting";
/// Name of the temporal anti-aliasing pass (renderers with temporal anti-aliasing only).
pub const PASS_TAA: &str = "taa";
/// Name of the tone-mapping and debug-view pass into the output.
pub const PASS_COMPOSITE: &str = "composite";
/// Name of the chunk-bounds pass (chunk-bounds view only).
pub const PASS_BOUNDS: &str = "bounds";
/// Name of the overlay pass (frames with an overlay only).
pub const PASS_OVERLAY: &str = "overlay";

/// Render and compute pipelines of the renderer, the overlay's included; all are created by
/// [`VoxelRenderer::new`](super::VoxelRenderer::new) and none later (contract §5.3 "No shader
/// stutter").
pub const PIPELINE_COUNT: usize = 11;

/// The transients and passes of the frame graph.
#[derive(Clone, Copy, Debug)]
pub(crate) struct GraphIds {
    pub(crate) albedo: TransientId,
    pub(crate) normal: TransientId,
    pub(crate) material: TransientId,
    pub(crate) depth: TransientId,
    pub(crate) overdraw: TransientId,
    pub(crate) hdr: TransientId,
    pub(crate) cull: PassId,
    pub(crate) gbuffer: PassId,
    pub(crate) overdraw_pass: PassId,
    pub(crate) lighting: PassId,
    pub(crate) taa: PassId,
    pub(crate) composite: PassId,
    pub(crate) bounds: PassId,
    pub(crate) overlay: PassId,
}

/// Every pipeline of the voxel renderer except the overlay's.
#[derive(Debug)]
pub(crate) struct Pipelines {
    pub(crate) cull_count: wgpu::ComputePipeline,
    pub(crate) cull_scan: wgpu::ComputePipeline,
    pub(crate) cull_scatter: wgpu::ComputePipeline,
    pub(crate) gbuffer_opaque: wgpu::RenderPipeline,
    pub(crate) gbuffer_cutout: wgpu::RenderPipeline,
    pub(crate) overdraw: wgpu::RenderPipeline,
    pub(crate) lighting: wgpu::RenderPipeline,
    pub(crate) taa: wgpu::RenderPipeline,
    pub(crate) composite: wgpu::RenderPipeline,
    pub(crate) bounds: wgpu::RenderPipeline,
}

/// The layouts of the bind groups that are made per frame (over transient textures).
#[derive(Debug)]
pub(crate) struct Layouts {
    pub(crate) lighting: wgpu::BindGroupLayout,
    pub(crate) taa: wgpu::BindGroupLayout,
    pub(crate) composite: wgpu::BindGroupLayout,
    /// Linear filtering, clamped to the edge: the history of temporal anti-aliasing.
    pub(crate) history_sampler: wgpu::Sampler,
}

/// The buffers of the frame: uniform, draw lists, counters, cull scratch, static indices.
#[derive(Debug)]
pub(crate) struct FrameBuffers {
    pub(crate) frame: wgpu::Buffer,
    pub(crate) taa: wgpu::Buffer,
    pub(crate) args: wgpu::Buffer,
    pub(crate) buckets: wgpu::Buffer,
    pub(crate) counters: wgpu::Buffer,
    pub(crate) scratch: wgpu::Buffer,
    pub(crate) indices: wgpu::Buffer,
}

/// Offsets of the cull scratch buffer, in words: item states `[0, max items)`, then one row of
/// `max_blocks` block sums per segment from `block_base`, then one total per segment from
/// `total_base`.
#[derive(Clone, Copy, Debug)]
pub(crate) struct ScratchLayout {
    pub(crate) max_blocks: u32,
    pub(crate) block_base: u32,
    pub(crate) total_base: u32,
    pub(crate) words: u64,
}

impl ScratchLayout {
    pub(crate) fn new(max_items: u32) -> Self {
        let max_blocks = max_items.div_ceil(CULL_WORKGROUP);
        let block_base = max_items;
        let total_base = block_base + SEGMENTS * max_blocks;
        Self {
            max_blocks,
            block_base,
            total_base,
            words: u64::from(total_base + SEGMENTS),
        }
    }
}

/// The bind groups over this frame's transient textures (and the history textures of temporal
/// anti-aliasing), kept while the textures stay.
#[derive(Debug)]
pub(crate) struct TransientBindings {
    /// The transient views, then the two history views if there are any.
    pub(crate) views: Vec<wgpu::TextureView>,
    pub(crate) lighting: wgpu::BindGroup,
    /// The composite pass over the lit image (no temporal anti-aliasing, or no history yet).
    pub(crate) composite: wgpu::BindGroup,
    /// Per history texture written: the resolve pass that reads the other one, and the
    /// composite pass over the one written.
    pub(crate) temporal: Option<[(wgpu::BindGroup, wgpu::BindGroup); 2]>,
}

/// Creates the frame buffers and fills the static index buffer.
pub(crate) fn create_buffers(
    gpu: &GpuContext,
    class_capacity: u32,
    scratch: &ScratchLayout,
) -> Result<FrameBuffers, RenderError> {
    use wgpu::BufferUsages as U;
    let slots = u64::from(DRAW_CLASSES) * u64::from(class_capacity);
    let buffers = capture(gpu, "voxel frame buffers", |device| {
        let buffer = |label: &str, size: u64, usage: U| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size,
                usage,
                mapped_at_creation: false,
            })
        };
        FrameBuffers {
            frame: buffer(
                "voxel frame",
                size_of::<GpuFrame>() as u64,
                U::UNIFORM | U::COPY_DST,
            ),
            taa: buffer(
                "voxel taa",
                size_of::<GpuTaa>() as u64,
                U::UNIFORM | U::COPY_DST,
            ),
            args: buffer(
                "voxel indirect args",
                slots * 20,
                U::STORAGE | U::INDIRECT | U::COPY_DST | U::COPY_SRC,
            ),
            buckets: buffer(
                "voxel buckets",
                slots * 8,
                U::STORAGE | U::VERTEX | U::COPY_DST | U::COPY_SRC,
            ),
            counters: buffer(
                "voxel counters",
                COUNTER_WORDS as u64 * 4,
                U::STORAGE | U::INDIRECT | U::COPY_DST | U::COPY_SRC,
            ),
            scratch: buffer(
                "voxel cull scratch",
                scratch.words * 4,
                U::STORAGE | U::COPY_DST | U::COPY_SRC,
            ),
            indices: buffer(
                "voxel quad indices",
                u64::from(MAX_ITEM_QUADS) * QUAD_INDEX_PATTERN.len() as u64 * 4,
                U::INDEX | U::COPY_DST,
            ),
        }
    })?;
    let indices: Vec<u32> = (0..MAX_ITEM_QUADS)
        .flat_map(|quad| QUAD_INDEX_PATTERN.map(|index| quad * QUAD_VERTICES + index))
        .collect();
    gpu.queue
        .write_buffer(&buffers.indices, 0, bytemuck::cast_slice(&indices));
    Ok(buffers)
}

/// The shader modules the pipelines are made of.
pub(crate) struct Modules<'a> {
    pub(crate) cull: &'a wgpu::ShaderModule,
    pub(crate) gbuffer: &'a wgpu::ShaderModule,
    pub(crate) lighting: &'a wgpu::ShaderModule,
    pub(crate) taa: &'a wgpu::ShaderModule,
    pub(crate) composite: &'a wgpu::ShaderModule,
    pub(crate) bounds: &'a wgpu::ShaderModule,
}

/// Everything [`create_pipelines`] makes.
pub(crate) struct Created {
    pub(crate) pipelines: Pipelines,
    pub(crate) layouts: Layouts,
    pub(crate) cull_layout: wgpu::BindGroupLayout,
    pub(crate) geometry_layout: wgpu::BindGroupLayout,
    pub(crate) texture_layout: wgpu::BindGroupLayout,
    /// Block textures: linear, repeating, anisotropic.
    pub(crate) sampler: wgpu::Sampler,
    /// The ground map: linear, clamped to the edge.
    pub(crate) ground_sampler: wgpu::Sampler,
}

fn uniform_entry(binding: u32, visibility: wgpu::ShaderStages) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: wgpu::BufferSize::new(size_of::<GpuFrame>() as u64),
        },
        count: None,
    }
}

fn storage_entry(
    binding: u32,
    visibility: wgpu::ShaderStages,
    read_only: bool,
) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn texture_entry(
    binding: u32,
    sample_type: wgpu::TextureSampleType,
    view_dimension: wgpu::TextureViewDimension,
) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Texture {
            sample_type,
            view_dimension,
            multisampled: false,
        },
        count: None,
    }
}

pub(crate) fn bind_group(
    device: &wgpu::Device,
    label: &str,
    layout: &wgpu::BindGroupLayout,
    resources: &[wgpu::BindingResource<'_>],
) -> wgpu::BindGroup {
    let entries: Vec<wgpu::BindGroupEntry<'_>> = (0..)
        .zip(resources)
        .map(|(binding, resource)| wgpu::BindGroupEntry {
            binding,
            resource: resource.clone(),
        })
        .collect();
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some(label),
        layout,
        entries: &entries,
    })
}

pub(crate) fn create_pipelines(
    device: &wgpu::Device,
    modules: &Modules<'_>,
    target_format: wgpu::TextureFormat,
    anisotropy: u16,
) -> Created {
    use wgpu::ShaderStages as S;
    let float = wgpu::TextureSampleType::Float { filterable: false };
    let d2 = wgpu::TextureViewDimension::D2;
    let layout = |label: &str, entries: &[wgpu::BindGroupLayoutEntry]| {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some(label),
            entries,
        })
    };
    let cull_layout = layout(
        "voxel cull",
        &[
            uniform_entry(0, S::COMPUTE),
            storage_entry(1, S::COMPUTE, true),
            storage_entry(2, S::COMPUTE, true),
            storage_entry(3, S::COMPUTE, false),
            storage_entry(4, S::COMPUTE, false),
            storage_entry(5, S::COMPUTE, false),
            storage_entry(6, S::COMPUTE, false),
        ],
    );
    let geometry_layout = layout(
        "voxel geometry",
        &[
            uniform_entry(0, S::VERTEX_FRAGMENT),
            storage_entry(1, S::VERTEX, true),
            storage_entry(2, S::VERTEX, true),
            storage_entry(3, S::VERTEX, true),
            storage_entry(4, S::VERTEX, true),
        ],
    );
    let filtering_sampler = |binding: u32| wgpu::BindGroupLayoutEntry {
        binding,
        visibility: S::FRAGMENT,
        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
        count: None,
    };
    // Block textures and their sampler; the ground map and its sampler.
    let texture_layout = layout(
        "voxel textures",
        &[
            texture_entry(
                0,
                wgpu::TextureSampleType::Float { filterable: true },
                wgpu::TextureViewDimension::D2Array,
            ),
            filtering_sampler(1),
            texture_entry(2, wgpu::TextureSampleType::Float { filterable: true }, d2),
            filtering_sampler(3),
        ],
    );
    let lighting_layout = layout(
        "voxel lighting",
        &[
            uniform_entry(0, S::FRAGMENT),
            texture_entry(1, float, d2),
            texture_entry(2, float, d2),
            texture_entry(3, float, d2),
            texture_entry(4, wgpu::TextureSampleType::Depth, d2),
        ],
    );
    let composite_layout = layout(
        "voxel composite",
        &[
            uniform_entry(0, S::FRAGMENT),
            texture_entry(1, float, d2),
            texture_entry(2, float, d2),
            texture_entry(3, float, d2),
            texture_entry(4, wgpu::TextureSampleType::Depth, d2),
            texture_entry(5, float, d2),
        ],
    );
    let taa_layout = layout(
        "voxel taa",
        &[
            uniform_entry(0, S::FRAGMENT),
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: S::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: wgpu::BufferSize::new(size_of::<GpuTaa>() as u64),
                },
                count: None,
            },
            texture_entry(2, float, d2),
            texture_entry(3, wgpu::TextureSampleType::Depth, d2),
            texture_entry(4, wgpu::TextureSampleType::Float { filterable: true }, d2),
            wgpu::BindGroupLayoutEntry {
                binding: 5,
                visibility: S::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
        ],
    );
    let pipeline_layout = |label: &str, groups: &[Option<&wgpu::BindGroupLayout>]| {
        device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some(label),
            bind_group_layouts: groups,
            immediate_size: 0,
        })
    };
    let cull_pipeline_layout = pipeline_layout("voxel cull", &[Some(&cull_layout)]);
    let geometry_pipeline_layout = pipeline_layout(
        "voxel geometry",
        &[Some(&geometry_layout), Some(&texture_layout)],
    );
    let bounds_pipeline_layout = pipeline_layout("voxel bounds", &[Some(&geometry_layout)]);
    let lighting_pipeline_layout = pipeline_layout("voxel lighting", &[Some(&lighting_layout)]);
    let taa_pipeline_layout = pipeline_layout("voxel taa", &[Some(&taa_layout)]);
    let composite_pipeline_layout = pipeline_layout("voxel composite", &[Some(&composite_layout)]);

    let compute = |entry: &str| {
        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(entry),
            layout: Some(&cull_pipeline_layout),
            module: modules.cull,
            entry_point: Some(entry),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        })
    };

    let bucket_attributes = [wgpu::VertexAttribute {
        format: wgpu::VertexFormat::Uint32x2,
        offset: 0,
        shader_location: 0,
    }];
    let bucket_buffers = [Some(wgpu::VertexBufferLayout {
        array_stride: 8,
        step_mode: wgpu::VertexStepMode::Instance,
        attributes: &bucket_attributes,
    })];
    let quad_primitive = wgpu::PrimitiveState {
        topology: wgpu::PrimitiveTopology::TriangleList,
        strip_index_format: None,
        front_face: wgpu::FrontFace::Ccw,
        cull_mode: Some(wgpu::Face::Back),
        unclipped_depth: false,
        polygon_mode: wgpu::PolygonMode::Fill,
        conservative: false,
    };
    // The cutout pipeline draws both sides: foliage shows the inside of its crown through the
    // gaps of its leaves (`fs_cutout` discards the back faces of everything else).
    let two_sided_primitive = wgpu::PrimitiveState {
        cull_mode: None,
        ..quad_primitive
    };
    let depth = wgpu::DepthStencilState {
        format: DEPTH_FORMAT,
        depth_write_enabled: Some(true),
        depth_compare: Some(wgpu::CompareFunction::Greater),
        stencil: wgpu::StencilState::default(),
        bias: wgpu::DepthBiasState::default(),
    };
    let gbuffer_targets = [
        Some(ALBEDO_FORMAT.into()),
        Some(NORMAL_FORMAT.into()),
        Some(MATERIAL_FORMAT.into()),
    ];
    let additive = wgpu::BlendComponent {
        src_factor: wgpu::BlendFactor::One,
        dst_factor: wgpu::BlendFactor::One,
        operation: wgpu::BlendOperation::Add,
    };
    let overdraw_targets = [Some(wgpu::ColorTargetState {
        format: OVERDRAW_FORMAT,
        blend: Some(wgpu::BlendState {
            color: additive,
            alpha: additive,
        }),
        write_mask: wgpu::ColorWrites::ALL,
    })];
    let voxel = |label: &str,
                 fragment: &str,
                 primitive: wgpu::PrimitiveState,
                 targets: &[Option<wgpu::ColorTargetState>],
                 depth_stencil: Option<wgpu::DepthStencilState>| {
        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some(label),
            layout: Some(&geometry_pipeline_layout),
            vertex: wgpu::VertexState {
                module: modules.gbuffer,
                entry_point: Some("vs_voxel"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                buffers: &bucket_buffers,
            },
            primitive,
            depth_stencil,
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: modules.gbuffer,
                entry_point: Some(fragment),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                targets,
            }),
            multiview_mask: None,
            cache: None,
        })
    };
    let target = [Some(wgpu::ColorTargetState::from(target_format))];
    let screen = |label: &str,
                  module: &wgpu::ShaderModule,
                  layout: &wgpu::PipelineLayout,
                  vertex: &str,
                  fragment: &str,
                  topology: wgpu::PrimitiveTopology,
                  targets: &[Option<wgpu::ColorTargetState>]| {
        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some(label),
            layout: Some(layout),
            vertex: wgpu::VertexState {
                module,
                entry_point: Some(vertex),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                buffers: &[],
            },
            primitive: wgpu::PrimitiveState {
                topology,
                ..wgpu::PrimitiveState::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module,
                entry_point: Some(fragment),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                targets,
            }),
            multiview_mask: None,
            cache: None,
        })
    };
    let pipelines = Pipelines {
        cull_count: compute("cull_count"),
        cull_scan: compute("cull_scan"),
        cull_scatter: compute("cull_scatter"),
        gbuffer_opaque: voxel(
            "voxel g-buffer opaque",
            "fs_opaque",
            quad_primitive,
            &gbuffer_targets,
            Some(depth.clone()),
        ),
        gbuffer_cutout: voxel(
            "voxel g-buffer cutout",
            "fs_cutout",
            two_sided_primitive,
            &gbuffer_targets,
            Some(depth),
        ),
        overdraw: voxel(
            "voxel overdraw",
            "fs_overdraw",
            quad_primitive,
            &overdraw_targets,
            None,
        ),
        lighting: screen(
            "voxel lighting",
            modules.lighting,
            &lighting_pipeline_layout,
            "vs_fullscreen",
            "fs_lighting",
            wgpu::PrimitiveTopology::TriangleList,
            &[Some(HDR_FORMAT.into())],
        ),
        taa: screen(
            "voxel taa",
            modules.taa,
            &taa_pipeline_layout,
            "vs_fullscreen",
            "fs_taa",
            wgpu::PrimitiveTopology::TriangleList,
            &[Some(HISTORY_FORMAT.into())],
        ),
        composite: screen(
            "voxel composite",
            modules.composite,
            &composite_pipeline_layout,
            "vs_fullscreen",
            "fs_composite",
            wgpu::PrimitiveTopology::TriangleList,
            &target,
        ),
        bounds: screen(
            "voxel bounds",
            modules.bounds,
            &bounds_pipeline_layout,
            "vs_bounds",
            "fs_bounds",
            wgpu::PrimitiveTopology::LineList,
            &target,
        ),
    };
    let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("voxel block textures"),
        address_mode_u: wgpu::AddressMode::Repeat,
        address_mode_v: wgpu::AddressMode::Repeat,
        address_mode_w: wgpu::AddressMode::ClampToEdge,
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        mipmap_filter: wgpu::MipmapFilterMode::Linear,
        lod_min_clamp: 0.0,
        lod_max_clamp: 32.0,
        compare: None,
        anisotropy_clamp: anisotropy,
        border_color: None,
    });
    let ground_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("voxel ground map"),
        address_mode_u: wgpu::AddressMode::ClampToEdge,
        address_mode_v: wgpu::AddressMode::ClampToEdge,
        address_mode_w: wgpu::AddressMode::ClampToEdge,
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        mipmap_filter: wgpu::MipmapFilterMode::Nearest,
        ..wgpu::SamplerDescriptor::default()
    });
    let history_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("voxel taa history"),
        address_mode_u: wgpu::AddressMode::ClampToEdge,
        address_mode_v: wgpu::AddressMode::ClampToEdge,
        address_mode_w: wgpu::AddressMode::ClampToEdge,
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        mipmap_filter: wgpu::MipmapFilterMode::Nearest,
        ..wgpu::SamplerDescriptor::default()
    });
    Created {
        pipelines,
        layouts: Layouts {
            lighting: lighting_layout,
            taa: taa_layout,
            composite: composite_layout,
            history_sampler,
        },
        cull_layout,
        geometry_layout,
        texture_layout,
        sampler,
        ground_sampler,
    }
}

pub(crate) fn graph_desc() -> (FrameGraphDesc, GraphIds) {
    let mut desc = FrameGraphDesc::new();
    let mut transient =
        |name: &str, format| desc.transient(TransientDesc::new(name, format, TransientSize::FULL));
    let albedo = transient("voxel.albedo", ALBEDO_FORMAT);
    let normal = transient("voxel.normal", NORMAL_FORMAT);
    let material = transient("voxel.material", MATERIAL_FORMAT);
    let depth = transient("voxel.depth", DEPTH_FORMAT);
    let overdraw = transient("voxel.overdraw", OVERDRAW_FORMAT);
    let hdr = transient("voxel.hdr", HDR_FORMAT);
    let cull = desc.compute_pass(PASS_CULL, &[], &[]);
    let gbuffer = desc.render_pass(PASS_GBUFFER, &[], &[albedo, normal, material, depth]);
    let overdraw_pass = desc.render_pass(PASS_OVERDRAW, &[], &[overdraw]);
    let lighting = desc.render_pass(PASS_LIGHTING, &[albedo, normal, material, depth], &[hdr]);
    // The history textures of temporal anti-aliasing live outside the transient pool.
    let taa = desc.render_pass(PASS_TAA, &[hdr, depth], &[]);
    let composite = desc.render_pass(PASS_COMPOSITE, &[hdr, albedo, normal, depth, overdraw], &[]);
    let bounds = desc.render_pass(PASS_BOUNDS, &[], &[]);
    let overlay = desc.render_pass(PASS_OVERLAY, &[], &[]);
    let ids = GraphIds {
        albedo,
        normal,
        material,
        depth,
        overdraw,
        hdr,
        cull,
        gbuffer,
        overdraw_pass,
        lighting,
        taa,
        composite,
        bounds,
        overlay,
    };
    (desc, ids)
}

/// The uniform buffers the screen passes bind.
pub(crate) struct ScreenUniforms<'a> {
    pub(crate) frame: &'a wgpu::Buffer,
    pub(crate) taa: &'a wgpu::Buffer,
}

/// The bind groups over `views`: the six transients (albedo, normal, material, depth,
/// overdraw, HDR) and, with temporal anti-aliasing, the two history textures after them.
pub(crate) fn transient_bindings(
    gpu: &GpuContext,
    layouts: &Layouts,
    uniforms: &ScreenUniforms<'_>,
    views: Vec<wgpu::TextureView>,
) -> Result<TransientBindings, RenderError> {
    let [albedo, normal, material, depth, overdraw, hdr] = [0, 1, 2, 3, 4, 5].map(|k| &views[k]);
    let history = (views.len() >= 8).then(|| [&views[6], &views[7]]);
    let (lighting, composite, temporal) = capture(gpu, "voxel screen bind groups", |device| {
        use wgpu::BindingResource::TextureView as View;
        let frame = uniforms.frame;
        let lighting = bind_group(
            device,
            "voxel lighting",
            &layouts.lighting,
            &[
                frame.as_entire_binding(),
                View(albedo),
                View(normal),
                View(material),
                View(depth),
            ],
        );
        let composite_over = |label: &str, color: &wgpu::TextureView| {
            bind_group(
                device,
                label,
                &layouts.composite,
                &[
                    frame.as_entire_binding(),
                    View(color),
                    View(albedo),
                    View(normal),
                    View(depth),
                    View(overdraw),
                ],
            )
        };
        let composite = composite_over("voxel composite", hdr);
        let temporal = history.map(|history| {
            [0, 1].map(|written: usize| {
                let resolve = bind_group(
                    device,
                    "voxel taa",
                    &layouts.taa,
                    &[
                        frame.as_entire_binding(),
                        uniforms.taa.as_entire_binding(),
                        View(hdr),
                        View(depth),
                        View(history[1 - written]),
                        wgpu::BindingResource::Sampler(&layouts.history_sampler),
                    ],
                );
                (
                    resolve,
                    composite_over("voxel composite of the history", history[written]),
                )
            })
        });
        (lighting, composite, temporal)
    })?;
    Ok(TransientBindings {
        views,
        lighting,
        composite,
        temporal,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_frame_graph_declaration_is_valid() {
        let (desc, ids) = graph_desc();
        desc.validate().unwrap();
        assert_eq!(desc.passes().len(), 8);
        assert!(ids.cull < ids.gbuffer && ids.gbuffer < ids.lighting);
        assert!(ids.lighting < ids.taa && ids.taa < ids.composite && ids.composite < ids.bounds);
        assert!(
            desc.usage(ids.depth)
                .contains(wgpu::TextureUsages::TEXTURE_BINDING)
        );
        let names: Vec<&str> = desc
            .passes()
            .iter()
            .map(|pass| pass.name.as_str())
            .collect();
        assert_eq!(
            names,
            [
                PASS_CULL,
                PASS_GBUFFER,
                PASS_OVERDRAW,
                PASS_LIGHTING,
                PASS_TAA,
                PASS_COMPOSITE,
                PASS_BOUNDS,
                PASS_OVERLAY
            ]
        );
    }

    #[test]
    fn the_scratch_layout_has_room_for_every_block() {
        let layout = ScratchLayout::new(32_768);
        assert_eq!(layout.max_blocks, 128);
        assert_eq!(layout.block_base, 32_768);
        assert_eq!(layout.total_base, 32_768 + 42 * 128);
        assert_eq!(layout.words, u64::from(layout.total_base) + 42);
        let small = ScratchLayout::new(1);
        assert_eq!(small.max_blocks, 1);
    }
}
