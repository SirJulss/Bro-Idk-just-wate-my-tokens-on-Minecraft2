//! The voxel renderer: the frame from uploads and culling to the output (contract §5.3, §7),
//! over what [`pipelines`](super::pipelines) creates at start-up.

use std::time::Instant;

use cp_core::det;
use thiserror::Error;

use super::camera::{Camera, FrameCamera, Lighting, cloud_anchor};
use super::cull::DrawList;
use super::ground::{GroundMap, GroundTexture};
#[cfg(test)]
use super::layout::COUNTER_WORDS;
use super::layout::{
    CULL_WORKGROUP, DRAW_CLASSES, EDGE_OVERLAP_PX, GpuDrawItem, GpuFrame, GpuObject,
    MAX_ITEM_DRAWS, QUAD_BYTES, SEGMENTS,
};
use super::pipelines::{
    FrameBuffers, GraphIds, Layouts, Modules, PIPELINE_COUNT, Pipelines, ScratchLayout,
    ScreenUniforms, TransientBindings, bind_group, create_buffers, create_pipelines, graph_desc,
    transient_bindings,
};
use super::scene::{SceneLimits, VoxelError, VoxelScene, pool_quads_for_bytes};
use super::stats::{CountersReadback, FrameStats, VisibleCounts};
use super::taa::{GpuTaa, History, is_cut, jitter_ndc};
use crate::debug_view::{DebugView, DebugViewSet, DebugViewState};
use crate::error::RenderError;
use crate::frame_graph::{FrameGraph, FrameTimings};
use crate::gpu::{DrawPath, GpuCaps, GpuContext, QualityTier};
use crate::material::MaterialTable;
use crate::overlay::{Overlay, OverlayRenderer};
use crate::scope::capture;
use crate::shader::ShaderLibrary;
use crate::texture::{TextureArray, TextureArrayKind};

/// How the voxel quads are submitted. All three use the same shaders and produce the same
/// image (contract §9, E2).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DrawMode {
    /// GPU culling and compaction, one `multi_draw_indexed_indirect_count` per draw class.
    A,
    /// CPU culling into a compact indirect buffer, one `multi_draw_indexed_indirect` per draw
    /// class.
    B,
    /// CPU culling, one `draw_indexed` per visible bucket (debugging, and devices whose
    /// indirect draws cannot carry a bucket index).
    Direct,
}

impl DrawMode {
    /// Every mode.
    pub const ALL: [Self; 3] = [Self::A, Self::B, Self::Direct];

    /// The number the shaders see (`frame.mode.y`): 0, 1, 2.
    #[must_use]
    pub const fn shader_value(self) -> u32 {
        match self {
            Self::A => 0,
            Self::B => 1,
            Self::Direct => 2,
        }
    }

    /// Stable name: `a`, `b`, `direct`.
    #[must_use]
    pub const fn id(self) -> &'static str {
        match self {
            Self::A => "a",
            Self::B => "b",
            Self::Direct => "direct",
        }
    }

    /// The mode named `id`.
    #[must_use]
    pub fn from_id(id: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|mode| mode.id() == id)
    }

    /// The default mode of a device: its draw path, path B falling back to direct draws when
    /// indirect draws cannot carry a bucket index.
    #[must_use]
    pub fn for_device(caps: &GpuCaps) -> Self {
        match caps.derived.draw_path {
            DrawPath::A => Self::A,
            DrawPath::B if caps.derived.can_indirect_first_instance => Self::B,
            DrawPath::B => Self::Direct,
        }
    }

    /// Whether the device can draw with this mode.
    #[must_use]
    pub fn is_supported(self, caps: &GpuCaps) -> bool {
        match self {
            Self::A => caps.derived.draw_path == DrawPath::A,
            Self::B => caps.derived.can_indirect_first_instance,
            Self::Direct => true,
        }
    }
}

/// A draw mode was asked for that the device cannot use.
#[derive(Clone, Copy, Debug, Error, PartialEq, Eq)]
#[error("draw mode {} is not supported on this device", mode.id())]
pub struct UnsupportedDrawMode {
    /// The mode asked for.
    pub mode: DrawMode,
}

/// Sizes and formats of a renderer, fixed at creation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VoxelRendererConfig {
    /// Format of the output (window surface or headless target).
    pub target_format: wgpu::TextureFormat,
    /// Bytes of the quad pool; rounded down to whole units of 256 quads (2 KiB). At most the
    /// device's storage binding limit.
    pub pool_bytes: u64,
    /// Draw items (chunk layers) at once.
    pub max_draw_items: u32,
    /// Objects (grids) at once.
    pub max_objects: u32,
    /// Bytes of quads copied into the pool per frame (at least one mesh per frame).
    pub upload_budget_bytes: u64,
    /// Anisotropic filtering of block textures, `1..=16`.
    pub anisotropy: u16,
    /// Temporal anti-aliasing (contract §7): every frame is jittered by a fraction of a pixel
    /// and accumulated with the reprojected history of the earlier ones. Off, every frame is
    /// rendered on its own, unjittered (exact images for tests).
    pub temporal_aa: bool,
}

impl VoxelRendererConfig {
    /// The pool of the high tier: 128 MiB, 16.7 million quads (contract §5.3, §7).
    pub const DEFAULT_POOL_BYTES: u64 = 128 << 20;
    /// Draw items of every tier.
    pub const DEFAULT_MAX_DRAW_ITEMS: u32 = 32_768;
    /// Objects of every tier.
    pub const DEFAULT_MAX_OBJECTS: u32 = 256;
    /// Upload budget of every tier: 4 MiB per frame.
    pub const DEFAULT_UPLOAD_BUDGET: u64 = 4 << 20;

    /// The configuration of a quality tier: pool 128 / 96 / 32 MiB and anisotropy 16 / 8 / 1
    /// for high / medium / low; temporal anti-aliasing on every tier.
    #[must_use]
    pub fn for_tier(tier: QualityTier, target_format: wgpu::TextureFormat) -> Self {
        let (pool_bytes, anisotropy) = match tier {
            QualityTier::High => (Self::DEFAULT_POOL_BYTES, 16),
            QualityTier::Medium => (96 << 20, 8),
            QualityTier::Low => (32 << 20, 1),
        };
        Self {
            target_format,
            pool_bytes,
            max_draw_items: Self::DEFAULT_MAX_DRAW_ITEMS,
            max_objects: Self::DEFAULT_MAX_OBJECTS,
            upload_budget_bytes: Self::DEFAULT_UPLOAD_BUDGET,
            anisotropy,
            temporal_aa: true,
        }
    }
}

/// What one frame shows and where it goes.
#[derive(Clone, Copy, Debug)]
pub struct FrameInput<'a> {
    /// The camera.
    pub camera: &'a Camera,
    /// Sun, sky and fog.
    pub lighting: &'a Lighting,
    /// The output view; its format must be the configured target format.
    pub target: &'a wgpu::TextureView,
    /// Size of the output in pixels.
    pub target_size: (u32, u32),
    /// Text and rectangles drawn last, if any.
    pub overlay: Option<&'a Overlay>,
}

/// The GPU-driven voxel renderer of contract §5.3.
#[derive(Debug)]
pub struct VoxelRenderer {
    config: VoxelRendererConfig,
    scene: VoxelScene,
    graph: FrameGraph,
    ids: GraphIds,
    pipelines: Pipelines,
    layouts: Layouts,
    buffers: FrameBuffers,
    scratch: ScratchLayout,
    class_capacity: u32,
    cull_bind_group: wgpu::BindGroup,
    geometry_bind_group: wgpu::BindGroup,
    texture_bind_group: wgpu::BindGroup,
    /// Layout and samplers of the texture bind group, which a new ground map rebuilds.
    texture_layout: wgpu::BindGroupLayout,
    block_sampler: wgpu::Sampler,
    ground_sampler: wgpu::Sampler,
    ground: GroundTexture,
    transient_bindings: Option<TransientBindings>,
    overlay: OverlayRenderer,
    materials: MaterialTable,
    albedo: TextureArray,
    draw_list: DrawList,
    mode: DrawMode,
    modes: Vec<DrawMode>,
    debug: DebugViewState,
    readback: CountersReadback,
    previous: Option<FrameCamera>,
    /// The history of temporal anti-aliasing (renderers with it only), at the output size.
    history: Option<History>,
    /// The debug view of the last frame: a new view starts a new history.
    last_view: Option<DebugView>,
    startup_ms: f64,
    /// Edge overlap of the quads in pixels: [`EDGE_OVERLAP_PX`], or what a test set.
    edge_overlap_px: f32,
}

impl VoxelRenderer {
    /// Creates the scene buffers, every pipeline and the frame graph.
    ///
    /// `materials` is indexed by the quads' material ids; `albedo` holds the block textures
    /// (an [`TextureArrayKind::Albedo`] array; all layers share one size).
    ///
    /// # Errors
    /// [`RenderError::Voxel`] with [`VoxelError::InvalidConfig`] for sizes the device cannot
    /// hold, [`RenderError::Shader`] or [`RenderError::Wgpu`] when a shader or pipeline is
    /// rejected (for example a target format that cannot be rendered to),
    /// [`RenderError::Device`] on a lost device.
    pub fn new(
        gpu: &GpuContext,
        config: &VoxelRendererConfig,
        materials: MaterialTable,
        albedo: TextureArray,
    ) -> Result<Self, RenderError> {
        let started = Instant::now();
        let limits = check_config(&gpu.caps, config, &albedo)?;
        let scene = VoxelScene::new(gpu, limits, config.upload_budget_bytes)?;
        let class_capacity = MAX_ITEM_DRAWS * config.max_draw_items;
        let scratch = ScratchLayout::new(config.max_draw_items);
        let buffers = create_buffers(gpu, class_capacity, &scratch)?;
        let library = ShaderLibrary::builtin();
        let cull_module = library.create_module(gpu, "voxel/cull.wgsl")?;
        let gbuffer_module = library.create_module(gpu, "voxel/gbuffer.wgsl")?;
        let lighting_module = library.create_module(gpu, "voxel/lighting.wgsl")?;
        let taa_module = library.create_module(gpu, "voxel/taa.wgsl")?;
        let composite_module = library.create_module(gpu, "voxel/composite.wgsl")?;
        let bounds_module = library.create_module(gpu, "voxel/bounds.wgsl")?;
        let modules = Modules {
            cull: &cull_module,
            gbuffer: &gbuffer_module,
            lighting: &lighting_module,
            taa: &taa_module,
            composite: &composite_module,
            bounds: &bounds_module,
        };
        let created = capture(gpu, "voxel pipelines", |device| {
            create_pipelines(device, &modules, config.target_format, config.anisotropy)
        })?;
        let overlay = OverlayRenderer::new(gpu, config.target_format)?;
        let (graph_desc, ids) = graph_desc();
        let graph = FrameGraph::new(gpu, graph_desc, (1, 1))?;
        let ground = capture(gpu, "voxel ground map", |device| {
            GroundTexture::new(device, &gpu.queue, &GroundMap::empty())
        })?;
        let (cull_bind_group, geometry_bind_group, texture_bind_group) =
            capture(gpu, "voxel bind groups", |device| {
                let cull = bind_group(
                    device,
                    "voxel cull",
                    &created.cull_layout,
                    &[
                        buffers.frame.as_entire_binding(),
                        scene.items_buffer().as_entire_binding(),
                        scene.objects_buffer().as_entire_binding(),
                        buffers.scratch.as_entire_binding(),
                        buffers.args.as_entire_binding(),
                        buffers.buckets.as_entire_binding(),
                        buffers.counters.as_entire_binding(),
                    ],
                );
                let geometry = bind_group(
                    device,
                    "voxel geometry",
                    &created.geometry_layout,
                    &[
                        buffers.frame.as_entire_binding(),
                        scene.items_buffer().as_entire_binding(),
                        scene.objects_buffer().as_entire_binding(),
                        scene.pool_buffer().as_entire_binding(),
                        materials.binding(),
                    ],
                );
                let textures = texture_bind_group(
                    device,
                    &created.texture_layout,
                    &albedo,
                    [&created.sampler, &created.ground_sampler],
                    &ground,
                );
                (cull, geometry, textures)
            })?;
        let readback = CountersReadback::new(gpu)?;
        let modes: Vec<DrawMode> = DrawMode::ALL
            .into_iter()
            .filter(|mode| mode.is_supported(&gpu.caps))
            .collect();
        let renderer = Self {
            config: *config,
            scene,
            graph,
            ids,
            pipelines: created.pipelines,
            layouts: created.layouts,
            buffers,
            scratch,
            class_capacity,
            cull_bind_group,
            geometry_bind_group,
            texture_bind_group,
            texture_layout: created.texture_layout,
            block_sampler: created.sampler,
            ground_sampler: created.ground_sampler,
            ground,
            transient_bindings: None,
            overlay,
            materials,
            albedo,
            draw_list: DrawList::default(),
            mode: DrawMode::for_device(&gpu.caps),
            modes,
            debug: DebugViewState::new(DebugViewSet::ALL),
            readback,
            previous: None,
            history: None,
            last_view: None,
            startup_ms: started.elapsed().as_secs_f64() * 1.0e3,
            edge_overlap_px: EDGE_OVERLAP_PX,
        };
        log::info!(
            "voxel renderer: {PIPELINE_COUNT} pipelines in {:.1} ms, pool {} MiB ({} quads), \
             {} draw items, draw mode {} (available: {:?})",
            renderer.startup_ms,
            (renderer.scene.limits().pool_quads * QUAD_BYTES) >> 20,
            renderer.scene.limits().pool_quads,
            config.max_draw_items,
            renderer.mode.id(),
            renderer
                .modes
                .iter()
                .map(|mode| mode.id())
                .collect::<Vec<_>>(),
        );
        Ok(renderer)
    }

    /// The configuration.
    pub fn config(&self) -> &VoxelRendererConfig {
        &self.config
    }

    /// The scene (objects, chunks, uploads).
    pub fn scene(&self) -> &VoxelScene {
        &self.scene
    }

    /// The scene, to change it.
    pub fn scene_mut(&mut self) -> &mut VoxelScene {
        &mut self.scene
    }

    /// The material table, for updates of material parameters (hot reload).
    pub fn materials(&self) -> &MaterialTable {
        &self.materials
    }

    /// The block texture array.
    pub fn albedo(&self) -> &TextureArray {
        &self.albedo
    }

    /// Replaces the ground map that colours grass and leaves and lays ground cover (a new
    /// renderer has an empty one, under which the shaders keep their defaults).
    ///
    /// # Errors
    /// [`RenderError::Voxel`] with [`VoxelError::InvalidGroundMap`] for a map outside its
    /// ranges, [`RenderError::Wgpu`] when the GPU refuses the texture.
    pub fn set_ground_map(&mut self, gpu: &GpuContext, map: &GroundMap) -> Result<(), RenderError> {
        map.validate()?;
        let (ground, textures) = capture(gpu, "voxel ground map", |device| {
            let ground = GroundTexture::new(device, &gpu.queue, map);
            let textures = texture_bind_group(
                device,
                &self.texture_layout,
                &self.albedo,
                [&self.block_sampler, &self.ground_sampler],
                &ground,
            );
            (ground, textures)
        })?;
        self.ground = ground;
        self.texture_bind_group = textures;
        Ok(())
    }

    /// The draw mode of the next frames.
    pub fn draw_mode(&self) -> DrawMode {
        self.mode
    }

    /// The modes this device supports, in the order A, B, direct.
    pub fn available_draw_modes(&self) -> &[DrawMode] {
        &self.modes
    }

    /// Selects the draw mode of the next frames.
    ///
    /// # Errors
    /// [`UnsupportedDrawMode`] when the device cannot use it; the mode stays.
    pub fn set_draw_mode(&mut self, mode: DrawMode) -> Result<(), UnsupportedDrawMode> {
        if self.modes.contains(&mode) {
            self.mode = mode;
            Ok(())
        } else {
            Err(UnsupportedDrawMode { mode })
        }
    }

    /// The debug view selection (every view of [`DebugView`] is supported).
    pub fn debug_view(&self) -> &DebugViewState {
        &self.debug
    }

    /// The debug view selection, to change it.
    pub fn debug_view_mut(&mut self) -> &mut DebugViewState {
        &mut self.debug
    }

    /// Render and compute pipelines created at start-up: [`PIPELINE_COUNT`].
    pub fn pipeline_count(&self) -> usize {
        PIPELINE_COUNT
    }

    /// CPU time [`Self::new`] took, shader compilation and pipeline creation included, in
    /// milliseconds.
    pub fn startup_ms(&self) -> f64 {
        self.startup_ms
    }

    /// The frame graph (transient pool, timing state).
    pub fn frame_graph(&self) -> &FrameGraph {
        &self.graph
    }

    /// GPU times of the newest frame whose timestamps have arrived; passes are named by the
    /// `PASS_*` constants.
    pub fn latest_timings(&self) -> Option<&FrameTimings> {
        self.graph.latest_timings()
    }

    /// The visible counts of the newest path-A frame whose counters have arrived.
    pub fn latest_gpu_counts(&self) -> Option<VisibleCounts> {
        self.readback.latest()
    }

    /// Whether the renderer accumulates frames with temporal anti-aliasing.
    pub fn temporal_aa(&self) -> bool {
        self.config.temporal_aa
    }

    /// Forgets the history of temporal anti-aliasing: the next frame stands alone, as after a
    /// camera cut. The renderer does this itself when the output size or the debug view
    /// changes and when the camera jumps.
    pub fn reset_temporal_history(&mut self) {
        if let Some(history) = &mut self.history {
            history.reset();
        }
    }

    /// Waits until the GPU has finished everything submitted, then takes the timings and
    /// counters that arrived (benchmarks and tests; the frame loop never waits).
    ///
    /// # Errors
    /// [`RenderError::Device`] when the device is lost.
    pub fn wait_for_gpu(&mut self, gpu: &GpuContext) -> Result<(), RenderError> {
        gpu.poll(wgpu::PollType::wait_indefinitely())?;
        self.graph.collect_timings();
        self.readback.collect();
        Ok(())
    }

    /// Renders one frame into `input.target`: uploads within the budget, the object table, the
    /// cull of the draw mode, G-buffer, lighting, tone mapping or the selected debug view,
    /// chunk bounds and the overlay.
    ///
    /// # Errors
    /// [`RenderError::Device`] on a lost device, [`RenderError::Wgpu`] when wgpu rejects the
    /// frame (for example a target of the wrong format), [`RenderError::FrameGraph`] for an
    /// internal sequencing error.
    pub fn render(
        &mut self,
        gpu: &GpuContext,
        input: &FrameInput<'_>,
    ) -> Result<FrameStats, RenderError> {
        let started = Instant::now();
        let size = (input.target_size.0.max(1), input.target_size.1.max(1));
        if self.graph.output_size() != size {
            self.graph.resize(size.0, size.1);
        }
        self.readback.collect();
        let uploads = self.scene.flush_uploads(gpu);
        let aspect = size.0 as f32 / size.1 as f32;
        let debug = self.debug.current();
        let frame_index = self.graph.frames_begun();
        let temporal = self.config.temporal_aa;
        let now = FrameCamera {
            camera: *input.camera,
            aspect,
            jitter: if temporal {
                jitter_ndc(frame_index, size)
            } else {
                [0.0, 0.0]
            },
        };
        let prev = self.previous.unwrap_or(now);
        if temporal {
            let resized = self
                .history
                .as_ref()
                .is_none_or(|history| history.size != size);
            if resized {
                self.history = Some(History::new(gpu, size)?);
            }
            if self.last_view != Some(debug) || is_cut(&now, &prev) {
                self.reset_temporal_history();
            }
        }
        self.last_view = Some(debug);
        self.scene.update_objects(gpu, &now, &prev);
        let item_count = self.scene.items_cpu().len() as u32;
        let mode = self.mode;
        let mut cpu_cull_ms = None;
        if mode != DrawMode::A {
            let cull_started = Instant::now();
            self.draw_list.build(
                self.scene.items_cpu(),
                self.scene.objects_cpu(),
                self.class_capacity,
            );
            for class in 0..DRAW_CLASSES {
                let first = u64::from(class * self.class_capacity);
                let args = &self.draw_list.args[class as usize];
                if args.is_empty() {
                    continue;
                }
                if mode == DrawMode::B {
                    gpu.queue.write_buffer(
                        &self.buffers.args,
                        first * 20,
                        bytemuck::cast_slice(args),
                    );
                }
                gpu.queue.write_buffer(
                    &self.buffers.buckets,
                    first * 8,
                    bytemuck::cast_slice(&self.draw_list.buckets[class as usize]),
                );
            }
            cpu_cull_ms = Some(cull_started.elapsed().as_secs_f64() * 1.0e3);
        }
        let uniform = self.frame_uniform(input, &now, size, item_count, frame_index, debug);
        gpu.queue
            .write_buffer(&self.buffers.frame, 0, bytemuck::bytes_of(&uniform));
        if let Some(history) = &self.history
            && temporal
        {
            let taa = GpuTaa::new(&now, &prev, history.frames);
            gpu.queue
                .write_buffer(&self.buffers.taa, 0, bytemuck::bytes_of(&taa));
        }
        if let Some(overlay) = input.overlay {
            self.overlay.prepare(gpu, overlay, size)?;
        }

        let ids = self.ids;
        let mut frame = self.graph.begin_frame(gpu)?;
        let mut views: Vec<wgpu::TextureView> = [
            ids.albedo,
            ids.normal,
            ids.material,
            ids.depth,
            ids.overdraw,
            ids.hdr,
        ]
        .map(|id| frame.view(id))
        .to_vec();
        if let Some(history) = self.history.as_ref().filter(|_| temporal) {
            views.extend(history.views.iter().cloned());
        }
        if self
            .transient_bindings
            .as_ref()
            .is_none_or(|bindings| bindings.views != views)
        {
            self.transient_bindings = Some(transient_bindings(
                gpu,
                &self.layouts,
                &ScreenUniforms {
                    frame: &self.buffers.frame,
                    taa: &self.buffers.taa,
                },
                views.clone(),
            )?);
        }
        let bindings = self
            .transient_bindings
            .as_ref()
            .expect("the transient bind groups were created above");
        let [albedo, normal, material, depth, overdraw, hdr] =
            [0, 1, 2, 3, 4, 5].map(|k| &views[k]);

        let mut readback_slot = None;
        if mode == DrawMode::A {
            frame
                .encoder()
                .clear_buffer(&self.buffers.counters, 0, None);
            if item_count > 0 {
                let blocks = item_count.div_ceil(CULL_WORKGROUP);
                let mut pass = frame.begin_compute_pass(ids.cull)?;
                pass.set_bind_group(0, &self.cull_bind_group, &[]);
                pass.set_pipeline(&self.pipelines.cull_count);
                pass.dispatch_workgroups(blocks, 1, 1);
                pass.set_pipeline(&self.pipelines.cull_scan);
                pass.dispatch_workgroups(SEGMENTS, 1, 1);
                pass.set_pipeline(&self.pipelines.cull_scatter);
                pass.dispatch_workgroups(blocks, 1, 1);
            }
            readback_slot = self.readback.acquire();
        }
        // The count draws of path A cost GPU time per slot of `max_count` on some drivers
        // (about 7.5 ns on the GTX 1080), so they get the scene's bound, not the capacity.
        let bounds = self.scene.draw_bounds();
        let draws = Draws {
            mode,
            buffers: &self.buffers,
            list: &self.draw_list,
            capacity: self.class_capacity,
            max_counts: [0, 1].map(|class: usize| {
                // At most the capacity, which is a u32.
                bounds[class].min(u64::from(self.class_capacity)) as u32
            }),
        };
        {
            let mut pass = frame.begin_render_pass(
                ids.gbuffer,
                &wgpu::RenderPassDescriptor {
                    color_attachments: &[
                        Some(cleared(albedo)),
                        Some(cleared(normal)),
                        Some(cleared(material)),
                    ],
                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                        view: depth,
                        depth_ops: Some(wgpu::Operations {
                            load: wgpu::LoadOp::Clear(0.0),
                            store: wgpu::StoreOp::Store,
                        }),
                        stencil_ops: None,
                    }),
                    ..Default::default()
                },
            )?;
            draws.bind(
                &mut pass,
                &self.geometry_bind_group,
                &self.texture_bind_group,
            );
            for class in 0..DRAW_CLASSES {
                pass.set_pipeline(if class == 0 {
                    &self.pipelines.gbuffer_opaque
                } else {
                    &self.pipelines.gbuffer_cutout
                });
                draws.record(&mut pass, class);
            }
        }
        if debug == DebugView::Overdraw {
            let mut pass = frame.begin_render_pass(
                ids.overdraw_pass,
                &wgpu::RenderPassDescriptor {
                    color_attachments: &[Some(cleared(overdraw))],
                    ..Default::default()
                },
            )?;
            draws.bind(
                &mut pass,
                &self.geometry_bind_group,
                &self.texture_bind_group,
            );
            pass.set_pipeline(&self.pipelines.overdraw);
            for class in 0..DRAW_CLASSES {
                draws.record(&mut pass, class);
            }
        }
        {
            let mut pass = frame.begin_render_pass(
                ids.lighting,
                &wgpu::RenderPassDescriptor {
                    color_attachments: &[Some(cleared(hdr))],
                    ..Default::default()
                },
            )?;
            pass.set_pipeline(&self.pipelines.lighting);
            pass.set_bind_group(0, &bindings.lighting, &[]);
            pass.draw(0..3, 0..1);
        }
        // Temporal anti-aliasing: resolve into the history texture `next`, which the composite
        // pass then tone-maps.
        let mut composite_group = &bindings.composite;
        let resolved = match (&self.history, &bindings.temporal) {
            (Some(history), Some(groups)) if temporal => {
                let written = history.next;
                let (resolve, composite) = &groups[written];
                let mut pass = frame.begin_render_pass(
                    ids.taa,
                    &wgpu::RenderPassDescriptor {
                        color_attachments: &[Some(cleared(&history.views[written]))],
                        ..Default::default()
                    },
                )?;
                pass.set_pipeline(&self.pipelines.taa);
                pass.set_bind_group(0, resolve, &[]);
                pass.draw(0..3, 0..1);
                composite_group = composite;
                true
            }
            _ => false,
        };
        {
            let mut pass = frame.begin_render_pass(
                ids.composite,
                &wgpu::RenderPassDescriptor {
                    color_attachments: &[Some(cleared(input.target))],
                    ..Default::default()
                },
            )?;
            pass.set_pipeline(&self.pipelines.composite);
            pass.set_bind_group(0, composite_group, &[]);
            pass.draw(0..3, 0..1);
        }
        if debug == DebugView::ChunkBounds && item_count > 0 {
            let mut pass = frame.begin_render_pass(
                ids.bounds,
                &wgpu::RenderPassDescriptor {
                    color_attachments: &[Some(loaded(input.target))],
                    ..Default::default()
                },
            )?;
            pass.set_pipeline(&self.pipelines.bounds);
            pass.set_bind_group(0, &self.geometry_bind_group, &[]);
            pass.draw(0..24, 0..item_count);
        }
        if input.overlay.is_some() {
            let mut pass = frame.begin_render_pass(
                ids.overlay,
                &wgpu::RenderPassDescriptor {
                    color_attachments: &[Some(loaded(input.target))],
                    ..Default::default()
                },
            )?;
            self.overlay.draw(&mut pass);
        }
        if let Some(slot) = readback_slot {
            self.readback
                .encode(frame.encoder(), &self.buffers.counters, slot);
        }
        frame.submit()?;
        if let Some(slot) = readback_slot {
            self.readback.submitted(slot, frame_index);
        }
        if resolved && let Some(history) = &mut self.history {
            history.advance();
        }
        self.previous = Some(now);
        self.scene.commit_poses();

        let scene = self.scene.stats();
        let visible = match mode {
            DrawMode::A => self.readback.latest(),
            DrawMode::B | DrawMode::Direct => Some(VisibleCounts {
                frame: frame_index,
                draws: [self.draw_list.draws(0), self.draw_list.draws(1)],
                quads: self.draw_list.visible_quads,
            }),
        };
        Ok(FrameStats {
            frame: frame_index,
            mode,
            objects: scene.objects,
            chunks: scene.chunks,
            draw_items: item_count,
            visible,
            cpu_cull_ms,
            cpu_frame_ms: started.elapsed().as_secs_f64() * 1.0e3,
            uploads,
        })
    }

    fn frame_uniform(
        &self,
        input: &FrameInput<'_>,
        now: &FrameCamera,
        size: (u32, u32),
        item_count: u32,
        frame_index: u64,
        debug: DebugView,
    ) -> GpuFrame {
        let aspect = size.0 as f32 / size.1 as f32;
        // With this frame's jitter: the lighting pass reconstructs the points the G-buffer
        // pass rasterised.
        let render_to_clip = now.render_to_clip();
        let clip_to_render = render_to_clip.inverse();
        let light = input.lighting;
        let rgb = |v: cp_core::math::Vec3| [v.x, v.y, v.z, 0.0];
        let (width, height) = (size.0 as f32, size.1 as f32);
        // Pixels per metre across the view ray at 1 m depth; `y_axis.y` is 1 / tan(fov_y / 2).
        let focal_px = 0.5 * height * input.camera.projection(aspect).y_axis.y;
        let (texture_width, texture_height) = self.albedo.size();
        let sun = light.sun_direction.normalize_or(cp_core::math::Vec3::Y);
        let up = light.up.normalize_or(cp_core::math::Vec3::Y);
        let altitude = input.camera.position.dot(up.as_dvec3()) - f64::from(light.fog_base);
        let anchor = cloud_anchor(input.camera.position);
        let temporal = self.config.temporal_aa;
        GpuFrame {
            render_to_clip: render_to_clip.as_mat4().to_cols_array_2d(),
            clip_to_render: clip_to_render.as_mat4().to_cols_array_2d(),
            sun_direction: [sun.x, sun.y, sun.z, det::cos(light.sun_angular_radius)],
            sun_color: rgb(light.sun_color),
            sky_zenith: rgb(light.sky_zenith),
            sky_horizon: rgb(light.sky_horizon),
            ground: rgb(light.ground),
            up: rgb(up),
            tints: light.tints.map(rgb),
            viewport: [width, height, 1.0 / width, 1.0 / height],
            light: [
                light.exposure,
                light.ambient,
                light.fog_density,
                cp_core::math::Z_NEAR,
            ],
            cull: [
                item_count,
                item_count.div_ceil(CULL_WORKGROUP),
                self.scratch.max_blocks,
                self.class_capacity,
            ],
            scratch: [
                self.scratch.block_base,
                self.scratch.total_base,
                // The dither pattern only needs the low bits.
                frame_index as u32,
                0,
            ],
            mode: [
                debug.shader_value(),
                self.mode.shader_value(),
                u32::from(self.config.target_format.is_srgb()),
                0,
            ],
            raster: [
                // Texture sides are at most the device's 2D limit, exact in f32.
                texture_width as f32,
                texture_height as f32,
                focal_px,
                self.edge_overlap_px,
            ],
            sky_sunset: rgb(light.sky_sunset),
            atmosphere: [
                light.fog_falloff,
                altitude as f32,
                light.horizon_fade[0],
                light.horizon_fade[1],
            ],
            anchor: [
                anchor.x,
                anchor.y,
                anchor.z,
                if temporal { 1.0 } else { 0.0 },
            ],
            ground_map: self.ground.uniform,
            sky_light: [
                light.sky_light.x,
                light.sky_light.y,
                light.sky_light.z,
                light.sky_gain,
            ],
        }
    }

    /// Sets the edge overlap of the quads in pixels (0 shows the cracks the overlap closes).
    #[cfg(test)]
    pub(crate) fn set_edge_overlap_px(&mut self, pixels: f32) {
        self.edge_overlap_px = pixels;
    }

    /// The GPU's draw lists of the last path-A frame and the CPU's for the same items, read
    /// back for tests: `(gpu, cpu)` per class as `(args, buckets)`.
    #[cfg(test)]
    pub(crate) fn read_back_draw_lists(
        &mut self,
        gpu: &GpuContext,
    ) -> Result<DrawListPair, RenderError> {
        let counters: Vec<u32> = read_words(gpu, &self.buffers.counters, 0, COUNTER_WORDS as u64)?;
        let mut gpu_lists = Vec::new();
        for class in 0..DRAW_CLASSES {
            let draws = u64::from(counters[class as usize]);
            let first = u64::from(class * self.class_capacity);
            let args = read_words(gpu, &self.buffers.args, first * 5, draws * 5)?;
            let buckets = read_words(gpu, &self.buffers.buckets, first * 2, draws * 2)?;
            gpu_lists.push((args, buckets));
        }
        self.draw_list.build(
            self.scene.items_cpu(),
            self.scene.objects_cpu(),
            self.class_capacity,
        );
        let cpu_lists = (0..DRAW_CLASSES as usize)
            .map(|class| {
                let args: Vec<u32> = bytemuck::cast_slice(&self.draw_list.args[class]).to_vec();
                let buckets: Vec<u32> =
                    bytemuck::cast_slice(&self.draw_list.buckets[class]).to_vec();
                (args, buckets)
            })
            .collect();
        Ok(DrawListPair {
            gpu: gpu_lists,
            cpu: cpu_lists,
            gpu_quads: u64::from(counters[super::layout::COUNTER_QUADS]),
            cpu_quads: self.draw_list.visible_quads,
        })
    }
}

#[cfg(test)]
impl VoxelRenderer {
    /// Runs the scan dispatch alone over `block_sums[segment][block]` (every row of the same
    /// length, at most the scratch layout's blocks per row) and returns the exclusive prefix
    /// sums per segment and the segment totals, as the GPU computed them.
    pub(crate) fn run_scan(
        &mut self,
        gpu: &GpuContext,
        block_sums: &[Vec<u32>],
    ) -> Result<(Vec<Vec<u32>>, Vec<u32>), RenderError> {
        let blocks = block_sums[0].len() as u32;
        assert!(blocks <= self.scratch.max_blocks && block_sums.len() == SEGMENTS as usize);
        let row_offset =
            |segment: u32| u64::from(self.scratch.block_base + segment * self.scratch.max_blocks);
        for (segment, row) in (0..).zip(block_sums) {
            gpu.queue.write_buffer(
                &self.buffers.scratch,
                row_offset(segment) * 4,
                bytemuck::cast_slice(row),
            );
        }
        let uniform = GpuFrame {
            cull: [0, blocks, self.scratch.max_blocks, self.class_capacity],
            scratch: [self.scratch.block_base, self.scratch.total_base, 0, 0],
            ..GpuFrame::default()
        };
        gpu.queue
            .write_buffer(&self.buffers.frame, 0, bytemuck::bytes_of(&uniform));
        capture(gpu, "test scan", |device| {
            let mut encoder =
                device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
                pass.set_bind_group(0, &self.cull_bind_group, &[]);
                pass.set_pipeline(&self.pipelines.cull_scan);
                pass.dispatch_workgroups(SEGMENTS, 1, 1);
            }
            gpu.queue.submit([encoder.finish()]);
        })?;
        let mut rows = Vec::new();
        for segment in 0..SEGMENTS {
            rows.push(read_words(
                gpu,
                &self.buffers.scratch,
                row_offset(segment),
                u64::from(blocks),
            )?);
        }
        let totals = read_words(
            gpu,
            &self.buffers.scratch,
            u64::from(self.scratch.total_base),
            u64::from(SEGMENTS),
        )?;
        Ok((rows, totals))
    }
}

/// The draw lists of both sides, for tests.
#[cfg(test)]
#[derive(Debug)]
pub(crate) struct DrawListPair {
    pub(crate) gpu: Vec<(Vec<u32>, Vec<u32>)>,
    pub(crate) cpu: Vec<(Vec<u32>, Vec<u32>)>,
    pub(crate) gpu_quads: u64,
    pub(crate) cpu_quads: u64,
}

/// Reads `words` u32 from `buffer` at word `first` (blocking; tests only).
#[cfg(test)]
fn read_words(
    gpu: &GpuContext,
    buffer: &wgpu::Buffer,
    first: u64,
    words: u64,
) -> Result<Vec<u32>, RenderError> {
    if words == 0 {
        return Ok(Vec::new());
    }
    let staging = capture(gpu, "test read-back", |device| {
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("test read-back"),
            size: words * 4,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        encoder.copy_buffer_to_buffer(buffer, first * 4, &staging, 0, words * 4);
        gpu.queue.submit([encoder.finish()]);
        staging
    })?;
    let (sender, receiver) = std::sync::mpsc::channel();
    staging.map_async(wgpu::MapMode::Read, .., move |result| {
        let _ = sender.send(result);
    });
    gpu.poll(wgpu::PollType::wait_indefinitely())?;
    match receiver.try_recv() {
        Ok(Ok(())) => {}
        _ => {
            return Err(RenderError::Readback {
                what: "test buffer".to_owned(),
                reason: "mapping failed".to_owned(),
            });
        }
    }
    let words: Vec<u32> = {
        let bytes = staging
            .get_mapped_range(..)
            .map_err(|error| RenderError::Readback {
                what: "test buffer".to_owned(),
                reason: error.to_string(),
            })?;
        bytemuck::cast_slice(&bytes).to_vec()
    };
    staging.unmap();
    Ok(words)
}

/// What the voxel draws of a pass need.
struct Draws<'a> {
    mode: DrawMode,
    buffers: &'a FrameBuffers,
    list: &'a DrawList,
    /// Draw slots per class.
    capacity: u32,
    /// `max_count` of the count draws of path A, per class.
    max_counts: [u32; 2],
}

impl Draws<'_> {
    fn bind(
        &self,
        pass: &mut wgpu::RenderPass<'_>,
        geometry: &wgpu::BindGroup,
        textures: &wgpu::BindGroup,
    ) {
        pass.set_bind_group(0, geometry, &[]);
        pass.set_bind_group(1, textures, &[]);
        pass.set_vertex_buffer(0, self.buffers.buckets.slice(..));
        pass.set_index_buffer(self.buffers.indices.slice(..), wgpu::IndexFormat::Uint32);
    }

    /// The draws of `class` in the pass's current pipeline.
    fn record(&self, pass: &mut wgpu::RenderPass<'_>, class: u32) {
        let first_slot = class * self.capacity;
        let args_offset = u64::from(first_slot) * 20;
        match self.mode {
            DrawMode::A => {
                let max_count = self.max_counts[class as usize];
                if max_count > 0 {
                    pass.multi_draw_indexed_indirect_count(
                        &self.buffers.args,
                        args_offset,
                        &self.buffers.counters,
                        u64::from(class) * 4,
                        max_count,
                    );
                }
            }
            DrawMode::B => {
                // wgpu's indirect validation issues one draw even for a count of zero.
                let count = self.list.draws(class);
                if count > 0 {
                    pass.multi_draw_indexed_indirect(&self.buffers.args, args_offset, count);
                }
            }
            DrawMode::Direct => {
                for (position, args) in (0..).zip(&self.list.args[class as usize]) {
                    let slot = first_slot + position;
                    pass.draw_indexed(0..args.index_count, 0, slot..slot + 1);
                }
            }
        }
    }
}

/// The bind group of the G-buffer pass's textures: the block textures and the ground map,
/// each with its sampler.
fn texture_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    albedo: &TextureArray,
    [block_sampler, ground_sampler]: [&wgpu::Sampler; 2],
    ground: &GroundTexture,
) -> wgpu::BindGroup {
    bind_group(
        device,
        "voxel textures",
        layout,
        &[
            wgpu::BindingResource::TextureView(albedo.view()),
            wgpu::BindingResource::Sampler(block_sampler),
            wgpu::BindingResource::TextureView(&ground.view),
            wgpu::BindingResource::Sampler(ground_sampler),
        ],
    )
}

fn cleared(view: &wgpu::TextureView) -> wgpu::RenderPassColorAttachment<'_> {
    wgpu::RenderPassColorAttachment {
        view,
        depth_slice: None,
        resolve_target: None,
        ops: wgpu::Operations {
            load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
            store: wgpu::StoreOp::Store,
        },
    }
}

fn loaded(view: &wgpu::TextureView) -> wgpu::RenderPassColorAttachment<'_> {
    wgpu::RenderPassColorAttachment {
        view,
        depth_slice: None,
        resolve_target: None,
        ops: wgpu::Operations {
            load: wgpu::LoadOp::Load,
            store: wgpu::StoreOp::Store,
        },
    }
}

/// Checks a configuration against the device and returns the scene limits.
fn check_config(
    caps: &GpuCaps,
    config: &VoxelRendererConfig,
    albedo: &TextureArray,
) -> Result<SceneLimits, VoxelError> {
    let invalid = |reason| Err(VoxelError::InvalidConfig { reason });
    let limits = &caps.enabled_limits;
    let binding = limits
        .max_storage_buffer_binding_size
        .min(limits.max_buffer_size);
    let pool_quads = pool_quads_for_bytes(config.pool_bytes);
    if pool_quads == 0 {
        return invalid("the quad pool must hold at least one unit of 256 quads (2 KiB)");
    }
    if pool_quads * QUAD_BYTES > binding {
        return invalid("the quad pool exceeds the storage buffer binding limit of the device");
    }
    if pool_quads > u64::from(u32::MAX) {
        return invalid("the quad pool must hold fewer than 2^32 quads");
    }
    if config.max_draw_items == 0 || config.max_draw_items > 1 << 20 {
        return invalid("max_draw_items must be in 1..=2^20");
    }
    let class_capacity = u64::from(MAX_ITEM_DRAWS) * u64::from(config.max_draw_items);
    let args_bytes = u64::from(DRAW_CLASSES) * class_capacity * 20;
    let items_bytes = u64::from(config.max_draw_items) * size_of::<GpuDrawItem>() as u64;
    if args_bytes > binding || items_bytes > binding {
        return invalid("max_draw_items needs buffers beyond the storage binding limit");
    }
    if config.max_objects == 0 || config.max_objects > 1 << 16 {
        return invalid("max_objects must be in 1..=65536");
    }
    if u64::from(config.max_objects) * size_of::<GpuObject>() as u64 > binding {
        return invalid("max_objects needs a buffer beyond the storage binding limit");
    }
    if !(1..=16).contains(&config.anisotropy) {
        return invalid("anisotropy must be in 1..=16");
    }
    if albedo.kind() != TextureArrayKind::Albedo {
        return invalid("the block texture array must be an albedo (sRGB) array");
    }
    Ok(SceneLimits {
        pool_quads,
        max_draw_items: config.max_draw_items,
        max_objects: config.max_objects,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn draw_modes_have_stable_ids_and_shader_values() {
        for mode in DrawMode::ALL {
            assert_eq!(DrawMode::from_id(mode.id()), Some(mode));
        }
        assert_eq!(DrawMode::ALL.map(DrawMode::shader_value), [0, 1, 2]);
        assert_eq!(DrawMode::from_id("c"), None);
        assert_eq!(
            UnsupportedDrawMode { mode: DrawMode::A }.to_string(),
            "draw mode a is not supported on this device"
        );
    }

    #[test]
    fn tiers_scale_the_pool_and_the_anisotropy() {
        let format = wgpu::TextureFormat::Rgba8UnormSrgb;
        let high = VoxelRendererConfig::for_tier(QualityTier::High, format);
        assert_eq!(high.pool_bytes, 128 << 20);
        assert_eq!(high.anisotropy, 16);
        let low = VoxelRendererConfig::for_tier(QualityTier::Low, format);
        assert!(low.pool_bytes < high.pool_bytes);
        assert_eq!(low.anisotropy, 1);
    }
}
