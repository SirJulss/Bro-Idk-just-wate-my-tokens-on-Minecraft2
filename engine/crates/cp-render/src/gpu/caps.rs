//! The capability record of a GPU device and what the engine concludes from it.
//!
//! Everything in this module is plain data and pure functions: an [`AdapterProfile`] can be
//! fabricated in a test, so the negotiation and every fallback decision can be checked without
//! a GPU.

use super::features::{WANTED_FEATURES, select_features};
use super::limits::select_limits;
use super::options::Dx12ShaderCompiler;

/// What wgpu reports about an adapter, before a device exists.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AdapterProfile {
    /// Name, vendor and device id, driver, backend and device type.
    pub info: wgpu::AdapterInfo,
    /// Every feature the adapter offers.
    pub features: wgpu::Features,
    /// The best limits the adapter supports.
    pub limits: wgpu::Limits,
    /// Where the adapter falls short of baseline WebGPU.
    pub downlevel: wgpu::DownlevelCapabilities,
}

impl AdapterProfile {
    /// Reads the profile of a real adapter.
    pub fn from_adapter(adapter: &wgpu::Adapter) -> Self {
        Self {
            info: adapter.get_info(),
            features: adapter.features(),
            limits: adapter.limits(),
            downlevel: adapter.get_downlevel_capabilities(),
        }
    }

    /// The device request the engine makes on this adapter (contract §4.5).
    ///
    /// Features: every optional feature the pipeline can use and the adapter has, nothing
    /// else. Limits: the WebGPU defaults with the pipeline's larger wishes, each clamped to the
    /// adapter's value. A device request with this plan cannot fail on a feature or a limit.
    pub fn plan_device(&self) -> DevicePlan {
        let mut features = select_features(self.features);
        // Immediates without a single byte of immediate data would be enabled but unusable;
        // the report should not show such a feature as won.
        if self.limits.max_immediate_size == 0 {
            features.remove(wgpu::Features::IMMEDIATES);
        }
        let limits = select_limits(features, &self.limits);
        DevicePlan { features, limits }
    }
}

/// Features and limits to pass to `wgpu::Adapter::request_device`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DevicePlan {
    /// `DeviceDescriptor::required_features`; a subset of the adapter's features.
    pub features: wgpu::Features,
    /// `DeviceDescriptor::required_limits`; never better than the adapter's limits.
    pub limits: wgpu::Limits,
}

/// One optional feature of the pipeline and whether this device has it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FeatureStatus {
    /// The feature flag (a single flag).
    pub feature: wgpu::Features,
    /// Name of the flag constant in `wgpu::Features`.
    pub name: &'static str,
    /// The adapter offers the feature.
    pub available: bool,
    /// The feature is enabled on the device. Implies `available`.
    pub enabled: bool,
    /// What the pipeline uses the feature for.
    pub used_for: &'static str,
    /// What the pipeline does without it.
    pub fallback: &'static str,
}

/// How the voxel renderer submits its draws (contract §4.5, §5.3 "Drawing").
///
/// Both paths draw indexed four-vertex quads with the same shaders. The vertex shader finds
/// its bucket `(draw item, first quad)` in an instance-step vertex attribute addressed by the
/// `first_instance` of the draw; it never reads `@builtin(instance_index)` and never depends
/// on a non-zero `base_vertex`, because DX12 count draws add neither to the builtins.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DrawPath {
    /// Path A: a compute pass culls the draw items and writes the indirect arguments, the
    /// buckets and their count; one `multi_draw_indexed_indirect_count` per view. The CPU
    /// never learns what is visible.
    ///
    /// Exactly the devices on the Vulkan backend with `MULTI_DRAW_INDIRECT_COUNT` and
    /// `INDIRECT_FIRST_INSTANCE` enabled.
    A,
    /// Path B: the same culling on the CPU into a compact indirect buffer with a known count,
    /// drawn with `multi_draw_indexed_indirect`. Every device that is not path A, DX12
    /// included.
    ///
    /// Without [`GpuDerived::can_indirect_first_instance`], indirect arguments cannot carry a
    /// bucket index, and the visible ranges are drawn one direct draw each.
    B,
}

impl DrawPath {
    /// Stable machine-readable name, used in the JSON report.
    pub fn id(self) -> &'static str {
        match self {
            Self::A => "a",
            Self::B => "b",
        }
    }

    /// Short description for people.
    pub fn description(self) -> &'static str {
        match self {
            Self::A => "A: GPU culling, multi_draw_indexed_indirect_count",
            Self::B => "B: CPU culling, multi_draw_indexed_indirect",
        }
    }
}

/// Starting point for the effect presets of a device (contract §7).
///
/// The tier says what class of hardware this is; it does not replace the individual
/// capabilities. An effect still has to check the [`GpuDerived`] booleans it depends on.
/// Tiers are ordered: `Low < Medium < High`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum QualityTier {
    /// A software rasteriser, or an adapter below baseline WebGPU (missing downlevel flags or
    /// limits under the WebGPU defaults). The designed pipeline cannot be expected to run.
    Low,
    /// A hardware GPU that runs the pipeline, but not of the baseline class: integrated,
    /// virtual or of unknown type, or a discrete GPU whose indirect draws cannot carry a
    /// bucket index (no [`GpuDerived::can_indirect_first_instance`]).
    Medium,
    /// A discrete GPU whose indirect draws carry the bucket index: the baseline class
    /// (GTX 1080), all effects. Draw path A or B; the path costs CPU time, not effects.
    High,
}

impl QualityTier {
    /// Stable machine-readable name, used in the JSON report.
    pub fn id(self) -> &'static str {
        match self {
            Self::Low => "low",
            Self::Medium => "medium",
            Self::High => "high",
        }
    }

    /// Short description for people.
    pub fn description(self) -> &'static str {
        match self {
            Self::Low => "software or sub-WebGPU adapter: minimal rendering only",
            Self::Medium => "hardware GPU below the baseline class: reduced effect presets",
            Self::High => "discrete GPU with indirect bucket draws: all effects",
        }
    }
}

/// Conclusions the renderer acts on, derived from the enabled features and limits, the
/// backend and the instance flags.
///
/// Each boolean describes the *device*: it is only `true` when the feature is enabled, not
/// merely available.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GpuDerived {
    /// `RenderPass::multi_draw_indirect_count` and `multi_draw_indexed_indirect_count` may be
    /// called.
    pub can_multi_draw_indirect_count: bool,
    /// Indirect draws may carry a non-zero `first_instance`, and it offsets the instance-step
    /// vertex attributes: the bucket index of both draw paths. `INDIRECT_FIRST_INSTANCE` is
    /// enabled on a device with indirect execution.
    pub can_indirect_first_instance: bool,
    /// In `draw_indirect` and `multi_draw_indirect`, `@builtin(vertex_index)` and
    /// `@builtin(instance_index)` include `first_vertex` and `first_instance` of the indirect
    /// arguments, and a non-zero `first_instance` is allowed.
    ///
    /// Diagnostic only: the renderer of contract §5.3 reads neither builtin offset. Vulkan
    /// adds both natively once `INDIRECT_FIRST_INSTANCE` is enabled. DX12 starts both builtins
    /// at zero; wgpu only adds the offsets while its indirect-call validation
    /// (`InstanceFlags::VALIDATION_INDIRECT_CALL`) is active. Other backends are not verified
    /// and reported as `false`.
    ///
    /// The rule is derived from the sources of wgpu 30 and has to be revisited with every wgpu
    /// update. The adapter tests `indirect_draws_deliver_first_vertex_and_instance_where_promised`
    /// (default instance flags) and `the_record_follows_the_flags_of_the_instance` (indirect-call
    /// validation switched the other way) hold every `true` against what the GPU really does.
    pub indirect_draws_respect_first_vertex_and_instance: bool,
    /// The same guarantee for `multi_draw_indirect_count`. Never `true` on DX12: wgpu 30 runs
    /// count draws without its validation pass, so the builtins start at zero there.
    pub indirect_count_draws_respect_first_vertex_and_instance: bool,
    /// The draw path the voxel renderer uses on this device.
    pub draw_path: DrawPath,
    /// Timestamps can be written at the beginning and end of render and compute passes.
    pub can_timestamp_passes: bool,
    /// `CommandEncoder::write_timestamp` may be called (timestamps between passes).
    pub can_timestamp_inside_encoders: bool,
    /// `write_timestamp` may be called inside render and compute passes.
    pub can_timestamp_inside_passes: bool,
    /// `Rg11b10Ufloat` can be a render target, so the HDR intermediates without alpha (bloom,
    /// half-resolution reflections) need 4 bytes per pixel instead of the 8 of `Rgba16Float`.
    pub hdr_rg11b10_renderable: bool,
    /// `R32Float`, `Rg32Float` and `Rgba32Float` textures can be sampled with linear filtering.
    pub float32_filterable: bool,
    /// The 16-bit normalised formats (`R16Unorm` and relatives) can be used; macro height maps
    /// need 2 bytes per texel instead of the 4 of `R32Float`.
    pub norm16_textures: bool,
    /// BC1 to BC7 compressed texture formats can be used.
    pub bc_compressed_textures: bool,
    /// `PrimitiveState::unclipped_depth` may be set (depth clamp instead of depth clip).
    pub can_unclip_depth: bool,
    /// Dual-source blend factors may be used.
    pub can_dual_source_blend: bool,
    /// Pipelines may declare immediate data; `enabled_limits.max_immediate_size` bytes exist.
    pub can_use_immediates: bool,
    /// Texture formats have the adapter's capabilities instead of only the WebGPU ones
    /// (`Adapter::get_texture_format_features`), read-write storage access included.
    pub adapter_format_features: bool,
}

impl GpuDerived {
    fn derive(
        profile: &AdapterProfile,
        instance_flags: wgpu::InstanceFlags,
        features: wgpu::Features,
        limits: &wgpu::Limits,
    ) -> Self {
        use wgpu::{DownlevelFlags, Features};

        let downlevel = profile.downlevel.flags;
        let indirect = downlevel.contains(DownlevelFlags::INDIRECT_EXECUTION);
        let first_instance = features.contains(Features::INDIRECT_FIRST_INSTANCE);
        // Mirrors the condition under which wgpu-core 30 creates its indirect validation,
        // which is also what feeds first_vertex/first_instance to DX12 shaders.
        let indirect_validation = instance_flags
            .contains(wgpu::InstanceFlags::VALIDATION_INDIRECT_CALL)
            && downlevel
                .contains(DownlevelFlags::INDIRECT_EXECUTION | DownlevelFlags::COMPUTE_SHADERS)
            && profile.limits.max_storage_buffers_per_shader_stage >= 2;

        let can_multi_draw_indirect_count =
            indirect && features.contains(Features::MULTI_DRAW_INDIRECT_COUNT);
        let indirect_respects = indirect
            && first_instance
            && match profile.info.backend {
                wgpu::Backend::Vulkan => true,
                wgpu::Backend::Dx12 => indirect_validation,
                _ => false,
            };
        let indirect_count_respects = can_multi_draw_indirect_count
            && first_instance
            && profile.info.backend == wgpu::Backend::Vulkan;

        // Contract §4.5, exactly: the backend and the two enabled features, nothing else. DX12
        // stays on path B because its count draws ignore first_instance in instance_index.
        let path_a = profile.info.backend == wgpu::Backend::Vulkan
            && features
                .contains(Features::MULTI_DRAW_INDIRECT_COUNT | Features::INDIRECT_FIRST_INSTANCE);
        let draw_path = if path_a { DrawPath::A } else { DrawPath::B };

        let timestamps = features.contains(Features::TIMESTAMP_QUERY);
        Self {
            can_multi_draw_indirect_count,
            can_indirect_first_instance: indirect && first_instance,
            indirect_draws_respect_first_vertex_and_instance: indirect_respects,
            indirect_count_draws_respect_first_vertex_and_instance: indirect_count_respects,
            draw_path,
            can_timestamp_passes: timestamps,
            can_timestamp_inside_encoders: timestamps
                && features.contains(Features::TIMESTAMP_QUERY_INSIDE_ENCODERS),
            can_timestamp_inside_passes: timestamps
                && features.contains(Features::TIMESTAMP_QUERY_INSIDE_PASSES),
            hdr_rg11b10_renderable: features.contains(Features::RG11B10UFLOAT_RENDERABLE),
            float32_filterable: features.contains(Features::FLOAT32_FILTERABLE),
            norm16_textures: features.contains(Features::TEXTURE_FORMAT_16BIT_NORM),
            bc_compressed_textures: features.contains(Features::TEXTURE_COMPRESSION_BC),
            can_unclip_depth: features.contains(Features::DEPTH_CLIP_CONTROL),
            can_dual_source_blend: features.contains(Features::DUAL_SOURCE_BLENDING),
            can_use_immediates: features.contains(Features::IMMEDIATES)
                && limits.max_immediate_size > 0,
            adapter_format_features: features
                .contains(Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES),
        }
    }

    /// The booleans with their field names and meanings, in declaration order (for reports).
    pub(crate) fn rows(&self) -> [(&'static str, bool, &'static str); 15] {
        [
            (
                "can_multi_draw_indirect_count",
                self.can_multi_draw_indirect_count,
                "multi_draw_(indexed_)indirect_count may be called",
            ),
            (
                "can_indirect_first_instance",
                self.can_indirect_first_instance,
                "first_instance of indirect draws offsets instance-step attributes (bucket)",
            ),
            (
                "indirect_draws_respect_first_vertex_and_instance",
                self.indirect_draws_respect_first_vertex_and_instance,
                "multi_draw_indirect: vertex_index and instance_index include first_vertex and \
                 first_instance",
            ),
            (
                "indirect_count_draws_respect_first_vertex_and_instance",
                self.indirect_count_draws_respect_first_vertex_and_instance,
                "the same for multi_draw_indirect_count",
            ),
            (
                "can_timestamp_passes",
                self.can_timestamp_passes,
                "timestamps at the begin and end of a pass",
            ),
            (
                "can_timestamp_inside_encoders",
                self.can_timestamp_inside_encoders,
                "CommandEncoder::write_timestamp between passes",
            ),
            (
                "can_timestamp_inside_passes",
                self.can_timestamp_inside_passes,
                "write_timestamp inside render and compute passes",
            ),
            (
                "hdr_rg11b10_renderable",
                self.hdr_rg11b10_renderable,
                "Rg11b10Ufloat as render target for HDR intermediates",
            ),
            (
                "float32_filterable",
                self.float32_filterable,
                "linear filtering of R32Float, Rg32Float and Rgba32Float",
            ),
            (
                "norm16_textures",
                self.norm16_textures,
                "R16Unorm and the other 16-bit normalised formats",
            ),
            (
                "bc_compressed_textures",
                self.bc_compressed_textures,
                "BC1 to BC7 texture formats",
            ),
            (
                "can_unclip_depth",
                self.can_unclip_depth,
                "PrimitiveState::unclipped_depth (depth clamp)",
            ),
            (
                "can_dual_source_blend",
                self.can_dual_source_blend,
                "dual-source blend factors",
            ),
            (
                "can_use_immediates",
                self.can_use_immediates,
                "immediate data in pipeline layouts (var<immediate>)",
            ),
            (
                "adapter_format_features",
                self.adapter_format_features,
                "adapter-specific texture format capabilities",
            ),
        ]
    }
}

/// The capability record of a GPU context: what the adapter offers, what the device has
/// enabled, and what the engine concludes from it.
///
/// Invariants (upheld by [`GpuCaps::new`] for consistent inputs): `enabled_features` is a
/// subset of `available_features`, and no field of `enabled_limits` is better than the same
/// field of `adapter_limits`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GpuCaps {
    /// Adapter identification: name, vendor and device id, driver, backend, device type.
    pub adapter: wgpu::AdapterInfo,
    /// The flags the wgpu instance of the device was created with
    /// ([`GpuInstance::flags`](super::GpuInstance::flags)), not the flags an instance created
    /// now would get. They matter for the report (validation layers distort timings) and for
    /// [`GpuDerived::indirect_draws_respect_first_vertex_and_instance`].
    pub instance_flags: wgpu::InstanceFlags,
    /// The DX12 shader compiler of that instance
    /// ([`GpuInstance::dx12_shader_compiler`](super::GpuInstance::dx12_shader_compiler));
    /// only DX12 devices use it. [`GpuCaps::new`] records the engine default, FXC; a context
    /// records its instance's choice through [`GpuCaps::with_dx12_shader_compiler`].
    pub dx12_shader_compiler: Dx12ShaderCompiler,
    /// Every feature the adapter offers.
    pub available_features: wgpu::Features,
    /// The features enabled on the device.
    pub enabled_features: wgpu::Features,
    /// The best limits the adapter supports.
    pub adapter_limits: wgpu::Limits,
    /// The limits of the device.
    pub enabled_limits: wgpu::Limits,
    /// Downlevel flags and shader model of the adapter.
    pub downlevel: wgpu::DownlevelCapabilities,
    /// Conclusions for the renderer.
    pub derived: GpuDerived,
    /// Starting point for the effect presets.
    pub tier: QualityTier,
}

impl GpuCaps {
    /// Builds the record for a device that was created on `profile` with the given features
    /// and limits, and derives the booleans and the tier.
    ///
    /// `instance_flags` must be the flags of the instance the device belongs to. wgpu cannot
    /// be asked for them afterwards, and a wrong value yields a wrong draw path on DX12.
    pub fn new(
        profile: AdapterProfile,
        instance_flags: wgpu::InstanceFlags,
        enabled_features: wgpu::Features,
        enabled_limits: wgpu::Limits,
    ) -> Self {
        let derived =
            GpuDerived::derive(&profile, instance_flags, enabled_features, &enabled_limits);
        let tier = quality_tier(&profile, &enabled_limits, &derived);
        Self {
            adapter: profile.info,
            instance_flags,
            dx12_shader_compiler: Dx12ShaderCompiler::default(),
            available_features: profile.features,
            enabled_features,
            adapter_limits: profile.limits,
            enabled_limits,
            downlevel: profile.downlevel,
            derived,
            tier,
        }
    }

    /// The record a device gets when it is created with [`AdapterProfile::plan_device`]:
    /// the complete negotiation as a pure function, usable without a GPU.
    ///
    /// ```
    /// use cp_render::gpu::{AdapterProfile, DrawPath, GpuCaps};
    /// use cp_render::wgpu;
    ///
    /// // A fabricated DX12 adapter that offers everything.
    /// let profile = AdapterProfile {
    ///     info: wgpu::AdapterInfo::new(wgpu::DeviceType::DiscreteGpu, wgpu::Backend::Dx12),
    ///     features: wgpu::Features::all(),
    ///     limits: wgpu::Limits::unlimited(),
    ///     downlevel: wgpu::DownlevelCapabilities::default(),
    /// };
    /// let caps = GpuCaps::negotiate(profile, wgpu::InstanceFlags::VALIDATION_INDIRECT_CALL);
    ///
    /// // Only what the pipeline can use is enabled, and limits stay at what it asks for.
    /// assert!(!caps.enabled_features.contains(wgpu::Features::SHADER_F64));
    /// assert!(!caps.enabled_features.contains(wgpu::Features::PIPELINE_CACHE));
    /// assert_eq!(caps.enabled_limits.max_buffer_size, 1 << 30);
    /// // DX12 has the count call, but draw path A is Vulkan only.
    /// assert!(caps.derived.can_multi_draw_indirect_count);
    /// assert_eq!(caps.derived.draw_path, DrawPath::B);
    /// ```
    pub fn negotiate(profile: AdapterProfile, instance_flags: wgpu::InstanceFlags) -> Self {
        let plan = profile.plan_device();
        Self::new(profile, instance_flags, plan.features, plan.limits)
    }

    /// The same record for a device whose instance compiles DX12 shaders with `compiler`.
    ///
    /// The compiler changes nothing that is derived; it is part of the record because the
    /// report has to say which one was in use (contract §4.5).
    #[must_use]
    pub fn with_dx12_shader_compiler(mut self, compiler: Dx12ShaderCompiler) -> Self {
        self.dx12_shader_compiler = compiler;
        self
    }

    /// The optional features of the pipeline, each with its availability on the adapter and
    /// its state on the device, in a fixed order.
    pub fn optional_features(&self) -> Vec<FeatureStatus> {
        WANTED_FEATURES
            .iter()
            .map(|wanted| FeatureStatus {
                feature: wanted.flag,
                name: wanted.name,
                available: self.available_features.contains(wanted.flag),
                enabled: self.enabled_features.contains(wanted.flag),
                used_for: wanted.used_for,
                fallback: wanted.fallback,
            })
            .collect()
    }
}

fn quality_tier(
    profile: &AdapterProfile,
    enabled: &wgpu::Limits,
    derived: &GpuDerived,
) -> QualityTier {
    // Only the limits the WebGPU specification defines count for the baseline. The wgpu-only
    // ones are zero on adapters that work perfectly well (wgpu itself zeroes them in its strict
    // compliance mode).
    let mut webgpu_minimum = wgpu::Limits::defaults();
    webgpu_minimum.zero_native_only();
    let baseline_webgpu =
        profile.downlevel.is_webgpu_compliant() && webgpu_minimum.check_limits(enabled);
    if profile.info.device_type == wgpu::DeviceType::Cpu || !baseline_webgpu {
        QualityTier::Low
    } else if profile.info.device_type == wgpu::DeviceType::DiscreteGpu
        && derived.can_indirect_first_instance
    {
        QualityTier::High
    } else {
        QualityTier::Medium
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::features::wanted_features;
    use crate::gpu::testing::{desktop_profile, indirect_validation, no_flags};
    use wgpu::{Backend, DeviceType, DownlevelFlags, Features};

    fn negotiate(profile: AdapterProfile) -> GpuCaps {
        GpuCaps::negotiate(profile, indirect_validation())
    }

    #[test]
    fn desktop_vulkan_gets_everything() {
        let caps = negotiate(desktop_profile(Backend::Vulkan, DeviceType::DiscreteGpu));
        assert_eq!(caps.enabled_features, wanted_features());
        assert!(caps.available_features.contains(caps.enabled_features));
        assert!(caps.enabled_limits.check_limits(&caps.adapter_limits));
        assert_eq!(caps.enabled_limits.max_buffer_size, 1 << 30);
        assert_eq!(caps.tier, QualityTier::High);

        let derived = caps.derived;
        assert_eq!(derived.draw_path, DrawPath::A);
        assert!(
            derived.rows().iter().all(|&(_, value, _)| value),
            "{derived:?}"
        );
        assert_eq!(caps.dx12_shader_compiler, Dx12ShaderCompiler::Fxc);
    }

    #[test]
    fn draw_path_is_exactly_the_contract_rule() {
        // A = Vulkan with MULTI_DRAW_INDIRECT_COUNT and INDIRECT_FIRST_INSTANCE enabled,
        // B = everything else, whatever the instance flags and the other features are.
        let pair = Features::MULTI_DRAW_INDIRECT_COUNT | Features::INDIRECT_FIRST_INSTANCE;
        for backend in Backend::ALL {
            let profile = desktop_profile(backend, DeviceType::DiscreteGpu);
            let limits = profile.plan_device().limits;
            for subset in [
                Features::empty(),
                Features::MULTI_DRAW_INDIRECT_COUNT,
                Features::INDIRECT_FIRST_INSTANCE,
                pair,
            ] {
                for others in [Features::empty(), wanted_features() - pair] {
                    for flags in [indirect_validation(), no_flags()] {
                        let caps =
                            GpuCaps::new(profile.clone(), flags, subset | others, limits.clone());
                        let expected = if backend == Backend::Vulkan && subset == pair {
                            DrawPath::A
                        } else {
                            DrawPath::B
                        };
                        assert_eq!(
                            caps.derived.draw_path, expected,
                            "{backend:?} {subset:?} {flags:?}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn the_dx12_compiler_is_recorded_and_derives_nothing() {
        let caps = negotiate(desktop_profile(Backend::Dx12, DeviceType::DiscreteGpu));
        let dxc = Dx12ShaderCompiler::DynamicDxc {
            path: "C:/dxc/dxcompiler.dll".to_owned(),
        };
        let with_dxc = caps.clone().with_dx12_shader_compiler(dxc.clone());
        assert_eq!(with_dxc.dx12_shader_compiler, dxc);
        assert_eq!(with_dxc.derived, caps.derived);
        assert_eq!(with_dxc.tier, caps.tier);
        assert_ne!(with_dxc, caps);
    }

    #[test]
    fn plan_and_negotiated_record_agree() {
        let profile = desktop_profile(Backend::Vulkan, DeviceType::DiscreteGpu);
        let plan = profile.plan_device();
        let caps = negotiate(profile.clone());
        assert_eq!(caps.enabled_features, plan.features);
        assert_eq!(caps.enabled_limits, plan.limits);
        assert_eq!(caps.adapter, profile.info);
        assert_eq!(caps.adapter_limits, profile.limits);
        assert_eq!(caps.available_features, profile.features);
        assert_eq!(caps.downlevel, profile.downlevel);
    }

    #[test]
    fn an_adapter_without_optional_features_degrades_to_path_b() {
        let mut profile = desktop_profile(Backend::Vulkan, DeviceType::DiscreteGpu);
        profile.features = Features::empty();
        let caps = negotiate(profile);
        assert_eq!(caps.enabled_features, Features::empty());
        assert_eq!(caps.enabled_limits.max_immediate_size, 0);
        assert!(
            caps.derived.rows().iter().all(|&(_, value, _)| !value),
            "{:?}",
            caps.derived
        );
        assert_eq!(caps.derived.draw_path, DrawPath::B);
        // Still a compliant discrete GPU, but without bucket indices in indirect draws it is
        // not of the baseline class.
        assert_eq!(caps.tier, QualityTier::Medium);
        assert!(
            caps.optional_features()
                .iter()
                .all(|status| !status.available && !status.enabled)
        );
    }

    #[test]
    fn optional_feature_rows_reflect_adapter_and_device() {
        let mut profile = desktop_profile(Backend::Vulkan, DeviceType::DiscreteGpu);
        profile
            .features
            .remove(Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES | Features::TIMESTAMP_QUERY);
        let caps = negotiate(profile);
        let rows = caps.optional_features();
        assert_eq!(rows.len(), WANTED_FEATURES.len());
        for row in &rows {
            assert_eq!(
                row.available,
                caps.available_features.contains(row.feature),
                "{}",
                row.name
            );
            assert_eq!(
                row.enabled,
                caps.enabled_features.contains(row.feature),
                "{}",
                row.name
            );
            assert!(
                !row.enabled || row.available,
                "{} enabled but unavailable",
                row.name
            );
        }
        let row = |name: &str| *rows.iter().find(|row| row.name == name).unwrap();
        assert!(!row("TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES").available);
        assert!(!row("TIMESTAMP_QUERY").available);
        assert!(rows.iter().all(|row| row.name != "PIPELINE_CACHE"));
        // Available on the adapter, but useless without the base feature: visible as such.
        let inside = row("TIMESTAMP_QUERY_INSIDE_PASSES");
        assert!(inside.available && !inside.enabled);
        assert!(row("MULTI_DRAW_INDIRECT_COUNT").enabled);
    }

    #[test]
    fn dx12_never_uses_the_count_path() {
        let caps = negotiate(desktop_profile(Backend::Dx12, DeviceType::DiscreteGpu));
        let derived = caps.derived;
        // The call exists on DX12, but its builtins would see vertex and instance zero.
        assert!(derived.can_multi_draw_indirect_count);
        assert!(derived.can_indirect_first_instance);
        assert!(!derived.indirect_count_draws_respect_first_vertex_and_instance);
        assert!(derived.indirect_draws_respect_first_vertex_and_instance);
        assert_eq!(derived.draw_path, DrawPath::B);
        assert_eq!(caps.tier, QualityTier::High);
    }

    #[test]
    fn dx12_without_indirect_validation_keeps_path_b_and_its_tier() {
        // wgpu's validation pass only feeds the builtins; the bucket is a vertex attribute.
        let profile = desktop_profile(Backend::Dx12, DeviceType::DiscreteGpu);
        let caps = GpuCaps::negotiate(profile, no_flags());
        assert!(
            !caps
                .derived
                .indirect_draws_respect_first_vertex_and_instance
        );
        assert!(caps.derived.can_indirect_first_instance);
        assert_eq!(caps.derived.draw_path, DrawPath::B);
        assert_eq!(caps.tier, QualityTier::High);
    }

    #[test]
    fn vulkan_does_not_depend_on_indirect_validation() {
        let profile = desktop_profile(Backend::Vulkan, DeviceType::DiscreteGpu);
        let caps = GpuCaps::negotiate(profile, no_flags());
        assert_eq!(caps.derived.draw_path, DrawPath::A);
    }

    #[test]
    fn missing_count_feature_selects_path_b() {
        let mut profile = desktop_profile(Backend::Vulkan, DeviceType::DiscreteGpu);
        profile.features.remove(Features::MULTI_DRAW_INDIRECT_COUNT);
        let caps = negotiate(profile);
        assert!(!caps.derived.can_multi_draw_indirect_count);
        assert!(caps.derived.can_indirect_first_instance);
        assert!(
            caps.derived
                .indirect_draws_respect_first_vertex_and_instance
        );
        assert_eq!(caps.derived.draw_path, DrawPath::B);
        assert_eq!(caps.tier, QualityTier::High);
    }

    #[test]
    fn missing_first_instance_selects_path_b_below_the_baseline() {
        let mut profile = desktop_profile(Backend::Vulkan, DeviceType::DiscreteGpu);
        profile.features.remove(Features::INDIRECT_FIRST_INSTANCE);
        let caps = negotiate(profile);
        assert!(caps.derived.can_multi_draw_indirect_count);
        assert!(!caps.derived.can_indirect_first_instance);
        assert!(
            !caps
                .derived
                .indirect_draws_respect_first_vertex_and_instance
        );
        assert!(
            !caps
                .derived
                .indirect_count_draws_respect_first_vertex_and_instance
        );
        assert_eq!(caps.derived.draw_path, DrawPath::B);
        assert_eq!(caps.tier, QualityTier::Medium);
    }

    #[test]
    fn missing_indirect_execution_rules_out_indirect_draws() {
        let mut profile = desktop_profile(Backend::Vulkan, DeviceType::DiscreteGpu);
        profile
            .downlevel
            .flags
            .remove(DownlevelFlags::INDIRECT_EXECUTION);
        let caps = negotiate(profile);
        assert!(!caps.derived.can_multi_draw_indirect_count);
        assert!(!caps.derived.can_indirect_first_instance);
        assert!(
            !caps
                .derived
                .indirect_draws_respect_first_vertex_and_instance
        );
        // Not baseline WebGPU any more.
        assert_eq!(caps.tier, QualityTier::Low);
        // The path rule looks at the backend and the two features only (contract §4.5). The
        // combination is synthetic: wgpu reports INDIRECT_EXECUTION on every Vulkan adapter.
        assert_eq!(caps.derived.draw_path, DrawPath::A);
    }

    #[test]
    fn other_backends_use_path_b() {
        for backend in [
            Backend::Metal,
            Backend::Gl,
            Backend::BrowserWebGpu,
            Backend::Noop,
        ] {
            let caps = negotiate(desktop_profile(backend, DeviceType::DiscreteGpu));
            assert_eq!(caps.derived.draw_path, DrawPath::B, "{backend:?}");
        }
    }

    #[test]
    fn timestamp_booleans_need_the_base_feature_on_the_device() {
        let profile = desktop_profile(Backend::Vulkan, DeviceType::DiscreteGpu);
        // A device that somebody created with the sub-features only.
        let features =
            Features::TIMESTAMP_QUERY_INSIDE_PASSES | Features::TIMESTAMP_QUERY_INSIDE_ENCODERS;
        let limits = profile.plan_device().limits;
        let caps = GpuCaps::new(
            profile.clone(),
            indirect_validation(),
            features,
            limits.clone(),
        );
        assert!(!caps.derived.can_timestamp_passes);
        assert!(!caps.derived.can_timestamp_inside_passes);
        assert!(!caps.derived.can_timestamp_inside_encoders);

        let features = Features::TIMESTAMP_QUERY | Features::TIMESTAMP_QUERY_INSIDE_PASSES;
        let caps = GpuCaps::new(profile, indirect_validation(), features, limits);
        assert!(caps.derived.can_timestamp_passes);
        assert!(caps.derived.can_timestamp_inside_passes);
        assert!(!caps.derived.can_timestamp_inside_encoders);
    }

    #[test]
    fn immediates_need_the_feature_and_a_non_zero_size() {
        // An adapter that has the feature but no immediate bytes: not requested at all.
        let mut profile = desktop_profile(Backend::Vulkan, DeviceType::DiscreteGpu);
        profile.limits.max_immediate_size = 0;
        let caps = negotiate(profile.clone());
        assert!(caps.available_features.contains(Features::IMMEDIATES));
        assert!(!caps.enabled_features.contains(Features::IMMEDIATES));
        assert_eq!(caps.enabled_limits.max_immediate_size, 0);
        assert!(!caps.derived.can_use_immediates);
        // Everything else is still requested.
        assert_eq!(
            caps.enabled_features,
            wanted_features() - Features::IMMEDIATES
        );

        // A device that somebody created with the feature but without bytes.
        let limits = profile.plan_device().limits;
        let caps = GpuCaps::new(profile, indirect_validation(), wanted_features(), limits);
        assert!(caps.enabled_features.contains(Features::IMMEDIATES));
        assert!(!caps.derived.can_use_immediates);

        // The feature without the limit raised on an adapter that has room.
        let profile = desktop_profile(Backend::Vulkan, DeviceType::DiscreteGpu);
        let caps = negotiate(profile);
        assert_eq!(caps.enabled_limits.max_immediate_size, 128);
        assert!(caps.derived.can_use_immediates);
    }

    #[test]
    fn booleans_describe_the_device_not_the_adapter() {
        // The adapter has everything, but the device was created with nothing enabled.
        let profile = desktop_profile(Backend::Vulkan, DeviceType::DiscreteGpu);
        let caps = GpuCaps::new(
            profile,
            indirect_validation(),
            Features::empty(),
            wgpu::Limits::defaults(),
        );
        assert!(caps.derived.rows().iter().all(|&(_, value, _)| !value));
        assert!(
            caps.optional_features()
                .iter()
                .all(|status| status.available && !status.enabled)
        );
    }

    #[test]
    fn tier_follows_the_device_type() {
        let tier = |device_type| negotiate(desktop_profile(Backend::Vulkan, device_type)).tier;
        assert_eq!(tier(DeviceType::DiscreteGpu), QualityTier::High);
        assert_eq!(tier(DeviceType::IntegratedGpu), QualityTier::Medium);
        assert_eq!(tier(DeviceType::VirtualGpu), QualityTier::Medium);
        assert_eq!(tier(DeviceType::Other), QualityTier::Medium);
        assert_eq!(tier(DeviceType::Cpu), QualityTier::Low);
    }

    #[test]
    fn tier_is_low_below_baseline_webgpu() {
        let mut profile = desktop_profile(Backend::Vulkan, DeviceType::DiscreteGpu);
        profile
            .downlevel
            .flags
            .remove(DownlevelFlags::COMPUTE_SHADERS);
        assert_eq!(negotiate(profile).tier, QualityTier::Low);

        let mut profile = desktop_profile(Backend::Vulkan, DeviceType::DiscreteGpu);
        profile.limits.max_texture_dimension_2d = 4096;
        assert_eq!(negotiate(profile).tier, QualityTier::Low);

        let mut profile = desktop_profile(Backend::Vulkan, DeviceType::DiscreteGpu);
        profile.downlevel.shader_model = wgpu::ShaderModel::Sm4;
        assert_eq!(negotiate(profile).tier, QualityTier::Low);

        // Anisotropic filtering is not part of WebGPU compliance.
        let mut profile = desktop_profile(Backend::Vulkan, DeviceType::DiscreteGpu);
        profile
            .downlevel
            .flags
            .remove(DownlevelFlags::ANISOTROPIC_FILTERING);
        assert_eq!(negotiate(profile).tier, QualityTier::High);
    }

    #[test]
    fn tier_ignores_limits_that_are_not_part_of_webgpu() {
        // What wgpu reports in its strict WebGPU mode: every wgpu-only limit is zero.
        let mut profile = desktop_profile(Backend::Vulkan, DeviceType::DiscreteGpu);
        profile.limits.zero_native_only();
        assert_eq!(
            profile
                .limits
                .max_buffers_and_acceleration_structures_per_shader_stage,
            0
        );
        let caps = negotiate(profile);
        assert!(caps.enabled_limits.check_limits(&caps.adapter_limits));
        assert_eq!(caps.tier, QualityTier::High);
    }

    #[test]
    fn tiers_are_ordered_and_named() {
        assert!(QualityTier::Low < QualityTier::Medium && QualityTier::Medium < QualityTier::High);
        let ids = [
            QualityTier::Low.id(),
            QualityTier::Medium.id(),
            QualityTier::High.id(),
        ];
        assert_eq!(ids, ["low", "medium", "high"]);
        assert_eq!([DrawPath::A, DrawPath::B].map(DrawPath::id), ["a", "b"]);
    }

    #[test]
    fn derived_rows_name_every_boolean_field() {
        // Guards the report: a boolean added to GpuDerived must also appear in rows(). The
        // Debug output lists all fields, so every `true`/`false` in it must have a row.
        let caps = negotiate(desktop_profile(Backend::Vulkan, DeviceType::DiscreteGpu));
        let debug = format!("{:?}", caps.derived);
        let booleans = debug.matches(": true").count() + debug.matches(": false").count();
        assert_eq!(booleans, caps.derived.rows().len());
        for (name, _, meaning) in caps.derived.rows() {
            assert!(
                debug.contains(&format!("{name}: ")),
                "{name} is not a field"
            );
            assert!(!meaning.is_empty());
        }
    }
}
