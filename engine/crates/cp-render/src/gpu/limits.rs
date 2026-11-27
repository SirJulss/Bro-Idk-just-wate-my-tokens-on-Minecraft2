//! The device limits the engine asks for (contract §4.5).
//!
//! The WebGPU defaults are too small for a few things the pipeline does: the quad pool in one
//! large storage buffer, block textures in texture arrays with many layers, the inputs of the
//! lighting pass, the single-mip storage views of the depth pyramid and the G-buffer targets.
//! Exactly the limits the contract lists are raised here, each to what the pipeline needs;
//! every other limit keeps its WebGPU default so that the device behaves the same on every
//! desktop adapter. Nothing is ever requested beyond what the adapter offers.

/// No buffer of the engine is larger than this: 4 GiB (contract §4.5).
///
/// Buffer offsets and sizes in shaders stay within `u32` addressing, and no buffer request can
/// depend on an adapter that reports more.
pub(crate) const BUFFER_BYTES_CEILING: u64 = 4 << 30;

/// Largest single buffer, in bytes: 1 GiB.
///
/// The largest buffer is the quad pool (contract §5.3: default 128 MiB, larger in higher
/// quality tiers); 1 GiB holds 134 million packed quads, is below the limit of every desktop
/// adapter on Vulkan and DX12 and therefore behaves identically on both backends.
pub(crate) const WANTED_BUFFER_BYTES: u64 = 1 << 30;

/// Largest storage buffer binding, in bytes: the whole quad pool in one binding.
pub(crate) const WANTED_STORAGE_BINDING_BYTES: u64 = WANTED_BUFFER_BYTES;

const _: () = assert!(
    WANTED_BUFFER_BYTES <= BUFFER_BYTES_CEILING
        && WANTED_STORAGE_BINDING_BYTES <= WANTED_BUFFER_BYTES
);

/// Layers of a 2D texture array: as many as the adapter has (the request is clamped to it).
pub(crate) const WANTED_TEXTURE_ARRAY_LAYERS: u32 = u32::MAX;

/// Bytes of immediate data: 128, the size DX12 offers and Vulkan guarantees, and the most the
/// contract allows a pipeline to rely on.
pub(crate) const WANTED_IMMEDIATE_BYTES: u32 = 128;

/// Sampled textures visible in one shader stage.
///
/// The deferred lighting pass alone reads the G-buffer, depth, shadow cascades, the light
/// clipmap, ambient occlusion, reflections and the atmosphere tables; the default of 16 leaves
/// no room for that.
pub(crate) const WANTED_SAMPLED_TEXTURES_PER_STAGE: u32 = 32;

/// Storage textures visible in one shader stage: the depth pyramid writes each of its mips
/// through its own single-mip storage view (contract §5.3), up to 12 at 1080p and above.
pub(crate) const WANTED_STORAGE_TEXTURES_PER_STAGE: u32 = 16;

/// Bytes per sample over all colour targets of one pass: room for the G-buffer (albedo, normal
/// and roughness, material, velocity) next to an HDR target, twice the default of 32.
pub(crate) const WANTED_COLOR_ATTACHMENT_BYTES_PER_SAMPLE: u32 = 64;

/// Bind groups of a pipeline layout; 8 is the maximum wgpu supports.
pub(crate) const WANTED_BIND_GROUPS: u32 = 8;

/// The limits the engine would like to have, given the features that will be enabled.
///
/// Starts from `Limits::default()` (the WebGPU defaults) and raises exactly the limits of
/// contract §4.5. `max_immediate_size` is only non-zero together with
/// [`wgpu::Features::IMMEDIATES`], as wgpu requires.
pub(crate) fn wanted_limits(features: wgpu::Features) -> wgpu::Limits {
    wgpu::Limits {
        max_buffer_size: WANTED_BUFFER_BYTES,
        max_storage_buffer_binding_size: WANTED_STORAGE_BINDING_BYTES,
        max_texture_array_layers: WANTED_TEXTURE_ARRAY_LAYERS,
        max_immediate_size: if features.contains(wgpu::Features::IMMEDIATES) {
            WANTED_IMMEDIATE_BYTES
        } else {
            0
        },
        max_sampled_textures_per_shader_stage: WANTED_SAMPLED_TEXTURES_PER_STAGE,
        max_storage_textures_per_shader_stage: WANTED_STORAGE_TEXTURES_PER_STAGE,
        max_color_attachment_bytes_per_sample: WANTED_COLOR_ATTACHMENT_BYTES_PER_SAMPLE,
        max_bind_groups: WANTED_BIND_GROUPS,
        ..wgpu::Limits::default()
    }
}

/// The limits to request from an adapter: [`wanted_limits`] clamped to what the adapter has.
///
/// Every field of the result is equal to or worse than the adapter's value, so the device
/// request cannot fail on a limit. On adapters below the WebGPU defaults the result is below
/// the defaults too.
pub(crate) fn select_limits(features: wgpu::Features, adapter: &wgpu::Limits) -> wgpu::Limits {
    wanted_limits(features).or_worse_values_from(adapter)
}

/// Every field of [`wgpu::Limits`] with its name, in declaration order.
pub(crate) fn limit_values(limits: &wgpu::Limits) -> Vec<(&'static str, u64)> {
    macro_rules! values {
        ($($field:ident),* $(,)?) => {{
            // The pattern is exhaustive on purpose: when a wgpu update adds a limit, this stops
            // compiling instead of the limit silently missing from the report.
            let wgpu::Limits { $($field),* } = limits;
            vec![$((stringify!($field), u64::from(*$field))),*]
        }};
    }
    values!(
        max_texture_dimension_1d,
        max_texture_dimension_2d,
        max_texture_dimension_3d,
        max_texture_array_layers,
        max_bind_groups,
        max_bind_groups_plus_vertex_buffers,
        max_bindings_per_bind_group,
        max_dynamic_uniform_buffers_per_pipeline_layout,
        max_dynamic_storage_buffers_per_pipeline_layout,
        max_sampled_textures_per_shader_stage,
        max_samplers_per_shader_stage,
        max_storage_buffers_per_shader_stage,
        max_storage_textures_per_shader_stage,
        max_uniform_buffers_per_shader_stage,
        max_binding_array_elements_per_shader_stage,
        max_binding_array_acceleration_structure_elements_per_shader_stage,
        max_binding_array_sampler_elements_per_shader_stage,
        max_uniform_buffer_binding_size,
        max_storage_buffer_binding_size,
        max_vertex_buffers,
        max_buffer_size,
        max_vertex_attributes,
        max_vertex_buffer_array_stride,
        max_inter_stage_shader_variables,
        min_uniform_buffer_offset_alignment,
        min_storage_buffer_offset_alignment,
        max_color_attachments,
        max_color_attachment_bytes_per_sample,
        max_compute_workgroup_storage_size,
        max_compute_invocations_per_workgroup,
        max_compute_workgroup_size_x,
        max_compute_workgroup_size_y,
        max_compute_workgroup_size_z,
        max_compute_workgroups_per_dimension,
        max_immediate_size,
        max_non_sampler_bindings,
        max_task_workgroup_total_count,
        max_task_workgroups_per_dimension,
        max_mesh_workgroup_total_count,
        max_mesh_workgroups_per_dimension,
        max_task_invocations_per_workgroup,
        max_task_invocations_per_dimension,
        max_mesh_invocations_per_workgroup,
        max_mesh_invocations_per_dimension,
        max_task_payload_size,
        max_mesh_output_vertices,
        max_mesh_output_primitives,
        max_mesh_output_layers,
        max_mesh_multiview_view_count,
        max_blas_primitive_count,
        max_blas_geometry_count,
        max_tlas_instance_count,
        max_acceleration_structures_per_shader_stage,
        max_buffers_and_acceleration_structures_per_shader_stage,
        max_multiview_view_count,
        max_ray_dispatch_count,
        max_ray_recursion_depth,
    )
}

/// How a limit value is shown in the readable report.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum LimitUnit {
    /// A plain count.
    Count,
    /// A size in bytes.
    Bytes,
}

/// A limit that matters to the pipeline and is listed in the readable report.
#[derive(Clone, Copy, Debug)]
pub(crate) struct ReportedLimit {
    /// Field name in `wgpu::Limits`.
    pub name: &'static str,
    /// Reads the field from a set of limits.
    pub get: fn(&wgpu::Limits) -> u64,
    /// How the value is printed.
    pub unit: LimitUnit,
    /// What the pipeline needs the limit for.
    pub used_for: &'static str,
}

macro_rules! reported {
    ($field:ident, $unit:ident, $used_for:literal) => {
        ReportedLimit {
            name: stringify!($field),
            get: |limits| u64::from(limits.$field),
            unit: LimitUnit::$unit,
            used_for: $used_for,
        }
    };
}

/// The limits shown in the readable report: the raised ones first, then those that bound the
/// design of the frame.
pub(crate) const REPORTED_LIMITS: &[ReportedLimit] = &[
    reported!(
        max_buffer_size,
        Bytes,
        "largest buffer: the quad pool (never above 4 GiB)"
    ),
    reported!(
        max_storage_buffer_binding_size,
        Bytes,
        "the quad pool bound as one storage buffer"
    ),
    reported!(max_texture_array_layers, Count, "block texture arrays"),
    reported!(max_immediate_size, Bytes, "immediate data per pass"),
    reported!(
        max_sampled_textures_per_shader_stage,
        Count,
        "inputs of the lighting pass"
    ),
    reported!(
        max_storage_textures_per_shader_stage,
        Count,
        "single-mip views of the depth pyramid"
    ),
    reported!(
        max_color_attachment_bytes_per_sample,
        Bytes,
        "G-buffer bytes per pixel"
    ),
    reported!(
        max_bind_groups,
        Count,
        "frame, pass, material and object bind groups"
    ),
    reported!(
        max_storage_buffers_per_shader_stage,
        Count,
        "pools bound by culling and drawing"
    ),
    reported!(
        max_buffers_and_acceleration_structures_per_shader_stage,
        Count,
        "storage + uniform + vertex buffers of one stage"
    ),
    reported!(
        max_uniform_buffer_binding_size,
        Bytes,
        "frame and pass constants"
    ),
    reported!(
        max_uniform_buffers_per_shader_stage,
        Count,
        "constant blocks of one stage"
    ),
    reported!(
        max_samplers_per_shader_stage,
        Count,
        "samplers of one stage"
    ),
    reported!(max_bindings_per_bind_group, Count, "highest binding index"),
    reported!(
        max_texture_dimension_2d,
        Count,
        "render targets, shadow atlas, macro maps"
    ),
    reported!(
        max_texture_dimension_3d,
        Count,
        "light volumes, froxel grids, cloud noise"
    ),
    reported!(max_color_attachments, Count, "G-buffer targets"),
    reported!(
        max_compute_invocations_per_workgroup,
        Count,
        "compute workgroup size"
    ),
    reported!(
        max_compute_workgroups_per_dimension,
        Count,
        "culling dispatch size"
    ),
    reported!(
        min_uniform_buffer_offset_alignment,
        Bytes,
        "uniform offsets in shared buffers (lower is better)"
    ),
    reported!(
        min_storage_buffer_offset_alignment,
        Bytes,
        "storage offsets in shared buffers (lower is better)"
    ),
];

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::features::wanted_features;
    use crate::gpu::testing::desktop_limits;

    fn presets() -> [(&'static str, wgpu::Limits); 5] {
        [
            ("webgpu defaults", wgpu::Limits::defaults()),
            ("downlevel", wgpu::Limits::downlevel_defaults()),
            ("webgl2", wgpu::Limits::downlevel_webgl2_defaults()),
            ("unlimited", wgpu::Limits::unlimited()),
            ("desktop", desktop_limits()),
        ]
    }

    fn feature_sets() -> [wgpu::Features; 3] {
        [
            wgpu::Features::empty(),
            wgpu::Features::IMMEDIATES,
            wanted_features(),
        ]
    }

    /// Small deterministic generator for the property test (xorshift64*).
    struct TestRng(u64);

    impl TestRng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 >> 12;
            self.0 ^= self.0 << 25;
            self.0 ^= self.0 >> 27;
            self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
        }

        /// A value around the interesting sizes: zero, small, near a power of two, or huge.
        fn limit(&mut self) -> u64 {
            let shift = self.next() % 40;
            match self.next() % 5 {
                0 => 0,
                1 => self.next() % 64,
                2 => (1 << shift) - 1,
                3 => 1 << shift,
                _ => self.next() % (1 << 34),
            }
        }

        fn limit_u32(&mut self) -> u32 {
            u32::try_from(self.limit().min(u64::from(u32::MAX))).unwrap()
        }
    }

    #[test]
    fn limit_values_lists_every_field_once_and_reads_the_right_one() {
        let limits = desktop_limits();
        let values = limit_values(&limits);
        assert_eq!(values.len(), 57);
        for (index, (name, _)) in values.iter().enumerate() {
            assert!(
                values[index + 1..].iter().all(|(other, _)| other != name),
                "{name} is listed twice"
            );
        }
        let value = |name: &str| values.iter().find(|(n, _)| *n == name).unwrap().1;
        assert_eq!(value("max_buffer_size"), limits.max_buffer_size);
        assert_eq!(value("max_texture_array_layers"), 2048);
        assert_eq!(value("min_storage_buffer_offset_alignment"), 32);
        assert_eq!(value("max_ray_recursion_depth"), 0);
    }

    #[test]
    fn reported_limits_use_real_field_names_and_matching_getters() {
        let limits = desktop_limits();
        let values = limit_values(&limits);
        for (index, row) in REPORTED_LIMITS.iter().enumerate() {
            let (_, expected) = values
                .iter()
                .find(|(name, _)| *name == row.name)
                .unwrap_or_else(|| panic!("{} is not a wgpu limit", row.name));
            assert_eq!((row.get)(&limits), *expected, "{}", row.name);
            assert!(
                REPORTED_LIMITS[index + 1..]
                    .iter()
                    .all(|other| other.name != row.name),
                "{} is reported twice",
                row.name
            );
            assert!(!row.used_for.is_empty());
        }
    }

    #[test]
    fn wanted_limits_raise_exactly_the_documented_fields() {
        let wanted = wanted_limits(wanted_features());
        let defaults = wgpu::Limits::defaults();
        let raised: Vec<(&str, u64)> = limit_values(&wanted)
            .into_iter()
            .zip(limit_values(&defaults))
            .filter(|((_, wanted), (_, default))| wanted != default)
            .map(|((name, wanted), _)| (name, wanted))
            .collect();
        // Contract §4.5 lists exactly these, in wgpu's declaration order here.
        assert_eq!(
            raised,
            [
                ("max_texture_array_layers", u64::from(u32::MAX)),
                ("max_bind_groups", 8),
                ("max_sampled_textures_per_shader_stage", 32),
                ("max_storage_textures_per_shader_stage", 16),
                ("max_storage_buffer_binding_size", 1 << 30),
                ("max_buffer_size", 1 << 30),
                ("max_color_attachment_bytes_per_sample", 64),
                ("max_immediate_size", 128),
            ]
        );
        // Raised means better: the WebGPU defaults stay within the wanted limits.
        assert!(defaults.check_limits(&wanted));
        assert_eq!(wgpu::Limits::default(), defaults);
    }

    #[test]
    fn no_buffer_above_four_gib_is_ever_requested() {
        assert_eq!(BUFFER_BYTES_CEILING, 4 * 1024 * 1024 * 1024);
        for (name, adapter) in presets() {
            for features in feature_sets() {
                for limits in [wanted_limits(features), select_limits(features, &adapter)] {
                    assert!(limits.max_buffer_size <= BUFFER_BYTES_CEILING, "{name}");
                    assert!(
                        limits.max_storage_buffer_binding_size <= limits.max_buffer_size,
                        "{name}"
                    );
                }
            }
        }
    }

    #[test]
    fn immediate_data_never_exceeds_128_bytes() {
        let roomy = wgpu::Limits {
            max_immediate_size: 4096,
            ..desktop_limits()
        };
        for features in feature_sets() {
            assert!(select_limits(features, &roomy).max_immediate_size <= 128);
        }
    }

    #[test]
    fn texture_array_layers_are_the_adapter_maximum() {
        for (name, adapter) in presets() {
            let enabled = select_limits(wanted_features(), &adapter);
            assert_eq!(
                enabled.max_texture_array_layers, adapter.max_texture_array_layers,
                "{name}"
            );
        }
    }

    #[test]
    fn immediate_bytes_are_only_requested_with_the_feature() {
        let adapter = desktop_limits();
        assert_eq!(wanted_limits(wgpu::Features::empty()).max_immediate_size, 0);
        assert_eq!(
            select_limits(wgpu::Features::empty(), &adapter).max_immediate_size,
            0
        );
        assert_eq!(
            select_limits(wgpu::Features::IMMEDIATES, &adapter).max_immediate_size,
            128
        );

        let small = wgpu::Limits {
            max_immediate_size: 64,
            ..desktop_limits()
        };
        assert_eq!(
            select_limits(wgpu::Features::IMMEDIATES, &small).max_immediate_size,
            64
        );
    }

    #[test]
    fn a_generous_adapter_grants_exactly_the_wanted_limits() {
        let adapter = wgpu::Limits::unlimited();
        for features in feature_sets() {
            let enabled = select_limits(features, &adapter);
            // Texture array layers are asked for as "the adapter maximum" (contract §4.5).
            let expected = wgpu::Limits {
                max_texture_array_layers: adapter.max_texture_array_layers,
                ..wanted_limits(features)
            };
            assert_eq!(enabled, expected);
        }
        let enabled = select_limits(wanted_features(), &desktop_limits());
        assert_eq!(enabled.max_buffer_size, 1 << 30);
        assert_eq!(enabled.max_storage_buffer_binding_size, 1 << 30);
        assert_eq!(enabled.max_texture_array_layers, 2048);
        assert_eq!(enabled.max_bind_groups, 8);
        assert_eq!(enabled.max_immediate_size, 128);
        assert_eq!(enabled.max_sampled_textures_per_shader_stage, 32);
        assert_eq!(enabled.max_storage_textures_per_shader_stage, 16);
        assert_eq!(enabled.max_color_attachment_bytes_per_sample, 64);
        // Not raised: stays at the WebGPU default although the adapter offers more.
        assert_eq!(enabled.max_texture_dimension_2d, 8192);
        assert_eq!(enabled.max_storage_buffers_per_shader_stage, 8);
        // "Lower is better" limits keep the default too; the adapter's 64 would be better.
        assert_eq!(enabled.min_uniform_buffer_offset_alignment, 256);
    }

    #[test]
    fn a_weak_adapter_clamps_every_raised_limit() {
        let adapter = wgpu::Limits::downlevel_webgl2_defaults();
        let enabled = select_limits(wanted_features(), &adapter);
        assert_eq!(enabled.max_buffer_size, adapter.max_buffer_size);
        assert_eq!(enabled.max_storage_buffer_binding_size, 0);
        assert_eq!(enabled.max_storage_buffers_per_shader_stage, 0);
        assert_eq!(enabled.max_texture_array_layers, 256);
        assert_eq!(enabled.max_bind_groups, 4);
        assert_eq!(enabled.max_immediate_size, 0);
        assert_eq!(
            enabled.max_sampled_textures_per_shader_stage,
            adapter.max_sampled_textures_per_shader_stage
        );
        assert_eq!(enabled.max_storage_textures_per_shader_stage, 0);
        assert_eq!(
            enabled.max_color_attachment_bytes_per_sample,
            adapter.max_color_attachment_bytes_per_sample
        );
        // Limits below the WebGPU default are taken over as well, not left at the default.
        assert_eq!(enabled.max_texture_dimension_2d, 2048);
        assert_eq!(enabled.max_compute_workgroups_per_dimension, 0);
    }

    #[test]
    fn selected_limits_never_exceed_the_adapter_presets() {
        for (name, adapter) in presets() {
            for features in feature_sets() {
                let enabled = select_limits(features, &adapter);
                // The same check wgpu runs on a device request.
                let mut failures = Vec::new();
                enabled.check_limits_with_fail_fn(&adapter, false, |limit, requested, allowed| {
                    failures.push(format!("{limit}: {requested} > {allowed}"));
                });
                assert!(failures.is_empty(), "{name}: {failures:?}");
            }
        }
    }

    #[test]
    fn selected_limits_are_the_minimum_of_wanted_and_adapter_for_random_adapters() {
        let mut rng = TestRng(0x9E37_79B9_7F4A_7C15);
        let bases = presets();
        for round in 0..4000 {
            let base = bases[round % bases.len()].1.clone();
            let adapter = wgpu::Limits {
                max_buffer_size: rng.limit(),
                max_storage_buffer_binding_size: rng.limit(),
                max_uniform_buffer_binding_size: rng.limit(),
                max_texture_array_layers: rng.limit_u32(),
                max_texture_dimension_2d: rng.limit_u32(),
                max_storage_buffers_per_shader_stage: rng.limit_u32(),
                max_buffers_and_acceleration_structures_per_shader_stage: rng.limit_u32(),
                max_bind_groups: rng.limit_u32(),
                max_immediate_size: rng.limit_u32(),
                max_color_attachments: rng.limit_u32(),
                min_uniform_buffer_offset_alignment: rng.limit_u32(),
                min_storage_buffer_offset_alignment: rng.limit_u32(),
                ..base
            };
            let features = feature_sets()[round % 3];
            let wanted = wanted_limits(features);
            let enabled = select_limits(features, &adapter);

            assert!(
                enabled.check_limits(&adapter),
                "round {round}: above the adapter"
            );

            let rows = limit_values(&enabled).into_iter();
            let rows = rows.zip(limit_values(&wanted)).zip(limit_values(&adapter));
            for (((name, enabled), (_, wanted)), (_, adapter)) in rows {
                // Alignments are minima: there the larger value is the worse one.
                let expected = if name.starts_with("min_") {
                    wanted.max(adapter)
                } else {
                    wanted.min(adapter)
                };
                assert_eq!(enabled, expected, "round {round}: {name}");
            }
        }
    }
}
