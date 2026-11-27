//! The optional wgpu features the engine's pipeline can use (contract §4.5).
//!
//! Every entry is optional: the pipeline has a fallback for each, and the capability report
//! shows which ones the device got. Adding a feature the pipeline starts to use means adding one
//! entry to [`WANTED_FEATURES`].

/// One optional feature, what the pipeline does with it and how it copes without it.
#[derive(Clone, Copy, Debug)]
pub(crate) struct WantedFeature {
    /// Exactly one feature flag.
    pub flag: wgpu::Features,
    /// Name of the flag constant in `wgpu::Features`.
    pub name: &'static str,
    /// Features that must be enabled together with `flag`; without them `flag` is useless and
    /// is not requested. Every flag listed here is itself an entry without requirements.
    pub requires: wgpu::Features,
    /// What the pipeline uses the feature for.
    pub used_for: &'static str,
    /// What the pipeline does when the feature is missing.
    pub fallback: &'static str,
}

/// The feature whitelist of contract §4.5, in the contract's order (also the report order).
///
/// `PIPELINE_CACHE` is deliberately absent: creating a `wgpu::PipelineCache` is `unsafe`, and
/// the workspace denies `unsafe` code.
pub(crate) const WANTED_FEATURES: &[WantedFeature] = &[
    WantedFeature {
        flag: wgpu::Features::MULTI_DRAW_INDIRECT_COUNT,
        name: "MULTI_DRAW_INDIRECT_COUNT",
        requires: wgpu::Features::empty(),
        used_for: "draw path A (Vulkan): GPU culling writes the indirect arguments and their count",
        fallback: "draw path B: culling on the CPU, multi_draw_indexed_indirect with a known count",
    },
    WantedFeature {
        flag: wgpu::Features::INDIRECT_FIRST_INSTANCE,
        name: "INDIRECT_FIRST_INSTANCE",
        requires: wgpu::Features::empty(),
        used_for: "first_instance of indirect draws selects the bucket of a draw (both draw paths)",
        fallback: "draw path B with one direct draw per visible range",
    },
    WantedFeature {
        flag: wgpu::Features::TIMESTAMP_QUERY,
        name: "TIMESTAMP_QUERY",
        requires: wgpu::Features::empty(),
        used_for: "GPU time per pass (timestamps at pass begin and end)",
        fallback: "no GPU timings, CPU frame times only",
    },
    WantedFeature {
        flag: wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS,
        name: "TIMESTAMP_QUERY_INSIDE_ENCODERS",
        requires: wgpu::Features::TIMESTAMP_QUERY,
        used_for: "timestamps between passes (uploads, whole frame)",
        fallback: "pass timings only",
    },
    WantedFeature {
        flag: wgpu::Features::TIMESTAMP_QUERY_INSIDE_PASSES,
        name: "TIMESTAMP_QUERY_INSIDE_PASSES",
        requires: wgpu::Features::TIMESTAMP_QUERY,
        used_for: "timing scopes inside a render or compute pass",
        fallback: "one timing per pass",
    },
    WantedFeature {
        flag: wgpu::Features::RG11B10UFLOAT_RENDERABLE,
        name: "RG11B10UFLOAT_RENDERABLE",
        requires: wgpu::Features::empty(),
        used_for: "Rg11b10Ufloat targets without alpha: bloom, half-resolution reflections",
        fallback: "Rgba16Float targets (8 instead of 4 bytes per pixel)",
    },
    WantedFeature {
        flag: wgpu::Features::FLOAT32_FILTERABLE,
        name: "FLOAT32_FILTERABLE",
        requires: wgpu::Features::empty(),
        used_for: "linear sampling of 32-bit float textures (height maps, depth pyramid)",
        fallback: "nearest sampling with interpolation in the shader",
    },
    WantedFeature {
        flag: wgpu::Features::TEXTURE_FORMAT_16BIT_NORM,
        name: "TEXTURE_FORMAT_16BIT_NORM",
        requires: wgpu::Features::empty(),
        used_for: "macro height maps in R16Unorm (2 bytes per texel)",
        fallback: "macro height maps in R32Float (4 bytes per texel)",
    },
    WantedFeature {
        flag: wgpu::Features::TEXTURE_COMPRESSION_BC,
        name: "TEXTURE_COMPRESSION_BC",
        requires: wgpu::Features::empty(),
        used_for: "BC-compressed textures for large maps",
        fallback: "uncompressed textures",
    },
    WantedFeature {
        flag: wgpu::Features::DEPTH_CLIP_CONTROL,
        name: "DEPTH_CLIP_CONTROL",
        requires: wgpu::Features::empty(),
        used_for: "unclipped depth for shadow cascades (casters in front of the near plane)",
        fallback: "longer cascade depth ranges",
    },
    WantedFeature {
        flag: wgpu::Features::DUAL_SOURCE_BLENDING,
        name: "DUAL_SOURCE_BLENDING",
        requires: wgpu::Features::empty(),
        used_for: "per-channel transmittance when blending atmosphere, fog and clouds",
        fallback: "single-channel transmittance",
    },
    WantedFeature {
        flag: wgpu::Features::IMMEDIATES,
        name: "IMMEDIATES",
        requires: wgpu::Features::empty(),
        used_for: "small per-pass constants as immediate data, at most 128 bytes (no bind group)",
        fallback: "a small uniform buffer",
    },
    WantedFeature {
        flag: wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES,
        name: "TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES",
        requires: wgpu::Features::empty(),
        used_for: "format capabilities beyond WebGPU, e.g. read-write storage textures",
        fallback: "the format capabilities WebGPU guarantees",
    },
];

/// All flags of [`WANTED_FEATURES`] together.
pub(crate) fn wanted_features() -> wgpu::Features {
    WANTED_FEATURES
        .iter()
        .fold(wgpu::Features::empty(), |all, wanted| all | wanted.flag)
}

/// The features to request from an adapter that offers `available`.
///
/// The result is a subset of both `available` and [`wanted_features`], so a device request
/// with it never fails because of a feature, and nothing the pipeline cannot use is enabled.
pub(crate) fn select_features(available: wgpu::Features) -> wgpu::Features {
    WANTED_FEATURES
        .iter()
        .filter(|wanted| available.contains(wanted.flag | wanted.requires))
        .fold(wgpu::Features::empty(), |enabled, wanted| {
            enabled | wanted.flag
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every subset of the wanted features, as adapter feature sets.
    fn wanted_subsets() -> impl Iterator<Item = wgpu::Features> {
        (0u32..1 << WANTED_FEATURES.len()).map(|mask| {
            WANTED_FEATURES
                .iter()
                .enumerate()
                .filter(|(bit, _)| mask & (1 << bit) != 0)
                .fold(wgpu::Features::empty(), |set, (_, wanted)| {
                    set | wanted.flag
                })
        })
    }

    #[test]
    fn table_names_match_the_wgpu_constants() {
        for wanted in WANTED_FEATURES {
            assert_eq!(
                wgpu::Features::from_name(wanted.name),
                Some(wanted.flag),
                "name and flag of {} disagree",
                wanted.name
            );
            assert_eq!(
                wanted.flag.iter().count(),
                1,
                "{} must be a single flag",
                wanted.name
            );
            assert!(!wanted.used_for.is_empty() && !wanted.fallback.is_empty());
        }
    }

    #[test]
    fn table_has_no_duplicates_and_covers_the_contract_list() {
        let mut seen = wgpu::Features::empty();
        for wanted in WANTED_FEATURES {
            assert!(
                !seen.intersects(wanted.flag),
                "{} is listed twice",
                wanted.name
            );
            seen |= wanted.flag;
        }
        // Contract §4.5, the whitelist exactly as listed there.
        let contract = wgpu::Features::MULTI_DRAW_INDIRECT_COUNT
            | wgpu::Features::INDIRECT_FIRST_INSTANCE
            | wgpu::Features::TIMESTAMP_QUERY
            | wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS
            | wgpu::Features::TIMESTAMP_QUERY_INSIDE_PASSES
            | wgpu::Features::RG11B10UFLOAT_RENDERABLE
            | wgpu::Features::FLOAT32_FILTERABLE
            | wgpu::Features::TEXTURE_FORMAT_16BIT_NORM
            | wgpu::Features::TEXTURE_COMPRESSION_BC
            | wgpu::Features::DEPTH_CLIP_CONTROL
            | wgpu::Features::DUAL_SOURCE_BLENDING
            | wgpu::Features::IMMEDIATES
            | wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES;
        assert_eq!(wanted_features(), contract);
    }

    #[test]
    fn the_report_order_is_the_order_of_the_contract() {
        let names: Vec<&str> = WANTED_FEATURES.iter().map(|wanted| wanted.name).collect();
        assert_eq!(
            names,
            [
                "MULTI_DRAW_INDIRECT_COUNT",
                "INDIRECT_FIRST_INSTANCE",
                "TIMESTAMP_QUERY",
                "TIMESTAMP_QUERY_INSIDE_ENCODERS",
                "TIMESTAMP_QUERY_INSIDE_PASSES",
                "RG11B10UFLOAT_RENDERABLE",
                "FLOAT32_FILTERABLE",
                "TEXTURE_FORMAT_16BIT_NORM",
                "TEXTURE_COMPRESSION_BC",
                "DEPTH_CLIP_CONTROL",
                "DUAL_SOURCE_BLENDING",
                "IMMEDIATES",
                "TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES",
            ]
        );
    }

    #[test]
    fn the_pipeline_cache_is_never_requested() {
        // Contract §4.5: creating a pipeline cache is `unsafe`.
        let cache = wgpu::Features::PIPELINE_CACHE;
        assert!(!wanted_features().contains(cache));
        assert!(!select_features(wgpu::Features::all()).contains(cache));
    }

    #[test]
    fn requirements_are_wanted_features_without_requirements_of_their_own() {
        for wanted in WANTED_FEATURES {
            for required in wanted.requires.iter() {
                let entry = WANTED_FEATURES
                    .iter()
                    .find(|candidate| candidate.flag == required)
                    .unwrap_or_else(|| panic!("{} requires an unlisted feature", wanted.name));
                assert!(
                    entry.requires.is_empty(),
                    "{} has a chained requirement",
                    wanted.name
                );
            }
        }
    }

    #[test]
    fn nothing_is_wanted_that_needs_the_experimental_opt_in() {
        // Experimental features need `DeviceDescriptor::experimental_features`, which the
        // context leaves disabled.
        assert!(!wanted_features().intersects(wgpu::Features::all_experimental_mask()));
    }

    #[test]
    fn an_adapter_without_optional_features_gets_none() {
        assert_eq!(
            select_features(wgpu::Features::empty()),
            wgpu::Features::empty()
        );
        // Features outside the table are never requested, however many the adapter has.
        let unrelated = wgpu::Features::all() - wanted_features();
        assert_eq!(select_features(unrelated), wgpu::Features::empty());
    }

    #[test]
    fn an_adapter_with_everything_gets_exactly_the_wanted_set() {
        assert_eq!(select_features(wgpu::Features::all()), wanted_features());
        assert_eq!(select_features(wanted_features()), wanted_features());
    }

    #[test]
    fn selection_is_a_subset_of_available_and_wanted_for_every_combination() {
        let noise = wgpu::Features::all() - wanted_features();
        for available in wanted_subsets() {
            for adapter in [available, available | noise] {
                let enabled = select_features(adapter);
                assert!(
                    adapter.contains(enabled),
                    "requested a feature the adapter lacks"
                );
                assert!(
                    wanted_features().contains(enabled),
                    "requested an unwanted feature"
                );
                for wanted in WANTED_FEATURES {
                    let expected = adapter.contains(wanted.flag | wanted.requires);
                    assert_eq!(enabled.contains(wanted.flag), expected, "{}", wanted.name);
                    // A feature is never enabled without what it depends on.
                    if enabled.contains(wanted.flag) {
                        assert!(enabled.contains(wanted.requires), "{}", wanted.name);
                    }
                }
            }
        }
    }

    #[test]
    fn timestamp_sub_features_need_the_base_feature() {
        let inside = wgpu::Features::TIMESTAMP_QUERY_INSIDE_PASSES
            | wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS;
        assert_eq!(select_features(inside), wgpu::Features::empty());
        assert_eq!(
            select_features(inside | wgpu::Features::TIMESTAMP_QUERY),
            inside | wgpu::Features::TIMESTAMP_QUERY
        );
        assert_eq!(
            select_features(wgpu::Features::TIMESTAMP_QUERY),
            wgpu::Features::TIMESTAMP_QUERY
        );
    }
}
