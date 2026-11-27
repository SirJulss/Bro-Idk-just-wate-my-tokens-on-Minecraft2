//! Options of GPU context creation and the backend selection rules (contract §4.5).

use super::GpuError;

/// Environment variable that forces a single backend: `vulkan` or `dx12` (case-insensitive).
///
/// It overrides [`GpuOptions::backends`]; an empty value counts as not set.
pub const BACKEND_ENV: &str = "CP_BACKEND";

/// Environment variable that selects DXC as the DX12 shader compiler: the path of
/// `dxcompiler.dll`. Unset or empty means FXC (contract §4.5).
pub const DXC_ENV: &str = "CP_DXC";

/// The shader compiler of the DX12 backend of an instance (contract §4.5).
///
/// The engine creates instances with [`Fxc`](Self::Fxc), or with
/// [`DynamicDxc`](Self::DynamicDxc) when [`DXC_ENV`] names the DLL; never with wgpu's default
/// `Auto`, which picks a different compiler depending on what happens to be installed. The
/// other two variants describe instances made from a hand-built descriptor. Vulkan devices
/// compile through naga to SPIR-V and do not use the setting.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub enum Dx12ShaderCompiler {
    /// FXC (`d3dcompiler_47.dll`), part of Windows.
    #[default]
    Fxc,
    /// DXC, loaded from the `dxcompiler.dll` at `path`.
    DynamicDxc {
        /// Path of `dxcompiler.dll` as given in [`DXC_ENV`].
        path: String,
    },
    /// DXC linked into the executable (wgpu feature `static-dxc`, not part of this build).
    StaticDxc,
    /// wgpu's own choice: static DXC, else a DXC on the `PATH`, else FXC.
    Auto,
}

impl Dx12ShaderCompiler {
    /// Stable machine-readable name, used in the JSON report.
    pub fn id(&self) -> &'static str {
        match self {
            Self::Fxc => "fxc",
            Self::DynamicDxc { .. } => "dynamic_dxc",
            Self::StaticDxc => "static_dxc",
            Self::Auto => "auto",
        }
    }

    /// Path of `dxcompiler.dll` for [`Self::DynamicDxc`].
    pub fn dxc_path(&self) -> Option<&str> {
        match self {
            Self::DynamicDxc { path } => Some(path),
            _ => None,
        }
    }

    /// The wgpu setting for this choice.
    pub fn to_wgpu(&self) -> wgpu::Dx12Compiler {
        match self {
            Self::Fxc => wgpu::Dx12Compiler::Fxc,
            Self::DynamicDxc { path } => wgpu::Dx12Compiler::DynamicDxc {
                dxc_path: path.clone(),
            },
            Self::StaticDxc => wgpu::Dx12Compiler::StaticDxc,
            Self::Auto => wgpu::Dx12Compiler::Auto,
        }
    }

    /// The choice a wgpu setting stands for.
    pub fn from_wgpu(compiler: &wgpu::Dx12Compiler) -> Self {
        match compiler {
            wgpu::Dx12Compiler::Fxc => Self::Fxc,
            wgpu::Dx12Compiler::DynamicDxc { dxc_path } => Self::DynamicDxc {
                path: dxc_path.clone(),
            },
            wgpu::Dx12Compiler::StaticDxc => Self::StaticDxc,
            wgpu::Dx12Compiler::Auto => Self::Auto,
        }
    }

    /// The compiler that [`DXC_ENV`] selects now.
    ///
    /// # Errors
    /// [`GpuError::DxcNotFound`] when `CP_DXC` names something that is not a file.
    pub fn from_env() -> Result<Self, GpuError> {
        let value = std::env::var_os(DXC_ENV).map(|value| value.to_string_lossy().into_owned());
        resolve_dx12_compiler(value.as_deref())
    }
}

/// Pure core of [`Dx12ShaderCompiler::from_env`]: FXC unless `env_value` names a file.
pub(crate) fn resolve_dx12_compiler(
    env_value: Option<&str>,
) -> Result<Dx12ShaderCompiler, GpuError> {
    let Some(path) = env_value.map(str::trim).filter(|path| !path.is_empty()) else {
        return Ok(Dx12ShaderCompiler::Fxc);
    };
    if std::path::Path::new(path).is_file() {
        Ok(Dx12ShaderCompiler::DynamicDxc {
            path: path.to_owned(),
        })
    } else {
        Err(GpuError::DxcNotFound {
            path: path.to_owned(),
        })
    }
}

/// Backends used when neither the options nor the environment name any.
pub(crate) const DEFAULT_BACKENDS: wgpu::Backends =
    wgpu::Backends::VULKAN.union(wgpu::Backends::DX12);

/// Preference order of backends: when several have a usable adapter, the earliest wins.
///
/// The contract fixes "Vulkan, then DX12"; the rest only matters when a caller explicitly
/// allows other backends.
pub(crate) const BACKEND_ORDER: [wgpu::Backend; 6] = [
    wgpu::Backend::Vulkan,
    wgpu::Backend::Dx12,
    wgpu::Backend::Metal,
    wgpu::Backend::Gl,
    wgpu::Backend::BrowserWebGpu,
    wgpu::Backend::Noop,
];

/// How to create the [`GpuContext`](super::GpuContext).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GpuOptions {
    /// Backends that may be used. `None` selects the engine default: Vulkan, then DX12.
    ///
    /// Among the allowed backends Vulkan is always preferred over DX12. The environment
    /// variable [`BACKEND_ENV`] overrides this field.
    pub backends: Option<wgpu::Backends>,
    /// Which adapter to prefer when a backend offers several (discrete or integrated GPU).
    pub power: wgpu::PowerPreference,
    /// Flags of the wgpu instance; the capability report shows them.
    ///
    /// The default is `InstanceFlags::from_build_config()`: API validation and debug labels in
    /// debug builds, indirect-call validation always. wgpu's `WGPU_*` environment variables are
    /// not read; a caller that wants them passes `from_build_config().with_env()`.
    pub flags: wgpu::InstanceFlags,
}

impl Default for GpuOptions {
    /// Default backends, the high-performance adapter (the baseline is a discrete GPU) and
    /// wgpu's instance flags for the build type.
    fn default() -> Self {
        Self {
            backends: None,
            power: wgpu::PowerPreference::HighPerformance,
            flags: wgpu::InstanceFlags::from_build_config(),
        }
    }
}

impl GpuOptions {
    /// The backends to use: `CP_BACKEND` if set, otherwise [`Self::backends`], otherwise
    /// Vulkan and DX12.
    ///
    /// # Errors
    /// [`GpuError::InvalidBackendOverride`] when `CP_BACKEND` is neither `vulkan` nor `dx12`.
    pub fn resolved_backends(&self) -> Result<wgpu::Backends, GpuError> {
        let env_override =
            std::env::var_os(BACKEND_ENV).map(|value| value.to_string_lossy().into_owned());
        resolve_backends(self.backends, env_override.as_deref())
    }
}

/// Pure core of [`GpuOptions::resolved_backends`]; the environment value is passed in so that
/// the rule can be tested without touching the process environment.
pub(crate) fn resolve_backends(
    configured: Option<wgpu::Backends>,
    env_override: Option<&str>,
) -> Result<wgpu::Backends, GpuError> {
    if let Some(raw) = env_override {
        let value = raw.trim();
        if !value.is_empty() {
            return match value.to_ascii_lowercase().as_str() {
                "vulkan" => Ok(wgpu::Backends::VULKAN),
                "dx12" => Ok(wgpu::Backends::DX12),
                _ => Err(GpuError::InvalidBackendOverride {
                    value: raw.to_owned(),
                }),
            };
        }
    }
    Ok(configured.unwrap_or(DEFAULT_BACKENDS))
}

/// Position of `backend` in [`BACKEND_ORDER`]; lower is preferred.
pub(crate) fn backend_rank(backend: wgpu::Backend) -> usize {
    BACKEND_ORDER
        .iter()
        .position(|&candidate| candidate == backend)
        .unwrap_or(BACKEND_ORDER.len())
}

/// Lower-case names of the backends in `backends`, in preference order: `"vulkan, dx12"`.
pub(crate) fn backend_list(backends: &wgpu::Backends) -> String {
    let names: Vec<&str> = BACKEND_ORDER
        .iter()
        .filter(|&&backend| backends.contains(wgpu::Backends::from(backend)))
        .map(|backend| backend.to_str())
        .collect();
    if names.is_empty() {
        "none".to_owned()
    } else {
        names.join(", ")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_is_vulkan_then_dx12() {
        let backends = resolve_backends(None, None).unwrap();
        assert_eq!(backends, wgpu::Backends::VULKAN | wgpu::Backends::DX12);
        assert!(backend_rank(wgpu::Backend::Vulkan) < backend_rank(wgpu::Backend::Dx12));
        assert_eq!(backend_list(&backends), "vulkan, dx12");
    }

    #[test]
    fn every_backend_has_a_distinct_rank() {
        for (expected, backend) in wgpu::Backend::ALL.into_iter().enumerate() {
            let rank = backend_rank(backend);
            assert!(
                rank < BACKEND_ORDER.len(),
                "{backend:?} is missing from BACKEND_ORDER"
            );
            let same_rank = wgpu::Backend::ALL
                .iter()
                .filter(|&&b| backend_rank(b) == rank)
                .count();
            assert_eq!(
                same_rank, 1,
                "{backend:?} shares its rank (entry {expected})"
            );
        }
    }

    #[test]
    fn env_override_selects_one_backend() {
        for value in ["vulkan", "VULKAN", " Vulkan\t"] {
            assert_eq!(
                resolve_backends(None, Some(value)).unwrap(),
                wgpu::Backends::VULKAN
            );
        }
        for value in ["dx12", "DX12", "Dx12 "] {
            assert_eq!(
                resolve_backends(None, Some(value)).unwrap(),
                wgpu::Backends::DX12
            );
        }
    }

    #[test]
    fn env_override_beats_the_options() {
        let configured = Some(wgpu::Backends::VULKAN);
        assert_eq!(
            resolve_backends(configured, Some("dx12")).unwrap(),
            wgpu::Backends::DX12
        );
        assert_eq!(
            resolve_backends(configured, None).unwrap(),
            wgpu::Backends::VULKAN
        );
    }

    #[test]
    fn empty_env_override_counts_as_unset() {
        for value in ["", "   ", "\t"] {
            assert_eq!(
                resolve_backends(None, Some(value)).unwrap(),
                DEFAULT_BACKENDS
            );
            assert_eq!(
                resolve_backends(Some(wgpu::Backends::DX12), Some(value)).unwrap(),
                wgpu::Backends::DX12
            );
        }
    }

    #[test]
    fn unknown_env_override_is_an_error_not_a_silent_default() {
        for value in ["gl", "dx11", "vulkan,dx12", "d3d12", "vk", "metal"] {
            match resolve_backends(None, Some(value)) {
                Err(GpuError::InvalidBackendOverride { value: reported }) => {
                    assert_eq!(reported, value);
                }
                other => panic!("{value:?} must be rejected, got {other:?}"),
            }
        }
    }

    #[test]
    fn backend_list_handles_the_empty_set() {
        assert_eq!(backend_list(&wgpu::Backends::empty()), "none");
        assert_eq!(backend_list(&wgpu::Backends::DX12), "dx12");
        assert_eq!(
            backend_list(&(wgpu::Backends::METAL | wgpu::Backends::VULKAN)),
            "vulkan, metal"
        );
    }

    #[test]
    fn default_options_prefer_the_fast_adapter() {
        let options = GpuOptions::default();
        assert_eq!(options.backends, None);
        assert_eq!(options.power, wgpu::PowerPreference::HighPerformance);
        assert_eq!(options.flags, wgpu::InstanceFlags::from_build_config());
    }

    #[test]
    fn fxc_is_the_dx12_compiler_unless_cp_dxc_names_a_file() {
        for unset in [None, Some(""), Some("  \t")] {
            assert_eq!(
                resolve_dx12_compiler(unset).unwrap(),
                Dx12ShaderCompiler::Fxc
            );
        }
        // Any existing file is accepted; whether it is a working DXC shows when wgpu loads it.
        let file = concat!(env!("CARGO_MANIFEST_DIR"), "/Cargo.toml");
        let compiler = resolve_dx12_compiler(Some(file)).unwrap();
        assert_eq!(compiler.id(), "dynamic_dxc");
        assert_eq!(compiler.dxc_path(), Some(file));
        assert!(matches!(
            compiler.to_wgpu(),
            wgpu::Dx12Compiler::DynamicDxc { dxc_path } if dxc_path == file
        ));
    }

    #[test]
    fn cp_dxc_without_a_file_is_an_error_not_a_silent_fallback() {
        let directory = env!("CARGO_MANIFEST_DIR");
        let missing = concat!(env!("CARGO_MANIFEST_DIR"), "/no-such-dxcompiler.dll");
        for path in [directory, missing] {
            match resolve_dx12_compiler(Some(path)) {
                Err(GpuError::DxcNotFound { path: reported }) => assert_eq!(reported, path),
                other => panic!("{path:?} must be rejected, got {other:?}"),
            }
        }
    }

    #[test]
    fn dx12_compiler_round_trips_through_wgpu() {
        let all = [
            Dx12ShaderCompiler::Fxc,
            Dx12ShaderCompiler::DynamicDxc {
                path: "dxcompiler.dll".to_owned(),
            },
            Dx12ShaderCompiler::StaticDxc,
            Dx12ShaderCompiler::Auto,
        ];
        for compiler in &all {
            assert_eq!(
                &Dx12ShaderCompiler::from_wgpu(&compiler.to_wgpu()),
                compiler
            );
        }
        let ids: Vec<&str> = all.iter().map(Dx12ShaderCompiler::id).collect();
        assert_eq!(ids, ["fxc", "dynamic_dxc", "static_dxc", "auto"]);
        assert_eq!(Dx12ShaderCompiler::default(), Dx12ShaderCompiler::Fxc);
    }
}
