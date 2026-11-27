//! The wgpu instance of a GPU context, together with what wgpu does not hand back about it.

use std::ops::Deref;

use super::options::Dx12ShaderCompiler;
use super::{GpuError, GpuOptions};

/// A wgpu instance that knows the flags and the DX12 shader compiler it was created with.
///
/// The flags of an instance decide how its devices behave: on DX12, for example, indirect
/// draws deliver `first_vertex` and `first_instance` to the builtins only under
/// `InstanceFlags::VALIDATION_INDIRECT_CALL`. wgpu 30 has no way to read the flags or the
/// compiler of an existing instance, so a capability record built on a bare `wgpu::Instance`
/// could only assume them. This type closes that gap: the only way to get one is
/// [`GpuContext::create_instance`](super::GpuContext::create_instance), which keeps what it
/// passes to wgpu, and [`GpuContext::with_instance`](super::GpuContext::with_instance)
/// accepts nothing else.
///
/// It dereferences to the `wgpu::Instance`, so `instance.create_surface(window)` and every
/// other method of the instance can be called on it directly. Cloning is cheap; all clones
/// are the same instance.
#[derive(Clone, Debug)]
pub struct GpuInstance {
    instance: wgpu::Instance,
    flags: wgpu::InstanceFlags,
    dx12_shader_compiler: Dx12ShaderCompiler,
}

impl GpuInstance {
    /// Creates the instance for `opts`: the backends of [`GpuOptions::resolved_backends`]
    /// that this build contains, the flags of [`GpuOptions::flags`] and the DX12 shader
    /// compiler of [`Dx12ShaderCompiler::from_env`].
    ///
    /// # Errors
    /// [`GpuError::InvalidBackendOverride`], [`GpuError::DxcNotFound`] or
    /// [`GpuError::BackendNotCompiled`].
    pub(crate) fn new(opts: &GpuOptions) -> Result<Self, GpuError> {
        Ok(Self::from_descriptor(instance_descriptor(opts)?))
    }

    /// Creates the instance `descriptor` describes.
    ///
    /// Every instance of this module comes from here, which is what makes [`Self::flags`] and
    /// [`Self::dx12_shader_compiler`] true by construction: both are taken from the very
    /// descriptor wgpu receives. `descriptor.backends` must contain a backend of this build,
    /// or wgpu panics.
    pub(crate) fn from_descriptor(descriptor: wgpu::InstanceDescriptor) -> Self {
        let flags = descriptor.flags;
        let dx12_shader_compiler =
            Dx12ShaderCompiler::from_wgpu(&descriptor.backend_options.dx12.shader_compiler);
        Self {
            instance: wgpu::Instance::new(descriptor),
            flags,
            dx12_shader_compiler,
        }
    }

    /// The flags the instance was created with ([`GpuOptions::flags`]).
    pub fn flags(&self) -> wgpu::InstanceFlags {
        self.flags
    }

    /// The shader compiler of the instance's DX12 backend: FXC unless `CP_DXC` named DXC
    /// when the instance was created.
    pub fn dx12_shader_compiler(&self) -> &Dx12ShaderCompiler {
        &self.dx12_shader_compiler
    }
}

impl Deref for GpuInstance {
    type Target = wgpu::Instance;

    fn deref(&self) -> &wgpu::Instance {
        &self.instance
    }
}

/// The descriptor of the instance for `opts`.
///
/// No display handle is passed: of the backends wgpu has, only GLES needs one, and the
/// engine's backends are Vulkan and DX12.
///
/// The DX12 shader compiler is always set explicitly (contract §4.5), so neither wgpu's
/// `Auto` default nor its `WGPU_DX12_COMPILER` variable decides it. The other backend options
/// are wgpu's, including its environment variables.
pub(crate) fn instance_descriptor(opts: &GpuOptions) -> Result<wgpu::InstanceDescriptor, GpuError> {
    let compiler = Dx12ShaderCompiler::from_env()?;
    let mut descriptor = wgpu::InstanceDescriptor::new_without_display_handle();
    descriptor.backends = backends_for(opts)?;
    descriptor.flags = opts.flags;
    descriptor.backend_options = wgpu::BackendOptions::from_env_or_default();
    descriptor.backend_options.dx12.shader_compiler = compiler.to_wgpu();
    Ok(descriptor)
}

/// The flags of an instance created for the default options.
///
/// What an existing instance has is [`GpuInstance::flags`], never this function.
#[cfg(test)]
pub(crate) fn default_instance_flags() -> wgpu::InstanceFlags {
    GpuOptions::default().flags
}

/// The backends to search for `opts`: the resolved request, reduced to what this build contains.
pub(crate) fn backends_for(opts: &GpuOptions) -> Result<wgpu::Backends, GpuError> {
    usable_backends(
        opts.resolved_backends()?,
        wgpu::Instance::enabled_backend_features(),
    )
}

/// The part of `requested` that is in `compiled`; an error when nothing is left, because wgpu
/// would then have no backend to create an instance with.
fn usable_backends(
    requested: wgpu::Backends,
    compiled: wgpu::Backends,
) -> Result<wgpu::Backends, GpuError> {
    let usable = requested & compiled;
    if usable.is_empty() {
        Err(GpuError::BackendNotCompiled {
            requested,
            compiled,
        })
    } else {
        Ok(usable)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_compiled_backends_are_used() {
        use wgpu::Backends;
        let desktop = Backends::VULKAN | Backends::DX12;
        assert_eq!(usable_backends(desktop, desktop).unwrap(), desktop);
        assert_eq!(
            usable_backends(desktop, Backends::DX12).unwrap(),
            Backends::DX12
        );
        assert_eq!(
            usable_backends(Backends::VULKAN | Backends::GL, desktop).unwrap(),
            Backends::VULKAN
        );
    }

    #[test]
    fn a_request_without_any_compiled_backend_is_an_error() {
        use wgpu::Backends;
        let compiled = Backends::VULKAN | Backends::DX12;
        for requested in [Backends::GL, Backends::METAL, Backends::empty()] {
            match usable_backends(requested, compiled) {
                Err(GpuError::BackendNotCompiled {
                    requested: reported,
                    compiled: built,
                }) => {
                    assert_eq!(reported, requested);
                    assert_eq!(built, compiled);
                }
                other => panic!("{requested:?} must be rejected, got {other:?}"),
            }
        }
        assert!(usable_backends(compiled, Backends::empty()).is_err());
    }

    #[test]
    fn the_default_backends_are_part_of_a_windows_build() {
        // The baseline platform must be able to use both backends of the contract.
        if cfg!(windows) {
            let compiled = wgpu::Instance::enabled_backend_features();
            assert!(compiled.contains(wgpu::Backends::VULKAN | wgpu::Backends::DX12));
        }
    }

    /// The descriptor for the default options, or `None` when this build has neither backend
    /// of the contract. Any other failure is a test failure.
    fn default_descriptor_or_skip(test: &str) -> Option<wgpu::InstanceDescriptor> {
        match instance_descriptor(&GpuOptions::default()) {
            Ok(descriptor) => Some(descriptor),
            Err(error @ GpuError::BackendNotCompiled { .. }) => {
                crate::gpu::testing::skip(test, &error);
                None
            }
            Err(error) => panic!("{test}: no instance descriptor: {error}"),
        }
    }

    #[test]
    fn an_instance_keeps_the_flags_it_was_created_with() {
        let test = "an_instance_keeps_the_flags_it_was_created_with";
        // Two sets that differ from each other, so at least one of them is not the default of
        // this build and environment.
        for flags in [
            wgpu::InstanceFlags::empty(),
            wgpu::InstanceFlags::VALIDATION_INDIRECT_CALL | wgpu::InstanceFlags::DISCARD_HAL_LABELS,
        ] {
            let Some(mut descriptor) = default_descriptor_or_skip(test) else {
                return;
            };
            descriptor.flags = flags;
            let instance = GpuInstance::from_descriptor(descriptor);
            assert_eq!(instance.flags(), flags);
            assert_eq!(instance.clone().flags(), flags);
        }
    }

    #[test]
    fn a_default_instance_has_the_default_flags_and_the_requested_backends() {
        let test = "a_default_instance_has_the_default_flags_and_the_requested_backends";
        let Some(descriptor) = default_descriptor_or_skip(test) else {
            return;
        };
        let opts = GpuOptions::default();
        assert_eq!(descriptor.flags, default_instance_flags());
        assert_eq!(descriptor.backends, backends_for(&opts).unwrap());
        assert!(descriptor.display.is_none());
        let compiler = Dx12ShaderCompiler::from_env().unwrap();
        assert_eq!(
            Dx12ShaderCompiler::from_wgpu(&descriptor.backend_options.dx12.shader_compiler),
            compiler
        );
        if std::env::var_os(crate::gpu::DXC_ENV).is_none() {
            assert_eq!(compiler, Dx12ShaderCompiler::Fxc);
        }

        let instance = GpuInstance::new(&opts).unwrap();
        assert_eq!(instance.flags(), default_instance_flags());
        assert_eq!(instance.dx12_shader_compiler(), &compiler);
    }

    #[test]
    fn the_flags_of_the_options_reach_the_instance() {
        let test = "the_flags_of_the_options_reach_the_instance";
        for flags in [
            wgpu::InstanceFlags::empty(),
            wgpu::InstanceFlags::VALIDATION_INDIRECT_CALL | wgpu::InstanceFlags::DEBUG,
        ] {
            let opts = GpuOptions {
                flags,
                ..GpuOptions::default()
            };
            match instance_descriptor(&opts) {
                Ok(descriptor) => assert_eq!(descriptor.flags, flags),
                Err(error @ GpuError::BackendNotCompiled { .. }) => {
                    crate::gpu::testing::skip(test, &error);
                    return;
                }
                Err(error) => panic!("{test}: no instance descriptor: {error}"),
            }
            assert_eq!(GpuInstance::new(&opts).unwrap().flags(), flags);
        }
    }

    #[test]
    fn it_can_be_used_where_a_wgpu_instance_is_expected() {
        fn takes_wgpu(instance: &wgpu::Instance) -> wgpu::Instance {
            instance.clone()
        }
        let test = "it_can_be_used_where_a_wgpu_instance_is_expected";
        let Some(descriptor) = default_descriptor_or_skip(test) else {
            return;
        };
        let instance = GpuInstance::from_descriptor(descriptor);
        // Deref coercion for arguments, auto-deref for methods.
        let raw = takes_wgpu(&instance);
        assert_eq!(raw, *instance);
        assert_eq!(raw, *instance.clone());
        // No device exists on this instance, so there is nothing in flight.
        assert!(instance.poll_all(false));
    }
}
