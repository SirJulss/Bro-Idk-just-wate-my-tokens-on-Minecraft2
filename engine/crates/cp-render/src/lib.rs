//! `cp-render`: the renderer of the Cubeplanet engine.
//!
//! Responsibility: the wgpu renderer, the frame graph and all shaders. Of the engine it knows
//! only the `cp-core` wire types; `cp-app` glues content and textures to it.
//!
//! Binding contract: `engine/docs/ARCHITEKTUR.md`, sections 2.4, 2.6, 4.5, 5.3 and 7.
//!
//! | module | contents |
//! |---|---|
//! | [`gpu`] | GPU context, capability negotiation and report (E0) |
//! | [`scope`] | error scopes: a GPU call whose failure is an `Err`, not a log line |
//! | [`shader`] | embedded WGSL library, `#include` preprocessor, naga validation, module creation |
//! | [`frame_graph`] | declared passes, transient texture pool, resize, per-pass GPU timings |
//! | [`headless`] | offscreen target and RGBA8 read-back for shots and tests |
//! | [`texture`] | block texture arrays (sRGB albedo, linear data) with caller-made mip chains |
//! | [`material`] | the material table storage buffer |
//! | [`overlay`] | own bitmap font and the text and rectangle overlay for debug HUD lines |
//! | [`debug_view`] | the debug views the voxel pipeline renders |
//! | [`voxel`] | the GPU-driven voxel pipeline: quad pool, draw items, objects, draw paths A, B and direct, G-buffer, lighting, tone mapping, calibration scenes |
//!
//! Everything that can fail on the device returns [`RenderError`]; pure checks have error
//! types of their own that convert into it.

pub mod debug_view;
pub mod error;
pub mod frame_graph;
pub mod gpu;
pub mod headless;
pub mod material;
pub mod overlay;
pub mod scope;
pub mod shader;
pub mod texture;
pub mod voxel;

pub use error::{RenderError, WgpuErrorKind};

/// The wgpu version this crate is built against.
///
/// The public API of [`gpu`] carries wgpu types (`Backends`, `PowerPreference`, `Instance`,
/// `Surface`, ...). Crates without a wgpu dependency of their own reach them through this
/// re-export and thereby always use the matching version.
pub use wgpu;
