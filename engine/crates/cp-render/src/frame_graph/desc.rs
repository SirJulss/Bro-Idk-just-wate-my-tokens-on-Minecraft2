//! Declarations of a frame graph: transient textures and passes, checked without a GPU.

use thiserror::Error;

/// Most passes a frame graph can have: every pass takes two timestamp queries, and a query
/// set holds at most `wgpu::QUERY_SET_MAX_QUERIES` (4096, contract §5.3).
pub const MAX_PASSES: usize = (wgpu::QUERY_SET_MAX_QUERIES / 2) as usize;

/// A transient texture of a frame graph, by declaration order.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct TransientId(pub(crate) usize);

impl TransientId {
    /// Position in [`FrameGraphDesc::transients`].
    pub fn index(self) -> usize {
        self.0
    }
}

/// A pass of a frame graph, by declaration order, which is also the execution order.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct PassId(pub(crate) usize);

impl PassId {
    /// Position in [`FrameGraphDesc::passes`].
    pub fn index(self) -> usize {
        self.0
    }
}

/// Whether a pass draws or dispatches.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PassKind {
    /// A render pass; the textures it writes are colour or depth attachments.
    Render,
    /// A compute pass; the textures it writes are storage textures.
    Compute,
}

/// The size of a transient texture, usually relative to the output of the frame.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TransientSize {
    /// `output * numerator / denominator` per axis, rounded up, at least one texel.
    Relative {
        /// Numerator, at least 1.
        numerator: u32,
        /// Denominator, at least 1.
        denominator: u32,
    },
    /// A size independent of the output (shadow maps, lookup tables), in texels.
    Fixed {
        /// Width, at least 1.
        width: u32,
        /// Height, at least 1.
        height: u32,
    },
}

impl TransientSize {
    /// The size of the output.
    pub const FULL: Self = Self::Relative {
        numerator: 1,
        denominator: 1,
    };
    /// Half the output per axis (rounded up).
    pub const HALF: Self = Self::Relative {
        numerator: 1,
        denominator: 2,
    };
    /// A quarter of the output per axis (rounded up).
    pub const QUARTER: Self = Self::Relative {
        numerator: 1,
        denominator: 4,
    };

    /// The size in texels for an output of `output` pixels; never zero.
    pub fn resolve(self, output: (u32, u32)) -> (u32, u32) {
        match self {
            Self::Relative {
                numerator,
                denominator,
            } => {
                let scale = |side: u32| {
                    let scaled = (u64::from(side) * u64::from(numerator))
                        .div_ceil(u64::from(denominator.max(1)));
                    u32::try_from(scaled).unwrap_or(u32::MAX).max(1)
                };
                (scale(output.0), scale(output.1))
            }
            Self::Fixed { width, height } => (width.max(1), height.max(1)),
        }
    }

    fn is_valid(self) -> bool {
        match self {
            Self::Relative {
                numerator,
                denominator,
            } => numerator >= 1 && denominator >= 1,
            Self::Fixed { width, height } => width >= 1 && height >= 1,
        }
    }
}

/// A texture that lives for one frame: written by a pass, read by later passes of the same
/// frame, its contents undefined at the start of the next. The graph creates it, keeps it in
/// its pool across frames and re-creates it when its size changes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TransientDesc {
    /// Name for labels and reports; unique within the graph.
    pub name: String,
    /// Texture format.
    pub format: wgpu::TextureFormat,
    /// Size, usually relative to the output.
    pub size: TransientSize,
    /// Mip levels, at least 1; clamped to the full chain of the resolved size.
    pub mip_level_count: u32,
    /// Usages beyond those the declarations imply (see [`FrameGraphDesc::usage`]), for
    /// example `COPY_SRC` for a read-back.
    pub extra_usage: wgpu::TextureUsages,
}

impl TransientDesc {
    /// A single-mip texture of `format` and `size` without extra usages.
    pub fn new(name: impl Into<String>, format: wgpu::TextureFormat, size: TransientSize) -> Self {
        Self {
            name: name.into(),
            format,
            size,
            mip_level_count: 1,
            extra_usage: wgpu::TextureUsages::empty(),
        }
    }

    /// The same texture with `count` mip levels.
    #[must_use]
    pub fn with_mip_levels(mut self, count: u32) -> Self {
        self.mip_level_count = count;
        self
    }

    /// The same texture with `usage` added to the implied usages.
    #[must_use]
    pub fn with_usage(mut self, usage: wgpu::TextureUsages) -> Self {
        self.extra_usage |= usage;
        self
    }
}

/// A declared pass: what it reads and writes of the transient textures.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PassDesc {
    /// Name for labels and timings; unique within the graph.
    pub name: String,
    /// Render or compute.
    pub kind: PassKind,
    /// Transients the pass samples or loads; an earlier pass must write each of them.
    pub reads: Vec<TransientId>,
    /// Transients the pass renders or stores into.
    pub writes: Vec<TransientId>,
}

/// Why a frame graph declaration or its use in a frame is wrong.
#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub enum FrameGraphError {
    /// Two passes or two transients have the same name.
    #[error("the frame graph has two {what} named {name:?}")]
    DuplicateName {
        /// `"passes"` or `"transients"`.
        what: &'static str,
        /// The name.
        name: String,
    },
    /// A pass refers to a transient the graph does not declare.
    #[error("pass {pass:?} uses transient {index}, which the frame graph does not declare")]
    UnknownTransient {
        /// The pass.
        pass: String,
        /// The index it uses.
        index: usize,
    },
    /// A pass reads a transient that no earlier pass writes in the frame.
    #[error("pass {pass:?} reads {transient:?} before any pass of the frame writes it")]
    ReadBeforeWrite {
        /// The pass.
        pass: String,
        /// The transient.
        transient: String,
    },
    /// A transient no pass uses.
    #[error("transient {transient:?} is declared but no pass uses it")]
    Unused {
        /// The transient.
        transient: String,
    },
    /// A size with a zero side, numerator or denominator.
    #[error("transient {transient:?} has the invalid size {size:?}")]
    InvalidSize {
        /// The transient.
        transient: String,
        /// The size.
        size: TransientSize,
    },
    /// Zero mip levels.
    #[error("transient {transient:?} needs at least one mip level")]
    NoMipLevels {
        /// The transient.
        transient: String,
    },
    /// More passes than one query set can time.
    #[error("{passes} passes exceed the {MAX_PASSES} passes of a frame graph")]
    TooManyPasses {
        /// Passes declared.
        passes: usize,
    },
    /// A pass id that does not belong to this graph.
    #[error("pass {index} does not belong to this frame graph")]
    UnknownPass {
        /// The index of the id.
        index: usize,
    },
    /// A pass was begun as the other kind.
    #[error("pass {pass:?} is declared as a {declared:?} pass but was begun as a {begun:?} pass")]
    WrongPassKind {
        /// The pass.
        pass: String,
        /// Its declared kind.
        declared: PassKind,
        /// The kind it was begun as.
        begun: PassKind,
    },
    /// A pass was begun after a pass that is declared after it, or twice in one frame.
    #[error(
        "pass {pass:?} was begun after {previous:?}; passes run in declaration order, each at \
         most once per frame"
    )]
    OutOfOrder {
        /// The pass begun.
        pass: String,
        /// The last pass begun before it.
        previous: String,
    },
}

/// The passes and transients of a frame graph, in execution order.
///
/// ```
/// use cp_render::frame_graph::{FrameGraphDesc, TransientDesc, TransientSize};
/// use cp_render::wgpu;
///
/// let mut graph = FrameGraphDesc::new();
/// let albedo = graph.transient(TransientDesc::new(
///     "gbuffer.albedo",
///     wgpu::TextureFormat::Rgba8UnormSrgb,
///     TransientSize::FULL,
/// ));
/// let depth = graph.transient(TransientDesc::new(
///     "depth",
///     wgpu::TextureFormat::Depth32Float,
///     TransientSize::FULL,
/// ));
/// let gbuffer = graph.render_pass("gbuffer", &[], &[albedo, depth]);
/// let lighting = graph.render_pass("lighting", &[albedo, depth], &[]);
/// graph.validate()?;
/// assert!(gbuffer < lighting);
/// assert!(graph.usage(depth).contains(wgpu::TextureUsages::RENDER_ATTACHMENT));
/// # Ok::<(), cp_render::frame_graph::FrameGraphError>(())
/// ```
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct FrameGraphDesc {
    transients: Vec<TransientDesc>,
    passes: Vec<PassDesc>,
}

impl FrameGraphDesc {
    /// An empty declaration.
    pub fn new() -> Self {
        Self::default()
    }

    /// Declares a transient texture.
    pub fn transient(&mut self, desc: TransientDesc) -> TransientId {
        self.transients.push(desc);
        TransientId(self.transients.len() - 1)
    }

    /// Declares the next render pass.
    pub fn render_pass(
        &mut self,
        name: impl Into<String>,
        reads: &[TransientId],
        writes: &[TransientId],
    ) -> PassId {
        self.pass(name.into(), PassKind::Render, reads, writes)
    }

    /// Declares the next compute pass.
    pub fn compute_pass(
        &mut self,
        name: impl Into<String>,
        reads: &[TransientId],
        writes: &[TransientId],
    ) -> PassId {
        self.pass(name.into(), PassKind::Compute, reads, writes)
    }

    fn pass(
        &mut self,
        name: String,
        kind: PassKind,
        reads: &[TransientId],
        writes: &[TransientId],
    ) -> PassId {
        self.passes.push(PassDesc {
            name,
            kind,
            reads: reads.to_vec(),
            writes: writes.to_vec(),
        });
        PassId(self.passes.len() - 1)
    }

    /// The transients in declaration order.
    pub fn transients(&self) -> &[TransientDesc] {
        &self.transients
    }

    /// The passes in execution order.
    pub fn passes(&self) -> &[PassDesc] {
        &self.passes
    }

    /// Checks the declaration: unique names, valid sizes, known transients, every transient
    /// used, every read preceded by a write in an earlier pass, at most [`MAX_PASSES`] passes.
    ///
    /// # Errors
    /// The first problem found, as a [`FrameGraphError`].
    pub fn validate(&self) -> Result<(), FrameGraphError> {
        if self.passes.len() > MAX_PASSES {
            return Err(FrameGraphError::TooManyPasses {
                passes: self.passes.len(),
            });
        }
        unique_names(
            "transients",
            self.transients.iter().map(|t| t.name.as_str()),
        )?;
        unique_names("passes", self.passes.iter().map(|p| p.name.as_str()))?;
        for transient in &self.transients {
            if !transient.size.is_valid() {
                return Err(FrameGraphError::InvalidSize {
                    transient: transient.name.clone(),
                    size: transient.size,
                });
            }
            if transient.mip_level_count == 0 {
                return Err(FrameGraphError::NoMipLevels {
                    transient: transient.name.clone(),
                });
            }
        }
        let mut written = vec![false; self.transients.len()];
        let mut used = vec![false; self.transients.len()];
        for pass in &self.passes {
            for id in pass.reads.iter().chain(&pass.writes) {
                if id.0 >= self.transients.len() {
                    return Err(FrameGraphError::UnknownTransient {
                        pass: pass.name.clone(),
                        index: id.0,
                    });
                }
                used[id.0] = true;
            }
            // Even a pass that changes a texture in place reads what an earlier pass wrote.
            if let Some(id) = pass.reads.iter().find(|id| !written[id.0]) {
                return Err(FrameGraphError::ReadBeforeWrite {
                    pass: pass.name.clone(),
                    transient: self.transients[id.0].name.clone(),
                });
            }
            for id in &pass.writes {
                written[id.0] = true;
            }
        }
        if let Some(index) = used.iter().position(|&used| !used) {
            return Err(FrameGraphError::Unused {
                transient: self.transients[index].name.clone(),
            });
        }
        Ok(())
    }

    /// The usages the texture of transient `id` is created with: `RENDER_ATTACHMENT` when a
    /// render pass writes it, `STORAGE_BINDING` when a compute pass writes it,
    /// `TEXTURE_BINDING` when a pass reads it, plus its [`TransientDesc::extra_usage`].
    ///
    /// # Panics
    /// When `id` does not belong to this declaration.
    pub fn usage(&self, id: TransientId) -> wgpu::TextureUsages {
        let mut usage = self.transients[id.0].extra_usage;
        for pass in &self.passes {
            if pass.writes.contains(&id) {
                usage |= match pass.kind {
                    PassKind::Render => wgpu::TextureUsages::RENDER_ATTACHMENT,
                    PassKind::Compute => wgpu::TextureUsages::STORAGE_BINDING,
                };
            }
            if pass.reads.contains(&id) {
                usage |= wgpu::TextureUsages::TEXTURE_BINDING;
            }
        }
        usage
    }
}

fn unique_names<'a>(
    what: &'static str,
    names: impl Iterator<Item = &'a str>,
) -> Result<(), FrameGraphError> {
    let mut seen = std::collections::BTreeSet::new();
    for name in names {
        if !seen.insert(name) {
            return Err(FrameGraphError::DuplicateName {
                what,
                name: name.to_owned(),
            });
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn color(name: &str) -> TransientDesc {
        TransientDesc::new(name, wgpu::TextureFormat::Rgba16Float, TransientSize::FULL)
    }

    #[test]
    fn sizes_resolve_relative_to_the_output_and_never_reach_zero() {
        assert_eq!(TransientSize::FULL.resolve((1920, 1080)), (1920, 1080));
        assert_eq!(TransientSize::HALF.resolve((1920, 1080)), (960, 540));
        assert_eq!(TransientSize::HALF.resolve((1919, 1079)), (960, 540));
        assert_eq!(TransientSize::QUARTER.resolve((1, 1)), (1, 1));
        let two_thirds = TransientSize::Relative {
            numerator: 2,
            denominator: 3,
        };
        assert_eq!(two_thirds.resolve((100, 10)), (67, 7));
        assert_eq!(TransientSize::FULL.resolve((0, 0)), (1, 1));
        let fixed = TransientSize::Fixed {
            width: 2048,
            height: 512,
        };
        assert_eq!(fixed.resolve((1, 1)), (2048, 512));
        let huge = TransientSize::Relative {
            numerator: u32::MAX,
            denominator: 1,
        };
        assert_eq!(huge.resolve((u32::MAX, 2)), (u32::MAX, u32::MAX));
    }

    #[test]
    fn usages_follow_the_declarations() {
        let mut graph = FrameGraphDesc::new();
        let hdr = graph.transient(color("hdr"));
        let bloom = graph.transient(color("bloom").with_usage(wgpu::TextureUsages::COPY_SRC));
        graph.render_pass("scene", &[], &[hdr]);
        graph.compute_pass("bloom", &[hdr], &[bloom]);
        graph.render_pass("compose", &[hdr, bloom], &[]);
        graph.validate().unwrap();
        assert_eq!(
            graph.usage(hdr),
            wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING
        );
        assert_eq!(
            graph.usage(bloom),
            wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC
        );
        assert_eq!(graph.passes()[1].kind, PassKind::Compute);
        assert_eq!(graph.transients()[1].name, "bloom");
        assert_eq!((hdr.index(), bloom.index()), (0, 1));
    }

    #[test]
    fn reads_need_an_earlier_write() {
        let mut graph = FrameGraphDesc::new();
        let a = graph.transient(color("a"));
        graph.render_pass("reader", &[a], &[]);
        graph.render_pass("writer", &[], &[a]);
        assert_eq!(
            graph.validate(),
            Err(FrameGraphError::ReadBeforeWrite {
                pass: "reader".to_owned(),
                transient: "a".to_owned()
            })
        );
        // A pass that changes a texture in place needs an earlier writer too.
        let mut graph = FrameGraphDesc::new();
        let a = graph.transient(color("a"));
        graph.compute_pass("in place", &[a], &[a]);
        assert!(matches!(
            graph.validate(),
            Err(FrameGraphError::ReadBeforeWrite { .. })
        ));
        let mut graph = FrameGraphDesc::new();
        let a = graph.transient(color("a"));
        graph.render_pass("writer", &[], &[a]);
        graph.compute_pass("in place", &[a], &[a]);
        assert_eq!(graph.validate(), Ok(()));
    }

    #[test]
    fn bad_declarations_are_reported() {
        let mut graph = FrameGraphDesc::new();
        let a = graph.transient(color("a"));
        graph.transient(color("a"));
        graph.render_pass("p", &[], &[a]);
        assert_eq!(
            graph.validate(),
            Err(FrameGraphError::DuplicateName {
                what: "transients",
                name: "a".to_owned()
            })
        );

        let mut graph = FrameGraphDesc::new();
        let a = graph.transient(color("a"));
        graph.render_pass("p", &[], &[a]);
        graph.render_pass("p", &[a], &[]);
        assert!(matches!(
            graph.validate(),
            Err(FrameGraphError::DuplicateName { what: "passes", .. })
        ));

        let mut graph = FrameGraphDesc::new();
        graph.transient(color("lonely"));
        assert_eq!(
            graph.validate(),
            Err(FrameGraphError::Unused {
                transient: "lonely".to_owned()
            })
        );

        let mut graph = FrameGraphDesc::new();
        graph.render_pass("p", &[], &[TransientId(3)]);
        assert_eq!(
            graph.validate(),
            Err(FrameGraphError::UnknownTransient {
                pass: "p".to_owned(),
                index: 3
            })
        );

        for size in [
            TransientSize::Relative {
                numerator: 1,
                denominator: 0,
            },
            TransientSize::Relative {
                numerator: 0,
                denominator: 1,
            },
            TransientSize::Fixed {
                width: 0,
                height: 4,
            },
        ] {
            let mut graph = FrameGraphDesc::new();
            let t = graph.transient(TransientDesc::new("t", wgpu::TextureFormat::R32Float, size));
            graph.compute_pass("p", &[], &[t]);
            assert!(
                matches!(graph.validate(), Err(FrameGraphError::InvalidSize { .. })),
                "{size:?}"
            );
        }

        let mut graph = FrameGraphDesc::new();
        let t = graph.transient(color("t").with_mip_levels(0));
        graph.render_pass("p", &[], &[t]);
        assert_eq!(
            graph.validate(),
            Err(FrameGraphError::NoMipLevels {
                transient: "t".to_owned()
            })
        );

        let mut graph = FrameGraphDesc::new();
        for index in 0..=MAX_PASSES {
            graph.compute_pass(format!("pass {index}"), &[], &[]);
        }
        assert_eq!(
            graph.validate(),
            Err(FrameGraphError::TooManyPasses {
                passes: MAX_PASSES + 1
            })
        );
        assert_eq!(MAX_PASSES, 2048);
    }
}
