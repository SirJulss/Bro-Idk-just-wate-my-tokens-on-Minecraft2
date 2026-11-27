//! A minimal logger for `cpe`: every record of the `log` crate that passes the filter becomes
//! one line on stderr, `[LEVEL target] message`.
//!
//! The engine's crates (targets `cp_*`, `cpe`) log at `info` and above by default, all other
//! crates (wgpu, naga, ...) at `warn` and above. `CP_LOG=<level>` ([`LOG_ENV`]) sets the
//! engine's level; other crates never get more verbose than `warn`, so a debug session is not
//! drowned in wgpu's own records.

use std::io::Write;
use std::sync::OnceLock;

use log::{LevelFilter, Log, Metadata, Record};

/// Environment variable with the engine's log level: `off`, `error`, `warn`, `info`, `debug`
/// or `trace` (any case).
pub const LOG_ENV: &str = "CP_LOG";

/// Which records reach stderr.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Filter {
    /// Level for the engine's own crates (`cp_*`, `cpe`).
    pub engine: LevelFilter,
    /// Level for every other crate.
    pub foreign: LevelFilter,
}

impl Filter {
    /// `info` for the engine, `warn` for everything else.
    pub const DEFAULT: Self = Self {
        engine: LevelFilter::Info,
        foreign: LevelFilter::Warn,
    };

    /// The filter for a `CP_LOG` value; `None` if it names no level.
    #[must_use]
    pub fn from_spec(spec: &str) -> Option<Self> {
        let engine = spec.trim().parse::<LevelFilter>().ok()?;
        Some(Self {
            engine,
            foreign: engine.min(LevelFilter::Warn),
        })
    }

    /// The most verbose level that a record with this target may have.
    #[must_use]
    pub fn level_for(&self, target: &str) -> LevelFilter {
        if is_engine_target(target) {
            self.engine
        } else {
            self.foreign
        }
    }

    /// The most verbose level of any target.
    #[must_use]
    pub fn max_level(&self) -> LevelFilter {
        self.engine.max(self.foreign)
    }
}

/// True for the targets of the engine's crates: module paths such as `cp_render::gpu::context`
/// and the binary `cpe`.
fn is_engine_target(target: &str) -> bool {
    let krate = target.split("::").next().unwrap_or(target);
    krate.starts_with("cp_") || krate == "cpe"
}

/// The process-wide logger; its filter is set once by [`install`].
struct StderrLogger {
    filter: OnceLock<Filter>,
}

static LOGGER: StderrLogger = StderrLogger {
    filter: OnceLock::new(),
};

impl StderrLogger {
    fn filter(&self) -> Filter {
        self.filter.get().copied().unwrap_or(Filter::DEFAULT)
    }
}

impl Log for StderrLogger {
    fn enabled(&self, metadata: &Metadata<'_>) -> bool {
        metadata.level() <= self.filter().level_for(metadata.target())
    }

    fn log(&self, record: &Record<'_>) {
        if self.enabled(record.metadata()) {
            // One write per record keeps lines of several threads apart. A closed stderr is no
            // reason to fail the command that logs.
            let line = format!(
                "[{} {}] {}\n",
                record.level(),
                record.target(),
                record.args()
            );
            let _ = std::io::stderr().lock().write_all(line.as_bytes());
        }
    }

    fn flush(&self) {
        let _ = std::io::stderr().flush();
    }
}

/// Installs the logger for this process with `filter`.
///
/// Returns `false` when a logger was installed before; the first one stays in place, with
/// its filter.
pub fn install(filter: Filter) -> bool {
    if LOGGER.filter.set(filter).is_err() || log::set_logger(&LOGGER).is_err() {
        return false;
    }
    log::set_max_level(filter.max_level());
    true
}

/// Installs the logger with the filter from [`LOG_ENV`], or [`Filter::DEFAULT`] when the
/// variable is unset or empty. An unreadable value is reported as a warning and the default
/// is used.
pub fn install_from_env() {
    let spec = std::env::var_os(LOG_ENV)
        .map(|value| value.to_string_lossy().into_owned())
        .filter(|value| !value.trim().is_empty());
    let parsed = spec.as_deref().map(Filter::from_spec);
    install(parsed.flatten().unwrap_or(Filter::DEFAULT));
    if let (Some(spec), Some(None)) = (&spec, parsed) {
        log::warn!(
            "{LOG_ENV}={spec:?} is not a log level (off, error, warn, info, debug, trace); \
             using info"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn levels_are_read_in_any_case() {
        assert_eq!(
            Filter::from_spec("debug"),
            Some(Filter {
                engine: LevelFilter::Debug,
                foreign: LevelFilter::Warn,
            })
        );
        assert_eq!(
            Filter::from_spec(" ERROR "),
            Some(Filter {
                engine: LevelFilter::Error,
                foreign: LevelFilter::Error,
            })
        );
        assert_eq!(
            Filter::from_spec("off").map(|filter| filter.max_level()),
            Some(LevelFilter::Off)
        );
        assert_eq!(Filter::from_spec("loud"), None);
        assert_eq!(Filter::from_spec(""), None);
    }

    #[test]
    fn engine_crates_and_other_crates_have_their_own_level() {
        let filter = Filter::from_spec("trace").expect("a level");
        for target in ["cp_render::gpu::context", "cp_content", "cpe"] {
            assert_eq!(filter.level_for(target), LevelFilter::Trace, "{target}");
        }
        for target in ["wgpu_hal::vulkan::instance", "naga::front", "cpx", "c"] {
            assert_eq!(filter.level_for(target), LevelFilter::Warn, "{target}");
        }
        assert_eq!(filter.max_level(), LevelFilter::Trace);
        assert_eq!(Filter::DEFAULT.level_for("cp_app"), LevelFilter::Info);
        assert_eq!(Filter::DEFAULT.level_for("wgpu_core"), LevelFilter::Warn);
    }
}
