//! Errors of the generator constructors.

/// Why a generator could not be built. Bad configuration is external input (later TOML), so
/// it is an error, never a panic (contract §2.6).
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum WorldgenError {
    /// A palette role names a state key that the lookup does not know.
    #[error("palette role `{role}`: block state `{key}` does not exist")]
    UnknownState {
        /// The palette role (`"grass"`, `"log"`, …).
        role: &'static str,
        /// The state key that was looked up.
        key: String,
    },
    /// A configuration value is out of its documented range.
    #[error("invalid generator configuration: {0}")]
    InvalidConfig(String),
}
