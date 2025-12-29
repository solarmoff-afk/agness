// Only compile the module if the feature is active
#[cfg(feature = "wgpu")]
pub mod wgpu;

/// Enumeration of available backends
/// [*] All variants exist regardless of enabled features to allow runtime selection logic
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendSelector {
    Wgpu,
}