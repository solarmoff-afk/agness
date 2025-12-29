use thiserror::Error;

/// Errors occurring during backend operation
#[derive(Error, Debug)]
pub enum BackendError {
    #[error("Initialization failed")]
    InitializationFailed,

    #[error("Resource creation failed")]
    ResourceCreation,

    #[error("Validation error")]
    Validation,

    #[error("Device lost")]
    DeviceLost,

    #[error("Out of memory")]
    OutOfMemory,
}