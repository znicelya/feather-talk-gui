mod devices;
mod jobs;
mod paths;
mod projects;
mod resources;

pub(crate) use {devices::*, jobs::*, paths::*, projects::*, resources::*};

use feathertalk_core::{AppError, ErrorCode, Result};

pub(super) async fn blocking<T: Send + 'static>(
    work: impl FnOnce() -> Result<T> + Send + 'static,
) -> Result<T> {
    tauri::async_runtime::spawn_blocking(work)
        .await
        .map_err(|e| desktop_error("background", e))?
}

pub(super) fn desktop_error(stage: &str, error: impl std::fmt::Display) -> AppError {
    AppError::new(ErrorCode::Internal, stage, error.to_string())
}
