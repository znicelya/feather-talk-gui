//! Synchronization only for explicitly instrumented, debug acceptance workers.
use feathertalk_core::{AppError, Cancellation, ErrorCode, Result};
use std::{
    fs,
    path::Path,
    thread,
    time::{Duration, Instant},
};

fn wait_for(condition: impl Fn() -> bool, phase: &str) -> Result<()> {
    let deadline = Instant::now() + Duration::from_secs(30);
    while !condition() {
        if Instant::now() >= deadline {
            return Err(error(format!("Timeout at {phase}")));
        }
        thread::sleep(Duration::from_millis(10));
    }
    Ok(())
}

fn error(e: impl std::fmt::Display) -> AppError {
    AppError::new(ErrorCode::Internal, "worker-acceptance", e.to_string())
}

pub(super) fn after_forward(root: &Path, cancel: &Cancellation) -> Result<()> {
    let directory = root.join(".worker-acceptance");
    if directory.join("cancel-after-forward").is_file() {
        fs::write(directory.join("forward-complete"), []).map_err(error)?;
        wait_for(
            || cancel.is_cancelled(),
            "awaiting worker Cancel during optimizer step",
        )?;
        fs::write(directory.join("cancel-observed"), []).map_err(error)?;
        wait_for(
            || directory.join("release-forward").is_file(),
            "awaiting test release",
        )?;
    }
    Ok(())
}

pub(super) fn after_checkpoint_event(root: &Path) -> Result<()> {
    let directory = root.join(".worker-acceptance");
    if directory.join("hold-checkpoint").is_file() {
        fs::write(directory.join("checkpoint-held"), []).map_err(error)?;
        // The harness normally terminates this owner. A failed harness must not hang it.
        wait_for(
            || directory.join("release-checkpoint").is_file(),
            "holding committed checkpoint",
        )?;
    }
    Ok(())
}
