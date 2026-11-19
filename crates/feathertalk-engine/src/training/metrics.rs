//! Durable sampled history. Offsets count records, never optimizer steps.
use super::{losses::LossParts, state::TrainingState};
use crate::ExecutionDevice;
pub use feathertalk_core::history::metrics_page;
use feathertalk_core::history::{MAX_METRIC_RECORD_BYTES, metrics_path, read_metric_record};
use feathertalk_core::*;
use std::{
    fs::{File, OpenOptions},
    io::{BufReader, Write},
    path::{Path, PathBuf},
};
#[cfg(test)]
thread_local! {pub(super) static READ_ONLY_WRITE:std::cell::Cell<bool>=const {std::cell::Cell::new(false)};}
fn error(e: impl std::fmt::Display) -> AppError {
    AppError::new(
        ErrorCode::ResourceCorrupt,
        "training-metrics-after-commit",
        e.to_string(),
    )
}
fn path(root: &Path) -> Result<PathBuf> {
    let root = std::fs::canonicalize(root).map_err(error)?;
    metrics_path(&root, &super::checkpoint::read_project(&root)?.metrics_path)
}
/// One writer per project history (independent readers remain available). elapsed_ms is cumulative monotonic run time, also after
/// restart. Open scans bounded records (constant memory) and rejects torn tails.
pub struct SampledMetrics {
    _lock: File,
    file: File,
    project: ProjectId,
    run: TrainingRunId,
    last_ms: Option<u64>,
    last_step: Option<u64>,
}
impl SampledMetrics {
    pub fn open(root: &Path, project: ProjectId, run: TrainingRunId) -> Result<Self> {
        if super::checkpoint::read_project(root)?.id != project {
            return Err(error("Wrong project root"));
        }
        let destination = path(root)?;
        // Lock a sidecar: Windows exclusive file locks otherwise prevent UI
        // readers from reading the history while the training writer is alive.
        let lock = OpenOptions::new()
            .create(true)
            .truncate(false)
            .read(true)
            .write(true)
            .open(
                std::fs::canonicalize(root)
                    .map_err(error)?
                    .join(".metrics-write.lock"),
            )
            .map_err(error)?;
        lock.try_lock().map_err(error)?;
        let file = OpenOptions::new()
            .create(true)
            .read(true)
            .append(true)
            .open(destination)
            .map_err(error)?;
        let mut reader = BufReader::new(file.try_clone().map_err(error)?);
        let mut last_ms: Option<u64> = None;
        let mut last_step = None;
        while let Some(m) = read_metric_record(&mut reader)? {
            if m.project_id != project {
                return Err(error("Metrics project mismatch"));
            }
            if m.run_id != run {
                continue;
            }
            if last_ms.is_some_and(|t| m.elapsed_ms < t.saturating_add(500))
                || last_step.is_some_and(|s| m.step <= s)
            {
                return Err(error("Nonmonotonic sampled history"));
            }
            last_ms = Some(m.elapsed_ms);
            last_step = Some(m.step);
        }
        Ok(Self {
            _lock: lock,
            file,
            project,
            run,
            last_ms,
            last_step,
        })
    }
    /// Resume the cumulative run clock from this value plus new wall time.
    pub fn elapsed_base_ms(&self) -> u64 {
        self.last_ms.unwrap_or(0)
    }
    /// Call only after complete_step. A write error does not undo that commit.
    /// Suppressed samples perform no loss readback and emit nothing.
    pub fn sample(
        &mut self,
        state: &TrainingState,
        parts: &LossParts,
        device: &ExecutionDevice,
        elapsed_ms: u64,
        batch_size: u32,
    ) -> Result<Option<TrainingMetrics>> {
        if state.metadata.global_step == 0
            || batch_size == 0
            || batch_size > state.metadata.actual_batch
        {
            return Err(error(
                "Metrics require a committed step and its actual nonempty batch",
            ));
        }
        if state.metadata.project_id != self.project || state.metadata.run_id != self.run {
            return Err(error("Metrics state identity mismatch"));
        }
        if self
            .last_ms
            .is_some_and(|t| elapsed_ms.saturating_sub(t) < 500)
            || self
                .last_step
                .is_some_and(|s| state.metadata.global_step <= s)
        {
            return Ok(None);
        }
        // Count attempted readbacks too, including validation/I/O failures.
        self.last_ms = Some(elapsed_ms);
        self.last_step = Some(state.metadata.global_step);
        let m = TrainingMetrics {
            project_id: self.project,
            run_id: self.run,
            step: state.metadata.global_step,
            epoch: state.metadata.epoch,
            elapsed_ms,
            lr: state.metadata.config.lr,
            losses: parts.to_scalars(),
            device: device.descriptor.clone(),
            precision: Precision::F32,
            batch_size,
        };
        m.validate()?;
        let mut bytes = serde_json::to_vec(&m).map_err(error)?;
        bytes.push(b'\n');
        if bytes.len() as u64 > MAX_METRIC_RECORD_BYTES {
            return Err(error("Metrics record exceeds 64 KiB"));
        }
        // Reopen rejects a partial tail after any interrupted append.
        #[cfg(test)]
        if READ_ONLY_WRITE.with(|flag| flag.replace(false)) {
            self.file = File::open(path(state.project_directory())?).map_err(error)?;
        }
        self.file.write_all(&bytes).map_err(error)?;
        self.file.sync_data().map_err(error)?;
        Ok(Some(m))
    }
}
