//! Durable sampled history. Offsets count records, never optimizer steps.
use super::{losses::LossParts, state::TrainingState};
use crate::ExecutionDevice;
use feathertalk_core::*;
use std::{
    fs::{File, OpenOptions},
    io::{BufRead, BufReader, Write},
    path::{Path, PathBuf},
};
const MAX_LINE: u64 = 65536;
const MAX_PAGE: u32 = 1000;
fn error(e: impl std::fmt::Display) -> AppError {
    AppError::new(
        ErrorCode::ResourceCorrupt,
        "training-metrics-after-commit",
        e.to_string(),
    )
}
fn path(root: &Path) -> Result<PathBuf> {
    let root = std::fs::canonicalize(root).map_err(error)?;
    let relative = super::checkpoint::read_project(&root)?.metrics_path;
    if relative.as_os_str().is_empty()
        || relative
            .components()
            .any(|c| !matches!(c, std::path::Component::Normal(_)))
    {
        return Err(error("Unsafe metrics path"));
    }
    let destination = root.join(relative);
    let parent = std::fs::canonicalize(
        destination
            .parent()
            .ok_or_else(|| error("Missing metrics parent"))?,
    )
    .map_err(error)?;
    if !parent.starts_with(&root) {
        return Err(error("Metrics path escapes project"));
    }
    if destination.exists()
        && !std::fs::canonicalize(&destination)
            .map_err(error)?
            .starts_with(&root)
    {
        return Err(error("Metrics file escapes project"));
    }
    Ok(destination)
}
fn record(reader: &mut BufReader<File>) -> Result<Option<TrainingMetrics>> {
    use std::io::Read;
    let mut bytes = Vec::new();
    let n = reader
        .take(MAX_LINE + 1)
        .read_until(b'\n', &mut bytes)
        .map_err(error)?;
    if n == 0 {
        return Ok(None);
    }
    if n as u64 > MAX_LINE {
        return Err(error("Metrics record exceeds 64 KiB"));
    }
    if bytes.last() != Some(&b'\n') {
        return Err(error(
            "Interrupted trailing metrics record; repair required before append",
        ));
    }
    let value: TrainingMetrics = serde_json::from_slice(&bytes).map_err(error)?;
    value.validate()?;
    Ok(Some(value))
}
pub fn metrics_page(root: &Path, request: &MetricsPageRequest) -> Result<MetricsPage> {
    if request.limit == 0 || request.limit > MAX_PAGE {
        return Err(error("Metrics page limit must be 1..=1000"));
    }
    if super::checkpoint::read_project(root)?.id != request.project_id {
        return Err(error("Wrong project root"));
    }
    let mut reader = BufReader::new(File::open(path(root)?).map_err(error)?);
    let mut offset = 0u64;
    let mut items = Vec::new();
    let mut more = false;
    while let Some(m) = record(&mut reader)? {
        if m.project_id != request.project_id {
            return Err(error("Metrics project mismatch"));
        }
        if m.run_id != request.run_id {
            continue;
        }
        if offset >= request.offset {
            if items.len() == request.limit as usize {
                more = true;
                break;
            }
            items.push(m);
        }
        offset = offset
            .checked_add(1)
            .ok_or_else(|| error("Metrics offset overflow"))?;
    }
    Ok(MetricsPage {
        project_id: request.project_id,
        run_id: request.run_id,
        next_offset: more.then_some(offset),
        items,
    })
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
        if destination.exists()
            && std::fs::symlink_metadata(&destination)
                .map_err(error)?
                .file_type()
                .is_symlink()
        {
            return Err(error("Metrics symlink rejected"));
        }
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
        while let Some(m) = record(&mut reader)? {
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
        if bytes.len() as u64 > MAX_LINE {
            return Err(error("Metrics record exceeds 64 KiB"));
        }
        // Reopen rejects a partial tail after any interrupted append.
        self.file.write_all(&bytes).map_err(error)?;
        self.file.sync_data().map_err(error)?;
        Ok(Some(m))
    }
}
