//! Bounded storage reader shared by the engine and the desktop host.
use crate::*;
use std::{
    fs::{self, File},
    io::{self, BufRead, BufReader, Read},
    path::{Component, Path, PathBuf},
};

pub const MAX_METRIC_RECORD_BYTES: u64 = 65_536;
const MAX_PAGE: u32 = 1000;

fn error(e: impl std::fmt::Display) -> AppError {
    AppError::new(
        ErrorCode::ResourceCorrupt,
        "training-metrics-after-commit",
        e.to_string(),
    )
}

/// Resolve an owned history file, including the not-yet-created first history.
pub fn metrics_path(root: &Path, relative: &Path) -> Result<PathBuf> {
    let root = root.canonicalize().map_err(error)?;
    if relative.as_os_str().is_empty()
        || relative
            .components()
            .any(|c| !matches!(c, Component::Normal(_)))
    {
        return Err(error("Unsafe metrics path"));
    }
    let destination = root.join(relative);
    let parent = destination
        .parent()
        .ok_or_else(|| error("Missing metrics parent"))?
        .canonicalize()
        .map_err(error)?;
    if !parent.starts_with(&root) {
        return Err(error("Metrics path escapes project"));
    }
    match fs::symlink_metadata(&destination) {
        Ok(meta) if !meta.is_file() || meta.file_type().is_symlink() => {
            return Err(error("Metrics must be a project-owned regular file"));
        }
        Ok(_)
            if !destination
                .canonicalize()
                .map_err(error)?
                .starts_with(&root) =>
        {
            return Err(error("Metrics file escapes project"));
        }
        Ok(_) => {}
        Err(e) if e.kind() == io::ErrorKind::NotFound => {}
        Err(e) => return Err(error(e)),
    }
    Ok(destination)
}

pub fn read_metric_record(reader: &mut impl BufRead) -> Result<Option<TrainingMetrics>> {
    let mut bytes = Vec::new();
    let n = reader
        .take(MAX_METRIC_RECORD_BYTES + 1)
        .read_until(b'\n', &mut bytes)
        .map_err(error)?;
    if n == 0 {
        return Ok(None);
    }
    if n as u64 > MAX_METRIC_RECORD_BYTES {
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
    let root = root.canonicalize().map_err(error)?;
    let manifest = root.join("project.json").canonicalize().map_err(error)?;
    if !manifest.starts_with(&root) {
        return Err(error("Project manifest escapes project"));
    }
    let project: ProjectManifest =
        serde_json::from_reader(File::open(manifest).map_err(error)?).map_err(error)?;
    if project.schema_version != 1 || project.id != request.project_id {
        return Err(error("Wrong project root"));
    }
    let mut page = MetricsPage {
        project_id: request.project_id,
        run_id: request.run_id,
        items: vec![],
        next_offset: None,
    };
    let file = match File::open(metrics_path(&root, &project.metrics_path)?) {
        Ok(file) => file,
        Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(page),
        Err(e) => return Err(error(e)),
    };
    let mut reader = BufReader::new(file);
    let mut offset = 0u64;
    while let Some(m) = read_metric_record(&mut reader)? {
        if m.project_id != request.project_id {
            return Err(error("Metrics project mismatch"));
        }
        if m.run_id != request.run_id {
            continue;
        }
        if offset >= request.offset {
            if page.items.len() == request.limit as usize {
                page.next_offset = Some(offset);
                break;
            }
            page.items.push(m);
        }
        offset = offset
            .checked_add(1)
            .ok_or_else(|| error("Metrics offset overflow"))?;
    }
    Ok(page)
}
