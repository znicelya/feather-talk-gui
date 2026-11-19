//! Bounded protocol framing shared without an engine dependency.
use crate::*;
use serde::{Deserialize, Serialize};
use std::{io::BufRead, path::PathBuf};
pub const MAX_MESSAGE_BYTES: usize = 1024 * 1024;
pub fn protocol_error(message: impl std::fmt::Display) -> AppError {
    AppError::new(ErrorCode::Protocol, "worker_transport", message.to_string())
}
/// Limit excludes the newline. EOF with an incomplete record is an error.
pub fn read_frame<R: BufRead>(reader: &mut R) -> Result<Option<Vec<u8>>> {
    let mut frame = Vec::new();
    loop {
        let bytes = reader.fill_buf().map_err(protocol_error)?;
        if bytes.is_empty() {
            return if frame.is_empty() {
                Ok(None)
            } else {
                Err(protocol_error("Truncated record at EOF"))
            };
        }
        let end = bytes.iter().position(|b| *b == b'\n');
        let take = end.unwrap_or(bytes.len());
        if take > MAX_MESSAGE_BYTES - frame.len() {
            return Err(protocol_error("Message exceeds 1 MiB"));
        }
        frame.extend_from_slice(&bytes[..take]);
        reader.consume(take + usize::from(end.is_some()));
        if end.is_some() {
            return Ok(Some(frame));
        }
    }
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProjectBinding {
    pub project_id: ProjectId,
    pub directory: PathBuf,
}
/// A file bound to the verified component pin. Hashes are checked again by the worker.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RuntimeFile {
    pub path: PathBuf,
    pub sha256: String,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RuntimeComponent {
    /// Copied from the verified resource catalog; None is for universal assets.
    pub target: Option<TargetPlatform>,
    pub pin: ResourcePin,
    pub root: PathBuf,
    pub files: Vec<RuntimeFile>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WorkerContext {
    pub protocol_version: u32,
    /// Exactly one accepted project per worker session in protocol v1.
    pub projects: Vec<ProjectBinding>,
    pub target: TargetPlatform,
    pub components: Vec<RuntimeComponent>,
    pub assets: PathBuf,
    pub ffmpeg: PathBuf,
    pub ffprobe: PathBuf,
    pub probe_runner: PathBuf,
    pub cuda_root: Option<PathBuf>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WorkerBootstrap {
    pub protocol_version: u32,
    pub context: WorkerContext,
}

/// A verified Hello-only session. Discovery never binds a project or loads models.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WorkerDiscovery {
    pub protocol_version: u32,
    pub target: TargetPlatform,
    pub components: Vec<RuntimeComponent>,
    pub cuda_root: Option<PathBuf>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum WorkerPrelude {
    Job(WorkerBootstrap),
    Discovery(WorkerDiscovery),
}
