use serde::{Deserialize, Serialize};
use std::{fmt, path::PathBuf};
pub type Result<T> = std::result::Result<T, AppError>;
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ErrorCode {
    InvalidConfig,
    ModelMismatch,
    ResourceMissing,
    ResourceCorrupt,
    UnsupportedBackend,
    DeviceInit,
    DeviceLost,
    OutOfMemory,
    NoFace,
    NoAudio,
    EncodeFailed,
    DiskFull,
    PermissionDenied,
    Busy,
    Protocol,
    Cancelled,
    Internal,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AppError {
    pub code: ErrorCode,
    pub stage: String,
    pub message: String,
    pub evidence: Vec<String>,
    pub actions: Vec<String>,
    pub log_path: Option<PathBuf>,
}
impl AppError {
    pub fn new(code: ErrorCode, stage: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            code,
            stage: stage.into(),
            message: message.into(),
            evidence: vec![],
            actions: vec![],
            log_path: None,
        }
    }
}
impl fmt::Display for AppError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.stage, self.message)
    }
}
impl std::error::Error for AppError {}
