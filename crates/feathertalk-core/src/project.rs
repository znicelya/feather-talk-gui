use crate::{AppError, ErrorCode, ModelIdentity, Result, TrainConfig};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use uuid::Uuid;
macro_rules! id_type {
    ($name:ident) => {
        #[derive(
            Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
        )]
        #[serde(transparent)]
        pub struct $name(pub Uuid);
        impl $name {
            pub fn new() -> Self {
                Self(Uuid::new_v4())
            }
        }
        impl Default for $name {
            fn default() -> Self {
                Self::new()
            }
        }
        impl std::fmt::Display for $name {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                self.0.fmt(f)
            }
        }
        impl std::str::FromStr for $name {
            type Err = uuid::Error;
            fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
                Uuid::parse_str(s).map(Self)
            }
        }
    };
}
id_type!(TaskId);
id_type!(ProjectId);
id_type!(TrainingRunId);
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NewProject {
    pub name: String,
    pub directory: PathBuf,
    pub source_video: PathBuf,
}
impl NewProject {
    pub fn validate(&self) -> Result<()> {
        if self.name.trim().is_empty()
            || self.directory.as_os_str().is_empty()
            || self.source_video.as_os_str().is_empty()
        {
            return Err(AppError::new(
                ErrorCode::InvalidConfig,
                "project",
                "Name, directory and one source video are required",
            ));
        }
        Ok(())
    }
}
/// Reference resolved only through a project's committed checkpoint index.
/// It deliberately carries no arbitrary checkpoint file path.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelRef {
    pub project_id: ProjectId,
    pub run_id: TrainingRunId,
    pub checkpoint_id: String,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SourceVideo {
    pub path: PathBuf,
    pub sha256: String,
    pub size_bytes: u64,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PreprocessingIdentity {
    pub version: String,
    pub source_sha256: String,
    pub config_sha256: String,
    pub cache_directory: PathBuf,
    pub component_identities: Vec<ModelIdentity>,
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CheckpointEntry {
    pub model_ref: ModelRef,
    pub relative_path: PathBuf,
    pub sha256: String,
    pub size_bytes: u64,
    pub epoch: u32,
    pub step: u64,
    pub config: TrainConfig,
    pub identity: ModelIdentity,
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProjectManifest {
    pub schema_version: u32,
    pub id: ProjectId,
    pub name: String,
    pub source_video: SourceVideo,
    pub preprocessing: Option<PreprocessingIdentity>,
    pub model_identity: Option<ModelIdentity>,
    /// Only atomically committed checkpoints belong in this index.
    pub checkpoints: Vec<CheckpointEntry>,
    /// Project-relative durable metrics/curve store.
    pub metrics_path: PathBuf,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProjectSummary {
    pub id: ProjectId,
    pub name: String,
    pub directory: PathBuf,
    pub source_video: PathBuf,
    pub latest_model: Option<ModelRef>,
}
