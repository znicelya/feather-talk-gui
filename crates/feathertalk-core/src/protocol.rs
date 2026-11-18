use crate::*;
use serde::{Deserialize, Serialize};
use std::{
    path::PathBuf,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
};
pub const PROTOCOL_VERSION: u32 = 1;
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BatchMode {
    #[default]
    Auto,
    Fixed,
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TrainRequest {
    #[serde(default)]
    pub batch_mode: BatchMode,
    pub project_id: ProjectId,
    pub config: TrainConfig,
    pub device: DevicePolicy,
    pub resume: Option<ModelRef>,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GenerateRequest {
    pub project_id: ProjectId,
    pub audio_path: PathBuf,
    pub output_path: PathBuf,
    pub model_ref: ModelRef,
    pub device: DevicePolicy,
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(
    tag = "kind",
    content = "request",
    rename_all = "snake_case",
    deny_unknown_fields
)]
pub enum JobSpec {
    Train(TrainRequest),
    Generate(GenerateRequest),
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum JobKind {
    Train,
    Generate,
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum WorkerRequest {
    Hello { protocol_version: u32 },
    Start { task_id: TaskId, job: JobSpec },
    Cancel { task_id: TaskId },
    Shutdown,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RequestKind {
    Hello,
    Start,
    Cancel,
    Shutdown,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ComponentCapability {
    pub name: String,
    pub version: String,
    pub available: bool,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WorkerCapabilities {
    pub engine_version: String,
    pub components: Vec<ComponentCapability>,
    pub training: bool,
    pub generation: bool,
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LossScalars {
    #[serde(serialize_with = "serialize_finite_metric")]
    pub full: f64,
    #[serde(serialize_with = "serialize_finite_metric")]
    pub mouth: f64,
    #[serde(serialize_with = "serialize_finite_metric")]
    pub temporal: f64,
    #[serde(serialize_with = "serialize_finite_metric")]
    pub temporal_mouth: f64,
    #[serde(serialize_with = "serialize_finite_metric")]
    pub perceptual: f64,
    #[serde(serialize_with = "serialize_finite_metric")]
    pub mouth_gradient: f64,
    #[serde(serialize_with = "serialize_finite_metric")]
    pub total: f64,
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TrainingMetrics {
    pub project_id: ProjectId,
    pub run_id: TrainingRunId,
    pub step: u64,
    pub epoch: u32,
    pub elapsed_ms: u64,
    #[serde(serialize_with = "serialize_finite_metric")]
    pub lr: f64,
    pub losses: LossScalars,
    pub device: DeviceDescriptor,
    pub precision: Precision,
    pub batch_size: u32,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExportResult {
    pub output_path: PathBuf,
    pub video_frames: u64,
    pub elapsed_ms: u64,
    pub device: DeviceDescriptor,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(
    tag = "kind",
    content = "artifact",
    rename_all = "snake_case",
    deny_unknown_fields
)]
pub enum ArtifactResult {
    Model(ModelRef),
    Video(ExportResult),
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum EventPayload {
    RequestRejected {
        request: RequestKind,
        requested_task_id: Option<TaskId>,
        error: AppError,
    },
    Ready {
        capabilities: WorkerCapabilities,
        devices: Vec<DeviceDescriptor>,
    },
    Started {
        job: JobKind,
    },
    Progress {
        stage: String,
        completed: u64,
        total: Option<u64>,
    },
    Metrics(TrainingMetrics),
    CheckpointCommitted {
        model_ref: ModelRef,
    },
    Completed {
        result: ArtifactResult,
    },
    Cancelled {
        checkpoint: Option<ModelRef>,
    },
    Failed {
        error: AppError,
    },
}
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(try_from = "WireEvent")]
pub struct WorkerEvent {
    pub protocol_version: u32,
    pub task_id: Option<TaskId>,
    pub seq: u64,
    pub payload: EventPayload,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct WireEvent {
    protocol_version: u32,
    task_id: Option<TaskId>,
    seq: u64,
    payload: EventPayload,
}
impl TryFrom<WireEvent> for WorkerEvent {
    type Error = AppError;
    fn try_from(w: WireEvent) -> Result<Self> {
        let e = Self {
            protocol_version: w.protocol_version,
            task_id: w.task_id,
            seq: w.seq,
            payload: w.payload,
        };
        e.validate()?;
        Ok(e)
    }
}
impl WorkerEvent {
    pub fn new(task_id: Option<TaskId>, seq: u64, payload: EventPayload) -> Result<Self> {
        WireEvent {
            protocol_version: PROTOCOL_VERSION,
            task_id,
            seq,
            payload,
        }
        .try_into()
    }
    pub fn validate(&self) -> Result<()> {
        if self.protocol_version != PROTOCOL_VERSION {
            return Err(AppError::new(
                ErrorCode::Protocol,
                "worker_event",
                "Unsupported protocol version",
            ));
        }
        if matches!(
            &self.payload,
            EventPayload::Ready { .. } | EventPayload::RequestRejected { .. }
        ) != self.task_id.is_none()
        {
            return Err(AppError::new(
                ErrorCode::Protocol,
                "worker_event",
                "Control responses must have no task ID; task events require one",
            ));
        }
        if let EventPayload::RequestRejected {
            request,
            requested_task_id,
            ..
        } = &self.payload
        {
            if matches!(request, RequestKind::Start | RequestKind::Cancel)
                != requested_task_id.is_some()
            {
                return Err(crate::worker::protocol_error(
                    "Rejected request ID does not match its kind",
                ));
            }
        }
        if let EventPayload::Metrics(metrics) = &self.payload {
            metrics.validate()?;
        }
        Ok(())
    }
}
// serde_json otherwise silently serializes NaN and infinities as null.
// Apply this guard to the scalars themselves so pages and bare payloads are safe too.
fn serialize_finite_metric<S: serde::Serializer>(
    value: &f64,
    serializer: S,
) -> std::result::Result<S::Ok, S::Error> {
    validate_finite_metric("metric scalar", *value).map_err(serde::ser::Error::custom)?;
    serializer.serialize_f64(*value)
}
fn validate_finite_metric(field: &str, value: f64) -> Result<()> {
    if !value.is_finite() {
        return Err(AppError::new(
            ErrorCode::Protocol,
            "metrics",
            format!("{field} must be finite"),
        ));
    }
    Ok(())
}
impl LossScalars {
    pub fn validate(&self) -> Result<()> {
        for (field, value) in [
            ("full", self.full),
            ("mouth", self.mouth),
            ("temporal", self.temporal),
            ("temporal_mouth", self.temporal_mouth),
            ("perceptual", self.perceptual),
            ("mouth_gradient", self.mouth_gradient),
            ("total", self.total),
        ] {
            validate_finite_metric(field, value)?;
        }
        Ok(())
    }
}
impl TrainingMetrics {
    pub fn validate(&self) -> Result<()> {
        validate_finite_metric("lr", self.lr)?;
        self.losses.validate()
    }
}
/// Observer implementations attach the task ID and monotonically increasing sequence.
/// Transport/UI throttling belongs to the worker, not the engine.
pub trait TaskObserver: Send + Sync {
    fn emit(&self, payload: EventPayload) -> Result<()>;
}
#[derive(Debug, Default, Clone)]
pub struct Cancellation(Arc<AtomicBool>);
impl Cancellation {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn cancel(&self) {
        self.0.store(true, Ordering::Release);
    }
    pub fn is_cancelled(&self) -> bool {
        self.0.load(Ordering::Acquire)
    }
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TaskStatus {
    Queued,
    Running,
    Completed,
    Cancelled,
    Failed,
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TaskSnapshot {
    pub task_id: TaskId,
    pub project_id: ProjectId,
    pub job: JobKind,
    pub status: TaskStatus,
    pub last_seq: u64,
    pub latest_event: Option<WorkerEvent>,
    pub result: Option<ArtifactResult>,
    pub error: Option<AppError>,
}
/// Offset into the durable metrics stream, not an epoch or training step.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MetricsPageRequest {
    pub project_id: ProjectId,
    pub run_id: TrainingRunId,
    pub offset: u64,
    pub limit: u32,
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MetricsPage {
    pub project_id: ProjectId,
    pub run_id: TrainingRunId,
    pub items: Vec<TrainingMetrics>,
    pub next_offset: Option<u64>,
}

impl Serialize for WorkerEvent {
    fn serialize<S: serde::Serializer>(
        &self,
        serializer: S,
    ) -> std::result::Result<S::Ok, S::Error> {
        use serde::ser::SerializeStruct;
        self.validate().map_err(serde::ser::Error::custom)?;
        let mut state = serializer.serialize_struct("WorkerEvent", 4)?;
        state.serialize_field("protocol_version", &self.protocol_version)?;
        state.serialize_field("task_id", &self.task_id)?;
        state.serialize_field("seq", &self.seq)?;
        state.serialize_field("payload", &self.payload)?;
        state.end()
    }
}
