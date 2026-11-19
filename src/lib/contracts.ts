export type ProjectId = string;
export type TrainingRunId = string;
export type TaskId = string;

export type ErrorCode =
  | "invalid_config" | "model_mismatch" | "resource_missing" | "resource_corrupt"
  | "unsupported_backend" | "device_init" | "device_lost" | "out_of_memory"
  | "no_face" | "no_audio" | "encode_failed" | "disk_full" | "permission_denied"
  | "busy" | "protocol" | "cancelled" | "internal";

export interface AppError {
  code: ErrorCode;
  stage: string;
  message: string;
  evidence: string[];
  actions: string[];
  log_path: string | null;
}

export type BackendKind = "flex" | "cpu" | "vulkan" | "cuda" | "metal";
export type Precision = "f32" | "f16" | "bf16";
export type DevicePolicy =
  | { policy: "auto" }
  | { policy: "explicit"; request: { backend: BackendKind; device_id: string } };

export interface DeviceDescriptor {
  id: string;
  name: string;
  vendor: string | null;
  driver: string | null;
  backend: BackendKind;
  memory_bytes: number | null;
}

export interface LossConfig {
  full: number;
  mouth: number;
  temporal: number;
  temporal_mouth: number;
  perceptual: number;
  mouth_gradient: number;
  temporal_kernel: number;
  temporal_stride: number;
}

export interface TrainConfig {
  /** JavaScript submissions must use an exact, nonnegative integer. */
  seed: number;
  prefetch_capacity: number;
  epochs: number;
  batch_size: number;
  lr: number;
  beta1: number;
  beta2: number;
  epsilon: number;
  weight_decay: number;
  save_every: number;
  loss: LossConfig;
}

export interface ModelIdentity {
  kind: string;
  architecture_version: string;
  source_sha256: string;
  weights_sha256: string;
}

export interface ModelRef {
  project_id: ProjectId;
  run_id: TrainingRunId;
  checkpoint_id: string;
}

export interface NewProject {
  name: string;
  directory: string;
  source_video: string;
}

export interface ProjectSummary extends NewProject {
  id: ProjectId;
  latest_model: ModelRef | null;
}

export interface CheckpointEntry {
  model_ref: ModelRef;
  relative_path: string;
  sha256: string;
  size_bytes: number;
  epoch: number;
  step: number;
  config: TrainConfig;
  identity: ModelIdentity;
}

export interface ProjectManifest {
  schema_version: number;
  id: ProjectId;
  name: string;
  source_video: { path: string; sha256: string; size_bytes: number };
  preprocessing: {
    version: string;
    source_sha256: string;
    config_sha256: string;
    cache_directory: string;
    component_identities: ModelIdentity[];
  } | null;
  model_identity: ModelIdentity | null;
  checkpoints: CheckpointEntry[];
  metrics_path: string;
}

export interface TargetPlatform {
  os: "windows" | "linux" | "macos";
  architecture: "x86_64" | "aarch64";
  backend: Exclude<BackendKind, "flex">;
}

export interface ResourcePin {
  id: string;
  version: string;
}

export interface ResourceSelection {
  target: TargetPlatform;
  resources: ResourcePin[];
}

export interface ResourceInventory {
  ready: boolean;
  items: {
    resource: ResourcePin;
    identity: ModelIdentity | null;
    path: string | null;
    ready: boolean;
    reason: string | null;
  }[];
}

export interface TrainRequest {
  batch_mode: "auto" | "fixed";
  project_id: ProjectId;
  config: TrainConfig;
  device: DevicePolicy;
  resume: ModelRef | null;
}

export interface GenerateRequest {
  project_id: ProjectId;
  audio_path: string;
  output_path: string;
  model_ref: ModelRef;
  device: DevicePolicy;
}

export type JobKind = "train" | "generate" | "prepare_resources";
export type RequestKind = "hello" | "start" | "cancel" | "shutdown";
export type TaskStatus = "queued" | "running" | "cancelling" | "completed" | "cancelled" | "failed";

export interface TrainingMetrics {
  project_id: ProjectId;
  run_id: TrainingRunId;
  step: number;
  epoch: number;
  elapsed_ms: number;
  lr: number;
  losses: Omit<LossConfig, "temporal_kernel" | "temporal_stride"> & { total: number };
  device: DeviceDescriptor;
  precision: Precision;
  batch_size: number;
}

export interface ExportResult {
  output_path: string;
  video_frames: number;
  elapsed_ms: number;
  device: DeviceDescriptor;
}

export type ArtifactResult =
  | { kind: "model"; artifact: ModelRef }
  | { kind: "video"; artifact: ExportResult }
  | { kind: "resources"; artifact: ResourceInventory };

export interface TaskProgress {
  stage: string;
  completed: number;
  total: number | null;
}

export type TaskPayload =
  | { type: "started"; job: JobKind }
  | ({ type: "progress" } & TaskProgress)
  | ({ type: "metrics" } & TrainingMetrics)
  | { type: "checkpoint_committed"; model_ref: ModelRef }
  | { type: "completed"; result: ArtifactResult }
  | { type: "cancelled"; checkpoint: ModelRef | null }
  | { type: "failed"; error: AppError };

export type ControlPayload =
  | {
      type: "ready";
      capabilities: {
        engine_version: string;
        components: { name: string; version: string; available: boolean }[];
        training: boolean;
        generation: boolean;
      };
      devices: DeviceDescriptor[];
    }
  | { type: "request_rejected"; request: RequestKind; requested_task_id: TaskId | null; error: AppError };

export interface TaskEvent {
  protocol_version: 1;
  task_id: TaskId;
  seq: number;
  payload: TaskPayload;
}

export type WorkerEvent = TaskEvent | {
  protocol_version: 1;
  task_id: null;
  seq: number;
  payload: ControlPayload;
};

export interface TaskSnapshot {
  task_id: TaskId;
  project_id: ProjectId | null;
  job: JobKind;
  run_id: TrainingRunId | null;
  status: TaskStatus;
  last_seq: number;
  latest_event: TaskEvent | null;
  result: ArtifactResult | null;
  error: AppError | null;
  progress: TaskProgress | null;
  metrics: TrainingMetrics | null;
  checkpoint: ModelRef | null;
}

export interface MetricsPageRequest {
  project_id: ProjectId;
  run_id: TrainingRunId;
  offset: number;
  limit: number;
}

export interface MetricsPage {
  project_id: ProjectId;
  run_id: TrainingRunId;
  items: TrainingMetrics[];
  next_offset: number | null;
}

export type PathChoice = "project_directory" | "source_video" | "driving_audio" | "output_video" | "resource_manifest";
