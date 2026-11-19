import { convertFileSrc, invoke } from "@tauri-apps/api/core";
import type {
  AppError, DeviceDescriptor, GenerateRequest, MetricsPage, MetricsPageRequest,
  NewProject, PathChoice, ProjectId, ProjectManifest, ProjectSummary, ResourceInventory,
  ResourceSelection, TargetPlatform, TaskId, TaskSnapshot, TrainConfig, TrainRequest,
} from "./contracts.ts";

export const api = {
  listProjects: () => invoke<ProjectSummary[]>("list_projects"),
  createProject: (project: NewProject) => invoke<ProjectSummary>("create_project", { project }),
  openProject: (directory: string) => invoke<ProjectSummary>("open_project", { directory }),
  getProject: (projectId: ProjectId) => invoke<ProjectManifest>("get_project", { projectId }),
  listDevices: () => invoke<DeviceDescriptor[]>("list_devices"),
  listResourceSelections: () => invoke<ResourceSelection[]>("list_resource_selections"),
  getResources: (target: TargetPlatform) => invoke<ResourceInventory>("get_resources", { target }),
  prepareResources: (selection: ResourceSelection) => invoke<TaskId>("prepare_resources", { selection }),
  importResource: (target: TargetPlatform, manifest: string) => invoke<TaskId>("import_resource", { target, manifest }),
  getTrainDefaults: () => invoke<TrainConfig>("get_train_defaults"),
  async startTraining(request: TrainRequest): Promise<TaskId> {
    if (!Number.isSafeInteger(request.config.seed) || request.config.seed < 0) {
      throw {
        code: "invalid_config",
        stage: "configuration",
        message: "随机种子必须是可精确表示的非负整数。",
        evidence: [],
        actions: ["请使用 0 到 9007199254740991 之间的整数。"],
        log_path: null,
      } satisfies AppError;
    }
    return invoke<TaskId>("start_training", { request });
  },
  startGeneration: (request: GenerateRequest) => invoke<TaskId>("start_generation", { request }),
  cancelTask: (taskId: TaskId) => invoke<void>("cancel_task", { taskId }),
  getTask: () => invoke<TaskSnapshot | null>("get_task"),
  readMetrics: (request: MetricsPageRequest) => invoke<MetricsPage>("read_metrics", { request }),
  selectPath: (kind: PathChoice) => invoke<string | null>("select_path", { kind }),
  async videoPreviewUrl(projectId: ProjectId, taskId: TaskId): Promise<string> {
    const path = await invoke<string>("allow_video_preview", { projectId, taskId });
    return convertFileSrc(path);
  },
  openOutputDirectory: (projectId: ProjectId, taskId: TaskId) =>
    invoke<void>("open_output_directory", { projectId, taskId }),
};
