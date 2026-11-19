import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { listen } from "@tauri-apps/api/event";
import { beforeEach, expect, it, vi } from "vitest";
import App from "@/App";
import { api } from "@/lib/api";
import type {
  AppError, CheckpointEntry, ModelIdentity, ProjectManifest, ProjectSummary, ResourceSelection,
  TaskEvent, TaskSnapshot, TrainConfig, TrainingMetrics,
} from "@/lib/contracts";

vi.mock("@/lib/api", () => ({ api: {
  listProjects: vi.fn(), getProject: vi.fn(), getTask: vi.fn(), listDevices: vi.fn(), cancelTask: vi.fn(),
  listResourceSelections: vi.fn(), getResources: vi.fn(), getTrainDefaults: vi.fn(), startTraining: vi.fn(), readMetrics: vi.fn(),
} }));
vi.mock("@tauri-apps/api/event", () => ({ listen: vi.fn() }));
// The test observes metrics and navigation; actual SVG sizing is checked in WebView2.
vi.mock("@/components/ui/chart", () => ({ ChartContainer: () => null, ChartTooltip: () => null, ChartTooltipContent: () => null }));

const config: TrainConfig = {
  seed: 42, prefetch_capacity: 2, epochs: 60, batch_size: 8, lr: 0.001, save_every: 5,
  beta1: 0.9, beta2: 0.999, epsilon: 1e-8, weight_decay: 0,
  loss: { full: 1, mouth: 4, temporal: 0.5, temporal_mouth: 4, perceptual: 0.01, mouth_gradient: 0.5,
    temporal_kernel: 5, temporal_stride: 1 },
};
const project: ProjectSummary = { id: "p", name: "训练人物", directory: "D:\\训练 人物", source_video: "D:\\人物视频.mp4", latest_model: null };
const manifest: ProjectManifest = { id: project.id, name: project.name, schema_version: 1, metrics_path: "metrics.jsonl",
  model_identity: null, preprocessing: null, checkpoints: [], source_video: { path: project.source_video, sha256: "test", size_bytes: 1 } };
const identity: ModelIdentity = { kind: "feather_talk", architecture_version: "1", source_sha256: "a", weights_sha256: "b" };
const checkpoint: CheckpointEntry = { model_ref: { project_id: project.id, run_id: "r", checkpoint_id: "saved" },
  relative_path: "checkpoints/saved", sha256: "hash", size_bytes: 100, epoch: 2, step: 4, identity,
  config: { ...config, epochs: 2, seed: 919, beta1: 0.8, loss: { ...config.loss, mouth: 3 } } };
const selection: ResourceSelection = { target: { os: "windows", architecture: "x86_64", backend: "cuda" }, resources: [] };
const callbacks = new Set<(event: { payload: TaskEvent }) => void>();
let snapshot: TaskSnapshot | null;

function publish(status: TaskSnapshot["status"], seq: number, payload: TaskEvent["payload"], extra: Partial<TaskSnapshot> = {}) {
  const latest_event: TaskEvent = { protocol_version: 1, task_id: "t", seq, payload };
  snapshot = {
    task_id: "t", project_id: project.id, job: "train", run_id: "r", status, last_seq: seq, latest_event,
    progress: null, result: null, error: null, metrics: null, checkpoint: null, ...extra,
  };
  act(() => { for (const callback of callbacks) callback({ payload: latest_event }); });
}

async function openTraining(buttonName = "一键训练") {
  render(<App />);
  fireEvent.click(screen.getByRole("button", { name: "训练" }));
  // The first navigation imports the lazy training page and its chart bundle.
  const button = await screen.findByRole("button", { name: buttonName }, { timeout: 5000 });
  await waitFor(() => expect(button).toBeEnabled());
  return button;
}

beforeEach(() => {
  vi.resetAllMocks();
  snapshot = null;
  callbacks.clear();
  vi.mocked(listen).mockImplementation(async (_name, cb) => {
    const callback = cb as (event: { payload: TaskEvent }) => void;
    callbacks.add(callback);
    return () => { callbacks.delete(callback); };
  });
  vi.mocked(api.listProjects).mockResolvedValue([project]);
  vi.mocked(api.getProject).mockResolvedValue(manifest);
  vi.mocked(api.listResourceSelections).mockResolvedValue([selection]);
  vi.mocked(api.getResources).mockResolvedValue({ ready: true, items: [] });
  vi.mocked(api.listDevices).mockResolvedValue([]);
  vi.mocked(api.getTask).mockImplementation(async () => snapshot);
  vi.mocked(api.getTrainDefaults).mockResolvedValue(config);
  vi.mocked(api.startTraining).mockResolvedValue("t");
  vi.mocked(api.readMetrics).mockImplementation(async ({ project_id, run_id }) => ({ project_id, run_id, items: [], next_offset: null }));
});

it("starts from scratch in one click with the engine defaults and automatic batch", async () => {
  fireEvent.click(await openTraining());
  await waitFor(() => expect(api.startTraining).toHaveBeenCalledExactlyOnceWith({
    project_id: project.id, resume: null, batch_mode: "auto", device: { policy: "auto" },
    config: expect.objectContaining({ epochs: 60, batch_size: 8, lr: 0.001, save_every: 5 }),
  }));
});

it("validates the exact seed and submits edited epochs, batch and learning rate in advanced mode", async () => {
  await openTraining();
  fireEvent.click(screen.getByText("高级设置", { selector: "button" }));
  fireEvent.change(screen.getByLabelText("训练总轮数"), { target: { value: "4" } });
  fireEvent.change(screen.getByLabelText("Batch size"), { target: { value: "2" } });
  fireEvent.change(screen.getByLabelText("学习率"), { target: { value: "0.00025" } });
  fireEvent.click(screen.getByRole("button", { name: "Adam 和损失权重" }));
  const seed = screen.getByLabelText("随机种子");
  fireEvent.change(seed, { target: { value: "9007199254740993" } });
  fireEvent.click(screen.getByRole("button", { name: "按参数训练" }));
  expect(seed).toHaveAttribute("aria-invalid", "true");
  expect(api.startTraining).not.toHaveBeenCalled();
  fireEvent.change(seed, { target: { value: "43" } });
  fireEvent.click(screen.getByRole("button", { name: "按参数训练" }));
  await waitFor(() => expect(api.startTraining).toHaveBeenCalledExactlyOnceWith(expect.objectContaining({
    batch_mode: "fixed", config: expect.objectContaining({ epochs: 4, batch_size: 2, lr: 0.00025, seed: 43 }),
  })));
});

it("restores pinned settings from the checkpoint and surfaces a restore failure", async () => {
  vi.mocked(api.getProject).mockResolvedValue({ ...manifest, checkpoints: [checkpoint] });
  const failure: AppError = { code: "model_mismatch", stage: "restore", message: "Checkpoint hash mismatch", evidence: [], actions: [], log_path: null };
  vi.mocked(api.startTraining).mockRejectedValueOnce(failure);
  await openTraining();
  fireEvent.click(await screen.findByText("从检查点续训", { selector: "button" }));
  fireEvent.click(screen.getByRole("button", { name: "继续训练" }));
  expect(await screen.findByText(failure.message)).toBeVisible();
  expect(api.startTraining).toHaveBeenCalledExactlyOnceWith(expect.objectContaining({ resume: checkpoint.model_ref,
    config: expect.objectContaining({ epochs: 3, seed: 919, beta1: 0.8, loss: checkpoint.config.loss }),
  }));
});

it("shows owned live loss, waits for the saved cancellation, then reloads both historical pages", async () => {
  await openTraining();
  const sample: TrainingMetrics = {
    project_id: project.id, run_id: "r", step: 4, epoch: 2, elapsed_ms: 1500, lr: 0.001,
    batch_size: 2, precision: "f32", device: { name: "GeForce", id: "cuda:0", backend: "cuda", vendor: null, driver: null, memory_bytes: null },
    losses: { full: 0.1, mouth: 0.07, temporal: 0.1, temporal_mouth: 0.1, perceptual: 0.1, mouth_gradient: 0.1, total: 0.73125 },
  };
  vi.mocked(api.readMetrics).mockImplementation(async ({ project_id, run_id, offset }) => ({ project_id, run_id,
    items: offset === 0 ? [{ ...sample, step: 3, elapsed_ms: 1000 }] : [sample], next_offset: offset === 0 ? 1 : null,
  }));
  publish("running", 1, { type: "metrics", ...sample }, { metrics: sample });
  expect(await screen.findByText("0.73125")).toBeVisible();
  vi.mocked(api.cancelTask).mockImplementation(async () => {
    if (snapshot) snapshot = { ...snapshot, status: "cancelling" };
  });
  fireEvent.click(screen.getByRole("button", { name: "取消并保存" }));
  expect(await screen.findByRole("button", { name: "等待任务收尾" })).toBeDisabled();
  expect(screen.queryByText("已取消并保存")).not.toBeInTheDocument();
  publish("cancelling", 2, { type: "checkpoint_committed", model_ref: checkpoint.model_ref }, { metrics: sample, checkpoint: checkpoint.model_ref });
  expect(screen.queryByText("已取消并保存")).not.toBeInTheDocument();
  publish("cancelled", 3, { type: "cancelled", checkpoint: checkpoint.model_ref }, { metrics: sample, checkpoint: checkpoint.model_ref });
  expect(await screen.findByText("已取消并保存")).toBeVisible();
  expect(api.readMetrics).toHaveBeenCalledWith(expect.objectContaining({ project_id: "p", run_id: "r", offset: 1 }));
});
