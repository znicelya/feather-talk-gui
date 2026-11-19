import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, expect, it, vi } from "vitest";
import { api } from "@/lib/api";
import type { AppError, ArtifactResult, CheckpointEntry, ProjectSummary, TaskSnapshot } from "@/lib/contracts";
import { GenerationPage } from "./GenerationPage";
import type { ComponentProps } from "react";

vi.mock("@/lib/api", () => ({ api: {
  selectPath: vi.fn(), startGeneration: vi.fn(), videoPreviewUrl: vi.fn(), openOutputDirectory: vi.fn(),
} }));

const project: ProjectSummary = { id: "p", name: "人物演员", directory: "D:\\演员 项目", source_video: "D:\\演员视频.mp4", latest_model: null };
const model_ref = { project_id: project.id, run_id: "r", checkpoint_id: "c" };
const trained = { model_ref, epoch: 1, step: 1 } as CheckpointEntry;
const audio = "D:\\音频 库\\配音文件.wav";
const output = "D:\\结果 库\\人物成品.mp4";
const video: ArtifactResult = { kind: "video", artifact: { output_path: output, video_frames: 3, elapsed_ms: 1200,
  device: { id: "cuda:0", name: "GeForce", backend: "cuda", driver: null, vendor: null, memory_bytes: null } } };
const base: TaskSnapshot = { task_id: "task", project_id: project.id, job: "generate", status: "running", run_id: null,
  last_seq: 1, latest_event: null, result: null, error: null, progress: null, checkpoint: null, metrics: null };

function state(overrides: Partial<ComponentProps<typeof GenerationPage>> = {}) {
  return <GenerationPage project={project} checkpoints={[trained]} checkpointId={model_ref.checkpoint_id}
    selectCheckpoint={vi.fn()} manifestError={null} busy={false} resourcesReady
    task={null} device={{ policy: "auto" }} start={(command) => command()}
    openProjects={vi.fn()} openTraining={vi.fn()} openResources={vi.fn()} {...overrides} />;
}

async function selectPaths() {
  fireEvent.click(screen.getByRole("button", { name: "选择音频" }));
  expect(await screen.findByText(audio)).toBeVisible();
  fireEvent.click(screen.getByRole("button", { name: "选择保存位置" }));
  expect(await screen.findByText(output)).toBeVisible();
}

beforeEach(() => {
  vi.resetAllMocks();
  vi.mocked(api.selectPath).mockImplementation(async (kind) => kind === "driving_audio" ? audio : output);
  vi.mocked(api.startGeneration).mockResolvedValue(base.task_id);
  vi.mocked(api.videoPreviewUrl).mockResolvedValue("http://asset.localhost/authorized.mp4");
  vi.mocked(api.openOutputDirectory).mockResolvedValue(undefined);
});

it("blocks an untrained checkpoint even though it exists", () => {
  render(state({ checkpoints: [{ ...trained, step: 0 }] }));
  expect(screen.getByText("检查点还未训练")).toBeVisible();
  const start = screen.getByRole("button", { name: "生成视频" });
  expect(start).toBeDisabled();
  fireEvent.click(start);
  expect(api.startGeneration).not.toHaveBeenCalled();
});

it("submits Unicode paths untouched and stops a second start while training owns the worker", async () => {
  const app = render(state());
  await selectPaths();
  app.rerender(state({ busy: true, task: { ...base, job: "train" } }));
  const start = screen.getByRole("button", { name: "生成视频" });
  expect(start).toBeDisabled();
  fireEvent.click(start);
  expect(api.startGeneration).not.toHaveBeenCalled();
  app.rerender(state());
  fireEvent.click(start);
  fireEvent.click(start);
  await waitFor(() => expect(api.startGeneration).toHaveBeenCalledExactlyOnceWith({
    project_id: project.id, model_ref, device: { policy: "auto" }, audio_path: audio, output_path: output,
  }));
  expect(api.selectPath).toHaveBeenNthCalledWith(1, "driving_audio");
  expect(api.selectPath).toHaveBeenNthCalledWith(2, "output_video");
});

it("only authorizes a matching completed output and drops the previous preview after a cancelled attempt", async () => {
  const old = { ...base, task_id: "old", status: "completed", result: video } satisfies TaskSnapshot;
  const { rerender } = render(state({ task: { ...old, project_id: "other" } }));
  expect(api.videoPreviewUrl).not.toHaveBeenCalled();
  rerender(state({ task: old }));
  expect(await screen.findByLabelText("生成视频预览")).toHaveAttribute("src", "http://asset.localhost/authorized.mp4");
  fireEvent.click(screen.getByRole("button", { name: "打开输出目录" }));
  expect(api.openOutputDirectory).toHaveBeenCalledExactlyOnceWith(project.id, old.task_id);
  await selectPaths();
  fireEvent.click(screen.getByRole("button", { name: "生成视频" }));
  await waitFor(() => expect(api.startGeneration).toHaveBeenCalledTimes(1));
  expect(screen.queryByLabelText("生成视频预览")).not.toBeInTheDocument();
  rerender(state({ busy: false, task: { ...base, status: "cancelled", result: video } }));
  expect(screen.queryByLabelText("生成视频预览")).not.toBeInTheDocument();
  expect(api.videoPreviewUrl).toHaveBeenCalledExactlyOnceWith(project.id, old.task_id);
});

it("shows distinct encoding and disk failures, preserves audio, and uses the save dialog on retry", async () => {
  const failure = (code: AppError["code"], message: string): AppError => ({ code, stage: "encode", message, evidence: [], actions: [], log_path: null });
  const { rerender } = render(state());
  await selectPaths();
  fireEvent.click(screen.getByRole("button", { name: "生成视频" }));
  await waitFor(() => expect(api.startGeneration).toHaveBeenCalledTimes(1));
  rerender(state({ task: { ...base, status: "failed", error: failure("encode_failed", "Encoder rejected the requested stream") } }));
  expect(await screen.findByText(/检查媒体组件/)).toBeVisible();
  expect(screen.getByText(audio)).toBeVisible();
  expect(screen.getByRole("button", { name: "生成视频" })).toBeDisabled();
  rerender(state({ task: { ...base, status: "failed", error: failure("disk_full", "No space left to write MP4") } }));
  expect(await screen.findByText(/释放项目和输出目录/)).toBeVisible();
  fireEvent.click(screen.getByRole("button", { name: "选择保存位置" }));
  await waitFor(() => expect(screen.getByRole("button", { name: "生成视频" })).toBeEnabled());
  expect(api.selectPath).toHaveBeenLastCalledWith("output_video");
});
