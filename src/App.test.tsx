import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, expect, it, vi } from "vitest";
import App from "./App";
import { api } from "./lib/api";
import type { ProjectManifest, ProjectSummary, ResourceInventory, ResourceSelection, TaskSnapshot } from "./lib/contracts";

vi.mock("./lib/api", () => ({ api: {
  listProjects: vi.fn(), selectPath: vi.fn(), openProject: vi.fn(), createProject: vi.fn(), getProject: vi.fn(),
  listResourceSelections: vi.fn(), getResources: vi.fn(), prepareResources: vi.fn(), importResource: vi.fn(),
  listDevices: vi.fn(), getTask: vi.fn(), cancelTask: vi.fn(),
} }));
vi.mock("@tauri-apps/api/event", () => ({ listen: vi.fn(async () => () => {}) }));

const selection: ResourceSelection = {
  target: { os: "windows", architecture: "x86_64", backend: "cpu" },
  resources: [{ id: "test-worker", version: "1" }],
};
const inventory: ResourceInventory = {
  ready: false,
  items: [{ resource: selection.resources[0], identity: null, path: null, ready: false, reason: "file not found" }],
};
const active: TaskSnapshot = {
  task_id: "t", project_id: "p", job: "train", run_id: "r", status: "running", last_seq: 1,
  latest_event: null, result: null, error: null, progress: null, metrics: null, checkpoint: null,
};

beforeEach(() => {
  vi.resetAllMocks();
  vi.mocked(api.getTask).mockResolvedValue(null);
  vi.mocked(api.listProjects).mockResolvedValue([]);
  vi.mocked(api.listResourceSelections).mockResolvedValue([selection]);
  vi.mocked(api.getResources).mockResolvedValue(inventory);
  vi.mocked(api.listDevices).mockResolvedValue([]);
});

it("chooses exactly one video; changing the selection replaces the first path", async () => {
  const project: ProjectSummary = { id: "p", name: "人物甲", directory: "D:\\人物 项目", source_video: "D:\\视频 2.mp4", latest_model: null };
  vi.mocked(api.selectPath).mockResolvedValueOnce(project.directory).mockResolvedValueOnce("D:\\视频 1.mp4")
    .mockResolvedValueOnce(project.source_video);
  vi.mocked(api.createProject).mockResolvedValue(project);
  vi.mocked(api.getProject).mockResolvedValue({
    schema_version: 1, id: project.id, name: project.name,
    source_video: { path: project.source_video, sha256: "test", size_bytes: 1 },
    preprocessing: null, model_identity: null, checkpoints: [], metrics_path: "metrics.jsonl",
  } satisfies ProjectManifest);
  vi.mocked(api.listProjects).mockResolvedValueOnce([]).mockResolvedValue([project]);
  render(<App />);
  fireEvent.click(screen.getByRole("button", { name: "新建人物" }));
  fireEvent.change(screen.getByLabelText("人物名称"), { target: { value: project.name } });
  fireEvent.click(screen.getByRole("button", { name: "选择目录" }));
  expect(await screen.findByText(project.directory)).toBeVisible();
  fireEvent.click(screen.getByRole("button", { name: "选择视频" }));
  expect(await screen.findByText("D:\\视频 1.mp4")).toBeVisible();
  fireEvent.click(screen.getByRole("button", { name: "更换视频" }));
  expect(await screen.findByText(project.source_video)).toBeVisible();
  expect(screen.queryByText("D:\\视频 1.mp4")).not.toBeInTheDocument();
  fireEvent.click(screen.getByRole("button", { name: "创建人物" }));
  await waitFor(() => expect(api.createProject).toHaveBeenCalledWith({
    name: project.name, directory: project.directory, source_video: project.source_video,
  }));
});

it("offers preparation for missing resources and blocks it while another task owns the worker", async () => {
  vi.mocked(api.getTask).mockResolvedValue(active);
  render(<App />);
  fireEvent.click(screen.getByRole("button", { name: "资源与设备" }));
  expect(await screen.findByText("资源未就绪")).toBeVisible();
  const prepare = screen.getByRole("button", { name: "准备资源" });
  expect(prepare).toBeDisabled();
  fireEvent.click(prepare);
  expect(api.prepareResources).not.toHaveBeenCalled();
});

it("requests only the missing catalog pins and guards a second start before native acknowledgement", async () => {
  vi.mocked(api.prepareResources).mockImplementation(() => new Promise(() => {}));
  render(<App />);
  fireEvent.click(screen.getByRole("button", { name: "资源与设备" }));
  const prepare = await screen.findByRole("button", { name: "准备资源" });
  await waitFor(() => expect(prepare).toBeEnabled());
  fireEvent.click(prepare);
  fireEvent.click(prepare);
  expect(api.prepareResources).toHaveBeenCalledExactlyOnceWith(selection);
});

it("reports a verified local inventory without starting a download", async () => {
  vi.mocked(api.getResources).mockResolvedValue({ ready: true, items: inventory.items.map((item) => ({ ...item, ready: true, reason: null })) });
  render(<App />);
  fireEvent.click(screen.getByRole("button", { name: "资源与设备" }));
  expect(await screen.findByText("资源已就绪，可离线使用")).toBeVisible();
  expect(screen.queryByRole("button", { name: "准备资源" })).not.toBeInTheDocument();
  expect(api.prepareResources).not.toHaveBeenCalled();
});
