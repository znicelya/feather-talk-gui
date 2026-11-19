import { lazy, Suspense, useEffect, useRef, useState } from "react";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { AppShell, type View } from "./components/AppShell";
import { ErrorDetails } from "./components/ErrorDetails";
import { TaskStatus } from "./components/TaskStatus";
import { ProjectPage } from "./features/projects/ProjectPage";
import { projectCheckpoints, useProjectManifest, useProjects } from "./features/projects/project-state";
import { ResourcesPage } from "./features/settings/ResourcesPage";
import { useDevices } from "./features/settings/device-state";
import { hasReadyResources, useResourceChecks } from "./features/settings/resource-state";
import type { DevicePolicy } from "./lib/contracts";
import { isActiveTask } from "./lib/events";
import { useTaskControl } from "./lib/task-control";
import "./App.css";

const TrainingPage = lazy(() => import("./features/training/TrainingPage").then(({ TrainingPage }) => ({ default: TrainingPage })));
const GenerationPage = lazy(() => import("./features/generation/GenerationPage").then(({ GenerationPage }) => ({ default: GenerationPage })));

export default function App() {
  const [view, setView] = useState<View>("projects");
  const [device, setDevice] = useState<DevicePolicy>({ policy: "auto" });
  const [chosen, choose] = useState<Record<string, string>>({});
  const projects = useProjects();
  const task = useTaskControl();
  const resource = useResourceChecks();
  const devices = useDevices();
  const active = isActiveTask(task.snapshot);
  const checkpoint = task.snapshot?.checkpoint;
  const project = projects.current;
  const loaded = useProjectManifest(project, project && checkpoint?.project_id === project.id ? checkpoint.checkpoint_id : undefined);
  const checkpoints = projectCheckpoints(project, loaded?.manifest ?? null);
  const selected = project ? chosen[project.id] : undefined;
  const checkpointId = checkpoints.find(({ model_ref }) => model_ref.checkpoint_id === selected)?.model_ref.checkpoint_id
    ?? checkpoints[checkpoints.length - 1]?.model_ref.checkpoint_id ?? "";
  const verifiedTask = useRef("");
  const refreshResources = resource.refresh;
  const refreshDevices = devices.refresh;

  useEffect(() => {
    if (!task.snapshot || active || task.snapshot.job !== "prepare_resources" || verifiedTask.current === task.snapshot.task_id) return;
    verifiedTask.current = task.snapshot.task_id;
    refreshResources();
    refreshDevices();
  }, [task.snapshot, active, refreshResources, refreshDevices]);

  function navigate(next: View) {
    setView(next);
    if (next === "resources" && !task.busy) refreshDevices();
  }

  function selectCheckpoint(id: string) {
    if (project) choose((current) => ({ ...current, [project.id]: id }));
  }

  const taskOwnerId = task.snapshot?.project_id;
  const owner = taskOwnerId ? projects.projects.find(({ id }) => id === taskOwnerId)?.name ?? taskOwnerId : null;
  const localTaskError = !!project && taskOwnerId === project.id && (view === "training" && task.snapshot?.job === "train" ||
    view === "generation" && task.snapshot?.job === "generate");
  return <AppShell view={view} onNavigate={navigate} project={project} task={<TaskStatus
    task={task.snapshot} owner={owner} onCancel={() => void task.cancel()} canCancel={task.canCancel} connected={task.connected}
  />}>
    {task.error && <div className="page-notice"><ErrorDetails error={task.error} /><Button type="button" variant="outline" onClick={task.reconnect}>重连任务服务</Button></div>}
    {task.cancelError && <div className="page-notice"><ErrorDetails error={task.cancelError} /></div>}
    {task.snapshot?.error && !localTaskError && <div className="page-notice"><ErrorDetails error={task.snapshot.error} /></div>}
    {view === "projects" && <ProjectPage store={projects} checkpoints={checkpoints} checkpointId={checkpointId}
      selectCheckpoint={selectCheckpoint} manifestError={loaded?.error ?? null} openTraining={() => navigate("training")} />}
    {view === "resources" && <ResourcesPage resources={resource} devices={devices} device={device}
      onDevice={setDevice} start={task.start} busy={task.busy} />}
    {view === "training" && <Suspense fallback={<Skeleton className="h-48 w-full" aria-label="正在打开训练页面" />}>
      <TrainingPage project={project} checkpoints={checkpoints} checkpointId={checkpointId} selectCheckpoint={selectCheckpoint}
        manifestError={loaded?.error ?? null} device={device} busy={task.busy} task={task.snapshot} start={task.start}
        resourcesReady={hasReadyResources(resource.checks, device)} resourcesLoading={resource.loading}
        openProjects={() => navigate("projects")} openResources={() => navigate("resources")} />
    </Suspense>}
    {view === "generation" && <Suspense fallback={<Skeleton className="h-48 w-full" aria-label="正在打开生成页面" />}>
      <GenerationPage project={project} checkpoints={checkpoints} checkpointId={checkpointId} selectCheckpoint={selectCheckpoint}
        manifestError={loaded?.error ?? null} device={device} busy={task.busy} task={task.snapshot} start={task.start}
        resourcesReady={hasReadyResources(resource.checks, device)} openProjects={() => navigate("projects")}
        openTraining={() => navigate("training")} openResources={() => navigate("resources")} />
    </Suspense>}
  </AppShell>;
}
