import { useEffect, useRef, useState } from "react";
import { FolderIcon } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Empty, EmptyContent, EmptyDescription, EmptyHeader, EmptyMedia, EmptyTitle } from "@/components/ui/empty";
import { AppShell, type View } from "./components/AppShell";
import { ErrorDetails } from "./components/ErrorDetails";
import { TaskStatus } from "./components/TaskStatus";
import { ProjectPage } from "./features/projects/ProjectPage";
import { projectCheckpoints, useProjectManifest, useProjects } from "./features/projects/project-state";
import { ResourcesPage } from "./features/settings/ResourcesPage";
import { useDevices } from "./features/settings/device-state";
import { useResourceChecks } from "./features/settings/resource-state";
import type { DevicePolicy } from "./lib/contracts";
import { isActiveTask } from "./lib/events";
import { useTaskControl } from "./lib/task-control";
import "./App.css";

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
  return <AppShell view={view} onNavigate={navigate} project={project} task={<TaskStatus
    task={task.snapshot} owner={owner} onCancel={() => void task.cancel()} canCancel={task.canCancel} connected={task.connected}
  />}>
    {task.error && <div className="page-notice"><ErrorDetails error={task.error} /><Button type="button" variant="outline" onClick={task.reconnect}>重连任务服务</Button></div>}
    {task.cancelError && <div className="page-notice"><ErrorDetails error={task.cancelError} /></div>}
    {task.snapshot?.error && <div className="page-notice"><ErrorDetails error={task.snapshot.error} /></div>}
    {view === "projects" && <ProjectPage store={projects} checkpoints={checkpoints} checkpointId={checkpointId}
      selectCheckpoint={selectCheckpoint} manifestError={loaded?.error ?? null} openTraining={() => navigate("training")} />}
    {view === "resources" && <ResourcesPage resources={resource} devices={devices} device={device}
      onDevice={setDevice} start={task.start} busy={task.busy} />}
    {(view === "training" || view === "generation") && <div className="page-flow">
      <div className="page-heading"><div><h1>{view === "training" ? "训练" : "视频生成"}</h1>
        <p>{project ? `当前人物：${project.name}` : "在人物项目中创建或选择人物。"}</p></div></div>
      <Empty>
        <EmptyHeader><EmptyMedia variant="icon"><FolderIcon /></EmptyMedia>
          <EmptyTitle>{project ? "此页面尚未开放" : "未选择人物"}</EmptyTitle>
          {project && <EmptyDescription>可以查看人物信息与本机资源。</EmptyDescription>}
        </EmptyHeader>
        <EmptyContent><Button type="button" variant="outline" size="lg" onClick={() => navigate("projects")}>查看人物项目</Button></EmptyContent>
      </Empty>
    </div>}
  </AppShell>;
}
