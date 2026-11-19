import { useEffect, useId, useState } from "react";
import { FolderIcon, TriangleAlertIcon } from "lucide-react";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Empty, EmptyContent, EmptyDescription, EmptyHeader, EmptyMedia, EmptyTitle } from "@/components/ui/empty";
import { Field, FieldGroup, FieldLabel } from "@/components/ui/field";
import { NativeSelect, NativeSelectOption } from "@/components/ui/native-select";
import { Skeleton } from "@/components/ui/skeleton";
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group";
import { CheckpointField } from "@/components/CheckpointField";
import { ErrorDetails } from "@/components/ErrorDetails";
import { backendLabel } from "@/features/settings/device-labels";
import { api } from "@/lib/api";
import type { AppError, CheckpointEntry, DevicePolicy, ProjectSummary, TaskSnapshot, TrainConfig } from "@/lib/contracts";
import { toAppError } from "@/lib/errors";
import { isActiveTask } from "@/lib/events";
import type { StartTask } from "@/lib/task-control";
import { LossChart } from "./LossChart";
import { TrainingForm } from "./TrainingForm";
import { useTrainingHistory } from "./use-training-history";

type TrainingProps = {
  project: ProjectSummary | null;
  checkpoints: CheckpointEntry[];
  checkpointId: string;
  selectCheckpoint: (id: string) => void;
  manifestError: AppError | null;
  device: DevicePolicy;
  busy: boolean;
  resourcesReady: boolean;
  resourcesLoading: boolean;
  task: TaskSnapshot | null;
  start: StartTask;
  openProjects: () => void;
  openResources: () => void;
};

export function TrainingPage(props: TrainingProps) {
  const project = props.project;
  return <div className="page-flow">
    <div className="page-heading"><div><h1>训练</h1><p>用一段视频训练人物模型，查看 loss 和检查点。</p></div></div>
    {project ? <ProjectTraining key={`${project.id}:${project.directory}`} {...props} project={project} /> : <Empty>
      <EmptyHeader><EmptyMedia variant="icon"><FolderIcon /></EmptyMedia><EmptyTitle>未选择人物</EmptyTitle>
        <EmptyDescription>先新建或打开人物项目，再开始训练。</EmptyDescription>
      </EmptyHeader>
      <EmptyContent><Button type="button" size="lg" variant="outline" onClick={props.openProjects}>查看人物项目</Button></EmptyContent>
    </Empty>}
  </div>;
}

function ProjectTraining({ project, checkpoints, checkpointId, selectCheckpoint, manifestError, device, busy, resourcesReady,
  resourcesLoading, task, start, openResources }: TrainingProps & { project: ProjectSummary }) {
  const id = useId();
  const [resume, setResume] = useState(false);
  const [historyRun, setHistoryRun] = useState("");
  const [attempt, retry] = useState(0);
  const [defaults, setDefaults] = useState<TrainConfig>();
  const [defaultsError, setDefaultsError] = useState<AppError>();
  const selected = checkpoints.find((checkpoint) => checkpoint.model_ref.checkpoint_id === checkpointId) ?? null;
  const checkpoint = resume ? selected : null;
  const ownTask = task?.job === "train" && task.project_id === project.id ? task : null;
  const active = isActiveTask(ownTask);
  const runs = new Map(checkpoints.map((entry) => [entry.model_ref.run_id, entry]));
  const run = active ? ownTask?.run_id ?? null : (runs.has(historyRun) ? historyRun : ownTask?.run_id ?? selected?.model_ref.run_id ?? null);
  const history = useTrainingHistory(project, run, ownTask);

  useEffect(() => {
    let alive = true;
    void api.getTrainDefaults().then((value) => {
      if (alive) { setDefaults(value); setDefaultsError(undefined); }
    }, (cause) => { if (alive) setDefaultsError(toAppError(cause, "读取训练默认值")); });
    return () => { alive = false; };
  }, [attempt]);

  return <>
    {ownTask?.error && <ErrorDetails error={ownTask.error} />}
    {manifestError && <ErrorDetails error={manifestError} />}
    {!resourcesReady && !resourcesLoading && <Alert>
      <TriangleAlertIcon />
      <AlertTitle>默认资源尚未就绪</AlertTitle>
      <AlertDescription>
        <p>从头训练前请准备计算设备所需的资源。续训时会检查该检查点绑定的版本。</p>
        <Button type="button" variant="outline" onClick={openResources}>查看资源与设备</Button>
      </AlertDescription>
    </Alert>}
    <div className="grid min-w-0 items-start gap-4 min-[78rem]:grid-cols-[minmax(0,5fr)_minmax(0,7fr)]">
      <div className="grid min-w-0 gap-4">
        <Card className="min-w-0">
          <CardHeader><CardTitle><h2 className="wrap-anywhere">{project.name}</h2></CardTitle>
            <CardDescription>训练视频与计算设备</CardDescription></CardHeader>
          <CardContent className="grid gap-4">
            <dl className="detail-list">
              <div><dt>训练视频</dt><dd className="path-text">{project.source_video}</dd></div>
              <div><dt>计算设备</dt><dd>{device.policy === "auto" ? "自动选择可用设备" : backendLabel[device.request.backend]}</dd></div>
            </dl>
            {checkpoints.length > 0 && <FieldGroup>
              <Field data-disabled={busy || undefined}>
                <FieldLabel id={`${id}-source`}>训练起点</FieldLabel>
                <ToggleGroup type="single" variant="outline" value={resume ? "resume" : "fresh"} aria-labelledby={`${id}-source`}
                  disabled={busy} className="flex-wrap" onValueChange={(value) => { if (value) setResume(value === "resume"); }}>
                  <ToggleGroupItem value="fresh">从头训练</ToggleGroupItem>
                  <ToggleGroupItem value="resume">从检查点续训</ToggleGroupItem>
                </ToggleGroup>
              </Field>
              {resume && <CheckpointField checkpoints={checkpoints} value={checkpointId} onChange={selectCheckpoint} disabled={busy} />}
            </FieldGroup>}
          </CardContent>
        </Card>
        {defaultsError ? <div className="grid gap-2"><ErrorDetails error={defaultsError} />
          <Button type="button" variant="outline" onClick={() => retry((n) => n + 1)}>重读默认参数</Button></div>
          : defaults ? <TrainingForm key={checkpoint?.model_ref.checkpoint_id ?? "fresh"} project={project} defaults={defaults}
            checkpoint={checkpoint} device={device} disabled={busy || (resume ? !checkpoint : !resourcesReady)} start={start}
            onStart={() => setHistoryRun("")} /> : <Skeleton className="h-40 w-full" aria-label="正在读取默认参数" />}
      </div>
      <div className="grid min-w-0 gap-3">
        {history.error && <div className="grid gap-2"><ErrorDetails error={history.error} />
          <Button type="button" variant="outline" onClick={history.retry}>重读训练记录</Button></div>}
        <LossChart points={history.points} loading={history.loading} active={active} scope={runs.size > 0 && <Field data-disabled={active || undefined}>
          <FieldLabel htmlFor={`${id}-history`}>曲线来源</FieldLabel>
          <NativeSelect id={`${id}-history`} disabled={active} value={active ? "" : historyRun} onChange={(event) => setHistoryRun(event.target.value)}>
            <NativeSelectOption value="">{active ? "当前训练" : "最近训练 / 所选检查点"}</NativeSelectOption>
            {[...runs].reverse().map(([key, entry]) => <NativeSelectOption key={key} value={key}>
              第 {entry.epoch} 轮 · 步 {entry.step} · {key.slice(0, 8)}
            </NativeSelectOption>)}
          </NativeSelect>
        </Field>} />
      </div>
    </div>
  </>;
}
