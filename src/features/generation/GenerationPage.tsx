import { useEffect, useRef, useState, type FormEvent } from "react";
import { FolderIcon, FolderOpenIcon, InfoIcon, VideoIcon } from "lucide-react";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { Empty, EmptyContent, EmptyDescription, EmptyHeader, EmptyMedia, EmptyTitle } from "@/components/ui/empty";
import { FieldGroup } from "@/components/ui/field";
import { Skeleton } from "@/components/ui/skeleton";
import { CheckpointField } from "@/components/CheckpointField";
import { ErrorDetails } from "@/components/ErrorDetails";
import { PathField } from "@/components/PathField";
import { TaskProgressView } from "@/components/TaskStatus";
import { backendLabel, deviceName } from "@/features/settings/device-labels";
import { api } from "@/lib/api";
import type { AppError, CheckpointEntry, DevicePolicy, ExportResult, ProjectSummary, TaskId, TaskSnapshot } from "@/lib/contracts";
import { toAppError } from "@/lib/errors";
import { isActiveTask } from "@/lib/events";
import type { StartTask } from "@/lib/task-control";

interface GenerationProps {
  project: ProjectSummary | null;
  checkpoints: CheckpointEntry[];
  checkpointId: string;
  selectCheckpoint: (id: string) => void;
  manifestError: AppError | null;
  device: DevicePolicy;
  busy: boolean;
  resourcesReady: boolean;
  task: TaskSnapshot | null;
  start: StartTask;
  openProjects: () => void;
  openTraining: () => void;
  openResources: () => void;
}

export function GenerationPage(props: GenerationProps) {
  const project = props.project;
  return <div className="page-flow">
    <div className="page-heading"><div><h1>视频生成</h1><p>选择人物检查点和一段音频，将结果保存为 MP4。</p></div></div>
    {project ? <ProjectGeneration key={`${project.id}:${project.directory}`} {...props} project={project} /> : <Empty>
      <EmptyHeader><EmptyMedia variant="icon"><FolderIcon /></EmptyMedia><EmptyTitle>未选择人物</EmptyTitle>
        <EmptyDescription>打开一个已训练的人物项目。</EmptyDescription></EmptyHeader>
      <EmptyContent><Button type="button" size="lg" variant="outline" onClick={props.openProjects}>查看人物项目</Button></EmptyContent>
    </Empty>}
  </div>;
}

function ProjectGeneration({ project, checkpoints, checkpointId, selectCheckpoint, manifestError,
  device, busy, resourcesReady, task, start, openTraining, openResources }: GenerationProps & { project: ProjectSummary }) {
  const [audio, setAudio] = useState<string | null>(null);
  const [output, setOutput] = useState<string | null>(null);
  const [pending, setPending] = useState(false);
  const [attempt, setAttempt] = useState<{ taskId: TaskId | null }>();
  const [error, setError] = useState<AppError | null>(null);
  const inFlight = useRef(false);
  const selected = checkpoints.find(({ model_ref }) => model_ref.checkpoint_id === checkpointId);
  const trained = !!selected && selected.step > 0;
  const owned = task?.job === "generate" && task.project_id === project.id && (!attempt || attempt.taskId === task.task_id) ? task : null;
  const result = owned?.status === "completed" && owned.result?.kind === "video" ? owned.result.artifact : null;
  const active = isActiveTask(owned);
  const blocked = busy || pending;

  async function work(stage: string, job: () => Promise<void>) {
    if (inFlight.current || blocked) return;
    inFlight.current = true;
    setPending(true);
    setError(null);
    try { await job(); }
    catch (cause) { setError(toAppError(cause, stage)); }
    finally { inFlight.current = false; setPending(false); }
  }

  function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!audio || !output || !selected || !trained) return;
    void work("开始视频生成", async () => {
      setAttempt({ taskId: null });
      // Each attempt spends one save selection. Retrying opens the native
      // save dialog again, including its existing-file confirmation.
      setOutput(null);
      const taskId = await start(() => api.startGeneration({
        project_id: project.id, model_ref: selected.model_ref, audio_path: audio, output_path: output, device,
      }));
      setAttempt({ taskId });
    });
  }

  return <>
    {manifestError && <ErrorDetails error={manifestError} />}
    {owned?.error && <ErrorDetails error={owned.error} />}
    {error && <ErrorDetails error={error} />}
    {!trained && !manifestError && <Alert>
      <InfoIcon />
      <AlertTitle>{selected ? "检查点还未训练" : "没有已训练的检查点"}</AlertTitle>
      <AlertDescription>
        <p>选择至少完成一个训练步的检查点，或先训练当前人物。</p>
        <Button type="button" variant="outline" onClick={openTraining}>前往训练</Button>
      </AlertDescription>
    </Alert>}
    {!resourcesReady && !manifestError && <Alert>
      <InfoIcon />
      <AlertTitle>默认资源未确认就绪</AlertTitle>
      <AlertDescription>
        <p>启动时会校验运行组件和所选检查点绑定的模型版本。</p>
        <Button type="button" variant="outline" onClick={openResources}>查看资源与设备</Button>
      </AlertDescription>
    </Alert>}
    <div className="grid min-w-0 items-start gap-4 min-[68rem]:grid-cols-[minmax(0,5fr)_minmax(0,7fr)]">
      <form onSubmit={submit} className="min-w-0">
        <Card>
          <CardHeader><CardTitle><h2 className="wrap-anywhere">{project.name}</h2></CardTitle>
            <CardDescription>音频与保存位置</CardDescription></CardHeader>
          <CardContent>
            <FieldGroup>
              <CheckpointField checkpoints={checkpoints} value={checkpointId} onChange={selectCheckpoint} disabled={blocked} />
              <PathField label="驱动音频" path={audio} action={audio ? "更换音频" : "选择音频"}
                empty="支持 WAV、MP3、M4A、AAC、FLAC、OGG 和 Opus。" disabled={blocked}
                onPick={() => void work("选择驱动音频", async () => {
                  const path = await api.selectPath("driving_audio");
                  if (path) setAudio(path);
                })} />
              <PathField label="输出文件" path={output} action="选择保存位置"
                empty={attempt ? "下一次生成前请选择新的保存位置，或通过保存对话框确认替换。" : "选择保存生成结果的 MP4 文件。"}
                disabled={blocked} onPick={() => void work("选择保存位置", async () => {
                  const path = await api.selectPath("output_video");
                  if (path) setOutput(path);
                })} />
            </FieldGroup>
          </CardContent>
          <CardFooter><Button type="submit" size="lg" disabled={blocked || !trained || !audio || !output}>
            <VideoIcon data-icon="inline-start" />生成视频
          </Button></CardFooter>
        </Card>
      </form>
      {owned && result ? <CompletedVideo taskId={owned.task_id} projectId={project.id} result={result} /> : <Card className="min-w-0">
        <CardHeader><CardTitle><h2>生成结果</h2></CardTitle><CardDescription>导出完成后可预览视频。</CardDescription></CardHeader>
        <CardContent>
          {active && owned?.progress ? <TaskProgressView progress={owned.progress} /> : <Empty>
            <EmptyHeader><EmptyMedia variant="icon"><VideoIcon /></EmptyMedia>
              <EmptyTitle>{active ? "正在准备视频生成" : owned?.status === "cancelled" ? "已取消本次生成" : "暂无生成结果"}</EmptyTitle>
              {owned?.status === "cancelled" && <EmptyDescription>可以保留所选音频，重新选择保存位置。</EmptyDescription>}
            </EmptyHeader>
          </Empty>}
        </CardContent>
      </Card>}
    </div>
  </>;
}

function CompletedVideo({ taskId, projectId, result }: { taskId: TaskId; projectId: string; result: ExportResult }) {
  const key = `${projectId}:${taskId}`;
  const [attempt, retry] = useState(0);
  const [preview, setPreview] = useState<{ key: string; url?: string; error?: AppError; unsupported?: boolean }>();
  const [openError, setOpenError] = useState<{ key: string; error: AppError }>();
  const current = preview?.key === key ? preview : undefined;

  useEffect(() => {
    let alive = true;
    void api.videoPreviewUrl(projectId, taskId).then((url) => {
      if (alive) setPreview({ key, url });
    }, (cause) => { if (alive) setPreview({ key, error: toAppError(cause, "预览成品视频") }); });
    return () => { alive = false; };
  }, [attempt, key, projectId, taskId]);

  return <Card className="min-w-0">
    <CardHeader>
      <CardTitle><h2>生成结果</h2></CardTitle>
      <CardDescription>{result.video_frames} 帧 · {deviceName(result.device)} · {backendLabel[result.device.backend]} · {(result.elapsed_ms / 1000).toFixed(1)} 秒</CardDescription>
    </CardHeader>
    <CardContent className="grid min-w-0 gap-4">
      <Badge variant="secondary">MP4 已导出</Badge>
      <p className="path-text">{result.output_path}</p>
      {!current && <Skeleton className="aspect-video w-full" aria-label="正在授权视频预览" />}
      {current?.url && !current.unsupported && <video key={key} className="max-h-[55vh] w-full rounded-md" src={current.url}
        controls preload="metadata" aria-label="生成视频预览" onError={() => setPreview({ key, unsupported: true })} />}
      {current?.unsupported && <Alert><InfoIcon /><AlertTitle>当前浏览器组件无法播放此视频</AlertTitle>
        <AlertDescription>打开输出目录，用系统播放器查看已导出的 MP4。</AlertDescription></Alert>}
      {current?.error && <ErrorDetails error={current.error} />}
      {openError?.key === key && <ErrorDetails error={openError.error} />}
    </CardContent>
    <CardFooter className="actions">
      <Button type="button" size="lg" variant="outline" onClick={() => {
        setOpenError(undefined);
        void api.openOutputDirectory(projectId, taskId).catch((cause) => setOpenError({ key, error: toAppError(cause, "打开输出目录") }));
      }}><FolderOpenIcon data-icon="inline-start" />打开输出目录</Button>
      {current?.error && <Button type="button" variant="outline" onClick={() => { setPreview(undefined); retry((n) => n + 1); }}>重新加载预览</Button>}
    </CardFooter>
  </Card>;
}
