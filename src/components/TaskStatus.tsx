import type { JobKind, TaskProgress, TaskSnapshot, TaskStatus as Status } from "../lib/contracts";
import { deviceName, backendLabel } from "../features/settings/device-labels";

const jobLabel: Record<JobKind, string> = { train: "训练", generate: "视频生成", prepare_resources: "准备资源" };
const statusLabel: Record<Status, string> = {
  queued: "正在准备", running: "运行中", cancelling: "正在取消", completed: "已完成", cancelled: "已取消", failed: "失败",
};
const stages: Record<string, string> = {
  "audio-features": "提取音频特征", render: "生成人物画面", encode: "封装视频",
  training: "训练模型", "training-checkpoint": "写入检查点",
  "resources/media": "媒体组件", "resources/worker": "运行组件", "resources/probe": "批量探测组件",
  "resources/public-assets": "公共模型", "resources/cuda-math": "CUDA 组件",
  "resources/download": "下载组件",
};

export function stageLabel(stage: string) {
  return stages[stage] ?? (stage.startsWith("batch-selected ") ? "已选择训练批量" : stage);
}

export function TaskProgressView({ progress }: { progress: TaskProgress }) {
  const total = progress.total && progress.total > 0 ? progress.total : null;
  const label = stageLabel(progress.stage);
  return <div className="task-progress">
    <span title={progress.stage}>{label}{total !== null && <small>{progress.completed} / {total}</small>}</span>
    {total !== null && <Progress aria-label={label} value={Math.max(0, Math.min(progress.completed / total * 100, 100))} />}
  </div>;
}

export function TaskStatus({ task, owner, onCancel, canCancel, connected }: {
  task: TaskSnapshot | null;
  owner: string | null;
  onCancel: () => void;
  canCancel: boolean;
  connected: boolean;
}) {
  const status = task ? statusLabel[task.status] : connected ? "无活动任务" : "等待任务服务";
  const cancelledWithCheckpoint = task?.status === "cancelled" && task.checkpoint;
  const resourceSubset = task?.result?.kind === "resources" && !task.result.artifact.ready;
  const metrics = task?.metrics;
  return <footer className="task-bar" aria-label="任务状态">
    <div className="task-title" role="status">
      <Badge variant={task?.status === "failed" ? "destructive" : task ? "secondary" : "outline"}>
        {task ? jobLabel[task.job] : "任务"}
      </Badge>
      <span><strong>{cancelledWithCheckpoint ? "已取消并保存" : resourceSubset ? "本次资源已准备" : status}</strong>
        {owner && <small>{owner}</small>}
        {metrics && <small>{deviceName(metrics.device)} · {backendLabel[metrics.device.backend]} · {metrics.precision.toUpperCase()}</small>}
      </span>
    </div>
    {task?.progress && (task.status === "running" || task.status === "cancelling") && <TaskProgressView progress={task.progress} />}
    {(canCancel || task?.status === "cancelling") && <Button type="button" variant="outline" size="lg" disabled={!canCancel} onClick={onCancel}>
      {task?.status === "cancelling" ? "等待任务收尾" : task?.job === "train" ? "取消并保存" : "取消任务"}
    </Button>}
  </footer>;
}
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
