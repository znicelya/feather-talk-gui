import type { ProjectId, TaskId, TrainingMetrics, TrainingRunId, WorkerEvent } from "@/lib/contracts";

export type RunOwner = { projectId: ProjectId; runId: TrainingRunId };
export type MetricOwner = RunOwner & { taskId: TaskId };
export type MetricPoint = TrainingMetrics & { transport?: { taskId: TaskId; seq: number } };

export const lossLabels = {
  total: "总 loss", full: "全图 L1", mouth: "嘴部 L1", temporal: "时序差分",
  temporal_mouth: "嘴部时序差分", perceptual: "VGG 感知 MSE", mouth_gradient: "嘴部梯度",
} satisfies Record<keyof TrainingMetrics["losses"], string>;

export function belongs(point: TrainingMetrics, owner: RunOwner) {
  return point.project_id === owner.projectId && point.run_id === owner.runId;
}

export function appendMetric(history: MetricPoint[], event: WorkerEvent, owner: MetricOwner): MetricPoint[] {
  if (event.task_id !== owner.taskId || event.payload.type !== "metrics" || !belongs(event.payload, owner)) return history;
  const previous = history[history.length - 1];
  if (previous && belongs(previous, owner) && (
    event.payload.elapsed_ms <= previous.elapsed_ms ||
    (previous.transport?.taskId === owner.taskId && event.seq <= previous.transport.seq)
  )) return history;
  const { type: _type, ...sample } = event.payload;
  return [...history, { ...sample, transport: { seq: event.seq, taskId: event.task_id } }];
}

/** The durable page is authoritative through its last elapsed timestamp. */
export function mergeMetricHistory(saved: TrainingMetrics[], live: MetricPoint[], owner: RunOwner): TrainingMetrics[] {
  const historical = saved.filter((point) => belongs(point, owner));
  const after = historical[historical.length - 1]?.elapsed_ms ?? -1;
  const tail = live.filter((point) => belongs(point, owner) && point.elapsed_ms > after)
    .map(({ transport: _transport, ...sample }) => sample);
  return [...historical, ...tail];
}

export function selectChartPoints(history: TrainingMetrics[], maxPoints: number): TrainingMetrics[] {
  if (maxPoints < 1) return [];
  if (history.length <= maxPoints) return history;
  if (maxPoints === 1) return history.slice(-1);
  const stride = (history.length - 1) / (maxPoints - 1);
  return Array.from({ length: maxPoints }, (_, i) => history[Math.round(i * stride)]);
}
