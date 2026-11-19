import { useEffect, useMemo, useState } from "react";
import { api } from "@/lib/api";
import type { AppError, ProjectSummary, TaskSnapshot, TrainingMetrics } from "@/lib/contracts";
import { toAppError } from "@/lib/errors";
import { isActiveTask, listenTaskEvents } from "@/lib/events";
import { appendMetric, belongs, mergeMetricHistory, type MetricPoint } from "./metrics";

const empty: TrainingMetrics[] = [];
const pageSize = 1000;

export function useTrainingHistory(project: ProjectSummary, runId: string | null, task: TaskSnapshot | null) {
  const projectId = project.id;
  const key = JSON.stringify([projectId, project.directory, runId]);
  const source = runId && task?.run_id === runId ? task : null;
  const taskId = source?.task_id ?? null;
  const active = isActiveTask(source);
  const bootstrap = source?.metrics;
  const [retry, setRetry] = useState(0);
  const [history, setHistory] = useState<{ key: string; points: TrainingMetrics[]; error: AppError | null }>();
  const [stream, setStream] = useState<{ key: string; taskId: string; points: MetricPoint[]; error: AppError | null }>();

  useEffect(() => {
    if (!runId || !taskId || !active) return;
    const owner = { projectId, runId, taskId };
    let alive = true;
    const unsubscribe = listenTaskEvents((event) => {
      if (!alive || event.task_id !== taskId || event.payload.type !== "metrics") return;
      setStream((previous) => {
        const before = previous?.key === key && previous.taskId === taskId ? previous.points : empty;
        const points = appendMetric(before, event, owner);
        return previous?.points === points ? previous : { key, taskId, points, error: null };
      });
    }).catch((cause) => {
      if (alive) setStream({ key, taskId, points: [], error: toAppError(cause, "订阅训练指标") });
      return undefined;
    });
    return () => { alive = false; void unsubscribe.then((stop) => stop?.()); };
  }, [active, key, projectId, runId, taskId, retry]);

  useEffect(() => {
    if (!runId) return;
    const owner = { projectId, runId };
    let alive = true;
    async function load() {
      const points: TrainingMetrics[] = [];
      let offset = 0;
      try {
        while (alive) {
          const page = await api.readMetrics({ project_id: owner.projectId, run_id: owner.runId, offset, limit: pageSize });
          if (!alive) return;
          if (page.project_id !== projectId || page.run_id !== runId || page.items.some((point) => !belongs(point, owner))) {
            throw new Error("历史记录与所选人物或训练不匹配。");
          }
          points.push(...page.items);
          if (page.next_offset === null) break;
          if (page.next_offset !== offset + page.items.length || page.next_offset <= offset) {
            throw new Error("历史记录的分页位置无效。");
          }
          offset = page.next_offset;
        }
        if (alive) setHistory({ key, points, error: null });
      } catch (cause) {
        if (alive) setHistory((previous) => ({ key, points: previous?.key === key ? previous.points : points, error: toAppError(cause, "读取训练历史") }));
      }
    }
    // Read once on entry and again after the worker exits to recover samples
    // that did not reach the stream. No disk polling during optimizer steps.
    void load();
    return () => { alive = false; };
  }, [active, key, projectId, runId, taskId, retry]);

  const saved = history?.key === key ? history : undefined;
  const current = stream?.key === key && stream.taskId === taskId ? stream : undefined;
  const live = current?.points ?? empty;
  const points = useMemo(() => {
    if (!runId) return empty;
    const owner = { projectId, runId };
    const tail = bootstrap && belongs(bootstrap, owner) && bootstrap.elapsed_ms > (live[live.length - 1]?.elapsed_ms ?? -1)
      ? [...live, bootstrap] : live;
    return mergeMetricHistory(saved?.points ?? empty, tail, owner);
  }, [bootstrap, live, projectId, runId, saved?.points]);
  return { points, error: saved?.error ?? (active ? current?.error : null), loading: !!runId && !saved, retry: () => setRetry((n) => n + 1) };
}
