import { describe, expect, it } from "vitest";
import type { TrainingMetrics, WorkerEvent } from "@/lib/contracts";
import { appendMetric, mergeMetricHistory, selectChartPoints } from "./metrics";

const metricOwner = { taskId: "new-worker", projectId: "person", runId: "run" };

function metric(step: number, elapsed_ms = step * 500): TrainingMetrics {
  return {
    project_id: metricOwner.projectId, run_id: metricOwner.runId,
    step, elapsed_ms, epoch: 1, lr: 0.001, batch_size: 1, precision: "f32",
    device: { id: "cuda:0", name: "GeForce", vendor: null, driver: null, backend: "cuda", memory_bytes: null },
    losses: { full: 0.1, mouth: 0.1, temporal: 0.1, temporal_mouth: 0.1, perceptual: 0.1, mouth_gradient: 0.1, total: 0.42 },
  };
}

function event(seq: number, sample: TrainingMetrics, task_id = metricOwner.taskId): WorkerEvent {
  return { protocol_version: 1, task_id, seq, payload: { type: "metrics", ...sample } };
}

describe("loss samples", () => {
  it("rejects duplicate, reordered, old task and wrong run or project events", () => {
    const sample = metric(3);
    const points = appendMetric([], event(4, sample), metricOwner);
    expect(points).toHaveLength(1);
    for (const stale of [event(4, sample), event(3, metric(4)), event(5, metric(4), "old"),
      event(5, { ...metric(4), run_id: "other" }), event(5, { ...metric(4), project_id: "other" })]) {
      expect(appendMetric(points, stale, metricOwner)).toEqual(points);
    }
  });

  it("keeps repeated steps from old checkpoint replay and merges overlapping historical pages", () => {
    const saved = [metric(1), metric(2), metric(3)];
    const replay = metric(2, 2000);
    const live = appendMetric([], event(1, replay), metricOwner);
    expect(mergeMetricHistory(saved, live, metricOwner)).toEqual([...saved, replay]);
    const reopened = [...saved, replay];
    expect(mergeMetricHistory(reopened, live, metricOwner)).toEqual(reopened);
    expect(mergeMetricHistory(reopened, live, { ...metricOwner, projectId: "other" })).toEqual([]);
  });

  it("limits display points, preserving endpoints and all saved samples", () => {
    const saved = Array.from({ length: 3000 }, (_, i) => metric(i + 1));
    const points = selectChartPoints(saved, 2000);
    expect(points.length).toBeLessThanOrEqual(2000);
    expect(points[0]).toBe(saved[0]);
    expect(points[points.length - 1]).toBe(saved[2999]);
    expect(saved).toHaveLength(3000);
  });
});
