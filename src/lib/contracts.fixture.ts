import fixture from "../../contracts/desktop-fixture.json" with { type: "json" };
import type { MetricsPageRequest, TaskSnapshot, TrainRequest, WorkerEvent } from "./contracts.ts";

// JSON imports widen literal strings. Constrain each discriminant while checking every field.
const [rejected, sampled, checkpoint, cancelled, failed, installed] = fixture.events;
export const fixtureEvents = [
  { ...rejected, protocol_version: 1, task_id: null, payload: {
    type: "request_rejected", request: "start", requested_task_id: rejected.payload.requested_task_id!,
    error: { ...rejected.payload.error!, code: "busy" },
  } },
  { ...sampled, protocol_version: 1, task_id: sampled.task_id!, payload: {
    ...sampled.payload, type: "metrics", precision: "f32", device: { ...sampled.payload.device!, backend: "flex" },
    project_id: sampled.payload.project_id!, run_id: sampled.payload.run_id!, step: sampled.payload.step!,
    epoch: sampled.payload.epoch!, elapsed_ms: sampled.payload.elapsed_ms!, lr: sampled.payload.lr!,
    losses: sampled.payload.losses!, batch_size: sampled.payload.batch_size!,
  } },
  { ...checkpoint, protocol_version: 1, task_id: checkpoint.task_id!, payload: {
    type: "checkpoint_committed", model_ref: checkpoint.payload.model_ref!,
  } },
  { ...cancelled, protocol_version: 1, task_id: cancelled.task_id!, payload: {
    type: "cancelled", checkpoint: cancelled.payload.checkpoint!,
  } },
  { ...failed, protocol_version: 1, task_id: failed.task_id!, payload: {
    type: "failed", error: { ...failed.payload.error!, code: "encode_failed" },
  } },
  { ...installed, protocol_version: 1, task_id: installed.task_id!, payload: {
    type: "completed", result: { ...installed.payload.result!, kind: "resources" },
  } },
] satisfies WorkerEvent[];

export const fixtureTraining = {
  ...fixture.train, batch_mode: "fixed",
  device: { policy: "explicit", request: { ...fixture.train.device.request, backend: "flex" } },
} satisfies TrainRequest;

export const fixtureSnapshot = {
  ...fixture.resource_snapshot,
  job: "prepare_resources", status: "completed",
  result: { ...fixture.resource_snapshot.result, kind: "resources" },
} satisfies TaskSnapshot;

export const fixtureMetricRequest = fixture.metrics_request satisfies MetricsPageRequest;
