import assert from "node:assert/strict";
import { test } from "node:test";
import { mockIPC } from "@tauri-apps/api/mocks";
import fixture from "../contracts/desktop-fixture.json" with { type: "json" };
import { api } from "../src/lib/api.ts";
import {
  fixtureEvents, fixtureMetricRequest, fixtureSnapshot, fixtureTraining,
} from "../src/lib/contracts.fixture.ts";
import { subscribeTask } from "../src/lib/events.ts";

test("shared desktop wire values reach the bridge unchanged", async () => {
  assert.deepEqual(fixtureEvents, fixture.events);
  assert.deepEqual(fixtureSnapshot, fixture.resource_snapshot);
  globalThis.window = {};
  const calls = [];
  mockIPC((command, args) => { calls.push([command, args]); });
  await api.startTraining(fixtureTraining);
  await api.readMetrics(fixtureMetricRequest);
  assert.deepEqual(calls, [
    ["start_training", { request: fixture.train }],
    ["read_metrics", { request: fixture.metrics_request }],
  ]);
  for (const seed of [Number.MAX_SAFE_INTEGER + 1, -1, 0.25, Infinity, "42"]) {
    await assert.rejects(api.startTraining({ ...fixtureTraining, config: { ...fixtureTraining.config, seed } }), {
      code: "invalid_config",
    });
  }
  assert.equal(calls.length, 2);
});

test("subscription survives an in-flight snapshot and a new worker with lower sequence", async () => {
  const [, metrics, , cancelled, failed] = fixtureEvents;
  const { type: _, ...sample } = metrics.payload;
  const initial = {
    task_id: metrics.task_id, job: "train", project_id: sample.project_id, run_id: sample.run_id,
    status: "running", last_seq: metrics.seq, latest_event: metrics, metrics: sample,
    result: null, progress: null, error: null, checkpoint: null,
  };
  const terminal = {
    ...initial, status: "cancelled", last_seq: cancelled.seq, latest_event: cancelled,
    checkpoint: cancelled.payload.checkpoint,
  };
  const nextTask = {
    ...initial, task_id: failed.task_id, job: "generate", run_id: null, status: "failed",
    last_seq: failed.seq, latest_event: failed, error: failed.payload.error, metrics: null,
  };
  let listener;
  let reads = 0;
  let unsubscribed = false;
  const entered = Promise.withResolvers();
  const response = Promise.withResolvers();
  let current = terminal;
  const seen = [];
  const errors = [];

  const connecting = subscribeTask((value) => seen.push(value), (error) => errors.push(error), {
    async listen(handler) {
      listener = handler;
      return () => { unsubscribed = true; };
    },
    snapshot() {
      assert.ok(listener);
      if (++reads === 1) {
        entered.resolve();
        return response.promise;
      }
      return Promise.resolve(structuredClone(current));
    },
  });
  await entered.promise;
  listener(cancelled);
  response.resolve(initial);
  const connection = await connecting;
  assert.deepEqual(seen, [initial, terminal]);
  listener(metrics);
  listener(cancelled);
  await Promise.resolve();
  assert.equal(reads, 2);
  current = nextTask;
  listener(failed);
  await connection.refresh();
  assert.deepEqual(seen, [initial, terminal, nextTask]);
  listener(cancelled);
  await connection.refresh();
  assert.equal(seen.length, 3);
  connection.stop();
  assert.ok(unsubscribed);
  const before = reads;
  await connection.refresh();
  assert.equal(reads, before);
  assert.deepEqual(errors, []);
});
