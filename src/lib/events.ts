import { useCallback, useEffect, useRef, useState } from "react";
import { listen } from "@tauri-apps/api/event";
import { api } from "./api.ts";
import type { AppError, TaskEvent, TaskSnapshot, WorkerEvent } from "./contracts.ts";
import { toAppError } from "./errors.ts";

export const taskEventName = "feathertalk://task";

export function listenTaskEvents(handler: (event: TaskEvent) => void): Promise<() => void> {
  return listen<WorkerEvent>(taskEventName, ({ payload }) => {
    if (payload.protocol_version === 1 && payload.task_id !== null) handler(payload);
  });
}

interface TaskTransport {
  listen: (handler: (event: TaskEvent) => void) => Promise<() => void>;
  snapshot: () => Promise<TaskSnapshot | null>;
}

const desktop: TaskTransport = { listen: listenTaskEvents, snapshot: api.getTask };

/** The host assigns project/run ownership. An event only invalidates its projection. */
export async function subscribeTask(
  onSnapshot: (snapshot: TaskSnapshot | null) => void,
  onError: (error: unknown) => void,
  transport: TaskTransport = desktop,
  onRead?: () => void,
) {
  let current: TaskSnapshot | null | undefined;
  let disposed = false;
  let pending = false;
  let reading: Promise<void> | null = null;

  function refresh(): Promise<void> {
    if (disposed) return Promise.resolve();
    pending = true;
    // Serialize reads; a notification during a read schedules another pass.
    reading ??= Promise.resolve().then(async () => {
      try {
        while (pending && !disposed) {
          pending = false;
          try {
            const next = await transport.snapshot();
            if (disposed) continue;
            onRead?.();
            if (next === current) continue;
            if (next && current?.task_id === next.task_id) {
              if (next.last_seq < current.last_seq) continue;
              if (next.last_seq === current.last_seq && next.status === current.status) continue;
            }
            current = next;
            onSnapshot(next);
          } catch (error) {
            if (!disposed) onError(error);
          }
        }
      } finally {
        reading = null;
      }
    });
    return reading;
  }

  const unlisten = await transport.listen((event) => {
    if (current?.task_id === event.task_id && event.seq <= current.last_seq) return;
    void refresh();
  });
  await refresh();
  return {
    refresh,
    stop() {
      disposed = true;
      unlisten();
    },
  };
}

type Connection = Awaited<ReturnType<typeof subscribeTask>>;

export function useTaskSnapshot() {
  const [snapshot, setSnapshot] = useState<TaskSnapshot | null>(null);
  const [error, setError] = useState<AppError | null>(null);
  const [connected, setConnected] = useState(false);
  const [attempt, reconnect] = useState(0);
  const connection = useRef<Promise<Connection | undefined> | null>(null);

  useEffect(() => {
    let disposed = false;
    const onError = (cause: unknown) => {
      if (disposed) return;
      setConnected(false);
      setError(toAppError(cause, "任务服务"));
    };
    const opening = subscribeTask(
      (next) => { if (!disposed) setSnapshot(next); }, onError, desktop,
      () => { if (!disposed) { setConnected(true); setError(null); } },
    ).catch((cause) => { onError(cause); return undefined; });
    connection.current = opening;
    return () => {
      disposed = true;
      connection.current = null;
      void opening.then((live) => live?.stop());
    };
  }, [attempt]);

  const refresh = useCallback(async () => (await connection.current)?.refresh(), []);
  return { snapshot, error, connected, refresh, reconnect: () => reconnect((n) => n + 1) };
}

export function isActiveTask(task: TaskSnapshot | null) {
  return task?.status === "queued" || task?.status === "running" || task?.status === "cancelling";
}
