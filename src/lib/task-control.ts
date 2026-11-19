import { useRef, useState } from "react";
import { api } from "./api";
import type { AppError, TaskId } from "./contracts";
import { isActiveTask, useTaskSnapshot } from "./events";
import { toAppError } from "./errors";

export type StartTask = (command: () => Promise<TaskId>) => Promise<TaskId>;

export function useTaskControl() {
  const live = useTaskSnapshot();
  const [pending, setPending] = useState(false);
  const lock = useRef(false);
  const [cancelError, setCancelError] = useState<AppError | null>(null);
  const busy = !live.connected || pending || isActiveTask(live.snapshot);
  const canCancel = live.connected && !pending && (live.snapshot?.status === "queued" || live.snapshot?.status === "running");

  async function exclusive<T>(command: () => Promise<T>) {
    lock.current = true;
    setPending(true);
    try {
      const value = await command();
      await live.refresh();
      return value;
    } finally {
      lock.current = false;
      setPending(false);
    }
  }

  const start: StartTask = async (command) => {
    if (lock.current || busy) throw {
      code: "busy", stage: "启动任务", message: "请等待任务服务就绪或当前任务结束。",
      evidence: [], actions: [], log_path: null,
    } satisfies AppError;
    setCancelError(null);
    return exclusive(command);
  };

  async function cancel() {
    if (lock.current || !canCancel || !live.snapshot) return;
    const id = live.snapshot.task_id;
    setCancelError(null);
    try { await exclusive(() => api.cancelTask(id)); }
    catch (cause) { setCancelError(toAppError(cause, "取消任务")); }
  }

  return { ...live, start, busy, canCancel, cancel, cancelError };
}
