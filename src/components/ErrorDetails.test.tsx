import { fireEvent, render, screen } from "@testing-library/react";
import { expect, it } from "vitest";
import type { AppError } from "../lib/contracts";
import { ErrorDetails } from "./ErrorDetails";

const cases: AppError[] = [
  { code: "out_of_memory", stage: "training-step", message: "CUDA allocation failed",
    evidence: ["RTX 2060; batch=8", "cudaErrorMemoryAllocation"], actions: [], log_path: null },
  { code: "no_face", stage: "landmarks", message: "No valid face at frame 42",
    evidence: ["00:00:01.680"], actions: ["检查第 42 帧是否有遮挡"], log_path: null },
  { code: "encode_failed", stage: "encode", message: "Could not initialize encoder",
    evidence: ["libx264: width not divisible by 2"], actions: [], log_path: "D:\\验收 项目\\导出.log" },
  { code: "internal", stage: "worker", message: "Worker exited unexpectedly",
    evidence: ["raw diagnostic: allocator symbol not found"], actions: [], log_path: null },
];

it.each(cases)("keeps the original $code diagnosis visible", (error) => {
  render(<ErrorDetails error={error} />);
  expect(screen.getByText(error.message)).toBeVisible();
  fireEvent.click(screen.getByText("详细诊断"));
  expect(screen.getByText(error.stage)).toBeVisible();
  for (const line of error.evidence) expect(screen.getByText(line)).toBeVisible();
  for (const action of error.actions) expect(screen.getByText(action)).toBeVisible();
  if (error.log_path) expect(screen.getByText(error.log_path)).toBeVisible();
  if (error.code === "out_of_memory") expect(screen.getByText(/减小 batch size/)).toBeVisible();
  else expect(screen.queryByText(/减小 batch size|显存或内存不足/)).not.toBeInTheDocument();
});
