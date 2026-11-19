import type { AppError, ErrorCode } from "./contracts";

const descriptions: Record<ErrorCode, [string, string?]> = {
  invalid_config: ["配置有误", "检查所选文件与参数，再试一次。"],
  model_mismatch: ["检查点与人物不匹配", "从当前人物的项目中重新选择检查点。"],
  resource_missing: ["资源缺失", "到资源与设备页面准备对应后端的资源。"],
  resource_corrupt: ["资源校验失败", "检查下方原因，再准备资源或导入匹配的本地包。"],
  unsupported_backend: ["所选后端不可用", "刷新设备并检查所选后端的组件和驱动，也可选择已检测到的 CPU。"],
  device_init: ["设备初始化失败", "检查所选后端的组件和驱动，再刷新设备。"],
  device_lost: ["计算设备已断开", "检查显卡驱动和系统事件，再从已提交的检查点恢复。"],
  out_of_memory: ["显存或内存不足", "减小 batch size，或选择内存足够的其他设备。"],
  no_face: ["找不到有效人脸", "检查报告时间附近的画面，确认人脸清晰且没有遮挡。"],
  no_audio: ["找不到可用音轨", "用播放器确认输入文件包含可播放的音频。"],
  encode_failed: ["视频编码失败", "查看编码器原始错误，检查媒体组件和目标文件。"],
  disk_full: ["磁盘空间不足", "释放项目和输出目录所在磁盘的空间，或选择其他输出位置。"],
  permission_denied: ["没有文件访问权限", "确认输入文件可读、项目和输出目录可写。"],
  busy: ["已有任务运行", "等待当前任务结束，或在任务栏取消它。"],
  protocol: ["后台进程通信失败", "检查资源中的运行组件是否与当前应用版本匹配。"],
  cancelled: ["操作已取消"],
  internal: ["操作失败", "查看详细诊断中的阶段和原始信息。"],
};

export function errorDescription(error: AppError) {
  return descriptions[error.code] ?? descriptions.internal;
}

export function toAppError(error: unknown, stage: string): AppError {
  if (error && typeof error === "object" && "code" in error && "stage" in error && "message" in error) {
    const value = error as Partial<AppError>;
    if (typeof value.code === "string" && Object.hasOwnProperty.call(descriptions, value.code)
      && typeof value.stage === "string" && typeof value.message === "string") {
      return {
        code: value.code, stage: value.stage, message: value.message,
        actions: Array.isArray(value.actions) ? value.actions.filter((item) => typeof item === "string") : [],
        evidence: Array.isArray(value.evidence) ? value.evidence.filter((item) => typeof item === "string") : [],
        log_path: typeof value.log_path === "string" ? value.log_path : null,
      };
    }
  }
  let message = error instanceof Error ? error.message : String(error);
  if (message === "[object Object]") {
    try { message = JSON.stringify(error); } catch { /* Keep the original string. */ }
  }
  return { code: "internal", stage, message, evidence: [], actions: [], log_path: null };
}
