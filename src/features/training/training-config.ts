import type { CheckpointEntry, LossConfig, TrainConfig } from "@/lib/contracts";
import { lossLabels } from "./metrics";

type ConfigNumber = Exclude<keyof TrainConfig, "loss">;
type LossNumber = Exclude<keyof LossConfig, "temporal_stride">;
export type TrainField = ConfigNumber | LossNumber;
export type TrainingFormValue = Record<TrainField, string>;
export type FormErrors = Partial<Record<TrainField, string>>;

const uintMax = 0xffff_ffff;
type Rule = { label: string; valid: (value: number) => boolean; hint: string; integer?: boolean };
const integer = (label: string, min: number, max: number): Rule => ({
  label, valid: (value) => Number.isSafeInteger(value) && value >= min && value <= max,
  hint: `请输入 ${min} 到 ${max} 之间的整数。`, integer: true,
});
const positive = (label: string): Rule => ({
  label, valid: (n) => Number.isFinite(Math.fround(n)) && Math.fround(n) > 0, hint: "请输入可用的正数。",
});
const nonnegative = (label: string): Rule => ({
  label, valid: (n) => Number.isFinite(Math.fround(n)) && n >= 0, hint: "请输入可用的非负数。",
});
const beta = (label: string): Rule => ({
  label, valid: (n) => n >= 0 && Math.fround(n) < 1, hint: "请输入大于或等于 0 且小于 1 的值。",
});

export const trainingFields: Record<TrainField, Rule> = {
  epochs: integer("训练总轮数", 1, uintMax), batch_size: integer("Batch size", 1, 1024),
  lr: positive("学习率"), save_every: integer("检查点保存间隔", 1, uintMax),
  prefetch_capacity: integer("预取批数", 0, 32), seed: integer("随机种子", 0, Number.MAX_SAFE_INTEGER),
  beta1: beta("Adam β1"), beta2: beta("Adam β2"), epsilon: positive("Adam ε"), weight_decay: nonnegative("权重衰减"),
  full: nonnegative(lossLabels.full), mouth: nonnegative(lossLabels.mouth), temporal: nonnegative(lossLabels.temporal),
  temporal_mouth: nonnegative(lossLabels.temporal_mouth), perceptual: nonnegative(lossLabels.perceptual),
  mouth_gradient: nonnegative(lossLabels.mouth_gradient),
  temporal_kernel: { label: "时序低通核", integer: true,
    valid: (n) => Number.isInteger(n) && n > 0 && n <= uintMax && n % 2 === 1, hint: "请输入正奇数。" },
};

const keys = Object.keys(trainingFields) as TrainField[];
export const operationalFields = ["epochs", "batch_size", "lr", "save_every", "prefetch_capacity"] as const;
export const adamFields = ["seed", "beta1", "beta2", "epsilon", "weight_decay"] as const;
export const lossFields = ["full", "mouth", "temporal", "temporal_mouth", "perceptual", "mouth_gradient", "temporal_kernel"] as const;
const pinned = new Set<TrainField>([...adamFields, ...lossFields]);

export function configFor(checkpoint: CheckpointEntry | null, defaults: TrainConfig): TrainConfig {
  return checkpoint ? {
    ...checkpoint.config,
    epochs: Math.min(uintMax, Math.max(checkpoint.config.epochs, checkpoint.epoch + 1)),
  } : defaults;
}

export function formValues(config: TrainConfig): TrainingFormValue {
  const values = { ...config, ...config.loss };
  return Object.fromEntries(keys.map((key) => [key, String(values[key])])) as TrainingFormValue;
}

export function validateTrainingForm(value: TrainingFormValue, checkpoint: CheckpointEntry | null = null): TrainConfig | FormErrors {
  const errors: FormErrors = {};
  const numbers = {} as Record<TrainField, number>;
  for (const field of keys) {
    const locked = checkpoint && pinned.has(field);
    const source = locked ? { ...checkpoint.config, ...checkpoint.config.loss } : null;
    const text = (source ? String(source[field]) : value[field]).trim();
    const rule = trainingFields[field];
    const syntax = rule.integer ? /^\d+$/ : /^[+-]?(?:\d+\.?\d*|\.\d+)(?:e[+-]?\d+)?$/i;
    const number = Number(text);
    if (!syntax.test(text) || !rule.valid(number)) errors[field] = rule.hint;
    numbers[field] = number;
  }
  if (checkpoint && numbers.epochs <= checkpoint.epoch) errors.epochs = `请输入大于 ${checkpoint.epoch} 的总轮数。`;
  if (Object.keys(errors).length) return errors;
  const { full, mouth, temporal, temporal_mouth, perceptual, mouth_gradient, temporal_kernel, ...config } = numbers;
  return { ...config, loss: { full, mouth, temporal, temporal_mouth, perceptual, mouth_gradient, temporal_kernel,
    temporal_stride: checkpoint?.config.loss.temporal_stride ?? 1 } };
}
