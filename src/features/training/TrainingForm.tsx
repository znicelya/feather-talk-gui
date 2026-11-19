import { useId, useRef, useState, type FormEvent } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";
import { Field, FieldDescription, FieldError, FieldGroup, FieldLabel, FieldLegend, FieldSet } from "@/components/ui/field";
import { Input } from "@/components/ui/input";
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group";
import { ErrorDetails } from "@/components/ErrorDetails";
import { api } from "@/lib/api";
import type { AppError, CheckpointEntry, DevicePolicy, ProjectSummary, TrainConfig, TrainRequest } from "@/lib/contracts";
import { toAppError } from "@/lib/errors";
import type { StartTask } from "@/lib/task-control";
import {
  adamFields, configFor, formValues, lossFields, operationalFields, trainingFields, validateTrainingForm,
  type FormErrors, type TrainField,
} from "./training-config";

export function TrainingForm({ project, defaults, checkpoint, device, disabled, start, onStart }: {
  project: ProjectSummary;
  defaults: TrainConfig;
  checkpoint: CheckpointEntry | null;
  device: DevicePolicy;
  disabled: boolean;
  start: StartTask;
  onStart: () => void;
}) {
  const prefix = useId();
  const base = configFor(checkpoint, defaults);
  const [advanced, setAdvanced] = useState(false);
  const [values, setValues] = useState(() => formValues(base));
  const [errors, setErrors] = useState<FormErrors>({});
  const [failure, setFailure] = useState<AppError | null>(null);
  const [pending, setPending] = useState(false);
  const submitting = useRef(false);
  const disabledFields = disabled || pending;

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (disabledFields || submitting.current) return;
    setFailure(null);
    const validated = validateTrainingForm(advanced ? values : formValues(base), checkpoint);
    if (!("loss" in validated)) {
      setErrors(validated);
      setAdvanced(true);
      return;
    }
    // Simple mode submits the authoritative defaults (or saved configuration).
    // Advanced mode submits exactly the validated edits and uses fixed batch.
    const request: TrainRequest = {
      project_id: project.id, config: advanced ? validated : base, batch_mode: advanced ? "fixed" : "auto",
      device, resume: checkpoint?.model_ref ?? null,
    };
    submitting.current = true;
    setPending(true);
    try { await start(() => api.startTraining(request)); onStart(); }
    catch (cause) { setFailure(toAppError(cause, checkpoint ? "从检查点续训" : "开始训练")); }
    finally { submitting.current = false; setPending(false); }
  }

  function numberField(key: TrainField, locked = false) {
    const id = `${prefix}-${key}`;
    const rule = trainingFields[key];
    const error = errors[key];
    return <Field key={key} data-invalid={!!error || undefined} data-disabled={disabledFields || locked || undefined}>
      <FieldLabel htmlFor={id}>{rule.label}</FieldLabel>
      <Input id={id} type="text" inputMode={rule.integer ? "numeric" : "decimal"} autoComplete="off"
        disabled={disabledFields} readOnly={locked} aria-invalid={!!error || undefined} aria-describedby={error ? `${id}-error` : undefined}
        value={values[key]} onChange={(event) => {
          setValues((current) => ({ ...current, [key]: event.target.value }));
          setErrors((current) => ({ ...current, [key]: undefined }));
        }} />
      {error && <FieldError id={`${id}-error`}>{error}</FieldError>}
    </Field>;
  }

  return <form onSubmit={(event) => void submit(event)}>
    <Card>
      <CardHeader>
        <CardTitle><h2>训练参数</h2></CardTitle>
        <CardDescription>{checkpoint ? "继续已有训练，保留模型、优化器和历史曲线。" : "从头训练此人物。首次训练会先准备视频和音频特征。"}</CardDescription>
      </CardHeader>
      <CardContent className="grid gap-4">
        <FieldGroup>
          <Field data-disabled={disabledFields || undefined}>
            <FieldLabel id={`${prefix}-mode`}>参数模式</FieldLabel>
            <ToggleGroup type="single" variant="outline" value={advanced ? "advanced" : "simple"}
              aria-labelledby={`${prefix}-mode`} disabled={disabledFields} onValueChange={(value) => {
                if (value) setAdvanced(value === "advanced");
              }}>
              <ToggleGroupItem value="simple">一键</ToggleGroupItem>
              <ToggleGroupItem value="advanced">高级设置</ToggleGroupItem>
            </ToggleGroup>
            <FieldDescription>{advanced ? "使用填写的批量。显存不足时可减小 Batch size 后重试。"
              : `训练至 ${base.epochs} 轮 · 目标批量 ${base.batch_size} · 学习率 ${base.lr} · 每 ${base.save_every} 轮保存`}</FieldDescription>
          </Field>
          {advanced && <>
            <FieldSet>
              <FieldLegend variant="label">训练与保存</FieldLegend>
              <FieldDescription>轮数是总轮数。Batch size 表示每批相邻帧对数量；预取批数填 0 可关闭预取。</FieldDescription>
              <FieldGroup className="grid grid-cols-[repeat(auto-fit,minmax(min(100%,11rem),1fr))] gap-4">
                {operationalFields.map((key) => numberField(key))}
              </FieldGroup>
            </FieldSet>
            <Collapsible defaultOpen={!!Object.keys(errors).find((key) => !operationalFields.some((field) => field === key))}>
              <CollapsibleTrigger asChild><Button type="button" variant="outline">Adam 和损失权重</Button></CollapsibleTrigger>
              <CollapsibleContent forceMount className="pt-4 data-[state=closed]:hidden">
                <FieldGroup>
                  <FieldSet>
                    <FieldLegend variant="label">随机种子与优化器</FieldLegend>
                    {checkpoint && <FieldDescription>续训沿用已保存的随机种子、Adam 设置和损失权重。</FieldDescription>}
                    <FieldGroup className="grid grid-cols-[repeat(auto-fit,minmax(min(100%,11rem),1fr))] gap-4">
                      {adamFields.map((key) => numberField(key, !!checkpoint))}
                    </FieldGroup>
                  </FieldSet>
                  <FieldSet>
                    <FieldLegend variant="label">损失权重</FieldLegend>
                    <FieldDescription>使用相邻帧，时序步长为 1。</FieldDescription>
                    <FieldGroup className="grid grid-cols-[repeat(auto-fit,minmax(min(100%,11rem),1fr))] gap-4">
                      {lossFields.map((key) => numberField(key, !!checkpoint))}
                    </FieldGroup>
                  </FieldSet>
                </FieldGroup>
              </CollapsibleContent>
            </Collapsible>
            {Object.entries(errors).some(([key, error]) => error && !operationalFields.some((field) => field === key)) &&
              <FieldError>请在「Adam 和损失权重」中检查标出的参数。</FieldError>}
          </>}
        </FieldGroup>
        {failure && <ErrorDetails error={failure} />}
      </CardContent>
      <CardFooter>
        <Button type="submit" size="lg" disabled={disabledFields}>
          {pending ? "正在提交" : advanced ? (checkpoint ? "按参数续训" : "按参数训练") : (checkpoint ? "继续训练" : "一键训练")}
        </Button>
      </CardFooter>
    </Card>
  </form>;
}
