import { useId } from "react";
import { Field, FieldDescription, FieldLabel } from "@/components/ui/field";
import { NativeSelect, NativeSelectOption } from "@/components/ui/native-select";
import type { CheckpointEntry } from "../lib/contracts";

export function CheckpointField({ checkpoints, value, onChange, disabled }: {
  checkpoints: CheckpointEntry[];
  value: string;
  onChange: (id: string) => void;
  disabled?: boolean;
}) {
  const id = useId();
  const disabledField = disabled || checkpoints.length === 0;
  return <Field data-disabled={disabledField || undefined}>
    <FieldLabel htmlFor={id}>人物检查点</FieldLabel>
    <NativeSelect id={id} value={value} onChange={(event) => onChange(event.target.value)} disabled={disabledField}>
      {!checkpoints.length && <NativeSelectOption value="">没有可用的检查点</NativeSelectOption>}
      {checkpoints.map(({ model_ref, epoch, step }) => <NativeSelectOption key={model_ref.checkpoint_id} value={model_ref.checkpoint_id}>
        第 {epoch} 轮 · 步 {step} · {model_ref.checkpoint_id.slice(0, 8)}
      </NativeSelectOption>)}
    </NativeSelect>
    {checkpoints.find(({ model_ref }) => model_ref.checkpoint_id === value)?.step === 0 && <FieldDescription>
      该检查点可以续训，完成一个训练步后即可用于生成视频。
    </FieldDescription>}
  </Field>;
}
