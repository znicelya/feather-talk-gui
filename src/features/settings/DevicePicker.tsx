import { useId } from "react";
import { Field, FieldDescription, FieldLabel } from "@/components/ui/field";
import { NativeSelect, NativeSelectOption } from "@/components/ui/native-select";
import type { DeviceDescriptor, DevicePolicy } from "../../lib/contracts";
import { backendLabel, deviceKey, deviceName } from "./device-labels";

export function DevicePicker({ devices, value, onChange, disabled }: {
  devices: DeviceDescriptor[];
  value: DevicePolicy;
  onChange: (value: DevicePolicy) => void;
  disabled?: boolean;
}) {
  const id = useId();
  const key = value.policy === "auto" ? "auto" : deviceKey({ backend: value.request.backend, id: value.request.device_id });
  const missing = key !== "auto" && !devices.some((device) => deviceKey(device) === key);
  return <Field data-disabled={disabled || undefined}>
    <FieldLabel htmlFor={id}>计算设备</FieldLabel>
    <NativeSelect id={id} disabled={disabled} value={key} onChange={(event) => {
      const selected = devices.find((device) => deviceKey(device) === event.target.value);
      if (selected) onChange({ policy: "explicit", request: { backend: selected.backend, device_id: selected.id } });
      else if (event.target.value === "auto") onChange({ policy: "auto" });
    }}>
      <NativeSelectOption value="auto">自动选择可用设备</NativeSelectOption>
      {missing && <NativeSelectOption value={key} disabled>上次选择的设备不可用</NativeSelectOption>}
      {devices.map((device) => <NativeSelectOption key={deviceKey(device)} value={deviceKey(device)}>
        {deviceName(device)} · {backendLabel[device.backend]}
      </NativeSelectOption>)}
    </NativeSelect>
    {missing && <FieldDescription>刷新设备，或改用自动选择或其他已检测到的设备。</FieldDescription>}
  </Field>;
}
