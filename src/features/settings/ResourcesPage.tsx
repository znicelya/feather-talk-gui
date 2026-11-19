import { useId, useRef, useState } from "react";
import { CheckCircle2Icon, CpuIcon, DownloadIcon, InfoIcon, PackageOpenIcon, RefreshCwIcon } from "lucide-react";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { Empty, EmptyDescription, EmptyHeader, EmptyMedia, EmptyTitle } from "@/components/ui/empty";
import { Field, FieldGroup, FieldLabel } from "@/components/ui/field";
import { NativeSelect, NativeSelectOption } from "@/components/ui/native-select";
import { Separator } from "@/components/ui/separator";
import { Skeleton } from "@/components/ui/skeleton";
import { ErrorDetails } from "../../components/ErrorDetails";
import { api } from "../../lib/api";
import type { AppError, DeviceDescriptor, DevicePolicy } from "../../lib/contracts";
import { toAppError } from "../../lib/errors";
import type { StartTask } from "../../lib/task-control";
import { DevicePicker } from "./DevicePicker";
import { backendLabel, deviceKey, deviceName, targetKey } from "./device-labels";
import { missingSelection, type useResourceChecks } from "./resource-state";

const resourceLabels: Record<string, string> = {
  "public-assets": "公共模型", media: "媒体组件", worker: "运行组件", probe: "批量探测组件", "cuda-math": "CUDA 数学库",
};

export function ResourcesPage({ resources, devices, device, onDevice, start, busy }: {
  resources: ReturnType<typeof useResourceChecks>;
  devices: { items: DeviceDescriptor[]; loading: boolean; error: AppError | null; refresh: () => void };
  device: DevicePolicy;
  onDevice: (policy: DevicePolicy) => void;
  start: StartTask;
  busy: boolean;
}) {
  const [target, setTarget] = useState("");
  const [pending, setPending] = useState(false);
  const [error, setError] = useState<AppError | null>(null);
  const inFlight = useRef(false);
  const targetId = useId();
  const current = resources.checks.find(({ selection }) => targetKey(selection.target) === target)
    ?? resources.checks.find(({ inventory }) => inventory?.ready) ?? resources.checks[0];
  const inventory = current?.inventory;
  const missing = current ? missingSelection(current) : null;
  const unconfigured = inventory?.items.some(({ resource }) => resource.version === "unconfigured");
  const blocked = busy || pending || resources.loading;

  async function prepare(importLocal: boolean) {
    if (!current || blocked || inFlight.current) return;
    inFlight.current = true;
    setPending(true);
    setError(null);
    try {
      if (importLocal) {
        const manifest = await api.selectPath("resource_manifest");
        if (manifest) await start(() => api.importResource(current.selection.target, manifest));
      } else if (missing) await start(() => api.prepareResources(missing));
    } catch (cause) { setError(toAppError(cause, importLocal ? "导入资源包" : "准备资源")); }
    finally { inFlight.current = false; setPending(false); }
  }

  return <div className="page-flow">
    <div className="page-heading">
      <div><h1>资源与设备</h1><p>准备本机使用的模型和组件，检查计算设备。</p></div>
      <Button variant="outline" size="lg" type="button" disabled={blocked} onClick={resources.refresh}>
        <RefreshCwIcon data-icon="inline-start" />重新校验资源
      </Button>
    </div>
    {resources.error && <ErrorDetails error={resources.error} />}
    {error && <ErrorDetails error={error} />}
    <Card>
      <CardHeader><CardTitle><h2>本地资源</h2></CardTitle><CardDescription>按所选后端校验版本和文件完整性。</CardDescription></CardHeader>
      <CardContent className="grid min-w-0 gap-4">
        {resources.loading && !current ? <Skeleton className="h-28 w-full" aria-label="正在校验本地资源" /> : current ? <>
          <FieldGroup>
            <Field data-disabled={pending || undefined}>
              <FieldLabel htmlFor={targetId}>资源后端</FieldLabel>
              <NativeSelect id={targetId} value={targetKey(current.selection.target)} disabled={pending} onChange={(event) => {
                setTarget(event.target.value); setError(null);
              }}>
                {resources.checks.map(({ selection }) => <NativeSelectOption key={targetKey(selection.target)} value={targetKey(selection.target)}>
                  {backendLabel[selection.target.backend]} · {selection.target.architecture}
                </NativeSelectOption>)}
              </NativeSelect>
            </Field>
          </FieldGroup>
          {current.error && <ErrorDetails error={current.error} />}
          {inventory && <>
            <Alert role="status">
              {resources.loading ? <RefreshCwIcon /> : inventory.ready ? <CheckCircle2Icon /> : <InfoIcon />}
              <AlertTitle>{resources.loading ? "正在校验本地资源" : inventory.ready ? "资源已就绪，可离线使用" : "资源未就绪"}</AlertTitle>
              {!resources.loading && !inventory.ready && <AlertDescription>{unconfigured
                ? "此后端仍缺少已配置的组件版本。可准备已配置的文件，或切换到其他后端。"
                : "准备缺失的组件，或导入与资源清单匹配的本地包。"}</AlertDescription>}
            </Alert>
            <ul className="resource-list">
              {inventory.items.map(({ resource, identity, path, ready, reason }, i) => <li key={`${resource.id}/${resource.version}/${i}`}>
                {i > 0 && <Separator />}
                <div className="resource-row">
                  <div className="min-w-0">
                    <strong>{resourceLabels[resource.id] ?? resource.id}</strong>
                    <p className="mono muted path-text">{resource.version === "unconfigured" ? "版本未配置" : resource.version}
                      {identity && ` · ${identity.architecture_version}`}</p>
                    {reason && <p className="diagnostic muted">{reason}</p>}
                    {path && <p className="path-text muted">{path}</p>}
                  </div>
                  <Badge variant={ready ? "secondary" : "outline"}>{ready ? "已校验" : "未就绪"}</Badge>
                </div>
              </li>)}
            </ul>
          </>}
        </> : <Empty>
          <EmptyHeader><EmptyMedia variant="icon"><PackageOpenIcon /></EmptyMedia><EmptyTitle>没有可用的本机资源清单</EmptyTitle>
            <EmptyDescription>安装与当前系统和架构匹配的组件清单，再重新校验。</EmptyDescription></EmptyHeader>
        </Empty>}
      </CardContent>
      {current && <CardFooter className="actions">
        {missing && <Button type="button" size="lg" disabled={blocked} onClick={() => void prepare(false)}><DownloadIcon data-icon="inline-start" />准备资源</Button>}
        <Button type="button" size="lg" variant="outline" disabled={blocked} onClick={() => void prepare(true)}>
          <PackageOpenIcon data-icon="inline-start" />导入本地资源包
        </Button>
      </CardFooter>}
    </Card>
    <Card>
      <CardHeader><CardTitle><h2>计算设备</h2></CardTitle><CardDescription>只列出本机已校验的运行组件实际报告的设备。</CardDescription></CardHeader>
      <CardContent className="grid gap-4">
        <FieldGroup><DevicePicker devices={devices.items} value={device} onChange={onDevice} disabled={busy || devices.loading} /></FieldGroup>
        {devices.error && <ErrorDetails error={devices.error} />}
        {devices.loading ? <p className="muted" role="status">正在检测设备…</p> : !devices.items.length && !devices.error &&
          <Alert role="status"><CpuIcon /><AlertTitle>尚未检测到计算设备</AlertTitle>
            <AlertDescription>准备所需后端的资源后刷新。也可以准备 CPU 资源。</AlertDescription></Alert>}
        {devices.items.length > 0 && <ul className="resource-list">
          {devices.items.map((descriptor, i) => <li key={deviceKey(descriptor)}>
            {i > 0 && <Separator />}
            <div className="resource-row">
              <div className="min-w-0"><strong className="path-text" title={descriptor.name}>{deviceName(descriptor)}</strong>
                <p className="muted path-text">{[descriptor.id, descriptor.vendor, descriptor.driver,
                  descriptor.memory_bytes === null ? null : `${(descriptor.memory_bytes / 2 ** 30).toFixed(1)} GiB`,
                ].filter(Boolean).join(" · ")}</p>
              </div>
              <Badge variant="secondary">{backendLabel[descriptor.backend]}</Badge>
            </div>
          </li>)}
        </ul>}
      </CardContent>
      <CardFooter><Button type="button" size="lg" variant="outline" disabled={blocked || devices.loading} onClick={devices.refresh}>
        <RefreshCwIcon data-icon="inline-start" />刷新设备
      </Button></CardFooter>
    </Card>
  </div>;
}
