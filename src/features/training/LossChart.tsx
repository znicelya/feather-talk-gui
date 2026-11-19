import { useMemo, useState, type ReactNode } from "react";
import { ChartNoAxesCombinedIcon } from "lucide-react";
import { CartesianGrid, Line, LineChart, XAxis, YAxis } from "recharts";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { ChartContainer, ChartTooltip, ChartTooltipContent, type ChartConfig } from "@/components/ui/chart";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";
import { Empty, EmptyDescription, EmptyHeader, EmptyMedia, EmptyTitle } from "@/components/ui/empty";
import { Skeleton } from "@/components/ui/skeleton";
import { backendLabel, deviceName } from "@/features/settings/device-labels";
import type { TrainingMetrics } from "@/lib/contracts";
import { lossLabels, selectChartPoints } from "./metrics";

const series = {
  total: { label: lossLabels.total, color: "var(--chart-1)" },
  full: { label: lossLabels.full, color: "var(--chart-2)" },
  mouth: { label: lossLabels.mouth, color: "var(--chart-3)" },
  temporal: { label: lossLabels.temporal, color: "var(--chart-4)" },
  temporal_mouth: { label: lossLabels.temporal_mouth, color: "var(--chart-4)" },
  perceptual: { label: lossLabels.perceptual, color: "var(--chart-5)" },
  mouth_gradient: { label: lossLabels.mouth_gradient, color: "var(--chart-3)" },
} satisfies ChartConfig;
const parts = ["full", "mouth", "temporal", "temporal_mouth", "perceptual", "mouth_gradient"] as const;
const number = new Intl.NumberFormat("zh-CN", { maximumSignificantDigits: 6 });

function elapsed(ms: number) {
  const seconds = Math.floor(ms / 1_000);
  const minutes = Math.floor(seconds / 60);
  return minutes ? `${minutes} 分 ${seconds % 60} 秒` : `${(ms / 1_000).toFixed(1)} 秒`;
}

export function LossChart({ points, loading, active, scope }: {
  points: TrainingMetrics[];
  loading: boolean;
  active: boolean;
  scope?: ReactNode;
}) {
  const [expanded, setExpanded] = useState(false);
  const latest = points[points.length - 1];
  const data = useMemo(() => selectChartPoints(points, 2000).map((point) => ({
    ...point.losses, step: point.step, epoch: point.epoch, elapsed_ms: point.elapsed_ms,
  })), [points]);

  return <Card className="min-w-0">
    <CardHeader>
      <CardTitle><h2>Loss 曲线</h2></CardTitle>
      <CardDescription>横轴为累计训练耗时，悬停或用方向键查看采样步。</CardDescription>
    </CardHeader>
    <CardContent className="grid min-w-0 gap-4">
      {scope}
      {latest ? <>
        <div className="flex flex-wrap items-baseline gap-x-4 gap-y-2">
          <span className="text-muted-foreground">{lossLabels.total}</span>
          <strong className="text-3xl tabular-nums">{number.format(latest.losses.total)}</strong>
          <Badge variant="secondary">{active ? "实时采样" : "历史采样"}</Badge>
        </div>
        <Collapsible open={expanded} onOpenChange={setExpanded} className="min-w-0">
          <ChartContainer config={series} className="h-60 min-h-48 w-full min-w-0" aria-label="训练损失随累计耗时变化">
            <LineChart accessibilityLayer data={data} margin={{ left: 0, right: 14, top: 8, bottom: 8 }}>
              <CartesianGrid vertical={false} />
              <XAxis dataKey="elapsed_ms" type="number" domain={["dataMin", "dataMax"]} tickLine={false} axisLine={false}
                tickFormatter={(value) => elapsed(Number(value))} minTickGap={24} />
              <YAxis width={48} tickLine={false} axisLine={false} tickFormatter={(value) => number.format(Number(value))} />
              <ChartTooltip isAnimationActive={false} content={<ChartTooltipContent
                labelFormatter={(_, payload) => {
                  const point = payload[0]?.payload as typeof data[number] | undefined;
                  return point ? `第 ${point.epoch} 轮 · 步 ${point.step} · ${elapsed(point.elapsed_ms)}` : undefined;
                }} />} />
              <Line dataKey="total" type="linear" stroke="var(--color-total)" strokeWidth={2} dot={data.length === 1} isAnimationActive={false} />
              {expanded && parts.map((key) => <Line key={key} dataKey={key} type="linear" stroke={`var(--color-${key})`}
                strokeDasharray={key === "temporal_mouth" || key === "mouth_gradient" ? "4 3" : undefined}
                dot={data.length === 1} isAnimationActive={false} />)}
            </LineChart>
          </ChartContainer>
          <CollapsibleTrigger asChild><Button type="button" variant="outline">分项 loss</Button></CollapsibleTrigger>
          <CollapsibleContent>
            <dl className="mt-4 grid grid-cols-[repeat(auto-fit,minmax(min(100%,8rem),1fr))] gap-3 text-sm">
              {parts.map((key) => <div key={key}><dt className="text-muted-foreground">{lossLabels[key]}</dt>
                <dd className="tabular-nums">{number.format(latest.losses[key])}</dd></div>)}
            </dl>
          </CollapsibleContent>
        </Collapsible>
      </> : loading ? <Skeleton className="h-44 w-full" aria-label="正在读取训练历史" /> : <Empty>
        <EmptyHeader>
          <EmptyMedia variant="icon"><ChartNoAxesCombinedIcon /></EmptyMedia>
          <EmptyTitle>{active ? "等待第一个训练采样" : "还没有 loss 记录"}</EmptyTitle>
          <EmptyDescription>{active ? "准备完成并执行训练步后，曲线会自动更新。" : "开始训练或选择已有检查点的历史。"}</EmptyDescription>
        </EmptyHeader>
      </Empty>}
    </CardContent>
    {latest && <CardFooter className="grid gap-3">
      <dl className="grid grid-cols-[repeat(auto-fit,minmax(min(100%,6rem),1fr))] gap-3 text-sm">
        <div><dt className="text-muted-foreground">轮次 / 步</dt><dd>{latest.epoch} / {latest.step}</dd></div>
        <div><dt className="text-muted-foreground">实际批量</dt><dd>{latest.batch_size} 对</dd></div>
        <div><dt className="text-muted-foreground">学习率</dt><dd>{latest.lr}</dd></div>
        <div><dt className="text-muted-foreground">累计耗时</dt><dd>{elapsed(latest.elapsed_ms)}</dd></div>
      </dl>
      <p className="path-text">{deviceName(latest.device)} · {backendLabel[latest.device.backend]} · {latest.precision.toUpperCase()}</p>
    </CardFooter>}
  </Card>;
}
