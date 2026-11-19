import { useState } from "react";
import { ChevronDownIcon, CircleAlertIcon, CopyIcon } from "lucide-react";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";
import type { AppError } from "../lib/contracts";
import { errorDescription } from "../lib/errors";

export function ErrorDetails({ error }: { error: AppError }) {
  const [copy, setCopy] = useState<{ error: AppError; status: string }>();
  const [title, suggestion] = errorDescription(error);
  const actions = error.actions.length ? error.actions : suggestion ? [suggestion] : [];

  async function copyDetails() {
    try {
      await navigator.clipboard.writeText([
        `${error.stage} / ${error.code}`, error.message, ...error.evidence, ...actions, error.log_path,
      ].filter(Boolean).join("\n"));
      setCopy({ error, status: "已复制诊断。" });
    } catch {
      setCopy({ error, status: "无法访问剪贴板，可直接选中下方文字复制。" });
    }
  }

  return (
    <div className="error-panel">
      <Alert variant="destructive">
        <CircleAlertIcon />
        <AlertTitle>{title}</AlertTitle>
        <AlertDescription className="min-w-0">
          <p className="diagnostic">{error.message}</p>
          {actions.length > 0 && <ul className="error-actions">{actions.map((action, i) => <li key={i}>{action}</li>)}</ul>}
        </AlertDescription>
      </Alert>
      <Collapsible>
        <CollapsibleTrigger asChild><Button variant="ghost" type="button">详细诊断<ChevronDownIcon data-icon="inline-end" /></Button></CollapsibleTrigger>
        <CollapsibleContent className="error-details">
        <dl className="detail-list">
          <div><dt>阶段</dt><dd className="mono">{error.stage}</dd></div>
          <div><dt>错误码</dt><dd className="mono">{error.code}</dd></div>
          {error.log_path && <div><dt>日志文件</dt><dd className="path-text">{error.log_path}</dd></div>}
        </dl>
        {error.evidence.length > 0 && <div className="raw-evidence" aria-label="原始证据">
          {error.evidence.map((item, i) => <pre key={i}>{item}</pre>)}
        </div>}
        <div className="actions">
          <Button type="button" variant="outline" onClick={() => void copyDetails()}><CopyIcon data-icon="inline-start" />复制诊断</Button>
          <small role="status">{copy?.error === error ? copy.status : ""}</small>
        </div>
        </CollapsibleContent>
      </Collapsible>
    </div>
  );
}
