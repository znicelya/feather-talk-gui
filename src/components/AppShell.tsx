import { useId, useRef, type ReactNode } from "react";
import { ChartNoAxesCombinedIcon, CpuIcon, FolderIcon, VideoIcon } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Separator } from "@/components/ui/separator";
import type { ProjectSummary } from "../lib/contracts";

const pages = {
  projects: { label: "人物项目", icon: FolderIcon },
  training: { label: "训练", icon: ChartNoAxesCombinedIcon },
  generation: { label: "视频生成", icon: VideoIcon },
  resources: { label: "资源与设备", icon: CpuIcon },
} as const;

export type View = keyof typeof pages;

export function AppShell({ view, onNavigate, project, task, children }: {
  view: View;
  onNavigate: (view: View) => void;
  project: ProjectSummary | null;
  task: ReactNode;
  children: ReactNode;
}) {
  const mainId = useId();
  const main = useRef<HTMLElement>(null);
  return <div className="app-shell">
    <a className="skip-link" href={`#${mainId}`} onClick={(event) => {
      event.preventDefault();
      main.current?.focus();
    }}>跳到工作区</a>
    <aside className="sidebar">
      <div className="brand">
        <img src="/feathertalk.svg" alt="" width="36" height="36" />
        <div><strong>FeatherTalk</strong><small>本地工作台</small></div>
      </div>
      <nav className="navigation" aria-label="工作区导航">
        {(Object.keys(pages) as View[]).map((page) => {
          const { icon: Icon, label } = pages[page];
          return <Button type="button" key={page} size="lg" className="justify-start gap-2.5" variant={view === page ? "secondary" : "ghost"}
            aria-current={view === page ? "page" : undefined} onClick={() => onNavigate(page)}>
            <Icon data-icon="inline-start" />{label}
          </Button>;
        })}
      </nav>
      <div className="rail-note"><Separator />一次运行一个任务</div>
    </aside>
    <div className="workspace">
      <header className="workspace-bar">
        <span className="workspace-label">工作区</span>
        <span className="workspace-project"><span>当前人物</span> <strong>{project?.name ?? "未选择"}</strong></span>
      </header>
      <main id={mainId} ref={main} tabIndex={-1} className="workspace-content">
        {children}
      </main>
      {task}
    </div>
  </div>;
}
