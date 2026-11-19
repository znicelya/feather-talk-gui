import { useRef, useState, type FormEvent } from "react";
import { ArrowRightIcon, FolderOpenIcon, FolderPlusIcon, PlusIcon } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { Empty, EmptyDescription, EmptyHeader, EmptyMedia, EmptyTitle } from "@/components/ui/empty";
import { Field, FieldGroup, FieldLabel } from "@/components/ui/field";
import { Input } from "@/components/ui/input";
import { Skeleton } from "@/components/ui/skeleton";
import { CheckpointField } from "../../components/CheckpointField";
import { ErrorDetails } from "../../components/ErrorDetails";
import { PathField } from "../../components/PathField";
import { api } from "../../lib/api";
import type { AppError, CheckpointEntry, ProjectSummary } from "../../lib/contracts";
import { toAppError } from "../../lib/errors";
import type { useProjects } from "./project-state";

export function ProjectPage({ store, checkpoints, checkpointId, selectCheckpoint, manifestError, openTraining }: {
  store: ReturnType<typeof useProjects>;
  checkpoints: CheckpointEntry[];
  checkpointId: string;
  selectCheckpoint: (id: string) => void;
  manifestError: AppError | null;
  openTraining: () => void;
}) {
  const [newProject, showForm] = useState(false);
  const [name, setName] = useState("");
  const [directory, setDirectory] = useState<string | null>(null);
  const [video, setVideo] = useState<string | null>(null);
  const [pending, setPending] = useState(false);
  const [error, setError] = useState<AppError | null>(null);
  const inFlight = useRef(false);
  const current = store.current;

  async function act(stage: string, work: () => Promise<void>) {
    if (inFlight.current) return;
    inFlight.current = true;
    setPending(true);
    setError(null);
    try { await work(); }
    catch (cause) { setError(toAppError(cause, stage)); }
    finally { inFlight.current = false; setPending(false); }
  }

  function clearForm() {
    showForm(false);
    setName("");
    setDirectory(null);
    setVideo(null);
  }

  function accept(project: ProjectSummary) {
    store.accept(project);
    clearForm();
  }

  function create(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!name.trim() || !directory || !video) return;
    void act("创建人物", async () => accept(await api.createProject({
      name: name.trim(), directory, source_video: video,
    })));
  }

  return <div className="page-flow">
    <div className="page-heading">
      <div><h1>人物项目</h1><p>为每个人物选择一段训练视频，管理模型与检查点。</p></div>
      <div className="actions">
        <Button type="button" size="lg" variant="outline" disabled={pending} onClick={() => void act("打开项目", async () => {
          const path = await api.selectPath("project_directory");
          if (path) accept(await api.openProject(path));
        })}><FolderOpenIcon data-icon="inline-start" />打开项目</Button>
        {!newProject && <Button type="button" size="lg" disabled={pending} onClick={() => { setError(null); showForm(true); }}>
          <PlusIcon data-icon="inline-start" />新建人物
        </Button>}
      </div>
    </div>
    {store.error && <ErrorDetails error={store.error} />}
    {!newProject && error && <ErrorDetails error={error} />}
    {newProject ? <form onSubmit={create}>
      <Card>
        <CardHeader><CardTitle><h2>新建人物</h2></CardTitle><CardDescription>选择项目目录和一段带音轨的人物视频。</CardDescription></CardHeader>
        <CardContent>
          <FieldGroup>
            <Field data-disabled={pending || undefined}>
              <FieldLabel htmlFor="person-name">人物名称</FieldLabel>
              <Input id="person-name" required autoFocus autoComplete="off" disabled={pending} value={name}
                onChange={(event) => setName(event.target.value)} />
            </Field>
            <PathField label="项目目录" path={directory} action="选择目录" empty="选择存放检查点与缓存的目录。"
              disabled={pending} onPick={() => void act("选择项目目录", async () => {
                const path = await api.selectPath("project_directory");
                if (path) setDirectory(path);
              })} />
            <PathField label="训练视频" path={video} action={video ? "更换视频" : "选择视频"} empty="每个人物只能选择一段视频。"
              disabled={pending} onPick={() => void act("选择训练视频", async () => {
                const path = await api.selectPath("source_video");
                if (path) setVideo(path);
              })} />
          </FieldGroup>
          {error && <div className="mt-4"><ErrorDetails error={error} /></div>}
        </CardContent>
        <CardFooter className="actions">
          <Button size="lg" disabled={pending || !name.trim() || !directory || !video}>创建人物</Button>
          <Button type="button" variant="outline" size="lg" disabled={pending} onClick={() => { clearForm(); setError(null); }}>返回项目列表</Button>
        </CardFooter>
      </Card>
    </form> : store.loading ? <Skeleton className="h-48 w-full" aria-label="正在读取人物项目" /> : !store.projects.length ? (
      <Card>
        <CardContent className="empty-workspace">
          <Empty>
            <EmptyHeader>
              <EmptyMedia variant="icon"><FolderPlusIcon /></EmptyMedia>
              <EmptyTitle><h2>{store.error ? "未能读取项目列表" : "还没有人物项目"}</h2></EmptyTitle>
              <EmptyDescription>{store.error ? "可以打开项目所在目录，重新登记已移动的人物。" : "新建一个人物，或打开已有的 FeatherTalk 项目目录。"}</EmptyDescription>
            </EmptyHeader>
          </Empty>
          <ol className="workflow" aria-label="制作顺序">
            <li><span>01</span>选择人物视频</li>
            <li><span>02</span>训练人物模型</li>
            <li><span>03</span>导入音频，导出视频</li>
          </ol>
        </CardContent>
      </Card>
    ) : <div className="project-grid">
      <Card>
        <CardHeader><CardTitle><h2>项目列表</h2></CardTitle><CardDescription>{store.projects.length} 个人物</CardDescription></CardHeader>
        <CardContent>
          <ul className="project-list">
            {store.projects.map((project) => <li key={project.id}>
              <Button size="lg" type="button" variant={project.id === current?.id ? "secondary" : "ghost"}
                className="h-auto w-full justify-start py-3" aria-pressed={project.id === current?.id}
                disabled={pending} onClick={() => store.select(project.id)}>
                <FolderOpenIcon data-icon="inline-start" /><span className="path-text text-left">{project.name}</span>
              </Button>
            </li>)}
          </ul>
        </CardContent>
      </Card>
      {current && <Card>
        <CardHeader>
          <CardTitle><h2 className="path-text">{current.name}</h2></CardTitle>
          <CardDescription><Badge variant="secondary">{checkpoints.length ? `${checkpoints.length} 个检查点` : "人物项目"}</Badge></CardDescription>
        </CardHeader>
        <CardContent className="grid gap-5">
          <dl className="detail-list">
            <div><dt>项目目录</dt><dd className="path-text">{current.directory}</dd></div>
            <div><dt>训练视频</dt><dd className="path-text">{current.source_video}</dd></div>
          </dl>
          {manifestError ? <ErrorDetails error={manifestError} /> : <CheckpointField
            checkpoints={checkpoints} value={checkpointId} onChange={selectCheckpoint} />}
        </CardContent>
        <CardFooter><Button type="button" size="lg" variant="outline" onClick={openTraining}>前往训练<ArrowRightIcon data-icon="inline-end" /></Button></CardFooter>
      </Card>}
    </div>}
  </div>;
}
