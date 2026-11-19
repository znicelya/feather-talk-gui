import { useEffect, useRef, useState } from "react";
import { api } from "../../lib/api";
import type { AppError, ProjectManifest, ProjectSummary } from "../../lib/contracts";
import { toAppError } from "../../lib/errors";

export function upsertProject(projects: ProjectSummary[], accepted: ProjectSummary) {
  return projects.some(({ id }) => id === accepted.id)
    ? projects.map((project) => project.id === accepted.id ? accepted : project)
    : [...projects, accepted];
}

export function projectCheckpoints(project: ProjectSummary | null, manifest: ProjectManifest | null) {
  return project && manifest?.id === project.id ? manifest.checkpoints : [];
}

export function useProjects() {
  const [projects, setProjects] = useState<ProjectSummary[]>([]);
  const [selected, select] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<AppError | null>(null);
  const [revision, setRevision] = useState(0);
  const mutations = useRef(0);

  useEffect(() => {
    let alive = true;
    const version = mutations.current;
    void api.listProjects().then((entries) => {
      if (!alive || version !== mutations.current) return;
      setProjects(entries);
      setError(null);
      select((id) => entries.find((entry) => entry.id === id)?.id ?? entries[0]?.id ?? null);
    }, (cause) => {
      if (alive && version === mutations.current) setError(toAppError(cause, "读取人物项目"));
    }).finally(() => { if (alive) setLoading(false); });
    return () => { alive = false; };
  }, [revision]);

  function accept(project: ProjectSummary) {
    ++mutations.current;
    setProjects((list) => upsertProject(list, project));
    select(project.id);
    // A new location can repair an index entry whose old directory was missing.
    setRevision((n) => n + 1);
  }

  return { projects, current: projects.find(({ id }) => id === selected) ?? null, select, accept, loading, error };
}

export function useProjectManifest(project: ProjectSummary | null, checkpointId?: string) {
  const [result, setResult] = useState<{ project: ProjectSummary; manifest: ProjectManifest | null; error: AppError | null }>();
  useEffect(() => {
    if (!project) return;
    let alive = true;
    void api.getProject(project.id).then((manifest) => {
      if (alive) setResult({ project, manifest, error: null });
    }, (cause) => {
      if (alive) setResult({ project, manifest: null, error: toAppError(cause, "读取人物详情") });
    });
    return () => { alive = false; };
  }, [project, checkpointId]);
  return result?.project === project ? result : undefined;
}
