import { expect, it } from "vitest";
import type { ProjectManifest, ProjectSummary } from "../../lib/contracts";
import { projectCheckpoints, upsertProject } from "./project-state";

const project: ProjectSummary = {
  id: "p1", name: "人物甲", directory: "D:\\原位置", source_video: "D:\\原位置\\人.mp4",
  latest_model: null,
};
const moved = { ...project, directory: "D:\\新位置", source_video: "D:\\新位置\\人.mp4" };

it("keeps the accepted project ID and paths when a project moves", () => {
  const other = { ...project, id: "p2" };
  expect(upsertProject([project, other], moved)).toEqual([moved, other]);
});

it("never uses the previous person's manifest after switching projects", () => {
  const old = { id: "p2", checkpoints: [{ model_ref: { project_id: "p2" }, step: 8 }] } as ProjectManifest;
  expect(projectCheckpoints(moved, old)).toEqual([]);
  expect(projectCheckpoints({ ...moved, id: "p2" }, old)).toEqual(old.checkpoints);
});
