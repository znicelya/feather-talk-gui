import { spawnSync } from "node:child_process";
import { constants, copyFileSync, existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { parseArgs } from "node:util";
import { bundleResources } from "./bundle-resources.mjs";
import { inspectExecutable, names, repo, targetBackend, targets } from "./stage-components.mjs";

const { values } = parseArgs({ options: {
  ...Object.fromEntries(["artifacts", "output", "base-catalog", "base-imports", "bundle-target", "python", "resource-target"]
    .map((key) => [key, { type: "string" }])),
  media: { type: "string" }, "cuda-math": { type: "string" },
} });
if (!values.artifacts || !values.output) throw new Error("Required: --artifacts DIRECTORY --output CATALOG.json");
const staged = resolve(values.artifacts);
const output = resolve(values.output);
const directory = dirname(output);
const packages = join(directory, "packages");
if (existsSync(output) || existsSync(packages)) throw new Error("Choose a new catalog staging directory");
const bundleTarget = values["bundle-target"] && targetBackend(values["bundle-target"], "cpu");
const builds = (bundleTarget ? [bundleTarget] : targets).flatMap((target) => target.backends.flatMap((backend) => {
  const path = join(staged, target.triple, backend);
  const record = join(path, "BUILD.json");
  if (!existsSync(record)) {
    if (bundleTarget) throw new Error(`Missing runtime pair for ${target.triple}/${backend}: ${path}`);
    return [];
  }
  const stamp = JSON.parse(readFileSync(record, "utf8"));
  if (stamp.target !== target.triple || stamp.backend !== backend || stamp.exit !== 0 || !stamp.base_revision) {
    throw new Error(`Invalid build record for ${target.triple}/${backend}: ${record}`);
  }
  for (const name of names(target)) inspectExecutable(join(path, name), target);
  return [path];
}));
if (builds.length === 0) throw new Error("No staged runtime builds found");
const resourceTarget = values["resource-target"] ?? values["bundle-target"];
if (bundleTarget && resourceTarget !== bundleTarget.triple) throw new Error("Bundle and resource target must match");
const optionalComponents = [];
if (values.media) {
  const target = targetBackend(resourceTarget, "cpu");
  for (const name of ["ffmpeg", "ffprobe"]) {
    inspectExecutable(join(resolve(values.media), "bin", name + (target.os === "windows" ? ".exe" : "")), target);
  }
  optionalComponents.push("--media", resourceTarget, resolve(values.media));
}
if (values["cuda-math"]) {
  targetBackend(resourceTarget, "cuda");
  optionalComponents.push("--cuda-math", resourceTarget, resolve(values["cuda-math"]));
}
mkdirSync(directory, { recursive: true });
const emptyImports = join(directory, "empty-imports.json");
if (!values["base-imports"]) writeFileSync(emptyImports, "[]\n", { flag: "wx" });
const built = spawnSync(values.python ?? process.env.FT_PACKAGING_PYTHON ?? "python", ["-B", join(repo, "tools/resources/package_runtime.py"),
  "--base-catalog", resolve(values["base-catalog"] ?? join(repo, "resources/development-catalog.json")),
  "--base-imports", resolve(values["base-imports"] ?? emptyImports), "--lockfile", join(repo, "Cargo.lock"), "--output", packages,
  ...builds.flatMap((path) => ["--build", path]),
  ...optionalComponents,
], { cwd: repo, stdio: "inherit", windowsHide: true });
if (built.error) throw built.error;
if (built.status !== 0) throw new Error(`Could not build verified catalog (exit ${built.status})`);
const packageCatalog = join(packages, "catalog.json");
if (bundleTarget) {
  const fullCatalog = JSON.parse(readFileSync(packageCatalog, "utf8"));
  const imports = JSON.parse(readFileSync(join(packages, "imports.json"), "utf8"));
  const { catalog, overlay } = bundleResources(fullCatalog, imports, bundleTarget, output);
  writeFileSync(output, JSON.stringify(catalog, null, 2) + "\n", { flag: "wx" });
  writeFileSync(join(directory, "tauri-overlay.json"), JSON.stringify(overlay, null, 2) + "\n", { flag: "wx" });
} else {
  copyFileSync(packageCatalog, output, constants.COPYFILE_EXCL);
}
console.log(`Verified runtime inventory: ${output}`);
