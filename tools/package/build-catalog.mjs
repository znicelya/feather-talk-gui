import { spawnSync } from "node:child_process";
import { constants, copyFileSync, existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { parseArgs } from "node:util";
import { inspectExecutable, repo, targetBackend, targets } from "./stage-components.mjs";

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
const builds = targets.flatMap(({ triple, backends }) => backends.map((backend) => join(staged, triple, backend)))
  .filter((path) => existsSync(join(path, "BUILD.json")));
if (builds.length === 0) throw new Error("No staged runtime builds found");
const resourceTarget = values["resource-target"] ?? values["bundle-target"];
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
copyFileSync(join(packages, "catalog.json"), output, constants.COPYFILE_EXCL);

if (values["bundle-target"]) {
  const triple = values["bundle-target"];
  const { os, architecture, bundles } = targetBackend(triple, "cpu");
  const catalog = JSON.parse(readFileSync(output, "utf8"));
  const selection = catalog.defaults.find(({ target }) => target.os === os && target.architecture === architecture && target.backend === "cpu");
  const imports = JSON.parse(readFileSync(join(packages, "imports.json"), "utf8"));
  const resources = { [output]: "bootstrap/catalog.json" };
  for (const role of ["worker", "probe"]) {
    const pin = selection?.resources.find(({ id }) => id === role);
    const key = pin && `${pin.id}/${pin.version}`;
    const manifest = key && imports[key];
    if (!manifest) throw new Error(`CPU ${role} has not been staged for ${triple}`);
    const files = JSON.parse(readFileSync(manifest, "utf8")).relative_files;
    for (const relative of ["manifest.json", ...files]) {
      resources[join(dirname(manifest), relative)] = `bootstrap/${key}/${relative}`;
    }
  }
  writeFileSync(join(directory, "tauri-overlay.json"), JSON.stringify({ bundle: { active: true, targets: bundles, resources } }, null, 2), { flag: "wx" });
}
console.log(`Verified runtime inventory: ${output}`);
