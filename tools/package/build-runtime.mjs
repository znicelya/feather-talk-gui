import { spawnSync } from "node:child_process";
import { copyFileSync, mkdirSync, mkdtempSync, writeFileSync } from "node:fs";
import { join, resolve } from "node:path";
import { parseArgs } from "node:util";
import { names, repo, stageComponents, targetBackend } from "./stage-components.mjs";

const { values } = parseArgs({ options: {
  target: { type: "string" }, backend: { type: "string" }, output: { type: "string" }, debug: { type: "boolean" },
  "signing-identity": { type: "string" },
} });
const { target, backend, output } = values;
if (!target || !backend || !output) throw new Error("Required: --target TRIPLE --backend BACKEND --output DIRECTORY");
const platform = targetBackend(target, backend);
const signing = values["signing-identity"];
if (signing && (platform.os !== "macos" || process.platform !== "darwin" || values.debug)) {
  throw new Error("Signing is available for release Mach-O binaries on macOS");
}
const env = { ...process.env };
if (platform.os === "macos") env.MACOSX_DEPLOYMENT_TARGET = platform.minimum_version;
const invoke = (program, args, stdio = "pipe") => {
  const result = spawnSync(program, args, { cwd: repo, encoding: "utf8", stdio, windowsHide: true, env });
  if (result.error) throw result.error;
  if (result.status !== 0) throw new Error(`${program} exited ${result.status ?? result.signal}\n${result.stderr ?? ""}`);
  return result.stdout?.trim();
};
const metadata = JSON.parse(invoke("cargo", ["metadata", "--no-deps", "--format-version", "1", "--locked"]));
const profile = values.debug ? "debug" : "release";
const feature = backend === "cpu" ? "flex" : backend;
const args = ["build", "--locked", "--target", target, "-p", "feathertalk-worker", "-p", "feathertalk-engine",
  "--no-default-features", "--features", `feathertalk-worker/${feature}`, "--bin", "feathertalk-worker", "--bin", "feathertalk-probe"];
if (!values.debug) args.push("--release");
invoke("cargo", args, "inherit");
const directory = resolve(output);
mkdirSync(directory, { recursive: true });
const input = mkdtempSync(join(directory, ".runtime-"));
for (const file of names(platform)) {
  const staged = join(input, file);
  copyFileSync(join(metadata.target_directory, target, profile, file), staged);
  if (signing) {
    invoke("codesign", ["--force", "--options", "runtime", "--timestamp", "--sign", signing, staged], "inherit");
    invoke("codesign", ["--verify", "--strict", staged], "inherit");
  }
}
writeFileSync(join(input, "BUILD.json"), JSON.stringify({
  profile, backend, target, exit: 0, base_revision: invoke("git", ["rev-parse", "HEAD"]),
  source_dirty: !!invoke("git", ["status", "--porcelain", "--untracked-files=normal"]),
  rustc: invoke("rustc", ["-vV"]), cargo_args: args,
  macos_deployment_target: platform.os === "macos" ? env.MACOSX_DEPLOYMENT_TARGET : null,
  signed_before_packaging: !!signing,
}, null, 2));
stageComponents({ target, backend, input, output: directory });
