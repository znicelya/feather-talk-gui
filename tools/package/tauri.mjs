import { spawn, spawnSync } from "node:child_process";
import { existsSync, mkdirSync, mkdtempSync } from "node:fs";
import { join, resolve } from "node:path";
import { inspectExecutable, repo, targetBackend } from "./stage-components.mjs";

const [command, ...args] = process.argv.slice(2);
const cli = join(repo, "node_modules/@tauri-apps/cli/tauri.js");
const cwd = repo;
const env = { ...process.env };

function run(program, args, stdio = "inherit") {
  const result = spawnSync(program, args, {
    cwd,
    env,
    stdio,
    encoding: "utf8",
    windowsHide: true,
  });
  if (result.error) throw result.error;
  if (result.status !== 0)
    throw new Error(
      `${program} exited ${result.status ?? result.signal}\n${result.stderr ?? ""}`.trim(),
    );
  return result.stdout;
}

function launch(args) {
  const child = spawn(process.execPath, [cli, ...args], {
    cwd,
    env,
    stdio: "inherit",
    windowsHide: true,
  });
  child.on("error", (error) => {
    console.error(error.message);
    process.exitCode = 1;
  });
  child.on("exit", (code) => {
    process.exitCode = code ?? 1;
  });
}

function node(script, args) {
  run(process.execPath, [join(repo, "tools/package", script), ...args]);
}

function requireInput(path) {
  if (!existsSync(path))
    throw new Error(
      `Missing bundle input: ${path}\nSet FT_BUNDLE_INPUTS to a prepared folder (see resources/README.md).`,
    );
  return path;
}

function prepare(options) {
  let triple;
  for (let i = 0; i < options.length; i++) {
    const arg = options[i];
    if (arg === "--target" || arg === "-t") triple = options[++i];
    else if (arg.startsWith("--target=") || arg.startsWith("-t="))
      triple = arg.split("=", 2)[1];
    else if (arg.startsWith("-t") && arg.length > 2) triple = arg.slice(2);
  }
  const native = !triple;
  triple ??= run("rustc", ["-vV"], "pipe").match(/^host:\s*(\S+)\s*$/m)?.[1];
  const platform = targetBackend(triple, "cpu");
  if (platform.os === "macos")
    env.MACOSX_DEPLOYMENT_TARGET = platform.minimum_version;

  const inputs = resolve(
    env.FT_BUNDLE_INPUTS ?? join(repo, ".artifacts/bundle-inputs"),
  );
  const imports = requireInput(join(inputs, "imports.json"));
  const base = join(inputs, "catalog.json");
  const media = join(inputs, triple, "media");
  const cuda = join(inputs, triple, "cuda-math");
  const ext = platform.os === "windows" ? ".exe" : "";
  for (const tool of ["ffmpeg", "ffprobe"])
    inspectExecutable(requireInput(join(media, "bin", tool + ext)), platform);
  for (const file of ["BUILD.json", "LICENSE"]) requireInput(join(media, file));
  const hasCuda = platform.backends.includes("cuda");
  if (hasCuda) {
    for (const file of ["nvvm/libdevice/libdevice.10.bc", "LICENSE"])
      requireInput(join(cuda, file));
  }
  if (
    platform.os ===
      { win32: "windows", linux: "linux", darwin: "macos" }[process.platform] &&
    (platform.architecture === "x86_64"
      ? process.arch === "x64"
      : process.arch === "arm64")
  ) {
    run(join(media, "bin", `ffprobe${ext}`), ["-version"], "pipe");
    const encoders = run(
      join(media, "bin", `ffmpeg${ext}`),
      ["-hide_banner", "-encoders"],
      "pipe",
    );
    if (
      !/^\s*V\S*\s+libx264\s/m.test(encoders) ||
      !/^\s*A\S*\s+aac\s/m.test(encoders)
    ) {
      throw new Error(`FFmpeg must support libx264 and AAC encoding: ${media}`);
    }
  }
  const root = join(repo, ".artifacts/bundles");
  mkdirSync(root, { recursive: true });
  const staging = env.FT_BUNDLE_STAGE_DIR
    ? resolve(env.FT_BUNDLE_STAGE_DIR)
    : mkdtempSync(join(root, `${triple}-`));
  const runtimes = env.FT_BUNDLE_RUNTIMES
    ? resolve(env.FT_BUNDLE_RUNTIMES)
    : join(staging, "runtimes");
  console.log(`Preparing FeatherTalk resources for ${triple}: ${staging}`);
  if (!env.FT_BUNDLE_RUNTIMES) {
    for (const backend of platform.backends) {
      node("build-runtime.mjs", [
        "--target",
        triple,
        "--backend",
        backend,
        "--output",
        runtimes,
        ...(env.FT_COMPONENT_SIGNING_IDENTITY
          ? ["--signing-identity", env.FT_COMPONENT_SIGNING_IDENTITY]
          : []),
      ]);
    }
  }
  node("build-catalog.mjs", [
    "--artifacts",
    runtimes,
    "--output",
    join(staging, "catalog.json"),
    "--base-imports",
    imports,
    ...(existsSync(base) ? ["--base-catalog", base] : []),
    "--bundle-target",
    triple,
    "--media",
    media,
    ...(hasCuda ? ["--cuda-math", cuda] : []),
  ]);
  return [
    ...(native ? ["--target", triple] : []),
    "--config",
    join(staging, "tauri-overlay.json"),
  ];
}

try {
  const split = args.indexOf("--");
  const options = split === -1 ? args : args.slice(0, split);
  if (
    !["build", "dev"].includes(command) ||
    options.some((arg) => ["-h", "--help", "-V", "--version"].includes(arg))
  ) {
    launch(process.argv.slice(2));
  } else {
    const overlay = prepare(options);
    const runner = split === -1 ? [] : args.slice(split + 1);
    const runnerOptions = runner.includes("--")
      ? runner.slice(0, runner.indexOf("--"))
      : runner;
    launch([
      command,
      ...options,
      ...overlay,
      "--",
      ...(runnerOptions.includes("--locked") ? [] : ["--locked"]),
      ...runner,
    ]);
  }
} catch (error) {
  console.error(error.message);
  process.exitCode = 1;
}
