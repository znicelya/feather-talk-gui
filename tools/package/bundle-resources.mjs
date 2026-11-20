import { createHash } from "node:crypto";
import { closeSync, fstatSync, openSync, readFileSync, readSync, realpathSync } from "node:fs";
import { dirname, isAbsolute, join, relative, sep } from "node:path";
import { isDeepStrictEqual } from "node:util";
import { inspectExecutable, names } from "./stage-components.mjs";

function portablePath(path) {
  if (/[\\:*?"<>|\x00-\x1f]/.test(path) || path.split("/").some((part) =>
    !part || /[. ]$/.test(part) || /^(con|prn|aux|nul|com\d|lpt\d)(\.|$)/i.test(part))) {
    throw new Error(`Invalid bundle path: ${path}`);
  }
  return path;
}

function pinKey({ id, version }) {
  if (![id, version].every((part) => /^[A-Za-z0-9_-][A-Za-z0-9_.-]*$/.test(part))) {
    throw new Error(`Invalid resource pin: ${id}/${version}`);
  }
  return portablePath(`${id}/${version}`);
}

function verifyFile(root, name, { size, sha256 }) {
  const path = join(root, portablePath(name));
  const rel = relative(root, realpathSync(path));
  if (isAbsolute(rel) || rel === ".." || rel.startsWith(`..${sep}`)) {
    throw new Error(`File escapes its package: ${path}`);
  }
  const fd = openSync(path, "r");
  try {
    const stat = fstatSync(fd);
    if (!stat.isFile() || stat.size !== size) throw new Error(`Size mismatch: ${path}`);
    const hash = createHash("sha256");
    const buffer = Buffer.alloc(1024 * 1024);
    for (let count; (count = readSync(fd, buffer)) !== 0;) hash.update(buffer.subarray(0, count));
    if (hash.digest("hex") !== sha256) throw new Error(`SHA-256 mismatch: ${path}`);
  } finally { closeSync(fd); }
  return path;
}

const modelFiles = ["feather_hubert", "scrfd_2_5g", "pfld_ghost_one", "mean_face", "vgg19"].flatMap((model) =>
  ["manifest.json", "config.json", model === "scrfd_2_5g" ? "model.bpk" : "model.safetensors"].map((file) => `${model}/${file}`));

function packageFiles(component, manifestPath, expected) {
  const { manifest, supplemental, executables, notices } = component;
  const root = realpathSync(dirname(manifestPath));
  const files = new Map([["manifest.json", { size: component.manifest_size, sha256: component.manifest_sha256 }]]);
  const folded = new Set(["manifest.json"]);
  function add(name, spec) {
    portablePath(name);
    if (name.toLowerCase() === ".component.json" || folded.has(name.toLowerCase())) {
      throw new Error(`Conflicting package inventory: ${manifestPath}: ${name}`);
    }
    folded.add(name.toLowerCase());
    files.set(name, spec);
  }
  for (const name of manifest.relative_files) add(name, { size: manifest.sizes[name], sha256: manifest.hashes[name] });
  for (const [name, spec] of Object.entries(supplemental)) add(name, spec);
  for (const name of [...expected.files, ...notices, ...executables]) {
    if (!files.has(name)) throw new Error(`Missing ${name} in ${manifestPath}`);
  }
  for (const name of expected.executables) {
    if (!executables.includes(name)) throw new Error(`Missing executable ${name} in ${manifestPath}`);
  }
  const checked = [...files].map(([name, spec]) => [name, verifyFile(root, name, spec)]);
  if (!isDeepStrictEqual(JSON.parse(readFileSync(join(root, "manifest.json"), "utf8")), manifest)) {
    throw new Error(`Manifest differs from catalog: ${manifestPath}`);
  }
  for (const name of executables) inspectExecutable(join(root, name), component.target);
  return checked;
}

export function bundleResources(catalog, imports, platform, output) {
  const { triple, os, architecture, backends, bundles } = platform;
  const resources = new Map([[output, "bootstrap/catalog.json"]]);
  const available = new Map(catalog.components.map((component) => [pinKey(component.pin), component]));
  if (available.size !== catalog.components.length) throw new Error("Duplicate catalog pin");
  const components = new Map();
  const defaults = [];
  const shared = new Map();
  const [worker, probe] = names(platform);
  const ext = os === "windows" ? ".exe" : "";
  for (const backend of backends) {
    const target = { os, architecture, backend };
    const selections = catalog.defaults.filter((s) => isDeepStrictEqual(s.target, target));
    if (selections.length !== 1) throw new Error(`Expected one default selection for ${triple}/${backend}`);
    const selection = selections[0];
    defaults.push(selection);
    const roles = [
      { id: "worker", kind: "worker", target, executables: [worker], files: ["BUILD.json", "Cargo.lock"] },
      { id: "probe", kind: "probe", target, executables: [probe], files: ["BUILD.json", "Cargo.lock"] },
      { id: "media", kind: "media", target: { ...target, backend: "any" }, executables: [`bin/ffmpeg${ext}`, `bin/ffprobe${ext}`], files: ["LICENSE"] },
      { id: "public-assets", kind: "model_bundle", target: null, executables: [], files: modelFiles },
    ];
    if (backend === "cuda") roles.push({ id: "cuda-math", kind: "cuda_runtime", target, executables: [], files: ["nvvm/libdevice/libdevice.10.bc", "LICENSE"] });
    for (const role of roles) {
      const pins = selection.resources.filter(({ id }) => id === role.id);
      if (pins.length !== 1) throw new Error(`Expected one ${role.id} pin for ${triple}/${backend}`);
      const key = pinKey(pins[0]);
      const component = available.get(key);
      const manifest = imports[key];
      if (!component || !manifest) throw new Error(`Missing local ${key} package for ${triple}/${backend}`);
      if (component.kind !== role.kind || !isDeepStrictEqual(component.target, role.target)) {
        throw new Error(`Incompatible ${key} package for ${triple}/${backend}`);
      }
      if (role.id === "public-assets" || role.id === "media") {
        if (shared.has(role.id) && shared.get(role.id) !== key) throw new Error(`Use one ${role.id} pin for all ${triple} backends`);
        shared.set(role.id, key);
      }
      if (components.has(key)) continue;
      for (const [name, source] of packageFiles(component, manifest, role)) {
        if (resources.has(source)) throw new Error(`File is mapped to more than one destination: ${source}`);
        resources.set(source, `bootstrap/${key}/${name}`);
      }
      components.set(key, component);
    }
    if (selection.resources.length !== roles.length) throw new Error(`Unexpected resource pins for ${triple}/${backend}`);
  }
  return {
    catalog: { ...catalog, components: [...components.values()], defaults },
    overlay: { bundle: { active: true, targets: bundles, resources: Object.fromEntries(resources) } },
  };
}
