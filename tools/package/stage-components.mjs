import assert from "node:assert/strict";
import { closeSync, constants, copyFileSync, existsSync, mkdirSync, openSync, readFileSync, readSync } from "node:fs";
import { join, resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { parseArgs } from "node:util";

export const repo = fileURLToPath(new URL("../../", import.meta.url));
export const targets = JSON.parse(readFileSync(join(repo, "resources/targets.json"), "utf8")).targets;

export function targetBackend(triple, backend) {
  const target = targets.find((row) => row.triple === triple && row.backends.includes(backend));
  if (!target) throw new Error(`Unsupported target/backend: ${triple}/${backend}`);
  return target;
}

export function names(target) {
  const ext = target.os === "windows" ? ".exe" : "";
  return [`feathertalk-worker${ext}`, `feathertalk-probe${ext}`];
}

export function inspectExecutable(path, target) {
  const file = openSync(path, "r");
  try {
    const header = Buffer.alloc(64);
    assert.equal(readSync(file, header, 0, header.length, 0), header.length, `Short executable: ${path}`);
    if (target.os === "windows") {
      assert.equal(header.toString("ascii", 0, 2), "MZ", path);
      const signature = Buffer.alloc(26);
      assert.equal(readSync(file, signature, 0, signature.length, header.readUInt32LE(60)), signature.length, path);
      assert.equal(signature.readUInt32LE(0), 0x4550, path);
      assert.equal(signature.readUInt16LE(4), 0x8664, `Expected x64 PE: ${path}`);
      assert.equal(signature.readUInt16LE(24), 0x20b, `Expected PE32+ executable: ${path}`);
    } else if (target.os === "linux") {
      assert.equal(header.readUInt32BE(0), 0x7f454c46, path);
      assert.deepEqual([...header.subarray(4, 6)], [2, 1], path);
      assert.equal(header.readUInt16LE(18), 62, `Expected x64 ELF: ${path}`);
      assert.ok([2, 3].includes(header.readUInt16LE(16)), `Expected ELF executable: ${path}`);
    } else {
      assert.equal(header.readUInt32LE(0), 0xfeedfacf, `Expected thin 64-bit Mach-O: ${path}`);
      assert.equal(header.readUInt32LE(4), target.architecture === "aarch64" ? 0x0100000c : 0x01000007, path);
      assert.equal(header.readUInt32LE(12), 2, `Expected Mach-O executable: ${path}`);
    }
  } finally { closeSync(file); }
}

export function stageComponents({ target: triple, backend, input, output }) {
  const target = targetBackend(triple, backend);
  const source = resolve(input);
  const destination = resolve(output, triple, backend);
  const stamp = JSON.parse(readFileSync(join(source, "BUILD.json"), "utf8"));
  if (stamp.target !== triple || stamp.backend !== backend || stamp.exit !== 0 || !stamp.base_revision) {
    throw new Error(`Expected a successful build record for ${triple}/${backend}`);
  }
  const exes = names(target);
  for (const name of exes) inspectExecutable(join(source, name), target);
  if (existsSync(destination)) throw new Error(`Stage already exists: ${destination}`);
  mkdirSync(destination, { recursive: true });
  for (const name of [...exes, "BUILD.json"]) {
    copyFileSync(join(source, name), join(destination, name), constants.COPYFILE_EXCL);
  }
  console.log(`Staged ${triple}/${backend}: ${destination}`);
  return destination;
}

if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) {
  const { values } = parseArgs({ options: Object.fromEntries(["target", "backend", "input", "output"].map((name) => [name, { type: "string" }])) });
  for (const name of ["target", "backend", "input", "output"]) if (!values[name]) throw new Error(`Missing --${name}`);
  stageComponents(values);
}
