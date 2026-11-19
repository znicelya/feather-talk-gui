import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { createHash } from "node:crypto";
import { mkdirSync, mkdtempDisposableSync, readFileSync, realpathSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { test } from "node:test";
import { names, repo, targets } from "./stage-components.mjs";

const json = (path) => JSON.parse(readFileSync(path, "utf8"));
const hash = (bytes) => createHash("sha256").update(bytes).digest("hex");

function write(path, bytes) {
  mkdirSync(dirname(path), { recursive: true });
  writeFileSync(path, bytes);
}

function executable({ os, architecture }) {
  const bytes = Buffer.alloc(128);
  if (os === "windows") {
    bytes.write("MZ");
    bytes.writeUInt32LE(64, 60);
    bytes.write("PE\0\0", 64);
    bytes.writeUInt16LE(0x8664, 68);
    bytes.writeUInt16LE(0x20b, 88);
  } else if (os === "linux") {
    bytes.write("\x7fELF\x02\x01", 0, "latin1");
    bytes.writeUInt16LE(2, 16);
    bytes.writeUInt16LE(62, 18);
  } else {
    bytes.writeUInt32LE(0xfeedfacf);
    bytes.writeUInt32LE(architecture === "aarch64" ? 0x0100000c : 0x01000007, 4);
    bytes.writeUInt32LE(2, 12);
  }
  return bytes;
}

function fixture(t, target, { missingBackend, badMedia, badModel } = {}) {
  const scratch = mkdtempDisposableSync(join(realpathSync(tmpdir()), "ft bundle-"));
  const root = scratch.path;
  t.after(() => scratch.remove());
  const assets = join(root, "inputs", "public-assets");
  const base = json(join(repo, "resources/development-catalog.json"));
  const c = base.components.find((c) => c.pin.id === "public-assets");
  // Keep the real catalog's file layout and replace bulky model bytes with fixtures.
  function entry(name) {
    const payload = Buffer.from(name);
    write(join(assets, name), payload);
    return { size: payload.length, sha256: hash(payload), urls: [] };
  }
  for (const name of c.manifest.relative_files) {
    const file = entry(name);
    c.manifest.hashes[name] = file.sha256;
    c.manifest.sizes[name] = file.size;
  }
  for (const name of Object.keys(c.supplemental)) c.supplemental[name] = entry(name);
  const manifest = Buffer.from(JSON.stringify(c.manifest));
  c.manifest_size = manifest.length;
  c.manifest_sha256 = hash(manifest);
  write(join(assets, "manifest.json"), manifest);
  if (badModel) {
    const name = Object.keys(c.supplemental)[0];
    write(join(assets, name), Buffer.alloc(c.supplemental[name].size));
  }
  write(join(root, "inputs", "catalog.json"), JSON.stringify(base));
  write(join(root, "inputs", "imports.json"), '["public-assets/manifest.json"]');
  const staged = join(root, "staged");
  for (const { triple, backends, ...platform } of targets) {
    for (const backend of backends) {
      if (triple === target.triple && backend === missingBackend) continue;
      const dir = join(staged, triple, backend);
      write(join(dir, "BUILD.json"), JSON.stringify({ target: triple, backend, base_revision: "fixture", exit: 0 }));
      for (const name of names(platform)) write(join(dir, name), executable(platform));
    }
  }
  const media = join(root, "inputs", target.triple, "media");
  write(join(media, "LICENSE"), "FFmpeg notice");
  write(join(media, "BUILD.json"), '{"source":"fixture"}');
  for (const name of ["ffmpeg", "ffprobe"]) {
    const ext = target.os === "windows" ? ".exe" : "";
    write(join(media, "bin", name + ext), executable(badMedia ? targets.find((t) => t.os !== target.os) : target));
  }
  const cuda = join(root, "inputs", target.triple, "cuda-math");
  if (target.backends.includes("cuda")) {
    write(join(cuda, "LICENSE"), "CUDA notice");
    write(join(cuda, "nvvm/libdevice/libdevice.10.bc"), "math fixture");
  }
  const out = join(root, "output");
  return {
    out,
    build({ withCuda = true } = {}) {
      const result = spawnSync(process.execPath, [join(repo, "tools/package/build-catalog.mjs"),
        "--artifacts", staged, "--output", join(out, "catalog.json"), "--bundle-target", target.triple,
        "--base-catalog", join(root, "inputs", "catalog.json"), "--base-imports", join(root, "inputs", "imports.json"),
        "--media", media, ...(withCuda && target.backends.includes("cuda") ? ["--cuda-math", cuda] : []),
      ], { cwd: repo, encoding: "utf8", windowsHide: true });
      if (result.error) throw result.error;
      return result;
    },
  };
}

for (const target of targets) {
  test(`bundle ${target.triple} contains the complete native selections`, (t) => {
    const { out, build } = fixture(t, target);
    const result = build();
    assert.equal(result.status, 0, result.stderr);
    const catalog = json(join(out, "catalog.json"));
    const { bundle } = json(join(out, "tauri-overlay.json"));
    assert.deepEqual(bundle.targets, target.bundles);
    assert.equal(bundle.resources[join(out, "catalog.json")], "bootstrap/catalog.json");
    assert.deepEqual(catalog.defaults.map((row) => row.target.backend).sort(), [...target.backends].sort());
    const destinations = new Set(Object.values(bundle.resources));
    const selected = new Set();
    for (const row of catalog.defaults) {
      assert.equal(row.target.os, target.os);
      assert.equal(row.target.architecture, target.architecture);
      const required = ["worker", "probe", "public-assets", "media"];
      if (row.target.backend === "cuda") required.push("cuda-math");
      assert.deepEqual(row.resources.map(({ id }) => id).sort(), required.sort());
      for (const pin of row.resources) selected.add(`${pin.id}/${pin.version}`);
    }
    assert.equal(catalog.components.length, 2 * target.backends.length + 2 + Number(target.backends.includes("cuda")));
    for (const c of catalog.components) {
      const key = `${c.pin.id}/${c.pin.version}`;
      assert.ok(selected.has(key));
      for (const file of ["manifest.json", ...c.manifest.relative_files, ...Object.keys(c.supplemental)]) {
        assert.ok(destinations.delete(`bootstrap/${key}/${file}`), `${key}/${file}`);
      }
    }
    assert.equal(Object.keys(bundle.resources).length, new Set(Object.values(bundle.resources)).size);
    assert.deepEqual([...destinations], ["bootstrap/catalog.json"]);
  });
}

test("refuse an installer when a native backend pair is missing", (t) => {
  const result = fixture(t, targets[0], { missingBackend: "vulkan" }).build();
  assert.notEqual(result.status, 0);
  assert.match(result.stderr, /vulkan/);
});

test("refuse an installer when no CUDA math package was supplied", (t) => {
  const result = fixture(t, targets[1]).build({ withCuda: false });
  assert.notEqual(result.status, 0);
  assert.match(result.stderr, /cuda-math/);
});

test("refuse media binaries built for another platform", (t) => {
  const result = fixture(t, targets[0], { badMedia: true }).build();
  assert.notEqual(result.status, 0);
  assert.match(result.stderr, /ffmpeg.exe/);
});

test("refuse a package whose pinned model notice bytes were changed", (t) => {
  const result = fixture(t, targets[2], { badModel: true }).build();
  assert.notEqual(result.status, 0);
  assert.match(result.stderr, /SHA-256 mismatch.*evidence/);
});
