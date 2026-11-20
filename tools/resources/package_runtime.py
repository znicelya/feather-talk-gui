"""Package successful local worker/probe builds alongside existing resources."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil

from package_local import digest


def load(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


TARGETS = {
    target["triple"]: (target["os"], target["architecture"], set(target["backends"]))
    for target in load(Path(__file__).resolve().parents[2] / "resources/targets.json")["targets"]
}


def encode(obj):
    return (json.dumps(obj, indent=2, sort_keys=True) + "\n").encode()


def hash_bytes(data):
    return hashlib.sha256(data).hexdigest()


def pin_key(pin):
    return f"{pin['id']}/{pin['version']}"


def package(kind, catalog_kind, paths, target, output, executables, notices, provenance):
    sizes = {name: path.stat().st_size for name, path in paths.items()}
    hashes = {name: digest(path) for name, path in paths.items()}
    inventory_id = hash_bytes(encode({"target": target, "sizes": sizes, "hashes": hashes}))
    manifest = dict(
        identity=dict(kind=kind, architecture_version="1", source_sha256=inventory_id, weights_sha256=inventory_id),
        relative_files=sorted(paths), sizes=sizes, hashes=hashes, urls={}, min_engine_version="0.1.0",
        provenance=provenance,
    )
    raw = encode(manifest)
    sha256 = hash_bytes(raw)
    pin = dict(id=kind, version=sha256[:16])
    root = output / pin_key(pin)
    root.mkdir(parents=True)
    for name, path in paths.items():
        copied = root / name
        copied.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, copied)
        if name in executables:
            shutil.copymode(path, copied)
        if copied.stat().st_size != sizes[name] or digest(copied) != hashes[name]:
            raise ValueError(f"Build input changed during packaging: {path}")
    manifest_path = root / "manifest.json"
    manifest_path.write_bytes(raw)
    component = dict(
        pin=pin, target=target, kind=catalog_kind, manifest=manifest, manifest_sha256=sha256, manifest_size=len(raw),
        manifest_urls=[], supplemental={}, executables=executables, notices=notices,
        redistribution_approved=False, license="NOASSERTION",
    )
    return component, str(manifest_path.resolve())


def package_worker(kind, directory, lockfile, target, output):
    binary = f"feathertalk-{kind}" + (".exe" if target["os"] == "windows" else "")
    paths = {binary: directory / binary, "BUILD.json": directory / "BUILD.json", "Cargo.lock": lockfile}
    return package(kind, kind, paths, target, output, [binary], ["BUILD.json", "Cargo.lock"],
                   "Local development build. BUILD.json records the build and Cargo.lock pins dependencies.")


def package_component(kind, triple, directory, output):
    os, arch, backends = TARGETS[triple]
    suffix = ".exe" if os == "windows" else ""
    if kind == "media":
        executables = [f"bin/{exe}{suffix}" for exe in ("ffmpeg", "ffprobe")]
        license_files = ["LICENSE", "BUILD.json"]
        files = executables + license_files
        catalog_kind, backend = "media", "any"
    else:
        if "cuda" not in backends:
            raise ValueError(f"No CUDA path for {triple}")
        executables = []
        license_files = ["LICENSE"]
        files = ["nvvm/libdevice/libdevice.10.bc", *license_files]
        catalog_kind, backend = "cuda_runtime", "cuda"
    target = dict(os=os, architecture=arch, backend=backend)
    return package(kind, catalog_kind, {name: directory / name for name in files}, target, output,
                   executables, license_files, "Local component inputs. No redistribution approval asserted.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("base-catalog", "base-imports", "lockfile", "output"):
        parser.add_argument("--" + flag, required=True, type=Path)
    parser.add_argument("--build", required=True, action="append", type=Path, help="Directory with binaries and BUILD.json")
    parser.add_argument("--media", nargs=2, metavar=("TRIPLE", "DIR"), action="append", default=[])
    parser.add_argument("--cuda-math", nargs=2, metavar=("TRIPLE", "DIR"), action="append", default=[])
    args = parser.parse_args()
    catalog = load(args.base_catalog)
    by_manifest = {c["manifest_sha256"]: c for c in catalog["components"]}
    imports = {}
    supplied = load(args.base_imports)
    for path in map(Path, supplied.values() if isinstance(supplied, dict) else supplied):
        path = args.base_imports.parent / path
        component = by_manifest.get(digest(path))
        if component is None:
            raise ValueError(f"Base manifest does not match the catalog: {path}")
        imports[pin_key(component["pin"])] = str(path.resolve())
    builds = []
    used_targets = set()
    for directory in args.build:
        record = load(directory / "BUILD.json")
        os, arch, backends = TARGETS[record["target"]]
        backend = record["backend"]
        key = os, arch, backend
        if record["exit"] != 0 or backend not in backends or key in used_targets:
            raise ValueError(f"Invalid, failed or duplicate build: {directory}")
        used_targets.add(key)
        target = dict(os=os, architecture=arch, backend=backend)
        matches = [s for s in catalog["defaults"] if s["target"] == target]
        if len(matches) != 1 or {"worker", "probe"} - {p["id"] for p in matches[0]["resources"]}:
            raise ValueError(f"No unique worker/probe selection for {target}")
        builds.append((directory, target, matches[0]))
    args.output.mkdir(parents=True)
    used_pins = {pin_key(c["pin"]) for c in catalog["components"]}

    def add(component, manifest_path, selections):
        pin = component["pin"]
        key = pin_key(pin)
        if key in used_pins:
            raise ValueError(f"Runtime pin already present: {key}")
        used_pins.add(key)
        catalog["components"].append(component)
        imports[key] = manifest_path
        for selection in selections:
            selection["resources"] = [p for p in selection["resources"] if p["id"] != pin["id"]] + [pin]

    for directory, target, selection in builds:
        for kind in ("worker", "probe"):
            add(*package_worker(kind, directory, args.lockfile, target, args.output), [selection])
    for kind, components in (("media", args.media), ("cuda-math", args.cuda_math)):
        used_targets = set()
        for triple, directory in components:
            if triple in used_targets:
                raise ValueError(f"More than one {kind} package for {triple}")
            used_targets.add(triple)
            os, arch, _ = TARGETS[triple]
            selections = [s for s in catalog["defaults"] if s["target"]["os"] == os
                          and s["target"]["architecture"] == arch
                          and (kind == "media" or s["target"]["backend"] == "cuda")]
            if not selections:
                raise ValueError(f"No {kind} selection for {triple}")
            add(*package_component(kind, triple, Path(directory), args.output), selections)
    (args.output / "catalog.json").write_bytes(encode(catalog))
    (args.output / "imports.json").write_bytes(encode(imports))


if __name__ == "__main__":
    main()
