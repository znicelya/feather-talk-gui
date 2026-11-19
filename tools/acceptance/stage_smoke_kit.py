"""Copy the resources for a portable Windows CPU/Vulkan smoke run."""

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shutil


def key(pin):
    parts = (pin["id"], pin["version"])
    if not all(re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]*", part) for part in parts):
        raise ValueError(f"Invalid component pin: {pin}")
    return "/".join(parts)


def payload_path(name):
    path = PurePosixPath(name)
    if not path.parts or path.is_absolute() or ".." in path.parts or any(c in name for c in "\\:"):
        raise ValueError(f"Invalid payload path: {name}")
    return path


def copy_verified(source, destination, size, expected, pin):
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as reader, destination.open("xb") as output:
        shutil.copyfileobj(reader, output)
    with destination.open("rb") as reader:
        actual = hashlib.file_digest(reader, "sha256").hexdigest()
    if destination.stat().st_size != size or actual != expected:
        raise ValueError(f"Content does not match the catalog for {pin}: {source.name}")


def stage(package: Path, runner: Path, output: Path):
    catalog_bytes = (package / "catalog.json").read_bytes()
    catalog = json.loads(catalog_bytes)
    imports = json.loads((package / "imports.json").read_bytes())
    components = {key(c["pin"]): c for c in catalog["components"]}
    selected = set()
    for backend in ("cpu", "vulkan"):
        target = {"os": "windows", "architecture": "x86_64", "backend": backend}
        rows = [row for row in catalog["defaults"] if row["target"] == target]
        if len(rows) != 1 or {p["id"] for p in rows[0]["resources"]} != {"public-assets", "media", "worker", "probe"}:
            raise ValueError(f"Expected a complete Windows x64 {backend} selection")
        selected.update(key(pin) for pin in rows[0]["resources"])

    for pin in sorted(selected):
        if pin not in components or pin not in imports:
            raise ValueError(f"Missing pinned component: {pin}")

    output.mkdir(parents=True)
    local = output / "packages"
    local.mkdir()
    portable = {}
    for pin in sorted(selected):
        c = components[pin]
        manifest = c["manifest"]
        relative = f"{pin}/manifest.json"
        source = package / imports[pin]
        copy_verified(source, local / relative, c["manifest_size"], c["manifest_sha256"], pin)
        for name in manifest["relative_files"]:
            path = payload_path(name)
            copy_verified(source.parent / path, local / pin / path, manifest["sizes"][name], manifest["hashes"][name], pin)
        for name, identity in c["supplemental"].items():
            path = payload_path(name)
            copy_verified(source.parent / path, local / pin / path, identity["size"], identity["sha256"], pin)
        portable[pin] = relative
    (local / "catalog.json").write_bytes(catalog_bytes)
    (local / "imports.json").write_text(json.dumps(portable, indent=2) + "\n", encoding="utf-8")
    shutil.copyfile(runner, output / "hardware_acceptance.exe")
    shutil.copyfile(Path(__file__).with_name("run.ps1"), output / "run.ps1")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("packages", "runner", "output"):
        parser.add_argument(f"--{option}", required=True, type=Path)
    args = parser.parse_args()
    try:
        stage(args.packages, args.runner, args.output)
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, f"Could not stage smoke kit: {error}\n")
    print(f"Portable smoke kit: {args.output.resolve()}")


if __name__ == "__main__":
    main()
