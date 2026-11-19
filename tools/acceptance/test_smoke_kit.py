import hashlib
import json
from pathlib import Path

import pytest

from stage_smoke_kit import stage


def sha(data):
    return hashlib.sha256(data).hexdigest()


@pytest.fixture
def sources(tmp_path):
    package = tmp_path / "catalog"
    package.mkdir()
    runner = tmp_path / "hardware.exe"
    runner.write_bytes(b"test runner")
    components, imports, defaults = [], {}, []
    for backend in ("cpu", "vulkan"):
        pins = []
        for role in ("public-assets", "media", "worker", "probe"):
            version = backend if role in {"worker", "probe"} else "common"
            pin = {"id": role, "version": version}
            pins.append(pin)
            key = f"{role}/{version}"
            if key in imports:
                continue
            root = tmp_path / "source" / key
            root.mkdir(parents=True)
            data = f"{role} for {version}".encode()
            (root / "data").write_bytes(data)
            manifest = {"relative_files": ["data"], "hashes": {"data": sha(data)}, "sizes": {"data": len(data)}}
            raw = json.dumps(manifest).encode()
            entry = root / "manifest.json"
            entry.write_bytes(raw)
            imports[key] = str(entry)
            components.append({"pin": pin, "manifest": manifest, "manifest_sha256": sha(raw), "manifest_size": len(raw),
                               "supplemental": {}})
        defaults.append({"target": {"os": "windows", "architecture": "x86_64", "backend": backend}, "resources": pins})
    (package / "catalog.json").write_text(json.dumps({"defaults": defaults, "components": components}), encoding="utf-8")
    (package / "imports.json").write_text(json.dumps(imports), encoding="utf-8")
    return package, runner, defaults, imports


def test_kit_imports_resolve_after_relocation(tmp_path, sources):
    package, runner, defaults, imports = sources
    kit = tmp_path / "first location"
    stage(package, runner, kit)
    moved = kit.rename(tmp_path / "moved location")
    local = moved / "packages"
    assert (local / "catalog.json").read_bytes() == (package / "catalog.json").read_bytes()
    assert (moved / "hardware_acceptance.exe").read_bytes() == runner.read_bytes()
    assert (moved / "run.ps1").is_file()
    paths = json.loads((local / "imports.json").read_text(encoding="utf-8"))
    assert set(paths) == {f"{p['id']}/{p['version']}" for row in defaults for p in row["resources"]}
    for key, relative in paths.items():
        path = Path(relative)
        assert not path.is_absolute()
        assert (local / path).read_bytes() == Path(imports[key]).read_bytes()
        assert (local / path).with_name("data").read_bytes() == Path(imports[key]).with_name("data").read_bytes()


@pytest.mark.parametrize("problem", ["missing-import", "same-size-corrupt-payload"])
def test_kit_rejects_incomplete_or_corrupt_resources(tmp_path, sources, problem):
    package, runner, _, imports = sources
    if problem == "missing-import":
        imports.pop("probe/vulkan")
        (package / "imports.json").write_text(json.dumps(imports), encoding="utf-8")
    else:
        payload = Path(imports["worker/vulkan"]).with_name("data")
        payload.write_bytes(b"x" * payload.stat().st_size)
    with pytest.raises(ValueError, match="probe/vulkan|worker/vulkan"):
        stage(package, runner, tmp_path / "invalid-kit")
