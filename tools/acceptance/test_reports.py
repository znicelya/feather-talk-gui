from copy import deepcopy
import json
from pathlib import Path

import pytest

from check_reports import validate_reports


MATRIX = json.loads((Path(__file__).resolve().parents[2] / "resources" / "targets.json").read_text())
SHA = "a" * 64


def acceptance(row, target):
    backend = row["backends"][0]
    hardware = row["hardware"]
    name = {"cpu": "Intel Core i5-10400", "amd": "AMD Radeon RX 6800", "apple": "Apple M2",
            "intel_arc": "Intel Arc B580", "nvidia": "NVIDIA GeForce RTX 2060"}[hardware]
    return {
        "schema_version": 1, "kind": "hardware", "path": row["id"], "target": row["target"],
        "system": {"os": target["os"], "architecture": target["architecture"],
                   "version": {"windows": "10.0.26100", "linux": "24.04", "macos": "13.7.8"}[target["os"]],
                   **({"distribution": "ubuntu"} if target["os"] == "linux" else {})},
        "device": {"backend": backend, "id": f"{backend}:0", "hardware": hardware, "name": name, "driver": "verified test runtime"},
        "checks": {key: {"status": "pass", "evidence": f"logs/{row['id']}-{key}.log"} for key in (
            "install", "first_start_download", "train", "resume", "export", "playback", "offline", "quality",
            "signing", "notarization")},
        "offline_mode": "network-disabled",
        "components": {role: {"sha256": SHA, "version": "test", "target": row["target"], "backend": backend}
                       for role in (["worker", "probe", "media", "cuda-math"] if backend == "cuda" else ["worker", "probe", "media"])},
        "models": {key: SHA for key in ("feather_hubert", "scrfd_2_5g", "pfld_ghost_one", "vgg19", "mean_face")},
        "media": {"source_sha256": SHA, "driving_audio_sha256": SHA, "python_reference_sha256": SHA},
        "catalog": {"sha256": SHA, "url": "https://download.valid.test/catalog.json", "production_download_verified": True},
    }


@pytest.fixture
def reports():
    by_target = {entry["triple"]: entry for entry in MATRIX["targets"]}
    builds = [{"schema_version": 1, "kind": "build", "target": triple, "bundles": target["bundles"],
               "status": "pass", "evidence": f"logs/{triple}-build.log"} for triple, target in by_target.items()]
    return builds + [acceptance(row, by_target[row["target"]]) for row in MATRIX["required_paths"]]


def test_complete_reports_cover_all_builds_and_runtime_paths(reports):
    result = validate_reports(MATRIX, reports, True)
    assert result.passed, result
    assert not result.missing


def test_compilation_does_not_count_as_gpu_execution(reports):
    builds = [item for item in reports if item["kind"] == "build"]
    result = validate_reports(MATRIX, builds, True)
    assert not result.passed
    assert "windows-intel-arc" in result.missing
    assert validate_reports(MATRIX, builds, False).passed


@pytest.mark.parametrize("case", ["cpu-as-amd", "intel-mac-gpu", "offline", "worker-target", "missing-mean-face", "invalid-url"])
def test_incomplete_or_wrong_runtime_evidence_fails_release(reports, case):
    path = "macos-intel-cpu" if case == "intel-mac-gpu" else "windows-amd"
    row = next(record for record in reports if record.get("path") == path)
    if case == "cpu-as-amd":
        row["device"].update(backend="cpu", name="Intel Core i5")
    elif case == "intel-mac-gpu":
        row["device"]["backend"] = "metal"
    elif case == "offline":
        row["offline_mode"] = "local-fixture-only"
    elif case == "worker-target":
        row["components"]["worker"]["target"] = "x86_64-unknown-linux-gnu"
    elif case == "missing-mean-face":
        del row["models"]["mean_face"]
    elif case == "invalid-url":
        row["catalog"]["url"] = "http://127.0.0.1:3456/catalog.json"
    result = validate_reports(MATRIX, deepcopy(reports), True)
    assert not result.passed
    assert path in result.missing, result


def test_windows_10_smoke_cannot_satisfy_win11_release_row(reports):
    row = next(item for item in reports if item.get("path") == "windows-cpu")
    row["system"]["version"] = "10.0.19045"
    result = validate_reports(MATRIX, reports, True)
    assert not result.passed
    assert "windows-cpu" in result.missing
