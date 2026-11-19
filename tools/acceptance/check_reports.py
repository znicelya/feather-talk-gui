"""Check the structure and coverage of locally collected release evidence."""

import argparse
from dataclasses import asdict, dataclass
import ipaddress
import json
from pathlib import Path, PurePosixPath
import re
from urllib.parse import urlsplit


MODELS = {"feather_hubert", "scrfd_2_5g", "pfld_ghost_one", "vgg19", "mean_face"}
CHECKS = {"install", "first_start_download", "train", "resume", "export", "playback", "offline", "quality"}


@dataclass
class ValidationResult:
    passed: bool
    verified: list[str]
    missing: list[str]
    failures: list[str]


def _sha(value):
    return isinstance(value, str) and re.fullmatch("[0-9a-f]{64}", value) is not None


def _text(value):
    return isinstance(value, str) and bool(value.strip())


def _mapping(value):
    return value if isinstance(value, dict) else {}


def _evidence(value):
    if not _text(value) or "\\" in value or ":" in value:
        return False
    path = PurePosixPath(value)
    return bool(path.parts) and not path.is_absolute() and ".." not in path.parts


def _production_url(value):
    if not _text(value):
        return False
    try:
        url = urlsplit(value)
        host = url.hostname or ""
        if url.scheme != "https" or not host or url.username or url.password or not url.path:
            return False
        try:
            return ipaddress.ip_address(host).is_global
        except ValueError:
            return "." in host and not host.rstrip(".").lower().endswith((".localhost", ".local"))
    except ValueError:
        return False


def _version_tuple(value):
    if isinstance(value, str) and re.fullmatch(r"[0-9]{1,8}(?:\.[0-9]{1,8}){0,3}", value):
        numbers = tuple(map(int, value.split(".")))
        return numbers + (0,) * (4 - len(numbers))
    return ()


def _hardware_issues(report, row, target):
    issues = []
    device = _mapping(report.get("device"))
    system = _mapping(report.get("system"))
    family = row["hardware"]
    backend = device.get("backend")
    name = str(device.get("name", "")).lower()
    markers = {"nvidia": ("nvidia", "geforce", "quadro", "tesla"), "amd": ("amd", "radeon"),
               "intel_arc": ("arc",), "apple": ("apple", "m1", "m2", "m3", "m4", "m5")}
    if (report.get("target") != target["triple"] or system.get("os") != target["os"]
            or system.get("architecture") != target["architecture"]
            or _version_tuple(system.get("version")) < _version_tuple(target["minimum_version"])
            or ("distribution" in target and system.get("distribution") != target["distribution"])):
        issues.append("wrong operating system, architecture or target")
    if (backend not in row["backends"] or device.get("hardware") != family or not _text(device.get("name"))
            or not _text(device.get("id")) or not _text(device.get("driver"))
            or (family in markers and not any(marker in name for marker in markers[family]))):
        issues.append("wrong or unverified execution device/backend/driver")
    components = _mapping(report.get("components"))
    for role in ["worker", "probe", "media"] + (["cuda-math"] if backend == "cuda" else []):
        component = _mapping(components.get(role))
        acceptable = (backend, "any") if role == "media" else (backend,)
        if (component.get("target") != target["triple"] or component.get("backend") not in acceptable
                or not _sha(component.get("sha256")) or not _text(component.get("version"))):
            issues.append(f"missing {role} identity for this target/backend")
    models = _mapping(report.get("models"))
    model_gaps = sorted(name for name in MODELS if not _sha(models.get(name)))
    if model_gaps:
        issues.append("missing model identities: " + ", ".join(model_gaps))
    media = _mapping(report.get("media"))
    if any(not _sha(media.get(key)) for key in ("source_sha256", "driving_audio_sha256", "python_reference_sha256")):
        issues.append("missing video, audio or Python reference identity")
    checks = _mapping(report.get("checks"))
    expected = CHECKS | ({"signing", "notarization"} if target["os"] == "macos" else set())
    for name in sorted(expected):
        check = _mapping(checks.get(name))
        if check.get("status") != "pass" or not _evidence(check.get("evidence")):
            issues.append(f"no passing {name} evidence")
    if report.get("offline_mode") != "network-disabled":
        issues.append("offline run did not use a disconnected test environment")
    catalog = _mapping(report.get("catalog"))
    if (not _production_url(catalog.get("url")) or not _sha(catalog.get("sha256"))
            or catalog.get("production_download_verified") is not True):
        issues.append("no verified production catalog download")
    return issues


def validate_reports(matrix: dict, reports: list[dict], require_complete: bool) -> ValidationResult:
    if matrix.get("schema_version") != 1:
        raise ValueError("Expected platform matrix schema version 1")
    targets = {entry["triple"]: entry for entry in matrix["targets"]}
    paths = {entry["id"]: entry for entry in matrix["required_paths"]}
    if not targets or not paths or len(targets) != len(matrix["targets"]) or len(paths) != len(matrix["required_paths"]):
        raise ValueError("Platform matrix must define unique build targets and hardware paths")
    if any(not _version_tuple(target.get("minimum_version")) for target in targets.values()):
        raise ValueError("Every target must define a numeric minimum_version")
    for path in paths.values():
        target = targets.get(path["target"])
        if not target or not path["backends"] or not set(path["backends"]).issubset(target["backends"]):
            raise ValueError(f"Invalid hardware path {path['id']}")
    evidence = {f"build:{triple}": [] for triple in targets} | {path: [] for path in paths}
    invalid = []
    verified = set()
    for report in reports:
        if not isinstance(report, dict) or report.get("schema_version") != 1:
            invalid.append("Unsupported report schema")
            continue
        if report.get("kind") == "build" and report.get("target") in targets:
            target = targets[report["target"]]
            row = f"build:{target['triple']}"
            bundles = report.get("bundles")
            if (report.get("status") == "pass" and isinstance(bundles, list)
                    and all(isinstance(bundle, str) for bundle in bundles)
                    and set(target["bundles"]).issubset(bundles) and _evidence(report.get("evidence"))):
                verified.add(row)
            else:
                evidence[row].append("expected installer formats and build log are missing")
        elif report.get("kind") == "hardware" and report.get("path") in paths:
            row = report["path"]
            issues = _hardware_issues(report, paths[row], targets[paths[row]["target"]])
            if issues:
                evidence[row].extend(issues)
            else:
                verified.add(row)
        else:
            invalid.append("Unknown build target or hardware path")
    missing = sorted(evidence.keys() - verified)
    failures = invalid + [f"{row}: {message}" for row in missing for message in sorted(set(evidence[row]))]
    # Local smoke evidence can be incomplete; the complete gate rejects any missing row.
    passed = not invalid and (not require_complete or not missing)
    return ValidationResult(passed, sorted(verified), missing, failures)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--reports", type=Path, required=True,
                        help="Directory of build-*.json and acceptance-*.json reports")
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    try:
        matrix = json.loads(args.matrix.read_text(encoding="utf-8"))
        if not args.reports.is_dir():
            raise ValueError("Report directory does not exist")
        files = sorted([*args.reports.glob("build-*.json"), *args.reports.glob("acceptance-*.json")])
        reports = []
        for file in files:
            report = json.loads(file.read_text(encoding="utf-8"))
            # A pointer without a retained nonempty log cannot satisfy a release gate.
            if args.require_complete:
                for record in [report, *_mapping(report.get("checks")).values()]:
                    record = _mapping(record)
                    path = record.get("evidence")
                    if _evidence(path):
                        log = (file.parent / path).resolve()
                        if not log.is_relative_to(file.parent.resolve()) or not log.is_file() or not log.stat().st_size:
                            record.pop("evidence", None)
            reports.append(report)
        result = validate_reports(matrix, reports, args.require_complete)
    except (OSError, ValueError, TypeError, KeyError, AttributeError) as error:
        parser.exit(2, f"Invalid acceptance input: {error}\n")
    print(json.dumps(asdict(result), indent=2, ensure_ascii=True))
    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
