"""Record the actual bundle files and their hashes, separately from runtime evidence."""

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def digest(path):
    with path.open("rb") as reader:
        return hashlib.file_digest(reader, "sha256").hexdigest()


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--target", required=True)
    cli.add_argument("--bundles", type=Path, required=True)
    cli.add_argument("--log", type=Path, required=True)
    args = cli.parse_args()
    matrix = json.loads((ROOT / "resources/targets.json").read_text(encoding="utf-8"))
    target = next((row for row in matrix["targets"] if row["triple"] == args.target), None)
    if not target or not args.log.is_file() or not args.log.stat().st_size:
        cli.error("Expected a supported target and a nonempty build log")
    patterns = {"nsis": "nsis/*-setup.exe", "deb": "deb/*.deb", "appimage": "appimage/*.AppImage",
                "app": "macos/*.app/Contents/Info.plist", "dmg": "dmg/*.dmg"}
    files = {}
    for bundle in target["bundles"]:
        matches = [p for p in args.bundles.glob(patterns[bundle]) if p.is_file() and p.stat().st_size]
        if len(matches) != 1:
            cli.error(f"Expected one {bundle} output in {args.bundles}, found {len(matches)}")
        files[bundle] = {"path": matches[0].relative_to(args.bundles).as_posix(), "sha256": digest(matches[0])}
    report = {"schema_version": 1, "kind": "build", "target": args.target,
              "status": "pass", "bundles": target["bundles"], "files": files, "evidence": args.log.name}
    output = args.log.parent / f"build-{args.target}.json"
    with output.open("x", encoding="utf-8") as writer:
        json.dump(report, writer, indent=2)
        writer.write("\n")
    print(output)


if __name__ == "__main__":
    main()
