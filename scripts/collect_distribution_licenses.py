"""Preserve installed wheel notices and metadata beside a Windows folder bundle."""

import argparse
import hashlib
import json
import re
from importlib.metadata import distributions
from pathlib import Path

NOTICE = re.compile(r"^(?:licen[cs]e|copying|notice|authors)(?:[._-].*)?$", re.IGNORECASE)


def collect(output: Path) -> tuple[int, int]:
    output.mkdir(parents=True, exist_ok=True)
    records = []
    file_count = 0
    for package in sorted(distributions(), key=lambda item: item.metadata.get("Name", "")):
        name = package.metadata.get("Name", "unknown")
        key = re.sub(r"[^A-Za-z0-9_.-]", "_", f"{name}-{package.version}")
        destination = output / key
        destination.mkdir(parents=True, exist_ok=True)
        metadata = package.read_text("METADATA")
        if metadata is not None:
            (destination / "METADATA").write_text(metadata, encoding="utf-8")
        notices = []
        for entry in package.files or ():
            if not NOTICE.fullmatch(entry.name) or entry.is_absolute() or ".." in entry.parts:
                continue
            original = package.locate_file(entry)
            if not original.is_file():
                continue
            body = original.read_bytes()
            target = destination.joinpath(*entry.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(body)
            notices.append(
                {
                    "path": target.relative_to(output).as_posix(),
                    "sha256": hashlib.sha256(body).hexdigest(),
                }
            )
            file_count += 1
        records.append(
            {
                "name": name,
                "version": package.version,
                "license": package.metadata.get("License-Expression")
                or package.metadata.get("License"),
                "notices": notices,
            }
        )
    (output / "manifest.json").write_text(
        json.dumps({"distributions": records}, indent=2) + "\n", encoding="utf-8"
    )
    return len(records), file_count


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(Path(__file__).resolve().parents[1]):
        parser.error("output must stay inside this repository")
    package_count, file_count = collect(output)
    print(f"Retained {file_count} notice files and metadata for {package_count} distributions.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
