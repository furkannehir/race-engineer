"""Generate versioned Windows build metadata from Pitward's authoritative version."""

import argparse
import json
import re
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from race_engineer.version import __version__

_VERSION = re.compile(
    r"^(?P<major>0|[1-9][0-9]*)\."
    r"(?P<minor>0|[1-9][0-9]*)\."
    r"(?P<patch>0|[1-9][0-9]*)"
    r"(?:(?P<stage>a|b|rc)(?P<serial>[1-9][0-9]*))?$"
)
_COMMIT = re.compile(r"^[0-9a-f]{40}$")


def git_commit(root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    commit = result.stdout.strip().lower()
    if not _COMMIT.fullmatch(commit):
        raise ValueError("git returned an invalid commit")
    return commit


def windows_version(version: str) -> tuple[int, int, int, int]:
    match = _VERSION.fullmatch(version)
    if match is None:
        raise ValueError(f"unsupported application version: {version}")
    return (
        int(match["major"]),
        int(match["minor"]),
        int(match["patch"]),
        int(match["serial"] or 0),
    )


def version_resource(version: str) -> str:
    numeric = windows_version(version)
    numbers = ", ".join(str(value) for value in numeric)
    return f"""VSVersionInfo(
  ffi=FixedFileInfo(filevers=({numbers}), prodvers=({numbers}), mask=0x3f,
    flags=0x0, OS=0x40004, fileType=0x1, subtype=0x0, date=(0, 0)),
  kids=[StringFileInfo([StringTable('040904B0', [
    StringStruct('CompanyName', 'Pitward contributors'),
    StringStruct('FileDescription', 'Pitward local race engineer'),
    StringStruct('FileVersion', '{version}'),
    StringStruct('InternalName', 'Pitward'),
    StringStruct('OriginalFilename', 'Pitward.exe'),
    StringStruct('ProductName', 'Pitward'),
    StringStruct('ProductVersion', '{version}')
  ])]), VarFileInfo([VarStruct('Translation', [1033, 1200])])]
)
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--commit")
    parser.add_argument("--tag")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    commit = (args.commit or git_commit(root)).lower()
    if not _COMMIT.fullmatch(commit):
        raise ValueError("commit must be a full lowercase Git SHA")
    if args.tag is not None and args.tag != f"v{__version__}":
        raise ValueError(
            f"release tag {args.tag!r} does not match application version v{__version__}"
        )
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    identity = {
        "schema_version": "pitward-build.v1",
        "version": __version__,
        "commit": commit,
        "built_at": datetime.now(UTC).isoformat(),
    }
    (output / "build-info.json").write_text(
        json.dumps(identity, indent=2) + "\n", encoding="utf-8"
    )
    (output / "windows-version.txt").write_text(
        version_resource(__version__), encoding="utf-8"
    )
    print(json.dumps(identity))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
