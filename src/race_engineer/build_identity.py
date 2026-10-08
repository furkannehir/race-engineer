"""Display-safe application build identity."""

import json
import re
from dataclasses import dataclass
from pathlib import Path

from race_engineer.version import __version__

_COMMIT = re.compile(r"^[0-9a-f]{7,40}$")


@dataclass(frozen=True)
class BuildIdentity:
    version: str
    commit: str


def read_build_identity(install_root: Path) -> BuildIdentity:
    path = install_root / "build-info.json"
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        version = raw.get("version")
        commit = raw.get("commit")
        if version != __version__ or not isinstance(commit, str) or not _COMMIT.fullmatch(commit):
            raise ValueError("invalid build identity")
        return BuildIdentity(version, commit[:12])
    except (OSError, ValueError, TypeError, AttributeError):
        return BuildIdentity(__version__, "development")
