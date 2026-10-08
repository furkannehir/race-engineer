"""Versioned, checksum-verified setup for Pitward's local AI components."""

import argparse
import hashlib
import json
import os
import shutil
import time
import urllib.request
import zipfile
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import Any, Literal

_CHUNK_SIZE = 1024 * 1024


@dataclass(frozen=True)
class Artifact:
    artifact_id: str
    kind: Literal["file", "zip"]
    url: str
    sha256: str
    size: int
    destination: Path
    license: str
    required: Path | None = None


def component_root(environ: dict[str, str] | None = None) -> Path:
    env = os.environ if environ is None else environ
    override = env.get("PITWARD_DATA_DIR")
    if override:
        return Path(override).expanduser().resolve() / "components"
    local = env.get("LOCALAPPDATA")
    if not local:
        raise OSError("LOCALAPPDATA is unavailable; set PITWARD_DATA_DIR")
    return (Path(local) / "Pitward" / "components").resolve()


def load_manifest() -> tuple[str, tuple[Artifact, ...]]:
    resource = files("race_engineer").joinpath("assets/components.v1.json")
    raw: dict[str, Any] = json.loads(resource.read_text(encoding="utf-8"))
    if raw.get("schema_version") != "pitward-components.v1":
        raise ValueError("unsupported component manifest")
    artifacts = tuple(
        Artifact(
            artifact_id=item["id"],
            kind=item["kind"],
            url=item["url"],
            sha256=item["sha256"],
            size=item["size"],
            destination=Path(item["destination"]),
            license=item["license"],
            required=Path(item["required"]) if item.get("required") else None,
        )
        for item in raw["artifacts"]
    )
    return str(raw["component_set"]), artifacts


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _valid_file(path: Path, artifact: Artifact, *, hash_content: bool) -> bool:
    if not path.is_file() or path.stat().st_size != artifact.size:
        return False
    return not hash_content or digest(path) == artifact.sha256


def _zip_installed(root: Path, artifact: Artifact) -> bool:
    if artifact.required is None:
        return False
    marker = root / artifact.destination / ".pitward-component.json"
    required = root / artifact.destination / artifact.required
    try:
        raw = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return required.is_file() and raw.get("sha256") == artifact.sha256


def artifact_installed(
    root: Path, artifact: Artifact, *, hash_content: bool = False
) -> bool:
    if artifact.kind == "file":
        return _valid_file(root / artifact.destination, artifact, hash_content=hash_content)
    return _zip_installed(root, artifact)


def _download(artifact: Artifact, target: Path) -> Path:
    temporary = target.with_name(target.name + ".partial")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    if temporary.exists() and temporary.stat().st_size >= artifact.size:
        if _valid_file(temporary, artifact, hash_content=True):
            return temporary
        temporary.unlink()
    offset = temporary.stat().st_size if temporary.exists() else 0
    request = urllib.request.Request(
        artifact.url,
        headers={
            "User-Agent": "Pitward component installer",
            **({"Range": f"bytes={offset}-"} if offset else {}),
        },
    )
    response = urllib.request.urlopen(request, timeout=60)
    if offset and getattr(response, "status", None) != 206:
        response.close()
        temporary.unlink()
        offset = 0
        request = urllib.request.Request(
            artifact.url, headers={"User-Agent": "Pitward component installer"}
        )
        response = urllib.request.urlopen(request, timeout=60)
    mode = "ab" if offset else "xb"
    downloaded = offset
    last_report = 0.0
    with response, temporary.open(mode) as output:
        while chunk := response.read(_CHUNK_SIZE):
            output.write(chunk)
            downloaded += len(chunk)
            now = time.monotonic()
            if now - last_report >= 2:
                print(
                    f"  {artifact.artifact_id}: {downloaded / artifact.size:.0%}",
                    flush=True,
                )
                last_report = now
    if not _valid_file(temporary, artifact, hash_content=True):
        raise RuntimeError(f"checksum_failed:{artifact.artifact_id}")
    return temporary


def _safe_extract(archive_path: Path, destination: Path, artifact: Artifact) -> None:
    staging = destination.with_name(destination.name + ".partial")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    root = staging.resolve()
    with zipfile.ZipFile(archive_path) as archive:
        for item in archive.infolist():
            target = (staging / item.filename).resolve()
            if root not in target.parents and target != root:
                raise RuntimeError(f"unsafe_archive_path:{artifact.artifact_id}")
            if item.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(item) as source, target.open("xb") as output:
                shutil.copyfileobj(source, output)
    marker = staging / ".pitward-component.json"
    marker.write_text(
        json.dumps({"id": artifact.artifact_id, "sha256": artifact.sha256}, indent=2)
        + "\n",
        encoding="utf-8",
    )
    if artifact.required is None or not (staging / artifact.required).is_file():
        raise RuntimeError(f"archive_missing_required_file:{artifact.artifact_id}")
    if destination.exists():
        shutil.rmtree(destination)
    staging.replace(destination)


def install(root: Path, artifacts: tuple[Artifact, ...]) -> None:
    downloads = root / ".downloads"
    for artifact in artifacts:
        if artifact_installed(root, artifact, hash_content=artifact.kind == "file"):
            print(f"Keeping {artifact.artifact_id}", flush=True)
            continue
        print(f"Downloading {artifact.artifact_id} ({artifact.license})", flush=True)
        filename = artifact.url.rsplit("/", 1)[-1]
        downloaded = _download(artifact, downloads / filename)
        destination = root / artifact.destination
        if artifact.kind == "zip":
            _safe_extract(downloaded, destination, artifact)
            downloaded.unlink()
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.replace(downloaded, destination)
    print(f"Pitward components are ready in {root}")


def _status(root: Path, artifacts: tuple[Artifact, ...], *, verify: bool) -> bool:
    ready = True
    for artifact in artifacts:
        installed = artifact_installed(root, artifact, hash_content=verify)
        print(f"{'ready' if installed else 'missing'}  {artifact.artifact_id}")
        ready = ready and installed
    return ready


def main() -> int:
    parser = argparse.ArgumentParser(description="Install Pitward local AI components")
    parser.add_argument("command", choices=("status", "verify", "install"))
    parser.add_argument("--root", type=Path)
    parser.add_argument("--accept-third-party-licenses", action="store_true")
    args = parser.parse_args()
    component_set, artifacts = load_manifest()
    root = (args.root.resolve() if args.root else component_root())
    print(f"Pitward component set {component_set}")
    if args.command == "install":
        if not args.accept_third_party_licenses:
            print(
                "Review the licenses shown in components.v1.json, then rerun with "
                "--accept-third-party-licenses."
            )
            return 2
        install(root, artifacts)
        return 0
    return 0 if _status(root, artifacts, verify=args.command == "verify") else 1


if __name__ == "__main__":
    raise SystemExit(main())
