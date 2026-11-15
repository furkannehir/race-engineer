"""Published GitHub releases and verified Windows installer downloads."""

import hashlib
import json
import re
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from http.client import HTTPResponse
from pathlib import Path
from typing import cast
from urllib.parse import urlsplit

REPOSITORY = "furkannehir/race-engineer"
RELEASES_URL = f"https://github.com/{REPOSITORY}/releases/latest"
LATEST_URL = f"https://api.github.com/repos/{REPOSITORY}/releases/latest"
_VERSION = re.compile(r"(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)(?:(a|b|rc)([1-9]\d*))?")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_MAX_INSTALLER = 2 * 1024**3


class UpdateError(Exception):
    """A display-safe update failure code."""


class UpdateCancelled(Exception):
    pass


@dataclass(frozen=True)
class AvailableUpdate:
    version: str
    filename: str
    url: str
    sha256: str
    size: int
    release_url: str


def version_key(value: str) -> tuple[int, int, int, int, int]:
    match = _VERSION.fullmatch(value)
    if match is None:
        raise UpdateError("update_version_invalid")
    major, minor, patch, stage, serial = match.groups()
    return (
        int(major),
        int(minor),
        int(patch),
        {"a": 0, "b": 1, "rc": 2, None: 3}[stage],
        int(serial or 0),
    )


def _check_cancelled(cancelled: Callable[[], bool]) -> None:
    if cancelled():
        raise UpdateCancelled


def _open(url: str) -> HTTPResponse:
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Pitward updater",
            "Accept": "application/vnd.github+json"
            if url == LATEST_URL
            else "application/octet-stream",
            "X-GitHub-Api-Version": "2026-03-10",
        },
    )
    response = cast(HTTPResponse, urllib.request.urlopen(request, timeout=15))
    if urlsplit(response.geturl()).scheme != "https":
        response.close()
        raise UpdateError("update_url_invalid")
    return response


def _read(url: str, limit: int, cancelled: Callable[[], bool]) -> bytes:
    _check_cancelled(cancelled)
    with _open(url) as response:
        data = response.read(limit + 1)
    _check_cancelled(cancelled)
    if len(data) > limit:
        raise UpdateError("update_metadata_invalid")
    return bytes(data)


def check_for_update(
    current_version: str, *, cancelled: Callable[[], bool] = lambda: False
) -> AvailableUpdate | None:
    """Only published stable releases with the exact repository asset names qualify."""
    try:
        raw = json.loads(_read(LATEST_URL, 1024**2, cancelled))
        if not isinstance(raw, dict):
            raise UpdateError("update_metadata_invalid")
        if raw.get("draft") is not False or raw.get("prerelease") is not False:
            return None
        tag = raw.get("tag_name")
        if not isinstance(tag, str) or not tag.startswith("v"):
            raise UpdateError("update_version_invalid")
        version = tag[1:]
        remote_key = version_key(version)
        if remote_key[3] != 3 or remote_key <= version_key(current_version):
            return None
        filename = f"Pitward-Setup-{version}-windows-x64.exe"
        prefix = f"https://github.com/{REPOSITORY}/releases/download/{tag}/"
        assets = raw.get("assets")
        if not isinstance(assets, list):
            raise UpdateError("update_assets_missing")

        def asset(name: str) -> dict[str, object]:
            matches = [
                item for item in assets if isinstance(item, dict) and item.get("name") == name
            ]
            if len(matches) != 1 or matches[0].get("browser_download_url") != prefix + name:
                raise UpdateError("update_assets_missing")
            return matches[0]

        installer = asset(filename)
        asset("SHA256SUMS.txt")
        size = installer.get("size")
        if type(size) is not int or not 0 < size <= _MAX_INSTALLER:
            raise UpdateError("update_metadata_invalid")
        checksums = _read(prefix + "SHA256SUMS.txt", 256 * 1024, cancelled).decode("utf-8-sig")
        entries = [line.split() for line in checksums.splitlines()]
        hashes = [
            parts[0] for parts in entries if len(parts) == 2 and parts[1].lstrip("*") == filename
        ]
        if len(hashes) != 1 or _SHA256.fullmatch(hashes[0]) is None:
            raise UpdateError("update_checksum_missing")
        digest = installer.get("digest")
        if digest is not None and digest != "sha256:" + hashes[0]:
            raise UpdateError("update_checksum_mismatch")
        return AvailableUpdate(
            version,
            filename,
            prefix + filename,
            hashes[0],
            size,
            f"https://github.com/{REPOSITORY}/releases/tag/{tag}",
        )
    except urllib.error.HTTPError as error:
        if error.code == 404 and error.geturl() == LATEST_URL:
            return None
        raise UpdateError("update_check_unavailable") from error
    except (OSError, ValueError) as error:
        raise UpdateError("update_check_unavailable") from error


def download_update(
    release: AvailableUpdate,
    directory: Path,
    *,
    cancelled: Callable[[], bool] = lambda: False,
    progress: Callable[[int], None] = lambda _: None,
) -> Path:
    """Write only to the owned staging folder; never modify installed program files."""
    _check_cancelled(cancelled)
    try:
        directory.mkdir(parents=True, exist_ok=True)
        root = directory.resolve()
        destination = (root / release.filename).resolve()
        partial = (root / (release.filename + ".partial")).resolve()
        if not destination.is_relative_to(root) or not partial.is_relative_to(root):
            raise UpdateError("update_cache_invalid")
        if destination.is_file() and destination.stat().st_size == release.size:
            with destination.open("rb") as stream:
                if hashlib.file_digest(stream, "sha256").hexdigest() == release.sha256:
                    _check_cancelled(cancelled)
                    progress(100)
                    return destination
        digest, received, last_progress = hashlib.sha256(), 0, -1
        with _open(release.url) as response, partial.open("wb") as output:
            while chunk := response.read(1024 * 1024):
                _check_cancelled(cancelled)
                received += len(chunk)
                if received > release.size:
                    raise UpdateError("update_checksum_mismatch")
                output.write(chunk)
                digest.update(chunk)
                value = min(99, received * 100 // release.size)
                if value != last_progress:
                    progress(value)
                    last_progress = value
        _check_cancelled(cancelled)
        if received != release.size or digest.hexdigest() != release.sha256:
            partial.unlink(missing_ok=True)
            raise UpdateError("update_checksum_mismatch")
        partial.replace(destination)
        progress(100)
        return destination
    except OSError as error:
        raise UpdateError("update_download_failed") from error


def installer_command(
    installer: Path, install_root: Path, config_path: Path, settings_path: Path
) -> list[str]:
    return [
        str(installer),
        "/NORESTART",
        f"/DIR={install_root}",
        f"/PITWARDCONFIG={config_path}",
        f"/PITWARDSETTINGS={settings_path}",
    ]
