"""Retire known managed model files only after their replacement has loaded."""

import hashlib
import json
import logging
import os
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib.resources import files
from pathlib import Path
from typing import Any, Literal

from pydantic import Field

from race_engineer.components import load_manifest
from race_engineer.config import ConversationConfig
from race_engineer.core.contracts import ContractModel
from race_engineer.installation import Installation
from race_engineer.radio_diagnostics import error_code

_LOGGER = logging.getLogger(__name__)
_JOURNAL = ".model-migrations.json"
_SCHEMA = "pitward-model-migrations.v1"


class RetiredModel(ContractModel):
    artifact_id: str = Field(pattern=r"^[a-z0-9.-]+$")
    destination: str = Field(pattern=r"^conversation/[A-Za-z0-9_.-]+\.gguf$")
    size: int = Field(gt=0, strict=True)
    sha256: str = Field(pattern=r"^[a-f0-9]{64}$")


class ModelMigration(ContractModel):
    migration_id: str = Field(pattern=r"^[a-z0-9.-]+$")
    model: str = Field(min_length=1)
    context_planner: str = Field(min_length=1)
    replacement_id: str = Field(pattern=r"^[a-z0-9.-]+$")
    retired: tuple[RetiredModel, ...] = Field(min_length=1)


@dataclass(frozen=True)
class CleanupResult:
    status: Literal["not_applicable", "completed", "deferred", "cancelled"]
    removed: tuple[str, ...] = ()
    bytes_freed: int = 0


class _Cancelled(Exception):
    pass


def load_migrations() -> tuple[ModelMigration, ...]:
    resource = files("race_engineer").joinpath("assets/components.v1.json")
    raw = json.loads(resource.read_text(encoding="utf-8"))
    migrations = tuple(
        ModelMigration.model_validate_json(json.dumps(item))
        for item in raw.get("model_migrations", [])
    )
    if len({m.migration_id for m in migrations}) != len(migrations):
        raise ValueError("duplicate_model_migration")
    return migrations


def _managed_path(root: Path, relative: str | Path) -> Path:
    """Resolve every destructive target and reject links within the owned root."""
    relative = Path(relative)
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise ValueError("model_cleanup_path_invalid")
    if root.is_symlink() or root.is_junction() or root.resolve() != root:
        raise ValueError("model_cleanup_root_redirected")
    target = root
    for part in relative.parts:
        target /= part
        if target.is_symlink() or target.is_junction():
            raise ValueError("model_cleanup_path_redirected")
    if not target.resolve().is_relative_to(root):
        raise ValueError("model_cleanup_path_invalid")
    return target


def _identity(path: Path) -> tuple[int, int, int, int]:
    stat = path.stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns


def _verified(
    path: Path, size: int, sha256: str, cancelled: Callable[[], bool]
) -> tuple[int, int, int, int] | None:
    before = _identity(path)
    if before[2] != size or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            if cancelled():
                raise _Cancelled
            digest.update(chunk)
    if cancelled():
        raise _Cancelled
    return before if digest.hexdigest() == sha256 and _identity(path) == before else None


def _record(root: Path, migration: ModelMigration, entry: dict[str, Any]) -> None:
    destination = _managed_path(root, _JOURNAL)
    if destination.exists():
        if destination.stat().st_size > 1024 * 1024:
            raise ValueError("model_cleanup_journal_invalid")
        raw = json.loads(destination.read_text(encoding="utf-8"))
        if not isinstance(raw, dict) or raw.get("schema_version") != _SCHEMA:
            raise ValueError("model_cleanup_journal_invalid")
        if not isinstance(raw.get("migrations"), dict):
            raise ValueError("model_cleanup_journal_invalid")
    else:
        raw = {"schema_version": _SCHEMA, "migrations": {}}
    raw["migrations"][migration.migration_id] = entry
    descriptor, filename = tempfile.mkstemp(prefix=".migration-", suffix=".tmp", dir=root)
    temporary = Path(filename)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(raw, stream, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, _managed_path(root, _JOURNAL))
    finally:
        temporary.unlink(missing_ok=True)


def retire_loaded_model(
    installation: Installation,
    config: ConversationConfig,
    loaded_model: Path,
    *,
    cancelled: Callable[[], bool] = lambda: False,
) -> CleanupResult:
    """Called only after an owned server reports successful model startup.

    Both replacement and retired files must match pinned sizes/checksums. Source
    launches, custom paths, external servers (excluded by caller), unknown files
    and simulator/speech assets never enter this cleanup path. Failure is best
    effort: it leaves the engineer usable and cleanup can retry on a later start.
    """
    if not installation.frozen:
        return CleanupResult("not_applicable")
    removed: list[str] = []
    freed = 0
    try:
        if cancelled():
            raise _Cancelled
        root = installation.components_root.absolute()
        _, artifacts = load_manifest()
        for migration in load_migrations():
            if (config.model, config.context_planner) != (
                migration.model,
                migration.context_planner,
            ):
                continue
            replacement = next(a for a in artifacts if a.artifact_id == migration.replacement_id)
            if replacement.kind != "file":
                raise ValueError("model_cleanup_replacement_invalid")
            target = _managed_path(root, replacement.destination)
            if loaded_model.resolve() != target or config.runtime.model_path.resolve() != target:
                continue
            protected = {a.destination for a in artifacts}
            candidates: list[tuple[RetiredModel, tuple[int, int, int, int]]] = []
            for old in migration.retired:
                if Path(old.destination) in protected:
                    raise ValueError("model_cleanup_active_artifact")
                path = _managed_path(root, old.destination)
                if not path.exists():
                    continue
                verified = _verified(path, old.size, old.sha256, cancelled)
                if verified is None:
                    raise ValueError("model_cleanup_unrecognized_file")
                candidates.append((old, verified))
            if not candidates:
                continue
            if _verified(target, replacement.size, replacement.sha256, cancelled) is None:
                raise ValueError("model_cleanup_replacement_unverified")
            entry = {
                "model": config.model,
                "replacement_id": replacement.artifact_id,
                "replacement_sha256": replacement.sha256,
                "loaded_at": datetime.now(UTC).isoformat(),
                "state": "cleanup_pending",
                "removed": [],
            }
            # A durable success record is required before touching any old file.
            _record(root, migration, entry)
            for old, verified in candidates:
                if cancelled():
                    raise _Cancelled
                path = _managed_path(root, old.destination)
                if _identity(path) != verified:
                    raise ValueError("model_cleanup_file_changed")
                path.unlink()
                removed.append(old.artifact_id)
                freed += old.size
                entry["removed"] = list(removed)
                _LOGGER.info(
                    "retired obsolete conversation model",
                    extra={
                        "event": "model_cleanup_removed",
                        "artifact_id": old.artifact_id,
                        "replacement_id": replacement.artifact_id,
                        "bytes_freed": old.size,
                    },
                )
            entry["state"] = "completed"
            _record(root, migration, entry)
        return CleanupResult("completed" if removed else "not_applicable", tuple(removed), freed)
    except _Cancelled:
        return CleanupResult("cancelled", tuple(removed), freed)
    except (OSError, ValueError, StopIteration, KeyError, TypeError) as error:
        _LOGGER.warning(
            "obsolete model cleanup deferred; engineer remains available",
            extra={
                "event": "model_cleanup_deferred",
                "reason": error_code(error),
            },
        )
        return CleanupResult("deferred", tuple(removed), freed)
