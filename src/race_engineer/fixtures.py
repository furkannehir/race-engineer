"""Versioned deterministic replay-fixture loader."""

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from race_engineer.core.contracts import (
    PolicyDecision,
    RaceEvent,
    SpeechIntent,
    TelemetryFrame,
)


class FixtureManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    fixture_version: Literal["race-fixture.v1", "race-fixture.v2"] = "race-fixture.v1"
    fixture_id: str = Field(min_length=1)
    description: str = Field(min_length=1)
    frames_file: str = "frames.jsonl"
    expected_events_file: str | None = None
    expected_intents_file: str | None = None
    expected_decisions_file: str | None = None

    @model_validator(mode="after")
    def validate_versioned_streams(self) -> "FixtureManifest":
        if self.fixture_version == "race-fixture.v1" and self.expected_decisions_file is not None:
            raise ValueError("policy decision streams require race-fixture.v2")
        return self


class FixtureBundle(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    manifest: FixtureManifest
    frames: tuple[TelemetryFrame, ...]
    expected_events: tuple[RaceEvent, ...] = ()
    expected_intents: tuple[SpeechIntent, ...] = ()
    expected_decisions: tuple[PolicyDecision, ...] = ()


def _safe_child(directory: Path, relative_name: str) -> Path:
    root = directory.resolve()
    child = (root / relative_name).resolve()
    if not child.is_relative_to(root):
        raise ValueError(f"fixture path escapes its directory: {relative_name!r}")
    return child


def _load_jsonl[ModelT: BaseModel](path: Path, model: type[ModelT]) -> tuple[ModelT, ...]:
    values: list[ModelT] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            values.append(model.model_validate_json(line))
        except (ValueError, json.JSONDecodeError) as error:
            raise ValueError(f"invalid {path.name}:{line_number}: {error}") from error
    return tuple(values)


def load_fixture(directory: Path) -> FixtureBundle:
    manifest_path = directory / "manifest.json"
    manifest = FixtureManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    frames = _load_jsonl(_safe_child(directory, manifest.frames_file), TelemetryFrame)
    events = (
        _load_jsonl(_safe_child(directory, manifest.expected_events_file), RaceEvent)
        if manifest.expected_events_file
        else ()
    )
    intents = (
        _load_jsonl(_safe_child(directory, manifest.expected_intents_file), SpeechIntent)
        if manifest.expected_intents_file
        else ()
    )
    decisions = (
        _load_jsonl(
            _safe_child(directory, manifest.expected_decisions_file),
            PolicyDecision,
        )
        if manifest.expected_decisions_file
        else ()
    )
    return FixtureBundle(
        manifest=manifest,
        frames=frames,
        expected_events=events,
        expected_intents=intents,
        expected_decisions=decisions,
    )
