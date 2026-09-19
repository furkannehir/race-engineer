"""Fixture-compatible recording of live telemetry and deterministic policy output."""

from collections.abc import Sequence
from pathlib import Path
from types import TracebackType
from typing import TextIO

from race_engineer.core.contracts import (
    PolicyDecision,
    RaceEvent,
    SpeechIntent,
    TelemetryFrame,
    Utterance,
    canonical_json,
)
from race_engineer.fixtures import FixtureManifest


class TelemetrySessionRecorder:
    """Creates a new replayable session directory without overwriting existing data."""

    def __init__(self, directory: Path, fixture_id: str, description: str) -> None:
        self._directory = directory
        self._fixture_id = fixture_id
        self._description = description
        self._frames_stream: TextIO | None = None
        self._events_stream: TextIO | None = None
        self._intents_stream: TextIO | None = None
        self._decisions_stream: TextIO | None = None
        self._utterances_stream: TextIO | None = None

    def __enter__(self) -> "TelemetrySessionRecorder":
        self._directory.mkdir(parents=True, exist_ok=False)
        manifest = FixtureManifest(
            fixture_version="race-fixture.v3",
            fixture_id=self._fixture_id,
            description=self._description,
            frames_file="frames.jsonl",
            expected_events_file="events.jsonl",
            expected_intents_file="intents.jsonl",
            expected_decisions_file="decisions.jsonl",
            expected_utterances_file="utterances.jsonl",
        )
        (self._directory / "manifest.json").write_text(
            manifest.model_dump_json(indent=2) + "\n",
            encoding="utf-8",
        )
        self._frames_stream = (self._directory / "frames.jsonl").open("x", encoding="utf-8")
        self._events_stream = (self._directory / "events.jsonl").open("x", encoding="utf-8")
        self._intents_stream = (self._directory / "intents.jsonl").open("x", encoding="utf-8")
        self._decisions_stream = (self._directory / "decisions.jsonl").open("x", encoding="utf-8")
        self._utterances_stream = (self._directory / "utterances.jsonl").open("x", encoding="utf-8")
        return self

    def write_frame(self, frame: TelemetryFrame) -> None:
        if self._frames_stream is None:
            raise RuntimeError("recorder is not open")
        self._frames_stream.write(canonical_json(frame) + "\n")
        self._frames_stream.flush()

    def write_events(self, events: Sequence[RaceEvent]) -> None:
        if self._events_stream is None:
            raise RuntimeError("recorder is not open")
        for event in events:
            self._events_stream.write(canonical_json(event) + "\n")
        self._events_stream.flush()

    def write_intents(self, intents: Sequence[SpeechIntent]) -> None:
        if self._intents_stream is None:
            raise RuntimeError("recorder is not open")
        for intent in intents:
            self._intents_stream.write(canonical_json(intent) + "\n")
        self._intents_stream.flush()

    def write_decisions(self, decisions: Sequence[PolicyDecision]) -> None:
        if self._decisions_stream is None:
            raise RuntimeError("recorder is not open")
        for decision in decisions:
            self._decisions_stream.write(canonical_json(decision) + "\n")
        self._decisions_stream.flush()

    def write_utterances(self, utterances: Sequence[Utterance]) -> None:
        if self._utterances_stream is None:
            raise RuntimeError("recorder is not open")
        for utterance in utterances:
            self._utterances_stream.write(canonical_json(utterance) + "\n")
        self._utterances_stream.flush()

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del exception_type, exception, traceback
        if self._frames_stream is not None:
            self._frames_stream.close()
        if self._events_stream is not None:
            self._events_stream.close()
        if self._intents_stream is not None:
            self._intents_stream.close()
        if self._decisions_stream is not None:
            self._decisions_stream.close()
        if self._utterances_stream is not None:
            self._utterances_stream.close()
        self._frames_stream = None
        self._events_stream = None
        self._intents_stream = None
        self._decisions_stream = None
        self._utterances_stream = None
