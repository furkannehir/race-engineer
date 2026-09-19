"""Fixture-compatible recording of normalized frames and derived events."""

from pathlib import Path
from types import TracebackType
from typing import TextIO

from race_engineer.core.contracts import RaceEvent, TelemetryFrame, canonical_json
from race_engineer.fixtures import FixtureManifest


class TelemetrySessionRecorder:
    """Creates a new recording directory and never overwrites an existing session."""

    def __init__(self, directory: Path, fixture_id: str, description: str) -> None:
        self._directory = directory
        self._fixture_id = fixture_id
        self._description = description
        self._frames_stream: TextIO | None = None
        self._events_stream: TextIO | None = None

    def __enter__(self) -> "TelemetrySessionRecorder":
        self._directory.mkdir(parents=True, exist_ok=False)
        manifest = FixtureManifest(
            fixture_id=self._fixture_id,
            description=self._description,
            frames_file="frames.jsonl",
            expected_events_file="events.jsonl",
        )
        (self._directory / "manifest.json").write_text(
            manifest.model_dump_json(indent=2) + "\n",
            encoding="utf-8",
        )
        self._frames_stream = (self._directory / "frames.jsonl").open("x", encoding="utf-8")
        self._events_stream = (self._directory / "events.jsonl").open("x", encoding="utf-8")
        return self

    def write_frame(self, frame: TelemetryFrame) -> None:
        if self._frames_stream is None:
            raise RuntimeError("recorder is not open")
        self._frames_stream.write(canonical_json(frame) + "\n")
        self._frames_stream.flush()

    def write_events(self, events: tuple[RaceEvent, ...]) -> None:
        if self._events_stream is None:
            raise RuntimeError("recorder is not open")
        for event in events:
            self._events_stream.write(canonical_json(event) + "\n")
        self._events_stream.flush()

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
        self._frames_stream = None
        self._events_stream = None
