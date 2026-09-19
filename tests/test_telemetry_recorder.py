from pathlib import Path

import pytest

from race_engineer.fixtures import load_fixture
from race_engineer.telemetry.iracing import IracingEventDeriver, normalize_sample
from race_engineer.telemetry.iracing.raw import IracingRawSample
from race_engineer.telemetry.recorder import TelemetrySessionRecorder


def test_recorder_creates_loadable_fixture_without_overwriting(
    tmp_path: Path,
    iracing_samples: tuple[IracingRawSample, IracingRawSample],
) -> None:
    frames = tuple(normalize_sample(sample) for sample in iracing_samples)
    assert frames[0] is not None and frames[1] is not None
    events = tuple(IracingEventDeriver().derive(frames[0], frames[1]))
    output = tmp_path / "recording"

    with TelemetrySessionRecorder(output, "recording-1", "Test recording.") as recorder:
        recorder.write_frame(frames[0])
        recorder.write_frame(frames[1])
        recorder.write_events(events)

    loaded = load_fixture(output)
    assert loaded.frames == frames
    assert loaded.expected_events == events
    with pytest.raises(FileExistsError):
        TelemetrySessionRecorder(output, "recording-2", "Must fail.").__enter__()
