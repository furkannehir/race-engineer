from pathlib import Path

import pytest

from race_engineer.telemetry.iracing.raw import IracingRawSample, load_raw_samples


@pytest.fixture
def iracing_samples() -> tuple[IracingRawSample, IracingRawSample]:
    root = Path(__file__).parents[1]
    samples = load_raw_samples(root / "fixtures" / "iracing" / "m1_sample" / "raw_samples.jsonl")
    assert len(samples) == 2
    return samples[0], samples[1]
