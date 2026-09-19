from pathlib import Path

from race_engineer.core.enums import RaceFlag, SessionPhase
from race_engineer.fixtures import load_fixture
from race_engineer.telemetry.iracing import IracingEventDeriver, normalize_sample
from race_engineer.telemetry.iracing.raw import IracingRawSample, load_raw_samples

ROOT = Path(__file__).parents[1]
FIXTURE_DIRECTORY = ROOT / "fixtures" / "iracing" / "m1_sample"


def test_recorded_samples_match_expected_normalized_frames_and_events() -> None:
    samples = load_raw_samples(FIXTURE_DIRECTORY / "raw_samples.jsonl")
    expected = load_fixture(FIXTURE_DIRECTORY)

    frames = tuple(normalize_sample(sample) for sample in samples)
    assert frames == expected.frames
    assert all(frame is not None for frame in frames)

    deriver = IracingEventDeriver()
    events = tuple(deriver.derive(frames[0], frames[1]))  # type: ignore[arg-type]
    assert events == expected.expected_events


def test_normalization_preserves_privacy_safe_ids_and_signed_race_gap() -> None:
    sample = load_raw_samples(FIXTURE_DIRECTORY / "raw_samples.jsonl")[0]
    frame = normalize_sample(sample)
    assert frame is not None
    assert frame.player.driver_id == "iracing:user:101"
    assert frame.opponents[0].driver_id == "iracing:user:102"
    assert frame.opponents[0].gap_to_player_s == -3.2
    assert frame.session_phase is SessionPhase.FORMATION


def test_incomplete_or_invalid_identity_is_not_fabricated() -> None:
    base = load_raw_samples(FIXTURE_DIRECTORY / "raw_samples.jsonl")[0]
    missing = IracingRawSample.model_validate({**base.model_dump(), "session_tick": None})
    invalid_player = IracingRawSample.model_validate({**base.model_dump(), "player_car_idx": -1})
    assert normalize_sample(missing) is None
    assert normalize_sample(invalid_player) is None


def test_flag_bits_and_caution_phase_are_normalized() -> None:
    base = load_raw_samples(FIXTURE_DIRECTORY / "raw_samples.jsonl")[1]
    caution = IracingRawSample.model_validate(
        {
            **base.model_dump(),
            "session_flags": 0x0008 | 0x0020 | 0x010000,
        }
    )
    frame = normalize_sample(caution)
    assert frame is not None
    assert frame.flags == (RaceFlag.YELLOW, RaceFlag.BLUE, RaceFlag.BLACK)
    assert frame.session_phase is SessionPhase.CAUTION
