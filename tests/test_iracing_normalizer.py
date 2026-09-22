from pathlib import Path

from race_engineer.core.enums import EventType, RaceFlag, SessionPhase
from race_engineer.fixtures import load_fixture
from race_engineer.telemetry.iracing import IracingEventDeriver, normalize_sample
from race_engineer.telemetry.iracing.raw import IracingRawSample, load_raw_samples

ROOT = Path(__file__).parents[1]
FIXTURE_DIRECTORY = ROOT / "fixtures" / "iracing" / "m1_sample"
IR001_FIXTURE_DIRECTORY = ROOT / "fixtures" / "iracing" / "ir001_false_blue_start"


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
            "player_position": 2,
            "car_idx_positions": [2, 1, 0],
            "car_idx_laps_completed": [5, 6, -1],
        }
    )
    frame = normalize_sample(caution)
    assert frame is not None
    assert frame.flags == (RaceFlag.YELLOW, RaceFlag.BLUE, RaceFlag.BLACK)
    assert frame.session_phase is SessionPhase.CAUTION


def test_ir001_false_start_blue_is_suppressed_until_lapping_traffic_exists() -> None:
    samples = load_raw_samples(IR001_FIXTURE_DIRECTORY / "raw_samples.jsonl")
    frames = tuple(normalize_sample(sample) for sample in samples)
    assert all(frame is not None for frame in frames)

    green, false_blue, classified_same_lap, genuine_blue = frames
    assert green is not None
    assert false_blue is not None
    assert classified_same_lap is not None
    assert genuine_blue is not None
    assert green.flags == (RaceFlag.GREEN,)
    assert false_blue.flags == (RaceFlag.GREEN,)
    assert classified_same_lap.flags == (RaceFlag.GREEN,)
    assert genuine_blue.flags == (RaceFlag.GREEN, RaceFlag.BLUE)

    deriver = IracingEventDeriver()
    assert deriver.derive(green, false_blue) == ()
    assert deriver.derive(false_blue, classified_same_lap) == ()
    events = deriver.derive(classified_same_lap, genuine_blue)
    flag_events = tuple(event for event in events if event.event_type is EventType.FLAG_CHANGED)
    assert len(flag_events) == 1
    assert flag_events[0].facts == {"added": ["blue"], "removed": []}


def test_non_race_blue_does_not_require_a_completed_lap_advantage() -> None:
    base = load_raw_samples(FIXTURE_DIRECTORY / "raw_samples.jsonl")[1]
    practice = IracingRawSample.model_validate(
        {
            **base.model_dump(),
            "session_type": "Practice",
            "session_flags": 0x0020,
            "car_idx_laps_completed": [5, 5, -1],
        }
    )

    frame = normalize_sample(practice)
    assert frame is not None
    assert frame.flags == (RaceFlag.BLUE,)
