from pathlib import Path

import pytest

from race_engineer.core.enums import EventType, SessionPhase
from race_engineer.fixtures import load_fixture

ROOT = Path(__file__).parents[1]
FIXTURE = ROOT / "fixtures" / "synthetic" / "green_flag"
M2_FIXTURE = ROOT / "fixtures" / "synthetic" / "m2_green_flag"


def test_green_flag_fixture_loads_all_versioned_streams() -> None:
    bundle = load_fixture(FIXTURE)
    assert bundle.manifest.fixture_id == "synthetic-green-flag-v1"
    assert len(bundle.frames) == 2
    assert bundle.frames[0].session_phase is SessionPhase.FORMATION
    assert bundle.frames[1].session_phase is SessionPhase.GREEN
    assert bundle.expected_events[0].event_type is EventType.SESSION_PHASE_CHANGED
    assert bundle.expected_intents[0].facts["message"] == "Green flag"


def test_fixture_paths_cannot_escape_their_directory(tmp_path: Path) -> None:
    fixture_dir = tmp_path / "fixture"
    fixture_dir.mkdir()
    (fixture_dir / "manifest.json").write_text(
        """{
          "fixture_version": "race-fixture.v1",
          "fixture_id": "unsafe",
          "description": "Unsafe path must be rejected.",
          "frames_file": "../frames.jsonl"
        }""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="escapes"):
        load_fixture(fixture_dir)


def test_m2_fixture_includes_expected_policy_intent() -> None:
    bundle = load_fixture(M2_FIXTURE)
    assert len(bundle.expected_events) == 2
    assert len(bundle.expected_intents) == 1
    assert bundle.expected_intents[0].facts["phase"] == "green"
