from pathlib import Path

import pytest

from race_engineer.core.enums import EventType, SessionPhase
from race_engineer.fixtures import load_fixture

ROOT = Path(__file__).parents[1]
FIXTURE = ROOT / "fixtures" / "synthetic" / "green_flag"
M2_FIXTURE = ROOT / "fixtures" / "synthetic" / "m2_green_flag"
M3_FIXTURE = ROOT / "fixtures" / "synthetic" / "m3_green_flag"


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
    assert bundle.manifest.fixture_version == "race-fixture.v2"
    assert len(bundle.expected_events) == 2
    assert len(bundle.expected_intents) == 1
    assert len(bundle.expected_decisions) == 2
    assert bundle.expected_intents[0].facts["phase"] == "green"


def test_v1_manifest_rejects_policy_decision_stream(tmp_path: Path) -> None:
    fixture_dir = tmp_path / "v1-with-decisions"
    fixture_dir.mkdir()
    (fixture_dir / "manifest.json").write_text(
        """{
          "fixture_version": "race-fixture.v1",
          "fixture_id": "invalid-v1",
          "description": "Version 1 cannot contain decisions.",
          "expected_decisions_file": "decisions.jsonl"
        }""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"race-fixture\.v2"):
        load_fixture(fixture_dir)


def test_m3_fixture_includes_expected_utterance() -> None:
    bundle = load_fixture(M3_FIXTURE)
    assert bundle.manifest.fixture_version == "race-fixture.v3"
    assert len(bundle.expected_utterances) == 1
    assert bundle.expected_utterances[0].text == "Green flag"


def test_v2_manifest_rejects_utterance_stream(tmp_path: Path) -> None:
    fixture_dir = tmp_path / "v2-with-utterances"
    fixture_dir.mkdir()
    (fixture_dir / "manifest.json").write_text(
        """{
          "fixture_version": "race-fixture.v2",
          "fixture_id": "invalid-v2",
          "description": "Version 2 cannot contain utterances.",
          "expected_utterances_file": "utterances.jsonl"
        }""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"race-fixture\.v3"):
        load_fixture(fixture_dir)
