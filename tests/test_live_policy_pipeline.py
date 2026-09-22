import asyncio
from datetime import UTC, datetime
from pathlib import Path

import pytest

from race_engineer.cli import _read_iracing, _replay_language, _replay_policy
from race_engineer.config import IracingTelemetryConfig
from race_engineer.fixtures import load_fixture
from race_engineer.memory import DurableHistory, SqliteDriverProfileRepository
from race_engineer.testing import RecordingTextToSpeechEngine

ROOT = Path(__file__).parents[1]


@pytest.fixture
def recording_tts(monkeypatch: pytest.MonkeyPatch) -> RecordingTextToSpeechEngine:
    engine = RecordingTextToSpeechEngine()
    monkeypatch.setattr("race_engineer.cli.tts_factory", lambda config: engine)
    return engine


class FixtureTelemetryAdapter:
    def __init__(self, config: IracingTelemetryConfig) -> None:
        del config
        frames = load_fixture(ROOT / "fixtures" / "synthetic" / "m2_green_flag").frames
        first_observed_at = frames[0].observed_at
        live_start = datetime.now(UTC)
        self._frames = tuple(
            frame.model_copy(
                update={"observed_at": live_start + (frame.observed_at - first_observed_at)}
            )
            for frame in frames
        )

    async def stream(self):
        for frame in self._frames:
            yield frame


def test_live_reader_records_policy_intents_and_decisions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    recording_tts: RecordingTextToSpeechEngine,
) -> None:
    monkeypatch.setattr("race_engineer.cli.IracingTelemetryAdapter", FixtureTelemetryAdapter)
    output = tmp_path / "live-policy"
    repository = SqliteDriverProfileRepository(tmp_path / "history.sqlite3")
    profile = repository.ensure_default_profile()
    history = DurableHistory(repository, profile.profile_id)

    frames = asyncio.run(
        _read_iracing(
            ROOT / "config" / "default.toml",
            limit=0,
            output=output,
            history=history,
        )
    )
    recording = load_fixture(output)

    assert frames == 2
    assert len(recording.expected_events) == 2
    assert len(recording.expected_intents) == 1
    assert len(recording.expected_decisions) == 2
    assert len(recording.expected_utterances) == 1
    assert recording.expected_intents[0].facts["phase"] == "green"
    assert recording.expected_utterances[0].text == "Green flag"
    assert [utterance.text for utterance in recording_tts.utterances] == ["Green flag"]
    summaries = repository.recent_session_history(profile.profile_id)
    assert len(summaries) == 1
    assert summaries[0].decisions == 2
    assert summaries[0].approved == 1 and summaries[0].suppressed == 1
    assert summaries[0].radio_completed == 1

    replay = asyncio.run(_replay_policy(ROOT / "config" / "default.toml", output))
    assert replay["expected_intents_match"] is True
    assert replay["expected_decisions_match"] is True
    language_replay = asyncio.run(_replay_language(ROOT / "config" / "default.toml", output))
    assert language_replay["expected_utterances_match"] is True


class FailingPolicy:
    def __init__(self, *args, **kwargs) -> None:
        del args, kwargs

    async def decide(self, context):
        del context
        raise RuntimeError("test policy failure")


def test_live_reader_continues_recording_when_policy_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    recording_tts: RecordingTextToSpeechEngine,
) -> None:
    monkeypatch.setattr("race_engineer.cli.IracingTelemetryAdapter", FixtureTelemetryAdapter)
    monkeypatch.setattr("race_engineer.cli.StrictRulePolicy", FailingPolicy)
    output = tmp_path / "policy-failure"

    frames = asyncio.run(
        _read_iracing(
            ROOT / "config" / "default.toml",
            limit=0,
            output=output,
        )
    )
    recording = load_fixture(output)

    assert frames == 2
    assert len(recording.frames) == 2
    assert len(recording.expected_events) == 2
    assert recording.expected_intents == ()
    assert recording.expected_decisions == ()
    assert recording.expected_utterances == ()
    assert recording_tts.utterances == []


class FailingLanguageGenerator:
    async def generate(self, intent):
        del intent
        raise RuntimeError("test language failure")


def test_live_reader_continues_when_language_generation_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    recording_tts: RecordingTextToSpeechEngine,
) -> None:
    monkeypatch.setattr("race_engineer.cli.IracingTelemetryAdapter", FixtureTelemetryAdapter)
    monkeypatch.setattr(
        "race_engineer.cli.language_factory",
        lambda config: FailingLanguageGenerator(),
    )
    output = tmp_path / "language-failure"

    frames = asyncio.run(
        _read_iracing(
            ROOT / "config" / "default.toml",
            limit=0,
            output=output,
        )
    )
    recording = load_fixture(output)

    assert frames == 2
    assert len(recording.expected_intents) == 1
    assert len(recording.expected_decisions) == 2
    assert recording.expected_utterances == ()
    assert recording_tts.utterances == []


def test_live_reader_continues_when_tts_initialization_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("race_engineer.cli.IracingTelemetryAdapter", FixtureTelemetryAdapter)

    def fail_tts_factory(config):
        del config
        raise RuntimeError("test TTS initialization failure")

    monkeypatch.setattr("race_engineer.cli.tts_factory", fail_tts_factory)
    output = tmp_path / "tts-start-failure"

    frames = asyncio.run(
        _read_iracing(
            ROOT / "config" / "default.toml",
            limit=0,
            output=output,
        )
    )
    recording = load_fixture(output)

    assert frames == 2
    assert len(recording.expected_intents) == 1
    assert len(recording.expected_utterances) == 1
