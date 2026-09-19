import asyncio
from pathlib import Path

import pytest

from race_engineer.cli import _read_iracing, _replay_language, _replay_policy
from race_engineer.config import IracingTelemetryConfig
from race_engineer.fixtures import load_fixture

ROOT = Path(__file__).parents[1]


class FixtureTelemetryAdapter:
    def __init__(self, config: IracingTelemetryConfig) -> None:
        del config
        self._frames = load_fixture(ROOT / "fixtures" / "synthetic" / "m2_green_flag").frames

    async def stream(self):
        for frame in self._frames:
            yield frame


def test_live_reader_records_policy_intents_and_decisions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("race_engineer.cli.IracingTelemetryAdapter", FixtureTelemetryAdapter)
    output = tmp_path / "live-policy"

    frames = asyncio.run(
        _read_iracing(
            ROOT / "config" / "default.toml",
            limit=0,
            output=output,
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


class FailingLanguageGenerator:
    async def generate(self, intent):
        del intent
        raise RuntimeError("test language failure")


def test_live_reader_continues_when_language_generation_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
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
