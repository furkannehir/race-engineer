import asyncio
from pathlib import Path

import pytest

from race_engineer.application.intelligence_cli import intelligence_replay
from race_engineer.cli import _parser
from race_engineer.core.intelligence import (
    ContextPacket,
    DriverTurn,
    EngineerBrief,
    GeneratedResponse,
    GroundedResponse,
)
from race_engineer.intelligence.local_model import LocalIntelligenceError
from race_engineer.intelligence.orchestrator import PreparedEngineerResponse

ROOT = Path(__file__).parents[1]
FIXTURE = ROOT / "fixtures/synthetic/intelligence"


class FixtureEngineer:
    def __init__(self, memory) -> None:
        self.memory = memory

    async def prepare(self, turn: DriverTurn) -> PreparedEngineerResponse:
        current = self.memory.latest_context()
        player_position = current.frame.player.position
        field_position = max(
            position
            for position in (
                player_position,
                *(opponent.position for opponent in current.frame.opponents),
            )
            if position is not None
        )
        is_last = player_position == field_position
        context = ContextPacket(
            turn_id=turn.turn_id,
            session_id=turn.session_id,
            generation=turn.generation,
            source_sequence=current.frame.sequence,
            assembled_at=current.frame.observed_at,
        )
        brief = EngineerBrief(
            turn_id=turn.turn_id,
            session_id=turn.session_id,
            generation=turn.generation,
            source_sequence=current.frame.sequence,
            goal="inform",
            language=turn.reply_language or "en",
            tone="calm_teammate",
        )
        response = GeneratedResponse(
            response_id=f"{turn.turn_id}:response",
            turn_id=turn.turn_id,
            session_id=turn.session_id,
            generation=turn.generation,
            language=brief.language,
            action="speak",
            speech_template="Yes, you are last." if is_last else "No, you are not last.",
        )
        return PreparedEngineerResponse(turn, context, brief, response)

    async def ground(self, prepared: PreparedEngineerResponse) -> GroundedResponse:
        return GroundedResponse(
            response_id=prepared.response.response_id,
            turn_id=prepared.turn.turn_id,
            session_id=prepared.turn.session_id,
            generation=prepared.turn.generation,
            source_sequence=prepared.context.source_sequence,
            language=prepared.brief.language,
            action="speak",
            text=prepared.response.speech_template,
        )


def test_intelligence_replay_uses_latest_fixture_history_by_default(monkeypatch, capsys):
    captured = {}

    def factory(config, memory):
        captured["samples"] = memory.sample_count
        return FixtureEngineer(memory)

    monkeypatch.setattr(
        "race_engineer.application.intelligence_cli.live_intelligence",
        factory,
    )

    code = asyncio.run(
        intelligence_replay(
            ROOT / "config/default.toml",
            FIXTURE,
            question="Am I last?",
        )
    )

    output = capsys.readouterr().out
    assert code == 0
    assert captured["samples"] == 3
    assert "frame 2/2" in output
    assert "Yes, you are last." in output


def test_intelligence_replay_can_select_not_last_frame_and_show_json(monkeypatch, capsys):
    monkeypatch.setattr(
        "race_engineer.application.intelligence_cli.live_intelligence",
        lambda config, memory: FixtureEngineer(memory),
    )

    code = asyncio.run(
        intelligence_replay(
            ROOT / "config/default.toml",
            FIXTURE,
            frame_index=0,
            question="Am I last?",
            json_output=True,
        )
    )

    output = capsys.readouterr().out
    assert code == 0
    assert '"frame_index": 0' in output
    assert '"text": "No, you are not last."' in output
    assert '"brief"' in output and '"generated"' in output


@pytest.mark.parametrize(
    ("reason", "expected_hint"),
    [
        ("model_unreachable", "server is not reachable"),
        ("model_timeout", "model timed out"),
        ("model_http_500", "server is reachable but rejected"),
    ],
)
def test_intelligence_replay_reports_local_model_failure(
    monkeypatch,
    capsys,
    reason,
    expected_hint,
):
    class FailingEngineer:
        async def prepare(self, turn):
            raise LocalIntelligenceError(reason)

    monkeypatch.setattr(
        "race_engineer.application.intelligence_cli.live_intelligence",
        lambda config, memory: FailingEngineer(),
    )

    code = asyncio.run(
        intelligence_replay(
            ROOT / "config/default.toml",
            FIXTURE,
            question="Am I last?",
        )
    )

    assert code == 1
    output = capsys.readouterr().out
    assert expected_hint in output
    if reason == "model_http_500":
        assert "Start it" not in output


def test_cli_accepts_documented_intelligence_replay_shape():
    args = _parser().parse_args(
        [
            "intelligence-replay",
            "--fixture",
            "fixtures/synthetic/intelligence",
            "--question",
            "Am I last?",
        ]
    )

    assert args.config == Path("config/default.toml")
    assert args.frame_index == -1
