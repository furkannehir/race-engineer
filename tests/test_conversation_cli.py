import asyncio
from pathlib import Path

import pytest

from race_engineer.application.conversation_cli import chat_replay
from race_engineer.conversation.local_model import ConversationModelError
from race_engineer.core.conversation import ConversationPlan

ROOT = Path(__file__).parents[1]


class Planner:
    def __init__(self, config):
        self.config = config

    async def plan(self, request):
        return ConversationPlan(language="en", queries=("position",), clarification="none")


def test_one_shot_replay_cli(monkeypatch, capsys):
    monkeypatch.setattr("race_engineer.application.conversation_cli.LocalQwenPlanner", Planner)
    code = asyncio.run(
        chat_replay(
            ROOT / "config/default.toml",
            ROOT / "fixtures/synthetic/conversation",
            frame_index=1,
            question="Where are we?",
            json_output=True,
        )
    )
    assert code == 0
    output = capsys.readouterr().out
    assert '"source_sequence": 200' in output
    assert "P5" in output
    assert '"mode": "replay"' in output


def test_interactive_controls_and_history_reset(monkeypatch, capsys):
    monkeypatch.setattr("race_engineer.application.conversation_cli.LocalQwenPlanner", Planner)
    lines = iter(
        ["", "/next", "Position?", "/frame 0", "Position?", "/next -1", "/reset", "/state", "/quit"]
    )
    monkeypatch.setattr("builtins.input", lambda prompt: next(lines))
    code = asyncio.run(
        chat_replay(ROOT / "config/default.toml", ROOT / "fixtures/synthetic/conversation")
    )
    output = capsys.readouterr().out
    assert code == 0
    assert "P5" in output and "P6" in output
    assert "history cleared" in output
    assert "Invalid input" in output


def test_model_failure_is_reported_without_traceback(monkeypatch, capsys):
    class FailingPlanner(Planner):
        async def plan(self, request):
            raise ConversationModelError("model_unreachable")

    monkeypatch.setattr(
        "race_engineer.application.conversation_cli.LocalQwenPlanner", FailingPlanner
    )
    code = asyncio.run(
        chat_replay(
            ROOT / "config/default.toml",
            ROOT / "fixtures/synthetic/conversation",
            question="Position?",
        )
    )
    assert code == 1
    assert "Start the local Qwen server" in capsys.readouterr().out


def test_json_requires_single_question():
    with pytest.raises(ValueError, match="requires --question"):
        asyncio.run(
            chat_replay(
                ROOT / "config/default.toml",
                ROOT / "fixtures/synthetic/conversation",
                json_output=True,
            )
        )
