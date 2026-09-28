import asyncio
import json
from pathlib import Path

import pytest

from race_engineer.application.conversation_cli import chat_replay
from race_engineer.config import load_config
from race_engineer.conversation.local_model import ConversationModelError
from race_engineer.core.conversation import ConversationPlan
from race_engineer.core.dialogue import QueryPart, SemanticProposal
from race_engineer.testing.dialogue import ScriptedSemanticJudge

ROOT = Path(__file__).parents[1]


class Planner:
    def __init__(self, config):
        self.config = config

    async def plan(self, request):
        return ConversationPlan(language="en", queries=("position",), clarification="none")


@pytest.fixture
def legacy_route(monkeypatch):
    config = load_config(ROOT / "config/default.toml")
    dialogue = config.conversation.dialogue.model_copy(update={"enabled": False})
    conversation = config.conversation.model_copy(update={"dialogue": dialogue})
    legacy = config.model_copy(update={"conversation": conversation})
    monkeypatch.setattr(
        "race_engineer.application.conversation_cli.load_config", lambda _path: legacy
    )


def test_one_shot_replay_cli(monkeypatch, capsys, legacy_route):
    monkeypatch.setattr("race_engineer.application.conversation_cli.conversation_planner", Planner)
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


def test_interactive_controls_and_history_reset(monkeypatch, capsys, legacy_route):
    monkeypatch.setattr("race_engineer.application.conversation_cli.conversation_planner", Planner)
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


def test_model_failure_is_reported_without_traceback(monkeypatch, capsys, legacy_route):
    class FailingPlanner(Planner):
        async def plan(self, request):
            raise ConversationModelError("model_unreachable")

    monkeypatch.setattr(
        "race_engineer.application.conversation_cli.conversation_planner", FailingPlanner
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


def test_one_shot_replay_can_use_ce05_preview(monkeypatch, capsys):
    config = load_config(ROOT / "config/default.toml")
    dialogue = config.conversation.dialogue.model_copy(update={"enabled": True})
    conversation = config.conversation.model_copy(update={"dialogue": dialogue})
    preview = config.model_copy(update={"conversation": conversation})
    judge = ScriptedSemanticJudge(
        SemanticProposal(
            language="en",
            requests=(QueryPart(part_id="q1", query="position"),),
            acts=("acknowledge",),
            model_id="scripted",
        )
    )
    monkeypatch.setattr(
        "race_engineer.application.conversation_cli.load_config", lambda _path: preview
    )
    monkeypatch.setattr(
        "race_engineer.application.conversation_cli.semantic_judge", lambda _config: judge
    )

    code = asyncio.run(
        chat_replay(
            ROOT / "config/default.toml",
            ROOT / "fixtures/synthetic/conversation",
            frame_index=1,
            question="That was dirty. Where are we?",
            json_output=True,
        )
    )

    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    assert payload["result"]["outcome"] == "answered"
    assert payload["text"] in {
        "Copy. P5 right now.",
        "Copy. We're running P5.",
        "Yeah, copy. P5 right now.",
        "Yeah, copy. We're running P5.",
    }
