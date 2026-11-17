import asyncio
from types import SimpleNamespace

import pytest
from test_lfm3_dialogue import setup

from race_engineer.config import ConversationConfig
from scripts import try_ministral as cli


@pytest.fixture
def demo(monkeypatch):
    _, memory, _ = setup()
    config = ConversationConfig(model="mistralai/Ministral-3-3B-Instruct-2512")
    args = SimpleNamespace(
        backend="cpu", language="en", port=18197, trace=False, question=None, raw=False,
    )
    state = {"events": [], "calls": [], "startup_failure": False, "timeout": False}
    monkeypatch.setattr(cli, "prepare", lambda _: (config, memory))
    monkeypatch.setattr(cli, "require_free_port", lambda _: None)

    class Server:
        def __init__(self, *_):
            self.process = object()
            self.effective = SimpleNamespace(backend="cpu")

        async def start(self):
            state["events"].append("start")
            if state["startup_failure"]:
                raise RuntimeError("startup_failed")

        async def aclose(self):
            state["events"].append("close")
            self.process = None

    class Dialogue:
        def __init__(self, *_args, **_kwargs):
            self.history = []

        def clear(self):
            self.history.clear()

        async def respond(self, question, language, turn_id):
            state["calls"].append((question, language, len(self.history)))
            self.history.append(question)
            row = {
                "raw_reply": "I'm here—what\u2019s next?", "duration_ms": 123,
                "application_validation": {"status": "rejected", "reason": "test_rejection"},
            }
            if state["timeout"]:
                row.update(raw_reply=None, failure="dialogue_timeout")
            return row

    monkeypatch.setattr(cli, "ConversationServer", Server)
    monkeypatch.setattr(cli, "RawConversation", Dialogue)
    monkeypatch.setattr(cli, "MinistralConversation", Dialogue)
    return args, state


def test_language_switch_and_clear_reset_history_without_reloading_server(
    monkeypatch, demo, capsys
):
    args, state = demo
    args.trace = True
    args.raw = True
    inputs = iter(["Radio check", "Position?", "/tr", "Orada mısın?", "/clear", "Sıra?", "/quit"])
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))
    assert asyncio.run(cli.chat(args)) == 0
    assert state["calls"] == [
        ("Radio check", "en", 0), ("Position?", "en", 1),
        ("Orada mısın?", "tr", 0), ("Sıra?", "tr", 0),
    ]
    assert state["events"] == ["start", "close"]
    output = capsys.readouterr().out
    assert "what\u2019s next?" in output and "\\u2019" not in output
    assert "App validation: rejected" in output


def test_startup_failure_still_closes_server(demo):
    args, state = demo
    state["startup_failure"] = True
    with pytest.raises(RuntimeError, match="startup_failed"):
        asyncio.run(cli.chat(args))
    assert state["events"] == ["start", "close"] and not state["calls"]


def test_timeout_stops_before_another_question_and_closes_server(monkeypatch, demo):
    args, state = demo
    state["timeout"] = True
    monkeypatch.setattr("builtins.input", lambda _: "Position?")
    assert asyncio.run(cli.chat(args)) == 1
    assert len(state["calls"]) == 1 and state["events"][-1] == "close"


@pytest.mark.parametrize("exception", [EOFError, KeyboardInterrupt])
def test_input_termination_closes_server_without_a_reader_thread(monkeypatch, demo, exception):
    args, state = demo

    def finish(_):
        raise exception

    monkeypatch.setattr("builtins.input", finish)
    if exception is KeyboardInterrupt:
        with pytest.raises(KeyboardInterrupt):
            asyncio.run(cli.chat(args))
    else:
        assert asyncio.run(cli.chat(args)) == 0
    assert state["events"] == ["start", "close"]
