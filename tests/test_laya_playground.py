import importlib
import json
from pathlib import Path

import pytest

from race_engineer.evaluation.radio_interpreter import TOPICS


@pytest.fixture
def playground(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return importlib.import_module("try_laya_interpreter")


class FakeAgent:
    def __init__(self):
        self.calls = []

    def predict(self, state, questions, *, lang):
        self.calls.append((state, questions, lang))
        return {"answers": {
            "purpose": {"choice": "presence", "probabilities": {"presence": 0.8}},
            "time_scope": {"choice": "none"},
            **{f"topic_{topic}": {"noul": 0.1} for topic in TOPICS},
        }, "usage": {"tokens": 42}}


def run_loop(playground, agent, lines, **kwargs):
    messages = iter(lines)
    output = []
    playground.interactive(
        agent, "en", read=lambda _: next(messages), write=output.append, **kwargs,
    )
    return output


def test_raw_output_contains_actual_response_and_unchanged_input(playground):
    agent = FakeAgent()
    output = []
    playground.show_decision(agent, "Are you there?", "en", (), raw=True, write=output.append)
    assert agent.calls[0][0] == "Are you there?"
    assert '"tokens": 42' in output[0] and '"probabilities"' in output[0]
    decoded = json.loads(output[1])
    assert decoded["interpretation"] == {"purpose": "presence", "time_scope": "none", "topics": []}
    assert decoded["duration_ms"] >= 0
    assert decoded["history_entries"] == 0


def test_messages_are_independent_by_default_and_language_can_change(playground):
    agent = FakeAgent()
    run_loop(playground, agent, ["Position.", "/lang tr", "Beni duyuyor musun?", "/quit"])
    assert agent.calls[1][0] == "Beni duyuyor musun?"
    assert agent.calls[1][2] == "tr"
    assert "yakıt" in agent.calls[1][1]["topic_fuel"]["instructions"]


def test_playground_can_restore_english_wording_without_changing_input_language(playground):
    agent = FakeAgent()
    output = run_loop(playground, agent, [
        "/lang tr", "Yakıt?", "/question-language english", "Yakıt?", "/quit",
    ])
    assert agent.calls[0][0] == agent.calls[1][0] == "Yakıt?"
    assert agent.calls[0][2] == agent.calls[1][2] == "tr"
    assert "yakıt" in agent.calls[0][1]["topic_fuel"]["instructions"]
    assert "fuel" in agent.calls[1][1]["topic_fuel"]["instructions"]
    results = [json.loads(line) for line in output if line.startswith("{")]
    assert [result["question_language"] for result in results] == ["tr", "en"]


def test_opt_in_history_and_authored_replies_are_bounded_and_can_be_reset(playground):
    agent = FakeAgent()
    run_loop(playground, agent, [
        "/history on", "Position.", "/engineer You are in the middle of the field.",
        "Are you there?", "/reset", "Fuel?", "/history off", "Hello.", "/quit",
    ])
    assert "driver: Position." in agent.calls[1][0]
    assert "engineer: You are in the middle of the field." in agent.calls[1][0]
    assert agent.calls[1][0].endswith("Current driver message:\nAre you there?")
    assert agent.calls[2][0] == "Fuel?" and agent.calls[3][0] == "Hello."
    agent = FakeAgent()
    run_loop(playground, agent, ["/history on", *[f"message {i}" for i in range(10)], "/quit"])
    assert "driver: message 2\n" not in agent.calls[-1][0]
    assert "driver: message 3\n" in agent.calls[-1][0]


def test_commands_and_empty_messages_do_not_call_model(playground):
    agent = FakeAgent()
    output = run_loop(playground, agent, [
        "", "/questions", "/help", "/unknown", "/engineer unsupported", "/raw on", "/quit",
    ])
    assert agent.calls == []
    assert any("Unknown command" in line for line in output)


def test_failed_prediction_does_not_poison_history(playground):
    class FailingAgent(FakeAgent):
        def predict(self, state, questions, *, lang):
            if "bad message" in state:
                raise RuntimeError("private test exception")
            return super().predict(state, questions, lang=lang)

    agent = FailingAgent()
    output = run_loop(playground, agent, ["/history on", "bad message", "Hello.", "/quit"])
    assert agent.calls[0][0] == "Hello."
    assert any("Interpretation failed (RuntimeError)" in line for line in output)
    assert not any("private test exception" in line for line in output)


def test_eof_exits_cleanly(playground):
    def read(_):
        raise EOFError

    output = []
    playground.interactive(FakeAgent(), "en", read=read, write=output.append)
    assert output[-1] == "Goodbye."
