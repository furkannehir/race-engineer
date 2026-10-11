import importlib
import json
from pathlib import Path

import pytest

from race_engineer.evaluation.mdeberta_interpreter import (
    decode_nli,
    hypotheses,
    nli_indices,
)
from race_engineer.evaluation.radio_interpreter import PURPOSES, SCOPES, TOPICS


def make_rows():
    rows = [[0.1, 0.6, 0.3] for _ in range(17)]
    rows[0] = [0.9, 0.05, 0.05]
    rows[9] = [0.8, 0.1, 0.1]
    rows[11] = [0.5, 0.4, 0.1]
    return rows


@pytest.fixture
def playground(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return importlib.import_module("try_mdeberta_interpreter")


class FakeNli:
    def __init__(self):
        self.calls = []

    def predict(self, text, language):
        self.calls.append((text, language))
        return make_rows()


def test_catalog_is_localized_but_ids_and_order_match_laya():
    for language in ("en", "tr"):
        catalog = hypotheses(language)
        assert tuple(catalog["purpose"]) == PURPOSES
        assert tuple(catalog["time_scope"]) == SCOPES
        assert tuple(catalog["topics"]) == TOPICS
        assert all(value for group in catalog.values() for value in group.values())
    assert "yakıt" in hypotheses("tr")["topics"]["fuel"]
    assert hypotheses("en") != hypotheses("tr")
    with pytest.raises(ValueError):
        hypotheses("de")


def test_checkpoint_label_order_is_explicit_not_assumed():
    assert nli_indices({0: "CONTRADICTION", 1: "ENTAILMENT", 2: "NEUTRAL"}) == (1, 2, 0)
    with pytest.raises(ValueError):
        nli_indices({0: "LABEL_0", 1: "LABEL_1", 2: "LABEL_2"})
    with pytest.raises(ValueError):
        nli_indices({1: "entailment", 2: "neutral", 3: "contradiction"})


def test_scoring_retains_neutral_and_does_not_hide_inconsistent_topics():
    result = decode_nli(make_rows(), "en")
    # This is deliberately an inconsistent result: show it, do not quietly repair it.
    assert result["interpretation"] == {
        "purpose": "presence", "time_scope": "none", "topics": ["fuel"],
    }
    assert result["topic_scores_uncalibrated"]["fuel"] == 0.5
    assert result["nli_probabilities"]["topics"]["fuel"]["neutral"] == 0.4
    rows = make_rows()
    rows[11] = [0.4, 0.5, 0.1]
    assert decode_nli(rows, "en")["interpretation"]["topics"] == []


@pytest.mark.parametrize("bad", [[], [0.2], [float("nan"), 0, 1], [1.1, 0, 0], [0.2, 0.2, 0.2]])
def test_invalid_model_probabilities_are_rejected(bad):
    rows = make_rows()
    rows[0] = bad
    with pytest.raises(ValueError):
        decode_nli(rows, "en")
    with pytest.raises(ValueError):
        decode_nli(rows[:-1], "en")


def test_show_decision_preserves_input_and_raw_flag(playground):
    agent = FakeNli()
    output = []
    playground.show_decision(agent, "Beni duyuyor musun?", "tr", raw=True, write=output.append)
    decoded = json.loads(output[0])
    assert agent.calls == [("Beni duyuyor musun?", "tr")]
    assert "nli_probabilities" in decoded
    assert decoded["duration_ms"] >= 0 and decoded["history_entries"] == 0
    output.clear()
    playground.show_decision(agent, "Yakıt?", "tr", question_language_mode="english",
                             write=output.append)
    assert agent.calls[-1] == ("Yakıt?", "en")
    assert "nli_probabilities" not in json.loads(output[0])
    for text in (" ", "x" * 1001):
        with pytest.raises(ValueError):
            playground.show_decision(agent, text, "en")


def test_interactive_commands_and_independent_messages(playground):
    messages = iter([
        "", "/questions", "/unknown", "Are you there?", "/lang tr", "/raw on",
        "Yakıt?", "/question-language english", "Fark ne?", "/raw off", "/help", "/quit",
    ])
    output = []
    agent = FakeNli()
    playground.interactive(agent, "en", read=lambda _: next(messages), write=output.append)
    assert agent.calls == [("Are you there?", "en"), ("Yakıt?", "tr"), ("Fark ne?", "en")]
    results = [json.loads(line) for line in output if line.startswith('{\n  "interpretation"')]
    assert len(results) == 3
    assert "nli_probabilities" not in results[0] and "nli_probabilities" in results[1]
    assert all(result["history_entries"] == 0 for result in results)
    assert any("Unknown command" in line for line in output)


def test_failure_is_sanitized_and_eof_is_graceful(playground):
    class FailingNli:
        def predict(self, text, language):
            raise RuntimeError("private prompt content")

    messages = iter(["Fuel?"])

    def read(_):
        try:
            return next(messages)
        except StopIteration as error:
            raise EOFError from error

    output = []
    playground.interactive(FailingNli(), "en", read=read, write=output.append)
    assert any("Interpretation failed (RuntimeError)" in line for line in output)
    assert not any("private prompt content" in line for line in output)
    assert output[-1] == "Goodbye."
