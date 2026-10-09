import json
from pathlib import Path

import pytest

from race_engineer.evaluation.radio_interpreter import (
    TOPICS,
    decision_questions,
    decode_decisions,
    score_case,
    state_for,
    summarize,
    text_state_for,
    validate_cases,
)

ROOT = Path(__file__).resolve().parents[1]


def dataset():
    return json.loads((ROOT / "fixtures/radio-interpreter/development.json").read_text("utf-8"))


def prediction(purpose="presence", scope="none", topics=()):
    return {"answers": {
        "purpose": {"choice": purpose}, "time_scope": {"choice": scope},
        **{f"topic_{topic}": {"noul": 0.9 if topic in topics else 0.1} for topic in TOPICS},
    }}


def test_dataset_is_bilingual_unique_and_development_only():
    cases = validate_cases(dataset())
    assert len(cases) == 28
    assert sum(case["language"] == "en" for case in cases) == 14
    assert sum(case["language"] == "tr" for case in cases) == 14
    invalid = dataset()
    invalid["split"] = "locked"
    with pytest.raises(ValueError, match="only_development"):
        validate_cases(invalid)
    invalid = dataset()
    invalid["cases"][1]["id"] = invalid["cases"][0]["id"]
    with pytest.raises(ValueError, match="duplicate"):
        validate_cases(invalid)


def test_questions_allow_multiple_topics_without_generation_or_phrase_matching():
    questions = decision_questions()
    assert set(questions) == {"purpose", "time_scope", *(f"topic_{topic}" for topic in TOPICS)}
    observed, _ = decode_decisions(prediction("mixed", "current", ("fuel", "classification")))
    assert observed == {
        "purpose": "mixed", "time_scope": "current", "topics": ["classification", "fuel"],
    }


def test_presence_after_position_is_scored_against_current_message():
    case = next(case for case in validate_cases(dataset())
                if case["id"] == "presence-after-position-en")
    state = state_for(case)
    assert state["current_driver_message"] == "Are you there?"
    assert len(state["recent_dialogue"]) == 2
    correct = score_case(case, prediction())
    wrong = score_case(case, prediction("race_information", "current", ("classification",)))
    assert correct["passed"] is True
    assert wrong["passed"] is False
    serialized = json.dumps(correct)
    assert "Are you there?" not in serialized and "middle of the field" not in serialized
    assert len(correct["state_sha256"]) == 64


def test_text_comparison_preserves_dialogue_and_marks_current_message():
    case = next(case for case in validate_cases(dataset())
                if case["id"] == "presence-after-position-en")
    text = text_state_for(case)
    assert "driver: What is my position?" in text
    assert text.endswith("Current driver message:\nAre you there?")
    assert text_state_for(validate_cases(dataset())[0]) == "Hey, are you there?"


@pytest.mark.parametrize("score", [float("nan"), float("inf"), -0.1, 1.1, True, "0.9", None])
def test_malformed_or_nonfinite_scores_are_rejected(score):
    raw = prediction()
    raw["answers"]["topic_fuel"]["noul"] = score
    with pytest.raises(ValueError, match="invalid_laya_score"):
        decode_decisions(raw)


def test_summary_keeps_errors_in_denominator_and_separates_languages():
    case = validate_cases(dataset())[0]
    results = [score_case(case, prediction()), {
        "case_id": "failed", "language": "tr", "error": "RuntimeError", "passed": False,
    }]
    summary = summarize(results)
    assert summary["attempted"] == 2 and summary["errors"] == 1
    assert summary["exact_accuracy"] == 0.5
    assert summary["by_language"]["en"]["exact_accuracy"] == 1
    assert summary["by_language"]["tr"]["exact_accuracy"] == 0
