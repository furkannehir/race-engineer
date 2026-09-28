from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from race_engineer.conversation import composer
from race_engineer.conversation.composer import (
    CompositionError,
    compose,
    compose_bounded,
    plan_utterance,
    validate_utterance,
)
from race_engineer.core.dialogue import (
    GroundedAnswer,
    ResponseDecision,
    UtteranceClause,
    UtterancePlan,
)


def decision(**updates) -> ResponseDecision:
    now = datetime.now(UTC)
    value = {
        "turn_id": "turn-r5",
        "response_id": "turn-r5:reply",
        "session_id": "live-r5",
        "generation": 0,
        "source_sequence": 10,
        "language": "en",
        "created_at": now,
        "expires_at": now + timedelta(seconds=30),
        "outcome": "answered",
        "reason": "resolved",
    }
    value.update(updates)
    return ResponseDecision.model_validate(value)


def fact(
    part_id: str = "position",
    *,
    query: str = "position",
    value: int | float = 7,
    unit: str = "position",
) -> GroundedAnswer:
    return GroundedAnswer.model_validate(
        {
            "part_id": part_id,
            "query": query,
            "status": "available",
            "value": value,
            "unit": unit,
            "source_sequence": 10,
        }
    )


def test_typed_plan_covers_each_grounded_fact_once_and_in_order():
    response = decision(
        acts=["acknowledge"],
        answers=[
            fact(),
            fact("gap", query="gap_behind", value=1.2, unit="s"),
        ],
        clarification="opponent",
        outcome="partial",
    )
    plan = plan_utterance(response)
    assert plan is not None
    assert plan.schema_version == "utterance-plan.v1"
    assert plan.tone == "neutral"
    assert tuple(clause.kind for clause in plan.clauses) == (
        "acknowledgment",
        "fact",
        "fact",
        "clarification",
    )
    assert tuple(clause.source_part_id for clause in plan.clauses if clause.kind == "fact") == (
        "position",
        "gap",
    )


def test_plan_contract_rejects_duplicate_or_unscoped_fact_clauses():
    with pytest.raises(ValidationError, match="spoken twice"):
        UtterancePlan(
            response_id="reply",
            language="en",
            tone="neutral",
            clauses=(
                UtteranceClause(clause_id="one", kind="fact", source_part_id="q1"),
                UtteranceClause(clause_id="two", kind="fact", source_part_id="q1"),
            ),
        )
    with pytest.raises(ValidationError, match="only fact clauses"):
        UtteranceClause(
            clause_id="bad",
            kind="acknowledgment",
            source_part_id="q1",
        )


def test_validator_rejects_missing_grounded_fact_and_scope_mismatch():
    response = decision(answers=[fact()])
    plan = plan_utterance(response)
    assert plan is not None
    missing = plan.model_copy(
        update={"clauses": (UtteranceClause(clause_id="failure", kind="failure"),)}
    )
    with pytest.raises(CompositionError, match="fact_coverage"):
        validate_utterance(missing, response, "No data.")
    wrong_scope = plan.model_copy(update={"response_id": "another:reply"})
    with pytest.raises(CompositionError, match="scope"):
        validate_utterance(wrong_scope, response, "P7 right now.")


def test_natural_composition_is_deterministic_and_keeps_exact_values():
    response = decision(
        acts=["acknowledge"],
        answers=[fact("gap", query="gap_behind", value=1.2, unit="s")],
    )
    first = compose(response)
    assert first == compose(response)
    assert first is not None
    assert "1.2" in first and ("Car behind" in first or "Gap behind" in first)
    assert "saw" not in first.lower() and "contact" not in first.lower()


def test_natural_composition_covers_turkish_and_relationship_truth():
    leading = GroundedAnswer(
        part_id="field",
        query="field_status",
        status="available",
        value=1,
        total=28,
        unit="position",
        field_relation="first",
        source_sequence=10,
    )
    not_leading = leading.model_copy(update={"value": 2})
    assert compose(decision(language="tr", answers=[leading])) == "Evet, P1. Lideriz."
    assert compose(decision(language="tr", answers=[not_leading])) == (
        "Hayır, P2. Önünde 1 araç var."
    )


def test_renderer_failure_or_radio_limit_uses_bounded_fallback(monkeypatch):
    response = decision(answers=[fact()])
    expected = compose_bounded(response)

    def too_long(plan, value):
        del plan, value
        return "word " * 100

    monkeypatch.setattr(composer, "render_utterance", too_long)
    assert compose(response) == expected


def test_no_reply_remains_silent_and_repeated_acknowledgment_is_brief():
    assert compose(decision(outcome="no_reply", acts=["close"])) is None
    response = decision(outcome="acknowledge", acts=["acknowledge"])
    assert compose(response, repeated_acknowledgment=True) == "Copy."
