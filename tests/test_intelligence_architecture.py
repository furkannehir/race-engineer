import asyncio
from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from race_engineer.core.intelligence import (
    ContextPacket,
    DriverTurn,
    EngineerBrief,
    EvidenceItem,
    EvidenceReference,
    GeneratedResponse,
)
from race_engineer.intelligence.grounding import GroundingError, StrictEvidenceGrounder
from race_engineer.intelligence.orchestrator import (
    EngineerOrchestrator,
    IntelligenceBoundaryError,
)

NOW = datetime(2026, 9, 28, 12, tzinfo=UTC)


def turn(text: str) -> DriverTurn:
    return DriverTurn(
        turn_id="turn-1",
        transcript=text,
        received_at=NOW,
        session_id="iracing:1",
        generation=2,
        asr_language="en",
    )


def evidence(
    evidence_id: str,
    metric: str,
    value: int | float,
    *,
    sequence: int = 10,
    kind: str = "measurement",
    unit: str | None = None,
) -> EvidenceItem:
    return EvidenceItem.model_validate(
        {
            "evidence_id": evidence_id,
            "session_id": "iracing:1",
            "source_sequence": sequence,
            "observed_at": NOW,
            "kind": kind,
            "subject": "player",
            "metric": metric,
            "value": value,
            "unit": unit,
            "source_fields": [metric],
            "valid_until": NOW + timedelta(seconds=3),
        }
    )


def packet(*items: EvidenceItem, sequence: int = 10) -> ContextPacket:
    return ContextPacket(
        turn_id="turn-1",
        session_id="iracing:1",
        generation=2,
        source_sequence=sequence,
        assembled_at=NOW + timedelta(milliseconds=100),
        evidence=items,
    )


class ScriptedContextEngineer:
    def __init__(self, initial: ContextPacket, refreshed: ContextPacket | None = None) -> None:
        self.initial = initial
        self.refreshed = refreshed or initial
        self.refresh_requests: list[tuple[str, ...]] = []

    async def analyze(self, request: DriverTurn) -> ContextPacket:
        del request
        return self.initial

    async def refresh(
        self,
        context: ContextPacket,
        evidence_ids: tuple[str, ...],
    ) -> ContextPacket:
        del context
        self.refresh_requests.append(evidence_ids)
        return self.refreshed


class ScriptedCoreEngineer:
    def __init__(self, result: EngineerBrief) -> None:
        self.result = result

    async def decide(self, request: DriverTurn, context: ContextPacket) -> EngineerBrief:
        del request, context
        return self.result


class ScriptedGenerator:
    def __init__(self, result: GeneratedResponse) -> None:
        self.result = result

    async def generate(
        self,
        request: DriverTurn,
        context: ContextPacket,
        brief: EngineerBrief,
    ) -> GeneratedResponse:
        del request, context, brief
        return self.result


def brief(goal: str, *evidence_ids: str) -> EngineerBrief:
    return EngineerBrief.model_validate(
        {
            "turn_id": "turn-1",
            "session_id": "iracing:1",
            "generation": 2,
            "source_sequence": 10,
            "goal": goal,
            "language": "en",
            "tone": "calm_teammate",
            "evidence_ids": evidence_ids,
            "guidance": ["answer naturally"] if goal != "silence" else [],
        }
    )


def response(
    speech: str,
    *references: EvidenceReference,
    action: str = "speak",
) -> GeneratedResponse:
    return GeneratedResponse.model_validate(
        {
            "response_id": "turn-1:reply",
            "turn_id": "turn-1",
            "session_id": "iracing:1",
            "generation": 2,
            "language": "en",
            "action": action,
            "speech_template": speech,
            "references": references,
        }
    )


def run_pipeline(
    driver_turn: DriverTurn,
    context: ScriptedContextEngineer,
    core_result: EngineerBrief,
    generated: GeneratedResponse,
):
    engine = EngineerOrchestrator(
        context,
        ScriptedCoreEngineer(core_result),
        ScriptedGenerator(generated),
        StrictEvidenceGrounder(),
    )
    return asyncio.run(engine.respond(driver_turn))


def test_direct_fact_uses_refreshed_evidence_not_the_initial_value():
    initial = packet(evidence("position", "position", 6, unit="position"))
    latest = packet(evidence("position", "position", 5, sequence=11, unit="position"), sequence=11)
    context = ScriptedContextEngineer(initial, latest)
    generated = response(
        "We're running P{{position}}.",
        EvidenceReference(placeholder="position", evidence_id="position", field="value"),
    )

    result = run_pipeline(turn("Where are we?"), context, brief("inform", "position"), generated)

    assert result.text == "We're running P5."
    assert result.source_sequence == 11
    assert result.evidence_ids == ("position",)
    assert context.refresh_requests == [("position",)]


def test_prepared_response_waits_until_delivery_to_refresh_evidence():
    initial = packet(evidence("position", "position", 6, unit="position"))
    latest = packet(evidence("position", "position", 5, sequence=11, unit="position"), sequence=11)
    context = ScriptedContextEngineer(initial, latest)
    engine = EngineerOrchestrator(
        context,
        ScriptedCoreEngineer(brief("inform", "position")),
        ScriptedGenerator(
            response(
                "P{{position}}.",
                EvidenceReference(
                    placeholder="position",
                    evidence_id="position",
                    field="value",
                ),
            )
        ),
        StrictEvidenceGrounder(),
    )

    prepared = asyncio.run(engine.prepare(turn("Position?")))
    assert context.refresh_requests == []
    result = asyncio.run(engine.ground(prepared))

    assert result.text == "P5."
    assert context.refresh_requests == [("position",)]


def test_analytical_question_uses_derived_time_series_evidence():
    loss = evidence("sector-loss", "sector_two_delta", 0.4, kind="derived", unit="s")
    context = ScriptedContextEngineer(packet(loss))
    generated = response(
        "Most of it is sector two. You're losing {{loss}} seconds under braking.",
        EvidenceReference(placeholder="loss", evidence_id="sector-loss", field="value"),
    )

    result = run_pipeline(
        turn("Why am I losing time?"),
        context,
        brief("analyze", "sector-loss"),
        generated,
    )

    assert result.text == "Most of it is sector two. You're losing 0.4 seconds under braking."


def test_human_interaction_does_not_require_a_telemetry_query():
    context = ScriptedContextEngineer(packet())
    generated = response("Yeah, don't get dragged into it. Keep it clean and focus on your exits.")

    result = run_pipeline(
        turn("He has no idea about racing."),
        context,
        brief("acknowledge"),
        generated,
    )

    assert result.text == (
        "Yeah, don't get dragged into it. Keep it clean and focus on your exits."
    )
    assert result.evidence_ids == ()
    assert context.refresh_requests == [()]


def test_generator_cannot_reference_evidence_the_core_did_not_approve():
    context = ScriptedContextEngineer(packet(evidence("position", "position", 7, unit="position")))
    generated = response(
        "P{{position}}.",
        EvidenceReference(placeholder="position", evidence_id="position", field="value"),
    )

    with pytest.raises(IntelligenceBoundaryError, match="unapproved"):
        run_pipeline(turn("Talk to me."), context, brief("acknowledge"), generated)


def test_generated_numeric_claims_require_evidence_placeholders():
    with pytest.raises(ValidationError, match="numeric telemetry"):
        response("We're running P7.")
    with pytest.raises(ValidationError, match="exactly one reference"):
        response("We're running P{{position}}.")


def test_grounder_rejects_expired_or_unapproved_refreshed_evidence():
    expired = evidence("position", "position", 7, unit="position").model_copy(
        update={"valid_until": NOW + timedelta(milliseconds=50)}
    )
    context = ScriptedContextEngineer(
        packet(evidence("position", "position", 6, unit="position")),
        packet(expired),
    )
    generated = response(
        "P{{position}}.",
        EvidenceReference(placeholder="position", evidence_id="position", field="value"),
    )
    with pytest.raises(GroundingError, match="stale"):
        run_pipeline(turn("Position?"), context, brief("inform", "position"), generated)
