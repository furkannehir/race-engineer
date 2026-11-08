import asyncio
import json
from pathlib import Path

import pytest
from pydantic import ValidationError
from test_context_engine import ScriptedPlanner, driver_turn, evidence_query, plan, race_context
from test_portable_engineer import FakeJsonModel, fuel_item, packet, turn

from race_engineer.config import ConversationConfig
from race_engineer.core.intelligence import EvidenceItem, GeneratedResponse, LimitationReference
from race_engineer.evaluation.intelligence import load_intelligence_evaluation_dataset
from race_engineer.intelligence.context_engine import QueryDrivenContextEngineer
from race_engineer.intelligence.context_planner import LFM_NATIVE_ID, PLANNER_V8_ID
from race_engineer.intelligence.grounding import GroundingError, StrictEvidenceGrounder
from race_engineer.intelligence.local_model import LocalIntelligenceError
from race_engineer.intelligence.orchestrator import EngineerOrchestrator
from race_engineer.intelligence.portable_engineer import PortableQwenEngineer
from race_engineer.intelligence.telemetry_memory import BoundedTelemetryMemory


def guarded(model):
    return PortableQwenEngineer(ConversationConfig(context_planner=PLANNER_V8_ID), model=model)


async def response(engineer, request, context):
    brief = await engineer.decide(request, context)
    generated = await engineer.generate(request, context, brief)
    return brief, generated, StrictEvidenceGrounder().ground(generated, context, brief)


@pytest.mark.parametrize(
    ("language", "scope", "expected"),
    [
        ("en", "current", "I can't assess that part from the available data."),
        ("tr", "current", "Mevcut verilerle bu kısmı değerlendiremiyorum."),
        ("en", "historical", "I don't have the historical data needed for that."),
        ("tr", "historical", "Bunun için gereken geçmiş veriler elimde yok."),
        ("en", "future_counterfactual", "I can't support that prediction with the available data."),
        ("tr", "future_counterfactual", "Mevcut verilerle bu tahmini yapamıyorum."),
    ],
)
def test_unknown_only_uses_application_statement_without_core_inference(language, scope, expected):
    model = FakeJsonModel(AssertionError("Core must not be called"))
    context = packet(unknowns=("requested_analysis_unavailable",)).model_copy(
        update={"temporal_scope": scope}
    )
    brief, generated, grounded = asyncio.run(
        response(guarded(model), turn("An unsupported question", language), context)
    )
    assert grounded.text == expected
    assert brief.unknown_ids == generated.limitation_references[0].unknown_ids == context.unknowns
    assert grounded.evidence_ids == ()
    assert model.calls == []


@pytest.mark.parametrize("language", ["en", "tr"])
def test_ambiguous_reference_asks_for_clarification(language):
    context = packet(unknowns=("requested_reference_ambiguous",))
    brief, generated, grounded = asyncio.run(
        response(guarded(FakeJsonModel(None)), turn(language=language), context)
    )
    assert brief.goal == generated.action == grounded.action == "clarify"
    assert grounded.text and grounded.text.endswith("?")


@pytest.mark.parametrize(
    ("language", "speech", "expected"),
    [
        ("en", "No damage has been reported. Fuel is {{a}} litres.", "Fuel is 0 litres."),
        ("tr", "Hasar bildirilmedi. Yakıt {{a}} litre.", "Yakıt 0 litre."),
        ("en", "Fuel is {{a}} litres; the tyres are healthy.", "Fuel is 0 litres"),
    ],
)
def test_partial_preserves_known_zero_and_removes_unbound_claims(
    language, speech, expected, caplog
):
    model = FakeJsonModel({"goal": "inform", "speech": speech})
    context = packet(fuel_item(value=0), unknowns=("requested_analysis_unavailable",))
    with caplog.at_level("INFO"):
        brief, generated, grounded = asyncio.run(
            response(guarded(model), turn(language=language), context)
        )
    assert grounded.text and grounded.text.endswith(expected)
    assert "damage" not in grounded.text and "Hasar" not in grounded.text
    assert "healthy" not in grounded.text
    assert brief.evidence_ids == ("fuel",)
    assert generated.limitation_references[0].unknown_ids == context.unknowns
    assert len(model.calls) == 1
    assert any(
        getattr(entry, "event", None) == "core_unsupported_clauses_removed"
        for entry in caplog.records
    )
    payload = json.loads(model.calls[0]["content"])
    assert payload["unsupported_data_policy"] == "grounded-limitations.v1"


def test_partial_missing_supported_answer_has_one_bounded_repair():
    model = FakeJsonModel(
        [
            {"goal": "inform", "speech": "No damage reported."},
            {"goal": "inform", "speech": "Fuel is {{a}} litres."},
        ]
    )
    context = packet(fuel_item(), unknowns=("requested_analysis_unavailable",))
    _, _, grounded = asyncio.run(response(guarded(model), turn(), context))
    assert grounded.text and grounded.text.endswith("Fuel is 30 litres.")
    assert len(model.calls) == 2


@pytest.mark.parametrize(
    "draft",
    [
        {"goal": "inform", "speech": "No damage reported."},
        {"goal": "silence", "speech": None},
    ],
)
def test_partial_cannot_drop_the_supported_answer_after_repair(draft):
    model = FakeJsonModel(draft)
    context = packet(fuel_item(), unknowns=("requested_analysis_unavailable",))
    with pytest.raises(LocalIntelligenceError, match="engineer_model_response_invalid"):
        asyncio.run(guarded(model).decide(turn(), context))
    assert len(model.calls) == 2


def test_v7_unknown_path_keeps_legacy_model_and_wire_schema():
    model = FakeJsonModel({"goal": "inform", "speech": "That information isn't available."})
    context = packet(unknowns=("requested_analysis_unavailable",))
    _, generated, _ = asyncio.run(
        response(PortableQwenEngineer(ConversationConfig(), model=model), turn(), context)
    )
    assert len(model.calls) == 1
    assert generated.limitation_references == ()
    assert "unsupported_data_policy" not in json.loads(model.calls[0]["content"])
    assert set(model.calls[0]["schema"]["oneOf"][1]["properties"]) == {"goal", "speech"}


def test_unknown_evidence_is_not_an_empty_or_zero_measurement():
    unknown = EvidenceItem.model_validate(
        fuel_item().model_dump()
        | {
            "kind": "unknown",
            "value": None,
            "unit": None,
        }
    )
    model = FakeJsonModel(None)
    context = packet(unknown, unknowns=(unknown.evidence_id,))
    _, generated, grounded = asyncio.run(response(guarded(model), turn(), context))
    assert generated.references == () and grounded.evidence_ids == ()
    assert model.calls == []


def test_limitation_binding_cannot_be_missing_forged_or_duplicated():
    context = packet(unknowns=("requested_analysis_unavailable",))
    brief, generated, _ = asyncio.run(response(guarded(FakeJsonModel(None)), turn(), context))
    with pytest.raises(ValidationError, match="every speech placeholder"):
        GeneratedResponse.model_validate(generated.model_dump() | {"limitation_references": ()})
    with pytest.raises(ValidationError, match="unique"):
        LimitationReference(placeholder="unavailable", unknown_ids=("x", "x"))
    forged = generated.model_copy(
        update={
            "limitation_references": (
                LimitationReference(placeholder="unavailable", unknown_ids=("forged",)),
            )
        }
    )
    with pytest.raises(GroundingError, match="response_did_not_acknowledge_brief_unknowns"):
        StrictEvidenceGrounder().ground(forged, context, brief)
    with pytest.raises(GroundingError, match="limitation_no_longer_supported"):
        StrictEvidenceGrounder().ground(
            generated, context.model_copy(update={"unknowns": ()}), brief
        )


@pytest.mark.parametrize("candidate", [PLANNER_V8_ID, LFM_NATIVE_ID])
def test_missing_query_is_refreshed_and_newly_available_data_blocks_stale_limitation(candidate):
    memory = BoundedTelemetryMemory()
    memory.update(race_context(1, 0, position=None))
    query_plan = plan(evidence_query("position", "position")).model_copy(
        update={
            "planner_id": candidate,
            "temporal_scope": "current",
        }
    )
    context = QueryDrivenContextEngineer(memory, ScriptedPlanner(query_plan))
    core = PortableQwenEngineer(
        ConversationConfig(context_planner=candidate), model=FakeJsonModel({"framing": "explain"})
    )
    engineer = EngineerOrchestrator(context, core, core, StrictEvidenceGrounder())

    async def run():
        prepared = await engineer.prepare(driver_turn())
        assert prepared.context.unknowns == ("position",)
        memory.update(race_context(2, 0.1, position=5))
        # The unknown query ID must be refreshed even though no numeric value was spoken.
        with pytest.raises(GroundingError, match="limitation_no_longer_supported"):
            await engineer.ground(prepared)

    asyncio.run(run())


def test_unsupported_validation_dataset_is_balanced_and_development_only():
    dataset = load_intelligence_evaluation_dataset(
        Path(__file__).resolve().parents[1]
        / "fixtures/intelligence-evaluation/development-unsupported-validation.json"
    )
    assert dataset.split == "development"
    assert dataset.language_counts() == {"en": 8, "tr": 8}


def test_v8_radio_check_does_not_reuse_old_factual_replies_as_evidence():
    model = FakeJsonModel({"goal": "acknowledge", "speech": "Copy, I'm here."})
    context = packet().model_copy(
        update={
            "temporal_scope": "social",
            "situation": ("driver_social_turn", "driver_radio_check"),
        }
    )
    request = turn("Are you there?").model_copy(
        update={
            "recent_dialogue": ("driver: Position?", "engineer: You're P1 of 1."),
        }
    )
    _, _, grounded = asyncio.run(response(guarded(model), request, context))
    assert grounded.text == "Copy, I'm here."
    assert json.loads(model.calls[0]["content"])["recent_dialogue"] == []


def test_internal_redacted_numbers_cannot_escape_into_spoken_text():
    model = FakeJsonModel(
        [
            {"goal": "acknowledge", "speech": "You're P[unverified_number]."},
            {"goal": "acknowledge", "speech": "Copy, I'm here."},
        ]
    )
    _, _, grounded = asyncio.run(response(guarded(model), turn("Hello"), packet()))
    assert grounded.text == "Copy, I'm here." and len(model.calls) == 2
