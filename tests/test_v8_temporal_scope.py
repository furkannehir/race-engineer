import asyncio
import hashlib
import json
from types import SimpleNamespace

import pytest
from test_portable_engineer import FakeJsonModel, fuel_item, packet
from test_portable_engineer import turn as core_turn
from test_qwen_understanding import Model, draft, memory, turn

from race_engineer.config import ConversationConfig
from race_engineer.core.intelligence import (
    EngineerBrief,
    EvidenceReference,
    GeneratedResponse,
)
from race_engineer.intelligence.capabilities import DeterministicRaceCapabilities
from race_engineer.intelligence.context_engine import QueryDrivenContextEngineer
from race_engineer.intelligence.context_planner import (
    PLANNER_ID,
    PLANNER_V7_ID,
    PLANNER_V8_ID,
    QwenContextQueryPlanner,
    _v7_prompt,
    _v8_prompt,
)
from race_engineer.intelligence.grounding import GroundingError, StrictEvidenceGrounder
from race_engineer.intelligence.orchestrator import (
    EngineerOrchestrator,
    IntelligenceBoundaryError,
    PreparedEngineerResponse,
)
from race_engineer.intelligence.portable_engineer import PortableQwenEngineer


def analyze(raw, candidate=PLANNER_V8_ID):
    state = memory()
    model = Model(raw)
    planner = QwenContextQueryPlanner(ConversationConfig(), model=model, planner_id=candidate)
    engine = QueryDrivenContextEngineer(state, planner, DeterministicRaceCapabilities(state))
    return engine, model


def test_v7_prompt_is_frozen_and_remains_default():
    assert PLANNER_ID == ConversationConfig().context_planner == PLANNER_V7_ID
    assert hashlib.sha256(_v7_prompt().encode()).hexdigest() == (
        "99f0439f704b44d3882663e5b8ca0d0f397bbd5c67a2a30e3f32ad47fa57fb0c"
    )
    assert "Only PRESENT race position" in _v8_prompt()
    assert "BEFORE matching time scope" not in _v8_prompt()
    # Do not solve the benchmark by copying its failing utterance into the prompt.
    assert "Yarışa kaçıncı başladık?" not in _v8_prompt()


def test_v8_uses_same_wire_schema_without_keyword_routing_or_another_model_call():
    old, old_model = analyze(draft(), PLANNER_V7_ID)
    new, new_model = analyze(draft())
    old_packet = asyncio.run(old.analyze(turn()))
    new_packet = asyncio.run(new.analyze(turn()))
    assert old_model.calls[0]["schema"] == new_model.calls[0]["schema"]
    assert old_model.calls[0]["content"] == new_model.calls[0]["content"]
    assert len(new_model.calls) == 1
    assert old_packet.temporal_scope is None
    assert new_packet.temporal_scope == "current"
    assert all(item.temporal_scope == "current" for item in new_packet.evidence)


def test_temporal_validation_is_bilingual_and_not_copied_into_prompt():
    from pathlib import Path

    from race_engineer.evaluation.intelligence import load_intelligence_evaluation_dataset

    dataset = load_intelligence_evaluation_dataset(
        Path(__file__).resolve().parents[1]
        / "fixtures/intelligence-evaluation/development-temporal-validation.json"
    )
    assert dataset.language_counts() == {"en": 5, "tr": 5}
    assert all(
        question not in _v8_prompt() for group in dataset.groups for question in group.questions
    )
    assert any(group.recent_dialogue for group in dataset.groups)


def test_temporal_provenance_survives_refresh_and_keeps_current_values_current():
    engine, _ = analyze(draft())

    async def run():
        current = await engine.analyze(turn())
        refreshed = await engine.refresh(current, [item.evidence_id for item in current.evidence])
        assert refreshed.temporal_scope == current.temporal_scope == "current"
        assert all(item.temporal_scope == "current" for item in refreshed.evidence)

    asyncio.run(run())


def test_historical_queries_are_labeled_from_operation_not_observation_time():
    engine, _ = analyze(
        draft(
            requested_facts=["past average speed"],
            temporal_scope="historical",
            capability_ids=[],
            queries=[{"signal_id": "player.speed_mps", "operation": "mean", "window_s": 10}],
        )
    )
    current = asyncio.run(engine.analyze(turn("What was the average speed?")))
    assert current.temporal_scope == "historical"
    assert current.evidence and all(
        item.temporal_scope == "historical" for item in current.evidence
    )


def test_unsupported_original_snapshot_does_not_receive_current_evidence():
    engine, _ = analyze(
        draft(
            requested_facts=["original lineup"],
            temporal_scope="historical",
            capability_ids=[],
            missing_information="unsupported_analysis",
        )
    )
    current = asyncio.run(engine.analyze(turn("An original-lineup question")))
    assert current.temporal_scope == "historical"
    assert current.evidence == ()
    assert current.unknowns == ("requested_analysis_unavailable",)


def response_for(context):
    brief = EngineerBrief(
        turn_id=context.turn_id,
        session_id=context.session_id,
        generation=context.generation,
        source_sequence=context.source_sequence,
        goal="inform",
        language="en",
        tone="calm",
        evidence_ids=("fuel",),
    )
    response = GeneratedResponse(
        response_id="temporal",
        turn_id=context.turn_id,
        session_id=context.session_id,
        generation=context.generation,
        language="en",
        action="speak",
        speech_template="Fuel: {{a}} litres.",
        references=(EvidenceReference(placeholder="a", evidence_id="fuel", field="value"),),
    )
    return brief, response


@pytest.mark.parametrize(
    "evidence_scope,allowed", [("current", False), (None, False), ("historical", True)]
)
def test_grounding_rejects_missing_or_incompatible_timing(evidence_scope, allowed):
    evidence = fuel_item().model_copy(update={"temporal_scope": evidence_scope})
    context = packet(evidence).model_copy(update={"temporal_scope": "historical"})
    brief, response = response_for(context)
    if allowed:
        assert StrictEvidenceGrounder().ground(response, context, brief).text == "Fuel: 30 litres."
    else:
        with pytest.raises(GroundingError, match="temporal_scope"):
            StrictEvidenceGrounder().ground(response, context, brief)


@pytest.mark.parametrize("candidate_scope", [None, "current"])
def test_only_opt_in_core_payload_changes_and_evidence_scope_reaches_bindings(candidate_scope):
    evidence = fuel_item().model_copy(update={"temporal_scope": candidate_scope})
    context = packet(evidence).model_copy(update={"temporal_scope": candidate_scope})
    model = FakeJsonModel({"goal": "inform", "speech": "Fuel: {{a}} litres."})
    core = PortableQwenEngineer(ConversationConfig(), model=model)
    asyncio.run(core.decide(core_turn("How much fuel now?"), context))
    payload = json.loads(model.calls[0]["content"])
    if candidate_scope is None:
        assert "requested_temporal_scope" not in payload
        assert "temporal_scope" not in payload["evidence"][0]
        assert "temporal_scope" not in payload["rendering_bindings"][0]
        assert "Temporal" not in model.calls[0]["system_prompt"]
    else:
        assert payload["requested_temporal_scope"] == "current"
        assert payload["evidence"][0]["temporal_scope"] == "current"
        assert payload["rendering_bindings"][0]["temporal_scope"] == "current"
        assert "when a fact belongs" in model.calls[0]["system_prompt"]


def test_refresh_cannot_drop_requested_time_before_grounding():
    context = packet(fuel_item().model_copy(update={"temporal_scope": "current"})).model_copy(
        update={"temporal_scope": "current"}
    )
    brief, response = response_for(context)
    prepared = PreparedEngineerResponse(core_turn(), context, brief, response)

    async def refresh(*args):
        return context.model_copy(update={"temporal_scope": None})

    orchestrator = EngineerOrchestrator(
        SimpleNamespace(refresh=refresh),
        None,
        None,
        StrictEvidenceGrounder(),
    )
    with pytest.raises(IntelligenceBoundaryError, match="temporal_scope_changed"):
        asyncio.run(orchestrator.ground(prepared))
