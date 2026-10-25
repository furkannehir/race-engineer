import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from race_engineer.cli import _parser
from race_engineer.config import ConversationConfig, PolicyContextConfig, load_config
from race_engineer.core.contracts import PlayerState, RaceContext, TelemetryFrame
from race_engineer.core.intelligence import ContextPlan, DriverTurn
from race_engineer.evaluation.intelligence import (
    ExpectedContextPlan,
    IntelligenceCandidateMetadata,
    load_intelligence_evaluation_dataset,
)
from race_engineer.evaluation.intelligence_runner import evaluate_intelligence_datasets
from race_engineer.intelligence.capabilities import DeterministicRaceCapabilities
from race_engineer.intelligence.context_engine import QueryDrivenContextEngineer
from race_engineer.intelligence.context_planner import (
    PLANNER_ID,
    PLANNER_V6_ID,
    PLANNER_V7_ID,
    QwenContextQueryPlanner,
)
from race_engineer.intelligence.planner_diagnostics import ContextPlanRejection
from race_engineer.intelligence.telemetry_memory import BoundedTelemetryMemory

ROOT = Path(__file__).parents[1]
DATASET = ROOT / "fixtures/intelligence-evaluation/development-understanding.json"
NOW = datetime(2026, 10, 12, tzinfo=UTC)


class Model:
    def __init__(self, raw):
        self.raw, self.calls = raw, []

    async def request(self, **kwargs):
        self.calls.append(kwargs)
        return self.raw


def draft(**updates):
    return {
        "social_kind": "none",
        "requested_facts": ["current classification"],
        "temporal_scope": "current",
        "capability_ids": ["current_classification"],
        "queries": [],
        "missing_information": "none",
        **updates,
    }


def memory(position=5):
    result = BoundedTelemetryMemory()
    result.update(
        RaceContext(
            frame=TelemetryFrame(
                source="test",
                session_id="understanding",
                sequence=1,
                observed_at=NOW,
                session_time_s=0,
                player=PlayerState(driver_id="player", position=position, speed_mps=30),
            )
        )
    )
    return result


def turn(text="Position."):
    return DriverTurn(
        turn_id="turn",
        transcript=text,
        received_at=NOW,
        session_id="understanding",
        generation=0,
        asr_language="en",
    )


def execute(raw, text="Position.", *, position=5, candidate=PLANNER_V7_ID):
    model = Model(raw)
    state = memory(position)
    planner = QwenContextQueryPlanner(ConversationConfig(), model=model, planner_id=candidate)
    engine = QueryDrivenContextEngineer(state, planner, DeterministicRaceCapabilities(state))
    return asyncio.run(engine.analyze(turn(text))), model


@pytest.mark.parametrize("kind", ["greeting", "radio_check", "vent", "acknowledgment", "other"])
def test_v7_social_meaning_reaches_core_without_any_telemetry(kind):
    packet, model = execute(
        draft(
            social_kind=kind,
            requested_facts=[],
            temporal_scope=None,
            capability_ids=[],
        ),
        "An authored social utterance",
    )
    assert packet.situation == ("driver_social_turn", f"driver_{kind}")
    assert packet.evidence == packet.unknowns == ()
    assert len(model.calls) == 1


def test_v7_mixed_radio_check_keeps_classification_evidence():
    packet, _ = execute(draft(social_kind="radio_check"))
    assert "driver_radio_check" in packet.situation
    assert "race_information_request" in packet.situation
    assert any(item.metric == "position" for item in packet.evidence)


def test_v7_known_unavailable_capability_is_executed_not_guessed_or_dropped():
    packet, model = execute(draft(), position=None)
    assert any("missing_player_position" in reason for reason in packet.unknowns)
    advertised = json.loads(model.calls[0]["content"])["capabilities"]
    classification = next(item for item in advertised if item["id"] == "current_classification")
    assert classification["available"] is False
    assert "player's current classified position" in classification["outputs"]


def test_v7_unsupported_measurement_is_factual_and_has_a_stable_unknown():
    packet, _ = execute(
        draft(
            requested_facts=["current car damage"],
            capability_ids=[],
            missing_information="unsupported_analysis",
        ),
        "Do I have any damage?",
    )
    assert packet.evidence == ()
    assert packet.situation == ("race_information_request",)
    assert packet.unknowns == ("requested_analysis_unavailable",)


def test_v7_partially_supported_turn_keeps_supported_evidence_and_unknown():
    packet, _ = execute(
        draft(
            requested_facts=["damage", "physical speed"],
            capability_ids=[],
            queries=[{"signal_id": "player.speed_mps", "operation": "latest", "window_s": None}],
            missing_information="unsupported_analysis",
        )
    )
    assert packet.evidence[0].metric == "speed_mps"
    assert packet.unknowns == ("requested_analysis_unavailable",)


def test_v7_empty_factual_plan_is_still_rejected_if_a_model_ignores_the_schema():
    with pytest.raises(ContextPlanRejection, match="planner_empty_selection"):
        execute(draft(capability_ids=[]))


@pytest.mark.parametrize("kind,comment", [("none", True), ("greeting", False)])
def test_v7_redundant_legacy_social_field_is_rejected(kind, comment):
    with pytest.raises(ContextPlanRejection, match="context_model_response_invalid"):
        execute(draft(social_kind=kind, social_comment=comment))


def test_v7_schema_starts_with_meaning_and_bounds_the_dynamic_catalog():
    _, model = execute(draft())
    schema = model.calls[0]["schema"]
    properties = schema["properties"]
    assert next(iter(properties)) == "social_kind"
    assert "social_comment" not in properties
    assert "$ref" not in json.dumps(schema)
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == set(properties)
    assert "none" in properties["social_kind"]["enum"]
    assert "current_classification" in properties["capability_ids"]["items"]["enum"]
    assert properties["queries"]["maxItems"] == 24


def test_v6_contract_remains_available_for_rollback():
    raw = draft()
    del raw["social_kind"]
    raw["social_comment"] = False
    _, model = execute(raw, candidate=PLANNER_V6_ID)
    assert PLANNER_ID == PLANNER_V7_ID
    assert next(iter(model.calls[0]["schema"]["oneOf"][0]["properties"])) == "social_comment"
    assert "social_kind" not in model.calls[0]["schema"]["oneOf"][0]["properties"]
    assert "outputs" not in json.loads(model.calls[0]["content"])["capabilities"][0]


def test_new_development_dataset_is_bilingual_and_includes_the_observed_failures():
    dataset = load_intelligence_evaluation_dataset(DATASET)
    assert dataset.split == "development"
    assert dataset.turn_count == 44
    assert dataset.language_counts() == {"en": 22, "tr": 22}
    questions = {question for group in dataset.groups for question in group.questions}
    assert {"Hey there.", "Position.", "Do I have any damage?"}.issubset(questions)
    assert any(group.recent_dialogue for group in dataset.groups)
    assert any(group.expected.evidence_outcome == "partial" for group in dataset.groups)


def test_non_social_empty_expectation_requires_an_explicit_missing_reason():
    with pytest.raises(ValidationError):
        ExpectedContextPlan(temporal_scope="current", evidence_outcome="available")
    accepted = ExpectedContextPlan(
        temporal_scope="current",
        evidence_outcome="unavailable",
        required_unknowns=("requested_analysis_unavailable",),
    )
    assert accepted.capability_ids == accepted.queries == ()
    with pytest.raises(ValidationError):
        ExpectedContextPlan(
            temporal_scope="current",
            evidence_outcome="partial",
            required_unknowns=("requested_analysis_unavailable",),
        )


def test_runner_does_not_count_a_greeting_as_a_correct_radio_check():
    dataset = load_intelligence_evaluation_dataset(DATASET)
    group = next(group for group in dataset.groups if group.id == "understand-radio-check-en")
    dataset = dataset.model_copy(update={"groups": (group,)})

    class GreetingPlanner:
        async def plan(self, request, *_):
            return ContextPlan(
                turn_id=request.turn_id,
                planner_id="scripted",
                temporal_scope="social",
                situation=("driver_social_turn", "driver_greeting"),
            )

    candidate = IntelligenceCandidateMetadata(
        candidate_id="scripted",
        runtime_profile="portable",
        candidate_kind="control",
        adapter="test",
        model="test",
    )
    results, _ = asyncio.run(
        evaluate_intelligence_datasets(
            ROOT,
            PolicyContextConfig(),
            GreetingPlanner(),
            candidate,
            (dataset,),
        )
    )
    assert all(result["purpose_passed"] for result in results)
    assert not any(result["plan_passed"] for result in results)
    assert not any(result["passed"] for result in results)
    assert all(result["observed_social_tags"] == ["driver_greeting"] for result in results)
    assert not any(result["situation_passed"] for result in results)
    assert all(result["common_contract_passed"] for result in results)


@pytest.mark.parametrize(
    "updates",
    [
        {"requested_facts": ["],"]},
        {"capability_ids": ["current_classification", "current_classification"]},
        {
            "queries": [
                {"signal_id": "player.speed_mps", "operation": "latest", "window_s": None},
                {"signal_id": "player.speed_mps", "operation": "latest", "window_s": None},
            ]
        },
    ],
)
def test_v7_rejects_malformed_inventory_and_duplicate_work(updates):
    with pytest.raises(ContextPlanRejection, match="context_model_response_invalid"):
        execute(draft(**updates))


def test_v7_is_default_and_explicit_candidates_are_validated():
    assert ConversationConfig().context_planner == PLANNER_V7_ID
    assert ConversationConfig(context_planner=PLANNER_V6_ID).context_planner == PLANNER_V6_ID
    assert ConversationConfig(context_planner=PLANNER_V7_ID).context_planner == PLANNER_V7_ID
    with pytest.raises(ValidationError):
        ConversationConfig(context_planner="unknown")
    args = _parser().parse_args(
        [
            "intelligence-replay",
            "--fixture",
            "fixtures/synthetic/intelligence",
            "--question",
            "Are you there?",
            "--candidate",
            PLANNER_V7_ID,
        ]
    )
    assert args.candidate == PLANNER_V7_ID


def test_live_factory_honors_explicit_candidate_without_changing_defaults(monkeypatch):
    from race_engineer.intelligence.factory import live_intelligence

    selected = []

    def planner(config, **kwargs):
        selected.append(kwargs["planner_id"])
        return object()

    monkeypatch.setattr("race_engineer.intelligence.factory.QwenContextQueryPlanner", planner)
    live_intelligence(ConversationConfig(), memory())
    live_intelligence(ConversationConfig(context_planner=PLANNER_V6_ID), memory())
    assert selected == [PLANNER_V7_ID, PLANNER_V6_ID]


def test_repository_config_and_direct_planner_default_to_v7():
    config = load_config(ROOT / "config/default.toml", environ={})
    assert config.conversation.context_planner == PLANNER_V7_ID == PLANNER_ID
    model = Model(draft())
    planner = QwenContextQueryPlanner(config.conversation, model=model)
    state = memory()
    result = asyncio.run(
        planner.plan(turn(), state.signal_catalog(), DeterministicRaceCapabilities(state).catalog())
    )
    assert result.planner_id == PLANNER_V7_ID
    assert "social_kind" in model.calls[0]["schema"]["properties"]


def test_live_factory_uses_actual_v7_contract_by_default(monkeypatch):
    from race_engineer.intelligence.factory import live_intelligence

    model = Model(
        draft(social_kind="radio_check", requested_facts=[], temporal_scope=None, capability_ids=[])
    )
    monkeypatch.setattr("race_engineer.intelligence.factory.LocalJsonModel", lambda _: model)
    engineer = live_intelligence(ConversationConfig(), memory())
    packet = asyncio.run(engineer._context.analyze(turn("Are you there?")))
    assert packet.situation == ("driver_social_turn", "driver_radio_check")
    assert packet.evidence == packet.unknowns == ()
    assert "social_kind" in model.calls[0]["schema"]["properties"]


def test_fresh_paraphrases_are_development_only_and_disjoint():
    original = load_intelligence_evaluation_dataset(DATASET)
    fresh = load_intelligence_evaluation_dataset(
        DATASET.with_name("development-understanding-validation.json")
    )
    assert fresh.split == "development"
    assert fresh.language_counts() == {"en": 6, "tr": 6}
    original_text = {q for g in original.groups for q in g.questions}
    assert not original_text.intersection(q for g in fresh.groups for q in g.questions)


@pytest.mark.parametrize(
    "updates,reason",
    [
        ({"capability_ids": ["not_advertised"]}, "planner_selected_unknown_capability"),
        (
            {"capability_ids": ["projected_rejoin_position"]},
            "planner_selected_wrong_capability_scope",
        ),
        (
            {"queries": [{"signal_id": "not_advertised", "operation": "latest", "window_s": None}]},
            "planner_selected_unknown_signal",
        ),
        (
            {"queries": [{"signal_id": "player.speed_mps", "operation": "mean", "window_s": 10}]},
            "planner_selected_invalid_query",
        ),
        (
            {"missing_information": "unsupported_projection"},
            "planner_missing_information_scope_mismatch",
        ),
    ],
)
def test_v7_flat_wire_schema_preserves_authoritative_validation(updates, reason):
    with pytest.raises(ContextPlanRejection, match=reason):
        execute(draft(**updates))
