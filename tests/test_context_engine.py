import asyncio
import json
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta

import pytest
from context_plan_fixtures import wire_plan

from race_engineer.config import ConversationConfig
from race_engineer.core.contracts import (
    OpponentState,
    PlayerState,
    RaceContext,
    TelemetryFrame,
)
from race_engineer.core.intelligence import (
    CapabilityDescriptor,
    CapabilityRequest,
    ContextPlan,
    DriverTurn,
    EvidenceQuery,
    SignalDescriptor,
)
from race_engineer.intelligence.capabilities import DeterministicRaceCapabilities
from race_engineer.intelligence.context_engine import (
    ContextEngineerError,
    QueryDrivenContextEngineer,
)
from race_engineer.intelligence.context_planner import QwenContextQueryPlanner
from race_engineer.intelligence.local_model import LocalIntelligenceError
from race_engineer.intelligence.planner_diagnostics import ContextPlanRejection
from race_engineer.intelligence.telemetry_memory import BoundedTelemetryMemory

NOW = datetime(2026, 9, 28, 12, tzinfo=UTC)


def race_context(
    sequence: int,
    session_time_s: float,
    *,
    position: int | None = 6,
    speed_mps: float = 20,
    opponent_positions: tuple[int, ...] = (),
) -> RaceContext:
    return RaceContext(
        frame=TelemetryFrame(
            source="test",
            session_id="iracing:1",
            sequence=sequence,
            observed_at=NOW + timedelta(seconds=session_time_s),
            session_time_s=session_time_s,
            player=PlayerState(
                driver_id="player",
                position=position,
                speed_mps=speed_mps,
            ),
            opponents=tuple(
                OpponentState(driver_id=f"opponent-{index}", position=value)
                for index, value in enumerate(opponent_positions, start=1)
            ),
            capabilities=("position", "speed"),
        )
    )


def driver_turn(text: str = "Where are we?") -> DriverTurn:
    return DriverTurn(
        turn_id="turn-context",
        transcript=text,
        received_at=NOW,
        session_id="iracing:1",
        generation=0,
        asr_language="en",
    )


def evidence_query(
    query_id: str,
    signal: str,
    operation: str = "latest",
    window_s: float | None = None,
) -> EvidenceQuery:
    return EvidenceQuery.model_validate(
        {
            "query_id": query_id,
            "selector": {"source": "player", "signal": signal},
            "operation": operation,
            "window_s": window_s,
        }
    )


class ScriptedPlanner:
    def __init__(self, result: ContextPlan) -> None:
        self.result = result
        self.signals: tuple[SignalDescriptor, ...] = ()
        self.capabilities: tuple[CapabilityDescriptor, ...] = ()

    async def plan(
        self,
        turn: DriverTurn,
        signals: Sequence[SignalDescriptor],
        capabilities: Sequence[CapabilityDescriptor],
    ) -> ContextPlan:
        del turn
        self.signals = tuple(signals)
        self.capabilities = tuple(capabilities)
        return self.result


def plan(
    *queries: EvidenceQuery,
    turn_id: str = "turn-context",
    capability_requests: tuple[CapabilityRequest, ...] = (),
    situation: tuple[str, ...] = (),
    unknowns: tuple[str, ...] = (),
) -> ContextPlan:
    return ContextPlan(
        turn_id=turn_id,
        planner_id="scripted",
        queries=queries,
        capability_requests=capability_requests,
        situation=situation,
        unknowns=unknowns,
    )


def test_context_engine_selects_and_refreshes_generic_direct_evidence():
    memory = BoundedTelemetryMemory()
    memory.update(race_context(1, 0, position=6))
    planner = ScriptedPlanner(plan(evidence_query("place", "position")))
    engine = QueryDrivenContextEngineer(memory, planner)

    initial = asyncio.run(engine.analyze(driver_turn()))
    assert initial.evidence[0].value == 6
    assert any(item.selector.signal == "position" for item in planner.signals)

    memory.update(race_context(2, 1, position=5))
    refreshed = asyncio.run(engine.refresh(initial, ("place",)))
    assert refreshed.evidence[0].value == 5
    assert refreshed.source_sequence == 2


def test_context_engine_executes_window_analysis_without_spoken_query_code():
    memory = BoundedTelemetryMemory(history_s=10)
    memory.update(race_context(1, 0, speed_mps=10))
    memory.update(race_context(2, 1, speed_mps=20))
    memory.update(race_context(3, 2, speed_mps=30))
    selected = evidence_query("speed-change", "speed_mps", "trend", 2)
    engine = QueryDrivenContextEngineer(memory, ScriptedPlanner(plan(selected)))

    packet = asyncio.run(engine.analyze(driver_turn("What is happening to our pace?")))

    assert packet.evidence[0].metric == "speed_mps"
    assert packet.evidence[0].value == 10
    assert packet.evidence[0].kind == "derived"


def test_context_engine_executes_and_refreshes_selected_race_capability():
    memory = BoundedTelemetryMemory()
    memory.update(race_context(1, 0, position=6, opponent_positions=(5,)))
    registry = DeterministicRaceCapabilities(memory)
    request = CapabilityRequest(
        request_id="c1",
        capability_id="current_classification",
    )
    planner = ScriptedPlanner(plan(capability_requests=(request,)))
    engine = QueryDrivenContextEngineer(memory, planner, registry)

    initial = asyncio.run(engine.analyze(driver_turn("Am I last?")))
    initial_values = {item.metric: item.value for item in initial.evidence}

    assert initial_values == {
        "position": 6,
        "classified_cars": 2,
        "is_last": True,
    }
    assert {
        "current_classification",
        "fuel_range",
        "projected_rejoin_position",
    } <= {item.capability_id for item in planner.capabilities}

    memory.update(race_context(2, 1, position=5, opponent_positions=(6,)))
    refreshed = asyncio.run(
        engine.refresh(
            initial,
            ("c1:player_position", "c1:last_position"),
        )
    )

    assert [item.value for item in refreshed.evidence] == [5, 6]
    assert refreshed.source_sequence == 2


def test_context_engine_allows_social_context_without_forcing_telemetry():
    memory = BoundedTelemetryMemory()
    memory.update(race_context(1, 0))
    selected = plan(situation=("driver_frustration",))
    engine = QueryDrivenContextEngineer(memory, ScriptedPlanner(selected))

    packet = asyncio.run(engine.analyze(driver_turn("He has no idea about racing.")))

    assert packet.evidence == ()
    assert packet.situation == ("driver_frustration",)


def test_context_engine_preserves_explicit_unknowns_and_missing_values():
    memory = BoundedTelemetryMemory()
    memory.update(race_context(1, 0, position=None))
    selected = plan(
        evidence_query("place", "position"),
        unknowns=("sector_comparison_not_available",),
    )
    engine = QueryDrivenContextEngineer(memory, ScriptedPlanner(selected))

    packet = asyncio.run(engine.analyze(driver_turn("Why are we slow?")))

    assert packet.evidence[0].kind == "unknown"
    assert packet.unknowns == ("sector_comparison_not_available", "place")


def test_context_packet_retains_every_unknown_from_a_bounded_plan():
    memory = BoundedTelemetryMemory()
    memory.update(race_context(1, 0, position=None))
    queries = tuple(evidence_query(f"e{index}", "position") for index in range(24))
    engine = QueryDrivenContextEngineer(memory, ScriptedPlanner(plan(*queries)))

    packet = asyncio.run(engine.analyze(driver_turn("What can you establish?")))

    assert len(packet.evidence) == 24
    assert packet.unknowns == tuple(f"e{index}" for index in range(24))


def test_context_engine_rejects_planner_scope_and_unknown_refresh_ids():
    memory = BoundedTelemetryMemory()
    memory.update(race_context(1, 0))
    wrong = QueryDrivenContextEngineer(
        memory,
        ScriptedPlanner(plan(turn_id="other-turn")),
    )
    with pytest.raises(ContextEngineerError, match="turn"):
        asyncio.run(wrong.analyze(driver_turn()))

    valid = QueryDrivenContextEngineer(
        memory,
        ScriptedPlanner(plan(evidence_query("place", "position"))),
    )
    packet = asyncio.run(valid.analyze(driver_turn()))
    with pytest.raises(ContextEngineerError, match="unknown"):
        asyncio.run(valid.refresh(packet, ("not-approved",)))


class FakeJsonModel:
    def __init__(self, result: object) -> None:
        self.result = wire_plan(result)
        self.captured: dict[str, object] | None = None

    async def request(self, **kwargs):
        self.captured = kwargs
        return self.result


def test_qwen_context_planner_receives_schema_not_raw_values_and_returns_queries():
    memory = BoundedTelemetryMemory()
    memory.update(race_context(1, 0, position=6))
    model = FakeJsonModel(
        {
            "queries": [
                {
                    "signal_id": "player.position",
                    "operation": "latest",
                    "window_s": None,
                }
            ],
        }
    )
    planner = QwenContextQueryPlanner(ConversationConfig(), model=model)
    capabilities = DeterministicRaceCapabilities(memory).catalog()

    result = asyncio.run(planner.plan(driver_turn(), memory.signal_catalog(), capabilities))

    assert result.planner_id == "qwen-context-v5"
    assert result.queries == (evidence_query("e1", "position"),)
    assert model.captured is not None
    payload = json.loads(str(model.captured["content"]))
    assert payload["transcript"] == "Where are we?"
    assert all("value" not in signal for signal in payload["signals"])
    assert payload["capabilities"][0]["id"] == "current_classification"
    assert "value" not in payload["capabilities"][0]
    schema = model.captured["schema"]
    assert all(branch["additionalProperties"] is False for branch in schema["oneOf"])
    assert '"$ref"' not in json.dumps(schema)
    assert list(payload)[-3:] == ["language", "recent_dialogue", "transcript"]


def test_model_selected_field_aggregate_can_supply_last_place_evidence():
    memory = BoundedTelemetryMemory()
    memory.update(race_context(1, 0, position=6))
    model = FakeJsonModel(
        {
            "queries": [
                {
                    "signal_id": "player.position",
                    "operation": "latest",
                },
                {
                    "signal_id": "field.position",
                    "operation": "maximum",
                },
            ],
        }
    )
    engine = QueryDrivenContextEngineer(
        memory,
        QwenContextQueryPlanner(ConversationConfig(), model=model),
    )

    packet = asyncio.run(engine.analyze(driver_turn("Am I last?")))

    assert [(item.subject, item.value) for item in packet.evidence] == [
        ("player", 6),
        ("field", 6),
    ]


def test_qwen_context_planner_selects_advertised_deterministic_capability():
    memory = BoundedTelemetryMemory()
    memory.update(race_context(1, 0, position=5, opponent_positions=(6,)))
    registry = DeterministicRaceCapabilities(memory)
    model = FakeJsonModel(
        {
            "temporal_scope": "current",
            "queries": [],
            "capability_ids": ["current_classification"],
        }
    )
    engine = QueryDrivenContextEngineer(
        memory,
        QwenContextQueryPlanner(ConversationConfig(), model=model),
        registry,
    )

    packet = asyncio.run(engine.analyze(driver_turn("Am I last?")))

    assert [item.evidence_id for item in packet.evidence] == [
        "c1:player_position",
        "c1:last_position",
        "c1:classified_cars",
        "c1:is_last",
    ]
    assert packet.evidence[-1].value is False
    assert model.captured is not None
    schema = model.captured["schema"]
    capability_ids = schema["oneOf"][1]["properties"]["capability_ids"]["items"]["enum"]
    assert capability_ids[:2] == ["current_classification", "fuel_range"]
    assert "relative_pace_behind" in capability_ids
    assert "projected_rejoin_position" not in capability_ids
    future_ids = schema["oneOf"][3]["properties"]["capability_ids"]["items"]["enum"]
    assert "projected_rejoin_position" in future_ids


def test_qwen_context_planner_verifies_driver_telemetry_claims():
    memory = BoundedTelemetryMemory()
    memory.update(race_context(1, 0, position=5))
    model = FakeJsonModel(
        {
            "queries": [
                {
                    "signal_id": "player.position",
                    "operation": "latest",
                },
                {
                    "signal_id": "field.position",
                    "operation": "maximum",
                },
            ],
        }
    )
    engine = QueryDrivenContextEngineer(
        memory,
        QwenContextQueryPlanner(ConversationConfig(), model=model),
    )

    packet = asyncio.run(engine.analyze(driver_turn("I'm P6 in a ten-car grid.")))

    assert [(item.subject, item.value) for item in packet.evidence] == [
        ("player", 5),
        ("field", 5),
    ]
    assert model.captured is not None
    assert "select evidence to verify it" in str(model.captured["system_prompt"])


def test_qwen_context_planner_marks_unsupported_pit_projection():
    memory = BoundedTelemetryMemory()
    memory.update(race_context(1, 0, position=5))
    model = FakeJsonModel(
        {
            "temporal_scope": "future_counterfactual",
            "queries": [],
            "missing_information": "unsupported_projection",
        }
    )
    planner = QwenContextQueryPlanner(ConversationConfig(), model=model)

    result = asyncio.run(
        planner.plan(
            driver_turn("Where would I be if I pit this lap?"),
            memory.signal_catalog(),
            DeterministicRaceCapabilities(memory).catalog(),
        )
    )

    assert result.queries == ()
    assert result.situation == ("race_information_request", "future_counterfactual")
    assert result.unknowns == ("requested_projection_unavailable",)
    assert model.captured is not None
    assert "Future questions require" in str(model.captured["system_prompt"])


def test_future_rejoin_capability_returns_its_authoritative_unavailable_reason():
    memory = BoundedTelemetryMemory()
    memory.update(race_context(1, 0, position=5))
    model = FakeJsonModel(
        {
            "temporal_scope": "future_counterfactual",
            "queries": [],
            "capability_ids": ["projected_rejoin_position"],
        }
    )
    engine = QueryDrivenContextEngineer(
        memory,
        QwenContextQueryPlanner(ConversationConfig(), model=model),
        DeterministicRaceCapabilities(memory),
    )

    packet = asyncio.run(engine.analyze(driver_turn("Where would I be if I pit this lap?")))

    assert packet.evidence[0].kind == "unknown"
    assert packet.unknowns == (
        "projected_rejoin_position:missing_rejoin_projection_model",
        "c1:rejoin_position",
    )


def test_qwen_context_planner_rejects_unadvertised_or_malformed_output():
    memory = BoundedTelemetryMemory()
    memory.update(race_context(1, 0))
    invented = FakeJsonModel(
        {
            "queries": [
                {
                    "signal_id": "player.tire_magic",
                    "operation": "latest",
                }
            ]
        }
    )
    planner = QwenContextQueryPlanner(ConversationConfig(), model=invented)
    with pytest.raises(LocalIntelligenceError, match="unknown_signal"):
        asyncio.run(planner.plan(driver_turn("How are the tires?"), memory.signal_catalog(), ()))

    malformed = QwenContextQueryPlanner(
        ConversationConfig(),
        model=FakeJsonModel({"queries": [], "answer": "You're P6"}),
    )
    with pytest.raises(LocalIntelligenceError, match="response_invalid"):
        asyncio.run(malformed.plan(driver_turn(), memory.signal_catalog(), ()))


def test_signal_schema_enumerates_complete_catalog_ids_and_operation_rules():
    memory = BoundedTelemetryMemory()
    memory.update(race_context(1, 0, opponent_positions=(5, 7)))
    model = FakeJsonModel({"purpose": "social", "temporal_scope": None})
    planner = QwenContextQueryPlanner(ConversationConfig(), model=model)
    asyncio.run(planner.plan(driver_turn(), memory.signal_catalog(), ()))

    payload = json.loads(model.captured["content"])
    by_id = {item["id"]: item for item in payload["signals"]}
    branches = [
        branch
        for schema in model.captured["schema"]["oneOf"][1:3]
        for branch in schema["properties"]["queries"]["items"]["oneOf"]
    ]
    advertised = set()
    for branch in branches:
        props = branch["properties"]
        identifiers = props["signal_id"]["enum"]
        advertised.update(identifiers)
        for identifier in identifiers:
            item = by_id[identifier]
            operations = props["operation"]["enum"]
            if item["source"] == "field":
                assert "latest" not in operations
                assert props["window_s"] == {"type": "null"}
            elif "latest" in operations:
                assert operations == ["latest"]
                assert props["window_s"] == {"type": "null"}
            else:
                assert "count" not in operations
                assert props["window_s"]["exclusiveMinimum"] == 0
    assert advertised == set(by_id)
    assert "context.position" not in advertised
    assert by_id["opponent_0.position"]["subject_id"] == "opponent-1"
    assert by_id["opponent_1.position"]["subject_id"] == "opponent-2"


def test_signal_binding_maps_opponent_id_without_model_authored_subject():
    memory = BoundedTelemetryMemory()
    memory.update(race_context(1, 0, opponent_positions=(5, 7)))
    model = FakeJsonModel(
        {
            "queries": [{"signal_id": "opponent_1.position", "operation": "latest"}],
        }
    )
    engine = QueryDrivenContextEngineer(
        memory, QwenContextQueryPlanner(ConversationConfig(), model=model)
    )
    packet = asyncio.run(engine.analyze(driver_turn()))
    assert packet.evidence[0].value == 7
    assert packet.evidence[0].subject == "opponent-2"


def test_empty_catalog_does_not_advertise_any_queries_or_capabilities():
    model = FakeJsonModel({"purpose": "social", "temporal_scope": None})
    planner = QwenContextQueryPlanner(ConversationConfig(), model=model)
    asyncio.run(planner.plan(driver_turn("Thanks"), (), ()))
    for branch in model.captured["schema"]["oneOf"]:
        properties = branch["properties"]
        assert properties["queries"]["maxItems"] == 0
        assert properties["capability_ids"]["maxItems"] == 0


def test_catalog_binding_is_order_independent_and_rebuilt_when_opponent_changes():
    memory = BoundedTelemetryMemory()
    memory.update(race_context(1, 0, opponent_positions=(5, 7)))
    signals = memory.signal_catalog()
    model = FakeJsonModel(
        {"queries": [{"signal_id": "opponent_0.position", "operation": "latest"}]}
    )
    planner = QwenContextQueryPlanner(ConversationConfig(), model=model)
    first = asyncio.run(planner.plan(driver_turn(), signals, ()))
    first_payload = model.captured["content"]
    reordered = asyncio.run(planner.plan(driver_turn(), tuple(reversed(signals)), ()))
    assert reordered.queries == first.queries
    assert model.captured["content"] == first_payload
    changed = tuple(
        item.model_copy(
            update={"selector": item.selector.model_copy(update={"subject_id": "new-opponent"})}
        )
        if item.selector.subject_id == "opponent-1"
        else item
        for item in signals
    )
    latest = asyncio.run(planner.plan(driver_turn(), changed, ()))
    assert latest.queries[0].selector.subject_id == "new-opponent"
    assert model.captured["content"] != first_payload
    assert "opponent-1" not in model.captured["content"]


@pytest.mark.parametrize(
    ("query", "counter", "reason"),
    [
        (
            {"signal_id": "context.position", "operation": "latest"},
            "unknown_signal_count",
            "unknown_signal",
        ),
        (
            {"signal_id": "field.position", "operation": "latest"},
            "invalid_query_count",
            "invalid_query",
        ),
        (
            {"signal_id": "player.position", "operation": "delta"},
            "invalid_query_count",
            "invalid_query",
        ),
        (
            {"signal_id": "player.position", "operation": "latest", "window_s": 10},
            "invalid_query_count",
            "invalid_query",
        ),
    ],
)
def test_invalid_query_preserves_known_selection_but_rejects_entire_turn(query, counter, reason):
    memory = BoundedTelemetryMemory()
    memory.update(race_context(1, 0))
    model = FakeJsonModel({"capability_ids": ["current_classification"], "queries": [query]})
    planner = QwenContextQueryPlanner(ConversationConfig(), model=model)
    with pytest.raises(ContextPlanRejection, match=reason) as caught:
        asyncio.run(
            planner.plan(
                driver_turn(),
                memory.signal_catalog(),
                DeterministicRaceCapabilities(memory).catalog(),
            )
        )
    diagnostic = caught.value.diagnostic
    assert diagnostic.capability_ids == ("current_classification",)
    assert diagnostic.temporal_scope == "current"
    assert diagnostic.queries == ()
    assert getattr(diagnostic, counter) == 1


def test_capability_and_raw_query_can_coexist_for_compound_question():
    memory = BoundedTelemetryMemory()
    memory.update(race_context(1, 0))
    model = FakeJsonModel(
        {
            "capability_ids": ["current_classification"],
            "queries": [{"signal_id": "player.speed_mps", "operation": "latest"}],
        }
    )
    engine = QueryDrivenContextEngineer(
        memory,
        QwenContextQueryPlanner(ConversationConfig(), model=model),
        DeterministicRaceCapabilities(memory),
    )
    packet = asyncio.run(engine.analyze(driver_turn("What is my position and speed?")))
    assert packet.evidence[0].metric == "speed_mps"
    assert any(item.metric == "position" for item in packet.evidence)


@pytest.mark.parametrize("question", ["Thanks", "Sağ ol", "What a clown!", "Yeter artık!"])
def test_model_selected_social_purpose_needs_no_telemetry_or_unknown(question):
    memory = BoundedTelemetryMemory()
    memory.update(race_context(1, 0))
    model = FakeJsonModel({"purpose": "social", "temporal_scope": None})
    engine = QueryDrivenContextEngineer(
        memory, QwenContextQueryPlanner(ConversationConfig(), model=model)
    )
    packet = asyncio.run(engine.analyze(driver_turn(question)))
    assert packet.evidence == ()
    assert packet.unknowns == ()
    assert packet.situation == ("driver_social_turn",)
    social = model.captured["schema"]["oneOf"][0]["properties"]
    assert social["missing_information"]["enum"] == ["none"]
    assert social["queries"]["maxItems"] == 0
    assert social["capability_ids"]["maxItems"] == 0


def test_mixed_social_and_factual_turn_keeps_both_purposes():
    memory = BoundedTelemetryMemory()
    memory.update(race_context(1, 0))
    planner = QwenContextQueryPlanner(
        ConversationConfig(),
        model=FakeJsonModel(
            {
                "purpose": "mixed",
                "queries": [{"signal_id": "player.speed_mps", "operation": "latest"}],
            }
        ),
    )
    packet = asyncio.run(
        QueryDrivenContextEngineer(memory, planner).analyze(
            driver_turn("That's ridiculous. What's my current speed?")
        )
    )
    assert packet.situation == ("driver_social_turn", "race_information_request")
    assert len(packet.evidence) == 1
    assert packet.unknowns == ()


@pytest.mark.parametrize(
    "raw",
    [
        {
            "purpose": "social",
            "temporal_scope": None,
            "missing_information": "unsupported_analysis",
        },
        {
            "purpose": "social",
            "temporal_scope": None,
            "queries": [{"signal_id": "player.speed_mps", "operation": "latest"}],
        },
        {
            "temporal_scope": "future_counterfactual",
            "queries": [{"signal_id": "player.position", "operation": "latest"}],
        },
        {
            "temporal_scope": "historical",
            "queries": [{"signal_id": "field.position", "operation": "mean"}],
        },
        {"missing_information": "unknown"},
        {"unknowns": ["unknown"]},
        {"purpose": "social", "temporal_scope": "current"},
        {"purpose": "race_information", "temporal_scope": None},
        {"missing_information": "unsupported_projection", "temporal_scope": "current"},
        {},
    ],
)
def test_invalid_purpose_scope_or_unknown_never_executes(raw, monkeypatch):
    memory = BoundedTelemetryMemory()
    memory.update(race_context(1, 0))
    monkeypatch.setattr(memory, "query_many", lambda _: pytest.fail("invalid plan executed"))
    planner = QwenContextQueryPlanner(ConversationConfig(), model=FakeJsonModel(raw))
    with pytest.raises(ContextPlanRejection):
        asyncio.run(QueryDrivenContextEngineer(memory, planner).analyze(driver_turn()))


def test_historical_signal_and_specific_unsupported_analysis_are_preserved():
    memory = BoundedTelemetryMemory()
    memory.update(race_context(1, 0))
    planner = QwenContextQueryPlanner(
        ConversationConfig(),
        model=FakeJsonModel(
            {
                "temporal_scope": "historical",
                "queries": [{"signal_id": "player.speed_mps", "operation": "mean", "window_s": 30}],
                "missing_information": "unsupported_analysis",
            }
        ),
    )
    selected = asyncio.run(planner.plan(driver_turn(), memory.signal_catalog(), ()))
    assert selected.queries[0].window_s == 30
    assert selected.unknowns == ("requested_analysis_unavailable",)


@pytest.mark.parametrize(
    "operation, expected", [("mean", 25), ("maximum", 30), ("minimum", 20), ("delta", 10)]
)
def test_model_selected_historical_operator_is_executed_without_substitution(operation, expected):
    memory = BoundedTelemetryMemory()
    for sequence, speed in enumerate((20, 25, 30)):
        memory.update(race_context(sequence, sequence * 10, speed_mps=speed))
    model = FakeJsonModel(
        {
            "temporal_scope": "historical",
            "queries": [{"signal_id": "player.speed_mps", "operation": operation, "window_s": 20}],
        }
    )
    engine = QueryDrivenContextEngineer(
        memory, QwenContextQueryPlanner(ConversationConfig(), model=model)
    )
    packet = asyncio.run(engine.analyze(driver_turn("Authored operation execution control")))
    assert len(packet.evidence) == 1
    assert packet.evidence[0].value == expected
    assert packet.unknowns == ()


def test_inventory_guides_the_model_but_never_becomes_evidence_or_session_context():
    memory = BoundedTelemetryMemory()
    memory.update(race_context(1, 0))
    model = FakeJsonModel(
        {
            "social_comment": True,
            "requested_facts": ["private copied driver words about speed"],
            "temporal_scope": "current",
            "queries": [{"signal_id": "player.speed_mps", "operation": "latest"}],
        }
    )
    packet = asyncio.run(
        QueryDrivenContextEngineer(
            memory, QwenContextQueryPlanner(ConversationConfig(), model=model)
        ).analyze(driver_turn("A different wording"))
    )
    assert packet.situation == ("driver_social_turn", "race_information_request")
    assert "private" not in packet.model_dump_json()
    assert "requested_facts" not in packet.model_dump_json()
    assert model.captured["max_tokens"] == 384
    social, *factual = model.captured["schema"]["oneOf"]
    assert list(social["properties"])[:2] == ["social_comment", "requested_facts"]
    assert "purpose" not in social["properties"]
    assert social["properties"]["social_comment"]["const"] is True
    assert social["properties"]["requested_facts"]["maxItems"] == 0
    for branch in factual:
        assert branch["properties"]["requested_facts"]["minItems"] == 1
        assert branch["properties"]["requested_facts"]["maxItems"] == 8


@pytest.mark.parametrize(
    "metadata",
    [
        {"social_comment": "true", "requested_facts": ["speed"]},
        {"social_comment": 1, "requested_facts": ["speed"]},
        {"social_comment": False},
        {"social_comment": False, "requested_facts": []},
        {"social_comment": False, "requested_facts": [""]},
        {"social_comment": False, "requested_facts": ["x" * 121]},
        {"social_comment": False, "requested_facts": ["speed"] * 9},
        {"social_comment": False, "requested_facts": [], "temporal_scope": None},
        {"social_comment": True, "requested_facts": ["speed"], "temporal_scope": None},
        {"social_comment": False, "requested_facts": ["speed"], "purpose": "mixed"},
    ],
)
def test_forged_inventory_metadata_is_rejected_before_execution(metadata, monkeypatch):
    memory = BoundedTelemetryMemory()
    memory.update(race_context(1, 0))
    monkeypatch.setattr(memory, "query_many", lambda _: pytest.fail("invalid draft executed"))
    model = FakeJsonModel(
        {**metadata, "queries": [{"signal_id": "player.speed_mps", "operation": "latest"}]}
    )
    engine = QueryDrivenContextEngineer(
        memory, QwenContextQueryPlanner(ConversationConfig(), model=model)
    )
    with pytest.raises(ContextPlanRejection):
        asyncio.run(engine.analyze(driver_turn()))
