import asyncio
import json
from pathlib import Path

import pytest
from test_context_engine import driver_turn, race_context
from test_intelligence_local_model import FakeConnection, payload

from race_engineer.application.intelligence_cli import _contexts
from race_engineer.config import AppConfig, ConversationConfig, load_config
from race_engineer.core.intelligence import DriverTurn
from race_engineer.fixtures import load_fixture
from race_engineer.intelligence.capabilities import DeterministicRaceCapabilities
from race_engineer.intelligence.context_engine import QueryDrivenContextEngineer
from race_engineer.intelligence.context_planner import LFM_NATIVE_ID
from race_engineer.intelligence.factory import live_intelligence
from race_engineer.intelligence.local_model import LocalIntelligenceError
from race_engineer.intelligence.native_lfm import LfmNativePlanner
from race_engineer.intelligence.native_model import (
    NativeToolCall,
    NativeToolReply,
)
from race_engineer.intelligence.planner_diagnostics import ContextPlanRejection
from race_engineer.intelligence.telemetry_memory import BoundedTelemetryMemory
from scripts.setup_lfm3_trial import trial_profile

ROOT = Path(__file__).parents[1]


def tool(name, **arguments):
    return NativeToolCall(call_id=f"call-{name}", name=name, arguments=arguments)


class Model:
    def __init__(self, *calls):
        self.result = NativeToolReply(calls=calls)
        self.requests = []

    async def request_tools(self, **kwargs):
        self.requests.append(kwargs)
        return self.result


def setup(*calls):
    memory = BoundedTelemetryMemory()
    memory.update(race_context(1, 90, position=6, opponent_positions=(1, 6)))
    memory.update(race_context(2, 100, position=5, opponent_positions=(1, 6)))
    registry = DeterministicRaceCapabilities(memory)
    model = Model(*calls)
    planner = LfmNativePlanner(ConversationConfig(context_planner=LFM_NATIVE_ID), model=model)
    return memory, registry, model, planner


def plan(planner, memory, registry, turn=None):
    return asyncio.run(
        planner.plan(turn or driver_turn(), memory.signal_catalog(), registry.catalog())
    )


def test_native_selection_reuses_capabilities_and_preserves_mixed_turn_and_history():
    memory, registry, model, planner = setup(
        tool("get_current_race_position"), tool("acknowledge_driver", kind="radio_check")
    )
    turn = driver_turn("Can you hear me, and where are we running?").model_copy(
        update={"recent_dialogue": ("driver: Fuel?", "engineer: Fuel checked.")}
    )
    result = plan(planner, memory, registry, turn)
    assert result.planner_id == LFM_NATIVE_ID and result.temporal_scope == "current"
    assert result.capability_requests[0].capability_id == "current_classification"
    assert {"driver_social_turn", "driver_radio_check", "race_information_request"}.issubset(
        result.situation
    )
    assert model.requests[0]["messages"][-1]["content"] == turn.transcript
    assert len(model.requests[0]["messages"]) == 2
    assert "driver: Fuel?" in model.requests[0]["messages"][0]["content"]
    assert "engineer: Fuel checked." in model.requests[0]["messages"][0]["content"]
    missing = next(
        t
        for t in model.requests[0]["tools"]
        if t["function"]["name"] == "report_unavailable_information"
    )
    assert "$defs" in missing["function"]["parameters"]  # Local type refs resolve for llama.cpp.


@pytest.mark.parametrize(
    "call,scope,operation",
    [
        (tool("read_current_telemetry", signal_id="player.speed_mps"), "current", "latest"),
        (
            tool(
                "summarize_recent_telemetry",
                signal_id="player.speed_mps",
                operation="mean",
                window_s=10,
            ),
            "historical",
            "mean",
        ),
        (
            tool("summarize_current_field", signal_id="field.position", operation="count"),
            "current",
            "count",
        ),
    ],
)
def test_native_raw_tools_bind_only_advertised_selectors(call, scope, operation):
    memory, registry, _, planner = setup(call)
    result = plan(planner, memory, registry)
    assert result.temporal_scope == scope and result.queries[0].operation == operation
    assert not result.capability_requests


@pytest.mark.parametrize(
    "calls,reason",
    [
        ((tool("run_shell_command", command="whoami"),), "native_unknown_tool"),
        ((tool("get_current_race_position", position=1),), "native_tool_arguments_invalid"),
        (
            (
                tool(
                    "summarize_recent_telemetry",
                    signal_id="player.speed_mps",
                    operation="mean",
                    window_s=True,
                ),
            ),
            "native_tool_arguments_invalid",
        ),
        ((tool("read_current_telemetry", signal_id="private.secret"),), "native_unknown_signal"),
        (
            (tool("read_current_telemetry", signal_id="field.position"),),
            "native_tool_arguments_invalid",
        ),
        (
            (tool("get_current_race_position"), tool("get_current_race_position")),
            "native_duplicate_tool_call",
        ),
        (
            (tool("get_current_race_position"), tool("get_recent_race_position_change")),
            "native_mixed_temporal_scopes",
        ),
    ],
)
def test_invalid_calls_reject_whole_plan_before_telemetry_execution(monkeypatch, calls, reason):
    memory, registry, _, planner = setup(*calls)
    monkeypatch.setattr(
        registry, "execute", lambda *_: pytest.fail("Invalid plan must not execute")
    )
    context = QueryDrivenContextEngineer(memory, planner, registry)
    with pytest.raises(ContextPlanRejection, match=reason):
        asyncio.run(context.analyze(driver_turn()))


def test_native_capability_results_keep_temporal_metadata_and_refresh_live_values():
    memory, registry, _, planner = setup(tool("get_current_race_position"))
    context = QueryDrivenContextEngineer(memory, planner, registry)

    async def run():
        packet = await context.analyze(driver_turn())
        assert packet.temporal_scope == "current"
        assert all(item.temporal_scope == "current" for item in packet.evidence)
        assert (
            next(
                e for e in packet.evidence if e.metric == "position" and e.subject == "player"
            ).value
            == 5
        )
        memory.update(race_context(3, 101, position=4, opponent_positions=(1, 6)))
        fresh = await context.refresh(packet, tuple(e.evidence_id for e in packet.evidence))
        assert fresh.source_sequence == 3
        assert (
            next(
                e for e in fresh.evidence if e.metric == "position" and e.subject == "player"
            ).value
            == 4
        )

    asyncio.run(run())


def test_human_fact_view_keeps_original_references_and_live_refresh():
    from test_portable_engineer import FakeJsonModel

    from race_engineer.intelligence.grounding import StrictEvidenceGrounder
    from race_engineer.intelligence.orchestrator import EngineerOrchestrator
    from race_engineer.intelligence.portable_engineer import PortableQwenEngineer

    memory, registry, _, planner = setup(tool("get_current_race_position"))
    context = QueryDrivenContextEngineer(memory, planner, registry)
    model = FakeJsonModel({"goal": "inform", "speech": "We're P{{position}}."})
    core = PortableQwenEngineer(ConversationConfig(context_planner=LFM_NATIVE_ID), model=model)
    engineer = EngineerOrchestrator(context, core, core, StrictEvidenceGrounder())

    async def run():
        prepared = await engineer.prepare(driver_turn())
        wire = json.loads(model.calls[0]["content"])
        assert wire["verified_facts"][0]["value"] == 5
        assert wire["verified_facts"][0]["meaning"] == "your current race position"
        assert wire["verified_facts"][0]["say"] == "{{position}}"
        assert "c1:player_position" not in model.calls[0]["content"]
        assert "classified_cars" not in model.calls[0]["content"]
        assert prepared.response.references[0].evidence_id == "c1:player_position"
        memory.update(race_context(3, 101, position=4, opponent_positions=(1, 6)))
        result = await engineer.ground(prepared)
        assert result.text == "We're P4."
        assert result.evidence_ids == ("c1:player_position",)

    asyncio.run(run())


def test_native_factory_runs_selection_execution_draft_and_grounding_on_same_model(monkeypatch):
    config = load_config(ROOT / "config/default.toml", environ={}).conversation
    memory = BoundedTelemetryMemory()
    contexts = _contexts(
        load_fixture(ROOT / "fixtures/synthetic/intelligence-capabilities"),
        AppConfig().policy.context,
    )
    for current in contexts:
        memory.update(current)
    selection = FakeConnection(
        raw=json.dumps(
            {
                "choices": [
                    {
                        "finish_reason": "tool_calls",
                        "message": {
                            "tool_calls": [
                                {
                                    "id": "position-call",
                                    "type": "function",
                                    "function": {
                                        "name": "get_current_race_position",
                                        "arguments": "{}",
                                    },
                                }
                            ]
                        },
                    }
                ]
            }
        ).encode()
    )
    reply = FakeConnection(
        raw=payload(json.dumps({"goal": "inform", "speech": "You're P{{position}}."}))
    )
    connections = iter((selection, reply))
    endpoints = []

    def connection(host, port, timeout):
        endpoints.append((host, port))
        return next(connections)

    monkeypatch.setattr("http.client.HTTPConnection", connection)
    frame = memory.latest_context().frame
    turn = DriverTurn(
        turn_id="native-full",
        transcript="Track position",
        received_at=frame.observed_at,
        session_id=frame.session_id,
        generation=0,
        asr_language="en",
        reply_language="en",
    )
    result = asyncio.run(live_intelligence(config, memory).respond(turn))
    assert result.evidence_ids and "5" in result.text
    assert endpoints == [("127.0.0.1", config.port)] * 2
    native_body, json_body = (c.requests[0][2] for c in (selection, reply))
    assert native_body["model"] == json_body["model"] == config.model
    assert native_body["tool_choice"] == "required" and "response_format" not in native_body
    assert json_body["response_format"]["type"] == "json_object" and "tools" not in json_body
    assert json_body["max_tokens"] == 320
    prompt = json_body["messages"][0]["content"]
    assert "JSON object" in prompt and "compose_engineer_reply" not in prompt
    schema = json_body["response_format"]["schema"]
    assert schema["properties"]["goal"]["const"] == "inform"
    assert schema["properties"]["speech"]["pattern"] == r'^[^0-9"\\\r\n]+$'
    assert selection.closed and reply.closed


def test_native_profile_uses_the_mvp_model_and_qwen_rollback_is_explicit():
    import tomllib

    source = (ROOT / "config/default.toml").read_text(encoding="utf-8")
    profile = AppConfig.model_validate(tomllib.loads(trial_profile(source, "gpu", native=True)))
    assert profile.conversation.context_planner == LFM_NATIVE_ID
    assert profile.conversation.port == 8092
    assert (
        AppConfig.model_validate(tomllib.loads(source)).conversation.context_planner
        == LFM_NATIVE_ID
    )
    rollback = load_config(ROOT / "config/qwen-legacy.toml", environ={})
    assert rollback.conversation.context_planner == "qwen-context-v7"
    assert rollback.conversation.model == "Qwen3-4B-Instruct-2507"


def test_native_trial_rejects_unvalidated_turkish_explicitly():
    memory, registry, model, planner = setup(tool("get_current_race_position"))
    with pytest.raises(LocalIntelligenceError, match="lfm_native_english_only"):
        plan(planner, memory, registry, driver_turn().model_copy(update={"asr_language": "tr"}))
    assert not model.requests


def test_native_reply_cannot_bypass_number_grounding_or_claim_unavailable_data():
    from test_portable_engineer import FakeJsonModel, fuel_item, packet, turn

    from race_engineer.intelligence.portable_engineer import PortableQwenEngineer

    config = ConversationConfig(context_planner=LFM_NATIVE_ID)
    wrong = FakeJsonModel({"goal": "inform", "speech": "Fuel {{a}}, plus 99 liters."})
    core = PortableQwenEngineer(config, model=wrong)
    with pytest.raises(LocalIntelligenceError, match="engineer_model_response_invalid"):
        asyncio.run(core.decide(turn("Fuel?"), packet(fuel_item())))
    assert len(wrong.calls) == 2  # Even a provider bypass cannot sneak in literal numbers.

    missing = packet().model_copy(
        update={
            "planner_id": LFM_NATIVE_ID,
            "temporal_scope": "historical",
            "unknowns": ("requested_analysis_unavailable",),
        }
    )
    brief = asyncio.run(core.decide(turn("Original starting position?"), missing))
    response = asyncio.run(core.generate(turn("Original starting position?"), missing, brief))
    assert brief.unknown_ids == missing.unknowns
    assert response.speech_template == "{{unavailable}}" and not response.references
    assert len(wrong.calls) == 3  # Invalid framing falls back to an application-owned limitation.
