"""Existing data/calculations exposed through the checked native dialogue path."""

import asyncio
import json
from pathlib import Path

import pytest
from test_lfm3_dialogue import Model, call
from test_ministral_dialogue import turn
from test_race_capabilities import context

from race_engineer.config import ConversationConfig, PolicyContextConfig
from race_engineer.core.enums import RaceFlag, SessionPhase
from race_engineer.fixtures import load_fixture
from race_engineer.intelligence.local_model import LocalIntelligenceError
from race_engineer.intelligence.telemetry_memory import BoundedTelemetryMemory
from race_engineer.policy import DefaultRaceContextBuilder
from scripts.lfm3_dialogue import ChatReply
from scripts.ministral_dialogue import MinistralDialogue
from scripts.ministral_telemetry import CAPABILITY_TOOLS, ReadOnlyRaceTools

ROOT = Path(__file__).parents[1]


def setup(*replies):
    memory = BoundedTelemetryMemory()
    builder = DefaultRaceContextBuilder(PolicyContextConfig())
    for frame in load_fixture(ROOT / "fixtures/synthetic/intelligence-capabilities").frames:
        memory.update(builder.update(frame, ()))
    model = Model(*replies)
    app = MinistralDialogue(ConversationConfig(), memory, model=model)
    return app, memory, model


def ask(app, question):
    return turn(question).model_copy(update={
        "session_id": app.memory.latest_context().frame.session_id,
    })


def test_real_fixture_exposes_implemented_calculations_but_no_strategy_or_action_tools():
    app, _, _ = setup()
    names = {tool["function"]["name"] for tool in app.read_tools.definitions()}
    assert set(CAPABILITY_TOOLS) <= names
    assert {"read_current_telemetry", "query_telemetry_history"} <= names
    assert not any("pit" in name or "set_" in name or "project" in name for name in names)


@pytest.mark.parametrize("tool,speech,expected", [
    ("get_fuel_range", "Fuel range is {{laps_remaining}} at {{context_fuel_trend_l_per_lap}}.",
     "Fuel range is 12 laps of fuel at 2 litres per lap."),
    ("get_recent_position_change", "{{direction}} {{positions_changed}} recently.",
     "gained 3 positions recently."),
    ("get_relative_pace_ahead", "We are {{state}}.", "We are closing."),
    ("get_relative_pace_behind", "The car behind is {{state}}.",
     "The car behind is falling back."),
    ("get_catch_time_ahead", "Estimated catch time is {{catch_time_s}} at this closing rate.",
     "Estimated catch time is 20 seconds at this closing rate."),
    ("get_race_classification",
     "{{is_last}}, you're {{player_position}}; last is {{last_position}}.",
     "No, you're P5; last is P6."),
])
def test_calculations_use_existing_core_and_refreshable_unit_bearing_values(tool, speech, expected):
    app, _, model = setup(ChatReply(None, (call(tool),)), ChatReply(speech))
    answer = asyncio.run(app.respond(ask(app, "Give me that race reading")))
    assert answer.text == expected and answer.evidence_ids
    result = json.loads(model.requests[1][0][-1]["content"])
    assert all(fact["reference"].startswith("{{") for fact in result["facts"])
    assert len(model.requests) == 2


def test_catch_time_behind_is_exposed_and_remains_unavailable_when_car_is_not_closing():
    app, _, model = setup(ChatReply(None, (call("get_catch_time_behind"),)))
    answer = asyncio.run(app.respond(ask(app, "When will they catch us?")))
    assert not answer.evidence_ids and "data" in answer.text
    assert len(model.requests) == 1


@pytest.mark.parametrize("signals,speech,expected", [
    (["player.fuel_l"], "Fuel is {{player_fuel_l}}.", "Fuel is 24 litres."),
    (["player.lap_number"], "We're on lap {{player_lap_number}}.", "We're on lap 7."),
    (["context.stint_lap"], "Stint lap {{context_stint_lap}}.", "Stint lap 4."),
    (["frame.session_phase"], "Session phase: {{frame_session_phase}}.", "Session phase: green."),
    (["opponent.position"], "That car is {{opponent_position}}.", "That car is P4."),
    (["player.fuel_l", "player.lap_number"],
     "Fuel {{player_fuel_l}}, lap {{player_lap_number}}.",
     "Fuel 24 litres, lap 7."),
])
def test_raw_adapter_values_are_reachable_without_a_new_calculation(signals, speech, expected):
    app, _, _ = setup(
        ChatReply(None, (call("read_current_telemetry", signals=signals),)), ChatReply(speech),
    )
    assert asyncio.run(app.respond(ask(app, "Read these values"))).text == expected


def test_flag_and_pit_boolean_reads_distinguish_missing_channels_from_false():
    app, memory, _ = setup()
    operation = call("read_current_telemetry", signals=["frame.flags", "player.in_pit_lane"])
    _, evidence, unknowns = app._tools((operation,))
    assert not evidence and len(unknowns) == 2
    latest = memory.latest_context()
    frame = latest.frame.model_copy(update={
        "sequence": 204, "session_time_s": 131,
        "capabilities": tuple(sorted({*latest.frame.capabilities, "flags", "pit"})),
        "flags": (RaceFlag.YELLOW,),
    })
    memory.update(latest.model_copy(update={"frame": frame}))
    _, evidence, unknowns = app._tools((operation,))
    assert not unknowns
    assert [item.claim for item in evidence] == ["yellow", "No"]


def test_window_queries_use_bounded_core_memory_and_field_count_is_not_grid_size():
    app, _, _ = setup()
    _, values, unknowns = app._tools((call(
        "query_telemetry_history", source="player", signal="position",
        operation="delta", window_s=30,
    ),))
    assert not unknowns and values[0].value == -3
    _, values, _ = app._tools((call(
        "query_telemetry_history", source="field", signal="position", operation="count",
    ),))
    assert values[0].value == 3 and values[0].claim == "3 cars with reported positions"


def test_missing_history_is_an_explicit_limitation_without_a_second_model_call():
    app, _, model = setup(ChatReply(None, (call(
        "query_telemetry_history", source="player", signal="fuel_l",
        operation="mean", window_s=120,
    ),)))
    answer = asyncio.run(app.respond(ask(app, "Average fuel level over two minutes?")))
    assert not answer.evidence_ids and len(model.requests) == 1


def test_missing_burn_does_not_hide_known_fuel_or_invent_a_range():
    app, memory, _ = setup(
        ChatReply(None, (call("get_fuel_range"),)), ChatReply("Fuel is {{player_fuel_l}}."),
    )
    memory.clear()
    memory.update(context(fuel_burn_l_per_lap=None))
    answer = asyncio.run(app.respond(ask(app, "Fuel situation?")))
    assert answer.text.startswith("Fuel is 20 litres.") and "data" in answer.text
    assert "laps" not in answer.text


def test_numeric_and_categorical_values_are_refreshed_after_generation():
    app, memory, model = setup()

    def later_reply():
        current = memory.latest_context()
        frame = current.frame.model_copy(update={
            "sequence": 204, "session_time_s": 131, "session_phase": SessionPhase.CAUTION,
            "player": current.frame.player.model_copy(update={"fuel_l": 22}),
        })
        memory.update(current.model_copy(update={"frame": frame}))
        return ChatReply("Fuel {{player_fuel_l}}. Phase {{frame_session_phase}}.")

    model.replies = iter((ChatReply(None, (call(
        "read_current_telemetry", signals=["player.fuel_l", "frame.session_phase"],
    ),)), later_reply))
    answer = asyncio.run(app.respond(ask(app, "Fuel and phase?")))
    assert answer.text == "Fuel 22 litres. Phase caution." and answer.source_sequence == 204


def test_new_readings_have_verified_recall_without_copying_numbers_into_chat_history():
    app, memory, _ = setup(
        ChatReply(None, (call("read_current_telemetry", signals=["player.fuel_l"]),)),
        ChatReply("Fuel is {{player_fuel_l}}."),
        ChatReply(None, (call("recall_radio_reading"),)),
        ChatReply("The earlier reading was {{player_fuel_l}}."),
    )

    async def run():
        first = await app.respond(ask(app, "Fuel?"))
        assert "24" not in json.dumps(list(app.history))
        current = memory.latest_context()
        memory.update(current.model_copy(update={"frame": current.frame.model_copy(update={
            "sequence": 204, "session_time_s": 131,
            "player": current.frame.player.model_copy(update={"fuel_l": 22}),
        })}))
        return first, await app.respond(ask(app, "What did you say?"))

    first, recalled = asyncio.run(run())
    assert first.text == "Fuel is 24 litres."
    assert recalled.text == "The earlier reading was 24 litres." and recalled.source_sequence == 203
    app.clear()
    assert not app._delivered_extended


@pytest.mark.parametrize("tool,speech,expected", [
    ("get_fuel_range", "Fuel is 24 litres, burning 2 litres per lap.",
     "Fuel is 24 litres, burning 2 litres per lap."),
    ("get_fuel_range", "Twelve laps of fuel.", "12 laps of fuel."),
    ("get_fuel_range", "You have {{laps_remaining}} laps left.", "You have 12 laps of fuel left."),
    ("get_recent_position_change", "We gained 3 positions in the last 30 seconds.",
     "We gained 3 positions in the last 30 seconds."),
    ("get_catch_time_ahead", "Twenty seconds at this rate.", "20 seconds at this rate."),
    ("check_if_driver_is_first", "You are P5. No, you are not first.",
     "You are P5. No, you are not first."),
])
def test_actual_model_number_formats_use_the_right_units_and_scope(tool, speech, expected):
    app, _, _ = setup(ChatReply(None, (call(tool),)), ChatReply(speech))
    assert asyncio.run(app.respond(ask(app, "Read the requested fact"))).text == expected


def test_ambiguous_equal_values_with_the_same_unit_require_explicit_references():
    app, memory, _ = setup()
    current = memory.latest_context()
    memory.update(current.model_copy(update={
        "frame": current.frame.model_copy(update={"sequence": 204, "session_time_s": 131}),
        "gap_ahead_s": 2.0, "gap_behind_s": 2.0,
    }))
    _, evidence, _ = app._tools((call("read_current_telemetry", signals=[
        "context.gap_ahead_s", "context.gap_behind_s",
    ]),))
    with pytest.raises(LocalIntelligenceError, match="ambiguous_reading"):
        app.bind_reading("The gap is 2 seconds.", evidence, language="en")


def test_wording_sees_only_current_tool_while_selection_retains_dialogue_context():
    app, _, model = setup(
        ChatReply(None, (call("read_current_telemetry", signals=["player.fuel_l"]),)),
        ChatReply("Fuel is 24 litres."),
        ChatReply(None, (call("read_current_telemetry", signals=["player.lap_number"]),)),
        ChatReply("Lap 7."),
    )

    async def run():
        await app.respond(ask(app, "Fuel?"))
        return await app.respond(ask(app, "Lap?"))

    assert asyncio.run(run()).text == "Lap 7."
    selection, wording = model.requests[2][0], model.requests[3][0]
    assert any(item.get("content") == "Fuel?" for item in selection)
    assert not any(item.get("content") == "Fuel?" for item in wording)
    assert not any("player_fuel_l" in (item.get("content") or "") for item in wording)


def test_boolean_change_rewords_against_fresh_values_instead_of_saying_yes_not_in_pits():
    app, memory, model = setup()
    latest = memory.latest_context()
    memory.update(latest.model_copy(update={"frame": latest.frame.model_copy(update={
        "sequence": 204, "capabilities": tuple(sorted({*latest.frame.capabilities, "pit"})),
    })}))

    def crossed_pit_entry():
        latest = memory.latest_context()
        memory.update(latest.model_copy(update={"frame": latest.frame.model_copy(update={
            "sequence": 205, "session_time_s": 131,
            "player": latest.frame.player.model_copy(update={"in_pit_lane": True}),
        })}))
        return ChatReply("No, we are not in the pit lane.")

    model.replies = iter((ChatReply(None, (call(
        "read_current_telemetry", signals=["player.in_pit_lane"],
    ),)), crossed_pit_entry, ChatReply("Yes, we are in the pit lane.")))
    answer = asyncio.run(app.respond(ask(app, "Are we in the pit lane?")))
    assert answer.text == "Yes, we are in the pit lane." and answer.source_sequence == 205
    repaired = json.loads(model.requests[2][0][-1]["content"])
    assert repaired["facts"][0]["value"] is True


@pytest.mark.parametrize("tool", ["get_relative_pace_ahead", "get_relative_pace_behind"])
def test_relative_pace_never_exposes_its_internal_numerical_rate(tool):
    app, _, _ = setup()
    messages, evidence, _ = app._tools((call(tool),))
    assert {item.metric for item in evidence} == {"relative_pace_state", "is_closing"}
    assert "gap_change_rate" not in json.dumps(messages)
    assert all(item.unit != "s/s" for item in evidence)


@pytest.mark.parametrize("source,signal", [
    ("context", "gap_ahead_s"), ("context", "gap_behind_s"),
    ("opponent", "gap_to_player_s"),
])
def test_seconds_window_gap_change_cannot_masquerade_as_a_lap_comparison(source, signal):
    app, _, model = setup(ChatReply(None, (call(
        "query_telemetry_history", source=source, signal=signal, operation="delta", window_s=30,
    ),)))
    answer = asyncio.run(app.respond(ask(app, "How much are we gaining per lap?")))
    assert not answer.evidence_ids and "data" in answer.text
    assert len(model.requests) == 1
    assert "lap_based_gap_change_unavailable" in json.dumps(app.last_trace)


def test_speed_and_derivative_queries_are_not_offered_to_the_model():
    app, _, _ = setup()
    definitions = app.read_tools.definitions()
    current = next(tool for tool in definitions
                   if tool["function"]["name"] == "read_current_telemetry")
    history = next(tool for tool in definitions
                   if tool["function"]["name"] == "query_telemetry_history")
    assert "player.speed_mps" not in current["function"]["parameters"]["properties"][
        "signals"]["items"]["enum"]
    assert "speed_mps" not in history["function"]["parameters"]["properties"]["signal"]["enum"]
    assert "trend" not in history["function"]["parameters"]["properties"]["operation"]["enum"]


@pytest.mark.parametrize("suffix", [
    " per second.", " per lap.", "/s.", "/lap.", " each lap.", " per completed lap.",
    ". Speed is in metres per second.", ". Speed is in meters per second.",
    ". Speed is in m/s.", ". Speed is in km/h.",
])
def test_rendered_speech_cannot_relabel_a_gap_or_add_speed_units(suffix):
    app, _, _ = setup(
        ChatReply(None, (call("read_current_telemetry", signals=["context.gap_ahead_s"]),)),
        ChatReply("Gap {{context_gap_ahead_s}}" + suffix),
        ChatReply("Gap {{context_gap_ahead_s}}."),
    )
    answer = asyncio.run(app.respond(ask(app, "Gap ahead?")))
    assert answer.text == "Gap 1 seconds."
    assert any(entry["stage"] == "validation" for entry in app.last_trace)


def test_invented_numerical_gap_rate_cannot_be_spoken_or_recalled():
    app, _, _ = setup(
        ChatReply(None, (call("get_relative_pace_ahead"),)),
        ChatReply("We are {{state}} at 0.05 seconds per second."),
        ChatReply("We are {{state}} at 0.05 seconds per second."),
    )
    with pytest.raises(LocalIntelligenceError, match="dialogue_reply_invalid"):
        asyncio.run(app.respond(ask(app, "Are we catching?")))
    assert not app.history and not app._delivered_extended


def test_position_change_cannot_be_presented_as_speed_history():
    app, _, _ = setup(
        ChatReply(None, (call("get_recent_position_change"),)),
        ChatReply("Our speed has been gaining positions over the last 30 seconds."),
        ChatReply(None, (call("report_unavailable_information", quantity="speed history"),)),
    )
    answer = asyncio.run(app.respond(ask(app, "How has our speed changed?")))
    assert not answer.evidence_ids and "data" in answer.text
    assert not app._delivered_extended


def test_relative_pace_yes_no_cannot_disagree_with_core_closing_state():
    app, _, _ = setup(
        ChatReply(None, (call("get_relative_pace_behind"),)),
        ChatReply("Yes, the car behind is falling back."),
        ChatReply("No, the car behind is falling back."),
    )
    answer = asyncio.run(app.respond(ask(app, "Is the car behind catching us?")))
    assert answer.text == "No, the car behind is falling back."


@pytest.mark.parametrize("speech", ["Seven.", "7.", "seven", "yedi"])
def test_one_known_lap_reading_accepts_a_bare_number_without_inferring_another_quantity(speech):
    app, _, _ = setup(
        ChatReply(None, (call("read_current_telemetry", signals=["player.lap_number"]),)),
        ChatReply(speech),
    )
    assert asyncio.run(app.respond(ask(app, "Which lap?"))).text.rstrip(".") == "7"


def test_unknown_or_wrong_bare_number_does_not_borrow_the_only_reading():
    app, _, _ = setup(
        ChatReply(None, (call("read_current_telemetry", signals=["player.lap_number"]),)),
        ChatReply("Sixteen."), ChatReply("Sixteen."),
    )
    with pytest.raises(LocalIntelligenceError, match="dialogue_reply_invalid"):
        asyncio.run(app.respond(ask(app, "Which lap?")))


def test_spelled_decimal_gap_binds_only_the_actual_verified_value():
    app, _, _ = setup(ChatReply(None, (call("get_gap_behind"),)),
                      ChatReply("One point six seconds behind us."))
    assert asyncio.run(app.respond(ask(app, "Gap behind?"))).text == "1.6 seconds behind us."


def test_field_lap_count_does_not_claim_to_count_reported_positions():
    app, _, _ = setup()
    _, evidence, _ = app._tools((call(
        "query_telemetry_history", source="field", signal="lap_number", operation="count",
    ),))
    assert evidence[0].claim == "3 cars with reported lap counts"


def test_recalled_fuel_fragment_has_explicit_past_framing_and_original_provenance():
    app, memory, _ = setup(
        ChatReply(None, (call("read_current_telemetry", signals=["player.fuel_l"]),)),
        ChatReply("24 litres remaining."),
        ChatReply(None, (call("recall_radio_reading", quantity="player.fuel_l"),)),
        ChatReply("24 litres remaining."),
    )

    async def run():
        await app.respond(ask(app, "Fuel level?"))
        current = memory.latest_context()
        memory.update(current.model_copy(update={"frame": current.frame.model_copy(update={
            "sequence": 204, "session_time_s": 131,
            "player": current.frame.player.model_copy(update={"fuel_l": 21}),
        })}))
        return await app.respond(ask(app, "What did you say?"))

    answer = asyncio.run(run())
    assert answer.text == "Last call was 24 litres remaining." and answer.source_sequence == 203


def test_new_fuel_recall_cannot_describe_the_old_value_as_current():
    app, _, _ = setup(
        ChatReply(None, (call("read_current_telemetry", signals=["player.fuel_l"]),)),
        ChatReply("24 litres remaining."),
        ChatReply(None, (call("recall_radio_reading", quantity="player.fuel_l"),)),
        ChatReply("24 litres remaining now."),
        ChatReply(None, (call("recall_radio_reading", quantity="player.fuel_l"),)),
        ChatReply("24 litres remaining now."),
    )

    async def run():
        await app.respond(ask(app, "Fuel level?"))
        with pytest.raises(LocalIntelligenceError, match="dialogue_reply_invalid"):
            await app.respond(ask(app, "What did you say?"))

    asyncio.run(run())
    assert len(app.history) == 1 and "now" not in json.dumps(list(app.history))


def test_catch_estimate_cannot_be_spoken_as_a_current_gap():
    app, _, _ = setup(
        ChatReply(None, (call("get_catch_time_ahead"),)),
        ChatReply("20 seconds ahead at this pace."),
        ChatReply("Estimated catch in 20 seconds if this closing rate continues."),
    )
    answer = asyncio.run(app.respond(ask(app, "When would we catch the car ahead?")))
    assert answer.text == "Estimated catch in 20 seconds if this closing rate continues."


@pytest.mark.parametrize("tool,position,opponents", [
    ("check_if_driver_is_first", 1, (2, 3)),
    ("check_if_driver_is_last", 6, (1, 5)),
])
def test_positive_rank_check_has_one_unambiguous_player_rank(tool, position, opponents):
    app, memory, _ = setup(
        ChatReply(None, (call(tool),)), ChatReply(f"Yes, you're P{position}."),
    )
    memory.clear()
    memory.update(context(position=position, opponent_positions=opponents))
    result = asyncio.run(app.respond(ask(app, "Check the requested rank")))
    assert result.text == f"Yes, you're P{position}."


def test_behind_catch_calculation_is_reachable_when_its_closing_inputs_exist():
    app, memory, _ = setup(
        ChatReply(None, (call("get_catch_time_behind"),)),
        ChatReply("Estimated catch from behind in {{catch_time_s}} at this rate."),
    )
    memory.clear()
    for sequence, time, gap in ((1, 100, 2.2), (2, 110, 2.0), (3, 120, 1.8), (4, 130, 1.6)):
        memory.update(context(sequence=sequence, session_time_s=time, gap_behind_s=gap))
    result = asyncio.run(app.respond(ask(app, "When would the car behind catch us?")))
    assert result.text == "Estimated catch from behind in 80 seconds at this rate."


def test_classification_preserves_core_output_meaning_and_source_not_just_metric_name():
    app, _, _ = setup()
    messages, _, _ = app._tools((call("get_race_classification"),))
    facts = {fact["reference"]: fact for fact in json.loads(messages[0]["content"])["facts"]}
    player, last = facts["{{player_position}}"], facts["{{last_position}}"]
    assert player["subject"] == "player" and last["subject"] == "field"
    assert "current classified position" in player["meaning"]
    assert "current field position" in last["meaning"]
    assert last["source_fields"] == ["field.position"]


@pytest.mark.parametrize("arguments", [
    {"signals": ["FuelLevel"]}, {"signals": ["player.fuel_l"], "set": 20},
    {"signals": ["frame.__dict__"]}, {"signals": ["player.fuel_l"] * 2},
    {"signals": ["opponent.position"], "opponent_position": True},
    {"signals": ["player.speed_mps"]},
])
def test_invalid_or_write_like_arguments_never_execute(arguments):
    app, _, _ = setup()
    with pytest.raises(LocalIntelligenceError, match="invalid_tool_arguments"):
        app._tools((call("read_current_telemetry", **arguments),))


@pytest.mark.parametrize("arguments", [
    {"source": "player", "signal": "speed_mps", "operation": "mean"},
    {"source": "player", "signal": "speed_mps", "operation": "latest", "window_s": 30},
    {"source": "player", "signal": "speed_mps", "operation": "mean", "window_s": 121},
    {"source": "player", "signal": "fuel_l", "operation": "count"},
    {"source": "opponent", "signal": "anything", "operation": "latest"},
    {"source": "player", "signal": "speed_mps", "operation": "latest"},
    {"source": "context", "signal": "gap_ahead_s", "operation": "trend", "window_s": 30},
])
def test_history_uses_the_existing_catalog_and_query_contract(arguments):
    memory = BoundedTelemetryMemory()
    memory.update(context(opponent_positions=()))
    with pytest.raises(LocalIntelligenceError, match="invalid_tool_arguments"):
        ReadOnlyRaceTools(memory).execute(call("query_telemetry_history", **arguments))


@pytest.mark.parametrize("speech", ["We have 999 litres.", "Fuel is twelve litres.",
                                    "Fuel is {{made_up}}."])
def test_unsupported_new_numbers_and_references_are_not_delivered(speech):
    app, _, _ = setup(ChatReply(None, (call("get_fuel_range"),)),
                      ChatReply(speech), ChatReply(speech))
    with pytest.raises(LocalIntelligenceError, match="dialogue_reply_invalid"):
        asyncio.run(app.respond(ask(app, "Fuel situation?")))
    assert not app.history and not app._delivered_extended
