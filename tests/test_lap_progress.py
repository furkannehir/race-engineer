"""Lap duration and race distance stay separate from fuel endurance."""

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_iracing_source import FakeSdk
from test_lfm3_dialogue import call
from test_ministral_telemetry import ask, setup
from test_race_capabilities import context, registry, request

from race_engineer.core.enums import SessionPhase
from race_engineer.intelligence.capabilities import DeterministicRaceCapabilities
from race_engineer.intelligence.telemetry_memory import BoundedTelemetryMemory
from race_engineer.telemetry.iracing import normalize_sample
from race_engineer.telemetry.iracing.raw import load_raw_samples
from race_engineer.telemetry.iracing.source import PyIrSdkSource
from scripts.lfm3_dialogue import ChatReply

ROOT = Path(__file__).parents[1]
VARIABLES = {
    "LapCompleted": 6, "LapLastLapTime": 92.5, "LapBestLapTime": 91.25,
    "SessionLapsTotal": 16, "SessionLapsRemainEx": 10,
    "SessionTimeRemain": 900.0, "SessionTimeTotal": 1800.0,
}


def race(*, player=None, frame=None, burn=2):
    base = context()
    return base.model_copy(update={
        "frame": base.frame.model_copy(update={
            "session_type": "Race", "session_phase": SessionPhase.GREEN,
            "session_laps_total": 16, "session_laps_remaining": 10,
            "session_time_remaining_s": 900.0,
            "player": base.frame.player.model_copy(update={
                "lap_number": 7, "laps_completed": 6, "fuel_l": 24,
                "last_lap_time_s": 92.5, "best_lap_time_s": 91.25, **(player or {}),
            }), **(frame or {}),
        }), "fuel_trend_l_per_lap": burn,
    })


def values(result):
    return {item.metric: item.value for item in result.evidence}


def demo(*replies):
    app, memory, model = setup(*replies)
    memory.clear()
    memory.update(race())
    return app, memory, model


def test_sdk_captures_lap_durations_and_session_limits_without_retaining_driver_names(monkeypatch):
    sdk = FakeSdk()
    sdk.var_headers_names.extend(VARIABLES)
    sdk.values.update(VARIABLES)
    monkeypatch.setattr("race_engineer.telemetry.iracing.source.importlib.import_module",
                        lambda _: SimpleNamespace(IRSDK=lambda: sdk))
    source = PyIrSdkSource()
    sample = source.read().sample
    assert sample.laps_completed == 6 and sample.last_lap_time_s == 92.5
    assert sample.best_lap_time_s == 91.25
    assert sample.session_laps_total == 16 and sample.session_laps_remaining == 10
    assert sample.session_time_remaining_s == 900 and sample.session_time_total_s == 1800
    assert set(VARIABLES) <= set(sample.available_variables)
    assert "Must Not Be Retained" not in sample.model_dump_json()
    assert sdk.freeze_calls == sdk.unfreeze_calls == 1
    source.close()


def test_normalization_separates_started_completed_laps_and_filters_sdk_sentinels():
    base = load_raw_samples(ROOT / "fixtures/iracing/m1_sample/raw_samples.jsonl")[0]
    raw = base.model_copy(update={
        "available_variables": tuple(sorted(set(base.available_variables) | set(VARIABLES))),
        "lap": 7, "laps_completed": 6, "last_lap_time_s": 92.5, "best_lap_time_s": 91.25,
        "session_laps_total": 16, "session_laps_remaining": 10,
        "session_time_remaining_s": 0, "session_time_total_s": 1800,
    })
    frame = normalize_sample(raw)
    assert frame is not None and frame.player.lap_number == 7
    assert frame.player.laps_completed == 6 and frame.player.last_lap_time_s == 92.5
    assert frame.session_laps_remaining == 10 and frame.session_time_remaining_s == 0
    missing = normalize_sample(raw.model_copy(update={
        "last_lap_time_s": -1, "best_lap_time_s": 0, "laps_completed": -1,
        "session_laps_total": 32767, "session_laps_remaining": 32767,
        "session_time_remaining_s": 604700, "session_time_total_s": 604800,
    }))
    assert missing is not None
    assert missing.player.last_lap_time_s is missing.player.best_lap_time_s is None
    assert missing.player.laps_completed is None
    assert missing.session_laps_total is missing.session_laps_remaining is None
    assert missing.session_time_remaining_s is missing.session_time_total_s is None


def test_lap_duration_and_last_vs_best_math_have_independent_provenance():
    core = registry(race())
    assert values(core.execute(request("last_lap_time"))) == {"last_lap_time_s": 92.5}
    assert values(core.execute(request("best_lap_time"))) == {"best_lap_time_s": 91.25}
    result = core.execute(request("lap_time_comparison"))
    assert values(result) == {"difference_s": 1.25, "comparison": "slower_than_best"}
    assert result.evidence[0].source_fields == ("player.last_lap_time_s", "player.best_lap_time_s")


def test_missing_best_lap_does_not_hide_last_lap_or_invent_a_zero_best():
    core = registry(race(player={"best_lap_time_s": None}))
    assert core.execute(request("last_lap_time")).status == "available"
    for capability in ("best_lap_time", "lap_time_comparison"):
        result = core.execute(request(capability))
        assert result.status == "unavailable"
        assert all(item.value is None for item in result.evidence)


def test_remaining_race_laps_and_fuel_endurance_are_distinct_calculations():
    core = registry(race())
    progress = values(core.execute(request("race_progress")))
    endurance = values(core.execute(request("fuel_range")))
    assert progress["current_lap"] == 7 and progress["laps_completed"] == 6
    assert progress["total_laps"] == 16 and progress["race_laps_remaining"] == 10
    assert endurance["fuel_laps_remaining"] == 12


@pytest.mark.parametrize("fuel,enough,margin", [(24, True, 2), (22, True, 0), (20, False, -2)])
def test_fuel_to_finish_includes_current_lap_and_explicit_reserve(fuel, enough, margin):
    result = registry(race(player={"fuel_l": fuel})).execute(request("fuel_to_finish"))
    facts = values(result)
    assert facts["scheduled_laps_remaining"] == 10 and facts["reserve_laps"] == 1
    assert facts["fuel_required_l"] == 22 and facts["fuel_margin_l"] == margin
    assert facts["enough_fuel"] is enough
    assert facts["fuel_shortfall_l"] == max(0, -margin)
    assert facts["fuel_surplus_l"] == max(0, margin)


def test_lapped_driver_fuel_budget_does_not_silently_adopt_leader_remaining_laps():
    result = registry(race(frame={"session_laps_remaining": 8})).execute(request("fuel_to_finish"))
    assert values(result)["scheduled_laps_remaining"] == 10


def test_timed_race_reports_clock_without_fabricating_laps_or_fuel_sufficiency():
    core = registry(race(frame={"session_laps_total": None, "session_laps_remaining": None}))
    progress = values(core.execute(request("race_progress")))
    assert progress["time_remaining_s"] == 900 and "race_laps_remaining" not in progress
    assert "total_laps" not in progress
    result = core.execute(request("fuel_to_finish"))
    assert result.status == "unavailable"
    assert result.unavailable_reason == "timed_or_unknown_race_distance"


@pytest.mark.parametrize("kwargs", [
    {"frame": {"session_type": "Practice"}}, {"burn": None}, {"burn": 0},
    {"player": {"laps_completed": None}}, {"player": {"laps_completed": 16}},
    {"frame": {"session_phase": SessionPhase.CHECKERED}},
])
def test_fuel_estimate_needs_active_race_distance_completed_count_and_positive_burn(kwargs):
    result = registry(race(**kwargs)).execute(request("fuel_to_finish"))
    assert result.status == "unavailable"
    assert all(item.kind == "unknown" for item in result.evidence)


@pytest.mark.parametrize("tool,speech,expected", [
    ("get_last_lap_time", "Last lap {{last_lap_time_s}}.", "Last lap one thirty two point five."),
    ("get_best_lap_time", "Session best {{best_lap_time_s}}.",
     "Session best one thirty one point two five."),
    ("compare_last_lap_to_best", "Last lap {{comparison}} by {{difference_s}}.",
     "Last lap slower than best by 1.25 seconds."),
    ("get_race_progress", "{{race_laps_remaining}} remaining in the race.",
     "10 laps remaining in the race."),
    ("get_remaining_race_laps", "{{race_laps_remaining}} left.", "10 laps left."),
    ("get_completed_lap_count", "{{laps_completed}} completed.", "6 laps completed."),
    ("get_total_race_laps", "Race distance {{total_laps}}.", "Race distance 16 laps."),
    ("get_remaining_race_time", "{{time_remaining_s}} remaining.", "900 seconds remaining."),
    ("get_last_lap_time", "Last lap 92.5 seconds per lap.", "Last lap one thirty two point five."),
    ("get_fuel_range", "{{laps_remaining}} remaining.", "12 laps of fuel remaining."),
    ("get_fuel_to_finish", "{{enough_fuel}}, allowing {{reserve_laps}} reserve.",
     "Yes, allowing 1 lap reserve."),
])
def test_ministral_tools_use_core_facts_not_model_arithmetic(tool, speech, expected):
    app, _, _ = demo(ChatReply(None, (call(tool),)), ChatReply(speech))
    assert asyncio.run(app.respond(ask(app, "Give me that reading"))).text == expected


def test_repeated_last_lap_samples_are_not_treated_as_clean_lap_history():
    app, _, _ = demo()
    messages, evidence, unknowns = app._tools((call(
        "query_telemetry_history", source="player", signal="last_lap_time_s",
        operation="mean", window_s=30,
    ),))
    assert not evidence and unknowns == ("completed_lap_history_required",)
    assert json.loads(messages[0]["content"])["status"] == "unavailable"


def test_old_recording_without_new_fields_returns_unavailable_not_fake_timing():
    memory = BoundedTelemetryMemory()
    memory.update(context())
    core = DeterministicRaceCapabilities(memory)
    assert core.execute(request("last_lap_time")).status == "unavailable"
    assert core.execute(request("race_progress")).status == "unavailable"


def test_focused_remaining_laps_cannot_leak_current_or_completed_lap_number():
    app, _, _ = demo()
    _, evidence, _ = app._tools((call("get_remaining_race_laps"),))
    assert len(evidence) == 1 and evidence[0].metric == "race_laps_remaining"
    assert evidence[0].value == 10


def test_timed_lap_count_tool_is_unavailable_instead_of_returning_current_lap():
    app, memory, _ = demo()
    memory.clear()
    memory.update(race(frame={"session_laps_total": None, "session_laps_remaining": None}))
    messages, evidence, unknowns = app._tools((call("get_remaining_race_laps"),))
    assert not evidence and unknowns == ("race_progress:missing_race_laps_remaining",)
    assert json.loads(messages[0]["content"])["status"] == "unavailable"


def test_shortfall_is_positive_in_speech_without_model_authored_absolute_value():
    app, memory, _ = demo(
        ChatReply(None, (call("get_fuel_to_finish"),)),
        ChatReply("{{enough_fuel}}, short by 17 litres allowing {{reserve_laps}} reserve."),
    )
    memory.clear()
    memory.update(race(player={"fuel_l": 5}))
    result = asyncio.run(app.respond(ask(app, "Enough fuel?")))
    assert result.text == "No, short by 17 litres allowing 1 lap reserve."


def test_fuel_range_cannot_claim_sufficiency_without_the_finish_calculation():
    app, _, _ = demo(
        ChatReply(None, (call("get_fuel_range"),)),
        ChatReply("You have 12 laps remaining. That's enough for 24 litres, you're good to go."),
        ChatReply(None, (call("get_fuel_to_finish"),)),
        ChatReply("{{enough_fuel}}, allowing {{reserve_laps}} reserve at current consumption."),
    )
    result = asyncio.run(app.respond(ask(app, "Is the fuel enough?")))
    assert result.text == "Yes, allowing 1 lap reserve at current consumption."
    assert any(entry.get("reason") == "dialogue_fuel_sufficiency_requires_calculation"
               for entry in app.last_trace)


def test_fuel_verdict_is_repaired_when_fuel_changes_during_wording():
    app, memory, model = demo()

    def fuel_changed():
        memory.update(race(player={"fuel_l": 20}, frame={"sequence": 43,
                      "session_time_s": 121}))
        return ChatReply("Yes, allowing {{reserve_laps}} reserve.")

    model.replies = iter((ChatReply(None, (call("get_fuel_to_finish"),)), fuel_changed,
                          ChatReply("No, allowing {{reserve_laps}} reserve.")))
    result = asyncio.run(app.respond(ask(app, "Enough to finish?")))
    assert result.text == "No, allowing 1 lap reserve." and result.source_sequence == 43
