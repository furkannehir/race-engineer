"""Radio lap-time wording preserves the verified duration and refresh/recall scope."""

import asyncio

import pytest
from test_lap_progress import demo, race
from test_lfm3_dialogue import call
from test_ministral_telemetry import ask

from race_engineer.intelligence.local_model import LocalIntelligenceError
from race_engineer.language.lap_times import format_lap_time
from scripts.lfm3_dialogue import ChatReply


@pytest.mark.parametrize("seconds,expected", [
    (92.5, "one thirty two point five"),
    (91.25, "one thirty one point two five"),
    (59.9, "fifty nine point nine"),
    (60, "one zero zero"),
    (65.003, "one oh five point zero zero three"),
    (92.500, "one thirty two point five"),
    (59.9996, "one zero zero"),
    (119.9996, "two zero zero"),
])
def test_radio_lap_time_preserves_fractional_digits_and_minute_carry(seconds, expected):
    assert format_lap_time(seconds) == expected


@pytest.mark.parametrize("seconds", [True, 0, -1, float("nan"), float("inf")])
def test_invalid_durations_never_become_radio_times(seconds):
    with pytest.raises(ValueError):
        format_lap_time(seconds)


def test_turkish_time_does_not_switch_to_english_words():
    assert format_lap_time(92.5, "tr") == "bir otuz iki virgül beş"


@pytest.mark.parametrize("speech", [
    "Last lap {{last_lap_time_s}}.",
    "Last lap 92.5 seconds.",
    "Last lap ninety two point five seconds.",
    "Last lap one thirty two point five.",
    "Last lap one thirty-two point five seconds per lap.",
])
def test_model_wording_is_bound_to_the_known_time_and_formatted_by_application(speech):
    app, _, _ = demo(ChatReply(None, (call("get_last_lap_time"),)), ChatReply(speech))
    answer = asyncio.run(app.respond(ask(app, "Last lap?")))
    assert answer.text == "Last lap one thirty two point five."
    assert answer.evidence_ids


def test_raw_numeric_evidence_and_non_lap_seconds_remain_unchanged():
    app, _, _ = demo()
    _, lap, _ = app._tools((call("get_last_lap_time"),))
    assert lap[0].value == 92.5 and lap[0].unit == "s"
    assert lap[0].claim == "one thirty two point five"
    _, comparison, _ = app._tools((call("compare_last_lap_to_best"),))
    assert comparison[0].value == 1.25 and comparison[0].claim == "1.25 seconds"
    _, clock, _ = app._tools((call("get_remaining_race_time"),))
    assert clock[0].value == 900 and clock[0].claim == "900 seconds"


def test_invented_spoken_lap_time_is_rejected():
    wrong = "Last lap one thirty three point five."
    app, _, _ = demo(ChatReply(None, (call("get_last_lap_time"),)),
                      ChatReply(wrong), ChatReply(wrong))
    with pytest.raises(LocalIntelligenceError, match="dialogue_reply_invalid"):
        asyncio.run(app.respond(ask(app, "Last lap?")))
    assert not app.history


def test_delivery_refresh_formats_new_lap_time_instead_of_preserving_old_words():
    app, memory, model = demo()

    def new_lap():
        memory.update(race(player={"last_lap_time_s": 93.2},
                           frame={"sequence": 43, "session_time_s": 121}))
        return ChatReply("Last lap one thirty two point five.")

    model.replies = iter((ChatReply(None, (call("get_last_lap_time"),)), new_lap))
    answer = asyncio.run(app.respond(ask(app, "Last lap?")))
    assert answer.text == "Last lap one thirty three point two."
    assert answer.source_sequence == 43


def test_recall_keeps_the_formatted_original_time_after_live_value_changes():
    app, memory, _ = demo(
        ChatReply(None, (call("get_last_lap_time"),)), ChatReply("Last lap 92.5 seconds."),
        ChatReply(None, (call("recall_radio_reading", quantity="get_last_lap_time"),)),
        ChatReply("Last call was one thirty two point five."),
    )

    async def run():
        await app.respond(ask(app, "Last lap?"))
        memory.update(race(player={"last_lap_time_s": 93.2},
                           frame={"sequence": 43, "session_time_s": 121}))
        return await app.respond(ask(app, "What did you say that lap was?"))

    answer = asyncio.run(run())
    assert answer.text == "Last call was one thirty two point five."
    assert answer.source_sequence == 42
