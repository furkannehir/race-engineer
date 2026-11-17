import asyncio
import json

import pytest
from test_context_engine import driver_turn, race_context
from test_lfm3_dialogue import Model, add_gaps, call

from race_engineer.config import ConversationConfig
from race_engineer.intelligence.local_model import LocalIntelligenceError
from race_engineer.intelligence.telemetry_memory import BoundedTelemetryMemory
from scripts.lfm3_dialogue import ChatReply
from scripts.ministral_dialogue import MinistralConversation, MinistralDialogue


def setup(*replies):
    config = ConversationConfig(model="mistralai/Ministral-3-3B-Instruct-2512")
    memory = BoundedTelemetryMemory()
    memory.update(race_context(1, 100, position=5, opponent_positions=(1, 6)))
    model = Model(*replies)
    return MinistralDialogue(config, memory, model=model), memory, model


def turn(question, *, language="en", generation=0):
    return driver_turn(question).model_copy(update={
        "reply_language": language, "asr_language": language, "generation": generation,
    })


def test_repeated_position_reads_changed_telemetry_instead_of_delivering_old_history():
    app, memory, model = setup(
        ChatReply(None, (call(),)), ChatReply("P5."),
        ChatReply("Still P5."), ChatReply(None, (call(),)), ChatReply("P4."),
    )

    async def run():
        await app.respond(turn("Position?"))
        memory.update(race_context(2, 101, position=4, opponent_positions=(1, 6)))
        return await app.respond(turn("Position?"))

    answer = asyncio.run(run())
    assert answer.text == "P4." and answer.source_sequence == 2
    assert [choice for _, choice in model.requests] == [
        "required", "none", "required", "required", "none",
    ]
    assert app._delivered["position"].value == 4


def test_clarification_uses_delivered_snapshot_while_new_request_uses_current_gap():
    app, memory, _ = setup(
        ChatReply(None, (call("get_gap_ahead"),)), ChatReply("One second behind the car ahead."),
        ChatReply(None, (call("recall_radio_reading"),)), ChatReply("That call was one second."),
        ChatReply(None, (call("get_gap_ahead"),)), ChatReply("The gap ahead is 3.2 seconds."),
    )
    add_gaps(memory)

    async def run():
        await app.respond(turn("Gap ahead?"))
        add_gaps(memory, sequence=3, ahead=3.2)
        clarification = await app.respond(turn("Was that one second or 1.6?"))
        trace = list(app.last_trace)
        current = await app.respond(turn("And what is the gap ahead now?"))
        return clarification, trace, current

    clarification, trace, current = asyncio.run(run())
    assert clarification.text == "That call was 1 second." and clarification.source_sequence == 2
    assert trace[-1]["temporal_scope"] == "historical"
    assert current.text == "The gap ahead is 3.2 seconds." and current.source_sequence == 3


def test_recall_stores_refreshed_delivered_value_not_original_generation_input():
    app, memory, model = setup()
    add_gaps(memory)

    def later_reply():
        add_gaps(memory, sequence=3, behind=2.2)
        return ChatReply("The gap behind is 1.6 seconds.")

    model.replies = iter((
        ChatReply(None, (call("get_gap_behind"),)), later_reply,
        ChatReply(None, (call("recall_radio_reading"),)), ChatReply("I said 2.2 seconds."),
    ))

    async def run():
        first = await app.respond(turn("Gap behind?"))
        recalled = await app.respond(turn("What did you say?"))
        return first, recalled

    first, recalled = asyncio.run(run())
    assert first.text == "The gap behind is 2.2 seconds."
    assert recalled.text == "I said 2.2 seconds." and recalled.source_sequence == 3


def test_recall_can_select_a_previously_reported_quantity_after_another_call():
    app, memory, _ = setup(
        ChatReply(None, (call("get_gap_ahead"),)), ChatReply("Gap ahead is 1 second."),
        ChatReply(None, (call(),)), ChatReply("P5."),
        ChatReply(None, (call("recall_radio_reading", quantity="gap_ahead"),)),
        ChatReply("The earlier gap ahead was 1 second."),
    )
    add_gaps(memory)

    async def run():
        for question in ("Gap ahead?", "Position?", "What did you say about the gap ahead?"):
            answer = await app.respond(turn(question))
        return answer

    assert asyncio.run(run()).text == "The earlier gap ahead was 1 second."


def test_rejected_draft_and_user_supplied_number_do_not_enter_verified_recall():
    app, _, model = setup(
        ChatReply(None, (call(),)), ChatReply("P99."), ChatReply("P99."),
        ChatReply(None, (call("recall_radio_reading"),)), ChatReply("You said P99."),
    )

    async def run():
        with pytest.raises(LocalIntelligenceError, match="dialogue_reply_invalid"):
            await app.respond(turn("I'm P99. Position?"))
        assert not app._delivered and not app.history
        return await app.respond(turn("Repeat your last measurement"))

    answer = asyncio.run(run())
    assert "99" not in answer.text and not answer.evidence_ids
    assert json.loads(model.requests[-1][0][-1]["content"])["status"] == "unavailable"


@pytest.mark.parametrize("reset", ["clear", "generation", "session"])
def test_recall_cannot_cross_clear_generation_or_session(reset):
    app, memory, _ = setup(
        ChatReply(None, (call(),)), ChatReply("P5."),
        ChatReply(None, (call("recall_radio_reading"),)), ChatReply("No earlier reading."),
    )

    async def run():
        await app.respond(turn("Position?"))
        request = turn("What did you say?")
        if reset == "clear":
            app.clear()
        elif reset == "generation":
            request = request.model_copy(update={"generation": 1})
        else:
            context = race_context(1, 100, position=9).model_copy(update={
                "frame": race_context(1, 100, position=9).frame.model_copy(
                    update={"session_id": "new-session"}
                )
            })
            memory.update(context)
            request = request.model_copy(update={"session_id": "new-session"})
        return await app.respond(request)

    assert not asyncio.run(run()).evidence_ids
    assert not app._delivered


@pytest.mark.parametrize("quantity", [[], {}, "fuel"])
def test_invalid_recall_arguments_are_rejected_before_any_lookup(quantity):
    app, _, _ = setup()
    with pytest.raises(LocalIntelligenceError, match="dialogue_invalid_tool_arguments"):
        app._tools((call("recall_radio_reading", quantity=quantity),))


def test_venting_uses_one_model_request_and_mixed_request_still_fetches_telemetry():
    app, memory, model = setup(
        ChatReply("I hear you. Keep your focus."),
        ChatReply(None, (call("get_gap_behind"),)), ChatReply("Copy. Gap behind is 1.6 seconds."),
    )
    add_gaps(memory)

    async def run():
        vent = await app.respond(turn("He has no idea about racing"))
        assert not vent.evidence_ids and len(model.requests) == 1
        return await app.respond(turn("Dirty move. What's the gap behind?"))

    assert asyncio.run(run()).text == "Copy. Gap behind is 1.6 seconds."
    assert len(model.requests) == 3


def test_unsupported_race_assessment_repairs_without_forcing_a_position_lookup():
    app, _, model = setup(
        ChatReply("Race is steady."), ChatReply("I can't assess the overall race from this data."),
    )
    answer = asyncio.run(app.respond(turn("How's the race going?")))
    assert "can't assess" in answer.text
    assert [choice for _, choice in model.requests] == ["required", "required"]


@pytest.mark.parametrize("language,speech", [
    ("en", "You're P5, fifth place."), ("tr", "P5, beşinci sıradayız."),
])
def test_repeated_position_is_one_verified_binding(language, speech):
    app, _, _ = setup(ChatReply(None, (call(),)), ChatReply(speech))
    answer = asyncio.run(app.respond(turn("Position?", language=language)))
    assert answer.evidence_ids == ("dialogue:player_position",) and answer.language == language


@pytest.mark.parametrize("speech,valid", [
    ("Arkadaki araç 1,6 saniye geride.", True),
    ("Arkadakiyle 1,6 saniye öndeyiz.", True),
    ("Arkadaşın 1,6 saniye önde.", False),
    ("1,6 saniye gerideyiz.", False),
])
def test_turkish_gap_units_and_known_direction_forms(speech, valid):
    app, memory, _ = setup(
        ChatReply(None, (call("get_gap_behind"),)), ChatReply(speech), ChatReply(speech),
    )
    add_gaps(memory)
    if valid:
        answer = asyncio.run(app.respond(turn("Arkayla fark?", language="tr")))
        assert "1.6" in answer.text
    else:
        with pytest.raises(LocalIntelligenceError, match="dialogue_reply_invalid"):
            asyncio.run(app.respond(turn("Arkayla fark?", language="tr")))


def test_checked_adapter_keeps_rejected_raw_draft_out_of_delivered_reply():
    _, memory, _ = setup()
    conversation = MinistralConversation(
        ConversationConfig(model="mistralai/Ministral-3-3B-Instruct-2512"), memory,
        model=Model(ChatReply("You're P99."), ChatReply("You're P99.")),
    )
    row = asyncio.run(conversation.respond("Position?", "en", "failure"))
    assert row["reply"] is None and row["raw_reply"] == "You're P99."
    assert row["application_validation"]["status"] == "rejected"
    assert not conversation.history


def test_social_operation_generates_free_speech_in_one_model_call():
    app, _, model = setup(ChatReply(None, (call(
        "reply_to_driver", message="I hear you. Keep your focus.",
    ),)))
    answer = asyncio.run(app.respond(turn("What a dirty move")))
    assert answer.text == "I hear you. Keep your focus."
    assert len(model.requests) == 1 and not app._delivered


def test_a_stale_social_measurement_can_retry_a_real_read_without_polluting_history():
    app, memory, model = setup(
        ChatReply(None, (call("reply_to_driver", message="Still P5."),)),
        ChatReply(None, (call(),)), ChatReply("You're P4."),
    )
    memory.update(race_context(2, 101, position=4, opponent_positions=(1, 6)))
    answer = asyncio.run(app.respond(turn("Position?")))
    assert answer.text == "You're P4." and len(model.requests) == 3
    assert "reply_to_driver" not in json.dumps(list(app.history))
    assert "P5" not in json.dumps(list(app.history))


def test_exact_measurement_is_in_verified_memory_not_working_chat_history():
    app, _, _ = setup(ChatReply(None, (call(),)), ChatReply("You're P5."))
    answer = asyncio.run(app.respond(turn("I'm P99? What's my position?")))
    assert answer.text == "You're P5."
    history = app.history[-1]
    assert history[0]["content"] == "I'm P99? What's my position?"
    assert "P5" not in json.dumps(history)
    assert app._delivered["position"].value == 5
    assert "recall_radio_reading" in history[-1]["content"]


def test_compacting_working_history_does_not_overwrite_the_actual_tool_trace():
    app, _, model = setup(ChatReply(None, (call(),)), ChatReply("P5."))
    asyncio.run(app.respond(turn("Position?")))
    traced = next(item for item in app.last_trace if item["stage"] == "tools")
    tool_result = json.loads(traced["results"][0]["content"])
    assert tool_result["status"] == "available" and tool_result["position"] == 5
    assert json.loads(model.requests[-1][0][-1]["content"])["position"] == 5
    history_result = next(message for message in app.history[-1] if message["role"] == "tool")
    assert json.loads(history_result["content"])["status"] == "historical_read_completed"


def test_historical_value_cannot_be_presented_as_current_and_can_retry_a_fresh_read():
    app, memory, _ = setup(
        ChatReply(None, (call("get_gap_ahead"),)), ChatReply("Gap ahead is 1 second."),
        ChatReply(None, (call("recall_radio_reading"),)), ChatReply("It's still 1 second now."),
        ChatReply(None, (call("get_gap_ahead"),)), ChatReply("Gap ahead is 2.4 seconds."),
    )
    add_gaps(memory)

    async def run():
        await app.respond(turn("Gap ahead?"))
        add_gaps(memory, sequence=3, ahead=2.4)
        return await app.respond(turn("Gap ahead now?"))

    answer = asyncio.run(run())
    assert answer.text == "Gap ahead is 2.4 seconds."
    assert any(x.get("reason") == "dialogue_historical_claim_scope" for x in app.last_trace)


def test_correct_car_relative_gap_is_not_mistaken_for_a_reversed_driver_claim():
    app, memory, _ = setup(
        ChatReply(None, (call("get_gap_behind"),)),
        ChatReply("You're followed by a car that's 1.6 seconds behind you."),
    )
    add_gaps(memory)
    answer = asyncio.run(app.respond(turn("Gap behind?")))
    assert "1.6 seconds behind you" in answer.text


def test_repeated_references_both_refresh_to_the_current_value():
    app, memory, model = setup()
    add_gaps(memory)

    def reply():
        add_gaps(memory, sequence=3, behind=2.8)
        return ChatReply("The gap is 1.6 seconds. He's 1.6 seconds back.")

    model.replies = iter((ChatReply(None, (call("get_gap_behind"),)), reply))
    answer = asyncio.run(app.respond(turn("Gap behind?")))
    assert answer.text.count("2.8") == 2 and "1.6" not in answer.text


@pytest.mark.parametrize("speech", ["1,6 saniyedir.", "1,6 saniyelik fark var."])
def test_turkish_inflected_seconds_are_not_left_as_unbound_numbers(speech):
    app, memory, _ = setup(ChatReply(None, (call("get_gap_behind"),)), ChatReply(speech))
    add_gaps(memory)
    answer = asyncio.run(app.respond(turn("Arkayla fark?", language="tr")))
    assert "1.6" in answer.text


def test_historical_fact_with_a_separate_followup_question_is_not_a_false_current_claim():
    app, memory, _ = setup(
        ChatReply(None, (call("get_gap_ahead"),)), ChatReply("Gap ahead is 1 second."),
        ChatReply(None, (call("recall_radio_reading"),)),
        ChatReply("I previously said 1.0 seconds. What do you need now?"),
    )
    add_gaps(memory)

    async def run():
        await app.respond(turn("Gap ahead?"))
        add_gaps(memory, sequence=3, ahead=1.9)
        return await app.respond(turn("Was that one second?"))

    answer = asyncio.run(run())
    assert answer.text == "I previously said 1 seconds. What do you need now?"
    assert answer.source_sequence == 2
