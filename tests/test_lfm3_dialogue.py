import asyncio
import json
from copy import deepcopy

import pytest
from test_context_engine import driver_turn, race_context
from test_intelligence_local_model import FakeConnection, install

from race_engineer.config import ConversationConfig
from race_engineer.intelligence.local_model import LocalIntelligenceError
from race_engineer.intelligence.native_model import NativeToolCall
from race_engineer.intelligence.telemetry_memory import BoundedTelemetryMemory
from scripts.lfm3_dialogue import ChatReply, LocalDialogueModel, NativeToolDialogue


def call(name="get_current_race_position", **arguments):
    return NativeToolCall(call_id="native-call", name=name, arguments=arguments)


class Model:
    def __init__(self, *replies):
        self.replies = iter(replies)
        self.requests = []

    async def request(self, messages, *, tool_choice):
        self.requests.append((deepcopy(messages), tool_choice))
        reply = next(self.replies)
        return reply() if callable(reply) else reply


def setup(*replies, position=5):
    config = ConversationConfig(model="LiquidAI/LFM3-1.7B", history_turns=2)
    memory = BoundedTelemetryMemory()
    memory.update(race_context(1, 100, position=position, opponent_positions=(1, 6)))
    model = Model(*replies)
    return NativeToolDialogue(config, memory, model=model), memory, model


@pytest.mark.parametrize("speech", ["You're P5.", "You're in 5th place.", "You're fifth.",
                                    "Your position is 5.", "5"])
def test_native_dialogue_keeps_original_user_tool_call_and_result_with_grounded_position(speech):
    chosen = ChatReply("Checking.", (call(),), "Need the current reading.")
    app, _, model = setup(chosen, ChatReply(speech))
    request = driver_turn("What position am I at the moment?")
    result = asyncio.run(app.respond(request))
    assert "5" in result.text and result.evidence_ids == ("dialogue:player_position",)
    first, finishing = model.requests
    assert first[1] == "auto" and finishing[1] == "none"
    assert finishing[0][1] == {"role": "user", "content": request.transcript}
    assert finishing[0][2] == chosen.message()
    tool = finishing[0][3]
    assert tool["role"] == "tool" and tool["tool_call_id"] == "native-call"
    assert json.loads(tool["content"])["position"] == 5
    assert finishing[0][2]["reasoning_content"] == "Need the current reading."
    assert len(app.history) == 1


def test_radio_needs_no_tools_and_english_on_is_not_a_turkish_number():
    app, _, model = setup(ChatReply("Yes, driver, I'm on the radio."))
    result = asyncio.run(app.respond(driver_turn("Are you there?")))
    assert result.text == "Yes, driver, I'm on the radio." and not result.evidence_ids
    assert len(model.requests) == 1
    app.clear()
    assert not app.history


def test_unavailable_tyre_result_cannot_become_a_model_authored_recommendation():
    app, _, model = setup(
        ChatReply(None, (call("report_unavailable_information", quantity="tyre advice"),)),
        ChatReply("Use the soft compound, it is fastest."),
    )
    result = asyncio.run(app.respond(driver_turn("What tire should we use next?")))
    assert "soft" not in result.text and not result.evidence_ids
    assert "not" in result.text or "don't" in result.text
    assert json.loads(model.requests[1][0][-1]["content"])["status"] == "unavailable"


def test_position_is_bound_to_observed_tool_fact_then_refreshed_before_delivery():
    app, memory, _ = setup(ChatReply(None, (call(),)))

    def later_reply():
        memory.update(race_context(2, 101, position=4, opponent_positions=(1, 6)))
        return ChatReply("You're P5.")

    app.model.replies = iter((ChatReply(None, (call(),)), later_reply))
    result = asyncio.run(app.respond(driver_turn("Position?")))
    assert result.text == "You're P4." and result.source_sequence == 2


def test_historical_position_requires_current_tool_evidence_before_it_can_be_spoken():
    app, memory, model = setup(
        ChatReply(None, (call(),)), ChatReply("You're P5."),
        ChatReply("You're P5."), ChatReply(None, (call(),)), ChatReply("You're P4."),
    )

    async def run():
        await app.respond(driver_turn("Position?"))
        memory.update(race_context(2, 101, position=4, opponent_positions=(1, 6)))
        return await app.respond(driver_turn("Where am I?").model_copy(update={"turn_id": "new"}))

    result = asyncio.run(run())
    assert result.text == "You're P4."
    assert [choice for _, choice in model.requests] == ["auto", "none", "auto", "auto", "none"]
    assert model.requests[2][0][-1]["content"] == "Where am I?"


def test_repair_can_explain_missing_support_without_being_forced_into_an_unrelated_tool():
    app, _, model = setup(
        ChatReply("I only know that you're P5."),
        ChatReply("I don't have enough information to assess race progress."),
    )
    question = "How's the race going?"
    result = asyncio.run(app.respond(driver_turn(question)))
    assert result.text == "I don't have enough information to assess race progress."
    assert not result.evidence_ids
    assert [choice for _, choice in model.requests] == ["auto", "auto"]
    assert all(messages[-1] == {"role": "user", "content": question}
               for messages, _ in model.requests)
    assert not any(stage["stage"] == "tools" for stage in app.last_trace)


def add_gaps(memory, sequence=2, behind=1.6, ahead=1.0):
    current = race_context(sequence, 100 + sequence, position=5, opponent_positions=(1, 6))
    opponents = tuple(
        item.model_copy(update={"gap_to_player_s": value})
        for item, value in zip(current.frame.opponents, (-ahead, behind), strict=True)
    )
    memory.update(current.model_copy(update={"frame": current.frame.model_copy(
        update={"opponents": opponents}
    )}))


@pytest.mark.parametrize("name,speech,expected", [
    ("get_gap_behind", "Gap behind is 1.6 seconds.", "Gap behind is 1.6 seconds."),
    ("get_gap_ahead", "Gap ahead is 1 second.", "Gap ahead is 1 second."),
    ("get_gap_ahead", "The car ahead is one second away.", "The car ahead is 1 second away."),
    ("get_gap_behind", "1.6", "1.6 seconds"),
])
def test_gap_tools_use_existing_capabilities_and_keep_seconds_distinct_from_position(
    name, speech, expected
):
    app, memory, model = setup(ChatReply(None, (call(name),)), ChatReply(speech))
    add_gaps(memory)
    question = "Gap ahead?" if name.endswith("ahead") else "Gap behind?"
    result = asyncio.run(app.respond(driver_turn(question)))
    assert result.text == expected and result.evidence_ids == ("dialogue:gap_s",)
    result_data = json.loads(model.requests[1][0][-1]["content"])
    assert "position" not in result_data and result_data["unit"] == "seconds"
    assert result_data["gap_seconds"] == (1 if name.endswith("ahead") else 1.6)


def test_gap_is_refreshed_before_delivery():
    app, memory, model = setup()
    add_gaps(memory)

    def reply():
        add_gaps(memory, sequence=3, behind=2.2)
        return ChatReply("Gap behind is 1.6 seconds.")

    model.replies = iter((ChatReply(None, (call("get_gap_behind"),)), reply))
    result = asyncio.run(app.respond(driver_turn("What's the gap behind?")))
    assert result.text == "Gap behind is 2.2 seconds." and result.source_sequence == 3


@pytest.mark.parametrize("speech", ["Gap behind is 1 second.", "You're P5."])
def test_gap_response_cannot_substitute_another_gap_or_historical_rank(speech):
    app, memory, _ = setup(ChatReply(None, (call("get_gap_behind"),)),
                           ChatReply(speech), ChatReply(speech))
    add_gaps(memory)
    with pytest.raises(LocalIntelligenceError, match="dialogue_reply_invalid"):
        asyncio.run(app.respond(driver_turn("Gap behind?")))
    assert not app.history


@pytest.mark.parametrize("name,speech", [
    ("get_gap_behind", "You're currently 1.6 seconds behind the nearest car ahead."),
    ("get_gap_ahead", "You're 1 second ahead of the nearest car in that direction."),
])
def test_correct_gap_number_cannot_validate_a_reversed_driver_relationship(name, speech):
    app, memory, _ = setup(ChatReply(None, (call(name),)), ChatReply(speech), ChatReply(speech))
    add_gaps(memory)
    with pytest.raises(LocalIntelligenceError, match="dialogue_reply_invalid"):
        asyncio.run(app.respond(driver_turn("Gap?")))
    assert any(item.get("reason") == "dialogue_gap_direction_mismatch"
               for item in app.last_trace)
    assert not app.history


@pytest.mark.parametrize("name,speech", [
    ("get_gap_behind", "You're 1.6 seconds ahead of the car behind."),
    ("get_gap_ahead", "You are 1 second behind the car ahead."),
])
def test_verified_driver_gap_direction_is_preserved(name, speech):
    app, memory, _ = setup(ChatReply(None, (call(name),)), ChatReply(speech))
    add_gaps(memory)
    result = asyncio.run(app.respond(driver_turn("Gap?")))
    assert result.text == speech


@pytest.mark.parametrize("name,speech,expected", [
    ("get_gap_behind", "You've got a 1.6-second cushion over the car behind.",
     "You've got a 1.6-second cushion over the car behind."),
    ("get_gap_behind", "You have a 1.6\u2011second cushion over the car behind.",
     "You have a 1.6\u2011second cushion over the car behind."),
    ("get_gap_ahead", "You're a second behind the nearest car ahead.",
     "You're 1 second behind the nearest car ahead."),
    ("get_gap_ahead", "You're 1 second behind. Want the gap to the next one behind?",
     "You're 1 second behind. Want the gap to the next one behind?"),
])
def test_natural_gap_formats_and_reference_pronouns_are_not_rejected(name, speech, expected):
    app, memory, model = setup(ChatReply(None, (call(name),)), ChatReply(speech))
    add_gaps(memory)
    answer = asyncio.run(app.respond(driver_turn("Gap?")))
    assert answer.text == expected
    assert answer.evidence_ids == ("dialogue:gap_s",) and len(model.requests) == 2


def test_natural_number_formats_do_not_allow_a_reversed_edge_claim():
    bad = "You've got a tight one-second edge over the car ahead."
    app, memory, _ = setup(
        ChatReply(None, (call("get_gap_ahead"),)), ChatReply(bad), ChatReply(bad)
    )
    add_gaps(memory)
    with pytest.raises(LocalIntelligenceError, match="dialogue_reply_invalid"):
        asyncio.run(app.respond(driver_turn("Gap ahead?")))
    assert any(item.get("reason") == "dialogue_gap_direction_mismatch" for item in app.last_trace)


def test_gap_to_trailing_car_does_not_prove_global_race_leadership():
    app, memory, model = setup(
        ChatReply(None, (call("get_gap_behind"),)),
        ChatReply("You're 1.6 seconds ahead of the car behind. So you're leading on the track."),
        ChatReply("The gap to the car behind is 1.6 seconds."),
    )
    add_gaps(memory)
    answer = asyncio.run(app.respond(driver_turn("Gap behind?")))
    assert answer.text == "The gap to the car behind is 1.6 seconds."
    assert len(model.requests) == 3
    assert any(e.get("reason") == "dialogue_unsupported_leadership_claim" for e in app.last_trace)


def test_rank_snapshot_does_not_prove_holding_ground():
    bad = "You're P5, so you're holding your ground."
    app, _, _ = setup(ChatReply(None, (call(),)), ChatReply(bad), ChatReply(bad))
    with pytest.raises(LocalIntelligenceError, match="dialogue_reply_invalid"):
        asyncio.run(app.respond(driver_turn("Position?")))
    assert any(e.get("reason") == "dialogue_unsupported_pace_claim" for e in app.last_trace)


def test_repeat_in_same_replay_frame_still_requires_current_tool_evidence():
    app, _, model = setup(
        ChatReply(None, (call(),)), ChatReply("You're P5."), ChatReply("Still P5."),
        ChatReply(None, (call(),)), ChatReply("You're P5."),
    )

    async def run():
        await app.respond(driver_turn("Position?"))
        repeat = driver_turn("Where am I?").model_copy(update={"turn_id": "repeat"})
        return await app.respond(repeat)

    answer = asyncio.run(run())
    assert answer.text == "You're P5." and answer.evidence_ids == ("dialogue:player_position",)
    assert len(model.requests) == 5
    assert any(e.get("reason") == "dialogue_current_evidence_required" for e in app.last_trace)
    assert model.requests[-2][1] == "auto"


@pytest.mark.parametrize("bad", ["You're P6.", "You're sixth.", "You have 5 liters.",
                                 "You're P5 of 6.", "Position {{fake}}."])
def test_unverified_numbers_and_references_fail_after_one_repair_without_entering_history(bad):
    app, _, model = setup(ChatReply(None, (call(),)), ChatReply(bad), ChatReply(bad))
    with pytest.raises(LocalIntelligenceError, match="dialogue_reply_invalid"):
        asyncio.run(app.respond(driver_turn("Position?")))
    assert len(model.requests) == 3 and not app.history


def test_entire_tool_batch_is_validated_before_any_python_tool_runs(monkeypatch):
    app, _, _ = setup(ChatReply(None, (call(), call("delete_files"))))
    calls = []
    monkeypatch.setattr(app.capabilities, "execute", lambda _: calls.append("executed"))
    with pytest.raises(LocalIntelligenceError, match="dialogue_invalid_tool_batch"):
        asyncio.run(app.respond(driver_turn("Position?")))
    assert calls == [] and not app.history


def test_model_cannot_execute_a_second_tool_batch():
    app, _, model = setup(ChatReply(None, (call(),)), ChatReply(None, (call(),)))
    with pytest.raises(LocalIntelligenceError, match="dialogue_tool_budget_exceeded"):
        asyncio.run(app.respond(driver_turn("Position?")))
    assert len(model.requests) == 2 and not app.history


@pytest.mark.parametrize("selection,reason", [
    (call("delete_files"), "dialogue_unknown_tool"),
    (call(position=99), "dialogue_invalid_tool_arguments"),
])
def test_unknown_tool_and_model_supplied_position_cannot_execute(selection, reason, monkeypatch):
    app, _, _ = setup(ChatReply(None, (selection,)))
    executed = []
    monkeypatch.setattr(app.capabilities, "execute", lambda _: executed.append(True))
    with pytest.raises(LocalIntelligenceError, match=reason):
        asyncio.run(app.respond(driver_turn("Position?")))
    assert not executed and not app.history


def test_unparsed_control_markup_is_never_delivered_as_speech():
    bad = ChatReply("<tool_call><function=get_current_race_position></function></tool_call>")
    app, _, _ = setup(bad, bad)
    with pytest.raises(LocalIntelligenceError, match="dialogue_reply_invalid"):
        asyncio.run(app.respond(driver_turn("Position?")))
    assert not app.history


@pytest.mark.parametrize("startup_fails", [False, True])
def test_native_trial_closes_owned_server_on_startup_or_question_failure(
    monkeypatch, startup_fails
):
    from pathlib import Path
    from types import SimpleNamespace

    from race_engineer.config import AppConfig
    from scripts import try_lfm3_dialogue as cli

    config = AppConfig(conversation=ConversationConfig(model="LiquidAI/LFM3-1.7B"))
    monkeypatch.setattr(cli, "load_config", lambda _: config)
    events = []

    class Server:
        def __init__(self, *_):
            pass

        async def start(self):
            events.append("start")
            if startup_fails:
                raise RuntimeError("startup_failed")

        async def aclose(self):
            events.append("close")

    class Dialogue:
        def __init__(self, *_):
            pass

        async def respond(self, _):
            raise LocalIntelligenceError("dialogue_reply_invalid")

    monkeypatch.setattr(cli, "ConversationServer", Server)
    monkeypatch.setattr(cli, "NativeToolDialogue", Dialogue)
    args = SimpleNamespace(config=Path("unused"), question="Position?", trace=False)
    if startup_fails:
        with pytest.raises(RuntimeError, match="startup_failed"):
            asyncio.run(cli.chat(args))
    else:
        assert asyncio.run(cli.chat(args)) == 1
    assert events == ["start", "close"]


def test_native_transport_separates_reasoning_and_sends_no_json_reply_schema(monkeypatch):
    raw = json.dumps({"choices": [{"finish_reason": "stop", "message": {
        "content": "<think>Internal draft.</think>Yes, I'm here.",
    }}]}).encode()
    connection = FakeConnection(raw=raw)
    install(monkeypatch, connection)
    result = asyncio.run(LocalDialogueModel(ConversationConfig()).request(
        [{"role": "user", "content": "Are you there?"}], tool_choice="auto",
    ))
    assert result.content == "Yes, I'm here."
    assert result.reasoning_content == "Internal draft."
    body = connection.requests[0][2]
    assert body["tool_choice"] == "auto" and "response_format" not in body
    assert body["reasoning_format"] == "deepseek"
    assert body["chat_template_kwargs"] == {"enable_thinking": False}
    assert connection.closed


def test_bounded_reasoning_experiment_is_explicit_and_does_not_inherit_none_effort(monkeypatch):
    connection = FakeConnection(raw=json.dumps({"choices": [{
        "finish_reason": "stop", "message": {"content": "I'm here."},
    }]}).encode())
    install(monkeypatch, connection)
    asyncio.run(LocalDialogueModel(ConversationConfig(), reasoning_budget=64).request(
        [{"role": "user", "content": "Are you there?"}], tool_choice="auto",
    ))
    body = connection.requests[0][2]
    assert body["reasoning_budget"] == 64
    assert body["chat_template_kwargs"] == {"enable_thinking": True}
    assert "reasoning_effort" not in body


@pytest.mark.parametrize("budget", [True, 0, 129])
def test_reasoning_experiment_cannot_be_unbounded(budget):
    with pytest.raises(ValueError, match="reasoning budget"):
        LocalDialogueModel(ConversationConfig(), reasoning_budget=budget)


@pytest.mark.parametrize("raw,reason", [
    (b"not-json", "dialogue_response_invalid"),
    (b"x" * 70000, "dialogue_response_too_large"),
    (b'{"choices":[{"finish_reason":"length","message":{"content":"hi"}}]}',
     "dialogue_response_incomplete"),
    (b'{"choices":[{"finish_reason":"stop","message":{"content":"<think>unfinished"}}]}',
     "dialogue_response_incomplete"),
], ids=["bad-json", "oversize", "truncated", "unfinished-think"])
def test_native_transport_rejects_invalid_or_truncated_output_and_closes(monkeypatch, raw, reason):
    connection = FakeConnection(raw=raw)
    install(monkeypatch, connection)
    with pytest.raises(LocalIntelligenceError, match=reason):
        asyncio.run(LocalDialogueModel(ConversationConfig()).request([], tool_choice="auto"))
    assert connection.closed
