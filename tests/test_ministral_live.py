import asyncio

import pytest
from test_context_engine import race_context
from test_lfm3_dialogue import Model, call
from test_ministral_dialogue import turn

from race_engineer.config import ConversationConfig
from race_engineer.intelligence.factory import live_intelligence
from race_engineer.intelligence.local_model import LocalIntelligenceError
from race_engineer.intelligence.ministral import MINISTRAL_NATIVE_ID, MinistralEngineer
from race_engineer.intelligence.native_dialogue import ChatReply
from race_engineer.intelligence.telemetry_memory import BoundedTelemetryMemory


def setup(*replies, history_turns=6):
    memory = BoundedTelemetryMemory()
    memory.update(race_context(1, 100, position=5, opponent_positions=(1, 6)))
    config = ConversationConfig(history_turns=history_turns)
    model = Model(*replies)
    return MinistralEngineer(config, memory, model=model), memory, model


def test_factory_is_lazy_and_defaults_to_packaged_ministral_without_telemetry():
    config = ConversationConfig()
    assert config.context_planner == MINISTRAL_NATIVE_ID
    engineer = live_intelligence(config, BoundedTelemetryMemory())
    assert isinstance(engineer, MinistralEngineer)
    assert engineer._engine is None
    assert engineer.__class__.__module__.startswith("race_engineer.")


def test_refresh_at_delivery_and_remember_only_the_value_that_was_spoken():
    app, memory, model = setup(
        ChatReply(None, (call(),)), ChatReply("You're P5."),
        ChatReply(None, (call("recall_radio_reading", quantity="position"),)),
        ChatReply("Last call was P4."),
    )

    async def run():
        prepared = await app.prepare(turn("Position?"))
        assert not app._engine.history and not app._engine._delivered
        memory.update(race_context(2, 101, position=4, opponent_positions=(1, 6)))
        grounded = await app.ground(prepared)
        assert grounded.text == "You're P4."
        assert not app._engine.history and not app._engine._delivered
        assert len(model.requests) == 2  # Delivery refresh does not invoke the LLM.
        app.delivered(prepared, grounded)
        assert app._engine._delivered["position"].value == 4
        assert len(app._engine.history) == 1
        recalled = await app.respond(turn("What did you say?"))
        assert recalled.text == "Last call was P4."
        assert recalled.source_sequence == 2
        with pytest.raises(LocalIntelligenceError, match="scope_mismatch"):
            app.delivered(prepared, grounded)

    asyncio.run(run())


@pytest.mark.parametrize("ground_before_cancel", [True, False])
def test_cancelled_or_unplayed_reply_is_never_in_verified_recall(ground_before_cancel):
    app, _, _ = setup(
        ChatReply(None, (call(),)), ChatReply("P5."),
        ChatReply(None, (call("recall_radio_reading", quantity="position"),)),
        ChatReply("Unavailable."),
    )

    async def run():
        prepared = await app.prepare(turn("Position?"))
        if ground_before_cancel:
            await app.ground(prepared)
        answer = await app.respond(turn("What did you say?"))
        assert "5" not in answer.text
        assert not answer.evidence_ids
        assert not app._engine._delivered
        assert len(app._engine.history) == 1
        with pytest.raises(LocalIntelligenceError, match="superseded"):
            await app.ground(prepared)

    asyncio.run(run())


def test_boolean_verdict_cannot_become_contradictory_before_delivery():
    app, memory, _ = setup(
        ChatReply(None, (call("check_if_driver_is_first"),)),
        ChatReply("{{is_first}}, you're {{player_position}}."),
    )

    async def run():
        prepared = await app.prepare(turn("Are we first?"))
        memory.update(race_context(2, 101, position=1, opponent_positions=(2, 6)))
        with pytest.raises(LocalIntelligenceError, match="boolean_state_changed"):
            await app.ground(prepared)
        assert not app._engine.history

    asyncio.run(run())


def test_session_change_before_delivery_rejects_old_reply():
    app, memory, _ = setup(ChatReply(None, (call(),)), ChatReply("P5."))

    async def run():
        prepared = await app.prepare(turn("Position?"))
        changed = race_context(2, 101).model_copy(update={
            "frame": race_context(2, 101).frame.model_copy(update={"session_id": "next-session"})
        })
        memory.update(changed)
        with pytest.raises(LocalIntelligenceError, match="session_changed"):
            await app.ground(prepared)
        assert not app._engine.history

    asyncio.run(run())


@pytest.mark.parametrize("scope", ["language", "generation"])
def test_new_language_or_generation_clears_delivered_history(scope):
    app, _, _ = setup(
        ChatReply(None, (call(),)), ChatReply("P5."),
        ChatReply(None, (call("reply_to_driver", message="Copy."),)),
    )

    async def run():
        await app.respond(turn("Position?"))
        assert app._engine._delivered
        request = turn("Copy", language="tr" if scope == "language" else "en",
                       generation=1 if scope == "generation" else 0)
        await app.respond(request)
        assert not app._engine._delivered
        assert len(app._engine.history) == 1

    asyncio.run(run())


def test_zero_history_does_not_fail_after_successful_delivery():
    app, _, _ = setup(ChatReply(None, (call(),)), ChatReply("P5."), history_turns=0)
    assert asyncio.run(app.respond(turn("Position?"))).text == "P5."
    assert not app._engine.history
