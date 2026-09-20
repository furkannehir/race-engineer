import asyncio
from datetime import timedelta
from pathlib import Path

import pytest
from pydantic import ValidationError

from race_engineer.config import ConversationConfig
from race_engineer.conversation.answers import render, retrieve
from race_engineer.conversation.local_model import ConversationModelError
from race_engineer.conversation.replay import ReplayRaceState
from race_engineer.conversation.session import ConversationSession
from race_engineer.core.conversation import ConversationPlan, RaceQuery
from race_engineer.fixtures import load_fixture

ROOT = Path(__file__).parents[1]
FIXTURE = ROOT / "fixtures" / "synthetic" / "conversation"


def plan(*queries, language="en", clarification="none"):
    return ConversationPlan(language=language, queries=queries, clarification=clarification)


class ScriptedPlanner:
    """Exercises orchestration, not natural-language quality (see the model eval)."""

    def __init__(self, *plans):
        self.plans = iter(plans)
        self.requests = []

    async def plan(self, request):
        self.requests.append(request)
        result = next(self.plans)
        if isinstance(result, Exception):
            raise result
        return result


@pytest.fixture
def state():
    return ReplayRaceState(load_fixture(FIXTURE))


def test_compound_question_and_bilingual_fact_rendering(state):
    planner = ScriptedPlanner(plan("position", "gap_behind", language="tr"))
    reply = asyncio.run(ConversationSession(planner, state.snapshot).ask("Kaçıncıyız, arkamız?"))
    assert reply.text == "Genel sıralamada 6. sıradasın. Arkadaki araç 2,4 saniye geride."
    assert reply.source_sequence == 100
    assert reply.mode == "replay"
    assert reply.status == "answered"
    assert [answer.value for answer in reply.answers] == [6, 2.4]


def test_followup_remembers_topic_but_uses_new_facts(state):
    planner = ScriptedPlanner(plan("gap_behind"), plan("gap_behind", language="tr"))
    session = ConversationSession(planner, state.snapshot)

    async def run():
        first = await session.ask("What about the car behind?")
        state.seek(1)
        second = await session.ask("Peki şimdi?")
        return first, second

    first, second = asyncio.run(run())
    assert first.answers[0].value == 2.4
    assert second.answers[0].value == 1.1
    assert planner.requests[1].history[0].plan.queries == (RaceQuery.GAP_BEHIND,)
    # Prompt memory has no telemetry values or historical answer text to accidentally reuse.
    assert "2.4" not in planner.requests[1].model_dump_json()


def test_retrieves_snapshot_again_after_model_finishes(state):
    class AdvancingPlanner:
        async def plan(self, request):
            state.seek(1)
            return plan("position")

    reply = asyncio.run(ConversationSession(AdvancingPlanner(), state.snapshot).ask("Position?"))
    assert reply.answers[0].value == 5
    assert reply.source_sequence == 200


def test_missing_facts_are_not_zeroes(state):
    state.seek(2)
    planner = ScriptedPlanner(plan("fuel_remaining", "gap_behind"))
    reply = asyncio.run(ConversationSession(planner, state.snapshot).ask("Fuel and behind?"))
    assert reply.status == "unavailable"
    assert all(answer.status == "missing" and answer.value is None for answer in reply.answers)
    assert "0.0" not in reply.text


def test_unsupported_reasoning_does_not_infer_strategy_from_current_values(state):
    planner = ScriptedPlanner(plan("fuel_to_finish", "gap_trend_behind", language="tr"))
    reply = asyncio.run(
        ConversationSession(planner, state.snapshot).ask("Yeter mi, yaklaşıyor mu?")
    )
    assert reply.status == "unavailable"
    assert all(answer.status == "unsupported" for answer in reply.answers)
    assert "kalan yarış mesafesi" in reply.text


def test_clarification_is_remembered_and_language_can_be_forced(state):
    planner = ScriptedPlanner(
        plan(clarification="opponent", language="en"), plan("gap_trend_behind")
    )
    session = ConversationSession(planner, state.snapshot)

    async def run():
        first = await session.ask("Is he gaining?", reply_language="tr")
        second = await session.ask("Behind.")
        return first, second

    first, _ = asyncio.run(run())
    assert first.status == "clarification"
    assert first.text == "Öndeki aracı mı, arkadakini mi kastediyorsun?"
    assert planner.requests[1].default_language == "tr"
    assert planner.requests[1].history[0].plan.clarification == "opponent"


@pytest.mark.parametrize("seconds", [-1, 4])
def test_stale_or_future_telemetry_is_rejected_before_inference(state, seconds):
    snapshot = state.snapshot()
    invalid = snapshot.model_copy(update={"as_of": snapshot.as_of + timedelta(seconds=seconds)})
    planner = ScriptedPlanner()
    reply = asyncio.run(ConversationSession(planner, lambda: invalid).ask("Position?"))
    assert reply.reason == "stale_snapshot"
    assert planner.requests == []


def test_telemetry_can_expire_during_inference(state):
    current = state.snapshot()

    class SlowPlanner:
        async def plan(self, request):
            nonlocal current
            current = current.model_copy(update={"as_of": current.as_of + timedelta(seconds=4)})
            return plan("position")

    reply = asyncio.run(ConversationSession(SlowPlanner(), lambda: current).ask("Position?"))
    assert reply.reason == "stale_snapshot"
    assert not reply.answers


def test_session_change_during_inference_discards_reply_and_history(state):
    current = state.snapshot()

    class ChangingPlanner:
        def __init__(self):
            self.requests = []

        async def plan(self, request):
            nonlocal current
            self.requests.append(request)
            frame = current.context.frame.model_copy(update={"session_id": "new-session"})
            context = current.context.model_copy(update={"frame": frame})
            current = current.model_copy(update={"context": context})
            return plan("position")

    planner = ChangingPlanner()
    session = ConversationSession(planner, lambda: current)

    async def run():
        first = await session.ask("Position?")
        second = await session.ask("And now?")
        return first, second

    first, second = asyncio.run(run())
    assert first.reason == "session_changed"
    assert second.status == "answered"
    assert planner.requests[1].history == ()


def test_memory_is_bounded_resettable_and_failures_do_not_pollute_it(state, caplog):
    planner = ScriptedPlanner(
        plan("position"),
        ConversationModelError("model_timeout"),
        plan("gap_behind"),
        plan("fuel_remaining"),
        plan("position"),
    )
    session = ConversationSession(planner, state.snapshot, ConversationConfig(history_turns=1))

    async def run():
        await session.ask("first private question")
        failed = await session.ask("second private question")
        await session.ask("third")
        await session.ask("fourth")
        session.reset()
        await session.ask("fifth")
        return failed

    reply = asyncio.run(run())
    assert reply.status == "model_error"
    assert reply.reason == "model_timeout"
    assert planner.requests[2].history[0].question == "first private question"
    assert planner.requests[3].history[0].question == "third"
    assert planner.requests[4].history == ()
    assert "private question" not in caplog.text


@pytest.mark.parametrize("question", ["", "  ", "x" * 1001])
def test_invalid_questions_never_reach_model(state, question):
    planner = ScriptedPlanner()
    with pytest.raises(ValidationError):
        asyncio.run(ConversationSession(planner, state.snapshot).ask(question))
    assert not planner.requests


@pytest.mark.parametrize(
    "fields",
    [
        {"language": "de", "queries": ["position"], "clarification": "none"},
        {"language": "en", "queries": ["execute_command"], "clarification": "none"},
        {"language": "en", "queries": ["position", "position"], "clarification": "none"},
        {"language": "en", "queries": [], "clarification": "none"},
        {"language": "en", "queries": ["position"], "clarification": "topic"},
        {"language": "en", "queries": ["position"], "clarification": "none", "value": 1},
    ],
)
def test_plan_rejects_invalid_or_fabricated_content(fields):
    with pytest.raises(ValidationError):
        ConversationPlan.model_validate(fields)


def test_all_queries_have_bilingual_renderers_for_available_and_missing_values(state):
    for index in (0, 1, 2):
        state.seek(index)
        for query in RaceQuery:
            for language in ("en", "tr"):
                assert render(retrieve(state.snapshot().context, query), language)


def test_replay_clock_and_frame_validation(state):
    assert state.snapshot().as_of == state.snapshot().context.frame.observed_at
    with pytest.raises(ValueError, match="frame index"):
        state.seek(-1)
    with pytest.raises(ValueError, match="frame index"):
        state.seek(3)
    fixture = load_fixture(FIXTURE)
    with pytest.raises(ValueError, match="at least one frame"):
        ReplayRaceState(fixture.model_copy(update={"frames": ()}))
    with pytest.raises(ValueError, match="ordered"):
        ReplayRaceState(fixture.model_copy(update={"frames": tuple(reversed(fixture.frames))}))


def test_conversation_config_rejects_remote_endpoint_configuration():
    with pytest.raises(ValidationError):
        ConversationConfig.model_validate({"host": "api.example.com"})
