import asyncio
from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError
from test_live_conversation import context

from race_engineer.application.live_conversation import live_dialogue
from race_engineer.config import AppConfig, SttConfig
from race_engineer.conversation.composer import compose
from race_engineer.conversation.dialogue_session import DialogueSession
from race_engineer.conversation.live import LiveRaceState, LiveTelemetryUnavailable
from race_engineer.conversation.qwen_v1_dialogue import QwenV1DialogueJudge, QwenV1DialoguePlan
from race_engineer.core.contracts import OpponentState
from race_engineer.core.conversation import RaceQuery
from race_engineer.core.dialogue import (
    GroundedAnswer,
    OpponentReference,
    QueryPart,
    ResponseDecision,
    SemanticProposal,
)
from race_engineer.core.speech_input import Transcription
from race_engineer.core.speech_output import SpeechOutputResult
from race_engineer.testing.dialogue import ScriptedSemanticJudge
from race_engineer.tts.live_radio import LiveRadio


def decision(**updates):
    now = datetime.now(UTC)
    value = {
        "turn_id": "turn-1",
        "response_id": "turn-1:reply",
        "session_id": "live-demo",
        "generation": 0,
        "source_sequence": 1,
        "language": "en",
        "created_at": now,
        "expires_at": now + timedelta(seconds=30),
        "outcome": "answered",
        "reason": "resolved",
    }
    value.update(updates)
    return ResponseDecision.model_validate(value)


@pytest.mark.parametrize(
    "value,expected",
    [
        (
            {"outcome": "acknowledge", "acts": ["acknowledge"]},
            "Understood. Stay focused.",
        ),
        (
            {"outcome": "acknowledge", "acts": ["acknowledge"], "language": "tr"},
            "Anlaşıldı. Yarışına odaklan.",
        ),
        ({"outcome": "no_reply", "acts": ["close"]}, None),
        (
            {"outcome": "unavailable", "reason": "model_error"},
            "The local conversation model couldn't process that. Please try again.",
        ),
    ],
)
def test_bounded_social_and_failure_wording(value, expected):
    assert compose(decision(**value)) == expected


def test_mixed_acknowledgment_keeps_grounded_fact_prominent():
    answer = GroundedAnswer(
        part_id="q1",
        query=RaceQuery.GAP_BEHIND,
        status="available",
        value=2.4,
        unit="s",
        source_sequence=1,
        opponent=OpponentReference(side="behind", driver_id="car-b"),
    )
    result = compose(decision(acts=["acknowledge"], answers=[answer]))
    assert result == "Yeah, copy. Car behind, 2.4 seconds."
    assert "saw" not in result.lower() and "contact" not in result.lower()
    assert (
        compose(decision(outcome="acknowledge", acts=["acknowledge"]), repeated_acknowledgment=True)
        == "Copy."
    )


def test_partial_answer_preserves_clarification_and_unavailable_part():
    answers = [
        GroundedAnswer(
            part_id="p",
            query="position",
            status="available",
            value=6,
            unit="position",
            source_sequence=1,
        ),
        GroundedAnswer(
            part_id="f", query="fuel_to_finish", status="unsupported", source_sequence=1
        ),
    ]
    text = compose(
        decision(outcome="partial", reason="unsupported", answers=answers, clarification="opponent")
    )
    assert "P6" in text and "finish" in text and "ahead or behind" in text


def test_live_refresh_uses_same_opponent_identity_and_preserves_acts():
    state = LiveRaceState(3)
    original = context()
    state.update(original)
    behind = min(
        (
            item
            for item in original.frame.opponents
            if item.gap_to_player_s is not None and item.gap_to_player_s > 0
        ),
        key=lambda item: item.gap_to_player_s,
    )
    response = decision(
        session_id=original.frame.session_id,
        generation=state.epoch,
        acts=["acknowledge"],
        answers=[
            GroundedAnswer(
                part_id="gap",
                query="gap_behind",
                status="available",
                value=behind.gap_to_player_s,
                unit="s",
                source_sequence=original.frame.sequence,
                opponent=OpponentReference(side="behind", driver_id=behind.driver_id),
            )
        ],
    )
    newer = context(sequence=original.frame.sequence + 1)
    opponents = tuple(
        item.model_copy(update={"gap_to_player_s": 1.1})
        if item.driver_id == behind.driver_id
        else item
        for item in newer.frame.opponents
    )
    state.update(
        newer.model_copy(update={"frame": newer.frame.model_copy(update={"opponents": opponents})})
    )
    refreshed = state.refresh_decision(response, state.epoch)
    assert refreshed.answers[0].value == 1.1
    assert refreshed.answers[0].opponent.driver_id == behind.driver_id
    assert refreshed.acts == ("acknowledge",)

    replacement = tuple(
        item.model_copy(update={"driver_id": "replacement"})
        if item.driver_id == behind.driver_id
        else item
        for item in opponents
    )
    latest = context(sequence=original.frame.sequence + 2)
    state.update(
        latest.model_copy(
            update={"frame": latest.frame.model_copy(update={"opponents": replacement})}
        )
    )
    with pytest.raises(LiveTelemetryUnavailable, match="opponent_changed"):
        state.refresh_decision(response, state.epoch)


def classified_context(position, total, sequence):
    base = context(sequence=sequence)
    opponents = tuple(
        OpponentState(
            driver_id=f"car-{place}",
            position=place,
            gap_to_player_s=float(place - position),
        )
        for place in range(1, total + 1)
        if place != position
    )
    frame = base.frame.model_copy(
        update={
            "player": base.frame.player.model_copy(update={"position": position}),
            "opponents": opponents,
            "capabilities": tuple(sorted({*base.frame.capabilities, "opponents"})),
        }
    )
    return base.model_copy(update={"frame": frame})


def test_field_status_is_grounded_and_refreshed_before_delivery():
    async def run():
        state = LiveRaceState(3)
        state.update(classified_context(4, 4, 10))
        judge = ScriptedSemanticJudge(
            SemanticProposal(
                language="en",
                requests=(QueryPart(part_id="field", query="field_status", field_relation="last"),),
                model_id="scripted",
            )
        )
        session = DialogueSession(judge, state.snapshot, generation=lambda: state.epoch)
        answer = await session.ask("Are we dead last?", output_mode="text")
        assert compose(answer) == "Affirm, P4. We're last."

        state.update(classified_context(3, 4, 11))
        refreshed = state.refresh_decision(answer, state.epoch)
        assert (refreshed.answers[0].value, refreshed.answers[0].total) == (3, 4)
        assert compose(refreshed) == "Negative, P3. 1 car behind."
        await session.aclose()

    asyncio.run(run())


def test_first_place_relation_stays_truthful_when_live_position_changes():
    async def run():
        state = LiveRaceState(3)
        state.update(classified_context(1, 4, 20))
        judge = ScriptedSemanticJudge(
            SemanticProposal(
                language="en",
                requests=(
                    QueryPart(part_id="field", query="field_status", field_relation="first"),
                ),
                model_id="scripted",
            )
        )
        session = DialogueSession(judge, state.snapshot, generation=lambda: state.epoch)
        answer = await session.ask("Am I first?", output_mode="text")
        assert compose(answer) == "Affirm, P1. We're leading."

        state.update(classified_context(2, 4, 21))
        refreshed = state.refresh_decision(answer, state.epoch)
        assert refreshed.answers[0].field_relation == "first"
        assert compose(refreshed) == "Negative, P2. 1 car ahead."
        await session.aclose()

    asyncio.run(run())


def test_hybrid_contract_rejects_unbounded_or_conflicting_output():
    with pytest.raises(ValidationError):
        QwenV1DialoguePlan(language="en", queries=("position",), clarification="opponent", acts=())
    with pytest.raises(ValidationError):
        QwenV1DialoguePlan(
            language="en", queries=("unsupported",), clarification="none", acts=("close",)
        )
    with pytest.raises(ValidationError):
        QwenV1DialoguePlan(language="en", queries=("field_status",), clarification="none", acts=())
    with pytest.raises(ValidationError):
        QwenV1DialoguePlan(
            language="en",
            queries=("position",),
            field_relation="first",
            clarification="none",
            acts=(),
        )
    with pytest.raises(ValidationError):
        QwenV1DialoguePlan.model_validate(
            {
                "language": "en",
                "queries": [],
                "clarification": "none",
                "acts": [],
                "mode": "request",
                "text": "We saw the contact",
            }
        )


def test_hybrid_judge_maps_v1_queries_and_social_act(monkeypatch):
    async def run():
        judge = QwenV1DialogueJudge(AppConfig().conversation)

        async def result(**kwargs):
            assert "current_utterance" in kwargs["content"]
            return {
                "schema_version": "qwen-v1-dialogue-plan.v2",
                "language": "en",
                "mode": "request",
                "queries": ["gap_behind"],
                "field_relation": None,
                "clarification": "none",
                "acts": ["acknowledge"],
            }

        monkeypatch.setattr(judge, "_request_json", result)
        from race_engineer.config import DialogueConfig
        from race_engineer.conversation.context_view import ConversationContextAssembler
        from race_engineer.core.dialogue import DialogueState
        from race_engineer.evaluation.dialogue import EvalWorld

        world = EvalWorld()
        request = ConversationContextAssembler(DialogueConfig()).assemble(
            world.snapshot(),
            DialogueState(session_id="dialogue-evaluation"),
            turn_id="turn",
            question="That was dirty. Gap behind?",
            received_at=world.now,
            deadline=world.now + timedelta(seconds=30),
            asr_language="en",
            reply_language=None,
        )
        proposal = await judge.judge(request)
        assert proposal.acts == ("acknowledge",)
        assert proposal.requests[0].query == "gap" and proposal.requests[0].reference == "behind"

    asyncio.run(run())


def test_hybrid_judge_preserves_requested_field_relation(monkeypatch):
    async def run():
        judge = QwenV1DialogueJudge(AppConfig().conversation)

        async def result(**kwargs):
            assert "field_relation" in kwargs["prompt"]
            return {
                "schema_version": "qwen-v1-dialogue-plan.v2",
                "language": "en",
                "mode": "request",
                "queries": ["field_status"],
                "field_relation": "first",
                "clarification": "none",
                "acts": [],
            }

        monkeypatch.setattr(judge, "_request_json", result)
        from race_engineer.config import DialogueConfig
        from race_engineer.conversation.context_view import ConversationContextAssembler
        from race_engineer.core.dialogue import DialogueState
        from race_engineer.evaluation.dialogue import EvalWorld

        world = EvalWorld()
        request = ConversationContextAssembler(DialogueConfig()).assemble(
            world.snapshot(),
            DialogueState(session_id="dialogue-evaluation"),
            turn_id="turn-first",
            question="Am I first?",
            received_at=world.now,
            deadline=world.now + timedelta(seconds=30),
            asr_language="en",
        )
        proposal = await judge.judge(request)
        assert proposal.requests == (
            QueryPart(part_id="q1", query="field_status", field_relation="first"),
        )

    asyncio.run(run())


def test_ce05_opt_in_live_mixed_reply_and_no_reply(monkeypatch, capsys):
    from test_stt_audio import tone

    async def run():
        state = LiveRaceState(3)
        state.update(context())
        spoken = []
        steps = iter(
            (
                Transcription(
                    status="transcribed",
                    text="That was dirty. Position?",
                    language="en",
                    audio_duration_s=1,
                ),
                Transcription(
                    status="transcribed", text="Thanks", language="en", audio_duration_s=1
                ),
            )
        )

        class Microphone:
            def __init__(self, config, **kwargs):
                self.clips = iter((tone(), tone(), None))

            async def next_clip(self, *, before_capture):
                before_capture()
                return next(self.clips)

            async def aclose(self):
                pass

        class Recognizer:
            async def start(self):
                pass

            async def transcribe(self, audio):
                return next(steps)

        class Speaker:
            async def start(self):
                pass

            async def speak(self, reply, *, before_playback):
                assert before_playback()
                spoken.append(reply)
                return SpeechOutputResult(
                    language=reply.language,
                    played=True,
                    audio_duration_s=1,
                    synthesis_ms=1,
                    playback_ms=1,
                )

        judge = ScriptedSemanticJudge(
            SemanticProposal.model_validate(
                {
                    "language": "en",
                    "model_id": "scripted",
                    "acts": ["acknowledge"],
                    "requests": [{"part_id": "p", "query": "position"}],
                }
            ),
            SemanticProposal(language="en", model_id="scripted", acts=("close",)),
        )
        monkeypatch.setattr(
            "race_engineer.application.live_conversation.PushToTalkMicrophone", Microphone
        )
        monkeypatch.setattr(
            "race_engineer.application.live_conversation.semantic_judge", lambda _: judge
        )
        config = AppConfig()
        config = config.model_copy(
            update={
                "conversation": config.conversation.model_copy(
                    update={
                        "dialogue": config.conversation.dialogue.model_copy(
                            update={"enabled": True}
                        )
                    }
                )
            }
        )
        radio = LiveRadio(None)
        await live_dialogue(config, SttConfig(), state, radio, Recognizer(), Speaker(), None)
        await radio.aclose()
        assert len(spoken) == 1
        assert spoken[0].text in {
            "Copy. P6 right now.",
            "Copy. We're running P6.",
            "Yeah, copy. P6 right now.",
            "Yeah, copy. We're running P6.",
        }
        assert len(judge.requests) == 2

    asyncio.run(run())
    assert capsys.readouterr().out.count("Engineer (en)") == 1
