import asyncio
from datetime import UTC, datetime, timedelta

import pytest
from test_live_conversation import context
from test_stt_audio import tone

from race_engineer.application.control import LiveControl
from race_engineer.application.live_conversation import live_dialogue
from race_engineer.config import AppConfig, ConversationConfig, SttConfig
from race_engineer.conversation.live import LiveRaceState
from race_engineer.core.intelligence import GroundedResponse
from race_engineer.core.speech_input import Transcription
from race_engineer.core.speech_output import SpeechOutputError, SpeechOutputResult
from race_engineer.tts.live_radio import LiveRadio


@pytest.mark.parametrize(
    ("scenario", "first_delivered"),
    [
        ("slow_fresh", True),
        ("stale", False),
        ("disconnect", False),
        ("session_change", False),
        ("timeout", False),
        ("generation_error", False),
        ("not_played", False),
        ("interrupted", False),
        ("muted", False),
    ],
)
def test_live_reply_budget_freshness_and_delivered_history(monkeypatch, scenario, first_delivered):
    wall = [datetime.now(UTC)]
    elapsed = [0.0]
    state = LiveRaceState(3, clock=lambda: wall[0])
    state.update(context(now=wall[0]))
    prepared = []
    played = []
    notices = []
    control = LiveControl()
    control.emit = lambda topic, value: notices.append((topic, value))

    class Microphone:
        count = 0

        def __init__(self, *args, **kwargs):
            pass

        async def next_clip(self, *, before_capture):
            self.count += 1
            if self.count == 2:
                # Recover a disconnected/changed session before the follow-up.
                state.update(context(now=wall[0], sequence=300))
                radio.reset(state.epoch)
                radio.set_muted(False)
            before_capture()
            return tone() if self.count <= 2 else None

        async def aclose(self):
            pass

    class Recognizer:
        async def start(self):
            pass

        async def transcribe(self, audio):
            return Transcription(
                status="transcribed", text="Are you there?", language="en", audio_duration_s=1
            )

    class Intelligence:
        async def prepare(self, turn):
            prepared.append(turn)
            return turn

        async def ground(self, turn):
            return GroundedResponse(
                response_id=turn.turn_id,
                turn_id=turn.turn_id,
                session_id=turn.session_id,
                generation=turn.generation,
                source_sequence=state.snapshot().context.frame.sequence,
                language="en",
                action="speak",
                text="Copy, I'm here.",
            )

    class Speaker:
        async def start(self):
            pass

        async def speak(self, reply, *, before_playback, preparation_timeout_s):
            assert 0 < preparation_timeout_s <= 15
            if len(prepared) == 1:
                delay = 16 if scenario == "timeout" else 6
                for tick in range(delay):
                    elapsed[0] += 1
                    wall[0] += timedelta(seconds=1)
                    if scenario != "stale":
                        state.update(context(now=wall[0], sequence=201 + tick))
                if scenario == "disconnect":
                    state.invalidate()
                elif scenario == "session_change":
                    state.update(context(now=wall[0], session_id="different-session"))
                if scenario == "generation_error":
                    raise SpeechOutputError("tts_generation_or_playback_failed")
                if scenario == "interrupted":
                    raise asyncio.CancelledError()
            if not before_playback():
                raise SpeechOutputError("tts_reply_expired")
            audible = len(prepared) != 1 or scenario != "not_played"
            if audible:
                played.append(reply.text)
            return SpeechOutputResult(
                language="en",
                played=audible,
                audio_duration_s=1,
                synthesis_ms=6000,
                playback_ms=1000 if audible else 0,
            )

    monkeypatch.setattr(
        "race_engineer.application.live_conversation.PushToTalkMicrophone", Microphone
    )
    monkeypatch.setattr(
        "race_engineer.application.live_conversation.time.perf_counter", lambda: elapsed[0]
    )

    async def run():
        nonlocal radio
        radio = LiveRadio(None)
        radio.set_muted(scenario == "muted")
        try:
            await live_dialogue(
                AppConfig(),
                SttConfig(),
                state,
                radio,
                Recognizer(),
                Speaker(),
                None,
                control,
                intelligence=Intelligence(),
            )
        finally:
            await radio.aclose()

    radio = None
    asyncio.run(run())
    assert len(prepared) == 2
    assert prepared[1].recent_dialogue == (
        ("driver: Are you there?", "engineer: Copy, I'm here.") if first_delivered else ()
    )
    assert len(played) == (2 if first_delivered else 1)
    if scenario in {
        "stale",
        "disconnect",
        "session_change",
        "timeout",
        "generation_error",
        "muted",
        "interrupted",
    }:
        assert any(topic == "notice" and "wasn't played" in value for topic, value in notices)


def test_delivery_config_does_not_relax_telemetry_freshness():
    config = ConversationConfig()
    assert config.reply_delivery_timeout_s == 15
    assert config.max_snapshot_age_s == 3
