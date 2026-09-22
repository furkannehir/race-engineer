import asyncio
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from race_engineer.application.live_conversation import LiveBridge, live_dialogue, voice_iracing
from race_engineer.config import AppConfig, SttConfig
from race_engineer.conversation.live import LiveRaceState, LiveTelemetryUnavailable
from race_engineer.conversation.replay import ReplayRaceState
from race_engineer.conversation.session import ConversationSession
from race_engineer.core.conversation import ConversationPlan
from race_engineer.core.speech_input import Transcription
from race_engineer.core.speech_output import SpeechOutputResult
from race_engineer.fixtures import load_fixture
from race_engineer.tts.live_radio import LiveRadio

ROOT = Path(__file__).parents[1]


def context(index=0, *, now=None, session_id="live-demo", sequence=None):
    replay = ReplayRaceState(load_fixture(ROOT / "fixtures/synthetic/conversation"))
    replay.seek(index)
    original = replay.snapshot().context
    updates = {
        "observed_at": now or datetime.now(UTC),
        "session_id": session_id,
        "is_replay": False,
    }
    if sequence is not None:
        updates["sequence"] = sequence
    return original.model_copy(update={"frame": original.frame.model_copy(update=updates)})


def test_live_snapshot_uses_wall_clock_and_never_fabricates_a_frame():
    now = datetime.now(UTC)
    clock = [now]
    state = LiveRaceState(3, clock=lambda: clock[0])
    with pytest.raises(LiveTelemetryUnavailable):
        state.snapshot()
    state.update(context(now=now))
    assert state.snapshot().mode == "live" and state.snapshot().as_of == now
    clock[0] += timedelta(seconds=4)
    assert not state.available
    with pytest.raises(LiveTelemetryUnavailable):
        state.snapshot()
    state.update(context(1, now=clock[0]))
    assert state.available and state.epoch == 1


@pytest.mark.parametrize("kind", ["disconnect", "session", "ticks", "replay"])
def test_invalidations_change_generation_even_when_ids_are_reused(kind):
    state = LiveRaceState(3)
    first = context()
    state.update(first)
    if kind == "disconnect":
        state.invalidate()
        state.update(first)
    elif kind == "session":
        state.update(context(session_id="new-session"))
    elif kind == "ticks":
        state.update(context(sequence=1))
    else:
        state.update(
            first.model_copy(update={"frame": first.frame.model_copy(update={"is_replay": True})})
        )
        assert not state.available
    assert state.epoch == 1


def test_planner_gets_current_facts_after_inference_and_queue_refresh():
    state = LiveRaceState(3)
    state.update(context())

    class Planner:
        async def plan(self, request):
            state.update(context(1))
            return ConversationPlan(language="en", queries=("position",), clarification="none")

    session = ConversationSession(Planner(), state.snapshot, generation=lambda: state.epoch)
    reply = asyncio.run(session.ask("Position?"))
    assert reply.mode == "live" and reply.answers[0].value == 5
    state.update(context(0, sequence=202))
    refreshed = state.refresh(reply, state.epoch)
    assert refreshed.answers[0].value == 6 and refreshed.source_sequence == 202


def test_reconnect_during_model_inference_discards_reply_and_clears_memory():
    state = LiveRaceState(3)
    state.update(context())
    requests = []

    class Planner:
        async def plan(self, request):
            requests.append(request)
            if len(requests) == 2:
                state.invalidate()
                state.update(context(1))  # Same session ID, new connection.
            return ConversationPlan(language="en", queries=("position",), clarification="none")

    session = ConversationSession(Planner(), state.snapshot, generation=lambda: state.epoch)

    async def run():
        await session.ask("Position?")
        assert (await session.ask("And now?")).reason == "session_changed"
        assert (await session.ask("Position?")).status == "answered"

    asyncio.run(run())
    assert requests[1].history and not requests[2].history


def test_disconnect_during_inference_never_returns_last_known_facts():
    state = LiveRaceState(3)
    state.update(context())

    class Planner:
        async def plan(self, request):
            state.invalidate()
            return ConversationPlan(language="en", queries=("position",), clarification="none")

    with pytest.raises(LiveTelemetryUnavailable):
        asyncio.run(ConversationSession(Planner(), state.snapshot).ask("Position?"))


def test_live_dialogue_refreshes_facts_and_closes_microphone(monkeypatch, capsys):
    from test_stt_audio import tone

    state = LiveRaceState(3)
    state.update(context())
    captured = []
    steps = []

    class Microphone:
        def __init__(self, config):
            self.clips = iter((tone(), None))

        async def next_clip(self, *, before_capture):
            before_capture()
            return next(self.clips)

        async def aclose(self):
            steps.append("microphone_closed")

    class Recognizer:
        async def start(self):
            pass

        async def transcribe(self, audio):
            state.update(context(1))
            return Transcription(
                status="transcribed", text="Kaçıncıyız?", language="tr", audio_duration_s=1
            )

    class Planner:
        def __init__(self, config):
            pass

        async def plan(self, request):
            await asyncio.sleep(0)
            return ConversationPlan(language="tr", queries=("position",), clarification="none")

    class Speaker:
        async def start(self):
            pass

        async def speak(self, reply, *, before_playback):
            assert before_playback()
            captured.append(reply)
            return SpeechOutputResult(
                language="tr", played=True, audio_duration_s=1, synthesis_ms=1, playback_ms=1
            )

    monkeypatch.setattr(
        "race_engineer.application.live_conversation.PushToTalkMicrophone", Microphone
    )
    monkeypatch.setattr("race_engineer.application.live_conversation.conversation_planner", Planner)

    async def run():
        radio = LiveRadio(None)
        await live_dialogue(AppConfig(), SttConfig(), state, radio, Recognizer(), Speaker(), None)
        await radio.aclose()

    asyncio.run(run())
    assert captured[0].language == "tr" and captured[0].answers[0].value == 5
    assert captured[0].mode == "live" and captured[0].source_sequence == 200
    assert steps == ["microphone_closed"]
    assert "Engineer (tr)" in capsys.readouterr().out


def test_live_recording_continues_while_conversation_is_blocked(tmp_path, monkeypatch):
    from race_engineer.cli import _read_iracing

    async def run():
        state = LiveRaceState(3)
        radio = LiveRadio(None)
        bridge = LiveBridge(state, radio)
        planning = asyncio.Event()
        release = asyncio.Event()
        newer_frame = asyncio.Event()

        class Adapter:
            def __init__(self, config, availability_sink):
                self.notify = availability_sink
                assert not config.include_replay

            async def stream(self):
                self.notify(True)
                yield context().frame
                await planning.wait()
                yield context(1).frame
                newer_frame.set()
                await release.wait()

        class Planner:
            async def plan(self, request):
                planning.set()
                await newer_frame.wait()
                return ConversationPlan(language="en", queries=("position",), clarification="none")

        monkeypatch.setattr("race_engineer.cli.IracingTelemetryAdapter", Adapter)
        reader = asyncio.create_task(
            _read_iracing(ROOT / "config/default.toml", 0, tmp_path / "live", live=bridge)
        )
        await asyncio.sleep(0)
        answer = await asyncio.wait_for(
            ConversationSession(Planner(), state.snapshot).ask("Position?"), 1
        )
        assert answer.answers[0].value == 5
        release.set()
        assert await reader == 2
        assert not state.available
        await radio.aclose()

    asyncio.run(run())
    assert len(load_fixture(tmp_path / "live").frames) == 2


def test_live_command_stops_all_components_when_stream_ends(monkeypatch):
    closed = []

    async def read(*args, **kwargs):
        await asyncio.sleep(0)
        return 0

    async def dialogue(*args):
        try:
            await asyncio.Event().wait()
        finally:
            closed.append("dialogue")

    class Recognizer:
        def __init__(self, config):
            pass

        async def aclose(self):
            closed.append("recognizer")

    monkeypatch.setattr("race_engineer.cli._read_iracing", read)
    monkeypatch.setattr("race_engineer.application.live_conversation.live_dialogue", dialogue)
    monkeypatch.setattr(
        "race_engineer.application.live_conversation.QwenSpeechRecognizer", Recognizer
    )
    assert asyncio.run(voice_iracing(ROOT / "config/default.toml", text_only=True)) == 0
    assert closed == ["dialogue", "recognizer"]


def test_stale_watchdog_invalidates_pending_radio_and_does_not_repeat():
    async def run():
        clock = [datetime.now(UTC)]
        state = LiveRaceState(3, clock=lambda: clock[0])
        radio = LiveRadio(None)
        bridge = LiveBridge(state, radio)
        bridge.update(context(now=clock[0]))
        clock[0] += timedelta(seconds=4)
        watchdog = asyncio.create_task(bridge.watch_freshness())
        await asyncio.sleep(0)
        assert state.epoch == 1 and not state.available
        bridge.availability_changed(False)
        assert state.epoch == 1
        watchdog.cancel()
        await asyncio.gather(watchdog, return_exceptions=True)
        await radio.aclose()

    asyncio.run(run())


def test_live_dialogue_survives_critical_capture_interrupt_without_submitting_audio(monkeypatch):
    from test_live_radio import submit
    from test_speech_playback_queue import RecordingEngine

    async def run():
        state = LiveRaceState(3)
        state.update(context())
        listening = asyncio.Event()
        automatic_done = asyncio.Event()
        stopped = []

        class Engine(RecordingEngine):
            async def speak(self, utterance):
                assert stopped == [True]
                result = await super().speak(utterance)
                automatic_done.set()
                return result

        class Microphone:
            def __init__(self, config):
                self.first = True

            async def next_clip(self, *, before_capture):
                if self.first:
                    self.first = False
                    before_capture()
                    listening.set()
                    try:
                        await asyncio.Event().wait()
                    finally:
                        stopped.append(True)
                await automatic_done.wait()
                return None  # ESC after the critical call.

            async def aclose(self):
                pass

        class Recognizer:
            async def start(self):
                pass

            async def transcribe(self, audio):
                raise AssertionError("cancelled capture must never reach transcription")

        monkeypatch.setattr(
            "race_engineer.application.live_conversation.PushToTalkMicrophone", Microphone
        )
        radio = LiveRadio(Engine())
        dialogue = asyncio.create_task(
            live_dialogue(AppConfig(), SttConfig(), state, radio, Recognizer(), None, None)
        )
        await listening.wait()
        await submit(radio, "critical", 100)
        await dialogue
        await radio.aclose()

    asyncio.run(asyncio.wait_for(run(), 2))
