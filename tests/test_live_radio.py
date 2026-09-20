import asyncio
from datetime import UTC, datetime, timedelta

import pytest
from test_speech_playback_queue import RecordingEngine, make_intent

from race_engineer.core.contracts import Utterance
from race_engineer.core.enums import InterruptionPolicy
from race_engineer.core.speech_input import SpeechInputError
from race_engineer.tts.live_radio import LiveRadio


async def submit(radio, name, priority=30, *, epoch=0, deadline=None):
    intent = make_intent(
        name,
        priority=priority,
        deadline=deadline or datetime.now(UTC) + timedelta(seconds=10),
        interruption_policy=(
            InterruptionPolicy.INTERRUPT_LOWER_PRIORITY
            if priority >= 90
            else InterruptionPolicy.NEVER
        ),
    )
    return await radio.submit(intent, Utterance(intent_id=name, text=name), epoch)


def test_critical_call_waits_for_cancel_cleanup_without_blocking_telemetry_submit():
    async def run():
        entered = asyncio.Event()
        cleanup_started = asyncio.Event()
        cleanup_allowed = asyncio.Event()
        engine = RecordingEngine()
        radio = LiveRadio(engine)

        async def reply():
            entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                cleanup_started.set()
                await cleanup_allowed.wait()

        answering = asyncio.create_task(radio.answer(reply, 0, 10))
        await entered.wait()
        assert await asyncio.wait_for(submit(radio, "critical", 100), 0.1)
        await cleanup_started.wait()
        # A second critical arrival must not cancel the first cancellation's cleanup.
        await submit(radio, "second-critical", 95)
        assert not engine.utterances
        cleanup_allowed.set()
        assert await answering == "interrupted"
        for _ in range(10):
            await asyncio.sleep(0)
        assert [item.text for item in engine.utterances] == ["critical", "second-critical"]
        await radio.aclose()

    asyncio.run(asyncio.wait_for(run(), 2))


def test_critical_cancels_capture_and_waits_until_microphone_is_stopped():
    async def run():
        claimed = asyncio.Event()
        stopped = asyncio.Event()
        engine = RecordingEngine()
        radio = LiveRadio(engine)

        async def capture():
            radio.claim_capture()
            claimed.set()
            try:
                await asyncio.Event().wait()
            finally:
                assert not engine.utterances
                await asyncio.sleep(0)
                stopped.set()
                radio.release_capture()

        capturing = asyncio.create_task(capture())
        await claimed.wait()
        await submit(radio, "routine")
        await asyncio.sleep(0)
        assert not engine.utterances
        await submit(radio, "critical", 100)
        await asyncio.gather(capturing, return_exceptions=True)
        assert stopped.is_set()
        for _ in range(10):
            await asyncio.sleep(0)
        assert [item.text for item in engine.utterances] == ["critical", "routine"]
        await radio.aclose()

    asyncio.run(asyncio.wait_for(run(), 2))


def test_queue_bounded_priority_order_expiry_and_capture_refusal():
    async def run():
        now = datetime.now(UTC)
        clock = [now]
        engine = RecordingEngine()
        radio = LiveRadio(engine, capacity=2, clock=lambda: clock[0])
        radio.claim_capture()
        assert await submit(radio, "low", 10, deadline=now + timedelta(seconds=2))
        assert await submit(radio, "expires", 20, deadline=now + timedelta(seconds=1))
        assert not await submit(radio, "dropped", 5)
        assert await submit(radio, "important", 70)
        with pytest.raises(SpeechInputError):
            radio.claim_capture()
        clock[0] += timedelta(seconds=1.5)
        radio.release_capture()
        for _ in range(10):
            await asyncio.sleep(0)
        assert [item.text for item in engine.utterances] == ["important"]
        await radio.aclose()

    asyncio.run(run())


def test_disconnect_flushes_pending_answers_and_cancels_playing_answer():
    async def run():
        radio = LiveRadio(None)
        started = asyncio.Event()
        cancelled = asyncio.Event()

        async def playing():
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()

        first = asyncio.create_task(radio.answer(playing, 0, 10))
        await started.wait()
        second = asyncio.create_task(radio.answer(playing, 0, 10))
        await asyncio.sleep(0)
        radio.reset(1)
        assert await first == "interrupted"
        assert await second == "invalidated"
        assert cancelled.is_set()
        assert await radio.answer(playing, 0, 10) == "invalidated"
        await radio.aclose()

    asyncio.run(asyncio.wait_for(run(), 2))


def test_failure_does_not_block_following_answers():
    async def run():
        radio = LiveRadio(None)

        async def broken():
            raise RuntimeError("test error")

        async def good():
            pass

        assert await radio.answer(broken, 0, 10) == "failed"
        assert await radio.answer(good, 0, 10) == "completed"
        await radio.aclose()

    asyncio.run(run())
