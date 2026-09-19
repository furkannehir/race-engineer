import asyncio
from datetime import UTC, datetime, timedelta

from race_engineer.core.contracts import PlaybackResult, SpeechIntent, Utterance
from race_engineer.core.enums import (
    InterruptionPolicy,
    MessageCategory,
    PlaybackStatus,
)
from race_engineer.tts import SpeechPlaybackQueue


def make_intent(
    intent_id: str,
    *,
    priority: int,
    deadline: datetime,
    interruption_policy: InterruptionPolicy = InterruptionPolicy.NEVER,
) -> SpeechIntent:
    return SpeechIntent(
        intent_id=intent_id,
        category=MessageCategory.SAFETY,
        priority=priority,
        deadline=deadline,
        interruption_policy=interruption_policy,
    )


class RecordingEngine:
    def __init__(self) -> None:
        self.utterances: list[Utterance] = []
        self.cancelled = False

    async def speak(self, utterance: Utterance) -> PlaybackResult:
        self.utterances.append(utterance)
        return PlaybackResult(intent_id=utterance.intent_id, status=PlaybackStatus.COMPLETED)

    async def cancel(self) -> None:
        self.cancelled = True


def test_expired_speech_is_not_played() -> None:
    now = datetime(2026, 9, 19, tzinfo=UTC)
    engine = RecordingEngine()
    results: list[PlaybackResult] = []

    async def scenario() -> bool:
        queue = SpeechPlaybackQueue(
            engine, capacity=2, result_sink=results.append, clock=lambda: now
        )
        accepted = await queue.submit(
            make_intent("expired", priority=30, deadline=now),
            Utterance(intent_id="expired", text="Stale message"),
        )
        await queue.aclose(drain=True)
        return accepted

    assert asyncio.run(scenario()) is False
    assert engine.utterances == []
    assert [result.status for result in results] == [PlaybackStatus.EXPIRED]


def test_critical_speech_interrupts_lower_priority_playback() -> None:
    results: list[PlaybackResult] = []

    async def scenario() -> tuple[list[str], bool]:
        started = asyncio.Event()
        cancelled = asyncio.Event()

        class InterruptibleEngine(RecordingEngine):
            async def speak(self, utterance: Utterance) -> PlaybackResult:
                self.utterances.append(utterance)
                if utterance.intent_id == "routine":
                    started.set()
                    await cancelled.wait()
                    return PlaybackResult(
                        intent_id=utterance.intent_id,
                        status=PlaybackStatus.CANCELLED,
                    )
                return PlaybackResult(
                    intent_id=utterance.intent_id,
                    status=PlaybackStatus.COMPLETED,
                )

            async def cancel(self) -> None:
                self.cancelled = True
                cancelled.set()

        engine = InterruptibleEngine()
        queue = SpeechPlaybackQueue(engine, capacity=2, result_sink=results.append)
        future = datetime.now(UTC) + timedelta(seconds=30)
        await queue.submit(
            make_intent("routine", priority=30, deadline=future),
            Utterance(intent_id="routine", text="Position five"),
        )
        await started.wait()
        await queue.submit(
            make_intent(
                "critical",
                priority=100,
                deadline=future,
                interruption_policy=InterruptionPolicy.INTERRUPT_ANY,
            ),
            Utterance(intent_id="critical", text="Caution, slow down"),
        )
        await queue.aclose(drain=True)
        return [utterance.intent_id for utterance in engine.utterances], engine.cancelled

    spoken, cancelled = asyncio.run(scenario())

    assert spoken == ["routine", "critical"]
    assert cancelled is True
    assert [result.status for result in results] == [
        PlaybackStatus.CANCELLED,
        PlaybackStatus.COMPLETED,
    ]


def test_more_important_speech_replaces_lower_priority_work_in_full_queue() -> None:
    async def scenario() -> list[str]:
        started = asyncio.Event()
        release = asyncio.Event()

        class HoldingEngine(RecordingEngine):
            async def speak(self, utterance: Utterance) -> PlaybackResult:
                self.utterances.append(utterance)
                if utterance.intent_id == "current":
                    started.set()
                    await release.wait()
                return PlaybackResult(
                    intent_id=utterance.intent_id,
                    status=PlaybackStatus.COMPLETED,
                )

        engine = HoldingEngine()
        queue = SpeechPlaybackQueue(engine, capacity=1)
        future = datetime.now(UTC) + timedelta(seconds=30)
        await queue.submit(
            make_intent("current", priority=50, deadline=future),
            Utterance(intent_id="current", text="Position four"),
        )
        await started.wait()
        await queue.submit(
            make_intent("low", priority=30, deadline=future),
            Utterance(intent_id="low", text="Routine update"),
        )
        await queue.submit(
            make_intent("high", priority=80, deadline=future),
            Utterance(intent_id="high", text="Blue flag"),
        )
        release.set()
        await queue.aclose(drain=True)
        return [utterance.intent_id for utterance in engine.utterances]

    assert asyncio.run(scenario()) == ["current", "high"]
