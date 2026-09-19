"""Bounded, expiring, priority-aware speech playback queue."""

import asyncio
import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime

from race_engineer.core.contracts import PlaybackResult, SpeechIntent, Utterance
from race_engineer.core.enums import InterruptionPolicy, PlaybackStatus
from race_engineer.core.interfaces import TextToSpeechEngine

_LOGGER = logging.getLogger(__name__)

PlaybackResultSink = Callable[[PlaybackResult], None]
Clock = Callable[[], datetime]


def _utc_now() -> datetime:
    return datetime.now(UTC)


@dataclass(frozen=True, slots=True)
class _PlaybackRequest:
    intent: SpeechIntent
    utterance: Utterance
    sequence: int
    enqueued_at: datetime


class SpeechPlaybackQueue:
    """Runs speech off the telemetry path and keeps pending work bounded."""

    def __init__(
        self,
        engine: TextToSpeechEngine,
        capacity: int,
        *,
        result_sink: PlaybackResultSink | None = None,
        clock: Clock = _utc_now,
    ) -> None:
        if capacity < 1:
            raise ValueError("speech queue capacity must be positive")
        self._engine = engine
        self._capacity = capacity
        self._result_sink = result_sink
        self._clock = clock
        self._pending: list[_PlaybackRequest] = []
        self._current: _PlaybackRequest | None = None
        self._next_sequence = 0
        self._wakeup = asyncio.Event()
        self._worker: asyncio.Task[None] | None = None
        self._closing = False

    def start(self) -> None:
        if self._worker is None:
            self._worker = asyncio.create_task(self._run(), name="speech-playback")

    def _emit(self, request: _PlaybackRequest, result: PlaybackResult) -> None:
        queue_wait_ms = (
            max(0.0, (result.started_at - request.enqueued_at).total_seconds() * 1_000)
            if result.started_at is not None
            else None
        )
        playback_ms = (
            max(0.0, (result.finished_at - result.started_at).total_seconds() * 1_000)
            if result.started_at is not None and result.finished_at is not None
            else None
        )
        _LOGGER.info(
            "speech playback finished",
            extra={
                "event": "playback_result",
                "intent_id": result.intent_id,
                "status": result.status.value,
                "error_code": result.error_code,
                "queue_wait_ms": queue_wait_ms,
                "playback_ms": playback_ms,
            },
        )
        if self._result_sink is not None:
            try:
                self._result_sink(result)
            except Exception as error:
                _LOGGER.exception(
                    "playback result sink failed; speech queue will continue",
                    extra={
                        "event": "playback_result_sink_failed",
                        "intent_id": result.intent_id,
                        "reason": type(error).__name__,
                    },
                )

    def _expired(self, request: _PlaybackRequest) -> None:
        self._emit(
            request,
            PlaybackResult(
                intent_id=request.intent.intent_id,
                status=PlaybackStatus.EXPIRED,
                finished_at=self._clock(),
            ),
        )

    @staticmethod
    def _sort_key(request: _PlaybackRequest) -> tuple[int, int]:
        return (-request.intent.priority, request.sequence)

    async def submit(self, intent: SpeechIntent, utterance: Utterance) -> bool:
        if self._closing:
            raise RuntimeError("speech queue is closing")
        if utterance.intent_id != intent.intent_id:
            raise ValueError("utterance and speech intent IDs do not match")
        self.start()

        request = _PlaybackRequest(intent, utterance, self._next_sequence, self._clock())
        self._next_sequence += 1
        if intent.deadline <= self._clock():
            self._expired(request)
            return False

        interrupted = False
        if self._current is not None:
            interrupted = intent.interruption_policy is InterruptionPolicy.INTERRUPT_ANY or (
                intent.interruption_policy is InterruptionPolicy.INTERRUPT_LOWER_PRIORITY
                and intent.priority > self._current.intent.priority
            )
        if interrupted:
            await self._engine.cancel()

        if len(self._pending) >= self._capacity:
            worst = max(self._pending, key=lambda pending: self._sort_key(pending))
            if self._sort_key(request) < self._sort_key(worst):
                self._pending.remove(worst)
                _LOGGER.warning(
                    "lower-priority speech dropped from full queue",
                    extra={
                        "event": "playback_dropped",
                        "intent_id": worst.intent.intent_id,
                        "reason": "superseded_in_queue",
                    },
                )
            else:
                _LOGGER.warning(
                    "speech dropped because playback queue is full",
                    extra={
                        "event": "playback_dropped",
                        "intent_id": intent.intent_id,
                        "reason": "queue_full",
                    },
                )
                return False

        self._pending.append(request)
        self._pending.sort(key=self._sort_key)
        self._wakeup.set()
        return True

    async def _run(self) -> None:
        while True:
            if not self._pending:
                if self._closing:
                    return
                self._wakeup.clear()
                await self._wakeup.wait()
                continue

            request = self._pending.pop(0)
            if request.intent.deadline <= self._clock():
                self._expired(request)
                continue

            self._current = request
            try:
                try:
                    result = await self._engine.speak(request.utterance)
                    if result.intent_id != request.intent.intent_id:
                        raise ValueError("TTS returned a result for a different intent")
                except Exception as error:
                    _LOGGER.exception(
                        "text-to-speech failed; playback queue will continue",
                        extra={
                            "event": "tts_failed",
                            "intent_id": request.intent.intent_id,
                            "reason": type(error).__name__,
                        },
                    )
                    result = PlaybackResult(
                        intent_id=request.intent.intent_id,
                        status=PlaybackStatus.FAILED,
                        error_code="engine_exception",
                    )
                self._emit(request, result)
            finally:
                self._current = None

    async def aclose(self, *, drain: bool) -> None:
        self._closing = True
        if not drain:
            self._pending.clear()
            await self._engine.cancel()
        self._wakeup.set()
        if self._worker is not None:
            await asyncio.shield(self._worker)
