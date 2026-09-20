"""One bounded audio scheduler for live automatic calls, answers, and microphone capture."""

import asyncio
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from race_engineer.core.contracts import SpeechIntent, Utterance
from race_engineer.core.enums import InterruptionPolicy, PlaybackStatus
from race_engineer.core.interfaces import TextToSpeechEngine
from race_engineer.core.speech_input import SpeechInputError
from race_engineer.core.speech_output import SpeechOutputError

_LOGGER = logging.getLogger(__name__)


@dataclass
class _Job:
    identifier: str
    priority: int
    sequence: int
    epoch: int
    deadline: datetime
    action: Callable[[], Awaitable[None]]
    done: asyncio.Future[str]


class LiveRadio:
    def __init__(
        self,
        engine: TextToSpeechEngine | None,
        *,
        capacity: int = 8,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        if capacity < 1:
            raise ValueError("radio capacity must be positive")
        self._engine = engine
        self._capacity = capacity
        self._clock = clock
        self._epoch = 0
        self._sequence = 0
        self._pending: list[_Job] = []
        self._current: _Job | None = None
        self._active: asyncio.Task[None] | None = None
        self._capture: asyncio.Task[object] | None = None
        self._worker: asyncio.Task[None] | None = None
        self._wakeup = asyncio.Event()
        self._closing = False

    @staticmethod
    def _cancel(task: asyncio.Task[object] | None) -> None:
        if task is not None and not task.done() and not task.cancelling():
            task.cancel()

    def _finish(self, job: _Job, outcome: str) -> None:
        if not job.done.done():
            job.done.set_result(outcome)
        _LOGGER.info(
            "live radio result",
            extra={
                "event": "live_radio_result",
                "request_id": job.identifier,
                "outcome": outcome,
            },
        )

    def reset(self, epoch: int) -> None:
        self._epoch = epoch
        for job in self._pending:
            self._finish(job, "invalidated")
        self._pending.clear()
        self._cancel(self._active)
        self._cancel(self._capture)
        self._wakeup.set()

    def claim_capture(self) -> None:
        if self._closing or self._current is not None or self._pending or self._capture is not None:
            raise SpeechInputError("radio_busy_release_and_retry")
        self._capture = asyncio.current_task()

    def release_capture(self) -> None:
        if self._capture is asyncio.current_task():
            self._capture = None
            self._wakeup.set()

    def _enqueue(
        self,
        identifier: str,
        priority: int,
        epoch: int,
        deadline: datetime,
        action: Callable[[], Awaitable[None]],
        *,
        interruption: InterruptionPolicy,
    ) -> _Job:
        future: asyncio.Future[str] = asyncio.get_running_loop().create_future()
        job = _Job(identifier, priority, self._sequence, epoch, deadline, action, future)
        self._sequence += 1
        if self._closing or epoch != self._epoch:
            self._finish(job, "invalidated")
            return job
        if deadline <= self._clock():
            self._finish(job, "expired")
            return job
        if len(self._pending) >= self._capacity:
            worst = min(self._pending, key=lambda item: (item.priority, -item.sequence))
            if priority <= worst.priority:
                self._finish(job, "dropped")
                return job
            self._pending.remove(worst)
            self._finish(worst, "superseded")
        self._pending.append(job)
        self._pending.sort(key=lambda item: (-item.priority, item.sequence))
        if (
            self._current is not None
            and self._active is not None
            and (
                interruption is InterruptionPolicy.INTERRUPT_ANY
                or (
                    interruption is InterruptionPolicy.INTERRUPT_LOWER_PRIORITY
                    and priority > self._current.priority
                )
            )
            and not self._active.done()
        ):
            self._cancel(self._active)
        # Critical automatic calls take the mic away, but playback waits for stream cleanup.
        if priority >= 90 and self._capture is not None and not self._capture.done():
            self._cancel(self._capture)
        if self._worker is None:
            self._worker = asyncio.create_task(self._run(), name="live-radio")
        self._wakeup.set()
        return job

    async def submit(self, intent: SpeechIntent, utterance: Utterance, epoch: int) -> bool:
        if intent.intent_id != utterance.intent_id:
            raise ValueError("speech intent and utterance IDs do not match")
        if self._engine is None:
            return False

        async def play() -> None:
            assert self._engine is not None
            result = await self._engine.speak(utterance)
            if (
                result.intent_id != intent.intent_id
                or result.status is not PlaybackStatus.COMPLETED
            ):
                raise SpeechOutputError("automatic_playback_failed")

        job = self._enqueue(
            intent.intent_id,
            intent.priority,
            epoch,
            intent.deadline,
            play,
            interruption=intent.interruption_policy,
        )
        # No playback/cancellation waits on the telemetry loop.
        return not job.done.done()

    async def answer(self, action: Callable[[], Awaitable[None]], epoch: int, ttl_s: float) -> str:
        job = self._enqueue(
            f"answer-{self._sequence}",
            50,
            epoch,
            self._clock() + timedelta(seconds=ttl_s),
            action,
            interruption=InterruptionPolicy.NEVER,
        )
        try:
            return await job.done
        except asyncio.CancelledError:
            if job in self._pending:
                self._pending.remove(job)
            if self._current is job and self._active is not None and not self._active.done():
                self._cancel(self._active)
            raise

    async def _run(self) -> None:
        while not self._closing:
            if self._capture is not None or not self._pending:
                self._wakeup.clear()
                await self._wakeup.wait()
                continue
            job = self._pending.pop(0)
            if job.epoch != self._epoch or job.deadline <= self._clock():
                self._finish(job, "expired")
                continue
            self._current = job

            async def invoke(action: Callable[[], Awaitable[None]] = job.action) -> None:
                await action()

            self._active = asyncio.create_task(invoke(), name="live-radio-playback")
            outcome = "completed"
            try:
                await self._active
            except asyncio.CancelledError:
                task = asyncio.current_task()
                if task is not None and task.cancelling():
                    raise
                outcome = "interrupted"
            except Exception as error:
                outcome = (
                    "expired"
                    if isinstance(error, SpeechOutputError) and str(error) == "tts_reply_expired"
                    else "failed"
                )
                _LOGGER.warning(
                    "live speech failed; telemetry continues",
                    extra={
                        "event": "live_speech_failed",
                        "reason": type(error).__name__,
                    },
                )
            finally:
                self._finish(job, outcome)
                self._current = None
                self._active = None

    async def aclose(self) -> None:
        self._closing = True
        self.reset(self._epoch + 1)
        if self._capture is not None:
            await asyncio.gather(self._capture, return_exceptions=True)
        if self._worker is not None:
            await self._worker
