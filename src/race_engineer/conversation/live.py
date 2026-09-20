"""Latest-only live snapshots with explicit connection generations and wall-clock freshness."""

from collections.abc import Callable
from datetime import UTC, datetime

from race_engineer.conversation.answers import render, retrieve
from race_engineer.core.contracts import RaceContext
from race_engineer.core.conversation import ConversationReply, RaceSnapshot


class LiveTelemetryUnavailable(Exception):
    pass


class LiveRaceState:
    def __init__(
        self, max_age_s: float, *, clock: Callable[[], datetime] = lambda: datetime.now(UTC)
    ) -> None:
        self._clock = clock
        self._max_age_s = max_age_s
        self._context: RaceContext | None = None
        self.epoch = 0

    @property
    def available(self) -> bool:
        if self._context is None:
            return False
        age = (self._clock() - self._context.frame.observed_at).total_seconds()
        return 0 <= age <= self._max_age_s and not self._context.frame.is_replay

    def invalidate(self) -> None:
        if self._context is not None:
            self.epoch += 1
            self._context = None

    def update(self, context: RaceContext) -> None:
        if context.frame.is_replay:
            self.invalidate()
            return
        if self._context is not None and (
            not self.available
            or context.frame.session_id != self._context.frame.session_id
            or context.frame.sequence <= self._context.frame.sequence
        ):
            self.invalidate()
        self._context = context

    def snapshot(self) -> RaceSnapshot:
        if not self.available or self._context is None:
            raise LiveTelemetryUnavailable("live_telemetry_unavailable")
        return RaceSnapshot(context=self._context, as_of=self._clock(), mode="live")

    def refresh(self, reply: ConversationReply, epoch: int) -> ConversationReply:
        snapshot = self.snapshot()
        frame = snapshot.context.frame
        if epoch != self.epoch or reply.session_id != frame.session_id:
            raise LiveTelemetryUnavailable("live_session_changed")
        if not reply.answers:
            return reply.model_copy(update={"source_sequence": frame.sequence})
        answers = tuple(retrieve(snapshot.context, answer.query) for answer in reply.answers)
        return reply.model_copy(
            update={
                "answers": answers,
                "text": " ".join(render(answer, reply.language) for answer in answers),
                "status": "answered"
                if any(a.status == "available" for a in answers)
                else "unavailable",
                "source_sequence": frame.sequence,
            }
        )
