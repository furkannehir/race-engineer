"""CE-04 session foundation, independently testable without live audio or model packages."""

import asyncio
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from uuid import uuid4

from pydantic import TypeAdapter

from race_engineer.config import DialogueConfig
from race_engineer.conversation.context_view import (
    ConversationContextAssembler,
    current_opponents,
    fresh,
)
from race_engineer.conversation.controller import DialogueController, failure_decision
from race_engineer.conversation.facts import DEFAULT_FACT_CATALOG, FactCatalog
from race_engineer.conversation.live import LiveTelemetryUnavailable
from race_engineer.conversation.state import apply_delivery, commit_decision, expire_state
from race_engineer.core.contracts import UtcDatetime
from race_engineer.core.conversation import RaceSnapshot, RadioLanguage
from race_engineer.core.dialogue import (
    DecisionReason,
    DeliveryEvent,
    DialogueState,
    OpponentReference,
    OutputMode,
    ResponseDecision,
    SemanticJudgeError,
    SemanticProposal,
)
from race_engineer.core.interfaces import SemanticJudge

_UTC_DATE = TypeAdapter(UtcDatetime)


class DialogueSession:
    """Single-event-loop owner of state. At most one judge task, with no pending queue."""

    def __init__(
        self,
        judge: SemanticJudge,
        snapshot: Callable[[], RaceSnapshot],
        config: DialogueConfig | None = None,
        *,
        generation: Callable[[], int] = lambda: 0,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
        fact_catalog: FactCatalog = DEFAULT_FACT_CATALOG,
    ) -> None:
        self._judge = judge
        self._snapshot = snapshot
        self._config = config or DialogueConfig()
        self._generation = generation
        self._clock = clock
        self._assembler = ConversationContextAssembler(self._config, fact_catalog)
        self._controller = DialogueController(self._config, fact_catalog)
        self._state = DialogueState()
        self._opponents: tuple[OpponentReference, ...] = ()
        self._last_frame: tuple[int, float] | None = None
        self._active_turn: str | None = None
        self._judge_task: asyncio.Task[SemanticProposal] | None = None
        self._invalidated: asyncio.Event | None = None
        self._invalid_reason: DecisionReason = "cancelled"

    @property
    def state(self) -> DialogueState:
        self._state = expire_state(self._state, self._now(), self._opponents, self._config)
        return self._state

    def _now(self) -> datetime:
        return _UTC_DATE.validate_python(self._clock())

    def cancel(self) -> None:
        """Invalidate in-flight interpretation; already committed meaning is retained."""
        self._invalid_reason = "cancelled"
        if self._invalidated is not None:
            self._invalidated.set()
        if self._judge_task is not None and not self._judge_task.done():
            self._judge_task.cancel()

    def reset(self) -> None:
        """Explicit reset/seek invalidates even when the simulator generation is unchanged."""
        self.cancel()
        self._invalid_reason = "session_changed"
        self._state = DialogueState(revision=self._state.revision + 1)
        self._opponents = ()
        self._last_frame = None

    def _synchronize(self, snapshot: RaceSnapshot, now: datetime) -> None:
        frame = snapshot.context.frame
        generation = self._generation()
        changed = (self._state.session_id, self._state.generation) != (frame.session_id, generation)
        rewind = self._last_frame is not None and (
            frame.sequence < self._last_frame[0] or frame.session_time_s < self._last_frame[1]
        )
        if changed or rewind:
            self.reset()
            self._state = DialogueState(
                session_id=frame.session_id,
                generation=generation,
                revision=self._state.revision,
            )
        self._last_frame = (frame.sequence, frame.session_time_s)
        self._opponents = current_opponents(snapshot)
        self._state = expire_state(self._state, now, self._opponents, self._config)

    def record_delivery(self, event: DeliveryEvent) -> bool:
        """Returns true only for a new valid transition; duplicate/old events are inert."""
        now = self._now()
        try:
            snapshot = self._snapshot()
        except LiveTelemetryUnavailable:
            self.reset()
            return False
        self._synchronize(snapshot, now)
        updated = apply_delivery(self._state, event, now)
        changed = updated != self._state
        self._state = updated
        return changed

    def _early_failure(
        self,
        turn_id: str,
        now: datetime,
        deadline: datetime,
        reason: DecisionReason,
        language: RadioLanguage,
        *,
        discarded: bool = False,
    ) -> ResponseDecision:
        return ResponseDecision(
            turn_id=turn_id,
            response_id=f"{turn_id}:reply",
            session_id=self._state.session_id or "unavailable",
            generation=self._state.generation,
            source_sequence=self._last_frame[0] if self._last_frame else 0,
            language=language,
            created_at=now,
            expires_at=deadline,
            outcome="discarded" if discarded else "unavailable",
            reason=reason,
        )

    def _reap(self, task: asyncio.Task[SemanticProposal]) -> None:
        # A cancelled/timed-out worker may finish later. Consume it, never commit its output.
        if not task.cancelled():
            task.exception()
        if self._judge_task is task:
            self._judge_task = None

    async def ask(
        self,
        question: str,
        *,
        asr_language: RadioLanguage | None = None,
        reply_language: RadioLanguage | None = None,
        output_mode: OutputMode = "speech",
        turn_started_at: datetime | None = None,
        origin_generation: int | None = None,
    ) -> ResponseDecision:
        question = question.strip()
        if not question or len(question) > 1000:
            raise ValueError("question must contain 1 to 1000 characters")
        if output_mode not in {"speech", "text"}:
            raise ValueError("unsupported output mode")
        now = self._now()
        started = _UTC_DATE.validate_python(turn_started_at) if turn_started_at is not None else now
        if started > now:
            raise ValueError("turn start cannot be in the future")
        deadline = started + timedelta(seconds=self._config.turn_timeout_s)
        turn_id = uuid4().hex
        language = reply_language or asr_language or self._config.default_language
        if self._active_turn is not None or (
            self._judge_task is not None and not self._judge_task.done()
        ):
            return self._early_failure(turn_id, now, deadline, "busy", language, discarded=True)
        if now >= deadline:
            return self._early_failure(turn_id, now, deadline, "deadline", language, discarded=True)
        try:
            snapshot = self._snapshot()
        except LiveTelemetryUnavailable:
            self.reset()
            return self._early_failure(turn_id, now, deadline, "stale_snapshot", language)
        self._synchronize(snapshot, now)
        if origin_generation is not None and origin_generation != self._state.generation:
            return self._early_failure(
                turn_id,
                now,
                deadline,
                "session_changed",
                language,
                discarded=True,
            )
        if not fresh(snapshot, self._config):
            return self._early_failure(turn_id, now, deadline, "stale_snapshot", language)
        request = self._assembler.assemble(
            snapshot,
            self._state,
            turn_id=turn_id,
            question=question,
            received_at=now,
            deadline=deadline,
            asr_language=asr_language,
            reply_language=reply_language,
        )
        self._active_turn = turn_id
        invalidated = asyncio.Event()
        self._invalidated = invalidated
        task = asyncio.create_task(self._judge.judge(request), name="semantic-judge")
        self._judge_task = task
        task.add_done_callback(self._reap)
        invalidation = asyncio.create_task(invalidated.wait(), name="dialogue-invalidation")
        try:
            done, _ = await asyncio.wait(
                {task, invalidation},
                timeout=max(0, (deadline - self._now()).total_seconds()),
                return_when=asyncio.FIRST_COMPLETED,
            )
            if invalidated.is_set():
                task.cancel()
                return failure_decision(request, self._now(), self._invalid_reason, discarded=True)
            if task not in done:
                task.cancel()
                return failure_decision(request, self._now(), "deadline", discarded=True)
            if task.cancelled():
                return failure_decision(request, self._now(), "cancelled", discarded=True)
            try:
                # Revalidate even a constructed typed instance at this untrusted boundary.
                proposal = SemanticProposal.model_validate(task.result().model_dump())
            except Exception:
                return failure_decision(request, self._now(), "model_error")
            now = self._now()
            if now >= deadline:
                return failure_decision(request, now, "deadline", discarded=True)
            try:
                current = self._snapshot()
            except LiveTelemetryUnavailable:
                self.reset()
                return failure_decision(request, now, "session_changed", discarded=True)
            self._synchronize(current, now)
            if invalidated.is_set():
                return failure_decision(request, now, "session_changed", discarded=True)
            if self._state.revision != request.dialogue.revision:
                return failure_decision(request, now, "state_changed", discarded=True)
            decision = self._controller.decide(request, proposal, current, now)
            self._state = commit_decision(self._state, decision, output_mode, self._config)
            return decision
        except asyncio.CancelledError:
            task.cancel()
            raise
        finally:
            invalidation.cancel()
            if self._active_turn == turn_id:
                self._active_turn = None
                self._invalidated = None
            await asyncio.gather(invalidation, return_exceptions=True)

    async def aclose(self) -> None:
        self.reset()
        task = self._judge_task
        if task is not None and not task.done():
            done, _ = await asyncio.wait({task}, timeout=1.0)
            if not done:
                raise SemanticJudgeError("judge_shutdown_timeout")
