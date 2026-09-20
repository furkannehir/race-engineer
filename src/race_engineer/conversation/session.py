"""Bounded conversational memory and fresh, traceable fact retrieval."""

import asyncio
import logging
from collections import deque
from collections.abc import Callable

from race_engineer.config import ConversationConfig
from race_engineer.conversation.answers import MESSAGES, render, retrieve
from race_engineer.conversation.local_model import ConversationModelError
from race_engineer.core.conversation import (
    ConversationReply,
    ConversationRequest,
    ConversationTurn,
    RaceSnapshot,
    RadioLanguage,
)
from race_engineer.core.interfaces import ConversationPlanner

_LOGGER = logging.getLogger(__name__)


class ConversationSession:
    def __init__(
        self,
        planner: ConversationPlanner,
        snapshot: Callable[[], RaceSnapshot],
        config: ConversationConfig | None = None,
        *,
        generation: Callable[[], int] = lambda: 0,
    ) -> None:
        self._config = config or ConversationConfig()
        self._planner = planner
        self._snapshot = snapshot
        self._generation = generation
        self._session_generation: int | None = None
        self._history: deque[ConversationTurn] = deque(maxlen=self._config.history_turns)
        self._session_id: str | None = None
        self._lock = asyncio.Lock()

    def reset(self) -> None:
        self._history.clear()
        self._session_id = None
        self._session_generation = None

    def _fresh(self, snapshot: RaceSnapshot) -> bool:
        age = (snapshot.as_of - snapshot.context.frame.observed_at).total_seconds()
        return 0 <= age <= self._config.max_snapshot_age_s

    @staticmethod
    def _failure(
        snapshot: RaceSnapshot, language: RadioLanguage, reason: str, *, model: bool = False
    ) -> ConversationReply:
        return ConversationReply(
            language=language,
            text=MESSAGES[language]["model_error" if model else reason],
            status="model_error" if model else "unavailable",
            session_id=snapshot.context.frame.session_id,
            source_sequence=snapshot.context.frame.sequence,
            mode=snapshot.mode,
            reason=reason,
        )

    async def ask(
        self, question: str, *, reply_language: RadioLanguage | None = None
    ) -> ConversationReply:
        async with self._lock:
            snapshot = self._snapshot()
            session_id = snapshot.context.frame.session_id
            generation = self._generation()
            if self._session_id != session_id or self._session_generation != generation:
                self.reset()
                self._session_id = session_id
                self._session_generation = generation
            language = reply_language or (
                self._history[-1].plan.language if self._history else self._config.default_language
            )
            request = ConversationRequest(
                question=question.strip(),
                history=tuple(self._history),
                default_language=language,
                reply_language=reply_language,
            )
            if not self._fresh(snapshot):
                return self._failure(snapshot, language, "stale_snapshot")
            try:
                plan = await self._planner.plan(request)
            except ConversationModelError as error:
                _LOGGER.warning(
                    "local conversation model failed",
                    extra={"event": "conversation_failed", "reason": str(error)},
                )
                return self._failure(snapshot, language, str(error), model=True)

            if reply_language is not None:
                plan = plan.model_copy(update={"language": reply_language})
            # The model only resolves meaning. Values always come from a new snapshot.
            current = self._snapshot()
            if current.context.frame.session_id != session_id or self._generation() != generation:
                self.reset()
                return self._failure(current, plan.language, "session_changed")
            if not self._fresh(current):
                return self._failure(current, plan.language, "stale_snapshot")
            self._history.append(ConversationTurn(question=request.question, plan=plan))
            answers = tuple(retrieve(current.context, query) for query in plan.queries)
            if plan.clarification != "none":
                text = MESSAGES[plan.language][plan.clarification]
                status = "clarification"
            else:
                text = " ".join(render(answer, plan.language) for answer in answers)
                status = (
                    "answered"
                    if any(answer.status == "available" for answer in answers)
                    else "unavailable"
                )
            return ConversationReply.model_validate(
                {
                    "language": plan.language,
                    "text": text,
                    "status": status,
                    "session_id": session_id,
                    "source_sequence": current.context.frame.sequence,
                    "mode": current.mode,
                    "answers": answers,
                }
            )
