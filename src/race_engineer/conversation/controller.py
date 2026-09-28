"""Deterministic dialogue decisions and proposed memory transitions."""

from datetime import datetime, timedelta

from race_engineer.config import DialogueConfig
from race_engineer.conversation.context_view import current_opponents, fresh
from race_engineer.conversation.facts import DEFAULT_FACT_CATALOG, FactCatalog
from race_engineer.core.conversation import RaceSnapshot
from race_engineer.core.dialogue import (
    ClarificationKind,
    DecisionOutcome,
    DecisionReason,
    DialogueTurnInput,
    DialogueUpdate,
    GroundedAnswer,
    OpponentReference,
    PendingClarification,
    QueryPart,
    RememberedRequest,
    ResponseDecision,
    SemanticProposal,
    TopicMemory,
    TurnMemory,
)


def failure_decision(
    request: DialogueTurnInput,
    now: datetime,
    reason: DecisionReason,
    *,
    discarded: bool = False,
) -> ResponseDecision:
    return ResponseDecision(
        turn_id=request.turn_id,
        response_id=f"{request.turn_id}:reply",
        session_id=request.context.session_id,
        generation=request.context.generation,
        source_sequence=request.context.source_sequence,
        language=request.reply_language or request.asr_language or request.default_language,
        created_at=now,
        expires_at=request.deadline,
        outcome="discarded" if discarded else "unavailable",
        reason=reason,
    )


class DialogueController:
    def __init__(
        self,
        config: DialogueConfig | None = None,
        fact_catalog: FactCatalog = DEFAULT_FACT_CATALOG,
    ) -> None:
        self._config = config or DialogueConfig()
        self._facts = fact_catalog

    def _requests(
        self,
        request: DialogueTurnInput,
        proposal: SemanticProposal,
    ) -> tuple[tuple[RememberedRequest, ...], bool, ClarificationKind, DecisionReason]:
        """Expand semantic follow-ups without parsing words or reusing numeric facts."""
        state = request.dialogue
        if proposal.abstain:
            return (), False, "topic", "uncertain"
        if proposal.mode == "clarification_answer":
            pending = state.pending
            if pending is None:
                return (), False, "none", "clarification_expired"
            if proposal.requests:
                if pending.requests and tuple(p.query for p in proposal.requests) != tuple(
                    p.query for p in pending.requests
                ):
                    return (), False, pending.kind, "clarification_failed"
                parts = proposal.requests
            else:
                parts = pending.requests
            remembered = tuple(RememberedRequest(part=part) for part in parts)
            if not parts:
                return (), False, pending.kind, "topic_unclear"
            reuse = False
        elif proposal.requests:
            return (
                tuple(RememberedRequest(part=part) for part in proposal.requests),
                False,
                (proposal.clarification),
                "resolved",
            )
        elif proposal.mode in {"follow_up", "correction"}:
            if state.topic is None:
                return (), False, "topic", "topic_unclear"
            remembered, reuse = state.topic.requests, True
        elif proposal.mode == "repeat":
            if not state.responses:
                return (), False, "topic", "topic_unclear"
            remembered, reuse = state.responses[-1].requests, True
        else:
            return (), False, proposal.clarification, "resolved"
        if proposal.reference is not None:
            if sum(item.part.query in {"gap", "gap_trend"} for item in remembered) != 1:
                return (), False, "topic", "topic_unclear"
            remembered = tuple(
                RememberedRequest(
                    part=QueryPart(
                        part_id=item.part.part_id,
                        query=item.part.query,
                        reference=proposal.reference,
                    )
                )
                if item.part.query in {"gap", "gap_trend"}
                else item
                for item in remembered
            )
            reuse = False
        return remembered, reuse, proposal.clarification, "resolved"

    def _anchor(
        self,
        item: RememberedRequest,
        reuse: bool,
        request: DialogueTurnInput,
        opponents: tuple[OpponentReference, ...],
        now: datetime,
    ) -> tuple[RememberedRequest, bool]:
        part = item.part
        if part.query not in {"gap", "gap_trend"}:
            return item, False
        anchor = item.opponent
        if reuse:
            if item.reference_expires_at is None or item.reference_expires_at <= now:
                return RememberedRequest(part=part), True
        elif part.reference == "active":
            topic = request.dialogue.topic
            anchors = (
                {
                    saved.opponent
                    for saved in topic.requests
                    if saved.opponent is not None
                    and saved.reference_expires_at is not None
                    and saved.reference_expires_at > now
                }
                if topic is not None
                else set()
            )
            anchor = next(iter(anchors)) if len(anchors) == 1 else None
            if anchor is None:
                return item, True
        elif part.reference == "unspecified":
            return item, True
        else:
            anchor = next(
                (
                    opponent
                    for opponent in request.context.opponents
                    if opponent.side == part.reference
                ),
                None,
            )
            if anchor is None:
                # The side is explicit; this is unavailable data, not an ambiguous pronoun.
                return item, False
        if anchor is None or anchor not in opponents:
            return RememberedRequest(part=part), True
        return RememberedRequest(
            part=QueryPart(part_id=part.part_id, query=part.query, reference=anchor.side),
            opponent=anchor,
            reference_expires_at=now + timedelta(seconds=self._config.reference_ttl_s),
        ), False

    def _answer(self, item: RememberedRequest, snapshot: RaceSnapshot) -> GroundedAnswer:
        part = item.part
        answer = self._facts.resolve_part(snapshot, part, item.opponent)
        return GroundedAnswer(
            **answer.model_dump(),
            part_id=part.part_id,
            opponent=item.opponent,
            source_sequence=snapshot.context.frame.sequence,
        )

    def decide(
        self,
        request: DialogueTurnInput,
        proposal: SemanticProposal,
        snapshot: RaceSnapshot,
        now: datetime,
    ) -> ResponseDecision:
        if snapshot.context.frame.session_id != request.context.session_id:
            return failure_decision(request, now, "session_changed", discarded=True)
        if now >= request.deadline:
            return failure_decision(request, now, "deadline", discarded=True)
        if not fresh(snapshot, self._config):
            return failure_decision(request, now, "stale_snapshot")
        remembered, reuse, clarification, reason = self._requests(request, proposal)
        answers: list[GroundedAnswer] = []
        unresolved: list[QueryPart] = []
        resolved: list[RememberedRequest] = []
        opponents = current_opponents(snapshot)
        for item in remembered:
            bound, unclear = self._anchor(item, reuse, request, opponents, now)
            if unclear:
                part = QueryPart(
                    part_id=item.part.part_id, query=item.part.query, reference="unspecified"
                )
                unresolved.append(part)
                resolved.append(RememberedRequest(part=part))
                clarification, reason = "opponent", "reference_unclear"
            else:
                resolved.append(bound)
                answers.append(self._answer(bound, snapshot))
        if (
            clarification != "none"
            and request.dialogue.pending is not None
            and proposal.mode in {"clarification_answer", "follow_up"}
        ):
            clarification, reason = "none", "clarification_failed"
        language = request.reply_language or proposal.language
        acts = proposal.acts
        if proposal.mode == "correction" and not acts:
            acts = ("acknowledge",)
        if proposal.mode == "repeat" and request.dialogue.responses:
            acts = request.dialogue.responses[-1].acts
        available = sum(answer.status == "available" for answer in answers)
        outcome: DecisionOutcome
        if available:
            outcome = (
                "partial"
                if (unresolved or available != len(answers) or clarification != "none")
                else "answered"
            )
        elif clarification != "none":
            outcome = "clarify"
        elif answers or unresolved or reason != "resolved":
            outcome = "unavailable"
        elif "close" in acts:
            outcome = "no_reply"
        elif "acknowledge" in acts:
            outcome = "acknowledge"
        else:
            outcome, reason = "unavailable", "topic_unclear"
        if reason == "resolved" and any(answer.status != "available" for answer in answers):
            reason = (
                "missing_facts" if any(a.status == "missing" for a in answers) else "unsupported"
            )
        # An intervening social remark need not erase an unanswered clarification.
        # Keep the original expiry; acknowledging a remark does not renew it.
        pending = (
            request.dialogue.pending
            if (
                proposal.mode == "request"
                and not proposal.requests
                and proposal.acts == ("acknowledge",)
                and clarification == "none"
            )
            else None
        )
        if clarification != "none":
            pending = PendingClarification(
                origin_turn_id=request.turn_id,
                kind=clarification,
                requests=tuple(unresolved),
                expires_at=now + timedelta(seconds=self._config.clarification_ttl_s),
            )
        topic = request.dialogue.topic
        if resolved:
            topic = TopicMemory(
                requests=tuple(resolved),
                expires_at=now + timedelta(seconds=self._config.topic_ttl_s),
            )
        elif proposal.mode == "request" and not acts:
            topic = None
        if reason == "clarification_failed":
            topic = None
        update = DialogueUpdate(
            expected_revision=request.dialogue.revision,
            topic=topic,
            pending=pending,
            turn=TurnMemory(
                turn_id=request.turn_id,
                received_at=request.received_at,
                language=language,
                question=request.question if self._config.retain_utterances else None,
                requests=tuple(resolved),
                acts=acts,
            ),
        )
        return ResponseDecision(
            turn_id=request.turn_id,
            response_id=f"{request.turn_id}:reply",
            session_id=request.context.session_id,
            generation=request.context.generation,
            source_sequence=snapshot.context.frame.sequence,
            language=language,
            created_at=now,
            expires_at=request.deadline,
            outcome=outcome,
            reason=reason,
            answers=tuple(answers),
            acts=acts,
            clarification=clarification,
            unresolved=tuple(unresolved),
            update=update,
        )
