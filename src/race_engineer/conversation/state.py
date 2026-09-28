"""Pure bounded memory reducers. DialogueSession owns the resulting state."""

from datetime import datetime, timedelta

from race_engineer.config import DialogueConfig
from race_engineer.core.dialogue import (
    DeliveryEvent,
    DialogueState,
    OpponentReference,
    OutputMode,
    ResponseDecision,
    ResponseMemory,
)

TERMINAL_DELIVERIES = frozenset({"completed", "interrupted", "cancelled", "expired", "failed"})


def expire_state(
    state: DialogueState,
    now: datetime,
    opponents: tuple[OpponentReference, ...],
    config: DialogueConfig,
) -> DialogueState:
    topic = state.topic
    if topic is not None and (
        topic.expires_at <= now
        or any(
            item.opponent is not None
            and (
                item.opponent not in opponents
                or item.reference_expires_at is None
                or item.reference_expires_at <= now
            )
            for item in topic.requests
        )
    ):
        topic = None
    pending = state.pending
    if pending is not None and pending.expires_at <= now:
        pending = None
    cutoff = now - timedelta(seconds=config.history_ttl_s)
    history = tuple(turn for turn in state.history if turn.received_at > cutoff)
    responses = tuple(
        response.model_copy(update={"delivery": "expired", "delivery_at": now})
        if response.delivery in {"accepted", "queued"} and response.expires_at <= now
        else response
        for response in state.responses
        if response.created_at > cutoff
    )
    if (topic, pending, history, responses) == (
        state.topic,
        state.pending,
        state.history,
        state.responses,
    ):
        return state
    return state.model_copy(
        update={
            "topic": topic,
            "pending": pending,
            "history": history,
            "responses": responses,
            "revision": state.revision + 1,
        }
    )


def commit_decision(
    state: DialogueState,
    decision: ResponseDecision,
    output_mode: OutputMode,
    config: DialogueConfig,
) -> DialogueState:
    update = decision.update
    if update is None:
        return state
    if (decision.session_id, decision.generation, update.expected_revision) != (
        state.session_id,
        state.generation,
        state.revision,
    ) or update.turn.turn_id != decision.turn_id:
        raise ValueError("dialogue transition scope changed")
    responses = tuple(
        response.model_copy(update={"delivery": "cancelled", "delivery_at": decision.created_at})
        if response.delivery not in TERMINAL_DELIVERIES
        else response
        for response in state.responses
    )
    if decision.outcome != "no_reply":
        responses += (
            ResponseMemory(
                turn_id=decision.turn_id,
                response_id=decision.response_id,
                created_at=decision.created_at,
                expires_at=decision.expires_at,
                requests=update.turn.requests,
                acts=decision.acts,
                language=decision.language,
                output_mode=output_mode,
            ),
        )
    return state.model_copy(
        update={
            "revision": state.revision + 1,
            "topic": update.topic,
            "pending": update.pending,
            "history": (*state.history, update.turn)[-config.history_turns :],
            "responses": responses[-config.history_turns :],
        }
    )


def apply_delivery(state: DialogueState, event: DeliveryEvent, now: datetime) -> DialogueState:
    if (event.session_id, event.generation) != (state.session_id, state.generation):
        return state
    for index, response in enumerate(state.responses):
        if (event.turn_id, event.response_id, event.output_mode) != (
            response.turn_id,
            response.response_id,
            response.output_mode,
        ):
            continue
        if (
            response.delivery in TERMINAL_DELIVERIES
            or event.status == response.delivery
            or not response.created_at <= event.occurred_at <= now
            or (response.delivery_at is not None and event.occurred_at < response.delivery_at)
        ):
            return state
        allowed = {
            "accepted": {"queued", "started", "cancelled", "expired", "failed"},
            "queued": {"started", "cancelled", "expired", "failed"},
            "started": {"completed", "interrupted", "cancelled", "failed"},
        }.get(response.delivery, set())
        if response.output_mode == "text":
            allowed = (allowed - {"started", "interrupted"}) | {"completed"}
        if event.status not in allowed:
            return state
        if event.status in {"queued", "started", "completed"} and (
            response.delivery != "started" and event.occurred_at >= response.expires_at
        ):
            return state
        updated = response.model_copy(
            update={
                "delivery": event.status,
                "delivery_at": event.occurred_at,
            }
        )
        responses = (*state.responses[:index], updated, *state.responses[index + 1 :])
        return state.model_copy(update={"responses": responses, "revision": state.revision + 1})
    return state
