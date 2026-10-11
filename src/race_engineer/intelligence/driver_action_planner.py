"""Bounded experimental social fast path, with full Qwen planning for everything else."""

import logging
import time
from collections.abc import Sequence
from typing import Protocol

from race_engineer.core.intelligence import (
    CapabilityDescriptor,
    ContextPlan,
    DriverTurn,
    SignalDescriptor,
)
from race_engineer.core.interfaces import ContextQueryPlanner
from race_engineer.intelligence.driver_action_client import DriverInterpreterError
from race_engineer.intelligence.driver_actions import (
    LABELS,
    ActionPrediction,
    ActionRequest,
    conversation_input,
)
from race_engineer.radio_diagnostics import radio_event, radio_stage

_LOGGER = logging.getLogger(__name__)
_FAST_ACTIONS = {"radio_check", "greeting", "vent"}


class DriverActionClassifier(Protocol):
    async def start(self) -> None: ...
    async def predict(self, request: ActionRequest) -> ActionPrediction: ...
    async def aclose(self) -> None: ...


def social_fast_action(prediction: ActionPrediction) -> tuple[str | None, str]:
    """Declared routing guard, not a calibration claim or a complete intent interpreter."""
    if prediction.input_truncated:
        return None, "input_truncated"
    if prediction.clarification_needed:
        return None, "clarification_needed"
    if len(prediction.actions) != 1 or prediction.actions[0] not in _FAST_ACTIONS:
        return None, "not_single_social_action"
    action = prediction.actions[0]
    if prediction.score(action) < max(0.85, prediction.threshold):
        return None, "social_score_below_guard"
    # Veto even sub-threshold signs of a factual clause or a feedback/correction turn.
    if prediction.score("request_information") >= 0.2:
        return None, "possible_information_request"
    if any(
        prediction.score(label) >= 0.35
        for label in (
            "correction",
            "preference_change",
            "communication_feedback",
            "out_of_scope",
            "unclear",
            "needs_clarification",
        )
    ):
        return None, "ambiguous_action_scores"
    return action, "single_social_action"


def live_action_input(turn: DriverTurn) -> str | None:
    dialogue: list[tuple[str, str]] = []
    for entry in turn.recent_dialogue[-6:]:
        role, separator, message = entry.partition(": ")
        if not separator or role not in {"driver", "engineer"} or len(message) > 2000:
            return None
        dialogue.append((role, message))
    return conversation_input(turn.transcript, dialogue)


class ActionAssistedContextPlanner:
    def __init__(self, classifier: DriverActionClassifier, fallback: ContextQueryPlanner) -> None:
        self._classifier = classifier
        self._fallback = fallback
        self._started = False
        self._unavailable_reason: str | None = None

    def _disable(self, reason: str) -> None:
        self._unavailable_reason = reason
        _LOGGER.warning(
            "experimental action interpreter unavailable; using Qwen planning",
            extra={"event": "driver_interpreter_unavailable", "reason": reason},
        )

    async def start(self) -> None:
        if self._started:
            return
        self._started = True
        started = time.perf_counter()
        try:
            await self._classifier.start()
        except DriverInterpreterError as error:
            self._disable(str(error))
        else:
            _LOGGER.info(
                "experimental action interpreter warmed on CPU",
                extra={
                    "event": "driver_interpreter_ready",
                    "backend": "trained_minilm_cpu",
                    "duration_ms": round((time.perf_counter() - started) * 1000, 3),
                    "scope": "social_fast_path_only",
                    "experimental": True,
                },
            )

    async def plan(
        self,
        turn: DriverTurn,
        signals: Sequence[SignalDescriptor],
        capabilities: Sequence[CapabilityDescriptor],
    ) -> ContextPlan:
        await self.start()
        reason = self._unavailable_reason
        text = live_action_input(turn)
        action: str | None = None
        if reason is None and text is not None:
            try:
                with radio_stage("driver_interpreter"):
                    prediction = await self._classifier.predict(
                        ActionRequest(request_id=turn.turn_id, input_text=text)
                    )
                radio_event(
                    "radio_driver_actions",
                    actions=prediction.actions,
                    clarification_needed=prediction.clarification_needed,
                    threshold=prediction.threshold,
                    input_truncated=prediction.input_truncated,
                    inference_duration_ms=round(prediction.duration_ms, 3),
                    scores_uncalibrated=dict(zip(LABELS, prediction.scores, strict=True)),
                )
                action, reason = social_fast_action(prediction)
            except DriverInterpreterError as error:
                reason = str(error)
                self._disable(reason)
        elif reason is None:
            reason = "unsupported_dialogue_format"
        radio_event(
            "radio_interpreter_route",
            route="social_fast_path" if action else "qwen_fallback",
            reason=reason,
            action=action,
            context_planner_skipped=action is not None,
        )
        if action is not None:
            return ContextPlan(
                turn_id=turn.turn_id,
                planner_id="trained-minilm-social-pilot",
                temporal_scope="social",
                situation=("driver_social_turn", f"driver_{action}"),
            )
        return await self._fallback.plan(turn, signals, capabilities)

    async def aclose(self) -> None:
        await self._classifier.aclose()
