"""Deterministic event-to-intent policy with no learned behavior."""

import logging
from collections.abc import Callable
from datetime import timedelta

from pydantic import JsonValue

from race_engineer.config import StrictPolicyConfig
from race_engineer.core.contracts import (
    CandidateMessage,
    PolicyDecision,
    RaceContext,
    RaceEvent,
    SpeechIntent,
)
from race_engineer.core.enums import (
    EventType,
    MessageCategory,
    PolicyDecisionOutcome,
    SessionPhase,
    Urgency,
)
from race_engineer.policy.scheduler import PolicyScheduler, ScheduledCandidate

DecisionSink = Callable[[PolicyDecision], None]

_PHASE_RULES: dict[str, tuple[int, str, str | None]] = {
    SessionPhase.FORMATION.value: (60, "race_control:formation", None),
    SessionPhase.GREEN.value: (70, "race_control:green", None),
    SessionPhase.CAUTION.value: (100, "race_control:yellow", "Caution, slow down"),
    SessionPhase.CHECKERED.value: (80, "race_control:checkered", None),
}

_FLAG_RULES: dict[str, tuple[int, str, str | None]] = {
    "red": (100, "race_control:red", "Red flag"),
    "yellow": (95, "race_control:yellow", "Yellow flag"),
    "black": (95, "race_control:black", "Black flag"),
    "checkered": (80, "race_control:checkered", None),
    "blue": (70, "race_control:blue", None),
    "green": (70, "race_control:green", None),
    "white": (60, "race_control:white", None),
}


def _priority_for_urgency(urgency: Urgency) -> int:
    return {
        Urgency.ROUTINE: 30,
        Urgency.IMPORTANT: 70,
        Urgency.CRITICAL: 100,
    }[urgency]


def _optional_int(value: JsonValue | None) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


class StrictRulePolicy:
    """Turns current-frame events into constrained and traceable speech intents."""

    def __init__(
        self,
        config: StrictPolicyConfig | None = None,
        decision_sink: DecisionSink | None = None,
    ) -> None:
        self._config = config or StrictPolicyConfig()
        self._scheduler = PolicyScheduler(self._config)
        self._decision_sink = decision_sink
        self._logger = logging.getLogger(__name__)

    @staticmethod
    def _candidate(
        event: RaceEvent,
        suffix: str,
        category: MessageCategory,
        facts: dict[str, JsonValue],
        priority: int,
        deduplication_key: str,
        critical_template: str | None = None,
        enabled: bool = True,
    ) -> ScheduledCandidate:
        return ScheduledCandidate(
            message=CandidateMessage(
                candidate_id=f"{event.event_id}:policy:{suffix}",
                category=category,
                facts=facts,
                base_priority=priority,
                created_at=event.occurred_at,
                expires_at=event.expires_at or event.occurred_at + timedelta(seconds=5),
                interruptible=priority < 90,
            ),
            deduplication_key=deduplication_key,
            critical_template=critical_template,
            enabled=enabled,
        )

    def _phase_candidate(self, event: RaceEvent) -> tuple[ScheduledCandidate, ...]:
        phase_value = event.facts.get("to", event.facts.get("phase"))
        if not isinstance(phase_value, str) or phase_value not in _PHASE_RULES:
            return ()
        priority, key, critical_template = _PHASE_RULES[phase_value]
        facts: dict[str, JsonValue] = {"phase": phase_value}
        previous_phase = event.facts.get("from")
        if isinstance(previous_phase, str):
            facts["previous_phase"] = previous_phase
        return (
            self._candidate(
                event,
                f"phase:{phase_value}",
                MessageCategory.SAFETY,
                facts,
                priority,
                key,
                critical_template,
            ),
        )

    def _flag_candidates(self, event: RaceEvent) -> tuple[ScheduledCandidate, ...]:
        added = event.facts.get("added")
        if not isinstance(added, list):
            return ()
        candidates: list[ScheduledCandidate] = []
        for value in added:
            if not isinstance(value, str) or value not in _FLAG_RULES:
                continue
            priority, key, critical_template = _FLAG_RULES[value]
            candidates.append(
                self._candidate(
                    event,
                    f"flag:{value}",
                    MessageCategory.SAFETY,
                    {"flag": value, "state": "shown"},
                    priority,
                    key,
                    critical_template,
                )
            )
        return tuple(candidates)

    def _position_candidate(self, event: RaceEvent) -> tuple[ScheduledCandidate, ...]:
        position = _optional_int(event.facts.get("to"))
        if position is None or position < 1:
            return ()
        facts: dict[str, JsonValue] = {"position": position}
        previous_position = _optional_int(event.facts.get("from"))
        if previous_position is not None and previous_position >= 1:
            facts["previous_position"] = previous_position
        return (
            self._candidate(
                event,
                f"position:{position}",
                MessageCategory.SITUATION,
                facts,
                40,
                "position",
                enabled=self._config.announce_position_changes,
            ),
        )

    def _pit_candidate(self, event: RaceEvent) -> tuple[ScheduledCandidate, ...]:
        in_pit_lane = event.facts.get("in_pit_lane")
        if not isinstance(in_pit_lane, bool):
            return ()
        return (
            self._candidate(
                event,
                f"pit:{int(in_pit_lane)}",
                MessageCategory.SITUATION,
                {"in_pit_lane": in_pit_lane},
                30,
                "pit_state",
                enabled=self._config.announce_pit_transitions,
            ),
        )

    def _fuel_candidate(self, event: RaceEvent) -> tuple[ScheduledCandidate, ...]:
        priority = _priority_for_urgency(event.urgency)
        return (
            self._candidate(
                event,
                "fuel",
                MessageCategory.STRATEGY,
                dict(event.facts),
                priority,
                "fuel_threshold",
                "Fuel critical" if priority >= 90 else None,
            ),
        )

    def _candidates_for(self, event: RaceEvent) -> tuple[ScheduledCandidate, ...]:
        match event.event_type:
            case EventType.SESSION_PHASE_CHANGED:
                return self._phase_candidate(event)
            case EventType.FLAG_CHANGED:
                return self._flag_candidates(event)
            case EventType.POSITION_CHANGED:
                return self._position_candidate(event)
            case EventType.PIT_STATE_CHANGED:
                return self._pit_candidate(event)
            case EventType.FUEL_THRESHOLD:
                return self._fuel_candidate(event)
            case _:
                return ()

    def _emit_decision(self, decision: PolicyDecision) -> None:
        self._logger.info(
            "policy candidate approved"
            if decision.outcome is PolicyDecisionOutcome.APPROVED
            else "policy candidate suppressed",
            extra={
                "event": "policy_decision",
                "decision_id": decision.decision_id,
                "candidate_id": decision.candidate_id,
                "session_id": decision.session_id,
                "source_sequence": decision.source_sequence,
                "outcome": decision.outcome.value,
                "reason": decision.reason.value,
                "priority": decision.priority,
                "intent_id": decision.intent_id,
            },
        )
        if self._decision_sink is not None:
            self._decision_sink(decision)

    async def decide(self, context: RaceContext) -> list[SpeechIntent]:
        candidates: list[ScheduledCandidate] = []
        for event in context.recent_events:
            if event.source_sequence == context.frame.sequence:
                candidates.extend(self._candidates_for(event))
        intents, decisions = self._scheduler.schedule(context.frame, tuple(candidates))
        for decision in decisions:
            self._emit_decision(decision)
        return intents
