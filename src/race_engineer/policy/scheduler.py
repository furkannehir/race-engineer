"""Deterministic priority, cooldown, expiry, and supersession scheduling."""

from dataclasses import dataclass
from datetime import datetime

from race_engineer.config import StrictPolicyConfig
from race_engineer.core.contracts import (
    CandidateMessage,
    PolicyDecision,
    SpeechIntent,
    TelemetryFrame,
)
from race_engineer.core.enums import (
    InterruptionPolicy,
    PolicyDecisionOutcome,
    PolicyDecisionReason,
    Tone,
)


@dataclass(frozen=True, slots=True)
class ScheduledCandidate:
    message: CandidateMessage
    deduplication_key: str
    critical_template: str | None = None
    enabled: bool = True


class PolicyScheduler:
    def __init__(self, config: StrictPolicyConfig | None = None) -> None:
        self._config = config or StrictPolicyConfig()
        self._session_id: str | None = None
        self._last_decided_at: datetime | None = None
        self._last_approved_by_key: dict[str, datetime] = {}

    def _reset(self, session_id: str) -> None:
        self._session_id = session_id
        self._last_decided_at = None
        self._last_approved_by_key.clear()

    def _cooldown_s(self, priority: int) -> float:
        if priority >= 90:
            return self._config.critical_cooldown_s
        if priority >= 70:
            return self._config.important_cooldown_s
        return self._config.routine_cooldown_s

    @staticmethod
    def _decision(
        candidate: ScheduledCandidate,
        frame: TelemetryFrame,
        outcome: PolicyDecisionOutcome,
        reason: PolicyDecisionReason,
        intent_id: str | None = None,
    ) -> PolicyDecision:
        message = candidate.message
        return PolicyDecision(
            decision_id=f"{message.candidate_id}:decision",
            candidate_id=message.candidate_id,
            session_id=frame.session_id,
            source_sequence=frame.sequence,
            decided_at=frame.observed_at,
            outcome=outcome,
            reason=reason,
            priority=message.base_priority,
            intent_id=intent_id,
        )

    def schedule(
        self,
        frame: TelemetryFrame,
        candidates: tuple[ScheduledCandidate, ...],
    ) -> tuple[list[SpeechIntent], tuple[PolicyDecision, ...]]:
        if frame.session_id != self._session_id:
            self._reset(frame.session_id)
        if self._last_decided_at is not None and frame.observed_at < self._last_decided_at:
            self._reset(frame.session_id)
        self._last_decided_at = frame.observed_at

        ranked = sorted(
            candidates, key=lambda candidate: candidate.message.base_priority, reverse=True
        )
        intents: list[SpeechIntent] = []
        decisions: list[PolicyDecision] = []
        seen_keys: set[str] = set()

        for candidate in ranked:
            message = candidate.message
            if not candidate.enabled:
                decisions.append(
                    self._decision(
                        candidate,
                        frame,
                        PolicyDecisionOutcome.SUPPRESSED,
                        PolicyDecisionReason.DISABLED,
                    )
                )
                continue
            if message.expires_at <= frame.observed_at:
                decisions.append(
                    self._decision(
                        candidate,
                        frame,
                        PolicyDecisionOutcome.SUPPRESSED,
                        PolicyDecisionReason.EXPIRED,
                    )
                )
                continue
            if candidate.deduplication_key in seen_keys:
                decisions.append(
                    self._decision(
                        candidate,
                        frame,
                        PolicyDecisionOutcome.SUPPRESSED,
                        PolicyDecisionReason.DUPLICATE,
                    )
                )
                continue
            seen_keys.add(candidate.deduplication_key)

            last_approved = self._last_approved_by_key.get(candidate.deduplication_key)
            cooldown_s = self._cooldown_s(message.base_priority)
            if (
                last_approved is not None
                and (frame.observed_at - last_approved).total_seconds() < cooldown_s
            ):
                decisions.append(
                    self._decision(
                        candidate,
                        frame,
                        PolicyDecisionOutcome.SUPPRESSED,
                        PolicyDecisionReason.COOLDOWN,
                    )
                )
                continue
            if len(intents) >= self._config.max_intents_per_frame:
                decisions.append(
                    self._decision(
                        candidate,
                        frame,
                        PolicyDecisionOutcome.SUPPRESSED,
                        PolicyDecisionReason.SUPERSEDED,
                    )
                )
                continue

            intent_id = f"{message.candidate_id}:speech"
            critical = message.base_priority >= 90
            intents.append(
                SpeechIntent(
                    intent_id=intent_id,
                    category=message.category,
                    facts=message.facts,
                    priority=message.base_priority,
                    tone=Tone.URGENT if critical else Tone.NEUTRAL,
                    max_words=self._config.max_words,
                    deadline=message.expires_at,
                    interruption_policy=(
                        InterruptionPolicy.INTERRUPT_ANY if critical else InterruptionPolicy.NEVER
                    ),
                    critical_template=candidate.critical_template if critical else None,
                    language=self._config.language,
                )
            )
            self._last_approved_by_key[candidate.deduplication_key] = frame.observed_at
            decisions.append(
                self._decision(
                    candidate,
                    frame,
                    PolicyDecisionOutcome.APPROVED,
                    PolicyDecisionReason.APPROVED,
                    intent_id,
                )
            )

        return intents, tuple(decisions)
