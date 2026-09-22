"""Failure-isolated runtime writer for privacy-safe decision and radio metadata."""

import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

from race_engineer.core.contracts import PlaybackResult, PolicyDecision
from race_engineer.memory.sqlite import DriverMemoryError, SqliteDriverProfileRepository

_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class DurableHistory:
    repository: SqliteDriverProfileRepository
    profile_id: str

    @classmethod
    def try_open(
        cls,
        path: Path,
        *,
        retention_days: int,
        clock: Callable[[], datetime] | None = None,
    ) -> "DurableHistory | None":
        current_time = clock or (lambda: datetime.now(UTC))
        try:
            if not 1 <= retention_days <= 3650:
                raise ValueError("history retention must be between 1 and 3650 days")
            # Runtime history is observational; lock contention must not stall telemetry.
            repository = SqliteDriverProfileRepository(
                path,
                clock=current_time,
                busy_timeout_ms=50,
            )
            profile = repository.ensure_default_profile()
            repository.prune_history(
                profile.profile_id,
                before=current_time() - timedelta(days=retention_days),
            )
            return cls(repository, profile.profile_id)
        except (DriverMemoryError, OSError, ValueError) as error:
            _LOGGER.warning(
                "durable history unavailable; live behavior will continue",
                extra={
                    "event": "history_unavailable",
                    "reason": str(error) or type(error).__name__,
                },
            )
            return None

    def record_decision(self, decision: PolicyDecision) -> None:
        try:
            self.repository.record_policy_decision(self.profile_id, decision)
        except (DriverMemoryError, OSError, ValueError) as error:
            _LOGGER.warning(
                "policy decision history write failed; live behavior will continue",
                extra={
                    "event": "history_decision_write_failed",
                    "decision_id": decision.decision_id,
                    "reason": str(error) or type(error).__name__,
                },
            )

    def record_playback_result(self, result: PlaybackResult) -> None:
        try:
            self.repository.record_playback_result(self.profile_id, result)
        except (DriverMemoryError, OSError, ValueError) as error:
            _LOGGER.warning(
                "radio outcome history write failed; live behavior will continue",
                extra={
                    "event": "history_outcome_write_failed",
                    "intent_id": result.intent_id,
                    "reason": str(error) or type(error).__name__,
                },
            )
