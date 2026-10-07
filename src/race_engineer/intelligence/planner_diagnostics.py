"""Bounded, catalog-only diagnostics for rejected plans, never executable fallbacks."""

from dataclasses import dataclass

from race_engineer.core.intelligence import ContextTemporalScope, EvidenceQuery
from race_engineer.intelligence.local_model import LocalIntelligenceError


@dataclass(frozen=True)
class RejectedPlanDiagnostic:
    """Known selections only; never retain arbitrary model text or unknown identifiers."""

    temporal_scope: ContextTemporalScope
    capability_ids: tuple[str, ...]
    queries: tuple[EvidenceQuery, ...]
    unknown_signal_count: int = 0
    unknown_capability_count: int = 0
    invalid_query_count: int = 0
    wrong_capability_scope_count: int = 0


class ContextPlanRejection(LocalIntelligenceError):
    """Reject the whole turn while retaining sanitized, non-executable diagnostic data."""

    def __init__(self, reason: str, diagnostic: RejectedPlanDiagnostic | None = None) -> None:
        super().__init__(reason)
        self.diagnostic = diagnostic
