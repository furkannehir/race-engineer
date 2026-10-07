"""Model-directed Context Engineer over generic bounded telemetry operations."""

from collections import OrderedDict
from collections.abc import Iterable, Sequence

from race_engineer.core.intelligence import (
    CapabilityRequest,
    CapabilityResult,
    ContextPacket,
    ContextPlan,
    DriverTurn,
    EvidenceItem,
)
from race_engineer.core.interfaces import (
    ContextQueryPlanner,
    RaceCapabilityRegistry,
    TelemetryMemory,
)


class ContextEngineerError(Exception):
    """A context plan or telemetry scope cannot be safely executed."""


class QueryDrivenContextEngineer:
    """Execute model-selected generic queries without a spoken command table."""

    def __init__(
        self,
        memory: TelemetryMemory,
        planner: ContextQueryPlanner,
        capabilities: RaceCapabilityRegistry | None = None,
        *,
        retained_plans: int = 8,
    ) -> None:
        if retained_plans < 1:
            raise ValueError("at least one context plan must be retained")
        self._memory = memory
        self._planner = planner
        self._capabilities = capabilities
        self._retained_plans = retained_plans
        self._plans: OrderedDict[str, tuple[DriverTurn, ContextPlan]] = OrderedDict()

    @staticmethod
    def _unknowns(
        plan: ContextPlan,
        evidence: Iterable[EvidenceItem],
        capability_unknowns: Iterable[str],
    ) -> tuple[str, ...]:
        values = [*plan.unknowns, *capability_unknowns]
        values.extend(item.evidence_id for item in evidence if item.kind == "unknown")
        return tuple(dict.fromkeys(values))

    def _execute_capabilities(
        self,
        requests: Sequence[CapabilityRequest],
    ) -> tuple[tuple[EvidenceItem, ...], tuple[str, ...]]:
        if not requests:
            return (), ()
        if self._capabilities is None:
            raise ContextEngineerError("race_capability_registry_unavailable")
        results: tuple[CapabilityResult, ...] = tuple(
            self._capabilities.execute(request) for request in requests
        )
        evidence = tuple(item for result in results for item in result.evidence)
        unknowns = tuple(
            f"{result.capability_id}:{result.unavailable_reason}"
            for result in results
            if result.status == "unavailable" and result.unavailable_reason is not None
        )
        return evidence, unknowns

    def _packet(
        self,
        turn: DriverTurn,
        plan: ContextPlan,
        evidence: tuple[EvidenceItem, ...],
        capability_unknowns: tuple[str, ...] = (),
    ) -> ContextPacket:
        current = self._memory.latest_context()
        frame = current.frame
        if frame.session_id != turn.session_id:
            raise ContextEngineerError("telemetry_session_mismatch")
        return ContextPacket(
            turn_id=turn.turn_id,
            session_id=turn.session_id,
            generation=turn.generation,
            source_sequence=frame.sequence,
            assembled_at=frame.observed_at,
            situation=plan.situation,
            evidence=evidence,
            unknowns=self._unknowns(plan, evidence, capability_unknowns),
        )

    async def analyze(self, turn: DriverTurn) -> ContextPacket:
        signals = self._memory.signal_catalog()
        capabilities = self._capabilities.catalog() if self._capabilities is not None else ()
        plan = await self._planner.plan(turn, signals, capabilities)
        if plan.turn_id != turn.turn_id:
            raise ContextEngineerError("context_plan_turn_mismatch")
        query_evidence = self._memory.query_many(plan.queries)
        capability_evidence, capability_unknowns = self._execute_capabilities(
            plan.capability_requests
        )
        evidence = (*query_evidence, *capability_evidence)
        self._plans[turn.turn_id] = (turn, plan)
        self._plans.move_to_end(turn.turn_id)
        while len(self._plans) > self._retained_plans:
            self._plans.popitem(last=False)
        return self._packet(turn, plan, evidence, capability_unknowns)

    async def refresh(
        self,
        packet: ContextPacket,
        evidence_ids: Sequence[str],
    ) -> ContextPacket:
        retained = self._plans.get(packet.turn_id)
        if retained is None:
            raise ContextEngineerError("context_plan_not_retained")
        turn, plan = retained
        if (
            packet.session_id != turn.session_id
            or packet.generation != turn.generation
            or packet.turn_id != turn.turn_id
        ):
            raise ContextEngineerError("context_packet_scope_mismatch")
        if len(set(evidence_ids)) != len(evidence_ids):
            raise ContextEngineerError("refresh_evidence_ids_not_unique")
        queries = {query.query_id: query for query in plan.queries}
        capability_evidence_ids: dict[str, CapabilityRequest] = {}
        if self._capabilities is not None:
            for request in plan.capability_requests:
                for evidence_id in self._capabilities.evidence_ids(request):
                    capability_evidence_ids[evidence_id] = request
        allowed = {*queries, *capability_evidence_ids}
        if not set(evidence_ids).issubset(allowed):
            raise ContextEngineerError("refresh_requested_unknown_evidence")
        selected_queries = tuple(
            queries[evidence_id] for evidence_id in evidence_ids if evidence_id in queries
        )
        refreshed = {
            item.evidence_id: item for item in self._memory.query_many(selected_queries)
        }
        selected_capabilities = tuple(
            request
            for request in plan.capability_requests
            if any(
                evidence_id in evidence_ids
                for evidence_id in capability_evidence_ids
                if capability_evidence_ids[evidence_id] == request
            )
        )
        capability_evidence, capability_unknowns = self._execute_capabilities(
            selected_capabilities
        )
        refreshed.update(
            (item.evidence_id, item)
            for item in capability_evidence
            if item.evidence_id in evidence_ids
        )
        evidence = tuple(refreshed[evidence_id] for evidence_id in evidence_ids)
        return self._packet(turn, plan, evidence, capability_unknowns)
