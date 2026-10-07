"""Measure the real two-inference text path without changing its prompts or decisions."""

import json
import re
import time
from collections import Counter
from collections.abc import Callable, Sequence
from hashlib import sha256

from race_engineer.config import ConversationConfig
from race_engineer.core.contracts import RaceContext
from race_engineer.core.intelligence import (
    ContextPacket,
    ContextPlan,
    DriverTurn,
    EngineerBrief,
    GeneratedResponse,
    GroundedResponse,
)
from race_engineer.evaluation.intelligence import (
    ExpectedContextPlan,
    IntelligenceEvaluationGroup,
)
from race_engineer.evaluation.intelligence_runner import TimedContextPlanner
from race_engineer.evaluation.system import distribution
from race_engineer.intelligence.capabilities import DeterministicRaceCapabilities
from race_engineer.intelligence.context_engine import QueryDrivenContextEngineer
from race_engineer.intelligence.context_planner import (
    PLANNER_ID,
    QwenContextQueryPlanner,
)
from race_engineer.intelligence.grounding import StrictEvidenceGrounder
from race_engineer.intelligence.local_model import (
    JsonModelClient,
    LocalJsonModel,
    ModelRequestMetrics,
)
from race_engineer.intelligence.orchestrator import EngineerOrchestrator
from race_engineer.intelligence.portable_engineer import PortableQwenEngineer
from race_engineer.intelligence.telemetry_memory import BoundedTelemetryMemory

type BenchmarkProgress = Callable[[dict[str, object], str | None], None]


def _reason(error: Exception) -> str:
    value = str(error)
    return value if re.fullmatch(r"[a-z][a-z0-9_]{0,119}", value) else "stage_failed"


class MeasuredModel:
    """Collect numeric provider metrics and wall time for every call, including retries."""

    def __init__(
        self, config: ConversationConfig, role: str, *, client: JsonModelClient | None = None
    ) -> None:
        self.role = role
        self.requests: list[dict[str, object]] = []
        self._metrics: ModelRequestMetrics = {}
        self._client = client or LocalJsonModel(config, on_metrics=self._record_metrics)

    def _record_metrics(self, metrics: ModelRequestMetrics) -> None:
        self._metrics = metrics

    async def request(
        self,
        *,
        system_prompt: str,
        content: str,
        schema: dict[str, object],
        max_tokens: int = 1024,
    ) -> object:
        self._metrics = {}
        started = time.perf_counter()
        reason: str | None = None
        try:
            return await self._client.request(
                system_prompt=system_prompt, content=content, schema=schema, max_tokens=max_tokens
            )
        except Exception as error:
            reason = _reason(error)
            raise
        finally:
            self.requests.append(
                {
                    "role": self.role,
                    "attempt": len(self.requests) + 1,
                    "wall_ms": round((time.perf_counter() - started) * 1000, 3),
                    "status": "error" if reason else "completed",
                    "reason": reason,
                    **self._metrics,
                }
            )


class _Stages:
    def __init__(self) -> None:
        self.current = "context"
        self.elapsed: dict[str, float] = {}


class _MeasuredContext:
    def __init__(self, delegate: QueryDrivenContextEngineer, stages: _Stages) -> None:
        self.delegate = delegate
        self.stages = stages
        self.packet: ContextPacket | None = None

    async def analyze(self, turn: DriverTurn) -> ContextPacket:
        self.stages.current = "context"
        started = time.perf_counter()
        try:
            self.packet = await self.delegate.analyze(turn)
            return self.packet
        finally:
            self.stages.elapsed["context"] = (time.perf_counter() - started) * 1000

    async def refresh(self, packet: ContextPacket, evidence_ids: Sequence[str]) -> ContextPacket:
        self.stages.current = "refresh"
        started = time.perf_counter()
        try:
            return await self.delegate.refresh(packet, evidence_ids)
        finally:
            self.stages.elapsed["refresh"] = (time.perf_counter() - started) * 1000


class _MeasuredCore:
    def __init__(self, delegate: PortableQwenEngineer, stages: _Stages) -> None:
        self.delegate = delegate
        self.stages = stages

    async def decide(self, turn: DriverTurn, context: ContextPacket) -> EngineerBrief:
        self.stages.current = "core"
        started = time.perf_counter()
        try:
            return await self.delegate.decide(turn, context)
        finally:
            self.stages.elapsed["core"] = (time.perf_counter() - started) * 1000

    async def generate(
        self, turn: DriverTurn, context: ContextPacket, brief: EngineerBrief
    ) -> GeneratedResponse:
        self.stages.current = "materialize"
        started = time.perf_counter()
        try:
            return await self.delegate.generate(turn, context, brief)
        finally:
            self.stages.elapsed["materialize"] = (time.perf_counter() - started) * 1000


class _MeasuredGrounder:
    def __init__(self, stages: _Stages) -> None:
        self.stages = stages

    def ground(
        self, response: GeneratedResponse, context: ContextPacket, brief: EngineerBrief
    ) -> GroundedResponse:
        self.stages.current = "grounding"
        started = time.perf_counter()
        try:
            return StrictEvidenceGrounder().ground(response, context, brief)
        finally:
            self.stages.elapsed["grounding"] = (time.perf_counter() - started) * 1000


def _plan_matches(plan: ContextPlan | None, expected: ExpectedContextPlan) -> bool:
    if plan is None:
        return False
    actual_queries = sorted(
        json.dumps(
            {
                **query.selector.model_dump(),
                "operation": query.operation,
                "window_s": query.window_s,
            },
            sort_keys=True,
        )
        for query in plan.queries
    )
    expected_queries = sorted(
        json.dumps(query.model_dump(), sort_keys=True) for query in expected.queries
    )
    social = "driver_social_turn" in plan.situation
    factual = "race_information_request" in plan.situation
    purpose = "mixed" if social and factual else "social" if social else "race_information"
    return (
        plan.temporal_scope == expected.temporal_scope
        and sorted(request.capability_id for request in plan.capability_requests)
        == sorted(expected.capability_ids)
        and actual_queries == expected_queries
        and (expected.purpose is None or purpose == expected.purpose)
    )


def _plan_diagnostic_value(plan: ContextPlan | None) -> dict[str, object] | None:
    if plan is None:
        return None
    queries = sorted(
        (
            {
                **query.selector.model_dump(mode="json"),
                "operation": query.operation,
                "window_s": query.window_s,
            }
            for query in plan.queries
        ),
        key=lambda query: json.dumps(query, sort_keys=True),
    )
    social = "driver_social_turn" in plan.situation
    factual = "race_information_request" in plan.situation
    purpose = "mixed" if social and factual else "social" if social else "race_information"
    return {
        "purpose": purpose,
        "temporal_scope": plan.temporal_scope,
        "capability_ids": sorted(request.capability_id for request in plan.capability_requests),
        "queries": queries,
    }


def _expected_plan_diagnostic_value(expected: ExpectedContextPlan) -> dict[str, object]:
    queries = sorted(
        (query.model_dump(mode="json") for query in expected.queries),
        key=lambda query: json.dumps(query, sort_keys=True),
    )
    return {
        "purpose": expected.purpose,
        "temporal_scope": expected.temporal_scope,
        "capability_ids": sorted(expected.capability_ids),
        "queries": queries,
    }


def _selection_diagnostic(
    observed: dict[str, object] | None, expected: dict[str, object]
) -> dict[str, object]:
    def requests(plan: dict[str, object] | None) -> Counter[str]:
        if plan is None:
            return Counter()
        capability_items = plan.get("capability_ids", [])
        query_items = plan.get("queries", [])
        capabilities = capability_items if isinstance(capability_items, list) else []
        queries = query_items if isinstance(query_items, list) else []
        values = Counter(
            f"capability:{value}" for value in capabilities if isinstance(value, str)
        )
        values.update(
            f"query:{json.dumps(value, sort_keys=True)}"
            for value in queries
            if isinstance(value, dict)
        )
        return values

    actual_requests, expected_requests = requests(observed), requests(expected)
    missing = sorted((expected_requests - actual_requests).elements())
    extra = sorted((actual_requests - expected_requests).elements())
    return {
        "missing": missing,
        "extra": extra,
        "missing_count": len(missing),
        "extra_count": len(extra),
    }


async def measure_intelligence_turn(
    config: ConversationConfig,
    contexts: tuple[RaceContext, ...],
    group: IntelligenceEvaluationGroup,
    question: str,
    *,
    run_index: int,
    repetition: int,
    planner_client: JsonModelClient | None = None,
    core_client: JsonModelClient | None = None,
    planner_id: str = PLANNER_ID,
    inventory_observer: Callable[[tuple[str, ...]], None] | None = None,
) -> tuple[dict[str, object], str | None]:
    """Use production adapters/orchestration; return content-free data plus transient text."""

    selected = len(contexts) - 1 if group.frame_index == -1 else group.frame_index
    if not 0 <= selected < len(contexts):
        raise ValueError("benchmark frame index is out of range")
    memory = BoundedTelemetryMemory()
    for context in contexts[: selected + 1]:
        memory.update(context)
    frame = memory.latest_context().frame
    turn = DriverTurn(
        turn_id=f"benchmark:{run_index}:{group.id}",
        transcript=question,
        received_at=frame.observed_at,
        session_id=frame.session_id,
        generation=0,
        asr_language=group.language,
        reply_language=group.language,
        recent_dialogue=group.recent_dialogue,
    )
    stages = _Stages()
    planner_model = MeasuredModel(config, "context", client=planner_client)
    core_model = MeasuredModel(config, "core", client=core_client)
    planner = TimedContextPlanner(
        QwenContextQueryPlanner(
            config,
            model=planner_model,
            planner_id=planner_id,
            on_request_inventory=inventory_observer,
        )
    )
    context_engine = _MeasuredContext(
        QueryDrivenContextEngineer(memory, planner, DeterministicRaceCapabilities(memory)), stages
    )
    core = _MeasuredCore(PortableQwenEngineer(config, model=core_model), stages)
    engineer = EngineerOrchestrator(context_engine, core, core, _MeasuredGrounder(stages))
    started = time.perf_counter()
    grounded: GroundedResponse | None = None
    failure: dict[str, str] | None = None
    try:
        grounded = await engineer.respond(turn)
    except Exception as error:
        failure = {"stage": stages.current, "reason": _reason(error)}
    elapsed_ms = (time.perf_counter() - started) * 1000
    packet = context_engine.packet
    unknowns = packet.unknowns if packet is not None else ()
    kinds = [item.kind for item in packet.evidence] if packet is not None else []
    outcome = (
        None
        if packet is None
        else "no_evidence"
        if not kinds
        else "unavailable"
        if all(kind == "unknown" for kind in kinds)
        else "partial"
        if "unknown" in kinds
        else "available"
    )
    observed_plan = _plan_diagnostic_value(planner.last_plan)
    expected_plan = _expected_plan_diagnostic_value(group.expected)
    result: dict[str, object] = {
        "case_id": group.id,
        "language": group.language,
        "question_id": sha256(question.encode("utf-8")).hexdigest()[:16],
        "run_index": run_index,
        "repetition": repetition,
        "call_state": "first_in_run" if run_index == 1 else "subsequent_in_run",
        "pipeline_completed": grounded is not None,
        "planner_exact_match": _plan_matches(planner.last_plan, group.expected),
        "expected_plan": expected_plan,
        "observed_plan": observed_plan,
        "observed_purpose": observed_plan["purpose"] if observed_plan else None,
        "temporal_scope_match": (
            observed_plan is not None
            and observed_plan["temporal_scope"] == expected_plan["temporal_scope"]
        ),
        "purpose_match": (
            group.expected.purpose is None
            or (observed_plan is not None and observed_plan["purpose"] == group.expected.purpose)
        ),
        "selection_difference": _selection_diagnostic(observed_plan, expected_plan),
        "evidence_expectation_match": (
            outcome == group.expected.evidence_outcome
            and set(group.expected.required_unknowns) == set(unknowns)
        ),
        "manual_reply_review_required": True,
        "failure": failure,
        "core_retries": max(0, len(core_model.requests) - 1),
        "model_requests": [*planner_model.requests, *core_model.requests],
        "timing_ms": {
            **{name: round(value, 3) for name, value in stages.elapsed.items()},
            "total": round(elapsed_ms, 3),
        },
        "reply_action": grounded.action if grounded is not None else None,
        "reply_word_count": len(grounded.text.split()) if grounded and grounded.text else 0,
    }
    return result, grounded.text if grounded is not None else None


def summarize_latency(results: list[dict[str, object]]) -> dict[str, object]:
    """Include failures in all-turn timing; completed turns are not semantic passes."""

    completed = [result for result in results if result["pipeline_completed"]]

    def timings(rows: list[dict[str, object]], name: str) -> list[float]:
        values: list[float] = []
        for row in rows:
            timing = row["timing_ms"]
            value = timing.get(name) if isinstance(timing, dict) else None
            if isinstance(value, (int, float)):
                values.append(float(value))
        return values

    reply_times = timings(completed, "total")
    return {
        "turns": len(results),
        "pipeline_completed": len(completed),
        "pipeline_failed": len(results) - len(completed),
        "planner_exact_matches": sum(bool(result["planner_exact_match"]) for result in results),
        "evidence_expectation_matches": sum(
            bool(result["evidence_expectation_match"]) for result in results
        ),
        "core_retries": sum(int(str(result["core_retries"])) for result in results),
        "all_turn_latency_ms": distribution(timings(results, "total")),
        "completed_turn_latency_ms": distribution(reply_times),
        "completed_under_5s": sum(value < 5000 for value in reply_times),
        "completed_under_10s": sum(value < 10000 for value in reply_times),
        "completed_at_least_30s": sum(value >= 30000 for value in reply_times),
        "stage_latency_ms": {
            name: distribution(timings(results, name))
            for name in ("context", "core", "materialize", "refresh", "grounding")
        },
        "first_in_run_latency_ms": distribution(timings(results[:1], "total")),
        "subsequent_in_run_latency_ms": distribution(timings(results[1:], "total")),
        "automatic_reply_accuracy": None,
    }
