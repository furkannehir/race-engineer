"""Content-free evaluation of Context Engineer candidates over race capabilities."""

import hashlib
import json
import re
import time
from collections import Counter, defaultdict
from collections.abc import Callable, Sequence
from pathlib import Path

from race_engineer.config import PolicyContextConfig
from race_engineer.core.contracts import RaceContext
from race_engineer.core.intelligence import (
    CapabilityDescriptor,
    ContextPlan,
    DriverTurn,
    EngineerLanguage,
    EvidenceQuery,
    SignalDescriptor,
)
from race_engineer.core.interfaces import ContextQueryPlanner
from race_engineer.evaluation.intelligence import (
    ExpectedContextPlan,
    ExpectedEvidenceQuery,
    IntelligenceCandidateMetadata,
    IntelligenceEvaluationDataset,
)
from race_engineer.evaluation.system import distribution
from race_engineer.fixtures import load_fixture
from race_engineer.intelligence.capabilities import (
    DeterministicRaceCapabilities,
    RaceCapabilityError,
)
from race_engineer.intelligence.context_engine import (
    ContextEngineerError,
    QueryDrivenContextEngineer,
)
from race_engineer.intelligence.local_model import LocalIntelligenceError
from race_engineer.intelligence.planner_diagnostics import (
    ContextPlanRejection,
    RejectedPlanDiagnostic,
)
from race_engineer.intelligence.telemetry_memory import (
    BoundedTelemetryMemory,
    TelemetryMemoryError,
)
from race_engineer.policy import DefaultRaceContextBuilder


class TimedContextPlanner:
    """Measure one candidate without changing its planner contract."""

    def __init__(self, planner: ContextQueryPlanner) -> None:
        self._planner = planner
        self.call_count = 0
        self.last_elapsed_ms: float | None = None
        self.last_plan: ContextPlan | None = None
        self.last_call_index: int | None = None
        self.last_rejection: RejectedPlanDiagnostic | None = None

    def reset_turn(self) -> None:
        self.last_elapsed_ms = None
        self.last_plan = None
        self.last_call_index = None
        self.last_rejection = None

    async def plan(
        self,
        turn: DriverTurn,
        signals: Sequence[SignalDescriptor],
        capabilities: Sequence[CapabilityDescriptor],
    ) -> ContextPlan:
        self.last_call_index = self.call_count
        self.call_count += 1
        started = time.perf_counter()
        try:
            self.last_plan = await self._planner.plan(turn, signals, capabilities)
            return self.last_plan
        except ContextPlanRejection as error:
            self.last_rejection = error.diagnostic
            raise
        finally:
            self.last_elapsed_ms = (time.perf_counter() - started) * 1000


def _question_id(question: str) -> str:
    return hashlib.sha256(question.encode("utf-8")).hexdigest()[:16]


def _query_value(query: EvidenceQuery | ExpectedEvidenceQuery) -> dict[str, object]:
    selector = query.selector if isinstance(query, EvidenceQuery) else query
    return {
        "source": selector.source,
        "signal": selector.signal,
        "subject_id": selector.subject_id,
        "operation": query.operation,
        "window_s": query.window_s,
    }


def _query_sort_key(value: dict[str, object]) -> tuple[str, str, str, str, float]:
    window = value["window_s"]
    return (
        str(value["source"]),
        str(value["subject_id"] or ""),
        str(value["signal"]),
        str(value["operation"]),
        float(window) if isinstance(window, (int, float)) else 0.0,
    )


def _plan_value(plan: ContextPlan | None) -> dict[str, object] | None:
    if plan is None:
        return None
    queries = sorted(
        (_query_value(query) for query in plan.queries),
        key=_query_sort_key,
    )
    return {
        "temporal_scope": plan.temporal_scope,
        "capability_ids": sorted(request.capability_id for request in plan.capability_requests),
        "queries": queries,
    }


def _purpose(plan: ContextPlan | None) -> str | None:
    if plan is None:
        return None
    social = "driver_social_turn" in plan.situation
    factual = "race_information_request" in plan.situation
    if social and factual:
        return "mixed"
    if social:
        return "social"
    return "race_information" if factual else None


def _expected_value(expected: ExpectedContextPlan) -> dict[str, object]:
    queries = sorted(
        (_query_value(query) for query in expected.queries),
        key=_query_sort_key,
    )
    return {
        "temporal_scope": expected.temporal_scope,
        "capability_ids": sorted(expected.capability_ids),
        "queries": queries,
    }


def _rejected_value(diagnostic: RejectedPlanDiagnostic | None) -> dict[str, object] | None:
    if diagnostic is None:
        return None
    return {
        "temporal_scope": diagnostic.temporal_scope,
        "capability_ids": sorted(diagnostic.capability_ids),
        "queries": sorted((_query_value(q) for q in diagnostic.queries), key=_query_sort_key),
        "unknown_signal_count": diagnostic.unknown_signal_count,
        "unknown_capability_count": diagnostic.unknown_capability_count,
        "invalid_query_count": diagnostic.invalid_query_count,
        "wrong_capability_scope_count": diagnostic.wrong_capability_scope_count,
    }


def _request_counts(plan: dict[str, object] | None) -> Counter[str]:
    if plan is None:
        return Counter()
    capabilities = plan.get("capability_ids")
    queries = plan.get("queries")
    result = (
        Counter(f"capability:{key}" for key in capabilities)
        if isinstance(capabilities, list)
        else Counter()
    )
    if isinstance(queries, list):
        result.update(f"query:{json.dumps(query, sort_keys=True)}" for query in queries)
    return result


def _selection_metrics(
    observed: dict[str, object] | None, expected: dict[str, object]
) -> dict[str, int | bool]:
    """Exact expected request coverage, not a claim about semantic equivalence or answers."""

    expected_counts = _request_counts(expected)
    observed_counts = _request_counts(observed)
    missing = sum((expected_counts - observed_counts).values())
    return {
        "expected_request_count": sum(expected_counts.values()),
        "matched_request_count": sum((expected_counts & observed_counts).values()),
        "missing_request_count": missing,
        "extra_request_count": sum((observed_counts - expected_counts).values()),
        "required_selection_present": observed is not None and missing == 0,
    }


def _failure_stage(error: Exception) -> str:
    if isinstance(error, ContextPlanRejection) or str(error).startswith("planner_"):
        return "plan_validation"
    if isinstance(error, LocalIntelligenceError):
        reason = str(error)
        if any(part in reason for part in ("unreachable", "timeout", "http_")):
            return "model_transport"
        return "model_response"
    return "evidence_execution"


def _evidence_outcome(evidence_kinds: tuple[str, ...]) -> str:
    if not evidence_kinds:
        return "no_evidence"
    unknown = sum(kind == "unknown" for kind in evidence_kinds)
    if unknown == len(evidence_kinds):
        return "unavailable"
    if unknown:
        return "partial"
    return "available"


def _safe_fixture(root: Path, configured: str) -> Path:
    resolved_root = root.resolve()
    fixture = (resolved_root / configured).resolve()
    if not fixture.is_relative_to(resolved_root):
        raise ValueError("intelligence evaluation fixture escapes the repository")
    return fixture


async def _evaluate_turn(
    timed: TimedContextPlanner,
    contexts: tuple[RaceContext, ...],
    *,
    dataset: IntelligenceEvaluationDataset,
    case_id: str,
    family: str,
    category: str,
    language: EngineerLanguage,
    item_index: int,
    question: str,
    frame_index: int,
    expected: ExpectedContextPlan,
    tags: tuple[str, ...],
    recent_dialogue: tuple[str, ...] = (),
) -> dict[str, object]:
    selected = len(contexts) - 1 if frame_index == -1 else frame_index
    if not 0 <= selected < len(contexts):
        raise ValueError("intelligence evaluation frame index is out of range")
    memory = BoundedTelemetryMemory()
    for context in contexts[: selected + 1]:
        memory.update(context)
    current = memory.latest_context()
    registry = DeterministicRaceCapabilities(memory)
    engine = QueryDrivenContextEngineer(memory, timed, registry)
    turn = DriverTurn(
        turn_id=f"evaluation:{dataset.dataset_id}:{case_id}:{item_index}",
        transcript=question,
        received_at=current.frame.observed_at,
        session_id=current.frame.session_id,
        generation=0,
        asr_language=language,
        reply_language=language,
        recent_dialogue=recent_dialogue,
    )

    timed.reset_turn()
    started = time.perf_counter()
    error_reason: str | None = None
    failure_stage: str | None = None
    packet = None
    try:
        packet = await engine.analyze(turn)
    except (
        ContextEngineerError,
        LocalIntelligenceError,
        RaceCapabilityError,
        TelemetryMemoryError,
        ValueError,
    ) as error:
        failure_stage = _failure_stage(error)
        # Pydantic/ValueError text may contain input values. Never persist that body.
        reason = str(error)
        error_reason = (
            reason if re.fullmatch(r"[a-z][a-z0-9_]{0,119}", reason) else f"{failure_stage}_failed"
        )
    total_ms = (time.perf_counter() - started) * 1000

    observed_plan = _plan_value(timed.last_plan)
    expected_plan = _expected_value(expected)
    observed_purpose = _purpose(timed.last_plan)
    purpose_passed = expected.purpose is None or expected.purpose == observed_purpose
    temporal_scope_passed = (
        observed_plan is not None
        and observed_plan["temporal_scope"] == expected_plan["temporal_scope"]
    )
    capability_selection_passed = (
        observed_plan is not None
        and observed_plan["capability_ids"] == expected_plan["capability_ids"]
    )
    query_selection_passed = (
        observed_plan is not None and observed_plan["queries"] == expected_plan["queries"]
    )
    plan_passed = (
        temporal_scope_passed
        and capability_selection_passed
        and query_selection_passed
        and purpose_passed
    )
    if packet is not None:
        observed_outcome = _evidence_outcome(tuple(item.kind for item in packet.evidence))
    elif failure_stage == "plan_validation":
        observed_outcome = "plan_rejected"
    elif failure_stage == "evidence_execution":
        observed_outcome = "execution_error"
    else:
        observed_outcome = "model_error"
    unknowns = tuple(packet.unknowns) if packet is not None else ()
    evidence_ids = {item.evidence_id for item in packet.evidence} if packet is not None else set()
    stable_unknowns = tuple(value for value in unknowns if value not in evidence_ids)
    reportable_unknowns = {
        "unknown",
        "requested_projection_unavailable",
        "requested_analysis_unavailable",
        "requested_reference_ambiguous",
        "semantic_candidate_abstained",
        *expected.required_unknowns,
        *(
            f"{item.capability_id}:{item.unavailable_reason}"
            for item in registry.catalog()
            if item.unavailable_reason is not None
        ),
    }
    outcome_passed = observed_outcome == expected.evidence_outcome
    unknowns_passed = packet is not None and set(expected.required_unknowns) == set(stable_unknowns)
    selection = _selection_metrics(observed_plan, expected_plan)
    required_evidence_passed = (
        bool(selection["required_selection_present"])
        and temporal_scope_passed
        and purpose_passed
        and outcome_passed
        and unknowns_passed
        and failure_stage is None
    )
    rejected_draft = _rejected_value(timed.last_rejection)
    planner_ms = timed.last_elapsed_ms
    execution_ms = max(0.0, total_ms - planner_ms) if planner_ms is not None else None
    return {
        "dataset_id": dataset.dataset_id,
        "split": dataset.split,
        "case_id": case_id,
        "family": family,
        "category": category,
        "language": language,
        "item": item_index,
        "question_id": _question_id(question),
        "tags": list(tags),
        "expected_plan": expected_plan,
        "expected_purpose": expected.purpose,
        "observed_purpose": observed_purpose,
        "purpose_passed": purpose_passed,
        "observed_plan": observed_plan,
        "rejected_draft": rejected_draft,
        "rejected_draft_selection": (
            _selection_metrics(rejected_draft, expected_plan) if rejected_draft else None
        ),
        "selection": selection,
        "required_evidence_passed": required_evidence_passed,
        "expected_evidence_outcome": expected.evidence_outcome,
        "observed_evidence_outcome": observed_outcome,
        "required_unknowns": list(expected.required_unknowns),
        "observed_unknowns": [
            value if value in reportable_unknowns else f"unrecognized_unknown:{_question_id(value)}"
            for value in stable_unknowns
        ],
        "unknown_evidence_count": len(unknowns) - len(stable_unknowns),
        "temporal_scope_passed": temporal_scope_passed,
        "capability_selection_passed": capability_selection_passed,
        "query_selection_passed": query_selection_passed,
        "plan_passed": plan_passed,
        "outcome_passed": outcome_passed,
        "unknowns_passed": unknowns_passed,
        "passed": plan_passed and outcome_passed and unknowns_passed,
        "reason": error_reason,
        "failure_stage": failure_stage,
        "planner_call": timed.last_call_index,
        "call_state": "first" if timed.last_call_index == 0 else "subsequent",
        "timing_ms": {
            "planner": round(planner_ms, 3) if planner_ms is not None else None,
            "execution": round(execution_ms, 3) if execution_ms is not None else None,
            "total": round(total_ms, 3),
        },
    }


def _score(results: list[dict[str, object]]) -> dict[str, object]:
    total = len(results)
    passed = sum(bool(result["passed"]) for result in results)
    plan_passed = sum(bool(result["plan_passed"]) for result in results)
    temporal_scope_passed = sum(bool(result["temporal_scope_passed"]) for result in results)
    capability_selection_passed = sum(
        bool(result["capability_selection_passed"]) for result in results
    )
    query_selection_passed = sum(bool(result["query_selection_passed"]) for result in results)
    outcome_passed = sum(bool(result["outcome_passed"]) for result in results)
    unknowns_passed = sum(bool(result["unknowns_passed"]) for result in results)
    supported = [
        result
        for result in results
        if isinstance(result["expected_plan"], dict)
        and result["expected_plan"].get("temporal_scope") != "social"
    ]
    accepted = [
        result
        for result in supported
        if isinstance(result["observed_plan"], dict)
        and bool(
            result["observed_plan"].get("capability_ids") or result["observed_plan"].get("queries")
        )
    ]
    accepted_correct = sum(bool(result["plan_passed"]) for result in accepted)
    purpose_cases = [result for result in results if result["expected_purpose"] is not None]
    return {
        "turns": total,
        "purpose_cases": len(purpose_cases),
        "purpose_accuracy": (
            round(
                sum(bool(result["purpose_passed"]) for result in purpose_cases)
                / len(purpose_cases),
                6,
            )
            if purpose_cases
            else None
        ),
        "passed": passed,
        "exact_turn_accuracy": round(passed / total, 6) if total else None,
        "exact_plan_accuracy": round(plan_passed / total, 6) if total else None,
        "temporal_scope_accuracy": (round(temporal_scope_passed / total, 6) if total else None),
        "capability_selection_accuracy": (
            round(capability_selection_passed / total, 6) if total else None
        ),
        "query_selection_accuracy": (round(query_selection_passed / total, 6) if total else None),
        "evidence_outcome_accuracy": round(outcome_passed / total, 6) if total else None,
        "unknown_reason_accuracy": round(unknowns_passed / total, 6) if total else None,
        "supported_turns": len(supported),
        "supported_accepted": len(accepted),
        "supported_coverage": (round(len(accepted) / len(supported), 6) if supported else None),
        "accepted_plan_accuracy": (
            round(accepted_correct / len(accepted), 6) if accepted else None
        ),
        "model_errors": sum(
            result["observed_evidence_outcome"] == "model_error" for result in results
        ),
        "transport_errors": sum(result["failure_stage"] == "model_transport" for result in results),
        "response_errors": sum(result["failure_stage"] == "model_response" for result in results),
        "plan_validation_errors": sum(
            result["failure_stage"] == "plan_validation" for result in results
        ),
        "execution_errors": sum(
            result["failure_stage"] == "evidence_execution" for result in results
        ),
        "required_evidence_accuracy": (
            round(sum(bool(result["required_evidence_passed"]) for result in results) / total, 6)
            if total
            else None
        ),
        "extra_evidence_turns": sum(
            isinstance(result["selection"], dict)
            and bool(result["selection"].get("extra_request_count"))
            for result in results
        ),
    }


def summarize_intelligence_results(
    results: list[dict[str, object]],
) -> dict[str, object]:
    by_language: dict[str, list[dict[str, object]]] = defaultdict(list)
    by_category: dict[str, list[dict[str, object]]] = defaultdict(list)
    failure_reasons: Counter[str] = Counter()
    planner_all: list[float] = []
    planner_first: list[float] = []
    planner_subsequent: list[float] = []
    model_prompt: list[float] = []
    model_generation: list[float] = []
    execution: list[float] = []
    total: list[float] = []
    for result in results:
        by_language[str(result["language"])].append(result)
        by_category[str(result["category"])].append(result)
        if not result["passed"]:
            failure_reasons[str(result["reason"] or "expectation_mismatch")] += 1
        timing = result["timing_ms"]
        if not isinstance(timing, dict):
            continue
        planner_ms = timing.get("planner")
        execution_ms = timing.get("execution")
        total_ms = timing.get("total")
        if isinstance(planner_ms, (int, float)):
            planner_all.append(float(planner_ms))
            if result["call_state"] == "first":
                planner_first.append(float(planner_ms))
            else:
                planner_subsequent.append(float(planner_ms))
        if isinstance(execution_ms, (int, float)):
            execution.append(float(execution_ms))
        if isinstance(total_ms, (int, float)):
            total.append(float(total_ms))
        requests = result.get("model_requests")
        if isinstance(requests, list):
            for request in requests:
                if not isinstance(request, dict):
                    continue
                prompt_ms = request.get("prompt_ms")
                generation_ms = request.get("generation_ms")
                if isinstance(prompt_ms, (int, float)):
                    model_prompt.append(float(prompt_ms))
                if isinstance(generation_ms, (int, float)):
                    model_generation.append(float(generation_ms))
    return {
        "overall": _score(results),
        "by_language": {key: _score(value) for key, value in sorted(by_language.items())},
        "by_category": {key: _score(value) for key, value in sorted(by_category.items())},
        "failure_reasons": dict(sorted(failure_reasons.items())),
        "latency_ms": {
            "planner_all": distribution(planner_all),
            "planner_first_call": distribution(planner_first),
            "planner_subsequent_calls": distribution(planner_subsequent),
            "model_prompt_processing": distribution(model_prompt),
            "model_generation": distribution(model_generation),
            "execution": distribution(execution),
            "total": distribution(total),
        },
    }


async def evaluate_intelligence_datasets(
    root: Path,
    policy_config: PolicyContextConfig,
    planner: ContextQueryPlanner,
    candidate: IntelligenceCandidateMetadata,
    datasets: tuple[IntelligenceEvaluationDataset, ...],
    *,
    progress: Callable[[dict[str, object]], None] | None = None,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    timed = TimedContextPlanner(planner)
    results: list[dict[str, object]] = []
    for dataset in datasets:
        fixture = load_fixture(_safe_fixture(root, dataset.fixture))
        builder = DefaultRaceContextBuilder(policy_config)
        contexts = tuple(builder.update(frame, ()) for frame in fixture.frames)
        for group in dataset.groups:
            for index, question in enumerate(group.questions, start=1):
                result = await _evaluate_turn(
                    timed,
                    contexts,
                    dataset=dataset,
                    case_id=group.id,
                    family=group.family,
                    category=group.category,
                    language=group.language,
                    item_index=index,
                    question=question,
                    frame_index=group.frame_index,
                    expected=group.expected,
                    tags=group.tags,
                    recent_dialogue=group.recent_dialogue,
                )
                results.append(result)
                if progress is not None:
                    progress(result)
    summary = summarize_intelligence_results(results)
    summary["candidate"] = candidate.model_dump(mode="json")
    return results, summary
