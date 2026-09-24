"""Deterministic conversation evaluation with stage timing and content-free reports."""

import hashlib
import time
from collections import Counter, defaultdict
from collections.abc import Callable
from pathlib import Path

from race_engineer.config import ConversationConfig
from race_engineer.conversation.replay import ReplayRaceState
from race_engineer.conversation.session import ConversationSession
from race_engineer.core.conversation import ConversationPlan, ConversationRequest
from race_engineer.core.interfaces import ConversationPlanner
from race_engineer.evaluation.conversation import (
    ConversationEvaluationDataset,
    ExpectedTurn,
)
from race_engineer.evaluation.system import distribution
from race_engineer.fixtures import load_fixture


class TimedPlanner:
    def __init__(self, planner: ConversationPlanner) -> None:
        self._planner = planner
        self.call_count = 0
        self.last_elapsed_ms: float | None = None
        self.last_plan: ConversationPlan | None = None
        self.last_call_index: int | None = None

    def reset_turn(self) -> None:
        self.last_elapsed_ms = None
        self.last_plan = None
        self.last_call_index = None

    async def plan(self, request: ConversationRequest) -> ConversationPlan:
        self.last_call_index = self.call_count
        self.call_count += 1
        started = time.perf_counter()
        try:
            self.last_plan = await self._planner.plan(request)
            return self.last_plan
        finally:
            self.last_elapsed_ms = (time.perf_counter() - started) * 1000


def _question_id(question: str) -> str:
    return hashlib.sha256(question.encode("utf-8")).hexdigest()[:16]


def _plan_value(plan: ConversationPlan | None) -> dict[str, object] | None:
    if plan is None:
        return None
    return {
        "language": plan.language,
        "queries": sorted(query.value for query in plan.queries),
        "clarification": plan.clarification,
    }


def _expected_value(expected: ExpectedTurn) -> dict[str, object]:
    return {
        "language": expected.language,
        "queries": sorted(query.value for query in expected.queries),
        "clarification": expected.clarification,
        "reply_status": expected.reply_status,
    }


async def _evaluate_turn(
    session: ConversationSession,
    planner: TimedPlanner,
    *,
    dataset: ConversationEvaluationDataset,
    case_id: str,
    family: str,
    category: str,
    item_index: int,
    question: str,
    expected: ExpectedTurn,
    tags: tuple[str, ...],
) -> dict[str, object]:
    planner.reset_turn()
    started = time.perf_counter()
    reply = await session.ask(question)
    total_ms = (time.perf_counter() - started) * 1000
    plan = planner.last_plan
    observed = _plan_value(plan)
    expected_plan = _expected_value(expected)
    expected_plan_only = {
        key: value for key, value in expected_plan.items() if key != "reply_status"
    }
    plan_passed = observed == expected_plan_only
    reply_passed = reply.status == expected.reply_status
    planner_ms = planner.last_elapsed_ms
    application_ms = max(0.0, total_ms - planner_ms) if planner_ms is not None else total_ms
    return {
        "dataset_id": dataset.dataset_id,
        "split": dataset.split,
        "case_id": case_id,
        "family": family,
        "category": category,
        "item": item_index,
        "question_id": _question_id(question),
        "tags": list(tags),
        "expected": expected_plan,
        "observed_plan": observed,
        "observed_reply_status": reply.status,
        "reason": reply.reason,
        "plan_passed": plan_passed,
        "reply_passed": reply_passed,
        "passed": plan_passed and reply_passed,
        "planner_call": planner.last_call_index,
        "warm_state": "cold" if planner.last_call_index == 0 else "warm",
        "timing_ms": {
            "planner": round(planner_ms, 3) if planner_ms is not None else None,
            "application": round(application_ms, 3),
            "total": round(total_ms, 3),
        },
    }


def _safe_fixture(root: Path, configured: str) -> Path:
    resolved_root = root.resolve()
    fixture = (resolved_root / configured).resolve()
    if not fixture.is_relative_to(resolved_root):
        raise ValueError("evaluation fixture path escapes the repository")
    return fixture


async def evaluate_datasets(
    root: Path,
    config: ConversationConfig,
    planner: ConversationPlanner,
    datasets: tuple[ConversationEvaluationDataset, ...],
    *,
    progress: Callable[[dict[str, object]], None] | None = None,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    timed = TimedPlanner(planner)
    results: list[dict[str, object]] = []
    for dataset in datasets:
        fixture = load_fixture(_safe_fixture(root, dataset.fixture))
        for group in dataset.groups:
            for index, question in enumerate(group.questions, start=1):
                state = ReplayRaceState(fixture)
                state.seek(group.frame_index)
                session = ConversationSession(timed, state.snapshot, config)
                result = await _evaluate_turn(
                    session,
                    timed,
                    dataset=dataset,
                    case_id=group.id,
                    family=group.family,
                    category=group.category,
                    item_index=index,
                    question=question,
                    expected=group.expected,
                    tags=group.tags,
                )
                results.append(result)
                if progress is not None:
                    progress(result)
        for sequence in dataset.sequences:
            state = ReplayRaceState(fixture)
            session = ConversationSession(timed, state.snapshot, config)
            for index, turn in enumerate(sequence.turns, start=1):
                if turn.frame_index is not None:
                    state.seek(turn.frame_index)
                result = await _evaluate_turn(
                    session,
                    timed,
                    dataset=dataset,
                    case_id=sequence.id,
                    family=sequence.family,
                    category=sequence.category,
                    item_index=index,
                    question=turn.question,
                    expected=turn.expected,
                    tags=sequence.tags,
                )
                results.append(result)
                if progress is not None:
                    progress(result)
    return results, summarize(results)


def _score(results: list[dict[str, object]]) -> dict[str, object]:
    total = len(results)
    passed = sum(bool(result["passed"]) for result in results)
    plans = sum(result["observed_plan"] is not None for result in results)
    plan_passed = sum(bool(result["plan_passed"]) for result in results)
    supported = [
        result
        for result in results
        if isinstance(result["expected"], dict)
        and result["expected"].get("clarification") == "none"
    ]
    accepted = [
        result
        for result in results
        if isinstance(result["observed_plan"], dict)
        and result["observed_plan"].get("clarification") == "none"
    ]
    supported_accepted = [result for result in supported if result in accepted]
    accepted_correct = sum(bool(result["plan_passed"]) for result in accepted)
    clarifications = [
        result
        for result in results
        if isinstance(result["expected"], dict)
        and result["expected"].get("clarification") != "none"
    ]
    correct_clarifications = sum(bool(result["plan_passed"]) for result in clarifications)
    return {
        "turns": total,
        "passed": passed,
        "exact_turn_accuracy": round(passed / total, 6) if total else None,
        "plans_returned": plans,
        "coverage": round(plans / total, 6) if total else None,
        "exact_plan_accuracy": round(plan_passed / total, 6) if total else None,
        "accepted_turns": len(accepted),
        "accepted_plan_accuracy": (
            round(accepted_correct / len(accepted), 6) if accepted else None
        ),
        "supported_turns": len(supported),
        "supported_accepted": len(supported_accepted),
        "supported_coverage": (
            round(len(supported_accepted) / len(supported), 6) if supported else None
        ),
        "clarification_turns": len(clarifications),
        "clarification_accuracy": (
            round(correct_clarifications / len(clarifications), 6) if clarifications else None
        ),
        "model_errors": sum(result["observed_reply_status"] == "model_error" for result in results),
    }


def summarize(results: list[dict[str, object]]) -> dict[str, object]:
    by_language: dict[str, list[dict[str, object]]] = defaultdict(list)
    by_category: dict[str, list[dict[str, object]]] = defaultdict(list)
    failure_reasons: Counter[str] = Counter()
    for result in results:
        expected = result["expected"]
        if isinstance(expected, dict):
            by_language[str(expected["language"])].append(result)
        by_category[str(result["category"])].append(result)
        if not result["passed"]:
            failure_reasons[str(result["reason"] or "expectation_mismatch")] += 1

    planner_all: list[float] = []
    planner_warm: list[float] = []
    application: list[float] = []
    total: list[float] = []
    for result in results:
        timing = result["timing_ms"]
        if not isinstance(timing, dict):
            continue
        planner_ms = timing.get("planner")
        if isinstance(planner_ms, (int, float)):
            planner_all.append(float(planner_ms))
            if result["warm_state"] == "warm":
                planner_warm.append(float(planner_ms))
        application_ms = timing.get("application")
        total_ms = timing.get("total")
        if isinstance(application_ms, (int, float)):
            application.append(float(application_ms))
        if isinstance(total_ms, (int, float)):
            total.append(float(total_ms))

    return {
        "overall": _score(results),
        "by_language": {key: _score(value) for key, value in sorted(by_language.items())},
        "by_category": {key: _score(value) for key, value in sorted(by_category.items())},
        "failure_reasons": dict(sorted(failure_reasons.items())),
        "latency_ms": {
            "planner_all": distribution(planner_all),
            "planner_warm": distribution(planner_warm),
            "application": distribution(application),
            "total": distribution(total),
        },
        "unmeasured_stages": [
            "capture_to_release",
            "asr",
            "tts_synthesis",
            "radio_wait",
            "first_audio",
        ],
    }
