"""CE-04 dialogue-eval.v1: independent of the frozen CE-02 single-turn scorer."""

import math
import re
import time
import unicodedata
from collections import Counter
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, model_validator

from race_engineer.config import DialogueConfig
from race_engineer.conversation.dialogue_session import DialogueSession
from race_engineer.core.contracts import (
    ContractModel,
    OpponentState,
    PlayerState,
    RaceContext,
    TelemetryFrame,
)
from race_engineer.core.conversation import RaceSnapshot
from race_engineer.core.dialogue import DeliveryEvent, DialogueTurnInput, SemanticProposal
from race_engineer.core.interfaces import SemanticJudge
from race_engineer.evaluation.system import distribution


class WorldSpec(ContractModel):
    position: int | None = Field(default=6, ge=1)
    lap: int = Field(default=3, ge=0)
    fuel: float | None = Field(default=32, ge=0, allow_inf_nan=False)
    ahead_id: str = "car-a"
    behind_id: str = "car-b"
    ahead_gap: float | None = Field(default=1.8, gt=0, allow_inf_nan=False)
    behind_gap: float | None = Field(default=2.4, gt=0, allow_inf_nan=False)


class EvalTurn(ContractModel):
    turn_id: str
    question: str = Field(min_length=1, max_length=1000)
    acceptable: tuple[SemanticProposal, ...] = Field(min_length=1)
    outcomes: tuple[str, ...] = Field(min_length=1)
    answers: tuple[str, ...] = ()  # query:status, exact multiset
    clarification: Literal["none", "topic", "opponent"] = "none"
    category: str
    elapsed_s: float = Field(default=1, ge=0, le=300, allow_inf_nan=False)
    world: WorldSpec | None = None
    reset: bool = False
    stale: bool = False
    delivery: Literal["completed", "interrupted", "none"] = "completed"
    reply_language: Literal["en", "tr"] | None = None

    @model_validator(mode="after")
    def consistent_language(self) -> "EvalTurn":
        if len({p.language for p in self.acceptable}) != 1:
            raise ValueError("one scored reply language per turn")
        return self


class Scenario(ContractModel):
    family: str
    description: str
    en: tuple[EvalTurn, ...] = Field(min_length=1)
    tr: tuple[EvalTurn, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def parallel_translations(self) -> "Scenario":
        if len(self.en) != len(self.tr):
            raise ValueError("translations stay together and have parallel turns")
        for turns in (self.en, self.tr):
            if len({t.turn_id for t in turns}) != len(turns):
                raise ValueError("turn IDs must be unique within dialogue")
        return self


class DialogueDataset(ContractModel):
    schema_version: Literal["dialogue-eval.v1"] = "dialogue-eval.v1"
    split: Literal["calibration", "locked"]
    revision: str
    scenarios: tuple[Scenario, ...] = Field(min_length=1)


def load_dialogues(path: Path) -> DialogueDataset:
    return DialogueDataset.model_validate_json(path.read_text(encoding="utf-8"))


def normalized(text: str) -> str:
    return re.sub(r"[^\w]+", " ", unicodedata.normalize("NFKC", text).casefold()).strip()


def audit_splits(datasets: tuple[DialogueDataset, ...]) -> dict[str, Any]:
    """Exact/family audit; near-text similarities are review items, never proof of semantics."""
    families: dict[str, str] = {}
    literals: dict[str, str] = {}
    counts: dict[str, int] = {}
    for dataset in datasets:
        if dataset.split in counts:
            raise ValueError("duplicate split")
        counts[dataset.split] = len(dataset.scenarios)
        for scenario in dataset.scenarios:
            if scenario.family in families:
                raise ValueError("scenario family leaked or duplicated")
            if re.match(r"^(en|tr|locked|calibration)[/:_-]", scenario.family):
                raise ValueError("family IDs must not encode split/language")
            families[scenario.family] = dataset.split
            for turn in (*scenario.en, *scenario.tr):
                text = normalized(turn.question)
                if text in literals and literals[text] != dataset.split:
                    raise ValueError("literal cross-split leakage")
                literals[text] = dataset.split
    return {
        "families": counts,
        "literal_cross_split_duplicates": 0,
        "semantic_review": "See fixtures/dialogue/README.md; not guaranteed by IDs.",
    }


def semantic_key(proposal: SemanticProposal) -> dict[str, Any]:
    result = proposal.model_dump(
        mode="json", exclude={"schema_version", "model_id", "calibration_id"}
    )
    result["requests"] = sorted((part.query, part.reference) for part in proposal.requests)
    result["acts"] = sorted(proposal.acts)
    return result


def semantic_correct(proposal: SemanticProposal | None, turn: EvalTurn) -> bool:
    return proposal is not None and any(
        semantic_key(proposal) == semantic_key(allowed) for allowed in turn.acceptable
    )


class EvalWorld:
    def __init__(self) -> None:
        self.now = datetime.now(UTC)
        self.sequence = 1
        self.generation = 0
        self.spec = WorldSpec()
        self.stale = False

    def current_generation(self) -> int:
        return self.generation

    def current_time(self) -> datetime:
        return self.now

    def advance(self, turn: EvalTurn) -> None:
        self.now += timedelta(seconds=turn.elapsed_s)
        self.sequence += 1
        self.generation += int(turn.reset)
        self.stale = turn.stale
        if turn.world is not None:
            self.spec = turn.world

    def snapshot(self) -> RaceSnapshot:
        spec = self.spec
        opponents = []
        if spec.ahead_gap is not None:
            opponents.append(
                OpponentState(driver_id=spec.ahead_id, gap_to_player_s=-spec.ahead_gap)
            )
        if spec.behind_gap is not None:
            opponents.append(
                OpponentState(driver_id=spec.behind_id, gap_to_player_s=spec.behind_gap)
            )
        frame = TelemetryFrame(
            source="synthetic",
            session_id="dialogue-evaluation",
            sequence=self.sequence,
            observed_at=self.now - timedelta(seconds=10 if self.stale else 0),
            session_time_s=float(self.sequence),
            player=PlayerState(
                driver_id="player", position=spec.position, lap_number=spec.lap, fuel_l=spec.fuel
            ),
            opponents=tuple(opponents),
        )
        return RaceSnapshot(
            context=RaceContext(
                frame=frame,
                gap_ahead_s=spec.ahead_gap,
                gap_behind_s=spec.behind_gap,
                fuel_trend_l_per_lap=2.7,
            ),
            as_of=self.now,
            mode="replay",
        )

    def facts_valid(self, answers: Any) -> bool:
        """Independent fixture truth, not the controller/fact resolver under test."""
        spec = self.spec
        facts = {
            "position": (spec.position, "position", None),
            "lap": (spec.lap, "lap", None),
            "fuel_remaining": (spec.fuel, "l", None),
            "fuel_consumption": (2.7, "l/lap", None),
            "gap_ahead": (spec.ahead_gap, "s", spec.ahead_id),
            "gap_behind": (spec.behind_gap, "s", spec.behind_id),
        }
        for answer in answers:
            if answer.status != "available":
                if answer.value is not None:
                    return False
                continue
            expected = facts.get(answer.query)
            if self.stale or expected is None or expected[0] is None:
                return False
            if (answer.value, answer.unit) != expected[
                :2
            ] or answer.source_sequence != self.sequence:
                return False
            if (answer.opponent.driver_id if answer.opponent else None) != expected[2]:
                return False
        return True


class CapturingJudge:
    def __init__(self, inner: SemanticJudge) -> None:
        self.inner = inner
        self.proposal: SemanticProposal | None = None
        self.request: DialogueTurnInput | None = None
        self.elapsed_ms = 0.0
        self.teacher: SemanticProposal | None = None
        self.error = False

    async def judge(self, request: DialogueTurnInput) -> SemanticProposal:
        self.request = request
        start = time.perf_counter()
        try:
            self.proposal = await self.inner.judge(request)
            return self.teacher or self.proposal
        except Exception:
            self.error = True
            if self.teacher is not None:
                return self.teacher
            raise
        finally:
            self.elapsed_ms = (time.perf_counter() - start) * 1000


def ratio_interval(success: int, count: int) -> dict[str, Any]:
    if count == 0:
        return {"numerator": success, "denominator": count, "rate": None, "wilson95": None}
    p, z = success / count, 1.959964
    denominator = 1 + z * z / count
    middle = (p + z * z / (2 * count)) / denominator
    radius = z * math.sqrt(p * (1 - p) / count + z * z / (4 * count * count)) / denominator
    return {
        "numerator": success,
        "denominator": count,
        "rate": p,
        "wilson95": [max(0, middle - radius), min(1, middle + radius)],
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def metrics(items: list[dict[str, Any]]) -> dict[str, Any]:
        judged = [r for r in items if r["judge_called"]]
        accepted = [r for r in judged if r["accepted"]]
        answerable = [r for r in judged if r["answerable"]]
        clarification = [r for r in judged if r["needs_clarification"]]
        return {
            "turns": len(items),
            "judged_turns": len(judged),
            "complete_turn_accuracy": ratio_interval(sum(r["passed"] for r in judged), len(judged)),
            "accepted_plan_accuracy": ratio_interval(
                sum(r["semantic_correct"] for r in accepted), len(accepted)
            ),
            "answerable_coverage": ratio_interval(
                sum(r["passed"] for r in answerable), len(answerable)
            ),
            "clarification_accuracy": ratio_interval(
                sum(r["passed"] for r in clarification), len(clarification)
            ),
            "errors": sum(r["error"] for r in judged),
            "abstentions": sum(r["abstained"] for r in judged),
            "guard_only_turns": len(items) - len(judged),
            "factual_guard_failures": sum(not r["facts_valid"] for r in items),
            "warm_total_ms": distribution([r["total_ms"] for r in judged if not r["cold_request"]]),
            "warm_judge_ms": distribution([r["judge_ms"] for r in judged if not r["cold_request"]]),
        }

    dialogue_ids = {r["dialogue_id"] for r in rows}
    return {
        "overall": metrics(rows),
        "languages": {
            lang: metrics([r for r in rows if r["language"] == lang]) for lang in ("en", "tr")
        },
        "categories": {
            cat: metrics([r for r in rows if r["category"] == cat])
            for cat in sorted({r["category"] for r in rows})
        },
        "whole_dialogues": ratio_interval(
            sum(all(r["passed"] for r in rows if r["dialogue_id"] == key) for key in dialogue_ids),
            len(dialogue_ids),
        ),
    }


async def evaluate_dialogues(
    dataset: DialogueDataset,
    judge: SemanticJudge,
    *,
    teacher_forced: bool = False,
    calibration_samples: list[tuple[DialogueTurnInput, dict[str, float], EvalTurn]] | None = None,
    progress: Any = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    first = True
    for scenario in dataset.scenarios:
        for language in ("en", "tr"):
            world = EvalWorld()
            capture = CapturingJudge(judge)
            session = DialogueSession(
                capture,
                world.snapshot,
                DialogueConfig.model_validate({"default_language": language}),
                generation=world.current_generation,
                clock=world.current_time,
            )
            try:
                for turn in getattr(scenario, language):
                    world.advance(turn)
                    capture.proposal, capture.request, capture.elapsed_ms, capture.error = (
                        None,
                        None,
                        0,
                        False,
                    )
                    capture.teacher = turn.acceptable[0] if teacher_forced else None
                    start = time.perf_counter()
                    decision = await session.ask(turn.question, reply_language=turn.reply_language)
                    total_ms = (time.perf_counter() - start) * 1000
                    semantics = semantic_correct(capture.proposal, turn)
                    facts_valid = world.facts_valid(decision.answers)
                    decision_ok = (
                        decision.outcome in turn.outcomes
                        and Counter(f"{a.query}:{a.status}" for a in decision.answers)
                        == Counter(turn.answers)
                        and decision.clarification == turn.clarification
                        and decision.language == turn.acceptable[0].language
                    )
                    if capture.request is not None and calibration_samples is not None:
                        scores = getattr(judge, "last_scores", {})
                        calibration_samples.append((capture.request, dict(scores), turn))
                    judged = capture.request is not None
                    row = {
                        "dialogue_id": f"{scenario.family}/{language}",
                        "turn_id": turn.turn_id,
                        "category": turn.category,
                        "language": turn.acceptable[0].language,
                        "judge_called": judged,
                        "semantic_correct": semantics,
                        "accepted": capture.proposal is not None and not capture.proposal.abstain,
                        "abstained": capture.proposal is not None and capture.proposal.abstain,
                        "error": capture.error or (judged and capture.proposal is None),
                        "answerable": any(a.endswith(":available") for a in turn.answers),
                        "needs_clarification": turn.clarification != "none",
                        "facts_valid": facts_valid,
                        "decision_correct": decision_ok,
                        "passed": (semantics or not judged) and facts_valid and decision_ok,
                        "actual": semantic_key(capture.proposal) if capture.proposal else None,
                        "outcome": decision.outcome,
                        "reason": decision.reason,
                        "judge_ms": round(capture.elapsed_ms, 3),
                        "total_ms": round(total_ms, 3),
                        "cold_request": judged and first,
                    }
                    if judged:
                        first = False
                    rows.append(row)
                    if progress:
                        progress(row)
                    if turn.delivery != "none" and decision.outcome not in {
                        "discarded",
                        "no_reply",
                    }:
                        for status in ("started", turn.delivery):
                            session.record_delivery(
                                DeliveryEvent.model_validate(
                                    {
                                        "turn_id": decision.turn_id,
                                        "response_id": decision.response_id,
                                        "session_id": decision.session_id,
                                        "generation": decision.generation,
                                        "output_mode": "speech",
                                        "status": status,
                                        "occurred_at": world.now,
                                    }
                                )
                            )
            finally:
                await session.aclose()
    return rows, summarize(rows)
