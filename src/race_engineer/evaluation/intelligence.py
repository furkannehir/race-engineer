"""Versioned bilingual datasets for Context Engineer candidate evaluation."""

from collections import Counter
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from race_engineer.core.conversation import MAX_DIALOGUE_HISTORY_ENTRIES
from race_engineer.core.intelligence import (
    ContextTemporalScope,
    EvidenceQuery,
    SignalSelector,
    TelemetryOperation,
    TelemetrySource,
)

type EvaluationSplit = Literal["development", "calibration", "locked"]
type ExpectedEvidenceOutcome = Literal["available", "unavailable", "no_evidence"]
type RuntimeProfile = Literal["portable", "enhanced"]
type CandidateKind = Literal["generative", "classifier", "temporal", "control"]


class IntelligenceEvaluationModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ExpectedEvidenceQuery(IntelligenceEvaluationModel):
    """A query expectation whose model-authored evidence ID is intentionally ignored."""

    source: TelemetrySource
    signal: str = Field(pattern=r"^[a-z][a-z0-9_]*$", max_length=80)
    subject_id: str | None = Field(default=None, min_length=1, max_length=160)
    operation: TelemetryOperation
    window_s: float | None = Field(default=None, gt=0, le=3600, allow_inf_nan=False)

    @model_validator(mode="after")
    def validate_query(self) -> "ExpectedEvidenceQuery":
        EvidenceQuery(
            query_id="evaluation-query",
            selector=SignalSelector(
                source=self.source,
                signal=self.signal,
                subject_id=self.subject_id,
            ),
            operation=self.operation,
            window_s=self.window_s,
        )
        return self


class ExpectedContextPlan(IntelligenceEvaluationModel):
    purpose: Literal["race_information", "social", "mixed"] | None = None
    temporal_scope: ContextTemporalScope
    capability_ids: tuple[str, ...] = Field(default=(), max_length=8)
    queries: tuple[ExpectedEvidenceQuery, ...] = Field(default=(), max_length=24)
    evidence_outcome: ExpectedEvidenceOutcome
    required_unknowns: tuple[str, ...] = Field(default=(), max_length=16)

    @model_validator(mode="after")
    def validate_plan(self) -> "ExpectedContextPlan":
        if self.purpose is not None and (self.purpose == "social") != (
            self.temporal_scope == "social"
        ):
            raise ValueError("only pure social expectations omit factual time scope")
        if len(set(self.capability_ids)) != len(self.capability_ids):
            raise ValueError("expected capability IDs must be unique")
        if self.temporal_scope == "social":
            if self.capability_ids or self.queries or self.evidence_outcome != "no_evidence":
                raise ValueError("social expectations cannot request telemetry evidence")
        elif not self.capability_ids and not self.queries:
            raise ValueError("non-social expectations require evidence work")
        if self.evidence_outcome == "unavailable" and not self.required_unknowns:
            raise ValueError("unavailable expectations require a stable unknown reason")
        if self.evidence_outcome != "unavailable" and self.required_unknowns:
            raise ValueError("only unavailable expectations carry unknown reasons")
        return self


class IntelligenceEvaluationGroup(IntelligenceEvaluationModel):
    """Independent paraphrases that each start with fresh telemetry/dialogue state."""

    id: str = Field(min_length=1, max_length=100, pattern=r"^[a-z0-9][a-z0-9-]*$")
    family: str = Field(min_length=1, max_length=100, pattern=r"^[a-z0-9][a-z0-9-]*$")
    category: str = Field(min_length=1, max_length=100, pattern=r"^[a-z0-9][a-z0-9-]*$")
    language: Literal["en", "tr"]
    questions: tuple[str, ...] = Field(min_length=1, max_length=100)
    expected: ExpectedContextPlan
    frame_index: int = Field(default=-1, ge=-1)
    tags: tuple[str, ...] = Field(default=(), max_length=20)
    recent_dialogue: tuple[str, ...] = Field(default=(), max_length=MAX_DIALOGUE_HISTORY_ENTRIES)

    @model_validator(mode="after")
    def validate_questions(self) -> "IntelligenceEvaluationGroup":
        if len(set(self.questions)) != len(self.questions):
            raise ValueError("evaluation questions must be unique within a group")
        return self


class IntelligenceEvaluationDataset(IntelligenceEvaluationModel):
    schema_version: Literal["intelligence-eval.v1"] = "intelligence-eval.v1"
    dataset_id: str = Field(min_length=1, max_length=100, pattern=r"^[a-z0-9][a-z0-9-]*$")
    revision: str = Field(min_length=1, max_length=100, pattern=r"^[A-Za-z0-9_.-]+$")
    split: EvaluationSplit
    fixture: str = Field(
        default="fixtures/synthetic/intelligence-capabilities",
        pattern=r"^[A-Za-z0-9_./-]+$",
    )
    groups: tuple[IntelligenceEvaluationGroup, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_dataset(self) -> "IntelligenceEvaluationDataset":
        identifiers = tuple(group.id for group in self.groups)
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("evaluation group IDs must be unique")
        families = tuple(group.family for group in self.groups)
        if len(set(families)) != len(families):
            raise ValueError("paraphrase families must be unique within a split")
        questions = tuple(question for group in self.groups for question in group.questions)
        if len(set(questions)) != len(questions):
            raise ValueError("evaluation questions must be unique within a dataset")
        return self

    @property
    def turn_count(self) -> int:
        return sum(len(group.questions) for group in self.groups)

    def language_counts(self) -> dict[str, int]:
        counts: Counter[str] = Counter()
        for group in self.groups:
            counts[group.language] += len(group.questions)
        return dict(sorted(counts.items()))


class IntelligenceCandidateMetadata(IntelligenceEvaluationModel):
    schema_version: Literal["intelligence-candidate.v1"] = "intelligence-candidate.v1"
    candidate_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_.-]*$", max_length=100)
    runtime_profile: RuntimeProfile
    candidate_kind: CandidateKind
    adapter: str = Field(min_length=1, max_length=100)
    model: str = Field(min_length=1, max_length=200)
    model_revision: str | None = Field(default=None, max_length=160)
    local_only: Literal[True] = True


def load_intelligence_evaluation_dataset(path: Path) -> IntelligenceEvaluationDataset:
    return IntelligenceEvaluationDataset.model_validate_json(path.read_text(encoding="utf-8"))


def validate_intelligence_split_families(
    datasets: tuple[IntelligenceEvaluationDataset, ...],
) -> None:
    ownership: dict[str, EvaluationSplit] = {}
    for dataset in datasets:
        for group in dataset.groups:
            previous = ownership.setdefault(group.family, dataset.split)
            if previous != dataset.split:
                raise ValueError(
                    f"paraphrase family {group.family!r} appears in {previous} and {dataset.split}"
                )
