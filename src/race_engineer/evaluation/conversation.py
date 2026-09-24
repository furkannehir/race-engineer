"""Versioned bilingual conversation datasets and content-free result scoring."""

from collections import Counter
from pathlib import Path
from typing import Literal, cast

from pydantic import BaseModel, ConfigDict, Field, model_validator

from race_engineer.core.conversation import RaceQuery, RadioLanguage

type ReplyStatus = Literal["answered", "clarification", "unavailable", "model_error"]
type Clarification = Literal["none", "topic", "opponent"]


class EvaluationModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ExpectedTurn(EvaluationModel):
    language: RadioLanguage
    queries: tuple[RaceQuery, ...] = Field(max_length=6)
    clarification: Clarification = "none"
    reply_status: ReplyStatus

    @model_validator(mode="after")
    def validate_plan_shape(self) -> "ExpectedTurn":
        if len(self.queries) != len(set(self.queries)):
            raise ValueError("expected queries must be unique")
        has_queries = bool(self.queries)
        if (self.clarification == "none") != has_queries:
            raise ValueError("expected queries or a clarification, never both or neither")
        if self.clarification != "none" and self.reply_status != "clarification":
            raise ValueError("clarification plans require clarification reply status")
        return self


class EvaluationGroup(EvaluationModel):
    """Independent paraphrases; every question starts with fresh dialogue history."""

    id: str = Field(min_length=1, max_length=100, pattern=r"^[a-z0-9][a-z0-9-]*$")
    family: str = Field(min_length=1, max_length=100, pattern=r"^[a-z0-9][a-z0-9-]*$")
    category: str = Field(min_length=1, max_length=100, pattern=r"^[a-z0-9][a-z0-9-]*$")
    questions: tuple[str, ...] = Field(min_length=1, max_length=100)
    expected: ExpectedTurn
    frame_index: int = Field(default=0, ge=0)
    tags: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_questions(self) -> "EvaluationGroup":
        if len(self.questions) != len(set(self.questions)):
            raise ValueError("evaluation questions must be unique within a group")
        return self


class EvaluationSequenceTurn(EvaluationModel):
    question: str = Field(min_length=1, max_length=1000)
    expected: ExpectedTurn
    frame_index: int | None = Field(default=None, ge=0)


class EvaluationSequence(EvaluationModel):
    """Ordered multi-turn dialogue; history is retained within the sequence."""

    id: str = Field(min_length=1, max_length=100, pattern=r"^[a-z0-9][a-z0-9-]*$")
    family: str = Field(min_length=1, max_length=100, pattern=r"^[a-z0-9][a-z0-9-]*$")
    category: str = Field(min_length=1, max_length=100, pattern=r"^[a-z0-9][a-z0-9-]*$")
    turns: tuple[EvaluationSequenceTurn, ...] = Field(min_length=1, max_length=20)
    tags: tuple[str, ...] = ()


class ConversationEvaluationDataset(EvaluationModel):
    schema_version: Literal["conversation-eval.v1"] = "conversation-eval.v1"
    dataset_id: str = Field(min_length=1, max_length=100, pattern=r"^[a-z0-9][a-z0-9-]*$")
    revision: str = Field(min_length=1, max_length=100, pattern=r"^[A-Za-z0-9_.-]+$")
    split: Literal["development", "calibration", "locked"]
    fixture: str = Field(
        default="fixtures/synthetic/conversation",
        pattern=r"^[A-Za-z0-9_./-]+$",
    )
    groups: tuple[EvaluationGroup, ...] = ()
    sequences: tuple[EvaluationSequence, ...] = ()

    @model_validator(mode="after")
    def validate_dataset(self) -> "ConversationEvaluationDataset":
        cases: tuple[EvaluationGroup | EvaluationSequence, ...] = (
            *self.groups,
            *self.sequences,
        )
        if not cases:
            raise ValueError("evaluation dataset must contain cases")
        identifiers = [case.id for case in cases]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("evaluation case IDs must be unique")
        families = [case.family for case in cases]
        if len(families) != len(set(families)):
            raise ValueError("paraphrase families must be unique within a split")
        questions = [question for group in self.groups for question in group.questions]
        questions.extend(turn.question for sequence in self.sequences for turn in sequence.turns)
        if len(questions) != len(set(questions)):
            raise ValueError("evaluation questions must be unique within a dataset")
        if self.split == "locked":
            counts = self.language_counts()
            if counts.get("en", 0) < 100 or counts.get("tr", 0) < 100:
                raise ValueError("locked datasets require at least 100 turns per language")
        return self

    def language_counts(self) -> dict[str, int]:
        counts: Counter[str] = Counter()
        for group in self.groups:
            counts[group.expected.language] += len(group.questions)
        for sequence in self.sequences:
            counts.update(turn.expected.language for turn in sequence.turns)
        return dict(sorted(counts.items()))

    @property
    def turn_count(self) -> int:
        return sum(len(group.questions) for group in self.groups) + sum(
            len(sequence.turns) for sequence in self.sequences
        )


def load_evaluation_dataset(path: Path) -> ConversationEvaluationDataset:
    return ConversationEvaluationDataset.model_validate_json(path.read_text(encoding="utf-8"))


def validate_split_families(datasets: tuple[ConversationEvaluationDataset, ...]) -> None:
    ownership: dict[str, str] = {}
    for dataset in datasets:
        cases = cast(
            tuple[EvaluationGroup | EvaluationSequence, ...],
            (*dataset.groups, *dataset.sequences),
        )
        for case in cases:
            previous = ownership.setdefault(case.family, dataset.split)
            if previous != dataset.split:
                raise ValueError(
                    f"paraphrase family {case.family!r} appears in {previous} and {dataset.split}"
                )
