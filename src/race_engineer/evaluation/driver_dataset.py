"""Review-only driver-meaning corpus contracts; not a live router or training loader."""

import hashlib
import json
import re
import unicodedata
from collections import Counter
from typing import Annotated, Literal

from pydantic import Field, model_validator

from race_engineer.core.contracts import ContractModel, UtcDatetime

type Text = Annotated[str, Field(min_length=1, max_length=1000, pattern=r"\S")]
type Identifier = Annotated[str, Field(pattern=r"^[a-z][a-z0-9_-]*$", max_length=100)]
type DriverAction = Literal[
    "radio_check", "request_information", "report_problem", "vent", "acknowledge",
    "correction", "preference_change", "communication_feedback", "out_of_scope", "unclear",
    "greeting",
]


class DialogueEntry(ContractModel):
    role: Literal["driver", "engineer"]
    text: Text


class AnnotationReview(ContractModel):
    status: Literal["draft", "approved", "rejected"]
    reviewer: Text | None = None
    reviewed_at: UtcDatetime | None = None

    @model_validator(mode="after")
    def validate_review(self) -> "AnnotationReview":
        if self.status == "draft":
            if self.reviewer is not None or self.reviewed_at is not None:
                raise ValueError("draft annotations cannot claim a completed review")
        elif self.reviewer is None or self.reviewed_at is None:
            raise ValueError("approved/rejected annotations require reviewer and timestamp")
        return self


class MeaningRequest(ContractModel):
    request_id: Identifier
    source_span: Text
    focus: Identifier  # Open vocabulary, not the six-topic experiment's enum.
    question: Text  # Intended question, never an answer or executable telemetry query.
    subject: Identifier
    reference_basis: Literal["explicit", "radio_convention", "dialogue", "unresolved"]
    time_scope: Literal["current", "historical", "future_counterfactual", "unclear"]
    window_seconds: float | None = Field(default=None, gt=0, le=3600, allow_inf_nan=False)
    resolution: Literal["clear", "ambiguous"]

    @model_validator(mode="after")
    def validate_reference(self) -> "MeaningRequest":
        if (self.subject == "unresolved") != (self.reference_basis == "unresolved"):
            raise ValueError("unresolved subjects and reference basis must agree")
        if self.resolution == "clear" and (
            self.subject == "unresolved" or self.focus == "unspecified"
            or self.time_scope == "unclear"
        ):
            raise ValueError("clear requests cannot contain unresolved required meaning")
        if self.window_seconds is not None and self.time_scope != "historical":
            raise ValueError("elapsed windows require historical scope")
        return self


class MeaningAnnotation(ContractModel):
    actions: tuple[DriverAction, ...] = Field(min_length=1, max_length=8)
    meaning: Text
    requests: tuple[MeaningRequest, ...] = Field(max_length=8)
    interpretation_status: Literal["clear", "ambiguous"]
    clarification_needed: bool
    ambiguities: tuple[Text, ...] = Field(max_length=8)
    rationale: Text

    @model_validator(mode="after")
    def validate_meaning(self) -> "MeaningAnnotation":
        if len(set(self.actions)) != len(self.actions):
            raise ValueError("duplicate driver actions")
        if bool(self.requests) != ("request_information" in self.actions):
            raise ValueError("information requests and request_information action must agree")
        if len({request.request_id for request in self.requests}) != len(self.requests):
            raise ValueError("duplicate request IDs")
        if (self.interpretation_status == "ambiguous") != bool(self.ambiguities):
            raise ValueError("ambiguity status must agree with explicit ambiguity notes")
        unresolved_request = any(request.resolution == "ambiguous" for request in self.requests)
        if unresolved_request and not self.clarification_needed:
            raise ValueError("unresolved requests must flag clarification")
        if ((unresolved_request or self.clarification_needed)
                and self.interpretation_status != "ambiguous"):
            raise ValueError("clarification cannot claim a clear interpretation")
        return self


class DriverMeaningExample(ContractModel):
    example_id: Identifier
    scenario_id: Identifier
    leakage_group: Identifier
    language: Literal["en", "tr"]
    speaker: Literal["driver"]
    text: Text
    recent_dialogue: tuple[DialogueEntry, ...] = Field(max_length=6)
    origin: Literal["assistant_authored"]
    prior_exposure: Literal["previously_probed", "newly_authored"]
    split: Literal["unassigned"]
    review: AnnotationReview
    annotation: MeaningAnnotation
    coverage_tags: tuple[Identifier, ...] = Field(default=(), max_length=12)
    input_kind: Literal["typed", "synthetic_asr"] = "typed"
    intended_text: Text | None = None

    @model_validator(mode="after")
    def validate_spans(self) -> "DriverMeaningExample":
        if len(set(self.coverage_tags)) != len(self.coverage_tags):
            raise ValueError("duplicate coverage tags")
        if (self.input_kind == "synthetic_asr") != (self.intended_text is not None):
            raise ValueError("synthetic ASR examples require a separate intended utterance")
        for request in self.annotation.requests:
            if request.source_span not in self.text:
                raise ValueError("request source span must occur in the current driver message")
            if request.reference_basis == "dialogue" and not self.recent_dialogue:
                raise ValueError("dialogue reference requires supplied dialogue")
        return self


def input_signature(example: DriverMeaningExample) -> str:
    """Detect punctuation/casing/spacing-only duplicates without merging distinct histories."""
    def normalize(text: str) -> str:
        value = unicodedata.normalize("NFKC", text).casefold()
        return " ".join(re.sub(r"[^\w\s]", "", value).split())

    return json.dumps({
        "language": example.language, "text": normalize(example.text),
        "history": [(entry.role, normalize(entry.text)) for entry in example.recent_dialogue],
    }, ensure_ascii=False, sort_keys=True)


class DriverMeaningDataset(ContractModel):
    schema_version: Literal["driver-meaning-dataset.v1"]
    dataset_id: Identifier
    revision: Text
    purpose: Literal["annotation_review"]
    context_assumptions: tuple[Text, ...] = Field(min_length=1, max_length=8)
    examples: tuple[DriverMeaningExample, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_examples(self) -> "DriverMeaningDataset":
        ids = [example.example_id for example in self.examples]
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate example IDs")
        signatures = [input_signature(example) for example in self.examples]
        if len(set(signatures)) != len(signatures):
            raise ValueError("duplicate full inputs")
        scenario_groups: dict[str, str] = {}
        for example in self.examples:
            previous = scenario_groups.setdefault(example.scenario_id, example.leakage_group)
            if previous != example.leakage_group:
                raise ValueError("related scenario examples must share a leakage group")
        return self


def dataset_summary(dataset: DriverMeaningDataset) -> dict[str, object]:
    """Content-free inventory, not model accuracy or permission to train."""
    return {
        "dataset_id": dataset.dataset_id,
        "revision": dataset.revision,
        "examples": len(dataset.examples),
        "languages": dict(Counter(example.language for example in dataset.examples)),
        "scenarios": len({example.scenario_id for example in dataset.examples}),
        "leakage_groups": len({example.leakage_group for example in dataset.examples}),
        "review_status": dict(Counter(example.review.status for example in dataset.examples)),
        "prior_exposure": dict(Counter(example.prior_exposure for example in dataset.examples)),
        "clarification_examples": sum(
            example.annotation.clarification_needed for example in dataset.examples
        ),
        "compound_examples": sum(
            len(example.annotation.requests) > 1 for example in dataset.examples
        ),
        "actions": dict(Counter(
            action for example in dataset.examples for action in example.annotation.actions
        )),
        "request_focuses": dict(Counter(
            request.focus for example in dataset.examples for request in example.annotation.requests
        )),
        "time_scopes": dict(Counter(
            request.time_scope for example in dataset.examples
            for request in example.annotation.requests
        )),
        "dialogue_examples": sum(bool(example.recent_dialogue) for example in dataset.examples),
        "synthetic_asr_examples": sum(
            example.input_kind == "synthetic_asr" for example in dataset.examples
        ),
        "coverage_tags": dict(Counter(
            tag for example in dataset.examples for tag in example.coverage_tags
        )),
        "training_ready": False,
        "split": "unassigned",
        "sha256": hashlib.sha256(dataset.model_dump_json().encode("utf-8")).hexdigest(),
    }
