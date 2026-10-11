"""Lightweight, shared contracts for the experimental trained action classifier."""

from collections.abc import Sequence
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

LABELS = (
    "radio_check",
    "request_information",
    "report_problem",
    "vent",
    "acknowledge",
    "correction",
    "preference_change",
    "communication_feedback",
    "out_of_scope",
    "unclear",
    "greeting",
    "needs_clarification",
)
MODEL_REPO = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
MODEL_REVISION = "e8f8c211226b894fcb81acc59f3b34ba3efd5f42"
MODEL_FILES = (
    "config.json",
    "model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "sentence_bert_config.json",
    "config_sentence_transformers.json",
    "modules.json",
    "1_Pooling/config.json",
    "README.md",
)


def conversation_input(text: str, dialogue: Sequence[tuple[str, str]] = ()) -> str:
    """Identical text-only formatting for training, playground and live inference."""
    history = "\n".join(f"{role}: {message}" for role, message in dialogue)
    return (
        f"Recent dialogue:\n{history}\n" if history else ""
    ) + f"Current driver message:\n{text}"


class ActionModelMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    schema_version: Literal["driver-action-pilot.v1"]
    base_repo: Literal["sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"]
    base_revision: Literal["e8f8c211226b894fcb81acc59f3b34ba3efd5f42"]
    labels: list[str]
    selected_candidate: Literal["frozen_encoder_trained_head", "fine_tuned_encoder"]
    selected_by: Literal["validation_exact_match_ties_prefer_frozen"]
    threshold: float = Field(gt=0, lt=1, allow_inf_nan=False)
    adapter_file: Literal["selected-adapter.safetensors"]
    trained_scope: Literal["conversational_actions_and_clarification_only"]
    draft_labels: Literal[True]
    promoted: Literal[False]


def load_action_metadata(directory: Path) -> ActionModelMetadata:
    metadata = ActionModelMetadata.model_validate_json((directory / "model.json").read_bytes())
    if metadata.labels != list(LABELS):
        raise ValueError("unexpected action label ordering")
    return metadata


class ActionPrediction(BaseModel):
    """Scores are uncalibrated, not confidence estimates or telemetry meaning."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    scores: list[Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]] = Field(
        min_length=len(LABELS), max_length=len(LABELS)
    )
    threshold: float = Field(gt=0, lt=1, allow_inf_nan=False)
    duration_ms: float = Field(ge=0, allow_inf_nan=False)
    input_truncated: bool

    @property
    def actions(self) -> tuple[str, ...]:
        return tuple(
            label
            for label, score in zip(LABELS[:-1], self.scores[:-1], strict=True)
            if score >= self.threshold
        )

    @property
    def clarification_needed(self) -> bool:
        return self.scores[-1] >= self.threshold

    def score(self, label: str) -> float:
        return self.scores[LABELS.index(label)]


class ActionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    request_id: str = Field(min_length=1, max_length=160)
    input_text: str = Field(min_length=1, max_length=16384)
