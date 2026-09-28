"""Contracts for evidence-driven contextual and conversational intelligence."""

import re
from typing import Literal

from pydantic import Field, JsonValue, model_validator

from race_engineer.core.contracts import Confidence, ContractModel, UtcDatetime

type EngineerLanguage = Literal["en", "tr"]
type EvidenceKind = Literal["measurement", "derived", "inference", "unknown"]
type EngineerGoal = Literal[
    "inform",
    "analyze",
    "coach",
    "acknowledge",
    "clarify",
    "silence",
]
type ResponseAction = Literal["speak", "clarify", "silence"]
type ReferenceField = Literal["value", "claim"]
type TelemetrySource = Literal["player", "context", "opponent"]
type TelemetryOperation = Literal["latest", "delta", "mean", "minimum", "maximum", "trend"]

_PLACEHOLDER = re.compile(r"\{\{([a-z][a-z0-9_]*)\}\}")
_RAW_NUMBER = re.compile(r"\d")


class DriverTurn(ContractModel):
    """One driver utterance scoped to one live/replay session generation."""

    schema_version: Literal["driver-turn.v1"] = "driver-turn.v1"
    turn_id: str = Field(min_length=1, max_length=160)
    transcript: str = Field(min_length=1, max_length=1000, pattern=r"\S")
    received_at: UtcDatetime
    session_id: str = Field(min_length=1, max_length=160)
    generation: int = Field(ge=0)
    asr_language: EngineerLanguage | None = None
    reply_language: EngineerLanguage | None = None
    recent_dialogue: tuple[str, ...] = Field(default=(), max_length=8)


class SignalSelector(ContractModel):
    """A simulator-independent path to one normalized numeric signal."""

    source: TelemetrySource
    signal: str = Field(pattern=r"^[a-z][a-z0-9_]*$", max_length=80)
    subject_id: str | None = Field(default=None, min_length=1, max_length=160)

    @model_validator(mode="after")
    def validate_subject(self) -> "SignalSelector":
        if (self.source == "opponent") != (self.subject_id is not None):
            raise ValueError("only opponent selectors require a subject ID")
        return self


class EvidenceQuery(ContractModel):
    """A generic numerical request; it is not a spoken intent or answer template."""

    query_id: str = Field(min_length=1, max_length=160)
    selector: SignalSelector
    operation: TelemetryOperation
    window_s: float | None = Field(default=None, gt=0, le=3600, allow_inf_nan=False)

    @model_validator(mode="after")
    def validate_window(self) -> "EvidenceQuery":
        if self.operation == "latest" and self.window_s is not None:
            raise ValueError("latest queries do not use a window")
        if self.operation != "latest" and self.window_s is None:
            raise ValueError("window operations require a duration")
        return self


class EvidenceItem(ContractModel):
    """A measured, derived, inferred, or explicitly unknown piece of race evidence."""

    schema_version: Literal["evidence-item.v1"] = "evidence-item.v1"
    evidence_id: str = Field(min_length=1, max_length=160)
    session_id: str = Field(min_length=1, max_length=160)
    source_sequence: int = Field(ge=0)
    observed_at: UtcDatetime
    kind: EvidenceKind
    subject: str = Field(min_length=1, max_length=160)
    metric: str = Field(min_length=1, max_length=160)
    value: JsonValue = None
    unit: str | None = Field(default=None, min_length=1, max_length=40)
    claim: str | None = Field(default=None, min_length=1, max_length=500)
    confidence: Confidence = 1.0
    source_fields: tuple[str, ...] = Field(default=(), max_length=16)
    valid_until: UtcDatetime | None = None

    @model_validator(mode="after")
    def validate_evidence(self) -> "EvidenceItem":
        if self.kind in {"measurement", "derived"} and self.value is None:
            raise ValueError("measured and derived evidence require a value")
        if self.kind == "inference" and self.claim is None:
            raise ValueError("inferred evidence requires a claim")
        if self.kind == "unknown" and (self.value is not None or self.claim is not None):
            raise ValueError("unknown evidence cannot carry a value or claim")
        if self.kind != "unknown" and not self.source_fields:
            raise ValueError("known evidence requires source provenance")
        if self.unit is not None and self.value is None:
            raise ValueError("units require a value")
        if self.valid_until is not None and self.valid_until <= self.observed_at:
            raise ValueError("evidence validity must extend beyond observation")
        return self


class ContextPacket(ContractModel):
    """The Context Engineer's evidence selection for one driver turn."""

    schema_version: Literal["context-packet.v1"] = "context-packet.v1"
    turn_id: str = Field(min_length=1, max_length=160)
    session_id: str = Field(min_length=1, max_length=160)
    generation: int = Field(ge=0)
    source_sequence: int = Field(ge=0)
    assembled_at: UtcDatetime
    situation: tuple[str, ...] = Field(default=(), max_length=16)
    evidence: tuple[EvidenceItem, ...] = Field(default=(), max_length=64)
    unknowns: tuple[str, ...] = Field(default=(), max_length=16)

    @model_validator(mode="after")
    def validate_scope(self) -> "ContextPacket":
        if len({item.evidence_id for item in self.evidence}) != len(self.evidence):
            raise ValueError("evidence IDs must be unique")
        if any(item.session_id != self.session_id for item in self.evidence):
            raise ValueError("all evidence must belong to the packet session")
        if any(item.source_sequence > self.source_sequence for item in self.evidence):
            raise ValueError("evidence cannot come from a future sequence")
        return self


class EngineerBrief(ContractModel):
    """The Core Engineer's evidence-backed communication decision."""

    schema_version: Literal["engineer-brief.v1"] = "engineer-brief.v1"
    turn_id: str = Field(min_length=1, max_length=160)
    session_id: str = Field(min_length=1, max_length=160)
    generation: int = Field(ge=0)
    source_sequence: int = Field(ge=0)
    goal: EngineerGoal
    language: EngineerLanguage
    tone: str = Field(min_length=1, max_length=80)
    evidence_ids: tuple[str, ...] = Field(default=(), max_length=32)
    guidance: tuple[str, ...] = Field(default=(), max_length=12)
    confidence: Confidence = 1.0

    @model_validator(mode="after")
    def validate_brief(self) -> "EngineerBrief":
        if len(set(self.evidence_ids)) != len(self.evidence_ids):
            raise ValueError("brief evidence IDs must be unique")
        if self.goal == "silence" and (self.evidence_ids or self.guidance):
            raise ValueError("a silent brief cannot request evidence or guidance")
        return self


class EvidenceReference(ContractModel):
    """One placeholder in generated speech, bound to one evidence field."""

    placeholder: str = Field(pattern=r"^[a-z][a-z0-9_]*$", max_length=80)
    evidence_id: str = Field(min_length=1, max_length=160)
    field: ReferenceField


class GeneratedResponse(ContractModel):
    """Qwen-authored speech with explicit evidence references for dynamic claims."""

    schema_version: Literal["generated-response.v1"] = "generated-response.v1"
    response_id: str = Field(min_length=1, max_length=160)
    turn_id: str = Field(min_length=1, max_length=160)
    session_id: str = Field(min_length=1, max_length=160)
    generation: int = Field(ge=0)
    language: EngineerLanguage
    action: ResponseAction
    speech_template: str | None = Field(default=None, min_length=1, max_length=1500)
    references: tuple[EvidenceReference, ...] = Field(default=(), max_length=32)

    @model_validator(mode="after")
    def validate_response(self) -> "GeneratedResponse":
        if self.action == "silence":
            if self.speech_template is not None or self.references:
                raise ValueError("silent responses cannot carry speech or references")
            return self
        if self.speech_template is None:
            raise ValueError("spoken responses require a speech template")
        placeholders = _PLACEHOLDER.findall(self.speech_template)
        reference_names = [reference.placeholder for reference in self.references]
        if len(reference_names) != len(set(reference_names)):
            raise ValueError("response placeholders must be unique")
        if sorted(placeholders) != sorted(reference_names):
            raise ValueError("every speech placeholder requires exactly one reference")
        text_without_placeholders = _PLACEHOLDER.sub("", self.speech_template)
        if _RAW_NUMBER.search(text_without_placeholders):
            raise ValueError("numeric telemetry must use an evidence placeholder")
        return self


class GroundedResponse(ContractModel):
    """The final validated text that may be handed to the radio/TTS layer."""

    schema_version: Literal["grounded-response.v1"] = "grounded-response.v1"
    response_id: str = Field(min_length=1, max_length=160)
    turn_id: str = Field(min_length=1, max_length=160)
    session_id: str = Field(min_length=1, max_length=160)
    generation: int = Field(ge=0)
    source_sequence: int = Field(ge=0)
    language: EngineerLanguage
    action: ResponseAction
    text: str | None = Field(default=None, min_length=1, max_length=1500)
    evidence_ids: tuple[str, ...] = Field(default=(), max_length=32)

    @model_validator(mode="after")
    def validate_grounded_response(self) -> "GroundedResponse":
        if (self.action == "silence") != (self.text is None):
            raise ValueError("only silent grounded responses omit text")
        if len(set(self.evidence_ids)) != len(self.evidence_ids):
            raise ValueError("grounded evidence IDs must be unique")
        return self
