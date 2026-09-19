"""Versioned, simulator-independent domain contracts."""

import json
from datetime import UTC, datetime
from typing import Annotated, Literal

from pydantic import (
    AfterValidator,
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    model_validator,
)

from race_engineer.core.enums import (
    EventType,
    InterruptionPolicy,
    MessageCategory,
    PlaybackStatus,
    PolicyDecisionOutcome,
    PolicyDecisionReason,
    PreferenceScope,
    PreferenceSource,
    RaceFlag,
    SessionPhase,
    Tone,
    Urgency,
)

NonNegativeFloat = Annotated[float, Field(ge=0, allow_inf_nan=False)]
Confidence = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
Priority = Annotated[int, Field(ge=0, le=100)]


def _normalize_utc(value: datetime) -> datetime:
    return value.astimezone(UTC)


UtcDatetime = Annotated[AwareDatetime, AfterValidator(_normalize_utc)]


class ContractModel(BaseModel):
    """Base for immutable contracts that reject accidental schema drift."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class PlayerState(ContractModel):
    driver_id: str = Field(min_length=1)
    lap_number: int | None = Field(default=None, ge=0)
    position: int | None = Field(default=None, ge=1)
    speed_mps: NonNegativeFloat | None = None
    fuel_l: NonNegativeFloat | None = None
    in_pit_lane: bool = False
    car_ahead_distance_m: NonNegativeFloat | None = None
    car_behind_distance_m: NonNegativeFloat | None = None


class OpponentState(ContractModel):
    driver_id: str = Field(min_length=1)
    position: int | None = Field(default=None, ge=1)
    lap_number: int | None = Field(default=None, ge=0)
    gap_to_player_s: float | None = Field(default=None, allow_inf_nan=False)


class TelemetryFrame(ContractModel):
    schema_version: Literal["telemetry-frame.v1"] = "telemetry-frame.v1"
    source: str = Field(min_length=1)
    session_id: str = Field(min_length=1)
    sequence: int = Field(ge=0)
    observed_at: UtcDatetime
    session_time_s: NonNegativeFloat
    session_phase: SessionPhase = SessionPhase.UNKNOWN
    player: PlayerState
    opponents: tuple[OpponentState, ...] = ()
    flags: tuple[RaceFlag, ...] = ()
    capabilities: tuple[str, ...] = ()
    is_replay: bool = False

    @model_validator(mode="after")
    def validate_ordered_sets(self) -> "TelemetryFrame":
        for name, values in (("flags", self.flags), ("capabilities", self.capabilities)):
            if len(values) != len(set(values)):
                raise ValueError(f"{name} must not contain duplicates")
        if tuple(sorted(self.capabilities)) != self.capabilities:
            raise ValueError("capabilities must be sorted for deterministic serialization")
        return self


class RaceEvent(ContractModel):
    schema_version: Literal["race-event.v1"] = "race-event.v1"
    event_id: str = Field(min_length=1)
    session_id: str = Field(min_length=1)
    source_sequence: int = Field(ge=0)
    occurred_at: UtcDatetime
    event_type: EventType
    facts: dict[str, JsonValue] = Field(default_factory=dict)
    confidence: Confidence = 1.0
    urgency: Urgency = Urgency.ROUTINE
    expires_at: UtcDatetime | None = None

    @model_validator(mode="after")
    def validate_expiry(self) -> "RaceEvent":
        if self.expires_at is not None and self.expires_at <= self.occurred_at:
            raise ValueError("expires_at must be later than occurred_at")
        return self


class RaceContext(ContractModel):
    schema_version: Literal["race-context.v1"] = "race-context.v1"
    frame: TelemetryFrame
    recent_events: tuple[RaceEvent, ...] = ()
    stint_lap: int | None = Field(default=None, ge=0)
    fuel_trend_l_per_lap: NonNegativeFloat | None = None
    gap_ahead_s: float | None = Field(default=None, allow_inf_nan=False)
    gap_behind_s: float | None = Field(default=None, allow_inf_nan=False)
    battle_state: str | None = None

    @model_validator(mode="after")
    def validate_session_consistency(self) -> "RaceContext":
        if any(event.session_id != self.frame.session_id for event in self.recent_events):
            raise ValueError("all recent events must belong to the frame session")
        return self


class CandidateMessage(ContractModel):
    schema_version: Literal["candidate-message.v1"] = "candidate-message.v1"
    candidate_id: str = Field(min_length=1)
    category: MessageCategory
    facts: dict[str, JsonValue] = Field(default_factory=dict)
    base_priority: Priority
    created_at: UtcDatetime
    expires_at: UtcDatetime
    interruptible: bool = True

    @model_validator(mode="after")
    def validate_expiry(self) -> "CandidateMessage":
        if self.expires_at <= self.created_at:
            raise ValueError("expires_at must be later than created_at")
        return self


class PolicyDecision(ContractModel):
    schema_version: Literal["policy-decision.v1"] = "policy-decision.v1"
    decision_id: str = Field(min_length=1)
    candidate_id: str = Field(min_length=1)
    session_id: str = Field(min_length=1)
    source_sequence: int = Field(ge=0)
    decided_at: UtcDatetime
    outcome: PolicyDecisionOutcome
    reason: PolicyDecisionReason
    priority: Priority
    intent_id: str | None = None

    @model_validator(mode="after")
    def validate_outcome(self) -> "PolicyDecision":
        approved = self.outcome is PolicyDecisionOutcome.APPROVED
        if approved and self.reason is not PolicyDecisionReason.APPROVED:
            raise ValueError("approved policy decisions require the approved reason")
        if not approved and self.reason is PolicyDecisionReason.APPROVED:
            raise ValueError("suppressed policy decisions require a suppression reason")
        if approved and self.intent_id is None:
            raise ValueError("approved policy decisions require an intent ID")
        if not approved and self.intent_id is not None:
            raise ValueError("suppressed policy decisions cannot reference an intent")
        return self


class SpeechIntent(ContractModel):
    schema_version: Literal["speech-intent.v1"] = "speech-intent.v1"
    intent_id: str = Field(min_length=1)
    category: MessageCategory
    facts: dict[str, JsonValue] = Field(default_factory=dict)
    priority: Priority
    tone: Tone = Tone.NEUTRAL
    max_words: int = Field(default=20, ge=1, le=50)
    deadline: UtcDatetime
    interruption_policy: InterruptionPolicy = InterruptionPolicy.NEVER
    critical_template: str | None = None
    language: str = Field(default="en", min_length=2, max_length=35)


class Utterance(ContractModel):
    schema_version: Literal["utterance.v1"] = "utterance.v1"
    intent_id: str = Field(min_length=1)
    text: str = Field(min_length=1, max_length=500)
    pronunciation_hints: dict[str, str] = Field(default_factory=dict)
    generator_metadata: dict[str, JsonValue] = Field(default_factory=dict)


class PreferenceCommand(ContractModel):
    schema_version: Literal["preference-command.v1"] = "preference-command.v1"
    command_id: str = Field(min_length=1)
    setting: str = Field(min_length=1)
    value: JsonValue
    source: PreferenceSource
    timestamp: UtcDatetime
    scope: PreferenceScope


class PlaybackResult(ContractModel):
    schema_version: Literal["playback-result.v1"] = "playback-result.v1"
    intent_id: str = Field(min_length=1)
    status: PlaybackStatus
    started_at: UtcDatetime | None = None
    finished_at: UtcDatetime | None = None
    error_code: str | None = None

    @model_validator(mode="after")
    def validate_timing(self) -> "PlaybackResult":
        if (
            self.started_at is not None
            and self.finished_at is not None
            and self.finished_at < self.started_at
        ):
            raise ValueError("finished_at must not be earlier than started_at")
        return self


def canonical_json(model: ContractModel) -> str:
    """Serialize a contract deterministically for snapshots and decision logs."""

    return json.dumps(
        model.model_dump(mode="json", exclude_none=False, by_alias=True),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
