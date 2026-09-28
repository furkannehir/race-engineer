"""Vendor-independent contracts for grounded, read-only race conversations."""

from enum import StrEnum
from typing import Literal

from pydantic import Field, model_validator

from race_engineer.core.contracts import ContractModel, RaceContext, UtcDatetime

type RadioLanguage = Literal["en", "tr"]
type FieldRelation = Literal[
    "first",
    "last",
    "cars_ahead",
    "cars_behind",
    "position_of_total",
]


class RaceQuery(StrEnum):
    POSITION = "position"
    FIELD_STATUS = "field_status"
    LAP = "lap"
    GAP_AHEAD = "gap_ahead"
    GAP_BEHIND = "gap_behind"
    FUEL_REMAINING = "fuel_remaining"
    FUEL_CONSUMPTION = "fuel_consumption"
    FUEL_TO_FINISH = "fuel_to_finish"
    GAP_TREND_AHEAD = "gap_trend_ahead"
    GAP_TREND_BEHIND = "gap_trend_behind"
    UNSUPPORTED = "unsupported"


class ConversationPlan(ContractModel):
    schema_version: Literal["conversation-plan.v1"] = "conversation-plan.v1"
    language: RadioLanguage
    queries: tuple[RaceQuery, ...] = Field(max_length=6)
    clarification: Literal["none", "topic", "opponent"]

    @model_validator(mode="after")
    def validate_queries(self) -> "ConversationPlan":
        if len(self.queries) != len(set(self.queries)):
            raise ValueError("queries must be unique")
        if (self.clarification == "none") != bool(self.queries):
            raise ValueError("supply queries or a clarification, never both or neither")
        return self


class ConversationTurn(ContractModel):
    """Remember meaning, not old telemetry values or generated answers."""

    question: str = Field(min_length=1, max_length=1000)
    plan: ConversationPlan


class ConversationRequest(ContractModel):
    question: str = Field(min_length=1, max_length=1000)
    history: tuple[ConversationTurn, ...] = ()
    default_language: RadioLanguage = "en"
    reply_language: RadioLanguage | None = None


class RaceSnapshot(ContractModel):
    context: RaceContext
    # Replay uses the simulator clock; live providers must use current UTC.
    as_of: UtcDatetime
    mode: Literal["replay", "live"]


class RaceAnswer(ContractModel):
    query: RaceQuery
    status: Literal["available", "missing", "unsupported"]
    value: int | float | None = None
    unit: Literal["position", "lap", "s", "l", "l/lap"] | None = None
    total: int | None = Field(default=None, ge=1)
    field_relation: FieldRelation | None = None

    @model_validator(mode="after")
    def validate_field_status(self) -> "RaceAnswer":
        if self.query is not RaceQuery.FIELD_STATUS:
            if self.total is not None or self.field_relation is not None:
                raise ValueError("only field status carries field comparison data")
            return self
        if self.status != "available":
            if self.total is not None:
                raise ValueError("unavailable field status cannot carry a total")
            return self
        if (
            not isinstance(self.value, int)
            or isinstance(self.value, bool)
            or self.unit != "position"
            or self.total is None
            or self.field_relation is None
            or self.value < 1
            or self.value > self.total
        ):
            raise ValueError("available field status needs a relation, valid position and total")
        return self


class ConversationReply(ContractModel):
    schema_version: Literal["conversation-reply.v1"] = "conversation-reply.v1"
    language: RadioLanguage
    text: str = Field(min_length=1, max_length=1500)
    status: Literal["answered", "clarification", "unavailable", "model_error"]
    session_id: str
    source_sequence: int
    mode: Literal["replay", "live"]
    answers: tuple[RaceAnswer, ...] = ()
    reason: str | None = None
