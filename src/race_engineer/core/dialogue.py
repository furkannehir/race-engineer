"""Immutable CE-04 dialogue contracts; no model, telemetry SDK, or persistence imports."""

from typing import Annotated, Literal

from pydantic import Field, model_validator

from race_engineer.core.contracts import ContractModel, UtcDatetime
from race_engineer.core.conversation import FieldRelation, RaceAnswer, RadioLanguage

type DialogueQuery = Literal[
    "position",
    "field_status",
    "lap",
    "gap",
    "gap_trend",
    "fuel_remaining",
    "fuel_consumption",
    "fuel_to_finish",
    "unsupported",
]
type OpponentSide = Literal["ahead", "behind"]
type ReferenceKind = Literal["none", "ahead", "behind", "active", "unspecified"]
type SocialAct = Literal["acknowledge", "close"]
type ClarificationKind = Literal["none", "topic", "opponent"]
type OutputMode = Literal["speech", "text"]
type DeliveryStatus = Literal[
    "accepted",
    "queued",
    "started",
    "completed",
    "interrupted",
    "cancelled",
    "expired",
    "failed",
]
type DecisionReason = Literal[
    "resolved",
    "reference_unclear",
    "topic_unclear",
    "uncertain",
    "clarification_failed",
    "clarification_expired",
    "missing_facts",
    "unsupported",
    "stale_snapshot",
    "session_changed",
    "state_changed",
    "deadline",
    "busy",
    "cancelled",
    "model_error",
]
type DecisionOutcome = Literal[
    "answered",
    "partial",
    "clarify",
    "acknowledge",
    "unavailable",
    "no_reply",
    "discarded",
]
type UtteranceClauseKind = Literal[
    "acknowledgment",
    "fact",
    "clarification",
    "failure",
]
type UtteranceTone = Literal["concise", "neutral", "supportive"]
Identifier = Annotated[str, Field(min_length=1, max_length=160)]


class QueryPart(ContractModel):
    part_id: Identifier
    query: DialogueQuery
    reference: ReferenceKind = "none"
    field_relation: FieldRelation | None = None

    @model_validator(mode="after")
    def validate_reference(self) -> "QueryPart":
        if (self.query in {"gap", "gap_trend"}) != (self.reference != "none"):
            raise ValueError("only opponent queries require a reference")
        if (self.query == "field_status") != (self.field_relation is not None):
            raise ValueError("field-status queries require exactly one field relation")
        return self


class SemanticProposal(ContractModel):
    schema_version: Literal["semantic-proposal.v2"] = "semantic-proposal.v2"
    language: RadioLanguage
    mode: Literal["request", "follow_up", "correction", "clarification_answer", "repeat"] = (
        "request"
    )
    requests: tuple[QueryPart, ...] = Field(default=(), max_length=6)
    reference: OpponentSide | None = None
    acts: tuple[SocialAct, ...] = Field(default=(), max_length=2)
    clarification: ClarificationKind = "none"
    abstain: bool = False
    model_id: Identifier
    calibration_id: Identifier | None = None

    @model_validator(mode="after")
    def validate_proposal(self) -> "SemanticProposal":
        if len({part.part_id for part in self.requests}) != len(self.requests):
            raise ValueError("request part IDs must be unique")
        if len(set(self.acts)) != len(self.acts):
            raise ValueError("response acts must be unique")
        if self.abstain:
            if self.requests or self.acts or self.reference or self.clarification != "none":
                raise ValueError("abstention cannot contain a proposed action")
        elif self.mode == "request" and not (
            self.requests or self.acts or self.clarification != "none"
        ):
            raise ValueError("a request needs a meaning or clarification")
        if self.reference is not None and (
            self.mode not in {"follow_up", "correction", "clarification_answer"} or self.requests
        ):
            raise ValueError("reference-only completion requires an elliptical turn")
        if self.mode in {"follow_up", "repeat"} and self.requests:
            raise ValueError("explicit queries use request or correction mode")
        if self.mode == "repeat" and (self.acts or self.clarification != "none"):
            raise ValueError("repeat cannot add acts or clarification")
        if "close" in self.acts and (
            self.requests
            or len(self.acts) != 1
            or self.mode != "request"
            or self.clarification != "none"
        ):
            raise ValueError("closing cannot discard another request")
        return self


class OpponentReference(ContractModel):
    side: OpponentSide
    driver_id: Identifier


class RememberedRequest(ContractModel):
    part: QueryPart
    opponent: OpponentReference | None = None
    reference_expires_at: UtcDatetime | None = None

    @model_validator(mode="after")
    def validate_anchor(self) -> "RememberedRequest":
        if (self.opponent is None) != (self.reference_expires_at is None):
            raise ValueError("opponent anchors require an expiry")
        if self.opponent is not None and self.part.reference != self.opponent.side:
            raise ValueError("stored reference must match the anchored side")
        return self


class TopicMemory(ContractModel):
    requests: tuple[RememberedRequest, ...] = Field(min_length=1, max_length=6)
    expires_at: UtcDatetime


class PendingClarification(ContractModel):
    origin_turn_id: Identifier
    kind: Literal["topic", "opponent"]
    requests: tuple[QueryPart, ...] = Field(default=(), max_length=6)
    expires_at: UtcDatetime


class TurnMemory(ContractModel):
    turn_id: Identifier
    received_at: UtcDatetime
    language: RadioLanguage
    question: str | None = Field(default=None, max_length=1000)
    requests: tuple[RememberedRequest, ...] = Field(default=(), max_length=6)
    acts: tuple[SocialAct, ...] = ()


class ResponseMemory(ContractModel):
    turn_id: Identifier
    response_id: Identifier
    created_at: UtcDatetime
    expires_at: UtcDatetime
    requests: tuple[RememberedRequest, ...] = Field(default=(), max_length=6)
    acts: tuple[SocialAct, ...] = ()
    language: RadioLanguage
    output_mode: OutputMode
    delivery: DeliveryStatus = "accepted"
    delivery_at: UtcDatetime | None = None


class DialogueState(ContractModel):
    schema_version: Literal["dialogue-state.v2"] = "dialogue-state.v2"
    session_id: Identifier | None = None
    generation: int = Field(default=0, ge=0)
    revision: int = Field(default=0, ge=0)
    topic: TopicMemory | None = None
    pending: PendingClarification | None = None
    history: tuple[TurnMemory, ...] = Field(default=(), max_length=12)
    responses: tuple[ResponseMemory, ...] = Field(default=(), max_length=12)


class QueryCapability(ContractModel):
    query: DialogueQuery
    reference: Literal["none", "ahead", "behind"] = "none"
    status: Literal["available", "missing", "unsupported"]


class EventSummary(ContractModel):
    event_id: Identifier
    category: str = Field(min_length=1, max_length=64)
    occurred_at: UtcDatetime


class ConversationContext(ContractModel):
    schema_version: Literal["conversation-context.v1"] = "conversation-context.v1"
    session_id: Identifier
    generation: int = Field(ge=0)
    source_sequence: int = Field(ge=0)
    observed_at: UtcDatetime
    as_of: UtcDatetime
    capabilities: tuple[QueryCapability, ...] = Field(max_length=12)
    opponents: tuple[OpponentReference, ...] = Field(default=(), max_length=2)
    recent_events: tuple[EventSummary, ...] = Field(default=(), max_length=16)
    battle_state: Literal["clear", "attacking", "defending", "sandwiched", "contested"] | None


class DialogueTurnInput(ContractModel):
    schema_version: Literal["dialogue-turn-input.v2"] = "dialogue-turn-input.v2"
    turn_id: Identifier
    question: str = Field(min_length=1, max_length=1000, pattern=r"\S")
    received_at: UtcDatetime
    deadline: UtcDatetime
    asr_language: RadioLanguage | None = None
    reply_language: RadioLanguage | None = None
    default_language: RadioLanguage = "en"
    context: ConversationContext
    dialogue: DialogueState

    @model_validator(mode="after")
    def validate_scope(self) -> "DialogueTurnInput":
        if self.deadline <= self.received_at:
            raise ValueError("turn deadline must follow receipt")
        if (self.context.session_id, self.context.generation) != (
            self.dialogue.session_id,
            self.dialogue.generation,
        ):
            raise ValueError("dialogue and race context must have the same scope")
        return self


class GroundedAnswer(RaceAnswer):
    schema_version: Literal["grounded-answer.v2"] = "grounded-answer.v2"
    part_id: Identifier
    source_sequence: int = Field(ge=0)
    opponent: OpponentReference | None = None
    value: int | float | None = Field(default=None, ge=0, allow_inf_nan=False)

    @model_validator(mode="after")
    def validate_value(self) -> "GroundedAnswer":
        if self.status == "available":
            if self.value is None or self.unit is None:
                raise ValueError("available answers need a value and unit")
        elif self.value is not None or self.unit is not None:
            raise ValueError("unavailable answers cannot contain values")
        return self


class DialogueUpdate(ContractModel):
    """A controller proposal. Only the owning session can commit it."""

    expected_revision: int = Field(ge=0)
    topic: TopicMemory | None
    pending: PendingClarification | None
    turn: TurnMemory


class ResponseDecision(ContractModel):
    schema_version: Literal["response-decision.v2"] = "response-decision.v2"
    turn_id: Identifier
    response_id: Identifier
    session_id: Identifier
    generation: int = Field(ge=0)
    source_sequence: int = Field(ge=0)
    language: RadioLanguage
    created_at: UtcDatetime
    expires_at: UtcDatetime
    outcome: DecisionOutcome
    reason: DecisionReason
    answers: tuple[GroundedAnswer, ...] = Field(default=(), max_length=6)
    acts: tuple[SocialAct, ...] = Field(default=(), max_length=2)
    clarification: ClarificationKind = "none"
    unresolved: tuple[QueryPart, ...] = Field(default=(), max_length=6)
    update: DialogueUpdate | None = None

    @model_validator(mode="after")
    def validate_decision(self) -> "ResponseDecision":
        if self.outcome in {"no_reply", "discarded"} and (
            self.answers or self.clarification != "none" or self.unresolved
        ):
            raise ValueError("silent decisions cannot carry an answer or clarification")
        if self.outcome == "discarded" and self.update is not None:
            raise ValueError("discarded turns cannot mutate memory")
        if len({answer.part_id for answer in self.answers}) != len(self.answers):
            raise ValueError("answer part IDs must be unique")
        if {a.part_id for a in self.answers} & {p.part_id for p in self.unresolved}:
            raise ValueError("a request part cannot be answered and unresolved")
        return self


class UtteranceClause(ContractModel):
    """One bounded piece of speech, optionally sourced from one grounded fact."""

    clause_id: Identifier
    kind: UtteranceClauseKind
    source_part_id: Identifier | None = None

    @model_validator(mode="after")
    def validate_source(self) -> "UtteranceClause":
        if (self.kind == "fact") != (self.source_part_id is not None):
            raise ValueError("only fact clauses require a grounded source part")
        return self


class UtterancePlan(ContractModel):
    """A local, non-generative speech plan compiled from a response decision."""

    schema_version: Literal["utterance-plan.v1"] = "utterance-plan.v1"
    response_id: Identifier
    language: RadioLanguage
    tone: UtteranceTone
    clauses: tuple[UtteranceClause, ...] = Field(min_length=1, max_length=10)
    max_words: int = Field(default=48, ge=8, le=80)

    @model_validator(mode="after")
    def validate_clauses(self) -> "UtterancePlan":
        if len({clause.clause_id for clause in self.clauses}) != len(self.clauses):
            raise ValueError("utterance clause IDs must be unique")
        fact_parts = [clause.source_part_id for clause in self.clauses if clause.kind == "fact"]
        if len(set(fact_parts)) != len(fact_parts):
            raise ValueError("a grounded fact cannot be spoken twice")
        return self


class DeliveryEvent(ContractModel):
    schema_version: Literal["dialogue-delivery.v1"] = "dialogue-delivery.v1"
    turn_id: Identifier
    response_id: Identifier
    session_id: Identifier
    generation: int = Field(ge=0)
    output_mode: OutputMode
    status: DeliveryStatus
    occurred_at: UtcDatetime


class SemanticJudgeError(Exception):
    """Adapter failure; session responses never expose exception text or transcripts."""
