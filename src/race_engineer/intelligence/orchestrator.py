"""One-turn orchestration across Context Engineer, Core Engineer, Qwen, and grounding."""

from dataclasses import dataclass

from race_engineer.core.intelligence import (
    ContextPacket,
    DriverTurn,
    EngineerBrief,
    GeneratedResponse,
    GroundedResponse,
)
from race_engineer.core.interfaces import (
    ContextEngineer,
    CoreEngineer,
    EngineerResponseGenerator,
    EvidenceGrounder,
)
from race_engineer.radio_diagnostics import radio_event, radio_evidence, radio_stage


class IntelligenceBoundaryError(Exception):
    """A component returned a valid shape with invalid cross-component scope."""


@dataclass(frozen=True)
class PreparedEngineerResponse:
    """A validated response awaiting final evidence refresh and grounding."""

    turn: DriverTurn
    context: ContextPacket
    brief: EngineerBrief
    response: GeneratedResponse


def _validate_context(turn: DriverTurn, context: ContextPacket) -> None:
    if (
        context.turn_id != turn.turn_id
        or context.session_id != turn.session_id
        or context.generation != turn.generation
    ):
        raise IntelligenceBoundaryError("context_scope_mismatch")


def _validate_brief(turn: DriverTurn, context: ContextPacket, brief: EngineerBrief) -> None:
    if (
        brief.turn_id != turn.turn_id
        or brief.session_id != turn.session_id
        or brief.generation != turn.generation
        or brief.source_sequence != context.source_sequence
    ):
        raise IntelligenceBoundaryError("brief_scope_mismatch")
    available = {item.evidence_id for item in context.evidence}
    if not set(brief.evidence_ids).issubset(available):
        raise IntelligenceBoundaryError("brief_referenced_unknown_evidence")
    if not set(brief.unknown_ids).issubset(context.unknowns):
        raise IntelligenceBoundaryError("brief_referenced_unknown_limitation")


def _validate_generated(
    turn: DriverTurn,
    brief: EngineerBrief,
    response: GeneratedResponse,
) -> None:
    if (
        response.turn_id != turn.turn_id
        or response.session_id != turn.session_id
        or response.generation != turn.generation
        or response.language != brief.language
    ):
        raise IntelligenceBoundaryError("generated_response_scope_mismatch")
    referenced = {reference.evidence_id for reference in response.references}
    if not referenced.issubset(set(brief.evidence_ids)):
        raise IntelligenceBoundaryError("generator_referenced_unapproved_evidence")
    unknowns = {
        item for reference in response.limitation_references for item in reference.unknown_ids
    }
    if unknowns != set(brief.unknown_ids):
        raise IntelligenceBoundaryError("generator_referenced_unapproved_limitation")


class EngineerOrchestrator:
    """Coordinate flexible intelligence while enforcing evidence and session boundaries."""

    def __init__(
        self,
        context_engineer: ContextEngineer,
        core_engineer: CoreEngineer,
        generator: EngineerResponseGenerator,
        grounder: EvidenceGrounder,
    ) -> None:
        self._context = context_engineer
        self._core = core_engineer
        self._generator = generator
        self._grounder = grounder

    async def prepare(self, turn: DriverTurn) -> PreparedEngineerResponse:
        with radio_stage("context"):
            context = await self._context.analyze(turn)
            _validate_context(turn, context)
            radio_evidence("radio_context_ready", context)
        with radio_stage("core"):
            brief = await self._core.decide(turn, context)
            _validate_brief(turn, context, brief)
            radio_event(
                "radio_core_decision",
                goal=brief.goal,
                language=brief.language,
                evidence_ids=brief.evidence_ids,
            )
        with radio_stage("response"):
            response = await self._generator.generate(turn, context, brief)
            _validate_generated(turn, brief, response)
        return PreparedEngineerResponse(turn, context, brief, response)

    async def ground(self, prepared: PreparedEngineerResponse) -> GroundedResponse:
        evidence_ids = tuple(reference.evidence_id for reference in prepared.response.references)
        if prepared.response.limitation_references:
            # Recheck unavailable queries/capabilities as well as spoken known facts.
            # A previously missing input may have arrived while Core was generating.
            evidence_ids = tuple(
                dict.fromkeys(
                    (
                        *evidence_ids,
                        *(
                            item.evidence_id
                            for item in prepared.context.evidence
                            if item.kind == "unknown"
                        ),
                    )
                )
            )
        with radio_stage("grounding"):
            refreshed = await self._context.refresh(prepared.context, evidence_ids)
            _validate_context(prepared.turn, refreshed)
            if refreshed.source_sequence < prepared.context.source_sequence:
                raise IntelligenceBoundaryError("refreshed_context_moved_backwards")
            if refreshed.temporal_scope != prepared.context.temporal_scope:
                raise IntelligenceBoundaryError("refreshed_context_temporal_scope_changed")
            radio_evidence("radio_evidence_refreshed", refreshed)
            return self._grounder.ground(
                prepared.response,
                refreshed,
                prepared.brief,
            )

    async def respond(self, turn: DriverTurn) -> GroundedResponse:
        return await self.ground(await self.prepare(turn))
