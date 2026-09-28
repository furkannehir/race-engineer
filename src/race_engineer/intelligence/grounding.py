"""Deterministic evidence substitution and final response validation."""

from collections.abc import Mapping

from race_engineer.core.intelligence import (
    ContextPacket,
    EngineerBrief,
    EvidenceItem,
    EvidenceReference,
    GeneratedResponse,
    GroundedResponse,
)


class GroundingError(Exception):
    """A generated response cannot be proven against its evidence packet."""


def _format_number(value: int | float) -> str:
    if isinstance(value, bool):
        raise GroundingError("boolean_is_not_numeric_evidence")
    if isinstance(value, int):
        return str(value)
    return f"{value:.3f}".rstrip("0").rstrip(".")


def _reference_value(reference: EvidenceReference, evidence: EvidenceItem) -> str:
    if evidence.kind == "unknown":
        raise GroundingError("unknown_evidence_cannot_be_spoken")
    if reference.field == "claim":
        if evidence.claim is None:
            raise GroundingError("referenced_claim_is_missing")
        return evidence.claim
    value = evidence.value
    if isinstance(value, bool) or value is None or isinstance(value, (dict, list)):
        raise GroundingError("referenced_value_is_not_scalar")
    return _format_number(value) if isinstance(value, (int, float)) else value


class StrictEvidenceGrounder:
    """Ground Qwen's template using only refreshed, Core-approved evidence."""

    def ground(
        self,
        response: GeneratedResponse,
        context: ContextPacket,
        brief: EngineerBrief,
    ) -> GroundedResponse:
        response_scope = (
            response.turn_id,
            response.session_id,
            response.generation,
        )
        context_scope = (context.turn_id, context.session_id, context.generation)
        brief_scope = (brief.turn_id, brief.session_id, brief.generation)
        if response_scope != context_scope or response_scope != brief_scope:
            raise GroundingError("response_scope_mismatch")
        if response.language != brief.language:
            raise GroundingError("response_language_mismatch")
        expected_action = (
            "silence"
            if brief.goal == "silence"
            else "clarify"
            if brief.goal == "clarify"
            else "speak"
        )
        if response.action != expected_action:
            raise GroundingError("response_action_mismatch")

        reference_ids = tuple(reference.evidence_id for reference in response.references)
        if len(set(reference_ids)) != len(reference_ids):
            raise GroundingError("evidence_cannot_be_referenced_twice")
        if set(reference_ids) != set(brief.evidence_ids):
            raise GroundingError("response_did_not_use_the_core_brief_evidence")

        if response.action == "silence":
            return GroundedResponse(
                response_id=response.response_id,
                turn_id=response.turn_id,
                session_id=response.session_id,
                generation=response.generation,
                source_sequence=context.source_sequence,
                language=response.language,
                action=response.action,
            )

        evidence_by_id: Mapping[str, EvidenceItem] = {
            item.evidence_id: item for item in context.evidence
        }
        text = response.speech_template
        assert text is not None
        for reference in response.references:
            evidence = evidence_by_id.get(reference.evidence_id)
            if evidence is None:
                raise GroundingError("referenced_evidence_is_missing")
            if evidence.valid_until is not None and context.assembled_at >= evidence.valid_until:
                raise GroundingError("referenced_evidence_is_stale")
            text = text.replace(
                "{{" + reference.placeholder + "}}",
                _reference_value(reference, evidence),
            )
        if "{{" in text or "}}" in text:
            raise GroundingError("unresolved_evidence_placeholder")
        if len(text.split()) > 60:
            raise GroundingError("grounded_response_exceeds_radio_limit")
        return GroundedResponse(
            response_id=response.response_id,
            turn_id=response.turn_id,
            session_id=response.session_id,
            generation=response.generation,
            source_sequence=context.source_sequence,
            language=response.language,
            action=response.action,
            text=text,
            evidence_ids=reference_ids,
        )
