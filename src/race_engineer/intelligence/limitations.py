"""Application-owned statements for unavailable facts, never a telemetry conclusion.

These are safety fallbacks for structured unknown states, not driver-phrase
matching or templates for supported race answers. Free-form prose still needs
semantic evaluation; a valid placeholder is not proof of every surrounding claim.
"""

from typing import Literal

from race_engineer.core.intelligence import ContextTemporalScope, EngineerLanguage

UNSUPPORTED_DATA_POLICY = "grounded-limitations.v1"


def needs_clarification(unknown_ids: tuple[str, ...]) -> bool:
    return bool(unknown_ids) and set(unknown_ids) == {"requested_reference_ambiguous"}


def limitation_statement(
    language: EngineerLanguage,
    temporal_scope: ContextTemporalScope | None,
    unknown_ids: tuple[str, ...],
    *,
    wording: Literal["standard", "radio", "explain"] = "standard",
) -> str:
    if needs_clarification(unknown_ids):
        return (
            "Ne kastettiğini netleştirebilir misin?"
            if language == "tr"
            else "Could you clarify what you mean?"
        )
    if wording != "standard":
        if temporal_scope == "historical":
            return (
                "O zamana ait veriler elimde yok."
                if language == "tr"
                else "I don't have that earlier data available."
            )
        if temporal_scope == "future_counterfactual":
            return (
                "Elimdeki verilerle bu tahmini yapamıyorum."
                if language == "tr"
                else "I can't make that prediction from the data I have."
            )
        if wording == "explain":
            return (
                "Bu bilgi, elimdeki verilerde bulunmuyor."
                if language == "tr"
                else "That information isn't available in the data I have here."
            )
        return (
            "Bu veri şu an elimde yok."
            if language == "tr"
            else "I don't have that data available here."
        )
    if temporal_scope == "historical":
        return (
            "Bunun için gereken geçmiş veriler elimde yok."
            if language == "tr"
            else "I don't have the historical data needed for that."
        )
    if temporal_scope == "future_counterfactual":
        return (
            "Mevcut verilerle bu tahmini yapamıyorum."
            if language == "tr"
            else "I can't support that prediction with the available data."
        )
    return (
        "Mevcut verilerle bu kısmı değerlendiremiyorum."
        if language == "tr"
        else "I can't assess that part from the available data."
    )
