"""Natural, bounded radio composition from deterministic dialogue decisions."""

from collections import Counter
from hashlib import blake2s

from race_engineer.conversation.answers import MESSAGES, render
from race_engineer.core.conversation import RaceAnswer, RaceQuery, RadioLanguage
from race_engineer.core.dialogue import (
    GroundedAnswer,
    ResponseDecision,
    UtteranceClause,
    UtterancePlan,
    UtteranceTone,
)

_ACTS: dict[RadioLanguage, dict[str, str]] = {
    "en": {
        "acknowledge": "Copy.",
        "acknowledge_help": "Copy. Focus forward.",
    },
    "tr": {
        "acknowledge": "Anlaşıldı.",
        "acknowledge_help": "Anlaşıldı. Önüne odaklan.",
    },
}

_NATURAL_ACKNOWLEDGMENTS: dict[RadioLanguage, dict[str, tuple[str, ...]]] = {
    "en": {
        "concise": ("Copy.",),
        "neutral": ("Copy.", "Yeah, copy."),
        "supportive": (
            "Yeah, copy. Keep your head down.",
            "Understood. Stay focused.",
        ),
    },
    "tr": {
        "concise": ("Anlaşıldı.",),
        "neutral": ("Anlaşıldı.", "Tamam, anlaşıldı."),
        "supportive": (
            "Tamam, anlaşıldı. Önüne odaklan.",
            "Anlaşıldı. Yarışına odaklan.",
        ),
    },
}

_FAILURES: dict[RadioLanguage, dict[str, str]] = {
    "en": {
        "clarification_failed": "I couldn't resolve that reference. Ask me again.",
        "clarification_expired": "That question expired. Ask me again.",
        "deadline": "That reply expired. Ask me again.",
        "busy": "I'm still handling the previous call. Ask me again.",
        "model_error": "The local conversation model couldn't process that. Please try again.",
        "topic_unclear": "I didn't understand which race information you need.",
    },
    "tr": {
        "clarification_failed": "O referansı çözemedim. Tekrar sor.",
        "clarification_expired": "O soru zaman aşımına uğradı. Tekrar sor.",
        "deadline": "Yanıtın süresi doldu. Tekrar sor.",
        "busy": "Önceki çağrıyı hâlâ işliyorum. Tekrar sor.",
        "model_error": "Yerel konuşma modeli soruyu işleyemedi. Lütfen tekrar dene.",
        "topic_unclear": "Hangi yarış bilgisini istediğini anlayamadım.",
    },
}


class CompositionError(Exception):
    """The natural plan failed closed and should use bounded legacy wording."""


def _race_answer(answer: GroundedAnswer) -> RaceAnswer:
    return RaceAnswer(
        query=answer.query,
        status=answer.status,
        value=answer.value,
        unit=answer.unit,
        total=answer.total,
        field_relation=answer.field_relation,
    )


def _fact(answer: GroundedAnswer, language: RadioLanguage) -> str:
    return render(_race_answer(answer), language)


def compose_bounded(
    decision: ResponseDecision,
    *,
    repeated_acknowledgment: bool = False,
) -> str | None:
    """The pre-R5 composer, retained as a deterministic safety fallback."""
    if decision.outcome in {"no_reply", "discarded"}:
        return None
    language = decision.language
    parts: list[str] = []
    if "acknowledge" in decision.acts:
        helpful = not decision.answers and not repeated_acknowledgment
        parts.append(_ACTS[language]["acknowledge_help" if helpful else "acknowledge"])
    parts.extend(_fact(answer, language) for answer in decision.answers)
    if decision.clarification != "none":
        parts.append(MESSAGES[language][decision.clarification])
    if not parts:
        parts.append(
            _FAILURES[language].get(
                decision.reason,
                MESSAGES[language].get(decision.reason, _FAILURES[language]["topic_unclear"]),
            )
        )
    return " ".join(parts)


def plan_utterance(
    decision: ResponseDecision,
    *,
    repeated_acknowledgment: bool = False,
) -> UtterancePlan | None:
    """Compile approved dialogue acts and grounded answers into typed speech clauses."""
    if decision.outcome in {"no_reply", "discarded"}:
        return None
    clauses: list[UtteranceClause] = []
    if "acknowledge" in decision.acts:
        clauses.append(UtteranceClause(clause_id="act:acknowledge", kind="acknowledgment"))
    clauses.extend(
        UtteranceClause(
            clause_id=f"fact:{answer.part_id}",
            kind="fact",
            source_part_id=answer.part_id,
        )
        for answer in decision.answers
    )
    if decision.clarification != "none":
        clauses.append(
            UtteranceClause(
                clause_id=f"clarification:{decision.clarification}",
                kind="clarification",
            )
        )
    if not clauses:
        clauses.append(UtteranceClause(clause_id="failure:reason", kind="failure"))
    tone: UtteranceTone = (
        "concise"
        if repeated_acknowledgment
        else "supportive"
        if "acknowledge" in decision.acts and not decision.answers
        else "neutral"
    )
    return UtterancePlan(
        response_id=decision.response_id,
        language=decision.language,
        tone=tone,
        clauses=tuple(clauses),
    )


def _select(plan: UtterancePlan, clause: UtteranceClause, options: tuple[str, ...]) -> str:
    if not options:
        raise CompositionError("empty_variant_set")
    seed = f"{plan.response_id}:{clause.clause_id}".encode()
    index = int.from_bytes(blake2s(seed, digest_size=2).digest(), "big") % len(options)
    return options[index]


def _number(value: int | float, unit: str | None, language: RadioLanguage) -> str:
    rendered = str(int(value)) if unit in {"position", "lap"} else f"{value:.1f}"
    return rendered.replace(".", ",") if language == "tr" else rendered


def _english_field(answer: GroundedAnswer) -> str:
    assert isinstance(answer.value, int) and answer.total is not None
    assert answer.field_relation is not None
    position, total = answer.value, answer.total
    ahead, behind = position - 1, total - position
    relation = answer.field_relation
    if total == 1:
        if relation == "cars_ahead":
            return "No cars ahead. We're the only classified car."
        if relation == "cars_behind":
            return "No cars behind. We're the only classified car."
        if relation == "position_of_total":
            return "P1 of 1, the only classified car."
        return "Affirm, P1. We're the only classified car."
    if relation == "first":
        if position == 1:
            return "Affirm, P1. We're leading."
        noun = "car" if ahead == 1 else "cars"
        return f"Negative, P{position}. {ahead} {noun} ahead."
    if relation == "last":
        if position == total:
            return f"Affirm, P{position}. We're last."
        noun = "car" if behind == 1 else "cars"
        return f"Negative, P{position}. {behind} {noun} behind."
    if relation == "cars_ahead":
        if ahead == 0:
            return "No cars ahead. P1, we're leading."
        noun = "car" if ahead == 1 else "cars"
        return f"{ahead} {noun} ahead. We're P{position}."
    if relation == "cars_behind":
        if behind == 0:
            return f"No cars behind. P{position}, we're last."
        noun = "car" if behind == 1 else "cars"
        return f"{behind} {noun} behind. We're P{position}."
    return f"P{position} of {total}."


def _turkish_field(answer: GroundedAnswer) -> str:
    assert isinstance(answer.value, int) and answer.total is not None
    assert answer.field_relation is not None
    position, total = answer.value, answer.total
    ahead, behind = position - 1, total - position
    relation = answer.field_relation
    if total == 1:
        if relation == "cars_ahead":
            return "Önümüzde araç yok. Sınıflandırılan tek araç biziz."
        if relation == "cars_behind":
            return "Arkamızda araç yok. Sınıflandırılan tek araç biziz."
        if relation == "position_of_total":
            return "Tek araç içinde P1."
        return "Evet, P1. Sınıflandırılan tek araç biziz."
    if relation == "first":
        if position == 1:
            return "Evet, P1. Lideriz."
        return f"Hayır, P{position}. Önünde {ahead} araç var."
    if relation == "last":
        if position == total:
            return f"Evet, P{position}. Son sıradayız."
        return f"Hayır, P{position}. Arkanda {behind} araç var."
    if relation == "cars_ahead":
        if ahead == 0:
            return "Önümüzde araç yok. P1, lideriz."
        return f"Önünde {ahead} araç var. P{position}'deyiz."
    if relation == "cars_behind":
        if behind == 0:
            return f"Arkanda araç yok. P{position}, son sıradayız."
        return f"Arkanda {behind} araç var. P{position}'deyiz."
    return f"Toplam {total} araç içinde P{position}."


def _natural_fact(
    answer: GroundedAnswer,
    language: RadioLanguage,
    plan: UtterancePlan,
    clause: UtteranceClause,
) -> str:
    if answer.status != "available":
        return _fact(answer, language)
    assert answer.value is not None
    if answer.query is RaceQuery.FIELD_STATUS:
        return _turkish_field(answer) if language == "tr" else _english_field(answer)
    value = _number(answer.value, answer.unit, language)
    variants: dict[RadioLanguage, dict[RaceQuery, tuple[str, ...]]] = {
        "en": {
            RaceQuery.POSITION: (f"P{value} right now.", f"We're running P{value}."),
            RaceQuery.LAP: (f"Lap {value}.", f"We're on lap {value}."),
            RaceQuery.GAP_AHEAD: (
                f"Car ahead, {value} seconds.",
                f"Gap ahead, {value} seconds.",
            ),
            RaceQuery.GAP_BEHIND: (
                f"Car behind, {value} seconds.",
                f"Gap behind, {value} seconds.",
            ),
            RaceQuery.FUEL_REMAINING: (
                f"{value} liters remaining.",
                f"Fuel remaining, {value} liters.",
            ),
            RaceQuery.FUEL_CONSUMPTION: (
                f"Fuel use is {value} liters per lap.",
                f"We're averaging {value} liters per lap.",
            ),
        },
        "tr": {
            RaceQuery.POSITION: (f"Şu an P{value}.", f"P{value}'deyiz."),
            RaceQuery.LAP: (f"{value}. turdayız.", f"Şu an {value}. tur."),
            RaceQuery.GAP_AHEAD: (
                f"Öndeki araçla fark {value} saniye.",
                f"Öndekiyle fark {value} saniye.",
            ),
            RaceQuery.GAP_BEHIND: (
                f"Arkadaki araçla fark {value} saniye.",
                f"Arkadakiyle fark {value} saniye.",
            ),
            RaceQuery.FUEL_REMAINING: (
                f"{value} litre yakıt kaldı.",
                f"Kalan yakıt {value} litre.",
            ),
            RaceQuery.FUEL_CONSUMPTION: (
                f"Tur başına {value} litre yakıyoruz.",
                f"Ortalama tüketim tur başına {value} litre.",
            ),
        },
    }
    options = variants[language].get(answer.query)
    if options is None:
        raise CompositionError("unregistered_available_fact")
    return _select(plan, clause, options)


def render_utterance(plan: UtterancePlan, decision: ResponseDecision) -> str:
    """Render only registered clauses; no model-authored wording enters this boundary."""
    answers = {answer.part_id: answer for answer in decision.answers}
    parts: list[str] = []
    for clause in plan.clauses:
        if clause.kind == "acknowledgment":
            parts.append(
                _select(
                    plan,
                    clause,
                    _NATURAL_ACKNOWLEDGMENTS[plan.language][plan.tone],
                )
            )
        elif clause.kind == "fact":
            assert clause.source_part_id is not None
            answer = answers.get(clause.source_part_id)
            if answer is None:
                raise CompositionError("unknown_fact_source")
            parts.append(_natural_fact(answer, plan.language, plan, clause))
        elif clause.kind == "clarification":
            if decision.clarification == "none":
                raise CompositionError("unexpected_clarification")
            parts.append(MESSAGES[plan.language][decision.clarification])
        elif clause.kind == "failure":
            parts.append(
                _FAILURES[plan.language].get(
                    decision.reason,
                    MESSAGES[plan.language].get(
                        decision.reason,
                        _FAILURES[plan.language]["topic_unclear"],
                    ),
                )
            )
    return " ".join(parts)


def validate_utterance(
    plan: UtterancePlan,
    decision: ResponseDecision,
    text: str,
) -> None:
    """Prove clause provenance and radio bounds before text reaches TTS."""
    if plan.response_id != decision.response_id or plan.language != decision.language:
        raise CompositionError("decision_scope_mismatch")
    kinds = Counter(clause.kind for clause in plan.clauses)
    fact_parts = tuple(clause.source_part_id for clause in plan.clauses if clause.kind == "fact")
    expected_parts = tuple(answer.part_id for answer in decision.answers)
    if fact_parts != expected_parts:
        raise CompositionError("grounded_fact_coverage_mismatch")
    if kinds["acknowledgment"] != int("acknowledge" in decision.acts):
        raise CompositionError("acknowledgment_mismatch")
    if kinds["clarification"] != int(decision.clarification != "none"):
        raise CompositionError("clarification_mismatch")
    expected_failure = (
        not decision.answers
        and "acknowledge" not in decision.acts
        and decision.clarification == "none"
    )
    if kinds["failure"] != int(expected_failure):
        raise CompositionError("failure_clause_mismatch")
    if not text.strip() or "\n" in text or "\r" in text:
        raise CompositionError("invalid_spoken_text")
    if len(text.split()) > plan.max_words:
        raise CompositionError("spoken_text_too_long")


def compose(decision: ResponseDecision, *, repeated_acknowledgment: bool = False) -> str | None:
    """Produce natural grounded speech, falling back on any plan/render invariant failure."""
    plan = plan_utterance(decision, repeated_acknowledgment=repeated_acknowledgment)
    if plan is None:
        return None
    try:
        text = render_utterance(plan, decision)
        validate_utterance(plan, decision, text)
        return text
    except (AssertionError, CompositionError, KeyError, ValueError):
        return compose_bounded(
            decision,
            repeated_acknowledgment=repeated_acknowledgment,
        )
