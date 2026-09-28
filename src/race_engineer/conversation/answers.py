"""Read-only fact retrieval and bilingual wording; models cannot supply values."""

from race_engineer.conversation.facts import DEFAULT_FACT_CATALOG
from race_engineer.core.contracts import RaceContext
from race_engineer.core.conversation import FieldRelation, RaceAnswer, RaceQuery, RadioLanguage

_AVAILABLE: dict[RadioLanguage, dict[RaceQuery, str]] = {
    "en": {
        RaceQuery.POSITION: "You're P{value} overall.",
        RaceQuery.LAP: "You're on lap {value}.",
        RaceQuery.GAP_AHEAD: "The car ahead is {value} seconds up the road.",
        RaceQuery.GAP_BEHIND: "The car behind is {value} seconds back.",
        RaceQuery.FUEL_REMAINING: "You have {value} liters of fuel.",
        RaceQuery.FUEL_CONSUMPTION: "Observed fuel use averages {value} liters per lap.",
    },
    "tr": {
        RaceQuery.POSITION: "Genel sıralamada {value}. sıradasın.",
        RaceQuery.LAP: "{value}. turdasın.",
        RaceQuery.GAP_AHEAD: "Öndeki araç {value} saniye ileride.",
        RaceQuery.GAP_BEHIND: "Arkadaki araç {value} saniye geride.",
        RaceQuery.FUEL_REMAINING: "{value} litre yakıtın var.",
        RaceQuery.FUEL_CONSUMPTION: "Gözlenen ortalama tüketim tur başına {value} litre.",
    },
}
_MISSING: dict[RadioLanguage, dict[RaceQuery, str]] = {
    "en": {
        RaceQuery.POSITION: "Your current position isn't available.",
        RaceQuery.FIELD_STATUS: "I can't confirm the complete running order right now.",
        RaceQuery.LAP: "Your current lap isn't available.",
        RaceQuery.GAP_AHEAD: "I don't have a reliable time gap to the car ahead.",
        RaceQuery.GAP_BEHIND: "I don't have a reliable time gap to the car behind.",
        RaceQuery.FUEL_REMAINING: "Your current fuel level isn't available.",
        RaceQuery.FUEL_CONSUMPTION: "I don't have a reliable fuel-consumption estimate yet.",
    },
    "tr": {
        RaceQuery.POSITION: "Güncel sıralama bilgisi yok.",
        RaceQuery.FIELD_STATUS: "Şu anda tam sıralamayı doğrulayamıyorum.",
        RaceQuery.LAP: "Güncel tur bilgisi yok.",
        RaceQuery.GAP_AHEAD: "Öndeki araçla güvenilir bir zaman farkı bilgim yok.",
        RaceQuery.GAP_BEHIND: "Arkadaki araçla güvenilir bir zaman farkı bilgim yok.",
        RaceQuery.FUEL_REMAINING: "Güncel yakıt seviyesi bilgisi yok.",
        RaceQuery.FUEL_CONSUMPTION: "Henüz güvenilir bir yakıt tüketimi tahminim yok.",
    },
}


_UNSUPPORTED: dict[RadioLanguage, dict[RaceQuery, str]] = {
    "en": {
        RaceQuery.FIELD_STATUS: "I can't compare your position with the field yet.",
        RaceQuery.FUEL_TO_FINISH: (
            "I can't estimate fuel to the finish yet; remaining race distance isn't supported."
        ),
        RaceQuery.GAP_TREND_AHEAD: "I can't tell whether you're gaining on the car ahead yet.",
        RaceQuery.GAP_TREND_BEHIND: "I can't tell whether the car behind is gaining yet.",
        RaceQuery.UNSUPPORTED: "I can't answer that part or make changes in this prototype yet.",
    },
    "tr": {
        RaceQuery.FIELD_STATUS: "Sıranı grubun tamamıyla henüz karşılaştıramıyorum.",
        RaceQuery.FUEL_TO_FINISH: (
            "Yakıtın finişe yetip yetmeyeceğini henüz hesaplayamıyorum; "
            "kalan yarış mesafesi desteklenmiyor."
        ),
        RaceQuery.GAP_TREND_AHEAD: "Öndeki araca yaklaşıp yaklaşmadığını henüz söyleyemiyorum.",
        RaceQuery.GAP_TREND_BEHIND: "Arkadaki aracın yaklaşıp yaklaşmadığını henüz söyleyemiyorum.",
        RaceQuery.UNSUPPORTED: (
            "Bu prototipte o kısmı henüz yanıtlayamıyorum veya değiştiremiyorum."
        ),
    },
}

MESSAGES: dict[RadioLanguage, dict[str, str]] = {
    "en": {
        "topic": "Do you mean position, gaps, or fuel?",
        "opponent": "Do you mean the car ahead or behind?",
        "stale_snapshot": "Telemetry isn't current. I can't give a reliable answer yet.",
        "session_changed": "The session changed while I was checking. Please ask again.",
        "model_error": "The local conversation model couldn't process that. Please try again.",
    },
    "tr": {
        "topic": "Sıralamayı mı, farkları mı, yakıtı mı soruyorsun?",
        "opponent": "Öndeki aracı mı, arkadakini mi kastediyorsun?",
        "stale_snapshot": "Telemetri güncel değil. Henüz güvenilir bir yanıt veremiyorum.",
        "session_changed": "Kontrol ederken oturum değişti. Lütfen tekrar sor.",
        "model_error": "Yerel konuşma modeli soruyu işleyemedi. Lütfen tekrar dene.",
    },
}


def retrieve(
    context: RaceContext,
    query: RaceQuery,
    *,
    field_relation: FieldRelation | None = None,
) -> RaceAnswer:
    return DEFAULT_FACT_CATALOG.resolve_query(
        context,
        query,
        field_relation=field_relation,
    )


def render(answer: RaceAnswer, language: RadioLanguage) -> str:
    if answer.status == "unsupported":
        return _UNSUPPORTED[language][answer.query]
    if answer.status == "missing":
        return _MISSING[language][answer.query]
    assert answer.value is not None
    if answer.query is RaceQuery.FIELD_STATUS:
        assert (
            isinstance(answer.value, int)
            and answer.total is not None
            and answer.field_relation is not None
        )
        position, total = answer.value, answer.total
        relation = answer.field_relation
        if language == "tr":
            if total == 1:
                if relation == "cars_ahead":
                    return "Önünde araç yok; sınıflandırılmış tek araç sensin."
                if relation == "cars_behind":
                    return "Arkanda araç yok; sınıflandırılmış tek araç sensin."
                if relation == "position_of_total":
                    return "Sınıflandırılmış 1 araç içinde 1. sıradasın."
                return "Evet. Sınıflandırılmış tek araç sensin; 1 araç içinde 1. sıradasın."
            ahead, behind = position - 1, total - position
            if relation == "first":
                if position == 1:
                    return f"Evet. Lideriz; {total} araç içinde 1. sıradayız."
                return f"Hayır. {total} araç içinde {position}. sıradasın; önünde {ahead} araç var."
            if relation == "last":
                if position == total:
                    return f"Evet. Şu anda sonuncusun; {total} araç içinde {position}. sıradasın."
                return (
                    f"Hayır. {total} araç içinde {position}. sıradasın; arkanda {behind} araç var."
                )
            if relation == "cars_ahead":
                if ahead == 0:
                    return f"Önünde araç yok. Lideriz; {total} araç içinde 1. sıradayız."
                return f"Önünde {ahead} araç var; {total} araç içinde {position}. sıradasın."
            if relation == "cars_behind":
                if behind == 0:
                    return (
                        f"Arkanda araç yok. Şu anda sonuncusun; {total} araç içinde "
                        f"{position}. sıradasın."
                    )
                return f"Arkanda {behind} araç var; {total} araç içinde {position}. sıradasın."
            return f"{total} araç içinde {position}. sıradasın."
        if total == 1:
            if relation == "cars_ahead":
                return "No cars ahead; you're the only classified car."
            if relation == "cars_behind":
                return "No cars behind; you're the only classified car."
            if relation == "position_of_total":
                return "You're P1 of 1, the only classified car."
            return "Yes. You're the only classified car, P1 of 1."
        ahead, behind = position - 1, total - position
        if relation == "first":
            if position == 1:
                return f"Yes. You're leading, P1 of {total}."
            suffix = "car is" if ahead == 1 else "cars are"
            return f"No. You're P{position} of {total}; {ahead} {suffix} ahead."
        if relation == "last":
            if position == total:
                return f"Yes. You're currently last, P{position} of {total}."
            suffix = "car is" if behind == 1 else "cars are"
            return f"No. You're P{position} of {total}; {behind} {suffix} behind."
        if relation == "cars_ahead":
            if ahead == 0:
                return f"No cars ahead. You're leading, P1 of {total}."
            noun = "car" if ahead == 1 else "cars"
            return f"{ahead} {noun} ahead. You're P{position} of {total}."
        if relation == "cars_behind":
            if behind == 0:
                return f"No cars behind. You're currently last, P{position} of {total}."
            noun = "car" if behind == 1 else "cars"
            return f"{behind} {noun} behind. You're P{position} of {total}."
        return f"You're P{position} of {total}."
    value = str(int(answer.value)) if answer.unit in {"position", "lap"} else f"{answer.value:.1f}"
    if language == "tr":
        value = value.replace(".", ",")
    return _AVAILABLE[language][answer.query].format(value=value)
