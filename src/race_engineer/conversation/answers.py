"""Read-only fact retrieval and bilingual wording; models cannot supply values."""

from race_engineer.core.contracts import RaceContext
from race_engineer.core.conversation import RaceAnswer, RaceQuery, RadioLanguage

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
        RaceQuery.LAP: "Your current lap isn't available.",
        RaceQuery.GAP_AHEAD: "I don't have a reliable time gap to the car ahead.",
        RaceQuery.GAP_BEHIND: "I don't have a reliable time gap to the car behind.",
        RaceQuery.FUEL_REMAINING: "Your current fuel level isn't available.",
        RaceQuery.FUEL_CONSUMPTION: "I don't have a reliable fuel-consumption estimate yet.",
    },
    "tr": {
        RaceQuery.POSITION: "Güncel sıralama bilgisi yok.",
        RaceQuery.LAP: "Güncel tur bilgisi yok.",
        RaceQuery.GAP_AHEAD: "Öndeki araçla güvenilir bir zaman farkı bilgim yok.",
        RaceQuery.GAP_BEHIND: "Arkadaki araçla güvenilir bir zaman farkı bilgim yok.",
        RaceQuery.FUEL_REMAINING: "Güncel yakıt seviyesi bilgisi yok.",
        RaceQuery.FUEL_CONSUMPTION: "Henüz güvenilir bir yakıt tüketimi tahminim yok.",
    },
}
_UNSUPPORTED: dict[RadioLanguage, dict[RaceQuery, str]] = {
    "en": {
        RaceQuery.FUEL_TO_FINISH: (
            "I can't estimate fuel to the finish yet; remaining race distance isn't supported."
        ),
        RaceQuery.GAP_TREND_AHEAD: "I can't tell whether you're gaining on the car ahead yet.",
        RaceQuery.GAP_TREND_BEHIND: "I can't tell whether the car behind is gaining yet.",
        RaceQuery.UNSUPPORTED: "I can't answer that part or make changes in this prototype yet.",
    },
    "tr": {
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


def retrieve(context: RaceContext, query: RaceQuery) -> RaceAnswer:
    player = context.frame.player
    value: int | float | None
    match query:
        case RaceQuery.POSITION:
            value, unit = player.position, "position"
        case RaceQuery.LAP:
            value, unit = player.lap_number, "lap"
        case RaceQuery.GAP_AHEAD:
            value, unit = context.gap_ahead_s, "s"
        case RaceQuery.GAP_BEHIND:
            value, unit = context.gap_behind_s, "s"
        case RaceQuery.FUEL_REMAINING:
            value, unit = player.fuel_l, "l"
        case RaceQuery.FUEL_CONSUMPTION:
            value, unit = context.fuel_trend_l_per_lap, "l/lap"
        case _:
            return RaceAnswer(query=query, status="unsupported")
    if value is None or value < 0:
        return RaceAnswer(query=query, status="missing")
    return RaceAnswer.model_validate(
        {"query": query, "status": "available", "value": value, "unit": unit}
    )


def render(answer: RaceAnswer, language: RadioLanguage) -> str:
    if answer.status == "unsupported":
        return _UNSUPPORTED[language][answer.query]
    if answer.status == "missing":
        return _MISSING[language][answer.query]
    assert answer.value is not None
    value = str(int(answer.value)) if answer.unit in {"position", "lap"} else f"{answer.value:.1f}"
    if language == "tr":
        value = value.replace(".", ",")
    return _AVAILABLE[language][answer.query].format(value=value)
