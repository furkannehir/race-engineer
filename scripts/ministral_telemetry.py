"""Read-only bridge from Ministral's trial tools to existing telemetry/core code.

No simulator commands, driver-phrase routing, model-authored arithmetic or new
strategy calculations. Reading tokens carry application-formatted units and are
resolved again against the refreshed evidence before delivery.
"""

import json
import re
from datetime import timedelta
from typing import Literal

from pydantic import Field, ValidationError

from race_engineer.core.contracts import ContractModel
from race_engineer.core.intelligence import (
    CapabilityRequest,
    EvidenceItem,
    EvidenceQuery,
    EvidenceReference,
    SignalSelector,
)
from race_engineer.intelligence.capabilities import (
    _POSITION_WINDOW_S,
    DeterministicRaceCapabilities,
)
from race_engineer.intelligence.local_model import LocalIntelligenceError
from race_engineer.intelligence.native_model import function_tool
from race_engineer.language.lap_times import format_lap_time

LAP_TIME_METRICS = {"last_lap_time_s", "best_lap_time_s"}

CAPABILITY_TOOLS = {
    "get_race_classification": "current_classification",
    "check_if_driver_is_first": "current_classification",
    "check_if_driver_is_last": "current_classification",
    "get_fuel_range": "fuel_range",
    "get_recent_position_change": "position_change",
    "get_relative_pace_ahead": "relative_pace_ahead",
    "get_relative_pace_behind": "relative_pace_behind",
    "get_catch_time_ahead": "catch_time_ahead",
    "get_catch_time_behind": "catch_time_behind",
    "get_last_lap_time": "last_lap_time",
    "get_best_lap_time": "best_lap_time",
    "compare_last_lap_to_best": "lap_time_comparison",
    "get_race_progress": "race_progress",
    "get_remaining_race_laps": "race_progress",
    "get_completed_lap_count": "race_progress",
    "get_total_race_laps": "race_progress",
    "get_remaining_race_time": "race_progress",
    "get_fuel_to_finish": "fuel_to_finish",
}

# Explicit normalized contract paths, not arbitrary SDK names/attributes.
SIGNALS = {
    "player.position": ("player", "position", "position"),
    "player.lap_number": ("player", "lap_number", None),
    "player.laps_completed": ("player", "laps_completed", "laps"),
    "player.last_lap_time_s": ("player", "last_lap_time_s", "s"),
    "player.best_lap_time_s": ("player", "best_lap_time_s", "s"),
    "player.fuel_l": ("player", "fuel_l", "l"),
    "player.in_pit_lane": ("player", "in_pit_lane", None),
    "player.car_ahead_distance_m": ("player", "car_ahead_distance_m", "m"),
    "player.car_behind_distance_m": ("player", "car_behind_distance_m", "m"),
    "context.stint_lap": ("context", "stint_lap", None),
    "context.fuel_trend_l_per_lap": ("context", "fuel_trend_l_per_lap", "l/lap"),
    "context.gap_ahead_s": ("context", "gap_ahead_s", "s"),
    "context.gap_behind_s": ("context", "gap_behind_s", "s"),
    "context.battle_state": ("context", "battle_state", None),
    "frame.session_time_s": ("frame", "session_time_s", "s"),
    "frame.session_phase": ("frame", "session_phase", None),
    "frame.flags": ("frame", "flags", None),
    "frame.is_replay": ("frame", "is_replay", None),
    "frame.session_type": ("frame", "session_type", None),
    "frame.session_laps_total": ("frame", "session_laps_total", "laps"),
    "frame.session_laps_remaining": ("frame", "session_laps_remaining", "laps"),
    "frame.session_time_remaining_s": ("frame", "session_time_remaining_s", "s"),
    "frame.session_time_total_s": ("frame", "session_time_total_s", "s"),
    "opponent.position": ("opponent", "position", "position"),
    "opponent.lap_number": ("opponent", "lap_number", None),
    "opponent.gap_to_player_s": ("opponent", "gap_to_player_s", "s"),
}
_CAPABILITY_REQUIREMENT = {
    "player.in_pit_lane": "pit", "frame.flags": "flags", "frame.is_replay": "replay",
}
_UNIT_LABELS = {
    "l": "litres", "l/lap": "litres per lap",
    "m": "metres", "s": "seconds", "laps": "laps",
    "positions": "positions", "cars": "cars with reported positions",
}
_UNIT_LABELS_TR = {
    "l": "litre", "l/lap": "litre/tur", "m": "metre",
    "s": "saniye", "laps": "tur", "positions": "sıra",
    "cars": "sırası bildirilen araç",
}
_STATE_LABELS_TR = {
    "closing": "yaklaşıyor", "losing_ground": "uzaklaşıyoruz",
    "falling_back": "uzaklaşıyor", "stable": "sabit", "gained": "kazanıldı",
    "lost": "kaybedildi", "unchanged": "değişmedi", "green": "yeşil",
    "yellow": "sarı", "red": "kırmızı", "blue": "mavi", "white": "beyaz",
    "black": "siyah", "checkered": "damalı", "caution": "sarı bayrak",
    "formation": "formasyon", "no active flags": "aktif bayrak yok",
}
_TOKEN = re.compile(r"\{\{([a-z][a-z0-9_]*)\}\}")
_NUMBER = r"(?P<number>-?\d+(?:[.,]\d+)?)"
_UNIT_PATTERNS = {
    "l/lap": r"(?:litres?|liters?|l)\s*(?:per\s+|/)(?:lap|tur)(?:dir)?",
    "l": r"(?:litres?|liters?|l|litre(?:dir)?)",
    "s": r"(?:seconds?|secs?|s|saniye)",
    "m": r"(?:metres?|meters?|m|metre)",
    "laps": r"(?:laps?|tur)",
    "lap": r"(?:laps?|tur)",
    "positions": r"(?:positions?|places?)",
    "cars": r"cars?(?:\s+with\s+reported\s+positions)?",
}
_SMALL = ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
          "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
          "sixteen", "seventeen", "eighteen", "nineteen")
_TENS = ("", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety")
_TR_SMALL = ("sıfır", "bir", "iki", "üç", "dört", "beş", "altı", "yedi", "sekiz", "dokuz")
_TR_TENS = ("", "on", "yirmi", "otuz", "kırk", "elli", "altmış", "yetmiş", "seksen", "doksan")
_TR_ORDINALS = ("", "birinci", "ikinci", "üçüncü", "dördüncü", "beşinci", "altıncı",
                "yedinci", "sekizinci", "dokuzuncu", "onuncu")


def number_words(value, language="en"):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if value != int(value):
        if not 0 <= value < 100:
            return None
        whole = number_words(int(value), language)
        fraction = f"{value:.3f}".rstrip("0").split(".")[1]
        if not fraction:
            return number_words(round(value), language)
        digits = _TR_SMALL if language == "tr" else _SMALL
        return whole + (" virgül " if language == "tr" else " point ") + " ".join(
            digits[int(digit)] for digit in fraction
        )
    value = int(value)
    if not 0 <= value < 100:
        return None
    if language == "tr":
        return _TR_SMALL[value] if value < 10 else _TR_TENS[value // 10] + (
            " " + _TR_SMALL[value % 10] if value % 10 else ""
        )
    return _SMALL[value] if value < 20 else _TENS[value // 10] + (
        " " + _SMALL[value % 10] if value % 10 else ""
    )


class CurrentRead(ContractModel):
    signals: list[str] = Field(min_length=1, max_length=8)
    opponent: Literal["ahead", "behind", "leader"] = "ahead"
    opponent_position: int | None = Field(default=None, ge=1, strict=True)


class HistoryRead(ContractModel):
    source: Literal["player", "context", "opponent", "field"]
    signal: str
    operation: Literal["latest", "delta", "mean", "minimum", "maximum", "count"]
    window_s: float | None = Field(default=None, gt=0, le=120, strict=True)
    opponent: Literal["ahead", "behind", "leader"] = "ahead"
    opponent_position: int | None = Field(default=None, ge=1, strict=True)


def placeholder(item):
    return item.evidence_id.rsplit(":", 1)[-1]


def extended_evidence(evidence):
    return bool(evidence) and all(
        item.evidence_id.startswith("ministral-read:") for item in evidence
    )


def display_claim(item, language="en"):
    value = item.value
    if isinstance(value, bool):
        return ("Evet" if value else "Hayır") if language == "tr" else "Yes" if value else "No"
    if isinstance(value, (int, float)):
        if item.metric in LAP_TIME_METRICS and item.unit == "s":
            return format_lap_time(value, language)
        number = str(value) if isinstance(value, int) else f"{value:.3f}".rstrip("0").rstrip(".")
        if item.metric == "fuel_laps_remaining":
            return number + (" turluk yakıt" if language == "tr" else
                             " lap of fuel" if value == 1 else " laps of fuel")
        if item.unit == "position":
            return "P" + number
        labels = _UNIT_LABELS_TR if language == "tr" else _UNIT_LABELS
        if item.unit == "cars" and item.metric == "lap_number":
            return number + (" tur sayısı bildirilen araç" if language == "tr"
                             else " cars with reported lap counts")
        label = "lap" if language == "en" and item.unit == "laps" and abs(value) == 1 else (
            labels.get(item.unit, item.unit)
        )
        return number + (" " + label if label else "")
    return (_STATE_LABELS_TR.get(str(value), str(value).replace("_", " "))
            if language == "tr" else str(value).replace("_", " "))


def reading_message(call, evidence, unknowns, meanings=None):
    meanings = meanings or {}
    value = {
        "status": "partial" if evidence and unknowns else "available" if evidence
                  else "unavailable",
        "operation": call.name,
        "parameters": call.arguments,
        "facts": [{"reference": "{{" + placeholder(item) + "}}",
                   "meaning": meanings.get(item.evidence_id, item.metric),
                   "subject": item.subject, "value": item.value,
                   "unit": item.unit, "spoken_value": item.claim,
                   "temporal_scope": item.temporal_scope,
                   "source_fields": list(item.source_fields),
                   "source_sequence": item.source_sequence} for item in evidence],
        "unavailable": list(unknowns),
        "instruction": "Use reference tokens verbatim for reported facts. They already "
                       "include their units. Never calculate, convert or type numbers.",
    }
    if unknowns:
        value["limitation_reference"] = "{{unavailable}}"
    if call.name == "query_telemetry_history" and call.arguments.get("operation") == "delta":
        value["interpretation"] = (
            "The numeric result is signed: positive means increasing, negative decreasing. "
            "Delta is the total change over the requested window. Preserve the sign's meaning."
        )
    if call.name in {"get_relative_pace_ahead", "get_relative_pace_behind"}:
        value["interpretation"] = (
            "Qualitative recent gap movement only: use the state and Yes/No verdict. "
            "This is NOT a completed-lap pace comparison. No numerical gap-change rate or "
            "gain/loss per lap is available; never infer one from this state."
        )
    if call.name in {"get_catch_time_ahead", "get_catch_time_behind"}:
        value["interpretation"] = (
            "Estimated time UNTIL the catch if the observed closing rate continues. "
            "It is NOT a current gap, lead, or time already gained. Say estimated catch in "
            "the reported time, not that somebody is that many seconds ahead/behind."
        )
    if call.name == "get_fuel_to_finish":
        value["interpretation"] = (
            "An estimate for the full scheduled player distance INCLUDING the declared reserve. "
            "Use enough_fuel for Yes/No and mention the reserve. fuel_required_l already "
            "includes it; do not add it again. Shortfall/surplus are positive amounts. This is not "
            "a guaranteed finish or a pit recommendation. Actual distance can be shorter."
        )
    if call.name in {"get_race_progress", "get_remaining_race_laps", "get_completed_lap_count",
                     "get_total_race_laps", "get_remaining_race_time"}:
        value["interpretation"] = (
            "race_laps_remaining is remaining RACE distance, never fuel range. "
            "For a timed race only the session clock is known: do not invent a lap count "
            "or claim that zero clock time means the player has crossed the finish. "
            "Missing outputs were not supplied by the simulator."
        )
    return [{"role": "tool", "tool_call_id": call.call_id, "content": json.dumps(value)}]


class ReadOnlyRaceTools:
    def __init__(self, memory):
        self.memory = memory
        self.capabilities = DeterministicRaceCapabilities(memory, include_rank_checks=True)
        self.language = "en"
        # Preserve existing output semantics, especially player's current rank
        # versus last occupied FIELD rank. Equal metric names are not equal facts.
        self.meanings = {
            f"ministral-read:{descriptor.capability_id}:{output.output_id}": output.description
            for descriptor in self.capabilities.catalog() for output in descriptor.outputs
        }

    def _window_evidence(self, duration, scope):
        frame = self.memory.latest_context().frame
        return EvidenceItem(
            evidence_id="ministral-read:parameter:window_s", session_id=frame.session_id,
            source_sequence=frame.sequence, observed_at=frame.observed_at, kind="derived",
            temporal_scope=scope, subject="query_window", metric="window_s", value=duration,
            unit="s", source_fields=("request.window_s",),
            valid_until=frame.observed_at + timedelta(seconds=3),
        )

    def definitions(self):
        descriptors = {item.capability_id: item for item in self.capabilities.catalog()}
        tools = []
        for name, capability in CAPABILITY_TOOLS.items():
            description = descriptors[capability].description
            if name in {"get_remaining_race_laps", "get_completed_lap_count",
                        "get_total_race_laps", "get_remaining_race_time"}:
                description = {
                    "get_remaining_race_laps": "Read ONLY laps LEFT in the RACE, not current "
                    "lap, completed laps or fuel range. Timed races return unavailable.",
                    "get_completed_lap_count": "Read ONLY the player's completed lap COUNT, "
                    "not current lap number, lap duration or remaining laps.",
                    "get_total_race_laps": "Read ONLY the scheduled total RACE lap limit. "
                    "Unavailable for timed-only races; never derive it from fuel range.",
                    "get_remaining_race_time": "Read ONLY time LEFT on the race session clock. "
                    "Clock expiry is not an exact finish crossing or remaining lap count.",
                }[name]
            elif name == "get_race_progress":
                description += " For a remaining/count/time question prefer its focused tool."
            if capability == "current_classification":
                description += (
                    " Use for first/last checks and classification comparisons. "
                    "Reported-car count is NOT starting grid size."
                )
                if name.startswith("check_if_driver_is_"):
                    description = (
                        "Authoritative current Yes/No check: whether the DRIVER is "
                        + ("FIRST/leading." if name.endswith("first")
                           else "LAST in classification.")
                        + " Also returns supporting current ranks. Use the returned Yes/No fact."
                    )
            if capability == "fuel_range":
                description += " Also returns current fuel and observed consumption when available."
            if capability.startswith("relative_pace_"):
                description += (
                    " Returns ONLY a qualitative state and Yes/No closing verdict. "
                    "Not lap pace or gain/loss per lap; no numerical gap-change rate is exposed."
                )
            tools.append(function_tool(name, description, {}, []))
        opponent_properties = {
            "opponent": {"type": "string", "enum": ["ahead", "behind", "leader"]},
            "opponent_position": {"type": "integer", "minimum": 1},
        }
        tools.append(function_tool(
            "read_current_telemetry",
            "Read normalized current fuel, lap NUMBER, pit state, metre distances, "
            "last/best lap TIMES, completed laps, race lap limit/remaining laps/time, "
            "stint lap, fuel burn, gaps, simple battle state, session time/phase, flags or "
            "opponent position/completed-lap count/signed gap. Select only requested signals. "
            "Use opponent_position for a specific classified car, otherwise opponent selects "
            "ahead/behind/leader. No speed/velocity, multi-lap pace trends, driver names, "
            "damage, weather or car controls. "
            "For fresh current position/gaps prefer their dedicated tools.",
            {"signals": {"type": "array", "items": {"type": "string", "enum": list(SIGNALS)},
                         "minItems": 1, "maxItems": 8, "uniqueItems": True},
             **opponent_properties}, ["signals"],
        ))
        tools.append(function_tool(
            "query_telemetry_history",
            "Use existing bounded numeric Telemetry Memory: latest, window delta/mean/minimum/"
            "maximum (window_s required, at most 120 seconds). Field count/mean/minimum/"
            "maximum aggregate CURRENT reported positions or lap numbers (no window). "
            "Player signals: position, lap_number, fuel_l, car_ahead_distance_m, "
            "car_behind_distance_m. Context: stint_lap, fuel_trend_l_per_lap, gap_ahead_s, "
            "gap_behind_s. Opponent: position, lap_number, gap_to_player_s; select opponent "
            "or opponent_position. Gap signals support ONLY latest, not window changes. "
            "Delta is TOTAL signed change. Positive means increasing, negative decreasing. "
            "No speed, numerical rates, lap-based gap change, arbitrary past point lookup "
            "or completed-lap timing.",
            {"source": {"type": "string", "enum": ["player", "context", "opponent", "field"]},
             "signal": {"type": "string", "enum": sorted({
                 signal for source, signal, _ in SIGNALS.values()
                 if source != "frame" and signal not in {"in_pit_lane", "battle_state"}
             })},
             "operation": {"type": "string", "enum": [
                 "latest", "delta", "mean", "minimum", "maximum", "count",
             ]}, "window_s": {"type": "number", "exclusiveMinimum": 0, "maximum": 120},
             **opponent_properties}, ["source", "signal", "operation"],
        ))
        return tools

    @staticmethod
    def is_tool(name):
        return name in {*CAPABILITY_TOOLS, "read_current_telemetry", "query_telemetry_history"}

    def _opponent(self, selection, position):
        if selection not in {"ahead", "behind", "leader"}:
            raise ValueError("invalid opponent selector")
        opponents = self.memory.latest_context().frame.opponents
        if position is not None:
            matches = [item for item in opponents if item.position == position]
            return matches[0] if len(matches) == 1 else None
        if selection == "leader":
            return next((item for item in opponents if item.position == 1), None)
        return self.capabilities._nearest_opponent(self.memory.latest_context(), selection)

    def _raw(self, path, opponent):
        source, signal, unit = SIGNALS[path]
        context = self.memory.latest_context()
        frame = context.frame
        required = _CAPABILITY_REQUIREMENT.get(path)
        if required and required not in frame.capabilities:
            return None
        owner = {"player": frame.player, "context": context, "frame": frame,
                 "opponent": opponent}[source]
        value = getattr(owner, signal) if owner is not None else None
        if path == "frame.flags":
            value = ", ".join(flag.value for flag in value) if value else "no active flags"
        elif hasattr(value, "value"):
            value = value.value
        if value is None or value == "unknown":
            return None
        subject = opponent.driver_id if source == "opponent" and opponent else source
        return EvidenceItem(
            evidence_id="ministral-read:raw:" + path.replace(".", "_"),
            session_id=frame.session_id, source_sequence=frame.sequence,
            observed_at=frame.observed_at, kind="measurement", temporal_scope="current",
            subject=subject, metric=signal, value=value, unit=unit,
            source_fields=(f"opponent:{subject}.{signal}" if source == "opponent" else path,),
            valid_until=frame.observed_at + timedelta(seconds=3),
        )

    def execute(self, call):
        if not self.is_tool(call.name):
            raise LocalIntelligenceError("dialogue_unknown_tool")
        try:
            evidence, unknowns = [], []
            if call.name in CAPABILITY_TOOLS:
                if call.arguments:
                    raise ValueError("calculation takes no arguments")
                capability = CAPABILITY_TOOLS[call.name]
                descriptor = next(item for item in self.capabilities.catalog()
                                  if item.capability_id == capability)
                result = self.capabilities.execute(CapabilityRequest(
                    request_id="ministral-read", capability_id=capability,
                ))
                evidence = [item.model_copy(update={
                    "evidence_id": "ministral-read:" + capability + ":" +
                                   item.evidence_id.rsplit(":", 1)[-1],
                    "temporal_scope": descriptor.temporal_scope,
                }) for item in result.evidence
                            if item.kind != "unknown" and item.metric != "gap_change_rate"]
                focused = {
                    "get_remaining_race_laps": "race_laps_remaining",
                    "get_completed_lap_count": "laps_completed",
                    "get_total_race_laps": "total_laps",
                    "get_remaining_race_time": "time_remaining_s",
                }.get(call.name)
                if focused:
                    evidence = [item for item in evidence if item.metric == focused]
                    if not evidence and result.status == "available":
                        unknowns.append("race_progress:missing_" + focused)
                if call.name == "get_fuel_to_finish":
                    evidence = [item for item in evidence
                                if item.metric not in {"fuel_l", "fuel_margin_l"}
                                and not (item.metric in {"fuel_shortfall_l", "fuel_surplus_l"}
                                         and item.value == 0)]
                if call.name.startswith("check_if_driver_is_"):
                    wanted = {"position", "is_first"} if call.name.endswith("first") else {
                        "position", "is_last",
                    }
                    evidence = [item for item in evidence if item.metric in wanted
                                and (item.metric != "position" or item.subject == "player")]
                if result.status == "unavailable":
                    unknowns.append(capability + ":" + result.unavailable_reason)
                elif capability == "position_change":
                    evidence.append(self._window_evidence(_POSITION_WINDOW_S,
                                                          descriptor.temporal_scope))
                elif capability.startswith("relative_pace_"):
                    state = next(item for item in evidence if item.metric == "relative_pace_state")
                    # Present the core's categorical closing result as an explicit
                    # Yes/No fact, so "Yes, they are falling back" cannot pass.
                    evidence.append(state.model_copy(update={
                        "evidence_id": "ministral-read:" + capability + ":is_closing",
                        "metric": "is_closing", "value": state.value == "closing",
                    }))
                if capability == "fuel_range":
                    for path in ("player.fuel_l", "context.fuel_trend_l_per_lap"):
                        item = self._raw(path, None)
                        if item is not None:
                            evidence.append(item)
            elif call.name == "read_current_telemetry":
                request = CurrentRead.model_validate(call.arguments)
                if len(set(request.signals)) != len(request.signals) or any(
                    path not in SIGNALS for path in request.signals
                ):
                    raise ValueError("invalid signals")
                opponent = self._opponent(request.opponent, request.opponent_position)
                for path in request.signals:
                    item = self._raw(path, opponent)
                    if item is None:
                        unknowns.append(path + ":missing_current_value")
                    else:
                        evidence.append(item)
            else:
                request = HistoryRead.model_validate(call.arguments)
                path = request.source + "." + request.signal
                if path not in SIGNALS and not (
                    request.source == "field" and request.signal in {"position", "lap_number"}
                ):
                    raise ValueError("signal is not exposed to driver conversation")
                if request.signal in {"gap_ahead_s", "gap_behind_s", "gap_to_player_s"} and (
                    request.operation != "latest"
                ):
                    # A seconds-based window is not a completed-lap comparison.
                    return (reading_message(call, (), ("lap_based_gap_change_unavailable",)),
                            (), ("lap_based_gap_change_unavailable",))
                if request.signal in {"last_lap_time_s", "best_lap_time_s"} and (
                    request.operation != "latest"
                ):
                    # Repeated SDK last/best values are samples, not distinct lap records.
                    return (reading_message(call, (), ("completed_lap_history_required",)),
                            (), ("completed_lap_history_required",))
                opponent = self._opponent(request.opponent, request.opponent_position)
                selector = SignalSelector(
                    source=request.source, signal=request.signal,
                    subject_id=(opponent.driver_id if opponent else "missing-opponent")
                    if request.source == "opponent" else None,
                )
                query = EvidenceQuery(
                    query_id="ministral-read:history:history_value", selector=selector,
                    operation=request.operation, window_s=request.window_s,
                )
                if request.source == "opponent" and opponent is None:
                    if "opponent." + request.signal not in SIGNALS:
                        raise ValueError("unknown opponent signal")
                    unknowns.append("opponent:missing_selected_car")
                else:
                    if selector not in [item.selector for item in self.memory.signal_catalog()]:
                        raise ValueError("signal is not in the normalized numeric catalog")
                    item = self.memory.query(query)
                    if item.kind == "unknown":
                        unknowns.append("history:missing_value_or_insufficient_window")
                    else:
                        evidence.append(item.model_copy(update={
                            "temporal_scope": "current",
                            "unit": "positions" if request.signal == "position"
                            and request.operation == "delta" else item.unit,
                        }))
                        if request.window_s is not None:
                            evidence.append(self._window_evidence(request.window_s, "current"))
            evidence = tuple(item.model_copy(update={"claim": display_claim(item, self.language)})
                             for item in evidence)
            return (reading_message(call, evidence, unknowns, self.meanings),
                    evidence, tuple(unknowns))
        except (ValidationError, ValueError, TypeError) as error:
            raise LocalIntelligenceError("dialogue_invalid_tool_arguments") from error


def bind_extended(speech, evidence):
    def token(item):
        return "{{" + placeholder(item) + "}}"

    def matching_number(number, candidates):
        matches = [item for item in candidates if not isinstance(item.value, bool)
                   and isinstance(item.value, (int, float))
                   and float(number.replace(",", ".")) == round(item.value, 3)]
        if len(matches) != 1:
            raise LocalIntelligenceError("dialogue_unverified_or_ambiguous_reading")
        return matches[0]

    # Prefer explicit evidence tokens, but accept the model's ordinary number/unit
    # wording using the same dimensional check as existing position/gap binding.
    # A value alone is never sufficient to bind two different metrics.
    numeric = [item for item in evidence if isinstance(item.value, (int, float))
               and not isinstance(item.value, bool)]
    lap_forms = {}
    for item in numeric:
        if item.metric in LAP_TIME_METRICS and item.unit == "s":
            for language in ("en", "tr"):
                form = format_lap_time(item.value, language)
                lap_forms.setdefault(form, []).append(item)
    for form in sorted(lap_forms, key=len, reverse=True):
        pattern = r"(?<![\w{])" + re.escape(form).replace(r"\ ", r"[-\s]+") + r"(?![\w}])"
        if re.search(pattern, speech, re.I):
            if len(lap_forms[form]) != 1:
                raise LocalIntelligenceError("dialogue_unverified_or_ambiguous_reading")
            speech = re.sub(pattern, lambda _, item=lap_forms[form][0]: token(item),
                            speech, flags=re.I)
    if len(evidence) == 1 and len(numeric) == 1:
        item = numeric[0]
        bare = speech.strip().rstrip(".!?").casefold()
        forms = {str(item.value).casefold(), f"{item.value:g}"}
        forms.update(filter(None, (number_words(item.value), number_words(item.value, "tr"))))
        if bare in forms:
            return token(item) + ("." if speech.rstrip().endswith(".") else "")
    for item in numeric:
        forms = [number_words(item.value), number_words(item.value, "tr")]
        if (item.unit is None and item.metric in {"lap_number", "stint_lap"}
                and item.value == int(item.value) and 0 < item.value < len(_TR_ORDINALS)):
            forms.append(_TR_ORDINALS[int(item.value)])
        for form in filter(None, forms):
            unit = _UNIT_PATTERNS.get(item.unit, re.escape(item.unit or ""))
            tail = r"(?=\s*" + unit + r"\b)" if item.unit else r"(?=\s+(?:laps?|tur\w*)\b)"
            speech = re.sub(r"\b" + re.escape(form).replace(r"\ ", r"[-\s]+") + r"\b" + tail,
                            f"{item.value:g}", speech, flags=re.I)
    for unit in sorted({item.unit for item in numeric if item.unit},
                       key=lambda unit: -len(_UNIT_PATTERNS.get(unit, unit))):
        candidates = [item for item in numeric if item.unit == unit]
        if unit == "position":
            pattern = r"\bP" + _NUMBER + r"\b|\b(?P<ordinal>\d+)(?:st|nd|rd|th)\b"

            def bind_rank(match, candidates=candidates):
                return token(matching_number(match.group("number") or match.group("ordinal"),
                                             candidates))

            speech = re.sub(pattern, bind_rank, speech, flags=re.I)
        else:
            pattern = (r"(?<![\w{.])" + _NUMBER + r"\s*" +
                       _UNIT_PATTERNS.get(unit, re.escape(unit)) + r"\b")

            def bind_unit(match, candidates=candidates):
                return token(matching_number(match.group("number"), candidates))

            speech = re.sub(pattern, bind_unit, speech, flags=re.I)
    # Counts such as 'lap 7' have no trailing unit. Require an output label too.
    counts = [item for item in numeric if item.unit is None]
    if counts:
        def bind_lap(match):
            return match.group("label") + token(matching_number(match.group("number"), counts))

        speech = re.sub(r"(?P<label>\b(?:lap|tur)\s*(?:number\s*)?#?\s*)" + _NUMBER + r"\b",
                        bind_lap, speech, flags=re.I)
        speech = re.sub(r"\b" + _NUMBER + r"\.\s*(?=tur\w*\b)",
                        lambda match: token(matching_number(match.group("number"), counts)) + ". ",
                        speech, flags=re.I)
        speech = re.sub(r"\b" + _NUMBER + r"\s+(?=laps?\b|tur\w*\b)",
                        lambda match: token(matching_number(match.group("number"), counts)) + " ",
                        speech, flags=re.I)
    # Categorical/boolean readings are refreshable too. Exact application-owned
    # labels may bind; other interpretations must use the explicit reference.
    labels = {}
    for item in evidence:
        if isinstance(item.value, (str, bool)):
            labels.setdefault(item.claim, []).append(item)
    for label in sorted(labels, key=len, reverse=True):
        if len(labels[label]) != 1:
            continue
        speech = re.sub(r"(?<![\w{])" + re.escape(label) + r"(?![\w}])",
                        lambda _, item=labels[label][0]: token(item), speech, flags=re.I)
    for item in numeric:
        if item.unit == "position":
            speech = speech.replace("P" + token(item), token(item))
        elif item.unit:
            speech = re.sub(re.escape(token(item)) + r"\s+" +
                            _UNIT_PATTERNS.get(item.unit, re.escape(item.unit)) + r"\b",
                            lambda _, item=item: token(item), speech, flags=re.I)
        if item.metric == "fuel_laps_remaining":
            speech = re.sub(re.escape(token(item)) + r"\s+of\s+fuel\b",
                            lambda _, item=item: token(item), speech, flags=re.I)
        if item.metric in LAP_TIME_METRICS:
            speech = re.sub(re.escape(token(item)) +
                            r"\s*(?:/\s*|per\s+|each\s+|every\s+)(?:completed\s+)?laps?\b",
                            lambda _, item=item: token(item), speech, flags=re.I)
    allowed = {placeholder(item) for item in evidence}
    used = set(_TOKEN.findall(speech)) - {"unavailable"}
    if not used or not used <= allowed:
        raise LocalIntelligenceError("dialogue_reading_reference_required")
    if "catch_time_s" in used and re.search(
        r"\{\{catch_time_s\}\}\s+(?:ahead|behind|back)\b", speech, re.I,
    ):
        raise LocalIntelligenceError("dialogue_catch_time_is_not_current_gap")
    if any(isinstance(item.value, bool) for item in evidence) and re.search(
        r"\b(?:yes|no|evet|hayır)\b", _TOKEN.sub("", speech), re.I,
    ):
        raise LocalIntelligenceError("dialogue_boolean_reference_required")
    # Numeric and categorical state both travel through refreshable references.
    # The ordinary grounder still rejects remaining raw/spelled-out numbers.
    for state, amount in (("direction", "positions_changed"), ("state", "gap_change_rate")):
        if amount in used and state in allowed and state not in used:
            raise LocalIntelligenceError("dialogue_state_reference_required")
    return speech


def extended_references(speech, evidence):
    used = set(_TOKEN.findall(speech))
    return tuple(EvidenceReference(
        placeholder=placeholder(item), evidence_id=item.evidence_id, field="claim",
    ) for item in evidence if placeholder(item) in used)
