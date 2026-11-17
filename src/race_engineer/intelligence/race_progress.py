"""Lap timing, race distance and conservative fuel calculations from known inputs."""

from dataclasses import dataclass
from datetime import timedelta

from race_engineer.core.contracts import RaceContext
from race_engineer.core.enums import SessionPhase
from race_engineer.core.intelligence import (
    CapabilityDescriptor,
    CapabilityOutputDescriptor,
    CapabilityRequest,
    CapabilityResult,
    EvidenceItem,
)

FUEL_RESERVE_LAPS = 1
Value = int | float | str | bool
Output = tuple[str, str, str | None]


@dataclass(frozen=True)
class ReadingSpec:
    description: str
    outputs: tuple[Output, ...]
    inputs: tuple[str, ...]
    uncertainty: str


SPECS = {
    "last_lap_time": ReadingSpec(
        "Read the simulator-reported last completed lap duration, not lap number or current speed.",
        (("last_lap_time_s", "simulator-reported last completed lap duration", "s"),),
        ("player.last_lap_time_s",),
        "SDK timing, not a claim that the lap was clean, valid for qualifying, or representative.",
    ),
    "best_lap_time": ReadingSpec(
        "Read the simulator-reported player's best lap duration in this session.",
        (("best_lap_time_s", "simulator-reported session best lap duration", "s"),),
        ("player.best_lap_time_s",),
        "Not an all-time, track-record or opponent best lap.",
    ),
    "lap_time_comparison": ReadingSpec(
        "Compare the last completed lap duration with the reported session best. "
        "Not a multi-lap improvement trend or a comparison to another driver.",
        (("difference_s", "absolute last-lap difference from reported session best", "s"),
         ("comparison", "slower_than_best, matches_best, or faster_than_reported_best", None)),
        ("player.last_lap_time_s", "player.best_lap_time_s"),
        "Two reported durations only; no clean-lap filter, consistency or opponent pace model.",
    ),
    "race_progress": ReadingSpec(
        "Read current/completed lap counts, total scheduled race laps and SDK-reported "
        "remaining race laps/time. Timed races have no exact remaining lap count.",
        (("current_lap", "player's current lap number, not completed count", "laps"),
         ("laps_completed", "player's completed lap count", "laps"),
         ("total_laps", "scheduled race lap limit", "laps"),
         ("race_laps_remaining", "SDK-reported remaining race laps, never fuel range", "laps"),
         ("time_remaining_s", "remaining session clock, not an exact finish time", "s")),
        ("player.lap_number", "player.laps_completed", "frame.session_laps_total",
         "frame.session_laps_remaining", "frame.session_time_remaining_s"),
        "Time expiry is not necessarily the finish crossing; lapped cars may finish sooner "
        "than their full scheduled distance. Missing values stay unavailable.",
    ),
    "fuel_to_finish": ReadingSpec(
        "Assess whether current fuel covers the remaining full scheduled player laps plus "
        "ONE lap of reserve at observed burn. Use for 'enough fuel?' and fuel follow-ups. "
        "Requires a race lap limit; unavailable for timed-only races or missing consumption.",
        (("fuel_l", "current fuel in the tank", "l"),
         ("scheduled_laps_remaining", "conservative full scheduled player laps still to drive, "
          "including the current lap; not fuel endurance", "laps"),
         ("reserve_laps", "explicit extra fuel reserve in laps", "laps"),
         ("fuel_required_l", "estimated fuel required INCLUDING the declared reserve", "l"),
         ("fuel_margin_l", "fuel minus requirement; negative means a shortfall", "l"),
         ("fuel_shortfall_l", "positive fuel shortfall, zero if sufficient", "l"),
         ("fuel_surplus_l", "positive fuel surplus above requirement, zero if insufficient", "l"),
         ("enough_fuel", "authoritative Yes/No against the declared distance and reserve", None)),
        ("player.fuel_l", "context.fuel_trend_l_per_lap", "player.laps_completed",
         "frame.session_laps_total", "frame.session_laps_remaining"),
        "Estimate at observed burn, counting full remaining laps plus one reserve lap. "
        "Time limits, being lapped or an earlier finish can shorten the actual distance. "
        "No fuel saving, pit strategy or guaranteed finish prediction.",
    ),
}


def _values(capability_id: str, context: RaceContext) -> tuple[dict[str, Value], str | None]:
    frame, player = context.frame, context.frame.player
    if capability_id in {"last_lap_time", "best_lap_time"}:
        field = "last_lap_time_s" if capability_id == "last_lap_time" else "best_lap_time_s"
        value = (player.last_lap_time_s if capability_id == "last_lap_time"
                 else player.best_lap_time_s)
        return ({field: value}, None) if value is not None else ({}, "missing_" + field)
    if capability_id == "lap_time_comparison":
        last, best = player.last_lap_time_s, player.best_lap_time_s
        if last is None or best is None:
            return {}, "missing_last_or_best_lap_time"
        delta = last - best
        return {"difference_s": abs(delta), "comparison": (
            "slower_than_best" if delta > 0 else "faster_than_reported_best" if delta < 0
            else "matches_best"
        )}, None
    if (frame.session_type or "").casefold() != "race":
        return {}, "race_session_not_confirmed"
    if capability_id == "race_progress":
        candidates = {
            "current_lap": player.lap_number, "laps_completed": player.laps_completed,
            "total_laps": frame.session_laps_total,
            "race_laps_remaining": frame.session_laps_remaining,
            "time_remaining_s": frame.session_time_remaining_s,
        }
        values: dict[str, Value] = {key: value for key, value in candidates.items()
                                    if value is not None}
        return (values, None) if values else ({}, "missing_race_progress")
    if frame.session_phase not in {SessionPhase.GREEN, SessionPhase.CAUTION}:
        return {}, "race_not_active_for_fuel_estimate"
    total, completed = frame.session_laps_total, player.laps_completed
    if total is None:
        return {}, "timed_or_unknown_race_distance"
    if completed is None or completed < 0 or completed >= total:
        return {}, "missing_or_finished_completed_lap_count"
    fuel, burn = player.fuel_l, context.fuel_trend_l_per_lap
    if fuel is None or burn is None or burn <= 0:
        return {}, "missing_fuel_or_positive_observed_burn"
    # Count the current unfinished lap in full. This deliberately avoids a
    # false precision estimate based on distance fraction or the leader's clock.
    remaining = max(total - completed, frame.session_laps_remaining or 0)
    required = (remaining + FUEL_RESERVE_LAPS) * burn
    return {
        "fuel_l": fuel, "scheduled_laps_remaining": remaining,
        "reserve_laps": FUEL_RESERVE_LAPS, "fuel_required_l": required,
        "fuel_margin_l": fuel - required, "enough_fuel": fuel >= required,
        "fuel_shortfall_l": max(0.0, required - fuel),
        "fuel_surplus_l": max(0.0, fuel - required),
    }, None


def progress_catalog(context: RaceContext, freshness_s: float) -> tuple[CapabilityDescriptor, ...]:
    descriptors = []
    for capability_id, spec in SPECS.items():
        _, reason = _values(capability_id, context)
        descriptors.append(CapabilityDescriptor(
            capability_id=capability_id, description=spec.description, temporal_scope="current",
            required_inputs=spec.inputs,
            outputs=tuple(CapabilityOutputDescriptor(
                output_id=key, description=description, unit=unit,
            ) for key, description, unit in spec.outputs),
            freshness_s=freshness_s, uncertainty=spec.uncertainty,
            available=reason is None, unavailable_reason=reason,
        ))
    return tuple(descriptors)


def execute_progress(
    request: CapabilityRequest, context: RaceContext, freshness_s: float,
) -> CapabilityResult:
    spec, frame = SPECS[request.capability_id], context.frame
    values, reason = _values(request.capability_id, context)
    evidence = tuple(EvidenceItem(
        evidence_id=f"{request.request_id}:{key}", session_id=frame.session_id,
        source_sequence=frame.sequence, observed_at=frame.observed_at,
        kind="unknown" if reason else "derived", temporal_scope="current", subject="player",
        metric=key, value=None if reason else values[key], unit=None if reason else unit,
        source_fields=spec.inputs, valid_until=frame.observed_at + timedelta(seconds=freshness_s),
    ) for key, _, unit in spec.outputs if reason or key in values)
    return CapabilityResult(
        request_id=request.request_id, capability_id=request.capability_id,
        status="unavailable" if reason else "available", evidence=evidence,
        unavailable_reason=reason,
    )
