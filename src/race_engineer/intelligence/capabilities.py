"""Typed deterministic race calculations exposed to the Context Engineer."""

from datetime import timedelta
from typing import Literal

from race_engineer.core.contracts import OpponentState, RaceContext
from race_engineer.core.intelligence import (
    CapabilityDescriptor,
    CapabilityOutputDescriptor,
    CapabilityRequest,
    CapabilityResult,
    EvidenceItem,
    EvidenceQuery,
    SignalSelector,
)
from race_engineer.core.interfaces import TelemetryMemory

_CLASSIFICATION = "current_classification"
_FUEL_RANGE = "fuel_range"
_POSITION_CHANGE = "position_change"
_GAP_AHEAD = "gap_ahead"
_GAP_BEHIND = "gap_behind"
_RELATIVE_PACE_AHEAD = "relative_pace_ahead"
_RELATIVE_PACE_BEHIND = "relative_pace_behind"
_CATCH_TIME_AHEAD = "catch_time_ahead"
_CATCH_TIME_BEHIND = "catch_time_behind"
_PIT_LOSS = "pit_loss_projection"
_PIT_STOP_DURATION = "pit_stop_duration_projection"
_PROJECTED_REJOIN = "projected_rejoin_position"

_POSITION_WINDOW_S = 30.0
_RELATIVE_PACE_WINDOW_S = 10.0
_MIN_CLOSING_RATE_S_PER_S = 0.005


class RaceCapabilityError(Exception):
    """A capability request violates the advertised registry contract."""


class DeterministicRaceCapabilities:
    """Calculate race facts from normalized telemetry without model-authored arithmetic."""

    def __init__(self, memory: TelemetryMemory, *, freshness_s: float = 3.0) -> None:
        if freshness_s <= 0 or freshness_s > 30:
            raise ValueError("capability freshness must be between zero and thirty seconds")
        self._memory = memory
        self._freshness_s = freshness_s

    @staticmethod
    def _classification_reason(context: RaceContext) -> str | None:
        return (
            None
            if context.frame.player.position is not None
            else "missing_player_position"
        )

    @staticmethod
    def _fuel_range_reason(context: RaceContext) -> str | None:
        if context.frame.player.fuel_l is None:
            return "missing_current_fuel"
        if context.fuel_trend_l_per_lap is None:
            return "missing_fuel_burn_per_lap"
        if context.fuel_trend_l_per_lap <= 0:
            return "invalid_fuel_burn_per_lap"
        return None

    @staticmethod
    def _window_query(
        query_id: str,
        *,
        source: Literal["context", "opponent"] = "context",
        signal: str,
        operation: str,
        window_s: float,
        subject_id: str | None = None,
    ) -> EvidenceQuery:
        return EvidenceQuery.model_validate(
            {
                "query_id": query_id,
                "selector": SignalSelector(
                    source=source,
                    signal=signal,
                    subject_id=subject_id,
                ),
                "operation": operation,
                "window_s": window_s,
            }
        )

    def _position_delta(self, query_id: str) -> EvidenceItem:
        return self._memory.query(
            EvidenceQuery(
                query_id=query_id,
                selector=SignalSelector(source="player", signal="position"),
                operation="delta",
                window_s=_POSITION_WINDOW_S,
            )
        )

    @staticmethod
    def _nearest_opponent(
        context: RaceContext,
        direction: str,
    ) -> OpponentState | None:
        candidates = tuple(
            opponent
            for opponent in context.frame.opponents
            if opponent.gap_to_player_s is not None
            and (
                opponent.gap_to_player_s < 0
                if direction == "ahead"
                else opponent.gap_to_player_s > 0
            )
        )
        return min(
            candidates,
            key=lambda opponent: abs(opponent.gap_to_player_s or 0),
            default=None,
        )

    def _gap_trend(self, direction: str, query_id: str) -> EvidenceItem:
        current = self._memory.latest_context()
        opponent = self._nearest_opponent(current, direction)
        if opponent is None:
            raise RaceCapabilityError(f"missing_current_gap_{direction}")
        evidence = self._memory.query(
            self._window_query(
                query_id,
                source="opponent",
                signal="gap_to_player_s",
                operation="trend",
                window_s=_RELATIVE_PACE_WINDOW_S,
                subject_id=opponent.driver_id,
            )
        )
        if (
            direction == "ahead"
            and isinstance(evidence.value, (int, float))
            and not isinstance(evidence.value, bool)
        ):
            return evidence.model_copy(update={"value": -evidence.value})
        return evidence

    def _position_change_reason(self, context: RaceContext) -> str | None:
        if context.frame.player.position is None:
            return "missing_player_position"
        if self._position_delta("catalog-position-change").kind == "unknown":
            return "insufficient_position_history_30s"
        return None

    def _gap_reason(self, context: RaceContext, direction: str) -> str | None:
        return (
            None
            if self._nearest_opponent(context, direction) is not None
            else f"missing_current_gap_{direction}"
        )

    def _relative_pace_reason(self, context: RaceContext, direction: str) -> str | None:
        gap_reason = self._gap_reason(context, direction)
        if gap_reason is not None:
            return gap_reason
        trend = self._gap_trend(direction, f"catalog-relative-{direction}")
        if trend.kind == "unknown" or trend.confidence < 0.8:
            return f"insufficient_gap_history_{direction}_10s"
        return None

    def _catch_time_reason(self, context: RaceContext, direction: str) -> str | None:
        reason = self._relative_pace_reason(context, direction)
        if reason is not None:
            return reason
        trend = self._gap_trend(direction, f"catalog-catch-{direction}").value
        if isinstance(trend, bool) or not isinstance(trend, (int, float)):
            return f"insufficient_gap_history_{direction}_10s"
        if trend >= -_MIN_CLOSING_RATE_S_PER_S:
            return (
                "player_not_closing_car_ahead"
                if direction == "ahead"
                else "car_behind_not_closing_player"
            )
        return None

    def catalog(self) -> tuple[CapabilityDescriptor, ...]:
        current = self._memory.latest_context()
        classification_reason = self._classification_reason(current)
        fuel_reason = self._fuel_range_reason(current)
        position_change_reason = self._position_change_reason(current)
        gap_ahead_reason = self._gap_reason(current, "ahead")
        gap_behind_reason = self._gap_reason(current, "behind")
        relative_ahead_reason = self._relative_pace_reason(current, "ahead")
        relative_behind_reason = self._relative_pace_reason(current, "behind")
        catch_ahead_reason = self._catch_time_reason(current, "ahead")
        catch_behind_reason = self._catch_time_reason(current, "behind")
        return (
            CapabilityDescriptor(
                capability_id=_CLASSIFICATION,
                description=(
                    "Calculate the player's current classification, the last occupied "
                    "classification, the number of cars with reported positions, and whether "
                    "the player is currently last."
                ),
                temporal_scope="current",
                required_inputs=("player.position", "field.position"),
                outputs=(
                    CapabilityOutputDescriptor(
                        output_id="player_position",
                        description="player's current classified position",
                        unit="position",
                    ),
                    CapabilityOutputDescriptor(
                        output_id="last_position",
                        description="last occupied current field position",
                        unit="position",
                    ),
                    CapabilityOutputDescriptor(
                        output_id="classified_cars",
                        description="cars with a reported current position",
                        unit="cars",
                    ),
                    CapabilityOutputDescriptor(
                        output_id="is_last",
                        description="authoritative current last-place relationship",
                        speakable=False,
                    ),
                ),
                freshness_s=self._freshness_s,
                uncertainty=(
                    "Exact over cars that currently report a classified position; cars without "
                    "a position are excluded."
                ),
                available=classification_reason is None,
                unavailable_reason=classification_reason,
            ),
            CapabilityDescriptor(
                capability_id=_FUEL_RANGE,
                description=(
                    "Calculate estimated laps remaining from current fuel divided by the "
                    "rolling observed fuel burn per completed lap."
                ),
                temporal_scope="current",
                required_inputs=("player.fuel_l", "context.fuel_trend_l_per_lap"),
                outputs=(
                    CapabilityOutputDescriptor(
                        output_id="laps_remaining",
                        description="estimated laps remaining at the observed average burn",
                        unit="laps",
                    ),
                ),
                freshness_s=self._freshness_s,
                uncertainty=(
                    "Uses the rolling average burn only; it has no caution, traffic, driving "
                    "style, reserve, or variance margin."
                ),
                available=fuel_reason is None,
                unavailable_reason=fuel_reason,
            ),
            CapabilityDescriptor(
                capability_id=_POSITION_CHANGE,
                description=(
                    "Calculate the player's net classified-position change over the last "
                    "30 seconds, returning an application-owned direction and magnitude."
                ),
                temporal_scope="historical",
                required_inputs=("player.position[30s]",),
                outputs=(
                    CapabilityOutputDescriptor(
                        output_id="direction",
                        description="whether positions were gained, lost, or unchanged",
                    ),
                    CapabilityOutputDescriptor(
                        output_id="positions_changed",
                        description="absolute number of classified positions changed",
                        unit="positions",
                    ),
                ),
                freshness_s=self._freshness_s,
                uncertainty=(
                    "Net change over a fixed rolling 30-second window; it does not attribute "
                    "passes, pit cycles, disconnects, or classification corrections."
                ),
                available=position_change_reason is None,
                unavailable_reason=position_change_reason,
            ),
            CapabilityDescriptor(
                capability_id=_GAP_AHEAD,
                description="Return the current gap to the nearest same-lap car ahead.",
                temporal_scope="current",
                required_inputs=("nearest_same_lap_opponent.gap_to_player_s",),
                outputs=(
                    CapabilityOutputDescriptor(
                        output_id="gap_s",
                        description="current gap to the nearest same-lap car ahead",
                        unit="s",
                    ),
                ),
                freshness_s=self._freshness_s,
                uncertainty=(
                    "Only same-lap opponents with a reported timing gap are considered."
                ),
                available=gap_ahead_reason is None,
                unavailable_reason=gap_ahead_reason,
            ),
            CapabilityDescriptor(
                capability_id=_GAP_BEHIND,
                description="Return the current gap to the nearest same-lap car behind.",
                temporal_scope="current",
                required_inputs=("nearest_same_lap_opponent.gap_to_player_s",),
                outputs=(
                    CapabilityOutputDescriptor(
                        output_id="gap_s",
                        description="current gap to the nearest same-lap car behind",
                        unit="s",
                    ),
                ),
                freshness_s=self._freshness_s,
                uncertainty=(
                    "Only same-lap opponents with a reported timing gap are considered."
                ),
                available=gap_behind_reason is None,
                unavailable_reason=gap_behind_reason,
            ),
            CapabilityDescriptor(
                capability_id=_RELATIVE_PACE_AHEAD,
                description=(
                    "Determine whether the player is closing on or losing ground to the "
                    "nearest same-lap car ahead from the last 10 seconds of gap movement."
                ),
                temporal_scope="current",
                required_inputs=("nearest_same_lap_opponent.gap_to_player_s[10s]",),
                outputs=(
                    CapabilityOutputDescriptor(
                        output_id="state",
                        description="closing, losing_ground, or stable relative to the car ahead",
                    ),
                    CapabilityOutputDescriptor(
                        output_id="gap_change_rate",
                        description="absolute rate of movement in the ahead gap per second",
                        unit="s/s",
                    ),
                ),
                freshness_s=self._freshness_s,
                uncertainty=(
                    "A rolling linear gap trend, not clean-air lap pace; traffic, incidents, "
                    "pit cycles, and timing jitter may dominate short windows."
                ),
                available=relative_ahead_reason is None,
                unavailable_reason=relative_ahead_reason,
            ),
            CapabilityDescriptor(
                capability_id=_RELATIVE_PACE_BEHIND,
                description=(
                    "Determine whether the nearest same-lap car behind is closing on or "
                    "falling back from the player using 10 seconds of gap movement."
                ),
                temporal_scope="current",
                required_inputs=("nearest_same_lap_opponent.gap_to_player_s[10s]",),
                outputs=(
                    CapabilityOutputDescriptor(
                        output_id="state",
                        description="closing, falling_back, or stable for the car behind",
                    ),
                    CapabilityOutputDescriptor(
                        output_id="gap_change_rate",
                        description="absolute rate of movement in the behind gap per second",
                        unit="s/s",
                    ),
                ),
                freshness_s=self._freshness_s,
                uncertainty=(
                    "A rolling linear gap trend, not clean-air lap pace; traffic, incidents, "
                    "pit cycles, and timing jitter may dominate short windows."
                ),
                available=relative_behind_reason is None,
                unavailable_reason=relative_behind_reason,
            ),
            CapabilityDescriptor(
                capability_id=_CATCH_TIME_AHEAD,
                description=(
                    "Project seconds until the player catches the nearest same-lap car ahead "
                    "if the current 10-second closing trend remains constant."
                ),
                temporal_scope="future_counterfactual",
                required_inputs=(
                    "nearest_same_lap_opponent.gap_to_player_s",
                    "nearest_same_lap_opponent.gap_to_player_s[10s]",
                ),
                outputs=(
                    CapabilityOutputDescriptor(
                        output_id="catch_time_s",
                        description="constant-trend time until catching the car ahead",
                        unit="s",
                    ),
                ),
                freshness_s=self._freshness_s,
                uncertainty=(
                    "Assumes the short-window gap trend remains constant and excludes traffic, "
                    "pit cycles, incidents, and lap differences."
                ),
                available=catch_ahead_reason is None,
                unavailable_reason=catch_ahead_reason,
            ),
            CapabilityDescriptor(
                capability_id=_CATCH_TIME_BEHIND,
                description=(
                    "Project seconds until the nearest same-lap car behind catches the player "
                    "if the current 10-second closing trend remains constant."
                ),
                temporal_scope="future_counterfactual",
                required_inputs=(
                    "nearest_same_lap_opponent.gap_to_player_s",
                    "nearest_same_lap_opponent.gap_to_player_s[10s]",
                ),
                outputs=(
                    CapabilityOutputDescriptor(
                        output_id="catch_time_s",
                        description="constant-trend time until the car behind catches the player",
                        unit="s",
                    ),
                ),
                freshness_s=self._freshness_s,
                uncertainty=(
                    "Assumes the short-window gap trend remains constant and excludes traffic, "
                    "pit cycles, incidents, and lap differences."
                ),
                available=catch_behind_reason is None,
                unavailable_reason=catch_behind_reason,
            ),
            CapabilityDescriptor(
                capability_id=_PIT_LOSS,
                description=(
                    "Project total pit-lane time loss for a future stop from a validated "
                    "track/car pit model."
                ),
                temporal_scope="future_counterfactual",
                required_inputs=("track.pit_loss_model",),
                outputs=(
                    CapabilityOutputDescriptor(
                        output_id="pit_loss_s",
                        description="projected time lost by taking the pit lane",
                        unit="s",
                    ),
                ),
                freshness_s=self._freshness_s,
                uncertainty="Unavailable until a validated track/car pit-loss model exists.",
                available=False,
                unavailable_reason="missing_pit_loss_model",
            ),
            CapabilityDescriptor(
                capability_id=_PIT_STOP_DURATION,
                description=(
                    "Project service duration for a future stop from requested fuel, tyres, "
                    "repairs, and simulator-specific service rules."
                ),
                temporal_scope="future_counterfactual",
                required_inputs=("pit.service_request", "simulator.service_rules"),
                outputs=(
                    CapabilityOutputDescriptor(
                        output_id="stop_duration_s",
                        description="projected stationary pit-service duration",
                        unit="s",
                    ),
                ),
                freshness_s=self._freshness_s,
                uncertainty="Unavailable until service requests and rules are normalized.",
                available=False,
                unavailable_reason="missing_pit_service_model",
            ),
            CapabilityDescriptor(
                capability_id=_PROJECTED_REJOIN,
                description=(
                    "Project the player's classified rejoin position after a hypothetical pit "
                    "stop using pit loss, service duration, and field trajectories."
                ),
                temporal_scope="future_counterfactual",
                required_inputs=(
                    "pit_loss_projection",
                    "pit_stop_duration_projection",
                    "field.trajectory_projection",
                ),
                outputs=(
                    CapabilityOutputDescriptor(
                        output_id="rejoin_position",
                        description="projected classified position after the stop",
                        unit="position",
                    ),
                ),
                freshness_s=self._freshness_s,
                uncertainty=(
                    "Unavailable until pit and field-trajectory models exist; current position "
                    "alone is not a rejoin projection."
                ),
                available=False,
                unavailable_reason="missing_rejoin_projection_model",
            ),
        )

    def _descriptor(self, capability_id: str) -> CapabilityDescriptor:
        descriptor = next(
            (
                item
                for item in self.catalog()
                if item.capability_id == capability_id
            ),
            None,
        )
        if descriptor is None:
            raise RaceCapabilityError("unknown_race_capability")
        return descriptor

    def evidence_ids(self, request: CapabilityRequest) -> tuple[str, ...]:
        descriptor = self._descriptor(request.capability_id)
        return tuple(
            f"{request.request_id}:{output.output_id}" for output in descriptor.outputs
        )

    @staticmethod
    def _unknown_evidence(
        request: CapabilityRequest,
        descriptor: CapabilityDescriptor,
        context: RaceContext,
    ) -> tuple[EvidenceItem, ...]:
        frame = context.frame
        valid_until = frame.observed_at + timedelta(seconds=descriptor.freshness_s)
        return tuple(
            EvidenceItem(
                evidence_id=f"{request.request_id}:{output.output_id}",
                session_id=frame.session_id,
                source_sequence=frame.sequence,
                observed_at=frame.observed_at,
                kind="unknown",
                subject="capability",
                metric=output.output_id,
                confidence=0.0,
                source_fields=descriptor.required_inputs,
                valid_until=valid_until,
            )
            for output in descriptor.outputs
        )

    def _unavailable(
        self,
        request: CapabilityRequest,
        descriptor: CapabilityDescriptor,
        context: RaceContext,
        reason: str,
    ) -> CapabilityResult:
        return CapabilityResult(
            request_id=request.request_id,
            capability_id=request.capability_id,
            status="unavailable",
            evidence=self._unknown_evidence(request, descriptor, context),
            unavailable_reason=reason,
        )

    def _classification(
        self,
        request: CapabilityRequest,
        descriptor: CapabilityDescriptor,
        context: RaceContext,
    ) -> CapabilityResult:
        reason = self._classification_reason(context)
        if reason is not None:
            return self._unavailable(request, descriptor, context, reason)
        frame = context.frame
        player_position = frame.player.position
        assert player_position is not None
        positions = tuple(
            position
            for position in (
                player_position,
                *(opponent.position for opponent in frame.opponents),
            )
            if position is not None
        )
        last_position = max(positions)
        valid_until = frame.observed_at + timedelta(seconds=descriptor.freshness_s)

        def item(
            output_id: str,
            *,
            subject: str,
            metric: str,
            value: int | bool,
            unit: str | None,
            source_fields: tuple[str, ...],
        ) -> EvidenceItem:
            return EvidenceItem(
                evidence_id=f"{request.request_id}:{output_id}",
                session_id=frame.session_id,
                source_sequence=frame.sequence,
                observed_at=frame.observed_at,
                kind="derived",
                subject=subject,
                metric=metric,
                value=value,
                unit=unit,
                source_fields=source_fields,
                valid_until=valid_until,
            )

        return CapabilityResult(
            request_id=request.request_id,
            capability_id=request.capability_id,
            status="available",
            evidence=(
                item(
                    "player_position",
                    subject="player",
                    metric="position",
                    value=player_position,
                    unit="position",
                    source_fields=("player.position",),
                ),
                item(
                    "last_position",
                    subject="field",
                    metric="position",
                    value=last_position,
                    unit="position",
                    source_fields=("field.position",),
                ),
                item(
                    "classified_cars",
                    subject="field",
                    metric="classified_cars",
                    value=len(positions),
                    unit="cars",
                    source_fields=("field.position",),
                ),
                item(
                    "is_last",
                    subject="player",
                    metric="is_last",
                    value=player_position == last_position,
                    unit=None,
                    source_fields=("player.position", "field.position"),
                ),
            ),
        )

    def _fuel_range(
        self,
        request: CapabilityRequest,
        descriptor: CapabilityDescriptor,
        context: RaceContext,
    ) -> CapabilityResult:
        reason = self._fuel_range_reason(context)
        if reason is not None:
            return self._unavailable(request, descriptor, context, reason)
        frame = context.frame
        fuel_l = frame.player.fuel_l
        burn_l_per_lap = context.fuel_trend_l_per_lap
        assert fuel_l is not None and burn_l_per_lap is not None
        evidence = EvidenceItem(
            evidence_id=f"{request.request_id}:laps_remaining",
            session_id=frame.session_id,
            source_sequence=frame.sequence,
            observed_at=frame.observed_at,
            kind="derived",
            subject="player",
            metric="fuel_laps_remaining",
            value=fuel_l / burn_l_per_lap,
            unit="laps",
            source_fields=("player.fuel_l", "context.fuel_trend_l_per_lap"),
            valid_until=frame.observed_at + timedelta(seconds=descriptor.freshness_s),
        )
        return CapabilityResult(
            request_id=request.request_id,
            capability_id=request.capability_id,
            status="available",
            evidence=(evidence,),
        )

    def _derived_item(
        self,
        request: CapabilityRequest,
        context: RaceContext,
        output_id: str,
        *,
        subject: str,
        metric: str,
        value: str | int | float | bool,
        unit: str | None,
        source_fields: tuple[str, ...],
        confidence: float = 1.0,
    ) -> EvidenceItem:
        frame = context.frame
        return EvidenceItem(
            evidence_id=f"{request.request_id}:{output_id}",
            session_id=frame.session_id,
            source_sequence=frame.sequence,
            observed_at=frame.observed_at,
            kind="derived",
            subject=subject,
            metric=metric,
            value=value,
            unit=unit,
            confidence=confidence,
            source_fields=source_fields,
            valid_until=frame.observed_at + timedelta(seconds=self._freshness_s),
        )

    def _position_change(
        self,
        request: CapabilityRequest,
        descriptor: CapabilityDescriptor,
        context: RaceContext,
    ) -> CapabilityResult:
        reason = self._position_change_reason(context)
        if reason is not None:
            return self._unavailable(request, descriptor, context, reason)
        delta = self._position_delta(f"{request.request_id}-position-delta")
        value = delta.value
        assert isinstance(value, (int, float)) and not isinstance(value, bool)
        direction = "gained" if value < 0 else "lost" if value > 0 else "unchanged"
        return CapabilityResult(
            request_id=request.request_id,
            capability_id=request.capability_id,
            status="available",
            evidence=(
                self._derived_item(
                    request,
                    context,
                    "direction",
                    subject="player",
                    metric="position_change_direction",
                    value=direction,
                    unit=None,
                    source_fields=("player.position[30s]",),
                    confidence=delta.confidence,
                ),
                self._derived_item(
                    request,
                    context,
                    "positions_changed",
                    subject="player",
                    metric="positions_changed",
                    value=abs(value),
                    unit="positions",
                    source_fields=("player.position[30s]",),
                    confidence=delta.confidence,
                ),
            ),
        )

    def _gap(
        self,
        request: CapabilityRequest,
        descriptor: CapabilityDescriptor,
        context: RaceContext,
        direction: str,
    ) -> CapabilityResult:
        reason = self._gap_reason(context, direction)
        if reason is not None:
            return self._unavailable(request, descriptor, context, reason)
        opponent = self._nearest_opponent(context, direction)
        assert opponent is not None and opponent.gap_to_player_s is not None
        value = abs(opponent.gap_to_player_s)
        return CapabilityResult(
            request_id=request.request_id,
            capability_id=request.capability_id,
            status="available",
            evidence=(
                self._derived_item(
                    request,
                    context,
                    "gap_s",
                    subject=f"car_{direction}",
                    metric="gap_s",
                    value=value,
                    unit="s",
                    source_fields=(
                        f"opponent:{opponent.driver_id}.gap_to_player_s",
                    ),
                ),
            ),
        )

    def _relative_pace(
        self,
        request: CapabilityRequest,
        descriptor: CapabilityDescriptor,
        context: RaceContext,
        direction: str,
    ) -> CapabilityResult:
        reason = self._relative_pace_reason(context, direction)
        if reason is not None:
            return self._unavailable(request, descriptor, context, reason)
        trend = self._gap_trend(direction, f"{request.request_id}-gap-trend")
        rate = trend.value
        assert isinstance(rate, (int, float)) and not isinstance(rate, bool)
        if rate < -_MIN_CLOSING_RATE_S_PER_S:
            state = "closing"
        elif rate > _MIN_CLOSING_RATE_S_PER_S:
            state = "losing_ground" if direction == "ahead" else "falling_back"
        else:
            state = "stable"
        subject = "player_vs_car_ahead" if direction == "ahead" else "car_behind"
        current_opponent = self._nearest_opponent(context, direction)
        assert current_opponent is not None
        source_fields = (
            f"opponent:{current_opponent.driver_id}.gap_to_player_s[10s]",
        )
        return CapabilityResult(
            request_id=request.request_id,
            capability_id=request.capability_id,
            status="available",
            evidence=(
                self._derived_item(
                    request,
                    context,
                    "state",
                    subject=subject,
                    metric="relative_pace_state",
                    value=state,
                    unit=None,
                    source_fields=source_fields,
                    confidence=trend.confidence,
                ),
                self._derived_item(
                    request,
                    context,
                    "gap_change_rate",
                    subject=subject,
                    metric="gap_change_rate",
                    value=abs(rate),
                    unit="s/s",
                    source_fields=source_fields,
                    confidence=trend.confidence,
                ),
            ),
        )

    def _catch_time(
        self,
        request: CapabilityRequest,
        descriptor: CapabilityDescriptor,
        context: RaceContext,
        direction: str,
    ) -> CapabilityResult:
        reason = self._catch_time_reason(context, direction)
        if reason is not None:
            return self._unavailable(request, descriptor, context, reason)
        opponent = self._nearest_opponent(context, direction)
        assert opponent is not None and opponent.gap_to_player_s is not None
        gap = abs(opponent.gap_to_player_s)
        trend = self._gap_trend(direction, f"{request.request_id}-catch-trend")
        rate = trend.value
        assert isinstance(rate, (int, float)) and not isinstance(rate, bool)
        return CapabilityResult(
            request_id=request.request_id,
            capability_id=request.capability_id,
            status="available",
            evidence=(
                self._derived_item(
                    request,
                    context,
                    "catch_time_s",
                    subject="car_ahead" if direction == "ahead" else "car_behind",
                    metric="catch_time_s",
                    value=gap / -rate,
                    unit="s",
                    source_fields=(
                        f"opponent:{opponent.driver_id}.gap_to_player_s",
                        f"opponent:{opponent.driver_id}.gap_to_player_s[10s]",
                    ),
                    confidence=trend.confidence,
                ),
            ),
        )

    def execute(self, request: CapabilityRequest) -> CapabilityResult:
        if request.arguments:
            raise RaceCapabilityError("capability_arguments_not_supported")
        descriptor = self._descriptor(request.capability_id)
        current = self._memory.latest_context()
        if request.capability_id == _CLASSIFICATION:
            return self._classification(request, descriptor, current)
        if request.capability_id == _FUEL_RANGE:
            return self._fuel_range(request, descriptor, current)
        if request.capability_id == _POSITION_CHANGE:
            return self._position_change(request, descriptor, current)
        if request.capability_id == _GAP_AHEAD:
            return self._gap(request, descriptor, current, "ahead")
        if request.capability_id == _GAP_BEHIND:
            return self._gap(request, descriptor, current, "behind")
        if request.capability_id == _RELATIVE_PACE_AHEAD:
            return self._relative_pace(request, descriptor, current, "ahead")
        if request.capability_id == _RELATIVE_PACE_BEHIND:
            return self._relative_pace(request, descriptor, current, "behind")
        if request.capability_id == _CATCH_TIME_AHEAD:
            return self._catch_time(request, descriptor, current, "ahead")
        if request.capability_id == _CATCH_TIME_BEHIND:
            return self._catch_time(request, descriptor, current, "behind")
        if request.capability_id in {_PIT_LOSS, _PIT_STOP_DURATION, _PROJECTED_REJOIN}:
            assert descriptor.unavailable_reason is not None
            return self._unavailable(
                request,
                descriptor,
                current,
                descriptor.unavailable_reason,
            )
        raise RaceCapabilityError("unknown_race_capability")
