"""Deterministic construction of policy-facing race context."""

from collections import deque
from collections.abc import Sequence

from race_engineer.config import PolicyContextConfig
from race_engineer.core.contracts import RaceContext, RaceEvent, TelemetryFrame


class DefaultRaceContextBuilder:
    """Accumulates bounded event history and small, replay-safe race-state features."""

    def __init__(self, config: PolicyContextConfig | None = None) -> None:
        self._config = config or PolicyContextConfig()
        self._session_id: str | None = None
        self._events: list[RaceEvent] = []
        self._previous_frame: TelemetryFrame | None = None
        self._stint_start_lap: int | None = None
        self._fuel_lap: int | None = None
        self._fuel_at_lap_start_l: float | None = None
        self._last_fuel_l: float | None = None
        self._fuel_burn_samples: deque[float] = deque(maxlen=self._config.fuel_trend_laps)

    def _reset(self, session_id: str) -> None:
        self._session_id = session_id
        self._events.clear()
        self._previous_frame = None
        self._stint_start_lap = None
        self._fuel_lap = None
        self._fuel_at_lap_start_l = None
        self._last_fuel_l = None
        self._fuel_burn_samples.clear()

    def _update_stint(self, frame: TelemetryFrame) -> int | None:
        lap = frame.player.lap_number
        if lap is None:
            return None

        previous = self._previous_frame
        left_pit_lane = (
            previous is not None and previous.player.in_pit_lane and not frame.player.in_pit_lane
        )
        if (
            left_pit_lane
            or (self._stint_start_lap is None and not frame.player.in_pit_lane)
            or (self._stint_start_lap is not None and lap < self._stint_start_lap)
        ):
            self._stint_start_lap = lap

        if self._stint_start_lap is None:
            return None
        return lap - self._stint_start_lap + 1

    def _update_fuel_trend(self, frame: TelemetryFrame) -> float | None:
        lap = frame.player.lap_number
        fuel_l = frame.player.fuel_l
        if lap is None or fuel_l is None:
            self._last_fuel_l = fuel_l
            return self._fuel_trend()

        refueled = (
            self._last_fuel_l is not None
            and fuel_l >= self._last_fuel_l + self._config.fuel_increase_reset_l
        )
        if (
            refueled
            or self._fuel_lap is None
            or self._fuel_at_lap_start_l is None
            or lap < self._fuel_lap
        ):
            self._fuel_burn_samples.clear()
            self._fuel_lap = lap
            self._fuel_at_lap_start_l = fuel_l
        elif lap > self._fuel_lap:
            completed_laps = lap - self._fuel_lap
            fuel_used_l = self._fuel_at_lap_start_l - fuel_l
            if fuel_used_l > 0:
                self._fuel_burn_samples.append(fuel_used_l / completed_laps)
            self._fuel_lap = lap
            self._fuel_at_lap_start_l = fuel_l

        self._last_fuel_l = fuel_l
        return self._fuel_trend()

    def _fuel_trend(self) -> float | None:
        if not self._fuel_burn_samples:
            return None
        return sum(self._fuel_burn_samples) / len(self._fuel_burn_samples)

    def _battle_context(
        self, frame: TelemetryFrame
    ) -> tuple[float | None, float | None, str | None]:
        gaps = tuple(
            opponent.gap_to_player_s
            for opponent in frame.opponents
            if opponent.gap_to_player_s is not None
        )
        if not gaps:
            return None, None, None

        ahead = min((-gap for gap in gaps if gap < 0), default=None)
        behind = min((gap for gap in gaps if gap > 0), default=None)
        if any(gap == 0 for gap in gaps):
            return ahead, behind, "contested"

        threshold = self._config.battle_gap_s
        close_ahead = ahead is not None and ahead <= threshold
        close_behind = behind is not None and behind <= threshold
        if close_ahead and close_behind:
            battle_state = "sandwiched"
        elif close_ahead:
            battle_state = "attacking"
        elif close_behind:
            battle_state = "defending"
        else:
            battle_state = "clear"
        return ahead, behind, battle_state

    def update(
        self,
        frame: TelemetryFrame,
        events: Sequence[RaceEvent],
    ) -> RaceContext:
        if frame.session_id != self._session_id:
            self._reset(frame.session_id)
        if any(event.session_id != frame.session_id for event in events):
            raise ValueError("policy context events must match the telemetry session")

        self._events.extend(events)
        self._events = [
            event
            for event in self._events
            if event.expires_at is None or event.expires_at > frame.observed_at
        ][-self._config.event_history_limit :]

        stint_lap = self._update_stint(frame)
        fuel_trend = self._update_fuel_trend(frame)
        gap_ahead_s, gap_behind_s, battle_state = self._battle_context(frame)
        context = RaceContext(
            frame=frame,
            recent_events=tuple(self._events),
            stint_lap=stint_lap,
            fuel_trend_l_per_lap=fuel_trend,
            gap_ahead_s=gap_ahead_s,
            gap_behind_s=gap_behind_s,
            battle_state=battle_state,
        )
        self._previous_frame = frame
        return context
