"""Pure conversion from selected iRacing fields to stable domain contracts."""

import math

from race_engineer.core.contracts import OpponentState, PlayerState, TelemetryFrame
from race_engineer.core.enums import RaceFlag, SessionPhase
from race_engineer.telemetry.iracing.raw import IracingDriverMetadata, IracingRawSample

FLAG_CHECKERED = 0x0001
FLAG_WHITE = 0x0002
FLAG_GREEN = 0x0004
FLAG_YELLOW = 0x0008
FLAG_RED = 0x0010
FLAG_BLUE = 0x0020
FLAG_YELLOW_WAVING = 0x0100
FLAG_CAUTION = 0x4000
FLAG_CAUTION_WAVING = 0x8000
FLAG_BLACK = 0x010000

SESSION_PARADE_LAPS = 3
SESSION_RACING = 4
SESSION_CHECKERED = 5
SESSION_COOL_DOWN = 6

CAPABILITY_VARIABLES: dict[str, frozenset[str]] = {
    "flags": frozenset({"SessionFlags"}),
    "fuel": frozenset({"FuelLevel"}),
    "gaps": frozenset({"CarIdxF2Time", "CarIdxLapCompleted", "CarIdxPosition", "PlayerCarIdx"}),
    "lap": frozenset({"Lap"}),
    "opponents": frozenset({"CarIdxPosition", "PlayerCarIdx"}),
    "pit": frozenset({"OnPitRoad"}),
    "position": frozenset({"PlayerCarPosition"}),
    "replay": frozenset({"IsReplayPlaying"}),
    "session": frozenset({"SessionNum", "SessionState", "SessionUniqueID"}),
    "speed": frozenset({"Speed"}),
    "traffic_distance": frozenset({"CarDistAhead", "CarDistBehind"}),
}


def _valid_non_negative(value: float | None) -> float | None:
    if value is None or not math.isfinite(value) or value < 0:
        return None
    return value


def _valid_positive_int(value: int | None) -> int | None:
    return value if value is not None and value > 0 else None


def _valid_lap(value: int | None) -> int | None:
    return value if value is not None and value >= 0 else None


def _at[T](values: tuple[T, ...], index: int) -> T | None:
    return values[index] if 0 <= index < len(values) else None


def _driver_id(driver: IracingDriverMetadata | None, car_idx: int) -> str:
    if driver is not None and driver.user_id is not None and driver.user_id > 0:
        return f"iracing:user:{driver.user_id}"
    return f"iracing:car:{car_idx}"


def _flags(bitfield: int | None) -> tuple[RaceFlag, ...]:
    if bitfield is None:
        return ()
    mapping = (
        (FLAG_CHECKERED, RaceFlag.CHECKERED),
        (FLAG_WHITE, RaceFlag.WHITE),
        (FLAG_GREEN, RaceFlag.GREEN),
        (FLAG_YELLOW | FLAG_YELLOW_WAVING | FLAG_CAUTION | FLAG_CAUTION_WAVING, RaceFlag.YELLOW),
        (FLAG_RED, RaceFlag.RED),
        (FLAG_BLUE, RaceFlag.BLUE),
        (FLAG_BLACK, RaceFlag.BLACK),
    )
    return tuple(flag for mask, flag in mapping if bitfield & mask)


def _session_phase(state: int | None, flags: tuple[RaceFlag, ...]) -> SessionPhase:
    if RaceFlag.CHECKERED in flags or state in {SESSION_CHECKERED, SESSION_COOL_DOWN}:
        return SessionPhase.CHECKERED
    if state == SESSION_PARADE_LAPS:
        return SessionPhase.FORMATION
    if state == SESSION_RACING:
        return SessionPhase.CAUTION if RaceFlag.YELLOW in flags else SessionPhase.GREEN
    return SessionPhase.UNKNOWN


def _same_lap_gap(
    sample: IracingRawSample,
    player_idx: int,
    opponent_idx: int,
) -> float | None:
    if (sample.session_type or "").casefold() != "race":
        return None
    player_lap = _at(sample.car_idx_laps_completed, player_idx)
    opponent_lap = _at(sample.car_idx_laps_completed, opponent_idx)
    player_f2 = _at(sample.car_idx_f2_time_s, player_idx)
    opponent_f2 = _at(sample.car_idx_f2_time_s, opponent_idx)
    if (
        player_lap is None
        or opponent_lap != player_lap
        or player_f2 is None
        or opponent_f2 is None
        or player_f2 < 0
        or opponent_f2 < 0
        or not math.isfinite(player_f2)
        or not math.isfinite(opponent_f2)
    ):
        return None
    return opponent_f2 - player_f2


def _opponents(sample: IracingRawSample, player_idx: int) -> tuple[OpponentState, ...]:
    drivers = {driver.car_idx: driver for driver in sample.drivers}
    opponents: list[OpponentState] = []
    for car_idx, position in enumerate(sample.car_idx_positions):
        driver = drivers.get(car_idx)
        if car_idx == player_idx or position <= 0 or (driver is not None and driver.is_pace_car):
            continue
        opponents.append(
            OpponentState(
                driver_id=_driver_id(driver, car_idx),
                position=position,
                lap_number=_valid_lap(_at(sample.car_idx_laps_completed, car_idx)),
                gap_to_player_s=_same_lap_gap(sample, player_idx, car_idx),
            )
        )
    return tuple(
        sorted(opponents, key=lambda opponent: (opponent.position or 10_000, opponent.driver_id))
    )


def normalize_sample(sample: IracingRawSample) -> TelemetryFrame | None:
    """Return `None` until the source exposes the minimum identity and timing fields."""

    if (
        sample.session_unique_id is None
        or sample.session_num is None
        or sample.session_num < 0
        or sample.session_tick is None
        or sample.session_time_s is None
        or sample.session_time_s < 0
        or sample.player_car_idx is None
        or sample.player_car_idx < 0
    ):
        return None

    variables = frozenset(sample.available_variables)
    capabilities = tuple(
        sorted(
            capability
            for capability, required in CAPABILITY_VARIABLES.items()
            if required.issubset(variables)
        )
    )
    drivers = {driver.car_idx: driver for driver in sample.drivers}
    player_idx = sample.player_car_idx
    flags = _flags(sample.session_flags)
    session_id = f"iracing:{sample.session_unique_id}:{sample.session_num}"
    return TelemetryFrame(
        source="iracing",
        session_id=session_id,
        sequence=sample.session_tick,
        observed_at=sample.observed_at,
        session_time_s=sample.session_time_s,
        session_phase=_session_phase(sample.session_state, flags),
        player=PlayerState(
            driver_id=_driver_id(drivers.get(player_idx), player_idx),
            lap_number=_valid_lap(sample.lap),
            position=_valid_positive_int(sample.player_position),
            speed_mps=_valid_non_negative(sample.speed_mps),
            fuel_l=_valid_non_negative(sample.fuel_l),
            in_pit_lane=bool(sample.on_pit_road),
            car_ahead_distance_m=_valid_non_negative(sample.car_ahead_distance_m),
            car_behind_distance_m=_valid_non_negative(sample.car_behind_distance_m),
        ),
        opponents=_opponents(sample, player_idx),
        flags=flags,
        capabilities=capabilities,
        is_replay=bool(sample.is_replay_playing),
    )
