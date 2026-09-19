"""Thin pyirsdk wrapper that owns shared-memory access and session-metadata filtering."""

import importlib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from time import perf_counter
from typing import Any, Protocol, cast

from race_engineer.telemetry.iracing.raw import IracingDriverMetadata, IracingRawSample

Clock = Callable[[], datetime]
MonotonicClock = Callable[[], float]

SELECTED_VARIABLES = (
    "CarDistAhead",
    "CarDistBehind",
    "CarIdxF2Time",
    "CarIdxLapCompleted",
    "CarIdxPosition",
    "FuelLevel",
    "IsReplayPlaying",
    "Lap",
    "OnPitRoad",
    "PlayerCarIdx",
    "PlayerCarPosition",
    "SessionFlags",
    "SessionNum",
    "SessionState",
    "SessionTick",
    "SessionTime",
    "SessionUniqueID",
    "Speed",
)


class IracingSdkUnavailableError(RuntimeError):
    """Raised when the optional Windows SDK binding cannot be imported."""


@dataclass(frozen=True, slots=True)
class IracingReadMetrics:
    buffer_wait_ms: float
    variable_read_ms: float
    metadata_refresh_ms: float
    total_ms: float
    metadata_refreshed: bool


@dataclass(frozen=True, slots=True)
class IracingReadResult:
    sample: IracingRawSample
    metrics: IracingReadMetrics


class IracingSource(Protocol):
    def connect(self) -> bool: ...

    @property
    def connected(self) -> bool: ...

    def read(self) -> IracingReadResult: ...

    def close(self) -> None: ...


def _optional_int(value: object) -> int | None:
    return int(value) if isinstance(value, int) and not isinstance(value, bool) else None


def _optional_float(value: object) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def _optional_bool(value: object) -> bool | None:
    return value if isinstance(value, bool) else None


def _int_tuple(value: object) -> tuple[int, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return ()
    return tuple(
        int(item) if isinstance(item, int) and not isinstance(item, bool) else -1 for item in value
    )


def _float_tuple(value: object) -> tuple[float, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return ()
    return tuple(
        float(item) if isinstance(item, (int, float)) and not isinstance(item, bool) else -1.0
        for item in value
    )


class PyIrSdkSource:
    """Reads one internally consistent sample from iRacing's Windows memory map."""

    def __init__(
        self,
        clock: Clock | None = None,
        monotonic: MonotonicClock = perf_counter,
    ) -> None:
        try:
            module = importlib.import_module("irsdk")
        except ImportError as error:
            raise IracingSdkUnavailableError(
                "pyirsdk is required for live iRacing telemetry; install the project dependencies"
            ) from error
        self._sdk: Any = module.IRSDK()
        self._clock = clock or (lambda: datetime.now(UTC))
        self._monotonic = monotonic
        self._metadata_update: int | None = None
        self._drivers: tuple[IracingDriverMetadata, ...] = ()
        self._session_types: dict[int, str] = {}

    def connect(self) -> bool:
        return bool(self._sdk.startup())

    @property
    def connected(self) -> bool:
        return bool(self._sdk.is_connected)

    def _read_value(self, key: str, available: frozenset[str]) -> object:
        return self._sdk[key] if key in available else None

    def _refresh_metadata(self) -> bool:
        update = _optional_int(self._sdk.session_info_update)
        if update is not None and update == self._metadata_update:
            return False

        drivers: list[IracingDriverMetadata] = []
        driver_info = self._sdk["DriverInfo"]
        if isinstance(driver_info, Mapping):
            raw_drivers = driver_info.get("Drivers")
            if isinstance(raw_drivers, Sequence):
                for raw_driver in raw_drivers:
                    if not isinstance(raw_driver, Mapping):
                        continue
                    car_idx = _optional_int(raw_driver.get("CarIdx"))
                    if car_idx is None or car_idx < 0:
                        continue
                    user_id = _optional_int(raw_driver.get("UserID"))
                    drivers.append(
                        IracingDriverMetadata(
                            car_idx=car_idx,
                            user_id=user_id if user_id is not None and user_id >= 0 else None,
                            is_pace_car=bool(raw_driver.get("CarIsPaceCar", 0)),
                        )
                    )

        session_types: dict[int, str] = {}
        session_info = self._sdk["SessionInfo"]
        if isinstance(session_info, Mapping):
            raw_sessions = session_info.get("Sessions")
            if isinstance(raw_sessions, Sequence):
                for raw_session in raw_sessions:
                    if not isinstance(raw_session, Mapping):
                        continue
                    session_num = _optional_int(raw_session.get("SessionNum"))
                    session_type = raw_session.get("SessionType")
                    if session_num is not None and isinstance(session_type, str):
                        session_types[session_num] = session_type

        self._drivers = tuple(sorted(drivers, key=lambda driver: driver.car_idx))
        self._session_types = session_types
        self._metadata_update = update
        return True

    def read(self) -> IracingReadResult:
        total_started = self._monotonic()
        buffer_wait_started = total_started
        self._sdk.freeze_var_buffer_latest()
        buffer_ready = self._monotonic()
        try:
            available = frozenset(str(name) for name in (self._sdk.var_headers_names or ()))
            values = {name: self._read_value(name, available) for name in SELECTED_VARIABLES}
        finally:
            self._sdk.unfreeze_var_buffer_latest()
        variables_read = self._monotonic()
        observed_at = self._clock()

        metadata_started = self._monotonic()
        metadata_refreshed = self._refresh_metadata()
        metadata_finished = self._monotonic()
        session_num = _optional_int(values["SessionNum"])
        sample = IracingRawSample(
            observed_at=observed_at,
            available_variables=tuple(sorted(available.intersection(SELECTED_VARIABLES))),
            session_unique_id=_optional_int(values["SessionUniqueID"]),
            session_num=session_num,
            session_tick=_optional_int(values["SessionTick"]),
            session_time_s=_optional_float(values["SessionTime"]),
            session_state=_optional_int(values["SessionState"]),
            session_flags=_optional_int(values["SessionFlags"]),
            session_type=self._session_types.get(session_num) if session_num is not None else None,
            is_replay_playing=_optional_bool(values["IsReplayPlaying"]),
            player_car_idx=_optional_int(values["PlayerCarIdx"]),
            lap=_optional_int(values["Lap"]),
            player_position=_optional_int(values["PlayerCarPosition"]),
            speed_mps=_optional_float(values["Speed"]),
            fuel_l=_optional_float(values["FuelLevel"]),
            on_pit_road=_optional_bool(values["OnPitRoad"]),
            car_ahead_distance_m=_optional_float(values["CarDistAhead"]),
            car_behind_distance_m=_optional_float(values["CarDistBehind"]),
            car_idx_positions=_int_tuple(values["CarIdxPosition"]),
            car_idx_laps_completed=_int_tuple(values["CarIdxLapCompleted"]),
            car_idx_f2_time_s=_float_tuple(values["CarIdxF2Time"]),
            drivers=self._drivers,
        )
        total_finished = self._monotonic()
        return IracingReadResult(
            sample=sample,
            metrics=IracingReadMetrics(
                buffer_wait_ms=(buffer_ready - buffer_wait_started) * 1000,
                variable_read_ms=(variables_read - buffer_ready) * 1000,
                metadata_refresh_ms=(metadata_finished - metadata_started) * 1000,
                total_ms=(total_finished - total_started) * 1000,
                metadata_refreshed=metadata_refreshed,
            ),
        )

    def close(self) -> None:
        self._sdk.shutdown()


def source_factory() -> IracingSource:
    return cast(IracingSource, PyIrSdkSource())
