"""Privacy-safe, replayable representation of the selected iRacing SDK fields."""

from pathlib import Path
from typing import Literal

from pydantic import Field

from race_engineer.core.contracts import ContractModel, UtcDatetime


class IracingDriverMetadata(ContractModel):
    car_idx: int = Field(ge=0)
    user_id: int | None = Field(default=None, ge=0)
    is_pace_car: bool = False


class IracingRawSample(ContractModel):
    schema_version: Literal["iracing-raw-sample.v1"] = "iracing-raw-sample.v1"
    observed_at: UtcDatetime
    available_variables: tuple[str, ...]
    session_unique_id: int | None = None
    session_num: int | None = None
    session_tick: int | None = Field(default=None, ge=0)
    session_time_s: float | None = Field(default=None, allow_inf_nan=False)
    session_state: int | None = None
    session_flags: int | None = Field(default=None, ge=0)
    session_type: str | None = None
    is_replay_playing: bool | None = None
    player_car_idx: int | None = None
    lap: int | None = None
    player_position: int | None = None
    speed_mps: float | None = Field(default=None, allow_inf_nan=False)
    fuel_l: float | None = Field(default=None, allow_inf_nan=False)
    on_pit_road: bool | None = None
    car_ahead_distance_m: float | None = Field(default=None, allow_inf_nan=False)
    car_behind_distance_m: float | None = Field(default=None, allow_inf_nan=False)
    car_idx_positions: tuple[int, ...] = ()
    car_idx_laps_completed: tuple[int, ...] = ()
    car_idx_f2_time_s: tuple[float, ...] = ()
    drivers: tuple[IracingDriverMetadata, ...] = ()


def load_raw_samples(path: Path) -> tuple[IracingRawSample, ...]:
    samples: list[IracingRawSample] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            samples.append(IracingRawSample.model_validate_json(line))
        except ValueError as error:
            raise ValueError(f"invalid {path.name}:{line_number}: {error}") from error
    return tuple(samples)
