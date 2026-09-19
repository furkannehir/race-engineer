"""Validated application configuration with explicit environment overrides."""

import os
import tomllib
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class ConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class PathsConfig(ConfigModel):
    data_dir: Path = Path("data")
    database_path: Path = Path("data/race_engineer.sqlite3")


class RuntimeConfig(ConfigModel):
    queue_capacity: int = Field(default=128, ge=1, le=100_000)
    component_timeout_s: float = Field(default=2.0, gt=0, allow_inf_nan=False)
    shutdown_timeout_s: float = Field(default=5.0, gt=0, allow_inf_nan=False)


class LoggingConfig(ConfigModel):
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    format: Literal["console", "json"] = "console"


class PrivacyConfig(ConfigModel):
    record_raw_telemetry: bool = False
    record_generator_prompts: bool = False
    record_generator_outputs: bool = False


class IracingTelemetryConfig(ConfigModel):
    sample_rate_hz: float = Field(default=10.0, ge=1.0, le=60.0, allow_inf_nan=False)
    reconnect_delay_s: float = Field(default=2.0, ge=0.1, le=60.0, allow_inf_nan=False)
    include_replay: bool = False


class TelemetryConfig(ConfigModel):
    iracing: IracingTelemetryConfig = IracingTelemetryConfig()


class AppConfig(ConfigModel):
    config_version: Literal["app-config.v1"] = "app-config.v1"
    paths: PathsConfig = PathsConfig()
    runtime: RuntimeConfig = RuntimeConfig()
    logging: LoggingConfig = LoggingConfig()
    privacy: PrivacyConfig = PrivacyConfig()
    telemetry: TelemetryConfig = TelemetryConfig()


def _parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"expected a boolean value, received {value!r}")


def _set_nested(data: dict[str, Any], section: str, key: str, value: object) -> None:
    nested = data.setdefault(section, {})
    if not isinstance(nested, dict):
        raise ValueError(f"configuration section {section!r} must be a table")
    nested[key] = value


def load_config(
    path: Path,
    environ: Mapping[str, str] | None = None,
) -> AppConfig:
    """Load TOML and apply the deliberately small public environment surface."""

    with path.open("rb") as stream:
        raw: dict[str, Any] = tomllib.load(stream)

    env = os.environ if environ is None else environ
    overrides: tuple[tuple[str, str, str, Callable[[str], object]], ...] = (
        ("RACE_ENGINEER_LOG_LEVEL", "logging", "level", str),
        ("RACE_ENGINEER_LOG_FORMAT", "logging", "format", str),
        ("RACE_ENGINEER_DATABASE_PATH", "paths", "database_path", Path),
        (
            "RACE_ENGINEER_RECORD_RAW_TELEMETRY",
            "privacy",
            "record_raw_telemetry",
            _parse_bool,
        ),
    )
    for environment_name, section, key, converter in overrides:
        if environment_name in env:
            converted = converter(env[environment_name])
            _set_nested(raw, section, key, converted)

    return AppConfig.model_validate(raw)
