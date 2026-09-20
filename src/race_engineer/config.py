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
    slow_read_warning_s: float = Field(default=0.05, gt=0, le=10.0, allow_inf_nan=False)
    sample_gap_warning_s: float = Field(default=0.25, gt=0, le=60.0, allow_inf_nan=False)


class TelemetryConfig(ConfigModel):
    iracing: IracingTelemetryConfig = IracingTelemetryConfig()


class PolicyContextConfig(ConfigModel):
    event_history_limit: int = Field(default=32, ge=1, le=1_000)
    fuel_trend_laps: int = Field(default=5, ge=1, le=20)
    fuel_increase_reset_l: float = Field(default=0.25, gt=0, le=100, allow_inf_nan=False)
    battle_gap_s: float = Field(default=1.5, gt=0, le=30, allow_inf_nan=False)


class StrictPolicyConfig(ConfigModel):
    max_intents_per_frame: int = Field(default=1, ge=1, le=10)
    routine_cooldown_s: float = Field(default=15.0, ge=0, le=600, allow_inf_nan=False)
    important_cooldown_s: float = Field(default=5.0, ge=0, le=600, allow_inf_nan=False)
    critical_cooldown_s: float = Field(default=0.0, ge=0, le=600, allow_inf_nan=False)
    announce_position_changes: bool = True
    announce_pit_transitions: bool = False
    max_words: int = Field(default=20, ge=1, le=50)
    language: str = Field(default="en", min_length=2, max_length=35)


class PolicyConfig(ConfigModel):
    context: PolicyContextConfig = PolicyContextConfig()
    strict: StrictPolicyConfig = StrictPolicyConfig()


class LanguageConfig(ConfigModel):
    adapter: Literal["deterministic"] = "deterministic"


class TtsConfig(ConfigModel):
    enabled: bool = True
    adapter: Literal["windows-sapi"] = "windows-sapi"
    voice: str | None = Field(default=None, min_length=1, max_length=200)
    rate: int = Field(default=0, ge=-10, le=10)
    volume: int = Field(default=100, ge=0, le=100)
    playback_timeout_s: float = Field(default=10.0, gt=0, le=120, allow_inf_nan=False)
    queue_capacity: int = Field(default=8, ge=1, le=100)


class ConversationConfig(ConfigModel):
    adapter: Literal["llama-cpp"] = "llama-cpp"
    model: Literal["Qwen3-4B-Instruct-2507"] = "Qwen3-4B-Instruct-2507"
    # The adapter connects only to literal 127.0.0.1, never a remote host or proxy.
    port: int = Field(default=8087, ge=1, le=65535)
    timeout_s: float = Field(default=30.0, gt=0, le=120, allow_inf_nan=False)
    history_turns: int = Field(default=6, ge=0, le=12)
    max_snapshot_age_s: float = Field(default=3.0, gt=0, le=30, allow_inf_nan=False)
    default_language: Literal["en", "tr"] = "en"


class SttConfig(ConfigModel):
    adapter: Literal["qwen3-asr"] = "qwen3-asr"
    python_path: Path = Path("data/stt-prototype/runtime/Scripts/python.exe")
    model_path: Path = Path("data/stt-prototype/Qwen3-ASR-0.6B-hf")
    device: Literal["cpu", "cuda"] = "cpu"
    threads: int = Field(default=8, ge=1, le=32)
    language: Literal["auto", "en", "tr"] = "auto"
    startup_timeout_s: float = Field(default=120, gt=0, le=600, allow_inf_nan=False)
    timeout_s: float = Field(default=45, gt=0, le=180, allow_inf_nan=False)
    input_device: int | None = Field(default=None, ge=0)
    sample_rate_hz: Literal[16000, 44100, 48000] = 16000
    ptt_key: str = Field(default="F8", pattern=r"^(F([1-9]|1[0-9]|2[0-4])|SPACE|RCTRL|RALT)$")
    max_capture_s: float = Field(default=15, ge=1, le=30, allow_inf_nan=False)
    min_capture_s: float = Field(default=0.2, ge=0.05, le=1, allow_inf_nan=False)
    silence_threshold_dbfs: float = Field(default=-42, ge=-80, le=-10, allow_inf_nan=False)


class RadioTtsConfig(ConfigModel):
    """Conversational speech only; automatic race calls retain their SAPI settings."""

    enabled: bool = True
    adapter: Literal["piper"] = "piper"
    python_path: Path = Path("data/tts-prototype/runtime/Scripts/python.exe")
    english_model_path: Path = Path("data/tts-prototype/voices/en_US-ljspeech-high.onnx")
    turkish_model_path: Path = Path("data/tts-prototype/voices/tr_TR-dfki-medium.onnx")
    output_device: int | None = Field(default=None, ge=0)
    volume: float = Field(default=0.8, ge=0, le=1, allow_inf_nan=False)
    length_scale: float = Field(default=1.0, ge=0.5, le=2, allow_inf_nan=False)
    threads: int = Field(default=4, ge=1, le=16)
    startup_timeout_s: float = Field(default=60, gt=0, le=180, allow_inf_nan=False)
    playback_timeout_s: float = Field(default=90, gt=0, le=180, allow_inf_nan=False)


class AppConfig(ConfigModel):
    config_version: Literal["app-config.v1"] = "app-config.v1"
    paths: PathsConfig = PathsConfig()
    runtime: RuntimeConfig = RuntimeConfig()
    logging: LoggingConfig = LoggingConfig()
    privacy: PrivacyConfig = PrivacyConfig()
    telemetry: TelemetryConfig = TelemetryConfig()
    policy: PolicyConfig = PolicyConfig()
    language: LanguageConfig = LanguageConfig()
    tts: TtsConfig = TtsConfig()
    conversation: ConversationConfig = ConversationConfig()
    stt: SttConfig = SttConfig()
    radio_tts: RadioTtsConfig = RadioTtsConfig()


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
