from pathlib import Path

import pytest
from pydantic import ValidationError

from race_engineer.config import load_config

ROOT = Path(__file__).parents[1]


def test_default_configuration_is_valid() -> None:
    config = load_config(ROOT / "config" / "default.toml", environ={})
    assert config.config_version == "app-config.v1"
    assert config.runtime.queue_capacity == 128
    assert config.privacy.record_raw_telemetry is False
    assert config.telemetry.iracing.sample_rate_hz == 10.0
    assert config.telemetry.iracing.include_replay is False
    assert config.telemetry.iracing.slow_read_warning_s == 0.05
    assert config.telemetry.iracing.sample_gap_warning_s == 0.25


def test_environment_overrides_are_explicit() -> None:
    config = load_config(
        ROOT / "config" / "default.toml",
        environ={
            "RACE_ENGINEER_LOG_LEVEL": "DEBUG",
            "RACE_ENGINEER_DATABASE_PATH": "custom/session.sqlite3",
            "RACE_ENGINEER_RECORD_RAW_TELEMETRY": "true",
            "UNRELATED_VALUE": "ignored",
        },
    )
    assert config.logging.level == "DEBUG"
    assert config.paths.database_path == Path("custom/session.sqlite3")
    assert config.privacy.record_raw_telemetry is True


def test_invalid_boolean_override_is_rejected() -> None:
    with pytest.raises(ValueError, match="expected a boolean"):
        load_config(
            ROOT / "config" / "default.toml",
            environ={"RACE_ENGINEER_RECORD_RAW_TELEMETRY": "perhaps"},
        )


def test_unknown_toml_key_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "invalid.toml"
    path.write_text('config_version = "app-config.v1"\nunknown = true\n', encoding="utf-8")
    with pytest.raises(ValidationError):
        load_config(path, environ={})
