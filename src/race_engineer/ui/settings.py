"""Explicit, local control-panel defaults; no inferred driver memory."""

import json
import os
import tempfile
from pathlib import Path
from typing import Literal

from pydantic import Field

from race_engineer.config import AppConfig, ConfigModel, PttBindingConfig
from race_engineer.stt.buttons import effective_binding, legacy_binding


class PanelSettings(ConfigModel):
    version: Literal[2] = 2
    ptt_binding: PttBindingConfig = PttBindingConfig()
    input_device: int | None = Field(default=None, ge=0)
    output_device: int | None = Field(default=None, ge=0)
    # Device identity prevents PortAudio index reordering silently choosing another device.
    input_name: str | None = None
    output_name: str | None = None
    reply_language: Literal["auto", "en", "tr"] = "auto"
    volume: int = Field(default=80, ge=0, le=100)
    announce_position_changes: bool = True
    announce_pit_transitions: bool = False

    @classmethod
    def from_config(cls, config: AppConfig) -> "PanelSettings":
        return cls(
            ptt_binding=effective_binding(config.stt),
            input_device=config.stt.input_device,
            output_device=config.radio_tts.output_device,
            volume=round(config.radio_tts.volume * 100),
            announce_position_changes=config.policy.strict.announce_position_changes,
            announce_pit_transitions=config.policy.strict.announce_pit_transitions,
        )

    def apply(self, config: AppConfig) -> AppConfig:
        raw = config.model_dump()
        raw["stt"].update(
            ptt_binding=self.ptt_binding.model_dump(), input_device=self.input_device
        )
        raw["radio_tts"].update(output_device=self.output_device, volume=self.volume / 100)
        raw["tts"]["volume"] = self.volume
        raw["policy"]["strict"].update(
            announce_position_changes=self.announce_position_changes,
            announce_pit_transitions=self.announce_pit_transitions,
        )
        return AppConfig.model_validate(raw)


def load_settings(path: Path, defaults: PanelSettings) -> PanelSettings:
    if not path.exists():
        return defaults
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict) and raw.get("version", 1) == 1:
        migrated = dict(raw)
        key = migrated.pop("ptt_key", "F8")
        migrated["version"] = 2
        migrated["ptt_binding"] = legacy_binding(str(key)).model_dump()
        raw = migrated
    return PanelSettings.model_validate(raw)


def save_settings(path: Path, settings: PanelSettings) -> None:
    """Atomic replacement; failed validation/writes leave the previous defaults intact."""
    settings = PanelSettings.model_validate(settings.model_dump())
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix="panel-", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(settings.model_dump_json(indent=2) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
