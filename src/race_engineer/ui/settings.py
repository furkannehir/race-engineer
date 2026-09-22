"""Explicit, local control-panel defaults; no inferred driver memory."""

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import Field

from race_engineer.config import AppConfig, ConfigModel, PttBindingConfig
from race_engineer.memory.models import CommunicationPreferences
from race_engineer.stt.buttons import effective_binding, legacy_binding


class PanelSettings(ConfigModel):
    version: Literal[3] = 3
    ptt_binding: PttBindingConfig = PttBindingConfig()
    input_device: int | None = Field(default=None, ge=0)
    output_device: int | None = Field(default=None, ge=0)
    # Device identity prevents PortAudio index reordering silently choosing another device.
    input_name: str | None = None
    output_name: str | None = None
    volume: int = Field(default=80, ge=0, le=100)

    @classmethod
    def from_config(cls, config: AppConfig) -> "PanelSettings":
        return cls(
            ptt_binding=effective_binding(config.stt),
            input_device=config.stt.input_device,
            output_device=config.radio_tts.output_device,
            volume=round(config.radio_tts.volume * 100),
        )

    def apply(
        self, config: AppConfig, preferences: CommunicationPreferences
    ) -> AppConfig:
        raw = config.model_dump()
        raw["stt"].update(
            ptt_binding=self.ptt_binding.model_dump(), input_device=self.input_device
        )
        raw["radio_tts"].update(output_device=self.output_device, volume=self.volume / 100)
        raw["tts"]["volume"] = self.volume
        raw["policy"]["strict"].update(
            announce_position_changes=preferences.announce_position_changes,
            announce_pit_transitions=preferences.announce_pit_transitions,
        )
        return AppConfig.model_validate(raw)


@dataclass(frozen=True)
class LoadedPanelSettings:
    settings: PanelSettings
    legacy_preferences: CommunicationPreferences | None = None


def load_settings_state(path: Path, defaults: PanelSettings) -> LoadedPanelSettings:
    if not path.exists():
        return LoadedPanelSettings(defaults)
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict) and raw.get("version", 1) == 1:
        migrated = dict(raw)
        key = migrated.pop("ptt_key", "F8")
        migrated["version"] = 2
        migrated["ptt_binding"] = legacy_binding(str(key)).model_dump()
        raw = migrated
    legacy: CommunicationPreferences | None = None
    if isinstance(raw, dict) and raw.get("version") == 2:
        migrated = dict(raw)
        legacy = CommunicationPreferences(
            profile_id="default",
            reply_language=migrated.pop("reply_language", "auto"),
            announce_position_changes=migrated.pop("announce_position_changes", True),
            announce_pit_transitions=migrated.pop("announce_pit_transitions", False),
        )
        migrated["version"] = 3
        raw = migrated
    return LoadedPanelSettings(PanelSettings.model_validate(raw), legacy)


def load_settings(path: Path, defaults: PanelSettings) -> PanelSettings:
    return load_settings_state(path, defaults).settings


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
