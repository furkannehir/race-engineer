"""Validated driver-profile and explicit-preference read models."""

from datetime import datetime
from typing import Literal

from pydantic import Field, model_validator

from race_engineer.core.contracts import ContractModel, UtcDatetime

PreferenceSetting = Literal[
    "announce_position_changes",
    "announce_pit_transitions",
    "reply_language",
]


class DriverProfile(ContractModel):
    schema_version: Literal["driver-profile.v1"] = "driver-profile.v1"
    profile_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
    display_name: str = Field(min_length=1, max_length=80, pattern=r"^[^\x00-\x1f]+$")
    created_at: UtcDatetime
    updated_at: UtcDatetime
    is_default: bool = False

    @model_validator(mode="after")
    def validate_timestamps(self) -> "DriverProfile":
        if self.updated_at < self.created_at:
            raise ValueError("updated_at must not precede created_at")
        return self


class CommunicationPreferences(ContractModel):
    schema_version: Literal["communication-preferences.v1"] = (
        "communication-preferences.v1"
    )
    profile_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
    announce_position_changes: bool = True
    announce_pit_transitions: bool = False
    reply_language: Literal["auto", "en", "tr"] = "auto"
    updated_at: UtcDatetime | None = None


class SessionHistorySummary(ContractModel):
    schema_version: Literal["session-history-summary.v1"] = "session-history-summary.v1"
    profile_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
    session_id: str = Field(min_length=1)
    first_decided_at: UtcDatetime
    last_decided_at: UtcDatetime
    decisions: int = Field(ge=0)
    approved: int = Field(ge=0)
    suppressed: int = Field(ge=0)
    radio_outcomes: int = Field(ge=0)
    radio_completed: int = Field(ge=0)
    radio_cancelled: int = Field(ge=0)
    radio_expired: int = Field(ge=0)
    radio_failed: int = Field(ge=0)


def validate_preference_value(setting: str, value: object) -> bool | str:
    if setting in {"announce_position_changes", "announce_pit_transitions"}:
        if type(value) is not bool:
            raise ValueError(f"{setting} requires a boolean")
        return value
    if setting == "reply_language":
        if value not in {"auto", "en", "tr"}:
            raise ValueError("reply_language must be auto, en, or tr")
        return str(value)
    raise ValueError("unsupported communication preference")


def default_preferences(
    profile_id: str, *, updated_at: datetime | None = None
) -> CommunicationPreferences:
    return CommunicationPreferences(profile_id=profile_id, updated_at=updated_at)
