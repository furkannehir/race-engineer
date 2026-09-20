"""Vendor-independent outcomes for bounded conversational audio playback."""

from typing import Literal

from pydantic import Field

from race_engineer.core.contracts import ContractModel
from race_engineer.core.conversation import RadioLanguage


class SpeechOutputResult(ContractModel):
    schema_version: Literal["speech-output.v1"] = "speech-output.v1"
    language: RadioLanguage
    played: bool
    audio_duration_s: float = Field(gt=0, le=60, allow_inf_nan=False)
    synthesis_ms: float = Field(ge=0, allow_inf_nan=False)
    playback_ms: float = Field(ge=0, allow_inf_nan=False)


class SpeechOutputError(Exception):
    """Contains only adapter-controlled reason codes, never text or worker output."""
