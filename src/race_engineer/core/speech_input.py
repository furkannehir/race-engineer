"""Local audio and transcription boundaries, independent of microphone/ASR libraries."""

from dataclasses import dataclass, field
from typing import Literal

from pydantic import Field, model_validator

from race_engineer.core.contracts import ContractModel


@dataclass(frozen=True)
class AudioClip:
    """Mono signed little-endian PCM16. Audio stays in memory and out of repr/logs."""

    pcm16: bytes = field(repr=False)
    sample_rate_hz: int = 16000

    def __post_init__(self) -> None:
        if self.sample_rate_hz not in {16000, 44100, 48000}:
            raise ValueError("unsupported sample rate")
        if not self.pcm16 or len(self.pcm16) % 2:
            raise ValueError("audio must contain complete PCM16 samples")
        if self.duration_s > 30:
            raise ValueError("audio exceeds the 30-second contract limit")

    @property
    def duration_s(self) -> float:
        return len(self.pcm16) / (2 * self.sample_rate_hz)


class Transcription(ContractModel):
    schema_version: Literal["transcription.v1"] = "transcription.v1"
    status: Literal["transcribed", "no_speech", "unsupported_language"]
    text: str = Field(default="", max_length=1000)
    language: Literal["en", "tr"] | None = None
    audio_duration_s: float = Field(ge=0, le=30, allow_inf_nan=False)
    elapsed_ms: float = Field(default=0, ge=0, allow_inf_nan=False)
    reason: str | None = None

    @model_validator(mode="after")
    def validate_text(self) -> "Transcription":
        if self.status == "transcribed":
            if not self.text.strip() or self.language is None:
                raise ValueError("transcribed speech needs text and a supported language")
        elif self.text or self.language is not None:
            raise ValueError("non-speech results must not carry text or language")
        return self


class SpeechInputError(Exception):
    """Stable reason code, never raw audio, transcript, or a library traceback."""
