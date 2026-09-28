"""Bounded PCM/WAV handling and an energy gate (not a speech/noise classifier)."""

import math
import sys
import wave
from array import array
from pathlib import Path
from typing import Literal, cast

from race_engineer.config import SttConfig
from race_engineer.core.speech_input import AudioClip, AudioDiagnostic, SpeechInputError


def samples(pcm16: bytes) -> array[int]:
    values = array("h", pcm16)
    if sys.byteorder != "little":
        values.byteswap()
    return values


def _dbfs(amplitude: float) -> float:
    if amplitude <= 0:
        return -96.0
    return max(-96.0, min(0.0, 20 * math.log10(amplitude / 32768)))


def analyze_audio(audio: AudioClip, config: SttConfig) -> AudioDiagnostic:
    """Measure a clip without retaining audio or interpreting its spoken content."""

    if audio.duration_s > config.max_capture_s:
        raise SpeechInputError("audio_too_long")
    values = samples(audio.pcm16)
    block_size = audio.sample_rate_hz // 50  # 20ms windows; remove DC offset per window.
    threshold = 32768 * 10 ** (config.silence_threshold_dbfs / 20)
    blocks: list[tuple[int, bool]] = []
    active_samples = 0
    squared_total = 0.0
    peak = 0.0
    for start in range(0, len(values), block_size):
        block = values[start : start + block_size]
        mean = sum(block) / len(block)
        centered = [value - mean for value in block]
        squared = sum(value**2 for value in centered)
        rms = math.sqrt(squared / len(block))
        active = rms >= threshold
        blocks.append((len(block), active))
        squared_total += squared
        peak = max(peak, max(abs(value) for value in centered))
        if active:
            active_samples += len(block)

    active_indexes = [index for index, (_, active) in enumerate(blocks) if active]
    if active_indexes:
        first_active, last_active = active_indexes[0], active_indexes[-1]
        leading_samples = sum(length for length, _ in blocks[:first_active])
        trailing_samples = sum(length for length, _ in blocks[last_active + 1 :])
    else:
        leading_samples = trailing_samples = len(values)

    result: Literal["accepted", "audio_too_short", "audio_below_threshold"]
    if audio.duration_s < config.min_capture_s:
        result = "audio_too_short"
    elif active_samples / audio.sample_rate_hz < 0.12:
        result = "audio_below_threshold"
    else:
        result = "accepted"

    duration_s = audio.duration_s
    return AudioDiagnostic(
        duration_s=round(duration_s, 4),
        sample_rate_hz=cast(Literal[16000, 44100, 48000], audio.sample_rate_hz),
        rms_dbfs=round(_dbfs(math.sqrt(squared_total / len(values))), 2),
        peak_dbfs=round(_dbfs(peak), 2),
        active_duration_s=round(active_samples / audio.sample_rate_hz, 4),
        active_ratio=round(active_samples / len(values), 4),
        leading_silence_s=round(leading_samples / audio.sample_rate_hz, 4),
        trailing_silence_s=round(trailing_samples / audio.sample_rate_hz, 4),
        activity_at_start=bool(blocks and blocks[0][1]),
        activity_at_end=bool(blocks and blocks[-1][1]),
        gate_result=result,
    )


def silence_reason(audio: AudioClip, config: SttConfig) -> str | None:
    result = analyze_audio(audio, config).gate_result
    return None if result == "accepted" else result


def load_wav(path: Path, *, max_duration_s: float = 15) -> AudioClip:
    try:
        with wave.open(str(path), "rb") as stream:
            rate, channels = stream.getframerate(), stream.getnchannels()
            if stream.getsampwidth() != 2 or channels not in {1, 2}:
                raise SpeechInputError("wav_requires_pcm16_mono_or_stereo")
            if rate not in {16000, 44100, 48000}:
                raise SpeechInputError("wav_unsupported_sample_rate")
            if stream.getnframes() / rate > max_duration_s:
                raise SpeechInputError("audio_too_long")
            raw = stream.readframes(stream.getnframes())
            if len(raw) != stream.getnframes() * channels * 2:
                raise SpeechInputError("wav_truncated")
    except (OSError, EOFError, wave.Error) as error:
        raise SpeechInputError("wav_unreadable") from error
    if channels == 2:
        stereo = samples(raw)
        mono = array(
            "h", (round((a + b) / 2) for a, b in zip(stereo[::2], stereo[1::2], strict=True))
        )
        if sys.byteorder != "little":
            mono.byteswap()
        raw = mono.tobytes()
    try:
        return AudioClip(raw, rate)
    except ValueError as error:
        raise SpeechInputError("wav_invalid_audio") from error
