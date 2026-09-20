"""Bounded PCM/WAV handling and an energy gate (not a speech/noise classifier)."""

import math
import sys
import wave
from array import array
from pathlib import Path

from race_engineer.config import SttConfig
from race_engineer.core.speech_input import AudioClip, SpeechInputError


def samples(pcm16: bytes) -> array[int]:
    values = array("h", pcm16)
    if sys.byteorder != "little":
        values.byteswap()
    return values


def silence_reason(audio: AudioClip, config: SttConfig) -> str | None:
    if audio.duration_s > config.max_capture_s:
        raise SpeechInputError("audio_too_long")
    if audio.duration_s < config.min_capture_s:
        return "audio_too_short"
    values = samples(audio.pcm16)
    block_size = audio.sample_rate_hz // 50  # 20ms windows; remove DC offset per window.
    threshold = 32768 * 10 ** (config.silence_threshold_dbfs / 20)
    active_samples = 0
    for start in range(0, len(values), block_size):
        block = values[start : start + block_size]
        mean = sum(block) / len(block)
        rms = math.sqrt(sum((value - mean) ** 2 for value in block) / len(block))
        if rms >= threshold:
            active_samples += len(block)
    if active_samples / audio.sample_rate_hz < 0.12:
        return "audio_below_threshold"
    return None


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
