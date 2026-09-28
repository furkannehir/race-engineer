"""Small original radio cues for push-to-talk interaction boundaries."""

import asyncio
import importlib
import importlib.resources
import io
import logging
import math
import sys
import wave
from array import array
from collections.abc import Callable
from functools import cache
from typing import Any, Literal

RadioCue = Literal["open", "close"]

_LOGGER = logging.getLogger(__name__)
_SAMPLE_RATE_HZ = 48_000
_CUE_ASSETS: dict[RadioCue, str] = {
    "open": "radio-open.wav",
    "close": "radio-close.wav",
}


def _sounddevice() -> Any:
    return importlib.import_module("sounddevice")


@cache
def _load_cue_pcm(cue: RadioCue) -> bytes:
    try:
        asset_name = _CUE_ASSETS[cue]
    except KeyError as error:
        raise ValueError("unknown radio cue") from error
    asset = importlib.resources.files("race_engineer").joinpath(
        "assets", "audio", asset_name
    )
    with wave.open(io.BytesIO(asset.read_bytes()), "rb") as stream:
        if (
            stream.getnchannels() != 1
            or stream.getsampwidth() != 2
            or stream.getframerate() != _SAMPLE_RATE_HZ
            or stream.getcomptype() != "NONE"
        ):
            raise ValueError(f"unsupported packaged radio cue format: {asset_name}")
        return stream.readframes(stream.getnframes())


def _scale_pcm(pcm: bytes, volume: float) -> bytes:
    samples = array("h")
    samples.frombytes(pcm)
    if sys.byteorder != "little":
        samples.byteswap()
    peak = max((abs(sample) for sample in samples), default=1)
    scale = 32767 * volume / max(1, peak)
    scaled = array(
        "h",
        (
            max(-32768, min(32767, round(sample * scale)))
            for sample in samples
        ),
    )
    if sys.byteorder != "little":
        scaled.byteswap()
    return scaled.tobytes()


def render_radio_cue(cue: RadioCue, volume: float) -> bytes:
    """Load and volume-scale an original packaged mono PCM16 radio cue."""
    if not 0 <= volume <= 1 or not math.isfinite(volume):
        raise ValueError("radio cue volume must be between zero and one")
    return _scale_pcm(_load_cue_pcm(cue), volume)


class RadioCuePlayer:
    """Best-effort cue playback; failures never block the live engineer."""

    def __init__(
        self,
        *,
        output_device: int | None = None,
        volume: float = 0.15,
        muted: Callable[[], bool] = lambda: False,
        enabled: bool = True,
    ) -> None:
        if not 0 <= volume <= 1 or not math.isfinite(volume):
            raise ValueError("radio cue volume must be between zero and one")
        self._output_device = output_device
        self._volume = volume
        self._muted = muted
        self._enabled = enabled
        self._available = True
        self._lock = asyncio.Lock()

    def _play_blocking(self, cue: RadioCue) -> None:
        pcm = render_radio_cue(cue, self._volume)
        sd = _sounddevice()
        with sd.RawOutputStream(
            samplerate=_SAMPLE_RATE_HZ,
            channels=1,
            dtype="int16",
            device=self._output_device,
        ) as stream:
            stream.write(pcm)

    async def play(self, cue: RadioCue) -> bool:
        if not self._enabled or not self._available or self._muted():
            return False
        async with self._lock:
            if not self._available or self._muted():
                return False
            try:
                await asyncio.to_thread(self._play_blocking, cue)
            except Exception as error:
                self._available = False
                _LOGGER.warning(
                    "radio cue playback unavailable; push-to-talk remains active",
                    extra={
                        "event": "radio_cue_unavailable",
                        "reason": type(error).__name__,
                    },
                )
                return False
        return True
