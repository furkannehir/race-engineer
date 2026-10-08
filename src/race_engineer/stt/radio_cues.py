"""Packaged, bounded radio cues for push-to-talk capture boundaries."""

import asyncio
import importlib
import importlib.resources
import io
import logging
import math
import sys
import threading
import wave
from array import array
from collections.abc import Callable
from functools import cache
from typing import Any, Literal

RadioCue = Literal["open", "close"]

_LOGGER = logging.getLogger(__name__)
_SAMPLE_RATE_HZ = 48_000
_CHUNK_BYTES = _SAMPLE_RATE_HZ // 50 * 2  # Check cancellation between 20 ms writes.
_CUE_ASSETS: dict[RadioCue, str] = {
    "open": "radio-open.wav",
    "close": "radio-close.wav",
}


def _sounddevice() -> Any:
    return importlib.import_module("sounddevice")


@cache
def _load_cue_pcm(cue: RadioCue) -> bytes:
    asset = importlib.resources.files("race_engineer.stt").joinpath(
        "assets", _CUE_ASSETS[cue]
    )
    with wave.open(io.BytesIO(asset.read_bytes()), "rb") as stream:
        if (
            stream.getnchannels() != 1
            or stream.getsampwidth() != 2
            or stream.getframerate() != _SAMPLE_RATE_HZ
            or stream.getcomptype() != "NONE"
            or not 0 < stream.getnframes() <= _SAMPLE_RATE_HZ
        ):
            raise ValueError("invalid packaged radio cue format or duration")
        pcm = stream.readframes(stream.getnframes())
        if len(pcm) != stream.getnframes() * 2:
            raise ValueError("truncated packaged radio cue")
        return pcm


def render_radio_cue(cue: RadioCue, volume: float = 1.0) -> bytes:
    """Read a packaged mono PCM16 cue, preserving its original gain."""
    if not math.isfinite(volume) or not 0 <= volume <= 1:
        raise ValueError("radio cue volume must be between zero and one")
    pcm = _load_cue_pcm(cue)
    values = array("h", pcm)
    if sys.byteorder != "little":
        values.byteswap()
    scaled = array("h", (round(value * volume) for value in values))
    if sys.byteorder != "little":
        scaled.byteswap()
    return scaled.tobytes()


class RadioCuePlayer:
    """Best-effort dedicated output; cancellation closes it before returning."""

    def __init__(
        self,
        *,
        output_device: int | None = None,
        volume: float = 0.8,
        muted: Callable[[], bool] = lambda: False,
        enabled: bool = True,
    ) -> None:
        if not math.isfinite(volume) or not 0 <= volume <= 1:
            raise ValueError("radio cue volume must be between zero and one")
        self._output_device = output_device
        self._volume = volume
        self._muted = muted
        self._enabled = enabled
        self._available = True
        self._lock = asyncio.Lock()

    def _play_blocking(self, cue: RadioCue, cancelled: threading.Event) -> bool:
        pcm = render_radio_cue(cue, self._volume)
        if cancelled.is_set() or self._muted():
            return False
        sd = _sounddevice()
        with sd.RawOutputStream(
            samplerate=_SAMPLE_RATE_HZ,
            channels=1,
            dtype="int16",
            device=self._output_device,
            blocksize=_SAMPLE_RATE_HZ // 50,
            latency="low",
        ) as stream:
            for offset in range(0, len(pcm), _CHUNK_BYTES):
                if cancelled.is_set() or self._muted():
                    stream.abort()
                    return False
                if stream.write(pcm[offset : offset + _CHUNK_BYTES]):
                    raise RuntimeError("radio_cue_output_underflow")
            if cancelled.is_set() or self._muted():
                stream.abort()
                return False
        return True

    def _report_failure(self, error: Exception) -> None:
        self._available = False
        _LOGGER.warning(
            "Radio cue playback unavailable; push-to-talk remains active.",
            extra={"event": "radio_cue_unavailable", "reason": type(error).__name__},
        )

    async def play(self, cue: RadioCue) -> bool:
        if not self._enabled or not self._available or not self._volume or self._muted():
            return False
        async with self._lock:
            if not self._available or self._muted():
                return False
            cancelled = threading.Event()
            worker = asyncio.create_task(asyncio.to_thread(self._play_blocking, cue, cancelled))
            try:
                return await asyncio.shield(worker)
            except asyncio.CancelledError:
                cancelled.set()
                # The radio cannot release capture while a cancelled worker still plays.
                # Further cancellation requests must not detach the output thread either.
                while not worker.done():
                    try:
                        await asyncio.shield(worker)
                    except asyncio.CancelledError:
                        continue
                    except Exception:
                        break
                try:
                    worker.result()
                except Exception as error:
                    self._report_failure(error)
                raise
            except Exception as error:
                self._report_failure(error)
                return False
