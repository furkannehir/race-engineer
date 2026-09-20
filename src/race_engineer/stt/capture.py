"""Windows push-to-talk and bounded, in-memory PortAudio capture."""

import asyncio
import ctypes
import importlib
import sys
import threading
import time
from collections.abc import Callable
from typing import Any

from race_engineer.config import SttConfig
from race_engineer.core.speech_input import AudioClip, SpeechInputError


def _sounddevice() -> Any:
    try:
        return importlib.import_module("sounddevice")
    except (ImportError, OSError) as error:
        raise SpeechInputError("microphone_dependency_missing") from error


def input_devices() -> list[dict[str, object]]:
    sd = _sounddevice()
    try:
        devices = sd.query_devices()
        hosts = sd.query_hostapis()
        default = sd.default.device[0]
        return [
            {
                "index": index,
                "name": device["name"],
                "host_api": hosts[device["hostapi"]]["name"],
                "input_channels": device["max_input_channels"],
                "default_sample_rate_hz": device["default_samplerate"],
                "default": index == default,
            }
            for index, device in enumerate(devices)
            if device["max_input_channels"] > 0
        ]
    except Exception as error:
        raise SpeechInputError("microphone_enumeration_failed") from error


def virtual_key(name: str) -> int:
    special = {"SPACE": 0x20, "RCTRL": 0xA3, "RALT": 0xA5}
    if name in special:
        return special[name]
    if name.startswith("F") and name[1:].isdigit() and 1 <= int(name[1:]) <= 24:
        return 0x70 + int(name[1:]) - 1
    raise ValueError("unsupported push-to-talk key")


def windows_key_reader() -> Callable[[int], bool]:
    if sys.platform != "win32":
        raise SpeechInputError("push_to_talk_requires_windows")
    user32 = ctypes.WinDLL("user32", use_last_error=True)
    user32.GetAsyncKeyState.argtypes = [ctypes.c_int]
    user32.GetAsyncKeyState.restype = ctypes.c_short
    # Read only the held-state bit, never the unreliable shared 'pressed since' bit.
    return lambda key: bool(user32.GetAsyncKeyState(key) & 0x8000)


class CaptureBuffer:
    def __init__(self, sample_rate_hz: int, max_duration_s: float) -> None:
        self._rate = sample_rate_hz
        self._limit = int(sample_rate_hz * max_duration_s) * 2
        self._lock = threading.Lock()
        self._data = bytearray()
        self._active = False
        self._error: str | None = None

    def begin(self) -> None:
        with self._lock:
            self._data.clear()
            self._error = None
            self._active = True

    def append(self, raw: bytes, *, overflow: bool = False) -> None:
        with self._lock:
            if not self._active or self._error:
                return
            if overflow or len(raw) % 2 or len(self._data) + len(raw) > self._limit:
                self._data.clear()
                self._error = "microphone_overflow" if overflow else "capture_too_long"
                return
            self._data.extend(raw)

    def discard(self) -> None:
        with self._lock:
            self._active = False
            self._data.clear()
            self._error = None

    def finish(self) -> AudioClip:
        with self._lock:
            self._active = False
            raw, error = bytes(self._data), self._error
            self._data.clear()
            self._error = None
        if error:
            raise SpeechInputError(error)
        if not raw:
            raise SpeechInputError("capture_empty")
        return AudioClip(raw, self._rate)


class PushToTalkMicrophone:
    def __init__(
        self,
        config: SttConfig,
        *,
        key_down: Callable[[int], bool] | None = None,
        stream_factory: Callable[..., Any] | None = None,
    ) -> None:
        self._config = config
        self._key = virtual_key(config.ptt_key)
        self._key_down = key_down or windows_key_reader()
        self._buffer = CaptureBuffer(config.sample_rate_hz, config.max_capture_s)
        self._stream: Any = None
        try:
            factory = stream_factory or _sounddevice().RawInputStream
            self._stream = factory(
                samplerate=config.sample_rate_hz,
                channels=1,
                dtype="int16",
                device=config.input_device,
                blocksize=config.sample_rate_hz // 50,
                callback=self._callback,
            )
        except SpeechInputError:
            raise
        except Exception as error:
            raise SpeechInputError("microphone_open_failed") from error

    def _callback(self, data: Any, frames: int, times: Any, status: Any) -> None:
        del frames, times
        self._buffer.append(bytes(data), overflow=bool(status))

    async def _operate(self, operation: Callable[[], Any]) -> None:
        # A cancelled to_thread await must not leave a late stream.start after cleanup.
        task = asyncio.create_task(asyncio.to_thread(operation))
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            await task
            raise

    async def next_clip(
        self,
        notify: Callable[[str], None] = print,
        *,
        before_capture: Callable[[], None] | None = None,
    ) -> AudioClip | None:
        # A key held during model inference must be released before a new recording.
        while self._key_down(self._key):
            if self._key_down(0x1B):
                return None
            await asyncio.sleep(0.01)
        while not self._key_down(self._key):
            if self._key_down(0x1B):
                return None
            await asyncio.sleep(0.01)
        if before_capture is not None:
            before_capture()
        self._buffer.begin()
        try:
            await self._operate(self._stream.start)
            notify("Listening... release the push-to-talk key to submit.")
            started = time.monotonic()
            while self._key_down(self._key):
                if self._key_down(0x1B):
                    self._buffer.discard()
                    return None
                if time.monotonic() - started >= self._config.max_capture_s:
                    self._buffer.discard()
                    raise SpeechInputError("capture_too_long")
                await asyncio.sleep(0.01)
        except (OSError, RuntimeError) as error:
            self._buffer.discard()
            raise SpeechInputError("microphone_capture_failed") from error
        finally:
            try:
                await self._operate(self._stream.stop)
            except Exception as error:
                self._buffer.discard()
                raise SpeechInputError("microphone_stop_failed") from error
        return self._buffer.finish()

    async def aclose(self) -> None:
        self._buffer.discard()
        if self._stream is not None:
            stream, self._stream = self._stream, None
            await asyncio.to_thread(stream.close)
