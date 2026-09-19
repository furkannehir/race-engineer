"""Local text-to-speech adapters and playback scheduling."""

import sys

from race_engineer.config import TtsConfig
from race_engineer.core.interfaces import TextToSpeechEngine
from race_engineer.tts.queue import SpeechPlaybackQueue
from race_engineer.tts.windows_sapi import WindowsSapiTextToSpeechEngine


def tts_factory(config: TtsConfig) -> TextToSpeechEngine:
    if not config.enabled:
        raise ValueError("text-to-speech is disabled")
    if config.adapter == "windows-sapi":
        if sys.platform != "win32":
            raise RuntimeError("the windows-sapi adapter requires Windows")
        return WindowsSapiTextToSpeechEngine(config)
    raise AssertionError(f"unsupported text-to-speech adapter: {config.adapter}")


__all__ = [
    "SpeechPlaybackQueue",
    "WindowsSapiTextToSpeechEngine",
    "tts_factory",
]
