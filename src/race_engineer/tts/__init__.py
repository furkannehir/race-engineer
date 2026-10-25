"""Local text-to-speech adapters and playback scheduling."""

from race_engineer.config import TtsConfig
from race_engineer.tts.chatterbox import ChatterboxAnnouncementEngine, ChatterboxSpeechBackend
from race_engineer.tts.queue import SpeechPlaybackQueue


def tts_factory(config: TtsConfig) -> ChatterboxAnnouncementEngine:
    if not config.enabled:
        raise ValueError("text-to-speech is disabled")
    return ChatterboxAnnouncementEngine(ChatterboxSpeechBackend(config))


__all__ = [
    "ChatterboxAnnouncementEngine",
    "ChatterboxSpeechBackend",
    "SpeechPlaybackQueue",
    "tts_factory",
]
