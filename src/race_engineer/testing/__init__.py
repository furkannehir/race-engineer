"""Deterministic test doubles that exercise the public adapter boundaries."""

from race_engineer.testing.fakes import (
    EchoPolicy,
    InMemoryContextBuilder,
    RecordingTextToSpeechEngine,
    ScriptedEventDeriver,
    SequenceTelemetryAdapter,
    TemplateLanguageGenerator,
)

__all__ = [
    "EchoPolicy",
    "InMemoryContextBuilder",
    "RecordingTextToSpeechEngine",
    "ScriptedEventDeriver",
    "SequenceTelemetryAdapter",
    "TemplateLanguageGenerator",
]
