"""Replaceable component boundaries for the race-engineer pipeline."""

from collections.abc import AsyncIterator, Sequence
from typing import Protocol

from race_engineer.core.contracts import (
    PlaybackResult,
    RaceContext,
    RaceEvent,
    SpeechIntent,
    TelemetryFrame,
    Utterance,
)


class TelemetryAdapter(Protocol):
    def stream(self) -> AsyncIterator[TelemetryFrame]: ...


class EventDeriver(Protocol):
    def derive(
        self,
        previous: TelemetryFrame | None,
        current: TelemetryFrame,
    ) -> Sequence[RaceEvent]: ...


class RaceContextBuilder(Protocol):
    def update(
        self,
        frame: TelemetryFrame,
        events: Sequence[RaceEvent],
    ) -> RaceContext: ...


class PolicyStrategy(Protocol):
    async def decide(self, context: RaceContext) -> list[SpeechIntent]: ...


class LanguageGenerator(Protocol):
    async def generate(self, intent: SpeechIntent) -> Utterance: ...


class TextToSpeechEngine(Protocol):
    async def speak(self, utterance: Utterance) -> PlaybackResult: ...

    async def cancel(self) -> None: ...
