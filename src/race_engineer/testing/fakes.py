"""Small deterministic implementations for contract and orchestration tests."""

from collections.abc import AsyncIterator, Sequence

from race_engineer.core.contracts import (
    PlaybackResult,
    RaceContext,
    RaceEvent,
    SpeechIntent,
    TelemetryFrame,
    Utterance,
)
from race_engineer.core.enums import (
    InterruptionPolicy,
    MessageCategory,
    PlaybackStatus,
    Tone,
    Urgency,
)


class SequenceTelemetryAdapter:
    def __init__(self, frames: Sequence[TelemetryFrame]) -> None:
        self._frames = tuple(frames)

    async def stream(self) -> AsyncIterator[TelemetryFrame]:
        for frame in self._frames:
            yield frame


class ScriptedEventDeriver:
    def __init__(self, events_by_sequence: dict[int, Sequence[RaceEvent]]) -> None:
        self._events_by_sequence = {
            sequence: tuple(events) for sequence, events in events_by_sequence.items()
        }

    def derive(
        self,
        previous: TelemetryFrame | None,
        current: TelemetryFrame,
    ) -> Sequence[RaceEvent]:
        del previous
        return self._events_by_sequence.get(current.sequence, ())


class InMemoryContextBuilder:
    def __init__(self, history_limit: int = 32) -> None:
        self._history_limit = history_limit
        self._events: list[RaceEvent] = []

    def update(
        self,
        frame: TelemetryFrame,
        events: Sequence[RaceEvent],
    ) -> RaceContext:
        self._events.extend(events)
        self._events = self._events[-self._history_limit :]
        return RaceContext(frame=frame, recent_events=tuple(self._events))


class EchoPolicy:
    """Turns only newly sourced fixture events into intents; not a production policy."""

    async def decide(self, context: RaceContext) -> list[SpeechIntent]:
        intents: list[SpeechIntent] = []
        for event in context.recent_events:
            if event.source_sequence != context.frame.sequence:
                continue
            message = event.facts.get("message")
            if not isinstance(message, str):
                continue
            intents.append(
                SpeechIntent(
                    intent_id=f"{event.event_id}:speech",
                    category=MessageCategory.SAFETY,
                    facts={"message": message},
                    priority={
                        Urgency.ROUTINE: 30,
                        Urgency.IMPORTANT: 70,
                        Urgency.CRITICAL: 100,
                    }[event.urgency],
                    tone=Tone.URGENT if event.urgency is Urgency.CRITICAL else Tone.NEUTRAL,
                    max_words=20,
                    deadline=event.expires_at or event.occurred_at,
                    interruption_policy=(
                        InterruptionPolicy.INTERRUPT_ANY
                        if event.urgency is Urgency.CRITICAL
                        else InterruptionPolicy.NEVER
                    ),
                )
            )
        return intents


class TemplateLanguageGenerator:
    async def generate(self, intent: SpeechIntent) -> Utterance:
        message = intent.facts.get("message")
        if not isinstance(message, str):
            raise ValueError("the deterministic generator requires a string 'message' fact")
        words = message.split()
        text = " ".join(words[: intent.max_words])
        return Utterance(
            intent_id=intent.intent_id,
            text=text,
            generator_metadata={"adapter": "deterministic-template"},
        )


class RecordingTextToSpeechEngine:
    def __init__(self) -> None:
        self.utterances: list[Utterance] = []
        self.cancelled = False

    async def speak(self, utterance: Utterance) -> PlaybackResult:
        self.utterances.append(utterance)
        return PlaybackResult(intent_id=utterance.intent_id, status=PlaybackStatus.COMPLETED)

    async def cancel(self) -> None:
        self.cancelled = True
