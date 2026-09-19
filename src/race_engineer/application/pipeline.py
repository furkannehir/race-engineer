"""Minimal async composition root used to prove the M0 boundaries."""

from dataclasses import dataclass

from race_engineer.core.contracts import TelemetryFrame
from race_engineer.core.interfaces import (
    EventDeriver,
    LanguageGenerator,
    PolicyStrategy,
    RaceContextBuilder,
    TelemetryAdapter,
    TextToSpeechEngine,
)


@dataclass(frozen=True, slots=True)
class PipelineResult:
    frames: int = 0
    events: int = 0
    intents: int = 0
    utterances: int = 0
    playbacks: int = 0


class RaceEngineerApplication:
    def __init__(
        self,
        telemetry: TelemetryAdapter,
        event_deriver: EventDeriver,
        context_builder: RaceContextBuilder,
        policy: PolicyStrategy,
        language: LanguageGenerator,
        tts: TextToSpeechEngine,
    ) -> None:
        self._telemetry = telemetry
        self._event_deriver = event_deriver
        self._context_builder = context_builder
        self._policy = policy
        self._language = language
        self._tts = tts

    async def run(self) -> PipelineResult:
        frames = events_count = intents_count = utterances = playbacks = 0
        previous: TelemetryFrame | None = None

        async for frame in self._telemetry.stream():
            derived_events = tuple(self._event_deriver.derive(previous, frame))
            context = self._context_builder.update(frame, derived_events)
            intents = await self._policy.decide(context)

            frames += 1
            events_count += len(derived_events)
            intents_count += len(intents)
            for intent in intents:
                utterance = await self._language.generate(intent)
                if utterance.intent_id != intent.intent_id:
                    raise ValueError("generator returned an utterance for a different intent")
                if len(utterance.text.split()) > intent.max_words:
                    raise ValueError("generator exceeded the intent word limit")
                utterances += 1
                playback = await self._tts.speak(utterance)
                if playback.intent_id != intent.intent_id:
                    raise ValueError("TTS returned a result for a different intent")
                playbacks += 1
            previous = frame

        return PipelineResult(
            frames=frames,
            events=events_count,
            intents=intents_count,
            utterances=utterances,
            playbacks=playbacks,
        )
