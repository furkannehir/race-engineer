"""Replaceable component boundaries for the race-engineer pipeline."""

from collections.abc import AsyncIterator, Callable, Sequence
from typing import Protocol

from race_engineer.core.contracts import (
    PlaybackResult,
    RaceContext,
    RaceEvent,
    SpeechIntent,
    TelemetryFrame,
    Utterance,
)
from race_engineer.core.conversation import ConversationPlan, ConversationReply, ConversationRequest
from race_engineer.core.intelligence import (
    ContextPacket,
    DriverTurn,
    EngineerBrief,
    EvidenceItem,
    EvidenceQuery,
    GeneratedResponse,
    GroundedResponse,
)
from race_engineer.core.speech_input import AudioClip, Transcription
from race_engineer.core.speech_output import SpeechOutputResult


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


class ConversationPlanner(Protocol):
    async def plan(self, request: ConversationRequest) -> ConversationPlan: ...


class ContextEngineer(Protocol):
    async def analyze(self, turn: DriverTurn) -> ContextPacket: ...

    async def refresh(
        self,
        packet: ContextPacket,
        evidence_ids: Sequence[str],
    ) -> ContextPacket: ...


class TelemetryMemory(Protocol):
    def update(self, context: RaceContext) -> None: ...

    def query(self, request: EvidenceQuery) -> EvidenceItem: ...

    def query_many(self, requests: Sequence[EvidenceQuery]) -> tuple[EvidenceItem, ...]: ...


class CoreEngineer(Protocol):
    async def decide(self, turn: DriverTurn, context: ContextPacket) -> EngineerBrief: ...


class EngineerResponseGenerator(Protocol):
    async def generate(
        self,
        turn: DriverTurn,
        context: ContextPacket,
        brief: EngineerBrief,
    ) -> GeneratedResponse: ...


class EvidenceGrounder(Protocol):
    def ground(
        self,
        response: GeneratedResponse,
        context: ContextPacket,
        brief: EngineerBrief,
    ) -> GroundedResponse: ...


class SpeechRecognizer(Protocol):
    async def start(self) -> None: ...

    async def transcribe(self, audio: AudioClip) -> Transcription: ...

    async def aclose(self) -> None: ...


class ConversationSpeaker(Protocol):
    async def start(self) -> None: ...

    async def speak(
        self,
        reply: ConversationReply,
        *,
        play_audio: bool = True,
        before_playback: Callable[[], bool] | None = None,
    ) -> SpeechOutputResult: ...

    async def aclose(self) -> None: ...


class LiveTelemetryBridge(Protocol):
    @property
    def available(self) -> bool: ...

    def availability_changed(self, available: bool) -> None: ...

    def update(self, context: RaceContext) -> None: ...

    async def submit(self, intent: SpeechIntent, utterance: Utterance) -> bool: ...
