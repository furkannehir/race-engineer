"""Native Ministral dialogue connected to the application's radio lifecycle."""

from race_engineer.config import ConversationConfig
from race_engineer.core.intelligence import DriverTurn, GroundedResponse
from race_engineer.intelligence.local_model import LocalIntelligenceError
from race_engineer.intelligence.ministral_dialogue import MinistralDialogue
from race_engineer.intelligence.native_dialogue import DialogueModel, Evidence
from race_engineer.intelligence.orchestrator import EngineerOrchestrator, PreparedEngineerResponse
from race_engineer.intelligence.telemetry_memory import BoundedTelemetryMemory
from race_engineer.radio_diagnostics import radio_event, radio_evidence, radio_stage

MINISTRAL_NATIVE_ID = "ministral-native-v1"


class MinistralEngineer(EngineerOrchestrator):
    """Select read-only tools, ground at delivery, remember only delivered speech.

    Construction is deliberately lazy: the panel starts before iRacing supplies
    a frame. The model lives in the existing owned llama.cpp server, not Python.
    No speech backend or audio queue is created here.
    """

    def __init__(
        self,
        config: ConversationConfig,
        memory: BoundedTelemetryMemory,
        *,
        model: DialogueModel | None = None,
    ) -> None:
        self._config = config
        self._memory = memory
        self._model = model
        self._engine: MinistralDialogue | None = None
        self._pending: PreparedEngineerResponse | None = None
        self._grounded: GroundedResponse | None = None
        self._spoken_evidence: Evidence = ()

    async def prepare(self, turn: DriverTurn) -> PreparedEngineerResponse:
        self._pending = None
        self._grounded = None
        self._spoken_evidence = ()
        if self._engine is None:
            self._engine = MinistralDialogue(self._config, self._memory, model=self._model)
        with radio_stage("native_dialogue"):
            await self._engine.respond(turn, commit=False)
        prepared = self._engine.last_prepared
        if prepared is None:
            raise LocalIntelligenceError("dialogue_response_not_prepared")
        self._pending = prepared
        radio_evidence("radio_context_ready", prepared.context)
        return prepared

    async def ground(self, prepared: PreparedEngineerResponse) -> GroundedResponse:
        if prepared is not self._pending or self._engine is None:
            raise LocalIntelligenceError("dialogue_prepared_response_superseded")
        engine = self._engine
        frame = self._memory.latest_context().frame
        if frame.session_id != prepared.turn.session_id:
            raise LocalIntelligenceError("dialogue_session_changed")
        if frame.sequence < prepared.context.source_sequence:
            raise LocalIntelligenceError("dialogue_telemetry_moved_backwards")
        with radio_stage("grounding"):
            _, fresh, unknowns = engine._tools(engine.pending_calls)
            if (
                bool(fresh) != bool(prepared.context.evidence)
                or unknowns != prepared.context.unknowns
            ):
                raise LocalIntelligenceError("dialogue_evidence_availability_changed")
            answer = engine._ground(
                prepared.turn,
                prepared.response.speech_template,
                prepared.context.evidence,
                unknowns,
                fresh_evidence=fresh,
            )
            self._spoken_evidence = fresh
            self._grounded = answer
            assert engine.last_prepared is not None
            radio_evidence("radio_evidence_refreshed", engine.last_prepared.context)
            return answer

    def delivered(self, prepared: PreparedEngineerResponse, grounded: GroundedResponse) -> None:
        """Called by the radio only after successful playback (or CLI delivery)."""
        if prepared is not self._pending or grounded is not self._grounded or self._engine is None:
            raise LocalIntelligenceError("dialogue_delivery_scope_mismatch")
        engine = self._engine
        messages = [dict(message) for message in engine.pending_messages]
        messages[-1] = {"role": "assistant", "content": grounded.text}
        engine.history.append(messages)
        engine.after_delivery(prepared.turn, engine.pending_calls, self._spoken_evidence, grounded)
        self._pending = None
        self._grounded = None
        self._spoken_evidence = ()
        radio_event("radio_native_dialogue_committed", source_sequence=grounded.source_sequence)
