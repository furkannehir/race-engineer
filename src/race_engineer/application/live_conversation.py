"""One live session: telemetry/policy, push-to-talk conversation, and coordinated radio."""

import asyncio
import logging
import time
from collections import deque
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

from race_engineer.application.control import LiveControl
from race_engineer.config import AppConfig, RadioTtsConfig, SttConfig, load_config
from race_engineer.conversation.factory import conversation_planner
from race_engineer.conversation.live import LiveRaceState, LiveTelemetryUnavailable
from race_engineer.conversation.session import ConversationSession
from race_engineer.core.contracts import PlaybackResult, RaceContext, SpeechIntent, Utterance
from race_engineer.core.conversation import ConversationReply, RadioLanguage
from race_engineer.core.enums import PlaybackStatus
from race_engineer.core.intelligence import DriverTurn
from race_engineer.core.interfaces import ConversationSpeaker, SpeechRecognizer
from race_engineer.core.speech_input import AudioClip, SpeechInputError
from race_engineer.intelligence.context_engine import ContextEngineerError
from race_engineer.intelligence.factory import live_intelligence
from race_engineer.intelligence.grounding import GroundingError
from race_engineer.intelligence.local_model import LocalIntelligenceError
from race_engineer.intelligence.orchestrator import (
    EngineerOrchestrator,
    IntelligenceBoundaryError,
    PreparedEngineerResponse,
)
from race_engineer.intelligence.telemetry_memory import (
    BoundedTelemetryMemory,
    TelemetryMemoryError,
)
from race_engineer.memory import DurableHistory
from race_engineer.observability import configure_logging
from race_engineer.stt.buttons import binding_label, legacy_binding
from race_engineer.stt.capture import PushToTalkMicrophone
from race_engineer.stt.qwen import QwenSpeechRecognizer
from race_engineer.stt.radio_cues import RadioCuePlayer
from race_engineer.tts import tts_factory
from race_engineer.tts.live_radio import LiveRadio
from race_engineer.tts.piper import PiperConversationSpeaker

_LOGGER = logging.getLogger(__name__)


class LiveBridge:
    def __init__(
        self,
        state: LiveRaceState,
        radio: LiveRadio,
        control: LiveControl | None = None,
        result_sink: Callable[[PlaybackResult], None] | None = None,
        telemetry_memory: BoundedTelemetryMemory | None = None,
    ) -> None:
        self.state = state
        self.radio = radio
        self.control = control
        self.result_sink = result_sink
        self.telemetry_memory = telemetry_memory
        self._reported_available = False

    def _report_availability(self) -> None:
        available = self.available
        if available != self._reported_available:
            self._reported_available = available
            if self.control:
                self.control.emit("telemetry", "ready" if available else "unavailable")

    @property
    def available(self) -> bool:
        return self.state.available

    def availability_changed(self, available: bool) -> None:
        if not available:
            epoch = self.state.epoch
            self.state.invalidate()
            if epoch != self.state.epoch:
                if self.telemetry_memory is not None:
                    self.telemetry_memory.clear()
                self.radio.reset(self.state.epoch)
                print(
                    "Live telemetry unavailable; old answers and pending calls discarded.",
                    flush=True,
                )
        self._report_availability()

    def update(self, context: RaceContext) -> None:
        available, epoch = self.available, self.state.epoch
        self.state.update(context)
        if self.telemetry_memory is not None:
            if self.state.epoch != epoch:
                self.telemetry_memory.clear()
            if self.state.available:
                try:
                    self.telemetry_memory.update(context)
                except TelemetryMemoryError as error:
                    self.telemetry_memory.clear()
                    _LOGGER.warning(
                        "intelligence telemetry memory reset; strict policy continues",
                        extra={
                            "event": "intelligence_memory_reset",
                            "reason": str(error),
                            "session_id": context.frame.session_id,
                            "source_sequence": context.frame.sequence,
                        },
                    )
        if self.state.epoch != epoch or not available:
            self.radio.reset(self.state.epoch)
            if self.available:
                print(
                    f"LIVE iRacing session {context.frame.session_id}; telemetry ready.", flush=True
                )
        self._report_availability()

    async def submit(self, intent: SpeechIntent, utterance: Utterance) -> bool:
        if not self.available:
            if self.result_sink is not None:
                try:
                    self.result_sink(
                        PlaybackResult(
                            intent_id=intent.intent_id,
                            status=PlaybackStatus.CANCELLED,
                            finished_at=datetime.now(UTC),
                            error_code="telemetry_unavailable",
                        )
                    )
                except Exception as error:
                    _LOGGER.warning(
                        "radio result sink failed; live telemetry will continue",
                        extra={
                            "event": "live_bridge_result_sink_failed",
                            "intent_id": intent.intent_id,
                            "reason": type(error).__name__,
                        },
                    )
            return False
        return await self.radio.submit(intent, utterance, self.state.epoch)

    async def watch_freshness(self) -> None:
        while True:
            if not self.available:
                self.availability_changed(False)
            await asyncio.sleep(0.1)


async def _capture(
    microphone: PushToTalkMicrophone,
    state: LiveRaceState,
    radio: LiveRadio,
    control: LiveControl | None = None,
    cues: RadioCuePlayer | None = None,
) -> tuple[AudioClip | None, int]:
    epoch = state.epoch

    def claim() -> None:
        nonlocal epoch
        if not state.available:
            raise SpeechInputError("live_telemetry_unavailable")
        radio.claim_capture()
        epoch = state.epoch
        if control and cues is None:
            control.emit("phase", "listening")

    async def opened() -> None:
        assert cues is not None
        await cues.play("open")
        if control:
            control.emit("phase", "listening")

    try:
        if cues is not None:
            return (
                await microphone.next_clip(
                    before_capture=claim,
                    after_stream_started=opened,
                    after_capture=lambda: cues.play("close"),
                ),
                epoch,
            )
        return await microphone.next_clip(before_capture=claim), epoch
    finally:
        # next_clip's finally has already stopped the physical microphone stream.
        radio.release_capture()


async def live_dialogue(
    config: AppConfig,
    stt: SttConfig,
    state: LiveRaceState,
    radio: LiveRadio,
    recognizer: SpeechRecognizer,
    speaker: ConversationSpeaker | None,
    reply_language: RadioLanguage | None,
    control: LiveControl | None = None,
    intelligence: EngineerOrchestrator | None = None,
    cues: RadioCuePlayer | None = None,
) -> None:
    from race_engineer.core.speech_output import SpeechOutputError

    session = (
        ConversationSession(
            conversation_planner(config.conversation),
            state.snapshot,
            config.conversation,
            generation=lambda: state.epoch,
        )
        if intelligence is None
        else None
    )
    recent_dialogue: deque[str] = deque(maxlen=config.conversation.history_entry_limit)
    turn_sequence = 0
    microphone: PushToTalkMicrophone | None = None
    try:
        print("Loading local speech models. Telemetry continues independently.", flush=True)
        await recognizer.start()
        if speaker is not None:
            try:
                await speaker.start()
            except SpeechOutputError as error:
                print(f"Conversational speech unavailable: {error}; text replies remain enabled.")
                speaker = None
                if control:
                    control.emit("notice", "Reply voice unavailable; check the voice installation.")
        microphone = (
            PushToTalkMicrophone(stt, exit_on_escape=False)
            if control
            else PushToTalkMicrophone(stt)
        )
        if control:
            control.emit("models", "ready" if speaker is not None else "no_reply_voice")
        print(
            f"Radio ready. Hold {binding_label(stt)} when telemetry is ready; "
            "ESC or Ctrl+C exits."
        )
        print("Critical calls interrupt speech/capture. Release the key and ask again afterwards.")
        while True:
            if control:
                control.emit("phase", "ready")
            try:
                capture = asyncio.create_task(_capture(microphone, state, radio, control, cues))
                try:
                    audio, epoch = await asyncio.shield(capture)
                except asyncio.CancelledError:
                    task = asyncio.current_task()
                    if task is not None and task.cancelling():
                        if not capture.done() and not capture.cancelling():
                            capture.cancel()
                        await asyncio.gather(capture, return_exceptions=True)
                        raise
                    print(
                        "Capture interrupted by a radio call or telemetry change. Please ask again."
                    )
                    continue
                if audio is None:
                    return
                started = time.perf_counter()
                if epoch != state.epoch or not state.available:
                    raise LiveTelemetryUnavailable()
                print("Transcribing...", flush=True)
                if control:
                    control.emit("phase", "transcribing")
                transcript = await recognizer.transcribe(audio)
                del audio
                if transcript.status != "transcribed":
                    print(f"No question submitted: {transcript.reason or transcript.status}.")
                    if control:
                        control.emit(
                            "notice", "No clear speech detected. Release PTT and try again."
                        )
                    continue
                if epoch != state.epoch or not state.available:
                    raise LiveTelemetryUnavailable()
                print(f"You ({transcript.language}): {transcript.text}", flush=True)
                if control:
                    control.emit("phase", "thinking")
                assert transcript.language is not None
                answer: ConversationReply | None = None
                prepared: PreparedEngineerResponse | None = None
                if intelligence is None:
                    assert session is not None
                    answer = await session.ask(
                        transcript.text,
                        reply_language=reply_language or transcript.language,
                    )
                else:
                    snapshot = state.snapshot()
                    frame = snapshot.context.frame
                    turn_sequence += 1
                    turn = DriverTurn(
                        turn_id=(
                            f"{frame.session_id}:{state.epoch}:{frame.sequence}:"
                            f"driver:{turn_sequence}"
                        ),
                        transcript=transcript.text,
                        received_at=datetime.now(UTC),
                        session_id=frame.session_id,
                        generation=state.epoch,
                        asr_language=transcript.language,
                        reply_language=reply_language or transcript.language,
                        recent_dialogue=tuple(recent_dialogue),
                    )
                    prepared = await intelligence.prepare(turn)
                if epoch != state.epoch or not state.available:
                    raise LiveTelemetryUnavailable()
                print(f"ASR + reply processing: {time.perf_counter() - started:.2f}s")

                async def deliver(
                    queued_answer: ConversationReply | None = answer,
                    queued_response: PreparedEngineerResponse | None = prepared,
                    answer_epoch: int = epoch,
                    question: str = transcript.text,
                ) -> None:
                    if queued_response is not None:
                        if state.epoch != answer_epoch or not state.available:
                            raise LiveTelemetryUnavailable("live_session_changed")
                        assert intelligence is not None
                        grounded = await intelligence.ground(queued_response)
                        if state.epoch != answer_epoch or not state.available:
                            raise LiveTelemetryUnavailable("live_session_changed")
                        if grounded.action == "silence":
                            recent_dialogue.extend((f"driver: {question}", "engineer: [silence]"))
                            return
                        assert grounded.text is not None
                        refreshed = ConversationReply(
                            language=grounded.language,
                            text=grounded.text,
                            status=(
                                "clarification" if grounded.action == "clarify" else "answered"
                            ),
                            session_id=grounded.session_id,
                            source_sequence=grounded.source_sequence,
                            mode="live",
                        )
                        recent_dialogue.extend(
                            (f"driver: {question}", f"engineer: {grounded.text}")
                        )
                    else:
                        assert queued_answer is not None
                        refreshed = state.refresh(queued_answer, answer_epoch)
                    observed_at = state.snapshot().context.frame.observed_at
                    print(f"Engineer ({refreshed.language}): {refreshed.text}", flush=True)
                    if speaker is not None:

                        def still_current() -> bool:
                            age = (datetime.now(UTC) - observed_at).total_seconds()
                            return (
                                state.available
                                and state.epoch == answer_epoch
                                and 0 <= age <= config.conversation.max_snapshot_age_s
                            )

                        await speaker.speak(refreshed, before_playback=still_current)

                outcome = await radio.answer(deliver, epoch, config.conversation.max_snapshot_age_s)
                if outcome != "completed":
                    print(f"Radio answer {outcome}; no audio retry. Release the key and ask again.")
                print(
                    f"Ready for another question. Hold {binding_label(stt)} to talk.",
                    flush=True,
                )
            except LiveTelemetryUnavailable:
                if session is not None:
                    session.reset()
                recent_dialogue.clear()
                print("Telemetry/session changed; that question was discarded. Please ask again.")
            except (
                ContextEngineerError,
                GroundingError,
                IntelligenceBoundaryError,
                LocalIntelligenceError,
                TelemetryMemoryError,
            ) as error:
                _LOGGER.warning(
                    "local intelligence failed; live telemetry continues",
                    extra={"event": "live_intelligence_failed", "reason": str(error)},
                )
                print("The local engineer couldn't process that safely. Please ask again.")
            except SpeechInputError as error:
                print(f"Speech input: {error}. Release the key before trying again.")
                if control:
                    control.emit("notice", f"Speech input: {error}. Release PTT and try again.")
    finally:
        if microphone is not None:
            await microphone.aclose()


async def voice_iracing(
    config_path: Path,
    *,
    limit: int = 0,
    output: Path | None = None,
    input_device: int | None = None,
    ptt_key: str | None = None,
    output_device: int | None = None,
    text_only: bool = False,
    reply_language: RadioLanguage | None = None,
    app_config: AppConfig | None = None,
    control: LiveControl | None = None,
    persist_history: bool = False,
    history_database_path: Path | None = None,
) -> int:
    # Existing recorder and policy pipeline are shared, not run in a second process.
    from race_engineer.cli import _read_iracing

    if limit < 0 or (output is not None and output.exists()):
        raise ValueError("invalid frame limit or recording directory already exists")
    config = app_config or load_config(config_path)
    if control is None:
        configure_logging(config.logging)
    input_settings = config.stt.model_dump()
    if input_device is not None:
        input_settings["input_device"] = input_device
    if ptt_key is not None:
        input_settings["ptt_key"] = ptt_key.upper()
        input_settings["ptt_binding"] = legacy_binding(ptt_key.upper()).model_dump()
    stt = SttConfig.model_validate(input_settings)
    output_settings = config.radio_tts.model_dump()
    if output_device is not None:
        output_settings["output_device"] = output_device
    radio_tts = RadioTtsConfig.model_validate(output_settings)
    cues = RadioCuePlayer(
        output_device=radio_tts.output_device,
        volume=radio_tts.volume,
        muted=(lambda: control.muted.is_set()) if control else (lambda: False),
        enabled=not text_only,
    )
    state = LiveRaceState(config.conversation.max_snapshot_age_s)
    telemetry_memory = BoundedTelemetryMemory()
    intelligence = live_intelligence(config.conversation, telemetry_memory)
    history = None
    if persist_history and config.history.enabled:
        history = DurableHistory.try_open(
            history_database_path or config.paths.database_path,
            retention_days=config.history.retention_days,
        )
    automatic = None
    if config.tts.enabled and not text_only:
        try:
            automatic = tts_factory(config.tts)
        except Exception as error:
            _LOGGER.warning(
                "automatic speech unavailable",
                extra={
                    "event": "live_automatic_speech_unavailable",
                    "reason": type(error).__name__,
                },
            )
    radio = LiveRadio(
        automatic,
        capacity=config.tts.queue_capacity,
        activity=lambda value: control.emit("speech", value) if control else None,
        result_sink=(history.record_playback_result if history is not None else None),
        engine_unavailable_reason=(
            "audio_disabled"
            if text_only or not config.tts.enabled
            else "engine_unavailable"
        ),
    )
    radio.set_muted(control.muted.is_set() if control else False)
    bridge = LiveBridge(
        state,
        radio,
        control,
        result_sink=(history.record_playback_result if history is not None else None),
        telemetry_memory=telemetry_memory,
    )
    recognizer = QwenSpeechRecognizer(stt)
    speaker = PiperConversationSpeaker(radio_tts) if radio_tts.enabled and not text_only else None
    tasks: list[asyncio.Task[object]] = []

    async def watch_controls() -> None:
        assert control is not None
        muted = control.muted.is_set()
        while not control.stop.is_set():
            current = control.muted.is_set()
            if current != muted:
                muted = current
                radio.set_muted(muted)
            await asyncio.sleep(0.05)

    try:
        print("LIVE mode. Waiting for iRacing. Do not also run read-iracing or voice-replay.")
        telemetry = asyncio.create_task(
            _read_iracing(
                config_path,
                limit,
                output,
                live=bridge,
                app_config=config,
                history=history,
            )
        )
        dialogue = asyncio.create_task(
            live_dialogue(
                config,
                stt,
                state,
                radio,
                recognizer,
                speaker,
                reply_language,
                control,
                intelligence,
                cues,
            )
        )
        watchdog = asyncio.create_task(bridge.watch_freshness())
        tasks.extend((telemetry, dialogue, watchdog))
        if control:
            tasks.append(asyncio.create_task(watch_controls()))
        done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            task.result()
        return 0
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        try:
            await radio.aclose()
        finally:
            try:
                await recognizer.aclose()
            finally:
                if speaker is not None:
                    await speaker.aclose()
