"""Local push-to-talk conversation with optional bilingual audio replies."""

import time
from collections.abc import Callable
from pathlib import Path

from race_engineer.config import RadioTtsConfig, SttConfig, load_config
from race_engineer.conversation.factory import conversation_planner
from race_engineer.conversation.replay import ReplayRaceState
from race_engineer.conversation.session import ConversationSession
from race_engineer.core.conversation import ConversationReply, RadioLanguage
from race_engineer.core.interfaces import ConversationSpeaker, SpeechRecognizer
from race_engineer.core.speech_input import AudioClip, SpeechInputError, Transcription
from race_engineer.core.speech_output import SpeechOutputError
from race_engineer.observability import configure_logging
from race_engineer.stt.audio import load_wav
from race_engineer.stt.buttons import binding_label, legacy_binding
from race_engineer.stt.capture import PushToTalkMicrophone
from race_engineer.stt.qwen import QwenSpeechRecognizer
from race_engineer.tts.piper import PiperConversationSpeaker


async def spoken_turn(
    recognizer: SpeechRecognizer,
    session: ConversationSession,
    audio: AudioClip,
    reply_language: RadioLanguage | None = None,
    *,
    on_transcript: Callable[[Transcription], None] | None = None,
) -> tuple[Transcription, ConversationReply | None]:
    transcript = await recognizer.transcribe(audio)
    if transcript.status != "transcribed":
        return transcript, None
    if on_transcript is not None:
        on_transcript(transcript)
    reply = await session.ask(transcript.text, reply_language=reply_language or transcript.language)
    return transcript, reply


async def transcribe_wav(config_path: Path, audio_path: Path) -> int:
    config = load_config(config_path)
    audio = load_wav(audio_path, max_duration_s=config.stt.max_capture_s)
    recognizer = QwenSpeechRecognizer(config.stt)
    try:
        result = await recognizer.transcribe(audio)
        print(result.model_dump_json(indent=2))
        return 0 if result.status == "transcribed" else 1
    finally:
        await recognizer.aclose()


async def voice_replay(
    config_path: Path,
    directory: Path,
    *,
    frame_index: int = 0,
    input_device: int | None = None,
    ptt_key: str | None = None,
    audio_path: Path | None = None,
    reply_language: RadioLanguage | None = None,
    text_only: bool = False,
    output_device: int | None = None,
) -> int:
    # Import here keeps the existing text client and dependency boundaries unchanged.
    from race_engineer.application.conversation_cli import _state_line
    from race_engineer.fixtures import load_fixture

    config = load_config(config_path)
    configure_logging(config.logging)
    settings = config.stt.model_dump()
    if input_device is not None:
        settings["input_device"] = input_device
    if ptt_key is not None:
        settings["ptt_key"] = ptt_key.upper()
        settings["ptt_binding"] = legacy_binding(ptt_key.upper()).model_dump()
    stt = SttConfig.model_validate(settings)
    speech_settings = config.radio_tts.model_dump()
    if output_device is not None:
        speech_settings["output_device"] = output_device
    radio_tts = RadioTtsConfig.model_validate(speech_settings)
    state = ReplayRaceState(load_fixture(directory), config.policy.context)
    state.seek(frame_index)
    session = ConversationSession(
        conversation_planner(config.conversation), state.snapshot, config.conversation
    )
    recognizer = QwenSpeechRecognizer(stt)
    speaker: ConversationSpeaker | None = (
        PiperConversationSpeaker(radio_tts) if radio_tts.enabled and not text_only else None
    )
    microphone: PushToTalkMicrophone | None = None
    print(_state_line(state))
    try:
        if audio_path is None:
            print("Loading local Qwen3-ASR; microphone capture has not started.", flush=True)
            await recognizer.start()
            if speaker is not None:
                print("Loading local English/Turkish Piper voices...", flush=True)
                try:
                    await speaker.start()
                except SpeechOutputError as error:
                    print(f"Speech output unavailable: {error}. Continuing with text replies.")
                    print("Run scripts/setup_radio_tts.py; see docs/conversational-speech.md.")
                    await speaker.aclose()
                    speaker = None
            microphone = PushToTalkMicrophone(stt)
            print(
                f"Ready. Hold {binding_label(stt)} to talk; "
                "ESC exits while waiting/listening."
            )
            print("Spoken + text replies." if speaker is not None else "Text replies only.")
            print("Replay stays on the selected frame. Ctrl+C stops processing/playback and exits.")
        while True:
            try:
                audio: AudioClip | None
                if audio_path is not None:
                    audio = load_wav(audio_path, max_duration_s=stt.max_capture_s)
                else:
                    assert microphone is not None
                    audio = await microphone.next_clip()
                if audio is None:
                    return 0
                started = time.perf_counter()
                print("Transcribing...", flush=True)
                transcript, reply = await spoken_turn(
                    recognizer,
                    session,
                    audio,
                    reply_language,
                    on_transcript=lambda result: print(
                        f"You ({result.language}): {result.text}", flush=True
                    ),
                )
                del audio
                playback_failed = False
                if reply is None:
                    print(f"No question submitted: {transcript.reason or transcript.status}.")
                else:
                    print(f"Engineer: {reply.text}", flush=True)
                    print(f"ASR + reply processing: {(time.perf_counter() - started):.2f}s")
                    if reply.status == "model_error":
                        print("Start the local conversation model; see docs/conversation.md.")
                    if speaker is not None:
                        try:
                            print("Speaking...", flush=True)
                            output = await speaker.speak(reply)
                            print(
                                f"Voice: {output.language}; synthesis {output.synthesis_ms:.0f}ms; "
                                f"audio {output.audio_duration_s:.2f}s."
                            )
                        except SpeechOutputError as error:
                            playback_failed = True
                            print(f"Speech output failed: {error}. The text reply is above.")
                            print("Check docs/conversational-speech.md or use --text-only.")
                if audio_path is not None:
                    return (
                        0
                        if reply is not None
                        and reply.status != "model_error"
                        and not playback_failed
                        else 1
                    )
                print(
                    f"Ready for another question. Hold {binding_label(stt)} to talk.",
                    flush=True,
                )
            except SpeechInputError as error:
                print(f"Speech input failed: {error}. See docs/speech-to-text.md.")
                if audio_path is not None:
                    return 1
    finally:
        try:
            if microphone is not None:
                await microphone.aclose()
        finally:
            try:
                await recognizer.aclose()
            finally:
                if speaker is not None:
                    await speaker.aclose()


async def radio_check(
    config_path: Path,
    *,
    language: RadioLanguage = "en",
    text: str | None = None,
    output_device: int | None = None,
    play_audio: bool = True,
) -> int:
    """Exercise synthesis/playback without ASR, microphone capture, or a language model."""
    settings = load_config(config_path).radio_tts.model_dump()
    if output_device is not None:
        settings["output_device"] = output_device
    config = RadioTtsConfig.model_validate(settings)
    if not config.enabled:
        print("Conversational speech is disabled in [radio_tts].")
        return 1
    defaults = {
        "en": "Radio check. Race engineer online.",
        "tr": "Telsiz kontrol. Yarış mühendisin hazır.",
    }
    reply = ConversationReply(
        text=text if text is not None else defaults[language],
        language=language,
        status="answered",
        session_id="radio-check",
        source_sequence=0,
        mode="replay",
    )
    speaker = PiperConversationSpeaker(config)
    try:
        result = await speaker.speak(reply, play_audio=play_audio)
        print(result.model_dump_json(indent=2))
        return 0
    finally:
        await speaker.aclose()
