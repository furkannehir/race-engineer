"""Headless administration and live-telemetry CLI."""

import argparse
import asyncio
import json
import logging
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path

from race_engineer.config import AppConfig, load_config
from race_engineer.core.contracts import (
    PlaybackResult,
    PolicyDecision,
    RaceEvent,
    SpeechIntent,
    TelemetryFrame,
    Utterance,
    canonical_json,
)
from race_engineer.core.enums import PlaybackStatus, PolicyDecisionOutcome
from race_engineer.core.interfaces import LiveTelemetryBridge
from race_engineer.fixtures import load_fixture
from race_engineer.language import language_factory
from race_engineer.memory import DurableHistory
from race_engineer.observability import configure_logging
from race_engineer.policy import DefaultRaceContextBuilder, StrictRulePolicy
from race_engineer.telemetry.iracing import IracingEventDeriver, IracingTelemetryAdapter
from race_engineer.telemetry.recorder import TelemetrySessionRecorder
from race_engineer.tts import SpeechPlaybackQueue, WindowsSapiTextToSpeechEngine, tts_factory

_LOGGER = logging.getLogger(__name__)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="race-engineer")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("list-input-devices", help="list microphone devices without recording")
    subparsers.add_parser("list-output-devices", help="list speaker/headphone output devices")
    radio = subparsers.add_parser("test-radio", help="test local English/Turkish Piper speech")
    radio.add_argument("--config", type=Path, required=True)
    radio.add_argument("--language", choices=("en", "tr"), default="en")
    radio.add_argument("--text", help="custom radio-check text")
    radio.add_argument("--output-device", type=int, help="index from list-output-devices")
    radio.add_argument(
        "--no-playback", action="store_true", help="synthesize without playing audio"
    )
    transcribe = subparsers.add_parser(
        "transcribe-wav", help="transcribe a local PCM16 WAV offline"
    )
    transcribe.add_argument("audio", type=Path)
    transcribe.add_argument("--config", type=Path, required=True)
    voice = subparsers.add_parser(
        "voice-replay", help="push-to-talk replay questions with local spoken and text replies"
    )
    voice.add_argument("directory", type=Path)
    voice.add_argument("--config", type=Path, required=True)
    voice.add_argument("--frame-index", type=int, default=0)
    voice.add_argument("--device", type=int, help="microphone index from list-input-devices")
    voice.add_argument("--ptt-key", help="hold key, e.g. F8, F9, RCTRL; default: config")
    voice.add_argument("--audio", type=Path, help="use one WAV file instead of microphone capture")
    voice.add_argument("--language", choices=("en", "tr"), help="force answer language")
    voice.add_argument(
        "--text-only", action="store_true", help="disable conversational audio output"
    )
    voice.add_argument("--output-device", type=int, help="speaker index from list-output-devices")

    live_voice = subparsers.add_parser(
        "voice-iracing",
        help="live iRacing telemetry, automatic calls, and push-to-talk conversation",
    )
    live_voice.add_argument("--config", type=Path, required=True)
    live_voice.add_argument(
        "--limit", type=int, default=0, help="frame limit; zero runs until exit"
    )
    live_voice.add_argument("--output", type=Path, help="new telemetry recording directory")
    live_voice.add_argument("--device", type=int, help="microphone index from list-input-devices")
    live_voice.add_argument("--ptt-key", help="hold key; default: config (F8)")
    live_voice.add_argument(
        "--output-device", type=int, help="Piper output device; SAPI uses default"
    )
    live_voice.add_argument(
        "--language", choices=("en", "tr"), help="force conversational language"
    )
    live_voice.add_argument("--text-only", action="store_true", help="mute ALL speech in live mode")

    chat = subparsers.add_parser(
        "chat-replay", help="ask the local Qwen model questions about a paused replay"
    )
    chat.add_argument("directory", type=Path)
    chat.add_argument("--config", type=Path, required=True)
    chat.add_argument("--frame-index", type=int, default=0)
    chat.add_argument("--question", help="ask once instead of opening an interactive conversation")
    chat.add_argument(
        "--language", choices=("en", "tr"), help="force reply language; default: auto"
    )
    chat.add_argument("--json", action="store_true", help="structured output with --question")

    validate = subparsers.add_parser("validate-config", help="validate a TOML configuration")
    validate.add_argument("--config", type=Path, required=True)

    inspect = subparsers.add_parser("inspect-fixture", help="validate and summarize a fixture")
    inspect.add_argument("directory", type=Path)

    replay_policy = subparsers.add_parser(
        "replay-policy",
        help="replay recorded events through the deterministic strict policy",
    )
    replay_policy.add_argument("directory", type=Path)
    replay_policy.add_argument("--config", type=Path, required=True)

    replay_language = subparsers.add_parser(
        "replay-language",
        help="replay recorded speech intents through the configured language generator",
    )
    replay_language.add_argument("directory", type=Path)
    replay_language.add_argument("--config", type=Path, required=True)

    list_voices = subparsers.add_parser(
        "list-tts-voices",
        help="list locally installed voices for the configured speech adapter",
    )
    list_voices.add_argument("--config", type=Path, required=True)

    test_tts = subparsers.add_parser(
        "test-tts",
        help="play a local race-engineer radio check",
    )
    test_tts.add_argument("--config", type=Path, required=True)
    test_tts.add_argument(
        "--text",
        default="Radio check. Race engineer online.",
        help="text to speak during the audio test",
    )

    read_iracing = subparsers.add_parser(
        "read-iracing",
        help="stream normalized telemetry from a running iRacing simulator",
    )
    read_iracing.add_argument("--config", type=Path, required=True)
    read_iracing.add_argument(
        "--limit",
        type=int,
        default=0,
        help="stop after this many frames; zero streams until interrupted",
    )
    read_iracing.add_argument(
        "--output",
        type=Path,
        help="record the live deterministic pipeline in a new fixture directory",
    )

    profile = subparsers.add_parser(
        "profile", help="inspect or change the local default driver profile"
    )
    profile.add_argument("--config", type=Path, required=True)
    profile_actions = profile.add_subparsers(dest="profile_action", required=True)
    profile_actions.add_parser("show", help="show the default profile and preferences")
    rename = profile_actions.add_parser("rename", help="rename the default profile")
    rename.add_argument("display_name")
    preference = profile_actions.add_parser(
        "set", help="set an explicit driver-scoped communication preference"
    )
    preference.add_argument(
        "setting",
        choices=(
            "announce_position_changes",
            "announce_pit_transitions",
            "reply_language",
        ),
    )
    preference.add_argument("value")
    profile_actions.add_parser("reset", help="reset communication preferences to defaults")

    history = subparsers.add_parser(
        "history", help="inspect privacy-safe policy and radio session history"
    )
    history.add_argument("--config", type=Path, required=True)
    history_actions = history.add_subparsers(dest="history_action", required=True)
    recent = history_actions.add_parser("recent", help="summarize recent recorded sessions")
    recent.add_argument("--limit", type=int, default=10)
    history_actions.add_parser("prune", help="apply the configured history retention period")
    return parser


def _parse_profile_value(setting: str, raw: str) -> bool | str:
    if setting in {"announce_position_changes", "announce_pit_transitions"}:
        normalized = raw.strip().lower()
        if normalized not in {"true", "false"}:
            raise ValueError(f"{setting} expects true or false")
        return normalized == "true"
    if setting == "reply_language" and raw in {"auto", "en", "tr"}:
        return raw
    raise ValueError("reply_language expects auto, en, or tr")


def _profile_action(
    config_path: Path,
    action: str,
    *,
    display_name: str | None = None,
    setting: str | None = None,
    value: str | None = None,
) -> dict[str, object]:
    from race_engineer.core.contracts import PreferenceCommand
    from race_engineer.core.enums import PreferenceScope, PreferenceSource
    from race_engineer.memory import SqliteDriverProfileRepository

    config = load_config(config_path)
    repository = SqliteDriverProfileRepository(config.paths.database_path)
    profile = repository.ensure_default_profile()
    if action == "rename":
        if display_name is None:
            raise ValueError("profile name is required")
        profile = repository.rename_profile(profile.profile_id, display_name)
    elif action == "set":
        if setting is None or value is None:
            raise ValueError("preference setting and value are required")
        repository.apply(
            profile.profile_id,
            PreferenceCommand(
                command_id=uuid.uuid4().hex,
                setting=setting,
                value=_parse_profile_value(setting, value),
                source=PreferenceSource.CLI,
                timestamp=datetime.now(UTC),
                scope=PreferenceScope.DRIVER,
            ),
        )
    elif action == "reset":
        repository.reset_preferences(profile.profile_id, source=PreferenceSource.CLI)
    elif action != "show":
        raise ValueError("unsupported profile action")
    return {
        "profile": profile.model_dump(mode="json"),
        "preferences": repository.preferences(profile.profile_id).model_dump(mode="json"),
    }


def _history_action(
    config_path: Path,
    action: str,
    *,
    limit: int = 10,
) -> dict[str, object]:
    from race_engineer.memory import SqliteDriverProfileRepository

    config = load_config(config_path)
    repository = SqliteDriverProfileRepository(config.paths.database_path)
    profile = repository.ensure_default_profile()
    result: dict[str, object] = {
        "profile_id": profile.profile_id,
        "retention_days": config.history.retention_days,
    }
    if action == "recent":
        result["sessions"] = [
            summary.model_dump(mode="json")
            for summary in repository.recent_session_history(
                profile.profile_id, limit=limit
            )
        ]
    elif action == "prune":
        result["deleted_decisions"] = repository.prune_history(
            profile.profile_id,
            before=datetime.now(UTC) - timedelta(days=config.history.retention_days),
        )
    else:
        raise ValueError("unsupported history action")
    return result


async def _read_iracing(
    config_path: Path,
    limit: int,
    output: Path | None,
    *,
    live: LiveTelemetryBridge | None = None,
    app_config: AppConfig | None = None,
    history: DurableHistory | None = None,
) -> int:
    if limit < 0:
        raise ValueError("--limit must be zero or greater")
    if output is not None and output.exists():
        raise FileExistsError(f"recording directory already exists: {output}")

    config = app_config or load_config(config_path)
    if app_config is None:
        configure_logging(config.logging)
    reset_requested = False

    def availability_changed(available: bool) -> None:
        nonlocal reset_requested
        if not available:
            reset_requested = True
        assert live is not None
        live.availability_changed(available)

    adapter = (
        IracingTelemetryAdapter(
            config.telemetry.iracing.model_copy(update={"include_replay": False}),
            availability_sink=availability_changed,
        )
        if live is not None
        else IracingTelemetryAdapter(config.telemetry.iracing)
    )
    event_deriver = IracingEventDeriver()
    context_builder = DefaultRaceContextBuilder(config.policy.context)
    pending_decisions: list[PolicyDecision] = []

    def capture_decision(decision: PolicyDecision) -> None:
        pending_decisions.append(decision)
        if history is not None:
            history.record_decision(decision)

    policy = StrictRulePolicy(config.policy.strict, decision_sink=capture_decision)
    language = language_factory(config.language)
    playback: SpeechPlaybackQueue | None = None
    if config.tts.enabled and live is None:
        try:
            playback = SpeechPlaybackQueue(
                tts_factory(config.tts),
                capacity=config.tts.queue_capacity,
                result_sink=(history.record_playback_result if history is not None else None),
            )
            playback.start()
        except Exception as error:
            _LOGGER.exception(
                "text-to-speech could not start; telemetry will continue",
                extra={
                    "event": "tts_start_failed",
                    "reason": type(error).__name__,
                },
            )
    recorder: TelemetrySessionRecorder | None = None
    previous: TelemetryFrame | None = None
    frames = 0
    stream = adapter.stream()
    try:
        async for frame in stream:
            if live is not None and (reset_requested or not live.available):
                previous = None
                context_builder = DefaultRaceContextBuilder(config.policy.context)
                policy = StrictRulePolicy(
                    config.policy.strict, decision_sink=capture_decision
                )
                pending_decisions.clear()
                reset_requested = False
            events = tuple(event_deriver.derive(previous, frame))
            if output is not None and recorder is None:
                fixture_id = f"{frame.session_id}:{frame.observed_at.strftime('%Y%m%dT%H%M%SZ')}"
                recorder = TelemetrySessionRecorder(
                    directory=output,
                    fixture_id=fixture_id,
                    description="Live iRacing telemetry and deterministic policy recording.",
                )
                recorder.__enter__()

            if recorder is not None:
                recorder.write_frame(frame)
                recorder.write_events(events)
            elif live is None:
                print(canonical_json(frame), flush=True)

            try:
                context = context_builder.update(frame, events)
                if live is not None:
                    live.update(context)
                intents = await policy.decide(context)
            except Exception as error:
                _LOGGER.exception(
                    "policy evaluation failed; telemetry will continue",
                    extra={
                        "event": "policy_failed",
                        "reason": type(error).__name__,
                        "session_id": frame.session_id,
                        "source_sequence": frame.sequence,
                    },
                )
                pending_decisions.clear()
                context_builder = DefaultRaceContextBuilder(config.policy.context)
                policy = StrictRulePolicy(
                    config.policy.strict,
                    decision_sink=capture_decision,
                )
            else:
                if recorder is not None:
                    if intents:
                        recorder.write_intents(intents)
                    if pending_decisions:
                        recorder.write_decisions(pending_decisions)
                pending_decisions.clear()

                utterances: list[Utterance] = []
                for intent in intents:
                    try:
                        utterance = await language.generate(intent)
                        if utterance.intent_id != intent.intent_id:
                            raise ValueError(
                                "generator returned an utterance for a different intent"
                            )
                        if len(utterance.text.split()) > intent.max_words:
                            raise ValueError("generator exceeded the intent word limit")
                    except Exception as error:
                        _LOGGER.exception(
                            "language generation failed; telemetry and policy will continue",
                            extra={
                                "event": "language_failed",
                                "reason": type(error).__name__,
                                "intent_id": intent.intent_id,
                            },
                        )
                        if history is not None:
                            history.record_playback_result(
                                PlaybackResult(
                                    intent_id=intent.intent_id,
                                    status=PlaybackStatus.FAILED,
                                    finished_at=datetime.now(UTC),
                                    error_code="language_generation_failed",
                                )
                            )
                    else:
                        utterances.append(utterance)
                        _LOGGER.info(
                            "utterance generated",
                            extra={
                                "event": "utterance_generated",
                                "intent_id": intent.intent_id,
                                "adapter": utterance.generator_metadata.get("adapter"),
                                "template_id": utterance.generator_metadata.get("template_id"),
                            },
                        )
                        speech_sink = live if live is not None else playback
                        if speech_sink is not None:
                            try:
                                queued = await speech_sink.submit(intent, utterance)
                            except Exception as error:
                                _LOGGER.exception(
                                    "speech could not be queued; telemetry will continue",
                                    extra={
                                        "event": "playback_submit_failed",
                                        "reason": type(error).__name__,
                                        "intent_id": intent.intent_id,
                                    },
                                )
                                if history is not None:
                                    history.record_playback_result(
                                        PlaybackResult(
                                            intent_id=intent.intent_id,
                                            status=PlaybackStatus.FAILED,
                                            finished_at=datetime.now(UTC),
                                            error_code="playback_submit_failed",
                                        )
                                    )
                            else:
                                if queued:
                                    _LOGGER.info(
                                        "utterance queued for speech",
                                        extra={
                                            "event": "playback_queued",
                                            "intent_id": intent.intent_id,
                                        },
                                    )
                        elif history is not None:
                            history.record_playback_result(
                                PlaybackResult(
                                    intent_id=intent.intent_id,
                                    status=(
                                        PlaybackStatus.CANCELLED
                                        if not config.tts.enabled
                                        else PlaybackStatus.FAILED
                                    ),
                                    finished_at=datetime.now(UTC),
                                    error_code=(
                                        "audio_disabled"
                                        if not config.tts.enabled
                                        else "speech_unavailable"
                                    ),
                                )
                            )
                if recorder is not None and utterances:
                    recorder.write_utterances(utterances)

            previous = frame
            frames += 1
            if limit and frames >= limit:
                break
    finally:
        try:
            await stream.aclose()
        finally:
            try:
                if live is not None:
                    live.availability_changed(False)
                if playback is not None:
                    try:
                        await asyncio.wait_for(
                            playback.aclose(drain=True),
                            timeout=config.runtime.shutdown_timeout_s,
                        )
                    except TimeoutError:
                        _LOGGER.warning(
                            "speech queue did not drain before shutdown",
                            extra={"event": "playback_shutdown_timeout"},
                        )
                        await playback.aclose(drain=False)
            finally:
                if recorder is not None:
                    recorder.__exit__(None, None, None)
    return frames


async def _replay_policy(config_path: Path, directory: Path) -> dict[str, object]:
    config = load_config(config_path)
    configure_logging(config.logging)
    fixture = load_fixture(directory)
    events_by_frame: dict[tuple[str, int], list[RaceEvent]] = {}
    for event in fixture.expected_events:
        events_by_frame.setdefault((event.session_id, event.source_sequence), []).append(event)

    decisions: list[PolicyDecision] = []
    builder = DefaultRaceContextBuilder(config.policy.context)
    policy = StrictRulePolicy(config.policy.strict, decision_sink=decisions.append)
    intents: list[SpeechIntent] = []
    for frame in fixture.frames:
        events = tuple(events_by_frame.get((frame.session_id, frame.sequence), ()))
        context = builder.update(frame, events)
        intents.extend(await policy.decide(context))

    approved = sum(decision.outcome is PolicyDecisionOutcome.APPROVED for decision in decisions)
    expected_match = (
        tuple(intents) == fixture.expected_intents
        if fixture.manifest.expected_intents_file is not None
        else None
    )
    expected_decisions_match = (
        tuple(decisions) == fixture.expected_decisions
        if fixture.manifest.expected_decisions_file is not None
        else None
    )
    return {
        "fixture_id": fixture.manifest.fixture_id,
        "frames": len(fixture.frames),
        "events": len(fixture.expected_events),
        "decisions": len(decisions),
        "approved": approved,
        "suppressed": len(decisions) - approved,
        "intents": len(intents),
        "expected_intents_match": expected_match,
        "expected_decisions_match": expected_decisions_match,
    }


async def _replay_language(config_path: Path, directory: Path) -> dict[str, object]:
    config = load_config(config_path)
    configure_logging(config.logging)
    fixture = load_fixture(directory)
    language = language_factory(config.language)
    utterances: list[Utterance] = []
    for intent in fixture.expected_intents:
        utterance = await language.generate(intent)
        if utterance.intent_id != intent.intent_id:
            raise ValueError("generator returned an utterance for a different intent")
        if len(utterance.text.split()) > intent.max_words:
            raise ValueError("generator exceeded the intent word limit")
        utterances.append(utterance)

    expected_match = (
        tuple(utterances) == fixture.expected_utterances
        if fixture.manifest.expected_utterances_file is not None
        else None
    )
    return {
        "fixture_id": fixture.manifest.fixture_id,
        "intents": len(fixture.expected_intents),
        "utterances": len(utterances),
        "expected_utterances_match": expected_match,
    }


async def _list_tts_voices(config_path: Path) -> tuple[str, ...]:
    config = load_config(config_path)
    configure_logging(config.logging)
    if not config.tts.enabled:
        raise ValueError("text-to-speech is disabled")
    if config.tts.adapter == "windows-sapi":
        return await WindowsSapiTextToSpeechEngine.installed_voices()
    raise AssertionError(f"unsupported text-to-speech adapter: {config.tts.adapter}")


async def _test_tts(config_path: Path, text: str) -> PlaybackResult:
    config = load_config(config_path)
    configure_logging(config.logging)
    engine = tts_factory(config.tts)
    utterance = Utterance(
        intent_id="tts-test",
        text=text,
        generator_metadata={"adapter": "manual-test"},
    )
    return await engine.speak(utterance)


def main() -> None:
    args = _parser().parse_args()
    match args.command:
        case "voice-iracing":
            from race_engineer.application.live_conversation import voice_iracing
            from race_engineer.core.speech_input import SpeechInputError

            try:
                live_code = asyncio.run(
                    voice_iracing(
                        args.config,
                        limit=args.limit,
                        output=args.output,
                        input_device=args.device,
                        ptt_key=args.ptt_key,
                        output_device=args.output_device,
                        text_only=args.text_only,
                        reply_language=args.language,
                        persist_history=True,
                    )
                )
            except SpeechInputError as error:
                print(f"Live speech input failed: {error}. See docs/live-conversation.md.")
                live_code = 1
            except (ValueError, OSError):
                print("Cannot start live conversation; check configuration and recording path.")
                live_code = 2
            except KeyboardInterrupt:
                live_code = 130
            raise SystemExit(live_code)
        case "list-output-devices" | "test-radio":
            from race_engineer.application.speech_cli import radio_check
            from race_engineer.core.speech_output import SpeechOutputError
            from race_engineer.tts.devices import output_devices

            try:
                if args.command == "list-output-devices":
                    print(json.dumps(output_devices(), indent=2, ensure_ascii=False))
                    radio_code = 0
                else:
                    radio_code = asyncio.run(
                        radio_check(
                            args.config,
                            language=args.language,
                            text=args.text,
                            output_device=args.output_device,
                            play_audio=not args.no_playback,
                        )
                    )
            except SpeechOutputError as error:
                print(f"Speech output failed: {error}. See docs/conversational-speech.md.")
                radio_code = 1
            except (ValueError, OSError):
                print("Cannot start radio check; check configuration and voice paths.")
                radio_code = 2
            except KeyboardInterrupt:
                radio_code = 130
            raise SystemExit(radio_code)
        case "list-input-devices" | "transcribe-wav" | "voice-replay":
            from race_engineer.application.speech_cli import transcribe_wav, voice_replay
            from race_engineer.core.speech_input import SpeechInputError
            from race_engineer.stt.capture import input_devices

            try:
                if args.command == "list-input-devices":
                    print(json.dumps(input_devices(), indent=2, ensure_ascii=False))
                    speech_code = 0
                elif args.command == "transcribe-wav":
                    speech_code = asyncio.run(transcribe_wav(args.config, args.audio))
                else:
                    speech_code = asyncio.run(
                        voice_replay(
                            args.config,
                            args.directory,
                            frame_index=args.frame_index,
                            input_device=args.device,
                            ptt_key=args.ptt_key,
                            audio_path=args.audio,
                            reply_language=args.language,
                            text_only=args.text_only,
                            output_device=args.output_device,
                        )
                    )
            except SpeechInputError as error:
                print(f"Speech input failed: {error}. See docs/speech-to-text.md.")
                speech_code = 1
            except (ValueError, OSError):
                print("Cannot start speech input; check configuration, fixture, and audio path.")
                speech_code = 2
            except KeyboardInterrupt:
                speech_code = 130
            raise SystemExit(speech_code)
        case "chat-replay":
            from race_engineer.application.conversation_cli import chat_replay

            try:
                result_code = asyncio.run(
                    chat_replay(
                        args.config,
                        args.directory,
                        frame_index=args.frame_index,
                        question=args.question,
                        reply_language=args.language,
                        json_output=args.json,
                    )
                )
            except KeyboardInterrupt:
                result_code = 130
            except (OSError, ValueError):
                print("Cannot start replay conversation; check config, fixture, and arguments.")
                result_code = 2
            raise SystemExit(result_code)
        case "validate-config":
            config = load_config(args.config)
            print(config.model_dump_json(indent=2))
        case "inspect-fixture":
            fixture = load_fixture(args.directory)
            print(
                json.dumps(
                    {
                        "fixture_id": fixture.manifest.fixture_id,
                        "frames": len(fixture.frames),
                        "expected_events": len(fixture.expected_events),
                        "expected_intents": len(fixture.expected_intents),
                        "expected_decisions": len(fixture.expected_decisions),
                        "expected_utterances": len(fixture.expected_utterances),
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
        case "replay-policy":
            result = asyncio.run(_replay_policy(args.config, args.directory))
            print(json.dumps(result, indent=2, sort_keys=True))
        case "replay-language":
            result = asyncio.run(_replay_language(args.config, args.directory))
            print(json.dumps(result, indent=2, sort_keys=True))
        case "list-tts-voices":
            voices = asyncio.run(_list_tts_voices(args.config))
            print(json.dumps({"voices": voices}, indent=2, sort_keys=True))
        case "test-tts":
            playback_result = asyncio.run(_test_tts(args.config, args.text))
            print(playback_result.model_dump_json(indent=2))
        case "read-iracing":
            config = load_config(args.config)
            history = (
                DurableHistory.try_open(
                    config.paths.database_path,
                    retention_days=config.history.retention_days,
                )
                if config.history.enabled
                else None
            )
            frames = asyncio.run(
                _read_iracing(args.config, args.limit, args.output, history=history)
            )
            if args.output is not None:
                print(f"recorded {frames} normalized frames to {args.output}")
        case "profile":
            from race_engineer.memory import DriverMemoryError

            try:
                result = _profile_action(
                    args.config,
                    args.profile_action,
                    display_name=getattr(args, "display_name", None),
                    setting=getattr(args, "setting", None),
                    value=getattr(args, "value", None),
                )
            except (DriverMemoryError, OSError, ValueError) as error:
                print(f"Driver profile update failed: {error}")
                raise SystemExit(2) from None
            print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))
        case "history":
            from race_engineer.memory import DriverMemoryError

            try:
                result = _history_action(
                    args.config,
                    args.history_action,
                    limit=getattr(args, "limit", 10),
                )
            except (DriverMemoryError, OSError, ValueError) as error:
                print(f"History query failed: {error}")
                raise SystemExit(2) from None
            print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))
        case _:
            raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    main()
