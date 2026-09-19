"""Headless administration and live-telemetry CLI."""

import argparse
import asyncio
import json
import logging
from pathlib import Path

from race_engineer.config import load_config
from race_engineer.core.contracts import (
    PlaybackResult,
    PolicyDecision,
    RaceEvent,
    SpeechIntent,
    TelemetryFrame,
    Utterance,
    canonical_json,
)
from race_engineer.core.enums import PolicyDecisionOutcome
from race_engineer.fixtures import load_fixture
from race_engineer.language import language_factory
from race_engineer.observability import configure_logging
from race_engineer.policy import DefaultRaceContextBuilder, StrictRulePolicy
from race_engineer.telemetry.iracing import IracingEventDeriver, IracingTelemetryAdapter
from race_engineer.telemetry.recorder import TelemetrySessionRecorder
from race_engineer.tts import SpeechPlaybackQueue, WindowsSapiTextToSpeechEngine, tts_factory

_LOGGER = logging.getLogger(__name__)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="race-engineer")
    subparsers = parser.add_subparsers(dest="command", required=True)

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
    return parser


async def _read_iracing(config_path: Path, limit: int, output: Path | None) -> int:
    if limit < 0:
        raise ValueError("--limit must be zero or greater")
    if output is not None and output.exists():
        raise FileExistsError(f"recording directory already exists: {output}")

    config = load_config(config_path)
    configure_logging(config.logging)
    adapter = IracingTelemetryAdapter(config.telemetry.iracing)
    event_deriver = IracingEventDeriver()
    context_builder = DefaultRaceContextBuilder(config.policy.context)
    pending_decisions: list[PolicyDecision] = []
    policy = StrictRulePolicy(config.policy.strict, decision_sink=pending_decisions.append)
    language = language_factory(config.language)
    playback: SpeechPlaybackQueue | None = None
    if config.tts.enabled:
        try:
            playback = SpeechPlaybackQueue(
                tts_factory(config.tts),
                capacity=config.tts.queue_capacity,
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
    try:
        async for frame in adapter.stream():
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
            else:
                print(canonical_json(frame), flush=True)

            try:
                context = context_builder.update(frame, events)
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
                    decision_sink=pending_decisions.append,
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
                        if playback is not None:
                            try:
                                queued = await playback.submit(intent, utterance)
                            except Exception as error:
                                _LOGGER.exception(
                                    "speech could not be queued; telemetry will continue",
                                    extra={
                                        "event": "playback_submit_failed",
                                        "reason": type(error).__name__,
                                        "intent_id": intent.intent_id,
                                    },
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
                if recorder is not None and utterances:
                    recorder.write_utterances(utterances)

            previous = frame
            frames += 1
            if limit and frames >= limit:
                break
    finally:
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
            frames = asyncio.run(_read_iracing(args.config, args.limit, args.output))
            if args.output is not None:
                print(f"recorded {frames} normalized frames to {args.output}")
        case _:
            raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    main()
