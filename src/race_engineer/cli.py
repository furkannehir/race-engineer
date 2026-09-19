"""Headless administration and live-telemetry CLI."""

import argparse
import asyncio
import json
from pathlib import Path

from race_engineer.config import load_config
from race_engineer.core.contracts import TelemetryFrame, canonical_json
from race_engineer.fixtures import load_fixture
from race_engineer.observability import configure_logging
from race_engineer.telemetry.iracing import IracingEventDeriver, IracingTelemetryAdapter
from race_engineer.telemetry.recorder import TelemetrySessionRecorder


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="race-engineer")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate-config", help="validate a TOML configuration")
    validate.add_argument("--config", type=Path, required=True)

    inspect = subparsers.add_parser("inspect-fixture", help="validate and summarize a fixture")
    inspect.add_argument("directory", type=Path)

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
        help="record frames and events in a new fixture directory instead of stdout",
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
    recorder: TelemetrySessionRecorder | None = None
    previous: TelemetryFrame | None = None
    frames = 0
    try:
        async for frame in adapter.stream():
            events = tuple(event_deriver.derive(previous, frame))
            if output is None:
                print(canonical_json(frame), flush=True)
            else:
                if recorder is None:
                    fixture_id = (
                        f"{frame.session_id}:{frame.observed_at.strftime('%Y%m%dT%H%M%SZ')}"
                    )
                    recorder = TelemetrySessionRecorder(
                        directory=output,
                        fixture_id=fixture_id,
                        description="Normalized live iRacing telemetry recording.",
                    )
                    recorder.__enter__()
                recorder.write_frame(frame)
                recorder.write_events(events)

            previous = frame
            frames += 1
            if limit and frames >= limit:
                break
    finally:
        if recorder is not None:
            recorder.__exit__(None, None, None)
    return frames


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
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
        case "read-iracing":
            frames = asyncio.run(_read_iracing(args.config, args.limit, args.output))
            if args.output is not None:
                print(f"recorded {frames} normalized frames to {args.output}")
        case _:
            raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    main()
