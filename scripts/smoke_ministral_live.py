"""Exercise the production Ministral factory offline, optionally synthesize without playback."""

import argparse
import asyncio
import json
import socket
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from race_engineer.application.intelligence_cli import _contexts
from race_engineer.config import load_config
from race_engineer.conversation.runtime import ConversationServer
from race_engineer.core.conversation import ConversationReply
from race_engineer.core.intelligence import DriverTurn
from race_engineer.fixtures import load_fixture
from race_engineer.intelligence.factory import live_intelligence
from race_engineer.intelligence.ministral import MINISTRAL_NATIVE_ID
from race_engineer.intelligence.telemetry_memory import BoundedTelemetryMemory
from race_engineer.tts.chatterbox import ChatterboxSpeechBackend

ROOT = Path(__file__).resolve().parents[1]
CASES = (
    ("en", "Radio check"),
    ("en", "Position?"),
    ("en", "What's my last lap time?"),
    ("en", "How many race laps remain?"),
    ("en", "Do we have enough fuel to finish?"),
    ("tr", "Telsiz kontrol"),
    ("tr", "Kaçıncıyız?"),
    ("tr", "Son tur zamanım ne?"),
)


async def run(args):
    app = load_config(args.config, environ={})
    if app.conversation.context_planner != MINISTRAL_NATIVE_ID:
        raise ValueError("This smoke test requires ministral-native-v1")
    # Refuse to borrow an existing model server or change its settings.
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", args.port))
    runtime = app.conversation.runtime.model_copy(update={
        "mode": args.backend, "fallback_to_cpu": False, "reuse_external_server": False,
    })
    config = app.conversation.model_copy(update={"runtime": runtime, "port": args.port})
    memory = BoundedTelemetryMemory()
    for context in _contexts(load_fixture(args.fixture), app.policy.context):
        memory.update(context)
    engineer = live_intelligence(config, memory)
    server = ConversationServer(ROOT, config)
    speech = ChatterboxSpeechBackend(app.tts) if args.synthesize else None
    report = {"model": config.model, "backend": args.backend, "turns": [],
              "synthetic_telemetry": True, "audio_played": False}
    print(f"Starting production factory: {config.model} ({args.backend})...", flush=True)
    try:
        await server.start()
        assert server.process is not None
        if speech:
            print("Starting local TTS; no microphone or audio playback.", flush=True)
            await speech.start()
        for index, (language, question) in enumerate(CASES, 1):
            frame = memory.latest_context().frame
            request = DriverTurn(
                turn_id=f"smoke:{index}", transcript=question, session_id=frame.session_id,
                received_at=datetime.now(UTC), generation=0,
                reply_language=language, asr_language=language,
            )
            print(f"[{index}/{len(CASES)}] {language}: {question}", flush=True)
            started = time.perf_counter()
            try:
                prepared = await engineer.prepare(request)
                answer = await engineer.ground(prepared)
            except Exception as error:
                row = {"language": language, "question": question, "failure": str(error),
                       "model_ms": round((time.perf_counter() - started) * 1000, 1)}
                engine = engineer._engine
                if engine is not None:
                    row["trace"] = engine.last_trace
                    row["model_metadata"] = getattr(engine.model, "last_metadata", {})
                report["turns"].append(row)
                print(f"  FAILED: {error}", flush=True)
                continue
            row = {"language": language, "question": question, "reply": answer.text,
                   "model_ms": round((time.perf_counter() - started) * 1000, 1),
                   "source_sequence": answer.source_sequence,
                   "evidence_ids": list(answer.evidence_ids)}
            if speech:
                result = await speech.speak(ConversationReply(
                    text=answer.text, language=answer.language, status="answered",
                    session_id=answer.session_id, source_sequence=answer.source_sequence,
                    mode="live",
                ), play_audio=False)
                row.update(tts_ms=result.synthesis_ms, audio_duration_s=result.audio_duration_s,
                           worker_pid=speech.worker_pid)
            # This script's printed text is its delivery boundary.
            engineer.delivered(prepared, answer)
            report["turns"].append(row)
            print(f"  {answer.text} ({row['model_ms']} ms)", flush=True)
    finally:
        if speech:
            await speech.aclose()
        await server.aclose()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    print(f"Saved {args.output}. Smoke results only; no accuracy or VR acceptance.", flush=True)
    if any("failure" in row for row in report["turns"]):
        raise SystemExit(1)


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config/default.toml")
    parser.add_argument("--fixture", type=Path,
                        default=ROOT / "fixtures/synthetic/ministral-lap-progress")
    parser.add_argument("--backend", choices=("gpu", "cpu"), default="gpu")
    parser.add_argument("--port", type=int, default=18201)
    parser.add_argument("--synthesize", action="store_true", help="local TTS, no playback")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "data/diagnostics/ministral-integration/live-smoke.json")
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
