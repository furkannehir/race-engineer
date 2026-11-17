"""Typed native LFM tool conversation: driver -> assistant tool call -> tool result -> reply."""

import argparse
import asyncio
import io
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from race_engineer.application.intelligence_cli import _contexts
from race_engineer.config import load_config
from race_engineer.conversation.runtime import ConversationServer
from race_engineer.core.intelligence import DriverTurn
from race_engineer.fixtures import load_fixture
from race_engineer.intelligence.telemetry_memory import BoundedTelemetryMemory

if __package__:
    from .lfm3_dialogue import NativeToolDialogue
else:
    from lfm3_dialogue import NativeToolDialogue

ROOT = Path(__file__).resolve().parents[1]


async def chat(args):
    config = load_config(args.config)
    if config.conversation.model != "LiquidAI/LFM3-1.7B":
        raise ValueError("This interactive trial must explicitly select LFM3")
    memory = BoundedTelemetryMemory()
    for context in _contexts(
        load_fixture(ROOT / "fixtures/synthetic/intelligence-capabilities"), config.policy.context
    ):
        memory.update(context)
    frame = memory.latest_context().frame
    dialogue = NativeToolDialogue(config.conversation, memory)
    server = ConversationServer(ROOT, config.conversation)
    print(
        f"Starting {config.conversation.model} ({config.conversation.runtime.mode})...", flush=True
    )
    failed = False
    try:
        await server.start()
        print("Native dialogue trial | synthetic telemetry. /clear resets; /quit exits.",
              flush=True)
        print(
            "Tools: current position, gaps ahead/behind, unsupported requests. Typed English only.",
            flush=True,
        )
        index = 0
        while True:
            try:
                question = args.question or await asyncio.to_thread(input, "Driver (en)> ")
            except EOFError:
                break
            question = question.strip()
            if question in {"/quit", "/exit"}:
                break
            if question == "/clear":
                dialogue.clear()
                print("Dialogue cleared.", flush=True)
                continue
            if not question:
                continue
            index += 1
            started = time.perf_counter()
            turn = DriverTurn(
                turn_id=f"native-dialogue-{index}", transcript=question,
                received_at=datetime.now(UTC), session_id=frame.session_id, generation=0,
                asr_language="en", reply_language="en",
            )
            try:
                answer = await dialogue.respond(turn)
                print(f"Engineer: {answer.text}", flush=True)
            except Exception as error:
                failed = True
                print(f"Question failed: {error}", flush=True)
                if any(reason in str(error) for reason in ("timeout", "unreachable")):
                    break
            finally:
                if args.trace:
                    print(json.dumps(dialogue.last_trace, ensure_ascii=True, indent=2), flush=True)
            print(f"{(time.perf_counter() - started) * 1000:.0f} ms", flush=True)
            if args.question:
                break
    finally:
        await server.aclose()
    return 1 if args.question and failed else 0


def main():
    if isinstance(sys.stdout, io.TextIOWrapper):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config/default.toml")
    parser.add_argument("--question", help="Single prompt instead of interactive chat")
    parser.add_argument("--trace", action="store_true", help="Show model replies and tool results")
    args = parser.parse_args()
    if args.question is not None and (
        not args.question.strip() or args.question.strip().startswith("/")
    ):
        parser.error("--question requires a nonempty driver message")
    try:
        return asyncio.run(chat(args))
    except KeyboardInterrupt:
        print("Trial stopped.")
        return 130
    except (OSError, RuntimeError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    raise SystemExit(main())
