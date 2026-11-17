"""Interactive Ministral chat using the model-comparison fixture and native tools."""

import argparse
import asyncio
import io
import json
import sys
from pathlib import Path

from race_engineer.application.intelligence_cli import _contexts
from race_engineer.conversation.runtime import ConversationServer
from race_engineer.evaluation.system import file_sha256
from race_engineer.fixtures import load_fixture
from race_engineer.intelligence.telemetry_memory import BoundedTelemetryMemory

if __package__:
    from .benchmark_qwen_cache import require_free_port
    from .compare_dialogue_models import CANDIDATES, ROOT, RawConversation, configuration
    from .lfm3_dialogue import LocalDialogueModel, NativeToolDialogue
    from .ministral_dialogue import MinistralConversation
else:
    from benchmark_qwen_cache import require_free_port
    from compare_dialogue_models import CANDIDATES, ROOT, RawConversation, configuration
    from lfm3_dialogue import LocalDialogueModel, NativeToolDialogue
    from ministral_dialogue import MinistralConversation

HELP = "/clear resets history; /en and /tr change language and reset; /quit exits; /help."
DEFAULT_FIXTURE = ROOT / "fixtures/synthetic/ministral-lap-progress"


def prepare(args):
    candidate = CANDIDATES["ministral"]
    if not candidate["path"].is_file():
        raise ValueError(
            "Ministral weights missing. Run: .\\.venv\\Scripts\\python.exe "
            "scripts\\setup_dialogue_comparison.py --model ministral"
        )
    print("Verifying cached Ministral weights...", flush=True)
    if file_sha256(candidate["path"]) != candidate["sha256"]:
        raise ValueError("Ministral checksum mismatch; model was not loaded")
    config, context_settings = configuration(candidate, args.backend, args.port)
    memory = BoundedTelemetryMemory()
    for context in _contexts(
        load_fixture(getattr(args, "fixture", None) or DEFAULT_FIXTURE), context_settings
    ):
        memory.update(context)
    return config, memory


async def chat(args):
    config, memory = prepare(args)
    require_free_port(config.port)
    candidate = CANDIDATES["ministral"]
    generation = {"max_tokens": 256, "seed": 7, **candidate["generation"]}
    if getattr(args, "raw", False):
        dialogue = RawConversation(
            LocalDialogueModel(config, generation=generation), NativeToolDialogue(config, memory),
            history_turns=config.history_turns,
        )
    else:
        dialogue = MinistralConversation(config, memory)
    server = ConversationServer(ROOT, config)
    language = args.language
    index, failed, fatal = 0, False, False
    print(f"Starting {config.model} ({args.backend})...", flush=True)
    try:
        await server.start()
        expected = "cuda" if args.backend == "gpu" else "cpu"
        if server.process is None or server.effective.backend != expected:
            raise RuntimeError("Demo requires its own server on the requested backend")
        frame = memory.latest_context().frame
        print(f"Ministral chat | fixture session {frame.session_id}, "
              f"sequence {frame.sequence}. Replay data, no audio.", flush=True)
        if getattr(args, "raw", False):
            print("Raw comparison mode; validation is diagnostic only.", flush=True)
        else:
            print("Checked prototype: fresh tools and verified radio recall. No audio.", flush=True)
        print("Tools: position/gaps, classification, fuel/range/to-finish, lap number, "
              "last/best lap times, last-versus-best, race laps/time remaining, states, "
              "qualitative gap movement/catch, bounded history, prior radio readings. "
              "No speed, opponent lap pace or car controls.", flush=True)
        print(HELP, flush=True)
        while True:
            try:
                # CLI input stays on the main thread so Ctrl+C cannot leave a
                # blocked input worker behind during asyncio executor shutdown.
                question = args.question if args.question is not None else input(
                    f"Driver ({language})> "
                )
            except EOFError:
                break
            question = question.strip()
            command = question.lower()
            if command in {"/quit", "/exit"}:
                break
            if command in {"/clear", "/en", "/tr"}:
                dialogue.clear()
                if command in {"/en", "/tr"}:
                    language = command[1:]
                print(f"History cleared. Reply language: {language}.", flush=True)
                continue
            if command == "/help":
                print(HELP, flush=True)
                continue
            if not question:
                continue
            if question.startswith("/"):
                print(f"Unknown command. {HELP}", flush=True)
                continue
            index += 1
            row = await dialogue.respond(question, language, f"ministral-chat-{index}")
            delivered = row.get("reply", row["raw_reply"])
            if delivered is not None:
                print(f"Engineer ({language}): {delivered}", flush=True)
                validation = row["application_validation"]
                if validation["status"] == "rejected":
                    print(
                        f"App validation: rejected ({validation['reason']}); raw reply shown.",
                        flush=True,
                    )
            else:
                failed = True
                reason = row.get("failure", "no_reply")
                print(f"Question failed: {reason}", flush=True)
                fatal = "timeout" in reason or "unreachable" in reason
            if args.trace:
                print(json.dumps(row, ensure_ascii=False, indent=2), flush=True)
            print(f"{row['duration_ms']:.0f} ms", flush=True)
            if args.question is not None or fatal:
                break
    finally:
        await server.aclose()
    return 1 if fatal or (args.question is not None and failed) else 0


def main():
    if isinstance(sys.stdout, io.TextIOWrapper):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("gpu", "cpu"), default="gpu")
    parser.add_argument("--language", choices=("en", "tr"), default="en")
    parser.add_argument("--port", type=int, default=18197)
    parser.add_argument("--trace", action="store_true", help="Show calls, results and validation")
    parser.add_argument("--raw", action="store_true", help="Original unfiltered comparison mode")
    parser.add_argument("--question", help="One question instead of interactive chat")
    parser.add_argument("--fixture", type=Path,
                        help="Normalized fixture/recording directory; default: telemetry demo")
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    if args.question is not None and (
        not args.question.strip() or args.question.strip().startswith("/")
    ):
        parser.error("--question requires a nonempty driver message")
    try:
        return asyncio.run(chat(args))
    except KeyboardInterrupt:
        print("Chat stopped.")
        return 130
    except (OSError, RuntimeError, ValueError) as error:
        print(f"Demo failed: {error}", file=sys.stderr)
        if args.backend == "gpu" and any(
            word in str(error).lower() for word in ("accelerated", "gpu", "cuda", "device", "probe")
        ):
            print("For an explicit CPU trial, add --backend cpu.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
