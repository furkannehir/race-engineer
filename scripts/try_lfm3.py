"""English-first LFM3 fixture chat and controlled comparison through real Context/Core."""

import argparse
import asyncio
import json
import time
from collections import deque
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

from race_engineer.application.intelligence_cli import _contexts
from race_engineer.config import load_config
from race_engineer.conversation.runtime import ConversationServer, ConversationTrialOptions
from race_engineer.core.intelligence import DriverTurn
from race_engineer.evaluation.intelligence_latency import (
    measure_intelligence_turn,
    summarize_latency,
)
from race_engineer.evaluation.system import file_sha256, files_fingerprint, gpu_memory_snapshot
from race_engineer.fixtures import load_fixture
from race_engineer.intelligence.capabilities import DeterministicRaceCapabilities
from race_engineer.intelligence.context_engine import QueryDrivenContextEngineer
from race_engineer.intelligence.context_planner import QwenContextQueryPlanner
from race_engineer.intelligence.factory import live_intelligence
from race_engineer.intelligence.grounding import StrictEvidenceGrounder
from race_engineer.intelligence.local_model import LocalJsonModel
from race_engineer.intelligence.orchestrator import EngineerOrchestrator
from race_engineer.intelligence.portable_engineer import PortableQwenEngineer
from race_engineer.intelligence.telemetry_memory import BoundedTelemetryMemory
from race_engineer.tts.chatterbox import ChatterboxSpeechBackend

if __package__:
    from .benchmark_qwen_cache import load_cases, new_output, parse_runtime_log, require_free_port
    from .lfm3_prompt import PROMPT_ID, EnglishPlannerClient
else:
    from benchmark_qwen_cache import load_cases, new_output, parse_runtime_log, require_free_port
    from lfm3_prompt import PROMPT_ID, EnglishPlannerClient

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "data/trials/lfm3-gpu.toml"


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def english_cases(config, full_suite):
    cases, inputs = load_cases(SimpleNamespace(case=None, full_suite=True, validation=True), config)
    selected = [case for case in cases if case[2].language == "en"]
    if not full_suite:
        seen = set()
        unique = []
        for case in selected:
            if case[2].id not in seen:
                seen.add(case[2].id)
                unique.append(case)
        selected = unique
    return selected, inputs


def model_config(path):
    if not path.resolve().is_relative_to(ROOT):
        raise ValueError("Trial configs must be inside the repository")
    config = load_config(path)
    runtime = config.conversation.runtime
    if runtime.reuse_external_server or runtime.fallback_to_cpu:
        raise ValueError("Trial requires an owned runtime and an explicit CPU/GPU selection")
    if runtime.context_size != 8192 or runtime.parallel != 1:
        raise ValueError("Keep the controlled 8192-token, single-slot comparison")
    if (
        config.conversation.model.startswith("LiquidAI/LFM3")
        and runtime.enable_thinking is not False
    ):
        raise ValueError("English-first LFM trial requires enable_thinking = false")
    return config


async def compare(args):
    config = model_config(args.config)
    cases, inputs = english_cases(config, args.full_suite)
    output = new_output(
        args.output
        or ROOT / ("data/benchmarks/lfm3-" + datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ"))
    )
    profiles = [("lfm3", args.config)]
    if args.baseline:
        profiles.append(("qwen-baseline", args.baseline))
    # Validate all profiles before starting either model or optional shared TTS.
    configs = [(label, model_config(path), path) for label, path in profiles]
    if any(c.tts != config.tts for _, c, _ in configs):
        raise ValueError("Comparison profiles must share identical TTS settings")
    speech = ChatterboxSpeechBackend(config.tts) if args.with_tts else None
    report = {
        "status": "running",
        "language": "en",
        "turkish_evaluated": False,
        "training_performed": False,
        "planner_schema_policy": config.conversation.context_planner,
        "core_reply_interface": "schema_constrained_json",
        "lfm_planner_prompt": (
            "lfm3-native-v1"
            if config.conversation.context_planner == "lfm3-native-v1"
            else PROMPT_ID
            if args.compact_planner
            else "unchanged-v8"
        ),
        "input_fingerprint": files_fingerprint(ROOT, inputs),
        "cases_per_model": len(cases),
        "repeats": args.repeats,
        "tts_resident": args.with_tts,
        "models": {},
        "limitations": [
            "Synthetic typed input, no STT, game/VR or audio playback.",
            "Whole-device memory samples are not isolated process peaks.",
            "Planner/evidence matches do not establish natural-language accuracy.",
            "Models require different runtime versions; timing is not a pure size comparison.",
        ],
    }
    try:
        if speech:
            print("Loading one shared Chatterbox worker for memory comparison...", flush=True)
            await speech.start()
            report["tts_worker_pid"] = speech.worker_pid
        for label, current, path in configs:
            conv = current.conversation
            if conv.context_planner not in {"qwen-context-v8", "lfm3-native-v1"}:
                raise ValueError("Use v8 or the explicit native LFM policy for comparison")
            require_free_port(conv.port)
            log = output / f"{label}-server.log"
            server = ConversationServer(
                ROOT, conv, trial=ConversationTrialOptions("auto", "f16", log)
            )
            data = {
                "model": conv.model,
                "planner_prompt": (
                    PROMPT_ID if args.compact_planner and label == "lfm3" else conv.context_planner
                ),
                "runtime": conv.runtime.model_dump(mode="json"),
                "config": str(path.relative_to(ROOT)),
                "results": [],
                "model_sha256": file_sha256(ROOT / conv.runtime.model_path),
                "gpu_before": gpu_memory_snapshot(),
            }
            report["models"][label] = data
            print(f"Starting {label}: {conv.model}...", flush=True)
            try:
                await server.start()
                assert server.process is not None and server.effective is not None
                data["effective_backend"] = server.effective.backend
                if conv.runtime.mode == "gpu" and server.effective.backend != "cuda":
                    raise ValueError("Requested CUDA was not actually selected")
                data["worker_pid"] = server.process.pid
                data["gpu_ready"] = gpu_memory_snapshot()
                data["allocations"] = parse_runtime_log(
                    log.read_text(encoding="utf-8", errors="replace")
                )
                with (output / f"{label}-review.jsonl").open("x", encoding="utf-8") as review:
                    for repeat in range(args.repeats):
                        for index, (dataset, contexts, group, question) in enumerate(cases, 1):
                            print(
                                f"[{label}] {repeat + 1}/{args.repeats} "
                                f"{index}/{len(cases)} {group.id}",
                                flush=True,
                            )
                            result, answer = await measure_intelligence_turn(
                                conv,
                                contexts,
                                group,
                                question,
                                run_index=index,
                                repetition=repeat + 1,
                                planner_id=conv.context_planner,
                                planner_client=(
                                    EnglishPlannerClient(conv)
                                    if args.compact_planner and label == "lfm3"
                                    else None
                                ),
                            )
                            result["dataset"] = dataset
                            data["results"].append(result)
                            data["summary"] = summarize_latency(data["results"])
                            review.write(
                                json.dumps(
                                    {
                                        "case": group.id,
                                        "question": question,
                                        "history": group.recent_dialogue,
                                        "reply": answer,
                                    },
                                    ensure_ascii=False,
                                )
                                + "\n"
                            )
                            review.flush()
                            save(output / "report.json", report)
                            print(
                                f"  {result['timing_ms']['total']:.0f} ms; "
                                f"plan={result['planner_exact_match']}; "
                                f"failure={result['failure']}",
                                flush=True,
                            )
                            if result["failure"] and any(
                                reason in str(result["failure"])
                                for reason in ("timeout", "unreachable")
                            ):
                                raise RuntimeError(
                                    "Transport failed; stopping to avoid stacked inference"
                                )
                data["gpu_after_turns"] = gpu_memory_snapshot()
                data["allocations"] = parse_runtime_log(
                    log.read_text(encoding="utf-8", errors="replace")
                )
            finally:
                await server.aclose()
                data["server_closed"] = server.process is None
        report["status"] = "completed"
    except BaseException as error:
        report["status"] = "failed"
        report["error"] = str(error)
        raise
    finally:
        if speech:
            await speech.aclose()
            report["tts_closed"] = speech.worker_pid is None
        save(output / "report.json", report)
        print(f"Report: {output / 'report.json'}", flush=True)


@asynccontextmanager
async def owned_server(config):
    server = ConversationServer(ROOT, config)
    try:
        await server.start()
        yield server
    finally:
        await server.aclose()


async def chat(args):
    config = model_config(args.config)
    if config.conversation.model != "LiquidAI/LFM3-1.7B":
        raise ValueError("This interactive trial must explicitly select LFM3")
    require_free_port(config.conversation.port)
    contexts = _contexts(load_fixture(args.fixture), config.policy.context)
    memory = BoundedTelemetryMemory()
    for context in contexts:
        memory.update(context)
    frame = memory.latest_context().frame
    if args.compact_planner:
        model = LocalJsonModel(config.conversation)
        planner = QwenContextQueryPlanner(
            config.conversation,
            model=EnglishPlannerClient(config.conversation),
            planner_id=config.conversation.context_planner,
        )
        context = QueryDrivenContextEngineer(memory, planner, DeterministicRaceCapabilities(memory))
        core = PortableQwenEngineer(config.conversation, model=model)
        engineer = EngineerOrchestrator(context, core, core, StrictEvidenceGrounder())
    else:
        engineer = live_intelligence(config.conversation, memory)
    history = deque(maxlen=config.conversation.history_entry_limit)
    single_question_failed = False
    print(
        f"Starting {config.conversation.model} ({config.conversation.runtime.mode})...", flush=True
    )
    async with owned_server(config.conversation):
        print(
            f"LFM3 English trial | {frame.session_id}, sequence {frame.sequence}. "
            "Synthetic telemetry; /clear resets history, /quit exits.",
            flush=True,
        )
        index = 0
        while True:
            try:
                question = args.question or await asyncio.to_thread(input, "Driver (en)> ")
            except EOFError:
                break
            if question.strip() in {"/quit", "/exit"}:
                break
            if question.strip() == "/clear":
                history.clear()
                continue
            if not question.strip():
                continue
            index += 1
            started = time.perf_counter()
            turn = DriverTurn(
                turn_id=f"lfm-chat-{index}",
                transcript=question.strip(),
                received_at=datetime.now(UTC),
                session_id=frame.session_id,
                generation=0,
                asr_language="en",
                reply_language="en",
                recent_dialogue=tuple(history),
            )
            try:
                response = await engineer.respond(turn)
                print(f"Engineer: {response.text or '[silence]'}", flush=True)
                if response.text:
                    history.extend((f"driver: {question.strip()}", f"engineer: {response.text}"))
            except Exception as error:
                single_question_failed = True
                print(f"Question failed: {error}", flush=True)
                if any(reason in str(error) for reason in ("timeout", "unreachable")):
                    break
            print(f"{(time.perf_counter() - started) * 1000:.0f} ms", flush=True)
            if args.question:
                break
    return 1 if args.question and single_question_failed else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument(
        "--native-tools",
        action="store_true",
        help="Use the separately configured native LFM tool adapter",
    )
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument(
        "--compact-planner",
        action="store_true",
        help="Explicit shorter English planner prompt, same v8 schema/grounding",
    )
    parser.add_argument(
        "--baseline", type=Path, help="Optional explicit Qwen profile, run sequentially"
    )
    parser.add_argument(
        "--with-tts", action="store_true", help="Keep shared speech resident, no playback"
    )
    parser.add_argument(
        "--full-suite", action="store_true", help="All EN paraphrases, not just one/group"
    )
    parser.add_argument("--repeats", type=int, default=1, choices=range(1, 6))
    parser.add_argument("--output", type=Path, help="New directory under data/benchmarks")
    parser.add_argument(
        "--fixture", type=Path, default=ROOT / "fixtures/synthetic/intelligence-capabilities"
    )
    parser.add_argument("--question", help="Single question instead of interactive chat")
    args = parser.parse_args()
    if args.config is None:
        args.config = (
            ROOT / "data/trials/lfm3-native-gpu.toml" if args.native_tools else DEFAULT_CONFIG
        )
    if args.native_tools and args.compact_planner:
        parser.error("--native-tools and --compact-planner use different planner interfaces")
    if (
        args.native_tools
        and load_config(args.config).conversation.context_planner != "lfm3-native-v1"
    ):
        parser.error("--native-tools requires an explicit lfm3-native-v1 profile")
    if args.question is not None and (
        not args.question.strip() or args.question.strip().startswith("/")
    ):
        parser.error("--question requires a nonempty driver question")
    if args.benchmark and args.question is not None:
        parser.error("Choose either --benchmark or --question")
    if not args.benchmark and (args.baseline or args.with_tts or args.full_suite or args.output):
        parser.error("--baseline, --with-tts, --full-suite and --output require --benchmark")
    if not args.config.is_absolute():
        args.config = ROOT / args.config
    if args.baseline and not args.baseline.is_absolute():
        args.baseline = ROOT / args.baseline
    if args.output and not args.output.is_absolute():
        args.output = ROOT / args.output
    try:
        result = asyncio.run(compare(args) if args.benchmark else chat(args))
        if result is not None:
            raise SystemExit(result)
    except (ValueError, RuntimeError, OSError) as error:
        parser.error(str(error))
    except KeyboardInterrupt:
        print("Trial stopped.")


if __name__ == "__main__":
    main()
