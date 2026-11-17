"""Compare raw native tool conversations, independently of application acceptance.

Runs typed synthetic telemetry only: no microphone, speech, UI or live model changes.
Raw completed turns remain in history even when the English application validator
rejects them. Tool choice is scored; speech correctness requires human review.
"""

import argparse
import asyncio
import json
import statistics
import time
from collections import deque
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path

from race_engineer.application.intelligence_cli import _contexts
from race_engineer.config import load_config
from race_engineer.conversation.runtime import ConversationServer, ConversationTrialOptions
from race_engineer.core.intelligence import DriverTurn
from race_engineer.evaluation.system import (
    file_sha256,
    files_fingerprint,
    gpu_memory_snapshot,
    process_metrics,
)
from race_engineer.fixtures import load_fixture
from race_engineer.intelligence.telemetry_memory import BoundedTelemetryMemory
from race_engineer.radio_diagnostics import error_code

if __package__:
    from .benchmark_qwen_cache import parse_runtime_log, require_free_port
    from .lfm3_dialogue import SYSTEM, ChatReply, LocalDialogueModel, NativeToolDialogue
    from .setup_dialogue_comparison import ASSETS, MODELS
else:
    from benchmark_qwen_cache import parse_runtime_log, require_free_port
    from lfm3_dialogue import SYSTEM, ChatReply, LocalDialogueModel, NativeToolDialogue
    from setup_dialogue_comparison import ASSETS, MODELS

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "fixtures/intelligence-evaluation/dialogue-model-comparison.json"
RUNTIME = ROOT / "data/lfm3-prototype"
CANDIDATES = {
    "lfm": {
        "model": "LiquidAI/LFM3-1.7B",
        "path": RUNTIME / "LFM3-1.7B-Q4_K_M.gguf",
        "revision": "cd0c74c575b5c416f3011b5b4c298077f26ca8c2",
        "sha256": "3793e1e6e2266919d6e015287221b41100175a98eddaba92e111aed3b05c93c4",
        "generation": {"temperature": 0},
    },
    "qwen4": {
        "model": "Qwen3-4B-Instruct-2507",
        "path": ROOT / "data/conversation-prototype/Qwen3-4B-Instruct-2507-Q4_K_M.gguf",
        "revision": "a06e946bb6b655725eafa393f4a9745d460374c9",
        "sha256": "3605803b982cb64aead44f6c1b2ae36e3acdb41d8e46c8a94c6533bc4c67e597",
        "generation": {"temperature": 0},
    },
    "qwen35": {
        "model": "Qwen/Qwen3.5-2B",
        "path": ASSETS / MODELS["qwen35"].name,
        "revision": MODELS["qwen35"].revision,
        "sha256": MODELS["qwen35"].sha256,
        "generation": {"temperature": 1.0, "top_p": 1.0, "top_k": 20,
                       "min_p": 0.0, "presence_penalty": 2.0, "repeat_penalty": 1.0},
    },
    "ministral": {
        "model": "mistralai/Ministral-3-3B-Instruct-2512",
        "path": ASSETS / MODELS["ministral"].name,
        "revision": MODELS["ministral"].revision,
        "sha256": MODELS["ministral"].sha256,
        "generation": {"temperature": 0.1},
    },
}
CANDIDATES["lfm-preset"] = {
    **CANDIDATES["lfm"],
    "generation": {"temperature": 0.1, "top_k": 50, "repeat_penalty": 1.1},
}
CANDIDATES["qwen35-greedy"] = {
    **CANDIDATES["qwen35"], "generation": {"temperature": 0},
}


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


class RawConversation:
    """No application-driven repairs: preserve raw responses and score separately."""

    def __init__(self, model, tools, *, history_turns=6):
        self.model, self.tools = model, tools
        self.history = deque(maxlen=history_turns)

    def clear(self):
        self.history.clear()

    async def respond(self, question, language, turn_id):
        system = SYSTEM.replace("in English.", "in Turkish." if language == "tr" else "in English.")
        # No English-only metric formatting rule is imposed on Turkish speech.
        system += f"\nReply only in {'Turkish' if language == 'tr' else 'English'}."
        current = [{"role": "user", "content": question}]
        messages = [{"role": "system", "content": system}]
        for group in self.history:
            messages.extend(group)
        messages.extend(current)
        row = {"question": question, "language": language, "turn_id": turn_id,
               "requests": [], "selected_tools": [], "raw_reply": None}
        started = time.perf_counter()

        async def request(choice):
            stamp = time.perf_counter()
            entry = {"tool_choice": choice}
            row["requests"].append(entry)
            try:
                reply = await self.model.request(messages, tool_choice=choice)
                entry["reply"] = reply.message()
                return reply
            finally:
                entry["duration_ms"] = round((time.perf_counter() - stamp) * 1000, 3)
                entry["metadata"] = getattr(self.model, "last_metadata", {})

        evidence, unknowns = (), ()
        try:
            reply = await request("auto")
            row["selected_tools"] = [call.name for call in reply.calls]
            if reply.calls:
                results, evidence, unknowns = self.tools._tools(reply.calls)
                row["tool_results"] = results
                current.extend([reply.message(), *results])
                messages.extend([reply.message(), *results])
                reply = await request("none")
                if reply.calls:
                    raise ValueError("comparison_second_tool_batch")
            if not isinstance(reply, ChatReply) or not reply.content:
                raise ValueError("comparison_empty_reply")
            row["raw_reply"] = reply.content
            # Diagnostic history is independent of application acceptance; otherwise
            # validator defects silently change the conversation seen by the model.
            current.append(reply.message())
            self.history.append(current)
        except Exception as error:
            row["failure"] = error_code(error)
        row["duration_ms"] = round((time.perf_counter() - started) * 1000, 3)
        row["application_validation"] = {
            "status": "not_applicable_turkish" if language == "tr" else "no_completed_reply"
        }
        if language == "en" and row["raw_reply"] is not None:
            frame = self.tools.memory.latest_context().frame
            turn = DriverTurn(
                turn_id=turn_id, transcript=question, session_id=frame.session_id,
                received_at=datetime.now(UTC), generation=0, asr_language="en", reply_language="en",
            )
            try:
                grounded = self.tools._ground(turn, row["raw_reply"], evidence, unknowns)
                row["application_validation"] = {"status": "accepted", "text": grounded.text}
            except Exception as error:
                row["application_validation"] = {"status": "rejected", "reason": error_code(error)}
        return row


def summarize(rows):
    result = {}
    for language in ("en", "tr"):
        selected = [row for row in rows if row["language"] == language]
        times = sorted(row["duration_ms"] for row in selected)
        if not times:
            continue
        result[language] = {
            "turns": len(selected),
            "completed": sum(row["raw_reply"] is not None for row in selected),
            "tool_matches": sum(row["tool_match"] for row in selected),
            "typed_median_ms": round(statistics.median(times), 3),
            "typed_p95_ms": times[min(len(times) - 1, int((len(times) - 1) * 0.95 + 0.5))],
            "model_calls": sum(len(row["requests"]) for row in selected),
            "application_rejections": sum(
                row["application_validation"]["status"] == "rejected" for row in selected
            ),
            "automatic_speech_accuracy": None,
        }
    return result


def configuration(candidate, backend, port):
    config = load_config(ROOT / "config/default.toml", environ={})
    data = config.conversation.model_dump()
    data.update(model=candidate["model"], port=port, timeout_s=60.0)
    data["runtime"].update(
        mode=backend,
        cpu_executable_path=RUNTIME / "llama-b12088-cpu/llama-server.exe",
        accelerated_executable_path=RUNTIME / "llama-b12088-cuda-13.4/llama-server.exe",
        accelerated_backend="cuda", device="CUDA0", model_path=candidate["path"],
        runtime_revision="b12088", model_revision=candidate["revision"],
        quantization="Q4_K_M", context_size=8192, parallel=1,
        gpu_layers=-1, enable_thinking=False, fallback_to_cpu=False, reuse_external_server=False,
    )
    return config.conversation.model_validate(data), config.policy.context


async def run_candidate(name, backend, args, dataset, output):
    candidate = CANDIDATES[name]
    if file_sha256(candidate["path"]) != candidate["sha256"]:
        raise ValueError(f"Model checksum mismatch: {name}")
    config, context_settings = configuration(candidate, backend, args.port)
    require_free_port(config.port)
    memory = BoundedTelemetryMemory()
    for context in _contexts(load_fixture(ROOT / dataset["fixture"]), context_settings):
        memory.update(context)
    executor = NativeToolDialogue(config, memory)
    prefix = f"{name}-{backend}"
    log = output / f"{prefix}-server.log"
    server = ConversationServer(ROOT, config, trial=ConversationTrialOptions("on", "f16", log))
    report = {
        "model": candidate["model"], "model_sha256": candidate["sha256"],
        "config": config.model_dump(mode="json"), "generation": candidate["generation"],
        "gpu_before": gpu_memory_snapshot(), "rows": [],
    }
    gpu_samples = []
    stop = asyncio.Event()

    async def sample_gpu():
        while not stop.is_set():
            gpu_samples.append(await asyncio.to_thread(gpu_memory_snapshot))
            with suppress(TimeoutError):
                await asyncio.wait_for(stop.wait(), timeout=0.5)

    sampler = asyncio.create_task(sample_gpu()) if backend == "gpu" else None
    try:
        await server.start()
        if server.process is None or server.effective.backend != backend.replace("gpu", "cuda"):
            raise ValueError("Comparison requires an owned server on the requested backend")
        report["gpu_ready"] = gpu_memory_snapshot()
        report["pid"] = server.process.pid
        for repeat in range(args.repeats):
            for sequence in dataset["sequences"]:
                if args.cpu_short and backend == "cpu" and sequence["id"] not in {
                    "reported-gaps", "gaps-tr",
                }:
                    continue
                # Same seed per repetition for every candidate; never tune on a failed question.
                generation = {"max_tokens": 256, "seed": 7 + repeat, **candidate["generation"]}
                dialogue = RawConversation(
                    LocalDialogueModel(config, generation=generation), executor
                )
                for index, case in enumerate(sequence["turns"]):
                    row = await dialogue.respond(
                        case["question"], sequence["language"],
                        f"{prefix}-{repeat}-{sequence['id']}-{index}",
                    )
                    row.update(sequence=sequence["id"], repeat=repeat + 1,
                               expected_tool=case["tool"])
                    row["tool_match"] = (
                        row["selected_tools"] == ([] if case["tool"] is None else [case["tool"]])
                        and "failure" not in row
                    )
                    report["rows"].append(row)
                    save(output / f"{prefix}.json", report)
                    print(json.dumps({
                        "candidate": prefix, "n": len(report["rows"]),
                        "question": row["question"], "tools": row["selected_tools"],
                        "tool_match": row["tool_match"], "reply": row["raw_reply"],
                        "ms": row["duration_ms"], "failure": row.get("failure"),
                    }, ensure_ascii=True), flush=True)
                    if row.get("failure") in {"dialogue_timeout", "dialogue_unreachable"}:
                        raise RuntimeError("Stopped after transport failure; no queued inference")
        report["process"] = process_metrics(server.process.pid)
    except Exception as error:
        report["failure"] = error_code(error)
        print(f"{prefix} stopped: {report['failure']}", flush=True)
    finally:
        stop.set()
        if sampler:
            await sampler
        await server.aclose()
        report["server_closed"] = server.process is None
        report["gpu_after_close"] = gpu_memory_snapshot()
        report["allocations"] = parse_runtime_log(
            log.read_text(encoding="utf-8", errors="replace") if log.is_file() else ""
        )
        report["gpu_samples"] = gpu_samples
        report["summary"] = summarize(report["rows"])
        save(output / f"{prefix}.json", report)
    return report


async def main_async(args):
    output = args.output.resolve()
    if not output.is_relative_to(ROOT / "data") or output.exists():
        raise ValueError("Choose a NEW output directory under data/")
    dataset = json.loads(DATASET.read_text(encoding="utf-8"))
    output.mkdir(parents=True)
    sources = [
        Path(__file__), ROOT / "scripts/lfm3_dialogue.py",
        ROOT / "scripts/setup_dialogue_comparison.py", DATASET,
        *ROOT.glob("src/race_engineer/**/*.py"),
    ]
    before = files_fingerprint(ROOT, sources)
    summary = {
        "schema": "native-dialogue-comparison.v1", "dataset": dataset["revision"],
        "fingerprint": before, "live_config_changed": False,
        "scope": "typed_synthetic_raw_native_tools_no_repairs",
        "limitations": [
            "Development examples, not held-out acceptance or race accuracy.",
            "Tool match does not establish relevant, correct, natural speech.",
            "English validator acceptance is diagnostic; Turkish validator is not implemented.",
            "Whole-device GPU samples include unrelated apps and can miss brief peaks.",
            "No STT, TTS, audio, microphone, UI or VR/game load tested.",
            "Per-model sampling presets differ; results compare usable configurations.",
        ], "runs": {},
    }
    for name in args.candidate or CANDIDATES:
        for backend in args.backend:
            print(f"Starting {name} on {backend}...", flush=True)
            report = await run_candidate(name, backend, args, dataset, output)
            summary["runs"][f"{name}-{backend}"] = {
                "summary": report["summary"], "failure": report.get("failure"),
                "allocations": report["allocations"], "server_closed": report["server_closed"],
            }
            summary["source_unchanged"] = files_fingerprint(ROOT, sources) == before
            save(output / "summary.json", summary)
            if not summary["source_unchanged"]:
                raise RuntimeError("Source changed during comparison; results are not comparable")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", action="append", choices=tuple(CANDIDATES))
    parser.add_argument("--backend", nargs="+", choices=("cpu", "gpu"), default=["gpu"])
    parser.add_argument("--repeats", type=int, choices=range(1, 4), default=1)
    parser.add_argument("--cpu-short", action="store_true", help="Two six-turn CPU conversations")
    parser.add_argument("--port", type=int, default=18196)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
