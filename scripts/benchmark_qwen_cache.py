"""Offline FP16/Q8 KV-cache trial with live v7 or an explicit experimental v8 candidate.

No live config, prompts, telemetry, STT or TTS is changed. Only owned servers on
a separate loopback port are used. Authored fixture questions/replies are saved
locally for review; no microphone or real-session text is collected.
"""

import argparse
import asyncio
import http.client
import json
import random
import re
import socket
import time
from datetime import UTC, datetime
from pathlib import Path

from race_engineer.application.intelligence_cli import _contexts
from race_engineer.config import load_config
from race_engineer.conversation.runtime import ConversationServer, ConversationTrialOptions
from race_engineer.evaluation.intelligence import load_intelligence_evaluation_dataset
from race_engineer.evaluation.intelligence_latency import (
    measure_intelligence_turn,
    summarize_latency,
)
from race_engineer.evaluation.system import (
    file_sha256,
    files_fingerprint,
    git_dirty,
    git_revision,
    gpu_memory_snapshot,
    machine_metadata,
    runtime_version,
)
from race_engineer.fixtures import load_fixture
from race_engineer.intelligence.context_planner import PLANNER_V7_ID, PLANNER_V8_ID
from race_engineer.intelligence.limitations import UNSUPPORTED_DATA_POLICY

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "data/benchmarks"
DATASETS = (
    ROOT / "fixtures/intelligence-evaluation/development-understanding.json",
    ROOT / "fixtures/intelligence-evaluation/development-controls.json",
)
DEFAULT_CASES = tuple(
    f"{case}-{language}"
    for language in ("en", "tr")
    for case in (
        "understand-radio-check",
        "understand-short-position",
        "understand-vent",
        "understand-starting-grid",
        "understand-target-correction",
        "understand-topic-change",
        "compound",
        "future",
        "historical-raw",
        "followup",
    )
)


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def new_output(path):
    path = path.resolve()
    if not path.is_relative_to(OUTPUT_ROOT.resolve()) or path == OUTPUT_ROOT.resolve():
        raise ValueError("output must be a NEW subdirectory of data/benchmarks")
    path.mkdir(parents=True, exist_ok=False)
    return path


def require_free_port(port):
    with socket.socket() as connection:
        if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            connection.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        try:
            connection.bind(("127.0.0.1", port))
        except OSError as error:
            raise ValueError(
                f"Trial port {port} is occupied; existing servers are left untouched"
            ) from error


def parse_runtime_log(text):
    """Read effective backend allocations, not just requested command-line flags."""
    result = {"cuda_model_mib": 0.0, "cuda_kv_mib": 0.0, "cuda_compute_mib": 0.0}
    proof = []
    for line in text.splitlines():
        lowered = line.lower()
        if match := re.search(r"\bn_ctx\s*=\s*(\d+)", line):
            result["context_size"] = int(match[1])
        if "flash attention enabled" in lowered:
            result["flash_attention_enabled"] = True
        elif "flash attention disabled" in lowered:
            result["flash_attention_enabled"] = False
        elif match := re.search(r"flash_attn\s*=\s*(enabled|disabled|on|off|1|0)\b", lowered):
            result["flash_attention_enabled"] = match[1] in {"enabled", "on", "1"}
        if match := re.search(r"K\s*\((\w+)\):.*V\s*\((\w+)\):", line):
            result["cache_k"], result["cache_v"] = match[1], match[2]
        if match := re.search(r"offloaded\s+(\d+)/(\d+)\s+layers", line):
            result["offloaded_layers"], result["total_layers"] = int(match[1]), int(match[2])
        if match := re.search(
            r"CUDA\d+\s+(model|KV|compute) buffer size\s*=\s*([\d.]+)\s*MiB", line
        ):
            key = f"cuda_{match[1].lower()}_mib"
            result[key] += float(match[2])
        if any(
            word in lowered
            for word in (
                "buffer size",
                "llama_kv_cache: size",
                "flash_attn",
                "flash attention",
                "offloaded",
            )
        ):
            proof.append(line)
    result["proof_lines"] = proof
    return result


def validate_runtime(observed, cache_type, context_size, require_flash=True):
    if observed.get("context_size") != context_size:
        raise ValueError("effective context size was not verified")
    if (observed.get("cache_k"), observed.get("cache_v")) != (cache_type, cache_type):
        raise ValueError("effective K/V cache precision was not verified")
    if observed.get("flash_attention_enabled") is not True and require_flash:
        raise ValueError("Flash Attention was not verified enabled")
    if (
        not observed.get("total_layers")
        or observed.get("offloaded_layers") != observed["total_layers"]
    ):
        raise ValueError("full GPU offload was not verified")
    if observed.get("cuda_kv_mib", 0) <= 0 or observed.get("cuda_model_mib", 0) <= 0:
        raise ValueError("CUDA model/cache allocation was not verified")


def runtime_properties(port):
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    try:
        connection.request("GET", "/props")
        response = connection.getresponse()
        if response.status != 200:
            raise ValueError("runtime properties unavailable")
        value = json.loads(response.read(262_145))
        settings = value["default_generation_settings"]
        return {
            "context_size": settings["n_ctx"],
            "seed": settings["params"]["seed"],
            "parallel": value["total_slots"],
            "model": value["model_alias"],
            "build_info": value.get("build_info"),
        }
    finally:
        connection.close()


def load_cases(args, config):
    selected = set(args.case or (() if args.full_suite else DEFAULT_CASES))
    cases, found, inputs = [], set(), []
    datasets = DATASETS + (
        (
            ROOT / "fixtures/intelligence-evaluation/development-understanding-validation.json",
            ROOT / "fixtures/intelligence-evaluation/development-temporal-validation.json",
            ROOT / "fixtures/intelligence-evaluation/development-unsupported-validation.json",
        )
        if getattr(args, "validation", False)
        else ()
    )
    for path in datasets:
        dataset = load_intelligence_evaluation_dataset(path)
        if dataset.split != "development":
            raise ValueError("cache benchmark uses only development fixtures")
        fixture_path = (ROOT / dataset.fixture).resolve()
        if not fixture_path.is_relative_to(ROOT / "fixtures/synthetic"):
            raise ValueError("cache benchmark permits only synthetic fixture data")
        contexts = _contexts(load_fixture(fixture_path), config.policy.context)
        inputs.extend((path, *(p for p in fixture_path.rglob("*") if p.is_file())))
        for group in dataset.groups:
            if selected and group.id not in selected:
                continue
            found.add(group.id)
            for question in group.questions if args.full_suite else group.questions[:1]:
                cases.append((dataset.dataset_id, contexts, group, question))
    if selected - found:
        raise ValueError(f"Unknown benchmark cases: {sorted(selected - found)}")
    if not cases:
        raise ValueError("no benchmark cases selected")
    return cases, tuple(inputs)


def trial_config(config, port, planner=None):
    candidate = planner or config.conversation.context_planner
    if candidate not in {PLANNER_V7_ID, PLANNER_V8_ID}:
        raise ValueError("This comparison requires a v7 or experimental v8 planner")
    runtime = config.conversation.runtime
    if runtime.mode != "gpu" or runtime.accelerated_backend != "cuda" or runtime.gpu_layers != -1:
        raise ValueError("This comparison requires the configured fully-offloaded CUDA Qwen")
    if runtime.accelerated_executable_path is None:
        raise ValueError("The local CUDA runtime path is not configured")
    if runtime.context_size != 8192 or runtime.parallel != 1:
        raise ValueError("Keep the 8192-token, single-slot baseline for this comparison")
    runtime = runtime.model_copy(update={"fallback_to_cpu": False, "reuse_external_server": False})
    return config.conversation.model_copy(
        update={"port": port, "runtime": runtime, "context_planner": candidate}
    )


async def verified_server(server, log_path, cache_type, config, seed):
    await server.start()
    if server.process is None or server.effective is None or server.effective.backend != "cuda":
        raise ValueError("Trial requires an owned CUDA process; external servers are forbidden")
    properties = await asyncio.to_thread(runtime_properties, config.port)
    if (
        properties["context_size"],
        properties["parallel"],
        properties["seed"],
        properties["model"],
    ) != (
        8192,
        1,
        seed,
        config.model,
    ):
        raise ValueError("Runtime properties do not match the controlled trial")
    for attempt in range(10):
        observed = parse_runtime_log(log_path.read_text(encoding="utf-8", errors="replace"))
        try:
            validate_runtime(observed, cache_type, config.runtime.context_size)
            return observed, properties
        except ValueError:
            if attempt == 9:
                raise
            await asyncio.sleep(0.1)


async def run_trial(args, config, cases, output, cache_type, round_index, *, preflight=False):
    label = "preflight-auto" if preflight else f"round-{round_index:02d}-{cache_type}"
    directory = output / label
    directory.mkdir()
    log_path = directory / "runtime.log"
    options = ConversationTrialOptions(
        "auto" if preflight else "on", cache_type, log_path, args.seed
    )
    server = ConversationServer(ROOT, config, trial=options)
    report = {
        "label": label,
        "cache_type": cache_type,
        "round": round_index,
        "status": "running",
        "results": [],
        "review_file": f"{label}/review.jsonl",
        "automatic_reply_accuracy": None,
        "gpu_samples": [],
    }
    path = directory / "result.json"

    async def sample(phase):
        report["gpu_samples"].append(
            {"phase": phase, **await asyncio.to_thread(gpu_memory_snapshot)}
        )

    async def exercise():
        require_free_port(config.port)
        print(f"[{label}] Starting owned CUDA server on {config.port}...", flush=True)
        await sample("before_server")
        started = time.perf_counter()
        observed, properties = await verified_server(
            server, log_path, cache_type, config, args.seed
        )
        report["startup_ms"] = (time.perf_counter() - started) * 1000
        report["effective_runtime"] = observed
        report["properties"] = properties
        report["pid"] = server.process.pid
        await sample("ready")
        print(
            f"[{label}] VERIFIED FA=on, K/V={cache_type}, KV={observed['cuda_kv_mib']:.1f} MiB",
            flush=True,
        )
        if preflight:
            return
        warmups = [
            next(case for case in cases if case[2].language == lang) for lang in ("en", "tr")
        ]
        for _, contexts, group, question in warmups:
            result, _ = await measure_intelligence_turn(
                config,
                contexts,
                group,
                question,
                run_index=0,
                repetition=0,
                planner_id=config.context_planner,
            )
            if result["failure"]:
                raise ValueError(f"warm-up failed: {result['failure']}")
        jobs = list(cases)
        random.Random(args.seed + round_index).shuffle(jobs)
        with (directory / "review.jsonl").open("x", encoding="utf-8") as review:
            for index, (dataset_id, contexts, group, question) in enumerate(jobs, 1):
                print(f"[{label}] {index}/{len(jobs)} {group.id}: processing...", flush=True)
                result, reply = await measure_intelligence_turn(
                    config,
                    contexts,
                    group,
                    question,
                    run_index=index,
                    repetition=round_index,
                    planner_id=config.context_planner,
                )
                result.update(
                    dataset_id=dataset_id,
                    call_state="after_bilingual_warmup",
                    history_entries=len(group.recent_dialogue),
                )
                report["results"].append(result)
                review.write(
                    json.dumps(
                        {
                            "case_id": group.id,
                            "question_id": result["question_id"],
                            "question": question,
                            "language": group.language,
                            "seeded_history": list(group.recent_dialogue),
                            "reply": reply,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                review.flush()
                report["summary"] = summarize_latency(report["results"])
                write_json(path, report)
                await sample(f"after_turn_{index}")
                print(
                    f"[{label}] {group.id}: {result['timing_ms']['total']:.0f} ms, "
                    f"plan={result['planner_exact_match']}, "
                    f"evidence={result['evidence_expectation_match']}",
                    flush=True,
                )
                failure = result["failure"]
                if failure and any(word in str(failure) for word in ("timeout", "unreachable")):
                    raise ValueError("transport failed; stopping to avoid stacking inference")
        report["status"] = "completed"

    try:
        await asyncio.wait_for(exercise(), args.timeout_s)
        report["status"] = "completed"
    except asyncio.CancelledError:
        report["status"] = "cancelled"
        raise
    except Exception as error:
        report["status"] = "failed"
        report["failure"] = {"type": type(error).__name__, "message": str(error)[:1500]}
        print(f"[{label}] FAILED: {report['failure']}", flush=True)
    finally:
        await server.aclose()
        await sample("after_server_close")
        report["summary"] = summarize_latency(report["results"])
        write_json(path, report)
    return report


def paired_comparison(trials):
    pairs, regressions, improvements, shared_failures = [], [], [], []
    rounds = sorted({trial["round"] for trial in trials})
    for round_index in rounds:
        selected = {trial["cache_type"]: trial for trial in trials if trial["round"] == round_index}
        if set(selected) != {"f16", "q8_0"} or any(
            t["status"] != "completed" for t in selected.values()
        ):
            continue
        left, right = selected["f16"], selected["q8_0"]
        for baseline, candidate in zip(left["results"], right["results"], strict=True):
            key = (baseline["case_id"], baseline["question_id"])
            if key != (candidate["case_id"], candidate["question_id"]):
                raise ValueError("trial question schedules differ")

            def passed(row):
                return all(
                    row[k]
                    for k in (
                        "pipeline_completed",
                        "planner_exact_match",
                        "evidence_expectation_match",
                    )
                )

            entry = {
                "round": round_index,
                "case_id": key[0],
                "question_id": key[1],
                "fp16_pass": passed(baseline),
                "q8_pass": passed(candidate),
            }
            pairs.append(entry)
            if entry["fp16_pass"] and not entry["q8_pass"]:
                regressions.append(entry)
            elif entry["q8_pass"] and not entry["fp16_pass"]:
                improvements.append(entry)
            elif not entry["fp16_pass"]:
                shared_failures.append(entry)
    summaries = {
        cache: summarize_latency(
            [row for trial in trials if trial["cache_type"] == cache for row in trial["results"]]
        )
        for cache in ("f16", "q8_0")
    }
    allocations = {
        trial["cache_type"]: trial["effective_runtime"]["cuda_kv_mib"]
        for trial in trials
        if "effective_runtime" in trial
    }
    return {
        "summaries": summaries,
        "paired_turns": len(pairs),
        "regressions": regressions,
        "improvements": improvements,
        "shared_failures": shared_failures,
        "verified_kv_saved_mib": allocations["f16"] - allocations["q8_0"]
        if len(allocations) == 2
        else None,
        "automatic_promotion": False,
        "manual_reply_review_required": True,
    }


async def benchmark(args):
    config = load_config(args.config)
    conversation = trial_config(config, args.port, args.planner)
    require_free_port(args.port)
    cases, inputs = load_cases(args, config)
    if {case[2].language for case in cases} != {"en", "tr"}:
        raise ValueError("Select both English and Turkish cases")
    source_files = (
        *ROOT.glob("src/race_engineer/**/*.py"),
        Path(__file__),
        ROOT / "scripts/benchmark_intelligence.py",
    )
    input_files = (args.config.resolve(), *inputs)
    source_hash = files_fingerprint(ROOT, source_files)
    input_hash = files_fingerprint(ROOT, input_files)
    executable = (ROOT / conversation.runtime.accelerated_executable_path).resolve()
    model = (ROOT / conversation.runtime.model_path).resolve()
    print("Fingerprinting the unchanged local Qwen model/runtime...", flush=True)
    model_hash = await asyncio.to_thread(file_sha256, model)
    output = new_output(args.output)
    report = {
        "schema_version": "qwen-cache-benchmark.v1",
        "created_at": datetime.now(UTC).isoformat(),
        "git_revision": git_revision(ROOT),
        "git_dirty": git_dirty(ROOT),
        "model": conversation.model,
        "model_sha256": model_hash,
        "runtime_version": await asyncio.to_thread(runtime_version, executable, 5),
        "runtime_exe_sha256": await asyncio.to_thread(file_sha256, executable),
        "planner": conversation.context_planner,
        "unsupported_data_policy": (
            UNSUPPORTED_DATA_POLICY if conversation.context_planner == PLANNER_V8_ID else None
        ),
        "config": conversation.model_dump(mode="json"),
        "machine": await asyncio.to_thread(machine_metadata),
        "seed": args.seed,
        "rounds": args.rounds,
        "questions_per_trial": len(cases),
        "source_sha256": source_hash,
        "inputs_sha256": input_hash,
        "trials": [],
        "limitations": [
            "Typed synthetic Context/Core/grounding path only; "
            "no STT, TTS, radio queue, playback or gameplay.",
            "Correctness checks cover expected plans/evidence, "
            "NOT automatic natural-language reply accuracy.",
            "History cases use authored previous dialogue, not a long self-generated conversation.",
            "V8 bypasses Core for fully unavailable answers; compare cache latency only "
            "within the same planner/policy and inspect per-turn model requests.",
            "Whole-device memory samples include other applications and are not measured peaks.",
            "Log buffer sizes exclude driver/context/graph overhead. No automatic promotion.",
        ],
    }
    print(f"Results: {output}", flush=True)
    report["preflight"] = await run_trial(
        args, conversation, cases, output, "f16", 0, preflight=True
    )
    if report["preflight"]["status"] == "completed":
        for round_index in range(1, args.rounds + 1):
            for cache in ("f16", "q8_0") if round_index % 2 else ("q8_0", "f16"):
                trial = await run_trial(args, conversation, cases, output, cache, round_index)
                report["trials"].append(trial)
                report["comparison"] = paired_comparison(report["trials"])
                write_json(output / "report.json", report)
                if trial["status"] != "completed":
                    break
            else:
                continue
            break
    report["source_changed_during_run"] = source_hash != files_fingerprint(ROOT, source_files)
    report["inputs_changed_during_run"] = input_hash != files_fingerprint(ROOT, input_files)
    report["valid_for_comparison"] = (
        not report["source_changed_during_run"]
        and not report["inputs_changed_during_run"]
        and report["preflight"]["status"] == "completed"
        and len(report["trials"]) == args.rounds * 2
        and all(t["status"] == "completed" for t in report["trials"])
    )
    write_json(output / "report.json", report)
    print(f"Report: {output / 'report.json'}\nLive Qwen configuration has NOT changed.", flush=True)
    return 0 if report["valid_for_comparison"] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config/qwen-legacy.toml")
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_ROOT / f"qwen-cache-{datetime.now(UTC):%Y%m%dT%H%M%SZ}",
    )
    parser.add_argument("--rounds", type=int, default=2, help="alternating f16/Q8 then Q8/f16")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--planner", choices=(PLANNER_V7_ID, PLANNER_V8_ID), default=PLANNER_V7_ID)
    parser.add_argument(
        "--validation",
        action="store_true",
        help="include understanding, temporal and unsupported validation; requires --full-suite",
    )
    parser.add_argument("--port", type=int, default=8097)
    parser.add_argument(
        "--timeout-s", type=float, default=900, help="bounded time per owned server trial"
    )
    parser.add_argument(
        "--case", action="append", help="select bilingual case IDs; repeat this option"
    )
    parser.add_argument(
        "--full-suite",
        action="store_true",
        help="use all questions rather than representative cases",
    )
    args = parser.parse_args()
    if args.validation and not args.full_suite:
        parser.error("--validation requires --full-suite")
    if (
        not 1 <= args.rounds <= 5
        or not 0 <= args.seed < 2**31
        or not 1024 <= args.port <= 65535
        or not 1 <= args.timeout_s <= 3600
    ):
        parser.error("rounds must be 1..5, seed 0..2^31-1, port 1024..65535, timeout-s 1..3600")
    try:
        return asyncio.run(benchmark(args))
    except KeyboardInterrupt:
        print(
            "Benchmark interrupted; owned server closed. Partial trial reports retained.",
            flush=True,
        )
        return 130
    except (OSError, ValueError, RuntimeError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    raise SystemExit(main())
