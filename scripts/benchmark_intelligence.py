"""Reproducible local text-response benchmark with managed server startup."""

import argparse
import asyncio
import csv
import io
import json
import os
import random
import shutil
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from race_engineer.application.intelligence_cli import _contexts
from race_engineer.config import load_config
from race_engineer.conversation.runtime import ConversationServer
from race_engineer.evaluation.intelligence import load_intelligence_evaluation_dataset
from race_engineer.evaluation.intelligence_latency import (
    measure_intelligence_turn,
    summarize_latency,
)
from race_engineer.evaluation.system import (
    files_fingerprint,
    git_dirty,
    git_revision,
    machine_metadata,
)
from race_engineer.fixtures import load_fixture
from race_engineer.intelligence.context_planner import (
    PLANNER_ID,
    PLANNER_V5_ID,
    PLANNER_V6_ID,
    PLANNER_V7_ID,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CASES = (
    "compound-en",
    "compound-tr",
    "historical-raw-en",
    "historical-raw-tr",
    "vent-en",
    "vent-tr",
)


def gpu_memory_snapshot() -> dict[str, object]:
    """Whole-device sampled memory, not per-process memory or a measured VRAM peak."""
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return {"available": False, "reason": "nvidia_smi_unavailable"}
    try:
        result = subprocess.run(
            [
                executable,
                "--query-gpu=index,name,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        if result.returncode != 0:
            return {"available": False, "reason": "gpu_query_failed"}
        devices = [
            {
                "index": int(index),
                "name": name.strip(),
                "used_mib": int(used),
                "total_mib": int(total),
            }
            for index, name, used, total in csv.reader(result.stdout.splitlines())
        ]
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return {"available": False, "reason": "gpu_query_failed"}
    return {"available": True, "devices": devices}


def verify_trial_runtime(server: ConversationServer, args: argparse.Namespace) -> None:
    if getattr(args, "require_owned_runtime", False) and server.process is None:
        raise ValueError("Controlled trial refuses an external server; use a free port")
    expected = getattr(args, "expect_backend", None)
    if expected is not None and (server.effective is None or server.effective.backend != expected):
        raise ValueError(f"Expected owned {expected} runtime; effective backend did not match")


async def benchmark(args: argparse.Namespace) -> dict[str, object]:
    dataset_path = args.dataset.resolve()
    if not dataset_path.is_relative_to(ROOT) or not dataset_path.is_file():
        raise ValueError("dataset must be a repository file")
    dataset = load_intelligence_evaluation_dataset(dataset_path)
    if dataset.split != "development":
        raise ValueError("this diagnostic benchmark uses only development datasets")
    if args.show_planner_inventory and not dataset.fixture.startswith("fixtures/synthetic/"):
        raise ValueError("planner inventory display is limited to synthetic fixtures")
    wanted = set(args.case or (group.id for group in dataset.groups))
    groups = [group for group in dataset.groups if group.id in wanted]
    if len(groups) != len(wanted):
        raise ValueError("unknown case ID for this dataset")
    fixture_path = (ROOT / dataset.fixture).resolve()
    if not fixture_path.is_relative_to(ROOT):
        raise ValueError("fixture escapes the repository")
    if args.output is not None and args.output.exists():
        raise ValueError("output already exists; choose a new filename")
    config = load_config(args.config)
    fixture = load_fixture(fixture_path)
    contexts = _contexts(fixture, config.policy.context)
    config_path = args.config.resolve()
    if not config_path.is_relative_to(ROOT):
        raise ValueError("benchmark config must be a repository file")
    input_files = (
        config_path,
        dataset_path,
        *(path for path in fixture_path.rglob("*") if path.is_file()),
    )
    source_files = (*ROOT.glob("src/race_engineer/**/*.py"), Path(__file__))
    source_before = files_fingerprint(ROOT, source_files)
    input_before = files_fingerprint(ROOT, input_files)
    machine = machine_metadata()
    server = ConversationServer(ROOT, config.conversation)
    if getattr(args, "show_server_logs", False):
        server.quiet = False
    results: list[dict[str, object]] = []
    gpu_observations: list[dict[str, object]] = []

    async def sample_gpu(phase: str) -> None:
        if getattr(args, "sample_gpu_memory", False):
            gpu_observations.append(
                {"phase": phase, **await asyncio.to_thread(gpu_memory_snapshot)}
            )

    await sample_gpu("before_server")
    order = random.Random(args.seed)
    started = time.perf_counter()
    runtime: dict[str, object] = {}
    try:
        print("Starting or reusing local Qwen server...", flush=True)
        await server.start()
        verify_trial_runtime(server, args)
        startup_ms = (time.perf_counter() - started) * 1000
        effective = server.effective
        runtime = {
            "ownership": "benchmark" if server.process is not None else "external_unmanaged",
            "effective_mode": effective.mode if effective is not None else None,
            "effective_backend": effective.backend if effective is not None else None,
            "effective_gpu_layers": effective.gpu_layers if effective is not None else None,
            "effective_device": effective.device if effective is not None else None,
            "startup_or_healthcheck_ms": round(startup_ms, 3),
        }
        await sample_gpu("after_server_start")
        for repetition in range(1, args.repeats + 1):
            turns = [(group, question) for group in groups for question in group.questions]
            order.shuffle(turns)
            for group, question in turns:
                index = len(results) + 1
                print(f"[{index}] repeat {repetition}, {group.id}: processing...", flush=True)
                request_inventory: list[tuple[str, ...]] = []
                result, reply = await measure_intelligence_turn(
                    config.conversation,
                    contexts,
                    group,
                    question,
                    run_index=index,
                    repetition=repetition,
                    planner_id=args.candidate,
                    inventory_observer=(
                        request_inventory.append if args.show_planner_inventory else None
                    ),
                )
                results.append(result)
                await sample_gpu(f"after_turn_{index}")
                print(json.dumps(result, ensure_ascii=True, sort_keys=True), flush=True)
                if args.show_replies:
                    print(
                        f"Question: {question}\nEngineer: {reply or '[no spoken reply]'}",
                        flush=True,
                    )
                if args.show_planner_inventory:
                    inventory_display = (
                        request_inventory[-1]
                        if request_inventory
                        else "[planner did not return an inventory]"
                    )
                    print(
                        f"Planner requested facts (temporary): {inventory_display}",
                        flush=True,
                    )
                failure = result["failure"]
                if isinstance(failure, dict) and (
                    "timeout" in str(failure["reason"]) or "unreachable" in str(failure["reason"])
                ):
                    # A timed-out HTTP thread/server task may still be executing. Do not
                    # contaminate subsequent samples or pile new work onto an unmanaged server.
                    print("Stopping after transport timeout/unreachable; partial report retained.")
                    break
            else:
                continue
            break
    finally:
        await server.aclose()
        await sample_gpu("after_server_close")
    return {
        "schema_version": "intelligence-latency-report.v2",
        "generated_at": datetime.now(UTC).isoformat(),
        "evaluation_scope": "typed_context_core_refresh_grounding",
        "excluded_stages": ["microphone", "stt", "radio_queue", "tts", "audio_playback"],
        "candidate_id": args.candidate,
        "model": config.conversation.model,
        "configured_runtime": config.conversation.runtime.model_dump(mode="json"),
        "model_request_timeout_s": config.conversation.timeout_s,
        "runtime": runtime,
        "gpu_observations": {
            "scope": "whole_device_including_other_applications",
            "peak_memory_measured": False,
            "samples": gpu_observations,
        },
        "dataset_id": dataset.dataset_id,
        "dataset_revision": dataset.revision,
        "seed": args.seed,
        "requested_repeats": args.repeats,
        "selected_cases": sorted(wanted),
        "expected_turns": args.repeats * sum(len(group.questions) for group in groups),
        "privacy": {
            "contains_questions": False,
            "contains_model_replies": False,
            "contains_request_inventory": False,
        },
        "reproducibility": {
            "git_revision": git_revision(ROOT),
            "git_dirty": git_dirty(ROOT),
            "machine": machine,
            "implementation_fingerprint_sha256": source_before,
            "implementation_changed_during_run": source_before
            != files_fingerprint(ROOT, source_files),
            "input_fingerprint_sha256": input_before,
            "inputs_changed_during_run": input_before != files_fingerprint(ROOT, input_files),
        },
        "summary": summarize_latency(results),
        "results": results,
    }


def main() -> None:
    # Windows pipes may otherwise default to a code page without Turkish characters.
    if isinstance(sys.stdout, io.TextIOWrapper):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config/default.toml")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=ROOT / "fixtures/intelligence-evaluation/development-controls.json",
    )
    parser.add_argument("--case", action="append", help="group ID; repeat to choose cases")
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--require-owned-runtime",
        action="store_true",
        help="refuse external/unverified servers for controlled hardware trials",
    )
    parser.add_argument(
        "--expect-backend",
        choices=("cpu", "cuda", "vulkan"),
        help="fail rather than benchmark the wrong effective backend",
    )
    parser.add_argument(
        "--show-server-logs",
        action="store_true",
        help="print managed server startup/offload diagnostics",
    )
    parser.add_argument(
        "--sample-gpu-memory",
        action="store_true",
        help="sample whole NVIDIA device memory outside per-turn timers",
    )
    parser.add_argument(
        "--candidate",
        choices=(PLANNER_V5_ID, PLANNER_V6_ID, PLANNER_V7_ID),
        default=PLANNER_ID,
        help="planner prompt candidate to benchmark",
    )
    parser.add_argument("--output", type=Path, help="new content-free JSON report")
    parser.add_argument(
        "--show-replies", action="store_true", help="show fixture replies for review"
    )
    parser.add_argument(
        "--show-planner-inventory",
        action="store_true",
        help="show temporary requested-fact labels for synthetic development questions",
    )
    args = parser.parse_args()
    if args.repeats < 1 or args.repeats > 20:
        parser.error("repeats must be between one and twenty")
    if (
        args.case is None
        and args.dataset == ROOT / "fixtures/intelligence-evaluation/development-controls.json"
    ):
        args.case = list(DEFAULT_CASES)
    try:
        report = asyncio.run(benchmark(args))
        print("SUMMARY\n" + json.dumps(report["summary"], indent=2), flush=True)
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("x", encoding="utf-8") as handle:
                handle.write(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
            print(f"Report: {args.output}", flush=True)
    except (OSError, RuntimeError, ValueError) as error:
        parser.error(str(error))
    summary = report["summary"]
    assert isinstance(summary, dict)
    raise SystemExit(1 if summary["pipeline_failed"] else 0)


if __name__ == "__main__":
    main()
