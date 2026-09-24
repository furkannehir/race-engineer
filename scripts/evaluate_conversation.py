"""Run explicit, local-only conversation quality and performance evaluation."""

import argparse
import asyncio
import json
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

from race_engineer.config import load_config
from race_engineer.conversation.factory import conversation_planner
from race_engineer.conversation.runtime import ConversationServer
from race_engineer.evaluation.conversation import (
    load_evaluation_dataset,
    validate_split_families,
)
from race_engineer.evaluation.runner import evaluate_datasets
from race_engineer.evaluation.system import (
    directory_fingerprint,
    file_sha256,
    files_fingerprint,
    git_dirty,
    git_revision,
    gpu_process_memory,
    machine_metadata,
    process_metrics,
    runtime_version,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "fixtures/conversation/cases.json"
EVALUATOR_REVISION = "ce-02.1"


def _timestamp() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _write_report(path: Path, report: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _progress(result: dict[str, object]) -> None:
    timing = result["timing_ms"]
    planner_ms = timing.get("planner") if isinstance(timing, dict) else None
    expected = result["expected"]
    language = expected["language"] if isinstance(expected, dict) else None
    print(
        json.dumps(
            {
                "case": result["case_id"],
                "item": result["item"],
                "language": language,
                "passed": result["passed"],
                "planner_ms": planner_ms,
                "warm_state": result["warm_state"],
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        flush=True,
    )


async def evaluate(
    config_path: Path,
    dataset_paths: tuple[Path, ...],
    *,
    output_path: Path | None = None,
    manage_server: bool = False,
    quiet: bool = False,
    run_label: str | None = None,
) -> int:
    config = load_config(config_path)
    datasets = tuple(load_evaluation_dataset(path) for path in dataset_paths)
    validate_split_families(datasets)
    started_at = _timestamp()
    wall_started = time.perf_counter()
    server_events: list[tuple[str, str]] = []
    server: ConversationServer | None = None
    startup_ms: float | None = None
    child_pid: int | None = None
    effective_compute = "external:unmanaged"
    selected_executable: Path | None = None
    child_metrics: dict[str, object] | None = None
    child_gpu: dict[str, object] | None = None
    try:
        if manage_server:
            server = ConversationServer(
                ROOT,
                config.conversation,
                notify=lambda *event: server_events.append(event),
            )
            startup_started = time.perf_counter()
            await server.start()
            startup_ms = (time.perf_counter() - startup_started) * 1000
            compute_events = [value for topic, value in server_events if topic == "compute"]
            if compute_events:
                effective_compute = compute_events[-1]
            if server.effective is not None:
                selected_executable = server.effective.executable
            if server.process is not None:
                child_pid = server.process.pid

        planner = conversation_planner(config.conversation)
        results, summary = await evaluate_datasets(
            ROOT,
            config.conversation,
            planner,
            datasets,
            progress=None if quiet else _progress,
        )
        if child_pid is not None:
            child_metrics = process_metrics(child_pid)
            child_gpu = gpu_process_memory(child_pid)
    finally:
        if server is not None:
            await server.aclose()

    finished_at = _timestamp()
    wall_ms = (time.perf_counter() - wall_started) * 1000
    runtime = config.conversation.runtime
    source_files = (
        Path(__file__).resolve(),
        *tuple((ROOT / "src/race_engineer").rglob("*.py")),
    )
    fallback_events = [
        value
        for topic, value in server_events
        if topic == "compute" and value.startswith("cpu:") and value != "cpu:cpu"
    ]
    report: dict[str, object] = {
        "schema_version": "conversation-eval-report.v1",
        "evaluator_revision": EVALUATOR_REVISION,
        "run_id": uuid.uuid4().hex,
        "run_label": run_label,
        "started_at": started_at,
        "finished_at": finished_at,
        "wall_ms": round(wall_ms, 3),
        "privacy": {
            "content_free_results": True,
            "question_text_recorded": False,
            "reply_text_recorded": False,
            "microphone_used": False,
            "network_inference_used": False,
        },
        "code": {
            "git_revision": git_revision(ROOT),
            "working_tree_dirty": git_dirty(ROOT),
            "source_fingerprint_sha256": files_fingerprint(ROOT, source_files),
            "configuration_sha256": file_sha256(config_path.resolve()),
        },
        "planner": {
            "adapter": config.conversation.adapter,
            "model": config.conversation.model,
            "model_revision": runtime.model_revision,
            "quantization": runtime.quantization,
            "timeout_s": config.conversation.timeout_s,
        },
        "runtime": {
            "declared_revision": runtime.runtime_revision,
            "reported_version": (
                runtime_version(selected_executable, runtime.probe_timeout_s)
                if selected_executable is not None
                else None
            ),
            "configured_mode": runtime.mode,
            "effective_compute": effective_compute,
            "lifecycle_requested": manage_server,
            "server_owned": child_pid is not None,
            "startup_ms": round(startup_ms, 3) if startup_ms is not None else None,
            "threads": runtime.threads,
            "context_size": runtime.context_size,
            "parallel": runtime.parallel,
            "gpu_layers": (
                server.effective.gpu_layers
                if server is not None and server.effective is not None
                else None
            ),
            "process": child_metrics,
            "gpu_process": child_gpu,
            "fallback_count": len(fallback_events),
            "fallback_reasons": fallback_events,
            "events": [
                {"topic": topic, "value": value}
                for topic, value in server_events
                if topic == "compute"
            ],
        },
        "machine": machine_metadata(),
        "datasets": [
            {
                "schema_version": dataset.schema_version,
                "dataset_id": dataset.dataset_id,
                "revision": dataset.revision,
                "split": dataset.split,
                "turns": dataset.turn_count,
                "languages": dataset.language_counts(),
                "file_sha256": file_sha256(path),
                "fixture_sha256": directory_fingerprint((ROOT / dataset.fixture).resolve()),
            }
            for dataset, path in zip(datasets, dataset_paths, strict=True)
        ],
        "summary": summary,
        "turns": results,
    }
    if output_path is not None:
        _write_report(output_path, report)
    print(
        json.dumps(
            {
                "schema_version": report["schema_version"],
                "datasets": [dataset.dataset_id for dataset in datasets],
                "report": str(output_path) if output_path is not None else None,
                "summary": summary,
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        flush=True,
    )
    overall = cast(dict[str, object], summary["overall"])
    return 0 if overall["passed"] == overall["turns"] else 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config/default.toml")
    parser.add_argument(
        "--dataset",
        "--cases",
        dest="datasets",
        action="append",
        type=Path,
        help="versioned dataset; repeat to evaluate multiple disjoint splits",
    )
    parser.add_argument("--output", type=Path, help="write a content-free JSON report")
    parser.add_argument(
        "--manage-server",
        action="store_true",
        help="start/reuse the configured local server and measure its lifecycle",
    )
    parser.add_argument("--quiet", action="store_true", help="print only the final summary")
    parser.add_argument("--run-label", help="optional non-sensitive label for this run")
    args = parser.parse_args()
    datasets = tuple(args.datasets or (DEFAULT_DATASET,))
    raise SystemExit(
        asyncio.run(
            evaluate(
                args.config,
                datasets,
                output_path=args.output,
                manage_server=args.manage_server,
                quiet=args.quiet,
                run_label=args.run_label,
            )
        )
    )


if __name__ == "__main__":
    main()
