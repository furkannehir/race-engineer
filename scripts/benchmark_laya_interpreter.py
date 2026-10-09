"""Screen the downloaded Laya checkpoint as an interpreter, never a live planner."""

import argparse
import hashlib
import io
import json
import os
import random
import sys
import time
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import cast

from race_engineer.evaluation.radio_interpreter import (
    decision_questions,
    score_case,
    state_for,
    summarize,
    text_state_for,
    validate_cases,
)
from race_engineer.evaluation.system import (
    distribution,
    files_fingerprint,
    git_dirty,
    git_revision,
    machine_metadata,
)

ROOT = Path(__file__).resolve().parents[1]
MODEL_REVISION = "e4e9ddf21a7b1903b7acffd8814ad4307bf63a67"


def repository_path(value: Path, *, directory: bool = False) -> Path:
    resolved = (ROOT / value).resolve()
    if not resolved.is_relative_to(ROOT):
        raise ValueError("experiment_path_escapes_repository")
    if not (resolved.is_dir() if directory else resolved.is_file()):
        raise ValueError("experiment_input_missing")
    return resolved


def run(args: argparse.Namespace) -> dict[str, object]:
    dataset_path = repository_path(args.dataset)
    model_path = repository_path(args.model, directory=True)
    package_path = repository_path(args.package_path, directory=True)
    output = (ROOT / args.output).resolve()
    if not output.is_relative_to(ROOT / "data"):
        raise ValueError("reports_must_be_inside_ignored_data_directory")
    if output.exists():
        raise ValueError("output_exists_choose_a_new_filename")
    if not 1 <= args.threads <= 64 or not 1 <= args.repeats <= 10:
        raise ValueError("invalid_threads_or_repeats")
    cases = validate_cases(json.loads(dataset_path.read_text(encoding="utf-8")))
    provenance = json.loads((model_path / "provenance.json").read_text(encoding="utf-8"))
    if provenance != {"repo": "convaiinnovations/laya-multilingual", "revision": MODEL_REVISION}:
        raise ValueError("unexpected_model_provenance")
    questions = decision_questions()
    source_files = (
        Path(__file__), ROOT / "src/race_engineer/evaluation/radio_interpreter.py",
        package_path / "laya/agent.py", package_path / "laya/common.py",
        package_path / "laya/__init__.py",
    )
    inputs = (dataset_path, model_path / "provenance.json", model_path / "rl_agent_config.json")
    source_fingerprint = files_fingerprint(ROOT, source_files)
    input_fingerprint = files_fingerprint(ROOT, inputs)
    print("Loading pinned Laya multilingual on CPU (offline; no Qwen or iRacing needed)...",
          flush=True)
    # This script belongs in the existing isolated PyTorch runtime, not the desktop venv.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    sys.path.insert(0, str(package_path))
    startup = time.perf_counter()
    import laya  # type: ignore[import-not-found]
    import torch  # type: ignore[import-not-found]

    if laya.__version__ != "0.3.20":
        raise ValueError("unexpected_laya_version")
    torch.set_num_threads(args.threads)
    agent = laya.load(str(model_path), device="cpu")
    startup_ms = (time.perf_counter() - startup) * 1000
    print(f"Loaded in {startup_ms / 1000:.2f}s. Running {len(cases) * args.repeats} turns...",
          flush=True)
    results: list[dict[str, object]] = []
    rng = random.Random(args.seed)
    for repetition in range(1, args.repeats + 1):
        ordered = list(cases)
        rng.shuffle(ordered)
        for case in ordered:
            state = text_state_for(case) if args.state_format == "text" else state_for(case)
            started = time.perf_counter()
            try:
                prediction = agent.predict(
                    state, questions, lang=case["language"],
                )
                elapsed_ms = (time.perf_counter() - started) * 1000
                result = score_case(case, prediction)
                result["error"] = None
            except Exception as error:
                elapsed_ms = (time.perf_counter() - started) * 1000
                result = {
                    "case_id": case["id"], "language": case["language"], "passed": False,
                    "error": type(error).__name__,
                }
            result.update(repetition=repetition, duration_ms=round(elapsed_ms, 3))
            result["model_input_sha256"] = hashlib.sha256(
                json.dumps(state, ensure_ascii=False, sort_keys=True).encode("utf-8")
            ).hexdigest()
            results.append(result)
            outcome = "PASS" if result["passed"] else "FAIL"
            print(f"[{len(results)}] repeat {repetition}: {case['id']} {outcome} "
                  f"({elapsed_ms / 1000:.3f}s)", flush=True)
    durations = [cast(float, row["duration_ms"]) for row in results]
    completed = [cast(float, row["duration_ms"]) for row in results if row.get("error") is None]
    report = {
        "schema_version": "radio-interpreter-screening.v1",
        "generated_at": datetime.now(UTC).isoformat(),
        "scope": "purpose_time_scope_and_multi_topic_component_only",
        "candidate": {
            "id": "laya-radio-decisions-v1", "package_version": laya.__version__,
            "model": provenance["repo"], "model_revision": MODEL_REVISION,
            "effective_device": "cpu", "threads": args.threads,
            "state_format": args.state_format,
            "python": sys.version.split()[0], "torch": version("torch"),
            "transformers": version("transformers"), "startup_ms": round(startup_ms, 3),
            "question_sha256": hashlib.sha256(
                json.dumps(questions, sort_keys=True).encode("utf-8")
            ).hexdigest(),
        },
        "reproducibility": {
            "input_fingerprint_sha256": input_fingerprint,
            "implementation_fingerprint_sha256": source_fingerprint,
            "inputs_unchanged": input_fingerprint == files_fingerprint(ROOT, inputs),
            "implementation_unchanged": source_fingerprint == files_fingerprint(ROOT, source_files),
            "git_revision": git_revision(ROOT), "git_dirty": git_dirty(ROOT),
            "machine": machine_metadata(), "seed": args.seed, "repeats": args.repeats,
        },
        "privacy": {"local_only": True, "contains_transcripts": False,
                    "contains_recent_dialogue": False},
        "summary": summarize(results),
        "latency_ms": {
            "all_attempts": distribution(durations), "completed": distribution(completed),
            "first_in_run": durations[0],
            "subsequent_completed": distribution([
                cast(float, row["duration_ms"]) for row in results[1:] if row.get("error") is None
            ]),
        },
        "limitations": [
            "Development screening, not a holdout or a full ContextPlan evaluation.",
            "Topic threshold 0.5 and model probabilities are uncalibrated diagnostics.",
            "No opponent/entity extraction, query operations or executable evidence requests.",
            "Authored histories, not actual sequential dialogue or live STT errors.",
            "No live telemetry, Core, STT, TTS, racing load or automatic promotion.",
        ],
        "live_routing_changed": False,
        "promoted": False,
        "results": results,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({"summary": report["summary"], "latency_ms": report["latency_ms"]},
                     indent=2), flush=True)
    print(f"Report: {output}", flush=True)
    return report


def main() -> int:
    if isinstance(sys.stdout, io.TextIOWrapper):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path,
                        default=Path("fixtures/radio-interpreter/development.json"))
    parser.add_argument("--model", type=Path, default=Path("data/dialogue-prototype/laya"))
    parser.add_argument("--package-path", type=Path,
                        default=Path("data/dialogue-prototype/packages"))
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--state-format", choices=("structured", "text"), default="structured")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = run(args)
    except (ValueError, OSError, ImportError) as error:
        print(f"Experiment could not run ({type(error).__name__}): {error}", file=sys.stderr)
        return 2
    summary = report["summary"]
    assert isinstance(summary, dict)
    return 0 if summary["exact_passes"] == summary["attempted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
