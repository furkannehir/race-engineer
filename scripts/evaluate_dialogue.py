"""Run CE-04 offline dialogue evaluation; never changes live model selection."""

import argparse
import asyncio
import json
import time
from datetime import UTC, datetime
from pathlib import Path

from race_engineer.config import load_config
from race_engineer.conversation.judges import EncoderSemanticJudge, QwenSemanticJudge
from race_engineer.conversation.runtime import ConversationServer
from race_engineer.conversation.semantic import ADAPTER_REVISION, compose_scores
from race_engineer.core.dialogue import SemanticProposal
from race_engineer.evaluation.dialogue import (
    audit_splits,
    evaluate_dialogues,
    load_dialogues,
    semantic_correct,
)
from race_engineer.evaluation.system import (
    file_sha256,
    files_fingerprint,
    git_dirty,
    git_revision,
    machine_metadata,
    process_metrics,
)

ROOT = Path(__file__).resolve().parents[1]


class OracleJudge:
    """Fixture harness check ONLY. Results never count as model accuracy."""

    def __init__(self, dataset):
        self.turns = [t for s in dataset.scenarios for t in (*s.en, *s.tr)]

    async def judge(self, request):
        matching = [t for t in self.turns if t.question == request.question]
        preferred = [t for t in matching if t.acceptable[0].language == request.default_language]
        return (preferred or matching)[0].acceptable[0]


class AbstainJudge:
    async def judge(self, request):
        return SemanticProposal(
            language=request.reply_language or request.default_language,
            model_id="abstain-control",
            abstain=True,
        )


def calibration_grid(samples):
    """Predeclared joint score/margin grid, ONLY on teacher-forced calibration inputs.

    These are acceptance thresholds, not probability calibration. All held-out
    evaluation subsequently runs autoregressively on the candidate's own state.
    """
    rows = []
    for threshold in (0.5, 0.65, 0.8):
        for margin in (0.0, 0.1, 0.2):
            languages = {}
            for language in ("en", "tr"):
                accepted = correct = answerable = covered = 0
                for request, scores, turn in samples:
                    if turn.acceptable[0].language != language:
                        continue
                    can_answer = any(a.endswith(":available") for a in turn.answers)
                    answerable += can_answer
                    if not scores:
                        continue
                    proposal = compose_scores(
                        request,
                        scores,
                        model_id="calibration",
                        threshold=threshold,
                        margin=margin,
                        calibration_id=None,
                    )
                    ok = semantic_correct(proposal, turn)
                    accepted += not proposal.abstain
                    correct += ok and not proposal.abstain
                    covered += can_answer and ok
                languages[language] = {
                    "accepted": accepted,
                    "correct": correct,
                    "answerable": answerable,
                    "covered": covered,
                }
            eligible = all(
                v["accepted"]
                and v["correct"] / v["accepted"] >= 0.95
                and v["answerable"]
                and v["covered"] / v["answerable"] >= 0.7
                for v in languages.values()
            )
            rows.append(
                {
                    "threshold": threshold,
                    "margin": margin,
                    "eligible": bool(eligible),
                    "languages": languages,
                }
            )
    # A failed grid still yields a declared diagnostic operating point, never qualification.
    selected = max(
        rows,
        key=lambda r: (
            r["eligible"],
            sum(v["correct"] for v in r["languages"].values()),
            -r["threshold"],
            -r["margin"],
        ),
    )
    return {
        "selected": selected,
        "grid": rows,
        "sample_count": len(samples),
        "method": "joint-threshold-grid.v1",
    }


async def run(args):
    calibration = load_dialogues(ROOT / "fixtures/dialogue/calibration.json")
    locked = load_dialogues(ROOT / "fixtures/dialogue/locked.json")
    audit = audit_splits((calibration, locked))
    config = load_config(ROOT / "config/default.toml").conversation
    config = config.model_copy(
        update={
            "port": args.port,
            "runtime": config.runtime.model_copy(update={"mode": "cpu", "threads": args.threads}),
        }
    )
    server = None
    encoder = None
    startup_ms = None
    calibration_result = None
    child_pid = None
    dataset = calibration if args.split == "calibration" else locked
    if args.candidate == "oracle":
        judge = OracleJudge(dataset)
    elif args.candidate == "abstain":
        judge = AbstainJudge()
    elif args.candidate == "qwen":
        server = ConversationServer(ROOT, config)
        judge = QwenSemanticJudge(config)
    else:
        encoder = EncoderSemanticJudge(
            python=args.python,
            worker=ROOT / "scripts/semantic_worker.py",
            model_path=ROOT / "data/dialogue-prototype" / args.candidate,
            package_path=ROOT / "data/dialogue-prototype/packages",
            candidate=args.candidate,
            threads=args.threads,
        )
        judge = encoder
    started = time.perf_counter()
    try:
        if server:
            await server.start()
            if server.process is None:
                raise RuntimeError("Evaluation needs an owned CPU server; choose an unused --port.")
            child_pid = server.process.pid
            startup_ms = (time.perf_counter() - started) * 1000
        if encoder:
            await encoder.start()
            child_pid = encoder.process.pid
            startup_ms = encoder.startup_ms
            if args.split == "locked":
                if args.calibration is None:
                    raise ValueError("--calibration required for locked encoder evaluation")
                saved = json.loads(args.calibration.read_text(encoding="utf-8"))
                if saved["candidate"] != args.candidate or saved["split"] != "calibration":
                    raise ValueError("calibration identity mismatch")
                expected_hash = file_sha256(ROOT / "fixtures/dialogue/calibration.json")
                if (
                    saved["dataset_sha256"] != expected_hash
                    or saved["adapter_revision"] != ADAPTER_REVISION
                ):
                    raise ValueError("calibration revision mismatch")
                calibration_result = saved["calibration"]
                encoder.threshold = calibration_result["selected"]["threshold"]
                encoder.margin = calibration_result["selected"]["margin"]
                encoder.calibration_id = file_sha256(args.calibration)
        samples = []

        def progress(row):
            print(
                json.dumps({k: row[k] for k in ("dialogue_id", "turn_id", "passed", "judge_ms")}),
                flush=True,
            )

        rows, summary = await evaluate_dialogues(
            dataset,
            judge,
            teacher_forced=encoder is not None and args.split == "calibration",
            calibration_samples=samples
            if encoder is not None and args.split == "calibration"
            else None,
            progress=None if args.quiet else progress,
        )
        if encoder and args.split == "calibration":
            calibration_result = calibration_grid(samples)
        input_files = (
            *(ROOT / "src/race_engineer/conversation").glob("*.py"),
            ROOT / "src/race_engineer/core/dialogue.py",
            ROOT / "src/race_engineer/evaluation/dialogue.py",
            ROOT / "scripts/evaluate_dialogue.py",
            ROOT / "scripts/semantic_worker.py",
        )
        metadata = (
            encoder.metadata
            if encoder
            else {
                "model": config.model if server else args.candidate,
                "threads": args.threads,
                "device": "cpu",
            }
        )
        if encoder:
            metadata["model"] = json.loads((encoder.model_path / "provenance.json").read_text())
        elif server:
            metadata.update(
                {
                    "model_revision": config.runtime.model_revision,
                    "runtime_revision": config.runtime.runtime_revision,
                    "quantization": config.runtime.quantization,
                    "model_sha256": file_sha256(ROOT / config.runtime.model_path),
                }
            )
        report = {
            "schema_version": "dialogue-report.v1",
            "adapter_revision": ADAPTER_REVISION,
            "created_at": datetime.now(UTC).isoformat(),
            "candidate": args.candidate,
            "split": args.split,
            "synthetic_control": args.candidate in {"oracle", "abstain"},
            "dataset_sha256": file_sha256(ROOT / f"fixtures/dialogue/{args.split}.json"),
            "dataset_revision": dataset.revision,
            "split_audit": audit,
            "git_revision": git_revision(ROOT),
            "git_dirty": git_dirty(ROOT),
            "source_fingerprint": files_fingerprint(ROOT, input_files),
            "runtime": metadata,
            "machine": machine_metadata(),
            "startup_ms": startup_ms,
            "process": process_metrics(child_pid) if child_pid else None,
            "total_wall_ms": (time.perf_counter() - started) * 1000,
            "calibration": calibration_result,
            "summary": summary,
            "results": rows,
            "limitations": [
                "Synthetic typed text, not ASR or listening evaluation.",
                "High-end host, not a minimum-hardware qualification.",
                "No simulator/audio co-residency or critical-radio playback measured.",
                "Calibration encoder states are teacher-forced; locked states are predicted.",
            ],
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if args.output.exists():
            raise ValueError("Refusing to overwrite an existing evaluation report")
        args.output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(summary["overall"], indent=2))
    finally:
        if encoder:
            await encoder.aclose()
        if server:
            await server.aclose()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate", choices=["oracle", "abstain", "qwen", "minilm", "laya"], required=True
    )
    parser.add_argument("--split", choices=["calibration", "locked"], default="locked")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--calibration", type=Path)
    parser.add_argument(
        "--python", type=Path, default=ROOT / "data/stt-prototype/runtime/Scripts/python.exe"
    )
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--port", type=int, default=8091)
    parser.add_argument("--quiet", action="store_true")
    asyncio.run(run(parser.parse_args()))
