"""Publish content-free CE-04 screening evidence without modifying live configuration."""

import argparse
import json
from pathlib import Path

from race_engineer.evaluation.routing import assess_candidate
from race_engineer.evaluation.system import file_sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Refusing to overwrite existing screening evidence")
    candidates = []
    seen = set()
    for path in args.reports:
        report = json.loads(path.read_text(encoding="utf-8"))
        if report["schema_version"] != "dialogue-report.v1" or report["split"] != "locked":
            raise ValueError("Only versioned locked reports may inform routing")
        if report["candidate"] in seen:
            raise ValueError("Duplicate candidate")
        seen.add(report["candidate"])
        candidates.append(
            {
                **{
                    key: report[key]
                    for key in (
                        "candidate",
                        "synthetic_control",
                        "dataset_revision",
                        "dataset_sha256",
                        "adapter_revision",
                        "source_fingerprint",
                        "runtime",
                        "machine",
                        "startup_ms",
                        "summary",
                        "calibration",
                        "limitations",
                    )
                },
                "raw_report_sha256": file_sha256(path),
                "raw_report_file": path.name,
                "resident_memory": {
                    "available": False,
                    "reason": (
                        "raw Windows PID identified the venv launcher; "
                        "use the dedicated worker probe"
                    ),
                },
                "assessment": assess_candidate(report),
                "failures": [
                    {
                        key: row[key]
                        for key in (
                            "dialogue_id",
                            "turn_id",
                            "category",
                            "semantic_correct",
                            "decision_correct",
                            "outcome",
                            "reason",
                            "accepted",
                            "abstained",
                            "error",
                        )
                    }
                    for row in report["results"]
                    if not row["passed"]
                ],
            }
        )
    for key in ("dataset_sha256", "adapter_revision", "source_fingerprint"):
        if len({candidate[key] for candidate in candidates}) != 1:
            raise ValueError(f"Comparison requires matching {key}")
    if seen != {"minilm", "laya", "qwen"}:
        raise ValueError("Screening requires all three candidates")
    evidence = {
        "schema_version": "dialogue-screening.v1",
        "candidates": candidates,
        "decision": {
            "production_primary": "existing-qwen-v1",
            "semantic_v2_live_enabled": False,
            "small_judge_primary": None,
            "fallback_enabled": False,
            "residency": "No new live model; evaluation workers are closed after each candidate.",
            "qualification": "No promotion from this pilot; see per-candidate evidence gates.",
            "next": "CE-05 opt-in integration work; new holdout evidence before activation.",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
