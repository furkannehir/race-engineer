"""Evidence gates, not an automatic live-model selector."""

from typing import Any


def assess_candidate(report: dict[str, Any]) -> dict[str, Any]:
    reasons: list[str] = []
    if report.get("synthetic_control") or report.get("split") != "locked":
        reasons.append("not_locked_model_evidence")
    summary = report["summary"]
    for language in ("en", "tr"):
        metrics = summary["languages"][language]
        if metrics["judged_turns"] < 100:
            reasons.append(f"{language}:insufficient_independent_screening_volume")
        for metric, target in (
            ("accepted_plan_accuracy", 0.95),
            ("answerable_coverage", 0.7),
            ("clarification_accuracy", 0.9),
        ):
            value = metrics[metric]["rate"]
            if value is None or value < target:
                reasons.append(f"{language}:{metric}")
        if metrics["errors"]:
            reasons.append(f"{language}:inference_errors")
    if summary["overall"]["factual_guard_failures"]:
        reasons.append("factual_guard_failure")
    if report["candidate"] in {"minilm", "laya"}:
        p95 = summary["overall"]["warm_total_ms"]["p95"]
        if p95 is None or p95 > 300:
            reasons.append("cpu_interpretation_p95")
        if not (report.get("calibration") or {}).get("selected", {}).get("eligible", False):
            reasons.append("no_qualifying_calibration")
    # Offline typed evidence cannot establish these separate live gates.
    reasons.extend(
        ("audio_and_radio_integration_unvalidated", "representative_hardware_unvalidated")
    )
    return {"candidate": report["candidate"], "promote_to_live": False, "reasons": reasons}
