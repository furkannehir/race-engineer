"""Try the trained action-classifier pilot locally; no telemetry, replies or saved prompts."""

import argparse
import io
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

from driver_pilot_runtime import ROOT, Encoder

from race_engineer.evaluation.driver_training import (
    LABELS,
    MODEL_REPO,
    MODEL_REVISION,
    conversation_input,
    threshold_predictions,
)

HELP = """Type an English or Turkish driver message and press Enter.
This predicts actions/clarification ONLY, not telemetry topics, full meaning or replies.
Messages are independent by default. Nothing you type is saved.
/history on | /history off   Include recent turns (up to six); off also clears history.
/engineer MESSAGE            Add an engineer turn when history is on.
/reset                       Clear history.
/help                        Show this help.
/quit                        Exit; Ctrl+C/end-of-input also exits.
Scores are uncalibrated, not trustworthy confidence estimates. No live app is changed.
"""


def load_metadata(path: Path) -> dict[str, Any]:
    path = path.resolve()
    if not path.is_relative_to(ROOT / "data"):
        raise ValueError("pilot model must be inside repository data/")
    raw = json.loads((path / "model.json").read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("invalid model metadata object")
    metadata: dict[str, Any] = raw
    expected = {
        "schema_version": "driver-action-pilot.v1",
        "base_repo": MODEL_REPO,
        "base_revision": MODEL_REVISION,
        "labels": list(LABELS),
        "adapter_file": "selected-adapter.safetensors",
        "trained_scope": "conversational_actions_and_clarification_only",
        "draft_labels": True,
        "promoted": False,
    }
    if any(metadata.get(key) != value for key, value in expected.items()):
        raise ValueError("unexpected pilot model metadata")
    threshold = metadata.get("threshold")
    if (
        isinstance(threshold, bool)
        or not isinstance(threshold, (int, float))
        or not math.isfinite(threshold)
        or not 0 <= threshold <= 1
    ):
        raise ValueError("invalid model threshold")
    return metadata


class LocalClassifier:
    def __init__(self, path: Path, threads: int) -> None:
        metadata = load_metadata(path)
        self.threshold: float = float(metadata["threshold"])
        self.runtime = Encoder(threads)
        from safetensors.torch import load_file  # type: ignore[import-not-found]

        self.runtime.restore(load_file(str(path / metadata["adapter_file"]), device="cpu"))

    def predict(self, text: str) -> list[float]:
        return self.runtime.probabilities([text])[0]


def display(classifier: LocalClassifier, text: str, history: list[tuple[str, str]]) -> None:
    before = time.perf_counter()
    scores = classifier.predict(conversation_input(text, history))
    elapsed = (time.perf_counter() - before) * 1000
    prediction = threshold_predictions([scores], classifier.threshold)[0]
    print(
        json.dumps(
            {
                "actions": [
                    label
                    for label, value in zip(LABELS[:-1], prediction[:-1], strict=True)
                    if value
                ],
                "clarification_needed": bool(prediction[-1]),
                "scores_uncalibrated": {
                    label: round(score, 4) for label, score in zip(LABELS, scores, strict=True)
                },
                "threshold": classifier.threshold,
                "duration_ms": round(elapsed, 3),
                "history_entries": len(history),
                "scope": "action_classifier_only_not_full_interpreter",
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


def main() -> int:
    for stream in (sys.stdin, sys.stdout):
        if isinstance(stream, io.TextIOWrapper):
            stream.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-dir", type=Path, default=ROOT / "data/driver-interpreter-pilot/run-20261011-seed17"
    )
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--prompt", help="One independent message, then exit.")
    args = parser.parse_args()
    if not 1 <= args.threads <= 64:
        parser.error("threads must be between 1 and 64")
    print("Loading local draft-label action pilot on CPU...", flush=True)
    try:
        classifier = LocalClassifier(args.model_dir.resolve(), args.threads)
    except (OSError, ValueError, ImportError):
        print(
            "Pilot could not load. Use the optional STT runtime Python, run the setup/training "
            "scripts, and check --model-dir. The live application is unaffected.",
            file=sys.stderr,
        )
        return 2
    if args.prompt is not None:
        if not args.prompt.strip():
            parser.error("prompt must not be empty")
        display(classifier, args.prompt, [])
        return 0
    print(HELP)
    history: list[tuple[str, str]] = []
    history_enabled = False
    while True:
        try:
            text = input("Driver> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break
        if not text:
            continue
        if text == "/quit":
            break
        if text == "/help":
            print(HELP)
            continue
        if text in ("/history on", "/history off"):
            history_enabled = text.endswith("on")
            if not history_enabled:
                history.clear()
            print(f"History {'on' if history_enabled else 'off'}.")
            continue
        if text == "/reset":
            history.clear()
            print("History cleared.")
            continue
        if text.startswith("/engineer "):
            if not history_enabled:
                print("Use /history on before adding engineer context.")
            else:
                history.append(("engineer", text.removeprefix("/engineer ").strip()))
                history[:] = history[-6:]
                print("Engineer turn added.")
            continue
        if text.startswith("/"):
            print("Unknown command; use /help.")
            continue
        display(classifier, text, history)
        if history_enabled:
            history.append(("driver", text))
            history[:] = history[-6:]
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
