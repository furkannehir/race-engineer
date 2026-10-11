"""Offline JSON-lines action inference worker; never imported by the desktop UI."""

import argparse
import contextlib
import io
import json
import sys
import time
from pathlib import Path
from typing import Any

from race_engineer.intelligence.driver_actions import (
    ActionPrediction,
    ActionRequest,
    load_action_metadata,
)
from race_engineer.intelligence.driver_encoder import Encoder


class ActionRuntime:
    def __init__(self, base: Path, model: Path, threads: int) -> None:
        self.metadata = load_action_metadata(model)
        self.encoder = Encoder(threads, base=base)
        from safetensors.torch import load_file  # type: ignore[import-not-found]

        self.encoder.restore(load_file(str(model / self.metadata.adapter_file), device="cpu"))
        # The first real request must not pay the first-kernel/allocator cost.
        self.predict("Current driver message:\nRadio check.")

    def predict(self, text: str) -> ActionPrediction:
        started = time.perf_counter()
        tokens = self.encoder.tokenizer(text, truncation=False)["input_ids"]
        scores = self.encoder.probabilities([text])[0]
        return ActionPrediction(
            scores=scores,
            threshold=self.metadata.threshold,
            duration_ms=(time.perf_counter() - started) * 1000,
            input_truncated=len(tokens) > 128,
        )


def main() -> int:
    for stream in (sys.stdin, sys.stdout):
        if isinstance(stream, io.TextIOWrapper):
            stream.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--threads", type=int, required=True)
    args = parser.parse_args()
    protocol_stream = sys.stdout

    def emit(payload: dict[str, Any]) -> None:
        print(
            json.dumps(payload, ensure_ascii=False, allow_nan=False),
            file=protocol_stream,
            flush=True,
        )

    with contextlib.redirect_stdout(sys.stderr):
        try:
            runtime = ActionRuntime(args.base, args.model, args.threads)
        except Exception:
            emit({"ready": False, "error": "action_model_load_failed"})
            return 1
        emit({"ready": True, "metadata": runtime.metadata.model_dump()})
        while line := sys.stdin.readline(65537):
            if len(line) > 65536 or not line.endswith("\n"):
                return 1
            try:
                request = ActionRequest.model_validate_json(line)
                result = runtime.predict(request.input_text)
            except Exception:
                # Never transmit exception details containing prompts or filesystem data.
                emit({"error": "action_inference_failed"})
                return 1
            emit({"request_id": request.request_id, "result": result.model_dump()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
