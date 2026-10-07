"""Optional local PyTorch/Laya semantic scorer JSON-lines worker."""

import argparse
import io
import json
import os
import sys
import time
from importlib.metadata import version
from pathlib import Path
from typing import Protocol


class ScoringBackend(Protocol):
    runtime_metadata: dict[str, object]

    def score(
        self,
        text: str,
        hypotheses: list[str],
        language: str | None,
    ) -> list[float]: ...


class MiniLmBackend:
    def __init__(self, model_path: Path, threads: int) -> None:
        import torch  # type: ignore[import-not-found]
        from transformers import (  # type: ignore[import-not-found]
            AutoModelForSequenceClassification,
            AutoTokenizer,
        )

        torch.set_num_threads(threads)
        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            local_files_only=True,
        )
        self._model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            local_files_only=True,
        )
        self._model.eval()
        self.runtime_metadata = {
            "python": sys.version.split()[0],
            "torch": version("torch"),
            "transformers": version("transformers"),
            "threads": threads,
        }
        labels = {str(value).lower(): key for key, value in self._model.config.id2label.items()}
        self._entailment = int(labels["entailment"])

    def score(
        self,
        text: str,
        hypotheses: list[str],
        language: str | None,
    ) -> list[float]:
        del language
        encoded = self._tokenizer(
            [text] * len(hypotheses),
            hypotheses,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt",
        )
        with self._torch.inference_mode():
            logits = self._model(**encoded).logits
            probabilities = self._torch.softmax(logits, dim=-1)[:, self._entailment]
        return [float(value) for value in probabilities.tolist()]


class LayaBackend:
    def __init__(self, model_path: Path, package_path: Path, threads: int) -> None:
        import torch

        torch.set_num_threads(threads)
        sys.path.insert(0, str(package_path))
        import laya  # type: ignore[import-not-found]

        self._agent = laya.load(str(model_path), device="cpu")
        self.runtime_metadata = {
            "python": sys.version.split()[0],
            "torch": version("torch"),
            "laya": version("laya"),
            "threads": threads,
        }

    def score(
        self,
        text: str,
        hypotheses: list[str],
        language: str | None,
    ) -> list[float]:
        questions = {
            f"h{index}": {
                "type": "noul",
                "instructions": hypothesis,
                "criteria": {
                    "false": "The driver does not request this.",
                    "true": "The driver requests this.",
                },
            }
            for index, hypothesis in enumerate(hypotheses)
        }
        result = self._agent.predict(text, questions, lang=language)
        answers = result.get("answers", {})
        return [float(answers[f"h{index}"]["noul"]) for index in range(len(hypotheses))]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("minilm", "laya"), required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--package-path", type=Path)
    parser.add_argument("--threads", type=int, default=8)
    return parser


def _valid_request(value: object) -> tuple[int, str, list[str], str | None]:
    if not isinstance(value, dict):
        raise ValueError("request must be an object")
    request_id = value.get("id")
    text = value.get("text")
    hypotheses = value.get("hypotheses")
    language = value.get("language")
    if not isinstance(request_id, int) or request_id < 1:
        raise ValueError("request ID is invalid")
    if not isinstance(text, str) or not 1 <= len(text) <= 1000:
        raise ValueError("request text is invalid")
    if (
        not isinstance(hypotheses, list)
        or not 1 <= len(hypotheses) <= _MAX_HYPOTHESES
        or not all(isinstance(item, str) and 1 <= len(item) <= 1000 for item in hypotheses)
    ):
        raise ValueError("request hypotheses are invalid")
    if language not in {None, "en", "tr"}:
        raise ValueError("request language is invalid")
    return request_id, text, hypotheses, language


_MAX_HYPOTHESES = 32


def main() -> None:
    if isinstance(sys.stdout, io.TextIOWrapper):
        sys.stdout.reconfigure(encoding="utf-8", errors="strict")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    args = _parser().parse_args()
    started = time.perf_counter()
    backend: ScoringBackend
    if args.backend == "minilm":
        backend = MiniLmBackend(args.model, args.threads)
    else:
        if args.package_path is None:
            raise SystemExit("Laya requires --package-path")
        backend = LayaBackend(args.model, args.package_path, args.threads)
    runtime = {
        **backend.runtime_metadata,
        "startup_ms": round((time.perf_counter() - started) * 1000, 3),
    }
    print(
        json.dumps({"status": "ready", "backend": args.backend, "runtime": runtime}),
        flush=True,
    )
    for line in sys.stdin:
        try:
            request_id, text, hypotheses, language = _valid_request(json.loads(line))
            scores = backend.score(text, hypotheses, language)
            response = {"id": request_id, "scores": scores}
        except Exception:
            response = {"id": None, "error": "semantic_worker_request_failed"}
        print(json.dumps(response, ensure_ascii=True, separators=(",", ":")), flush=True)


if __name__ == "__main__":
    main()
