"""Inspect local mDeBERTa zero-shot radio interpretations; no generated replies or saved text."""

import argparse
import io
import json
import os
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Protocol

from race_engineer.evaluation.mdeberta_interpreter import (
    MODEL_FILES,
    MODEL_REPO,
    MODEL_REVISION,
    TOPIC_THRESHOLD,
    decode_nli,
    hypotheses,
    nli_indices,
)
from race_engineer.evaluation.radio_interpreter import question_language_for

ROOT = Path(__file__).resolve().parents[1]
HELP = """Type an English or Turkish message, then press Enter.
This is classification, not chat. No Qwen, telemetry, microphone, or TTS is involved.
Every message is independent. Nothing is saved to files.
/lang en | /lang tr       Choose input language (same multilingual checkpoint).
/question-language matched | english   Choose hypothesis wording.
/raw on | /raw off       Show all entailment/neutral/contradiction probabilities.
/questions               Show the exact NLI hypotheses (not driver commands).
/help                    Show this help.
/quit                    Exit; Ctrl+C or end-of-input also exits.
Purpose/scope are forced highest-score choices, NOT reliable confidence estimates.
Topics use full three-way entailment >= 0.5; scores are not calibrated or Laya-comparable.
"""


class NliPredictor(Protocol):
    def predict(self, text: str, language: str) -> Sequence[Sequence[float]]: ...


class LocalNli:
    def __init__(self, threads: int) -> None:
        if not 1 <= threads <= 64:
            raise ValueError("threads must be between 1 and 64")
        path = (ROOT / "data/dialogue-prototype/mdeberta").resolve()
        if not path.is_relative_to(ROOT / "data"):
            raise ValueError("model must remain inside repository data")
        if any(not (path / name).is_file() for name in (*MODEL_FILES, "provenance.json")):
            raise ValueError("model is missing; run scripts/setup_mdeberta.py first")
        if json.loads((path / "provenance.json").read_text(encoding="utf-8")) != {
            "repo": MODEL_REPO, "revision": MODEL_REVISION,
        }:
            raise ValueError("unexpected checkpoint provenance")
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
        import torch  # type: ignore[import-not-found]
        from transformers import (  # type: ignore[import-not-found]
            AutoModelForSequenceClassification,
            AutoTokenizer,
        )

        torch.set_num_threads(threads)
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True,
                                                       trust_remote_code=False)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            str(path), local_files_only=True, use_safetensors=True, trust_remote_code=False,
        ).to(device="cpu", dtype=torch.float32).eval()
        self.indices = nli_indices(self.model.config.id2label)

    def predict(self, text: str, language: str) -> Sequence[Sequence[float]]:
        candidates = [value for group in hypotheses(language).values() for value in group.values()]
        encoded = self.tokenizer([text] * len(candidates), candidates, padding=True,
                                 truncation="only_first", max_length=256, return_tensors="pt")
        with self.torch.inference_mode():
            probabilities = self.torch.softmax(self.model(**encoded).logits, dim=-1)
        rows: list[list[float]] = probabilities[:, list(self.indices)].tolist()
        return rows


def show_decision(
    agent: NliPredictor, text: str, language: str, *, raw: bool = False,
    question_language_mode: str = "matched", write: Callable[[str], None] = print,
) -> None:
    if not 1 <= len(text.strip()) <= 1000:
        raise ValueError("message must contain 1-1000 nonblank characters")
    hypothesis_language = question_language_for(language, question_language_mode)
    started = time.perf_counter()
    rows = agent.predict(text, hypothesis_language)
    duration_ms = round((time.perf_counter() - started) * 1000, 3)
    result = decode_nli(rows, hypothesis_language)
    if not raw:
        result.pop("nli_probabilities")
    result.update({"duration_ms": duration_ms, "language_hint": language,
                   "question_language": hypothesis_language, "history_entries": 0,
                   "topic_threshold": TOPIC_THRESHOLD})
    write(json.dumps(result, ensure_ascii=False, indent=2))


def interactive(
    agent: NliPredictor, language: str, *, raw: bool = False,
    question_language_mode: str = "matched", read: Callable[[str], str] = input,
    write: Callable[[str], None] = print,
) -> None:
    write(HELP)
    while True:
        try:
            text = read(f"Driver ({language})> ").strip()
            if not text:
                continue
            if text == "/quit":
                return
            if text == "/help":
                write(HELP)
            elif text in {"/lang en", "/lang tr"}:
                language = text.rsplit(" ", 1)[1]
                write(f"Input language: {language}.")
            elif text in {"/question-language matched", "/question-language english"}:
                question_language_mode = text.rsplit(" ", 1)[1]
                write(f"Hypothesis wording mode: {question_language_mode}.")
            elif text in {"/raw on", "/raw off"}:
                raw = text.endswith(" on")
                write(f"Raw response display: {'on' if raw else 'off'}.")
            elif text == "/questions":
                write(json.dumps(hypotheses(question_language_for(
                    language, question_language_mode,
                )), ensure_ascii=False, indent=2))
            elif text.startswith("/"):
                write("Unknown command. Type /help.")
            else:
                try:
                    show_decision(agent, text, language, raw=raw,
                                  question_language_mode=question_language_mode, write=write)
                except Exception as error:
                    write(f"Interpretation failed ({type(error).__name__}); no reply generated.")
        except (EOFError, KeyboardInterrupt):
            write("Goodbye.")
            return


def main() -> int:
    for stream in (sys.stdin, sys.stdout):
        if isinstance(stream, io.TextIOWrapper):
            stream.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--language", choices=("en", "tr"), default="en")
    parser.add_argument("--question-language", choices=("english", "matched"), default="matched")
    parser.add_argument("--threads", type=int, choices=range(1, 65), default=8, metavar="1-64")
    parser.add_argument("--prompt", help="Inspect one message and exit.")
    parser.add_argument("--raw", action="store_true")
    args = parser.parse_args()
    if args.prompt is not None and not 1 <= len(args.prompt.strip()) <= 1000:
        parser.error("--prompt must contain 1-1000 nonblank characters")
    print("Loading pinned mDeBERTa NLI on CPU, offline. No Qwen or iRacing needed...", flush=True)
    started = time.perf_counter()
    try:
        agent = LocalNli(args.threads)
    except KeyboardInterrupt:
        return 130
    except Exception as error:
        print(f"mDeBERTa could not load ({type(error).__name__}): {error}", file=sys.stderr)
        print("Use data\\stt-prototype\\runtime\\Scripts\\python.exe; "
              "run scripts\\setup_mdeberta.py if the model is missing.", file=sys.stderr)
        return 2
    print(f"Ready in {time.perf_counter() - started:.2f}s. Scores are uncalibrated.", flush=True)
    if args.prompt is None:
        interactive(agent, args.language, raw=args.raw,
                    question_language_mode=args.question_language)
    else:
        try:
            show_decision(agent, args.prompt.strip(), args.language, raw=args.raw,
                          question_language_mode=args.question_language)
        except KeyboardInterrupt:
            return 130
        except Exception as error:
            print(f"Interpretation failed ({type(error).__name__}).", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
