"""Type messages and inspect the pinned local Laya radio decisions, without saving text."""

import argparse
import io
import json
import os
import sys
import time
from collections import deque
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Protocol

from race_engineer.evaluation.radio_interpreter import (
    decision_questions,
    decode_decisions,
    question_language_for,
    text_state_for,
)

ROOT = Path(__file__).resolve().parents[1]
MODEL_REVISION = "e4e9ddf21a7b1903b7acffd8814ad4307bf63a67"
HELP = """Type an English or Turkish message, then press Enter.
This displays interpretations, not a spoken/chat reply. No telemetry is connected.
Messages are independent by default; nothing is saved to files.

/lang en | /lang tr       Choose input language (same multilingual checkpoint).
/question-language matched | english   Match question wording, or keep the English baseline.
/raw on | /raw off       Show/hide the full Laya response and its probability scores.
/questions               Show the exact decision questions sent to Laya.
/history on | off        Include previous messages as context (off clears them).
/engineer <text>         Add an authored engineer reply when history is enabled.
/reset                   Clear the in-memory dialogue.
/help                    Show this help.
/quit                    Exit. Ctrl+C or end-of-input also exits.
Scores are uncalibrated diagnostics, not trustworthy confidence guarantees.
"""


class DecisionAgent(Protocol):
    def predict(
        self, state: str, questions: dict[str, object], *, lang: str,
    ) -> object: ...


def load_agent(threads: int) -> DecisionAgent:
    if not 1 <= threads <= 64:
        raise ValueError("threads must be between 1 and 64")
    model = (ROOT / "data/dialogue-prototype/laya").resolve()
    package = (ROOT / "data/dialogue-prototype/packages").resolve()
    for path in (model, package):
        if not path.is_relative_to(ROOT) or not path.is_dir():
            raise ValueError("pinned Laya model/package is missing inside data/dialogue-prototype")
    provenance = json.loads((model / "provenance.json").read_text(encoding="utf-8"))
    if provenance != {"repo": "convaiinnovations/laya-multilingual", "revision": MODEL_REVISION}:
        raise ValueError("unexpected Laya checkpoint provenance")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    sys.path.insert(0, str(package))
    import laya  # type: ignore[import-not-found]
    import torch  # type: ignore[import-not-found]

    if laya.__version__ != "0.3.20":
        raise ValueError("unexpected Laya package version (requires the pinned 0.3.20)")
    torch.set_num_threads(threads)
    agent: DecisionAgent = laya.load(str(model), device="cpu")
    return agent


def show_decision(
    agent: DecisionAgent, text: str, language: str, history: Sequence[str], *,
    raw: bool, question_language_mode: str = "matched", write: Callable[[str], None] = print,
) -> None:
    if not 1 <= len(text) <= 1000:
        raise ValueError("message must be between 1 and 1000 characters")
    state = text_state_for({"text": text, "recent_dialogue": list(history)})
    question_language = question_language_for(language, question_language_mode)
    started = time.perf_counter()
    result = agent.predict(state, decision_questions(question_language), lang=language)
    duration_ms = round((time.perf_counter() - started) * 1000, 3)
    # Preserve access to the real response even if its shape fails our decoder.
    if raw:
        write("Raw Laya response:\n" + json.dumps(result, ensure_ascii=False, indent=2))
    observed, scores = decode_decisions(result)
    write(json.dumps({
        "interpretation": observed, "topic_scores_uncalibrated": scores,
        "duration_ms": duration_ms, "language_hint": language,
        "question_language": question_language,
        "history_entries": len(history),
    }, ensure_ascii=False, indent=2))


def interactive(
    agent: DecisionAgent, language: str, *, raw: bool = False,
    question_language_mode: str = "matched",
    read: Callable[[str], str] = input, write: Callable[[str], None] = print,
) -> None:
    history: deque[str] = deque(maxlen=6)
    use_history = False
    write(HELP)
    while True:
        try:
            text = read(f"Driver ({language})> ").strip()
        except (EOFError, KeyboardInterrupt):
            write("Goodbye.")
            return
        if not text:
            continue
        if text == "/quit":
            return
        if text == "/help":
            write(HELP)
        elif text == "/questions":
            write(json.dumps(decision_questions(question_language_for(
                language, question_language_mode,
            )), ensure_ascii=False, indent=2))
        elif text in {"/question-language matched", "/question-language english"}:
            question_language_mode = text.rsplit(" ", 1)[1]
            write(f"Question wording mode: {question_language_mode}.")
        elif text in {"/lang en", "/lang tr"}:
            language = text.rsplit(" ", 1)[1]
            write(f"Input language: {language}; question language: "
                  f"{question_language_for(language, question_language_mode)}.")
        elif text in {"/raw on", "/raw off"}:
            raw = text.endswith(" on")
            write(f"Raw response display: {'on' if raw else 'off'}.")
        elif text in {"/history on", "/history off"}:
            use_history = text.endswith(" on")
            if not use_history:
                history.clear()
            write(f"History: {'on' if use_history else 'off'}.")
        elif text == "/reset":
            history.clear()
            write("Dialogue cleared.")
        elif text.startswith("/engineer "):
            reply = text.removeprefix("/engineer ").strip()
            if not use_history or not 1 <= len(reply) <= 1000:
                write("Enable /history on first; supply an engineer reply of 1-1000 characters.")
            else:
                history.append(f"engineer: {reply}")
                write("Authored reply added to in-memory history.")
        elif text.startswith("/"):
            write("Unknown command. Type /help.")
        else:
            try:
                show_decision(agent, text, language, tuple(history), raw=raw,
                              question_language_mode=question_language_mode, write=write)
            except KeyboardInterrupt:
                write("Goodbye.")
                return
            except Exception as error:
                write(f"Interpretation failed ({type(error).__name__}); no reply generated.")
                continue
            if use_history:
                history.append(f"driver: {text}")


def main() -> int:
    for stream in (sys.stdin, sys.stdout):
        if isinstance(stream, io.TextIOWrapper):
            stream.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--language", choices=("en", "tr"), default="en")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--prompt", help="Inspect one message and exit instead of opening a loop.")
    parser.add_argument("--raw", action="store_true", help="Also print the complete Laya response.")
    parser.add_argument("--question-language", choices=("english", "matched"), default="matched",
                        help="Use questions matching the input language, or the English baseline.")
    args = parser.parse_args()
    if args.prompt is not None and not 1 <= len(args.prompt.strip()) <= 1000:
        parser.error("--prompt must contain 1-1000 nonblank characters")
    print("Loading pinned Laya multilingual on CPU, offline. No Qwen or iRacing needed...",
          flush=True)
    started = time.perf_counter()
    try:
        agent = load_agent(args.threads)
    except KeyboardInterrupt:
        return 130
    except Exception as error:
        print(f"Laya could not load ({type(error).__name__}): {error}", file=sys.stderr)
        print("Use .\\data\\stt-prototype\\runtime\\Scripts\\python.exe for this playground.",
              file=sys.stderr)
        return 2
    print(f"Ready in {time.perf_counter() - started:.2f}s. Scores are uncalibrated.", flush=True)
    if args.prompt is not None:
        try:
            show_decision(agent, args.prompt.strip(), args.language, (), raw=args.raw,
                          question_language_mode=args.question_language)
        except KeyboardInterrupt:
            return 130
        except Exception as error:
            print(f"Interpretation failed ({type(error).__name__}).", file=sys.stderr)
            return 1
    else:
        interactive(agent, args.language, raw=args.raw,
                    question_language_mode=args.question_language)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
