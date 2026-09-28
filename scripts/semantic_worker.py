"""Owned, offline CPU inference worker. Optional packages never enter the app process."""

import argparse
import contextlib
import importlib.metadata
import json
import os
import sys
from pathlib import Path


class ContextLimit(Exception):
    pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", choices=["minilm", "laya"], required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--threads", type=int, required=True)
    args = parser.parse_args()
    if not args.model.is_dir() or not 1 <= args.threads <= 32:
        raise ValueError("invalid worker configuration")
    os.environ.update(
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        HF_HUB_DISABLE_TELEMETRY="1",
        TOKENIZERS_PARALLELISM="false",
    )
    protocol = sys.stdout
    with contextlib.redirect_stdout(sys.stderr):
        import torch

        torch.set_num_threads(args.threads)
        torch.set_num_interop_threads(1)
        if args.candidate == "minilm":
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
            model = AutoModelForSequenceClassification.from_pretrained(
                args.model,
                local_files_only=True,
                use_safetensors=True,
            ).eval()
            labels = {
                str(label).lower(): int(index) for index, label in model.config.id2label.items()
            }
            if "entailment" not in labels or "contradiction" not in labels:
                raise ValueError("NLI label mapping missing")
        else:
            from laya import Agent

            model = Agent(str(args.model.resolve()), device="cpu", fast=False, compile=False)
            tokenizer = model.tok
    versions = {
        name: importlib.metadata.version(name)
        for name in ("torch", "transformers", "safetensors", "huggingface-hub")
    }
    if args.candidate == "laya":
        versions["laya"] = importlib.metadata.version("laya")
    print(
        json.dumps(
            {
                "ready": args.candidate,
                "versions": versions,
                "threads": args.threads,
                "device": "cpu",
                "dtype": "float32",
            }
        ),
        file=protocol,
        flush=True,
    )
    while True:
        line = sys.stdin.buffer.readline(65537)
        if not line:
            return
        try:
            if len(line) > 65536 or not line.endswith(b"\n"):
                raise ValueError("oversized message")
            message = json.loads(line)
            state, hypotheses = message["state"], message["hypotheses"]
            if (
                not isinstance(state, str)
                or not isinstance(hypotheses, dict)
                or len(hypotheses) > 40
            ):
                raise ValueError("invalid message")
            with contextlib.redirect_stdout(sys.stderr), torch.inference_mode():
                if args.candidate == "minilm":
                    encoded = tokenizer(
                        [state] * len(hypotheses),
                        list(hypotheses.values()),
                        padding=True,
                        truncation=False,
                        return_tensors="pt",
                    )
                    if encoded["input_ids"].shape[1] > 512:
                        raise ContextLimit()
                    batches = []
                    for start in range(0, len(hypotheses), 8):
                        logits = model(
                            **{k: v[start : start + 8] for k, v in encoded.items()}
                        ).logits
                        # Three-way NLI: neutral is uncertainty, not evidence for entailment.
                        batches.extend(logits.softmax(-1)[:, labels["entailment"]].tolist())
                    scores = dict(zip(hypotheses, batches, strict=True))
                else:
                    # Reserve room for all question/options tokens. Reject, never silently
                    # truncate away the current request or its context.
                    if len(tokenizer(state, add_special_tokens=False)["input_ids"]) > 800:
                        raise ContextLimit()
                    questions = {
                        key: {"type": "noul", "instructions": text}
                        for key, text in hypotheses.items()
                    }
                    result = model.predict_batch(
                        [state], questions, batch_size=8, max_len=1024, head_max_len=192
                    )[0]
                    scores = {key: result["answers"][key]["noul"] for key in hypotheses}
            print(json.dumps({"id": message["id"], "scores": scores}), file=protocol, flush=True)
        except ContextLimit:
            print(
                json.dumps({"id": message["id"], "error": "context_limit"}),
                file=protocol,
                flush=True,
            )
        except Exception:
            print('{"error":"inference_failed"}', file=protocol, flush=True)
            return


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print('{"error":"startup_failed"}', flush=True)
        sys.exit(1)
