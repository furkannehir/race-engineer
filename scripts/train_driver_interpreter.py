"""Run a bounded CPU-only draft-label pilot; never changes or promotes the live interpreter."""

import argparse
import io
import json
import random
import statistics
import sys
import time
from importlib.metadata import version
from pathlib import Path
from typing import Any

from driver_pilot_runtime import ROOT, Encoder

from race_engineer.evaluation.driver_dataset import DriverMeaningDataset
from race_engineer.evaluation.driver_training import (
    ACTION_DESCRIPTIONS,
    LABELS,
    MODEL_REPO,
    MODEL_REVISION,
    classification_metrics,
    grouped_split,
    majority_predictions,
    model_input,
    select_threshold,
    split_manifest,
    targets,
    threshold_predictions,
)


def evaluate(
    rows: list[Any],
    truth: list[list[int]],
    scores: list[list[float]],
    threshold: float,
) -> dict[str, Any]:
    predicted = threshold_predictions(scores, threshold)
    result = classification_metrics(truth, predicted)
    result["languages"] = {}
    for language in ("en", "tr"):
        indices = [i for i, row in enumerate(rows) if row.language == language]
        result["languages"][language] = (
            classification_metrics(
                [truth[i] for i in indices],
                [predicted[i] for i in indices],
            )
            if indices
            else None
        )
    result["failures"] = [
        {
            "example_id": row.example_id,
            "expected": [label for label, value in zip(LABELS, expected, strict=True) if value],
            "predicted": [label for label, value in zip(LABELS, observed, strict=True) if value],
        }
        for row, expected, observed in zip(rows, truth, predicted, strict=True)
        if expected != observed
    ]
    return result


def main() -> int:
    if isinstance(sys.stdout, io.TextIOWrapper):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-draft-labels", action="store_true")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=ROOT / "fixtures/driver-meaning/review-expanded.v1.json",
        help="Corpus to train; defaults to the preserved v1 pilot corpus.",
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="New run directory inside data/."
    )
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--head-steps", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    if not (1 <= args.steps <= 400 and 1 <= args.head_steps <= 2000 and 1 <= args.batch_size <= 64):
        parser.error("steps 1-400, head-steps 1-2000, batch-size 1-64 required")
    output = args.output.resolve()
    if not output.is_relative_to(ROOT / "data") or output == ROOT / "data" or output.exists():
        parser.error("--output must be a new named run directory inside repository data/")
    try:
        dataset = DriverMeaningDataset.model_validate_json(
            args.dataset.read_text(encoding="utf-8"),
        )
    except (OSError, ValueError):
        parser.error("could not read or validate --dataset; no model has been loaded")
    try:
        splits = grouped_split(dataset, args.seed, allow_draft_labels=args.allow_draft_labels)
    except ValueError as error:
        parser.error(str(error))
    manifest = split_manifest(dataset, splits, args.seed)
    print(
        "PILOT ONLY: draft synthetic labels; actions/clarification, not full question parsing.",
        flush=True,
    )
    print({name: len(indices) for name, indices in splits.items()}, flush=True)
    print("Loading pinned sentence encoder on CPU...", flush=True)
    started = time.perf_counter()
    runtime = Encoder(args.threads)
    torch = runtime.torch
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    # Reset the randomly initialized head after setting the seed.
    runtime.head.reset_parameters()
    texts = [model_input(row) for row in dataset.examples]
    truth = [targets(row) for row in dataset.examples]
    raw_lengths = runtime.tokenizer(texts, truncation=False, padding=False, return_length=True)[
        "length"
    ]
    truncated_ids = [
        row.example_id
        for row, length in zip(dataset.examples, raw_lengths, strict=True)
        if length > 128
    ]
    print(
        "Encoding frozen features (all text, no validation/test labels used in fitting)...",
        flush=True,
    )
    features = runtime.encode(texts)
    label_tensor = torch.tensor(truth, dtype=torch.float32)
    train_ids = splits["train"]
    training_y = label_tensor[train_ids]
    positives = training_y.sum(0)
    pos_weight = ((len(train_ids) - positives) / positives.clamp(min=1)).clamp(max=10)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    rng = random.Random(args.seed)
    warm_batches = [rng.choices(train_ids, k=args.batch_size) for _ in range(args.head_steps)]
    adapter_batches = [rng.choices(train_ids, k=args.batch_size) for _ in range(args.steps)]
    validation_ids = splits["validation"]
    validation_truth = [truth[index] for index in validation_ids]
    validation_rows = [dataset.examples[index] for index in validation_ids]

    def feature_scores(head: Any, ids: list[int]) -> list[list[float]]:
        with torch.inference_mode():
            result: list[list[float]] = torch.sigmoid(head(features[ids])).tolist()
        return result

    descriptions = runtime.encode(list(ACTION_DESCRIPTIONS))
    prototype_scores = (features @ descriptions.T).clamp(min=0, max=1).tolist()
    prototype_threshold = select_threshold(
        [prototype_scores[index] for index in validation_ids],
        validation_truth,
    )
    optimizer = torch.optim.AdamW(runtime.head.parameters(), lr=0.02, weight_decay=0.01)
    print(f"Fitting classification head: {args.head_steps} steps...", flush=True)
    for step, indices in enumerate(warm_batches, 1):
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(runtime.head(features[indices]), label_tensor[indices])
        loss.backward()
        optimizer.step()
        if step % 100 == 0 or step == args.head_steps:
            print(f"  head step {step}/{args.head_steps}, loss={loss.item():.4f}", flush=True)
    initial_head = runtime.snapshot(include_backbone=False)
    # Matched extra head updates for the frozen-body control.
    control = torch.nn.Linear(384, len(LABELS))
    control.load_state_dict(runtime.head.state_dict())
    optimizer = torch.optim.AdamW(control.parameters(), lr=0.001, weight_decay=0.01)
    for indices in adapter_batches:
        optimizer.zero_grad(set_to_none=True)
        loss_fn(control(features[indices]), label_tensor[indices]).backward()
        optimizer.step()
    control_weights = {
        f"head.{name}": value.detach().clone() for name, value in control.state_dict().items()
    }
    control_validation_scores = feature_scores(control, validation_ids)
    control_threshold = select_threshold(control_validation_scores, validation_truth)
    control_validation = evaluate(
        validation_rows, validation_truth, control_validation_scores, control_threshold
    )
    runtime.restore(initial_head)
    runtime.adapt_last_layers()
    adapter_parameters = [
        parameter for parameter in runtime.backbone.parameters() if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        [
            {"params": adapter_parameters, "lr": 0.00002},
            {"params": runtime.head.parameters(), "lr": 0.001},
        ],
        weight_decay=0.01,
    )
    best: dict[str, Any] | None = None
    checkpoints = {max(1, args.steps // 2), args.steps}
    print(f"Fine-tuning last two encoder layers: {args.steps} steps, CPU...", flush=True)
    for step, indices in enumerate(adapter_batches, 1):
        optimizer.zero_grad(set_to_none=True)
        embeddings = runtime.embedding(runtime.tokenize([texts[index] for index in indices]))
        loss = loss_fn(runtime.head(embeddings), label_tensor[indices])
        loss.backward()
        torch.nn.utils.clip_grad_norm_([*adapter_parameters, *runtime.head.parameters()], 1)
        optimizer.step()
        if step % 10 == 0 or step == args.steps:
            print(f"  adapter step {step}/{args.steps}, loss={loss.item():.4f}", flush=True)
        if step in checkpoints:
            scores = runtime.probabilities([texts[index] for index in validation_ids])
            threshold = select_threshold(scores, validation_truth)
            metrics = evaluate(validation_rows, validation_truth, scores, threshold)
            print(
                f"  validation exact={metrics['exact_match']:.4f}, "
                f"micro-F1={metrics['micro_f1']:.4f}, threshold={threshold}",
                flush=True,
            )
            if best is None or metrics["exact_match"] > best["validation"]["exact_match"]:
                best = {
                    "step": step,
                    "threshold": threshold,
                    "validation": metrics,
                    "weights": runtime.snapshot(include_backbone=True),
                }
    assert best is not None
    runtime.restore(best["weights"])
    # Candidate selection and thresholds are final BEFORE obtaining test predictions.
    selected = (
        "fine_tuned_encoder"
        if best["validation"]["exact_match"] > control_validation["exact_match"]
        else "frozen_encoder_trained_head"
    )
    test_ids = splits["test"]
    test_rows = [dataset.examples[index] for index in test_ids]
    test_truth = [truth[index] for index in test_ids]
    print("Evaluating the group-held-out test once for each fixed candidate...", flush=True)
    fine_test_scores = runtime.probabilities([texts[index] for index in test_ids])
    fine_test = evaluate(test_rows, test_truth, fine_test_scores, best["threshold"])
    control_test = evaluate(
        test_rows, test_truth, feature_scores(control, test_ids), control_threshold
    )
    prototype_test = evaluate(
        test_rows, test_truth, [prototype_scores[index] for index in test_ids], prototype_threshold
    )
    naive_test = classification_metrics(
        test_truth,
        majority_predictions([truth[index] for index in train_ids], len(test_ids)),
    )
    from safetensors.torch import save_file  # type: ignore[import-not-found]

    output.mkdir(parents=True, exist_ok=False)
    (output / "split-manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    save_file(control_weights, str(output / "frozen-head.safetensors"))
    save_file(best["weights"], str(output / "fine-tuned-adapter.safetensors"))
    selected_weights = best["weights"] if selected == "fine_tuned_encoder" else control_weights
    selected_threshold = (
        best["threshold"] if selected == "fine_tuned_encoder" else control_threshold
    )
    if selected != "fine_tuned_encoder":
        # Restore adapted blocks to pinned base for benchmarking the selected frozen model.
        runtime = Encoder(args.threads)
    runtime.restore(selected_weights)
    metadata = {
        "schema_version": "driver-action-pilot.v1",
        "base_repo": MODEL_REPO,
        "base_revision": MODEL_REVISION,
        "labels": list(LABELS),
        "selected_candidate": selected,
        "selected_by": "validation_exact_match_ties_prefer_frozen",
        "threshold": selected_threshold,
        "adapter_file": "selected-adapter.safetensors",
        "trained_scope": "conversational_actions_and_clarification_only",
        "draft_labels": True,
        "promoted": False,
    }
    save_file(selected_weights, str(output / metadata["adapter_file"]))
    (output / "model.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    durations: list[float] = []
    for index in test_ids[: min(40, len(test_ids))]:
        before = time.perf_counter()
        runtime.probabilities([texts[index]])
        durations.append((time.perf_counter() - before) * 1000)
    report = {
        "schema_version": "driver-training-report.v1",
        "metadata": metadata,
        "split_manifest": manifest,
        "settings": vars(args)
        | {
            "output": str(output),
            "dataset": str(args.dataset.resolve()),
        },
        "runtime": {
            "python": sys.version.split()[0],
            "torch": version("torch"),
            "transformers": version("transformers"),
            "device": "cpu",
        },
        "truncated_input_ids": truncated_ids,
        "prototype_baseline": {"threshold": prototype_threshold, "test": prototype_test},
        "no_input_majority_baseline": {"test": naive_test},
        "frozen_encoder_trained_head": {
            "threshold": control_threshold,
            "validation": control_validation,
            "test": control_test,
        },
        "fine_tuned_encoder": {
            "threshold": best["threshold"],
            "selected_step": best["step"],
            "validation": best["validation"],
            "test": fine_test,
        },
        "cpu_single_message_ms": {
            "samples": len(durations),
            "median": round(statistics.median(durations), 3),
            "p95": round(sorted(durations)[min(len(durations) - 1, int(len(durations) * 0.95))], 3),
            "includes": "tokenization_encoder_and_action_head_not_STT_Qwen_TTS",
        },
        "elapsed_seconds": round(time.perf_counter() - started, 2),
        "limitations": [
            "All labels are authored drafts, not independently human-reviewed ground truth.",
            "Group IDs are provisional; semantic overlap across compound families may remain.",
            "Singleton-group labels lack positive held-out coverage; see per-label support.",
            "No request topics, predicates, references, scopes or full ContextPlans are learned.",
            "Known/exposed synthetic corpus; not a locked release or real-race evaluation.",
        ],
    }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "selected": selected,
                "test_exact_match": {
                    "prototype": prototype_test["exact_match"],
                    "frozen": control_test["exact_match"],
                    "fine_tuned": fine_test["exact_match"],
                },
                "cpu_ms": report["cpu_single_message_ms"],
                "report": str(output / "report.json"),
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
