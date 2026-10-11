"""Pure contracts/helpers for an explicitly opt-in, draft-label classification pilot."""

import hashlib
import math
import random
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

from race_engineer.evaluation.driver_dataset import DriverMeaningDataset, DriverMeaningExample
from race_engineer.intelligence.driver_actions import (
    LABELS as LABELS,
)
from race_engineer.intelligence.driver_actions import (
    MODEL_FILES as MODEL_FILES,
)
from race_engineer.intelligence.driver_actions import (
    MODEL_REPO as MODEL_REPO,
)
from race_engineer.intelligence.driver_actions import (
    MODEL_REVISION as MODEL_REVISION,
)
from race_engineer.intelligence.driver_actions import (
    conversation_input as conversation_input,
)

ACTION_DESCRIPTIONS = (
    "Checking whether the race engineer can hear the driver. Telsiz ses kontrolü.",
    "Asking for racing information, analysis or advice. Yarış bilgisi veya tavsiye istiyor.",
    "Reporting a problem with the car. Araçta bir sorun bildiriyor.",
    "Venting frustration about racing or another driver. Yarışla ilgili sinirini ifade ediyor.",
    "Acknowledging or thanking the engineer. Mühendisi anladığını söylüyor veya teşekkür ediyor.",
    "Correcting the meaning of a previous question. Önceki sorusunun anlamını düzeltiyor.",
    "Requesting a communication preference change. İletişim tercihini değiştirmesini istiyor.",
    "Telling the engineer to stop nagging and let them race. "
    "Tekrarlanan uyarıları kesmesini istiyor.",
    "Asking for something unrelated to racing. Yarışla ilgisiz bir şey istiyor.",
    "An unclear or unintelligible driver message. Sürücü mesajının anlamı belirsiz.",
    "Greeting the race engineer. Yarış mühendisine selam veriyor.",
    "The driver request cannot be understood without clarification. "
    "İsteği açıklığa kavuşturmak gerekiyor.",
)


def model_input(example: DriverMeaningExample) -> str:
    """Only observed text and dialogue; never tags, gold labels or hidden ASR intent."""
    return conversation_input(
        example.text, [(entry.role, entry.text) for entry in example.recent_dialogue]
    )


def targets(example: DriverMeaningExample) -> list[int]:
    return [
        int(
            label in example.annotation.actions
            if label != "needs_clarification"
            else example.annotation.clarification_needed
        )
        for label in LABELS
    ]


def grouped_split(
    dataset: DriverMeaningDataset,
    seed: int = 17,
    *,
    allow_draft_labels: bool = False,
) -> dict[str, list[int]]:
    if any(row.review.status == "rejected" for row in dataset.examples):
        raise ValueError("rejected annotations are not allowed in the pilot")
    if not allow_draft_labels and any(row.review.status != "approved" for row in dataset.examples):
        raise ValueError("draft-label pilot requires explicit --allow-draft-labels")
    groups: dict[str, list[int]] = defaultdict(list)
    label_groups: dict[str, set[str]] = defaultdict(set)
    for index, row in enumerate(dataset.examples):
        groups[row.leakage_group].append(index)
        for label, value in zip(LABELS, targets(row), strict=True):
            if value:
                label_groups[label].add(row.leakage_group)
    # Singleton-group actions cannot honestly appear on both sides of the split.
    # Keep them in train, report their missing positive evaluation coverage.
    train_groups = {next(iter(values)) for values in label_groups.values() if len(values) == 1}
    rng = random.Random(seed)
    if any(not label_groups[label] for label in LABELS):
        raise ValueError("target has no positive examples")
    assigned_group = dict.fromkeys(train_groups, "train")
    required = {
        label: (
            ("train",)
            if len(label_groups[label]) == 1
            else ("train", "test")
            if len(label_groups[label]) == 2
            else ("train", "validation", "test")
        )
        for label in LABELS
    }
    priority = sorted(groups)
    rng.shuffle(priority)
    order = {group: index for index, group in enumerate(priority)}
    attempts = 0

    def cover() -> bool:
        nonlocal attempts
        attempts += 1
        if attempts > 10000:
            return False
        pending: list[tuple[int, str, str, set[str]]] = []
        for label, destinations in required.items():
            available = label_groups[label] - assigned_group.keys()
            missing = [
                destination
                for destination in destinations
                if not any(
                    assigned_group.get(group) == destination for group in label_groups[label]
                )
            ]
            if len(available) < len(missing):
                return False
            pending.extend(
                (len(available), label, destination, available) for destination in missing
            )
        if not pending:
            return True
        _, _, destination, available = min(pending, key=lambda item: item[:3])
        for group in sorted(available, key=lambda group: order[group]):
            assigned_group[group] = destination
            if cover():
                return True
            del assigned_group[group]
        return False

    if not cover():
        raise ValueError("cannot satisfy group-disjoint label coverage; add independent families")
    # Stratification uses labels only, before any model fit or evaluation.
    remaining = sorted(set(groups) - assigned_group.keys())
    rng.shuffle(remaining)
    total = len(dataset.examples)
    assigned = {
        name: sorted(group for group, split in assigned_group.items() if split == name)
        for name in ("train", "validation", "test")
    }
    for split in ("validation", "test"):
        count = sum(len(groups[group]) for group in assigned[split])
        while remaining and count < total * 0.15:
            group = remaining.pop()
            assigned[split].append(group)
            count += len(groups[group])
    assigned["train"].extend(remaining)
    result = {
        split: sorted(index for group in group_ids for index in groups[group])
        for split, group_ids in assigned.items()
    }
    if any(not indices for indices in result.values()):
        raise ValueError("not enough independent groups for train/validation/test")
    return result


def split_manifest(
    dataset: DriverMeaningDataset,
    splits: Mapping[str, Sequence[int]],
    seed: int,
) -> dict[str, object]:
    result: dict[str, object] = {}
    for name, indices in splits.items():
        rows = [dataset.examples[index] for index in indices]
        support = {
            label: sum(targets(row)[column] for row in rows) for column, label in enumerate(LABELS)
        }
        result[name] = {
            "count": len(rows),
            "ids": [row.example_id for row in rows],
            "groups": sorted({row.leakage_group for row in rows}),
            "languages": dict(Counter(row.language for row in rows)),
            "positive_support": support,
            "no_positive_coverage": [label for label, count in support.items() if count == 0],
        }
    return {
        "schema_version": "driver-training-split.v1",
        "seed": seed,
        "dataset_sha256": hashlib.sha256(dataset.model_dump_json().encode("utf-8")).hexdigest(),
        "source_review_status": dict(Counter(row.review.status for row in dataset.examples)),
        "evaluation_kind": "exploratory_synthetic_group_holdout_not_release_acceptance",
        "splits": result,
    }


def coverage_audit(dataset: DriverMeaningDataset, seed: int = 17) -> dict[str, Any]:
    """Read-only exploratory split feasibility; never fit models or assign corpus splits."""
    splits = grouped_split(dataset, seed, allow_draft_labels=True)
    result: dict[str, Any] = {}
    for name, indices in splits.items():
        rows = [dataset.examples[index] for index in indices]
        support = {
            label: sum(targets(row)[column] for row in rows) for column, label in enumerate(LABELS)
        }
        pure_vents = [row for row in rows if row.annotation.actions == ("vent",)]
        result[name] = {
            "count": len(rows),
            "languages": dict(Counter(row.language for row in rows)),
            "positive_support": support,
            "no_positive_coverage": [label for label, count in support.items() if not count],
            "pure_vent_examples": len(pure_vents),
            "pure_vent_groups": sorted({row.leakage_group for row in pure_vents}),
        }
    return {
        "seed": seed,
        "writes_splits": False,
        "training_ready": False,
        "all_targets_have_positive_coverage": all(
            not value["no_positive_coverage"] for value in result.values()
        ),
        "positive_group_counts": {
            label: len({row.leakage_group for row in dataset.examples if targets(row)[column]})
            for column, label in enumerate(LABELS)
        },
        "splits": result,
        "warning": "Draft synthetic labels and provisional groups; "
        "not independent acceptance data.",
    }


def classification_metrics(
    truth: Sequence[Sequence[int]],
    predicted: Sequence[Sequence[int]],
) -> dict[str, Any]:
    if not truth or len(truth) != len(predicted):
        raise ValueError("invalid metric row counts")
    if any(
        len(row) != len(LABELS) or any(value not in (0, 1) for value in row)
        for row in (*truth, *predicted)
    ):
        raise ValueError("invalid metric target vectors")
    per_label = {}
    total_tp = total_fp = total_fn = 0
    supported_f1 = []
    for column, label in enumerate(LABELS):
        tp = sum(
            row[column] == pred[column] == 1 for row, pred in zip(truth, predicted, strict=True)
        )
        fp = sum(
            row[column] == 0 and pred[column] == 1
            for row, pred in zip(truth, predicted, strict=True)
        )
        fn = sum(
            row[column] == 1 and pred[column] == 0
            for row, pred in zip(truth, predicted, strict=True)
        )
        support = tp + fn
        f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
        per_label[label] = {
            "support": support,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "f1": round(f1, 4) if support else None,
        }
        if support:
            supported_f1.append(f1)
        total_tp, total_fp, total_fn = total_tp + tp, total_fp + fp, total_fn + fn
    denominator = 2 * total_tp + total_fp + total_fn
    return {
        "rows": len(truth),
        "exact_match": round(
            sum(list(row) == list(pred) for row, pred in zip(truth, predicted, strict=True))
            / len(truth),
            4,
        ),
        "micro_f1": round(2 * total_tp / denominator, 4) if denominator else 0.0,
        "macro_f1_supported_labels_only": round(sum(supported_f1) / len(supported_f1), 4)
        if supported_f1
        else None,
        "unsupported_labels": [
            label for label, value in per_label.items() if value["support"] == 0
        ],
        "per_label": per_label,
    }


def majority_predictions(
    training_truth: Sequence[Sequence[int]],
    count: int,
) -> list[list[int]]:
    """No-input sanity baseline; fit one constant vector on training labels only."""
    if count < 1:
        raise ValueError("prediction count must be positive")
    classification_metrics(training_truth, training_truth)  # Validate target width/values.
    vector = [
        int(sum(row[column] for row in training_truth) * 2 > len(training_truth))
        for column in range(len(LABELS))
    ]
    return [vector.copy() for _ in range(count)]


def threshold_predictions(scores: Sequence[Sequence[float]], threshold: float) -> list[list[int]]:
    if not 0 <= threshold <= 1 or any(
        len(row) != len(LABELS)
        or any(not math.isfinite(value) or not 0 <= value <= 1 for value in row)
        for row in scores
    ):
        raise ValueError("invalid prediction scores")
    return [[int(value >= threshold) for value in row] for row in scores]


def select_threshold(
    scores: Sequence[Sequence[float]],
    truth: Sequence[Sequence[int]],
) -> float:
    # Predeclared small global grid; no per-label fitting to tiny validation counts.
    thresholds = (0.3, 0.4, 0.5, 0.6, 0.7)
    return max(
        thresholds,
        key=lambda threshold: (
            classification_metrics(truth, threshold_predictions(scores, threshold))["exact_match"],
            -abs(threshold - 0.5),
        ),
    )
