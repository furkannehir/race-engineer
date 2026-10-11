import importlib
import json
from collections import Counter
from pathlib import Path

import pytest

from race_engineer.evaluation.driver_dataset import DriverMeaningDataset
from race_engineer.evaluation.driver_training import (
    LABELS,
    MODEL_REPO,
    MODEL_REVISION,
    classification_metrics,
    conversation_input,
    grouped_split,
    majority_predictions,
    model_input,
    select_threshold,
    split_manifest,
    targets,
    threshold_predictions,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def corpus():
    return DriverMeaningDataset.model_validate_json(
        (ROOT / "fixtures/driver-meaning/review-expanded.v1.json").read_text(encoding="utf-8"),
    )


def vector(*labels):
    return [int(label in labels) for label in LABELS]


def test_draft_labels_require_opt_in_and_review_status_is_not_changed(corpus):
    before = corpus.model_dump_json()
    with pytest.raises(ValueError, match="explicit --allow-draft-labels"):
        grouped_split(corpus)
    grouped_split(corpus, allow_draft_labels=True)
    assert corpus.model_dump_json() == before
    first = corpus.examples[0]
    rejected_row = first.model_copy(
        update={
            "review": first.review.model_copy(update={"status": "rejected"}),
        }
    )
    rejected = corpus.model_copy(update={"examples": (rejected_row, *corpus.examples[1:])})
    with pytest.raises(ValueError, match="rejected"):
        grouped_split(rejected, allow_draft_labels=True)


def test_split_is_reproducible_balanced_complete_and_group_disjoint(corpus):
    splits = grouped_split(corpus, allow_draft_labels=True)
    assert splits == grouped_split(corpus, allow_draft_labels=True)
    assert sorted(index for indices in splits.values() for index in indices) == list(range(800))
    groups = {
        name: {corpus.examples[index].leakage_group for index in indices}
        for name, indices in splits.items()
    }
    assert not groups["train"] & groups["test"]
    assert not groups["train"] & groups["validation"]
    assert not groups["validation"] & groups["test"]
    assert {name: len(indices) for name, indices in splits.items()} == {
        "train": 552,
        "validation": 126,
        "test": 122,
    }
    for indices in splits.values():
        languages = Counter(corpus.examples[index].language for index in indices)
        assert languages["en"] == languages["tr"]
    manifest = split_manifest(corpus, splits, 17)
    assert manifest["source_review_status"] == {"draft": 800}
    unsupported = manifest["splits"]["test"]["no_positive_coverage"]
    assert set(unsupported) == {"radio_check", "acknowledge", "out_of_scope", "greeting"}


def test_model_input_uses_only_observed_text_and_role_separated_history(corpus):
    noise = next(row for row in corpus.examples if row.intended_text)
    assert model_input(noise) == f"Current driver message:\n{noise.text}"
    contextual = next(row for row in corpus.examples if row.example_id == "dm21-en")
    assert model_input(contextual) == conversation_input(
        contextual.text,
        [(turn.role, turn.text) for turn in contextual.recent_dialogue],
    )
    assert "engineer:" in model_input(contextual)
    assert contextual.annotation.rationale not in model_input(contextual)
    assert contextual.example_id not in model_input(contextual)
    assert targets(contextual) == vector("communication_feedback")
    ambiguous = next(row for row in corpus.examples if row.example_id == "dm10-en")
    assert targets(ambiguous) == vector("request_information", "needs_clarification")


def test_metrics_report_unsupported_labels_without_hiding_false_positives():
    metrics = classification_metrics(
        [vector("vent"), vector("request_information")],
        [vector("vent", "greeting"), vector("request_information")],
    )
    assert metrics["exact_match"] == 0.5
    assert metrics["micro_f1"] == 0.8
    assert metrics["macro_f1_supported_labels_only"] == 1.0
    assert metrics["per_label"]["greeting"] == {
        "support": 0,
        "tp": 0,
        "fp": 1,
        "fn": 0,
        "f1": None,
    }


@pytest.mark.parametrize(
    "truth,predicted",
    [([], []), ([vector("vent")], []), ([[2] * len(LABELS)], [[0] * len(LABELS)]), ([[1]], [[1]])],
)
def test_metrics_reject_invalid_vectors(truth, predicted):
    with pytest.raises(ValueError):
        classification_metrics(truth, predicted)


def test_majority_baseline_uses_training_labels_only_and_loses_ties():
    predictions = majority_predictions(
        [vector("request_information"), vector("request_information", "vent")],
        3,
    )
    assert predictions == [vector("request_information")] * 3
    predictions[0][0] = 1
    assert predictions[1][0] == 0
    with pytest.raises(ValueError):
        majority_predictions([], 1)
    with pytest.raises(ValueError):
        majority_predictions([vector("vent")], 0)


def test_original_pilot_majority_baseline_exposes_class_imbalance(corpus):
    splits = grouped_split(corpus, allow_draft_labels=True)
    truth = [targets(corpus.examples[index]) for index in splits["test"]]
    predicted = majority_predictions(
        [targets(corpus.examples[index]) for index in splits["train"]],
        len(truth),
    )
    assert predicted == [vector("request_information")] * 122
    assert classification_metrics(truth, predicted)["exact_match"] == 0.4262


def test_threshold_selection_is_validation_only_and_prefers_middle_on_ties():
    scores = [[0.05] * len(LABELS)]
    scores[0][LABELS.index("vent")] = 0.45
    assert select_threshold(scores, [vector("vent")]) == 0.4
    assert select_threshold([[0.9] * len(LABELS)], [[1] * len(LABELS)]) == 0.5
    assert threshold_predictions(scores, 0.4) == [vector("vent")]


@pytest.mark.parametrize(
    "scores,threshold",
    [
        ([[float("nan")] * len(LABELS)], 0.5),
        ([[1.1] * len(LABELS)], 0.5),
        ([[0.2]], 0.5),
        ([[0.5] * len(LABELS)], float("nan")),
    ],
)
def test_invalid_probabilities_are_rejected(scores, threshold):
    with pytest.raises(ValueError):
        threshold_predictions(scores, threshold)


@pytest.fixture
def playground(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    return importlib.import_module("try_driver_interpreter")


def test_metadata_is_pinned_and_cannot_load_arbitrary_adapters(playground, monkeypatch, tmp_path):
    monkeypatch.setattr(playground, "ROOT", tmp_path)
    path = tmp_path / "data/run"
    path.mkdir(parents=True)
    metadata = {
        "schema_version": "driver-action-pilot.v1",
        "base_repo": MODEL_REPO,
        "base_revision": MODEL_REVISION,
        "labels": list(LABELS),
        "adapter_file": "selected-adapter.safetensors",
        "threshold": 0.5,
        "trained_scope": "conversational_actions_and_clarification_only",
        "draft_labels": True,
        "promoted": False,
    }
    model = path / "model.json"
    model.write_text(json.dumps(metadata), encoding="utf-8")
    assert playground.load_metadata(path) == metadata
    for key, value in (
        ("labels", list(reversed(LABELS))),
        ("base_revision", "wrong"),
        ("adapter_file", "../../outside.safetensors"),
        ("threshold", True),
        ("threshold", float("nan")),
    ):
        model.write_text(json.dumps(metadata | {key: value}), encoding="utf-8")
        with pytest.raises(ValueError):
            playground.load_metadata(path)
    with pytest.raises(ValueError, match="inside repository"):
        playground.load_metadata(tmp_path)


def test_playground_supports_optional_bounded_history_and_no_forced_action(
    playground,
    monkeypatch,
    capsys,
):
    inputs = iter(
        [
            "/history on",
            "/engineer The car behind is closing.",
            "I know what I'm doing.",
            "/history off",
            "Hello?",
            "/quit",
        ]
    )
    seen = []

    class FakeClassifier:
        threshold = 0.5

        def __init__(self, *args):
            pass

        def predict(self, text):
            seen.append(text)
            return [0.1] * len(LABELS)

    monkeypatch.setattr(playground, "LocalClassifier", FakeClassifier)
    monkeypatch.setattr("sys.argv", ["try_driver_interpreter"])
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))
    assert playground.main() == 0
    output = capsys.readouterr().out
    assert '"actions": []' in output and '"scope": "action_classifier_only' in output
    assert "engineer: The car behind is closing." in seen[0]
    assert seen[1] == "Current driver message:\nHello?"


def test_playground_load_failure_is_actionable_and_non_mutating(playground, monkeypatch, capsys):
    def fail(*args):
        raise OSError("private path must not be printed")

    monkeypatch.setattr(playground, "LocalClassifier", fail)
    monkeypatch.setattr("sys.argv", ["try_driver_interpreter"])
    assert playground.main() == 2
    output = capsys.readouterr().err
    assert "optional STT runtime Python" in output and "private path" not in output


def test_training_refuses_drafts_before_loading_encoder(monkeypatch, tmp_path, corpus, capsys):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    trainer = importlib.import_module("train_driver_interpreter")
    monkeypatch.setattr(trainer, "ROOT", tmp_path)
    source = tmp_path / "fixtures/driver-meaning/review-expanded.v1.json"
    source.parent.mkdir(parents=True)
    source.write_text(corpus.model_dump_json(), encoding="utf-8")
    output = tmp_path / "data/new-run"

    def fail(*args):
        pytest.fail("must reject missing opt-in before any model load")

    monkeypatch.setattr(trainer, "Encoder", fail)
    monkeypatch.setattr("sys.argv", ["train_driver_interpreter", "--output", str(output)])
    with pytest.raises(SystemExit) as error:
        trainer.main()
    assert error.value.code == 2 and not output.exists()
    assert "--allow-draft-labels" in capsys.readouterr().err


def test_training_never_overwrites_existing_run(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    trainer = importlib.import_module("train_driver_interpreter")
    monkeypatch.setattr(trainer, "ROOT", tmp_path)
    output = tmp_path / "data/existing-run"
    output.mkdir(parents=True)
    sentinel = output / "report.json"
    sentinel.write_text("original result", encoding="utf-8")
    monkeypatch.setattr(
        "sys.argv", ["train_driver_interpreter", "--allow-draft-labels", "--output", str(output)]
    )
    with pytest.raises(SystemExit) as error:
        trainer.main()
    assert error.value.code == 2
    assert sentinel.read_text(encoding="utf-8") == "original result"


def test_adapter_restore_requires_complete_head_and_either_full_or_no_body(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    runtime_module = importlib.import_module("driver_pilot_runtime")
    calls = []

    class FakeLayer:
        def state_dict(self):
            return {"weight": 1, "bias": 2}

    class FakeBackbone:
        class LayerContainer:
            def __init__(self):
                self.layer = [FakeLayer() for _ in range(12)]

        encoder = LayerContainer()

        def load_state_dict(self, weights, strict=False):
            calls.append(("backbone", weights))

    class FakeHead:
        def load_state_dict(self, weights):
            calls.append(("head", weights))

    encoder = object.__new__(runtime_module.Encoder)
    encoder.backbone, encoder.head = FakeBackbone(), FakeHead()
    head = {"head.weight": 1, "head.bias": 2}
    for invalid in (
        {"head.weight": 1},
        head | {"other": 3},
        head | {"backbone.encoder.layer.10.weight": 3},
    ):
        with pytest.raises(ValueError, match="unexpected adapter"):
            encoder.restore(invalid)
    assert calls == []
    encoder.restore(head)
    assert calls[-1] == ("head", {"weight": 1, "bias": 2})
    full = head | {
        f"backbone.encoder.layer.{index}.{name}": 3
        for index in (10, 11)
        for name in ("weight", "bias")
    }
    encoder.restore(full)
    assert len(calls[-2][1]) == 4
