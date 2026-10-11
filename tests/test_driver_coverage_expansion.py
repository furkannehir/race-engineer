import copy
import importlib
import json
from collections import Counter
from pathlib import Path

import pytest

from race_engineer.evaluation.driver_dataset import DriverMeaningDataset, dataset_summary
from race_engineer.evaluation.driver_training import LABELS, coverage_audit, grouped_split

ROOT = Path(__file__).resolve().parents[1]
V1 = ROOT / "fixtures/driver-meaning/review-expanded.v1.json"
V2 = ROOT / "fixtures/driver-meaning/review-expanded.v2.json"
CATALOG = ROOT / "fixtures/driver-meaning/coverage-catalog.v2.json"


@pytest.fixture(scope="module")
def corpus():
    return DriverMeaningDataset.model_validate_json(V2.read_text(encoding="utf-8"))


@pytest.fixture
def builder(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    return importlib.import_module("build_driver_dataset")


def test_v2_preserves_all_v1_examples_and_adds_600_bilingual_drafts(corpus):
    parent = DriverMeaningDataset.model_validate_json(V1.read_text(encoding="utf-8"))
    assert corpus.examples[:800] == parent.examples
    summary = dataset_summary(corpus)
    assert summary["examples"] == 1400 and summary["languages"] == {"en": 700, "tr": 700}
    assert summary["scenarios"] == 157 and summary["leakage_groups"] == 100
    assert summary["review_status"] == {"draft": 1400}
    assert summary["training_ready"] is False and summary["split"] == "unassigned"
    assert all(
        row.origin == "assistant_authored" and row.review.reviewer is None
        for row in corpus.examples
    )
    assert summary["dialogue_examples"] == 228
    assert summary["clarification_examples"] == 80
    assert summary["synthetic_asr_examples"] == 28


def test_targeted_action_counts_are_genuinely_expanded(corpus):
    assert dataset_summary(corpus)["actions"] == {
        "radio_check": 62,
        "request_information": 708,
        "vent": 114,
        "unclear": 80,
        "report_problem": 132,
        "acknowledge": 52,
        "preference_change": 102,
        "correction": 82,
        "out_of_scope": 52,
        "communication_feedback": 92,
        "greeting": 50,
    }
    new = corpus.examples[800:]
    assert Counter(row.language for row in new) == {"en": 300, "tr": 300}
    assert sum(row.annotation.actions == ("vent",) for row in new) == 70
    assert sum(row.prior_exposure == "previously_probed" for row in new) == 40
    assert len({row.scenario_id for row in new}) == 60


def test_v2_is_reproducible_from_an_untouched_parent_and_authored_catalog(builder, corpus):
    parent = json.loads(V1.read_text(encoding="utf-8"))
    original = copy.deepcopy(parent)
    catalog = json.loads(CATALOG.read_text(encoding="utf-8"))
    assert len(catalog["families"]) == 60
    assert all(len(family["variants"]) == 5 for family in catalog["families"])
    actual = builder.build_dataset(parent, catalog)
    assert parent == original and actual == corpus
    assert V2.read_text(encoding="utf-8") == actual.model_dump_json(indent=2) + "\n"


@pytest.mark.parametrize("expected", [True, "700", 0, -1])
def test_builder_rejects_invalid_expected_counts(builder, expected):
    parent = json.loads(V1.read_text(encoding="utf-8"))
    catalog = json.loads(CATALOG.read_text(encoding="utf-8"))
    catalog["examples_per_language"] = expected
    with pytest.raises(ValueError, match="invalid examples_per_language"):
        builder.build_dataset(parent, catalog)


def test_semantic_contrasts_are_not_just_negative_word_keywords(corpus):
    rows = {row.example_id: row for row in corpus.examples}
    for language in ("en", "tr"):
        move = rows[f"ex-v2-door-slam-vent-01-{language}"]
        visibility = rows[f"ex-v2-dirty-windscreen-01-{language}"]
        assert move.annotation.actions == ("vent",)
        assert visibility.annotation.actions == ("report_problem",)
        assert move.annotation.requests == visibility.annotation.requests == ()
        neutral = rows[f"ex-v2-steering-vibration-01-{language}"]
        emotional = rows[f"ex-v2-vibration-and-vent-01-{language}"]
        assert neutral.annotation.actions == ("report_problem",)
        assert emotional.annotation.actions == ("vent", "report_problem")
        assert neutral.leakage_group == emotional.leakage_group
        past = rows[f"ex-v2-long-answer-feedback-01-{language}"]
        future = rows[f"ex-v2-prefer-number-first-01-{language}"]
        assert past.annotation.actions == ("communication_feedback",)
        assert future.annotation.actions == ("preference_change",)


def test_weather_requests_greetings_vents_and_out_of_scope_are_distinguished(corpus):
    rows = {row.example_id: row for row in corpus.examples}
    for language in ("en", "tr"):
        current = rows[f"ex-v2-casual-weather-question-01-{language}"]
        greeting = rows[f"ex-v2-weather-greeting-01-{language}"]
        forecast = rows[f"ex-v2-rain-forecast-question-01-{language}"]
        holiday = rows[f"ex-v2-holiday-weather-01-{language}"]
        assert current.annotation.actions == forecast.annotation.actions == ("request_information",)
        assert current.annotation.requests[0].time_scope == "current"
        assert forecast.annotation.requests[0].time_scope == "future_counterfactual"
        assert greeting.annotation.actions == ("greeting",) and not greeting.annotation.requests
        assert current.leakage_group == greeting.leakage_group == "current-weather"
        assert holiday.annotation.actions == ("out_of_scope",) and not holiday.annotation.requests
        followup = rows[f"ex-v2-weather-followup-01-{language}"]
        check = rows[f"ex-v2-presence-after-weather-01-{language}"]
        assert followup.recent_dialogue == check.recent_dialogue
        assert followup.leakage_group == check.leakage_group
        assert check.annotation.actions == ("radio_check",)
        assert followup.annotation.actions == ("request_information",)
        assert followup.annotation.requests[0].reference_basis == "dialogue"


def test_missing_meaning_is_not_the_same_as_unsupported_telemetry(corpus):
    rows = {row.example_id: row for row in corpus.examples}
    unclear = rows["ex-v2-missing-comparison-01-en"].annotation
    understood = rows["ex-v2-rain-forecast-question-01-en"].annotation
    assert unclear.clarification_needed and unclear.requests[0].resolution == "ambiguous"
    assert not understood.clarification_needed and understood.requests[0].resolution == "clear"
    social = rows["ex-v2-ambiguous-reaction-01-en"].annotation
    assert social.interpretation_status == "ambiguous" and not social.clarification_needed
    # User-supplied failures stay in the notes, not falsely authored as new gold.
    assert not any(
        row.text.casefold().strip(" .?!")
        in (
            "such a dirty move",
            "how's the weather today",
        )
        for row in corpus.examples[800:]
    )


def test_every_pilot_target_has_positive_coverage_without_cross_group_leakage(corpus):
    before = corpus.model_dump_json()
    audit = coverage_audit(corpus)
    assert audit["all_targets_have_positive_coverage"] is True
    assert audit["writes_splits"] is False and audit["training_ready"] is False
    assert min(audit["positive_group_counts"].values()) >= 4
    splits = grouped_split(corpus, allow_draft_labels=True)
    groups = {
        name: {corpus.examples[index].leakage_group for index in indices}
        for name, indices in splits.items()
    }
    assert not groups["train"] & groups["validation"]
    assert not groups["train"] & groups["test"]
    assert not groups["validation"] & groups["test"]
    assert sorted(index for indices in splits.values() for index in indices) == list(range(1400))
    for value in audit["splits"].values():
        assert value["languages"]["en"] == value["languages"]["tr"]
        assert value["no_positive_coverage"] == []
        assert all(value["positive_support"][label] > 0 for label in LABELS)
        assert value["pure_vent_examples"] > 0
    assert corpus.model_dump_json() == before


def test_inspector_can_check_v2_coverage_without_importing_model_runtime(monkeypatch, capsys):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    inspector = importlib.import_module("inspect_driver_dataset")
    before = V2.read_bytes()
    monkeypatch.setattr(
        "sys.argv", ["inspect_driver_dataset", "--dataset", str(V2), "--pilot-split-check"]
    )
    assert inspector.main() == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["examples"] == 1400
    assert summary["pilot_split_check"]["all_targets_have_positive_coverage"] is True
    assert V2.read_bytes() == before


def test_builder_v2_check_does_not_write_either_edition(builder, monkeypatch, capsys):
    before = (V1.read_bytes(), V2.read_bytes())
    monkeypatch.setattr("sys.argv", ["build_driver_dataset", "--edition", "v2", "--check"])
    assert builder.main() == 0
    assert json.loads(capsys.readouterr().out.split("Authored drafts only:")[0])["examples"] == 1400
    assert (V1.read_bytes(), V2.read_bytes()) == before


def test_training_honors_explicit_corpus_and_still_requires_opt_in(monkeypatch, tmp_path, capsys):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    trainer = importlib.import_module("train_driver_interpreter")
    monkeypatch.setattr(trainer, "ROOT", tmp_path)
    original = trainer.grouped_split
    seen = []

    def spy(dataset, *args, **kwargs):
        seen.append(len(dataset.examples))
        return original(dataset, *args, **kwargs)

    monkeypatch.setattr(trainer, "grouped_split", spy)
    monkeypatch.setattr(
        "sys.argv",
        [
            "train_driver_interpreter",
            "--dataset",
            str(V2),
            "--output",
            str(tmp_path / "data/new-run"),
        ],
    )
    with pytest.raises(SystemExit) as error:
        trainer.main()
    assert error.value.code == 2 and seen == [1400]
    assert "--allow-draft-labels" in capsys.readouterr().err


def test_training_invalid_corpus_fails_before_loading_any_model(monkeypatch, tmp_path, capsys):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    trainer = importlib.import_module("train_driver_interpreter")
    monkeypatch.setattr(trainer, "ROOT", tmp_path)
    invalid = tmp_path / "invalid.json"
    invalid.write_text('{"private-invalid-data": true}', encoding="utf-8")
    monkeypatch.setattr(
        "sys.argv",
        [
            "train_driver_interpreter",
            "--dataset",
            str(invalid),
            "--allow-draft-labels",
            "--output",
            str(tmp_path / "data/new-run"),
        ],
    )
    with pytest.raises(SystemExit) as error:
        trainer.main()
    assert error.value.code == 2
    output = capsys.readouterr().err
    assert "no model has been loaded" in output and "private-invalid-data" not in output


def test_inspector_failed_split_check_is_safe(monkeypatch, capsys):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    inspector = importlib.import_module("inspect_driver_dataset")

    def fail(*args):
        raise ValueError("private source detail")

    monkeypatch.setattr(inspector, "coverage_audit", fail)
    monkeypatch.setattr("sys.argv", ["inspect_driver_dataset", "--pilot-split-check"])
    assert inspector.main() == 2
    output = capsys.readouterr().err
    assert "Cannot construct" in output and "private source detail" not in output
