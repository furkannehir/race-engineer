import copy
import importlib
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from race_engineer.evaluation.driver_dataset import DriverMeaningDataset, dataset_summary

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "fixtures/driver-meaning/review-draft.v1.json"
EXPANDED = ROOT / "fixtures/driver-meaning/review-expanded.v1.json"
CATALOG = ROOT / "fixtures/driver-meaning/expansion-catalog.v1.json"


def raw_dataset():
    return json.loads(DATASET.read_text(encoding="utf-8"))


def corpus():
    return DriverMeaningDataset.model_validate(raw_dataset())


def expanded_corpus():
    return DriverMeaningDataset.model_validate_json(EXPANDED.read_text(encoding="utf-8"))


def test_seed_is_driver_only_bilingual_unreviewed_and_not_training_ready():
    dataset = corpus()
    summary = dataset_summary(dataset)
    assert summary["examples"] == 42
    assert summary["languages"] == {"en": 21, "tr": 21}
    assert summary["scenarios"] == 21 and summary["leakage_groups"] == 13
    assert summary["review_status"] == {"draft": 42}
    assert summary["training_ready"] is False and summary["split"] == "unassigned"
    assert all(example.speaker == "driver" for example in dataset.examples)
    assert all(example.origin == "assistant_authored" for example in dataset.examples)
    assert all(example.review.reviewer is None for example in dataset.examples)
    assert len(summary["sha256"]) == 64
    assert "Are you there?" not in json.dumps(summary)


def test_related_languages_and_context_contrasts_share_leakage_groups():
    examples = corpus().examples
    for scenario in {example.scenario_id for example in examples}:
        family = [example for example in examples if example.scenario_id == scenario]
        assert {example.language for example in family} == {"en", "tr"}
        assert len({example.leakage_group for example in family}) == 1
    for language in ("en", "tr"):
        family = [
            example for example in examples
            if example.language == language and example.leakage_group == "reachability-family"
        ]
        assert len(family) == 3 and len({example.text for example in family}) == 1
        assert {example.annotation.requests[0].focus for example in family} == {
            "unspecified", "fuel_sufficiency", "catch_feasibility",
        }
        assert sum(example.annotation.clarification_needed for example in family) == 1


def test_predicates_compounds_socials_and_open_vocabulary_are_preserved():
    examples = {example.example_id: example for example in corpus().examples}
    first = examples["dm03-en"].annotation.requests[0]
    last = examples["dm02-en"].annotation.requests[0]
    assert first.focus == last.focus and first.question != last.question
    compound = examples["dm09-en"].annotation
    assert len(compound.requests) == 2
    assert compound.actions == ("request_information",)
    mixed = examples["dm08-en"].annotation
    assert mixed.actions == ("vent", "request_information") and len(mixed.requests) == 1
    assert examples["dm01-en"].annotation.requests == ()
    assert examples["dm06-en"].annotation.requests == ()
    assert examples["dm07-en"].annotation.interpretation_status == "ambiguous"
    assert not examples["dm07-en"].annotation.clarification_needed
    assert examples["dm16-en"].annotation.requests[0].focus == "tyre_stint_feasibility"
    assert examples["dm19-en"].annotation.requests[0].subject == "car_behind"
    assert examples["dm14-en"].annotation.requests[0].window_seconds == 20


def test_back_off_feedback_requires_reminder_context_and_is_not_permanent_preference():
    examples = {example.example_id: example for example in corpus().examples}
    for language in ("en", "tr"):
        standalone = examples[f"dm07-{language}"]
        contextual = examples[f"dm21-{language}"]
        assert standalone.text == contextual.text
        assert standalone.leakage_group == contextual.leakage_group
        assert standalone.annotation.actions == ("unclear",)
        assert standalone.annotation.interpretation_status == "ambiguous"
        assert contextual.annotation.actions == ("communication_feedback",)
        assert contextual.annotation.interpretation_status == "clear"
        assert contextual.annotation.requests == ()
        assert not contextual.annotation.clarification_needed
        assert len(contextual.recent_dialogue) == 4
        assert contextual.review.status == "draft"
        assert contextual.origin == "assistant_authored"
        assert "permanent" in contextual.annotation.rationale
        assert examples[f"dm18-{language}"].annotation.actions == ("preference_change",)


def test_duplicate_ids_full_inputs_and_mismatched_scenario_groups_are_rejected():
    raw = raw_dataset()
    raw["examples"][1]["example_id"] = raw["examples"][0]["example_id"]
    with pytest.raises(ValidationError, match="duplicate example IDs"):
        DriverMeaningDataset.model_validate(raw)
    raw = raw_dataset()
    duplicate = copy.deepcopy(raw["examples"][0])
    duplicate["example_id"] = "different-id"
    raw["examples"].append(duplicate)
    with pytest.raises(ValidationError, match="duplicate full inputs"):
        DriverMeaningDataset.model_validate(raw)
    raw = raw_dataset()
    raw["examples"][1]["leakage_group"] = "different-group"
    with pytest.raises(ValidationError, match="share a leakage group"):
        DriverMeaningDataset.model_validate(raw)


@pytest.mark.parametrize("changes", [
    {"speaker": "engineer"}, {"split": "test"}, {"origin": "real_session_log"},
    {"review": {"status": "approved"}},
    {"review": {"status": "draft", "reviewer": "imaginary-reviewer"}},
    {"invented_extra_field": "ignored"},
])
def test_schema_rejects_speaker_split_provenance_and_fake_review(changes):
    raw = raw_dataset()
    raw["examples"][0].update(changes)
    with pytest.raises(ValidationError):
        DriverMeaningDataset.model_validate(raw)


@pytest.mark.parametrize("changes", [
    {"source_span": "not in the current message"},
    {"reference_basis": "dialogue"},
    {"subject": "unresolved"},
    {"focus": "unspecified"},
    {"time_scope": "unclear"},
    {"window_seconds": 20},
    {"resolution": "ambiguous"},
])
def test_request_consistency_and_span_are_checked(changes):
    raw = raw_dataset()
    raw["examples"][2]["annotation"]["requests"][0].update(changes)
    with pytest.raises(ValidationError):
        DriverMeaningDataset.model_validate(raw)


@pytest.mark.parametrize("changes", [
    {"actions": ["radio_check", "radio_check"]},
    {"actions": ["request_information"]},
    {"interpretation_status": "ambiguous"},
    {"clarification_needed": True},
])
def test_annotation_consistency_is_checked(changes):
    raw = raw_dataset()
    raw["examples"][0]["annotation"].update(changes)
    with pytest.raises(ValidationError):
        DriverMeaningDataset.model_validate(raw)


def test_expansion_has_800_balanced_drafts_and_preserves_the_original_seed():
    dataset = expanded_corpus()
    summary = dataset_summary(dataset)
    assert summary["examples"] == 800
    assert summary["languages"] == {"en": 400, "tr": 400}
    assert summary["scenarios"] == 97 and summary["leakage_groups"] == 59
    assert summary["review_status"] == {"draft": 800}
    assert summary["training_ready"] is False
    assert dataset.examples[:42] == corpus().examples
    assert all(row.origin == "assistant_authored" and row.split == "unassigned"
               for row in dataset.examples)
    assert all(row.review.reviewer is None for row in dataset.examples)
    assert set(summary["actions"]) == {
        "radio_check", "request_information", "report_problem", "vent", "acknowledge",
        "correction", "preference_change", "communication_feedback", "out_of_scope", "unclear",
        "greeting",
    }


def test_expansion_covers_compounds_history_uncertainty_code_switch_and_noise():
    dataset = expanded_corpus()
    summary = dataset_summary(dataset)
    assert summary["compound_examples"] == 72
    assert summary["dialogue_examples"] == 78
    assert summary["clarification_examples"] == 30
    assert summary["synthetic_asr_examples"] == 18
    assert summary["coverage_tags"]["code-switch"] == 10
    mixed_scope = [row for row in dataset.examples
                   if len({request.time_scope for request in row.annotation.requests}) > 1]
    assert len(mixed_scope) == 10
    recoverable = [row for row in dataset.examples if "recoverable-noise" in row.coverage_tags]
    unrecoverable = [row for row in dataset.examples if "unrecoverable-noise" in row.coverage_tags]
    assert len(recoverable) == 10 and len(unrecoverable) == 8
    assert all(row.intended_text and row.text != row.intended_text
               for row in (*recoverable, *unrecoverable))
    assert all(not row.annotation.requests and row.annotation.clarification_needed
               for row in unrecoverable)
    assert len(summary["request_focuses"]) >= 40


def test_authored_families_are_bilingual_grouped_and_not_prefix_cartesian_padding():
    dataset = expanded_corpus()
    rows = {row.example_id: row for row in dataset.examples}
    catalog = json.loads(CATALOG.read_text(encoding="utf-8"))
    assert len(catalog["families"]) == 76
    assert sum(len(family["variants"]) for family in catalog["families"]) == 379
    for family in catalog["families"]:
        for index, variant in enumerate(family["variants"], 1):
            for language in ("en", "tr"):
                row = rows[f"ex-{family['id']}-{index:02d}-{language}"]
                assert row.text == variant[language]
                assert row.leakage_group == family["leakage_group"]
                for request in row.annotation.requests:
                    assert request.source_span in row.text
                    if len(row.annotation.requests) > 1:
                        assert request.source_span == variant["spans"][language][request.request_id]
    for scenario in {row.scenario_id for row in dataset.examples}:
        family_rows = [row for row in dataset.examples if row.scenario_id == scenario]
        assert {row.language for row in family_rows} == {"en", "tr"}
        assert len({row.leakage_group for row in family_rows}) == 1


def test_normalized_input_duplicates_are_rejected_without_merging_context_contrasts():
    raw = raw_dataset()
    duplicate = copy.deepcopy(raw["examples"][0])
    duplicate["example_id"] = "punctuation-only-duplicate"
    duplicate["text"] = "  ARE   YOU THERE!!!  "
    raw["examples"].append(duplicate)
    with pytest.raises(ValidationError, match="duplicate full inputs"):
        DriverMeaningDataset.model_validate(raw)
    assert expanded_corpus()  # Same question with distinct histories remains valid.


@pytest.mark.parametrize("changes", [
    {"coverage_tags": ["dialogue", "dialogue"]},
    {"input_kind": "synthetic_asr"},
    {"intended_text": "Hidden intended question without synthetic-ASR provenance"},
])
def test_expansion_metadata_cannot_claim_real_or_unmarked_asr(changes):
    raw = raw_dataset()
    raw["examples"][0].update(changes)
    with pytest.raises(ValidationError):
        DriverMeaningDataset.model_validate(raw)


@pytest.fixture
def inspector(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    return importlib.import_module("inspect_driver_dataset")


def test_cli_summary_and_language_review_listing(inspector, monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["inspect_driver_dataset"])
    assert inspector.main() == 0
    assert json.loads(capsys.readouterr().out)["examples"] == 1400
    monkeypatch.setattr("sys.argv", ["inspect_driver_dataset", "--language", "tr", "--list"])
    assert inspector.main() == 0
    output = capsys.readouterr().out
    assert "dm01-tr [tr]" in output and "Beni duyuyor musun?" in output
    assert "dm01-en [en]" not in output
    monkeypatch.setattr("sys.argv", ["inspect_driver_dataset", "--example", "dm11-tr"])
    assert inspector.main() == 0
    example = json.loads(capsys.readouterr().out)
    assert example["annotation"]["requests"][0]["focus"] == "fuel_sufficiency"
    assert len(example["recent_dialogue"]) == 2


def test_cli_review_filters_pagination_and_coverage(inspector, monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", [
        "inspect_driver_dataset", "--language", "tr", "--action", "communication_feedback",
        "--tag", "dialogue", "--limit", "2", "--offset", "1", "--list",
    ])
    assert inspector.main() == 0
    output = capsys.readouterr().out
    displayed = [line for line in output.splitlines() if " [tr] " in line]
    assert len(displayed) == 2 and all("communication_feedback" in line for line in displayed)
    assert "offset=1, limit=2" in output
    monkeypatch.setattr("sys.argv", [
        "inspect_driver_dataset", "--focus", "speed_maximum", "--language", "en",
        "--group", "speed-minute-statistics", "--list", "--coverage",
    ])
    assert inspector.main() == 0
    output = capsys.readouterr().out
    assert len([line for line in output.splitlines() if " [en] " in line]) == 5
    assert '"request_focuses"' in output
    monkeypatch.setattr("sys.argv", ["inspect_driver_dataset", "--limit", "0"])
    with pytest.raises(SystemExit) as error:
        inspector.main()
    assert error.value.code == 2


@pytest.fixture
def builder(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    return importlib.import_module("build_driver_dataset")


def test_expansion_is_reproducible_and_does_not_mutate_seed(builder):
    seed = raw_dataset()
    original = copy.deepcopy(seed)
    catalog = json.loads(CATALOG.read_text(encoding="utf-8"))
    actual = builder.build_dataset(seed, catalog)
    assert seed == original
    assert actual == expanded_corpus()
    assert EXPANDED.read_text(encoding="utf-8") == actual.model_dump_json(indent=2) + "\n"


def test_builder_checks_output_and_refuses_to_overwrite_an_edited_file(
    builder, monkeypatch, capsys, tmp_path,
):
    monkeypatch.setattr("sys.argv", ["build_driver_dataset", "--check"])
    assert builder.main() == 0
    capsys.readouterr()
    edited = tmp_path / "edited.json"
    edited.write_text("do not overwrite", encoding="utf-8")
    monkeypatch.setattr(builder, "OUTPUT", edited)
    monkeypatch.setattr("sys.argv", ["build_driver_dataset", "--write"])
    assert builder.main() == 2
    assert edited.read_text(encoding="utf-8") == "do not overwrite"
    assert "refusing to overwrite" in capsys.readouterr().out


def test_builder_rejects_seed_drift_wrong_counts_and_missing_compound_spans(builder):
    catalog = json.loads(CATALOG.read_text(encoding="utf-8"))
    seed = raw_dataset()
    seed["revision"] = "changed"
    with pytest.raises(ValueError, match="seed revision"):
        builder.build_dataset(seed, catalog)
    seed = raw_dataset()
    truncated = copy.deepcopy(catalog)
    truncated["families"].pop()
    with pytest.raises(ValueError, match="exactly 400"):
        builder.build_dataset(seed, truncated)
    broken = copy.deepcopy(catalog)
    compound = next(family for family in broken["families"]
                    if len(family["annotation"].get("requests", [])) > 1)
    del compound["variants"][0]["spans"]
    with pytest.raises(ValueError, match="compound variants"):
        builder.build_dataset(seed, broken)


def test_cli_unknown_example_missing_file_and_invalid_dataset_are_safe(
    inspector, monkeypatch, capsys, tmp_path,
):
    monkeypatch.setattr("sys.argv", ["inspect_driver_dataset", "--example", "missing"])
    assert inspector.main() == 1
    capsys.readouterr()
    monkeypatch.setattr("sys.argv", ["inspect_driver_dataset", "--dataset", str(tmp_path / "none")])
    assert inspector.main() == 2
    capsys.readouterr()
    path = tmp_path / "invalid.json"
    raw = raw_dataset()
    raw["examples"][0]["text"] = "private test content"
    raw["examples"][0]["speaker"] = "private invalid speaker"
    path.write_text(json.dumps(raw), encoding="utf-8")
    monkeypatch.setattr("sys.argv", ["inspect_driver_dataset", "--dataset", str(path)])
    assert inspector.main() == 2
    output = capsys.readouterr().err
    assert "validation error" in output and "private" not in output
