import asyncio
import json
from collections.abc import Sequence
from pathlib import Path

import pytest
from context_plan_fixtures import wire_plan
from pydantic import ValidationError

from race_engineer.application import intelligence_evaluation as evaluation_application
from race_engineer.cli import _parser
from race_engineer.config import ConversationConfig, PolicyContextConfig
from race_engineer.core.intelligence import (
    CapabilityDescriptor,
    CapabilityRequest,
    ContextPlan,
    DriverTurn,
    EvidenceQuery,
    SignalDescriptor,
    SignalSelector,
)
from race_engineer.evaluation.intelligence import (
    ExpectedContextPlan,
    ExpectedEvidenceQuery,
    IntelligenceCandidateMetadata,
    IntelligenceEvaluationDataset,
    load_intelligence_evaluation_dataset,
    validate_intelligence_split_families,
)
from race_engineer.evaluation.intelligence_runner import (
    evaluate_intelligence_datasets,
)
from race_engineer.intelligence.context_planner import (
    PLANNER_V5_ID,
    PLANNER_V6_ID,
    QwenContextQueryPlanner,
)
from race_engineer.intelligence.local_model import LocalIntelligenceError

ROOT = Path(__file__).parents[1]
DATASETS = tuple(
    ROOT / "fixtures" / "intelligence-evaluation" / name
    for name in ("development.json", "calibration.json", "locked.json")
)
CAPABILITIES = {
    "current_classification",
    "fuel_range",
    "position_change",
    "gap_ahead",
    "gap_behind",
    "relative_pace_ahead",
    "relative_pace_behind",
    "catch_time_ahead",
    "catch_time_behind",
    "pit_loss_projection",
    "pit_stop_duration_projection",
    "projected_rejoin_position",
}


def _candidate() -> IntelligenceCandidateMetadata:
    return IntelligenceCandidateMetadata(
        candidate_id="scripted-context-v1",
        runtime_profile="portable",
        candidate_kind="control",
        adapter="scripted",
        model="test-control",
        model_revision="1",
    )


class ScriptedContextPlanner:
    async def plan(
        self,
        turn: DriverTurn,
        signals: Sequence[SignalDescriptor],
        capabilities: Sequence[CapabilityDescriptor],
    ) -> ContextPlan:
        del signals, capabilities
        if turn.transcript == "private classification question":
            return ContextPlan(
                turn_id=turn.turn_id,
                planner_id="scripted-context-v1",
                capability_requests=(
                    CapabilityRequest(
                        request_id="c1",
                        capability_id="current_classification",
                    ),
                ),
            )
        if turn.transcript == "private speed question":
            return ContextPlan(
                turn_id=turn.turn_id,
                planner_id="scripted-context-v1",
                queries=(
                    EvidenceQuery(
                        query_id="e1",
                        selector=SignalSelector(source="player", signal="speed_mps"),
                        operation="latest",
                    ),
                ),
            )
        if turn.transcript == "private social remark":
            return ContextPlan(
                turn_id=turn.turn_id,
                planner_id="scripted-context-v1",
                temporal_scope="social",
            )
        return ContextPlan(
            turn_id=turn.turn_id,
            planner_id="scripted-context-v1",
            temporal_scope="future_counterfactual",
            capability_requests=(
                CapabilityRequest(
                    request_id="c1",
                    capability_id="projected_rejoin_position",
                ),
            ),
        )


class FailingContextPlanner:
    async def plan(
        self,
        turn: DriverTurn,
        signals: Sequence[SignalDescriptor],
        capabilities: Sequence[CapabilityDescriptor],
    ) -> ContextPlan:
        del turn, signals, capabilities
        raise LocalIntelligenceError("context_model_unreachable")


class ExpectedContextPlanner:
    def __init__(self, plans: dict[str, ExpectedContextPlan]) -> None:
        self._plans = plans

    async def plan(
        self,
        turn: DriverTurn,
        signals: Sequence[SignalDescriptor],
        capabilities: Sequence[CapabilityDescriptor],
    ) -> ContextPlan:
        del signals, capabilities
        expected = self._plans[turn.transcript]
        return ContextPlan(
            turn_id=turn.turn_id,
            planner_id="dataset-control-v1",
            temporal_scope=expected.temporal_scope,
            situation={
                "social": ("driver_social_turn",),
                "race_information": ("race_information_request",),
                "mixed": ("driver_social_turn", "race_information_request"),
                None: (),
            }[expected.purpose],
            queries=tuple(
                EvidenceQuery(
                    query_id=f"e{index}",
                    selector=SignalSelector(
                        source=query.source,
                        signal=query.signal,
                        subject_id=query.subject_id,
                    ),
                    operation=query.operation,
                    window_s=query.window_s,
                )
                for index, query in enumerate(expected.queries, start=1)
            ),
            capability_requests=tuple(
                CapabilityRequest(
                    request_id=f"c{index}",
                    capability_id=capability_id,
                )
                for index, capability_id in enumerate(expected.capability_ids, start=1)
            ),
        )


def _runner_dataset() -> IntelligenceEvaluationDataset:
    return IntelligenceEvaluationDataset.model_validate(
        {
            "dataset_id": "intelligence-runner-test",
            "revision": "1",
            "split": "development",
            "fixture": "fixtures/synthetic/intelligence-capabilities",
            "groups": [
                {
                    "id": "classification-control",
                    "family": "classification-control-family",
                    "category": "classification",
                    "language": "en",
                    "questions": ["private classification question"],
                    "expected": {
                        "temporal_scope": "current",
                        "capability_ids": ["current_classification"],
                        "evidence_outcome": "available",
                    },
                },
                {
                    "id": "speed-control",
                    "family": "speed-control-family",
                    "category": "raw-signal",
                    "language": "tr",
                    "questions": ["private speed question"],
                    "expected": {
                        "temporal_scope": "current",
                        "queries": [
                            {
                                "source": "player",
                                "signal": "speed_mps",
                                "operation": "latest",
                            }
                        ],
                        "evidence_outcome": "available",
                    },
                },
                {
                    "id": "social-control",
                    "family": "social-control-family",
                    "category": "social",
                    "language": "en",
                    "questions": ["private social remark"],
                    "expected": {
                        "temporal_scope": "social",
                        "evidence_outcome": "no_evidence",
                    },
                },
                {
                    "id": "rejoin-control",
                    "family": "rejoin-control-family",
                    "category": "strategy",
                    "language": "tr",
                    "questions": ["private rejoin question"],
                    "expected": {
                        "temporal_scope": "future_counterfactual",
                        "capability_ids": ["projected_rejoin_position"],
                        "evidence_outcome": "unavailable",
                        "required_unknowns": [
                            "projected_rejoin_position:missing_rejoin_projection_model"
                        ],
                    },
                },
            ],
        }
    )


def test_committed_intelligence_datasets_are_bilingual_disjoint_and_cover_registry() -> None:
    datasets = tuple(load_intelligence_evaluation_dataset(path) for path in DATASETS)
    validate_intelligence_split_families(datasets)

    assert all(dataset.language_counts()["en"] > 0 for dataset in datasets)
    assert all(dataset.language_counts()["tr"] > 0 for dataset in datasets)
    covered = {
        capability_id
        for dataset in datasets
        for group in dataset.groups
        for capability_id in group.expected.capability_ids
    }
    assert covered == CAPABILITIES
    assert sum(dataset.turn_count for dataset in datasets) == 32


def test_committed_expectations_match_deterministic_fixture_execution() -> None:
    datasets = tuple(load_intelligence_evaluation_dataset(path) for path in DATASETS)
    expected = {
        question: group.expected
        for dataset in datasets
        for group in dataset.groups
        for question in group.questions
    }

    results, summary = asyncio.run(
        evaluate_intelligence_datasets(
            ROOT,
            PolicyContextConfig(),
            ExpectedContextPlanner(expected),
            _candidate(),
            datasets,
        )
    )

    assert len(results) == 32
    assert all(result["passed"] for result in results)
    assert summary["overall"]["exact_turn_accuracy"] == 1.0


def test_expected_query_applies_real_evidence_query_validation() -> None:
    with pytest.raises(ValidationError, match="field selectors require an aggregate"):
        ExpectedEvidenceQuery.model_validate(
            {"source": "field", "signal": "position", "operation": "latest"}
        )


def test_context_candidate_runner_is_content_free_and_scores_evidence() -> None:
    results, summary = asyncio.run(
        evaluate_intelligence_datasets(
            ROOT,
            PolicyContextConfig(),
            ScriptedContextPlanner(),
            _candidate(),
            (_runner_dataset(),),
        )
    )

    assert len(results) == 4
    assert all(result["passed"] for result in results)
    assert summary["overall"]["exact_turn_accuracy"] == 1.0
    assert summary["overall"]["supported_coverage"] == 1.0
    assert summary["by_language"]["en"]["turns"] == 2
    assert summary["by_language"]["tr"]["turns"] == 2
    assert summary["candidate"]["candidate_id"] == "scripted-context-v1"
    serialized = json.dumps(results)
    assert "private classification question" not in serialized
    assert "private speed question" not in serialized
    assert "private social remark" not in serialized
    assert "private rejoin question" not in serialized


def test_context_candidate_failure_is_recorded_without_transcript() -> None:
    dataset = IntelligenceEvaluationDataset.model_validate(
        {
            "dataset_id": "intelligence-failure-test",
            "revision": "1",
            "split": "development",
            "fixture": "fixtures/synthetic/intelligence-capabilities",
            "groups": [
                {
                    "id": "failure-control",
                    "family": "failure-control-family",
                    "category": "classification",
                    "language": "en",
                    "questions": ["private failure transcript"],
                    "expected": {
                        "temporal_scope": "current",
                        "capability_ids": ["current_classification"],
                        "evidence_outcome": "available",
                    },
                }
            ],
        }
    )
    results, summary = asyncio.run(
        evaluate_intelligence_datasets(
            ROOT,
            PolicyContextConfig(),
            FailingContextPlanner(),
            _candidate(),
            (dataset,),
        )
    )

    assert results[0]["observed_evidence_outcome"] == "model_error"
    assert results[0]["reason"] == "context_model_unreachable"
    assert summary["overall"]["model_errors"] == 1
    assert summary["overall"]["transport_errors"] == 1
    assert summary["overall"]["plan_validation_errors"] == 0
    assert results[0]["unknowns_passed"] is False
    assert "private failure transcript" not in json.dumps(results)


class FixedModel:
    def __init__(self, raw: object) -> None:
        self.raw = wire_plan(raw)

    async def request(self, **kwargs):
        return self.raw


def _evaluate_classification(raw):
    dataset = _runner_dataset().model_copy(update={"groups": _runner_dataset().groups[:1]})
    planner = QwenContextQueryPlanner(ConversationConfig(), model=FixedModel(raw))
    return asyncio.run(
        evaluate_intelligence_datasets(
            ROOT, PolicyContextConfig(), planner, _candidate(), (dataset,)
        )
    )


def test_rejected_selection_diagnostic_never_counts_as_an_accepted_plan():
    results, summary = _evaluate_classification(
        {
            "temporal_scope": "current",
            "capability_ids": ["current_classification", "private guessed capability"],
            "queries": [
                {"signal_id": "private driver utterance copied here", "operation": "latest"},
                {"signal_id": "player.speed_mps", "operation": "latest"},
            ],
        }
    )
    result = results[0]
    assert result["observed_plan"] is None
    assert result["observed_evidence_outcome"] == "plan_rejected"
    assert result["failure_stage"] == "plan_validation"
    assert result["rejected_draft"]["capability_ids"] == ["current_classification"]
    assert result["rejected_draft"]["unknown_capability_count"] == 1
    assert result["rejected_draft"]["unknown_signal_count"] == 1
    assert result["rejected_draft"]["queries"][0]["signal"] == "speed_mps"
    assert result["rejected_draft_selection"]["required_selection_present"] is True
    assert result["selection"]["required_selection_present"] is False
    assert not result["passed"] and not result["required_evidence_passed"]
    assert summary["overall"]["supported_accepted"] == 0
    assert summary["overall"]["model_errors"] == 0
    assert summary["overall"]["plan_validation_errors"] == 1
    assert "private" not in json.dumps(results)


def test_extra_evidence_is_visible_without_relaxing_exact_pass():
    results, summary = _evaluate_classification(
        {
            "capability_ids": ["current_classification"],
            "queries": [{"signal_id": "player.speed_mps", "operation": "latest"}],
        }
    )
    result = results[0]
    assert result["observed_evidence_outcome"] == "available"
    assert result["required_evidence_passed"] is True
    assert result["passed"] is False
    assert result["selection"] == {
        "expected_request_count": 1,
        "matched_request_count": 1,
        "missing_request_count": 0,
        "extra_request_count": 1,
        "required_selection_present": True,
    }
    assert summary["overall"]["exact_turn_accuracy"] == 0
    assert summary["overall"]["required_evidence_accuracy"] == 1
    assert summary["overall"]["extra_evidence_turns"] == 1


def test_unrelated_evidence_does_not_earn_required_evidence_credit():
    results, summary = _evaluate_classification(
        {
            "queries": [{"signal_id": "player.speed_mps", "operation": "latest"}],
        }
    )
    assert results[0]["selection"]["missing_request_count"] == 1
    assert results[0]["selection"]["extra_request_count"] == 1
    assert summary["overall"]["required_evidence_accuracy"] == 0


def test_duplicate_queries_are_counted_as_extra_not_silently_deduplicated():
    results, _ = _evaluate_classification(
        {
            "capability_ids": ["current_classification"],
            "queries": [{"signal_id": "player.speed_mps", "operation": "latest"}] * 2,
        }
    )
    assert results[0]["selection"]["extra_request_count"] == 2


class UnknownScriptedPlanner(ScriptedContextPlanner):
    def __init__(self, unknown: str) -> None:
        self.unknown = unknown

    async def plan(self, *args):
        selected = await super().plan(*args)
        return selected.model_copy(update={"unknowns": (self.unknown,)})


def test_generic_social_unknown_remains_a_real_failure():
    dataset = _runner_dataset().model_copy(update={"groups": _runner_dataset().groups[2:3]})
    planner = UnknownScriptedPlanner("unknown")
    results, _ = asyncio.run(
        evaluate_intelligence_datasets(
            ROOT, PolicyContextConfig(), planner, _candidate(), (dataset,)
        )
    )
    assert results[0]["plan_passed"] is True
    assert results[0]["unknowns_passed"] is False
    assert results[0]["required_evidence_passed"] is False
    assert results[0]["passed"] is False


def test_evaluation_sanitizes_malformed_model_body():
    results, _ = _evaluate_classification(
        {
            "queries": [{"signal_id": "private text", "operation": "private raw content"}],
        }
    )
    assert results[0]["failure_stage"] == "plan_validation"
    assert results[0]["rejected_draft"] is None
    assert "private" not in json.dumps(results)


def test_evaluation_hashes_unrecognized_unknown_text_without_accepting_it():
    dataset = _runner_dataset().model_copy(update={"groups": _runner_dataset().groups[:1]})
    results, _ = asyncio.run(
        evaluate_intelligence_datasets(
            ROOT,
            PolicyContextConfig(),
            UnknownScriptedPlanner("private driver transcript copied by model"),
            _candidate(),
            (dataset,),
        )
    )
    assert results[0]["observed_unknowns"][0].startswith("unrecognized_unknown:")
    assert results[0]["unknowns_passed"] is False
    assert "private" not in json.dumps(results)


@pytest.mark.parametrize("invent_signal", [False, True])
def test_request_inventory_is_not_persisted_in_accepted_or_rejected_reports(invent_signal):
    results, _ = _evaluate_classification(
        {
            "social_comment": False,
            "requested_facts": ["private driver wording copied into model inventory"],
            "capability_ids": ["current_classification"],
            "queries": (
                [{"signal_id": "player.invented", "operation": "latest"}] if invent_signal else []
            ),
        }
    )
    assert results[0]["passed"] is not invent_signal
    assert "private" not in json.dumps(results)
    assert "requested_facts" not in json.dumps(results)


def test_intelligence_evaluation_cli_accepts_repeated_datasets() -> None:
    args = _parser().parse_args(
        [
            "evaluate-intelligence",
            "--dataset",
            "fixtures/intelligence-evaluation/development.json",
            "--dataset",
            "fixtures/intelligence-evaluation/calibration.json",
        ]
    )

    assert args.command == "evaluate-intelligence"
    assert len(args.dataset) == 2
    assert args.candidate == "qwen-context-v6"
    assert args.profile == "portable"


@pytest.mark.parametrize("candidate_id", [PLANNER_V5_ID, PLANNER_V6_ID])
def test_intelligence_evaluation_cli_accepts_the_separately_versioned_planner(candidate_id):
    args = _parser().parse_args(
        [
            "evaluate-intelligence",
            "--dataset",
            "fixtures/intelligence-evaluation/development-controls.json",
            "--candidate",
            candidate_id,
        ]
    )
    assert args.candidate == candidate_id


@pytest.mark.parametrize("candidate_id", [None, PLANNER_V5_ID, PLANNER_V6_ID])
def test_evaluation_application_builds_reproducible_content_free_report(
    monkeypatch: pytest.MonkeyPatch,
    candidate_id,
) -> None:
    async def fake_evaluate(*args, **kwargs):
        del args, kwargs
        return ([{"question_id": "hashed-only"}], {"overall": {"turns": 1}})

    monkeypatch.setattr(
        evaluation_application,
        "evaluate_intelligence_datasets",
        fake_evaluate,
    )
    monkeypatch.setattr(
        evaluation_application,
        "files_fingerprint",
        lambda *args: "fingerprint",
    )
    monkeypatch.setattr(evaluation_application, "git_revision", lambda root: "a" * 40)
    monkeypatch.setattr(evaluation_application, "git_dirty", lambda root: True)
    monkeypatch.setattr(
        evaluation_application,
        "machine_metadata",
        lambda: {"cpu": "test"},
    )

    candidate_options = {} if candidate_id is None else {"candidate_id": candidate_id}
    report = asyncio.run(
        evaluation_application.run_intelligence_evaluation(
            ROOT,
            ROOT / "config" / "default.toml",
            (Path("fixtures/intelligence-evaluation/development.json"),),
            **candidate_options,
        )
    )

    assert report["schema_version"] == "intelligence-eval-report.v2"
    assert report["candidate"]["candidate_id"] == (candidate_id or PLANNER_V6_ID)
    assert report["datasets"][0]["turns"] == 12
    assert report["reproducibility"]["input_fingerprint_sha256"] == "fingerprint"
    assert report["privacy"] == {
        "local_only": True,
        "report_contains_transcripts": False,
        "report_contains_model_replies": False,
    }


@pytest.mark.parametrize(
    "filename, turn_count",
    [("development-controls.json", 18), ("development-composition.json", 20)],
)
def test_supplemental_development_controls_execute_and_forward_private_history(
    filename, turn_count
):
    dataset = load_intelligence_evaluation_dataset(
        ROOT / "fixtures/intelligence-evaluation" / filename
    )
    assert dataset.turn_count == turn_count
    assert dataset.language_counts() == {"en": turn_count // 2, "tr": turn_count // 2}
    validate_intelligence_split_families(
        (dataset, *(load_intelligence_evaluation_dataset(path) for path in DATASETS))
    )
    expected_by_question = {
        question: group.expected for group in dataset.groups for question in group.questions
    }
    seen_history = {}

    class HistoryControl(ExpectedContextPlanner):
        async def plan(self, turn, signals, capabilities):
            seen_history[turn.transcript] = turn.recent_dialogue
            return await super().plan(turn, signals, capabilities)

    results, summary = asyncio.run(
        evaluate_intelligence_datasets(
            ROOT,
            PolicyContextConfig(),
            HistoryControl(expected_by_question),
            _candidate(),
            (dataset,),
        )
    )
    assert summary["overall"]["passed"] == turn_count
    assert summary["overall"]["purpose_cases"] == turn_count
    assert summary["overall"]["purpose_accuracy"] == 1
    report = json.dumps(results, ensure_ascii=False)
    for group in dataset.groups:
        for question in group.questions:
            assert seen_history[question] == group.recent_dialogue
            assert question not in report
        for line in group.recent_dialogue:
            assert line not in report


def test_planner_contrast_development_dataset_is_balanced_and_uses_fresh_families():
    dataset = load_intelligence_evaluation_dataset(
        ROOT / "fixtures/intelligence-evaluation/development-planner-contrasts.json"
    )
    assert dataset.turn_count == 8
    assert dataset.language_counts() == {"en": 4, "tr": 4}
    assert {group.category for group in dataset.groups} == {
        "vent",
        "future",
        "compound",
        "mixed",
    }
    validate_intelligence_split_families(
        (dataset, *(load_intelligence_evaluation_dataset(path) for path in DATASETS))
    )


def test_correct_evidence_with_wrong_purpose_does_not_pass_mixed_control():
    original = _runner_dataset()
    group = original.groups[0]
    dataset = original.model_copy(
        update={
            "groups": (
                group.model_copy(
                    update={"expected": group.expected.model_copy(update={"purpose": "mixed"})}
                ),
            )
        }
    )
    planner = QwenContextQueryPlanner(
        ConversationConfig(),
        model=FixedModel(
            {"purpose": "race_information", "capability_ids": ["current_classification"]}
        ),
    )
    results, summary = asyncio.run(
        evaluate_intelligence_datasets(
            ROOT, PolicyContextConfig(), planner, _candidate(), (dataset,)
        )
    )
    result = results[0]
    assert result["selection"]["required_selection_present"] is True
    assert result["outcome_passed"] is True
    assert result["purpose_passed"] is False
    assert result["plan_passed"] is False
    assert result["required_evidence_passed"] is False
    assert result["passed"] is False
    assert summary["overall"]["purpose_accuracy"] == 0


@pytest.mark.parametrize("purpose, scope", [("mixed", "social"), ("social", "current")])
def test_expected_purpose_must_agree_with_scope(purpose, scope):
    with pytest.raises(ValidationError, match="pure social"):
        ExpectedContextPlan(
            purpose=purpose,
            temporal_scope=scope,
            evidence_outcome="no_evidence",
        )
