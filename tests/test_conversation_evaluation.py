import asyncio
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from race_engineer.config import ConversationConfig
from race_engineer.core.conversation import ConversationPlan
from race_engineer.evaluation.conversation import (
    ConversationEvaluationDataset,
    load_evaluation_dataset,
    validate_split_families,
)
from race_engineer.evaluation.runner import evaluate_datasets
from race_engineer.evaluation.system import distribution

ROOT = Path(__file__).parents[1]
DATASETS = tuple(
    ROOT / "fixtures" / "conversation" / name
    for name in ("cases.json", "holdout.json", "calibration.json", "locked.json")
)


class ScriptedPlanner:
    def __init__(self, *plans: ConversationPlan) -> None:
        self._plans = iter(plans)
        self.requests = []

    async def plan(self, request):
        self.requests.append(request)
        return next(self._plans)


def _plan(*queries: str) -> ConversationPlan:
    return ConversationPlan(language="en", queries=queries, clarification="none")


def test_committed_datasets_are_valid_and_family_disjoint() -> None:
    datasets = tuple(load_evaluation_dataset(path) for path in DATASETS)
    validate_split_families(datasets)

    locked = datasets[-1]
    assert locked.split == "locked"
    assert locked.turn_count == 200
    assert locked.language_counts() == {"en": 100, "tr": 100}


def test_expected_queries_must_be_unique() -> None:
    with pytest.raises(ValidationError, match="expected queries must be unique"):
        ConversationEvaluationDataset.model_validate(
            {
                "dataset_id": "duplicate-query-test",
                "revision": "1",
                "split": "development",
                "groups": [
                    {
                        "id": "duplicate-query",
                        "family": "duplicate-query-family",
                        "category": "position",
                        "questions": ["Position?"],
                        "expected": {
                            "language": "en",
                            "queries": ["position", "position"],
                            "reply_status": "answered",
                        },
                    }
                ],
            }
        )


def test_paraphrase_family_cannot_cross_splits() -> None:
    base = {
        "revision": "1",
        "fixture": "fixtures/synthetic/conversation",
        "groups": [
            {
                "id": "same-id",
                "family": "same-family",
                "category": "position",
                "questions": ["Position?"],
                "expected": {
                    "language": "en",
                    "queries": ["position"],
                    "reply_status": "answered",
                },
            }
        ],
    }
    development = ConversationEvaluationDataset.model_validate(
        {**base, "dataset_id": "development-test", "split": "development"}
    )
    calibration = ConversationEvaluationDataset.model_validate(
        {**base, "dataset_id": "calibration-test", "split": "calibration"}
    )
    with pytest.raises(ValueError, match="appears in development and calibration"):
        validate_split_families((development, calibration))


def test_fake_planner_evaluation_is_content_free_and_tracks_history() -> None:
    dataset = ConversationEvaluationDataset.model_validate(
        {
            "dataset_id": "runner-test",
            "revision": "1",
            "split": "development",
            "fixture": "fixtures/synthetic/conversation",
            "groups": [
                {
                    "id": "position-forms",
                    "family": "position-forms-family",
                    "category": "position",
                    "questions": ["Position?", "Where are we?"],
                    "expected": {
                        "language": "en",
                        "queries": ["position"],
                        "reply_status": "answered",
                    },
                }
            ],
            "sequences": [
                {
                    "id": "gap-followup",
                    "family": "gap-followup-family",
                    "category": "followup",
                    "turns": [
                        {
                            "question": "What's the gap behind?",
                            "expected": {
                                "language": "en",
                                "queries": ["gap_behind"],
                                "reply_status": "answered",
                            },
                        },
                        {
                            "question": "And now?",
                            "frame_index": 1,
                            "expected": {
                                "language": "en",
                                "queries": ["gap_behind"],
                                "reply_status": "answered",
                            },
                        },
                    ],
                }
            ],
        }
    )
    planner = ScriptedPlanner(
        _plan("position"),
        _plan("position"),
        _plan("gap_behind"),
        _plan("gap_behind"),
    )

    results, summary = asyncio.run(
        evaluate_datasets(ROOT, ConversationConfig(), planner, (dataset,))
    )

    assert len(results) == 4
    assert all(result["passed"] for result in results)
    assert summary["overall"] == {
        "turns": 4,
        "passed": 4,
        "exact_turn_accuracy": 1.0,
        "plans_returned": 4,
        "coverage": 1.0,
        "exact_plan_accuracy": 1.0,
        "accepted_turns": 4,
        "accepted_plan_accuracy": 1.0,
        "supported_turns": 4,
        "supported_accepted": 4,
        "supported_coverage": 1.0,
        "clarification_turns": 0,
        "clarification_accuracy": None,
        "model_errors": 0,
    }
    assert planner.requests[0].history == ()
    assert planner.requests[1].history == ()
    assert planner.requests[2].history == ()
    assert len(planner.requests[3].history) == 1
    serialized = json.dumps(results)
    assert "Position?" not in serialized
    assert "What's the gap behind?" not in serialized


def test_latency_distribution_uses_interpolated_percentiles() -> None:
    assert distribution([10.0, 20.0, 30.0, 40.0]) == {
        "count": 4,
        "min": 10.0,
        "mean": 25.0,
        "p50": 25.0,
        "p95": 38.5,
        "max": 40.0,
    }
