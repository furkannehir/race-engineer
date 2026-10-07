import asyncio
import json
from argparse import Namespace
from pathlib import Path

import pytest

from race_engineer.application.intelligence_cli import _contexts
from race_engineer.config import load_config
from race_engineer.evaluation.intelligence import load_intelligence_evaluation_dataset
from race_engineer.evaluation.intelligence_latency import (
    MeasuredModel,
    measure_intelligence_turn,
    summarize_latency,
)
from race_engineer.fixtures import load_fixture
from race_engineer.intelligence.local_model import LocalIntelligenceError

ROOT = Path(__file__).parents[1]


class FixedModel:
    def __init__(self, *replies):
        self.replies = iter(replies)
        self.calls = 0

    async def request(self, **kwargs):
        self.calls += 1
        reply = next(self.replies)
        if isinstance(reply, Exception):
            raise reply
        return reply


def setup():
    config = load_config(ROOT / "config/default.toml")
    dataset = load_intelligence_evaluation_dataset(
        ROOT / "fixtures/intelligence-evaluation/development-controls.json"
    )
    fixture = load_fixture(ROOT / dataset.fixture)
    contexts = _contexts(fixture, config.policy.context)
    group = next(group for group in dataset.groups if group.id == "compound-en")
    return config.conversation, contexts, group


def planner_reply(extra=False):
    return {
        "social_comment": False,
        "requested_facts": ["remaining fuel laps", "current speed"],
        "temporal_scope": "current",
        "capability_ids": ["fuel_range", "current_classification"] if extra else ["fuel_range"],
        "queries": [{"signal_id": "player.speed_mps", "operation": "latest", "window_s": None}],
        "missing_information": "none",
    }


def core_reply():
    return {
        "goal": "inform",
        "tone": "calm_teammate",
        "guidance": [],
        "confidence": 1,
        "speech_template": "Speed: {{speed}} metres per second. Fuel range: {{fuel}} laps.",
        "references": [
            {"placeholder": "speed", "evidence_id": "e1", "field": "value"},
            {"placeholder": "fuel", "evidence_id": "c1:laps_remaining", "field": "value"},
        ],
    }


def measure(planner, core):
    config, contexts, group = setup()
    return asyncio.run(
        measure_intelligence_turn(
            config,
            contexts,
            group,
            group.questions[0],
            run_index=1,
            repetition=1,
            planner_client=planner,
            core_client=core,
        )
    )


def test_benchmark_measures_both_model_calls_refresh_and_grounding_without_extra_inference():
    planner, core = FixedModel(planner_reply()), FixedModel(core_reply())
    result, reply = measure(planner, core)
    assert planner.calls == core.calls == 1
    assert reply == "Speed: 51 metres per second. Fuel range: 12 laps."
    assert result["pipeline_completed"] is True
    assert result["planner_exact_match"] is True
    assert result["evidence_expectation_match"] is True
    assert result["core_retries"] == 0
    assert [request["role"] for request in result["model_requests"]] == ["context", "core"]
    assert set(result["timing_ms"]) == {
        "context",
        "core",
        "materialize",
        "refresh",
        "grounding",
        "total",
    }
    assert result["manual_reply_review_required"] is True
    serialized = json.dumps(result)
    assert reply not in serialized
    assert "remaining fuel laps" not in serialized
    assert "requested_facts" not in serialized


def test_benchmark_distinguishes_successful_reply_from_incorrect_extra_plan_selection():
    result, reply = measure(FixedModel(planner_reply(extra=True)), FixedModel(core_reply()))
    assert reply is not None
    assert result["pipeline_completed"] is True
    assert result["planner_exact_match"] is False
    summary = summarize_latency([result])
    assert summary["pipeline_completed"] == 1
    assert summary["planner_exact_matches"] == 0
    assert summary["automatic_reply_accuracy"] is None


def test_benchmark_counts_the_existing_core_repair_call():
    invalid = {**core_reply(), "speech_template": ""}
    core = FixedModel(invalid, core_reply())
    result, reply = measure(FixedModel(planner_reply()), core)
    assert reply is not None
    assert result["core_retries"] == 1
    assert core.calls == 2
    assert [request["attempt"] for request in result["model_requests"]] == [1, 1, 2]


@pytest.mark.parametrize("reason", ["model_timeout", "model_response_incomplete"])
def test_benchmark_preserves_failed_stage_timing_without_counting_it_as_a_fast_reply(reason):
    core = FixedModel(core_reply())
    result, reply = measure(FixedModel(LocalIntelligenceError(reason)), core)
    assert reply is None
    assert core.calls == 0
    assert result["failure"] == {"stage": "context", "reason": "context_" + reason}
    assert result["pipeline_completed"] is False
    summary = summarize_latency([result])
    assert summary["pipeline_failed"] == 1
    assert summary["completed_turn_latency_ms"]["count"] == 0
    assert summary["all_turn_latency_ms"]["count"] == 1


def test_measured_model_copies_only_provider_metrics_not_request_or_reply_content():
    config, _, _ = setup()
    client = FixedModel({"private_response": "not reportable"})
    measured = MeasuredModel(config, "context", client=client)
    asyncio.run(measured.request(system_prompt="private prompt", content="secret", schema={}))
    trace = measured.requests[0]
    assert trace["status"] == "completed"
    assert trace["role"] == "context"
    assert "private" not in json.dumps(trace)
    assert "secret" not in json.dumps(trace)


def test_summary_reports_completed_slow_tails_separately_from_failed_turns():
    result, _ = measure(FixedModel(planner_reply()), FixedModel(core_reply()))
    fast = {**result, "timing_ms": {"total": 4000}}
    slow = {**result, "timing_ms": {"total": 43000}, "core_retries": 1}
    failed = {**result, "timing_ms": {"total": 30000}, "pipeline_completed": False}
    summary = summarize_latency([fast, slow, failed])
    assert summary["pipeline_completed"] == 2
    assert summary["completed_under_5s"] == 1
    assert summary["completed_at_least_30s"] == 1
    assert summary["all_turn_latency_ms"]["count"] == 3
    assert summary["completed_turn_latency_ms"]["count"] == 2
    assert summary["core_retries"] == 1


def test_benchmark_stops_after_timeout_and_closes_only_its_managed_runtime(monkeypatch):
    from scripts import benchmark_intelligence as script

    calls = []
    runtime = []

    class Server:
        process = None
        effective = None

        def __init__(self, *args):
            pass

        async def start(self):
            runtime.append("reused external")

        async def aclose(self):
            runtime.append("closed lifecycle")

    async def fail(*args, **kwargs):
        calls.append(kwargs)
        return {
            "pipeline_completed": False,
            "planner_exact_match": False,
            "evidence_expectation_match": False,
            "core_retries": 0,
            "timing_ms": {"context": 30000, "total": 30000},
            "failure": {"stage": "context", "reason": "context_model_timeout"},
        }, None

    monkeypatch.setattr(script, "ConversationServer", Server)
    monkeypatch.setattr(script, "measure_intelligence_turn", fail)
    monkeypatch.setattr(script, "machine_metadata", lambda: {})
    monkeypatch.setattr(script, "git_revision", lambda _: None)
    monkeypatch.setattr(script, "git_dirty", lambda _: True)
    args = Namespace(
        config=ROOT / "config/default.toml",
        dataset=ROOT / "fixtures/intelligence-evaluation/development-controls.json",
        case=["compound-en", "compound-tr"],
        repeats=2,
        seed=7,
        show_replies=False,
        output=None,
    )
    report = asyncio.run(script.benchmark(args))
    assert len(calls) == 1
    assert runtime == ["reused external", "closed lifecycle"]
    assert report["expected_turns"] == 4
    assert report["summary"]["turns"] == 1
    assert report["runtime"]["ownership"] == "external_unmanaged"
    assert report["runtime"]["effective_mode"] is None
    assert report["reproducibility"]["implementation_changed_during_run"] is False


def test_benchmark_refuses_to_overwrite_an_existing_report_before_server_start(tmp_path):
    from scripts.benchmark_intelligence import benchmark

    output = tmp_path / "existing.json"
    output.write_text("user data", encoding="utf-8")
    args = Namespace(
        config=ROOT / "config/default.toml",
        dataset=ROOT / "fixtures/intelligence-evaluation/development-controls.json",
        case=["compound-en"],
        repeats=1,
        seed=7,
        show_replies=False,
        output=output,
    )
    with pytest.raises(ValueError, match="output already exists"):
        asyncio.run(benchmark(args))
    assert output.read_text(encoding="utf-8") == "user data"
